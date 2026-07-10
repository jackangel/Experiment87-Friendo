"""
Memory Router

Handles memory retrieval and injection into model inference.

Key Features:
- Attention-based memory retrieval using query embeddings
- Multiple injection modes:
  * state_fusion: Blend SSM carry states (most efficient)
  * kv_injection: Inject high-saliency KV cache entries
  * context_prepend: Traditional RAG-style (least efficient)
- Adaptive retrieval based on query context
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from .consolidation import MemoryConsolidation
from .core import MemoryEntry


class MemoryRouter:
    """
    Routes relevant memories into model inference
    
    The router uses the model's current state (carry states and embeddings)
    to retrieve relevant memories and inject them into the forward pass.
    """
    
    def __init__(self,
                 consolidation: MemoryConsolidation,
                 device: str = 'cuda',
                 default_injection_mode: str = 'state_fusion'):
        """
        Args:
            consolidation: MemoryConsolidation instance managing memories
            device: Device for operations
            default_injection_mode: 'state_fusion', 'kv_injection', or 'context_prepend'
        """
        self.consolidation = consolidation
        self.device = device
        self.default_injection_mode = default_injection_mode
        
        # Validate injection mode
        valid_modes = ['state_fusion', 'kv_injection', 'context_prepend']
        if default_injection_mode not in valid_modes:
            raise ValueError(f"Invalid injection_mode: {default_injection_mode}. "
                           f"Must be one of {valid_modes}")
        
        print(f"[MemoryRouter] Initialized with injection_mode={default_injection_mode}")
    
    def retrieve_memories(self,
                         query_embedding: torch.Tensor,
                         max_memories: int = 3,
                         episodic_ratio: float = 0.4) -> List[Tuple[MemoryEntry, float, str]]:
        """
        Retrieve relevant memories for a query
        
        Args:
            query_embedding: Query embedding vector (typically from final carry state)
            max_memories: Maximum number of memories to retrieve
            episodic_ratio: Proportion of results from episodic memory (0.0-1.0)
        
        Returns:
            List of (memory, similarity, source) tuples
        """
        # Calculate how many from each source
        episodic_k = max(1, int(max_memories * episodic_ratio))
        semantic_k = max_memories - episodic_k + 1  # Get extra for ranking
        
        # Retrieve from both memory stores
        results = self.consolidation.retrieve_combined(
            query_embedding,
            episodic_k=episodic_k,
            semantic_k=semantic_k,
            total_k=max_memories
        )
        
        if results:
            print(f"[MemoryRouter] Retrieved {len(results)} memories:")
            for i, (memory, sim, source) in enumerate(results):
                print(f"  {i+1}. [{source:9s}] similarity={sim:.3f}, "
                      f"importance={memory.importance_score:.3f}, "
                      f"tokens={len(memory.tokens)}")
        
        return results
    
    def inject_memories(self,
                       query_embedding: torch.Tensor,
                       current_carry_states: Optional[List[torch.Tensor]],
                       current_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
                       max_memories: int = 3,
                       injection_mode: Optional[str] = None,
                       fusion_weight: float = 0.3) -> Tuple[List[torch.Tensor], Any]:
        """
        Retrieve and inject memories into the model's current state
        
        Args:
            query_embedding: Query embedding from current context
            current_carry_states: Current SSM carry states (one per layer)
            current_kv: Current KV cache (one per layer)
            max_memories: Max number of memories to retrieve
            injection_mode: Override default injection mode
            fusion_weight: Weight for memory fusion (0.0-1.0, typically 0.2-0.4)
        
        Returns:
            modified_carry_states: Carry states with memory injected
            injection_info: Dict with injection statistics
        """
        # Use default injection mode if not specified
        injection_mode = injection_mode or self.default_injection_mode
        
        # Retrieve relevant memories
        memories = self.retrieve_memories(query_embedding, max_memories=max_memories)
        
        if not memories:
            print("[MemoryRouter] No memories retrieved, returning original states")
            return current_carry_states, {'num_memories': 0, 'injection_mode': injection_mode}
        
        # Apply injection based on mode
        if injection_mode == 'state_fusion':
            modified_carry = self._fuse_carry_states(current_carry_states, memories, fusion_weight)
            modified_kv = current_kv  # KV cache unchanged
            
        elif injection_mode == 'kv_injection':
            modified_carry = current_carry_states  # Carry states unchanged
            modified_kv = self._inject_kv_cache(current_kv, memories)
            
        elif injection_mode == 'context_prepend':
            # This mode requires token manipulation, handled externally
            raise NotImplementedError("context_prepend mode must be handled by caller (requires tokenizer)")
        
        else:
            raise ValueError(f"Invalid injection_mode: {injection_mode}")
        
        injection_info = {
            'num_memories': len(memories),
            'injection_mode': injection_mode,
            'avg_similarity': sum(sim for _, sim, _ in memories) / len(memories),
            'sources': [source for _, _, source in memories],
            'fusion_weight': fusion_weight if injection_mode == 'state_fusion' else None
        }
        
        return modified_carry, injection_info
    
    def _fuse_carry_states(self,
                          current_carry: Optional[List[torch.Tensor]],
                          memories: List[Tuple[MemoryEntry, float, str]],
                          fusion_weight: float) -> List[torch.Tensor]:
        """
        Fuse current carry states with retrieved memory states
        
        This is the key innovation: we blend compressed SSM representations
        rather than concatenating raw tokens.
        
        Formula:
            fused_state = (1 - α) * current_state + α * weighted_memory_blend
            where α = fusion_weight, typically 0.2-0.4
        
        Args:
            current_carry: Current SSM carry states per layer
            memories: Retrieved memories with similarity scores
            fusion_weight: Weight for memory contribution (0.0-1.0)
        
        Returns:
            Fused carry states
        """
        if current_carry is None or len(memories) == 0:
            return current_carry
        
        # Extract memories and normalize weights by similarity
        memory_entries = [mem for mem, _, _ in memories]
        similarities = torch.tensor([sim for _, sim, _ in memories], device=self.device)
        weights = F.softmax(similarities, dim=0)  # Normalize similarities to sum to 1
        
        fused_carry = []
        
        for layer_idx, carry in enumerate(current_carry):
            if carry is None:
                fused_carry.append(None)
                continue
            
            carry = carry.to(self.device)
            
            # Collect memory carry states for this layer
            memory_carries = []
            for memory in memory_entries:
                if layer_idx < len(memory.carry_states):
                    mem_carry = memory.carry_states[layer_idx]
                    if mem_carry is not None:
                        memory_carries.append(mem_carry.to(self.device))
            
            if len(memory_carries) == 0:
                fused_carry.append(carry)
                continue
            
            # Weighted blend of memory states
            memory_blend = sum(w * mc for w, mc in zip(weights, memory_carries))
            
            # Fuse current state with memory blend
            # current state (1 - fusion_weight) + memory blend (fusion_weight)
            fused = (1.0 - fusion_weight) * carry + fusion_weight * memory_blend
            
            fused_carry.append(fused.detach())
        
        print(f"[MemoryRouter] Fused {len(memories)} memories into carry states "
              f"(fusion_weight={fusion_weight:.2f})")
        
        return fused_carry
    
    def _inject_kv_cache(self,
                        current_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
                        memories: List[Tuple[MemoryEntry, float, str]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Inject high-saliency KV cache entries from memories
        
        This prepends memory KV entries to the current KV cache,
        allowing the model to attend to important past information.
        
        Note: This increases KV cache size, so use carefully.
        
        Args:
            current_kv: Current KV cache per layer
            memories: Retrieved memories
        
        Returns:
            Modified KV cache with memory entries prepended
        """
        if current_kv is None or len(memories) == 0:
            return current_kv
        
        memory_entries = [mem for mem, _, _ in memories]
        modified_kv = []
        
        for layer_idx, (k, v, s, kr) in enumerate(current_kv):
            # Collect memory KV entries for this layer
            memory_kvs = []
            for memory in memory_entries:
                if layer_idx < len(memory.kv_cache):
                    mem_k, mem_v, mem_s, mem_kr = memory.kv_cache[layer_idx]
                    memory_kvs.append((
                        mem_k.to(self.device),
                        mem_v.to(self.device),
                        mem_s.to(self.device),
                        mem_kr.to(self.device)
                    ))
            
            if len(memory_kvs) == 0:
                modified_kv.append((k, v, s, kr))
                continue
            
            # Concatenate memory KV entries with current KV
            mem_k_all = torch.cat([mkv[0] for mkv in memory_kvs], dim=2)
            mem_v_all = torch.cat([mkv[1] for mkv in memory_kvs], dim=2)
            mem_s_all = torch.cat([mkv[2] for mkv in memory_kvs], dim=1)
            mem_kr_all = torch.cat([mkv[3] for mkv in memory_kvs], dim=2)
            
            # Prepend to current KV cache
            k_full = torch.cat([mem_k_all, k], dim=2)
            v_full = torch.cat([mem_v_all, v], dim=2)
            s_full = torch.cat([mem_s_all, s], dim=1)
            kr_full = torch.cat([mem_kr_all, kr], dim=2)
            
            modified_kv.append((k_full, v_full, s_full, kr_full))
        
        print(f"[MemoryRouter] Injected {len(memories)} memory KV entries into cache")
        
        return modified_kv
    
    def create_query_embedding(self, carry_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Create a query embedding from current carry states
        
        Uses the final layer's carry state as the query embedding
        (this is already a compressed representation of the sequence)
        
        Args:
            carry_states: List of carry states from all layers
        
        Returns:
            Query embedding vector (D,)
        """
        if not carry_states or carry_states[-1] is None:
            raise ValueError("Cannot create query embedding from None carry states")
        
        final_carry = carry_states[-1].to(self.device)
        
        # Average across batch dimension if present
        if final_carry.dim() == 2:
            query_embedding = final_carry.mean(dim=0)  # (D,)
        else:
            query_embedding = final_carry.view(-1)
        
        return query_embedding
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return self.consolidation.get_stats()


class MemoryAugmentedGenerator:
    """
    Wrapper for memory-augmented generation
    
    Automatically retrieves and injects relevant memories during generation.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 memory_router: MemoryRouter,
                 device: str = 'cuda'):
        """
        Args:
            model: The SSMTransformer model
            tokenizer: Tokenizer instance
            memory_router: MemoryRouter for retrieval/injection
            device: Device for operations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.memory_router = memory_router
        self.device = device
    
    def generate_with_memory(self,
                            context_ids: List[int],
                            max_new_tokens: int = 256,
                            chunk_size: int = 512,
                            temperature: float = 0.8,
                            repetition_penalty: float = 1.2,
                            top_k: int = 50,
                            top_p: float = 0.9,
                            memory_retrieval_interval: int = 50,
                            max_memories_per_retrieval: int = 3,
                            fusion_weight: float = 0.3) -> Tuple[List[int], Dict[str, Any]]:
        """
        Generate text with memory augmentation
        
        Periodically retrieves and injects relevant memories during generation.
        
        Args:
            context_ids: Input token IDs
            max_new_tokens: Max tokens to generate
            chunk_size: Context chunk size for processing
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty factor
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            memory_retrieval_interval: Retrieve memories every N tokens
            max_memories_per_retrieval: Max memories to retrieve each time
            fusion_weight: Weight for memory state fusion
        
        Returns:
            generated_ids: Complete token sequence (context + generated)
            generation_info: Statistics about generation and memory usage
        """
        self.model.eval()
        
        with torch.inference_mode():
            # Process initial context
            # (Use existing block recurrent processing from the model)
            # For simplicity, we'll process the full context first
            
            context_tensor = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            _, carry_states, kv_cache, _ = self.model(
                x=context_tensor,
                carry_states=None,
                is_training=False,
                past_key_values=None,
                use_cache=True,
                abs_pos_offset=0
            )
            
            generated_ids = context_ids.copy()
            abs_pos_offset = len(context_ids)
            tokens_generated = 0
            memory_retrieval_count = 0
            
            # Generation loop with periodic memory retrieval
            while tokens_generated < max_new_tokens:
                # Periodic memory retrieval
                if tokens_generated % memory_retrieval_interval == 0:
                    try:
                        query_embedding = self.memory_router.create_query_embedding(carry_states)
                        carry_states, injection_info = self.memory_router.inject_memories(
                            query_embedding=query_embedding,
                            current_carry_states=carry_states,
                            current_kv=kv_cache,
                            max_memories=max_memories_per_retrieval,
                            fusion_weight=fusion_weight
                        )
                        if injection_info['num_memories'] > 0:
                            memory_retrieval_count += 1
                    except Exception as e:
                        print(f"[MemoryAugmentedGenerator] Memory retrieval failed: {e}")
                
                # Standard token generation
                last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=self.device)
                logits, carry_states, kv_cache, _ = self.model(
                    x=last_token,
                    carry_states=carry_states,
                    is_training=False,
                    past_key_values=kv_cache,
                    use_cache=True,
                    abs_pos_offset=abs_pos_offset
                )
                
                abs_pos_offset += 1
                
                # Sampling (convert to float32 for stability)
                next_token_logits = logits[0, -1].float().clone()
                
                # Apply penalties
                if repetition_penalty != 1.0:
                    for token in set(generated_ids):
                        if next_token_logits[token] < 0:
                            next_token_logits[token] *= repetition_penalty
                        else:
                            next_token_logits[token] /= repetition_penalty
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_ids.append(next_token)
                tokens_generated += 1
                
                # Check stopping conditions
                if next_token == self.tokenizer.tokenizer.eot_token:
                    break
            
            generation_info = {
                'tokens_generated': tokens_generated,
                'memory_retrievals': memory_retrieval_count,
                'memory_stats': self.memory_router.get_memory_stats()
            }
            
            return generated_ids, generation_info
