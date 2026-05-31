"""
Latent Thinking Module

Implements internal reasoning loops before committing to output.
Instead of immediately outputting the next token, the model can
engage in "thinking loops" to refine its reasoning.

Key Concepts:
- Uncertainty Detection: Model assesses confidence in current answer
- Thinking Loops: Cycles through network with <think> tokens
- Adaptive Depth: Learns how many loops are needed
- Confidence Refinement: Stops when confident enough
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class ThinkingController(nn.Module):
    """
    Decides whether to output immediately or engage in latent thinking
    
    Analyzes the model's hidden state to assess confidence and determine
    if additional reasoning is needed before committing to an output.
    """
    
    def __init__(self, dim: int, max_thinking_depth: int = 5):
        """
        Args:
            dim: Model hidden dimension
            max_thinking_depth: Maximum number of thinking loops allowed
        """
        super().__init__()
        self.dim = dim
        self.max_thinking_depth = max_thinking_depth
        
        # Confidence predictor from final hidden state
        # Outputs 0-1 score: 1.0 = very confident, 0.0 = very uncertain
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Thinking depth predictor
        # Predicts optimal number of thinking loops needed
        self.depth_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, max_thinking_depth),
            nn.Softmax(dim=-1)
        )
        
        # Uncertainty features extractor
        # Identifies what makes the model uncertain
        self.uncertainty_analyzer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 8)  # 8 uncertainty types
        )
        
        print(f"[ThinkingController] Initialized with max_depth={max_thinking_depth}")
    
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analyze hidden state to determine thinking requirements
        
        Args:
            hidden_state: (B, L, D) - final layer hidden state
        
        Returns:
            confidence: (B,) - confidence scores (0-1)
            should_think: (B,) - binary decision to engage thinking
            thinking_depth: (B,) - predicted optimal thinking depth (1 to max_depth)
            uncertainty_types: (B, 8) - types of uncertainty detected
        """
        # Use last token's hidden state for decision
        last_hidden = hidden_state[:, -1, :]  # (B, D)
        
        # Compute confidence
        confidence = self.confidence_head(last_hidden).squeeze(-1)  # (B,)
        
        # Predict optimal thinking depth
        depth_probs = self.depth_predictor(last_hidden)  # (B, max_depth)
        thinking_depth = torch.argmax(depth_probs, dim=-1) + 1  # 1 to max_depth
        
        # Analyze uncertainty types
        uncertainty_types = self.uncertainty_analyzer(last_hidden)  # (B, 8)
        
        # Decision: should we think?
        # Low confidence OR high uncertainty → should think
        should_think = confidence < 0.7  # Threshold
        
        return confidence, should_think, thinking_depth, uncertainty_types
    
    def compute_thinking_loss(self, 
                             predicted_confidence: torch.Tensor,
                             predicted_depth: torch.Tensor,
                             target_confidence: torch.Tensor,
                             target_depth: torch.Tensor) -> torch.Tensor:
        """
        Supervision loss for learning when and how much to think
        
        Args:
            predicted_confidence: Model's confidence predictions
            predicted_depth: Model's depth predictions
            target_confidence: Ground truth confidence (from annotations)
            target_depth: Ground truth optimal depth
        
        Returns:
            loss: Combined supervision loss
        """
        # Confidence prediction loss
        confidence_loss = F.mse_loss(predicted_confidence, target_confidence)
        
        # Depth prediction loss
        depth_loss = F.mse_loss(
            predicted_depth.float(),
            target_depth.float()
        )
        
        # Combined loss
        total_loss = confidence_loss + 0.5 * depth_loss
        
        return total_loss


class LatentThinkingWrapper(nn.Module):
    """
    Wraps an SSM model to add latent thinking capability
    
    When the model is uncertain, it engages in internal thinking loops
    before committing to an output. This allows for deeper reasoning
    on complex problems while maintaining fast inference on simple ones.
    """
    
    def __init__(self, 
                 base_model,
                 tokenizer,
                 max_thinking_depth: int = 5,
                 thinking_threshold: float = 0.7):
        """
        Args:
            base_model: Base SSMTransformer model
            tokenizer: Tokenizer instance (must have <think> token)
            max_thinking_depth: Maximum number of thinking loops
            thinking_threshold: Confidence threshold below which to think
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_thinking_depth = max_thinking_depth
        self.thinking_threshold = thinking_threshold
        
        # Thinking controller
        self.thinking_controller = ThinkingController(
            base_model.dim,
            max_thinking_depth
        )
        
        # Add special thinking token to vocabulary
        # This token is used during internal reasoning loops
        self.think_token_id = tokenizer.vocab_size  # New token ID
        
        # Extend embedding layer to include <think> token
        old_vocab_size = base_model.tok_embeddings.num_embeddings
        new_vocab_size = old_vocab_size + 1
        
        # Get device of base model
        model_device = next(base_model.parameters()).device
        
        # Create new embedding layer with extended vocabulary
        new_embeddings = nn.Embedding(new_vocab_size, base_model.dim)
        new_embeddings.weight.data[:old_vocab_size] = base_model.tok_embeddings.weight.data
        # Initialize <think> token embedding (average of all tokens)
        new_embeddings.weight.data[old_vocab_size] = base_model.tok_embeddings.weight.data.mean(dim=0)
        
        # Move to same device as base model
        new_embeddings = new_embeddings.to(model_device)
        self.base_model.tok_embeddings = new_embeddings
        
        # Update output layer to handle new vocabulary size
        old_output_weight = base_model.output.weight.data
        new_output = nn.Linear(base_model.dim, new_vocab_size, bias=False)
        new_output.weight.data[:old_vocab_size] = old_output_weight
        # Initialize <think> output (average)
        new_output.weight.data[old_vocab_size] = old_output_weight.mean(dim=0)
        
        # Move to same device as base model and tie weights
        new_output = new_output.to(model_device)
        new_output.weight = new_embeddings.weight
        self.base_model.output = new_output
        
        # Thinking refinement layers
        # These transform carry states during thinking iterations
        self.thinking_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_model.dim, base_model.dim),
                nn.LayerNorm(base_model.dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(max_thinking_depth)
        ])
        
        # Move thinking components to same device as base model
        self.thinking_controller = self.thinking_controller.to(model_device)
        self.thinking_refiners = self.thinking_refiners.to(model_device)
        
        print(f"[LatentThinking] Initialized with max_depth={max_thinking_depth}, "
              f"threshold={thinking_threshold}")
        print(f"[LatentThinking] Extended vocabulary: {old_vocab_size} → {new_vocab_size}")
        print(f"[LatentThinking] <think> token ID: {self.think_token_id}")
    
    @property
    def dim(self):
        """Expose base model's dimension"""
        return self.base_model.dim
    
    @property
    def num_layers(self):
        """Expose base model's number of layers"""
        return self.base_model.num_layers
    
    @property
    def num_heads(self):
        """Expose base model's number of heads"""
        return self.base_model.num_heads
    
    def forward(self, x=None, inputs_embeds=None, carry_states=None, past_key_values=None, 
                use_cache=False, abs_pos_offset=0, is_training=True):
        """
        Standard forward pass (no thinking during training)
        
        This maintains compatibility with existing training code.
        Thinking is only engaged during inference.
        
        Args:
            x: Input token IDs (or None if using inputs_embeds)
            inputs_embeds: Pre-computed embeddings for fuzzy training (optional)
            carry_states: SSM carry states
            past_key_values: KV cache
            use_cache: Whether to cache KV states
            abs_pos_offset: Absolute position offset for RoPE
            is_training: Training mode flag
        """
        return self.base_model(
            x=x,
            inputs_embeds=inputs_embeds,
            carry_states=carry_states,
            is_training=is_training,
            past_key_values=past_key_values,
            use_cache=use_cache,
            abs_pos_offset=abs_pos_offset
        )
    
    def forward_with_thinking(self,
                             x: torch.Tensor,
                             carry_states: Optional[List[torch.Tensor]] = None,
                             past_key_values: Optional[List] = None,
                             abs_pos_offset: int = 0,
                             enable_thinking: bool = True,
                             verbose: bool = False) -> Tuple:
        """
        Forward pass with latent thinking capability
        
        When uncertain, the model engages in thinking loops before output.
        
        Args:
            x: Input tokens (B, L)
            carry_states: SSM carry states
            past_key_values: KV cache
            abs_pos_offset: Absolute position offset for RoPE
            enable_thinking: Whether to engage thinking mode
            verbose: Print thinking process details
        
        Returns:
            logits: Final token predictions
            carry_states: Updated carry states (includes thinking refinement)
            past_key_values: Updated KV cache
            thinking_trace: Dict with thinking metadata
        """
        # Initial forward pass
        logits, carry_states, past_key_values, mod_loss = self.base_model(
            x=x,
            carry_states=carry_states,
            is_training=False,
            past_key_values=past_key_values,
            use_cache=True,
            abs_pos_offset=abs_pos_offset
        )
        
        if not enable_thinking or self.training:
            return logits, carry_states, past_key_values, {
                'thinking_depth': 0,
                'initial_confidence': 1.0,
                'final_confidence': 1.0,
                'confidence_gain': 0.0
            }
        
        # Get hidden state from final layer (before output projection)
        # We approximate this by embedding the input tokens
        with torch.no_grad():
            # Embed the tokens to get hidden representation
            h = self.base_model.tok_embeddings(x)  # (B, L, D)
            # Apply final norm as approximation of final hidden state
            hidden_state = self.base_model.norm(h)  # (B, L, D)
        
        # Analyze with thinking controller
        confidence, should_think, predicted_depth, uncertainty = self.thinking_controller(
            hidden_state
        )
        
        initial_confidence = confidence.mean().item()
        
        thinking_trace = {
            'thinking_depth': 0,
            'initial_confidence': initial_confidence,
            'final_confidence': initial_confidence,
            'confidence_gain': 0.0,
            'confidence_trajectory': [initial_confidence],
            'uncertainty_types': uncertainty.tolist() if uncertainty is not None else []
        }
        
        # If confident enough, return immediately
        if not should_think.any() or initial_confidence >= self.thinking_threshold:
            if verbose:
                print(f"[Thinking] Confident ({initial_confidence:.3f}), no thinking needed")
            return logits, carry_states, past_key_values, thinking_trace
        
        # Determine thinking depth
        actual_depth = min(predicted_depth.max().item(), self.max_thinking_depth)
        
        if verbose:
            print(f"[Thinking] Uncertain ({initial_confidence:.3f}), engaging {actual_depth} thinking loops...")
        
        # Track previous confidence to detect if thinking helps
        prev_confidence = initial_confidence
        no_improvement_count = 0
        
        # Engage thinking loops
        for depth_idx in range(actual_depth):
            # Create thinking token input
            think_token = torch.full(
                (x.size(0), 1),
                self.think_token_id,
                dtype=torch.long,
                device=x.device
            )
            
            # Refine carry states through thinking layer
            refined_carry = []
            for layer_idx, carry in enumerate(carry_states):
                if carry is not None:
                    refined = self.thinking_refiners[depth_idx](carry)
                    refined_carry.append(refined)
                else:
                    refined_carry.append(None)
            
            # Forward pass with thinking token
            think_logits, refined_carry, past_key_values, _ = self.base_model(
                x=think_token,
                carry_states=refined_carry,
                is_training=False,
                past_key_values=past_key_values,
                use_cache=True,
                abs_pos_offset=abs_pos_offset + 1 + depth_idx
            )
            
            # Update states
            carry_states = refined_carry
            logits = think_logits
            
            # Check confidence improvement
            with torch.no_grad():
                # Embed the thinking token and approximate hidden state
                think_h = self.base_model.tok_embeddings(think_token)
                think_hidden = self.base_model.norm(think_h)
                new_confidence, _, _, _ = self.thinking_controller(think_hidden)
                new_confidence_val = new_confidence.mean().item()
            
            thinking_trace['thinking_depth'] = depth_idx + 1
            thinking_trace['confidence_trajectory'].append(new_confidence_val)
            
            if verbose:
                print(f"  Loop {depth_idx+1}: confidence {new_confidence_val:.3f}")
            
            # Check if thinking is actually helping
            confidence_delta = new_confidence_val - prev_confidence
            if confidence_delta <= 0.01:  # No meaningful improvement
                no_improvement_count += 1
                if no_improvement_count >= 2:  # Stop if no improvement for 2 consecutive loops
                    if verbose:
                        print(f"  No confidence improvement, stopping early (untrained thinking system)")
                    break
            else:
                no_improvement_count = 0  # Reset if we see improvement
            
            prev_confidence = new_confidence_val
            
            # Early stopping if confident
            if new_confidence_val >= self.thinking_threshold:
                if verbose:
                    print(f"  Confident after {depth_idx+1} loops, stopping early")
                break
        
        thinking_trace['final_confidence'] = thinking_trace['confidence_trajectory'][-1]
        thinking_trace['confidence_gain'] = (
            thinking_trace['final_confidence'] - thinking_trace['initial_confidence']
        )
        
        if verbose:
            print(f"[Thinking] Complete: {thinking_trace['thinking_depth']} loops, "
                  f"confidence {initial_confidence:.3f} → {thinking_trace['final_confidence']:.3f}")
        
        return logits, carry_states, past_key_values, thinking_trace
    
    def generate_with_thinking(self,
                              context_ids: List[int],
                              device: str,
                              max_new_tokens: int = 256,
                              temperature: float = 0.8,
                              top_k: int = 50,
                              top_p: float = 0.9,
                              thinking_threshold: Optional[float] = None,
                              verbose: bool = False) -> Tuple[List[int], Dict]:
        """
        Generate text with latent thinking
        
        Args:
            context_ids: Input token IDs
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            thinking_threshold: Override default thinking threshold
            verbose: Print thinking details
        
        Returns:
            generated_ids: Complete token sequence
            generation_info: Statistics about thinking during generation
        """
        self.eval()
        threshold = thinking_threshold or self.thinking_threshold
        
        if verbose:
            print(f"[LatentThinking] Starting generation with {len(context_ids)} context tokens")
        
        with torch.inference_mode():
            # Process context
            if verbose:
                print(f"[LatentThinking] Processing context...")
            context_tensor = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(device)
            _, carry_states, kv_cache, _ = self.base_model(
                x=context_tensor,
                carry_states=None,
                is_training=False,
                past_key_values=None,
                use_cache=True,
                abs_pos_offset=0
            )
            
            if verbose:
                print(f"[LatentThinking] Context processed, starting token generation...")
            
            generated_ids = context_ids.copy()
            abs_pos_offset = len(context_ids)
            
            # Track thinking statistics
            total_thinking_steps = 0
            thinking_events = []
            
            # Generation loop
            for step in range(max_new_tokens):
                last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
                
                # Generate with thinking
                logits, carry_states, kv_cache, trace = self.forward_with_thinking(
                    x=last_token,
                    carry_states=carry_states,
                    past_key_values=kv_cache,
                    abs_pos_offset=abs_pos_offset,
                    enable_thinking=True,
                    verbose=verbose
                )
                
                abs_pos_offset += 1 + trace['thinking_depth']
                total_thinking_steps += trace['thinking_depth']
                
                if trace['thinking_depth'] > 0:
                    thinking_events.append({
                        'step': step,
                        'depth': trace['thinking_depth'],
                        'confidence_gain': trace['confidence_gain']
                    })
                
                # Sample next token
                next_token_logits = logits[0, -1].float()
                
                # Apply sampling
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
                
                if verbose and (step < 5 or step % 50 == 0):
                    print(f"[LatentThinking] Step {step+1}: token {next_token}, depth {trace['thinking_depth']}")
                
                # Stop conditions
                if next_token == self.tokenizer.tokenizer.eot_token:
                    if verbose:
                        print(f"[LatentThinking] EOT token reached at step {step+1}")
                    break
            
            if verbose:
                print(f"[LatentThinking] Generation complete: {len(generated_ids) - len(context_ids)} tokens")
            
            generation_info = {
                'tokens_generated': len(generated_ids) - len(context_ids),
                'total_thinking_steps': total_thinking_steps,
                'thinking_events': thinking_events,
                'avg_thinking_depth': total_thinking_steps / max(1, len(thinking_events)) if thinking_events else 0
            }
            
            return generated_ids, generation_info
