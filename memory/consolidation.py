"""
Memory Consolidation Engine

Implements biologically-inspired memory consolidation:
- Episodic (short-term) → Semantic (long-term) transition
- Importance-based retention (like cognitive forgetting gates)
- Memory interference reduction through merging
- Time-based decay

Inspired by human memory consolidation during sleep and rest periods.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import time
import numpy as np
from .core import SSMStateMemoryBank, MemoryEntry


class MemoryConsolidation:
    """
    Manages the transition of memories from episodic to semantic storage
    
    Episodic Buffer: Recent memories (fast access, limited capacity)
    Semantic Storage: Consolidated long-term memories (slower access, larger capacity)
    
    Consolidation happens when:
    1. Episodic buffer reaches capacity
    2. Memory importance exceeds threshold
    3. Memory is frequently accessed
    4. Periodic consolidation (e.g., during idle time)
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 episodic_capacity: int = 50,
                 consolidation_threshold: float = 0.7,
                 decay_rate: float = 0.95,
                 access_threshold: int = 3,
                 merge_similarity_threshold: float = 0.9):
        """
        Args:
            device: Device for operations
            episodic_capacity: Max number of episodic memories before consolidation
            consolidation_threshold: Min importance score to promote to semantic
            decay_rate: Time decay factor (per hour)
            access_threshold: Min access count to promote to semantic
            merge_similarity_threshold: Similarity threshold for merging similar memories
        """
        self.device = device
        self.episodic_capacity = episodic_capacity
        self.consolidation_threshold = consolidation_threshold
        self.decay_rate = decay_rate
        self.access_threshold = access_threshold
        self.merge_similarity_threshold = merge_similarity_threshold
        
        # Episodic buffer (short-term, recent memories)
        self.episodic_buffer: List[MemoryEntry] = []
        
        # Semantic storage (long-term, consolidated memories)
        self.semantic_memory = SSMStateMemoryBank(
            max_memories=1000,
            saliency_percentile=0.8,
            device=device
        )
        
        print(f"[MemoryConsolidation] Initialized with episodic_capacity={episodic_capacity}, "
              f"consolidation_threshold={consolidation_threshold}, decay_rate={decay_rate}")
    
    def add_episodic_memory(self, memory: MemoryEntry) -> str:
        """
        Add a new memory to the episodic buffer
        
        Args:
            memory: Memory entry to add
        
        Returns:
            memory_id: ID of the added memory
        """
        self.episodic_buffer.append(memory)
        
        # Trigger consolidation if buffer is full
        if len(self.episodic_buffer) >= self.episodic_capacity:
            print(f"[MemoryConsolidation] Episodic buffer full ({len(self.episodic_buffer)}), "
                  f"triggering consolidation...")
            self.consolidate()
        
        return memory.id
    
    def consolidate(self, force: bool = False):
        """
        Consolidate episodic memories to semantic storage
        
        Process:
        1. Apply time-based decay to importance scores
        2. Identify memories worthy of semantic promotion
        3. Merge similar memories to reduce interference
        4. Clear episodic buffer of consolidated/forgotten memories
        
        Args:
            force: If True, consolidate all memories regardless of thresholds
        """
        if len(self.episodic_buffer) == 0:
            print("[MemoryConsolidation] No episodic memories to consolidate")
            return
        
        current_time = time.time()
        promoted_count = 0
        forgotten_count = 0
        merged_count = 0
        
        print(f"\n[MemoryConsolidation] Starting consolidation of {len(self.episodic_buffer)} "
              f"episodic memories...")
        
        # Process each episodic memory
        for memory in self.episodic_buffer:
            # Apply time-based decay
            hours_elapsed = (current_time - memory.timestamp) / 3600.0
            time_decay = self.decay_rate ** hours_elapsed
            decayed_importance = memory.importance_score * time_decay
            
            # Decision criteria for promotion
            should_promote = (
                force or
                decayed_importance >= self.consolidation_threshold or
                memory.access_count >= self.access_threshold
            )
            
            if should_promote:
                # Check for similar existing semantic memories
                similar_memories = self.semantic_memory.search_by_embedding(
                    memory.embedding,
                    top_k=3,
                    min_similarity=self.merge_similarity_threshold
                )
                
                if similar_memories and not force:
                    # Merge with most similar memory
                    most_similar, similarity = similar_memories[0]
                    self._merge_memories(memory, most_similar)
                    merged_count += 1
                else:
                    # Promote as new semantic memory
                    self.semantic_memory.store_memory(
                        carry_states=memory.carry_states,
                        past_kv=memory.kv_cache,
                        tokens=memory.tokens,
                        metadata={
                            **memory.metadata,
                            'promoted_from_episodic': True,
                            'original_timestamp': memory.timestamp,
                            'promotion_time': current_time,
                            'decayed_importance': decayed_importance,
                            'access_count': memory.access_count
                        }
                    )
                    promoted_count += 1
            else:
                # Memory doesn't meet promotion criteria (forgotten)
                forgotten_count += 1
        
        # Clear episodic buffer
        self.episodic_buffer.clear()
        
        stats = self.semantic_memory.get_memory_stats()
        print(f"[MemoryConsolidation] Consolidation complete:")
        print(f"  - Promoted: {promoted_count}")
        print(f"  - Merged: {merged_count}")
        print(f"  - Forgotten: {forgotten_count}")
        print(f"  - Total semantic memories: {stats['num_memories']}")
        print(f"  - Avg importance: {stats['avg_importance']:.3f}\n")
    
    def _merge_memories(self, new_memory: MemoryEntry, existing_memory: MemoryEntry):
        """
        Merge a new memory with an existing similar memory (constructive interference)
        
        Strategy:
        - Weighted blend of carry states based on importance
        - Combine saliency maps (union of high-saliency tokens)
        - Update metadata (merge entities, topics, etc.)
        - Increase importance score
        
        This mimics how repeated exposure to similar information strengthens memory.
        """
        # Calculate blend weight based on relative importance
        total_importance = new_memory.importance_score + existing_memory.importance_score
        new_weight = new_memory.importance_score / total_importance
        existing_weight = existing_memory.importance_score / total_importance
        
        # Blend carry states
        merged_carry_states = []
        for new_carry, exist_carry in zip(new_memory.carry_states, existing_memory.carry_states):
            if new_carry is None or exist_carry is None:
                merged_carry_states.append(None)
                continue
            
            # Ensure same device
            new_carry = new_carry.to(self.device)
            exist_carry = exist_carry.to(self.device)
            
            # Weighted blend
            merged = new_weight * new_carry + existing_weight * exist_carry
            merged_carry_states.append(merged.detach())
        
        # Update memory embedding (blend)
        new_emb = new_memory.embedding.to(self.device)
        exist_emb = existing_memory.embedding.to(self.device)
        merged_embedding = F.normalize(
            new_weight * new_emb + existing_weight * exist_emb,
            dim=0
        )
        
        # Merge metadata
        merged_metadata = {**existing_memory.metadata, **new_memory.metadata}
        if 'merge_count' in merged_metadata:
            merged_metadata['merge_count'] += 1
        else:
            merged_metadata['merge_count'] = 1
        
        # Update existing memory with merged data
        existing_memory.carry_states = merged_carry_states
        existing_memory.embedding = merged_embedding.detach()
        existing_memory.importance_score = min(1.0, total_importance * 1.1)  # Boost importance
        existing_memory.access_count += new_memory.access_count
        existing_memory.metadata = merged_metadata
        
        print(f"[MemoryConsolidation] Merged memory {new_memory.id[:8]} into {existing_memory.id[:8]}, "
              f"new importance: {existing_memory.importance_score:.3f}")
    
    def retrieve_from_episodic(self, query_embedding: torch.Tensor, top_k: int = 2) -> List[Tuple[MemoryEntry, float]]:
        """
        Search episodic buffer by embedding similarity
        
        Args:
            query_embedding: Query vector (D,)
            top_k: Number of results to return
        
        Returns:
            List of (memory, similarity_score) tuples
        """
        if len(self.episodic_buffer) == 0:
            return []
        
        query_embedding = query_embedding.to(self.device)
        
        # Compute similarities
        similarities = []
        for memory in self.episodic_buffer:
            mem_emb = memory.embedding.to(self.device)
            
            # Cosine similarity
            query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
            mem_norm = F.normalize(mem_emb.unsqueeze(0), dim=-1)
            sim = (query_norm * mem_norm).sum().item()
            
            similarities.append((memory, sim))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def retrieve_from_semantic(self, query_embedding: torch.Tensor, top_k: int = 3, min_similarity: float = 0.5) -> List[Tuple[MemoryEntry, float]]:
        """
        Search semantic memory by embedding similarity
        
        Args:
            query_embedding: Query vector (D,)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (memory, similarity_score) tuples
        """
        return self.semantic_memory.search_by_embedding(
            query_embedding,
            top_k=top_k,
            min_similarity=min_similarity
        )
    
    def retrieve_combined(self, 
                         query_embedding: torch.Tensor,
                         episodic_k: int = 2,
                         semantic_k: int = 3,
                         total_k: int = 5) -> List[Tuple[MemoryEntry, float, str]]:
        """
        Retrieve from both episodic and semantic memory, then combine and rank
        
        Returns:
            List of (memory, similarity, source) tuples where source is 'episodic' or 'semantic'
        """
        # Retrieve from both stores
        episodic_results = self.retrieve_from_episodic(query_embedding, top_k=episodic_k)
        semantic_results = self.retrieve_from_semantic(query_embedding, top_k=semantic_k)
        
        # Tag with source
        combined = []
        for memory, sim in episodic_results:
            combined.append((memory, sim, 'episodic'))
        for memory, sim in semantic_results:
            combined.append((memory, sim, 'semantic'))
        
        # Sort by similarity (episodic memories get slight recency boost)
        def score_key(item):
            memory, sim, source = item
            recency_boost = 0.05 if source == 'episodic' else 0.0
            return sim + recency_boost
        
        combined.sort(key=score_key, reverse=True)
        
        return combined[:total_k]
    
    def get_episodic_count(self) -> int:
        """Get number of episodic memories"""
        return len(self.episodic_buffer)
    
    def get_semantic_count(self) -> int:
        """Get number of semantic memories"""
        return len(self.semantic_memory.memories)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        semantic_stats = self.semantic_memory.get_memory_stats()
        
        episodic_importance = (
            sum(m.importance_score for m in self.episodic_buffer) / len(self.episodic_buffer)
            if self.episodic_buffer else 0.0
        )
        
        return {
            'episodic_count': len(self.episodic_buffer),
            'semantic_count': semantic_stats['num_memories'],
            'total_count': len(self.episodic_buffer) + semantic_stats['num_memories'],
            'episodic_avg_importance': episodic_importance,
            'semantic_avg_importance': semantic_stats['avg_importance'],
            'semantic_avg_access': semantic_stats['avg_access_count'],
            'total_tokens': semantic_stats['total_tokens']
        }
    
    def save_to_disk(self, filepath: str):
        """Save both episodic and semantic memories to disk"""
        # Move episodic to CPU
        cpu_episodic = [m.to_cpu() for m in self.episodic_buffer]
        
        save_dict = {
            'episodic_buffer': cpu_episodic,
            'consolidation_threshold': self.consolidation_threshold,
            'decay_rate': self.decay_rate,
            'access_threshold': self.access_threshold
        }
        
        # Save episodic buffer
        torch.save(save_dict, filepath.replace('.pt', '_episodic.pt'))
        
        # Save semantic memory
        self.semantic_memory.save_to_disk(filepath.replace('.pt', '_semantic.pt'))
        
        print(f"[MemoryConsolidation] Saved {len(self.episodic_buffer)} episodic and "
              f"{len(self.semantic_memory.memories)} semantic memories")
    
    def load_from_disk(self, filepath: str):
        """Load both episodic and semantic memories from disk"""
        # Load episodic buffer
        episodic_dict = torch.load(filepath.replace('.pt', '_episodic.pt'), map_location='cpu')
        self.episodic_buffer = episodic_dict['episodic_buffer']
        self.consolidation_threshold = episodic_dict['consolidation_threshold']
        self.decay_rate = episodic_dict['decay_rate']
        self.access_threshold = episodic_dict['access_threshold']
        
        # Load semantic memory
        self.semantic_memory.load_from_disk(filepath.replace('.pt', '_semantic.pt'))
        
        print(f"[MemoryConsolidation] Loaded {len(self.episodic_buffer)} episodic and "
              f"{len(self.semantic_memory.memories)} semantic memories")
