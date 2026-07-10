"""
Core SSM State Memory Bank

Stores compressed conversation/document states using:
- SSM carry states (highly compressed sequence representation)
- High-saliency KV cache entries (only important tokens)
- Memory embeddings for retrieval
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
import uuid


@dataclass
class MemoryEntry:
    """A single memory entry with compressed state and metadata"""
    
    id: str
    carry_states: List[torch.Tensor]  # SSM hidden states per layer
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]  # Compressed KV cache
    saliency_map: torch.Tensor  # Saliency scores for stored tokens
    token_indices: torch.Tensor  # Which token positions were kept
    tokens: List[int]  # Original token sequence (for reference)
    embedding: torch.Tensor  # Memory embedding for retrieval (derived from final carry state)
    timestamp: float  # When was this memory created
    access_count: int  # How often has this been retrieved
    importance_score: float  # Overall importance (based on saliency)
    metadata: Dict[str, Any]  # Additional metadata (topic, entities, summary, etc.)
    
    def to_device(self, device):
        """Move memory entry to specified device"""
        return MemoryEntry(
            id=self.id,
            carry_states=[c.to(device) if c is not None else None for c in self.carry_states],
            kv_cache=[(k.to(device), v.to(device), s.to(device), kr.to(device)) 
                     for k, v, s, kr in self.kv_cache],
            saliency_map=self.saliency_map.to(device),
            token_indices=self.token_indices.to(device),
            tokens=self.tokens,
            embedding=self.embedding.to(device),
            timestamp=self.timestamp,
            access_count=self.access_count,
            importance_score=self.importance_score,
            metadata=self.metadata
        )
    
    def to_cpu(self):
        """Move memory entry to CPU for storage"""
        return MemoryEntry(
            id=self.id,
            carry_states=[c.cpu() if c is not None else None for c in self.carry_states],
            kv_cache=[(k.cpu(), v.cpu(), s.cpu(), kr.cpu()) 
                     for k, v, s, kr in self.kv_cache],
            saliency_map=self.saliency_map.cpu(),
            token_indices=self.token_indices.cpu(),
            tokens=self.tokens,
            embedding=self.embedding.cpu(),
            timestamp=self.timestamp,
            access_count=self.access_count,
            importance_score=self.importance_score,
            metadata=self.metadata
        )


class SSMStateMemoryBank:
    """
    Memory bank that stores compressed SSM states with saliency-based filtering
    
    Key Features:
    - Stores only high-saliency tokens (top 20% by default)
    - Uses final SSM carry state as memory embedding
    - Tracks access patterns and importance scores
    - Automatic memory management (LRU eviction)
    """
    
    def __init__(self, 
                 max_memories: int = 1000,
                 saliency_percentile: float = 0.8,
                 device: str = 'cuda'):
        """
        Args:
            max_memories: Maximum number of memories to store
            saliency_percentile: Keep tokens above this percentile (0.8 = top 20%)
            device: Default device for operations
        """
        self.max_memories = max_memories
        self.saliency_percentile = saliency_percentile
        self.device = device
        self.memories: List[MemoryEntry] = []
        self.memory_index: Dict[str, int] = {}  # id -> index mapping
        
        print(f"[SSMStateMemoryBank] Initialized with max_memories={max_memories}, "
              f"saliency_percentile={saliency_percentile}")
    
    def store_memory(self,
                     carry_states: List[torch.Tensor],
                     past_kv: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                     tokens: List[int],
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new memory entry with saliency-based compression
        
        Args:
            carry_states: SSM hidden states from all layers
            past_kv: KV cache with saliency scores (k, v, saliency_scores, k_rope)
            tokens: Original token sequence
            metadata: Optional metadata (topic, entities, summary, etc.)
        
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        # Extract saliency scores (average across all layers)
        saliency_scores = []
        for k, v, s, kr in past_kv:
            saliency_scores.append(s)
        
        # Average saliency across layers and batch dimension
        avg_saliency = torch.stack(saliency_scores, dim=0).mean(dim=0)  # (B, L)
        avg_saliency = avg_saliency.mean(dim=0)  # (L,)
        
        # Filter KV cache to only high-saliency tokens
        saliency_threshold = torch.quantile(avg_saliency, self.saliency_percentile)
        high_sal_mask = avg_saliency >= saliency_threshold
        high_sal_indices = high_sal_mask.nonzero(as_tuple=True)[0]
        
        # Compress KV cache
        compressed_kv = []
        for k, v, s, kr in past_kv:
            # k, v, kr: (B, H, L, D), s: (B, L)
            k_compressed = k[:, :, high_sal_indices, :]
            v_compressed = v[:, :, high_sal_indices, :]
            s_compressed = s[:, high_sal_indices]
            kr_compressed = kr[:, :, high_sal_indices, :]
            compressed_kv.append((k_compressed, v_compressed, s_compressed, kr_compressed))
        
        # Create memory embedding from final carry state
        # Use mean across the feature dimension
        final_carry = carry_states[-1]  # Last layer's carry state (B, D)
        if final_carry.dim() == 2:
            memory_embedding = final_carry.mean(dim=0)  # (D,)
        else:
            memory_embedding = final_carry.view(-1)  # Flatten if different shape
        
        # Calculate importance score (mean saliency)
        importance_score = avg_saliency.mean().item()
        
        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            carry_states=[c.detach().clone() for c in carry_states],
            kv_cache=compressed_kv,
            saliency_map=avg_saliency[high_sal_indices].detach().clone(),
            token_indices=high_sal_indices.detach().clone(),
            tokens=tokens,
            embedding=memory_embedding.detach().clone(),
            timestamp=time.time(),
            access_count=0,
            importance_score=importance_score,
            metadata=metadata or {}
        )
        
        # Move to CPU for storage if we're approaching max capacity
        if len(self.memories) >= self.max_memories * 0.7:
            memory = memory.to_cpu()
        
        # Add to memory bank
        self.memories.append(memory)
        self.memory_index[memory_id] = len(self.memories) - 1
        
        # Evict oldest low-importance memories if at capacity
        if len(self.memories) > self.max_memories:
            self._evict_memories()
        
        return memory_id
    
    def retrieve_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by its ID"""
        if memory_id not in self.memory_index:
            return None
        
        idx = self.memory_index[memory_id]
        memory = self.memories[idx]
        
        # Update access count
        memory.access_count += 1
        
        # Move to device if needed
        if memory.embedding.device.type == 'cpu' and self.device != 'cpu':
            return memory.to_device(self.device)
        
        return memory
    
    def search_by_embedding(self, 
                           query_embedding: torch.Tensor, 
                           top_k: int = 5,
                           min_similarity: float = 0.0) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for similar memories using embedding similarity
        
        Args:
            query_embedding: Query embedding vector (D,)
            top_k: Number of top results to return
            min_similarity: Minimum cosine similarity threshold
        
        Returns:
            List of (memory, similarity_score) tuples, sorted by similarity
        """
        if len(self.memories) == 0:
            return []
        
        # Ensure query is on the same device
        query_embedding = query_embedding.to(self.device)
        
        # Compute cosine similarities
        similarities = []
        for memory in self.memories:
            mem_emb = memory.embedding.to(self.device)
            
            # Normalize embeddings
            query_norm = F.normalize(query_embedding.unsqueeze(0), dim=-1)
            mem_norm = F.normalize(mem_emb.unsqueeze(0), dim=-1)
            
            # Cosine similarity
            sim = (query_norm * mem_norm).sum().item()
            
            if sim >= min_similarity:
                similarities.append((memory, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = similarities[:top_k]
        
        # Update access counts
        for memory, _ in results:
            memory.access_count += 1
        
        return results
    
    def get_all_memories(self) -> List[MemoryEntry]:
        """Get all stored memories"""
        return self.memories.copy()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        if len(self.memories) == 0:
            return {
                'num_memories': 0,
                'avg_importance': 0.0,
                'avg_access_count': 0.0,
                'total_tokens': 0
            }
        
        return {
            'num_memories': len(self.memories),
            'avg_importance': sum(m.importance_score for m in self.memories) / len(self.memories),
            'avg_access_count': sum(m.access_count for m in self.memories) / len(self.memories),
            'total_tokens': sum(len(m.tokens) for m in self.memories),
            'oldest_timestamp': min(m.timestamp for m in self.memories),
            'newest_timestamp': max(m.timestamp for m in self.memories)
        }
    
    def _evict_memories(self):
        """
        Evict low-importance, rarely-accessed memories using a combined score
        
        Score = importance * 0.5 + (access_count / max_access) * 0.3 + recency * 0.2
        """
        if len(self.memories) <= self.max_memories:
            return
        
        # Calculate retention scores
        current_time = time.time()
        max_access = max(m.access_count for m in self.memories) or 1
        max_age = max(current_time - m.timestamp for m in self.memories) or 1
        
        scored_memories = []
        for idx, memory in enumerate(self.memories):
            recency = 1.0 - ((current_time - memory.timestamp) / max_age)
            access_score = memory.access_count / max_access
            
            # Combined score: importance, access frequency, and recency
            retention_score = (
                memory.importance_score * 0.5 +
                access_score * 0.3 +
                recency * 0.2
            )
            
            scored_memories.append((idx, memory, retention_score))
        
        # Sort by retention score (descending)
        scored_memories.sort(key=lambda x: x[2], reverse=True)
        
        # Keep top memories
        num_to_keep = int(self.max_memories * 0.9)  # Keep 90% after eviction
        memories_to_keep = scored_memories[:num_to_keep]
        
        # Update memory list and index
        self.memories = [m for _, m, _ in memories_to_keep]
        self.memory_index = {m.id: idx for idx, m in enumerate(self.memories)}
        
        print(f"[SSMStateMemoryBank] Evicted {len(scored_memories) - num_to_keep} memories, "
              f"kept {num_to_keep} highest-scoring memories")
    
    def clear(self):
        """Clear all memories"""
        self.memories.clear()
        self.memory_index.clear()
        print("[SSMStateMemoryBank] Cleared all memories")
    
    def save_to_disk(self, filepath: str):
        """Save memory bank to disk"""
        # Move all memories to CPU before saving
        cpu_memories = [m.to_cpu() for m in self.memories]
        
        save_dict = {
            'memories': cpu_memories,
            'max_memories': self.max_memories,
            'saliency_percentile': self.saliency_percentile
        }
        
        torch.save(save_dict, filepath)
        print(f"[SSMStateMemoryBank] Saved {len(self.memories)} memories to {filepath}")
    
    def load_from_disk(self, filepath: str):
        """Load memory bank from disk"""
        save_dict = torch.load(filepath, map_location='cpu')
        
        self.memories = save_dict['memories']
        self.max_memories = save_dict['max_memories']
        self.saliency_percentile = save_dict['saliency_percentile']
        
        # Rebuild index
        self.memory_index = {m.id: idx for idx, m in enumerate(self.memories)}
        
        print(f"[SSMStateMemoryBank] Loaded {len(self.memories)} memories from {filepath}")
