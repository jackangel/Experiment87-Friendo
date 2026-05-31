"""
Meta-Cognition Pattern Memory

Stores and retrieves learned reasoning patterns that can be
transferred across domains. This enables the model to explicitly
recognize and apply abstract reasoning strategies.

Key Concepts:
- Reasoning Patterns: Extracted carry state trajectories from successful solutions
- Cross-Domain Transfer: Patterns learned in one domain applied to another
- Pattern Similarity: Embedding-based retrieval of relevant patterns
- Success Tracking: Monitor which patterns work in which contexts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import uuid
import time


@dataclass
class ReasoningPattern:
    """
    A learned reasoning pattern extracted from model behavior
    
    Patterns are carry state trajectories that represent abstract
    reasoning skills (e.g., modus ponens, chain rule, analogy)
    """
    
    id: str
    name: str  # e.g., "modus_ponens", "chain_rule", "analogical_reasoning"
    domain: str  # e.g., "logic", "math", "language", "code"
    
    # Pattern signature: sequence of carry states representing the reasoning process
    carry_trajectory: List[torch.Tensor]  # Carry states through time
    
    # Pattern embedding for similarity-based retrieval
    pattern_embedding: torch.Tensor
    
    # Metadata
    examples: List[str] = field(default_factory=list)  # Example problems
    success_rate: float = 1.0  # Success rate when applied
    applicability_score: float = 0.5  # How broadly applicable (0-1)
    creation_time: float = field(default_factory=time.time)
    
    # Domain transfer tracking
    source_domains: List[str] = field(default_factory=list)  # Where learned
    transferred_domains: List[str] = field(default_factory=list)  # Where applied
    transfer_success_count: int = 0
    transfer_attempt_count: int = 0
    
    def to_device(self, device: str):
        """Move pattern to specified device"""
        return ReasoningPattern(
            id=self.id,
            name=self.name,
            domain=self.domain,
            carry_trajectory=[c.to(device) for c in self.carry_trajectory],
            pattern_embedding=self.pattern_embedding.to(device),
            examples=self.examples,
            success_rate=self.success_rate,
            applicability_score=self.applicability_score,
            creation_time=self.creation_time,
            source_domains=self.source_domains,
            transferred_domains=self.transferred_domains,
            transfer_success_count=self.transfer_success_count,
            transfer_attempt_count=self.transfer_attempt_count
        )
    
    def to_cpu(self):
        """Move pattern to CPU for storage"""
        return self.to_device('cpu')


class PatternMemoryBank:
    """
    Stores and retrieves learned reasoning patterns
    
    Unlike conversation memory (content-based), this stores
    abstract reasoning patterns that transfer across domains.
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 max_patterns: int = 500,
                 trajectory_length: int = 5):
        """
        Args:
            device: Default device for operations
            max_patterns: Maximum number of patterns to store
            trajectory_length: Number of carry states to store per pattern
        """
        self.device = device
        self.max_patterns = max_patterns
        self.trajectory_length = trajectory_length
        
        # Pattern storage
        self.patterns: Dict[str, ReasoningPattern] = {}
        
        # Index patterns by domain for fast retrieval
        self.domain_index: Dict[str, List[str]] = {}
        
        # Track pattern usage statistics
        self.retrieval_count: Dict[str, int] = {}
        self.application_count: Dict[str, int] = {}
        
        print(f"[PatternMemoryBank] Initialized with max_patterns={max_patterns}, "
              f"trajectory_length={trajectory_length}")
    
    def extract_pattern(self,
                       carry_trajectory: List[torch.Tensor],
                       problem_text: str,
                       domain: str,
                       pattern_name: Optional[str] = None,
                       success: bool = True) -> str:
        """
        Extract a reasoning pattern from a solution trajectory
        
        Args:
            carry_trajectory: Sequence of carry states during reasoning
            problem_text: The problem that was solved
            domain: Domain of the problem (logic, math, code, etc.)
            pattern_name: Optional name for the pattern
            success: Whether this was a successful solution
        
        Returns:
            pattern_id: ID of extracted pattern
        """
        if not success:
            return None  # Only extract from successful solutions
        
        # Truncate trajectory to max length
        if len(carry_trajectory) > self.trajectory_length:
            # Sample evenly across trajectory
            indices = torch.linspace(0, len(carry_trajectory)-1, self.trajectory_length).long()
            carry_trajectory = [carry_trajectory[i] for i in indices]
        
        # Create pattern embedding (average of trajectory)
        pattern_embedding = torch.stack(carry_trajectory).mean(dim=0)
        
        # Generate pattern ID
        pattern_id = str(uuid.uuid4())
        if pattern_name is None:
            pattern_name = f"{domain}_pattern_{len(self.patterns)}"
        
        # Create pattern
        pattern = ReasoningPattern(
            id=pattern_id,
            name=pattern_name,
            domain=domain,
            carry_trajectory=[c.detach().cpu().clone() for c in carry_trajectory],
            pattern_embedding=pattern_embedding.detach().cpu().clone(),
            examples=[problem_text],
            success_rate=1.0,
            applicability_score=0.5,  # Initial neutral score
            source_domains=[domain],
            transferred_domains=[]
        )
        
        # Store pattern
        self.patterns[pattern_id] = pattern
        
        # Update domain index
        if domain not in self.domain_index:
            self.domain_index[domain] = []
        self.domain_index[domain].append(pattern_id)
        
        # Initialize statistics
        self.retrieval_count[pattern_id] = 0
        self.application_count[pattern_id] = 0
        
        # Evict old patterns if at capacity
        if len(self.patterns) > self.max_patterns:
            self._evict_patterns()
        
        print(f"[PatternMemory] Extracted pattern '{pattern_name}' from domain '{domain}' "
              f"(trajectory_len={len(carry_trajectory)})")
        
        return pattern_id
    
    def find_similar_patterns(self,
                             query_trajectory: List[torch.Tensor],
                             top_k: int = 5,
                             cross_domain: bool = True,
                             source_domain: Optional[str] = None,
                             min_similarity: float = 0.3) -> List[Tuple[ReasoningPattern, float]]:
        """
        Find patterns similar to a given carry trajectory
        
        Args:
            query_trajectory: Current reasoning trajectory
            top_k: Number of patterns to return
            cross_domain: If True, search across all domains
            source_domain: If specified, prioritize patterns from this domain
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of (pattern, similarity_score) tuples
        """
        if len(self.patterns) == 0:
            return []
        
        # Compute query embedding
        query_embedding = torch.stack(query_trajectory).mean(dim=0).to(self.device)
        
        # Compute similarities
        similarities = []
        for pattern_id, pattern in self.patterns.items():
            pattern_emb = pattern.pattern_embedding.to(self.device)
            
            # Cosine similarity
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                pattern_emb.unsqueeze(0)
            ).item()
            
            if sim < min_similarity:
                continue
            
            # Weight by pattern quality
            quality_weight = (
                pattern.applicability_score * 0.4 +
                pattern.success_rate * 0.4 +
                min(pattern.transfer_success_count / max(1, pattern.transfer_attempt_count), 1.0) * 0.2
            )
            
            weighted_sim = sim * quality_weight
            
            # Boost same-domain patterns if specified
            if source_domain and pattern.domain == source_domain:
                weighted_sim *= 1.2
            
            similarities.append((pattern, weighted_sim, sim))
            
            # Track retrieval
            self.retrieval_count[pattern_id] += 1
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(p, raw_sim) for p, weighted_sim, raw_sim in similarities[:top_k]]
    
    def apply_pattern(self,
                     pattern: ReasoningPattern,
                     current_carry: List[torch.Tensor],
                     fusion_weight: float = 0.3) -> List[torch.Tensor]:
        """
        Apply a pattern to current reasoning state
        
        Blends pattern's carry trajectory with current state to
        inject the reasoning pattern into the model's processing.
        
        Args:
            pattern: Pattern to apply
            current_carry: Current carry states
            fusion_weight: How much pattern influence (0-1)
        
        Returns:
            Modified carry states with pattern applied
        """
        # Track application
        self.application_count[pattern.id] += 1
        
        # Apply pattern states layer by layer
        modified_carry = []
        for layer_idx, current in enumerate(current_carry):
            if current is None:
                modified_carry.append(None)
                continue
            
            # Use corresponding pattern state if available
            if layer_idx < len(pattern.carry_trajectory):
                pattern_state = pattern.carry_trajectory[layer_idx].to(self.device)
                
                # Ensure dimensions match (handle batch dimension)
                if current.dim() == 2 and pattern_state.dim() == 1:
                    pattern_state = pattern_state.unsqueeze(0).expand_as(current)
                elif current.dim() == 1 and pattern_state.dim() == 2:
                    pattern_state = pattern_state[0]
                
                # Blend current state with pattern
                fused = (1 - fusion_weight) * current + fusion_weight * pattern_state
                modified_carry.append(fused.detach())
            else:
                modified_carry.append(current)
        
        return modified_carry
    
    def record_transfer(self, 
                       pattern_id: str,
                       target_domain: str,
                       success: bool):
        """
        Record that a pattern was transferred to a new domain
        
        Updates pattern's success rate and applicability score.
        
        Args:
            pattern_id: ID of pattern that was applied
            target_domain: Domain where pattern was applied
            success: Whether the application was successful
        """
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        
        # Update transfer tracking
        pattern.transfer_attempt_count += 1
        if success:
            pattern.transfer_success_count += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        new_success = 1.0 if success else 0.0
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * new_success
        
        # Track domain transfer
        if target_domain not in pattern.transferred_domains:
            pattern.transferred_domains.append(target_domain)
        
        # Update applicability (more domains → more applicable)
        unique_domains = len(set(pattern.source_domains + pattern.transferred_domains))
        pattern.applicability_score = min(1.0, unique_domains / 5.0)  # Max at 5 domains
        
        status = "SUCCESS" if success else "FAILED"
        print(f"[PatternMemory] Pattern '{pattern.name}' transfer "
              f"({pattern.domain} → {target_domain}): {status} "
              f"(success_rate={pattern.success_rate:.2f}, "
              f"applicability={pattern.applicability_score:.2f})")
    
    def get_pattern(self, pattern_id: str) -> Optional[ReasoningPattern]:
        """Retrieve a pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_domain(self, domain: str) -> List[ReasoningPattern]:
        """Get all patterns from a specific domain"""
        if domain not in self.domain_index:
            return []
        return [self.patterns[pid] for pid in self.domain_index[domain]]
    
    def get_stats(self) -> Dict:
        """Get statistics about the pattern memory bank"""
        if len(self.patterns) == 0:
            return {
                'num_patterns': 0,
                'avg_success_rate': 0.0,
                'avg_applicability': 0.0,
                'total_transfers': 0
            }
        
        total_transfers = sum(p.transfer_attempt_count for p in self.patterns.values())
        successful_transfers = sum(p.transfer_success_count for p in self.patterns.values())
        
        return {
            'num_patterns': len(self.patterns),
            'domains': list(self.domain_index.keys()),
            'avg_success_rate': sum(p.success_rate for p in self.patterns.values()) / len(self.patterns),
            'avg_applicability': sum(p.applicability_score for p in self.patterns.values()) / len(self.patterns),
            'total_transfers': total_transfers,
            'successful_transfers': successful_transfers,
            'transfer_success_rate': successful_transfers / max(1, total_transfers)
        }
    
    def _evict_patterns(self):
        """Evict low-quality patterns when at capacity"""
        if len(self.patterns) <= self.max_patterns:
            return
        
        # Score patterns by quality
        scored_patterns = []
        for pid, pattern in self.patterns.items():
            # Quality score: success rate, applicability, and usage
            usage_score = (self.retrieval_count.get(pid, 0) + self.application_count.get(pid, 0)) / 100.0
            quality = (
                pattern.success_rate * 0.4 +
                pattern.applicability_score * 0.3 +
                min(usage_score, 1.0) * 0.3
            )
            scored_patterns.append((pid, quality))
        
        # Sort by quality
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top patterns
        num_to_keep = int(self.max_patterns * 0.9)
        patterns_to_keep = set(pid for pid, _ in scored_patterns[:num_to_keep])
        
        # Remove low-quality patterns
        for pid in list(self.patterns.keys()):
            if pid not in patterns_to_keep:
                pattern = self.patterns[pid]
                del self.patterns[pid]
                if pattern.domain in self.domain_index:
                    self.domain_index[pattern.domain].remove(pid)
                if pid in self.retrieval_count:
                    del self.retrieval_count[pid]
                if pid in self.application_count:
                    del self.application_count[pid]
        
        print(f"[PatternMemory] Evicted {len(scored_patterns) - num_to_keep} low-quality patterns")
    
    def save_to_disk(self, filepath: str):
        """Save pattern bank to disk"""
        # Move all patterns to CPU
        cpu_patterns = {pid: p.to_cpu() for pid, p in self.patterns.items()}
        
        save_dict = {
            'patterns': cpu_patterns,
            'domain_index': self.domain_index,
            'retrieval_count': self.retrieval_count,
            'application_count': self.application_count,
            'max_patterns': self.max_patterns,
            'trajectory_length': self.trajectory_length
        }
        
        torch.save(save_dict, filepath)
        print(f"[PatternMemory] Saved {len(self.patterns)} patterns to {filepath}")
    
    def load_from_disk(self, filepath: str):
        """Load pattern bank from disk"""
        save_dict = torch.load(filepath, map_location='cpu')
        
        self.patterns = save_dict['patterns']
        self.domain_index = save_dict['domain_index']
        self.retrieval_count = save_dict['retrieval_count']
        self.application_count = save_dict['application_count']
        self.max_patterns = save_dict['max_patterns']
        self.trajectory_length = save_dict['trajectory_length']
        
        print(f"[PatternMemory] Loaded {len(self.patterns)} patterns from {filepath}")


class PatternDetector(nn.Module):
    """
    Detects which reasoning patterns are currently active
    
    Analyzes carry state trajectory to identify abstract patterns.
    """
    
    def __init__(self, dim: int, num_pattern_types: int = 20):
        """
        Args:
            dim: Model hidden dimension
            num_pattern_types: Number of pattern types to detect
        """
        super().__init__()
        self.num_pattern_types = num_pattern_types
        
        # Pattern classification head (multi-label)
        self.pattern_classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_pattern_types),
            nn.Sigmoid()  # Multi-label: multiple patterns can be active
        )
        
        # Learnable pattern prototypes
        # These represent abstract reasoning patterns
        self.pattern_prototypes = nn.Parameter(
            torch.randn(num_pattern_types, dim) * 0.02
        )
        
        # Pattern type names (for interpretability)
        self.pattern_names = [
            'deduction', 'induction', 'analogy', 'abstraction',
            'composition', 'decomposition', 'transformation', 'optimization',
            'search', 'backtracking', 'recursion', 'iteration',
            'causality', 'correlation', 'classification', 'generation',
            'verification', 'correction', 'refinement', 'synthesis'
        ]
    
    def forward(self, carry_states: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect active patterns from carry states
        
        Args:
            carry_states: Carry states from all layers
        
        Returns:
            pattern_activations: (num_patterns,) - active pattern scores
            pattern_embeddings: (num_patterns, dim) - pattern representations
        """
        # Use last layer's carry state
        last_carry = carry_states[-1]  # (B, D)
        
        # Handle batch dimension
        if last_carry.dim() == 2:
            last_carry = last_carry.mean(dim=0)  # Average over batch
        
        # Classify active patterns
        pattern_activations = self.pattern_classifier(last_carry)  # (num_patterns,)
        
        return pattern_activations, self.pattern_prototypes
    
    def get_active_pattern_names(self, pattern_activations: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """
        Get names of active patterns above threshold
        
        Args:
            pattern_activations: Pattern activation scores
            threshold: Activation threshold
        
        Returns:
            List of active pattern names
        """
        active_indices = (pattern_activations > threshold).nonzero(as_tuple=True)[0]
        return [self.pattern_names[i] for i in active_indices.tolist()]
