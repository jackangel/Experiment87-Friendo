"""
Meta-Cognitive Controller

High-level controller that manages pattern selection, application,
and cross-domain transfer. This is the "executive function" that makes
decisions about which patterns to activate and when.

Key Responsibilities:
- Domain Detection: Identify current reasoning domain
- Pattern Selection: Choose relevant patterns from memory
- Transfer Decision: Decide when to apply cross-domain patterns
- Pattern Application: Inject patterns into reasoning process
- Success Monitoring: Track which patterns work
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .patterns import PatternMemoryBank, PatternDetector, ReasoningPattern


class MetaCognitiveController(nn.Module):
    """
    High-level controller for meta-cognitive reasoning
    
    Manages the selection and application of reasoning patterns,
    enabling cross-domain knowledge transfer.
    """
    
    def __init__(self,
                 base_model,
                 pattern_bank: PatternMemoryBank,
                 num_domains: int = 10):
        """
        Args:
            base_model: Base SSMTransformer model
            pattern_bank: PatternMemoryBank for storing patterns
            num_domains: Number of reasoning domains to support
        """
        super().__init__()
        self.base_model = base_model
        self.pattern_bank = pattern_bank
        self.num_domains = num_domains
        
        # Pattern detector
        self.pattern_detector = PatternDetector(
            base_model.dim,
            num_pattern_types=20
        )
        
        # Domain classifier
        # Identifies which domain the current problem belongs to
        self.domain_classifier = nn.Sequential(
            nn.Linear(base_model.dim, base_model.dim // 2),
            nn.LayerNorm(base_model.dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_model.dim // 2, num_domains),
            nn.Softmax(dim=-1)
        )
        
        # Domain names for interpretability
        self.domain_names = [
            'logic', 'mathematics', 'language', 'code', 'reasoning',
            'analogy', 'causality', 'planning', 'creativity', 'meta'
        ]
        
        # Pattern selection policy
        # Decides whether to apply a given pattern in current context
        self.pattern_selector = nn.Sequential(
            nn.Linear(base_model.dim * 2 + num_domains, base_model.dim // 2),
            nn.LayerNorm(base_model.dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_model.dim // 2, 1),
            nn.Sigmoid()  # Probability of applying pattern
        )
        
        # Transfer confidence estimator
        # Estimates likelihood of successful cross-domain transfer
        self.transfer_confidence = nn.Sequential(
            nn.Linear(base_model.dim * 2 + num_domains * 2, base_model.dim // 4),
            nn.LayerNorm(base_model.dim // 4),
            nn.GELU(),
            nn.Linear(base_model.dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Move meta-cognitive components to same device as base model
        model_device = next(base_model.parameters()).device
        self.pattern_detector = self.pattern_detector.to(model_device)
        self.domain_classifier = self.domain_classifier.to(model_device)
        self.pattern_selector = self.pattern_selector.to(model_device)
        self.transfer_confidence = self.transfer_confidence.to(model_device)
        
        print(f"[MetaCognitive] Initialized with {num_domains} domains, "
              f"{len(pattern_bank.patterns)} patterns")
    
    def forward(self, x, carry_states=None, past_key_values=None,
                use_cache=False, abs_pos_offset=0, is_training=True):
        """
        Standard forward pass (delegates to base model)
        
        Meta-cognition is only active during generation, not training.
        """
        return self.base_model(
            x=x,
            carry_states=carry_states,
            is_training=is_training,
            past_key_values=past_key_values,
            use_cache=use_cache,
            abs_pos_offset=abs_pos_offset
        )
    
    def forward_with_metacognition(self,
                                   x: torch.Tensor,
                                   carry_states: Optional[List[torch.Tensor]] = None,
                                   past_key_values: Optional[List] = None,
                                   abs_pos_offset: int = 0,
                                   enable_pattern_transfer: bool = True,
                                   pattern_fusion_weight: float = 0.3,
                                   transfer_threshold: float = 0.6,
                                   verbose: bool = False) -> Tuple:
        """
        Forward pass with meta-cognitive pattern management
        
        Args:
            x: Input tokens
            carry_states: SSM carry states
            past_key_values: KV cache
            abs_pos_offset: Absolute position offset
            enable_pattern_transfer: Whether to apply patterns
            pattern_fusion_weight: How strongly to apply patterns (0-1)
            transfer_threshold: Confidence threshold for pattern application
            verbose: Print metacognitive decisions
        
        Returns:
            logits: Token predictions
            carry_states: Modified carry states (with patterns)
            past_key_values: KV cache
            meta_info: Metacognitive information
        """
        # Standard forward pass
        logits, carry_states, past_key_values, mod_loss = self.base_model(
            x=x,
            carry_states=carry_states,
            is_training=False,
            past_key_values=past_key_values,
            use_cache=True,
            abs_pos_offset=abs_pos_offset
        )
        
        if not enable_pattern_transfer or len(carry_states) == 0:
            return logits, carry_states, past_key_values, {
                'patterns_applied': [],
                'current_domain': None,
                'domain_confidence': 0.0
            }
        
        # Detect current domain
        last_carry = carry_states[-1]
        if last_carry.dim() == 2:
            last_carry_pooled = last_carry.mean(dim=0)  # Pool over batch
        else:
            last_carry_pooled = last_carry
        
        domain_probs = self.domain_classifier(last_carry_pooled)
        current_domain_idx = torch.argmax(domain_probs).item()
        current_domain = self.domain_names[current_domain_idx]
        domain_confidence = domain_probs[current_domain_idx].item()
        
        # Detect active patterns
        pattern_activations, pattern_prototypes = self.pattern_detector(carry_states)
        
        # Find similar patterns from memory
        similar_patterns = self.pattern_bank.find_similar_patterns(
            carry_states,
            top_k=3,
            cross_domain=True,
            source_domain=current_domain
        )
        
        meta_info = {
            'current_domain': current_domain,
            'domain_confidence': domain_confidence,
            'active_pattern_types': self.pattern_detector.get_active_pattern_names(pattern_activations),
            'patterns_applied': [],
            'patterns_considered': []
        }
        
        if verbose:
            print(f"\n[MetaCognition] Domain: {current_domain} (conf={domain_confidence:.3f})")
            if meta_info['active_pattern_types']:
                print(f"[MetaCognition] Active patterns: {', '.join(meta_info['active_pattern_types'])}")
        
        # Consider each similar pattern for application
        for pattern, similarity in similar_patterns:
            is_cross_domain = pattern.domain != current_domain
            
            meta_info['patterns_considered'].append({
                'pattern_name': pattern.name,
                'pattern_domain': pattern.domain,
                'similarity': similarity,
                'cross_domain': is_cross_domain
            })
            
            # Build context for selection decision
            pattern_emb = pattern.pattern_embedding.to(last_carry_pooled.device)
            selection_context = torch.cat([
                last_carry_pooled,
                pattern_emb,
                domain_probs
            ])
            
            # Decide whether to apply this pattern
            apply_prob = self.pattern_selector(selection_context).item()
            
            # For cross-domain transfer, require higher confidence
            if is_cross_domain:
                # Estimate transfer confidence
                source_domain_idx = self.domain_names.index(pattern.domain)
                source_domain_vec = F.one_hot(
                    torch.tensor(source_domain_idx),
                    num_classes=self.num_domains
                ).float().to(last_carry_pooled.device)
                
                target_domain_vec = domain_probs
                
                transfer_context = torch.cat([
                    last_carry_pooled,
                    pattern_emb,
                    source_domain_vec,
                    target_domain_vec
                ])
                
                transfer_conf = self.transfer_confidence(transfer_context).item()
                
                # Adjust application probability based on transfer confidence
                apply_prob = apply_prob * transfer_conf
                
                if verbose:
                    print(f"  [Cross-Domain] {pattern.name} ({pattern.domain} → {current_domain}): "
                          f"apply_prob={apply_prob:.3f}, transfer_conf={transfer_conf:.3f}")
            
            # Apply pattern if confidence exceeds threshold
            if apply_prob > transfer_threshold:
                # Apply pattern to carry states
                carry_states = self.pattern_bank.apply_pattern(
                    pattern,
                    carry_states,
                    fusion_weight=apply_prob * pattern_fusion_weight
                )
                
                meta_info['patterns_applied'].append({
                    'pattern_name': pattern.name,
                    'pattern_domain': pattern.domain,
                    'similarity': similarity,
                    'apply_prob': apply_prob,
                    'cross_domain': is_cross_domain,
                    'fusion_weight': apply_prob * pattern_fusion_weight
                })
                
                if verbose:
                    transfer_str = f" [TRANSFER: {pattern.domain}→{current_domain}]" if is_cross_domain else ""
                    print(f"  ✓ Applied: {pattern.name}{transfer_str} "
                          f"(conf={apply_prob:.3f}, weight={apply_prob * pattern_fusion_weight:.3f})")
        
        if verbose and not meta_info['patterns_applied']:
            print(f"[MetaCognition] No patterns applied (threshold={transfer_threshold})")
        
        return logits, carry_states, past_key_values, meta_info
    
    def extract_pattern_from_trajectory(self,
                                       carry_trajectory: List[List[torch.Tensor]],
                                       problem_text: str,
                                       domain: str,
                                       pattern_name: Optional[str] = None,
                                       success: bool = True) -> Optional[str]:
        """
        Extract and store a reasoning pattern from a solution trajectory
        
        Args:
            carry_trajectory: List of carry states at each step
            problem_text: Problem that was solved
            domain: Domain of the problem
            pattern_name: Optional name for the pattern
            success: Whether solution was successful
        
        Returns:
            pattern_id: ID of extracted pattern, or None if unsuccessful
        """
        if not success:
            return None
        
        # Flatten trajectory (take last layer from each step)
        flattened_trajectory = [states[-1] for states in carry_trajectory if states]
        
        if len(flattened_trajectory) == 0:
            return None
        
        # Extract pattern
        pattern_id = self.pattern_bank.extract_pattern(
            carry_trajectory=flattened_trajectory,
            problem_text=problem_text,
            domain=domain,
            pattern_name=pattern_name,
            success=success
        )
        
        return pattern_id
    
    def record_pattern_application(self,
                                  pattern_id: str,
                                  target_domain: str,
                                  success: bool):
        """
        Record the result of applying a pattern
        
        Args:
            pattern_id: ID of pattern that was applied
            target_domain: Domain where pattern was applied
            success: Whether application was successful
        """
        self.pattern_bank.record_transfer(pattern_id, target_domain, success)
    
    def get_metacognitive_stats(self) -> Dict:
        """Get comprehensive metacognitive statistics"""
        pattern_stats = self.pattern_bank.get_stats()
        
        return {
            'pattern_bank': pattern_stats,
            'domains': self.domain_names,
            'num_pattern_types': self.pattern_detector.num_pattern_types
        }
    
    def save_patterns(self, filepath: str):
        """Save pattern bank to disk"""
        self.pattern_bank.save_to_disk(filepath)
    
    def load_patterns(self, filepath: str):
        """Load pattern bank from disk"""
        self.pattern_bank.load_from_disk(filepath)


def generate_with_metacognition(model: MetaCognitiveController,
                                tokenizer,
                                context_ids: List[int],
                                device: str,
                                max_new_tokens: int = 256,
                                temperature: float = 0.8,
                                top_k: int = 50,
                                top_p: float = 0.9,
                                enable_patterns: bool = True,
                                pattern_fusion_weight: float = 0.3,
                                verbose: bool = False) -> Tuple[List[int], Dict]:
    """
    Generate text with meta-cognitive pattern management
    
    Args:
        model: MetaCognitiveController instance
        tokenizer: Tokenizer
        context_ids: Input token IDs
        device: Device
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        enable_patterns: Whether to use pattern transfer
        pattern_fusion_weight: Pattern influence strength
        verbose: Print metacognitive decisions
    
    Returns:
        generated_ids: Generated token sequence
        generation_info: Statistics about pattern usage
    """
    model.eval()
    
    with torch.inference_mode():
        # Process context
        context_tensor = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(device)
        _, carry_states, kv_cache, _ = model.base_model(
            x=context_tensor,
            carry_states=None,
            is_training=False,
            past_key_values=None,
            use_cache=True,
            abs_pos_offset=0
        )
        
        generated_ids = context_ids.copy()
        abs_pos_offset = len(context_ids)
        
        # Track pattern usage
        pattern_applications = []
        domain_sequence = []
        
        # Generation loop
        for step in range(max_new_tokens):
            last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
            
            # Generate with metacognition
            logits, carry_states, kv_cache, meta_info = model.forward_with_metacognition(
                x=last_token,
                carry_states=carry_states,
                past_key_values=kv_cache,
                abs_pos_offset=abs_pos_offset,
                enable_pattern_transfer=enable_patterns,
                pattern_fusion_weight=pattern_fusion_weight,
                verbose=verbose
            )
            
            abs_pos_offset += 1
            
            # Track domain
            if meta_info['current_domain']:
                domain_sequence.append(meta_info['current_domain'])
            
            # Track pattern applications
            if meta_info['patterns_applied']:
                pattern_applications.extend(meta_info['patterns_applied'])
            
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
            
            # Stop conditions
            if next_token == tokenizer.tokenizer.eot_token:
                break
        
        # Compile generation info
        generation_info = {
            'tokens_generated': len(generated_ids) - len(context_ids),
            'pattern_applications': len(pattern_applications),
            'unique_patterns_used': len(set(p['pattern_name'] for p in pattern_applications)),
            'cross_domain_transfers': sum(1 for p in pattern_applications if p['cross_domain']),
            'domain_sequence': domain_sequence,
            'pattern_details': pattern_applications
        }
        
        return generated_ids, generation_info
