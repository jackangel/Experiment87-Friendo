"""
Cognitive Memory Manager — paragraph-level KV cache snapshots for generation.
"""


# =============================================================================
# COGNITIVE MEMORY MANAGER
# =============================================================================

class CognitiveMemoryManager:
    def __init__(self, device, max_paragraphs=50):
        self.device = device
        self.max_paragraphs = max_paragraphs  # Prevent unbounded growth
        self.paragraph_states = []
        self.paragraph_tokens = []

    def save_paragraph_state(self, carry_states, past_key_values, tokens):
        # Keep states on GPU to avoid CPU-GPU transfers (much faster)
        # Only move to CPU if approaching max_paragraphs limit
        should_cpu = len(self.paragraph_states) >= self.max_paragraphs * 0.8

        if should_cpu:
            cpu_carry = [c.detach().cpu().clone() if c is not None else None for c in carry_states] if carry_states else None
            cpu_kv = []
            if past_key_values:
                for entry in past_key_values:
                    # Meta-phase KV can be None when every token EXITed at
                    # inference before the shared layer ever ran. Skip it.
                    if entry is None:
                        cpu_kv.append(None)
                    else:
                        k, v, s, kr = entry
                        cpu_kv.append((k.detach().cpu().clone(), v.detach().cpu().clone(), s.detach().cpu().clone(), kr.detach().cpu().clone()))
            else:
                cpu_kv = None
        else:
            # Keep on GPU for faster access
            cpu_carry = [c.detach().clone() if c is not None else None for c in carry_states] if carry_states else None
            cpu_kv = []
            if past_key_values:
                for entry in past_key_values:
                    # Meta-phase KV can be None when every token EXITed at
                    # inference before the shared layer ever ran. Skip it.
                    if entry is None:
                        cpu_kv.append(None)
                    else:
                        k, v, s, kr = entry
                        cpu_kv.append((k.detach().clone(), v.detach().clone(), s.detach().clone(), kr.detach().clone()))
            else:
                cpu_kv = None

        self.paragraph_states.append({
            'carry_states': cpu_carry,
            'past_key_values': cpu_kv
        })
        self.paragraph_tokens.append(tokens)

        # Enforce max paragraphs limit (keep most recent)
        if len(self.paragraph_states) > self.max_paragraphs:
            self.paragraph_states = self.paragraph_states[-self.max_paragraphs:]
            self.paragraph_tokens = self.paragraph_tokens[-self.max_paragraphs:]

    def get_paragraph_state(self, idx):
        snap = self.paragraph_states[idx]
        dev_carry = [c.to(self.device) if c is not None else None for c in snap['carry_states']] if snap['carry_states'] else None
        dev_kv = []
        if snap['past_key_values']:
            for entry in snap['past_key_values']:
                # Meta-phase KV can be None when every token EXITed at
                # inference before the shared layer ever ran. Pass through.
                if entry is None:
                    dev_kv.append(None)
                else:
                    k, v, s, kr = entry
                    dev_kv.append((k.to(self.device), v.to(self.device), s.to(self.device), kr.to(self.device)))
        else:
            dev_kv = None
        return dev_carry, dev_kv
