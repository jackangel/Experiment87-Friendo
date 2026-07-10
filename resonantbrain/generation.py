"""
Text generation — block-recurrent autoregressive sampling with carry/KV states.
"""

import torch
import torch.nn.functional as F

from .data import apply_sampling_penalties
from .memory import CognitiveMemoryManager


# =============================================================================
# GENERATION
# =============================================================================

def generate_block_recurrent(model, context_ids, tokenizer, device,
                             max_new_tokens=256, chunk_size=512,
                             temperature=0.8, repetition_penalty=1.2,
                             top_k=50, top_p=0.9, enable_rewind=True,
                             stop_sequence=None, max_paragraph_cache=50):
    model.eval()
    memory_manager = CognitiveMemoryManager(device, max_paragraphs=max_paragraph_cache)

    with torch.inference_mode():
        generated_ids = context_ids.copy()

        paragraphs = [context_ids[i:i + chunk_size] for i in range(0, len(context_ids), chunk_size)]

        # Process context chunks with proper position tracking
        carry_states = None
        past_key_values = None
        cumulative_pos = 0
        for chunk in paragraphs:
            if len(chunk) == 0:
                continue
            chunk_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
            _, carry_states, past_key_values = model(
                x=chunk_tensor, carry_states=carry_states, is_training=False,
                past_key_values=past_key_values, use_cache=True, abs_pos_offset=cumulative_pos
            )
            cumulative_pos += len(chunk)
            memory_manager.save_paragraph_state(carry_states, past_key_values, chunk)

        tokens_generated = 0
        context_length = len(generated_ids)  # Track where new generation starts

        # Initialize with the last paragraph state and correct position
        if len(memory_manager.paragraph_tokens) > 0:
            active_carry, active_kv = memory_manager.get_paragraph_state(len(paragraphs) - 1)
            abs_pos_offset = cumulative_pos
        else:
            active_carry, active_kv = None, None
            abs_pos_offset = 0

        while tokens_generated < max_new_tokens:
            last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
            logits, active_carry, active_kv = model(
                x=last_token, carry_states=active_carry, is_training=False,
                past_key_values=active_kv, use_cache=True, abs_pos_offset=abs_pos_offset
            )

            # Increment position for next iteration
            abs_pos_offset += 1

            # Convert to float32 for numerical stability during sampling
            next_token_logits = logits[0, -1].float().clone()
            next_token_logits = apply_sampling_penalties(
                next_token_logits, generated_ids, repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p
            )
            probs = F.softmax(next_token_logits / temperature, dim=-1)

            if torch.isnan(probs).any():
                print(f"\n[ERROR] NaN detected in sampling probabilities at token {tokens_generated}!")
                print(f"[ERROR] Last logits stats: min={next_token_logits.min():.2f}, max={next_token_logits.max():.2f}, mean={next_token_logits.mean():.2f}")
                print(f"[ERROR] Temperature: {temperature}, Last token: {generated_ids[-1]}")
                next_token = tokenizer.tokenizer.eot_token
            else:
                next_token = torch.multinomial(probs, 1).item()

            generated_ids.append(next_token)
            tokens_generated += 1

            #print(f"\n[DEBUG] Token {tokens_generated}: {next_token} -> '{tokenizer.decode([next_token])}'")

            if next_token == tokenizer.tokenizer.eot_token:
                #print(f"[DEBUG] EOT token detected. Breaking. EOT={tokenizer.tokenizer.eot_token}")
                break

            if stop_sequence:
                # Only check newly generated tokens, not the context
                new_tokens_only = generated_ids[context_length:]
                check_len = min(len(new_tokens_only), 10)
                recent_text = tokenizer.decode(new_tokens_only[-check_len:])
                if stop_sequence in recent_text:
                    print(f"[DEBUG] Stop sequence '{stop_sequence}' found in '{recent_text}'. Breaking.")
                    break

    return generated_ids
