"""
Interactive ChatML chat loop for instruction-tuned inference.
"""

import torch

from .generation import generate_block_recurrent
from .tokenizer import CHAT_START, CHAT_END


# =============================================================================
# CHAT MODE
# =============================================================================

def chat_mode(model, tokenizer, device, chunk_size=512, temperature=0.7,
              repetition_penalty=1.3, top_k=50, top_p=0.9,
              max_new_tokens=512, stop_sequence=CHAT_END):
    """Interactive chat loop using ChatML format."""
    model.eval()
    print("\n" + "=" * 60)
    print("CHAT MODE - Type 'quit' or 'exit' to end session")
    print("=" * 60)
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("Ending chat session.")
            break
        if not user_input:
            continue

        prompt = f"{CHAT_START}user\n{user_input}{CHAT_END}\n{CHAT_START}assistant\n"
        input_ids = tokenizer.encode(prompt)

        print("Assistant: ", end="", flush=True)

        try:
            generated_ids = generate_block_recurrent(
                model, input_ids, tokenizer, device,
                max_new_tokens=max_new_tokens, chunk_size=chunk_size,
                temperature=temperature, repetition_penalty=repetition_penalty,
                top_k=top_k, top_p=top_p, stop_sequence=stop_sequence
            )

            # Extract only the model's response (after the prompt).
            response_ids = generated_ids[len(input_ids):]
            response_text = tokenizer.decode(response_ids)

            if stop_sequence in response_text:
                response_text = response_text.split(stop_sequence, 1)[0]

            print(response_text)
        except Exception as e:
            print(f"\n[ERROR during generation]: {e}")
            import traceback
            traceback.print_exc()

        print()
