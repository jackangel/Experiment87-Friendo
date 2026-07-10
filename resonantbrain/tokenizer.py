"""
Tiktoken tokenizer wrapper and ChatML constant definitions.
"""

import tiktoken

# ==========================================
# CHATML CONSTANTS
# ==========================================

# Using these variants prevents tiktoken's strict `<|...|>` regex from failing
CHAT_START = "<im_start>"
CHAT_END = "<im_end>"


class TiktokenTokenizer:
    def __init__(self, encoding_name="gpt2"):
        print(f"Loading tiktoken encoding: '{encoding_name}'...")
        base_tokenizer = tiktoken.get_encoding(encoding_name)

        # Explicitly register special tokens so they aren't split into characters
        special_tokens = {
            CHAT_START: base_tokenizer.n_vocab,
            CHAT_END: base_tokenizer.n_vocab + 1
        }

        self.tokenizer = tiktoken.Encoding(
            name="custom_chatml",
            pat_str=base_tokenizer._pat_str,
            mergeable_ranks=base_tokenizer._mergeable_ranks,
            special_tokens={**base_tokenizer._special_tokens, **special_tokens}
        )
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, ids):
        return self.tokenizer.decode(ids)
