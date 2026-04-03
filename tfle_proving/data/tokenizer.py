"""Simple BPE-lite tokenizer: 256 bytes + top-N bigrams."""

from __future__ import annotations

from collections import Counter


class BigramTokenizer:
    """Byte-level tokenizer extended with common bigrams.

    Vocab = 256 bytes + top-N bigrams. Simple, fast, no training needed.
    """

    def __init__(self, text: str, n_bigrams: int = 512):
        # Count all bigrams
        bigram_counts = Counter()
        for i in range(len(text) - 1):
            bigram_counts[text[i : i + 2]] += 1

        # Top N bigrams that aren't single bytes repeated
        self.bigrams = [
            bg for bg, _ in bigram_counts.most_common(n_bigrams * 2)
            if len(set(bg)) > 1  # skip "aa", "  ", etc.
        ][:n_bigrams]

        self.bigram_to_id = {bg: 256 + i for i, bg in enumerate(self.bigrams)}
        self.id_to_token = {i: chr(i) for i in range(256)}
        for bg, idx in self.bigram_to_id.items():
            self.id_to_token[idx] = bg

        self.vocab_size = 256 + len(self.bigrams)

    def encode(self, text: str) -> list[int]:
        tokens = []
        i = 0
        while i < len(text):
            if i + 1 < len(text):
                bigram = text[i : i + 2]
                if bigram in self.bigram_to_id:
                    tokens.append(self.bigram_to_id[bigram])
                    i += 2
                    continue
            tokens.append(ord(text[i]) % 256)
            i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.id_to_token.get(t, "?") for t in tokens)
