# SPDX-License-Identifier: MIT
"""DistilBERT's uncased WordPiece tokenizer in plain Python (no ``tokenizers``/``transformers``).

It reproduces the pipeline of ``distilbert-base-uncased``'s ``tokenizer.json``, which upstream
Kirin runs through ``AutoTokenizer(texts, padding=True)``:

1. special tokens written literally in the text (``[CLS]``, ``[MASK]``, ...) are matched first,
   on the raw text, like the fast tokenizer's added-token splitter;
2. ``BertNormalizer``: drop control characters and U+FFFD, map whitespace to spaces, pad CJK
   ideographs with spaces, strip accents (NFD, drop non-spacing marks), lowercase;
3. ``BertPreTokenizer``: split on whitespace, isolate every punctuation character;
4. ``WordPiece``: greedy longest-match-first with the ``##`` continuation prefix; a word longer
   than 100 characters, or with an unmatched piece, becomes ``[UNK]``;
5. ``[CLS] ... [SEP]``, then right-padding with ``[PAD]`` (id 0) to the longest text, with an
   attention mask that is 1 on real tokens.

The sandbox tests compare it with the Hugging Face tokenizer on a set of tricky strings.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
MAX_CHARS_PER_WORD = 100
MAX_LENGTH = 512  # DistilBERT's learned position table


def _is_whitespace(ch: str) -> bool:
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_chinese_char(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF
            or 0x2F800 <= cp <= 0x2FA1F)


class WordPieceTokenizer:
    """``tokenizer(texts) -> (input_ids [B, L] int32, attention_mask [B, L] bool)``."""

    def __init__(self, vocab_text: str):
        """``vocab_text``: the contents of ``vocab.txt`` (one token per line, id = line number)."""
        tokens = vocab_text.split("\n")
        if tokens and tokens[-1] == "":
            tokens.pop()
        self.vocab_text = vocab_text
        self.vocab = {token: index for index, token in enumerate(tokens)}
        missing = [t for t in SPECIAL_TOKENS if t not in self.vocab]
        if missing:
            raise ValueError(f"the vocabulary lacks the special tokens {missing}")
        self.pad_id, self.unk_id = self.vocab["[PAD]"], self.vocab["[UNK]"]
        self.cls_id, self.sep_id = self.vocab["[CLS]"], self.vocab["[SEP]"]
        self._special = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))

    @classmethod
    def from_file(cls, vocab_path: str | Path) -> WordPieceTokenizer:
        return cls(Path(vocab_path).read_text(encoding="utf-8"))

    # -- pipeline stages ------------------------------------------------------------------
    @staticmethod
    def normalize(text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            elif _is_chinese_char(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        text = unicodedata.normalize("NFD", "".join(out))
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        # Per character, like the Rust normalizer: str.lower() would apply the context-dependent
        # Greek final-sigma rule (word-final Σ -> ς), which the reference tokenizer does not.
        return "".join(ch.lower() for ch in text)

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        words, current = [], []
        for ch in text:
            if _is_whitespace(ch):
                if current:
                    words.append("".join(current))
                    current = []
            elif _is_punctuation(ch):
                if current:
                    words.append("".join(current))
                    current = []
                words.append(ch)
            else:
                current.append(ch)
        if current:
            words.append("".join(current))
        return words

    def wordpiece(self, word: str) -> list[int]:
        if len(word) > MAX_CHARS_PER_WORD:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end, match = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    match = self.vocab[piece]
                    break
                end -= 1
            if match is None:
                return [self.unk_id]
            ids.append(match)
            start = end
        return ids

    def encode(self, text: str) -> list[int]:
        """Token ids of one text, with ``[CLS]``/``[SEP]``."""
        ids, position = [self.cls_id], 0
        for special in self._special.finditer(text):
            ids += self._encode_plain(text[position:special.start()])
            ids.append(self.vocab[special.group()])
            position = special.end()
        ids += self._encode_plain(text[position:])
        ids.append(self.sep_id)
        if len(ids) > MAX_LENGTH:
            raise ValueError(f"text is {len(ids)} tokens long; DistilBERT takes at most {MAX_LENGTH}")
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        return [i for word in self.pre_tokenize(self.normalize(text)) for i in self.wordpiece(word)]

    def __call__(self, texts) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) for t in texts]
        if not encoded:
            raise ValueError("no texts to tokenize")
        length = max(len(e) for e in encoded)
        input_ids = np.full((len(encoded), length), self.pad_id, np.int32)
        mask = np.zeros((len(encoded), length), bool)
        for row, ids in enumerate(encoded):
            input_ids[row, :len(ids)] = ids
            mask[row, :len(ids)] = True
        return input_ids, mask
