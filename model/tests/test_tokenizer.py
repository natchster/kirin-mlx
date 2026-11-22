"""The plain-Python WordPiece tokenizer on a tiny vocabulary (weight-free). The parity tests
compare it with the Hugging Face tokenizer on the real vocabulary."""

from __future__ import annotations

import numpy as np
import pytest
from kirin_mlx.tokenizer import MAX_LENGTH, WordPieceTokenizer

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "horse", "run", "##s", "##ning", ",", ".", "cafe",
         "日", "本", "!", "un", "##aff", "##able", "x"]


@pytest.fixture(scope="module")
def tok():
    return WordPieceTokenizer("\n".join(VOCAB) + "\n")


def ids(tok, *words):
    return [tok.vocab[w] for w in words]


def test_basic_and_wordpiece(tok):
    assert tok.encode("a horse runs") == ids(tok, "[CLS]", "a", "horse", "run", "##s", "[SEP]")
    assert tok.encode("unaffable") == ids(tok, "[CLS]", "un", "##aff", "##able", "[SEP]")
    assert tok.encode("running") == ids(tok, "[CLS]", "run", "##ning", "[SEP]")


def test_lowercase_accents_punctuation(tok):
    assert tok.encode("A HORSE, Café!") == ids(tok, "[CLS]", "a", "horse", ",", "cafe", "!", "[SEP]")


def test_cjk_chars_are_split(tok):
    assert tok.encode("日本") == ids(tok, "[CLS]", "日", "本", "[SEP]")


def test_unknown_and_long_words(tok):
    assert tok.encode("zebra") == ids(tok, "[CLS]", "[UNK]", "[SEP]")
    assert tok.encode("runz") == ids(tok, "[CLS]", "[UNK]", "[SEP]")  # an unmatched piece makes the word [UNK]
    assert tok.encode("x" * 101) == ids(tok, "[CLS]", "[UNK]", "[SEP]")


def test_whitespace_control_and_empty(tok):
    assert tok.encode(" \t\n ") == ids(tok, "[CLS]", "[SEP]")
    assert tok.encode("") == ids(tok, "[CLS]", "[SEP]")
    # NUL and the zero-width space (category Cf) are dropped, not turned into separators
    assert tok.encode("a\x00 ​horse") == ids(tok, "[CLS]", "a", "horse", "[SEP]")
    assert tok.encode("ru​ns") == ids(tok, "[CLS]", "run", "##s", "[SEP]")


def test_literal_special_tokens(tok):
    assert tok.encode("a [MASK] horse") == ids(tok, "[CLS]", "a", "[MASK]", "horse", "[SEP]")
    assert tok.encode("[mask]")[1:-1] != ids(tok, "[MASK]")  # matched case-sensitively, on the raw text


def test_batch_padding_and_mask(tok):
    input_ids, mask = tok(["a horse", "a horse runs"])
    assert input_ids.dtype == np.int32 and mask.dtype == bool
    np.testing.assert_array_equal(mask, [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    assert input_ids[0, 4] == tok.pad_id and input_ids[0, 5] == tok.pad_id


def test_too_long_raises(tok):
    with pytest.raises(ValueError, match="at most"):
        tok.encode("a " * MAX_LENGTH)


def test_missing_special_tokens():
    with pytest.raises(ValueError, match="special tokens"):
        WordPieceTokenizer("a\nb\n")
