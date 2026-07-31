"""A deliberately simple whitespace/word-piece-free tokenizer.

This is *engineering scaffolding*, not research. It lowercases, splits on
whitespace after separating punctuation, and maps tokens to integer ids over a
fixed vocabulary. Special tokens (separators, the consciousness/memory slots,
parse-tree markers) are reserved up front so model code can refer to them by
name.

TODO(tokenizer): swap for a real subword tokenizer (e.g. the one NAOMI uses, or
a BPE model) once the surface vocabulary grows beyond toy size.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

# Reserved special tokens. Order matters only in that ids are assigned in this
# order starting at 0; downstream code looks them up by string.
PAD = "[PAD]"
UNK = "[UNK]"
CONSC = "[CONSC]"   # slot that carries the consciousness vector
MEM = "[MEM]"       # slot that carries the retrieved-memory vector
SEP = "[SEP]"       # generic separator (e.g. between passage text and parse)
ANS = "[ANS]"       # marks the start of the answer/response region
EOS = "[EOS]"       # end of sequence / end of answer
TREE = "[TREE]"     # parse serialization: tree start
TREE_END = "[/TREE]"  # parse serialization: tree end
NODE = "[NODE]"     # parse serialization: node boundary

SPECIAL_TOKENS: List[str] = [PAD, UNK, CONSC, MEM, SEP, ANS, EOS, TREE, TREE_END, NODE]

_TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


def basic_tokenize(text: str) -> List[str]:
    """Lowercase and split ``text`` into word/number/punctuation tokens."""
    return _TOKEN_RE.findall(text.lower())


class SimpleTokenizer:
    """A fixed-vocabulary integer tokenizer.

    The vocabulary is the reserved special tokens followed by whatever tokens
    were seen during :meth:`build`. Out-of-vocabulary tokens map to ``[UNK]``.
    """

    def __init__(self, token_to_id: Dict[str, int]) -> None:
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, texts: Iterable[str], extra_tokens: Iterable[str] = ()) -> "SimpleTokenizer":
        """Build a tokenizer from a corpus of raw ``texts``.

        Args:
            texts: Raw strings to harvest surface tokens from.
            extra_tokens: Additional literal tokens to guarantee in the vocab
                (e.g. NSM prime keys, parse labels).
        """
        token_to_id: Dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)
        for tok in extra_tokens:
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)
        for text in texts:
            for tok in basic_tokenize(text):
                if tok not in token_to_id:
                    token_to_id[tok] = len(token_to_id)
        return cls(token_to_id)

    # -- properties ----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    def id_of(self, token: str) -> int:
        """Id of a literal/special token (falls back to ``[UNK]``)."""
        return self.token_to_id.get(token, self.token_to_id[UNK])

    # -- encode / decode -----------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Encode raw text into ids using whitespace tokenization."""
        return [self.id_of(tok) for tok in basic_tokenize(text)]

    def encode_tokens(self, tokens: Iterable[str]) -> List[int]:
        """Encode an already-tokenized sequence (e.g. a serialized parse)."""
        return [self.id_of(tok) for tok in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids back to a space-joined string (best-effort, for debugging)."""
        return " ".join(self.id_to_token.get(int(i), UNK) for i in ids)
