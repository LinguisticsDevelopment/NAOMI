"""Pluggable "sentence -> input object" encoders.

The input object is just the token-id sequence the model's step function
consumes; the model owns the embedding table, so encoders only need to produce
ids. This keeps the *source* of structure swappable:

* :class:`TokenInputEncoder` — the **default**. Plain tokenization, no parser.
  Chosen as the spine because the rule-based parser is experimental/inconsistent.
* :class:`ParserInputEncoder` — **optional / unstable**. Wraps the experimental
  ``quantum_parser`` to produce a structure-aware token stream (with semantic
  role markers). Any failure falls back to plain tokenization, so a flaky parse
  never breaks training.

TODO(input): a third encoder — "chained transformers" — is the user's fallback
if rule parsing proves too messy: a learned transformer that maps a sentence to
the input object. It would implement this same interface.
"""

from __future__ import annotations

import abc
import os
import sys
from typing import List, Optional, Tuple

from .structure import align_structure, role_id
from .tokenizer import SimpleTokenizer

# A structured encoding: aligned (token ids, role ids, depths, meanings) of equal
# length. Role ids index the dedicated structure vocab (structure.py); meanings is
# a per-token bag of NSM prime ids (the word's meaning tree). Both are structure
# layered on top of the words — never replacing them.
Structured = Tuple[List[int], List[int], List[int], List[List[int]]]


class AbstractInputEncoder(abc.ABC):
    """Maps a sentence to a list of token ids (the "input object")."""

    @abc.abstractmethod
    def encode(self, sentence: str) -> List[int]:
        raise NotImplementedError

    def encode_structured(self, sentence: str) -> Structured:
        """Tokens + per-token (role, depth, meaning). Default: no structure."""
        ids = self.encode(sentence)
        return ids, [0] * len(ids), [0] * len(ids), [[] for _ in ids]


class TokenInputEncoder(AbstractInputEncoder):
    """Default encoder: plain tokenization. No parser, no structure."""

    def __init__(self, tokenizer: SimpleTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, sentence: str) -> List[int]:
        return self.tokenizer.encode(sentence)


class ParserInputEncoder(AbstractInputEncoder):
    """Optional encoder that runs the experimental ``quantum_parser``.

    Produces a serialized, role-annotated token stream. The parser is treated as
    untrusted: construction or parsing failures degrade gracefully to plain
    tokenization (a logged note once), so the loop keeps running.

    Args:
        tokenizer: Vocabulary (must already include node/relation label tokens).
        grammar_path: Path to a quantum_parser grammar JSON. If ``None``, uses
            the repo's English grammar.
    """

    def __init__(self, tokenizer: SimpleTokenizer, grammar_path: Optional[str] = None,
                 meaning_resolver=None) -> None:
        from .thought import MockMeaningResolver
        self.tokenizer = tokenizer
        self._fallback = TokenInputEncoder(tokenizer)
        self._warned = False
        self._adapter = None
        self._grammar_path = grammar_path
        self._resolver = meaning_resolver or MockMeaningResolver()
        self._init_adapter()

    def _init_adapter(self) -> None:
        try:
            qp_root = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "quantum_parser")
            )
            if qp_root not in sys.path:
                sys.path.insert(0, qp_root)
            from src.parser.pos_tagger import tag_sentence  # type: ignore
            from src.parser.quantum_parser import QuantumParser  # type: ignore

            grammar = self._grammar_path or os.path.join(qp_root, "grammars", "english.json")
            self._tag = tag_sentence
            self._parser = QuantumParser(grammar)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            self._note(f"quantum_parser unavailable ({exc}); using plain tokenization.")
            self._adapter = None

    def _note(self, msg: str) -> None:
        if not self._warned:
            print(f"[ParserInputEncoder] {msg}")
            self._warned = True

    def encode(self, sentence: str) -> List[int]:
        # Plain token ids (the words the model embeds). Structure rides alongside
        # via encode_structured; this keeps `encode` lossless and consistent.
        return self.tokenizer.encode(sentence)

    def _parse_tree(self, sentence: str):
        """Best-effort parse tree, or None on any failure (parser is untrusted)."""
        if getattr(self, "_parser", None) is None:
            return None
        try:
            from .quantum_adapter import hypothesis_to_tree  # local import

            words = self._tag(sentence)
            hyp = self._parser.parse(words).best_hypothesis()
            return hypothesis_to_tree(hyp, sentence) if hyp is not None else None
        except Exception as exc:  # pragma: no cover - parser is experimental
            self._note(f"parse failed ({exc}); feeding tokens without structure.")
            return None

    def encode_structured(self, sentence: str) -> Structured:
        from .thought import build_thought  # local import (avoids cycle)
        tree = self._parse_tree(sentence)
        if tree is not None:
            build_thought(sentence, tree, self._resolver)  # attach meaning trees
        tokens, roles, depths, meanings = align_structure(sentence, tree)
        return (self.tokenizer.encode_tokens(tokens),
                [role_id(r) for r in roles], depths, meanings)


def make_input_encoder(name: str, tokenizer: SimpleTokenizer) -> AbstractInputEncoder:
    """Factory: build an input encoder by config name."""
    name = name.lower()
    if name == "token":
        return TokenInputEncoder(tokenizer)
    if name == "parser":
        return ParserInputEncoder(tokenizer)
    raise ValueError(f"Unknown input encoder: {name!r}")
