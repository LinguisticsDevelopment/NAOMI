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
        from .meaning import NSMMeaningResolver
        self.tokenizer = tokenizer
        self._fallback = TokenInputEncoder(tokenizer)
        self._warned = False
        self._adapter = None
        self._grammar_path = grammar_path
        self._resolver = meaning_resolver or NSMMeaningResolver()
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

    def _parse_graph(self, sentence: str):
        """Best-effort flat hypothesis graph (keeps coordination/negation edges).

        The tree view (:meth:`_parse_tree`) drops inter-clause structure; this
        retains every typed edge for :func:`nsm_ct.clause.extract_discourse`.
        Returns ``None`` on any failure (the parser is untrusted).
        """
        if getattr(self, "_parser", None) is None:
            return None
        try:
            from .quantum_adapter import hypothesis_to_graph  # local import

            words = self._tag(sentence)
            hyp = self._parser.parse(words).best_hypothesis()
            return hypothesis_to_graph(hyp) if hyp is not None else None
        except Exception as exc:  # pragma: no cover - parser is experimental
            self._note(f"graph parse failed ({exc}); no discourse structure.")
            return None

    def encode_structured(self, sentence: str) -> Structured:
        from .thought import build_thought  # local import (avoids cycle)
        tree = self._parse_tree(sentence)
        if tree is not None:
            build_thought(sentence, tree, self._resolver)  # attach meaning trees
        tokens, roles, depths, meanings = align_structure(sentence, tree)
        return (self.tokenizer.encode_tokens(tokens),
                [role_id(r) for r in roles], depths, meanings)


class GroundedMeaningEncoder(AbstractInputEncoder):
    """Feed the model **sense-resolved grounded meaning**, not just raw tokens (M26.2).

    This is the first wiring of the roadmap's step (A): the grounding work (M17-M25)
    placed word-senses over named NSM axes; here each content word's meaning bag is
    filled from its resolved sense's *grounded* NSM-prime signature, so what the Mind
    pools into its memory write carries grounded meaning rather than a bare token id.

    Design (deliberately minimal, no model/pipeline surgery):

    * **Tokens/ids stay plain.** ``encode`` is ordinary tokenization; the model still
      owns the token embedding table. We only populate the per-token *meaning* channel
      that the pipeline already threads end to end (``encode_structured`` meanings ->
      ``item_meaning [B,T,L,M]`` -> ``prime_embedding(input_meaning).sum(2)`` in
      ``model.step``). So swapping this encoder in changes the meaning source and
      nothing else.
    * **Sense resolution is MFS (most-frequent-sense), context-free.** At encode time
      (host side, before the model) there is no ``(state, memory)`` context to drive
      the coherence WSD, so we take WordNet's most-frequent sense -- the first sense
      ``GroundedWordNetSenseInventory.senses(word)[0]`` (WordNet orders synsets MFS
      first). The runtime, context-dependent coherence resolver
      (``IterativeSenseResolver``) writing into the loop is the next milestone; this
      lands the static grounded channel it will later refine.
    * **Signature -> meaning ids.** The resolved sense's grounded primes (name -> weight
      from ``ground.sense_graph.gloss_prime_weights``) are ranked by weight and the top
      ``MAX_MEANING_PRIMES`` are mapped to the model's meaning ids via
      ``thought.meaning_prime_id`` (= ``nsm_primes.prime_index(name) + 1``; id 0 is pad).

    Honest note: ``prime_embedding`` is zero-initialized, so a fresh (untrained) model
    is unaffected by this channel at step 0 -- the grounded meaning only *matters once
    trained*. What this encoder guarantees is that the grounded signal is present,
    correctly aligned, and connected to the learning path.
    """

    def __init__(self, tokenizer: SimpleTokenizer, inventory=None, max_primes: Optional[int] = None) -> None:
        from .thought import MAX_MEANING_PRIMES
        from .wsd import GroundedWordNetSenseInventory

        self.tokenizer = tokenizer
        self.inventory = inventory or GroundedWordNetSenseInventory()
        self.max_primes = max_primes or MAX_MEANING_PRIMES
        self._cache: dict = {}  # word -> meaning-id list (per-word MFS signature is stable)

    def encode(self, sentence: str) -> List[int]:
        # Plain token ids; the grounded meaning rides alongside via encode_structured.
        return self.tokenizer.encode(sentence)

    def _meaning_ids(self, word: str) -> List[int]:
        """The MFS grounded sense's top primes for ``word``, as model meaning ids."""
        if word in self._cache:
            return self._cache[word]
        from .thought import meaning_prime_id

        senses = self.inventory.senses(word)
        primes = senses[0].primes if senses else {}          # MFS = first sense
        # rank primes by grounded weight; take the strongest few for the bag
        top = sorted(primes.items(), key=lambda kv: -kv[1])[: self.max_primes]
        ids = [meaning_prime_id(name) for name, _ in top]
        self._cache[word] = ids
        return ids

    def encode_structured(self, sentence: str) -> Structured:
        from .tokenizer import basic_tokenize

        # basic_tokenize is exactly what the tokenizer's encode() splits on, so tokens,
        # ids, and meanings stay index-aligned (one token = one word = one meaning bag).
        tokens = basic_tokenize(sentence)
        ids = self.tokenizer.encode_tokens(tokens)
        meanings = [self._meaning_ids(tok) for tok in tokens]
        n = len(ids)
        # roles/depths are left neutral (0): this encoder grounds MEANING, not parse
        # structure -- those remain the parser encoder's job.
        return ids, [0] * n, [0] * n, meanings


def make_input_encoder(name: str, tokenizer: SimpleTokenizer) -> AbstractInputEncoder:
    """Factory: build an input encoder by config name."""
    name = name.lower()
    if name == "token":
        return TokenInputEncoder(tokenizer)
    if name == "parser":
        return ParserInputEncoder(tokenizer)
    if name == "grounded":
        return GroundedMeaningEncoder(tokenizer)
    raise ValueError(f"Unknown input encoder: {name!r}")
