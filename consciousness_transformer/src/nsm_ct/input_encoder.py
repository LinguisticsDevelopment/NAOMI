"""The parser front-end: "sentence -> parse tree / discourse graph".

What survives of the old pluggable-encoder layer after the token-stack removal:
the clause perception path (:func:`nsm_ct.clause_reactor.build_clause_batch`)
uses :class:`ParserInputEncoder` for its parse tree (``_parse_tree``) and flat
discourse graph (``_parse_graph``); tokenization exists only as the parser's
text reader (the §0g decision — the *model* never embeds tokens).

* :class:`ParserInputEncoder` — wraps the experimental ``quantum_parser``.
  The parser is treated as untrusted: any failure degrades gracefully to plain
  tokenization, so a flaky parse never breaks the loop.
* :class:`TokenInputEncoder` — the trivial fallback (plain token ids).

TODO(input): a learned parser — a trained encoder mapping a sentence to the
same parse/meaning objects — would implement this same interface (the membrane
stays deterministic; see mind/grammar.py for the owned controlled-English path).
"""

from __future__ import annotations

import abc
import os
import sys
from typing import Dict, List, Optional, Tuple

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


_SENTENCE_END_TOKENS = {".", "?", "!"}


def _split_sentences(sentence: str) -> List[str]:
    """Split whitespace-tokenized text on sentence-final punctuation tokens.

    Mirrors quantum_parser's own tokenization (``sentence.split()`` in
    ``pos_tagger.tag_sentence``): a sentence boundary is a bare ``.``/``?``/
    ``!`` token, exactly as every sentence in this codebase is already
    written (space-separated from its final punctuation). Single-sentence
    input yields exactly one segment, so callers can special-case "len == 1"
    to keep that path byte-identical.
    """
    tokens = sentence.split()
    sentences: List[str] = []
    current: List[str] = []
    for tok in tokens:
        current.append(tok)
        if tok in _SENTENCE_END_TOKENS:
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences or [sentence]


def _merge_graphs(graphs: List["object"]):
    """Concatenate per-sentence :class:`~nsm_ct.quantum_adapter.HypGraph`\\ s.

    Node indices are offset by each preceding graph's node count so every
    node/edge index stays unique and internally consistent; nothing else about
    a sentence's nodes/edges is touched. The result satisfies exactly the
    graph API :func:`nsm_ct.clause.extract_discourse` uses: ``.nodes``,
    ``.edges``, ``node()``, ``token()``, ``label()``, ``edges_of()`` — all
    inherited for free since the merged object is a real ``HypGraph``.

    Per-node subtype flags (M50, ``HypGraph.flags``) are carried through under
    the same index offset, defensively via ``getattr`` so a caller-supplied
    graph without a ``flags`` attribute (pre-M50) still merges fine.
    """
    from .quantum_adapter import HypGraph  # local import (optional adapter)

    nodes: List[Tuple[int, str, Optional[str]]] = []
    edges: List[Tuple[str, int, int]] = []
    roots: List[int] = []
    flags: Dict[int, List[str]] = {}
    offset = 0
    for g in graphs:
        idx_map = {idx: idx + offset for idx, _label, _token in g.nodes}
        nodes.extend((idx_map[idx], label, token) for idx, label, token in g.nodes)
        edges.extend((etype, idx_map.get(p, p), idx_map.get(c, c)) for etype, p, c in g.edges)
        roots.extend(idx_map.get(r, r) for r in g.roots)
        for idx, fl in getattr(g, "flags", {}).items():
            flags[idx_map.get(idx, idx)] = fl
        offset += len(g.nodes)
    return HypGraph(nodes=nodes, edges=edges, roots=roots, flags=flags)



# ---------------------------------------------------------------------------
# Spanish Freeze Test (dev/ROADMAP_LONG_TERM.md): the language/grammar seam.
#
# clause.py's ``_PREP_RELATION`` (English preposition-token -> role-label map)
# is a bare module-level dict, not a parameter threaded through
# extract_clauses/extract_discourse -- there is no call-signature seam to
# inject a Spanish role map through. clause.py is explicitly out of scope for
# this task (FILES OWNED); the two options are (a) edit clause.py to
# parameterize ``_prep_relation``, or (b) extend the SAME dict from outside
# the file. (b) is safe here specifically because English and Spanish
# preposition tokens are disjoint strings (verified: "en"/"al"/"a"/"del"/"de"
# never collide with clause.py's English keys "in"/"on"/"at"/"to"/"into"/
# "from"/"by"/"near"/"inside") -- adding Spanish entries can only ever affect
# Spanish text, never change what an English preposition resolves to. This is
# recorded here as the seam decision, not silently done: a "proper" fix
# (parameterizing ``_prep_relation`` so callers pass their own role map)
# would touch clause.py and is left for a follow-up if a THIRD language ever
# makes the disjoint-keys assumption break.
# ---------------------------------------------------------------------------
_SPANISH_PREP_RELATION = {
    "en": "PLACE", "al": "PLACE", "a": "PLACE",   # locative / motion-destination
    "de": "SOURCE", "del": "SOURCE",              # "from" / "of" (de+el)
    "por": "PLACE",
}
_spanish_prep_relation_installed = False


def _install_spanish_prep_relation() -> None:
    """Idempotently extend ``nsm_ct.clause._PREP_RELATION`` with the Spanish
    map above (see the module-level note). Called once, lazily, only when a
    Spanish :class:`ParserInputEncoder` is actually constructed -- an
    English-only process never imports this path and clause.py's dict never
    changes for it."""
    global _spanish_prep_relation_installed
    if _spanish_prep_relation_installed:
        return
    from . import clause as _clause
    _clause._PREP_RELATION.update(_SPANISH_PREP_RELATION)
    _spanish_prep_relation_installed = True


class ParserInputEncoder(AbstractInputEncoder):
    """Optional encoder that runs the experimental ``quantum_parser``.

    Produces a serialized, role-annotated token stream. The parser is treated as
    untrusted: construction or parsing failures degrade gracefully to plain
    tokenization (a logged note once), so the loop keeps running.

    Args:
        tokenizer: Vocabulary (must already include node/relation label tokens).
        grammar_path: Path to a quantum_parser grammar JSON. If ``None``, uses
            the repo's English (or, if ``lang="es"``, Spanish) grammar.
        lang: ``"en"`` (default, byte-identical to before this parameter
            existed) or ``"es"`` -- selects the Spanish tagger
            (``tag_spanish_sentence``) and grammar (``grammars/spanish.json``)
            and installs the Spanish preposition->role extension (see
            :func:`_install_spanish_prep_relation`). No other behavior
            differs; unknown values raise (fail loud, not silently English).
    """

    def __init__(self, tokenizer: SimpleTokenizer, grammar_path: Optional[str] = None,
                 meaning_resolver=None, lang: str = "en") -> None:
        from .meaning import NSMMeaningResolver
        if lang not in ("en", "es"):
            raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
        self.tokenizer = tokenizer
        self._fallback = TokenInputEncoder(tokenizer)
        self._warned = False
        self._adapter = None
        self._grammar_path = grammar_path
        self._lang = lang
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
            from src.parser.quantum_parser import QuantumParser  # type: ignore

            if self._lang == "es":
                from src.parser.pos_tagger import tag_spanish_sentence as _tag  # type: ignore
                default_grammar = os.path.join(qp_root, "grammars", "spanish.json")
                _install_spanish_prep_relation()
            else:
                from src.parser.pos_tagger import tag_sentence as _tag  # type: ignore
                default_grammar = os.path.join(qp_root, "grammars", "english.json")

            grammar = self._grammar_path or default_grammar
            self._tag = _tag
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

    def _parse_graph_one(self, sentence: str):
        """Best-effort flat hypothesis graph for a SINGLE sentence.

        Returns ``None`` on any failure (the parser is untrusted).
        """
        try:
            from .quantum_adapter import hypothesis_to_graph  # local import

            words = self._tag(sentence)
            hyp = self._parser.parse(words).best_hypothesis()
            return hypothesis_to_graph(hyp) if hyp is not None else None
        except Exception as exc:  # pragma: no cover - parser is experimental
            self._note(f"graph parse failed ({exc}); no discourse structure.")
            return None

    def _parse_graph(self, sentence: str):
        """Best-effort flat hypothesis graph (keeps coordination/negation edges).

        The tree view (:meth:`_parse_tree`) drops inter-clause structure; this
        retains every typed edge for :func:`nsm_ct.clause.extract_discourse`.

        quantum_parser has no cross-sentence grammar rule: fed multi-sentence
        text directly, it folds sentence 2 into sentence 1's argument
        structure (e.g. in "mary went to the garden . she found the ball .",
        "she" comes out as an OBJECT of "went" and sentence 2 never yields a
        hypothesis root). So multi-sentence input (a sentence-final
        ``.``/``?``/``!`` token before the end, per :func:`_split_sentences`)
        is split and each sentence parsed on its own, then the per-sentence
        graphs are merged (:func:`_merge_graphs`, node indices offset) into
        one graph. Single-sentence input takes the exact same path as before
        this existed (``_split_sentences`` returns one segment) — byte-
        identical behavior for every existing caller.
        """
        if getattr(self, "_parser", None) is None:
            return None
        sentences = _split_sentences(sentence)
        if len(sentences) <= 1:
            return self._parse_graph_one(sentence)
        graphs = [g for g in (self._parse_graph_one(s) for s in sentences) if g is not None]
        if not graphs:
            return None
        if len(graphs) == 1:
            return graphs[0]
        return _merge_graphs(graphs)

    def _parse_topk_one(self, sentence: str, k: int = 4) -> Tuple[List["object"], List[float], float]:
        """M55a opt-in path (default OFF -- purely additive; no existing
        caller of ``_parse_graph``/``_parse_graph_one``/``_parse_tree`` is
        touched): the top-``k`` parse hypotheses for a SINGLE sentence, with
        their structural scores and the top1-top2 margin.

        ``QuantumParser.parse`` already deduplicates structurally-equivalent
        hypotheses at every grammar-rule step (its own ``is_equivalent``
        pass) and returns ``chart.hypotheses`` sorted by ``(score,
        completeness_key)`` (``ParseChart.sort_hypotheses`` /
        :func:`~quantum_parser.scorer.completeness_key`) with only complete
        parses kept when any exist -- so ``chart.hypotheses[:k]`` is already
        "K genuinely different readings" with no new dedup logic needed
        here (see ``scripts/probe_m55_hyp_survey.py``'s survey for which
        sentence SHAPES actually exercise this in our grammar).

        Returns ``(graphs, scores, margin)`` -- ``graphs`` are up to ``k``
        :class:`~nsm_ct.quantum_adapter.HypGraph` views (the same flat,
        ``extract_discourse``-ready shape :meth:`_parse_graph` returns),
        ``scores`` their matching structural scores, ``margin`` is
        ``scores[0] - scores[1]`` (``0.0`` if fewer than 2 hypotheses).
        ``([], [], 0.0)`` on any parser failure or multi-sentence input (the
        parser has no cross-sentence grammar rule -- see :meth:`_parse_graph`'s
        docstring; top-K exposure is single-sentence only, which is all any
        garden-path sentence needs).
        """
        if getattr(self, "_parser", None) is None:
            return [], [], 0.0
        if len(_split_sentences(sentence)) > 1:
            return [], [], 0.0
        try:
            from .quantum_adapter import hypothesis_to_graph  # local import

            words = self._tag(sentence)
            chart = self._parser.parse(words)
            hyps = chart.hypotheses[:k]
            graphs = [hypothesis_to_graph(h) for h in hyps]
            scores = [float(h.score) for h in hyps]
            margin = (scores[0] - scores[1]) if len(scores) > 1 else 0.0
            return graphs, scores, margin
        except Exception as exc:  # pragma: no cover - parser is experimental
            self._note(f"top-k parse failed ({exc}); no hypotheses.")
            return [], [], 0.0

    def encode_structured(self, sentence: str) -> Structured:
        from .thought import build_thought  # local import (avoids cycle)
        tree = self._parse_tree(sentence)
        if tree is not None:
            build_thought(sentence, tree, self._resolver)  # attach meaning trees
        tokens, roles, depths, meanings = align_structure(sentence, tree)
        return (self.tokenizer.encode_tokens(tokens),
                [role_id(r) for r in roles], depths, meanings)


