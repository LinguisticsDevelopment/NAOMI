"""Reader: sentence -> ReadResult (a Tree: every non-punctuation token a node, uncommitted senses).

Pattern-free by owner decision: no word lists, regexes or heuristics that encode a language live
here. Language knowledge comes only from (a) the deterministic grammar parser
(``quantum_parser`` + ``grammars/<lang>.json``) and (b) the lexicon. This file (1) normalises text
exactly like the corpus pipeline, (2) runs the parser under a wall-clock cap, (3) builds the tree
from ALL of its hypothesis-graph edges (``naomi.tree.attach_graph``: language-neutral traversal,
the grammar's edge types become the relation labels, anything unattached is marked UNATTACHED and
diagnosed, never repaired), (4) fills lemma / candidate senses (single seam: ``sense_candidates``),
(5) falls back to the frozen v6 encoder's old-style clauses, converted as far as they go and marked
``encoder_v6_legacy``, (6) caches.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .tree import ROOT_TOKEN, attach_graph, tree_from_dict, tree_to_dict, walk
from .types import (KIND_NAME, KIND_WORD, REL_ROOT, REL_UNATTACHED, SOURCE_LEGACY, SOURCE_NONE,
                    SOURCE_PARSER, ClauseTree, Filler, Node, ReadResult, Tree)

CT_DIR = Path(__file__).resolve().parents[2]  # .../consciousness_transformer

PARSER_MAX_HYPOTHESES = 4000
# tagger categories (the pipeline's own UPOS names) -> WordNet part-of-speech letters (lexicon access)
_WN_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
# tagger categories whose tokens are names: variables with no meaning point (owner: numerals too, for now)
_NAME_POS = ("PROPN", "NUM")
_PUNCT_POS = "PUNCT"


class _ParserTimeout(Exception):
    pass


def normalize(text: str) -> str:
    """Lowercase + whitespace/punctuation tokenisation, exactly the corpus pipeline's."""
    from nsm_ct.corpus import _tokenize_words
    return " ".join(_tokenize_words(text))


# --------------------------------------------------------------------------- wordnet
@lru_cache(maxsize=50000)
def _wn_lemma(token: str, pos: Optional[str]) -> str:
    try:
        from nltk.corpus import wordnet as wn
        for p in ([pos] if pos else ["v", "n", "a", "r"]):
            m = wn.morphy(token, p)
            if m:
                return m
    except Exception:
        pass
    return token


@lru_cache(maxsize=50000)
def _wn_senses(lemma: str, pos: Optional[str]) -> Tuple[str, ...]:
    try:
        from nltk.corpus import wordnet as wn
        ss = []
        if pos:
            ss = list(wn.synsets(lemma, pos=pos))
            if pos == "a":
                ss += list(wn.synsets(lemma, pos="s"))
        rest = [s for s in wn.synsets(lemma) if s not in ss]
        return tuple(s.name() for s in (ss + rest)[:5])
    except Exception:
        return ()


def sense_candidates(token: str, lemma: str, wn_pos: Optional[str]) -> List[str]:
    """THE single seam for candidate senses (best-first, uncommitted). Called for every non-name
    token, function words included; empty list if the lexicon knows nothing."""
    return list(_wn_senses(lemma, wn_pos))


def make_node(rel: str, token: str, idx: Optional[int], pos: str, label: Optional[str] = None,
              subtypes: Optional[List[str]] = None) -> Node:
    token = (token or "").lower()
    if pos in _NAME_POS:
        return Node(rel=rel, token=token, token_index=idx, lemma=token, pos=pos, kind=KIND_NAME,
                    label=label, subtypes=list(subtypes or []))
    lemma_pos = _WN_POS.get(pos) or ("v" if pos == "AUX" else None)
    lemma = _wn_lemma(token, lemma_pos) if lemma_pos else token
    return Node(rel=rel, token=token, token_index=idx, lemma=lemma, pos=pos, kind=KIND_WORD,
                senses=sense_candidates(token, lemma, _WN_POS.get(pos)), label=label,
                subtypes=list(subtypes or []))


def synthetic_root() -> Node:
    return Node(rel=REL_ROOT, token=ROOT_TOKEN, token_index=None, lemma=ROOT_TOKEN, pos=ROOT_TOKEN)


class Reader:
    def __init__(self, lang: str = "en", encoder_ckpt: str = "runs/arms/v6_all_hard_seed0.pt",
                 parser_seconds: float = 5.0, cache_path: Optional[str] = None,
                 use_parser: bool = True, use_encoder: bool = True):
        self.lang = lang
        self.encoder_ckpt = self._resolve(encoder_ckpt)
        self.parser_seconds = float(parser_seconds)
        self.use_parser, self.use_encoder = use_parser, use_encoder
        self.cache_path = self._resolve(cache_path) if cache_path else None
        self._parser = None
        self._usvs = None
        self._enc = None  # (model, ckpt, em)
        self._cache: Dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    # ---- paths / cache -----------------------------------------------------
    @staticmethod
    def _resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else CT_DIR / p

    def _load_cache(self) -> None:
        with open(self.cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self._cache[rec["k"]] = rec["v"]
                except Exception:
                    continue

    def _cache_put(self, key: str, val: dict) -> None:
        self._cache[key] = val
        if self.cache_path:  # write-through (append)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": val}) + "\n")

    def save_cache(self) -> None:
        """Rewrite the (write-through) cache file compactly."""
        if not self.cache_path:
            return
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for k, v in self._cache.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        os.replace(tmp, self.cache_path)

    # ---- lazy resources ----------------------------------------------------
    def _get_parser(self):
        if self._parser is None:
            from nsm_ct.input_encoder import ParserInputEncoder
            from nsm_ct.nsm_primes import PRIME_NAMES
            from nsm_ct.structure import PARSE_LABELS
            from nsm_ct.tokenizer import SimpleTokenizer
            tok = SimpleTokenizer.build([], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
            self._parser = ParserInputEncoder(tok, lang=self.lang)
        return self._parser

    def _get_encoder(self):
        if self._enc is None:
            import torch
            from nsm_ct import encoder_model as em
            from nsm_ct.ground.usvs import load_usvs
            torch.set_num_threads(1)  # oversubscribed threads make beam decode ~10x slower here
            ckpt = torch.load(str(self.encoder_ckpt), map_location="cpu", weights_only=False)
            model = em.EncoderModel(ckpt["pos_vocab"], ckpt["role_vocab"], d_axes=ckpt["d_axes"],
                                    hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                                    controller_hidden=ckpt["d_model"])
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._usvs = load_usvs(str(CT_DIR / "data" / "usvs"), extended=False)  # v6 encoder was trained on the 607 base axes
            self._enc = (model, ckpt, em)
        return self._enc

    def tag_full(self, norm: str) -> Tuple[List[str], List[str], List[List[str]]]:
        """(tokens, UPOS tags, tagger subtype names) of a normalised sentence."""
        words = self._get_parser()._tag(norm)
        return ([w.text.lower() for w in words], [w.pos.name for w in words],
                [[getattr(st, "name", str(st)) for st in (w.subtypes or [])] for w in words])

    def tag(self, norm: str) -> Tuple[List[str], List[str]]:
        """(tokens, UPOS tags) of a normalised sentence, from the pipeline's own tagger."""
        t, p, _ = self.tag_full(norm)
        return t, p

    # ---- parser path -------------------------------------------------------
    def _raw_graph(self, norm: str) -> dict:
        """The parser's own hypothesis graph, JSON-able and cached: nodes (index, label, token),
        typed edges (type, parent, child) in the parser's order, and its unconsumed roots.
        ``{"fail": True}`` when the parser raised, hit the cap, or produced no hypothesis."""
        key = "g|" + norm
        if key in self._cache:
            return self._cache[key]
        parser = self._get_parser()
        out: dict = {"fail": True}
        use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")
        old = None
        if use_alarm:
            def _h(signum, frame):
                raise _ParserTimeout()
            old = signal.signal(signal.SIGALRM, _h)
            signal.setitimer(signal.ITIMER_REAL, self.parser_seconds + 1.0)
        try:
            graph = parser._parse_graph(norm, max_hypotheses=PARSER_MAX_HYPOTHESES,
                                        max_seconds=self.parser_seconds)
            if graph is not None:
                out = {"fail": False, "nodes": [list(n) for n in graph.nodes],
                       "edges": [list(e) for e in graph.edges], "roots": list(graph.roots)}
        except BaseException as exc:  # noqa: BLE001 -- cap hit, no hypothesis, anything
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
        self._cache_put(key, out)
        return out

    def _from_parser(self, norm: str, sentence: str) -> Optional[Tree]:
        raw = self._raw_graph(norm)
        if raw.get("fail"):
            return None
        tokens, pos, subs = self.tag_full(norm)
        labels = {n[0]: n[1] for n in raw["nodes"]}
        diags: List[dict] = []
        for idx, _lab, tok in raw["nodes"]:
            if not (0 <= idx < len(tokens)) or (tok or "").lower() != tokens[idx]:
                diags.append({"kind": "graph_token_mismatch", "index": idx, "token": tok})

        def mk(i: int) -> Node:
            return make_node(REL_UNATTACHED, tokens[i], i, pos[i], labels.get(i), subs[i])

        root, d = attach_graph(tokens, pos, [tuple(e) for e in raw["edges"]], raw["roots"], mk,
                               lambda i: pos[i] == _PUNCT_POS)
        return Tree(sentence=sentence, root=root, source=SOURCE_PARSER, diagnostics=diags + d)

    # ---- encoder path (legacy format) --------------------------------------
    def _legacy_clauses(self, norm: str, tokens: List[str]) -> List[ClauseTree]:
        model, ckpt, em = self._get_encoder()
        sc = [{"index": i, "token": t, "sense_candidates": list(c)}
              for i, t in enumerate(tokens) for c in [self._usvs.senses_of(t)] if c]
        pos = self.tag(norm)[1]
        rec = {"text": norm, "tokens": tokens, "pos": pos, "token_sense_candidates": sc}
        feats = em.build_features(rec, self._usvs, ckpt["pos_vocab"], ckpt["hash_buckets"])
        forest = em.beam_decode(model, feats, beam_width=8, k=8, policy="model")
        out = []
        for cl in (forest[0] if forest else {"clauses": []}).get("clauses", []):
            pi = cl.get("predicate", {}).get("token_index")
            if pi is None or not (0 <= pi < len(tokens)):
                continue
            c = ClauseTree(predicate=Filler("PREDICATE", tokens[pi], tokens[pi], pi))
            for r in cl.get("roles", []):
                ai = r.get("token_index")
                if ai is not None and 0 <= ai < len(tokens):
                    c.args.append(Filler(str(r.get("relation", "ARG")).upper(), tokens[ai], tokens[ai], ai))
            out.append(c)
        return out

    @staticmethod
    def legacy_to_tree(sentence: str, clauses: List[ClauseTree], tokens: List[str], pos: List[str],
                       subs: List[List[str]]) -> Optional[Tree]:
        """Old clauses -> Tree as far as they go: tokens the encoder placed become nodes (first
        clause predicate = root, later predicates and every token it did not place hang from the
        root as UNATTACHED; args keep their OLD role labels, which is why source is legacy)."""
        if not clauses:
            return None
        placed: Dict[int, Node] = {}
        diags: List[dict] = []

        def node(rel: str, i: int) -> Node:
            return make_node(rel, tokens[i], i, pos[i], None, subs[i])

        root = None
        for ci, c in enumerate(clauses):
            pi = c.predicate.token_index
            if pos[pi] == _PUNCT_POS:
                continue
            if pi in placed:
                diags.append({"kind": "extra_edge", "type": REL_UNATTACHED, "index": pi, "token": tokens[pi]})
                continue
            pn = node(REL_ROOT if root is None else REL_UNATTACHED, pi)
            placed[pi] = pn
            if root is None:
                root = pn
            else:
                root.children.append(pn)
            for a in c.args:
                if pos[a.token_index] == _PUNCT_POS:
                    continue
                if a.token_index in placed:
                    diags.append({"kind": "extra_edge", "type": a.role, "index": a.token_index,
                                  "token": a.token})
                    continue
                an = node(a.role, a.token_index)
                placed[a.token_index] = an
                pn.children.append(an)
        for i in range(len(tokens)):
            if i not in placed and pos[i] != _PUNCT_POS:
                root.children.append(node(REL_UNATTACHED, i))
                diags.append({"kind": "unattached_token", "index": i, "token": tokens[i], "pos": pos[i]})
        for n, _d, _p in walk(root):
            n.children.sort(key=lambda x: x.token_index)
        return Tree(sentence=sentence, root=root, source=SOURCE_LEGACY, diagnostics=diags)

    # ---- public ------------------------------------------------------------
    def _cached_tree(self, kind: str, sentence: str, build) -> Optional[Tree]:
        key = f"{kind}|{sentence}"
        if key not in self._cache:
            t = build()
            self._cache_put(key, {"tree": tree_to_dict(t) if t is not None else None})
        d = self._cache[key]["tree"]
        return tree_from_dict(d) if d is not None else None

    def read_parser(self, sentence: str, sentence_index=None) -> Optional[ReadResult]:
        """Parser-only reading, or None if the parser failed (error, cap, no hypothesis)."""
        t0 = time.perf_counter()
        t = self._cached_tree("tp", sentence, lambda: self._from_parser(normalize(sentence), sentence))
        if t is None:
            return None
        return ReadResult(sentence=sentence, tree=t, source=SOURCE_PARSER, seconds=time.perf_counter() - t0)

    def read_encoder(self, sentence: str, sentence_index=None) -> ReadResult:
        """Legacy encoder-only reading (v6 old-format clauses converted to a Tree)."""
        def build():
            norm = normalize(sentence)
            tokens, pos, subs = self.tag_full(norm)
            try:
                cl = self._legacy_clauses(norm, tokens)
            except Exception:
                cl = []
            return self.legacy_to_tree(sentence, cl, tokens, pos, subs)
        t0 = time.perf_counter()
        t = self._cached_tree("tl", sentence, build)
        if t is None:
            return ReadResult(sentence=sentence, source=SOURCE_NONE, seconds=time.perf_counter() - t0)
        return ReadResult(sentence=sentence, tree=t, source=SOURCE_LEGACY, seconds=time.perf_counter() - t0)

    def read(self, sentence: str, sentence_index=None) -> ReadResult:
        """Parser tree; the legacy encoder's converted tree if the parser fails."""
        if self.use_parser:
            rr = self.read_parser(sentence, sentence_index)
            if rr is not None:
                return rr
        if self.use_encoder:
            rr = self.read_encoder(sentence, sentence_index)
            if rr.tree is not None:
                return rr
        return ReadResult(sentence=sentence, source=SOURCE_NONE)

    def read_question(self, question: str) -> ReadResult:
        """Alias of :meth:`read`: questions get no special treatment here."""
        return self.read(question)
