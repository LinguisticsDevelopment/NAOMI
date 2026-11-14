"""Reader: sentence -> ReadResult (ClauseTrees with uncommitted sense candidates).

Pattern-free by owner decision: no English word lists, regexes or heuristics live here.
Language knowledge comes only from (a) the deterministic grammar parser
(``quantum_parser`` + ``grammars/english.json``) and (b) what the learned encoder has
learned. This file only (1) normalises text exactly like the corpus pipeline, (2) runs the
parser under a wall-clock cap and converts ITS output faithfully (every edge keeps the
parser's own label), (3) falls back to the encoder's rank-1 tree when the parser fails,
(4) fills lemma / candidate senses from WordNet (lexicon access), (5) caches.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .types import ClauseTree, Filler, ReadResult

CT_DIR = Path(__file__).resolve().parents[2]  # .../consciousness_transformer

PARSER_MAX_HYPOTHESES = 4000
# structural UPOS categories (tagger labels, not words)
_FUNCTION_POS = {"AUX", "DET", "ADP", "PART", "PRON", "CCONJ", "SCONJ", "PUNCT", "NUM", "SYM", "INTJ"}
_WN_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


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


def make_filler(role: str, token: str, idx: Optional[int], pos: List[str]) -> Filler:
    token = (token or "").lower()
    p = pos[idx] if idx is not None and 0 <= idx < len(pos) else None
    if p == "AUX":
        lemma, wn_pos = _wn_lemma(token, "v"), None
    elif p in _FUNCTION_POS or p == "PROPN":
        lemma, wn_pos = token, None
    else:
        wn_pos = _WN_POS.get(p) or ("v" if role == "PREDICATE" else None)
        lemma = _wn_lemma(token, wn_pos)
    senses: List[str] = []
    if p not in _FUNCTION_POS and p != "PROPN" and token.isalpha():
        senses = list(_wn_senses(lemma, wn_pos))
    return Filler(role=role.upper(), token=token, lemma=lemma, token_index=idx, senses=senses)


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
        """Rewrite the cache file compactly (the cache is already write-through)."""
        if not self.cache_path:
            return
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for k, v in self._cache.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        os.replace(tmp, self.cache_path)

    @staticmethod
    def _clause_from_dict(c: dict) -> ClauseTree:
        return ClauseTree(predicate=Filler(**c["predicate"]), args=[Filler(**a) for a in c["args"]],
                          negated=c["negated"], source=c["source"], sentence=c["sentence"],
                          sentence_index=c["sentence_index"])

    @classmethod
    def _result_from_dict(cls, d: dict) -> ReadResult:
        return ReadResult(sentence=d["sentence"], clauses=[cls._clause_from_dict(c) for c in d["clauses"]],
                          source=d["source"], seconds=d["seconds"])

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
            self._usvs = load_usvs(str(CT_DIR / "data" / "usvs"))
            self._enc = (model, ckpt, em)
        return self._enc

    def tag(self, norm: str) -> Tuple[List[str], List[str]]:
        """(tokens, UPOS tags) of a normalised sentence, from the pipeline's own tagger."""
        words = self._get_parser()._tag(norm)
        return [w.text.lower() for w in words], [w.pos.name for w in words]

    # ---- parser path -------------------------------------------------------
    def _raw_parse(self, norm: str) -> dict:
        """The parser's own output, JSON-able and cached: its discourse clauses
        (predicate, [(relation, token)]), discourse links, and the flat hypothesis graph.
        ``{"fail": True}`` when the parser raised, hit the cap, or produced no clause."""
        key = "raw|" + norm
        if key in self._cache:
            return self._cache[key]
        from nsm_ct.clause import extract_discourse
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
            clauses, links = extract_discourse(graph)
            if clauses:
                out = {"fail": False,
                       "clauses": [[c.predicate if isinstance(c.predicate, str) else c.predicate.token,
                                    [[r, n.token] for r, n in c.args]] for c in clauses],
                       "links": [[l.coordinator, l.i, l.j] for l in links],
                       "nodes": [list(n) for n in graph.nodes],
                       "edges": [list(e) for e in graph.edges]}
        except BaseException as exc:  # noqa: BLE001 -- cap hit, no hypothesis, anything
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
        self._cache_put(key, out)
        return out

    def _from_parser(self, raw: dict, tokens: List[str], pos: List[str]) -> List[ClauseTree]:
        """Carry the parser's output through under its own labels: its clauses and args, the
        NOT link as ``negated``, and every hypothesis-graph edge the clause extractor did
        not already surface (attached to the clause holding the edge's parent node; a
        parent with SUBJECT/OBJECT edges but no clause becomes its own clause)."""
        gtok = {n[0]: (n[2] or "").lower() for n in raw["nodes"]}
        used: set = set()

        def locate(tok):
            tok = (tok or "").lower()
            cands = [i for i, t in gtok.items() if t == tok and i not in used and 0 <= i < len(tokens)]
            cands = cands or [i for i, t in enumerate(tokens) if t == tok and i not in used]
            cands = cands or [i for i, t in enumerate(tokens) if t == tok]
            if not cands:
                return None
            used.add(cands[0])
            return cands[0]

        out: List[ClauseTree] = []
        for pred, args in raw["clauses"]:
            pf = make_filler("PREDICATE", pred, locate(pred), pos)
            c = ClauseTree(predicate=pf, source="parser")
            for rel, tok in args:
                if tok:
                    c.args.append(make_filler(rel, tok, locate(tok), pos))
            out.append(c)
        for coord, i, j in raw["links"]:
            if coord == "NOT":
                for k in {i, j}:
                    if 0 <= k < len(out):
                        out[k].negated = True

        def held(c):
            return {c.predicate.token_index} | {a.token_index for a in c.args}

        by_parent: Dict[int, list] = {}
        for etype, p, ch in raw["edges"]:
            by_parent.setdefault(p, []).append((etype, ch))
        for p, edges in by_parent.items():
            owner = next((c for c in out if c.predicate.token_index == p), None) \
                or next((c for c in out if p in held(c)), None)
            if owner is None and 0 <= p < len(tokens):
                # a modifier node pointing AT a clause predicate (parser: MODIFICATION edge)
                tgt = [c for c in out for _e, ch in edges if c.predicate.token_index == ch]
                if tgt and not any(e in ("SUBJECT", "OBJECT") for e, _ in edges):
                    lab = next((n[1] for n in raw["nodes"] if n[0] == p), "MODIFIER")
                    for c in tgt:
                        if p not in held(c):
                            c.args.append(make_filler(lab, tokens[p], p, pos))
                    continue
            if owner is None:
                if not any(e in ("SUBJECT", "OBJECT") for e, _ in edges) or not 0 <= p < len(tokens):
                    continue
                owner = ClauseTree(predicate=make_filler("PREDICATE", tokens[p], p, pos), source="parser")
                out.append(owner)
            for etype, ch in edges:
                if 0 <= ch < len(tokens) and ch not in held(owner) and pos[ch] != "PUNCT":
                    owner.args.append(make_filler(etype, tokens[ch], ch, pos))
        return out

    # ---- encoder path ------------------------------------------------------
    def _from_encoder(self, norm: str, tokens: List[str], pos: List[str]) -> List[ClauseTree]:
        model, ckpt, em = self._get_encoder()
        sc = [{"index": i, "token": t, "sense_candidates": list(c)}
              for i, t in enumerate(tokens) for c in [self._usvs.senses_of(t)] if c]
        rec = {"text": norm, "tokens": tokens, "pos": pos, "token_sense_candidates": sc}
        feats = em.build_features(rec, self._usvs, ckpt["pos_vocab"], ckpt["hash_buckets"])
        forest = em.beam_decode(model, feats, beam_width=8, k=8, policy="model")
        out = []
        for cl in (forest[0] if forest else {"clauses": []}).get("clauses", []):
            pi = cl.get("predicate", {}).get("token_index")
            if pi is None or not (0 <= pi < len(tokens)):
                continue
            c = ClauseTree(predicate=make_filler("PREDICATE", tokens[pi], pi, pos), source="encoder")
            for r in cl.get("roles", []):
                ai = r.get("token_index")
                if ai is not None and 0 <= ai < len(tokens):
                    c.args.append(make_filler(r.get("relation", "ARG"), tokens[ai], ai, pos))
            out.append(c)
        return out

    # ---- public ------------------------------------------------------------
    def _finish(self, sentence, clauses, source, sentence_index, t0) -> ReadResult:
        for c in clauses:
            c.sentence, c.sentence_index, c.source = sentence, sentence_index, source
        return ReadResult(sentence=sentence, clauses=clauses, source=source if clauses else "none",
                          seconds=time.perf_counter() - t0)

    def _cached(self, kind: str, sentence: str, sentence_index, build) -> ReadResult:
        key = f"{kind}|{sentence}"
        if key not in self._cache:
            self._cache_put(key, dataclasses.asdict(build()))
        rr = self._result_from_dict(self._cache[key])
        for c in rr.clauses:
            c.sentence_index = sentence_index
        return rr

    def read_parser(self, sentence: str, sentence_index=None) -> Optional[ReadResult]:
        """Parser-only reading, or None if the parser failed (error, cap, no clause)."""
        def build():
            t0, norm = time.perf_counter(), normalize(sentence)
            raw = self._raw_parse(norm)
            if raw.get("fail"):
                return ReadResult(sentence=sentence, source="none", seconds=time.perf_counter() - t0)
            tokens, pos = self.tag(norm)
            return self._finish(sentence, self._from_parser(raw, tokens, pos), "parser", None, t0)
        rr = self._cached("p", sentence, sentence_index, build)
        return rr if rr.clauses else None

    def read_encoder(self, sentence: str, sentence_index=None) -> ReadResult:
        """Encoder-only reading (rank-1 beam tree)."""
        def build():
            t0, norm = time.perf_counter(), normalize(sentence)
            tokens, pos = self.tag(norm)
            try:
                clauses = self._from_encoder(norm, tokens, pos)
            except Exception:
                clauses = []
            return self._finish(sentence, clauses, "encoder", None, t0)
        return self._cached("e", sentence, sentence_index, build)

    def read_both(self, sentence: str, sentence_index=None) -> Tuple[Optional[ReadResult], ReadResult]:
        return self.read_parser(sentence, sentence_index), self.read_encoder(sentence, sentence_index)

    def read(self, sentence: str, sentence_index=None) -> ReadResult:
        """Parser tree; the encoder's rank-1 tree if the parser fails."""
        if self.use_parser:
            rr = self.read_parser(sentence, sentence_index)
            if rr is not None:
                return rr
        if self.use_encoder:
            rr = self.read_encoder(sentence, sentence_index)
            if rr.clauses:
                return rr
        return ReadResult(sentence=sentence, source="none")

    def read_question(self, question: str) -> ReadResult:
        """Alias of :meth:`read`: questions get no special treatment here."""
        return self.read(question)

