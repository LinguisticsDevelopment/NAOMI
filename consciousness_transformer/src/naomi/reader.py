"""Reader: sentence -> ReadResult (ClauseTrees with uncommitted sense candidates).

Hybrid: the deterministic grammar parser (the "teacher") is tried first under a
wall-clock cap; on failure the learned encoder's rank-1 tree is used instead.
Both outputs are converted to :class:`naomi.types.ClauseTree`; a deterministic
post-pass adds negation (both readers drop it) and a question pass marks the
wh-slot as the hole.

Does NOT import the retired ``nsm_ct.clause_reactor`` / ``corpus`` /
``curriculum2`` / ``episode`` modules.
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
NEG_CUES = {"not", "n't", "never", "no", "nothing", "nobody", "none", "neither", "nor",
            "cannot", "without", "nowhere", "n’t"}
AUX_BARE = {"do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "may", "might", "must"}
WH_WORDS = {"who", "what", "where", "when", "which", "whom", "whose", "why", "how"}
DETERMINERS = {"the", "a", "an"}
FUNCTION_POS = {"AUX", "DET", "ADP", "PART", "PRON", "CCONJ", "SCONJ", "PUNCT", "NUM", "SYM", "INTJ"}
_WH_ADV_ROLE = {"where": "PLACE", "when": "TIME", "why": "REASON", "how": "MANNER"}
_WN_POS = {"NOUN": "n", "PROPN": None, "VERB": "v", "ADJ": "a", "ADV": "r"}


def _norm(text: str) -> str:
    """Light tokenisation for raw (untokenised) text: lowercase, split final/inner
    punctuation and the n't contraction the way the parser's tagger expects."""
    import re
    t = text.strip().lower().replace("\u2019", "'")
    t = re.sub(r"(\w)'s\b", r"\1 's", t)  # possessive / "'s" clitic -> own token
    t = re.sub(r"\bcan't\b", "can n't", t)
    t = re.sub(r"\bwon't\b", "will n't", t)
    t = re.sub(r"\b(\w+)n't\b", r"\1 n't", t)
    t = re.sub(r"([.?!,;:\"])", r" \1 ", t)
    return re.sub(r"\s+", " ", t).strip()


class _ParserTimeout(Exception):
    pass


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
            ss = list(wn.synsets(lemma, pos="a" if pos == "a" else pos))
            if pos == "a":
                ss += list(wn.synsets(lemma, pos="s"))
        rest = [s for s in wn.synsets(lemma) if s not in ss]
        return tuple(s.name() for s in (ss + rest)[:5])
    except Exception:
        return ()


# --------------------------------------------------------------------------- conversion
class Reader:
    def __init__(self, lang: str = "en", encoder_ckpt: str = "runs/arms/v6_all_hard_seed0.pt",
                 parser_seconds: float = 5.0, cache_path: Optional[str] = None,
                 use_parser: bool = True, use_encoder: bool = True):
        self.lang = lang
        self.encoder_ckpt = self._resolve(encoder_ckpt)
        self.parser_seconds = float(parser_seconds)
        self.use_parser = use_parser
        self.use_encoder = use_encoder
        self.cache_path = self._resolve(cache_path) if cache_path else None
        self._parser = None
        self._usvs = None
        self._enc = None  # (model, ckpt, em)
        self._cache: Dict[str, dict] = {}
        self._cache_dirty = False
        self._cfg = f"p{int(use_parser)}e{int(use_encoder)}"
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    # ---- paths / cache -----------------------------------------------------
    @staticmethod
    def _resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else CT_DIR / p

    def _key(self, kind: str, sentence: str) -> str:
        return f"{kind}|{self._cfg}|{sentence}"

    def _load_cache(self) -> None:
        with open(self.cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
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
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for k, v in self._cache.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        os.replace(tmp, self.cache_path)

    @staticmethod
    def _result_from_dict(d: dict) -> ReadResult:
        def fl(x):
            return Filler(**x)
        clauses = [ClauseTree(predicate=fl(c["predicate"]), args=[fl(a) for a in c["args"]],
                              negated=c["negated"], source=c["source"], sentence=c["sentence"],
                              sentence_index=c["sentence_index"]) for c in d["clauses"]]
        return ReadResult(sentence=d["sentence"], clauses=clauses, source=d["source"], seconds=d["seconds"])

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
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            ckpt = torch.load(str(self.encoder_ckpt), map_location="cpu", weights_only=False)
            model = em.EncoderModel(ckpt["pos_vocab"], ckpt["role_vocab"], d_axes=ckpt["d_axes"],
                                    hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                                    controller_hidden=ckpt["d_model"])
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._usvs = load_usvs(str(CT_DIR / "data" / "usvs"))
            self._enc = (model, ckpt, em)
        return self._enc

    # ---- parser path -------------------------------------------------------
    def _tag(self, sentence: str):
        words = self._get_parser()._tag(sentence)
        return [w.text.lower() for w in words], [w.pos.name for w in words]

    def _parse_clauses(self, sentence: str):
        """Returns clause list or None (failure). Hard wall-clock cap."""
        from nsm_ct.clause import extract_discourse
        parser = self._get_parser()
        if getattr(parser, "_parser", None) is None:
            return None
        use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")
        old = None
        if use_alarm:
            def _h(signum, frame):
                raise _ParserTimeout()
            old = signal.signal(signal.SIGALRM, _h)
            signal.setitimer(signal.ITIMER_REAL, self.parser_seconds + 1.0)
        try:
            graph = parser._parse_graph(sentence, max_hypotheses=PARSER_MAX_HYPOTHESES,
                                        max_seconds=self.parser_seconds)
            clauses, _links = extract_discourse(graph)
            return clauses
        except BaseException as exc:  # noqa: BLE001 -- cap hit, no hypothesis, anything
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)

    def _filler(self, role, token, idx, pos) -> Filler:
        token = (token or "").lower()
        p = pos[idx] if idx is not None and 0 <= idx < len(pos) else None
        wn_pos = _WN_POS.get(p) if p else None
        if role == "PREDICATE" and wn_pos is None and p not in FUNCTION_POS and p != "PROPN":
            wn_pos = "v"
        if p == "AUX":
            lemma = _wn_lemma(token, "v")
        else:
            lemma = token if p in FUNCTION_POS or p == "PROPN" else _wn_lemma(token, wn_pos)
        senses: List[str] = []
        if p not in FUNCTION_POS and p != "PROPN" and token.isalpha() and token not in WH_WORDS:
            senses = list(_wn_senses(lemma, wn_pos))
        # the verb "be"/"do"/"have" as predicate keep senses (cheap); aux pos excluded above
        return Filler(role=role.upper(), token=token, lemma=lemma, token_index=idx, senses=senses)

    def _from_parser(self, sentence: str, pclauses, tokens, pos) -> List[ClauseTree]:
        used = set()

        def locate(tok, after=-1):
            tok = (tok or "").lower()
            cands = [i for i, t in enumerate(tokens) if t == tok and i not in used]
            if not cands:
                cands = [i for i, t in enumerate(tokens) if t == tok]
            if not cands:
                return None
            later = [i for i in cands if i > after]
            i = (later or cands)[0]
            used.add(i)
            return i

        out = []
        for c in pclauses:
            pred_tok = c.predicate if isinstance(c.predicate, str) else getattr(c.predicate, "token", "")
            pi = locate(pred_tok)
            pred = self._filler("PREDICATE", pred_tok, pi, pos)
            args, seen = [], set()
            for rel, node in c.args:
                tk = (node.token or "").lower()
                if not tk or (rel == "DESCRIPTION" and tk in DETERMINERS):
                    continue
                if (rel, tk) in seen:
                    continue
                seen.add((rel, tk))
                ai = locate(tk)
                args.append(self._filler(rel, tk, ai, pos))
            out.append(ClauseTree(predicate=pred, args=args, source="parser", sentence=sentence))
        return out

    # ---- encoder path ------------------------------------------------------
    def _from_encoder(self, sentence: str, tokens, pos) -> List[ClauseTree]:
        model, ckpt, em = self._get_encoder()
        sc = []
        for i, t in enumerate(tokens):
            c = self._usvs.senses_of(t)
            if c:
                sc.append({"index": i, "token": t, "sense_candidates": list(c)})
        rec = {"text": sentence, "tokens": tokens, "pos": pos, "token_sense_candidates": sc}
        feats = em.build_features(rec, self._usvs, ckpt["pos_vocab"], ckpt["hash_buckets"])
        forest = em.beam_decode(model, feats, beam_width=8, k=8, policy="model")
        top = forest[0] if forest else {"clauses": []}
        out = []
        for cl in top.get("clauses", []):
            pnode = cl.get("predicate", {})
            pi = pnode.get("token_index")
            if pi is None or not (0 <= pi < len(tokens)):
                continue
            pred = self._filler("PREDICATE", tokens[pi], pi, pos)
            args = []
            for r in cl.get("roles", []):
                ai = r.get("token_index")
                if ai is None or not (0 <= ai < len(tokens)):
                    continue
                if r.get("relation") == "DESCRIPTION" and tokens[ai] in DETERMINERS:
                    continue
                args.append(self._filler(r.get("relation", "ARG"), tokens[ai], ai, pos))
            out.append(ClauseTree(predicate=pred, args=args, source="encoder", sentence=sentence))
        return out

    # ---- post-passes -------------------------------------------------------
    def _repair_bare_aux(self, clauses: List[ClauseTree], tokens, pos) -> List[ClauseTree]:
        """Parser quirk: "the dog did not eat the bone" -> did(dog). Re-head on the
        main verb (first later VERB-tagged token) and take the next nominal as OBJECT.
        Chosen over falling back to the encoder because it is deterministic and cheap."""
        out = []
        taken = {c.predicate.token_index for c in clauses}
        for c in clauses:
            p = c.predicate
            if p.lemma in AUX_BARE or p.token in AUX_BARE:
                pi = p.token_index if p.token_index is not None else -1
                vi = next((i for i in range(pi + 1, len(tokens))
                           if pos[i] == "VERB" and i not in taken), None)
                if vi is not None:
                    newp = self._filler("PREDICATE", tokens[vi], vi, pos)
                    c.predicate = newp
                    have_obj = c.arg("OBJECT") is not None
                    if not have_obj:
                        ni = next((i for i in range(vi + 1, len(tokens))
                                   if pos[i] in ("NOUN", "PROPN", "PRON")), None)
                        if ni is not None and ni not in {a.token_index for a in c.args}:
                            c.args.append(self._filler("OBJECT", tokens[ni], ni, pos))
                    taken.add(vi)
            out.append(c)
        return out

    def _negation(self, clauses: List[ClauseTree], tokens) -> None:
        cues = [i for i, t in enumerate(tokens) if t in NEG_CUES]
        if not cues or not clauses:
            return
        spans = []
        for c in clauses:
            idxs = [c.predicate.token_index] + [a.token_index for a in c.args]
            idxs = [i for i in idxs if i is not None]
            spans.append((min(idxs), max(idxs), c.predicate.token_index) if idxs else None)
        for cue in cues:
            if len(clauses) == 1:
                clauses[0].negated = True
                continue
            best, bkey = None, None
            for ci, sp in enumerate(spans):
                if sp is None:
                    continue
                lo, hi, pi = sp
                d = 0 if lo <= cue <= hi else min(abs(cue - lo), abs(cue - hi))
                key = (d, abs(cue - pi) if pi is not None else 99)
                if bkey is None or key < bkey:
                    best, bkey = ci, key
            if best is not None:
                clauses[best].negated = True

    # ---- public API --------------------------------------------------------
    def _finish(self, sentence, clauses, tokens, pos, sentence_index):
        clauses = self._repair_bare_aux(clauses, tokens, pos)
        self._negation(clauses, tokens)
        for c in clauses:
            c.sentence_index = sentence_index
        return clauses

    def _read_raw(self, sentence: str, sentence_index=None) -> ReadResult:
        t0 = time.perf_counter()
        orig, sentence = sentence, _norm(sentence)
        tokens, pos = self._tag(sentence)
        pk = self._key("pfail", sentence)
        clauses: List[ClauseTree] = []
        source = "none"
        parser_clauses: List[ClauseTree] = []
        if self.use_parser and pk not in self._cache:
            pc = self._parse_clauses(sentence)
            if pc:
                parser_clauses = self._from_parser(sentence, pc, tokens, pos)
            ok = any(c.arg("SUBJECT") is not None for c in parser_clauses)
            if ok:
                clauses, source = parser_clauses, "parser"
            else:
                self._cache_put(pk, {"fail": True})
        if source == "none" and self.use_encoder:
            try:
                clauses = self._from_encoder(sentence, tokens, pos)
            except Exception:
                clauses = []
            if clauses:
                source = "encoder"
        if source == "none" and parser_clauses:
            clauses, source = parser_clauses, "parser"
        clauses = self._finish(sentence, clauses, tokens, pos, sentence_index)
        for c in clauses:
            c.source = source
            c.sentence = orig
        return ReadResult(sentence=orig, clauses=clauses, source=source,
                          seconds=time.perf_counter() - t0)

    def read(self, sentence: str, sentence_index=None) -> ReadResult:
        key = self._key("s", sentence)
        if key in self._cache:
            rr = self._result_from_dict(self._cache[key])
            for c in rr.clauses:
                c.sentence_index = sentence_index
            return rr
        rr = self._read_raw(sentence, sentence_index)
        self._cache_put(key, dataclasses.asdict(rr))
        return rr

    def read_question(self, question: str) -> ReadResult:
        key = self._key("q", question)
        if key in self._cache:
            return self._result_from_dict(self._cache[key])
        t0 = time.perf_counter()
        rr = self._read_raw(question)
        tokens, pos = self._tag(_norm(question))
        rr.clauses = self._mark_hole(question, rr.clauses, tokens, pos)
        rr.seconds = time.perf_counter() - t0
        self._cache_put(key, dataclasses.asdict(rr))
        return rr

    # ---- question hole -----------------------------------------------------
    def _mark_hole(self, question, clauses, tokens, pos) -> List[ClauseTree]:
        wh_i = next((i for i, t in enumerate(tokens) if t in WH_WORDS), None)
        if wh_i is None:
            return clauses[:1] if clauses else clauses
        wh = tokens[wh_i]
        src = clauses[0].source if clauses else "none"

        def is_wh(f):
            return f.token == wh and f.token_index in (wh_i, None)

        # main clause: first non-bare-aux predicate (copula "is" counts as main if nothing else)
        def bare_aux(c):
            return c.predicate.lemma in AUX_BARE or c.predicate.token in AUX_BARE

        main = next((c for c in clauses if not bare_aux(c) and c.predicate.token != "have"
                     or (not bare_aux(c) and len(clauses) == 1)), None)
        if main is None:
            main = clauses[0] if clauses else None
        if main is None:
            # nothing parsed: bare tree from the wh word alone
            hole = Filler(role=_WH_ADV_ROLE.get(wh, "SUBJECT"), token=wh, lemma=wh,
                          token_index=wh_i, is_hole=True)
            pi = next((i for i, p in enumerate(pos) if p == "VERB"), None)
            if pi is None:
                return []
            pred = self._filler("PREDICATE", tokens[pi], pi, pos)
            return [ClauseTree(predicate=pred, args=[hole], source="none", sentence=question)]

        # gather wh filler from any clause (remove it from where it was)
        wh_f = None
        for c in clauses:
            for a in list(c.args):
                if is_wh(a) or (a.token == wh and wh_f is None):
                    wh_f = a
                    c.args.remove(a)
        # drop other clauses' stray aux-only shells; keep only main
        # (do-support shells are bare auxiliaries with no content once wh is removed)
        main.negated = main.negated or any(c.negated for c in clauses if c is not main and bare_aux(c))
        # re-head do-support / bare aux in main
        if bare_aux(main):
            pass  # already repaired in _finish when a later verb exists
        # "what did X do" with no verb after the aux: re-head on the final "do"
        if bare_aux(main) and wh == "what":
            di = next((i for i in range(len(tokens) - 1, -1, -1) if tokens[i] == "do"
                       and i != main.predicate.token_index), None)
            if di is not None:
                subj = main.arg("SUBJECT")
                if subj is None:
                    si = next((i for i in range((main.predicate.token_index or 0) + 1, di)
                               if pos[i] in ("NOUN", "PROPN", "PRON")), None)
                    if si is not None:
                        main.args.append(self._filler("SUBJECT", tokens[si], si, pos))
                main.predicate = self._filler("PREDICATE", "do", di, pos)
        # "what did X do": predicate is the verb "do"
        if (main.predicate.lemma == "do" and wh in ("what",)) and (
                wh_f is not None or True):
            main.predicate.is_hole = True
            main.predicate.role = "PREDICATE"
            main.args = [a for a in main.args if a.role not in ("OBJECT",) or not is_wh(a)]
            main.source = src
            return [main]

        if wh in _WH_ADV_ROLE:
            role = _WH_ADV_ROLE[wh]
            # re-role an existing wh-adverb filler (parser: SPECIFICATION where)
            existing = next((a for a in main.args if a.token == wh), None)
            if existing is not None:
                existing.role, existing.is_hole = role, True
            else:
                main.args.append(Filler(role=role, token=wh, lemma=wh, token_index=wh_i, is_hole=True))
        else:
            wf = wh_f or Filler(role="SUBJECT", token=wh, lemma=wh, token_index=wh_i)
            has_subj = any(a.role == "SUBJECT" for a in main.args)
            next_is_verb = wh_i + 1 < len(tokens) and wh_i + 1 == main.predicate.token_index
            if wh == "whose":
                role = "DESCRIPTION"
            elif not has_subj or next_is_verb:
                role = "SUBJECT"
            else:
                role = "OBJECT"
                if any(a.role == "OBJECT" for a in main.args):
                    role = "INDIRECT_OBJECT"
            wf.role, wf.is_hole = role, True
            wf.senses = []
            main.args = [a for a in main.args if a.role != role or a.is_hole] if role == "SUBJECT" else main.args
            main.args.append(wf)
        # a hole must be unique
        seen = False
        for a in main.args:
            if a.is_hole:
                if seen:
                    a.is_hole = False
                seen = True
        main.source = src
        return [main]
