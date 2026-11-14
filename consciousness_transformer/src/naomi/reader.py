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
import hashlib
import json
import os
import re
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
# version of the post-pass code: part of the key of FINAL cached results (raw parser
# output is cached independently, so reader post-pass edits never force a re-parse)
_CODE_VER = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:8]
COPULAS = {"be", "become", "seem", "look", "feel", "appear", "remain", "call", "name"}
_NUMWORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty "
    "ninety hundred thousand million dozen".split())
_QUANT_WORDS = frozenset("several few many both half twice once dozens hundreds thousands".split())
_CLITIC_HOSTS = frozenset("it he she that there here what who where how let".split())
_LOCATIVE = frozenset("in at on to into from near under over behind inside outside by beside "
                      "through across around toward towards onto off out up down".split())
AUX_ALL = AUX_BARE | {"is", "are", "was", "were", "am", "be", "been", "being", "have", "has", "had"}
_ATTR_NOUNS = frozenset("color colour kind type sort size shape name".split())
_QUANT_ADJ = frozenset("long old far tall big high fast often wide deep heavy large small short "
                       "much many".split())
_SUBORD = frozenset("when after before because while as if until since although though".split())
_TITLES = frozenset("mr mrs ms dr".split())
_CLAUSE_MARKS = frozenset("and but so that because while when after before until if although though then".split())
_SUBJECTISH = frozenset("he she they it i we you the a an his her their".split())


def _is_num(t: str) -> bool:
    if t in _NUMWORDS:
        return True
    if re.fullmatch(r"\$?\d[\d,.:]*(st|nd|rd|th|%|s)?", t):
        return True
    parts = t.split("-")
    return len(parts) > 1 and all(p in _NUMWORDS for p in parts)


def _norm(text: str, lower: bool = True) -> str:
    """Light tokenisation for raw (untokenised) text: lowercase, split final/inner
    punctuation and the n't contraction the way the parser's tagger expects."""
    t = text.strip().replace("\u2019", "'")
    if lower:
        t = t.lower()
    t = re.sub(r"(\w)'s\b", r"\1 's", t)  # possessive / "'s" clitic -> own token
    t = re.sub(r"\b(mr|mrs|ms|dr)\.", r"\1", t)  # titles keep their period out of the way
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
        self._cfg = f"p{int(use_parser)}e{int(use_encoder)}v3"
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    # ---- paths / cache -----------------------------------------------------
    @staticmethod
    def _resolve(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else CT_DIR / p

    def _key(self, kind: str, sentence: str) -> str:
        if kind in ("s", "q"):  # final results depend on this file's post-pass code
            return f"{kind}|{self._cfg}|{_CODE_VER}|{sentence}"
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
                if k[:2] in ("s|", "q|") and f"|{_CODE_VER}|" not in k:
                    continue  # stale final result from an older post-pass version
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        os.replace(tmp, self.cache_path)

    @staticmethod
    def _clause_from_dict(c: dict) -> ClauseTree:
        def fl(x):
            return Filler(**x)
        return ClauseTree(predicate=fl(c["predicate"]), args=[fl(a) for a in c["args"]],
                          negated=c["negated"], source=c["source"], sentence=c["sentence"],
                          sentence_index=c["sentence_index"])

    @classmethod
    def _result_from_dict(cls, d: dict) -> ReadResult:
        clauses = [cls._clause_from_dict(c) for c in d["clauses"]]
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
    def _tag(self, sentence: str, orig: Optional[str] = None):
        """(tokens, pos) of the normalised sentence. The tagger sees lowercase text, so a
        capitalised non-initial word ("Mark", "Tim") is re-tagged PROPN when ``orig`` is given."""
        words = self._get_parser()._tag(sentence)
        tokens, pos = [w.text.lower() for w in words], [w.pos.name for w in words]
        if orig is not None:
            ot = _norm(orig, lower=False).split()
            if len(ot) == len(tokens) and all(a.lower() == b for a, b in zip(ot, tokens)):
                for i in range(1, len(tokens)):
                    if ot[i][:1].isupper() and pos[i] in ("NOUN", "VERB", "ADJ", "ADV", "AUX", "X"):
                        pos[i] = "PROPN"
        return tokens, pos

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

    # ---- surface passes (deterministic, token/POS driven; the parser is noisy) ---------
    @staticmethod
    def _clause_with(clauses, idx):
        for c in clauses:
            if c.predicate.token_index == idx or any(a.token_index == idx for a in c.args):
                return c
        return None

    @staticmethod
    def _np_head(tokens, pos, start):
        """Index of the head noun of the NP that begins at ``start`` (None if no NP)."""
        j, head = start, None
        while j < len(tokens):
            p = pos[j]
            if p == "DET":
                if head is not None:
                    break
            elif p == "ADJ" or p == "NUM":
                if head is not None:
                    break
            elif p in ("NOUN", "PROPN"):
                if head is not None and p == "PROPN" and pos[head] == "NOUN":
                    break
                head = j
            else:
                break
            j += 1
        return head

    def _possessives(self, clauses, tokens, pos) -> None:
        """"jim 's father": keep the possessor as a POSSESSOR filler on the clause that
        holds the possessed noun (re-attaching the noun if the parser dropped it)."""
        if not clauses:
            return
        for i, t in enumerate(tokens):
            if t != "'s" or i == 0 or pos[i - 1] in ("PRON", "PUNCT", "AUX", "DET", "ADP", "CCONJ", "SCONJ", "PART") \
                    or tokens[i - 1] in _CLITIC_HOSTS:
                continue
            n = self._np_head(tokens, pos, i + 1)
            if n is None:
                continue
            p, holder, prole = i - 1, None, None
            for c in clauses:
                for a in list(c.args):
                    if a.token_index in (p, i):
                        holder, prole = c, prole or a.role
                        c.args.remove(a)
            tgt = self._clause_with(clauses, n)
            if tgt is None:
                tgt = holder or clauses[0]
                tgt.args.append(self._filler(prole or "OBJECT", tokens[n], n, pos))
            tgt.args.append(self._filler("POSSESSOR", tokens[p], p, pos))

    def _copulas(self, clauses, tokens, pos) -> None:
        """Keep the predicate nominal / adjective of be, become, seem, look, feel, call,
        name as an ATTRIBUTE filler (both readers drop it)."""
        for c in clauses:
            pr = c.predicate
            if pr.lemma not in COPULAS or pr.token_index is None:
                continue
            for a in c.args:
                if a.role == "COMPLEMENT":
                    a.role = "ATTRIBUTE"
            if any(a.role in ("OBJECT", "ATTRIBUTE") for a in c.args):
                continue
            j = pr.token_index + 1
            while j < len(tokens) and (pos[j] in ("PART", "ADV", "DET")
                                       or tokens[j] in ("so", "very", "too", "really", "quite")):
                j += 1
            if j < len(tokens) and tokens[j] in ("named", "called") and pr.lemma == "be":
                j += 1
            elif pr.lemma in ("look", "seem", "feel", "appear") and j < len(tokens) and tokens[j] in ("like", "as"):
                j += 1
                while j < len(tokens) and pos[j] == "DET":
                    j += 1
            if j >= len(tokens) or pos[j] not in ("ADJ", "NOUN", "PROPN", "NUM"):
                continue
            k = j
            while k + 1 < len(tokens) and pos[k + 1] in ("ADJ", "NOUN", "PROPN"):
                k += 1
            if any(a.token_index == k for a in c.args):
                continue
            c.args.append(self._filler("ATTRIBUTE", tokens[k], k, pos))

    def _quantities(self, clauses, tokens, pos) -> None:
        """Numerals / quantifiers become a QUANTITY filler on the clause that holds the
        counted noun (the noun keeps its own role, the unit is found by token index)."""
        if not clauses:
            return
        for i, t in enumerate(tokens):
            if not (_is_num(t) or t in _QUANT_WORDS) or pos[i] in ("PUNCT", "VERB", "AUX", "ADP"):
                continue
            if t == "one" and (i == 0 or tokens[i - 1] in {".", ",", "!", "?", "and", "or", "of"}):
                continue
            j = i + 1
            while j < len(tokens) and pos[j] in ("ADJ", "NUM"):
                j += 1
            if j >= len(tokens) or pos[j] != "NOUN":
                continue
            tgt = self._clause_with(clauses, j)
            if tgt is None:
                continue
            for c in clauses:
                c.args = [a for a in c.args if a.token_index != i]
            tgt.args.append(self._filler("QUANTITY", t, i, pos))

    # ---- surface question builder --------------------------------------------------
    _PERSONAL = frozenset("he she they it i we you him them me us".split())

    def _nps(self, tokens, pos, lo, hi):
        """Noun phrases in tokens[lo..hi]: dicts head/poss/adjs/prep (token indices)."""
        out, i = [], lo
        while i <= hi:
            p = pos[i]
            prep = i - 1 if i > lo and pos[i - 1] == "ADP" else None
            if p == "PRON" and tokens[i] in self._PERSONAL:
                out.append({"head": i, "poss": None, "adjs": [], "prep": prep})
                i += 1
                continue
            if p not in ("DET", "ADJ", "NUM", "NOUN", "PROPN"):
                i += 1
                continue
            j, head, poss, adjs = i, None, None, []
            while j <= hi:
                q = pos[j]
                if q == "DET":
                    if head is not None:
                        break
                elif q == "ADJ":
                    if head is not None:
                        break
                    adjs.append(j)
                elif q == "NUM":
                    pass
                elif q in ("NOUN", "PROPN"):
                    if head is not None and q == "PROPN" and pos[head] == "NOUN":
                        break
                    head = j
                    if j + 1 <= hi and tokens[j + 1] == "'s":
                        poss, head, adjs, j = j, None, [], j + 1
                else:
                    break
                j += 1
            if head is None and poss is not None:
                head, poss = poss, None
            if head is not None:
                out.append({"head": head, "poss": poss, "adjs": adjs, "prep": prep})
            i = max(j, i + 1)
        return out

    def _surface_question(self, question, tokens, pos) -> Optional[ClauseTree]:
        n = len(tokens)
        end = max((i for i in range(n) if pos[i] != "PUNCT"), default=-1)
        wh_i = next((i for i, t in enumerate(tokens) if t in WH_WORDS), None)
        if wh_i is None or wh_i > 3 or end < 0:
            return None
        wh = tokens[wh_i]
        mk = lambda role, i: self._filler(role, tokens[i], i, pos)  # noqa: E731
        args: List[Filler] = []
        j, kind, hole_role, cn = wh_i + 1, None, None, None
        if wh == "how":
            if j <= end and tokens[j] in ("many", "much") or (j <= end and tokens[j] in _QUANT_ADJ):
                hole_role, j = "QUANTITY", j + 1
                if tokens[j - 1] in ("many", "much") and j <= end and pos[j] == "NOUN":
                    cn, j = j, j + 1      # the counted noun, e.g. "how many SONS did ..."
            else:
                hole_role = "MANNER"
        elif wh in _WH_ADV_ROLE:
            hole_role = _WH_ADV_ROLE[wh]
        elif wh == "whose":
            hole_role = "POSSESSOR"
        elif wh in ("what", "which") and j <= end and pos[j] == "NOUN" and tokens[j] not in ("way",):
            kind, j = tokens[j], j + 1
            if kind in ("day", "time", "year", "month", "date", "week", "season", "hour"):
                hole_role, kind = "TIME", None
            elif kind in _ATTR_NOUNS:
                hole_role = "ATTRIBUTE"
        # subordinate clause -> optional CONTEXT
        sub = next((i for i in range(j, end + 1) if tokens[i] in _SUBORD and (
            pos[i] in ("SCONJ", "ADV") or any(pos[x] == "VERB" for x in range(i + 1, end + 1)))), None)
        main_end = (sub - 1) if sub is not None else end
        k, aux = j, None
        if k <= main_end and (pos[k] == "AUX" or tokens[k] in AUX_ALL):
            aux, k = k, k + 1
        verbs = [i for i in range(k, main_end + 1) if pos[i] == "VERB" or (
            pos[i] == "AUX" and tokens[i] in ("have", "has", "had", "do", "does", "did") and aux is not None and i > aux)]
        verbs = [i for i in verbs if not (pos[i] == "AUX" and tokens[i] in ("have", "has", "had")
                                          and any(x > i for x in verbs) and tokens[i + 1:i + 2] != ["to"])]
        be_form = aux is not None and tokens[aux] in ("is", "are", "was", "were", "am", "be", "been")
        if be_form and kind is None and wh in ("what", "which") and k <= main_end:
            # "what was the NAME of the cat tim met" -> copular, kind-noun in the NP
            kk = k + 1 if pos[k] == "DET" else k
            if kk <= main_end and tokens[kk] in _ATTR_NOUNS and pos[kk] == "NOUN":
                kind, hole_role = tokens[kk], "ATTRIBUTE"
                args.append(Filler(role="KIND", token=kind, lemma=kind, token_index=kk))
                k = kk + 1
                verbs = []
                main_end = next((i for i in range(k, main_end + 1) if pos[i] == "VERB"), main_end + 1) - 1
        elif kind in _ATTR_NOUNS and be_form:   # "what color was the bike tim wanted": copular
            verbs = []
        do_at = next((i for i in range(main_end, (aux or 0), -1) if tokens[i] == "do"
                      and not any(pos[x] == "VERB" and x != i for x in range(k, main_end + 1))), None)
        do_pred = (wh == "what" and aux is not None and tokens[aux] in ("do", "does", "did")
                   and do_at is not None and do_at > aux)
        if do_pred:
            main_end = do_at       # words after the final "do" ("everyday after school") are dropped
            verbs = []
        v = None if do_pred else (verbs[0] if verbs else None)
        copular = v is None and not do_pred
        if copular and (aux is None or tokens[aux] in AUX_BARE):
            return None
        stranded = None
        if pos[end] == "ADP" and end > wh_i and wh in ("who", "what", "which", "whom"):
            stranded = tokens[end]
        # predicate
        if do_pred:
            pred = self._filler("PREDICATE", "do", main_end, pos)
            pred.is_hole = True
            body_hi = main_end - 1
        elif copular:
            pred = self._filler("PREDICATE", tokens[aux], aux, pos)
            body_hi = main_end
        else:
            pred = self._filler("PREDICATE", tokens[v], v, pos)
            body_hi = main_end
        pre_hi = (v - 1) if v is not None else body_hi
        pre = self._nps(tokens, pos, k, pre_hi) if (aux is not None or v is None) else []
        subj_hole = False
        if not copular and not do_pred:
            subj_hole = aux is None or not pre
        if do_pred:
            subj_hole = False
        if cn is not None:
            args.append(mk("SUBJECT" if (subj_hole or copular) else "OBJECT", cn))
        # who/what hole role (non-adverbial)
        if hole_role is None:
            if copular:
                hole_role = "SUBJECT"
            elif stranded:
                hole_role = "PLACE" if stranded in _LOCATIVE else stranded.upper()
            else:
                hole_role = "SUBJECT" if subj_hole else "OBJECT"
        if v is not None and pred.lemma in ("look", "seem", "feel", "become") and hole_role in ("MANNER", "OBJECT", "SUBJECT") \
                and (wh == "how" or "like" in tokens[v:end + 1]):
            hole_role = "ATTRIBUTE"
        # fillers
        if do_pred:
            hole = pred
            for np_ in self._nps(tokens, pos, k, body_hi):
                args.append(mk("SUBJECT", np_["head"]))
                if np_["poss"] is not None:
                    args.append(mk("POSSESSOR", np_["poss"]))
        else:
            hole = Filler(role=hole_role, token=wh, lemma=wh, token_index=wh_i, is_hole=True)
            if kind and not any(x.role == "KIND" for x in args):
                args.append(Filler(role="KIND", token=kind, lemma=kind, token_index=wh_i + 1))
            if copular:
                nps = self._nps(tokens, pos, k, body_hi)
                attr_i = next((i for i in range(k, body_hi + 1) if (pos[i] == "ADJ" or (
                    pos[i] == "ADV" and tokens[i].endswith("er") and len(tokens[i]) > 4))
                    and not any(i in x["adjs"] or i == x["head"] for x in nps)), None)
                first, seen_subj = True, False
                for np_ in nps:
                    h = np_["head"]
                    if tokens[h] in _ATTR_NOUNS and hole_role not in _WH_ADV_ROLE.values():
                        kind = kind or tokens[h]
                        args.append(Filler(role="KIND", token=tokens[h], lemma=tokens[h], token_index=h))
                        hole.role = "ATTRIBUTE"
                        if np_["poss"] is not None:
                            args.append(mk("SUBJECT", np_["poss"]))
                        continue
                    prole = np_["prep"] is not None and tokens[np_["prep"]]
                    if hole_role in _WH_ADV_ROLE.values() or hole_role == "QUANTITY":
                        role = "SUBJECT" if first else "OBJECT"
                    elif hole.role == "ATTRIBUTE" and kind in _ATTR_NOUNS:
                        role = "SUBJECT"
                    elif prole == "of" and not first:
                        role = "POSSESSOR"
                    elif prole:
                        role = "PLACE" if prole in _LOCATIVE else prole.upper()
                    elif kind in _ATTR_NOUNS and hole.role == "ATTRIBUTE":
                        role = "SUBJECT"
                    else:
                        role = "OBJECT"
                    first = False
                    args.append(mk(role if (role != "SUBJECT" or not seen_subj) else "CONTEXT", h))
                    seen_subj = seen_subj or role == "SUBJECT"
                    if np_["poss"] is not None:
                        args.append(mk("POSSESSOR", np_["poss"]))
                    args.extend(mk("DESCRIPTION", a) for a in np_["adjs"])
                if attr_i is not None and not any(a.role == "ATTRIBUTE" for a in args) and hole.role != "ATTRIBUTE":
                    args.append(mk("ATTRIBUTE", attr_i))
                elif attr_i is not None and hole.role == "ATTRIBUTE":
                    args.append(mk("DESCRIPTION", attr_i))
            else:
                for np_ in pre:
                    args.append(mk("SUBJECT", np_["head"]))
                    if np_["poss"] is not None:
                        args.append(mk("POSSESSOR", np_["poss"]))
                for np_ in self._nps(tokens, pos, v + 1, body_hi):
                    prole = tokens[np_["prep"]] if np_["prep"] is not None else None
                    role = ("PLACE" if prole in _LOCATIVE else prole.upper()) if prole else "OBJECT"
                    args.append(mk(role, np_["head"]))
                    if np_["poss"] is not None:
                        args.append(mk("POSSESSOR", np_["poss"]))
                    args.extend(mk("DESCRIPTION", a) for a in np_["adjs"])
                for x in verbs[1:3]:
                    if x <= main_end:
                        args.append(mk("XCOMP", x))
        args.append(hole) if not do_pred else None
        if sub is not None:
            for i in range(sub + 1, end + 1):
                if pos[i] in ("NOUN", "PROPN") or (pos[i] == "VERB"):
                    args.append(mk("CONTEXT", i))
        # drop exact duplicates / the wh word itself
        seen, uniq = set(), []
        for a in args:
            key = (a.role, a.token_index, a.is_hole)
            if key in seen or (not a.is_hole and a.token == wh):
                continue
            seen.add(key)
            uniq.append(a)
        if not do_pred and not any(a.is_hole for a in uniq):
            return None
        neg = any(t in NEG_CUES for t in tokens)
        return ClauseTree(predicate=pred, args=uniq, negated=neg, source="parser", sentence=question)

    # ---- public API --------------------------------------------------------
    def _finish(self, sentence, clauses, tokens, pos, sentence_index):
        clauses = self._repair_bare_aux(clauses, tokens, pos)
        self._possessives(clauses, tokens, pos)
        self._copulas(clauses, tokens, pos)
        self._quantities(clauses, tokens, pos)
        self._negation(clauses, tokens)
        for c in clauses:
            c.sentence_index = sentence_index
        return clauses

    @staticmethod
    def _segments(tokens, pos):
        """Split a long sentence at clause boundaries (and/but/so/,/because/when/...) so the
        parser sees one finite clause at a time. A boundary needs a verb on its left and a
        subject-like token followed by a verb on its right; the marker itself is dropped."""
        n = len(tokens)
        verbish = lambda i: pos[i] in ("VERB", "AUX")  # noqa: E731
        marks = []
        for i, t in enumerate(tokens):
            if 0 < i < n - 2 and (t in _CLAUSE_MARKS or (t == "," and tokens[i + 1] in _CLAUSE_MARKS | _SUBJECTISH)):
                marks.append(i)
        if not marks:
            return None
        bounds, start = [], 0
        for m in marks:
            if m - start < 2:
                continue
            nxt = next((x for x in marks if x > m), n)
            left_ok = any(verbish(i) for i in range(start, m))
            right = range(m + 1, nxt)
            first_v = next((i for i in right if verbish(i)), None)
            right_ok = first_v is not None and any(
                pos[i] in ("NOUN", "PROPN", "PRON", "DET") for i in range(m + 1, first_v))
            if left_ok and right_ok:
                bounds.append((start, m))
                start = m + 1
        bounds.append((start, n))
        if len(bounds) < 2:
            return None
        return [(lo, hi) for lo, hi in bounds if hi - lo >= 2]

    def _raw_clauses(self, sentence: str, tokens, pos, allow_split: bool = True):
        """Parser/encoder trees before any post-pass; cached by normalised sentence."""
        rk = self._key("raw", sentence)
        if rk in self._cache:
            d = self._cache[rk]
            return [self._clause_from_dict(c) for c in d["clauses"]], d["source"]
        segs = self._segments(tokens, pos) if allow_split else None
        if segs:
            out, srcs = [], set()
            for lo, hi in segs:
                sub = tokens[lo:hi]
                cl, sr = self._raw_clauses(" ".join(sub), sub, pos[lo:hi], allow_split=False)
                for c in cl:
                    for f in [c.predicate] + c.args:
                        if f.token_index is not None:
                            f.token_index += lo
                out += cl
                srcs.add(sr)
            source = "parser" if "parser" in srcs else ("encoder" if "encoder" in srcs else "none")
            self._cache_put(rk, {"source": source, "clauses": [dataclasses.asdict(c) for c in out]})
            return out, source
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
        self._cache_put(rk, {"source": source, "clauses": [dataclasses.asdict(c) for c in clauses]})
        return clauses, source

    def _read_raw(self, sentence: str, sentence_index=None, tagged=None) -> ReadResult:
        t0 = time.perf_counter()
        orig, sentence = sentence, _norm(sentence)
        tokens, pos = tagged or self._tag(sentence, orig)
        clauses, source = self._raw_clauses(sentence, tokens, pos)
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
        norm = _norm(question)
        tokens, pos = self._tag(norm, question)
        qc = self._surface_question(question, tokens, pos)
        if qc is not None:
            rr = ReadResult(sentence=question, clauses=[qc], source="parser")
        else:  # no recognisable wh-shape: parser tree + the old hole marking
            rr = self._read_raw(question, tagged=(tokens, pos))
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
