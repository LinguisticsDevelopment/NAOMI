"""Tree scorers: grade any candidate tree for a sentence, as a percentage with reasons.

(A) :func:`score_tree` is reference-free: it checks how much of the sentence's POS-tagged
    structure a tree accounts for. Uses only structural UPOS categories (no word lists).
(B) :func:`score_against` is reference-based: graded agreement between two trees over
    (predicate token index, role, filler token index) edges.

Both report machine-readable diagnostics: a list of dicts with a ``kind`` key. "Dropped
token" diagnostics report what the tree is missing, they are not language rules.

Weights of the coverage percentage (documented, adjustable):
    verb 0.35 + noun 0.30 + modifier 0.15 + structure 0.20
A component with no applicable tokens (e.g. no adjectives) is left out and the remaining
weights are renormalised. Reference score: 0.7 * edge F1 + 0.3 * predicate recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .types import ClauseTree

VERB_LIKE = {"VERB", "AUX"}
NOMINAL_LIKE = {"NOUN", "PROPN", "PRON"}
MODIFIER_LIKE = {"ADJ", "ADV", "NUM"}
IGNORED = {"PUNCT", "SPACE"}
WEIGHTS = {"verb": 0.35, "noun": 0.30, "modifier": 0.15, "structure": 0.20}
REF_WEIGHTS = {"edge_f1": 0.7, "predicate_recall": 0.3}


@dataclass
class TreeScore:
    percent: float
    components: Dict[str, Optional[float]]            # each in [0, 1] or None if not applicable
    diagnostics: List[dict] = field(default_factory=list)


@dataclass
class RefScore:
    percent: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    predicate_recall: float
    predicate_precision: float
    diagnostics: List[dict] = field(default_factory=list)
    graded: Optional[float] = None                    # usvs_graded overall (0-1), if requested


def _idx(f) -> Optional[int]:
    return f.token_index


def score_tree(sentence_tokens: Sequence[str], pos_tags: Sequence[str],
               clauses: Sequence[ClauseTree]) -> TreeScore:
    n = len(sentence_tokens)
    diags: List[dict] = []
    pred_idx, filler_idx = set(), set()
    valid_clauses = 0
    seen_pred: Dict[int, int] = {}

    def d(kind, i, **kw):
        diags.append({"kind": kind, "token": sentence_tokens[i] if 0 <= i < n else None,
                      "index": i, **kw})

    for ci, c in enumerate(clauses):
        ok = True
        pi = _idx(c.predicate)
        if pi is None or not (0 <= pi < n):
            diags.append({"kind": "phantom_token_index", "clause": ci, "role": "PREDICATE",
                          "index": pi, "token": c.predicate.token})
            ok = False
        else:
            pred_idx.add(pi)
            if pos_tags[pi] not in VERB_LIKE:
                d("predicate_not_verb_like", pi, clause=ci, pos=pos_tags[pi])
                ok = False
            if pi in seen_pred:
                d("duplicate_token_index", pi, clause=ci, role="PREDICATE")
                ok = False
            seen_pred[pi] = ci
            if sentence_tokens[pi] != c.predicate.token:
                d("phantom_token_index", pi, clause=ci, role="PREDICATE", claimed=c.predicate.token)
                ok = False
        used = {pi} if pi is not None else set()
        for a in c.args:
            ai = _idx(a)
            if ai is None or not (0 <= ai < n):
                diags.append({"kind": "phantom_token_index", "clause": ci, "role": a.role,
                              "index": ai, "token": a.token})
                ok = False
                continue
            if ai in used:
                d("duplicate_token_index", ai, clause=ci, role=a.role)
                ok = False
            elif sentence_tokens[ai] != a.token:
                d("phantom_token_index", ai, clause=ci, role=a.role, claimed=a.token)
                ok = False
            used.add(ai)
            filler_idx.add(ai)
        if c.arg("SUBJECT") is None:
            diags.append({"kind": "clause_without_subject", "clause": ci,
                          "predicate": c.predicate.token, "index": pi})
        valid_clauses += ok

    covered = pred_idx | filler_idx

    def cov(cats, accept, kind):
        want = [i for i in range(n) if pos_tags[i] in cats]
        if not want:
            return None
        hit = [i for i in want if i in accept]
        for i in want:
            if i not in accept:
                d(kind, i, pos=pos_tags[i])
        return len(hit) / len(want)

    comps: Dict[str, Optional[float]] = {
        "verb": cov(VERB_LIKE, covered, "verb_without_clause"),
        "noun": cov(NOMINAL_LIKE, filler_idx | pred_idx, "noun_not_in_tree"),
        "modifier": cov(MODIFIER_LIKE, covered, "modifier_not_in_tree"),
        "structure": (valid_clauses / len(clauses)) if clauses else (0.0 if n else None),
    }
    for i in range(n):
        p = pos_tags[i]
        if p not in VERB_LIKE | NOMINAL_LIKE | MODIFIER_LIKE | IGNORED and i not in covered:
            d("token_dropped", i, pos=p)
    if not clauses and n:
        diags.append({"kind": "no_clause"})
    used_w = {k: w for k, w in WEIGHTS.items() if comps[k] is not None}
    tot = sum(used_w.values())
    pct = 100.0 * sum(w * comps[k] for k, w in used_w.items()) / tot if tot else 0.0
    return TreeScore(percent=pct, components=comps, diagnostics=diags)


def _edges(clauses: Sequence[ClauseTree]) -> Tuple[set, set]:
    preds, edges = set(), set()
    for c in clauses:
        pi = _idx(c.predicate)
        if pi is None:
            continue
        preds.add(pi)
        for a in c.args:
            if _idx(a) is not None:
                edges.add((pi, a.role, _idx(a)))
    return preds, edges


def _as_tree(clauses: Sequence[ClauseTree]) -> dict:
    def node(f):
        return {"token_index": f.token_index, "grounding": {"type": "sense", "candidates": list(f.senses)}}
    return {"clauses": [{"predicate": node(c.predicate), "roles": [{"relation": a.role, **node(a)} for a in c.args]}
                        for c in clauses]}


def score_against(reference_clauses: Sequence[ClauseTree], candidate_clauses: Sequence[ClauseTree],
                  tokens: Optional[Sequence[str]] = None, usvs=None) -> RefScore:
    """Agreement of a candidate with a reference. If ``usvs`` (a loaded USVS) is given,
    the near-miss-aware ``nsm_ct.usvs_graded`` overall similarity is added as ``graded``."""
    rp, re_ = _edges(reference_clauses)
    cp, ce = _edges(candidate_clauses)
    diags: List[dict] = []
    for p in sorted(rp - cp):
        diags.append({"kind": "clause_dropped", "index": p})
    for p in sorted(cp - rp):
        diags.append({"kind": "clause_extra", "index": p})
    cand_by_pf = {(p, f): r for p, r, f in ce}
    ref_by_pf = {(p, f): r for p, r, f in re_}
    for p, r, f in sorted(re_ - ce):
        if (p, f) in cand_by_pf:
            diags.append({"kind": "role_wrong", "predicate": p, "filler": f, "ref_role": r,
                          "cand_role": cand_by_pf[(p, f)]})
        elif p in cp:
            diags.append({"kind": "filler_missing", "predicate": p, "filler": f, "role": r})
    for p, r, f in sorted(ce - re_):
        if (p, f) not in ref_by_pf and p in rp:
            diags.append({"kind": "filler_extra", "predicate": p, "filler": f, "role": r})
    tp = len(re_ & ce)
    prec = tp / len(ce) if ce else (1.0 if not re_ else 0.0)
    rec = tp / len(re_) if re_ else (1.0 if not ce else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    prec_p = len(rp & cp) / len(cp) if cp else (1.0 if not rp else 0.0)
    rec_p = len(rp & cp) / len(rp) if rp else (1.0 if not cp else 0.0)
    pct = 100.0 * (REF_WEIGHTS["edge_f1"] * f1 + REF_WEIGHTS["predicate_recall"] * rec_p)
    graded = None
    if usvs is not None:
        from nsm_ct.usvs_graded import usvs_tree_similarity
        graded = float(usvs_tree_similarity(_as_tree(candidate_clauses), _as_tree(reference_clauses),
                                      {"tokens": list(tokens or [])}, usvs)["graded_overall"])
    return RefScore(percent=pct, edge_precision=prec, edge_recall=rec, edge_f1=f1,
                    predicate_recall=rec_p, predicate_precision=prec_p, diagnostics=diags, graded=graded)
