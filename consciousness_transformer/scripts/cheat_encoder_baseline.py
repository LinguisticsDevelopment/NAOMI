"""The encoder's cheat baseline: a ~30-line POS heuristic (dev/
AUDIT_2026-10-15.md section A.1), scored through the SAME rank-1 edge-F1
code path as the learned encoder (``nsm_ct.encoder_model.score_record`` /
``aggregate_recall`` -- not a second, parallel metric implementation).

The audit measured this heuristic by hand at edge-F1 0.40 (P 0.37, R 0.42)
against the encoder's 0.622 on the same 98-sentence v2 holdout, but never
committed it as runnable code or wired it into the eval scripts -- so every
encoder number since has shipped with no floor beside it, breaking CLAUDE.md's
"every capability curriculum ships with its cheat-baseline at floor" law.
This script is that floor, and ``--pos-vocab``-free: it reads ``record["pos"]``
directly off each gold record (the project's own tagger already wrote it at
gold-build time -- see ``nsm_ct.encoder_model.build_features``'s own
``record["pos"]`` read), falling back to nltk's tagger only when a record
has no ``pos`` field at all.

Heuristic (the whole "model"):
  * PREDICATE = the first VERB token; else the first AUX token; else none.
  * SUBJECT   = the nearest nominal (NOUN/PROPN/PRON) BEFORE the predicate.
  * OBJECT    = the nearest nominal AFTER the predicate.
  * DESCRIPTION = every ADJ/DET/ADV/NUM token, each its own role.
  * PLACE     = the nominal immediately after an ADP (preposition) token.
  * grounding-type = "sense" if the token has any sense candidates
    (``record["token_sense_candidates"]``), else "entity" for a PROPN,
    else "reference" -- a real signal (candidates ARE the input), not a
    free pass: the heuristic is never told which candidate/relation is
    right, only whether retrieval found any at all.

Usage:
    python scripts/cheat_encoder_baseline.py \\
        --gold runs/encoder_gold_v4b_16k.jsonl --holdout runs/holdout_sentences.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.encoder_model import RecordRecall, aggregate_recall, score_record  # noqa: E402
from nsm_ct.encoder_train_util import load_gold  # noqa: E402

_NOMINAL = {"NOUN", "PROPN", "PRON"}


def _gtype_for(record: dict, idx: Optional[int]) -> str:
    if idx is None:
        return "prime"
    for e in record.get("token_sense_candidates", []) or []:
        if e.get("index") == idx and e.get("sense_candidates"):
            return "sense"
    pos = record["pos"][idx] if idx < len(record.get("pos", [])) else "X"
    return "entity" if pos == "PROPN" else "reference"


def _node(idx: Optional[int], record: dict, relation: Optional[str] = None) -> dict:
    node = {"token_index": idx, "grounding": {"type": _gtype_for(record, idx)}}
    if relation is not None:
        node["relation"] = relation
    return node


def _pos_tags(record: dict) -> List[str]:
    """``record["pos"]`` when present (every gold record this project builds
    has it -- see module docstring); nltk's tagger as a pure fallback for a
    record that somehow lacks it (e.g. a hand-written fixture)."""
    pos = record.get("pos")
    if pos:
        return pos
    import nltk  # noqa: E402  -- lazy: only hit on the fallback path

    tagged = nltk.pos_tag(record["tokens"])
    _MAP = {
        "NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN",
        "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
        "MD": "AUX",
        "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
        "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
        "DT": "DET", "PDT": "DET", "WDT": "DET",
        "IN": "ADP", "TO": "ADP",
        "CD": "NUM",
        "PRP": "PRON", "PRP$": "PRON", "WP": "PRON",
    }
    return [_MAP.get(tag, "X") for _tok, tag in tagged]


def heuristic_forest(record: dict) -> List[dict]:
    """-> a one-tree forest in the SAME normalized clause/node shape
    ``beam_decode``/``score_record`` use (``nsm_ct.encoder_model``'s
    ``_dump_forest`` is the other example of this shape) -- see module
    docstring for the rule set."""
    pos = _pos_tags(record)
    T = len(pos)

    pred_idx = next((i for i in range(T) if pos[i] == "VERB"), None)
    if pred_idx is None:
        pred_idx = next((i for i in range(T) if pos[i] == "AUX"), None)

    roles: List[dict] = []
    if pred_idx is not None:
        subj_idx = next((i for i in range(pred_idx - 1, -1, -1) if pos[i] in _NOMINAL), None)
        obj_idx = next((i for i in range(pred_idx + 1, T) if pos[i] in _NOMINAL), None)
        if subj_idx is not None:
            roles.append(_node(subj_idx, record, "SUBJECT"))
        if obj_idx is not None:
            roles.append(_node(obj_idx, record, "OBJECT"))

    for i, p in enumerate(pos):
        if p in ("ADJ", "DET", "ADV", "NUM"):
            roles.append(_node(i, record, "DESCRIPTION"))
        elif p == "ADP":
            # "the nominal after an ADP" -- skip over any DET/ADJ/NUM in
            # between (a bare "ADP"->"NOUN" index check almost never fires
            # on real prose, where a determiner sits between "on"/"in"/
            # "at" and its head noun) to the first real nominal.
            j = i + 1
            while j < T and pos[j] in ("DET", "ADJ", "NUM"):
                j += 1
            if j < T and pos[j] in _NOMINAL:
                roles.append(_node(j, record, "PLACE"))

    predicate = _node(pred_idx, record, "PREDICATE")
    return [{"clauses": [{"predicate": predicate, "roles": roles}]}]


def evaluate_heuristic(records: Sequence[dict]) -> Dict[str, float]:
    scores: List[RecordRecall] = []
    for record in records:
        forest = heuristic_forest(record)
        scores.append(score_record(record, forest))
    return aggregate_recall(scores)


def load_holdout_records(gold_path: str, holdout_path: str) -> List[dict]:
    with open(holdout_path, encoding="utf-8") as f:
        holdout_texts = {line.strip() for line in f if line.strip()}
    records = load_gold(gold_path)
    kept = [r for r in records if r.get("text", "").strip() in holdout_texts]
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=str, required=True, help="Gold jsonl (nsm_ct.encoder_train_util.load_gold format).")
    ap.add_argument("--holdout", type=str, default=None,
                     help="One sentence per line (scripts/make_holdout.py's output); only gold "
                          "records whose text matches are scored. Omit to score the WHOLE gold file.")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    if args.holdout:
        records = load_holdout_records(args.gold, args.holdout)
        print(f"=== cheat_encoder_baseline: {len(records)} holdout records matched "
              f"(of {sum(1 for _ in open(args.holdout))} holdout sentences) in {args.gold} ===")
    else:
        records = load_gold(args.gold)
        print(f"=== cheat_encoder_baseline: {len(records)} records loaded from {args.gold} ===")

    if not records:
        print("no records to score -- nothing to do.")
        return

    agg = evaluate_heuristic(records)
    p, r = agg["edge_precision"], agg["edge_recall"]
    f1 = (2 * p * r / (p + r)) if (p + r) else float("nan")
    print(f"\nPOS-HEURISTIC CHEAT BASELINE (n={agg['n_records']}):")
    print(f"  edge_precision: {p:.3f}")
    print(f"  edge_recall:    {r:.3f}")
    print(f"  edge_F1:        {f1:.3f}")
    print(f"  struct-exact (structure_recall): {agg['structure_recall']:.3f}")
    print(f"  sense_recall: {agg['sense_recall']:.3f}  slot_recall: {agg['slot_recall']:.3f}  "
          f"all_gold_recalled_rate: {agg['all_gold_recalled_rate']:.3f}")

    if args.json_out:
        out = dict(agg)
        out["edge_f1"] = f1
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote -> {args.json_out}")


if __name__ == "__main__":
    main()
