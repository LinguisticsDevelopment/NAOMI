#!/usr/bin/env python3
"""Calibration harness for the LLM parse judge (dev/PARSE_JUDGE.md).

Three sets, built from `runs/encoder_gold_v2.jsonl` (+ `runs/complement_probe.txt`
when present):

  (i)   50 teacher GOLD top trees (first 50 records, deterministic) --
        expected mostly good.
  (ii)  50 CORRUPTED gold trees (next 50 records, one of three corruptions
        each: SUBJECT/OBJECT token swap, predicate replaced by an adjacent
        non-verb token, or a phantom role attached) -- expected bad.
  (iii) the 77 ENCODER COMPLEMENT trees, reconstructed from
        `runs/complement_probe.txt`'s "ALL ..." section (that file has no
        machine-readable twin -- see `parse_complement_probe_txt` for the
        reconstruction and its known lossiness) when that file exists.

Reports per-set good-rate and, for the corrupted set, the corrupted-
detection rate (= bad-rate, since every element there is a known corruption).

Always runs `--backend mock`. Runs `--backend haiku` too, but only if
ANTHROPIC_API_KEY is set in the environment -- if it is not, this script
does NOT prompt for one; it prints the exact command to run it later.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.parse_judge import HaikuJudge, MockJudge, judge_many_with_prefilter  # noqa: E402
from nsm_ct.tree_render import normalize_gold_tree  # noqa: E402

GOLD_PATH = ROOT / "runs" / "encoder_gold_v2.jsonl"
COMPLEMENT_PROBE_TXT = ROOT / "runs" / "complement_probe.txt"
SEED = 20260908

RecordTree = Tuple[dict, dict]


def load_gold(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def gold_tree_of(record: dict) -> dict:
    trees = record.get("lattice", {}).get("trees", [])
    if not trees:
        return {"clauses": []}
    return normalize_gold_tree(record, trees[0])


# ---------------------------------------------------------------------------
# (ii) corruption
# ---------------------------------------------------------------------------

VERB_LIKE = {"VERB", "AUX"}


def _corrupt_swap_subject_object(record: dict, tree: dict) -> Optional[dict]:
    """Swap the token_index of a SUBJECT and an OBJECT role in the first
    clause that has both -- a structurally invisible corruption (no phantom,
    no POS violation) that only a semantic judge can catch."""
    t = copy.deepcopy(tree)
    for clause in t.get("clauses", []):
        roles = clause.get("roles", [])
        subj = next((r for r in roles if r.get("relation") == "SUBJECT"), None)
        obj = next((r for r in roles if r.get("relation") == "OBJECT"), None)
        if subj and obj and subj.get("token_index") is not None and obj.get("token_index") is not None:
            subj["token_index"], obj["token_index"] = obj["token_index"], subj["token_index"]
            return t
    return None


def _corrupt_predicate_to_non_verb(record: dict, tree: dict) -> Optional[dict]:
    """Replace the predicate's token_index with the next non-VERB/AUX token
    after it -- the "may as predicate" / "was as PLACE" failure mode
    `runs/complement_probe.txt`'s best-5 exposed in a POS-only judge."""
    t = copy.deepcopy(tree)
    pos_list = record.get("pos", [])
    for clause in t.get("clauses", []):
        pred = clause.get("predicate", {}) or {}
        idx = pred.get("token_index")
        if idx is None:
            continue
        for cand in range(idx + 1, len(pos_list)):
            if pos_list[cand] not in VERB_LIKE:
                pred["token_index"] = cand
                pred["grounding"] = {"type": "entity"}
                return t
    return None


def _corrupt_phantom_role(record: dict, tree: dict) -> Optional[dict]:
    """Attach an extra role pointing at an out-of-range token_index -- J3's
    phantom-node check."""
    t = copy.deepcopy(tree)
    clauses = t.get("clauses", [])
    if not clauses:
        return None
    clauses[0].setdefault("roles", []).append(
        {"relation": "PHANTOM", "token_index": len(record.get("tokens", [])) + 5,
         "grounding": {"type": "entity"}})
    return t


_CORRUPTIONS = [_corrupt_swap_subject_object, _corrupt_predicate_to_non_verb, _corrupt_phantom_role]

# The subject/object swap is BY DESIGN structurally invisible (no phantom
# node, no POS violation) -- it is the corruption a POS-only/structural
# judge (MockJudge) is not expected to reliably catch, which is the whole
# point of needing a semantic (LLM) judge. `build_corrupted_set`'s default
# `kinds` includes it for a realistic calibration report; tests that need a
# 100%-detectable corrupted set pass `kinds=_STRUCTURAL_CORRUPTIONS`.
_STRUCTURAL_CORRUPTIONS = [_corrupt_predicate_to_non_verb, _corrupt_phantom_role]


def build_corrupted_set(records: List[dict], seed: int, kinds: List = None) -> List[RecordTree]:
    rng = random.Random(seed)
    kinds = list(kinds) if kinds is not None else list(_CORRUPTIONS)
    out = []
    for record in records:
        tree = gold_tree_of(record)
        order = list(kinds)
        rng.shuffle(order)
        for kind in order:
            corrupted = kind(record, tree)
            if corrupted is not None:
                out.append((record, corrupted))
                break
        else:
            # nothing applicable (e.g. no SUBJECT/OBJECT and no predicate
            # token) -- fall back to the phantom corruption, which always
            # applies to any non-empty clause list.
            out.append((record, tree))
    return out


# ---------------------------------------------------------------------------
# (iii) reconstruct the encoder complement set from complement_probe.txt
# ---------------------------------------------------------------------------

_BLOCK_HEADER_RE = re.compile(r"^\[(\d+)\] (.*)$")
_POS_LINE_RE = re.compile(r"^  POS: (.*)$")
_CLAUSE_RE = re.compile(r"^  clause (\d+) \((\w+)\):$")
_NODE_RE = re.compile(r"^    (\S+): (.*)\((.*)\)$")
_J_LINE_RE = re.compile(r"^  J1\(")


def _parse_node_detail(detail: str, tokens: list, claimed: set) -> Tuple[Optional[int], dict]:
    """`detail` is what `tree_render.node_detail_compact` (via the probe
    script's `render_node`) produced -- e.g. `sense:thief.n.01`, `prime:YOU`,
    `reference:memory`, `entity`, `None`. Recovers a best-effort
    (token_index, grounding) pair; the surface token itself was already
    consumed by the caller to resolve token_index."""
    if detail.startswith("sense:"):
        sense = detail[len("sense:"):]
        candidates = None if sense == "?" else [sense]
        return None, {"type": "sense", "candidates": candidates}
    if detail.startswith("prime:"):
        return None, {"type": "prime", "prime": detail[len("prime:"):], "candidates": None}
    if detail.startswith("reference:") or detail.startswith("elision:"):
        gtype, source = detail.split(":", 1)
        return None, {"type": gtype, "source": source, "candidates": None}
    if detail == "entity":
        return None, {"type": "entity", "candidates": None}
    return None, {"type": None, "candidates": None}


def _resolve_index(tokens: list, token_text: str, claimed: set) -> Optional[int]:
    if token_text == "<none>":
        return None
    for i, tok in enumerate(tokens):
        if tok == token_text and i not in claimed:
            claimed.add(i)
            return i
    return None


def parse_complement_probe_txt(path: Path) -> List[RecordTree]:
    """Best-effort reconstruction of (record, tree) pairs from the
    human-readable diagnostic dump `probe_encoder_complement.py` writes.
    LOSSY: the dump prints surface tokens, not `token_index`, so indices are
    recovered by a left-to-right consume-on-match walk over the block's own
    POS line (the same convention `encoder_model._predicate_token_index`
    uses) -- a repeated word can occasionally resolve to the wrong instance.
    Only the sense id (not the full candidate list) survives per sense node.
    Returns [] if the file is missing or has no parseable "ALL ..." section."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    marker = re.search(r"^----- ALL .* -----$", text, re.MULTILINE)
    if marker is None:
        return []
    lines = text[marker.end():].splitlines()

    out: List[RecordTree] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _BLOCK_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        sentence = m.group(2)
        i += 1
        while i < n and not _POS_LINE_RE.match(lines[i]) and not _BLOCK_HEADER_RE.match(lines[i]):
            i += 1
        if i >= n or not _POS_LINE_RE.match(lines[i]):
            continue  # malformed block -- skip
        pos_line = _POS_LINE_RE.match(lines[i]).group(1)
        tokens, pos = [], []
        for tp in pos_line.split(" "):
            if "/" not in tp:
                continue
            tok, tag = tp.rsplit("/", 1)
            tokens.append(tok)
            pos.append(tag)
        i += 1

        claimed: set = set()
        clauses = []
        cur_clause = None
        while i < n and not _BLOCK_HEADER_RE.match(lines[i]):
            line = lines[i]
            cm = _CLAUSE_RE.match(line)
            nm = _NODE_RE.match(line)
            if cm:
                cur_clause = {"predicate": {"token_index": None, "grounding": {"type": "entity"}},
                              "roles": [], "utterance_kind": cm.group(2)}
                clauses.append(cur_clause)
            elif nm and cur_clause is not None:
                relation, token_text, detail = nm.group(1), nm.group(2), nm.group(3)
                idx = _resolve_index(tokens, token_text, claimed)
                _, grounding = _parse_node_detail(detail, tokens, claimed)
                node = {"relation": relation, "token_index": idx, "grounding": grounding}
                if relation == "PREDICATE":
                    cur_clause["predicate"] = node
                else:
                    cur_clause["roles"].append(node)
            elif _J_LINE_RE.match(line):
                cur_clause = None
            i += 1

        record = {"text": sentence, "tokens": tokens, "pos": pos, "token_sense_candidates": []}
        out.append((record, {"clauses": clauses}))
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report_set(name: str, pairs: List[RecordTree], judge, batch: bool) -> None:
    if not pairs:
        print(f"{name}: n=0 (skipped)")
        return
    verdicts = judge_many_with_prefilter(judge, pairs, batch=batch)
    n = len(verdicts)
    good = sum(1 for v in verdicts if v.verdict == "good")
    print(f"{name}: n={n} good={good} ({good/n:.1%}) bad={n-good} ({(n-good)/n:.1%})")


def run_backend(backend_name: str, judge, gold_pairs, corrupted_pairs, complement_pairs,
                 batch: bool) -> None:
    print(f"\n=== backend={backend_name} model={judge.model} batch={batch} ===")
    report_set("(i)   gold (expected mostly good)", gold_pairs, judge, batch)
    report_set("(ii)  corrupted (expected bad -- this IS the corrupted-detection rate)",
                corrupted_pairs, judge, batch)
    report_set("(iii) encoder complement (reconstructed from complement_probe.txt)",
                complement_pairs, judge, batch)
    if isinstance(judge, HaikuJudge) and judge.total_requests:
        print(f"  usage: requests={judge.total_requests} input_tokens={judge.total_input_tokens} "
              f"output_tokens={judge.total_output_tokens}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "haiku", "both"], default="both")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API for haiku")
    args = ap.parse_args()

    if not GOLD_PATH.exists():
        print(f"ERROR: {GOLD_PATH} missing")
        return 1
    records = load_gold(GOLD_PATH)
    if len(records) < 100:
        print(f"ERROR: need >=100 gold records for sets (i)+(ii), have {len(records)}")
        return 1

    gold_records = records[:50]
    corrupt_records = records[50:100]
    gold_pairs = [(r, gold_tree_of(r)) for r in gold_records]
    corrupted_pairs = build_corrupted_set(corrupt_records, SEED)
    complement_pairs = parse_complement_probe_txt(COMPLEMENT_PROBE_TXT)
    print(f"sets: (i) gold n={len(gold_pairs)}  (ii) corrupted n={len(corrupted_pairs)}  "
          f"(iii) encoder complement n={len(complement_pairs)}"
          + ("" if complement_pairs else f"  ({COMPLEMENT_PROBE_TXT} missing/unparseable -- skipped)"))

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if args.backend in ("mock", "both"):
        run_backend("mock", MockJudge(), gold_pairs, corrupted_pairs, complement_pairs, batch=False)

    if args.backend == "haiku" and not have_key:
        print("\nANTHROPIC_API_KEY is not set -- not running --backend haiku here (not asking for a "
              "key). Run it with a key set, e.g.:\n"
              "  ANTHROPIC_API_KEY=... python scripts/calibrate_judge.py --backend haiku"
              + (" --batch" if args.batch else ""))
        return 0

    if args.backend in ("haiku", "both") and have_key:
        run_backend("haiku", HaikuJudge(), gold_pairs, corrupted_pairs, complement_pairs, batch=args.batch)
    elif args.backend == "both" and not have_key:
        print("\nANTHROPIC_API_KEY is not set -- skipping --backend haiku (leaving it for a run with "
              "a key set). Exact command:\n"
              "  ANTHROPIC_API_KEY=... python scripts/calibrate_judge.py --backend haiku"
              + (" --batch" if args.batch else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
