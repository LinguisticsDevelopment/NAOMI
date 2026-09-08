#!/usr/bin/env python3
"""Generate hard-case gold at scale from typed-slot templates (lead
directive, 2026-09-08).

The 16 hand-authored drafts (`scripts/build_hand_gold_draft.py`,
`dev/HAND_GOLD_DRAFT.md`) proved the v2 lattice schema is human-writable for
the constructions the deterministic teacher can never emit gold for
(imperatives, interjections, elision/fragments, additive/focus, quantity
fragments, synthesized subjects) -- but 16 records cannot train anything.
This script GENERALIZES each of those 16 exemplars into a template with
typed slots (`src/nsm_ct/hard_gold_templates.py`), fills the slots from
word pools built from the live USVS + WordNet, and gates every generated
record through the SAME check `scripts/hand_gold.py`'s drafts pass:
`check_record` -- schema validity, oracle linearization, action legality,
skeleton round-trip.

Splits:
  - train        -- most templates x fillers.
  - test_filler  -- the SAME templates as train, fillers never drawn for train.
  - test_template -- 2 whole templates per family, held out of train entirely.

Run:  python scripts/gen_hard_gold.py [--per-family N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from hand_gold import check_record, hand_gold_record, load_usvs_default  # noqa: E402
from nsm_ct.hard_gold_templates import (  # noqa: E402
    FAMILIES, Template, build_pools, templates_by_family,
)

OUT_DIR = _HERE.parent / "runs"
SPLITS = ("train", "test_filler", "test_template")


def sample_combo(template: Template, pools: Dict[str, List[str]], rng: random.Random) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    for placeholder, pool_key in template.slots.items():
        vals[placeholder] = rng.choice(pools[pool_key])
    for group in template.distinct:
        for _ in range(20):
            values = [vals[p] for p in group]
            if len(set(values)) == len(values):
                break
            for p in group:
                vals[p] = rng.choice(pools[template.slots[p]])
    return vals


def combo_key(template: Template, vals: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(vals[p] for p in sorted(template.slots))


_REASON_PATTERNS = [
    (re.compile(r"does not point at word"), "token_index doesn't point at word"),
    (re.compile(r"does not dereference"), "ctx handle/ref doesn't dereference"),
    (re.compile(r"resolved grounding must have candidates=null"), "resolved grounding with candidates"),
    (re.compile(r"entity must have candidates=null"), "entity grounding with candidates"),
    (re.compile(r"node candidates disagree"), "candidates disagree with token_sense_candidates"),
    (re.compile(r"has no token_sense_candidates entry"), "sense slot missing token_sense_candidates entry"),
    (re.compile(r"bad utterance_kind"), "bad utterance_kind"),
    (re.compile(r"bad grounding\.type"), "bad grounding.type"),
    (re.compile(r"retrieval\.method missing"), "retrieval.method missing"),
    (re.compile(r"bad retrieval\.source"), "bad retrieval.source"),
    (re.compile(r"grounded predicate with null surface"), "grounded predicate with null surface"),
    (re.compile(r"elided predicate must have"), "elided predicate has non-null surface"),
    (re.compile(r"predicate_token_index out of range"), "predicate_token_index out of range"),
    (re.compile(r"token_index out of range"), "token_index out of range"),
    (re.compile(r"token_index without a surface word"), "token_index without a surface word"),
    (re.compile(r"ILLEGAL at"), "illegal oracle action"),
    (re.compile(r"skeleton mismatch"), "round-trip skeleton mismatch"),
    (re.compile(r"does not end in STOP"), "derivation doesn't end in STOP"),
    (re.compile(r"^\(.*Error.*\)$|Error\("), "linearize exception"),
]


def bucket_reason(problem: str) -> str:
    for pat, label in _REASON_PATTERNS:
        if pat.search(problem):
            return label
    return f"other: {problem[:60]}"


def render_summary(rec: dict) -> str:
    lines = [f"  text: {rec['text']!r}"]
    tree = rec["lattice"]["trees"][0]
    for clause in tree["clauses"]:
        pg = clause["predicate_grounding"]
        lines.append(f"    clause kind={clause.get('utterance_kind','proposition')} "
                      f"predicate={clause.get('predicate')!r} gtype={pg['type']}")
        for role in clause["roles"]:
            g = role["grounding"]
            lines.append(f"      {role['relation']:12s} word={role['word']!r:14} gtype={g['type']}")
    if rec.get("context"):
        lines.append(f"    context: {[c['text'] for c in rec['context']]}")
    return "\n".join(lines)


def generate(per_family: int, seed: int) -> Tuple[Dict[str, List[dict]], dict]:
    usvs = load_usvs_default()
    pools, pool_report = build_pools(usvs)
    rng = random.Random(seed)
    by_family = templates_by_family()

    out: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    seen_surfaces: set = set()
    fail_reasons: Counter = Counter()
    family_stats: Dict[str, dict] = {}
    samples: Dict[str, List[dict]] = {f: [] for f in FAMILIES}

    def try_fill(family: str, t: Template, split: str, target: int,
                 used_combo_keys: set, stats: dict) -> None:
        attempts, got, max_attempts = 0, 0, target * 60 + 60
        while got < target and attempts < max_attempts:
            attempts += 1
            vals = sample_combo(t, pools, rng)
            key = combo_key(t, vals)
            if key in used_combo_keys:
                continue
            rendered = t.build(vals, usvs)
            stats["generated"] += 1
            if rendered.text in seen_surfaces:
                used_combo_keys.add(key)
                continue
            rec = hand_gold_record(rendered.text, rendered.clauses, usvs=usvs,
                                    context=rendered.context)
            report = check_record(rec, usvs, strict=False)
            used_combo_keys.add(key)
            if not report["ok"]:
                stats["failed"] += 1
                for p in report["problems"]:
                    fail_reasons[bucket_reason(p)] += 1
                continue
            seen_surfaces.add(rendered.text)
            rec["meta"] = {"source": "hard_gold_gen", "family": family,
                            "template_id": t.id, "split": split}
            out[split].append(rec)
            stats["passed"] += 1
            stats["splits"][split] += 1
            if len(samples[family]) < 2:
                samples[family].append(rec)
            got += 1

    for family in FAMILIES:
        templates = by_family[family]
        n_held = min(2, len(templates))
        held_out = rng.sample(templates, n_held) if n_held else []
        held_ids = {t.id for t in held_out}
        train_templates = [t for t in templates if t.id not in held_ids]

        stats = {"generated": 0, "passed": 0, "failed": 0,
                 "splits": {s: 0 for s in SPLITS},
                 "templates": [t.id for t in templates],
                 "held_out_templates": sorted(held_ids)}

        n_train_t = max(len(train_templates), 1)
        train_target = -(-per_family // n_train_t)
        filler_target = max(1, -(-max(1, per_family // 5) // n_train_t))
        for t in train_templates:
            used = set()
            try_fill(family, t, "train", train_target, used, stats)
            try_fill(family, t, "test_filler", filler_target, used, stats)

        n_held_t = max(len(held_out), 1)
        held_target = max(1, -(-max(1, per_family // 5) // n_held_t))
        for t in held_out:
            used = set()
            try_fill(family, t, "test_template", held_target, used, stats)

        family_stats[family] = stats

    report = {"pools": pool_report, "families": family_stats,
              "fail_reasons": fail_reasons, "samples": samples}
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out, report = generate(args.per_family, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        path = out_dir / f"hard_gold_{split}.jsonl"
        with path.open("w") as fh:
            for rec in out[split]:
                fh.write(json.dumps(rec) + "\n")
        print(f"wrote {path} ({len(out[split])} records)")

    all_path = out_dir / "hard_gold_all.jsonl"
    with all_path.open("w") as fh:
        for split in SPLITS:
            for rec in out[split]:
                fh.write(json.dumps(rec) + "\n")
    total = sum(len(out[s]) for s in SPLITS)
    print(f"wrote {all_path} ({total} records)")

    print("\n=== pool sizes ===")
    for k in sorted(report["pools"].sizes):
        print(f"  {k:16s} n={report['pools'].sizes[k]:3d}  {report['pools'].sources.get(k,'')}")
    if report["pools"].dropped:
        print("\n  dropped (curated but failed the ground-filter):")
        for k, v in report["pools"].dropped.items():
            print(f"    {k}: {v}")

    print("\n=== families x templates ===")
    hdr = f"{'family':20s} {'templates':10s} {'generated':10s} {'passed':8s} {'failed':8s}  splits(train/filler/tmpl)"
    print(hdr)
    for family in FAMILIES:
        s = report["families"][family]
        sp = s["splits"]
        print(f"{family:20s} {len(s['templates']):10d} {s['generated']:10d} "
              f"{s['passed']:8d} {s['failed']:8d}  "
              f"{sp['train']:5d}/{sp['test_filler']:5d}/{sp['test_template']:5d}   "
              f"held_out={s['held_out_templates']}")

    print("\n=== failure reason histogram ===")
    if not report["fail_reasons"]:
        print("  (no failures)")
    for reason, n in report["fail_reasons"].most_common():
        print(f"  {n:5d}  {reason}")

    print("\n=== 2 rendered samples per family ===")
    for family in FAMILIES:
        print(f"\n-- {family} --")
        for rec in report["samples"][family]:
            print(render_summary(rec))

    total_failed = sum(s["failed"] for s in report["families"].values())
    print(f"\ntotal generated={sum(s['generated'] for s in report['families'].values())} "
          f"passed={sum(s['passed'] for s in report['families'].values())} failed={total_failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
