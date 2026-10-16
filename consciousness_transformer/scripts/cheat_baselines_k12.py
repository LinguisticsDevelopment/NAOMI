"""K-12 MC-benchmark cheat baselines (dev/reports/K12_BENCHMARK_FIX.md).

dev/reports/AUDIT_2026-10-15.md section C: the FairytaleQA entity MC benchmark's
``cross-story`` distractor policy lets "pick the option that appears
verbatim in the passage" score ~97% (553/584 outright + 31 ties, 0 misses
on the committed ``runs/fairytaleqa_episodes.jsonl``), because only the
GOLD answer is ever a passage substring. ``scripts/eval_prose.py`` only
ever reported random/majority floors -- this script adds the two cheats
that actually matter for an MC benchmark built from real prose:
``passage-substring`` and ``word-overlap``, alongside random/majority, over
ANY episodes jsonl (``scripts/convert_fairytaleqa.py``'s or
``scripts/convert_mctest.py``'s output -- both share the one
:class:`~nsm_ct.episode.Episode` schema).

Only episodes with ``options is not None`` are scorable (an entity-type
FairytaleQA episode that didn't get >=2 distractors keeps ``options=None``;
every MCTest episode has 4 options) -- this script reports how many of the
loaded episodes were kept for exactly that reason, then reports all four
baselines over the kept set.

Usage:
    python scripts/cheat_baselines_k12.py --episodes runs/fairytaleqa_episodes.jsonl
    python scripts/cheat_baselines_k12.py --episodes runs/fairytaleqa_episodes_samepassage.jsonl
    python scripts/cheat_baselines_k12.py --episodes runs/mctest_mc160_episodes.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.episode import Episode  # noqa: E402

_EPISODE_FIELDS = {f.name for f in dataclasses.fields(Episode)}


def load_episodes(path: str) -> List[Episode]:
    episodes: List[Episode] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kwargs = {k: v for k, v in row.items() if k in _EPISODE_FIELDS}
            episodes.append(Episode(**kwargs))
    return episodes


def filter_scorable(episodes: List[Episode]) -> List[Episode]:
    return [e for e in episodes if e.options is not None and e.answer_idx is not None]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def random_baseline(episodes: List[Episode]) -> float:
    """Expected accuracy of a uniform-random pick, per episode's own
    option count -- mean of ``1/n_options``."""
    if not episodes:
        return float("nan")
    return sum(1.0 / len(e.options) for e in episodes) / len(episodes)


def majority_baseline(episodes: List[Episode]) -> Tuple[float, str]:
    """Always guess the single most common gold answer TEXT across the
    whole set (the standard majority-class floor for a varying-option-set
    MC task)."""
    if not episodes:
        return float("nan"), ""
    counts = Counter(e.answer_text for e in episodes)
    modal_text, modal_count = counts.most_common(1)[0]
    return modal_count / len(episodes), modal_text


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def passage_substring_baseline(episodes: List[Episode], rng: random.Random) -> Tuple[float, int, int, int]:
    """"Pick the option that occurs verbatim in the passage" (dev/
    AUDIT_2026-10-15.md's own cheat). For each episode: the TIED set is
    every option whose normalized text is a substring of the normalized
    passage. Zero matches -> a MISS (scored 0, the cheat gives no signal).
    One match -> UNIQUE (scored 1 iff it's the gold). >1 match -> a TIE,
    broken uniformly at random among the tied options (expected credit
    1/|tie| if gold is among them, else 0) -- matches the audit's own
    "553 unique-correct, 31 ties, 0 misses" accounting. Returns
    ``(accuracy, n_unique_correct, n_ties, n_misses)``."""
    if not episodes:
        return float("nan"), 0, 0, 0
    total = 0.0
    n_unique_correct = 0
    n_ties = 0
    n_misses = 0
    for e in episodes:
        passage = _norm(" ".join(e.context))
        gold_norm = _norm(e.answer_text)
        hits = [i for i, opt in enumerate(e.options) if _norm(opt) and _norm(opt) in passage]
        if not hits:
            n_misses += 1
            continue
        if len(hits) == 1:
            if hits[0] == e.answer_idx:
                n_unique_correct += 1
                total += 1.0
            continue
        n_ties += 1
        if e.answer_idx in hits:
            total += 1.0 / len(hits)
    return total / len(episodes), n_unique_correct, n_ties, n_misses


def word_overlap_baseline(episodes: List[Episode], rng: random.Random) -> float:
    """"Option with max token overlap with the passage" -- for each option,
    the count of its own distinct lowercased tokens that also occur
    anywhere in the passage's tokens; the option(s) with the MAX count win,
    ties broken uniformly at random among the tied options (same
    tie-break convention as ``passage_substring_baseline``, so the two
    cheats are directly comparable)."""
    if not episodes:
        return float("nan")
    total = 0.0
    for e in episodes:
        passage_toks = set(_norm(" ".join(e.context)).split())
        scores = []
        for opt in e.options:
            opt_toks = set(_norm(opt).split())
            scores.append(len(opt_toks & passage_toks))
        best = max(scores) if scores else 0
        tied = [i for i, s in enumerate(scores) if s == best]
        if e.answer_idx in tied:
            total += 1.0 / len(tied)
    return total / len(episodes)


def report(name: str, episodes: List[Episode], seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    scorable = filter_scorable(episodes)
    print(f"\n=== {name} ===")
    print(f"loaded: {len(episodes)}  scorable (options is not None): {len(scorable)}")
    if not scorable:
        print("no scorable episodes -- nothing to report.")
        return {}

    r = random_baseline(scorable)
    maj_acc, maj_text = majority_baseline(scorable)
    sub_acc, n_unique, n_ties, n_miss = passage_substring_baseline(scorable, rng)
    ov_acc = word_overlap_baseline(scorable, rng)

    print(f"random               : {r:.3f}")
    print(f"majority (={maj_text!r}) : {maj_acc:.3f}")
    print(f"passage-substring    : {sub_acc:.3f}  "
          f"(unique-correct={n_unique} ties={n_ties} misses={n_miss}, n={len(scorable)})")
    print(f"word-overlap         : {ov_acc:.3f}")
    print(f"gate (substring <= random + 0.10): "
          f"{'PASS' if sub_acc <= r + 0.10 else 'FAIL'}  ({sub_acc:.3f} vs {r + 0.10:.3f})")

    return {
        "n_loaded": len(episodes), "n_scorable": len(scorable),
        "random": r, "majority": maj_acc, "majority_text": maj_text,
        "passage_substring": sub_acc, "passage_substring_unique_correct": n_unique,
        "passage_substring_ties": n_ties, "passage_substring_misses": n_miss,
        "word_overlap": ov_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=str, required=True, help="Episodes jsonl (Episode schema).")
    ap.add_argument("--name", type=str, default=None, help="Label for the report (default: the file path).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=str, default=None, help="Optional: also write the report dict as JSON.")
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)
    result = report(args.name or args.episodes, episodes, seed=args.seed)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote -> {args.json_out}")


if __name__ == "__main__":
    main()
