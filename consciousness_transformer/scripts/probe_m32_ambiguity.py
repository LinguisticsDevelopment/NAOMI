"""M32 gap measurement — does sense-correct grounding beat MFS on the
ambiguity-bearing curriculum? No training: this is a pure USVS-space scoring
probe over ``nsm_ct.episode.generate_ambiguity_episodes``.

For each generated episode the homograph's VALUE vector is grounded three
ways:

  (a) WORD  — ``usvs_handle(word, d)``: the plain word-level handle (no sense
      resolution at all — what a token/word-only perception path would use).
  (b) MFS   — ``usvs_sense_handle(mfs_sense, d)``: always the most-frequent
      sense (``wn.synsets(word)[0]``), regardless of context.
  (c) GOLD  — ``usvs_sense_handle(gold_sense, d)``: the sense the episode's
      context actually establishes.

Each grounding is scored against the episode's two MC options (via
``usvs_handle`` on the option words) by cosine similarity; a grounding
"wins" an episode if it ranks the correct option's cosine strictly above the
wrong option's. Reported over ALL episodes and over the SENSE-FLIPPED half
(``mfs_sense != gold_sense``) where MFS and GOLD necessarily disagree — the
subset that actually tests whether sense choice matters.

Gate: GOLD beats MFS by a real margin on the sense-flipped subset (word-level
and MFS should be close to each other there, since MFS-grounding ignores
context exactly like the word-level handle when the two disagree).

Run:
    python scripts/probe_m32_ambiguity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.episode import generate_ambiguity_episodes  # noqa: E402
from nsm_ct.usvs_bridge import usvs_handle, usvs_sense_handle  # noqa: E402

N_EPISODES = 400
D = 256
SEED = 0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both are unit-norm (usvs_bridge always normalizes)


def score_episode(ep, value_vec: np.ndarray, d: int = D) -> bool:
    """True if ``value_vec`` ranks the correct option above the wrong one."""
    opt_vecs = [usvs_handle(o, d) for o in ep.options]
    if any(v is None for v in opt_vecs):
        return None  # option word unknown to USVS; drop
    scores = [_cos(value_vec, v) for v in opt_vecs]
    correct = ep.answer_idx
    wrong = 1 - correct  # binary options by construction
    return scores[correct] > scores[wrong]


def run(n_episodes: int = N_EPISODES, d: int = D, seed: int = SEED) -> dict:
    """Run the probe over ``n_episodes`` generated episodes.

    Returns ``{"episodes": [...], "families": [...], "rows": [...], "acc": {...}, "gap": float}``
    so both ``main()`` (printing) and tests (assertions, on a small N) can share the logic.
    """
    episodes = generate_ambiguity_episodes(n_episodes, seed=seed)
    families = sorted({e.meta["family"] for e in episodes})

    rows = []  # (label, subset_name, n, wins)
    acc = {}
    for label, grounder in (
        ("WORD", lambda ep: usvs_handle(ep.meta["homograph"], d)),
        ("MFS", lambda ep: usvs_sense_handle(ep.meta["mfs_sense"], d)),
        ("GOLD", lambda ep: usvs_sense_handle(ep.meta["gold_sense"], d)),
    ):
        for subset_name, subset in (
            ("all", episodes),
            ("flipped (mfs!=gold)", [e for e in episodes if e.meta["mfs_sense"] != e.meta["gold_sense"]]),
            ("unflipped (mfs==gold)", [e for e in episodes if e.meta["mfs_sense"] == e.meta["gold_sense"]]),
        ):
            wins, total = 0, 0
            for ep in subset:
                v = grounder(ep)
                if v is None:
                    continue
                r = score_episode(ep, v, d)
                if r is None:
                    continue
                total += 1
                wins += int(r)
            a = wins / total if total else float("nan")
            rows.append((label, subset_name, total, wins))
            acc[(label, subset_name)] = a

    gap = acc.get(("GOLD", "flipped (mfs!=gold)"), float("nan")) - acc.get(
        ("MFS", "flipped (mfs!=gold)"), float("nan")
    )
    return {"episodes": episodes, "families": families, "rows": rows, "acc": acc, "gap": gap}


def main() -> None:
    result = run()
    episodes, families, rows, acc, gap = (
        result["episodes"], result["families"], result["rows"], result["acc"], result["gap"]
    )
    print(f"generated {len(episodes)} episodes")
    print(f"homograph families: {families}")

    header = f"{'grounding':<8}{'subset':<24}{'n':>6}{'wins':>6}{'acc':>10}"
    print(header)
    print("-" * len(header))
    for label, subset_name, total, wins in rows:
        a = acc[(label, subset_name)]
        print(f"{label:<8}{subset_name:<24}{total:>6}{wins:>6}{a:>10.3f}")

    print()
    print(f"GOLD - MFS gap on sense-flipped subset: {gap:+.3f}")
    print("GATE PASS" if gap > 0.10 else "GATE FAIL (no real gap)")


if __name__ == "__main__":
    main()
