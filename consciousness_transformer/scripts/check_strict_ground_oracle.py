"""Confirmation-run check (mask-fix, Step 2): does `strict_ground=True`
(the decode-time-only mask tightening that drops GROUND once `i>=T`, see
`encoder_model.legal_action_types`) ever mask out a gold oracle action?

Replays every gold record's `linearize_tree` oracle steps through
`legal_action_types(strict_ground=True)` and asserts each oracle action was
legal at its state. Prints PASS/FAIL; on FAIL, dumps the first offending
record + step (this tells us whether the oracle ever GROUNDs at `i>=T` --
it should not).

Usage:
    python scripts/check_strict_ground_oracle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct import encoder_model as em

_GOLD_PATH = Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"


def main() -> int:
    with open(_GOLD_PATH) as f:
        records = [json.loads(line) for line in f]

    checked = 0
    failures = []
    for rec_idx, record in enumerate(records):
        T = len(record["tokens"])
        for tree_idx, tree in enumerate(record["lattice"]["trees"]):
            steps = em.linearize_tree(record, tree)
            open_clause = False
            has_clause = False
            i = 0
            for step_idx, s in enumerate(steps):
                legal = em.legal_action_types(open_clause, i, T, has_clause, strict_ground=True)
                checked += 1
                if s.action not in legal:
                    failures.append({
                        "record_idx": rec_idx,
                        "tree_idx": tree_idx,
                        "step_idx": step_idx,
                        "action": s.action,
                        "token_index": s.token_index,
                        "i": i,
                        "T": T,
                        "open_clause": open_clause,
                        "has_clause": has_clause,
                        "legal_at_state": legal,
                        "tokens": record["tokens"],
                    })
                if s.action == "OPEN_CLAUSE":
                    open_clause = True
                elif s.action == "CLOSE_CLAUSE":
                    open_clause = False
                    has_clause = True
                if s.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and s.token_index is not None:
                    i = s.token_index + 1

    print(f"checked {checked} oracle steps across {len(records)} gold records under strict_ground=True")
    if not failures:
        print("PASS: strict_ground=True never masks out a gold oracle action "
              "(the oracle never GROUNDs at i>=T in this corpus).")
        return 0

    print(f"FAIL: {len(failures)} gold oracle action(s) masked out under strict_ground=True")
    print("\nfirst offending record + step:")
    print(json.dumps(failures[0], indent=2, default=str))
    if len(failures) > 1:
        print(f"\n... and {len(failures) - 1} more failure(s) (not shown)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
