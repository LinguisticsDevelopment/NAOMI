"""Evaluate Mind v3 (and the v2b checkpoint on the same items) on one split: the v2 evaluation code with the thinker dispatch.
    python scripts/eval_mind_v3.py --dir mind_v3 --split val --topk 1 --models v3=runs/mind_v3/thinker_v3_s1_best.pt v2b=runs/mind_v2b/thinker_s2b_all_best.pt --main v3
runs/mind_v3 holds symlinks to the v2b build (stories.pkl, pairs_*.pkl, items_meta.json, vocab.json, read_cache); outputs are
runs/mind_v3/eval_<split><tag>.json, eval_rows_<split><tag>.jsonl.  Run on the test split ONCE, after every choice is made on val.
--handread N writes the seeded hand-read sample (N/2 low-ROUGE answered, N/2 abstained) after the evaluation."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v2 as m2  # noqa: E402
from naomi import mind_v3 as m3  # noqa: E402
import eval_mind_v2 as E  # noqa: E402
import train_mind_v3 as T3  # noqa: E402


def take(flag, default, cast):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return v
    return default


def handread(split, main, n, tag):
    rows = [json.loads(l) for l in open(CT / "runs" / "mind_v3" / f"eval_rows_{split}{tag}.jsonl")]
    rng = random.Random(0)
    low = [r for r in rows if r.get(f"{main}_answered") and m2.rouge_multi(r[f"{main}_text_a"], r["golds"]) <= 0.05]
    ab = [r for r in rows if not r.get(f"{main}_answered")]
    out = {"low": [], "abstain": [], "n_low_total": len(low), "n_abstain_total": len(ab)}
    for key, pool in (("low", low), ("abstain", ab)):
        for r in rng.sample(pool, min(n // 2, len(pool))):
            out[key].append({"id": r["id"], "question": r["question"], "reference": r["reference"], "answer": r.get(f"{main}_text_a"), "p_abstain": r.get(f"{main}_p_abstain"),
                             "band": r["band"], "oracle_text": r["oracle_subtree_text"], "cheat_text": r["cheat_text"]})
    json.dump(out, open(CT / "runs" / "mind_v3" / f"eval_handread_{n}{tag}.json", "w"), indent=1)


def main():
    topk = take("--topk", 1, int)
    hr = take("--handread", 0, int)
    E.m2 = m3.Mind(topk)
    E.T = T3
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "val"
    tag = sys.argv[sys.argv.index("--out_tag") + 1] if "--out_tag" in sys.argv else ""
    mainm = sys.argv[sys.argv.index("--main") + 1]
    E.main()
    if hr:
        handread(split, mainm, hr, tag)


if __name__ == "__main__":
    main()
