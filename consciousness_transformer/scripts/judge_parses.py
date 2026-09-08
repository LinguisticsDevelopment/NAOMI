#!/usr/bin/env python3
"""Judge a gold or encoder-decoded parse tree per sentence with the LLM
parse judge (dev/PARSE_JUDGE.md) -- the admission gate for self-training on
the encoder's own good parses (see `nsm_ct.parse_judge` module docstring).

For each input record, writes back a `judge` field:
    {"verdict": "good"|"bad", "problems": [...], "confidence": float,
     "backend": "haiku"|"mock", "model": str}

Usage:
    python scripts/judge_parses.py --in runs/encoder_gold_v2.jsonl \\
        --out runs/judged.jsonl --tree-source gold --backend mock

    python scripts/judge_parses.py --in runs/some_sentences.jsonl \\
        --out runs/judged.jsonl --tree-source encoder --backend haiku --batch \\
        --filter good
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.parse_judge import HaikuJudge, MockJudge, judge_many_with_prefilter  # noqa: E402
from nsm_ct.tree_render import normalize_gold_tree  # noqa: E402

# Standard per-token pricing (Claude Haiku 4.5, cached: 2026-09-08). Batches
# API = 50% of these rates.
HAIKU_INPUT_PER_M = 1.0
HAIKU_OUTPUT_PER_M = 5.0


def load_gold(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def gold_tree_of(record: dict) -> dict:
    trees = record.get("lattice", {}).get("trees", [])
    if not trees:
        return {"clauses": []}
    return normalize_gold_tree(record, trees[0])


def encoder_tree_of(record: dict, model, usvs, pos_vocab, hash_buckets) -> dict:
    from nsm_ct import encoder_model as em

    feats = em.build_features(record, usvs, pos_vocab, hash_buckets)
    forest = em.beam_decode(model, feats, beam_width=8, k=8, policy="model")
    return forest[0] if forest else {"clauses": []}


def load_encoder(checkpoint: str):
    import torch

    from nsm_ct import encoder_model as em

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = em.EncoderModel(ckpt["pos_vocab"], ckpt["role_vocab"], d_axes=ckpt["d_axes"],
                             hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                             controller_hidden=ckpt["d_model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["pos_vocab"], ckpt["hash_buckets"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--tree-source", choices=["gold", "encoder"], default="gold")
    ap.add_argument("--backend", choices=["haiku", "mock"], default="mock")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (haiku only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter", choices=["good", "bad"], default=None,
                     help="write out only records whose verdict matches")
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "encoder_colab.pt"))
    ap.add_argument("--usvs-dir", default=str(ROOT / "data" / "usvs"))
    args = ap.parse_args()

    records = load_gold(args.in_path)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"loaded {len(records)} records from {args.in_path}")

    if args.tree_source == "gold":
        trees = [gold_tree_of(r) for r in records]
    else:
        from nsm_ct.ground.usvs import load_usvs

        print("loading encoder checkpoint + usvs ...")
        model, pos_vocab, hash_buckets = load_encoder(args.checkpoint)
        usvs = load_usvs(args.usvs_dir)
        trees = [encoder_tree_of(r, model, usvs, pos_vocab, hash_buckets) for r in records]

    judge = HaikuJudge() if args.backend == "haiku" else MockJudge()
    if args.batch and args.backend != "haiku":
        print("--batch has no effect for --backend mock (no API calls to batch)")

    verdicts = judge_many_with_prefilter(judge, list(zip(records, trees)), batch=args.batch)

    n_good = sum(1 for v in verdicts if v.verdict == "good")
    n_bad = len(verdicts) - n_good
    print(f"verdicts: good={n_good} bad={n_bad} (of {len(verdicts)})")

    out_records = []
    for record, tree, verdict in zip(records, trees, verdicts):
        record = dict(record)
        record["judge"] = {
            "verdict": verdict.verdict,
            "problems": verdict.problems,
            "confidence": verdict.confidence,
            "backend": judge.name,
            "model": judge.model,
        }
        if args.filter is not None and verdict.verdict != args.filter:
            continue
        out_records.append(record)

    with open(args.out_path, "w") as f:
        for record in out_records:
            f.write(json.dumps(record) + "\n")
    print(f"wrote {len(out_records)} records -> {args.out_path}"
          + (f" (filtered to verdict={args.filter})" if args.filter else ""))

    if isinstance(judge, HaikuJudge) and judge.total_requests:
        per_m = 0.5 if args.batch else 1.0
        cost = (judge.total_input_tokens * HAIKU_INPUT_PER_M
                + judge.total_output_tokens * HAIKU_OUTPUT_PER_M) / 1_000_000 * per_m
        print(f"haiku usage: requests={judge.total_requests} "
              f"input_tokens={judge.total_input_tokens} output_tokens={judge.total_output_tokens}")
        print(f"estimated cost: ${cost:.4f}"
              f" (${'0.50' if args.batch else '1.00'}/M input, ${'2.50' if args.batch else '5.00'}/M output)")
        per_1k = cost / max(1, judge.total_requests) * 1000
        print(f"estimated cost per 1,000 judged trees: ${per_1k:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
