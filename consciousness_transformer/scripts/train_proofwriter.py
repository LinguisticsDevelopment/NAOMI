"""Train the learned controller on real ProofWriter (verification mode, M8 step 2).

Facts + rules + a query -> {true, false, idk} (Unknown). The controller ingests
the theory + query and scores its generated answer-vector against the three answer
atoms (the existing contrastive MC head); training is the standard answer loss.
Reports held-out accuracy BY REASONING DEPTH — the breadth result on real data.

Run:
    python scripts/fetch_proofwriter.py
    python scripts/train_proofwriter.py [--train-per-depth 250] [--epochs 150]
        [--dim 64] [--hops 5] [--depths 0,1,2,3]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_psyche import compute_clause_psyche_losses  # noqa: E402
from nsm_ct.mind.controller import MindController  # noqa: E402
from nsm_ct.mind.datasets import proofwriter as pw  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _load_items(split: str, depths, per_depth: int):
    items = []
    base = pw.default_data_dir()
    for d in depths:
        path = os.path.join(base, f"owa-depth{d}-{split}.jsonl")
        if not os.path.exists(path):
            print(f"missing {path} — run scripts/fetch_proofwriter.py"); sys.exit(1)
        exs = [pw.parse_record(r) for r in pw.load_records(path, limit=per_depth)]
        items += pw.flatten(exs)
    return items


def _accuracy_by_depth(model, items, codec):
    batch = pw.build_pw_batch(items, codec)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    pred = out["answer_logits"].argmax(-1).tolist()
    agg = collections.defaultdict(lambda: [0, 0])
    tot = [0, 0]
    for it, p in zip(items, pred):
        gold, depth = it[3], it[4]
        hit = (p == gold)
        agg[depth][0] += hit; agg[depth][1] += 1
        tot[0] += hit; tot[1] += 1
    return tot, agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-depth", type=int, default=250)
    ap.add_argument("--test-per-depth", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hops", type=int, default=5)
    ap.add_argument("--depths", type=str, default="0,1,2,3")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    depths = args.depths.split(",")

    codec = TPRCodec(dim=args.dim)
    train_items = _load_items("train", depths, args.train_per_depth)
    test_items = _load_items("test", depths, args.test_per_depth)
    print(f"ProofWriter verification: {len(train_items)} train / {len(test_items)} test "
          f"questions (depths {depths}, dim={args.dim}, hops={args.hops})")
    train = pw.build_pw_batch(train_items, codec)
    model = MindController(codec, hidden=96, hops=args.hops, halting=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    import collections as C
    maj = C.Counter(it[3] for it in test_items).most_common(1)[0][1]
    maj_acc = maj / len(test_items)

    def report(tag):
        tot, agg = _accuracy_by_depth(model, test_items, codec)
        bydepth = "  ".join(f"d{d}={agg[d][0]/max(agg[d][1],1):.2f}" for d in sorted(agg))
        print(f"  [{tag}] held-out={tot[0]/tot[1]:.3f} (maj {maj_acc:.3f}) | {bydepth}", flush=True)

    n = len(train_items)
    bs = args.batch_size
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        last = 0.0
        for i in range(0, n, bs):
            mb = train.subset(perm[i:i + bs])
            out = model(mb)
            loss = compute_clause_psyche_losses(out, mb, model)
            opt.zero_grad(); loss["total"].backward(); opt.step()
            last = float(loss["total"].detach())
        if epoch % 5 == 4 or epoch == 0:           # periodic held-out eval (captured even if capped)
            print(f"  epoch {epoch+1:3d} loss={last:.3f}", flush=True)
            report(f"epoch {epoch+1}")
            if args.save:
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                torch.save({"sd": model.state_dict(), "dim": args.dim, "hops": args.hops}, args.save)

    print("\n=== FINAL ===")
    tot, agg = _accuracy_by_depth(model, test_items, codec)
    print(f"held-out accuracy: {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.3f}")
    print("by reasoning depth:")
    for d in sorted(agg):
        c, n2 = agg[d]
        print(f"  depth {d}: {c}/{n2} = {c/max(n2,1):.3f}")
    print(f"majority-class baseline: {maj}/{len(test_items)} = {maj_acc:.3f}")
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"sd": model.state_dict(), "dim": args.dim, "hops": args.hops}, args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
