"""Train ClausePsyche on the curriculum — it GENERATES a clause meaning-object.

Perception (parse → grounded clause stream) is the fixed shared pathway; only the
GRU controller + generators train, by Frobenius to the gold clause matrix +
a decode cross-entropy + a consciousness-state consistency term. Reports per-level
exact clause-decode accuracy (the non-gameable metric replacing MC accuracy).

Run:
    python scripts/train_clause_psyche.py [--episodes 240] [--epochs 60] [--dim 64]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_psyche import (  # noqa: E402
    ClausePsyche,
    clause_decode_accuracy,
    compute_clause_psyche_losses,
)
from nsm_ct.clause_reactor import build_clause_batch  # noqa: E402
from nsm_ct.dataset import PARSE_LABELS, split_episodes  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _per_level(model, batch, episodes):
    with torch.no_grad():
        ok = (model(batch)["answer_logits"].argmax(-1) == batch.answer).tolist()
    agg = collections.defaultdict(lambda: [0, 0])
    for ep, hit in zip(episodes, ok):
        key = f"L{ep.level}"
        if ep.level == 7:
            key += "-res" if ep.meta.get("resolved") else "-may"
        agg[key][0] += int(hit); agg[key][1] += 1
    return {k: tuple(v) for k, v in sorted(agg.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--save", type=str, default="", help="checkpoint path (optional)")
    args = ap.parse_args()
    torch.manual_seed(0)

    eps = CurriculumGenerator(max_level=args.max_level, seed=0).generate(args.episodes)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=args.dim)

    tr, va = split_episodes(eps, 0.2, seed=0)
    print(f"encoding {len(tr)} train / {len(va)} val episodes (fixed perception)...")
    train = build_clause_batch(tr, parser, resolver, codec)
    val = build_clause_batch(va, parser, resolver, codec)

    model = ClausePsyche(codec)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"ClausePsyche params (reaction + generators only): "
          f"{sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        model.train()
        out = model(train)
        loss = compute_clause_psyche_losses(out, train, model)
        opt.zero_grad(); loss["total"].backward(); opt.step()
        if epoch % 10 == 9 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vo = model(val)
            print(f"epoch {epoch+1:3d} | frob={float(loss['frobenius']):.2f} "
                  f"decode={float(loss['decode']):.3f} cons={float(loss['consistency']):.4f} "
                  f"| train_dec={clause_decode_accuracy(out, train):.3f} "
                  f"val_dec={clause_decode_accuracy(vo, val):.3f}")

    model.eval()
    levels = _per_level(model, val, va)
    print("per-level val clause-decode:",
          "  ".join(f"{k}={c}/{n}={c/n:.2f}" for k, (c, n) in levels.items()))
    if args.save:
        torch.save({"state_dict": model.state_dict(), "dim": args.dim}, args.save)
        print(f"saved checkpoint -> {args.save}")


if __name__ == "__main__":
    main()
