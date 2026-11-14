"""M52 gate — does multi-arg transfer data cost the old curriculum anything?

Two sequential arms, same config/seed:
  control: old curriculum only (CurriculumGenerator levels 1-6)
  mixed:   2/3 old + 1/3 transfer (curriculum2.generate_transfer_episodes)

Gate: mixed arm's old-subset val accuracy within noise of control's, AND
transfer-subset val accuracy well above chance (1/num_options), per-level.

Usage: python scripts/probe_m52_transfer.py [--episodes 1500] [--dim 48]
       [--epochs 60]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

torch.set_num_threads(1)

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import generate_transfer_episodes  # noqa: E402
from nsm_ct.episode import CurriculumGenerator, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _is_transfer(e) -> bool:
    return str(e.meta.get("kind", "")).startswith("transfer")


def run_arm(name: str, episodes, dim: int, epochs: int):
    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, resolver, codec)
    va = build_clause_batch(va_eps, parser, resolver, codec)

    torch.manual_seed(0)
    model = ClauseReactor(dim=dim)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])
    gold_va = torch.tensor([e.answer_idx for e in va_eps])

    t0 = time.time()
    for i in range(epochs):
        out = model(tr)
        loss = F.cross_entropy(out["answer_logits"], gold_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 20 == 0 or i == 0:
            with torch.no_grad():
                acc = (model(va)["answer_logits"].argmax(-1) == gold_va).float().mean()
            print(f"  [{name}] epoch {i+1:3d} loss={loss.item():.3f} val={acc:.3f}", flush=True)

    with torch.no_grad():
        pred = model(va)["answer_logits"].argmax(-1)
    total = float((pred == gold_va).float().mean())
    print(f"  [{name}] val total={total:.3f}  time={(time.time()-t0)/60:.1f} min", flush=True)
    per_level = {}
    for i, e in enumerate(va_eps):
        per_level.setdefault(str(e.meta.get("kind", "old")), []).append(bool(pred[i] == e.answer_idx))
    for lv in sorted(per_level):
        w = per_level[lv]
        print(f"    level {lv}: {sum(w)}/{len(w)} = {sum(w)/len(w):.2f}")
    return va_eps, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    print(f"=== control: old-only (L1-6), {args.episodes} eps, dim={args.dim}, "
          f"{args.epochs} epochs ===", flush=True)
    old = CurriculumGenerator(max_level=6, seed=11).generate(args.episodes)
    run_arm("control", old, args.dim, args.epochs)

    print("\n=== mixed: 2/3 old + 1/3 transfer ===", flush=True)
    n_new = args.episodes // 3
    mixed = (CurriculumGenerator(max_level=6, seed=11).generate(args.episodes - n_new)
             + list(generate_transfer_episodes(n_new, seed=22)))
    mixed = [mixed[i] for i in np.random.RandomState(7).permutation(len(mixed))]
    va_eps, pred = run_arm("mixed", mixed, args.dim, args.epochs)

    for label, keep in (("old-subset", lambda e: not _is_transfer(e)),
                        ("transfer-subset", _is_transfer)):
        idx = [i for i, e in enumerate(va_eps) if keep(e)]
        if idx:
            hits = sum(bool(pred[i] == va_eps[i].answer_idx) for i in idx)
            print(f"mixed {label}: {hits}/{len(idx)} = {hits/len(idx):.3f}")


if __name__ == "__main__":
    main()
