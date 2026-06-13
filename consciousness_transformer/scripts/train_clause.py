"""Train the token-free clause reactor on the curriculum.

Perception (parse → clause → TPR triple) is fixed/grounded; only the GRU reaction
policy trains, by a contrastive answer loss over the fixed option meaning-vectors.

Run:
    python scripts/train_clause.py [--episodes 240] [--epochs 60] [--dim 64]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.dataset import PARSE_LABELS, split_episodes  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _acc(out, batch):
    return float((out["answer_logits"].argmax(-1) == batch.answer).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--max-level", type=int, default=6)
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
    print(f"clause streams: T_train={train.entity.shape[1]} K={train.options.shape[1]} "
          f"| params learn ONLY the reaction policy")

    model = ClauseReactor(dim=args.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"reactor params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(args.epochs):
        model.train()
        out = model(train)
        loss = F.cross_entropy(out["answer_logits"], train.answer)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 9 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vo = model(val)
            print(f"epoch {epoch+1:3d} | loss={float(loss):.3f} "
                  f"train_acc={_acc(out, train):.3f} val_acc={_acc(vo, val):.3f} "
                  f"resp@q={float(out['respond_position'].mean()):.2f}")


if __name__ == "__main__":
    main()
