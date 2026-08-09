"""M53b -- resolver training script: Track A (CorefHead) vs Track B
(SharedScorer) vs the --gold-binding ceiling (no resolver, M53a's placeholder
gold binding), on the SAME mixed curriculum. dev/RESOLVER_BUILD_PLAN.md Phase 2
"Agent 3"; follows scripts/probe_m52_transfer.py's arm pattern.

Mixed curriculum: 1/2 old L1-6 + 1/4 transfer + 1/4 pronoun (deterministic given
--episodes/--seed). Reports: val accuracy overall + per curriculum kind
(includes the pronoun_binding level), RESOLVER BINDING ACCURACY (chose the gold
antecedent) overall and on the anti-recency half vs the scripted nearest-entity
baseline (0.500 / 0.000), the resolver's param count, and its margin
distribution.

Usage:
    python scripts/train_resolver.py --track A
    python scripts/train_resolver.py --track B
    python scripts/train_resolver.py --gold-binding       # the ceiling arm
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
from nsm_ct.curriculum2 import (  # noqa: E402
    generate_pronoun_episodes,
    generate_transfer_episodes,
    nearest_entity_baseline,
)
from nsm_ct.episode import CurriculumGenerator, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.resolver import make_resolver  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

AUX_WEIGHT = 0.5   # resolver cross-entropy weight added to the answer loss (training-script side, not the model)


def build_mixed_curriculum(n_episodes: int, seed: int):
    """1/2 old L1-6 + 1/4 transfer + 1/4 pronoun (RESOLVER_BUILD_PLAN Phase 2),
    deterministic given (n_episodes, seed)."""
    n_pronoun = n_episodes // 4
    n_transfer = n_episodes // 4
    n_old = n_episodes - n_pronoun - n_transfer
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    transfer = generate_transfer_episodes(n_transfer, seed=seed + 1)
    pronoun = generate_pronoun_episodes(n_pronoun, seed=seed + 2)
    episodes = old + transfer + pronoun
    order = np.random.RandomState(seed + 3).permutation(len(episodes))
    return [episodes[i] for i in order]


def resolver_binding_stats(out, batch, va_eps):
    """RESOLVER BINDING ACCURACY overall + anti-recency half, and the raw margin
    list, read off one resolver-carrying forward pass + the episodes' meta."""
    if "resolver_logits" not in out:
        return None
    cand_gold = batch.cand_gold
    has_cand = cand_gold >= 0
    pred_idx = out["resolver_logits"].argmax(-1)
    correct = (pred_idx == cand_gold) & has_cand
    margins = []
    hits = anti_hits = anti_total = total = 0
    for i, e in enumerate(va_eps):
        row_mask = has_cand[i]
        if not bool(row_mask.any()):
            continue
        t = int(row_mask.nonzero()[0, 0])
        total += 1
        is_hit = bool(correct[i, t])
        hits += int(is_hit)
        margins.append(float(out["resolver_margin"][i, t]))
        if e.meta.get("antecedent_recency") == "old":
            anti_total += 1
            anti_hits += int(is_hit)
    if total == 0:
        return None
    return {
        "overall_acc": hits / total, "n": total,
        "anti_recency_acc": (anti_hits / anti_total) if anti_total else float("nan"),
        "n_anti_recency": anti_total,
        "margins": margins,
    }


def run_arm(name: str, track, episodes, dim: int, epochs: int, seed: int, hidden: int = 128):
    """``track``: "A" | "B" | None (None = --gold-binding ceiling, no resolver)."""
    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return None
    meaning_resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, meaning_resolver, codec)
    va = build_clause_batch(va_eps, parser, meaning_resolver, codec)

    torch.manual_seed(seed)
    resolver = make_resolver(track, dim, hidden) if track else None
    model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver)
    n_resolver_params = sum(p.numel() for p in resolver.parameters()) if resolver else 0
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])
    gold_va = torch.tensor([e.answer_idx for e in va_eps])

    t0 = time.time()
    model.train()
    for i in range(epochs):
        out = model(tr)
        loss = F.cross_entropy(out["answer_logits"], gold_tr)
        if resolver is not None and "resolver_logits" in out:
            cg = tr.cand_gold
            has_cand = cg >= 0
            if bool(has_cand.any()):
                aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
                loss = loss + AUX_WEIGHT * aux
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 20 == 0 or i == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(va)["answer_logits"].argmax(-1) == gold_va).float().mean()
            model.train()
            print(f"  [{name}] epoch {i+1:3d} loss={loss.item():.3f} val={acc:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        out_va = model(va)
    pred = out_va["answer_logits"].argmax(-1)
    total_acc = float((pred == gold_va).float().mean())
    print(f"  [{name}] val total={total_acc:.3f}  resolver_params={n_resolver_params}  "
          f"time={(time.time()-t0)/60:.2f} min", flush=True)

    per_kind = {}
    for i, e in enumerate(va_eps):
        per_kind.setdefault(str(e.meta.get("kind", "old")), []).append(bool(pred[i] == gold_va[i]))
    for k in sorted(per_kind):
        w = per_kind[k]
        print(f"    kind {k}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")

    binding = resolver_binding_stats(out_va, va, va_eps)
    if binding is not None:
        m = np.array(binding["margins"]) if binding["margins"] else np.array([0.0])
        print(f"  [{name}] RESOLVER BINDING ACCURACY overall={binding['overall_acc']:.3f} "
              f"(n={binding['n']}) anti-recency={binding['anti_recency_acc']:.3f} "
              f"(n={binding['n_anti_recency']}) vs baseline 0.500/0.000", flush=True)
        print(f"  [{name}] margin distribution: min={m.min():.3f} p25={np.percentile(m, 25):.3f} "
              f"median={np.median(m):.3f} p75={np.percentile(m, 75):.3f} max={m.max():.3f}", flush=True)
    return {"total_acc": total_acc, "n_resolver_params": n_resolver_params, "binding": binding}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B"], default=None)
    ap.add_argument("--gold-binding", action="store_true",
                     help="baseline/ceiling arm: no resolver, M53a placeholder gold binding")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.gold_binding and args.track is None:
        raise SystemExit("pass --track A|B or --gold-binding")

    episodes = build_mixed_curriculum(args.episodes, args.seed)
    baseline = nearest_entity_baseline(episodes)
    print(f"nearest-entity baseline: overall={baseline['accuracy']:.3f} (n={baseline['n']}) "
          f"anti-recency={baseline['anti_recency_accuracy']:.3f} (n={baseline['n_anti_recency']})",
          flush=True)

    if args.gold_binding:
        print(f"=== gold-binding ceiling: no resolver, {args.episodes} eps, dim={args.dim} ===",
              flush=True)
        run_arm("gold-binding", None, episodes, args.dim, args.epochs, args.seed, args.hidden)
    else:
        print(f"=== track {args.track}: {args.episodes} eps, dim={args.dim} ===", flush=True)
        run_arm(f"track-{args.track}", args.track, episodes, args.dim, args.epochs, args.seed, args.hidden)


if __name__ == "__main__":
    main()
