"""Probe: is the clause reactor's curriculum accuracy template overfitting?

CurriculumGenerator (episode.py, levels 1-6) draws every context sentence from
a tiny, fixed set of rigid templates ("X is in the Y .", "X moved to the Z .").
Because perception runs sentences through the experimental quantum_parser
(ParserInputEncoder._parse_graph -> nsm_ct.clause.extract_discourse ->
nsm_ct.clause_reactor.build_clause_batch), the reactor's reported accuracy
gains could in principle be keyed off the exact surface form rather than the
underlying (entity, relation, value) triple.

This script measures that honestly with :mod:`nsm_ct.curriculum2`, which
generates the SAME facts/logic through two DISJOINT surface template sets
(A, B — see curriculum2.TEMPLATES and its DROPPED_TEMPLATES for what didn't
survive quantum_parser verification), in three arms:

  (i)   train A   / val A    -- the status-quo shape (single fixed template family)
  (ii)  train A   / val B    -- unseen templates at val time: the overfit test
  (iii) train A+B / val A+B  -- mixed training: the fix, if (ii) drops sharply

Perception is fixed/grounded (usvs meaning source, dim 48); only the
ClauseReactor's GRU reaction policy trains, exactly as in
scripts/train_clause.py -- this script re-uses that machinery (ClauseReactor,
build_clause_batch) rather than re-implementing it, and only swaps the episode
source for nsm_ct.curriculum2.generate_varied_episodes.

Run:
    python scripts/probe_template_overfit.py [--episodes 240] [--epochs 70] [--dim 48]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import generate_varied_episodes, verify_templates  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _acc(out, batch) -> float:
    return float((out["answer_logits"].argmax(-1) == batch.answer).float().mean())


def _per_level(model, batch, episodes):
    """Val accuracy broken down by curriculum level (matches train_clause.py)."""
    with torch.no_grad():
        correct = (model(batch)["answer_logits"].argmax(-1) == batch.answer).tolist()
    agg = collections.defaultdict(lambda: [0, 0])
    for ep, ok in zip(episodes, correct):
        agg[f"L{ep.level}"][0] += int(ok)
        agg[f"L{ep.level}"][1] += 1
    return {k: (c, n) for k, (c, n) in sorted(agg.items())}


def report_parse_success() -> None:
    print("=" * 78)
    print("PARSE-SUCCESS TABLE -- curriculum2 templates through the real quantum_parser")
    print("=" * 78)
    results = verify_templates(("A", "B"))
    if not results:
        print("  quantum_parser unavailable in this environment; cannot verify templates.")
        print()
        return
    for t, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {t}")
    n_ok = sum(results.values())
    print(f"  -> {n_ok}/{len(results)} = {n_ok / len(results):.3f} parse-success rate "
          f"(kept templates only; see curriculum2.DROPPED_TEMPLATES for what failed)")
    print()


def _build_parser(episodes) -> ParserInputEncoder:
    texts = [t for e in episodes
             for t in e.context + [e.question] + (e.options or []) + (e.post_context or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    return ParserInputEncoder(tok)


def train_arm(label, train_eps, val_eps, *, dim, epochs, seed=0):
    """Train a fresh ClauseReactor on ``train_eps``, report val accuracy on
    ``val_eps``. Mirrors scripts/train_clause.py's loop exactly (full-batch
    AdamW, lr=3e-3, meaning_source='usvs') so results are comparable to it.
    """
    torch.manual_seed(seed)
    parser = _build_parser(train_eps + val_eps)
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    print(f"--- arm: {label} --- train={len(train_eps)} val={len(val_eps)}")
    train = build_clause_batch(train_eps, parser, resolver, codec, "usvs")
    val = build_clause_batch(val_eps, parser, resolver, codec, "usvs")

    model = ClauseReactor(dim=dim)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for epoch in range(epochs):
        model.train()
        out = model(train)
        loss = F.cross_entropy(out["answer_logits"], train.answer)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 9 or epoch == 0:
            model.eval()
            with torch.no_grad():
                to_, vo = model(train), model(val)
            print(f"  epoch {epoch + 1:3d} | loss={float(loss):.3f} "
                  f"train_acc={_acc(to_, train):.3f} val_acc={_acc(vo, val):.3f}")

    model.eval()
    with torch.no_grad():
        vo = model(val)
    final_acc = _acc(vo, val)
    levels = _per_level(model, val, val_eps)
    print(f"  final val_acc={final_acc:.3f}  per-level: "
          + "  ".join(f"{k}={c}/{n}={c / n:.2f}" for k, (c, n) in levels.items()))
    print()
    return final_acc, levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=240, help="total episodes per arm's train pool")
    ap.add_argument("--val-episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--max-level", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report_parse_success()

    n_tr, n_va = args.episodes, args.val_episodes

    # Arms (i) & (ii) share a train pool (template set A) and, thanks to
    # curriculum2's RNG bookkeeping, val_A and val_B (same seed=1000) draw the
    # SAME underlying facts/entities/options -- they differ ONLY in surface
    # template -- so any accuracy gap between them isolates the template
    # effect from ordinary sampling variance.
    train_A = generate_varied_episodes(n_tr, seed=args.seed, template_set="A",
                                        max_level=args.max_level)
    val_A = generate_varied_episodes(n_va, seed=1000, template_set="A",
                                      max_level=args.max_level)
    val_B = generate_varied_episodes(n_va, seed=1000, template_set="B",
                                      max_level=args.max_level)

    # Arm (iii): mixed training AND mixed (held-out) validation -- every
    # sentence independently drawn from A or B.
    train_AB = generate_varied_episodes(n_tr, seed=args.seed + 2000, template_set="AB",
                                         max_level=args.max_level)
    val_AB = generate_varied_episodes(n_va, seed=3000, template_set="AB",
                                       max_level=args.max_level)

    print("=" * 78)
    print("THREE-ARM TEMPLATE-OVERFIT PROBE")
    print("=" * 78)
    results = {}
    results["i: train A / val A"] = train_arm(
        "i: train A / val A (status quo)", train_A, val_A, dim=args.dim, epochs=args.epochs)
    results["ii: train A / val B"] = train_arm(
        "ii: train A / val B (unseen templates)", train_A, val_B, dim=args.dim, epochs=args.epochs)
    results["iii: train A+B / val A+B"] = train_arm(
        "iii: train A+B / val A+B (mixed)", train_AB, val_AB, dim=args.dim, epochs=args.epochs)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, (acc, levels) in results.items():
        per_level_str = "  ".join(f"{k}={c}/{n}={c / n:.2f}" for k, (c, n) in levels.items())
        print(f"{name:28s} val_acc={acc:.3f}   {per_level_str}")

    acc_i = results["i: train A / val A"][0]
    acc_ii = results["ii: train A / val B"][0]
    acc_iii = results["iii: train A+B / val A+B"][0]
    print()
    print(f"overfit gap        (i - ii)  : {acc_i - acc_ii:+.3f}  "
          f"(positive => accuracy drops on unseen templates => template overfitting)")
    print(f"mixed-train recovery (iii - ii): {acc_iii - acc_ii:+.3f}  "
          f"(positive => training on both templates recovers the drop)")


if __name__ == "__main__":
    main()
