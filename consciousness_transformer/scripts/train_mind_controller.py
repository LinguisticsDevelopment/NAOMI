"""Train the M3 MindController with teacher op-trace supervision (soft→discrete).

The proven focus-chaining controller (ClausePsyche) trained with the answer signal
PLUS the M2 teacher's gold supervision (relation-to-follow CE + optional halt CE),
annealed high→low. Reports held-out per-level decode, op-trace (relation) match,
abstain P/R, and — the headline — the L12 deep-chain decode vs the hops=0
single-pass baseline (reproducing §0n: the loop is necessary AND sufficient for depth).

Run:
    python scripts/train_mind_controller.py [--episodes 400] [--epochs 200]
        [--dim 48] [--hops 5] [--halting] [--baseline]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_psyche import abstain_prf, clause_decode_accuracy  # noqa: E402
from nsm_ct.clause_reactor import build_clause_batch  # noqa: E402
from nsm_ct.data_structures import ParseNode, ParseTree  # noqa: E402
from nsm_ct.dataset import split_episodes  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.mind import teacher  # noqa: E402
from nsm_ct.mind.controller import MindController, relation_match  # noqa: E402
from nsm_ct.mind.controller_losses import combined_loss  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


class _StubResolver:
    """Deterministic, dependency-free word→meaning (reasoning levels need no parser)."""
    def resolve(self, word: str) -> ParseTree:
        return ParseTree(root=ParseNode(label=word, token=word))


def _supervision(eps, hops, device):
    s = teacher.build_supervision(eps, hops)
    return {"rel_targets": torch.from_numpy(s["rel_targets"]).to(device),
            "depth": torch.from_numpy(s["depth"]).to(device),
            "answerable": torch.from_numpy(s["answerable"]).to(device)}


def _per_level(out, episodes):
    pred = out["answer_logits"].argmax(-1).tolist()
    abst = (out["abstain_prob"] >= 0.5).tolist()
    agg = collections.defaultdict(lambda: [0, 0])
    for ep, p, ab in zip(episodes, pred, abst):
        hit = (p == ep.answer_idx)
        ok = ab if not ep.answerable else (hit and not ab)
        agg[f"L{ep.level}"][0] += int(ok); agg[f"L{ep.level}"][1] += 1
    return {k: (c, n) for k, (c, n) in sorted(agg.items())}


def _train(eps_tr, eps_va, codec, hops, halting, epochs, device, supervise=True):
    tr = build_clause_batch(eps_tr, None, _StubResolver(), codec).to(device)
    va = build_clause_batch(eps_va, None, _StubResolver(), codec).to(device)
    sup_tr = _supervision(eps_tr, hops, device)
    sup_va = _supervision(eps_va, hops, device)
    model = MindController(codec, hidden=96, hops=hops, halting=halting).to(device)
    codebook = model.relation_codebook
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tag = f"hops={hops}" + ("+sup" if supervise else "")
    for epoch in range(epochs):
        model.train()
        frac = epoch / max(epochs - 1, 1)
        temperature = 2.0 * (1 - frac) + 0.3 * frac          # soft -> sharp
        w_rel = (2.0 * (1 - frac) + 0.5 * frac) if supervise else 0.0
        w_halt = (1.0 if (halting and supervise) else 0.0)
        out = model(tr)
        loss = combined_loss(out, tr, model, sup_tr, codebook,
                             temperature=temperature, w_rel=w_rel, w_halt=w_halt,
                             w_prior=0.3 if halting else 0.0)
        opt.zero_grad(); loss["total"].backward(); opt.step()
        if epoch % 40 == 39 or epoch == 0:
            rm = relation_match(out, sup_tr["rel_targets"], codebook) if supervise else 0.0
            print(f"  [{tag}] epoch {epoch+1:3d} loss={float(loss['total'].detach()):.3f} "
                  f"train_rel_match={rm:.2f}", flush=True)
    model.eval()
    with torch.no_grad():
        vo = model(va)
    return model, va, vo, sup_va, codebook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hops", type=int, default=5)
    ap.add_argument("--halting", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="also train hops=0 single-pass")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")

    alleps = CurriculumGenerator(max_level=13, seed=0).generate(args.episodes)
    eps = [e for e in alleps if e.level in (9, 10, 11, 12, 13)]
    tr, va = split_episodes(eps, 0.25, seed=0)
    codec = TPRCodec(dim=args.dim)
    print(f"M3 controller: {len(tr)} train / {len(va)} val reasoning episodes "
          f"(hops={args.hops}, halting={args.halting})")

    model, vb, vo, sup_va, codebook = _train(tr, va, codec, args.hops, args.halting, args.epochs, device)
    levels = _per_level(vo, va)
    print("per-level val (L11=abstain):",
          "  ".join(f"{k}={c}/{n}={c/n:.2f}" for k, (c, n) in levels.items()))
    print(f"overall decode (answerable): {clause_decode_accuracy(vo, vb):.3f}")
    print(f"op-trace (relation-to-follow) match: {relation_match(vo, sup_va['rel_targets'], codebook):.3f}")
    prf = abstain_prf(vo, vb)
    print(f"abstain P/R: {prf['precision']:.2f}/{prf['recall']:.2f}")

    if args.baseline:
        _, vb0, vo0, _, _ = _train(tr, va, codec, 0, False, args.epochs, device, supervise=False)
        l12 = _per_level(vo, va).get("L12", (0, 1))
        l12_0 = _per_level(vo0, va).get("L12", (0, 1))
        print("\n--- multi-hop ablation (L12 deep is-a chains) ---")
        print(f"  hops=0 single-pass : {l12_0[0]}/{l12_0[1]} = {l12_0[0]/max(l12_0[1],1):.2f}")
        print(f"  hops={args.hops} focus-chaining: {l12[0]}/{l12[1]} = {l12[0]/max(l12[1],1):.2f}")


if __name__ == "__main__":
    main()
