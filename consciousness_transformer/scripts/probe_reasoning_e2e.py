"""Stage R4 — the reasoning north star, end to end.

For a conditional, a transitivity, and an unanswerable episode: the trained
ClausePsyche reads the grounded premises, runs its inference loop, and either
GENERATES the derived answer or ABSTAINS ("I don't know") — shown beside the
oracle's gold derivation chain (the justification a correct answer aligns with).

Run:
    python scripts/train_clause_psyche.py --hops 3 --save /tmp/psyche_r.pt
    python scripts/probe_reasoning_e2e.py --ckpt /tmp/psyche_r.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_psyche import ClausePsyche  # noqa: E402
from nsm_ct.clause_reactor import build_clause_batch  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="/tmp/psyche_r.pt")
    args = ap.parse_args()
    if not os.path.exists(args.ckpt):
        print(f"no checkpoint at {args.ckpt}; run train_clause_psyche.py --save first.")
        return

    ck = torch.load(args.ckpt, weights_only=False)
    codec = TPRCodec(dim=ck["dim"])
    model = ClausePsyche(codec, hops=ck.get("hops", 3))
    model.load_state_dict(ck["state_dict"]); model.eval()
    resolver = NSMMeaningResolver()

    gen = CurriculumGenerator(max_level=11, seed=123)
    pool = gen.generate(120)
    picks = [next(e for e in pool if e.level == lvl) for lvl in (9, 10, 11)]
    batch = build_clause_batch(picks, None, resolver, codec)
    with torch.no_grad():
        out = model(batch)
        place = out["place_filler"]
        pick = torch.einsum("bd,bkd->bk", torch.nn.functional.normalize(place, -1),
                            torch.nn.functional.normalize(batch.options, -1)).argmax(-1)
        abstained = (out["abstain_prob"] >= 0.5).tolist()

    for i, ep in enumerate(picks):
        print(f"=== L{ep.level} ===")
        for s in ep.context:
            print(f"   {s}")
        print(f"   Q: {ep.question}")
        if abstained[i]:
            verdict = "ABSTAIN: 'I don't know'"
        else:
            verdict = f"ANSWER: {ep.options[int(pick[i])]!r}"
        correct = (abstained[i] == (not ep.answerable)) and (
            ep.answerable is False or ep.options[int(pick[i])] == ep.answer_text)
        print(f"   model -> {verdict}   (gold {ep.answer_text!r}, {'OK' if correct else 'X'})")
        for derived, rule, support in (ep.gold_chain or []):
            print(f"   justification: {derived} via {rule} from {list(support)}")
        print()


if __name__ == "__main__":
    main()
