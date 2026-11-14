"""Run the M4 two-loop system long — both loops — and then TALK to the trained brain.

Each round: the **subconscious** loop self-trains the M3 controller (replay +
generated episodes, accumulating iterations) and consolidates what it was told
into LTM + pre-derives multi-hop facts (offline inference); held-out decode /
op-trace are tracked. Checkpointable, so a long run resumes across wall-clock caps.
At the end it **talks**: the *learned controller* answers held-out questions
(natural language in via the membrane, answer + faithful reasoning out).

Run:
    python scripts/run_mind_loops.py --rounds 12 --steps 30 --save runs/mind.pt
    python scripts/run_mind_loops.py --rounds 12 --resume runs/mind.pt --save runs/mind.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.mind import membrane, ops, teacher  # noqa: E402
from nsm_ct.mind.conscious_loop import ConsciousLoop  # noqa: E402
from nsm_ct.mind.controller import MindController  # noqa: E402
from nsm_ct.mind.knowledge import KnowledgeGraph  # noqa: E402
from nsm_ct.mind.subconscious_loop import SubconsciousLoop  # noqa: E402
from nsm_ct.mind.verbalize import verbalize_trace  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

_REASONING = (9, 10, 11, 12, 13)


def _talk(controller, codec, val, n=6):
    """The learned controller answers held-out questions; verbalize the reasoning."""
    loop = ConsciousLoop(KnowledgeGraph(codec=codec), controller=controller, codec=codec)
    print("\n=== talking to the trained brain (held-out; learned controller answers) ===")
    hits = 0
    shown = 0
    for ep in val:
        if ep.level not in (9, 10, 12):
            continue
        out = loop.respond(ep)                              # the LEARNED brain answers
        correct = (out["answer"] == ep.answer_text)
        hits += int(correct)
        if shown < n:
            shown += 1
            res = teacher.replay(ep)                        # provenance for the because-chain
            support = next((s.support for s in res["trace"] if s.op == ops.INFER), [])
            q = membrane.render_query(*ep.meta["query"])
            print(f"\n  Q: \"{q}\"")
            print(f"  learned answer: {out['answer']}   ({'correct' if correct else 'WRONG; gold=' + ep.answer_text})")
            print(f"  reasoning: {verbalize_trace(tuple(ep.meta['query']), ep.answer_text, support)}")
    total = sum(1 for e in val if e.level in (9, 10, 12))
    print(f"\n  learned-controller accuracy on held-out L9/L10/L12: {hits}/{total} = {hits/max(total,1):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--episodes-per-round", type=int, default=160)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hops", type=int, default=5)
    ap.add_argument("--no-halting", action="store_true")
    ap.add_argument("--total-rounds", type=int, default=24, help="anneal horizon")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--talk-only", action="store_true", help="load a checkpoint and just talk")
    args = ap.parse_args()
    torch.manual_seed(0)

    codec = TPRCodec(dim=args.dim)
    val = [e for e in CurriculumGenerator(max_level=13, seed=999).generate(200)
           if e.level in _REASONING]

    ltm = KnowledgeGraph(codec=codec)
    controller = MindController(codec, hidden=96, hops=args.hops, halting=not args.no_halting)
    sub = SubconsciousLoop(ltm, controller, codec=codec, seed=0, total_rounds=args.total_rounds)
    ckpt = args.resume or (args.save if args.talk_only else "")
    if ckpt and os.path.exists(ckpt):
        sub.load_state_dict(torch.load(ckpt, weights_only=False))
        print(f"loaded {ckpt} at round {sub._round}")

    if args.talk_only:
        _talk(controller, codec, val)
        return

    print(f"two-loop run: {args.rounds} rounds x {args.steps} steps "
          f"(hops={args.hops}, halting={not args.no_halting}); {len(val)} held-out val")
    hist = sub.run(args.rounds, episodes_per_round=args.episodes_per_round,
                   steps=args.steps, val=val, verbose=True, save_path=args.save)
    if args.save:
        print(f"checkpoint at {args.save} (round {sub._round})")

    first, last = hist[0], hist[-1]
    print("\n--- both loops, across rounds ---")
    print(f"  val decode   : {first.get('val_decode', 0):.2f} -> {last.get('val_decode', 0):.2f}")
    print(f"  val op-trace : {first.get('val_optrace_match', 0):.2f} -> {last.get('val_optrace_match', 0):.2f}")
    print(f"  LTM facts    : {first['ltm_facts']} -> {last['ltm_facts']} (told + offline-inferred)")

    _talk(controller, codec, val)


if __name__ == "__main__":
    main()
