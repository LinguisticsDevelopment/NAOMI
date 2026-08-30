"""M57 checkpoint evaluation tool (RESEARCH_NOTES M-ES1, "once checkpointing
lands"): loads a frozen checkpoint (scripts/train_instances.py --save /
scripts/train_writeback.py --save) and evaluates it on a FRESHLY GENERATED
episode mix -- the same generator knobs scripts/train_instances.py's
``--episodes``/``--rich-frac``/``--inverse-frac``/``--seed`` expose, so this
tool doesn't require reusing the exact episode set the checkpoint was
trained on the way ``scripts/train_instances.py --load`` does. This is the
tool the M58 zero-shot prose test and the Spanish Freeze Test's true
frozen-model comparison are meant to extend (RESEARCH_NOTES M-ES1's "once
checkpointing lands" note) -- see this module's own tail for what that
extension will need beyond what's built here.

Reuses ``train_instances.py``'s own ``build_instance_curriculum`` +
``run_arm(load=...)`` verbatim -- ONE report implementation, not a second
hand-rolled one -- rather than reimplementing curriculum generation or the
report block here. ``--dim``/``--hidden`` are accepted only for the header
print before the checkpoint is loaded; ``run_arm(load=...)`` overrides them
from the checkpoint's own config regardless (see its docstring), so a
mismatched value here can't silently build a wrong-shaped eval batch.

Usage:
    python scripts/eval_checkpoint.py --ckpt /tmp/m57_smoke.pt --episodes 60 --seed 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import train_instances  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path from --save.")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--inverse-frac", type=float, default=0.3,
                     help="Fraction of instance episodes that are inverse-query, same knob as "
                          "train_instances.py's own --inverse-frac.")
    ap.add_argument("--rich-frac", type=float, default=0.0,
                     help="Fraction of the mix that is RICH-EPISODE curriculum, same knob as "
                          "train_instances.py's own --rich-frac.")
    ap.add_argument("--rich-inverse-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0,
                     help="Curriculum-generation seed for this FRESH mix -- independent of whatever "
                          "seed the checkpoint was trained with (recorded in its own config).")
    ap.add_argument("--dim", type=int, default=48,
                     help="Cosmetic only -- overridden by the checkpoint's own dim once loaded.")
    ap.add_argument("--hidden", type=int, default=128,
                     help="Cosmetic only -- overridden by the checkpoint's own hidden once loaded.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--audit", type=int, default=0,
                     help="Forwarded to run_arm's --audit: after eval, print a provenance explain() "
                          "trail for the first N held-out episodes. 0 (default) = no-op.")
    args = ap.parse_args()

    episodes = train_instances.build_instance_curriculum(
        args.episodes, args.seed, inverse_frac=args.inverse_frac,
        rich_frac=args.rich_frac, rich_inverse_frac=args.rich_inverse_frac)
    n_wb = sum(1 for e in episodes if e.meta.get("kind") == "writeback")
    n_inst = sum(1 for e in episodes if e.meta.get("kind") == "instance")
    n_rich = sum(1 for e in episodes if e.meta.get("kind") == "rich")
    print(f"=== eval_checkpoint {args.ckpt}: {args.episodes} eps ({n_wb} writeback, {n_inst} instance, "
          f"{n_rich} rich, inverse_frac={args.inverse_frac}, rich_frac={args.rich_frac}), "
          f"batch_size={args.batch_size} (dim/hidden taken from the checkpoint) ===", flush=True)

    result = train_instances.run_arm(
        "frozen", None, episodes, args.dim, epochs=0, seed=args.seed, hidden=args.hidden,
        batch_size=args.batch_size, audit=args.audit, load=args.ckpt)
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# Seams for the M58 zero-shot prose test / the Spanish Freeze Test's true
# frozen-model comparison (both named in RESEARCH_NOTES M-ES1's "once
# checkpointing lands" note -- neither is built here, this is plumbing):
#
#   - This tool only evaluates on train_instances.py's own synthetic
#     curriculum generator (build_instance_curriculum). A zero-shot PROSE
#     test needs a second batch-building path from raw text (not
#     CurriculumGenerator episodes) into the same ClauseBatch shape the
#     loaded model expects -- build_clause_batch's parser/meaning_resolver/
#     codec machinery is reusable, but episodes would have to come from
#     somewhere other than curriculum2's generators.
#   - The Spanish Freeze Test needs an ES-language episode source (or
#     prose) fed through the SAME frozen English-trained model -- config's
#     ``meaning_source`` field is recorded per checkpoint precisely so a
#     future ``--meaning-source spanish`` flag here could select
#     nsm_ct.meaning_es.SpanishMeaningResolver instead of NSMMeaningResolver
#     without touching the checkpoint or its model at all (the interlingua
#     claim this whole test is about: one frozen mind, a swapped perception
#     front-end).
#   - Nothing here compares TWO checkpoints (e.g. an EN arm vs an ES arm)
#     side by side -- today's tool loads and reports on exactly one.
# --------------------------------------------------------------------------
