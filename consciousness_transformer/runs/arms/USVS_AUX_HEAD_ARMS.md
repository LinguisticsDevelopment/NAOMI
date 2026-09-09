# USVS auxiliary head -- arm pair (v4b_788, seed 0)

Branch `usvs-aux-head`. Follow-up to `dev/USVS_GRADED_SCORING.md` S5.4
("the encoder has NO node embedding and NEVER scores sense candidates ...
the nearest hook would be a new Linear(controller_hidden, d_axes) head").

Both arms: `--gold runs/encoder_gold_v4b.jsonl --n-train 788 --seed 0
--subset-seed 0 --max-steps 6000 --holdout-file runs/holdout_sentences.txt
--eval-gold runs/encoder_gold_v2.jsonl --eval-gold-alt
runs/encoder_gold_v4b.jsonl --eval-every 250 --keep-best --beam-width 6 --k 6
--metric both`, run concurrently with `OMP_NUM_THREADS=1`.

- arm1: default loss (`runs/arms/v4b_788_arm1_default_seed0.log`, `.pt.curve.tsv`)
- arm2: `--loss usvs-soft --loss-temperature 0.25 --aux-usvs 0.5`
  (`runs/arms/v4b_788_arm2_auxusvs0.5_seed0.log`, `.pt.curve.tsv`)

Checkpoints (~3.9 MB each, over the 2 MB commit threshold, so not
committed) are reproducible from these logs/commands; the `[best]` block
in each log carries the full `evaluate_full` metric dict.

## Node-emitting actions (the aux head's hook)

Per `linearize_tree`, every tree node -- PREDICATE or role alike -- is
created by exactly one of three actions, dispatched on `grounding.type`
(`encoder_model.NODE_EMIT_ACTIONS`):

- `GROUND` -- a `sense` or `entity` node (a resolved content word).
- `EMIT_SYNTH_SLOT` -- a `prime` node (the imperative's synthesized
  addressee YOU, or "I"/"me"/"myself" resolving to the SPEAKER prime).
- `EMIT_UNRESOLVED_SLOT` -- a `reference` or `elision` node (an
  unresolved slot with a `retrieval.source`).

`ATTACH` is declared in `ACTION_TYPES` but is never legal
(`legal_action_types`) and never emitted by the oracle -- excluded on
purpose.

## Results (BEST checkpoint per arm, `evaluate_full`'s rank-1/committed view)

| arm | best_step | edge-F1 v4b (alt) | graded-F v4b (alt) | edge-F1 v2 (holdout) | graded-F v2 (holdout) | head-cosine v4b | head-cosine v2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1: default loss | 1250 | 0.4906 | 0.5810 | 0.3914 | 0.4590 | n/a | n/a |
| 2: usvs-soft + aux-usvs 0.5 | 4000 | 0.4376 | 0.5404 | 0.3405 | 0.4138 | 0.7975 | 0.6379 |

(best-of-6 oracle view, for reference: arm1 v4b graded_f=0.6798 v2
graded_f=0.5674; arm2 v4b graded_f=0.6523 v2 graded_f=0.5365 -- same
direction: arm2 below arm1 on every column, best-of-k or rank-1.)

`dev_rank1_f1` (the cheap --eval-every proxy used for --keep-best
selection) peaked similarly for both arms (arm1 0.3356 @1250, arm2 0.3296
@4000) -- the arms are not far apart on the SELECTION signal, but the
full holdout eval (wider beam, both edge and graded metrics, on both
target files) shows arm2 consistently ~0.03-0.06 below arm1 on every one
of the four columns.

## Verdict

**The USVS-space loss does NOT move the graded metric up -- it moves it
down, modestly but consistently, on both edge-F1 and graded-F, on both v2
and v4b targets, at this single seed/single weight (W=0.5).** The
auxiliary head itself clearly DOES learn the USVS space it was trained
against (head-cosine 0.64-0.80 on gold-teacher-forced replay, far above
the smoke run's early value of ~0.30) -- so the hook works as designed --
but that learned signal does not translate into a better decoded tree at
this setting.

Caveat: arm2 bundles TWO changes at once, exactly as specified
(`--loss usvs-soft --aux-usvs 0.5`), so this result cannot cleanly
separate "the soft CE targets hurt" from "the aux cosine term hurts" from
"the interaction hurts" -- an aux-usvs-only arm (no `--loss usvs-soft`)
against the same default-loss baseline would isolate that, and is the
natural follow-up. Single seed, single weight: not swept.
