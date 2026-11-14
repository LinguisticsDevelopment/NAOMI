# Encoder retrained on gold v6 (seed 1) — `v6_all_hard`

Arm `v6_all_hard`, SEED 1 replicate (sibling seed-0 job runs the identical
arm on branch `encoder-v6-seed1`'s pair, `encoder-v6-seed0`, in a separate
container — not read or waited on here).

## Config (matches `v4b_all_hard_0` exactly, gold swapped v4b_16k -> v6)

```
python scripts/train_encoder.py \
  --gold runs/encoder_gold_v6.jsonl,runs/hard_gold_train.jsonl \
  --n-train-per-file 22990,1083 --n-train 22990 \
  --seed 1 --subset-seed 1 \
  --max-steps 12000 --max-seconds 172800 \
  --holdout-file runs/holdout_sentences.txt \
  --eval-gold runs/eval_gold_v6_combined.jsonl \
  --eval-gold-alt runs/encoder_gold_v4b_16k.jsonl \
  --eval-every 250 --dev-holdout-file runs/holdout_dev_sentences.txt --keep-best \
  --extra-eval runs/hard_gold_test_filler.jsonl,runs/hard_gold_test_template.jsonl \
  --beam-width 6 --k 6 --metric both \
  --out runs/arms/v6_all_hard_seed1.pt
```

Both the model-init seed and the subset seed are `1` (`--seed 1 --subset-seed
1`), matching `v4b_all_hard_0`'s pattern of `seed == subset_seed`.

**`--eval-gold` wiring note (deviation from the literal branch instructions,
required for correctness):** gold v6's records are passage-level blocks, not
single sentences (`encoder_gold_v6.jsonl` itself has 0/98 exact-text matches
against `holdout_sentences.txt` — the held-out sentences were already
excluded from the v6 source corpus upstream, confirmed by substring-checking
all 194 held-out sentences against every v6 passage: only 1/194 appears as a
substring anywhere, i.e. no real train/test leakage). `--eval-gold` /
`--dev-holdout-file` matching in `train_encoder.py` is exact-text against a
single gold file, so `runs/holdout_gold_v6.jsonl` (94 test records) and
`runs/holdout_dev_gold_v6.jsonl` (98 dev records) — the pre-built
sentence-level eval targets for v6 — were concatenated into
`runs/eval_gold_v6_combined.jsonl` and passed as the single `--eval-gold`
file, which is what the script's dev/test matching actually needs (it
always matches `--eval-gold`, never the training `--gold`, against both
`--holdout-file` and `--dev-holdout-file`). This reproduced `matched=98
missing=0` (dev) and `matched=94 missing=4` (test) — exactly the 94/98 the
task brief named.

## Pilot (200 steps) and timing

First 200-step pilot run was invalidated by an orphaned duplicate process
left running from an earlier (wrongly-exited) wait loop — killed, gold-v6
leakage checked, `--eval-gold` corrected, re-run clean. Clean pilot:
165.5s training wall-clock for 200 steps (includes one real 98-record dev
eval) -> projected ~2.7-3h training + ~0.4h fixed final-eval overhead,
safely under the 6h cutoff. **No step-cap cut was needed.**

Actual full run: **training wall-clock 8701.6s** (12,000 steps, throughput
1.3791 steps/s), **total wall-clock 10,307.6s (~2.86h)** including all
evals. Best checkpoint at **step 11,500**, kept-best dev rank1-F1 = 0.6411
(final-step dev rank1-F1 was 0.6210, i.e. keep-best did find and keep a
better point than the final step).

## Scores — kept-best checkpoint

### `runs/holdout_gold_v6.jsonl` (94 v6 test sentences)

| metric | v6 cheat (POS heuristic) | v6_all_hard seed1 [best] rank-1 |
|---|---|---|
| edge precision | 0.288 | 0.698 |
| edge recall | 0.288 | 0.657 |
| edge F1 | 0.288 | **0.677** |
| struct-exact (rank-1) | 0.000 | 0.202 |
| clause_count (rank-1) | — | 0.891 |

Full `[best]` holdout (best-of-6 + rank1 + forest width), v6 targets:
`rank1_edge_precision=0.6983 rank1_edge_recall=0.6571 rank1_edge_f1=0.6771
rank1_structure_recall=0.2021 rank1_clause_count=0.8913 rank1_graded_f=0.7585`.
(best-of-6, non-rank1: `edge_precision=0.7368 edge_recall=0.7251
clause_count=0.9289`.)

### Old v4b test holdout — `encoder_gold_v4b_16k.jsonl` restricted to the 98 sentences (`--eval-gold-alt`)

| metric | v4b cheat (POS heuristic) | v6_all_hard seed1 [best] rank-1 | v4b_all_hard seed0 (prior best, never committed) |
|---|---|---|---|
| edge F1 (rank-1) | 0.268 | 0.583 | 0.622 |
| struct-exact (rank-1) | 0.000 | 0.143 | 0.153 |
| clause_count (rank-1) | — | 0.767 | 0.805 |

Full `[best]` holdout_alt: `rank1_edge_precision=0.5506 rank1_edge_recall=0.6204
rank1_edge_f1=0.5834 rank1_structure_recall=0.1429 rank1_clause_count=0.7670
rank1_graded_f=0.6620`.

**Reading the two tables together:** on its own (v6) test target the encoder
clears its cheat baseline by +0.389 edge-F1 (0.677 vs 0.288) — comfortably
above floor, curriculum looks valid by the CLAUDE.md cheat-baseline law.
Scored against the *old* v4b targets, this v6-trained model comes in at
0.583, below `v4b_all_hard_0`'s 0.622 on the same 98 sentences — expected:
it was never trained on v4b-style gold, and v6's rebuilt clause extraction
changes what "correct" edges look like, so the two test targets are not
interchangeable. The v6 number (0.677) is the one to track going forward;
the alt number is continuity-only, per `run_encoder_arms.sh`'s existing
convention for every `*_hard` arm.

### Extra-eval (hard-gold test splits, per family, rank-1 edge-F1) — kept-best

| family | filler (n) | template (n) |
|---|---|---|
| additive_focus | 0.750 (10) | 0.750 (24) |
| content_interjection | 0.950 (30) | 0.925 (30) |
| elision | 0.661 (14) | 0.588 (28) |
| imperative | 0.854 (50) | 0.708 (32) |
| pure_interjection | 1.000 (25) | 0.983 (30) |
| quantity | 0.657 (26) | 0.515 (30) |
| speaker_prime | 0.398 (34) | 0.315 (32) |
| synth_subject | 0.646 (28) | 0.537 (25) |

Same shape as every prior `*_hard` arm: `speaker_prime` and `quantity` are
the weak families (worse than `v4b_all_hard_0`'s template-split numbers,
e.g. speaker_prime 0.315 vs 0.405 there) while `pure_interjection` /
`content_interjection` are near-ceiling. Not investigated further here —
flagging for whoever looks at family-level regressions next, since v6's
retrained weights are uniformly a bit weaker on hard-gold template forms
than `v4b_all_hard_0`, even though they're stronger on v6's own holdout.

### Random baseline (holdout, v6 targets)

`edge_precision=0.0142 edge_recall=0.0084` (n=94) — floor confirmed, as
expected.

## Surprises / things that would have silently gone wrong

1. **`--eval-gold runs/encoder_gold_v6.jsonl` matches 0/98** — the obvious
   first command (mirroring `v4b_all_hard`'s pattern of pointing `--eval-gold`
   at the training-adjacent gold file) silently produces `nan` dev/test
   metrics the whole run, because gold v6 is passage-granular. Caught before
   the full run by inspecting the pilot's printed `matched=` line — this is
   exactly the kind of silent failure CLAUDE.md's "perfect-looking results
   get held-out-atom tests" rule is about, except here it would have been a
   silent *nan*, not a perfect-looking fake pass, so still worth a flag for
   whoever automates this arm for other gold files later (an assertion on
   `matched > 0` in `train_encoder.py` would catch this class of bug for
   free).
2. First pilot run was contaminated by my own process-management bug (an
   earlier wait loop exited on a 540s timeout without the job having
   actually finished; a second pilot was started without killing the first,
   so both competed for the same 4 CPUs). Caught via `ps aux` before trusting
   the timing; no training data or checkpoints affected, just wasted ~10 min
   of calibration wall-clock.
3. Checkpoint file is **6.4 MB**, not the ~1.4 MB the task brief estimated
   (that number is the policy's own parameter count at fp32; the saved
   `.pt` also carries optimizer state + eval metadata, which roughly
   quadruples it). Force-added anyway as specified.

## Files pushed

- `runs/arms/v6_all_hard_seed1.pt` (kept-best, step 11500) — force-added
- `runs/arms/v6_all_hard_seed1.pt.last.pt` (final-step) — force-added
- `runs/arms/v6_all_hard_seed1.log`, `.pt.curve.tsv` — force-added
- `runs/eval_gold_v6_combined.jsonl` — force-added (the dev+test eval-gold
  fix above; needed to reproduce this run's `--eval-gold` matching)
