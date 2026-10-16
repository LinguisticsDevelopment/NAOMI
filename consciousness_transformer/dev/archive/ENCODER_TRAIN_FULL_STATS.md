# Encoder full English train — stats (2026-09-04)

Full (non-smoke) train of `nsm_ct.encoder_model.EncoderModel` on all 985
records of `runs/encoder_gold_v2.jsonl` (spec `dev/ENCODER_MODEL_SPEC.md`).
Checkpoint: `runs/encoder_full.pt`.

## Config

Stratified 80/10/10 split (seed 0): train=788, dev=98, test=98 records
(2531 / 329 / 241 teacher-forced gold trees respectively).

`d_model=128` (full Stage-i default, not the smoke `d_model=64`), 15 epochs,
batch size 32, Adam lr 1e-3, beam width 6, k=6.

**Deviation from the script's own `--hash-buckets` full-mode default
(32768):** the script's `else` branch (non-`--smoke`) defaults to
`hash_buckets=32768`, which — because `tok_emb` is a plain
`nn.Embedding(hash_buckets, d_tok=32)`, not the two-hash compositional
scheme the spec's budget table assumes — costs `32768*32*4B ≈ 4.19MB`
just for the token table, pushing the whole model to **~5.02MB**, well past
the spec's own "whole model stays ≤~2MB" ceiling (S2.2). We overrode
`--hash-buckets 4096` (keeping the full `d_model=128`) to land at:

- **policy params: 342,834 → 1.371 MB fp32** — matches the spec's own
  "learned policy ≈0.33M → ~1.3MB" figure (S2.2 table) and is comfortably
  under the ≤2MB whole-model ceiling. (Strictly this is a hair over 1.0MB,
  same as the spec's own quoted number — "sub-MB" in the spec text is
  loose; we did not shrink `d_model` to force it under 1,000,000 bytes
  since the spec explicitly prioritizes `d_model=128` for the full run.)

Flag for whoever revisits `train_encoder.py`: its non-smoke default
`hash_buckets=32768` should probably become 4096–8192, or the token
embedding should switch to the spec's two-hash compositional scheme,
so the *un-overridden* full-mode invocation doesn't silently blow the
budget.

## Wall-clock

- Training: **1165.9s (19.4 min)**, 15/15 epochs completed, no early
  time-cutoff (`--max-seconds 1500` was not hit).
- Training + full train/dev/test/random-baseline eval (beam decode over
  all 3 splits): **1900.2s (31.7 min) total**.
- All 15 epochs ran; no reduction was needed.

## Loss curve (epoch-avg teacher-forced CE, action-type + typed-arg +
grounding-type + source, per spec S3.2)

| epoch | avg_loss |
|---|---|
| 0 | 40.527 |
| 1 | 23.279 |
| 2 | 18.991 |
| 3 | 16.512 |
| 4 | 14.533 |
| 5 | 12.905 |
| 6 | 11.546 |
| 7 | 10.420 |
| 8 | 9.475 |
| 9 | 8.669 |
| 10 | 7.965 |
| 11 | 7.377 |
| 12 | 6.889 |
| 13 | 6.410 |
| 14 | 6.037 |

Monotonic, smooth, still descending at epoch 14 (no plateau) — sense/slot
heads are well fit already (see below) but more epochs would likely keep
improving the action-type head further. No dev-based early stopping was
implemented (out of scope for a one-turn run; loop only has a wall-clock
cutoff) — 15 epochs finished inside budget so it was never invoked.

## Held-out candidate-set recall (spec S6)

| split | n_records | sense_recall (n sites) | slot_recall (n sites) | structure_recall (n gold trees) | all_gold_recalled_rate |
|---|---|---|---|---|---|
| train | 788 | 0.9332 (2681) | 0.9776 (3843) | 0.0000 (2531) | 0.0000 |
| dev | 98 | 0.9417 (360) | 0.9685 (603) | 0.0000 (329) | 0.0000 |
| test | 98 | 0.9313 (291) | 0.9783 (322) | 0.0000 (241) | 0.0000 |
| test, **random-legal baseline** | 98 | 0.0412 (291) | 0.0000 (322) | 0.0000 (241) | 0.0000 |

Sense recall and slot recall are both far above the random baseline
(93–94% vs 4% sense; 97–98% vs 0% slot) and consistent across train/dev/
test with no train→test gap — the model has genuinely learned *where* to
open a sense-grounding site and *where* to posit an unresolved
reference/elision slot. Structure recall is 0% on **every** split,
including train (the set it was directly optimized on), and the random
baseline is also 0% — so the model provides zero measured lift on
structure, same as chance.

## Structure-recall diagnosis: why 0%, and is it the metric?

Not primarily a metric-strictness artifact. Direct inspection of
decoded forests (train records the model was fit on, beam width 6,
`max_clauses` first at its 6-clause default then relaxed to 50) shows:

- **Every** emitted tree runs all the way out to whatever `max_clauses`
  cap `beam_decode` is given (6 by default; 50 when relaxed), vs. gold
  trees with 1–4 clauses (median ~3). This is not "close but capped
  slightly early" — it is unbounded: raising the cap from 6 to 50 just
  moves the wall from 6 to 50 clauses, it never converges to a natural
  stopping point.
- **Root cause:** the 7-action transition system (spec S2.1) has **no
  terminal/STOP action**. `legal_action_types()` returns `["OPEN_CLAUSE"]`
  unconditionally whenever no clause is open — including after the buffer
  is fully consumed (`i >= T`). There is no legal action that means "the
  sentence is done, emit no more clauses." A derivation only stops because
  `beam_decode`'s decode loop force-terminates a beam once
  `len(clauses) >= max_clauses` or `steps_taken >= max_steps` (S implementation,
  not spec-mandated) — an artificial cap, not a learned or even
  representable "done" state. So at inference the policy keeps legally
  re-opening clauses forever; teacher forcing never has to learn a "stop"
  signal either, because the *oracle* derivation for a gold tree simply
  ends after its last `CLOSE_CLAUSE` and the loss is never computed past
  that point (§3.1) — the model is never taught what comes after the last
  gold clause, and the transition system doesn't allow "nothing" as an
  answer.
- This over-generation (6–50 mostly-degenerate clauses per emitted tree,
  many with empty/`None`-token-index predicates) is why `clause-structure
  equality` (exact same clause *set*) never matches: gold trees are compact
  (1–4 real clauses) and every emitted tree is a large superset padded with
  spurious clauses.

**This is architectural** (missing STOP semantics in the action inventory /
mask), not something 15 vs. 30 epochs or a larger `d_model` would fix on
its own — per the task's instruction, we did **not** change the model
architecture or the action inventory to add one.

### A fairer diagnostic metric (added as a side script, not part of the
eval contract — `scripts/eval_encoder.py` / `encoder_model.evaluate` /
`score_record` are unmodified)

Computed on a 60-record train sample + full dev + full test, per gold
tree, beam width 6, k=6:

| split | strict tree-in-forest recall | best-tree node-recall (lenient) | best-tree clause-boundary F1 | clause-COUNT match rate |
|---|---|---|---|---|
| train (n=60 recs, 171 gold trees) | 0/171 = 0.0000 | 0.7351 | 0.1804 | 4/171 = 0.0234 |
| dev (329 gold trees) | 0/329 = 0.0000 | 0.6722 | 0.2182 | 1/329 = 0.0030 |
| test (241 gold trees) | 0/241 = 0.0000 | 0.6678 | 0.1587 | 0/241 = 0.0000 |

- **"Best-tree node-recall"** (for each gold tree, the best-matching
  emitted tree's fraction of gold `(role, token_index, grounding_type)`
  triples it contains) looks encouraging (0.66–0.74) but is **inflated by
  the same over-generation bug**: an emitted "tree" with 6–50 clauses
  covering most token indices under most role labels is likely to contain
  any given gold triple *by sheer coverage*, not because it correctly
  identified that triple's structural site. It is a recall-only number
  with no precision term, so it rewards over-generation — precision would
  be poor (most of an emitted tree's 50–90 nodes are not in any gold tree).
- **"Best-tree clause-boundary F1"** (predicate-token-index sets per
  clause, precision+recall, best-matching emitted tree) is the more honest
  fairer metric: **0.16–0.22**. It still credits partial/near-miss
  segmentation, and it is not fooled by pure over-generation the way plain
  recall is, since spurious extra clauses hurt precision. It confirms the
  strict 0% is real: even with generous partial credit for "did you at
  least open a clause boundary near where gold did," the model is only
  ~16–22% of the way there.

Bottom line: sense/slot grounding-site prediction is strong and
generalizes cleanly; clause/forest structure is not learned in any
meaningful sense at Stage-i — the decoder degenerates to running out the
clause cap every time because the action space has no way to say "done."
The 0% strict number is the correct headline; the fairer F1 (not the
inflated recall) is the number to trust as a diagnostic, and it says the
same thing the strict metric does, just less severely.

## Is the encoder "trained"?

- **Sense/slot grounding: yes**, functionally trained — 93–98% held-out
  recall vs. 0–4% random, flat across train/dev/test (no overfitting gap),
  loss still descending smoothly at epoch 15.
- **Structure/forest emission: no** — 0% on every split including train,
  because the transition system cannot represent "stop," not because of
  insufficient data or epochs.

**Next lever:** add a terminal action (e.g. an explicit `DONE`/`STOP`
action legal only when the buffer is exhausted and no clause is open,
or make `CLOSE_CLAUSE` sufficient to end the derivation once `i >= T`
and let the decode loop treat "no legal action but buffer-exhausted" as
completion instead of only capping on `max_clauses`/`max_steps`) so a
derivation can actually terminate on its own. That is an action-space /
transition-system change, which is out of scope for this run (task said
not to change the model architecture or eval contract) but is the
concrete, well-isolated fix the diagnosis points to — everything else
(features, loss, masking for the actions that do exist, sense/slot
supervision) is working as intended.
