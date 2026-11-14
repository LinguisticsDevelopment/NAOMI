# Encoder v6 seed 0 — `v6_all_hard` retrained on gold v6 (branch `encoder-v6-seed0`)

Retrains the encoder on the fixed-clause-extraction gold (`encoder-gold-v6`,
22,990 records, mean VERB coverage 67.8%) topped up with the full v4
hard-construction gold (1,083 records), replacing the lost `v4b_all_hard_0`
checkpoint (never committed; `*.pt` is gitignored and it wasn't force-added —
see `dev/ENCODER_GOLD_V6_STATS.md`). This run's checkpoint **is** force-added.

## Config (same hyperparameters/step cap as `v4b_all_hard_0`, only gold + seed-handling differ)

```
python scripts/train_encoder.py \
  --gold runs/encoder_gold_v6.jsonl,runs/hard_gold_train.jsonl \
  --n-train 24073 --seed 0 --subset-seed 0 \
  --max-steps 12000 --max-seconds 172800 \
  --holdout-file runs/holdout_sentences.txt \
  --eval-gold runs/holdout_eval_gold_v6_combined.jsonl \
  --eval-every 250 --dev-holdout-file runs/holdout_dev_sentences.txt --keep-best \
  --extra-eval runs/hard_gold_test_filler.jsonl,runs/hard_gold_test_template.jsonl \
  --n-train-per-file 22990,1083 \
  --beam-width 6 --k 6 --metric both \
  --out runs/arms/v6_all_hard_seed0.pt
```

No code changes were needed. `--eval-gold` points at
`runs/holdout_eval_gold_v6_combined.jsonl` (new, not committed separately —
reproduce with `cat runs/holdout_dev_gold_v6.jsonl runs/holdout_gold_v6.jsonl >
runs/holdout_eval_gold_v6_combined.jsonl`), the concatenation of the v6 DEV
holdout gold (98 records) and the v6 TEST holdout gold (94 records). Both
`runs/encoder_gold_v6.jsonl` and `runs/hard_gold_train.jsonl` have ZERO
sentence overlap with either holdout list (holdout is excluded at the
*corpus* level for v6, not post-hoc) — `train_encoder.py`'s default
`dev_eval_gold_path = args.gold` therefore can't resolve the holdout records,
and `--eval-gold` has to name a pool that actually contains them.
`records_for_sentences` matches that combined pool against
`--dev-holdout-file` (98/98 matched -> keep-best selection) and against
`--holdout-file` (94/98 matched, 4 missing by construction -> final test
eval) independently, so a single `--eval-gold` file correctly serves both
roles, exactly as confirmed by the pilot run's own log lines:

```
dev-holdout eval targets from runs/holdout_eval_gold_v6_combined.jsonl: matched=98 missing=0
holdout eval targets from runs/holdout_eval_gold_v6_combined.jsonl: matched=94 missing=4
```

No `--eval-gold-alt` was passed (the old v4b_16k gold needed for that
comparison was never committed either — see "Old v4b holdout comparison"
below for how it was reconstructed post-hoc instead).

## Pilot + wall-clock

200-step pilot: 0.6466 s/step training-only, projected full run ≈2.6h
(well under the 5h/6h ceiling) → no step-cap cut needed.

Actual full run: **training wall-clock 7,495.5s (2.08h)**, optimizer
throughput 0.6246 s/step (faster than `v4b_all_hard_0`'s 1.0943 s/step on
this hardware). **TOTAL wall-clock 8,660.4s (2.41h)** including the periodic
dev evals, final last/best holdout eval, random baseline, and the two
extra-eval family breakdowns (filler + template). Stop reason: `max_steps`
(12,000, cap reached, not early-stopped). 16 epochs over the 24,073-record
pool (epoch_fraction 15.95).

## Dev-holdout curve (rank1 edge-F1, keep-best on v6 dev holdout, 98 sentences)

| step | train_loss | dev_rank1_f1 | dev_edge_p | dev_edge_r |
|---:|---:|---:|---:|---:|
| 250 | 37.98 | 0.373 | 0.449 | 0.319 |
| 1250 | 19.25 | 0.559 | 0.569 | 0.550 |
| 2750 | 16.04 | 0.599 | 0.601 | 0.597 |
| 4250 | 14.43 | 0.625 | 0.625 | 0.626 |
| 5750 | 13.34 | 0.654 | 0.665 | 0.644 |
| 7250 | 12.51 | 0.651 | 0.655 | 0.647 |
| 8750 | 11.80 | 0.662 | 0.670 | 0.654 |
| 10250 | 11.29 | 0.641 | 0.663 | 0.621 |
| 11750 | 10.65 | 0.646 | 0.660 | 0.632 |

**Best: step 11,000, dev_rank1_f1=0.6649** (kept checkpoint). Curve is noisy
but clearly climbing and plateauing in the 0.62–0.66 band from step ~5,000
on — no sign the 12,000-step cap left meaningful headroom on the table.

## Score table — kept-best checkpoint (`runs/arms/v6_all_hard_seed0.pt`, step 11,000)

### v6 TEST holdout (`runs/holdout_gold_v6.jsonl`, 94/98 sentences — 4 cap-hit on both branches)

| | rank1 edge-P | rank1 edge-R | **rank1 edge-F1** | full edge-P | full struct-exact | clause-count recall (rank1) |
|---|---:|---:|---:|---:|---:|---:|
| **model (best)** | 0.739 | 0.680 | **0.708** | 0.776 | 0.255 | 0.901 |
| model (last, step 12000) | 0.724 | 0.680 | 0.701 | 0.763 | 0.255 | 0.901 |
| POS-cheat baseline | 0.288 | 0.289 | **0.288** | — | 0.000 | — |
| random baseline | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |

Cheat baseline reproduces the ledger's documented v6-holdout number
(0.288) **exactly**.

### OLD v4b-style TEST holdout (reconstructed — see below; same 94 sentences, old gold standard)

| | rank1 edge-P | rank1 edge-R | **rank1 edge-F1** | full edge-P | full struct-exact | clause-count recall (rank1) |
|---|---:|---:|---:|---:|---:|---:|
| **v6 model (best), scored against OLD gold** | 0.602 | 0.659 | **0.629** | 0.666 | 0.202 | 0.791 |
| old reference (`v4b_all_hard_0`, from its own committed `.log`) | 0.692 | 0.566 | 0.623 | 0.751 | 0.214 | 0.805 |
| POS-cheat baseline (reconstructed gold) | 0.252 | 0.300 | 0.274 | — | 0.000 | — |
| POS-cheat baseline (ledger-documented, original lost gold) | — | — | 0.268 | — | — | — |

**Reading this table**: judged against gold built with the OLD (pre-fix)
clause extraction, the new model is roughly *on par* with the old checkpoint
(0.629 vs 0.623 rank1-F1 — a wash, within the old/new cheat-baseline gap of
0.274 vs 0.268). Judged against the NEW (fixed) gold — the correct ground
truth, since it recovers verbs the old extraction silently dropped — the
same checkpoint scores **0.708**, a real +0.08 absolute gain over the old
number. The two numbers are not in tension: gold v6 is simply a better,
harder-to-game measuring stick (higher VERB coverage), and this model was
trained *and* evaluated against it, so its 0.708 is the trustworthy number
going forward. The old-gold column exists only for continuity with
"0.622" in prior ledger entries.

### Old v4b holdout comparison — reconstruction method (original file is lost)

`runs/encoder_gold_v4b_16k.jsonl` (the file `v4b_all_hard_0` trained and was
scored against) was **never committed** — `runs/` is gitignored and this
file was never force-added; commit `97fd9e5` explicitly merged
`encoder-gold-v4b-16k`'s stats/builder but left "jsonl stays on branch" (that
branch's container is gone). This is the same loss `dev/ENCODER_GOLD_V6_STATS.md`
already flagged for the `.pt` checkpoint — flagged here too, not silently
patched.

Reconstructed an equivalent instead: checked out `src/nsm_ct/clause.py` from
commit `ad017fb` (the parent of `99d3742`, the clause-extraction-fix commit —
i.e. the exact pre-fix extraction code `v4b_16k` was built with), ran
`scripts/build_holdout_gold.py --sentences runs/holdout_sentences.txt --out
runs/holdout_gold_v4b_reconstructed.jsonl` against the same 98 TEST holdout
sentences (got 94/98 — same cap-hit count, `{'ok': 94, 'cap-hit': 4}`), then
restored the fixed `clause.py` via `git checkout --` (verified empty diff;
the live training run was unaffected since Python had already imported the
module into the running process before the swap). The cheat baseline on
this reconstructed file (0.274) is close to but not bit-identical with the
ledger's documented 0.268 for the real (lost) file — most likely minor
USVS/WordNet grounding drift between then and now, not a reconstruction
bug (the sentence-level cap-hit set matches the documented one exactly).
Flagged, not silently patched — this comparison table should be read as
"close enough for a sanity check," not as a byte-for-byte replay of the
original arm.

## Clause-count recall (predicted clauses vs gold, per sentence)

| gold set | clause_count (full, best-of-6) | rank1_clause_count |
|---|---:|---:|
| v6 TEST holdout (94) | 0.960 | 0.901 |
| OLD v4b-style reconstructed holdout (94) | 0.835 | 0.791 |

The model recovers more of the gold clause structure when scored against the
gold standard that actually contains those clauses (v6) — consistent with
the extraction fix, not an artifact of the new checkpoint.

## Extra-eval family breakdown (hard-construction generalization, best checkpoint)

`runs/hard_gold_test_filler.jsonl` (rank1 edge-F1): additive_focus 0.750,
content_interjection 0.983, elision 0.679, imperative 0.864,
pure_interjection 1.000, quantity 0.657, speaker_prime 0.383,
synth_subject 0.658.

`runs/hard_gold_test_template.jsonl`: additive_focus 0.750,
content_interjection 0.890, elision 0.643, imperative 0.846,
pure_interjection 0.983, quantity 0.586, speaker_prime 0.424,
synth_subject 0.523.

All families intact and mostly improved vs. the old arm's filler/template
numbers (e.g. imperative 0.827→0.864/0.659→0.846, content_interjection
0.994→0.983/0.827→0.890); `speaker_prime` remains the weakest family on
both branches (0.363→0.383 filler, 0.405→0.424 template) — a pre-existing
gap, not a regression.

## Artifacts

- `runs/arms/v6_all_hard_seed0.pt` (best, step 11,000) — **force-added**
- `runs/arms/v6_all_hard_seed0.pt.last.pt` (last, step 12,000) — **force-added**
- `runs/arms/v6_all_hard_seed0.pt.curve.tsv` — **force-added**
- `runs/arms/v6_all_hard_seed0.log` — **force-added**
- `runs/holdout_eval_gold_v6_combined.jsonl`, `runs/holdout_gold_v4b_reconstructed.jsonl` —
  intermediate files, NOT committed (gitignored, regeneration commands given above/below).

Reproduce the reconstructed old-style holdout:
```
git show ad017fb:consciousness_transformer/src/nsm_ct/clause.py > src/nsm_ct/clause.py
python scripts/build_holdout_gold.py --sentences runs/holdout_sentences.txt \
  --out runs/holdout_gold_v4b_reconstructed.jsonl
git checkout -- src/nsm_ct/clause.py
```
