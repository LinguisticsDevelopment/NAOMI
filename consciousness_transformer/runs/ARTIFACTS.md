# runs/ — artifact index

Written during the 2026-11 integration cleanup. Lists every artifact force-added
to `integration-2026-11` that a future agent would otherwise have to go hunting
for on a side branch. For the branches these came from and their verdicts, see
`BRANCHES.md` at repo root. For the encoder's full result history, see
`RESEARCH_NOTES.md`'s dated entries (search for "gold v6", "seed 0"/"seed 1").

## THE reference encoder checkpoints

- **`runs/arms/v6_all_hard_seed0.pt`** (6.4MB) — **THE reference encoder.**
  Trained by `scripts/train_encoder.py` on `encoder_gold_v6.jsonl.gz` +
  `hard_gold_train.jsonl` (1:9 mix), seed 0, 12K-step cap, keep-best (step
  11,000). Rank-1 edge-F1 **0.708** (P 0.739 / R 0.680) on the 94-sentence
  v6 holdout vs POS-cheat baseline 0.288 and random 0.000; clause-count
  recall 0.901. Weak families: `speaker_prime`, `quantity`.
  Load: `torch.load("runs/arms/v6_all_hard_seed0.pt", map_location="cpu",
  weights_only=False)` → dict with `model_state`, `pos_vocab`, `role_vocab`,
  `d_axes`, `hash_buckets`, `d_model`, `config`, `loss_curve`, `metrics`,
  `random_baseline_test`. Build the model from `config` + vocabs, then
  `model.load_state_dict(ckpt["model_state"])`. Full writeup:
  `dev/reports/ENCODER_V6_SEED0.md`.
- **`runs/arms/v6_all_hard_seed0.pt.curve.tsv`** — per-step dev-loss curve
  for the run above (used to confirm step 11,000 was the keep-best point).
- **`runs/arms/v6_all_hard_seed0.log`** — full stdout of the training run.
- **`runs/arms/v6_all_hard_seed1.pt`** (6.4MB) — second-seed checkpoint,
  same recipe, seed 1, keep-best step 11,500. Rank-1 edge-F1 **0.677** on
  its own v6 holdout (cheat floor 0.288). Use for cross-seed variance, not
  as the reference. Same load procedure. Writeup:
  `dev/reports/ENCODER_V6_SEED1.md`.
- **`runs/arms/v6_all_hard_seed1.pt.curve.tsv`**, **`.log`** — curve/log for seed 1.

  `.pt.last.pt` duplicates (identical weights, last-step instead of
  keep-best) were dropped for both seeds — use the `.pt` file.

## Gold (teacher-built candidate-lattice training/eval data)

- **`runs/encoder_gold_v6.jsonl.gz`** (17MB) — the v6 teacher gold: 22,990
  records (76.0% yield of 30,232 attempted), built by
  `scripts/build_encoder_gold_v2.py` on the clause-extraction-fix'd
  `clause.py` (B1+C1+B2) over the full Gutenberg population (16,217 sent.)
  plus a seeded 125-whole-story K-12 FairytaleQA subset (13,999 sent.),
  holdout sentences excluded at the corpus level (`--exclude-file`). Mean
  VERB coverage 67.8% (vs v4b-16k's 53.5%). Load: `gzip.open(...)` then
  one JSON object per line (candidate-lattice shape — see
  `dev/design/ENCODER_IO_CONTRACT_V2.md`). Full stats:
  `dev/reports/ENCODER_GOLD_V6_STATS.md`.
- **`runs/holdout_gold_v6.jsonl`** (94 records) — gold built from
  `holdout_sentences.txt` (the frozen v2/v4b test-split holdout, 98
  sentences, 4 cap-hit) on the fixed `clause.py`. This is the **test**
  split the 0.708/0.677 numbers above are scored against.
- **`runs/holdout_dev_gold_v6.jsonl`** (98 records) — same, but for
  `holdout_dev_sentences.txt` (the dev split). Used for keep-best selection
  during v6 training, not for the reported test numbers.
- **`runs/eval_gold_v6_combined.jsonl`** — `holdout_gold_v6` +
  `holdout_dev_gold_v6` concatenated; `train_encoder.py --eval-gold` needs
  one file, and v6 gold is passage-granular so pointing `--eval-gold` at
  the training file (the old `v4b_all_hard` pattern) silently matches 0/98.
  Use this file as `--eval-gold` when re-running/extending v6 training.
- **`runs/hard_gold_train.jsonl`** (1,083 records) — the hand/template
  hard-gold mix (8 families: pure/content interjection, imperative,
  additive, elision, quantity, synth_subject, speaker_prime) built by
  `scripts/gen_hard_gold.py`, widened to v4 (all families). Mixed into
  every encoder training run at ~1:9 (hard:main) since `encoder-arms4-a`.
- **`runs/holdout_sentences.txt`** / **`runs/holdout_dev_sentences.txt`**
  (98 lines each) — the frozen plain-text sentence lists that
  `holdout_gold_v6.jsonl`/`holdout_dev_gold_v6.jsonl` (and all prior
  holdout golds) were built from. Keep these frozen across encoder
  versions so results stay comparable; only the *gold* gets rebuilt when
  `clause.py` changes.
- **`runs/encoder_gold_v5_k12.jsonl.gz`** (4.6MB, gzipped during this
  cleanup — was committed uncompressed at 37MB) — the superseded v5 K-12
  gold (6,160 records, 73.2% yield), built before the clause-extraction
  fix. Superseded by v6 above; kept for the yield-rate comparison in
  `dev/reports/ENCODER_GOLD_V5_K12_STATS.md`, not for training.

## K-12 / FairytaleQA corpus + episodes

- **`data/corpus/k12_fairytaleqa_{train,val,test}.txt`** — the sentence
  corpus (30,690 sentences, 4,082 sections) extracted from the FairytaleQA
  story files, FairytaleQA's own train/val/test story split, built by
  `scripts/build_k12_fairytaleqa_corpus.py`.
- **`runs/k12_fairytaleqa_manifest.jsonl`** (30,690 lines) — sidecar
  manifest giving each sentence's story id / section / split, for
  traceability back to the source story. Built by the same script.
- **`data/k12_samples/fairytaleqa/`**, **`fairytaleqa_full_sample/`** — the
  committed FairytaleQA story+question CSVs (small, curated subsets; see
  each dir's `SOURCE.md`/`MANIFEST.json`/`story_meta.csv`) that
  `scripts/convert_fairytaleqa.py` and the test suite
  (`tests/test_fairytaleqa.py`, offline, no network) run against.
- **`runs/fairytaleqa_episodes.jsonl`** / **`fairytaleqa_episodes_samepassage.jsonl`**
  (3,000 lines each) — entity-MC Episodes converted from FairytaleQA by
  `scripts/convert_fairytaleqa.py`, cross-story vs same-passage distractor
  policy (same-passage is the fixed default — see
  `dev/reports/K12_BENCHMARK_FIX.md`; cross-story is kept for the cheat
  comparison in that report).
  Load with `nsm_ct`'s `Episode` dataclass (`src/nsm_ct/episode.py`, slimmed 2026-11-14; original loader code in `archive/src/nsm_ct/episode.py`) — one
  JSON Episode per line.
- **`runs/mctest_mc160_episodes.jsonl`** (400 lines) — MCTest mc160
  train+dev converted the same way, native in-story distractors.

## E2E QA spike (archived, superseded — see GOALS.md "Answer from memory" goal)

- **`runs/e2e_qa_spike_ftqa.pt`** / **`e2e_qa_spike_mctest.pt`** (~700KB
  each) — small QA-head checkpoints from `archive/scripts/e2e_qa_spike.py`'s (archived 2026-11-14)
  passage-capped (n=25 / n≈12) run over teacher-built (not encoder-built)
  episodes. Model and memory-severed control were identical (0.400/0.400)
  — data-starved, not evidence either way. Not reference checkpoints; kept
  only so the spike is reproducible without re-running the parse build.
  Superseded 2026-11-14: the spike's QA adapter and memory-severed control
  were flawed (see GOALS.md), so these files are not evidence either way.
- **`runs/e2e_qa_spike_ftqa.json`** / **`e2e_qa_spike_mctest.json`** — the
  spike's result/metrics dump (accuracy vs random/cheat/memory-severed
  controls, parse coverage, answerable-from-memory ceiling). Full report:
  `dev/reports/E2E_QA_SPIKE.md`.

## Encoder vs. parser head-to-head (the complement-coverage proof)

- **`runs/encoder_vs_parser_judged.jsonl`** (375KB) — per-sentence output
  of `scripts/encoder_vs_parser_probe.py`'s 2,000-sentence unseen-prose run
  (1,500 FairytaleQA val/test + 500 Gutenberg not in gold v6/hard-gold) plus
  `archive/scripts/_apply_judgments.py`'s (archived) hand-judgments (no LLM) merged in: 80
  complement + 80 matched + 40 calibration sentences judged
  USABLE/PARTIAL/WRONG. Encoder any-correct on the complement 63.8% vs.
  POS-cheat 37.5%; hybrid coverage 84.9% (+26.6pp over teacher-only). Full
  writeup: `dev/reports/ENCODER_VS_PARSER.md`.

## NAOMI minimal loop — first end-to-end benchmark (2026-11-14)

Produced by `scripts/naomi_run.py bench` over the episode files above
(MCTest mc160: 400 items; FairytaleQA same-passage: 222 items, max 3 per
story). No training; reader is parser + encoder v6 fallback. Ledger entry:
"Answer from memory: minimal read->remember->recall loop..." in
`RESEARCH_NOTES.md`.

- **`runs/naomi_bench_mctest.jsonl`**, **`runs/naomi_bench_ftqa.jsonl`** —
  one JSON record per item, including the wrong-passage runs. Score with
  `python scripts/naomi_run.py report --in runs/naomi_bench_mctest.jsonl`.
- **`runs/naomi_failures_mctest.jsonl`**, **`runs/naomi_failures_ftqa.jsonl`**
  — the failed items, kept for the failure-cause analysis.
- **`runs/naomi_bench_{mctest,ftqa}_v2.jsonl`**, **`naomi_failures_*_v2.jsonl`**,
  **`naomi_bench_*_v2_shuffled.jsonl`** — the second pass (question-side
  fixes) and its question-shuffled control (`bench --shuffle-questions`).
- **`runs/naomi_read_cache.jsonl.gz`** (3.5MB; `gunzip -k` it to
  `runs/naomi_read_cache.jsonl` before running, the plain file is 46MB and
  not tracked) — on-disk read cache for the
  reader (sentence -> read result); `bench` uses it by default
  (`--cache runs/naomi_read_cache.jsonl`), which makes re-runs fast. Delete
  it to force a fresh read.
- Reproduce: `python scripts/naomi_run.py bench --dataset mctest --wrong-passage`
  (resumable, stops starting new items after `--time-limit` s, default 110;
  re-run to continue), then `report`.

## Not included here

`runs/arms/*.pt.last.pt` (last-step duplicates of the two v6 checkpoints
above — dropped), and every other `runs/arms/*.log`/`*.pt.curve.tsv` from
earlier (v2/v3/v4b/arms1-4) scaling and ablation runs, which were already
on `integration-2026-11` via earlier merges and are historical record, not
artifacts a new agent needs to load. See `RESEARCH_NOTES.md` for what each
of those runs found.

## Gold v7 (new tree format)

The grammar parser's trees (`naomi.types.Tree`: every non-punctuation token one node, nested clauses, function words as
grounded nodes, unattached tokens recorded as UNATTACHED) over the gold-v6 population, for encoder/decoder design and
training. Built by `scripts/build_gold_v7.py` (recipe and statistics: `dev/reports/GOLD_V7_STATS.md`).

- **`runs/gold_v7.jsonl.gz`** (18.3MB, 29,421 records), **`runs/gold_v7_failures.jsonl.gz`** (616 failures with kind, no tree),
  **`runs/holdout_gold_v7.jsonl`** (94) + `.failures` (4), **`runs/holdout_dev_gold_v7.jsonl`** (98) + `.failures` (empty),
  **`runs/build_gold_v7.log`**.
- Load: `for l in gzip.open(p,"rt"): r = json.loads(l); t = naomi.tree.tree_from_dict(r["tree"])`.
  Record: id, text, source, tokens (token, pos incl. punctuation), tree, coverage (score_coverage), diagnostics (counts), n_tokens, n_attached, parse_seconds.
- In them: candidate sense ids per node (`sense` is None everywhere), parser edge types as relations, UNATTACHED pieces (only ~30% of tokens
  hang from the main root). Not in them: committed senses, encoder output, any answer/QA signal; holdout sentences are excluded from the main file.
- **v6 gold (`encoder_gold_v6.jsonl.gz`) remains the gold the v6 encoder checkpoint was trained on**; v7 does not replace it.
