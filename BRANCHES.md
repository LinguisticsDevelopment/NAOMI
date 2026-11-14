# Branch inventory (2026-11-14 integration cleanup)

Written during the `integration-2026-11` cleanup. 98 remote branches
existed when this inventory was first drawn up (`git branch -r`); a 99th,
`encoder-vs-parser`, was pushed and merged into `integration-2026-11`
later in the same cleanup (see its row below) — counts reflect all 99.
`master` had been stale since 2025 and the de
facto mainline was `claude/m27-m28-cleanup` (last real merge 2026-10-15).
This file is the one-time audit of all of them. It does not get maintained
going forward — `master` (fast-forwarded to `integration-2026-11`; see `CLAUDE.md`) is now the single
place work lands, so this list should only shrink as the deletions below
happen, never grow back to this size.

## Methodology

For every branch: `git merge-base integration-2026-11 <branch>` plus
`git rev-list --count integration-2026-11..<branch>` (ahead-count). **Ahead
= 0 means the branch tip is a literal git ancestor of `integration-2026-11`
— fully contained, verdict MERGED automatically.** (The repo had to be
unshallowed first — `git fetch --unshallow` — or every old branch reads as
unrelated history and these numbers are garbage; this cloud session started
from a shallow clone.)

For every branch with ahead > 0, I diffed it against its merge-base,
excluding `RESEARCH_NOTES.md`/`CURRENT_STATE.md` (which diverge for
uninteresting reasons — see below) to see what *real* files it uniquely
touches, then checked whether that content (or its evolved descendant) is
already present on `integration-2026-11`. Verdicts:

- **MERGED** — fully contained (ahead=0), or (for the 5 branches named in
  the cleanup task) merged into `integration-2026-11` in this session.
- **ARTIFACT-ONLY** — the unique content is logs/checkpoints/data from an
  experiment whose result is already narrated in `RESEARCH_NOTES.md` and
  superseded by a later run; nothing a future agent needs to go fetch.
- **SUPERSEDED** — real code/design existed, but it already landed on
  mainline (verified: the current file is a more-evolved version of the
  same thing) or the approach was explicitly abandoned/falsified in the
  ledger. Confirmed per-branch by diffing the specific file against
  `integration-2026-11`'s current version, not just checking it exists.
- **KEEP** — real content not yet on `integration-2026-11` and not
  superseded. Only one branch landed here (see below).

**Why root `RESEARCH_NOTES.md` was excluded from the diff, and why ahead
counts for Aug/Sept-2026 branches look large:** this repo went through a
2026-10-16 doc consolidation (`docs-cleanup`) that collapsed a growing
root-level `RESEARCH_NOTES.md` copy into a one-line pointer and made
`consciousness_transformer/RESEARCH_NOTES.md` the only canonical ledger.
Every branch forked *before* that consolidation still carries the old,
large root `RESEARCH_NOTES.md` as a live file, which diverges from
`integration-2026-11`'s pointer version on nearly every line — that's
*not* unique unmerged work, just an artifact of when the branch forked. I
hit this same pattern merging the 5 branches by hand (see `CLAUDE.md`/the
commit log) and applied the same read to the other 93.

## The branches

| Branch | Last commit | Ahead of `integration-2026-11` | Verdict | Why |
|---|---|---|---|---|
| `master` | 2025-11-25 | 0 | **MERGED** | stale since 2025; fully contained (ahead=0) - the root of everything, never advanced |
| `claude/nsm-explication-corpus-vMlqe` | 2026-06-13 | 0 | **MERGED** | founding commit set; fully contained (ahead=0) |
| `claude/nsm-consciousness-transformer-vMlqe` | 2026-07-31 | 0 | **MERGED** | founding commit set; fully contained (ahead=0) |
| `m59-battery-logs` | 2026-08-30 | 1 | **ARTIFACT-ONLY** | ablation battery incl. no-LTM arm; result narrated |
| `m57c-battery-logs` | 2026-08-30 | 1 | **ARTIFACT-ONLY** | same battery, re-run; result narrated |
| `m57b3-battery-logs` | 2026-08-30 | 1 | **ARTIFACT-ONLY** | ablation battery logs (cheat/forcegold/forcewrong/nogold/normal); result narrated |
| `m57c2-battery-logs` | 2026-08-30 | 1 | **ARTIFACT-ONLY** | same battery, re-run; result narrated |
| `m60-battery-logs` | 2026-08-31 | 1 | **ARTIFACT-ONLY** | battery + m60_checkpoint.pt; result narrated, checkpoint superseded |
| `executor-lofo-logs` | 2026-08-31 | 1 | **ARTIFACT-ONLY** | LOFO gate #1 run logs; result narrated (inconclusive as instrumented) |
| `executor-lofo2-logs` | 2026-08-31 | 1 | **ARTIFACT-ONLY** | LOFO gate #2 run logs; result narrated (GO, 4/4 informative cells) |
| `parser-round2-work` | 2026-09-01 | 0 | **MERGED** | fully contained (ahead=0) |
| `m58f3-gc-fix` | 2026-09-02 | 0 | **MERGED** | fully contained (ahead=0) |
| `m58f2-corpus-cap` | 2026-09-02 | 1 | **SUPERSEDED** | corpus.py parse-cap handling superseded by the M58g hoisted-default fix already on mainline |
| `m58f-parse-cap-fix` | 2026-09-02 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-probe-logs` | 2026-09-02 | 1 | **ARTIFACT-ONLY** | M62 gold-volume probe csv/doc; superseded |
| `encoder-probe-logs2` | 2026-09-02 | 0 | **MERGED** | fully contained (ahead=0) |
| `prose-training-run1` | 2026-09-02 | 1 | **ARTIFACT-ONLY** | prose_v1.pt checkpoint (M61 prose-training payoff); result narrated, not current focus (encoder path superseded prose path) |
| `converter-memfix-work` | 2026-09-02 | 1 | **SUPERSEDED** | no unique content beyond stale ledger snapshot divergence |
| `m58g-parser-cap-hoist-logs` | 2026-09-02 | 0 | **MERGED** | fully contained (ahead=0); this IS the M58g hoisted-cap fix now on mainline |
| `encoder-gold-v1` | 2026-09-03 | 3 | **ARTIFACT-ONLY** | first teacher-gold corpus (1259 records); superseded by v2/v4b/v6, result narrated |
| `encoder-handgold-v1` | 2026-09-03 | 1 | **SUPERSEDED** | hand-gold draft + appraisal-as-emotion-table design; superseded by the appraisal grounding decision (literal-sense + learned connotation, no table) |
| `decoder-model` | 2026-09-04 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-model` | 2026-09-04 | 1 | **SUPERSEDED** | original 199K-param smoke encoder + smoke checkpoint; code evolved far beyond this on mainline already |
| `decoder-design` | 2026-09-04 | 0 | **MERGED** | fully contained (ahead=0); DECODER_DESIGN.md already in dev/archive |
| `spanish-gold-v2` | 2026-09-04 | 1 | **SUPERSEDED** | USVS Spanish lemma extension (wordnet.py/usvs.py) already landed and evolved further; script/stats doc already on mainline |
| `encoder-gold-v2` | 2026-09-04 | 1 | **ARTIFACT-ONLY** | candidate-lattice gold v2 (985 records); superseded by v4b/v6, result narrated |
| `encoder-schema-v2` | 2026-09-04 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-swap-test` | 2026-09-04 | 1 | **SUPERSEDED** | SWAP_TEST_RESULTS.md + eval_encoder_on.py already present on mainline (dev/reports/) |
| `encoder-model-spec` | 2026-09-04 | 0 | **MERGED** | fully contained (ahead=0); ENCODER_MODEL_SPEC.md already in dev/design |
| `encoder-train-full` | 2026-09-04 | 1 | **ARTIFACT-ONLY** | encoder_full.pt checkpoint superseded by v4b/v6; ENCODER_TRAIN_FULL_STATS.md already archived |
| `encoder-handgold-v2` | 2026-09-04 | 1 | **SUPERSEDED** | independent 2nd hand-gold draft; same supersession as v1 |
| `encoder-grammar-final` | 2026-09-04 | 0 | **MERGED** | fully contained (ahead=0) |
| `spanish-swap-feasibility` | 2026-09-04 | 1 | **SUPERSEDED** | feasibility probe; SPANISH_SWAP_FEASIBILITY.md already on mainline, probe superseded by the actual acceptance test (SWAP_TEST_RESULTS.md) |
| `encoder-diag` | 2026-09-05 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-stop-fix` | 2026-09-05 | 1 | **SUPERSEDED** | the real STOP-action fix; code since evolved far beyond this snapshot on mainline |
| `encoder-stop-fix2` | 2026-09-05 | 1 | **SUPERSEDED** | explicitly called out in the ledger as a redundant duplicate of encoder-stop-fix — "ignore/dedupe" |
| `colab-full-notebook` | 2026-09-05 | 1 | **SUPERSEDED** | unified Colab notebook superseded by direct CPU training; results narrated (COLAB RUN 1/2 entries) |
| `decoder-trained-code` | 2026-09-05 | 1 | **SUPERSEDED** | decoder_trained.py/train_decoder.py/test already present on mainline verbatim (landed via direct commits, not this branch) |
| `colab-encoder-notebook` | 2026-09-05 | 1 | **SUPERSEDED** | encoder-only Colab notebook superseded by colab-full-notebook, itself superseded by direct CPU training (train_encoder.py/arms) |
| `encoder-fix-v2` | 2026-09-06 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-aliasfix` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | aliasfix_output.txt diagnostic log; result narrated (ROOT CAUSE: beam_decode aliasing bug) |
| `encoder-mask-fix` | 2026-09-06 | 1 | **SUPERSEDED** | the strict_ground mask fix was explicitly found dead-on-arrival ("CORRECTION: FAILED + was NOT gold-safe") — do not resurrect |
| `encoder-diag-logs` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | diag_output.txt; result narrated (DIAGNOSIS: structure-0.000 is real) |
| `encoder-schedsamp` | 2026-09-06 | 2 | **SUPERSEDED** | scheduled-sampling approach; RUN-2 result explicitly falsified both "train more/bigger" levers |
| `encoder-audit-logs` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | audit_output.txt + audit_decode.py; result narrated (beam_decode aliasing bug found here) |
| `richer-gold-phase3` | 2026-09-06 | 2 | **SUPERSEDED** | subordination extraction; verdict was "plateaus, verbatim round-trip not reachable this way" — abandoned by design, not just superseded |
| `colab-trained-ckpts` | 2026-09-06 | 2 | **ARTIFACT-ONLY** | Colab run-1 encoder/decoder checkpoints; numbers narrated (edge_precision 0.10, decoder token-F1 0.36), superseded by v6 checkpoints |
| `decoder-memory-step1` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | decoder memory-step probe log/script; decoder ON HOLD, result narrated (23% coverage bottleneck) |
| `encoder-rescore-logs` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | rescore diagnostic log; superseded |
| `decoder-lossaudit-logs` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | decoder teacher-forced loss audit log/script; decoder is ON HOLD, result narrated |
| `encoder-lossaudit-logs` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | teacher-forced eval log; superseded |
| `colab-trained-ckpts-v2` | 2026-09-06 | 1 | **ARTIFACT-ONLY** | Colab run-2 checkpoints; numbers narrated (FLAT vs run-1), superseded by v6 |
| `gold-qa` | 2026-09-07 | 1 | **ARTIFACT-ONLY** | gold QA analysis report/script; superseded |
| `corpus-expand` | 2026-09-07 | 0 | **MERGED** | fully contained (ahead=0) |
| `hand-gold-draft` | 2026-09-07 | 0 | **MERGED** | fully contained (ahead=0) |
| `richer-gold-phase1` | 2026-09-07 | 3 | **SUPERSEDED** | modifier-extraction (DESCRIPTION/SPECIFICATION) concept; verified already landed and further evolved in current clause.py |
| `decoder-structmatch` | 2026-09-07 | 1 | **ARTIFACT-ONLY** | cycle-consistency structure-match evaluator result log; decoder ON HOLD, result narrated |
| `encoder-schedsamp-results` | 2026-09-07 | 3 | **SUPERSEDED** | same scheduled-sampling results, falsified |
| `k12-scout` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0); K12_CORPUS_SCOUT.md already in dev/reports |
| `hard-gold-gen` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `decisions-d1-d6` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0); D1/D2 decisions already in GOALS.md Decisions log |
| `encoder-arms3-b` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-arms3-a` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `llm-parse-judge` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0); note: `test_parse_judge.py` needs `pydantic`, which is not currently a declared dependency — pre-existing gap, flagged here, not fixed in this cleanup |
| `k12-fairytaleqa` | 2026-09-08 | 2 | **ARTIFACT-ONLY** | fairytaleqa_episodes_head1000.jsonl sample variant; superseded by the full fairytaleqa_episodes.jsonl already on mainline |
| `encoder-arms2-a` | 2026-09-08 | 3 | **ARTIFACT-ONLY** | arm checkpoints/logs from the v4b scaling-curve run; numbers narrated, superseded by v6 |
| `encoder-arms2-b` | 2026-09-08 | 3 | **ARTIFACT-ONLY** | arm checkpoints/logs (hard-gold mix); numbers narrated, superseded by v6 |
| `encoder-arms2-c` | 2026-09-08 | 4 | **ARTIFACT-ONLY** | arm checkpoints/logs (v2/margin); numbers narrated, superseded by v6 |
| `hard-gold-gen-v4` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0); this is the hard-gold family widening that `runs/hard_gold_train.jsonl` is built from |
| `hard-gold-gen-v3` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `hard-gold-gen-v2` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-train-arms` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-arms-seed0` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-arms-seed1` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `usvs-graded-scoring` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-gold-v4b-16k` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-gold-v3-small` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `k12-fairytaleqa-audit` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-train-arms-v2` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-gold-v4-small` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-gold-v4b-small` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-gold-v4b-variants` | 2026-09-08 | 0 | **MERGED** | fully contained (ahead=0) |
| `usvs-aux-head` | 2026-09-09 | 0 | **MERGED** | fully contained (ahead=0); USVS aux-head honest-negative result already in ledger |
| `encoder-arms4-a` | 2026-09-09 | 0 | **MERGED** | fully contained (ahead=0); this is the branch whose checkpoint (`v4b_all_hard_0.pt`) was **lost** (never force-added) — see `RESEARCH_NOTES.md` and `GOALS.md`'s Read goal |
| `encoder-arms4-b` | 2026-09-09 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-complement-probe` | 2026-09-09 | 2 | **ARTIFACT-ONLY** | complement-probe report updates; superseded |
| `k12-bench-fix` | 2026-10-15 | 0 | **MERGED** | merged via docs-cleanup; fully contained (ahead=0) |
| `gold-coverage-audit` | 2026-10-15 | 0 | **MERGED** | merged via docs-cleanup; fully contained (ahead=0) |
| `director-2026-10-15` | 2026-10-15 | 0 | **MERGED** | fully contained (ahead=0) |
| `encoder-gold-v5-k12` | 2026-10-15 | 0 | **MERGED** | merged into `integration-2026-11` in this cleanup |
| `clause-extraction-fix` | 2026-10-15 | 0 | **MERGED** | merged via docs-cleanup; fully contained (ahead=0); the B1+C1+B2 fix gold v6 is built on |
| `claude/m27-m28-cleanup` | 2026-10-15 | 0 | **MERGED** | former de facto mainline; fully contained (ahead=0) |
| `docs-cleanup` | 2026-10-16 | 0 | **MERGED** | the base `integration-2026-11` was built from; fully contained (ahead=0) |
| `e2e-qa-spike` | 2026-10-16 | 0 | **MERGED** | merged into `integration-2026-11` in this cleanup |
| `encoder-gold-v6` | 2026-10-16 | 0 | **MERGED** | merged into `integration-2026-11` in this cleanup |
| `encoder-v6-seed0` | 2026-10-17 | 0 | **MERGED** | merged into `integration-2026-11` in this cleanup — THE reference checkpoint |
| `encoder-v6-seed1` | 2026-10-17 | 0 | **MERGED** | merged into `integration-2026-11` in this cleanup |
| `claude/review-recent-pushes-or4o2e` | 2026-11-14 | 2 | **KEEP** | stray branch from an unrelated review session today, forked from an old (pre-docs-cleanup) point; only unique content is `.claude/settings.json` (git-push permission rules) — not merged here since it's outside this task's scope and touches repo permissions; owner should decide whether to port the settings.json rule |
| `encoder-vs-parser` | 2026-11-14 | 0 | **MERGED** | pushed after this inventory's first pass; merge-base was `c411b59` (already an ancestor), so the merge brought in exactly its own 5 files — encoder-vs-parser head-to-head result now in `GOALS.md`'s Read goal |

Counts: **52 MERGED**, **28 ARTIFACT-ONLY**, **18 SUPERSEDED**, **1 KEEP**
(99 total).

## Recommended deletion

Every branch above except `claude/review-recent-pushes-or4o2e` (KEEP) is
either fully contained in `integration-2026-11` or has nothing left in it
that isn't already captured here or in the ledger. None of this list was
deleted by me — deleting branches is a judgment call for the owner, not an
autonomous cleanup action. If the owner agrees with the verdicts above, the
one-liner to delete all 98 of them:

```bash
git push origin --delete \
  master claude/nsm-explication-corpus-vMlqe claude/nsm-consciousness-transformer-vMlqe \
  m59-battery-logs m57c-battery-logs m57b3-battery-logs m57c2-battery-logs \
  m60-battery-logs executor-lofo-logs executor-lofo2-logs parser-round2-work \
  m58f3-gc-fix m58f2-corpus-cap m58f-parse-cap-fix encoder-probe-logs \
  encoder-probe-logs2 prose-training-run1 converter-memfix-work \
  m58g-parser-cap-hoist-logs encoder-gold-v1 encoder-handgold-v1 decoder-model \
  encoder-model decoder-design spanish-gold-v2 encoder-gold-v2 encoder-schema-v2 \
  encoder-swap-test encoder-model-spec encoder-train-full encoder-handgold-v2 \
  encoder-grammar-final spanish-swap-feasibility encoder-diag encoder-stop-fix \
  encoder-stop-fix2 colab-full-notebook decoder-trained-code colab-encoder-notebook \
  encoder-fix-v2 encoder-aliasfix encoder-mask-fix encoder-diag-logs \
  encoder-schedsamp encoder-audit-logs richer-gold-phase3 colab-trained-ckpts \
  decoder-memory-step1 encoder-rescore-logs decoder-lossaudit-logs \
  encoder-lossaudit-logs colab-trained-ckpts-v2 gold-qa corpus-expand \
  hand-gold-draft richer-gold-phase1 decoder-structmatch encoder-schedsamp-results \
  k12-scout hard-gold-gen decisions-d1-d6 encoder-arms3-b encoder-arms3-a \
  llm-parse-judge k12-fairytaleqa encoder-arms2-a encoder-arms2-b encoder-arms2-c \
  hard-gold-gen-v4 hard-gold-gen-v3 hard-gold-gen-v2 encoder-train-arms \
  encoder-arms-seed0 encoder-arms-seed1 usvs-graded-scoring encoder-gold-v4b-16k \
  encoder-gold-v3-small k12-fairytaleqa-audit encoder-train-arms-v2 \
  encoder-gold-v4-small encoder-gold-v4b-small encoder-gold-v4b-variants \
  usvs-aux-head encoder-arms4-a encoder-arms4-b encoder-complement-probe \
  k12-bench-fix gold-coverage-audit director-2026-10-15 encoder-gold-v5-k12 \
  clause-extraction-fix claude/m27-m28-cleanup docs-cleanup e2e-qa-spike \
  encoder-gold-v6 encoder-v6-seed0 encoder-v6-seed1 encoder-vs-parser
```

(98 branch names — every row above with a verdict of MERGED, ARTIFACT-ONLY,
or SUPERSEDED. `claude/review-recent-pushes-or4o2e` is deliberately left
off this list.)

Before running it: `encoder-arms4-a`'s own checkpoint
(`runs/arms/v4b_all_hard_0.pt`) is **not on any branch** — it was never
committed (see the MERGED note above and `GOALS.md`'s Read goal) — deleting
`encoder-arms4-a` loses nothing further, that checkpoint is already gone.
Nothing else on this list carries anything not already reachable from
`integration-2026-11` or summarized in `RESEARCH_NOTES.md`.
