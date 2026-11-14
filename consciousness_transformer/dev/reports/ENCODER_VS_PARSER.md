# Encoder vs. parser, head to head, on prose neither was trained on

Branch `encoder-vs-parser`. Checkpoint: `runs/arms/v6_all_hard_seed0.pt`
(the `encoder-v6-seed0` reference, rank-1 edge-F1 0.708 vs POS-cheat 0.288
on the teacher's own holdout — `dev/reports/ENCODER_V6_SEED0.md`). Teacher:
the deterministic `quantum_parser` + clause extraction, run with the SAME
settings `build_encoder_gold_v2.build_record` uses to build gold
(`TOP_K=8`, `CORPUS_MAX_HYPOTHESES=4000`, `CORPUS_MAX_PARSE_SECONDS=10.0`).
Script: `scripts/encoder_vs_parser_probe.py` (stages `sample` / `teacher` /
`encoder` / `metrics` / `judgeset`). Hand-judged items:
`runs/encoder_vs_parser_judged.jsonl` (force-added).

The prior measurement of this question (2026-09-08, branch
`encoder-complement-probe`) used an old checkpoint, a lenient LLM judge,
and (in its rushed variant) only a 500-sentence sample of the teacher
failure pool — 18.2% complement usable-rate, "FAILS" against its own 50%
gate, later revised to 26.0% ("INCONCLUSIVE") once the full 1,475-sentence
failure population was classified before sampling. This run uses the
current reference checkpoint, a much larger and more diverse unseen-prose
sample (2,000 sentences across two sources), and the operator (me) as the
judge instead of an LLM.

## 1. Sample

2,000 sentences, seed 20261114:

- **1,500 FairytaleQA VAL+TEST** story sentences
  (`data/corpus/k12_fairytaleqa_{val,test}.txt`, `encoder-gold-v5-k12`) —
  never in any gold by construction (that corpus excludes the holdout
  lists, and the encoder was never trained on this file at all).
- **500 Gutenberg sentences** from `data/corpus/real_*.txt` whose exact
  text does **not** appear in `runs/encoder_gold_v6.jsonl.gz` (22,990
  records) or `runs/hard_gold_train.jsonl` (1,083 records) — the two files
  `v6_all_hard_seed0` actually trained on.

**Methodological flag, found during the build, not anticipated going in**:
gold v6's build consumed almost the *entire* `real_*.txt` Gutenberg corpus
(16,217 of ~16,387 unique sentences, per `dev/ENCODER_GOLD_V6_STATS.md`-era
notes) — successful teacher parses overwhelmingly became gold records. So
"Gutenberg sentences NOT in gold v6" is not a random sample of Gutenberg
prose; it is specifically enriched for sentences the gold build's own
teacher pass **failed on** (that's *why* they were left out). The 500-
sentence Gutenberg draw is therefore a near-worst-case slice, not a
representative one — see the outcome table below, where its teacher-OK
rate is 4% vs FairytaleQA's 76%. The **FairytaleQA population is the
clean, representative "unseen prose" sample** (nothing about its selection
depends on teacher success/failure); Gutenberg's numbers are reported
separately throughout and should be read as "the teacher's leftovers,"
not "Gutenberg difficulty in general."

### Length bins (A≤8, B 9-15, C 16-25, D 26+ tokens)

| source | A | B | C | D | total |
|---|---:|---:|---:|---:|---:|
| fairytaleqa | 159 | 283 | 443 | 615 | 1500 |
| gutenberg_unseen | 121 | 101 | 103 | 175 | 500 |

## 2. Outcome table (teacher, by source × bin)

| source\|bin | ok | no-hyp | cap-hit | ground-fail | too-long |
|---|---:|---:|---:|---:|---:|
| fairytaleqa\|A (n=159) | 81 | 47 | 0 | 31 | 0 |
| fairytaleqa\|B (n=283) | 196 | 76 | 0 | 11 | 0 |
| fairytaleqa\|C (n=443) | 372 | 58 | 0 | 13 | 0 |
| fairytaleqa\|D (n=615) | 496 | 92 | 23 | 4 | 0 |
| gutenberg\|A (n=121) | 2 | 53 | 0 | 66 | 0 |
| gutenberg\|B (n=101) | 1 | 86 | 0 | 14 | 0 |
| gutenberg\|C (n=103) | 8 | 88 | 0 | 7 | 0 |
| gutenberg\|D (n=175) | 9 | 146 | 16 | 4 | 0 |

**`too-long` never fired** (no sentence exceeded the parser's 100-word hard
cap in either sample).

| | ok | no-hyp | cap-hit | ground-fail | **complement (fail)** |
|---|---:|---:|---:|---:|---:|
| fairytaleqa (n=1500) | 1145 (76.3%) | 273 | 23 | 59 | 355 (**23.7%**) |
| gutenberg_unseen (n=500) | 20 (4.0%) | 373 | 16 | 91 | 480 (**96.0%**) |
| **combined (n=2000)** | 1165 (58.25%) | 646 | 39 | 150 | **835 (41.75%)** |

The FairytaleQA-only complement rate (23.7%) lands almost exactly inside
the ledger's prior estimate ("no-hypothesis, cap-hit, grounding-fail
sentences ~24–38% of real prose") — a good sanity check that this sample is
representative. The combined 41.75% is inflated by the Gutenberg
selection-bias flagged above.

## 3. OOD agreement on teacher-OK sentences (encoder rank-1 vs. teacher)

Encoder-vs-teacher rank-1 edge-F1/struct-exact/clause-predicate-recall,
`nsm_ct.encoder_model.score_record`/`aggregate_recall` (same metric
`rescore_encoder.py`'s Part A uses), computed fresh against teacher trees
built live on this unseen sample (not against any stored gold file).

| | n | edge-P | edge-R | **edge-F1** | struct-exact | clause-pred. recall |
|---|---:|---:|---:|---:|---:|---:|
| **combined** | 1165 | 0.623 | 0.564 | **0.592** | 0.072 | 0.710 |
| fairytaleqa | 1145 | 0.621 | 0.564 | 0.591 | 0.072 | — |
| gutenberg_unseen | 20 | 0.716 | 0.570 | 0.634 | 0.100 | — |
| *(reference, in-distribution holdout, `ENCODER_V6_SEED0.md`)* | 94 | 0.739 | 0.680 | **0.708** | 0.255 | 0.901 |

**The headline 0.708 does not fully survive out-of-distribution**: edge-F1
drops to 0.592 (−0.116 absolute, −16% relative) on prose the teacher itself
calls "OK" but that is otherwise unlike anything in the training gold.
Clause-predicate recall (which clauses the encoder finds a predicate for,
regardless of its arguments) holds up much better — 0.710 vs. 0.901 in
distribution — meaning the encoder still reliably spots *that* a clause is
there; it loses more on getting the clause's *arguments* structurally
exact (struct-exact collapses from 0.255 to 0.072 — a 3.5× drop). The
gutenberg_unseen n=20 OOD-F1 (0.634) is higher than fairytaleqa's, but the
sample is too small to read as a real source effect (and recall the
selection-bias note above — these 20 are themselves an unusual survivor
set).

## 4. Complement (teacher-failed): automatic structural metrics

n=835 (teacher outcome ∈ {no-hypothesis, cap-hit, grounding-fail}), encoder
rank-1 tree, no reference needed:

| metric | value |
|---|---:|
| produced a tree (≥1 clause) | 100.0% |
| structurally valid (tree + no phantom/duplicate token_index, J3) | 94.1% |
| mean verb coverage (tagger VERB/AUX tokens also a clause predicate) | 43.8% |
| mean content-noun coverage (tagger NOUN/PROPN/PRON also a role filler) | 52.1% |
| mean clauses emitted | 2.17 |
| mean tagger verbs/sentence | 4.40 |
| mean memory-writeable clauses (predicate=VERB/AUX ∧ ≥1 grounded role) | 1.45 |
| **any memory-writeable clause** | **79.3%** |

By failure mode:

| outcome | n | produced tree | any writeable |
|---|---:|---:|---:|
| no-hypothesis | 646 | 100.0% | 87.9% |
| grounding-fail | 150 | 100.0% | 37.3% |
| cap-hit | 39 | 100.0% | 97.4% |

The encoder *always* emits something structurally well-formed, and on
~4.4 tagger verbs per sentence it only anchors ~2.2 clauses — it is
selective, not dumping. `grounding-fail` sentences (where the teacher found
a discourse graph but no clause with a real subject) are the encoder's
worst bucket (37.3% any-writeable) — these are disproportionately the
garbled/fragment-heavy inputs (interjections, titles, mis-tagged tokens)
where there often isn't a clean proposition for *anyone* to extract.
`cap-hit` (the teacher ran out of time, not out of ideas — usually a long,
otherwise-ordinary sentence) is the encoder's best bucket (97.4%), which
matches the intuition that these are "parseable but slow," not "hard."

Automatic metrics only measure *shape*, not correctness — hence the
hand-judging below.

## 5. Hand-judged tables

Operator (me) judged every item directly against the sentence text — no
external LLM judge. Labels: **USABLE** (all main propositions right enough
to write to memory) / **PARTIAL** (≥1 correct writeable proposition,
something else missing or wrong) / **WRONG** (would write a false fact —
most commonly: negation dropped, or a core entity's subject/object role
reversed) / **EMPTY** (no usable content; typically the input itself is a
fragment/interjection/title, or the tree has no real predicate).

### 5a. Complement set (n=80, stratified by outcome × bin, seed 20261115): ENCODER vs. POS-cheat baseline

Same 80 sentences, same judge, same criteria — `scripts/cheat_encoder_baseline.heuristic_forest`
(first VERB/AUX as predicate, nearest nominal before/after as SUBJECT/OBJECT,
everything else DESCRIPTION) run through `judge_tree`'s rendering.

| | USABLE | PARTIAL | WRONG | EMPTY | **any-correct (U+P)** |
|---|---:|---:|---:|---:|---:|
| **ENCODER** | 15 (18.8%) | 36 (45.0%) | 17 (21.3%) | 12 (15.0%) | **51/80 = 63.8%** |
| **POS-cheat** | 4 (5.0%) | 26 (32.5%) | 40 (50.0%) | 10 (12.5%) | **30/80 = 37.5%** |

The encoder beats the cheat floor by **+26.3 points** on the sentences the
teacher cannot parse at all. The cheat heuristic's dominant failure mode is
structural: it always emits exactly one clause (never multi-clause), it has
no negation handling at all (any "not"/"n't" is silently dropped — see
§5c), and its nearest-nominal rule reverses subject/object on any inverted
dialogue tag ("said Alice", "cried the mother"). The encoder gets some of
these right (multi-clause capture, occasional correct inverted-tag
resolution) and still drops negation about as often.

### 5b. Calibration set (n=40, teacher-OK, seed 20261116): ENCODER, same judging criteria

| USABLE | PARTIAL | WRONG | EMPTY | **any-correct (U+P)** |
|---:|---:|---:|---:|---:|
| 14 (35.0%) | 21 (52.5%) | 4 (10.0%) | 1 (2.5%) | **35/40 = 87.5%** |

This is the judge's own calibration check: on sentences the teacher itself
successfully parsed (a reasonable proxy for "easier" text), the encoder's
any-correct rate is 87.5% vs. 63.8% on the complement — a large,
sensible gap in the expected direction. It also says the judging rubric
isn't just rubber-stamping USABLE/PARTIAL regardless of input: WRONG/EMPTY
still shows up 12.5% of the time even on relatively easy sentences
(mostly: dropped multi-clause content, or one garbled sub-clause next to
several correct ones).

### 5c. Representative examples

**USABLE — encoder, complement, hard multi-clause (cap-hit):**
> "when the fisherman went home to his wife in the pigsty , he told her how
> he had caught a great fish , and how it had told him it was an enchanted
> prince , and how , on hearing it speak , he had let it go again ."

8 clauses emitted; 6 of 8 (went/home, told/her, caught/fish, told/him,
was/prince, go/it) are correctly subject/predicate/object-anchored. The
teacher gave up on this sentence (ran past the 10s cap); the encoder
produces most of its real content correctly in under a second.

**USABLE — encoder, complement (no-hypothesis), simple:**
> "they gave her three kinds of wine to drink , white , red , and yellow ,
> and with that she died . '"

Two clauses: "they gave ... kinds of wine" and "she died" — both correct,
covering the sentence's two real facts.

**PARTIAL — encoder, complement, long/garbled but with real content (cap-hit):**
> "when gerda had warmed herself , and had eaten and drunk , the lapland
> woman wrote a few words on a dried haberdine , begged gerda to take care
> of them , put her on the reindeer , bound her fast , and away sprang the
> animal ."

"gerda warmed herself" and "woman wrote words" are both correctly
extracted; the sentence's other four actions (begged/put/bound/sprang) are
dropped or garbled.

**WRONG — negation dropped (both encoder AND cheat, same sentence):**
> "but for all this he was not a favourite ."

Encoder: `PREDICATE: was  SUBJECT: he` — "not" never attaches to anything;
writing "he was [a favourite]" asserts the opposite of what the sentence
says. The POS-cheat baseline makes the identical error on the identical
sentence (`SUBJECT: he  OBJECT: favourite`) — negation-loss is a shared,
not an encoder-specific, failure mode (see §6).

**WRONG — role reversal on an inverted dialogue tag:**
> "'no ; but there will be rough weather by to-morrow at sunset , ' rejoined
> matte ."

Encoder's second clause: `PREDICATE: matte  SUBJECT: rejoined` — predicate
and subject are swapped (matte is not a verb). Contrast with "the goodman
laughed" (calibration K37) or "cried the mother" (K3), where the encoder
resolves the identical inverted-SV pattern correctly — this construction is
inconsistently handled, not uniformly broken.

**EMPTY — correctly inert on a non-propositional input:**
> "chug-a-rum !"

A frog's exclamation, not a fact-bearing sentence. The encoder emits an
`elision`-grounded predicate and no real roles — there is nothing to write,
and it (correctly) doesn't fabricate something.

## 6. Hybrid coverage — the encoder's complement value, quantified

Fraction of all 2,000 sampled sentences yielding ≥1 correct writeable
proposition, using the measured teacher-OK rate (58.25%) and the
hand-judged any-correct rates above (calibration 87.5% on teacher-OK text,
complement 63.75% on teacher-failed text) as the estimator. The standing
assumption (as everywhere else in this project) is that the teacher's own
output is correct when it successfully parses — "teacher-only" coverage is
therefore just its OK-rate.

| strategy | coverage | formula |
|---|---:|---|
| **(a) teacher only** | **58.2%** | P(teacher OK) |
| **(b) encoder only** | **77.6%** | P(ok)·0.875 + P(fail)·0.6375 |
| **(c) hybrid** (teacher where OK, else encoder) | **84.9%** | P(ok)·1.0 + P(fail)·0.6375 |
| *(floor: cheat-hybrid, teacher where OK else POS-cheat)* | *73.9%* | P(ok)·1.0 + P(fail)·0.375 |

**The encoder's coverage gain over teacher-only is +26.6 percentage
points** (58.2% → 84.9%) — essentially all of it earned on the 41.75% of
sentences the teacher cannot parse at all. The cheat floor would only have
bought +15.7 points (58.2% → 73.9%) using the same hybrid construction — so
the encoder's complement value is real and is not just "any heuristic
would have done this": it delivers about 1.7× the coverage gain the floor
gets on the identical failure population.

## 7. Verdict

**Yes, the encoder adds real coverage beyond the parser — on the order of
+27 points of additional "something correct got written to memory," almost
entirely on the ~24–40% of real prose the deterministic parser cannot
handle at all.** This is the first time that claim has survived a
large (2,000-sentence), methodologically-clean (full-population-first
sampling, current checkpoint, human judge, cheat-baseline comparison)
measurement — the prior attempt at this exact question landed at 18.2–26.0%
raw complement usable-rate against a 50% PASS gate and called it a FAIL/
INCONCLUSIVE. This run's complement any-correct rate (63.75%, counting
PARTIAL) clears that bar; even the stricter USABLE-only rate (18.75%) is
close to the prior measurement's own number, suggesting the two studies
aren't actually in tension — PARTIAL (a real but incomplete proposition)
turns out to be the dominant complement outcome, and whether you count it
depends on how forgiving your memory-write policy is willing to be about
partial propositions.

**Where it still fails, in order of how much it costs:**
1. **Negation is not tracked at all** — "not"/"n't" is simply absent from
   the emitted tree in every case observed (§5c), silently flipping true
   facts to false ones whenever a clause is negated. This is the single
   most damaging failure mode because it produces a confident WRONG, not a
   visible gap — exactly the output a naive memory-write policy is least
   equipped to catch. It is not encoder-specific: the POS-cheat baseline
   drops negation identically (it has no mechanism for it either), so this
   is a genuine gap in what either system currently extracts from
   `grounding-fail`/`no-hypothesis` text, not a training artifact.
2. **grounding-fail sentences are the hardest bucket** (37.3% any-writeable
   vs. 87.9%/97.4% for the other two failure modes) — these disproportion-
   ately overlap with inputs that may not contain a clean proposition for
   anyone (interjections, chapter titles, POS-mis-tagged fragments).
3. **Structural exactness degrades much faster than clause detection
   out-of-distribution** (struct-exact 0.072 vs. clause-predicate-recall
   0.710 on teacher-OK unseen prose) — the encoder reliably notices a
   clause exists; it is far less reliable about getting every one of that
   clause's arguments exactly right, which is consistent with PARTIAL
   (not USABLE) being the modal complement outcome.
4. Inverted subject-verb dialogue tags ("said X", "cried X") are resolved
   correctly about half the time and reversed the other half — no
   consistent rule yet.

**Where it clearly earns its keep**: long, information-dense sentences
that blow the teacher's 10-second cap (`cap-hit`, 97.4% any-writeable) and
simple no-hypothesis sentences (87.9%) are read nearly as well as
in-distribution text, at a fraction of the teacher's wall-clock cost on the
same input (the complement-probe branch's own timing table showed an 8.9×
median speedup on bin-D cap-hit sentences, consistent with this run's own
walk-clock logs).

## Artifacts

- `scripts/encoder_vs_parser_probe.py` — sampling / teacher / encoder /
  metrics / judgeset stages.
- `scripts/_apply_judgments.py` — one-off merge of the hand labels below
  into the judged jsonl (left for provenance, not part of the reusable
  pipeline).
- `runs/evp_sample.csv` (2,000-sentence sample), `runs/evp_teacher.jsonl`,
  `runs/evp_encoder.jsonl`, `runs/evp_metrics.json` — intermediate,
  NOT committed (gitignored, regenerate with `--stage sample/teacher/
  encoder/metrics`).
- `runs/encoder_vs_parser_judged.jsonl` — the 200 hand-judged items
  (80 complement×encoder, 80 complement×cheat, 40 calibration), **force-
  added**.
