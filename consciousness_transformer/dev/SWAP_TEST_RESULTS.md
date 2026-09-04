# Grammar-Swap Acceptance Test — EN-trained encoder on never-seen Spanish

**Question.** Does grounding transfer cross-lingually by data-swap alone? Take
the candidate-lattice encoder trained ONLY on English gold data, freeze the
weights, and evaluate it on Spanish records (Spanish senses + Spanish grammar
swapped into the memory/candidate input, same USVS, same architecture, same
candidate-set-recall metric). No retraining, no changes to
`encoder_model.py` or the metric.

## Setup

- Branch: `claude/m27-m28-cleanup` (USVS build, multilingual `usvs.py` /
  `wordnet.py`, `build_encoder_gold_es.py`).
- USVS built fresh via `python scripts/build_usvs.py`.
  - Fingerprint: **`e0daef638b640dd5`** (matches expected — English build is
    intact under the multilingual code path).
  - `senses_of('perro')[:3]` → `['dog.n.01', 'rotter.n.01']` — Spanish
    lemma resolves to the correct English WordNet synset (`dog.n.01`
    first). `senses_of('dog')[:3]` → `['dog.n.01', 'frump.n.01', 'dog.n.03']`.
- Checkpoint: `runs/encoder_full.pt`, pulled from `origin/encoder-train-full`
  (342,834 policy params), trained on English only.
- Spanish gold: `runs/spanish_gold_v2.jsonl`, pulled from
  `origin/spanish-gold-v2`, 208 records, never used in training.
- `scripts/eval_encoder.py` recomputes the English train/dev/test split
  internally (via `train_encoder.stratified_split`) and can't point at an
  arbitrary gold file, so a minimal `scripts/eval_encoder_on.py` was added
  that takes `--checkpoint`/`--gold` and calls `encoder_model.evaluate` /
  `score_record` verbatim — no metric logic was reimplemented or changed.

## Results

### Spanish (never-seen, EN-trained weights, n=208 records)

| Metric | Model | Random-legal baseline | n sites |
|---|---|---|---|
| Sense recall | **1.000** (308/308) | ~0.07–0.10 (23–31/308, baseline draw varies run-to-run) | 308 |
| Slot recall | N/A (no reference/elision sites in this gold set) | N/A | 0 |
| Structure recall | 0.000 (0/346) | 0.000 (0/346) | 346 |

(Random baseline is drawn from `policy="random"` over the same legal-action
mask as the model; its exact hit count has minor run-to-run jitter from
Python's non-deterministic set/dict iteration order in tie-breaking, but it
stays in the ~7–10% band across runs — nowhere near the model.)

### English held-out test split (reference, from prior training run —
reported as context, not regenerated here; `runs/encoder_gold_v2.jsonl` is
generated training data and isn't checked into any branch, so re-deriving
this split was out of scope for this test)

| Metric | Model |
|---|---|
| Sense recall | 0.931 |
| Slot recall | 0.978 |
| Structure recall | 0.000 |

### Side-by-side

| Metric | English (EN test) | Spanish (never-seen) | Spanish random baseline |
|---|---|---|---|
| Sense recall | 0.931 | **1.000** | ~0.07–0.10 |
| Slot recall | 0.978 | N/A (0 sites) | N/A |
| Structure recall | 0.000 | 0.000 | 0.000 |

## Interpretation

**Sense recall** here is a *candidate-set-recall*, not an exact-sense-ID
metric: it asks whether the model correctly decided that a given token
position is a sense-grounded argument (as opposed to entity/reference/
elision), not which of the (up to several) WordNet candidates it picked.
That's exactly the structural, language-agnostic decision the swap test is
designed to isolate: "does this argument slot need grounding in the shared
semantic space" — a decision driven by clause structure/role (e.g. a PLACE
role generally resolves to a sense site), which is invariant across English
and Spanish once the grammar and lexicon are swapped in as memory input.

The model gets **every one of the 308 Spanish sense sites right** (1.000),
against a random-legal baseline of ~0.07–0.10 — a ~10-13x gap, and if anything
higher than the English figure (0.931). Spanish sense sites in this gold
set skew toward nouns/places (predicate verbs fall back to entity-grounding
due to a base-lemma gap — see caveats), which are structurally simpler/more
uniform than the full English site mix, plausibly explaining the higher
Spanish number rather than indicating Spanish is "easier" in some deeper
sense.

**Slot recall** is not computable on this Spanish set — it contains zero
reference/elision grounding sites, so the metric is 0/0 (NaN) for both
model and random. This is a property of `spanish_gold_v2.jsonl`'s
templates, not a model result; no slot-transfer claim can be made from this
run.

**Structure recall** is 0.000 on Spanish, identically 0.000 to the English
reference. Per the task context this is the known missing-STOP-action bug
(the decoder over-generates clauses and never matches the gold tree
skeleton exactly) — it fires identically regardless of language, so the
Spanish 0% is the *same pre-existing architectural gap*, not new evidence of
cross-lingual failure.

## Verdict

**Grounding transferred cross-lingually by data-swap alone, on the signal
this test can actually measure (sense recall).** With the same frozen
English-trained weights, swapping in Spanish senses + grammar as memory
input drove sense recall to 1.000 (308/308) — at or above the English
reference (0.931), and roughly 10-13x the random-legal baseline (~0.07–0.10).
The encoder is not memorizing English surface parses; it is applying a
learned *grammar → candidate-grounding* policy that holds when the grammar
and lexicon underneath it are swapped for a language never seen in
training.

The other two metrics don't extend or contradict this: structure recall is
0% in both languages from the same known decoder bug (no cross-lingual
signal either way), and slot recall is simply not present in this Spanish
gold set (no sites to measure). So the clean result is sense-recall-only,
not the full three-metric picture originally hoped for.

## Caveats

- **Biggest caveat: this is a one-metric result.** Slot recall — the other
  metric the task flagged as a "clean cross-lingual signal" — has zero
  measurable sites in `spanish_gold_v2.jsonl`, so the swap test could only
  be run on sense recall, not sense+slot jointly as intended. A Spanish
  gold set with reference/elision sites would be needed to close that gap.
- Sense recall here measures grounding-type site correctness (is this a
  sense-grounded slot), not exact WordNet-sense selection; a model that
  gets the site right but the specific sense wrong still scores a hit. This
  is the same metric semantics used for the English reference number, so
  the comparison is apples-to-apples, but it's a coarser signal than "picks
  the right sense."
- Spanish predicate verbs fall back to entity-grounding (a base-lemma
  coverage gap in the Spanish lexicon build, not an encoder property), so
  the 308 Spanish sense sites evaluated are concentrated in nouns/places
  rather than spanning the same site mix as the English reference set — the
  two 0.93 vs 1.00 numbers are not drawn from directly comparable site
  distributions.
- The random baseline has minor run-to-run jitter (~0.075–0.101 across
  repeated runs) from non-deterministic tie-breaking order; it is reported
  as a band, not a single fixed value, though it is consistently far below
  the model's 1.000.
- English test-split numbers (0.931/0.978/0.000) are quoted from the task's
  prior training-run context, not regenerated in this session —
  `encoder_gold_v2.jsonl` (the English gold set) is build-script-generated
  data not checked into any branch, so re-deriving the exact held-out split
  was out of scope for a coding-light, no-retrain acceptance test.
