# Encoder training-data strategy

Status: RESEARCH / recommendation (2026-09-06, director). Question from lead:
how to improve the encoder without burning compute manufacturing a training set.
Context: encoder edge-F1 0.70 (decode) / 0.888 teacher-forced ceiling / 0.212
whole-tree-exact, trained on 788 English gold sentences (of 985). It is unknown
whether the encoder is data-limited, quality-limited, capacity-limited, or
decode-limited.

## Framing: four distinct levers, only one is "more data"
1. **Decode gap (FREE, no data).** 0.888 TF vs 0.70 decode = ~19 pts the model
   already knows but loses in beam search (exposure bias). Recover via better
   decoding (confidence-gating we already built; scheduled sampling / DAgger-style
   training on the model's own prefixes). Cheapest single win. Do regardless.
2. **Data quantity.** 788 is small. Only worth spending on IF the scaling curve
   is still climbing (see GATE).
3. **Data quality.** Gold is teacher-parser output, UNAUDITED. Some of the 0.30
   edge "errors" may be the model right / teacher wrong. More *noisy* teacher
   gold multiplies teacher errors. Auditing may matter more than volume.
4. **Capacity.** 343K params — untested as a limit.

## GATE (do this before manufacturing anything): data-scaling curve
Train the encoder on 200 / 400 / 788 gold sentences (same everything else),
plot edge-F1 + TF-accuracy vs n. Cheap (3 short runs).
- Still climbing steeply at 788 -> data-limited -> options A/B below pay off.
- Flattening -> NOT data-limited -> spend on the decode gap (#1) and/or a
  capacity sweep instead; manufacturing data would be wasted compute.
This is the "don't burn compute until you know you need it" discipline made
literal. Everything downstream branches on it.

## If data-limited, in cost order (cheapest first)

### A. Deterministic augmentation of EXISTING gold (near-free, best fit)
Structure is INVARIANT under content substitution, so we can multiply 788 ->
10k+ (sentence, gold-tree) pairs with GUARANTEED-correct trees and zero teacher
runs:
- **Sense-preserving lexical swap:** at a grounded content node, swap the word
  for another lemma of a same-POS sense (via USVS / WordNet). "the dog ran" ->
  "the cat ran": identical tree shape, the SUBJECT node's grounding changes to
  the new sense (which USVS already has a vector for). Teaches the encoder that
  structure is content-agnostic and grounding is sense-driven -- directly
  reduces overfitting to specific words (a plausible chunk of the 0.63-style
  gap seen in the decoder, and of encoder brittleness).
- **Clause recombination:** stitch two gold clauses into a new multi-clause
  record (trees compose; the buffer walk stays monotonic). Grows multi-clause
  coverage where whole-tree-exact is hardest.
- **Punctuation/casing/contraction normalization variants.**
Cost: pure CPU string+lookup work, no model, no teacher, correct-by-construction.
This is the strongest "no burn" lever. Risk: augmentation distribution drift ->
keep an un-augmented held-out test.

### B. Scale the teacher over more public-domain text (CPU-cheap, real sentences)
The teacher parser is deterministic and free to run. "Manufacturing" more gold =
running the EXISTING parser over more free text (Project Gutenberg children's
lit, McGuffey readers). Not expensive compute -- it's the parser we already have.
Pair with a **gold-quality audit** (sample N trees, hand-check) to know the noise
floor before trusting volume. Inherits teacher errors + MFS grounding; best as a
supplement to A, not a replacement.

### C. Self-training / bootstrapping (once encoder is stronger)
Run the (fixed, decent) encoder on unlabeled text; keep only HIGH-CONFIDENCE
predictions (tight candidate lattice / high beam margin) as new silver gold.
Gets data beyond the teacher with no human labels. Risk: reinforces own errors
-> gate hard on confidence + keep the teacher/augmented gold as the anchor.
Defer until after the decode gap + A are exhausted.

## Explicitly NOT doing
- No training on Spanish/other languages -- Spanish stays the held-out zero-shot
  exam (grammar-swap acceptance test). Cross-lingual is measured, never trained.
- No fabricated/LLM-generated gold -- would inject exactly the distributional
  guessing the thesis rejects.
- The K-12 graded-reader corpus is for the COMPREHENSION phase, not encoder
  structure training (though running the teacher over McGuffey is B-territory).

## Recommended sequence
1. Run the scaling-curve GATE + grab the free decode-gap win in parallel (both
   need no new data).
2. If data-limited: ship augmentation (A) -> retrain -> re-measure the curve.
3. Add teacher-scaled real text (B) with a quality audit only if A's curve says
   more real data still helps.
4. Self-training (C) last, as a force-multiplier once the base is strong.

Net answer to the lead: "more data" is premature until the scaling curve says so;
the biggest *free* win is the decode gap, and the best *no-burn* data lever is
deterministic sense-preserving augmentation of the gold we already have.
