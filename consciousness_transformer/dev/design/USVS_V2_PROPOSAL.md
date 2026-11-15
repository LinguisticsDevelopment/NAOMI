# USVS v2 proposal — named dimensions + a graph-derived block with importance and sliding values

Status: PROPOSAL for owner decision (2026-11-15). Nothing is built. The current
607-dimension space stays frozen; v2 is built beside it and replaces it only if it wins.

Background (read these for the evidence): `dev/reports/USVS_CONTENT_DIAGNOSIS.md`
(baseline scorecard), `USVS_ALTERNATIVES.md` (survey), `USVS_PROBES.md` (the three
read-only probes this proposal rests on), `USVS_DESIGN_HISTORY.md` (what must not break).

## 1. The method in plain words

Which part of the architecture: the meaning space for CONTENT words (function words
already have their lexicon and operator dimensions; unchanged).

A content word's meaning becomes two blocks:

- **Named block (given, unchanged):** the existing 607 named dimensions. They say what
  kind of thing a word is. Measured strength: synonym and category structure.
- **Graph block (new, derived):** about 100 dimensions computed from the dictionary's
  relation graph, with recorded opposites as negative links. Every placed word gets a
  **sliding value** on each (a signed, continuous position) and an **importance** (how
  much that dimension matters to that word). Opposites end up on opposite sides, and the
  graph spreads that to words whose opposites were never recorded.

What is given by hand: nothing new. What is derived: the graph block, by a calculation
that is transparent, logged and repeatable (owner ruling: fitting is acceptable if
transparent and traceable, not opaque random weights). What is learned later: nothing
here; the encoder/decoder only consume the space.

Operators act on a word's position in the graph block as a whole:

- "not" reflects it (moves it to the other side of the middle);
- "very"/"extremely" move it further from the middle, "slightly" closer to it.

So "slightly unhappy", "unhappy", "extremely unhappy" are points along one line, and
"not happy" is across the middle from "happy" — near, but not identical to, "unhappy".

What would show it is working: held-out opposites separate (today a synonym pair is
closer than an opposite pair only 8% of the time), category structure does not degrade,
and degree/negation phrases come out in the right order.

## 2. Evidence from the probes (held-out pairs)

| | synonym closer than opposite | category (hypernym) score |
|---|---|---|
| today | 0.08 | 0.95 |
| clean-up only | 0.13 | 0.94 |
| importance + position from recorded opposites only | 0.10 | 0.96 |
| graph block alone (100 dims) | 0.71 - 0.75 | lost |
| named block + graph block | 0.30 - 0.50 | 0.97 - 0.94 |

- Only the graph generalises beyond recorded pairs, and it does so for nouns and verbs
  as well as adjectives (0.69 / 0.73 / 0.70).
- Importance + position gives the right ordering for very / slightly / not on recorded
  scales, but has no source of positions elsewhere. The graph block is that source.
- The graph block alone loses category structure and leaves 13-21% of senses unplaced.

## 3. Design

1. **Graph.** Nodes: dictionary senses. Positive links: similar-to, also-see,
   derivational, pertainym, attribute, verb-group, entailment/cause, hypernym (low
   weight). Negative links: antonym pointers. All from the dictionary — no word lists.
2. **Graph block.** k dimensions (start k = 100; choose by the scorecard). Two
   constructions were probed: a closed-form spectral calculation and a fitted embedding
   with a stated objective, fixed seed, single thread (bit-identical on re-run). Step 1
   below decides between them.
3. **Sliding value.** The sense's coordinate on each graph dimension, signed.
4. **Importance.** Non-negative, per sense per dimension. Starting definition: the
   magnitude of the sense's coordinate relative to the sense's own total (so importance
   says where the word "lives"), with senses the graph could not place getting zero
   importance on the whole block. Alternatives to test in step 1: importance from the
   number/strength of the sense's links.
5. **Comparison.** Named block as today (IDF-weighted). Graph block: compare only on
   dimensions important to both words, and use a measure that sees distance from the
   middle (so degree registers), not direction alone. The two blocks combine with one
   stated weight; the probes bracket it (opposites 0.30 at equal weight, 0.50 at double,
   with category score 0.97 -> 0.94).
6. **Operators.** "not" = reflect the graph-block position; degree words = scale its
   distance from the middle. They keep their own operator dimensions as now. Operator
   words never change importance.
7. **Storage.** Additive: the named block and its fingerprint are untouched; the graph
   block ships as a separate file with its own provenance record (graph edges used,
   construction, seed, k, version). Senses with no place in the graph block are marked
   "not placed", which is different from "in the middle".

## 4. What this changes in existing decisions

- "Senses are non-negative": still true of the named block; the graph block is signed.
- "Every axis named and interpretable": the graph dimensions are NOT individually
  nameable (probe: topic clusters for the spectral version, 1 of 15 nameable for the
  fitted one). The block as a whole is "position relative to opposites and neighbours".
  This bends the naming rule — owner decision.
- "No trained per-word values": relaxed by owner ruling to "transparent, traceable,
  repeatable".
- Content words still get no value on operator dimensions (unchanged).
- Encoder checkpoints must be retrained (already required by the new tree format).

## 5. Pre-mortem — how this fails

| Failure | Sign we'd see | What we do |
|---|---|---|
| Reflection does not land near the opposite | "not X" no closer to X's opposite than to random words | drop operator-on-block; keep negation as its own operator dimension only |
| Degree does not register | "very X" and "X" indistinguishable under the new measure, or the measure wrecks synonym scores | keep degree on operator dimensions only |
| Graph block hurts what works | category or synonym scores fall more than 0.02 at the weight needed for opposites | lower the weight; if opposites then fall under 0.25, abandon |
| Coverage too low on real text | over 20% of content tokens unplaced in a full build | add a documented fallback (inherit from nearest placed relative) or abandon |
| Gains are an artefact of leakage | held-out results vanish when whole clusters, not just edges, are held out | abandon |
| Not repeatable | two builds differ | use the closed-form construction only |
| It does not matter downstream | round-trip training does no better with v2 than v1 | keep v1; v2 stays a side experiment |

Abandon criteria are fixed now: any "abandon" row above, or no improvement over
"named block + recorded-opposite lookup" on held-out opposites of at least +0.15.

## 6. Staged plan (each stage ends in a go / no-go for the owner)

1. **Two cheap checks (read-only, one agent, ~2 h).** Using the probe's existing graph
   block: (a) does reflecting a word land near its recorded opposite — held-out; (b)
   which magnitude-aware comparison makes very / slightly / not order correctly without
   lowering the synonym and category scores; (c) cluster-level hold-out to rule out
   leakage; (d) spectral vs fitted on those three. No files in the repo change.
2. **Build v2 beside v1 (one agent, ~4 h)** only if stage 1 passes: full-graph build,
   provenance file, loader option; full scorecard vs baseline.
3. **Wire into the tree scorer and re-run its test set** (dog/cat, bone/stone,
   happy/unhappy/sad, very/slightly/not). Owner decides whether v2 becomes the default.

## 7. Questions for the owner

1. May the graph block's dimensions be unnamed individually, with the block documented
   as a whole? (Recommended: yes — the probes say individual names would be forced.)
2. Should "not happy" land near "unhappy" but not on it? (Recommended: yes; reflection
   gives that naturally.)
3. Proceed with stage 1 only, and decide again after it? (Recommended: yes.)
