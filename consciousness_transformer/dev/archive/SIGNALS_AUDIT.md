# Signals audit — what we've tried, what's untried, what's excluded

Self-audit of every signal source for the meaning-space minimization (M17–M28.0).
"Signal" = anything that constrains where a sense sits on the named axes or how
axes are chosen. Kept current as Step B lands; every new signal gets a row here
with its held-out delta vs the M28.0 baseline table.

M28.0 baseline (3k gloss corpus, 234 axes): placement syn>ant **0.693**, dict AUC
synonym **0.867** / similar **0.748** / **hypernym 0.727** / antonym 0.261(raw),
syn>rand uniform ~0.85–0.88 across POS regions. Improvement targets, in order:
**(T1) hypernym AUC — flat since M19.4**; **(T2) antonym edge breadth outside
nouns** (v-v: 5 test pairs, a-a: 10); (T3) similar 0.748.

## 1. Tried — signals in the space today

| signal | role | outcome (held-out) | where |
|---|---|---|---|
| WordNet glosses (decomposition) | the grounding anchor | grounding 0.126→0.255 w/ derived basis; foundation of everything | M17 |
| synonym edges | propagation close-edges | THE placement win: 0.404→0.693 | M19.2 |
| similar_to | propagation close-edges | in the winning set | M19.0/.2 |
| antonym edges | 4 attempts | in-coordinate: below chance (M17.3/M18.1); comparison-penalty 0.638 (M18.3); pole-anchor+propagation 0.693 (M19.2); conclusion: **a signed relation, not a position** | M17–M25 |
| is_a / hypernym | steering + eval | containment/AUC ~0.72–0.73, **flat through every milestone** (=T1) | M18.2+ |
| lexname (45 cats) | feature → named axes | in the honest-minimal mix; not load-bearing as a block | M19.1/M20.1 |
| attribute (adj↔noun dims) | 126 named axes + pole anchoring | real but coverage-capped (8% of words, 35% of adjectives) | M19.1/.2 |
| derivational | close-edges | best sense-node close set (w/ meronym): 0.497 vs 0.424 | M23 |
| meronym/holonym | close-edges | modest lift with derivational | M23 |
| gloss-overlap edges | close-edges | **DROPPED** — added nothing, raised random | M23 |
| verb_group | relation store | present; never individually ablated (do this in Step B) | M19.0 |
| morphological negation (un-/dis-/-less) | polarity flip | small (+0.05 with poles) | M18.1 |
| gloss magnitude (high/low) | polarity | small; recovers negation stopwords | M18.1 |
| sense frequency (lemma.count) | MFS ordering | used for sense choice, not placement | M22 |
| also_see | close-edges | **WIN** — similar +0.015 solo / +0.018 combined; in default set | M28.1 |
| domains (topic/region/usage) | feature → named axes | **WIN** — random overlap −0.032 (M21 objective); hypernym_cos −0.006 cost | M28.1 |
| satellite-cluster antonymy | antonym edge store | **WIN** — 2,553 new pairs (~11×), consistency 0.781; filter artifacts pre-M29 | M28.1 |
| genus–differentia gloss parse | close-edges | ❌ target moved (+0.010 hyp_cos) but similar −0.021 / random +0.022 — hypernymy is directional; re-route to relational store | M28.1 |
| entailment | close-edges | ❌ regresses similar — sequence, not sameness | M28.1 |
| pertainym | close-edges | null (within noise) | M28.1 |
| cause | close-edges | null (within noise) | M28.1 |
| verb_group (solo) | close-edges | null — 872/1015 pairs M24-collide with synonym test pairs (redundant with synonymy) | M28.1 |

## 2. Tried — methods on top of signals (documented outcomes)

| method | outcome |
|---|---|
| MDL basis derivation | works — a real defining vocabulary, converges with corpus size (M17.2/M18.0) |
| multi-signal *axis selection* | flat — relatedness levers are coordinate+edges, not which words are atomic (M18.2, honest negative) |
| sense-node primary space | correct per-sense grounding, loses to word-graph held-out — **closed** (M22–23) |
| per-word contrastive optimization | fails antonyms, fragile — documented negative (M21b) |
| joint training on ALL signals | loses to propagation on every metric; more training = worse — **closed** (M25) |
| normalization | tanh best for antonyms (0.756), standardize best all-around, minmax backfires (M20) |
| Null/sparse masks | the unrelated-separation win: random 0.32→0.03 (M21) |
| threshold-gated fusion | separation + discrimination together; product gate fails (M22) |
| leakage audit | M20/M22 leaks found+fixed; the M24 rule is standing law |

## 3. Untried — available signals, mapped to targets

*(M28.1 swept items 1–7 of the original list — genus, satellite, pertainyms,
entailment+cause, also_see, domains, verb_group solo — results now in section 1.)*

WordNet-internal remainder:

1. **morphosemantic links** (Princeton standoff file, small download) — types the
   derivational edges (agent/result/instrument); only if derivational-typed
   ablation suggests the types matter.

External, nltk-available (gated, currently untriggered):

2. **VerbNet classes** — gate was verb-region weakness; M28.0 shows v-v syn>rand
   0.848 ≈ nouns and M28.1 exposed no verb gap, so **hold**.
3. **FrameNet frames** — same gate, **hold**.
4. **Longman Defining Vocabulary** — not a placement signal; the external check
   on the derived basis (M17.2 predicted Longman-style; measure actual overlap).
5. **SemCor** — reserved for the WSD-vs-MFS gate, not placement.

Deferred by decision (future bridges, not needed for the English space):

6. OMW multilingual colexification — the cross-language bridge, out of scope now.
7. CLICS³ colexification — stretch goal behind OMW.
8. Wiktionary (translations, etymology families) — deferred.

## 4. Excluded on principle (do not revisit without a thesis change)

- **Distributional/co-occurrence signals of any kind** (word2vec-style): would
  smuggle opaque, unnamed structure into the axes — the core thesis violation.
- **ConceptNet as a placement signal**: crowd noise; at most an extra held-out
  eval set.
- **Trained per-word values**: M25 closed door (overfits, loses to propagation).
- **More axes for their own sake**: M20 closed door (the tail is noise).

## 5. Step B outcome (M28.1) → Step C

The sweep ran 2026-08-07 (harness `ground/ablation.py` + 3 parallel agents;
RESEARCH_NOTES M28.1). Hit rate 2.5 wins / 6.5 negatives out of nine signals.
Default placement set is now `signal_combined` (also_see close-edges + domain
feature axes + satellite antonym store): similar +0.018, random −0.029,
hypernym_cos −0.008, rest within noise.

Step C (next): freeze this signal set; filter the satellite pairs (compass/
off-sense artifacts); carry genus as a **directed relation** in the artifact's
relational store (its close-edge form is the documented negative); full-sense
placement; publish the artifact + English sense→coordinate dictionary.
