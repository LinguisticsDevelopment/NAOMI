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

WordNet-internal (zero new dependencies — Step B order):

1. **Gloss genus–differentia parsing → T1.** Glosses are near-formulaic
   ("a/an <genus> that/of <differentia>"); extracting the genus head gives a
   second, independent hypernym signal (the current one is synset pointers only)
   and differentia terms give feature evidence. Directly attacks the flat 0.727.
2. **Satellite-cluster indirect antonymy → T2.** WordNet adjectives: satellites
   orbit head adjectives via similar_to; heads carry the antonym. Propagating
   antonymy through the full satellite cluster (damp~wet, wet⊥dry ⇒ damp⊥dry)
   multiplies antonym coverage principled-ly. (M18.3 did one synonym hop; this is
   the sense-level structural version.)
3. **pertainyms** (dental→tooth) — cross-POS close-edges, same family as
   derivational (which earned its keep).
4. **verb entailment + cause** (snore→sleep, show→see) — directed edges; small
   counts but clean semantics; candidate for the signed/relational store too.
5. **also_see** — weak close-edges; cheap to ablate.
6. **topic/region/usage domains** — feature relations (axis-naming candidates,
   like lexname).
7. **verb_group individual ablation** — present since M19.0, never measured alone.
8. **morphosemantic links** (Princeton standoff file, small download) — types the
   derivational edges (agent/result/instrument); only if derivational-typed
   ablation suggests the types matter.

External, nltk-available (gated, currently untriggered):

9. **VerbNet classes** — gate was verb-region weakness; M28.0 shows v-v syn>rand
   0.848 ≈ nouns, so **hold** unless a Step B ablation exposes a verb gap.
10. **FrameNet frames** — same gate, **hold**.
11. **Longman Defining Vocabulary** — not a placement signal; the external check
    on the derived basis (M17.2 predicted Longman-style; measure actual overlap).
12. **SemCor** — reserved for the WSD-vs-MFS gate, not placement.

Deferred by decision (future bridges, not needed for the English space):

13. OMW multilingual colexification — the cross-language bridge, out of scope now.
14. CLICS³ colexification — stretch goal behind OMW.
15. Wiktionary (translations, etymology families) — deferred.

## 4. Excluded on principle (do not revisit without a thesis change)

- **Distributional/co-occurrence signals of any kind** (word2vec-style): would
  smuggle opaque, unnamed structure into the axes — the core thesis violation.
- **ConceptNet as a placement signal**: crowd noise; at most an extra held-out
  eval set.
- **Trained per-word values**: M25 closed door (overfits, loses to propagation).
- **More axes for their own sake**: M20 closed door (the tail is noise).

## 5. Step B execution order (from this audit)

1. Genus–differentia gloss parse (→T1, biggest flat target)
2. Satellite-cluster antonymy (→T2)
3. pertainyms + entailment/cause + also_see + domains + verb_group solo ablations
   (cheap batch; keep what moves the table)
4. Re-run minimality; update the M28.x table; then Step C (artifact + dictionary).

Each lands as: `wordnet.py` wrapper → `RelationGraph` relation → held-out ablation
vs M28.0 → RESEARCH_NOTES entry (win or documented negative).
