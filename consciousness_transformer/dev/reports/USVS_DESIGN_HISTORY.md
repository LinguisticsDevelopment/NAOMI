# USVS design history (read-only research, 2026-11-15)

Purpose: before anyone changes USVS for function words / negation, record exactly what was decided, why, and what was tried and
rejected. Sources: `RESEARCH_NOTES.md` (cited as "RN §heading (line)"), `dev/archive/{SEMANTIC_MAPPING_PLAN,SIGNALS_AUDIT,MILESTONE_MAP,
INTEGRATION_PLAN}.md`, `dev/design/ROADMAP_LONG_TERM.md`, `MIND_ARCHITECTURE.md`, `archive/README_pre_refactor.md`, `GOALS.md`, the code in
`src/nsm_ct/ground/` and `archive/src/nsm_ct/ground/`, and the built artifact `data/usvs/` (fingerprint `e0daef638b640dd5`).
Convention: **RECORD** = what a ledger/doc/code comment says. **MEASURED-NOW** = I computed it today from the artifact or code (scratch
scripts only; no repo file touched). **INFERENCE** = my reading; flagged, not recorded anywhere.

---------------------------------------------------------------------------------------------------------------------------------
## 0. Orientation: the design in one paragraph (RECORD)

USVS = "a minimal set of non-overlapping, interpretable NAMED axes over which WORD-senses are *placed* by deterministic propagation"
(RN "The next layer" (1477-1521); README_pre_refactor.md l.43). Hard invariant from M19 (RN §0u, 1189): **every axis is named; no anonymous
word2vec dimensions; axes never rotate or mix.** What the coordinate captures: *gradient "how-alike" meaning* (synonymy AUC 0.86,
hypernymy 0.72, similarity 0.73, unrelated pairs ~0). What it explicitly does NOT capture and delegates to structure ON the space:
**opposition ("a relation, not a position"), composition ("an operation, not a point"), reasoning ("dynamics")** (RN 1489-1500). The
3 "closed doors" (do not re-open without new information): more axes for their own sake (M20), sense-nodes as primary space (M22-23),
trained per-word embeddings (M25) (RN 1518-1521; SIGNALS_AUDIT §4). GOALS.md goal 1 says "the grounded lexical space (USVS) is settled and
not being revisited". Everything below is consistent with that, but note: **the record says nothing about function words, because the whole
M17-M29 arc was content-word/WordNet-driven; function words were *excluded by construction* (see §3.3).**

---------------------------------------------------------------------------------------------------------------------------------
## 1. Timeline (Q1)  -- measured result / decision per step

| Step | Built | Measured (as recorded) | Decision |
|---|---|---|---|
| **Pre-M17** (RN "9. Reconciling NSM primes with 51 anchors" (1045)) | `nsm_primes.py` 65-prime table; NAOMI had 51 older anchors | "nobody has decided which (or how) is the basis" -- open | Primes = seed; basis to be *derived* (M17) |
| **M17** RN §0s (1078) | `ground/`: gloss decomposition to primes, MDL basis search, deterministic clause==word reduction | grounding 0.126->0.255, syn>ant 0.227->0.455 (n=22, **below chance**), round-trip exact 0.909, DeepNSM agreement 0.072->0.059 ("demonstrates independence") | Basis is **derived by MDL, seeded by NSM-65**, not assumed. Antonyms "need explicit antonym **edges**" (l.1117-1120) |
| **M18** §0t (1142) | 10k gloss-vocab corpus; `polarity.py` (signed pole pairs GOOD/BAD->EVAL etc. + gloss-cue magnitude + morphological negation); multisignal basis; graph closeness | polarity coords syn>ant 0.413->0.466 (still < chance); basis-selection flat (honest neg); **graph-aware closeness 0.333 / 0.392 / 0.638** (coord / polarity / graph, n=230 held-out) | The relational web (signed edges) does what coordinates can't. M18.1 notes cue list "recovers the negation words decomposition drops as stopwords" (l.1160) |
| **M19** §0u (1189) | one unified relation store; 234 named axes (65 primes + 126 attribute dims + 43 lexname); SVD dimensionality; placement = anchored coordinate + label propagation; minimality; dictionary reconstruction | intrinsic dim ~46 (90% energy), effective ~17; **plain-cosine syn>ant 0.404 -> 0.693** (attr axes signed by gloss magnitude + propagation); "~30 axes reproduce 95%" (variance-ranked; later retracted); dictionary AUC syn 0.857 / similar 0.734 / hyper 0.722 / antonym-by-distance 0.275 | "Antonymy now lives in the position" (l.1212) -- **but see M20/M25 for the later correction to this**. Axes never rotate |
| **M20** §0v (1232) | `normalize.py`; honest (leave-one-out) minimality | random-pair cos **0.32 (dense, 137/234 axes active)**; z-score -> 0.007; **tanh(z)** -> 0.145 and best syn>ant (raw 0.673 -> tanh **0.756** held-out; originally mis-reported 0.94, leaked, see M24); honest minimality curve **peaks ~0.83 at K~80-120 then falls to 0.756 at all 214** ("the last ~100 axes hurt") | Default = tanh (bounded [-1,1], the owner's request). Standardize is the all-round alternative, surfaced as a choice. "30 axes" was a variance-ranking artifact |
| **M21** §0w (1278) | `sparse_value.py` (value + Null mask per word, IDF-weighted cosine), `contrastive.py` | **WIN** random sim 0.32 -> 0.08 -> 0.03 (words occupy ~2.9 applicable axes); **honest neg** per-word syn>ant 0.39 -> ~0.48 (< chance; good/bad collapses to 1.0 even anchored) | Null/bipolar representation and relational propagation are **complementary**; per-word optimisation does NOT replace propagation for antonyms. (Archived, not in USVS) |
| **M22** §0x (1314) | `sense_graph.py` (synset nodes), `fusion.py` threshold-gated propagation | sim>ant 0.632 (**later shown leaked**), threshold gate cut random 0.198 -> 0.116 (a product gate failed: disc 0.495) | Sense nodes kept as an idea; see M23 |
| **M23** §0y (1349) | derivational / meronym / gloss-overlap close edges | held-out sense-node syn>ant **0.497** (best set = similar+deriv+mero); gloss-overlap dropped (adds nothing, raises random); "sense-node propagation barely beats chance" | Word-graph beats sense-nodes (0.756 vs ~0.50). Closed door |
| **M24** §0z (1393) | full leakage audit | two real leaks: M22 0.632 -> ~0.50; M20 0.94 -> **0.756** (raw 0.828->0.673, standardize 0.767->0.625, tanh 0.938->0.756). M19.2 0.693 and M18.3 0.638 were clean | **House rule**: a metric must not propagate over the pairs it scores (`train_pairs=`) |
| **M25** (1434) | `joint_place.py`: train per-word values on all relations | syn AUC 0.60 / syn>ant 0.56 / random 0.27 vs propagation 0.72 / 0.73 / 0.18; more training = worse (0.60->0.54) | Deterministic propagation over named axes is THE mechanism; trained per-word values = closed door. Mask caps density; "masked per-word values can't push words that share all applicable axes to opposite" |
| **Post-grounding roadmap** (1477) | -- | "Opposition is a relation, not a position... the only lever that has ever moved it is a **signed relational edge** used at scoring/composition time" | (A) sense-resolve into Mind, (B) compose clauses with "antonymy/negation as signed operations", (C) antonymy as a first-class relational layer, (D) project Mind state onto named axes. B/C never built |
| **M27** (1523) | deleted token stack | -- | grounding signal belongs in meaning-graph fillers |
| **M28.0** (1553) | baseline harness `probe_m28_baseline.py` (3k, 234 axes, 17 s) | placement 0.404 -> 0.693; hypernym AUC 0.727 flat since M19.4; v-v only 5 and a-a only 10 test antonym pairs vs 139 n-n | Target list: hypernym, antonym edge breadth, similar |
| **M28.1** (1584) | `ablation.py` + 9 signals | 2.5 wins / 6.5 negatives. WIN: also_see (+0.015 similar), domain features (random -0.032), **satellite-cluster antonymy (2,553 new pairs, 0.781 vs original 0.695, placement-inert)**. NEG: genus-differentia as *close edge* (hyp +0.010 but similar -0.021, random +0.022), entailment, pertainym, cause, verbgroup | "Hypernymy, like antonymy, is *directional structure* -- belongs in the relational store, not closeness" |
| **M29** (1635) | `usvs.py` + `scripts/build_usvs.py`; 9,946-word placed core, 607 axes, 117,659 senses, 11,317 tiered antonym edges, 1,546 directed genus edges, `dictionary.jsonl.gz` | held-out @10k: syn AUC 0.868->0.885, similar 0.747->0.803, random 0.255->0.233, syn>ant 0.695->0.676 (883 vs 251 test antonym pairs); 72 s build | Frozen artifact. "Semantic mapping is frozen" (SEMANTIC_MAPPING_PLAN status). Production build uses ALL edges; quality numbers only from held-out harness |
| **M30** (1680) | SemCor WSD gate, training-free | usvs-idf polysemous 0.353 vs MFS 0.727 vs random 0.274; unsure-slice 0.382 vs MFS 0.584; **SimLex-999 rho 0.333 cosine -> 0.380 + antonym store -> 0.417 core+edges** (word2vec ~0.41-0.44) | MFS keeps the WSD slot; "stop pointing USVS at context-WSD". The SimLex +0.047 is the one number showing the two-part design (coords + signed edges) earning its keep |
| **M31/M31b** (1788) | `usvs_bridge`, handles | concept deref top-1 0.316->0.962 (d=256); reactor val 0.698->0.885 | USVS becomes default `meaning_source` |
| **M32-M34** | ambiguity curriculum; trained sense chooser | gold-sense ceiling 1.00 on 4 families; LOFO 3/4 | -- |
| **M40/M42** (2087 / 2169) | chooser on raw 607 axes; grounding-depth diagnosis | raw ceiling 0.813 (projection exonerated). Old signatures 2-6 axes; **SOMETHING in 66,875 (57%) of senses**, "cosine is substantially a genericity match"; IDF 50/62 -> 53/62 | Grounding depth = #1 open item |
| **M44** (2218) | gloss-word enrichment | negative with M24 guard (51/62 vs 53/62); root cause: `sense_prime_weights` kept only literal-prime leaves (HEAD discarded) | The lever is the prime whitelist, not query-time blending |
| **M46** (2284) | `sense_usvs_weights`: senses grounded IN the space = genus (w=1.0) + gloss words (w=0.7), 1/sqrt(n_senses) discount, top-24 sparsified; headword's own lemma excluded; prime decomposition = fallback only | **Parity**: 62-proxy 50/52 vs 50/53; SemCor 0.237 vs 0.232; gold ceiling 0.810 vs 0.812; nnz/sense ~4 -> ~24; fallback = 1,208 senses (1.0%) | Adopted "on architectural grounds, NOT a performance win": "USVS is our Semantic Metalanguage, we don't need a vestigial lossy one" (owner) |
| **M49 / 2026-09 / 2026-11** | parser, encoder, usvs_graded scoring (RN 4997-5151: USVS-space *metric* kept; USVS-space *loss* negative at 788 and 8,000), naomi loop | -- | USVS consumed as sense vectors (`sense_dense`) by `encoder_model.py:236`, `usvs_graded.py:255`. **Nothing live consumes core coords or the antonym store** (grep: only `build_usvs.py` spot-check) -- MEASURED-NOW |

---------------------------------------------------------------------------------------------------------------------------------
## 2. Antonyms (Q2)

### 2.1 How antonymy is represented in the artifact today (MEASURED-NOW + code)
Three separate mechanisms, only the third is relational:
1. **Anchored sign on `attr:` axes only (core words only).** `placement.anchored_coordinate` (placement.py:37-48): for each word,
   `sign = gloss_polarity(w)`; if non-zero, every `attr:<dim>` axis the word has (WordNet `attribute()` relation, i.e. adjective->noun
   dimension: hot->temperature) is set to exactly **+1 or -1** instead of +1. Then ordinary label propagation (alpha=0.7, 20 iters) over
   synonym+similar+also_see spreads those signs to neighbours.
2. **Pole pairs are NOT collapsed in USVS.** GOOD/BAD, BIG/SMALL, MUCH_MANY/LITTLE_FEW remain separate unsigned prime axes (axes list is
   `PRIME_NAMES` verbatim, `axes.py:assemble`). The signed-pole layout (`POLARITY_PAIRS`, `signed_axes`, `polarity_vector`,
   `negation_base`) exists in `polarity.py` but is only used by archived M18 closeness/probes -- MEASURED-NOW + audit.
3. **Signed antonym-edge store**, `antonym_edges_tiered` (usvs.py:285): 11,317 `(a, b, tier)` pairs. Query API `USVS.antonyms_of(word,
   tiers=DEFAULT)`; default floor excludes `satellite_satellite`. **Placement-inert**: edges never feed placement (signal_satellite.py
   docstring: "Antonym edges do not currently feed placement"). They are "consulted at comparison/composition time" -- a design intent
   (ROADMAP_LONG_TERM §3) with **no live consumer** today.

MEASURED-NOW on the shipped artifact: 0 of 2,785,213 sense nonzeros are negative (usvs.py:275 `np.where(blend >= cut, blend, 0)` zeroes
negatives when top-24-sparsifying), so **senses carry no sign at all; signs exist only in `core_coords`**: 356,369 negative entries,
6,670 of 9,946 core words, 220 `attr:` axes (0 prime axes, 0 `lex:`). Examples: hot/cold cos -0.013 (attr:temperature +0.36/-0.33);
good/bad cos 0.20 (attr:quality +0.37/-0.29); possible/impossible 0.39 (attr:possibleness +0.35/-0.34); big/small cos 0.50 (separate BIG
and SMALL axes). The audit's "mean antonym core-pair cosine 0.47 vs random 0.28" also stands.

### 2.2 What `ground/polarity.py` contributes (Q2 sub-question) -- MEASURED-NOW
- `gloss_polarity` (polarity.py:77) counts cue words in the **first-sense gloss**: `_POS_CUES` 38 words (high, more, large, big, strong,
  above, fast, long, hot, heavy, deep, wide, rich, presence...) vs `_NEG_CUES` 42 (not, no, without, lacking, absence, opposite, less, low,
  small, weak, below, slow, short, cold, light, shallow, narrow, poor, empty...) (polarity.py:48-62); net>0 -> +, <0 -> -, 0 -> 0.
  Cue words are English; the file calls them "a fixed, interpretable lexicon (NOT the antonym edges)".
- Reach, at the 10k production vocab: only **414 core words have any `attribute` relation**, **130** of those get a non-zero sign (84 negative),
  touching **217 attr axes** (164 with a negative-signed word). Propagation then spreads the sign to 6,670 words / 220 axes. At 3k vocab:
  59 of 3,000 words actually signed (32 negative).
- Ablation run today (3k corpus, `evaluate_placement`, same split, alpha 0.7): cues present **anchored 0.404 -> placed 0.693**; cues removed
  (`gloss_polarity` forced to 0) **anchored 0.353 -> placed 0.652** (n_test_syn 2,927, n_test_ant 230). So removing the lists costs
  ~0.04 held-out syn>ant but does not collapse antonymy to chance: without a sign every attr axis is +1 for both members of an antonym pair, so
  antonyms differ only through the unrelated parts of the vectors / propagation. Nothing else in USVS would break (senses and the edge store
  never read it). `negation_base` (affix stripping) is **not called by the build** (audit; grep: only `polarity_vector`).
- Record on strength of the signal: SIGNALS_AUDIT rows "gloss magnitude: small; recovers negation stopwords" and "morphological
  negation: small (+0.05 with poles)" (M18.1).

### 2.3 Alternatives tried (RECORD, numbers from the ledger)
| Alternative | Where | Result | Verdict |
|---|---|---|---|
| Plain coordinate cosine over decomposition | M17.3 (1112) | syn>ant 0.227->0.455 | below chance; antonyms "share nearly all gloss structure" -> rated similar |
| **Signed pole pairs** (GOOD/BAD->+-EVAL, BIG/SMALL->+-SIZE, MUCH/LITTLE->+-QTY) + **morphological negation flip** (un-/dis-/-less flips base's poles) | M18.1 (1158) | 0.413 -> 0.466 (n=397) | helps but < chance. **Not carried into USVS** (separate unsigned prime axes remain) |
| Gloss-cue **magnitude axis** (hot="high" vs cold="low") | M18.1 | included above; "recovers negation words decomposition drops as stopwords" | re-used as the sign source for attr axes (M19.2) |
| Graph-aware closeness: coordinate cosine minus antonym penalty from train edges (one synonym hop) | M18.3 (1173) | **0.638** vs 0.333/0.392; clean (circularity-free) | works but needs train edges next to the test pair (coverage-capped); "the antonym solution" at that time |
| **Anchored attr-axis sign + propagation** | M19.2 (1207) | **0.693** plain cosine, beating M18.3's comparison-time penalty | **KEPT** -- this is what is in USVS |
| tanh normalisation | M20 | 0.673 raw -> **0.756** (honest) | best for antonyms; USVS ships raw cos + tanh available in harness (`normalize.py`); built USVS coords are max |1.0| (tanh-bounded) |
| Null-aware sparse (value, mask) per word; antonym prime pairs on ONE signed axis (good=+1, bad=-1) | M21 (1278) | unrelated separation WIN (random 0.32 -> 0.03); antonym 0.39 -> 0.48 | not adopted for antonyms; good/bad "collapses to 1.0 even anchored" |
| Per-word contrastive training (syn->+1, ant->-1, random->0) | M21b | ~0.48 | closed; documented negative |
| Fused Null + propagation, threshold-gated; product gate | M22 (1314) | 0.632 (leaked) -> ~0.50; product gate 0.495 | closed |
| Sense-node close edges (derivational/meronym/gloss-overlap) | M23 | best 0.497 | closed |
| Joint training on all signals | M25 | 0.56 vs 0.73 | closed; "masked per-word values can't push words that share all applicable axes to opposite" |
| Antonym-pair axes / a NOT prime as an antonym device | **Not tried as a placement device.** NOT is a seed prime axis but is never populated (see §3). Morphological negation was the only NOT-like device, via sign flips (M18.1), not via the NOT axis | -- | INFERENCE: this is the open seam for the new work |
| **Signed relational edge store (tiered)** | M28.1 -> M29 | 2,553 new satellite pairs scored 0.781 by the existing geometry; SimLex 0.333 -> 0.380 (+0.047) with edges | **KEPT** as the antonym layer |

### 2.4 The tiers (usvs.py:147-152, 285-325; M28.1 §1584; signal_satellite.py docstring) -- RECORD + code reading
WordNet adjectives: **head** synsets (pos `a`) carry direct antonym pointers; **satellite** synsets (pos `s`) orbit a head via `similar_to`
and have none. Tiers, strongest first (MEASURED-NOW counts): `direct` 702 (lemma-level pointer, or a head adjective's own antonym),
`satellite_head` 2,681 (a satellite word <-> the antonym lemma of its head, e.g. damp <-> dry), `satellite_satellite` 7,934 (word <-> the
satellites of the opposed head, e.g. damp <-> arid; the code also assigns head-word <-> opposed satellites to this tier). Default query
tiers = direct + satellite_head (3,383). Reason for tiering: M28.1 sampled artifact classes (compass-direction chains such as
north/south, off-first-sense pairs such as minor/star) concentrate in satellite_satellite; "tiering replaces naive filtering, consumers pick
their floor" (RN 1651). Caveat recorded at M29: lemma-level tiers inherit sense conflation (`antonyms("hot")` includes "bad" via a slang sense)
-- "the WSD layer, not the store, is where that resolves".

### 2.5 Owner statement vs record
"Antonyms are much more complicated than only occupying one dimension and we did a lot of reasoning for that" is exactly what the record
shows: five distinct attempts (above), the conclusion "opposition is a relation, not a position" (RN 1489-1494), and the standing rule that
antonymy is delegated to "signed edges" at comparison/composition time (SEMANTIC_MAPPING_PLAN "Done means"). The one thing the record does
**not** say is that antonym pairs should be separated by a *sign flip on a shared axis in the coordinate*; the sign-on-attr-axes mechanism is
a **partial anchor** (M19.2) whose own author warns "attribute-pole anchoring is partial" (attr coverage 8% of words, 35% of adjectives, RN 1224).

---------------------------------------------------------------------------------------------------------------------------------
## 3. The prime axes (Q3)

### 3.1 Intent (RECORD)
- Primes were the **seed**, not a limit: "NSM primes = degrees of freedom (axes); WordNet = the LTM dictionary"; "the basis is **derived, not
  assumed**" and "seeded by NSM-65" (M17, RN 1087-1091). `axes.py` docstring: "the NSM primes (**semantic floor**) + attribute dimensions + 44
  lexname categories". Prime decomposition was the grounding mechanism ("a reduced clause lexicalizes back to the word it defines",
  "decomposition to the prime floor", `definition_graph.py` docstring) before M46; after M46 it is the OOV fallback only (1,208 senses).
- Primes were never intended as "populated axes for every word". I found **no passage** that states a prime axis must be populated, or that a
  specific one may be empty. Only `nsm_primes.py` flags canonical-list uncertainty (`TODO(canonical-list)`: DON'T WANT vs WANT, `(IS) MINE` vs
  HAVE, allolexes). RN M17 also records: "molecules are grounded *points* (not axes)... molecule-grounded words have an empty prime
  coordinate" (l.1126-1127) -- the nearest explicit statement that a word/axis can be empty.
- The MDL basis search (M17.2) promoted **15 non-NSM primitives** (act, make, person, give, feeling, state, life... a "Longman-style defining
  vocabulary", "the predicted 'more than just NSM'") and the 10k run found act, having, relating, state, person, particular, substance, things,
  number, cause, make (RN 1104-1107, 1155). These promoted axes were an `extra_axes` device in the decomposition and are **not** in the
  607-axis USVS (the USVS axis set = primes + `attr:` + `lex:`).

### 3.2 Why 26 primes are empty -- mechanism (MEASURED-NOW; the record has no explicit decision)
The 26 empty prime axes (I, YOU, THIS, THE_SAME, SOME, DONT_WANT, DO, BE_SOMEWHERE, THERE_IS, BE_SOMEONE_SOMETHING, MINE, BEFORE, AFTER,
A_LONG_TIME, A_SHORT_TIME, FOR_SOME_TIME, ABOVE, BELOW, NEAR, NOT, MAYBE, CAN, BECAUSE, IF, VERY, MORE; matches the audit) are explained by two
facts I verified:
1. `meaning._STOPWORDS` (meaning.py:43, 149 English words, "excluded from gloss decomposition") contains the single-word exponent of **18 of the
   26** (i/me, you, this, some, do, be, have(=MINE), before, after, above, below, near, not, can, because, if, very, more). The other 8 are multi-word
   exponents (`the same`, `don't want`, `there is`, `a long time`, `a short time`, `for some time`, `be somewhere`) or `maybe`, which never
   appear as a single token leaf.
2. `ground/corpus.gloss_vocabulary` (corpus.py:38) filters by the same `_STOPWORDS`, so none of these words is a core word (MEASURED-NOW: `not,
   before, after, if, because, this, can, more, very, i, you, above, near, some, do, be, have, the, and` are all absent from the 9,946 core
   words; some have WordNet senses, e.g. can 9, do 16, have 20, but `if`, `because`, `this`, `you`, `the`, `and` have 0).
   Therefore neither the core word nor any gloss decomposition can ever emit these prime labels. Only the 39-word "other" primes whose
   exponents are content-like (SOMEONE, GOOD, KNOW, WHEN...) populate.
INFERENCE: this is **by omission**, an unintended side effect of reusing the English stopword list (itself a hand-written English list in
code, the thing the 2026-11-14 owner rules now forbid) for a content-word pipeline. The only place the record touches it is M18.1: the cue
lexicon "recovers the negation words decomposition drops as stopwords" (RN 1160-1161) -- i.e. the project **knew** negation words were dropped
and worked around it with sign cues rather than with the NOT axis. No ledger entry lists "empty prime axes" as an open item (grep of
`empty|never loaded|dead axes|unused axes` in RN: none).

### 3.3 Why SOMETHING / PART / ONE / WHEN are near-constant (MEASURED-NOW + RECORD)
Current senses: SOMETHING 115,764 (98.4%), ONE 114,588 (97.4%), PART 114,370 (97.2%), WHEN 97,011 (82.4%); core words: 8,020 / 7,439 / 7,230 / 6,966
of 9,946 (median core word has 560 non-zero axes, max |value| 1.0). Record:
- M20 (1239): placed coordinate is **dense** (137/234 axes active per word) because propagation spreads support ("the coordinate densifies").
- M30 (1688), M42 (2169): "ubiquitous primes like SOMETHING otherwise dominate"; SOMETHING in 57% of *old* signatures = "a genericity match";
  IDF reweighting is the recorded fix (50/62 -> 53/62), later opt-in not default.
- M44 (2229): "generic primes win because decomposition keeps whatever whitelisted prime it stumbles into".
- INFERENCE: the SOMETHING fallback leaf of the resolver (`meaning.py` resolution step 6, and the exponent maps `one`, `part`, `when~time`), plus
  M46's inheritance of dense *core* coordinates through genus/gloss blends, raised SOMETHING from 57% (old signatures) to 98% (current). M46's
  entry reports parity metrics but does not mention this increase. PART/ONE/WHEN are common words/definitional filler ("part of", "one who", "time").
  Consumers handle it by IDF-weighting (usvs_graded, M30) -- not by removing the axes.

---------------------------------------------------------------------------------------------------------------------------------
## 4. Function words / closed class: recorded decisions (Q4)

Chronologically (all RECORD unless marked):
1. **M17-M29: function words excluded from USVS by construction** (stopword filter, §3.2). No decision written down as such.
2. **`NSMMeaningResolver`** (meaning.py:135-153): precedence "prime exponent -> molecule -> explication store -> WordNet person-sense ->
   gloss decomposition -> SOMETHING/SOMEONE fallback". `_PRIME_EXPONENT_MAP` flattens allolexes (`I~ME`, `SOMETHING~THING`, `MUCH~MANY`...) to
   74 English keys. So a prime exponent resolves to a **prime leaf label, outside the vector space** (audit §Headline).
3. **D2 (2026-09-07, "DECISIONS LOCKED", RN 4527-4535; original `dev/CURRENT_STATE.md`; hand-gold analysis `dev/archive/HAND_GOLD_DRAFT.md` §D1-D6)**:
   interjections ground to a REAL sense (content ones -> WordNet; pure ones ugh/ah/hey/oops... -> gloss-grounded into USVS:
   `PURE_INTERJECTION_GLOSSES` usvs.py:124, 20 entries -> `interj.<word>.01`, "pure_interjection_senses": 20, added *after* the fingerprint so
   inert). `utterance_kind=interjection`; **no appraisal node** (an earlier appraisal design `I FEEL something good/bad` was STRIPPED,
   ENCODER_IO_CONTRACT_V2 §6, "valence DERIVED from structure never imported").
4. **D3 (same date)**: "add prime I (me->I); YOU exists". The speaker is a resolved `prime:"I"` grounding, symmetric with the synthesized
   imperative addressee `prime:"YOU"`; `encoder_model.PRIMES = ["YOU","I","<UNK_PRIME>"]`; `clause.FIRST_PERSON_SINGULAR` (clause.py:~467)
   routes i/me/myself. Contract: ENCODER_IO_CONTRACT_V2 §5. Decision text: HAND_GOLD_DRAFT §D3 ("Symmetry argues for the prime; consistency with the 1,259
   teacher records argues against"). Note this is a **grounding-type** (`prime`), whose gold node vector is a reserved block with **no USVS axes**
   (usvs_graded.py:268; audit §5) -- so I/YOU are "primes" in the tree but unpopulated in USVS.
5. **D4 (same date)**: "add role labels QUANTITY/ADDITIVE/FOCUS (**structural only, NOT USVS coords**)" (RN 4532; clause.py:455-463
   `MODIFIER_RELATIONS`). These are modifier-role labels for bare quantifiers ("more !"), "too"/"also".
6. **D6 (same date)**: elision slots retain the surface carrier ("keep 'did' token_index for tense/polarity", RN 4535; encoder_model.py:293;
   hand_gold.py:371). The only recorded statement about **tense/polarity**: an auxiliary carrier keeps it; no tense axis or field exists.
7. **Prepositions**: collapse into role labels via a hand table in `clause.py` (`_PREP_RELATION`, PLACE/SOURCE/OF/WITH...), and in gold v6 only 39.5%
   of tokens are tree nodes (ADP 3.6%, CCONJ 9%, PART/SCONJ 0%) (RN 5452 "Read: hand patterns removed..."). "the preposition token itself is not
   grounded (flushed by the oracle's terminal SHIFT), matching teacher-gold treatment of function words" (HAND_GOLD_DRAFT l.244).
8. **Negation**: `MODIFIER 'not'` edge -> `DiscourseLink("NOT","NOT")` (clause.py:848-858); dropped by both encoder and POS cheat as "dominant
   failure: negation shared gap" (RN 5441); marked per clause as `negated` by `src/naomi/reader.py:249` (types.py:37). RN 5452: "negation only as an
   undifferentiated MODIFIER edge (needs its own label); possessive 's not emitted; numerals only as DESCRIPTION"; NEXT (awaiting owner): enrich tree
   format, keep prepositions/negation/connectors as tokens.
9. **2026-11-14 owner design rules** (CLAUDE.md "Design rules"): closed-class word defined by its position on USVS dimensions stored as
   per-language data; special dimensions may be added by hand (negation, known/new, not-known, possession, time...); no grammar-label/code
   path for negation/questions/tense. This **supersedes** D4 (structural labels) and D3 (reserved-block prime) in spirit; no ledger entry yet
   reconciles them. The audit (`USVS_FUNCTION_WORD_AUDIT.md`) is the first document to examine the gap.
10. **Original decision docs**: `dev/CURRENT_STATE.md` "DECISIONS LOCKED" (no longer contains D3/D4 text -- grep finds none; the ledger copy RN 4527 +
   `dev/archive/HAND_GOLD_DRAFT.md` D1-D6 and `dev/design/ENCODER_IO_CONTRACT_V2.md` §5-§6 hold them).
- Searches for `speaker_prime`, `synth_subject`, "numeral", "determiner" decisions: `speaker_prime` appears only as a weak encoder **family**
  (GOALS.md l.85); no separate design decision found for numbers/determiners/wh-words/pronouns beyond the above. The only parser-level closed-class
  list is `quantum_parser/src/parser/pos_tagger.py WORD_TAG_DICT` (294 words, hand-written; RN 5452 flags it).

---------------------------------------------------------------------------------------------------------------------------------
## 5. Operators and logic: how negation / connectives were meant to be represented (Q5)

All RECORD unless marked.
- **Operators are NODES, not flags** (RN §0g/0j 383-390; `meaning_graph.py:240-247` comment). Reason: the old parser flag list
  (`List[SubType]` NEGATIVE/L_NOT) "is not deconvolvable because a flag carries no binding to which child / relation / order set it".
- `apply_operator` (meaning_graph.py:250-288): `M_op = bind(self_role, filler(op_name)) + sum_i bind(role_vec(i,"OP_ARG"), unit(clause_i.handle))`,
  stored on an OPERATOR node with `OPERATES_ON` edges. Operator labels from MIND_ARCHITECTURE.md: `{NOT, MAYBE, AND, OR, IF}`. `NOT`
  additionally sets the target clause's `meta["truth"]="FALSE"` (a lossless tag "for read-time filtering"). `read_operator` recovers label
  (score ~0.99) and the clause argument (cos ~0.99) -- gate in RN 383-390. The wrapped clause vector is untouched (`allclose`).
- **Truth tags** (clause.py:399-420, 930-955; RN §0i 276): `{TRUE, FALSE, MAYBE}`, `tag(M,v) = M + bind(role(TRUTH), filler(v))`,
  recoverable by unbind + cleanup; "store as OR, then decide truth": disjuncts start MAYBE, a negation re-tags the refuted one FALSE and the
  survivor TRUE, nothing deleted ("overwrite but don't forget"); unresolved query -> first-class `MAYBE` answer.
- **Connectives as relations between clauses** (clause.py:386-395): `_COORD_PRIME = {OR: MAYBE, BECAUSE: BECAUSE, IF: IF, WHEN: WHEN, NOT: NOT, AND:
  None, BUT: None}`; "NSM has no AND/OR exponent -- disjunction *is* MAYBE". Each disjunct is a separate lossless clause matrix; the coordinator's prime
  atom is bound on a CONNECTIVE role (`_CONNECTIVE_POS=2`) of the edges tensor; `DiscourseLink(coordinator, prime, i, j)`.
- **TPR encoding** (tpr.py + meaning_graph): primes are atoms `filler_vec(label)`; role positions are orthonormal Q-columns with a per-relation
  random sign (tpr.py:117-122); an operator applied to a clause = the matrix above (a *bound, scoped* node), a truth tag = an additive bound
  adjective on the clause matrix; both are exactly decodable by unbinding. **These atoms are keyed by label, independent of USVS axes** (INFERENCE:
  which is why NOT works in the Mind while the NOT *axis* in USVS is empty).
- **Neural side**: the executor/reactor has a learned "negate" strength: `value_gate = write - decide_truth` (RN §0i 342; OP_INVENTORY.md l.59,
  76: `decide_truth: Linear(hidden+d,1)`, executor `S_neg` scalar). L7/L8 results: unresolved->MAYBE 1.00, resolved-by-negation 0.86,
  negation-removal 0.70. That is the only **graded (0..1) negation** in the record, and it is a learned scalar write-strength, not a position in USVS.
- **Graded truth / graded MAYBE**: **no recorded position.** Searches (`graded|fuzzy|probabilistic|degree`) hit nothing about truth degrees.
  MAYBE is a discrete atom/tag. The graded notions that DO exist concern *similarity* (tanh-bounded axes in [-1,1] by owner request at M20; Null mask +
  IDF-cosine at M21).
- **Current minimal loop**: `src/naomi/memory.py:165` `apply_operator(..., "NOT", ...)` when `tree.negated` -- negation is an operator node over a
  clause (docstring l.6-7), marked by the reader (`reader.py:249`). The 2026-11-14 owner rule ("behaviour must fall out of where the word sits
  [in USVS]; no code path for negation") is in tension with this and with `_COORD_PRIME`/`MODIFIER 'not'`; the record contains no reconciliation.
- Roadmap intent (ROADMAP_LONG_TERM §3; README_pre_refactor l.108-111, 141): "composition over grounded primes (the central problem), clauses as
  operations (negation/antonymy as **signed operations**), not points"; held-out gate "composed-clause coordinates reconstruct entailment direction
  + negation flip on unseen clauses". **Never built**: no code in `src` or `archive` performs it (listed as open).

---------------------------------------------------------------------------------------------------------------------------------
## 6. Minimality (Q6)

| Piece | Finding (RECORD) |
|---|---|
| **M17.2 `basis_search.py`** (archive) | MDL basis on a 619-word closed vocab: 15 interpretable primitives promoted; MDL 45669 -> 36770, grounding 0.127 -> 0.250; at 10k the basis converges (act, having, relating, state, person, particular, substance, things, number, cause, make); M18.2 multi-signal steering **flat** (syn_cos 0.258->0.251) |
| **M19.1 `axes.dimensionality_spectrum`** (live, axes.py:121) | SVD of word x axis: 234 candidate axes -> intrinsic dim ~**46** (90% energy), effective (participation ratio) ~**17**. Eigenspectrum of the synonym-affinity was "degenerate"; the word x axis SVD is the "right matrix" |
| **M19.3 `minimality.py`** | variance-ranked: ~**30 named axes reproduce 95%** of fidelity (full 0.69; K=5 0.61, K=30 0.66). Minimal set = primes + lexname (SOMETHING, ONE, PART, BODY, lex:noun.act, WHEN, PEOPLE, KIND...). **Retracted by M20** |
| **M20.1 `honest_minimality.py`** | contribution-ranked (leave-one-out) on the normalized space: full 0.756; curve **peaks ~0.83 at K~80-120** then declines; "all 214 -> 0.756 -- the last ~100 axes hurt"; lexname block "not load-bearing as a block (drop -> no loss)". "The minimum-axes number it reports is honest even if larger than 30" |
| **SEMANTIC_MAPPING_PLAN** | "'Minimum redundancy' = the contribution-ranked minimality curve, **re-run after each signal lands**" (l.14-15). **I found no ledger entry that re-ran it** after M20.1 -- M28.1 reported only the ablation table, and M29 shipped 607 axes without a curve |
| **Pruning of attr axes** | Only the assembly-time filter `min_attribute_freq=2` (axes.py) ("frequency-filtered so noise drops out") was applied. Domain features then *added* ~370 axes (M28.1 win: random -0.032, hypernym_cos -0.006). No leave-one-out pruning of the 498 `attr:` axes, nor of the 44 `lex:` axes, was ever attempted on the 607-axis space. MEASURED-NOW (audit): 21 attr axes have 0 senses, 42 have <=50 |
| **Standing rule** | "More axes for their own sake: M20 closed door (the tail is noise)" (SIGNALS_AUDIT §4) -- **but** the shipped USVS is 607 axes vs the M20 peak of K~80-120 on 234; the record doesn't reconcile these (INFERENCE: domain axes were judged on random-overlap, a different metric) |

Verdict: the record's number for "how many independent axes are needed" is **~17 effective / ~46 for 90% energy / peak fidelity at ~80-120 named
axes (on a 234-axis space and for syn-vs-ant held-out fidelity)**; it was never re-measured for 607.

---------------------------------------------------------------------------------------------------------------------------------
## 7. Senses vs words (Q7)

- **Core words (9,946)**: one coordinate per *word* over 607 axes. Features (`axes.feature_matrix`): prime leaves of `naive_decompose(word)` which
  uses the **first WordNet sense's gloss only** ("first-sense bootstrap; WSD deferred", RN 1128), `attribute` dims (words with adjective->noun
  relation), `lex:` category of the word; attribute axes signed (§2); then propagation over synonym (22,856) + similar (3,988) + also_see (1,172)
  edges at alpha 0.7, 20 iters -- relations are at word level (union over senses), so a core coordinate is "a mixture over all its senses"
  (usvs.py `_mean_unit_coord` comment). Domain features add ~370 axes. Dense (median 560 non-zero axes).
- **Senses (117,659 + 20)**: `sense_usvs_weights` (usvs.py:240): genus component = ambiguity-discounted mean of unit core coords of the direct
  hypernyms' single-token lemmas (w 1.0) + gloss content words' coords (w 0.7), each scaled by confidence = mean(1/sqrt(n_senses)), top-24 axes kept
  (negatives dropped), max-normalized, plus its `lex:` axis = 1.0. The synset's own lemmas are deliberately excluded (62-proxy 50->44 when included).
  ~24 axes/sense, sparse CSR.
- **`prime_fallback_senses` = 1,208 (1.0%)** -- MEASURED-NOW reproduced exactly: senses where `sense_usvs_weights` returns `{}` because **no single-token
  hypernym lemma and no gloss content word is in the 9,946-word core**, so the code falls back to `sense_prime_weights` (gloss->prime decomposition,
  usvs.py:557). By POS: n 673, s 208, r 123, a 104, v 100. Typical: `abiding.s.01 "unceasing"`, `abrocoma.n.01 "abrocomes"`, `accurately.r.01 "with few
  mistakes"`, `acinonyx.n.01 "cheetahs"` (rare taxa / inflected-form glosses / stopword-only glosses).
- **`pure_interjection_senses` = 20** (usvs.py:124; D2 decision): ugh, ah, hey, ouch, phew, pooh, yuck, oops, hurray, argh, eek, gosh, yikes, shh, bah, hmm, huh, aha,
  whoa, eh have no WordNet synset; each is given a one-line gloss ("an exclamation expressing disgust or horror") and run through the same
  sense pipeline, minted as `interj.<word>.01` **after** the fingerprint is computed (so fingerprint-inert).
- Spanish lemmas (OMW) are additively merged into the lemma index only; grounding never reads them (usvs.py comment).
- `senses_of_surface` (usvs.py): function-tag tokens (`_FUNCTION_POS_TAGS`: AUX/DET/ADP/PART/PRON/CCONJ/SCONJ/PUNCT/NUM/SYM/INTJ) **never go through morphy** ("was ->
  wa -> washington.n.02 is a category error") and look up the raw surface only; empty -> caller emits `entity`. This is the code-level statement that
  "function words are grammar, not content" -- the closest thing to a recorded function-word policy in the lexical layer.

---------------------------------------------------------------------------------------------------------------------------------
## 8. Things a function-word / negation extension must NOT break (settled), and things left open

### 8.1 Settled (with evidence)
1. **Every axis named, interpretable; axes never rotate/mix; no opaque dimensions** (M19 hard invariant; SEMANTIC_MAPPING_PLAN "method constraints, do not
   relitigate"). New dimensions (NOT, known/new, tense...) must be named axes.
2. **Deterministic, learning-free placement by propagation over typed relational edges; no free per-word trained values** (M25: 0.56 vs 0.73; M21b ~0.48;
   closed doors). Hand-added special dimensions are compatible; training their values is the rejected path.
3. **M24 held-out rule**: any syn>ant / relatedness metric must exclude its scored pairs from propagation; production build uses all edges, quality only
   from the harness. A new axis must be judged held-out (the harness `ablation.py` already does; the 2026-11-14 note lowers validation bar for experiments).
4. **Antonymy = a signed relation consulted at comparison/composition time, not a position** (RN 1489-1494; SimLex +0.047 from edges). Do not try to
   encode antonyms as one sign-flipped axis on every word; the evidence is 5 failed attempts plus the 0.40->0.69 gain only from partial attr-axis anchoring.
5. **Tiered antonym edges keep provenance** (direct/satellite_head/satellite_satellite; default floor excludes satellite_satellite).
6. **Pole prime pairs GOOD/BAD, BIG/SMALL, MUCH_MANY/LITTLE_FEW stay as separate unsigned axes** in USVS (the signed collapse was never adopted); the audit relies
   on this too. Any change alters the fingerprint and invalidates `usvs_bridge`-style raw-607 consumers (chooser trained at d=607, RN M40).
7. **Senses are grounded by genus + differentia IN the space; the headword's own lemma is excluded; prime decomposition is fallback only** (M46, owner call).
8. **Senses are non-negative** (top-k sparsify drops negatives). Consumers (`encoder_model.py:236`, `usvs_graded.py:255`) rely on `sense_dense >= 0` and
   IDF weighting. Adding signed values to senses is a schema change.
9. **Pure interjections are post-fingerprint, additive** (D2) and the fingerprint is computed from axes+meta -- extension rows should be added the same way to
   keep the 607-axis fingerprint (and any pinned `usvs_meta.json`) stable, else rebuild + re-pin (`runs/ARTIFACTS.md` mentions artifacts).
10. **Operators are nodes on a reserved orthonormal role, not flags** (clause.py/meaning_graph.py comments; 0.99 deconvolution gate), truth adjectives are
    lossless tags, `MAYBE` is a first-class atom ("disjunction is MAYBE"), and "overwrite but don't forget" (never delete a refuted clause).
11. **`unit(clause_handle)` must be bound** (norm issue) and the encoder/hand-gold schema is frozen to `prime` = {YOU, I, UNK} + reserved role vocabulary
    (D3/D4): changing it requires gold rebuild (encoder v6 seeds 0.708 / 0.677).
12. **Function-tag tokens are never morphy-lemmatized** (usvs.py `senses_of_surface`).
13. **IDF weighting is the standing mitigation for ubiquitous axes** (SOMETHING/PART/ONE/WHEN) -- do not read raw cosines as meaningful on those.

### 8.2 Left open / explicit gaps (where the record says so)
1. Composition over grounded primes: "the central open problem" (§2; RN 1508-1516; README_pre_refactor l.141); negation/antonymy "as signed operations" at
   composition time **never built**. Held-out gate (negation-flip on unseen clauses) proposed, not run.
2. Antonymy as a first-class relational layer consulted at comparison time (post-grounding item C): store exists, **no live consumer**.
3. Graded truth / graded negation: **no position recorded** (only the learned scalar S_neg / decide_truth).
4. Which axes are load-bearing in the 607-axis space (minimality re-run) -- required by SEMANTIC_MAPPING_PLAN but not in ledger.
5. Grounding depth (M42's "#1 open item"); M44 conclusion: the prime whitelist, not query-time blending, is the lever (partly resolved by M46 parity).
6. Empty prime axes / function-word grounding: **not on any open-items list; discovered by the 2026-11-14 audit**. Cause = shared English `_STOPWORDS`.
7. Tense: no axis or field; D6 only says "carrier keeps tense/polarity". Possession (`'s`, MINE), question/known-ness, numerals scale: unaddressed.
8. Reconciling D3/D4 (reserved-block I/YOU; structural QUANTITY/ADDITIVE/FOCUS labels) with the 2026-11-14 "meaning lives in USVS only" rule (RN 5452 "NEXT awaiting owner").
9. `NSM canonical-list` uncertainties (DON'T WANT, (IS) MINE vs HAVE, allolexes) flagged `TODO(canonical-list)` in nsm_primes.py, never closed.
10. Lemma-level antonym tiers inherit sense conflation; resolution deferred to "the WSD layer" (M29 caveat).
11. Reader/encoder drop negation; "needs its own label" (RN 5452) -- conflicts with the owner rule against dedicated negation code paths.

---------------------------------------------------------------------------------------------------------------------------------
## 9. Reproduction notes (scratch only)
Scripts: `/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/{q,abl,cnt,fb}.py` (artifact stats; cue ablation; signed-word
count at 10k; fallback-sense census). Numbers reproduced today: negatives 0/2,785,213 in senses; 356,369 negative core entries, 6,670 words, 220 attr axes;
signed-word counts 130/414; ablation 0.693 -> 0.652; prime-fallback 1,208.
