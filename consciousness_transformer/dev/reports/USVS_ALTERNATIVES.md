# USVS content words: alternative ways to define the space (survey, 2026-11-15)

Research-and-options only. No code or data was changed. Nothing here is a recommendation; section 6 lists candidates without picking one.
Evidence tags: **[V]** = title/authors/venue/year and the one-line claim were checked against a search result or the publisher/arXiv/ACL page
today; **[V-s]** = bibliographic details checked but the specific claim only from secondary summaries/abstracts; **[U]** = unverified (listed again
in section 9). **MEASURED-NOW** = I computed it today with NLTK WordNet 3.0 in a scratch script (outside the repo); **RECORD** = what the project's own
reports say (cited as "history s2.3" = `USVS_DESIGN_HISTORY.md` section 2.3, etc.); **INFERENCE** = my reasoning.

---------------------------------------------------------------------------------------------------------------------------------
## 0. Executive summary (for the project owner)

1. The space works for "how alike are two words" (synonym AUC ~0.88, hypernymy ~0.72). It fails at two things the new scorer exposed: it cannot separate opposites, and it cannot cleanly separate unrelated words.
2. The causes are mechanical, not conceptual: three prime axes sit on ~98% of senses, senses carry no sign at all (0 negative values in 2.79M entries), and signs on core words depend on an English cue-word list.
3. The literature has two broad ideas for opposites: put each opposite pair on its own two-ended axis (semantic differential, SemAxis, POLAR), or keep opposition as a relation and consult it when comparing (what the project does now). Both are old; neither is free.
4. WordNet already contains the raw material for two-ended axes. MEASURED-NOW: 3,846 head adjective synsets carry a direct antonym pointer; they form ~2,000 head-to-head pairs; 14,532 of 18,156 adjective synsets (80%) sit in such bipolar clusters; the other ~3,600 are almost all "pertainym" adjectives (relational, like "atomic") with no opposite.
5. The catch is the number of axes: ~2,000 raw pair-axes is far from "minimum". Pair-axes would have to be merged (e.g., via WordNet's own `attribute` links: 620 adjective heads have one) or limited to a smaller core. Merging is the unresolved design question.
6. Verbs, nouns and adverbs barely have antonym pointers (996 verb synsets, 1,922 noun synsets, 630 adverb synsets). Two-ended axes alone will therefore cover adjectives well and everything else weakly.
7. Opposites come in different kinds (alive/dead cannot be "more or less"; buy/sell are two views of one event; tie/untie reverse an action). A single sign on a single axis fits only the gradable kind. This agrees with the owner's own caveat.
8. Taxonomy-based coordinates (is-a paths, information content) would help noun pairs like dog/cat vs. unrelated nouns, and verbs are shallow (mean depth 2.5), so eat/chew gets little. They give nothing for adjectives: 0 of 18,156 have a hypernym.
9. Human-rated norms (VAD, Lancaster sensorimotor, Binder 65 attributes, THINGS dimensions) give few, named, human-checked dimensions, and VAD is signed. But they cover words not senses, are mostly licensed for research only (non-redistribution for NRC VAD), and some exceed "derived from dictionaries" in a way only the owner can rule on.
10. Methods that inject antonymy into vectors (retrofitting, counter-fitting, Attract-Repel, SPINE, sparse coding) are the best-evidenced ones, but they learn per-word values or create unnamed dimensions: they fail requirements 2 and 3 as stated. They are useful as an evaluation baseline, not as the design.
11. How many dimensions does meaning need? Semantic differential: 3 (evaluation, potency, activity). Interpretable object similarity: 49 (THINGS). Binder: 65 hand-chosen. Lancaster: 11. Embeddings: intrinsic dimension ~10-30. The project's own estimate: ~17 effective, ~46 for 90% energy. A named, non-rotated set is necessarily redundant versus a rotated one, so "minimal" can realistically mean "no useless axes", not "the smallest possible rank".
12. Cheap first step with no downside: drop or down-weight the ubiquitous axes (IDF/pruning). It is reversible and is a precondition for any sign-based fix to show up in cosine.
13. The design that looks structurally closest to the owner's intuition is "typed hybrid": category axes (what kind of thing), scale axes (two-ended, signed, from WordNet clusters), and operator axes (closed-class only). It is also the most work.
14. The project has already tried: gloss-cue signs on attribute axes (kept, 0.40 -> 0.69), pole pairs on three prime axes (not adopted), per-word trained values (closed), sense-node propagation (closed). Pair-axes from WordNet clusters as a placement device were NOT tried (history s2.3, row "Antonym-pair axes").
15. Evaluation sets exist for exactly the weaknesses seen: SimLex-999 (similarity vs relatedness), WordSim-353 split, SimVerb-3500, HyperLex, BLESS, the Nguyen et al. antonym/synonym set (15,642 pairs), the GRE "most contrasting word" set. All are free to download for research; some licences are unstated.
16. Any change that makes senses signed breaks a recorded invariant (senses non-negative, history s8.1 item 8) and changes the pinned fingerprint; this should be an additive layer or a new version, not an edit.
17. Pole choice is the hard part of two-ended axes: for hot/cold either end can be "+". Orientation does not matter inside one axis but matters if axes are later merged; a data-derived rule (e.g., the more frequent member is the "+" end) exists but is a convention.
18. Cross-language: WordNet antonym pointers sit on lemmas, but head-to-head bipolar cluster membership sits on synsets, so synset-level opposition could transfer through sense identifiers (INFERENCE; needs a check on Open Multilingual Wordnet).
19. Five questions (section 7) decide most of this: is human-rated data acceptable, may one axis hold both a meaning position and be acted on by operator words, how many axes is "minimum", is a signed sense layer allowed, and what accuracy gain is worth a rebuild.
20. Suggested order, if the owner wants to proceed: run the read-only probes in section 8 first (a day of work, no build), then decide.

---------------------------------------------------------------------------------------------------------------------------------
## 1. Background: what the project has, tried and must not break (condensed)

RECORD (history report). 607 axes = 65 NSM primes + 498 `attr:` + 44 `lex:`; placement by label propagation over synonym/similar/also_see edges; core words (9,946)
have signed coordinates only on `attr:` axes (220 axes carry negatives); senses (117,659) are genus + gloss blends, top-24, non-negative. Antonymy is a tiered
edge store (11,317 pairs) with no placement role. Five prior attempts on antonyms (history s2.3): plain cosine (0.455), signed pole prime pairs and morphological
flips (0.466, not adopted), graph-aware penalty (0.638), anchored attr-axis sign + propagation (0.693, kept), Null-aware sparse bipolar words (0.48), per-word
contrastive training (0.48), joint training (0.56). Rules: every axis named; deterministic placement; no per-word trained values; no English lists in code; operator
dimensions only for closed-class words (owner, 2026-11-15).

Two facts shape every option below:
- **MEASURED-NOW (history s2.2 ablation):** removing the cue-word list costs only ~0.04 held-out syn>ant (0.693 -> 0.652). Signs are a modest, not a decisive, lever as currently done.
- **INFERENCE:** signed values cannot separate two words if most of each vector is the same generic mass. With SOMETHING/ONE/PART on ~98% of senses and cosine dominated by "genericity" (RN M42), a sign flip on one axis changes a small fraction of the dot product. Family A (de-emphasising ubiquitous axes) is therefore a precondition for B and C to show their effect.

WordNet facts used below (MEASURED-NOW, WordNet 3.0 via NLTK): 117,659 synsets (82,115 n, 13,767 v, 18,156 adj incl. 10,693 satellites, 3,621 adv); 3,992 lemma-level antonym pairs
(2,012 adjective, 1,076 noun, 549 verb, 355 adverb); 3,846 of 7,463 head adjective synsets have a lemma antonym; 1,999 distinct head-to-head antonymous pairs; 14,532 adjective
synsets (80%) are in bipolar clusters; 3,617 heads are outside any cluster, 3,560 of those pertainym-only; 940 synsets have an `attribute` pointer (620 adjective, 320 noun).

---------------------------------------------------------------------------------------------------------------------------------
## 2. Family sections

Requirement key: R1 minimum named set; R2 every axis named, no rotation; R3 deterministic, data-derived, no word lists in code, no trained per-word values; R4 cross-lingual via sense ids;
R5 opposites as opposite ends (as far as possible); R6 operator dimensions separate, closed-class only; R7 evidence-based, planned, improvements must be real.

### A. Keep the space; handle opposition and discrimination outside it

**How it works.** Leave coordinates alone. (a) Consult the antonym relation when comparing two nodes (what `score.py` already does: recorded opposites count 0). (b) Reweight axes at comparison time: IDF
or entropy weights so ubiquitous axes count little (already used by `usvs_graded`, M30). (c) Prune axes: drop axes that are on nearly all senses or on almost none, or rank by leave-one-out contribution (the
honest-minimality procedure of M20.1, never re-run on 607 axes, history s6).

**References.** Retrieval-style IDF is standard (no specific citation needed). Widdows (2003) "Orthogonal negation in vector spaces for modelling word-meanings and document retrieval", ACL 2003, pp. 136-143 [V] is the
classical "remove a meaning component" idea; Mohammad, Dorr, Hirst, Turney (2013) "Computing Lexical Contrast", Computational Linguistics 39(3):555-590 [V] reports that more than 90% of GRE "most contrasting word"
pairs are NOT listed as WordNet antonyms [V], i.e., a pointer store alone under-covers human-judged opposites.

**Data / open / multilingual.** All internal. IDF needs only the artifact. Antonym relation: WordNet lemma pointers (open; lemma-level, mostly not carried into other-language wordnets, INFERENCE).

**Dimensions / coverage.** 607 -> fewer if pruned (project record: honest fidelity peaked at ~80-120 axes on a 234-axis space, history s6; not measured on 607). Coverage of the opposition fix is exactly the pointer coverage: 3,992 lemma pairs,
~11k with tiers.

**Opposites / near-synonyms / unrelated / hypernyms / polysemy.** Opposites: separated only where a recorded edge exists, so happy/sad (not a recorded pair) stays ~0.85. Near-synonyms: unchanged. Unrelated: improves with IDF/pruning (M42: IDF 50/62 -> 53/62; M21's sparse
form took random similarity 0.32 -> 0.03). Hypernyms: unchanged. Polysemy: lemma-level edges inherit sense conflation (history s2.4, M29 caveat).

**Fit.** R1 partial (pruning moves toward it); R2 pass; R3 pass; R4 partial (edges are lemma-level); R5 fail (opposition stays a relation; this is the owner's intuition's opposite); R6 pass; R7 pass (lowest risk).

**History.** This is the current position (history s2.5, RN 1489-1494). New within it: actually re-running minimality on 607 axes, and applying IDF/pruning inside the stored vectors (not only at query time).

**Risks.** Leaves the owner's stated intuition unmet; relation store has no live consumer except the scorer; coverage ceiling (>90% of human contrast pairs are not WordNet antonyms [V]).

**Probe (read-only).** From `data/usvs/`: (1) compute for each axis its sense frequency and IDF; recompute the scorer's example pairs (dog/cat vs eat/chew, happy/unhappy/sad, unrelated pairs) with the top-k most ubiquitous axes zeroed, for k = 0, 3, 10, 30; (2) run the leave-one-out contribution ranking on the 607 axes using the held-out harness; (3) report how many of SimLex-999's lowest-scored pairs are WordNet antonym pairs.

### B. Signed attribute scales at the sense level

**How it works.** Carry the existing `attr:` sign down to senses and remove the cue-word list. Today only core words get signed `attr:` values (gloss-cue sign, then propagation); senses are rebuilt as non-negative blends. Variant B1: store sign in sense vectors for attribute axes (schema change). Variant B2: derive the sign from data
instead of cue words: for an antonym head pair that shares an attribute noun (e.g., hot/cold -> temperature) orient the two heads +/-; satellites inherit via `similar_to`; orientation fixed by a data convention (e.g., the member with the higher WordNet/SemCor tag count is "+", linguistic markedness) rather than words like "high"/"low".

**References.** Sheinman, Fellbaum, Julien, Schulam, Tokunaga (2013) "Large, huge or gigantic? Identifying and encoding intensity relations among adjectives in WordNet", Language Resources and Evaluation 47(3):797-816 [V] adds an intensity relation to WordNet dumbbells (ordered scales per attribute) mined from web patterns: relevant for graded position along a scale, but it is a separate resource and web-derived. Gross & Miller (1990) "Adjectives in WordNet", Int. J. of Lexicography 3(4) [V-s: authorship and issue verified; page claims not]: the head/satellite bipolar organisation. Kennedy & McNally (2005) "Scale structure, degree modification, and the semantics of gradable predicates", Language 81(2):345-381 [V]: scales may be open or closed; closed-scale adjectives (full/empty, open/closed) behave differently from open-scale ones (tall/short).

**Data / open / multilingual.** WordNet `attribute`, `antonym`, `similar_to` (open). Sense-level sign is carried by synset identity, so the sign layer itself is language-neutral once synset-level (not lemma-level) opposition is used (INFERENCE). Tag counts for orientation come from SemCor counts shipped with WordNet (English only; INFERENCE that other languages would need another convention or none, since orientation is arbitrary).

**Dimensions / coverage.** No new axes: the existing 498 `attr:` axes (21 have 0 senses, 42 <=50 senses, history s6). Coverage: only adjectives with an `attribute` pointer (620 heads, 940 synsets incl. nouns) plus their satellites via `similar_to`; MEASURED-NOW (history s2.2) only 414 of 9,946 core words have an attribute relation at all. Verbs/nouns: essentially none.

**Opposites.** Pairs that share an attribute noun get opposite signs: hot/cold, big/small if WordNet links them. happy/unhappy: WordNet's `attribute` for happy is not standard (INFERENCE: check; many evaluative adjectives have no attribute pointer, which is why quality/good/bad was special-cased by prime pole axes). Near-synonyms: same sign, high cosine. Unrelated: unchanged, since sign adds no discrimination there. Hypernyms: unchanged. Polysemy: sign is per sense, which is an improvement over today's per-word sign.

**Fit.** R1 pass (no new axes); R2 pass; R3 pass for B2 (orientation convention must be stated), fail-ish for B1 if cue words stay; R4 partial/pass; R5 partial (only attribute-linked adjectives); R6 pass; R7 pass (small, testable).

**History.** Extends M19.2 (anchored sign, 0.693 held-out) and removes M18.1's cue lists. Different from what failed: M18.1/M21 collapsed three prime pole axes or operated per word; here the sign is per sense on existing attribute axes and propagated by existing edges. Honest expectation (history s2.2): the cue list was worth only ~0.04; a better sign source cannot be expected to give a big gain on its own.

**Risks.** Breaks "senses are non-negative" (history s8.1 #8) for every consumer (`encoder_model.py:236`, `usvs_graded.py:255`); fingerprint change; attribute coverage is only ~8% of words / 35% of adjectives (RN 1224); orientation convention; sign vs magnitude confusion on non-gradable pairs.

**Probe.** Count, over WordNet: adjective head pairs that share an `attribute` noun (hot/cold-type) vs. those that do not (good/bad, happy/unhappy-type); fraction of the 3,846 antonym heads reachable; orientation rule consistency (does the tag-count rule give the same sign for the same attribute across independent pairs?); then compute the sign agreement with the existing gloss-cue signs for the 130 words that have them (a free consistency check: if the data rule recovers the cue-list signs, the list can go).

### C. Antonym-pair scales as dimensions (semantic differential family)

**How it works.** Each antonym pair (or each WordNet adjective cluster) defines a two-ended axis; other words are placed by their relation to the two poles (nearer pole, signed projection). Three concrete versions exist in the literature:
- **Semantic differential** (Osgood, Suci, Tannenbaum 1957, *The Measurement of Meaning*, Univ. of Illinois Press [V]): people rate concepts on bipolar adjective scales; factor analysis repeatedly yields three general factors, evaluation, potency, activity [V-s via the 1975 cross-cultural book and encyclopedia summaries; the 1957 book's own chapter on dimensionality (p. 31) was not read]. Osgood, May, Miron (1975), *Cross-Cultural Universals of Affective Meaning* [V-s].
- **SemAxis** (An, Kwak, Ahn, ACL 2018, "SemAxis: A Lightweight Framework to Characterize Domain-Specific Word Semantics Beyond Sentiment") [V]: an axis is the difference between the average vectors of two pole word sets; 732 axes built from ConceptNet antonym pairs; a word is scored by cosine with the axis. FrameAxis (Kwak, An, Jing, Ahn, PeerJ Computer Science 7:e644, 2021) [V]: antonym-pair "microframes" give bias and intensity of a text on each axis. Both sit on pretrained embeddings: the poles are named, the geometry underneath is learned.
- **POLAR** (Mathew, Sikdar, Lemmerich, Strohmaier, WWW 2020, pp. 1548-1558) [V]: transforms existing embeddings into a space whose dimensions are polar-opposite pairs taken from WordNet-style sources; post-processing, interpretable by construction, performance preserved per the authors [V from abstract].
- **WordNet adjective organisation** (above): head synsets with direct antonym pointers, satellites attached by similar_to; "dumbbell" structure; relational (pertainym) adjectives are outside it [V-s].
- **Conceptual spaces** (Gärdenfors 2000, *Conceptual Spaces: The Geometry of Thought*, MIT Press [V-s]): quality dimensions grouped into domains, some with natural zero or two ends; this supplies the "domain + scale" language used in H.

**Data / open / multilingual.** WordNet (open); ConceptNet 5.5 (Speer, Chin, Havasi, AAAI 2017 [V]; open, multilingual, antonym edges). For WordNet-only construction (no embeddings) the data are internal and deterministic. Cross-lingual: bipolar cluster membership is synset-level (INFERENCE: transferable), lemma antonym pointers are not (INFERENCE).

**Dimensions / coverage (MEASURED-NOW).** Raw: 1,999 head-head antonymous pairs (a ceiling on pair-axes from adjectives); 3,992 lemma antonym pairs overall. Adjective coverage 80% of adjective synsets (14,532/18,156) lie in clusters; adjectives are ~15% of all synsets (nouns 70%, verbs 12%, adverbs 3%), so
coverage of the whole sense inventory is ~12% by direct cluster membership plus whatever propagation adds. Verb/noun/adverb antonym pointers: 996 / 1,922 / 630 synsets. Merging pair-axes into fewer dimensions is the open step (candidates: merge by shared attribute noun; by `also_see`/similar links between clusters; by lexicographer file; each is a data rule, each needs checking).

**Opposites / synonyms / unrelated / hypernyms / polysemy.** Opposites: separated by construction on their shared axis (opposite signs) and by nothing else. Near-synonyms: same side of the same axis. Unrelated words: placed on different axes, so cosine ~0 (good for the random-pair floor) provided no ubiquitous dimension shares mass. Hypernyms: no help (adjectives have no hypernyms). Polysemy: a word that sits in two clusters (hot = temperature and = spicy/stolen/popular) gets mass on two axes; that is right at sense level and wrong at word level (project's core coords are word-level).
**Known failure modes.** (1) Words related to both ends (e.g., "lukewarm", "fine", "okay"): near the axis origin or on a neighbour axis; pole assignment by "nearer pole" is unstable there. (2) Non-gradable/complementary opposites (alive/dead, open/shut): the axis exists but intermediate values are meaningless (Kennedy & McNally closed scales [V]). (3) Converses (buy/sell, give/receive, husband/wife) and reversives (tie/untie, rise/fall): not two ends of a scale but two perspectives or a direction of change; Cruse (1986) *Lexical Semantics* [V-s] distinguishes complementaries, antonyms (polar, overlapping, equipollent), directional opposites (antipodals, counterparts, reversives) and relational opposites/converses. (4) Multiple opposites for one word ("big" vs "little" vs "small" vs "tiny"): handled by clusters, not pairs. (5) Embedding-based versions (SemAxis, POLAR) inherit all corpus artifacts of the embedding, and a recorded observation about contrast measures: pole words co-occur, so distributional axes can put antonyms close (Mohammad et al. 2013 [V] motivates thesaurus-structure use for this reason).

**Fit.** R1 fail/partial (raw ~2,000 axes unless merged); R2 pass (named by their pole words) for WordNet-only construction, partial for SemAxis/POLAR (named but over a learned basis); R3 pass for WordNet-only, fail for POLAR/SemAxis (pretrained embeddings are trained); R4 partial; R5 pass, within gradable adjectives only; R6 pass (operators stay separate; but see owner question O2); R7 partial: untested as a placement device here.

**History.** Not tried as a placement device (history s2.3). Closest relatives that failed or were not adopted: M18.1 signed poles on three prime axes (0.466), M21 words as signed values on shared axes with Null mask (0.48), M25 trained values. What would be different: one axis PER cluster (not shared), signs from WordNet pointers (not cue words), values assigned to senses, ubiquitous axes removed so the sign is not drowned.

**Risks.** Axis-count explosion; incompatible with the "minimal set" goal unless merged; each merge is a hand-judgment unless purely data-driven; poor coverage outside adjectives; loses the clean prime-basis story (primes become one family among many); interactions with M20's finding that "the last ~100 axes hurt".

**Probe.** From WordNet only: (1) build the 1,999 head-pair graph; merge by attribute noun and by also_see/similar edges; report the number of connected components, the size distribution, and how many axes cover 90% of in-cluster adjective senses (a power-law tail would suggest a small core); (2) for each cluster report whether both poles are populated and how many satellites are on each side; (3) list the 100 largest clusters and hand-inspect 20 for coherence (happy/unhappy, good/bad, hot/cold, fast/slow...); (4) simulate: signed placement of adjective senses on cluster axes, then measure syn>ant AUC on the Nguyen et al. set restricted to adjectives and on SimLex adjectives.

### D. Feature-norm and property-based interpretable spaces

**How it works.** Use human-rated attributes as the named dimensions: each word/concept gets a vector of rating values on a small set of named properties. Sources:
- **McRae, Cree, Seidenberg, McNorgan (2005)**, Behavior Research Methods 37(4):547-559 [V]: feature production norms, 541 concrete concepts, ~725 participants [V]; a later featural semantic space of 2,526 dimensions is described in secondary sources [V-s]. Too small for a lexicon, and features are free-text properties, not a compact named set.
- **Binder et al. (2016)**, "Toward a brain-based componential semantic representation", Cognitive Neuropsychology 33(3-4):130-174 [V]: ~65 experiential attributes (sensory, motor, spatial, temporal, affective, social, cognitive) rated for 535 English words [V-s; the lab page lists "65-dimensional attribute ratings for 535 English words", another paper says 68 features: unresolved]. Open download from the lab page [V-s]. From memory several attributes split poles into separate unsigned scales (e.g., large vs small, fast vs slow, pleasant vs unpleasant) [U]; if true this is the same pattern as NSM's BIG and SMALL axes.
- **Lancaster Sensorimotor Norms** (Lynott, Connell, Brysbaert, Brand, Carney 2020), Behavior Research Methods 52(3):1271-1291 [V]: 39,707 concepts rated on six perceptual modalities (touch, hearing, smell, taste, vision, interoception) and five action effectors (11 dimensions); data on OSF; licence stated inconsistently across repositories (CC BY vs CC BY-NC) [V-s; check the OSF page]. Strengths, not directions: unsigned.
- **Affective norms:** Warriner, Kuperman, Brysbaert (2013) 13,915 lemmas, Behavior Research Methods 45(4):1191-1207 [V]; Mohammad (2018) NRC VAD, 20,000 words, ACL 2018 pp. 174-184 [V]; NRC VAD v2: >55,000 English terms and machine translations into ~108 languages, free for research, no redistribution, translations sometimes wrong [V from the project page via search]. Valence is signed (negative to positive): the only family in D where happy/unhappy fall on opposite ends of a named dimension by data.
- **THINGS / SPoSE:** Hebart, Zheng, Pereira, Baker (2020), Nature Human Behaviour 4(11):1173-1185 [V]: 49 human-interpretable, non-negative dimensions from 1.46M odd-one-out triplets over 1,854 objects [V-s for the count of triplets and the name SPoSE, which the abstract search did not state]. Data on OSF (licence not checked) [V-s].
- **Conceptual spaces** (Gärdenfors 2000 [V-s]) gives the philosophy (quality dimensions with domains) rather than a dataset.

**Data / open / multilingual.** Human-rated, per word (not per sense), almost all English; NRC VAD has automatic translations; Lancaster has some non-English counterparts (not checked, [U]). Licences range from open to research-only. Whether ratings count as "data" under R3 is owner question O1.

**Dimensions / coverage.** 3 (VAD), 11 (Lancaster), ~65 (Binder), 49 (THINGS) vs 607 now. Coverage: Lancaster/VAD tens of thousands of words but not senses and not closed-class words; Binder/McRae/THINGS hundreds to ~1,800 words. To cover 117,659 senses the values would have to be propagated through WordNet exactly like current attr values (deterministic), which is R3-compatible but adds a propagation step on rated seeds. Rated seeds are not senses: each word-level rating would have to be assigned to a sense (first sense? all senses? both are approximations, INFERENCE).

**Opposites / synonyms / unrelated / hypernyms / polysemy.** Opposites: separated on valence (VAD) for evaluative pairs only; hot/cold, big/small, fast/slow not separated unless the attribute set has directional scales (Lancaster: no, strength only). Near-synonyms: close. Unrelated: well separated on perceptual/affective profile for concrete words, poorly for abstract. Hypernyms: only through shared profile. Polysemy: word-level ratings blur senses.

**Fit.** R1 pass (few dims) -- the strongest family on R1; R2 pass (dimension names are the rating scales); R3 partial (data-derived and deterministic once propagated, but human-rated rather than from a dictionary: O1; no trained values); R4 partial/fail (rated in one language; translations automatic); R5 partial (valence only, plus whichever other signed scales the norm set provides); R6 pass; R7 partial: uncertain gain for abstract words and function-adjacent vocabulary.

**History.** Not tried. The project's own `attr:` axes are the WordNet analogue of Binder-style properties, but data-derived. Valence has appeared only as GOOD/BAD prime axes and as the owner-removed appraisal node (D2).

**Risks.** Licences (non-redistribution of NRC VAD would prevent shipping coordinates derived from it, INFERENCE from the stated terms); small coverage of abstract/relational vocabulary; word-vs-sense mismatch; adds an English-centric human data dependency to an otherwise lexical pipeline.

**Probe.** Download-free first: compute the overlap between each norm's word list and the 9,946 core words and 117,659 senses' lemmas (counts only). Then correlate each rated dimension with the existing USVS axes on the overlap (is "valence" already recoverable from `attr:quality` and GOOD/BAD? if cosine/regression R^2 is high, the dimension is redundant, if low, it is new information). Report which words have rating but no WordNet sense (fixes cannot propagate).

### E. Taxonomy-derived coordinates

**How it works.** Place a sense by its position in the is-a hierarchy. Similarity measures are classical: Resnik (1995), "Using information content to evaluate semantic similarity in a taxonomy", IJCAI-95 pp. 448-452 [V] (similarity = information content of the most specific common ancestor; reported r = 0.79 with WordNet + Brown frequencies per a secondary summary [V-s]); Lin (1998), "An information-theoretic definition of similarity", ICML pp. 296-304 [V; formula not read]; Wu & Palmer (1994), "Verb semantics and lexical selection", ACL 32, pp. 133-138 [V] (depth of the lowest common subsumer); Seco, Veale, Hayes (2004), intrinsic information content from structure alone, ECAI pp. 1089-1090 [V]. These are comparison measures, not coordinates. To make coordinates, one uses ancestor indicators (a sense is a vector over the named ancestor synsets at chosen depths) or "domain axes" such as the project's own 44 `lex:` categories.
Learned geometric versions: Poincare embeddings (Nickel & Kiela, NeurIPS 2017) [V] and box embeddings / box lattice (Vilnis, Li, Murty, McCallum, ACL 2018, pp. 263-272) [V]. Both are trained and produce unnamed dimensions (the box version has box corners, not named ones): R2 and R3 fail as stated. (A deterministic hyperbolic placement of a tree exists in the literature, Sarkar 2011 [U]; it would still have unnamed coordinates.)

**Data / open / multilingual.** WordNet hypernym graph (open); other wordnets via synset ids (OMW: Bond & Foster 2013, "Linking and extending an open multilingual wordnet", ACL pp. 1352-1362 [V]: >26 languages, >2M senses, open licences). Hypernym structure transfers by synset id, which is a real cross-lingual advantage for this family.

**Dimensions / coverage (MEASURED-NOW).** Distinct noun ancestors at hierarchy depth index 2/3/4/5: 26 / 254 / 2,274 / 8,523 (so a "category axes at depth 3" set would be ~250 named axes; depth 4 already exceeds the whole current space). Coverage: nouns and verbs. Adjectives: 0 of 18,156 have a hypernym; adverbs none either. Verbs are shallow: 13,767 synsets, mean maximal depth 2.5, 559 root-level verb synsets. Example scores (Wu-Palmer, MEASURED-NOW): dog/cat 0.857; eat/chew 0.20 (eat.v.01 depth 2, chew.v.01 depth 5), so a plain path measure also makes dog/cat far closer than eat/chew: the weakness the scorer found would persist for verbs under this family.

**Opposites / synonyms / unrelated / hypernyms / polysemy.** Opposites: not represented; antonyms such as hot/cold are co-hyponyms-or-unrelated in the tree. Near-synonyms: same node. Unrelated: well separated for nouns (distant subtrees). Hypernyms: represented explicitly (the strength of this family: HyperLex/BLESS-type tests). Polysemy: handled by sense nodes (project closed this door as the primary space, M22-23: "sense-node propagation barely beats chance" for the close-edge version; that was propagation, not ancestor coordinates, so not the same experiment).

**Fit.** R1 partial (depth-3 cut = ~250 axes; deeper cuts blow up); R2 pass for ancestor axes (each axis is an actual synset), fail for learned embeddings; R3 pass for ancestor/IC, fail for Poincare/box; R4 pass (synset ids); R5 fail (no scales); R6 pass; R7 partial.

**History.** Hypernymy as a relation: 1,546 directed genus edges; M28.1: "Hypernymy, like antonymy, is directional structure -- belongs in the relational store, not closeness"; genus-differentia as a close edge: hypernym +0.010 but similar -0.021 and random +0.022 (negative). Domain features (lexicographer-like) added ~370 axes and helped random-pair separation (M28.1). What would be different: ancestors as axes (not propagation edges), chosen depth cut, and using this ONLY for nouns/verbs inside a typed hybrid (H).

**Risks.** Depth is not comparable across branches (WordNet is uneven); verbs are flat; no adjective/adverb story; ancestor axes are highly collinear (each ancestor implies its parent), so the effective dimension is not the axis count; extra axes again conflict with R1 and with "the last ~100 axes hurt".

**Probe.** From WordNet: for the 9,946 core words' first senses, compute the ancestor-set cosine at depths 3-5 and compare to the artifact's cosine on the scorer's examples and on the SimLex noun and verb subsets; report the fraction of core words with no ancestor information (adjectives, adverbs, function-adjacent); report effective rank of the ancestor matrix (participation ratio, as in the project's M19.1) to estimate how many independent axes it truly adds.

### F. Decompositional approaches (NSM explications, frames, qualia, ConceptNet)

**How it works.** Define meaning through explicit decomposition into smaller named parts.
- **NSM** (Wierzbicka; Goddard): 65 primes [V-s] plus "semantic molecules" [V-s: Goddard & Wierzbicka, *Words and Meanings*, OUP 2014; Goddard 2012/2016 on molecules]: non-primitive meanings (e.g., hands, water, long, children) used as building blocks in explications. The project already uses primes as seed axes. For gradable adjectives and antonyms I could NOT verify how NSM explicates them: the searches showed good/bad and big/small are primes (so their opposition is built into the prime set), and I found no source on NSM treatment of non-prime opposites (e.g., hot/cold, alive/dead) [U].
- **FrameNet** (Baker, Fillmore, Lowe, "The Berkeley FrameNet Project", COLING-ACL 1998, pp. 86-90) [V]: frames with named frame elements; a frame is a named, interpretable unit, but there are hundreds to ~1,200 frames, each organised around situations, not scales. [count U]
- **VerbNet** (Kipper-Schuler 2005, PhD thesis, Univ. of Pennsylvania) [V-s; thesis not retrieved]: verb classes (Levin-based) with thematic roles and mappings to WordNet, PropBank and FrameNet [V-s]. A class membership axis for verbs (a named category set) complements WordNet's shallow verb hierarchy.
- **Pustejovsky qualia** (*The Generative Lexicon*, MIT Press 1995) [V-s]: four roles (formal, constitutive, telic, agentive) per noun; a typed structure, not a set of scales, and no resource with broad coverage that is open and uniform [U for any such resource].
- **ConceptNet** (Speer, Chin, Havasi 2017 [V]): typed relations (IsA, HasProperty, Antonym, UsedFor, ...), multilingual; edges are crowd/derived and noisy.

**Data / open / multilingual.** FrameNet, VerbNet, ConceptNet open; frames have been built for other languages in separate projects [U]; ConceptNet is multilingual by construction.

**Dimensions / coverage.** Primes: 65 (already in). Frames: hundreds; VerbNet: a few hundred classes [U]; ConceptNet relation types: ~34 [U] but edges not axes. None gives a broad-coverage set of scale dimensions. Coverage of frames over WordNet senses: partial via the FrameNet-WordNet mappings [U].

**Opposites etc.** Frames/VerbNet: group verbs by situation, with no handling of opposites except where a frame has "Opposite"-type relations (not verified) [U]. NSM: opposition is partly primitive-level (GOOD/BAD, BIG/SMALL are primes) and otherwise requires explications; the explicit use of semantic molecules for adjectives was not found.

**Fit.** R1 partial; R2 pass for NSM/frames; R3 partial (explications are hand-written; hand-written explications are language resources, but writing them for 117k senses is not data-derived); R4 partial (NSM is explicitly cross-linguistic: its main selling point; frames partly); R5 partial (only primes); R6 pass; R7 low for the cost.

**History.** The project began here (M17 decomposed glosses to primes, then M46 "USVS is our Semantic Metalanguage, we don't need a vestigial lossy one" -- adopted on architectural grounds). Gloss-to-prime decomposition is now the OOV fallback (1,208 senses, 1.0%). FrameNet/VerbNet/qualia were not tried.

**Risks.** Hand effort scale; coverage; frames are not scales; verbs only for VerbNet.

**Probe.** Compute coverage of VerbNet classes and FrameNet LUs over the project's verb senses (counts only; if VerbNet class membership covers >80% of the 13,767 verb synsets via its WordNet mappings it could become a verb category axis set, otherwise skip); read the NSM publications on adjective/antonym explications and note exactly which opposites are primitives vs defined (closing the [U] above).

### G. Post-processing methods that inject antonymy/synonymy into vectors

**How it works.** Start from distributional vectors; move them so that linked words get closer and antonyms further.
- **Retrofitting** (Faruqui, Dodge, Jauhar, Dyer, Hovy, Smith, NAACL 2015, pp. 1606-1615) [V]: nudges vectors of lexicon-linked words together (WordNet, FrameNet, PPDB); synonyms/attract only.
- **Counter-fitting** (Mrksic et al., NAACL 2016, pp. 142-148) [V]: injects antonymy (repel) and synonymy (attract) constraints; reported new state of the art on SimLex-999 at the time [V].
- **Attract-Repel** (Mrksic, Vulic, O Seaghdha, Leviant, Reichart, Gasic, Korhonen, Young; TACL 5:309-324, 2017, titled "Semantic Specialization of Distributional Word Vector Spaces using Monolingual and Cross-Lingual Constraints") [V]: monolingual and cross-lingual constraints; six languages.
- **Lexical-contrast embeddings:** Ono, Miwa, Sasaki, NAACL 2015 pp. 984-989 [V bibliographic only]; Nguyen, Schulte im Walde, Vu, ACL 2016 short, pp. 454-459 [V]; AntSynNET (Nguyen et al., EACL 2017 pp. 76-85) [V].
- **Sparse/interpretable recodings:** Sparse overcomplete vectors, "SPOWV" per the task (Faruqui, Tsvetkov, Yogatama, Dyer, Smith, ACL 2015, pp. 1491-1500) [V]; SPINE (Subramanian et al., AAAI 2018) [V]; Non-negative sparse embedding NNSE (Murphy, Talukdar, Mitchell, COLING 2012, pp. 1933-1950) [V]. These produce sparse, partly human-interpretable (via intrusion tests) but unnamed dimensions. Interpretability scoring by categories: Senel et al. (2018), "Semantic structure and interpretability of word embeddings", IEEE/ACM TASLP 26(10):1769-1779, with the SEMCAT set of >6,500 words in 110 categories [V]. Hollis & Westbury (2016), Psychon. Bull. Rev. 23(6):1744-1756 [V bibliographic]: principal components of embeddings correlate with affect (valence/arousal/dominance) [V-s].

**Data / open / multilingual.** Need a trained embedding as input; constraints from WordNet/PPDB (open). Attract-Repel is explicitly cross-lingual.

**Dimensions.** Input dimensionality (e.g., 300) or a larger sparse space (SPINE/SPOWV thousands). Axes unnamed.

**Opposites etc.** Counter-fitting/Attract-Repel are the best-evidenced for separating antonyms from synonyms while keeping relatedness; SimLex gains are documented. Sparse recodings do not address antonymy.

**Fit.** R1 fail/partial; R2 fail (dimensions unnamed, rotated); R3 fail (trained embedding, learned per-word values, M25 reproduces this failure mode: "more training = worse"); R4 pass for Attract-Repel; R5 partial; R6 pass; R7 well-evidenced in the literature but evidence is on embeddings, not on this space.

**History.** Closest relatives: M21b/M25 (trained per-word values; closed doors). The M30 result that adding the signed antonym store lifted SimLex from 0.380 to 0.417 puts this project's two-part design near word2vec's ~0.41-0.44 range (RECORD); published counter-fitting numbers are higher but are on strong embeddings and unverified here [U for numbers].

**What could still be used.** Deterministic pieces: IDF-like reweighting; constraints as a data source; Attract-Repel/counter-fitting as an external baseline for evaluation only (what score do trained methods get on the same antonym/synonym sets as a ceiling?). SEMCAT-style category intrusion as an interpretability check for any new axis set (named categories in, check each axis). I found no method in this family that is deterministic and yields named axes.

**Risks.** Would reintroduce what the project closed (trained per-word values; M25).

**Probe.** None needed for the design. For calibration: run an off-the-shelf counter-fitted vector set (if obtainable offline) on the project's evaluation harness to get an external ceiling for the syn>ant metric.

### H. Hybrid designs: typed dimensions, per-POS subspaces, domain + scale

**How it works.** Treat the space as several typed blocks rather than one flat basis: (1) category/domain axes (what kind of thing: from lexicographer files, a depth-cut of the noun hierarchy, VerbNet classes for verbs), (2) scale axes (two-ended, signed: from adjective clusters, `attribute` links, possibly affect/sensorimotor scales), (3) operator axes (negation, degree, known/new, tense: closed-class words only, already built for function words, 614 axes with the seven extension axes), (4) optionally relational/role axes (GOAL, SOURCE). Similarity is then block-wise (e.g., domain match gate x scale agreement) or a weighted sum, and each block can have its own normalisation (e.g., IDF on categories, none on scales). Gärdenfors' domains/dimensions [V-s] and NSM's primes-plus-molecules layering are precedents for typed structure. Per-POS subspaces: adjectives on scales, nouns on categories+properties, verbs on classes+event structure, adverbs inherit from the adjective they derive from (derivational pointers; INFERENCE).

**Data.** Entirely WordNet-derivable for blocks (1) and (2) (lexname, hypernym ancestors, adjective clusters, attribute), VerbNet optional. All synset-id based, hence transferable to other wordnets (INFERENCE; subject to the lemma-level caveat on antonym pointers).

**Dimensions / coverage.** Category block ~44 (lex) to ~250 (depth-3 ancestors); scale block from ~hundreds (after merging) to ~2,000 (raw); operator block ~15-20. Plausibly 300-600 named axes in total, but with far less overlap (each sense uses few axes in each block), which is the intent: low effective dimension per word, not a small global axis count.

**Opposites etc.** Opposites: same category block, opposite signs in the scale block (happy/unhappy: same domain, opposite ends). Near-synonyms: same category, same side, close. Unrelated: different category block and/or different scales. Hypernyms: category block, with scale values inherited down. Polysemy: per sense. Non-gradable, converse, reversive opposites: still not covered by scales; remain relations (or get typed "complementary" scales with two values only; INFERENCE).

**Fit.** R1 partial (many axes, few active per word; "minimum" must be defined, O3); R2 pass; R3 pass (all derived from WordNet); R4 partial-pass; R5 partial (gradable adjectives, plus any signed scales found for nouns/verbs); R6 pass (operators typed and separate by construction, which also eases R6 concerns about conflating dimension types); R7 low-risk only if built additively behind a version flag.

**History.** The current space already mixes types in one block (primes, `attr:`, `lex:`), with cosine over all of them (history s0). M28.1 domain features helped; M20 showed the `lex:` block "not load-bearing as a block". Typed blocks with separate comparison rules were not tried; the owner's 2026-11-15 operator decisions already separate a fourth type.

**Risks.** Largest build; comparison rule becomes a design surface (two-stage, weights); the learned encoder/decoder and the memory model currently consume one flat vector (`sense_dense`), so the interface changes.

**Probe.** Take the five blocks as separate matrices over the same senses, measure each block's contribution on the scorer's examples (dog/cat, eat/chew, happy/unhappy/sad, unrelated) and on SimLex-999 by POS (block-wise cosine and a few simple combiners), without changing the artifact; report which combiner makes opposites and unrelated pairs behave. This costs reading the artifact, building scratch matrices, and no repo edit.

---------------------------------------------------------------------------------------------------------------------------------
## 3. How many dimensions does lexical meaning need? What can "minimal" mean?

- **Semantic differential:** 3 factors (evaluation, potency, activity) recur across scales, concepts and (per Osgood, May, Miron 1975) cultures [V-s]. This is the lowest-dimensional claim and it is about affective/connotative meaning, not denotation: it cannot distinguish dog from cat.
- **Hebart et al. (2020):** 49 reproducible, human-nameable dimensions explain most explainable variance in similarity judgements of 1,854 objects [V]. Object concepts only.
- **Binder et al. (2016):** ~65 hand-chosen experiential attributes [V]; **Lancaster:** 11 sensorimotor dimensions [V]; **VAD:** 3.
- **Feature norms:** McRae norms need thousands of distinct features (2,526 reported) to cover 541 concepts [V-s]: the property space is large and sparse.
- **Embeddings:** intrinsic dimension estimates for 300-d GloVe/word2vec of roughly 10-30 using TwoNN (GloVe ~24.8, word2vec ~24.75, fastText ~13.2; Kataiwa, Cho, Ohki 2025, arXiv:2503.02142) [V-s; single estimator, preprint]. Principal components of skip-gram embeddings align with valence/arousal/dominance, frequency and age of acquisition (Hollis & Westbury 2016 [V-s]).
- **This project:** effective dimension ~17 (participation ratio), ~46 for 90% of energy (M19.1), fidelity peaking at K ~ 80-120 named axes on the 234-axis space (M20.1); never re-measured on 607 (history s6).
- **MDS of similarity judgements:** the classical result for concrete categories is a handful to a few dozen dimensions; the Hebart 49 is the modern version of this [V]. I did not retrieve a specific older MDS paper [U].

**What "minimal" can realistically mean.** (a) Affect only: 3. (b) Object similarity as people judge it: ~50. (c) Distinctions a word-sense inventory needs (kind of thing + properties + scales): hundreds. The effective dimension of a named basis is always lower than its axis count because named axes cannot be rotated to remove redundancy (R2): so the practical target is not "smallest rank" but "every axis earns its place": pruned by leave-one-out contribution (M20.1 method), with no axis that is on ~every sense. That reading is compatible with all three families the owner's requirements favour (A, B, H).

---------------------------------------------------------------------------------------------------------------------------------
## 4. Decision table (families vs the owner's requirements)

P = pass, ~ = partial, F = fail. Cost and risk are my relative estimates (INFERENCE); reversibility = ease of backing out (additive layer vs rebuild of the 607 base + fingerprint).

| Family | R1 minimum | R2 named | R3 determ. / data | R4 cross-lingual | R5 opposites as ends | R6 operators separate | Cost | Risk | Reversible |
|---|---|---|---|---|---|---|---|---|---|
| A. keep + IDF/prune/relation | ~ | P | P | ~ (lemma edges) | F (kept as relation) | P | low | low | yes |
| B. signed sense-level attr scales | P | P | P (B2) / F (cue list) | ~ | ~ (attr-linked adjectives only) | P | low-med | med (sense schema, fingerprint) | partly (additive layer) |
| C. antonym-pair / cluster axes (WordNet-only) | F raw ~2,000 / ~ merged | P | P | ~ (synset-level) | P (gradable adjectives) | P | med-high | med-high | additive block: yes |
| C'. SemAxis/POLAR over embeddings | ~ | ~ (named, learned basis) | F | ~ | P | P | med | med | yes |
| D. rated norms (VAD, sensorimotor, Binder, THINGS) | P | P | ~ (human data; O1) | F-~ | ~ (valence only) | P | med | med (licences, coverage) | yes |
| E. taxonomy coordinates (ancestors/IC) | ~ | P (ancestor axes) / F (hyperbolic, box) | P / F | P (synset ids) | F | P | low-med | low-med | yes |
| F. NSM/frames/VerbNet/qualia | ~ | P | ~ (hand explications) | ~ | ~ (primes only) | P | high | med | n/a |
| G. retrofit/counter-fit/Attract-Repel/SPINE | F | F | F | P (Attract-Repel) | ~ | P | med | high (closed door, M25) | yes |
| H. typed hybrid (categories + scales + operators) | ~ (see O3) | P | P | ~-P | ~-P | P | high | med | only if additive |

---------------------------------------------------------------------------------------------------------------------------------
## 5. Evaluation sets for the observed weaknesses

| Weakness | Set | What it tests | Open? | Status |
|---|---|---|---|---|
| similarity vs relatedness (incl. antonyms low) | **SimLex-999** (Hill, Reichart, Korhonen, Computational Linguistics 41(4):665-695, 2015) [V] | similarity not association; ~500 raters, adj/verb/noun, 999 pairs; antonym handling detail [U] | free download; licence not checked | project already uses it (M30, rho 0.333 -> 0.417) |
| relatedness | **WordSim-353** (Finkelstein et al., ACM TOIS 20(1):116-131, 2002) [V]; split into similarity and relatedness by Agirre et al. (NAACL 2009, pp. 19-27) [V] | mixed similarity+relatedness; the split isolates them | CC BY 4.0 per the dataset page [V-s] | standard, noun-heavy, older |
| relatedness (comparative ratings) | **MEN** (Bruni, Tran, Baroni, JAIR 49:1-47, 2014) [V] | relatedness, 50-point comparative score | on the author page [V-s] | concrete nouns mostly |
| verb similarity | **SimVerb-3500** (Gerz, Vulic, Hill, Reichart, Korhonen, EMNLP 2016, pp. 2173-2182) [V] | 3,500 verb pairs, covers VerbNet classes | arXiv/ACL [V] | relevant to eat/chew |
| hypernymy (graded) | **HyperLex** (Vulic, Gerz, Kiela, Hill, Korhonen, Computational Linguistics 43(4):781-835, 2017) [V] | 2,616 pairs graded entailment | on the first author's page [V-s] | tests the hypernym AUC (0.72) |
| hypernymy / co-hyponymy / meronymy / random | **BLESS** (Baroni & Lenci, GEMS 2011, pp. 1-10) [V] | typed relations plus controlled random pairs; no synonymy/antonymy [V-s] | public | tests unrelated-word separation on nouns |
| antonym vs synonym | **Nguyen et al. set** (Nguyen, Schulte im Walde, Vu, ACL 2016 and EACL 2017) [V]; 15,642 balanced pairs (7,816 syn, 7,826 ant) across adj, verb, noun; WordNet + Wordnik; Stuttgart dataset page [V-s] | the exact discrimination the scorer flags | open download (licence not checked) | best match for the antonym weakness; note overlap with WordNet pointers (see caveat below) |
| contrast / opposites | **GRE "most contrasting word"** (Mohammad, Dorr, Hirst, Turney 2013 [V]; earlier Mohammad, Dorr, Hirst 2008, EMNLP pp. 982-991 [V]) | picking the most contrasting option among five | availability not confirmed [U] | tests beyond-WordNet opposites (>90% of its pairs are not WordNet antonyms [V]) |
| lexical contrast | **Lexical Contrast Dataset** (Stuttgart, listed on the same page) [V-s] | antonym-synonym with POS split | open | alternative to the above |
| "LZQZ" and "Turney" antonym sets | not identified by title/author; searches returned nothing [U] | -- | -- | ask the owner for the exact reference |

Caveat for any WordNet-derived set: if the antonym test pairs come from WordNet and the placement uses WordNet antonym pointers, the metric must hold out the scored pairs (house rule M24, `train_pairs=`). The GRE set and SimLex are better external checks.

---------------------------------------------------------------------------------------------------------------------------------
## 6. The most promising candidates (stated neutrally; no winner)

**Candidate 1: Clean up and keep (A, plus the minimality re-run).** Prune/re-weight the ubiquitous axes inside the stored space, re-run honest minimality on 607 axes, keep antonymy as a relation, make the scorer's antonym consult sense-level and complete.
*Right choice if:* (a) zeroing the top-k ubiquitous axes already moves unrelated-pair similarity down and makes dog/cat vs eat/chew behave (probe A1); (b) the antonym store covers the pairs the scorer meets (probe A3: most scorer-relevant opposites are in the store); (c) the owner holds that opposition is a relation (as the history concludes) and R5 is a preference, not a rule.

**Candidate 2: Signed senses on the existing attribute scales, signs from WordNet pointers (B2).** Additive signed layer or v2 of the sense matrix; data-derived orientation; no new axes.
*Right choice if:* (a) the data rule reproduces the 130 cue-list signs and extends signs to many more adjective senses (probe B1); (b) removing the generic axes (Candidate 1) shows that sign flips then move antonym cosines substantially (otherwise the 0.04-sized effect of the cue list repeats); (c) the owner accepts signed senses (O4) and that most evaluative opposites (good/bad, happy/unhappy) may lack `attribute` pointers and so remain relation-only.

**Candidate 3: WordNet-cluster scale axes, merged (C, WordNet-only).** One two-ended axis per merged adjective cluster, senses placed with signs from head/satellite membership, built as an extra named block next to the existing one.
*Right choice if:* (a) merging the 1,999 head pairs yields a few hundred coherent axes that cover most in-cluster senses (probe C1; a long tail of tiny clusters would argue against); (b) held-out adjective antonym discrimination clearly beats Candidates 1-2 (probe C4); (c) the owner allows a flat count of several hundred scale axes as long as each sense activates few; (d) pole orientation convention and complementary/converse exceptions are acceptable as documented limits.

**Candidate 4: Typed hybrid with rated or taxonomy scales where WordNet has none (H, with E and/or D).** Blocks for categories, scales, operators; rated norms (e.g., valence) as the signed scale for evaluative vocabulary that lacks `attribute` pointers; taxonomy ancestors only for nouns/verbs.
*Right choice if:* (a) block-wise scratch comparison (probe H1) shows a combiner that fixes opposites, unrelated pairs and dog/cat vs eat/chew simultaneously, which no single block does; (b) the owner accepts human-rated data (O1) and a licence that blocks redistribution of derived coordinates or finds a permissively licensed substitute; (c) the interface change (blocks instead of one vector) is acceptable to the encoder, decoder and memory model, which currently consume one flat vector.

Excluded as primary candidates: G (fails R2/R3 by its construction and repeats closed doors M21b/M25), F in full (hand-explication scale), and learned geometric versions of E (R2/R3).

---------------------------------------------------------------------------------------------------------------------------------
## 7. Questions only the owner can answer (8)

1. **O1. Is hand-curated human rating data acceptable as "data"?** (VAD, Lancaster, Binder, THINGS are human-rated lexical norms, not dictionaries, mostly English, some research-only licensed.) If not, Family D is out; if yes, which licences are acceptable?
2. **O2. May a two-ended scale carry a meaning position AND be acted on by operator words?** (E.g., "very happy" scaling the happy end and "not happy" acting on it, while "unhappy" is the other end.) Your 2026-11-15 rule keeps operators on closed-class-only dimensions; does that also forbid an operator word from multiplying a meaning dimension at composition time?
3. **O3. What does "minimum" mean here?** Fewest axes in total (e.g., ~50-100, as the project's own curve and Hebart's 49 suggest), or fewest active axes per word (e.g., 3-5 on a larger named set, as in the typed hybrid)? The two lead to different families.
4. **O4. Is a signed sense layer acceptable?** It breaks the recorded "senses are non-negative" invariant and changes the artifact fingerprint; additive v2 file with the old one kept, or in-place?
5. **O5. How should unrelated-word separation and opposite separation be weighed?** If only one can be improved in the next step, which matters more for the tree scorer and the encoder/decoder pair?
6. **O6. How far should non-gradable opposites (alive/dead), converses (buy/sell) and reversives (tie/untie) be expected to come out of positions?** Is it acceptable that these stay relations while gradable opposites become scale ends?
7. **O7. Orientation of a scale (which end is "+")**: acceptable as an arbitrary but data-derived convention (e.g., unmarked/more frequent member first), or must it carry meaning (e.g., positive = presence, negative = absence)?
8. **O8. Is the current fingerprint / 607-axis base frozen?** (GOALS.md says "settled and not being revisited".) Any option beyond Candidate 1 reopens it; which version policy applies to the trained encoder (v6 seeds 0.708/0.677) and the chooser trained at d=607 (history s8.1 #6)?

---------------------------------------------------------------------------------------------------------------------------------
## 8. Cheap read-only feasibility probes worth running (ranked by information per hour)

All read the existing artifact (`data/usvs/`) and/or WordNet; none builds or changes anything.
1. **P1 (A):** zero the top-k most ubiquitous axes (k = 0, 3, 10, 30) on stored senses; recompute the scorer's dog/cat, eat/chew, happy/unhappy/sad, random-pair examples and SimLex-999 rho. (Answers: how much of the weakness is just generic mass.)
2. **P2 (A):** leave-one-out contribution ranking on the 607 axes (re-run M20.1); report how many axes can be dropped without loss.
3. **P3 (B):** can a WordNet-derived orientation rule reproduce the 130 cue-list signs? Count adjective heads reachable via attribute vs not (good/bad, happy/unhappy-type).
4. **P4 (C):** head-pair graph, merge by attribute/also_see, report component count and size distribution, coverage curve of senses by number of axes (MEASURED-NOW already: 1,999 head pairs, 14,532/18,156 adjective synsets in clusters).
5. **P5 (C/B):** simulate signed cluster placement for adjective senses and score adjective antonym vs synonym AUC on the Nguyen set (held-out) and SimLex adjectives.
6. **P6 (E):** ancestor-set cosine at depths 3-5 for noun/verb senses; effective rank; compare on SimLex nouns/verbs and SimVerb-3500.
7. **P7 (D):** vocabulary overlap counts of VAD/Lancaster/Binder/THINGS word lists with the core words and senses; regress each rated dimension on the existing axes (redundant or new?).
8. **P8 (H):** block-wise comparison scratch matrices and a few combiners.
9. **P9 (cross-lingual):** check in Open Multilingual Wordnet whether antonym and similar_to pointers exist for a second language (e.g., Spanish, which the project already loads) and whether synset-level cluster structure survives.

---------------------------------------------------------------------------------------------------------------------------------
## 9. Items I could not verify (tagged [U] or [V-s] above)

- Osgood et al. (1957) factor labels (evaluation, potency, activity): confirmed only through encyclopedia/secondary summaries and the 1975 cross-cultural book listing; the 1957 chapter itself was not read. The "30 cultures / 620 words" figure came from a Wikipedia-derived page and is not relied on.
- Gross & Miller (1990) page numbers and exact claims; Fellbaum's later WordNet book chapter on adjectives (not searched).
- Lin (1998) formula and results; Resnik's r values (secondary).
- How NSM explicates gradable adjectives and non-prime antonyms: not found.
- "LZQZ" antonym/synonym set and a "Turney" antonym set: could not identify; GRE set availability.
- Binder et al.: 65 vs 68 attributes; that several attributes split poles into separate scales (from memory).
- Lancaster Norms licence (CC BY vs CC BY-NC vs CC BY-NC-SA, per repository); non-English versions.
- THINGS/Hebart: licence; the name SPoSE and the triplet count came from secondary pages; the PLOS ONE 2019 reference was not found.
- NRC VAD licence details beyond the project page summary (research use free; no redistribution; translation quality).
- McRae feature count (2,526) and participant number are from secondary sources; norms licence not checked.
- SimLex-999's exact antonym-handling instructions; MEN pair count; SimLex/MEN/HyperLex/BLESS download licences.
- Kipper-Schuler (2005) thesis text; FrameNet frame count and VerbNet class count; FrameNet-WordNet mapping coverage; ConceptNet relation count (~34).
- Counter-fitting and Attract-Repel published SimLex scores (not read).
- Sarkar (2011) deterministic hyperbolic tree embedding (named from memory only).
- Kataiwa et al. 2025 intrinsic-dimension numbers: preprint, single estimator, from search snippets; any MDS-of-similarity classical results.
- Whether OMW carries antonym/similar_to pointers across languages (INFERENCE in this report).
- Chen et al. ACL 2015 (contrast embeddings, 92% F on GRE): seen only as a citation in a search snippet.
