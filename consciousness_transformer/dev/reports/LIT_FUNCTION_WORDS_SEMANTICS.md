# Function words, negation and grammatical meaning: what the literature says

Literature review, 2026-11-15. Web-verified where marked. Verification key:
**[V]** = bibliographic details confirmed in at least one search result;
**[V-abs]** = the claim I cite was confirmed from an abstract or a primary page;
**[V-2nd]** = claim confirmed only from a secondary source (course notes, summary page);
**[unverified]** = could not confirm; do not rely on it.
Nothing below is cited from memory alone. Web queries used linguistics/NLP terms only.

## Executive summary (for the project owner)

1. **H1 (function words are a closed class definable in the space, with a few special dimensions): PARTLY supported.** Formal semantics, NSM and typology all agree function words are few and systematic. But formal semantics treats them as operators over meanings (types), not as points, so "a dimension on the node" captures degree and feature-like meaning (polarity, definiteness, tense) but not scope or binding.
2. **The decisive structural fact:** what the literature calls "function-word meaning" splits into (a) feature-like meaning (polarity, definiteness, tense, number, person) and (b) relational/procedural meaning (connectives, quantifier scope, wh-binding, anaphora). (a) fits dimensions. (b) needs the tree, which you already have.
3. **H2 (negation as a graded 0..1 dimension): PARTLY supported, and one dimension alone will break.** Evidence for gradedness is real: negators shift rather than flip sentiment (Zhu et al. 2014), "not cold" sits nearer "cold" than "hot" (Aina et al. 2018), and not-A is a mitigated, intermediate meaning.
4. **What breaks a single negation dimension:** (i) scope ("every student did not fail"; AMR itself is criticised for one flat :polarity), (ii) double negation ("not unhappy" is not "happy", Hermann et al. 2013), (iii) contrary vs contradictory ("unhappy" vs "not happy", Horn; Krifka disputes it), (iv) degree words interact with negation non-additively (Kiritchenko and Mohammad 2016: parts' polarities do not reliably predict the phrase).
5. **Negation as operator/matrix: supported theoretically (Montague/GQ, Coecke et al.) and in small experiments** (Rimell et al. 2017 learn bilinear adjective negators; Hermann et al. 2013), but a learned antonym-map is empirically the wrong target, because not-A is not the antonym (Aina et al. 2018, abstract).
6. **Reconciliation exists and is the recommended design (Option C below):** keep an operator NODE in the tree (carries scope and nesting), whose effect on its argument is a displacement along a few named dimensions (carries degree). One generic composition rule, no word-specific code.
7. **Probing evidence that negation is a linear direction in neural nets: weak or unverified.** I found papers on a linear "truth direction" and on negation mechanisms in LMs, but could not confirm a paper showing "negation = one direction". I do not rely on it.
8. **H3 (other function words fall out of the space): PARTLY supported.** Definite/indefinite as KNOWN vs NEW matches Heim/Kamp (novelty vs familiarity), i.e. a memory-lookup instruction; wh-words as "unknown" variables match Hamblin/Karttunen and indeterminate-pronoun work; tense as relations among time points (Reichenbach) is feature-like only if you keep three time slots.
9. **Connectives (because/if/and/but) are procedural/relational** (Blakemore, Wilson and Sperber): best as predicate nodes with two clause-valued roles, whose own point sits on NSM BECAUSE / IF dimensions, not as a dimension on a clause.
10. **Numbers (from ONE and TWO): only partly.** NSM (Goddard 2009) says low numbers (1-3ish) explicate from primes, but larger numbers and the open-ended sequence depend on COUNTING as a procedure. Embedding studies show magnitude is recoverable but fragile (Wallace et al. 2019).
11. **H4 (same treatment across languages): SUPPORTED with a caveat.** Typology (Dryer WALS; Miestamo 2005) shows negation is always expressed by some morpheme (particle, affix, auxiliary), never by word order alone; so "a morpheme fills the NEG dimension" is cross-linguistically safe. Caveat: double negation / negative concord and negative indefinites vary, so the operator must be able to appear more than once per clause without compounding.
12. **Ready-made inventories exist:** UD features (Polarity, Tense, Definite, PronType, Number, Poss...) and UniMorph (23 dimensions, 212+ features) are good checklists for which special dimensions to define. UD's Polarity is binary (Neg, sometimes Pos) and so is AMR's :polarity.
13. **Existing meaning representations have no graded negation.** AMR: :polarity -. UMR: modal-strength values with polarity (graded epistemic strength, but verified only second-hand). MRS: negation as a scopal predicate. Nobody I verified encodes "slightly negated" as such.
14. **The "unhappy" case:** the literature supports treating un- as contradictory-plus-pragmatic-strengthening (Krifka 2007) OR as contrary (Horn); both agree not-A and un-A differ in how strongly they are pushed toward the antonym. A signed A-axis plus a NEG-degree dimension can represent both readings without new machinery.
15. **Bottom line:** adopt hybrid C: few named dimensions (NEG, KNOWN, WH/UNKNOWN, time-relation, POSS, QUANT) for graded content, plus ordinary tree nodes for operators and relations, so scope and nesting live in the tree and degree lives in the space.

---

## 1. Formal semantics: function words as operators

**What the main lines say.**
- Montague grammar interprets natural language by a model-theoretic semantics with a systematic syntax-semantics relation. Proper treatment of quantification (Montague 1973) is the source of treating determiners and noun phrases as higher-type objects. [V-2nd: confirmed only via the SEP overview; I did not retrieve the 1973 text.] Janssen and Zimmermann, "Montague Semantics", Stanford Encyclopedia of Philosophy (rev. 2025). https://plato.stanford.edu/entries/montague-semantics/ [V-abs]
- Generalized quantifier theory: Barwise and Cooper (1981), "Generalized Quantifiers and Natural Language", Linguistics and Philosophy 4(2):159-219, doi:10.1007/BF00350139. [V] Determiners (every, some, no, most, the, a) denote relations between sets; a noun phrase denotes a set of sets. Function words are operators, content words supply the sets.
- Type-driven composition: Heim and Kratzer (1998), Semantics in Generative Grammar, Blackwell. [V] Three composition operations: function application, predicate modification and a conjunctive operation (from the lecture slides found: https://www.grace.umd.edu/~pietro/research/slides/FormAndComposition.pdf [V-2nd]).
- Polarity: Ladusaw (1980), Polarity Sensitivity as Inherent Scope Relations, Garland. [V] Negative polarity items are licensed in downward-entailing contexts; downward entailment is a property of functions over sets, not of whole sentences. The same search noted that the purely semantic version of NPI licensing is now widely considered untenable [V-2nd].

**Which function words are standardly operators and why.** Negation (a function from propositions/properties to their complement), quantifiers and determiners (relations between sets), connectives (functions on propositions), tense operators or tense-as-variable (see Q4), question operators (see Q4), degree morphemes (functions on gradable predicates, see Q3). They are treated as functions because they have no stable referent and their effect is the same whatever the content word is: they change how content is combined. Their contribution is invariant across the arguments they take, which is exactly why they are "closed class".

**Evidence for/against hypotheses.** Supports H1's "closed class, small number" part. Against "defined as a point": in this tradition a function word has a TYPE (a function), and the tradition has no analogue of "position on dimensions". Scope (which operator applies to which sub-tree) is a first-class phenomenon.

**Takeaway.** Keep operators as tree nodes; give them a point for graded content.

---

## 2. Vector-space and distributional treatments, incl. negation

### 2a. Compositional models with matrices/tensors
- Baroni and Zamparelli (2010), "Nouns are Vectors, Adjectives are Matrices: Representing Adjective-Noun Constructions in Semantic Space", EMNLP 2010, pp. 1183-1193. https://aclanthology.org/D10-1115/ [V-abs] Nouns are vectors, adjectives are learned linear maps; the model beat rivals at reconstructing unseen adjective-noun vectors.
- Socher, Huval, Manning and Ng (2012), "Semantic Compositionality through Recursive Matrix-Vector Spaces", EMNLP 2012, pp. 1201-1211. https://aclanthology.org/D12-1110/ [V-abs] Each parse-tree node has a vector (meaning) and a matrix (how it modifies neighbours); evaluated on sentiment of adverb-adjective pairs, among others.
- Coecke, Sadrzadeh and Clark (2010/2011), "Mathematical Foundations for a Compositional Distributional Model of Meaning", Linguistic Analysis 36:345-384; arXiv:1003.4394. https://arxiv.org/abs/1003.4394 [V-abs] Word meanings are vectors/tensors; grammatical types (pregroup) determine tensor order; restricting scalars to Booleans gives a Montague-style semantics. Function words become tensors of higher order.
- Socher et al. (2013), "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank", EMNLP 2013. https://aclanthology.org/D13-1170.pdf [V; negation-tracking claim V-2nd]
- Kruszewski, Paperno and Baroni (2015), "Deriving Boolean structures from distributional vectors", TACL 3:375-388. https://aclanthology.org/Q15-1027/ [V-abs] Maps distributional vectors into a Boolean structure where entailment is inclusion; relevant if "not" should be a Boolean complement.

### 2b. Negation in vector spaces
- Widdows (2003), "Orthogonal Negation in Vector Spaces for Modelling Word-Meanings and Document Retrieval", ACL 2003, pp. 136-143. https://aclanthology.org/P03-1018/ [V-abs] Negation as orthogonality (project the query away from the negated term). Works for removing unwanted word senses ("web NOT internet"). This is "not" = remove, not "not" = oppose.
- Hermann, Grefenstette and Blunsom (2013), ""Not not bad" is not "bad": A distributional account of negation", Workshop on Continuous Vector Space Models and their Compositionality, arXiv:1306.2158. https://arxiv.org/abs/1306.2158 [V-abs] Argues current models lack logical operations; proposes a three-part representation (I did not verify what the three parts are) and concludes non-linearities should be removed so that negation can be captured. Takeaway: double negation must be representable and must not collapse to the original.
- Zhu, Guo, Mohammad and Kiritchenko (2014), "An Empirical Study on the Effect of Negation Words on Sentiment", ACL 2014 (P14-1029). https://aclanthology.org/P14-1029 [V-abs] Compares a reversing hypothesis (negation flips sign) with a shifting hypothesis (negation changes the value by a constant amount); negators have different shifting power on positive vs negative phrases. This is the most direct empirical evidence for graded negation: **shifting beats pure reversal** is the paper's framing; I did not verify the exact numbers.
- Kiritchenko and Mohammad (2016), "Sentiment Composition of Words with Opposing Polarities", NAACL 2016. https://aclanthology.org/N16-1128/ [V-abs] For most phrases, knowing constituent polarity and POS is not enough to predict the phrase's sentiment. Evidence that composition with a single polarity number is lossy.
- Rimell, Mabona, Bulat and Kiela (2017), "Learning to Negate Adjectives with Bilinear Models", EACL 2017, pp. 71-78. https://aclanthology.org/E17-2012/ [V-abs] Learns a mapping from an adjective to its antonym in embedding space; class-conditional bilinear nets do best, and giving the model the semantic domain (e.g. a centroid of temperature words) helps. An operator with a domain parameter, i.e. the operator needs to know which axis is relevant.
- Pham, Lazaridou and Baroni (2015), "A Multitask Objective to Inject Lexical Contrast into Distributional Semantics", ACL 2015 (P15-2004). [V-abs] Reports promising results on a negation operator mapping adjectives to antonyms.
- Aina, Bernardi and Fernandez (2018), "A distributional study of negated adjectives and antonyms", CLiC-it 2018. https://staff.fnwi.uva.nl/r.fernandezrovira/papers/2018/AinaEtal-clicit2018.pdf [V-abs, read the abstract and introduction text] Finding: a negated adjective (not cold) is typically more similar to the adjective (cold) than to its antonym (hot); the effect is weaker for affixal antonyms (happy/unhappy). It also states that modelling negation as a map from adjective to antonym is a simplification made in some computational work (naming Pham et al. 2015 and Rimell et al. 2017), and supports the mitigation hypothesis (not large is intermediate between large and its antonym).
- Conversational negation: I did not find a distinct paper under that name in these searches; the closest verified concept is the mitigation/intermediate-meaning account above (Aina et al. citing Jespersen, Horn 1972, Giora 2006) [V-2nd]. Treat "conversational negation" as [unverified] as a labelled line of work.

### 2c. Negation/polarity as a DIMENSION
- Sentiment-style polarity is the standard case of a polarity dimension (Zhu et al. above: negation shifts a polarity value; Socher et al. 2013: tree models track negation scope).
- Osgood, Suci and Tannenbaum (1957), The Measurement of Meaning, University of Illinois Press. https://www.press.uillinois.edu/books/?id=p745393 [V] The semantic differential: bipolar adjective scales (sweet/sour, good/bad) whose factor analysis repeatedly yields evaluation, potency and activity, cross-culturally [V-2nd]. This is the strongest precedent for signed interpretable axes, and it is for antonym poles, not for negation.
- Probing: Marks and Tegmark, "The Geometry of Truth", arXiv:2310.06824 [V (title/arXiv id), content V-2nd]: LLMs linearly represent truth/falsehood of statements. Zhou, Zhou, Jia and May (2026), "How Language Models Process Negation", arXiv:2605.03052 [V-abs]: late-layer attention shortcuts cause errors; ablating them helps; both a suppressive and a constructive mechanism operate (the constructive one, building a representation of the whole negative phrase, is more prominent) in Mistral-7B and Llama-3.1-8B. **The abstract does not claim a negation direction.** A search summary said hidden states become linearly separable along a "not" direction, but I could not confirm this from the primary page: [unverified]. I found no verified paper showing that negation itself is one linear direction.

### Which has better support?
- Negation-as-dimension (graded shift of a polarity/feature value): supported by Zhu 2014 (shifting), Aina 2018 (not-A stays near A), and Osgood's bipolar axes. Fails where the phrase has structure (Kiritchenko 2016) and for double negation (Hermann 2013).
- Negation-as-operator: supported theoretically (Q1) and by small learned models (Rimell 2017), but the target "antonym" is wrong per Aina 2018; operator models need a domain argument (Rimell).
- **They reconcile.** An operator that moves a point along a negation direction is just an operator whose matrix is a rank-one displacement plus identity. Rimell's bilinear model with a domain vector is exactly "which axis to push along". So: operator in the tree, displacement in the space.

**Takeaway.** Gradedness is real; one global scalar is not enough; scope and nesting need tree structure; the operator should be generic and its parameters should be dimensions.

---

## 3. Graded negation, antonymy, degree

- **Contrary vs contradictory.** Horn, A Natural History of Negation (1989; reissued CSLI 2001). [V as a book; claims via 2nd-hand sources] Treats happy/unhappy as contraries (a gap between them); not-happy is contradictory in form but is pragmatically strengthened toward the contrary. Krifka (2007), "Negated antonyms: Creating and filling the gap", in Sauerland and Stateva (eds.), Presupposition and Implicature in Compositional Semantics, Palgrave, pp. 163-177 [V]: argues antonyms are literally contradictory and the gap arises from pragmatic strengthening, applied when the negated form is more complex than its antonym [V-2nd: based on summaries, not the chapter].
- **Negative strengthening is lexically variable.** Gotzner, Solt and Benz (2018), "Scalar Diversity, Negative Strengthening, and Adjectival Semantics", Frontiers in Psychology 9:1659, doi:10.3389/fpsyg.2018.01659 [V-abs]: scale structure (type of gradable adjective) affects how likely scalar implicature and negative strengthening are. Ruytenbeek et al. 2017 (Glossa) tests Horn vs Krifka [V-2nd, found via the Krifka search; I did not open it].
- **Degree semantics for gradable adjectives.** Kennedy (2007), "Vagueness and Grammar: The Semantics of Relative and Absolute Gradable Adjectives", Linguistics and Philosophy 30(1):1-45 [V-abs]: adjectives map entities to degrees on scales; relative adjectives use scales without endpoints, absolute adjectives use scales with endpoints (closed scales); the standard of comparison is selected by an economy principle for absolutes. Implication: "negating an adjective" operates on a SCALE with a standard, so graded negation depends on the adjective's scale type (open/closed), not only on the negation word.
- **Degree words + negation.** I found no verified source that formalises "not very happy" / "slightly unhappy" in a single-dimension model. Kiritchenko and Mohammad (2016) is the nearest evidence that composition is not additive. Marked as a gap.
- **Antonyms are distributionally similar.** Pham et al. (2015) [V-abs]: distributional models struggle to separate antonyms from synonyms because both occur in similar contexts. Mrkšić et al. (2016), "Counter-fitting Word Vectors to Linguistic Constraints", NAACL 2016 [V-abs]: post-process vectors with synonym/antonym constraints. Mohammad, Dorr and Hirst (2008), "Computing Word-Pair Antonymy", EMNLP 2008, pp. 982-991, https://aclanthology.org/D08-1103/ [V-abs]. Schwartz, Reichart and Rappoport (2015), "Symmetric Pattern Based Word Embeddings for Improved Word Similarity Prediction", CoNLL 2015, pp. 258-267 [V-abs]. Takeaway: antonyms share all dimensions except the one(s) on which they are opposed, so an interpretable space should make that one axis explicit, which is what a signed dimension does.
- **Interpretable polar axes.** Osgood et al. (1957) [V]; Gardenfors, Conceptual Spaces: The Geometry of Thought (MIT Press, 2000) and The Geometry of Meaning: Semantics Based on Conceptual Spaces (MIT Press, 2014) [V as books; search found no function-word analysis in them, so they are precedent for geometric word-class semantics, covering nouns, adjectives, verbs and prepositions, not for function words].
- **Structure of antonymy (gradable/complementary/converse/reversive).** I did not verify a primary source for this typology within this session (Cruse and Lyons are the usual references): [unverified]. The project owner's remark that antonymy is more complicated than one dimension is consistent with everything above: at least three behaviours exist (gradable pairs on a scale, complementary pairs with no middle, converses that swap roles), and only the first is a "signed dimension".

**Does a single graded negation dimension capture slightly/very/not + adjective?** Partly. It captures "not" shifting toward the antonym (Zhu; Aina). It does not capture: not-very vs very-not; a closed-scale adjective such as "full" where not-full is nearly complementary; double negation; or the un-/not difference without extra information. What breaks is composition order, which is a tree property.

**Takeaway.** Use a signed axis for the adjective's own scale and a separate NEG degree for the operator's push; let the tree encode order.

---

## 4. Function words as procedures: dynamic semantics, relevance theory, tense, variables

- **Definite vs indefinite.** Kamp (1981) DRT and Heim (1982) File Change Semantics: indefinites introduce a new discourse referent; definites require a familiar one; both are updates on a stored discourse state. Source: Stanford Encyclopedia, "Discourse Representation Theory" https://plato.stanford.edu/entries/discourse-representation-theory/ [V-abs for the DRT summary; V-2nd for Heim's novelty/familiarity wording, which the SEP search did not quote]. Implication: "the = known, a = new" (hypothesis H3) is almost exactly the Heim/Kamp novelty-familiarity condition, and in this project it maps to a memory operation (lookup vs write), a strong fit.
- **Procedural meaning for connectives.** Blakemore (1987), Semantic Constraints on Relevance, Blackwell [V]; Wilson and Sperber (1993), "Linguistic Form and Relevance", Lingua 90:1-25 [V]. They distinguish conceptual from procedural encoding (instructions on how to process) and apply the latter to discourse connectives and particles [V-2nd: the page snippets confirm the distinction and its extension to discourse particles, not the specific example of "so"]. Implication: because/if/but are instructions about how to relate two contents, not concepts.
- **Tense.** Reichenbach (1947), Elements of Symbolic Logic, section 51 "The Tenses of Verbs", pp. 287-298 [V-2nd]: tense relates speech time S, event time E and reference time R. Partee (1973), "Some structural analogies between tenses and pronouns in English" [V-2nd]: tenses behave like pronouns (deictic, anaphoric, bindable). Bjorkman, "Some structural disanalogies between pronouns and tenses" (Canadian Journal of Linguistics, Cambridge) argues most of the analogies can be explained otherwise [V-2nd]. Implication: tense is relations among time points. As dimensions: three time slots plus an order relation; the "operator" view (past as an operator on propositions) and the "variable" view both exist, and the variable view is easier to put in a graph.
- **Pronouns as variables.** Partee 1973 and DRT (discourse referents) [V-2nd]. The project's "names are variables with no meaning point" matches.
- **Wh-words as variables/unknowns.** Hamblin (1973), "Questions in Montague English", Foundations of Language 10:41-53 [V]: a question denotes the set of propositions that answer it. Karttunen (1977), "Syntax and semantics of questions", Linguistics and Philosophy 1(1):3-44, doi:10.1007/BF00351935 [V]: denotation is the set of true answers. Kratzer and Shimoyama (2002), "Indeterminate Pronouns: The View from Japanese" [V-2nd]: indeterminate pronouns denote sets of alternatives and receive existential, universal, interrogative, NPI or free-choice readings depending on the operator they associate with. Haspelmath (1997), Indefinite Pronouns, OUP [V-2nd]: implicational semantic map of indefinite functions.
  **Direct fit with H3:** "who = someone + not-known" is, in effect, the Hamblin/Kratzer-Shimoyama view: the wh-word is the same indeterminate as someone, and what makes it a question is the operator or context it sits under. It is evidence AGAINST giving who a separate hand-made class.

**Takeaway.** Several H3 meanings are best described as instructions over memory/discourse state (known/new, unknown, bound variable). The project's memory architecture (read/remember/recall) is a natural home; a dimension is the "flag", the memory operation is the "procedure".

---

## 5. Natural Semantic Metalanguage

- Primes: 65 (current), grouped (English exponents, from the Wikipedia table [V-2nd; consistent with Goddard and Wierzbicka 2014, Words and Meanings, OUP]): substantives (I, you, someone, people, something/thing, body); relational substantives (kind, part); determiners (this, the same, other/else/another); quantifiers (one, two, some, all, much/many, little/few); evaluators (good, bad); descriptors (big, small); mental predicates (think, know, want, don't want, feel, see, hear); speech (say, words, true); actions/events (do, happen, move); existence and possession (be somewhere, there is, be someone/something, is mine); life/death (live, die); time (when/time, now, before, after, a long time, a short time, for some time, moment); space (where/place, here, above, below, far, near, side, inside, touch); **logical concepts (not, maybe, can, because, if)**; **intensifier/augmentor (very, more)**; similarity (like/as/way). https://en.wikipedia.org/wiki/Natural_semantic_metalanguage [V-2nd]. So in NSM, NOT, IF, BECAUSE, CAN, MAYBE, VERY, MORE, THIS, THE SAME, ONE, TWO, SOME, ALL, BEFORE, AFTER are all PRIMES, not function words to be defined; "possession" is the prime (IS) MINE; and "known" is KNOW.
- Combinatorics: primes combine in a limited set of universal syntactic frames, with valency options per prime (e.g. four frames for SAY) [V-2nd]. Semantic molecules: intermediate non-prime words used inside explications [V-2nd]. Allolexy (different surface forms of one prime): the Wikipedia article does not mention it [V-2nd; I could not confirm details of allolexy from a primary source in this session].
- Grammar is in scope: Goddard (1997), "Semantic primes and grammatical categories", Australian Journal of Linguistics 17(1):1-41, doi:10.1080/07268609708599543 [V]. Argues that all 55 primes then posited are frequently found as components of grammatically encoded meanings (pronoun systems, indefinites, classifiers, evidentials, locational deixis, tense systems, diminutives and augmentatives, modality), against the idea that grammatical and lexical semantics need different toolkits. The Oxford Handbook of Linguistic Analysis (2nd ed.) chapter by Goddard has a section on NSM grammatical semantics [V-2nd]. Wierzbicka (1996), Semantics: Primes and Universals, OUP [V].
- Numbers: Goddard (2009), "The conceptual semantics of numbers and counting: An NSM analysis", Functions of Language 16(2):193-224, doi:10.1075/fol.16.2.02god [V-abs]. Low numbers (one, two, three) can be explicated directly from primes without counting; larger numbers and the open-ended sequence depend on counting as an activity; abstract numerals (5, 15, 27) are analysed last.
- **Which NSM explications for negation, tense, definiteness, questions, possession did I verify?** None. The searches returned no actual NSM explication of tense, the definite article or questions: [unverified]. What is verified is that NSM treats NOT as a prime with a fixed syntax and says grammar draws on primes.

**Takeaway for design.** NSM supports H1 in a stronger form than expected: many function meanings ARE primes (NOT, IF, BECAUSE, MAYBE, VERY, MORE, THIS, THE SAME, ONE, TWO, SOME, ALL, BEFORE, AFTER). Since USVS is seeded from the primes, these words land on prime dimensions already; what NSM does not give is a recipe for graded NOT (NSM primes are discrete) or for scope.

---

## 6. Prepositions and spatial/temporal relations

- Tyler and Evans (2003), The Semantics of English Prepositions: Spatial Scenes, Embodied Meaning and Cognition, CUP [V]. "Principled polysemy": senses of spatial particles are related and motivated, derived from a proto-scene; chapter 4 builds the network for "over". A review notes the diachronic-proto-scene choice may not match speakers' perception [V-2nd].
- Zwarts (1997), "Vectors as relative positions: a compositional semantics of modified PPs", Journal of Semantics [V-2nd]; Zwarts and Winter (2000), "Vector Space Semantics: A Model-Theoretic Analysis of Locative Prepositions", Journal of Logic, Language and Information 9(2):169-211 [V]; Zwarts (2017) survey in Language and Linguistics Compass [V-2nd]. Prepositions denote vectors (figure relative to ground), which explains modification ("one meter behind").
- Image schemas (Johnson, Lakoff) and Herskovits: [unverified in this session].
- Computational grounding: not examined beyond the above.

**Takeaway.** Spatial prepositions are naturally vector-valued relations between two nodes (figure, ground), so they are relation nodes with two role slots. A direction vector in the space (above/below/near/far/inside already exist as NSM space primes) is a good fit; metaphorical senses (time) are by principled polysemy.

---

## 7. Numbers

- Goddard 2009 [V-abs] (above): low numbers from primes, larger numbers need counting.
- Hurford (1975), The Linguistic Theory of Numerals, CUP [V]: generative treatment of numeral systems across languages (English, French, Mixtec, Hawaiian, Danish, Welsh, Yoruba) [V-2nd]. Hurford (1987), Language and Number: The Emergence of a Cognitive System, Blackwell [V]: contents include the meanings of numerals and standardisation of complex numerals to a fixed base [V-2nd]; author's scanned copy at https://www.lel.ed.ac.uk/~jim/l_and_n. I did not verify the precise compositional rules.
- Wallace, Wang, Li, Singh and Gardner (2019), "Do NLP Models Know Numbers? Probing Numeracy in Embeddings", EMNLP-IJCNLP 2019, pp. 5307-5315, https://aclanthology.org/D19-1534/ [V-abs]: GloVe and word2vec encode magnitude up to about 1,000; character-level methods best overall; probes fail to extrapolate beyond training range.

**Takeaway.** "Numbers from ONE and TWO" is supported for small numbers; for large numbers, a counting/recursion rule (a number is a tree of operations, base and multiplier) is needed, not a point. Treat numerals as trees (e.g. twenty-three = a nested structure) rather than dimensions.

---

## 8. Cross-linguistic typology and inventories

- Negation: Dryer, WALS chapter 112 "Negative Morphemes" https://wals.info/chapter/112 [V-abs]: types of morpheme signalling clausal negation (particle, affix, verb, double negation, ...); no known language realises negation by word order or intonation alone. Miestamo (2005), Standard Negation: The Negation of Declarative Verbal Main Clauses in a Typological Perspective, Mouton de Gruyter, doi:10.1515/9783110197631 [V-abs]: 297-language sample; symmetric vs asymmetric negation; 79% of languages have just one type of asymmetry [V-2nd]. Haspelmath (1997) on negative indefinite pronouns [V-2nd].
- Grammatical categories are cross-linguistically recurring: Bybee, Perkins and Pagliuca (1994), The Evolution of Grammar: Tense, Aspect, and Modality in the Languages of the World, University of Chicago Press [V]: 76 languages, 25 phyla; recurring grammaticalization paths, universal set of TAM grams [V-abs from the abstract].
- Inventories usable as a checklist:
  - Universal Dependencies morphological features: Polarity (value Neg; Slovene lists Neg and Pos), Tense (Pres, Fut, ...), Definite (Def, Indef), PronType (Prs, Int, Rel, Dem, Tot, Neg, Ind), Number, Poss, Number[psor]; extensible with language-specific features [V-2nd, from language-specific pages; the canonical page https://universaldependencies.org/u/feat/ was not retrieved]. De Marneffe et al., "Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection", arXiv:2004.10643 [V].
  - UniMorph: Sylak-Glassman et al. (2015), "A Language-Independent Feature Schema for Inflectional Morphology", ACL 2015: 23 dimensions of meaning, 212+ features, including Polarity [V-2nd; https://github.com/unimorph/unimorph.github.io/blob/master/schema.md]. UniMorph 2.0: arXiv:1810.11101 [V].
  - Could these serve as the minimal "special dimensions"? **Largely yes as a checklist, with two caveats.** (i) They are categorical features of word forms (inflectional morphology), not graded; the project would turn each into a dimension. (ii) UD's PronType (Int, Rel, Neg, Ind) already shows that wh-words, negative words and indefinites form one family of pronoun types, which is consistent with the indeterminate-pronoun analysis in Q4.
- Universality of the inventory: supported at the level of categories (Bybee et al.; UniMorph's size suggests the set is finite but large), but each language realises them with different morphemes and with different degrees of obligatoriness. A language-neutral dimension set is plausible; a language-neutral set of surface forms is not.

**Takeaway for H4.** Safe: negation as "some morpheme fills NEG". Needs care: double negation and negative concord.

---

## 9. Existing meaning representations

- AMR: Banarescu et al. (2013), "Abstract Meaning Representation for Sembanking", Linguistic Annotation Workshop, https://aclanthology.org/W13-2322.pdf [V]. Negation = :polarity - on the negated concept; questions use the concept amr-unknown (as the filler of the asked role, e.g. "Whose toy did the girl find?" puts amr-unknown as :poss of toy); possession via :poss; quantities via :quant, :unit, :scale [V-2nd, from the search summaries and lecture slides; the guidelines repository https://github.com/amrisi/amr-guidelines was not opened]. A 2019 paper argues :polarity - is inadequate for scope and proposes a scoped extension (AMR+, arXiv:1908.01355 "Separating Argument Structure from Logical Structure in AMR") [V-2nd]. Tense, aspect and definiteness are not encoded in AMR (as far as the 2013 paper's inventory shows; unverified in detail).
- UMR: Van Gysel et al. (2021), "Designing a Uniform Meaning Representation for Natural Language Processing", KI - Kunstliche Intelligenz 35(3/4) [V]. Sentence-level graph with aspectuality; document-level graph has temporal, modal and coreference relations; temporal dependencies replace tense, modal dependencies replace modality; polarity and epistemic strength are captured on modal edges with lattice-organised values based on Boye's work [V-2nd]. **UMR is the closest verified system to graded negation** (a graded epistemic strength lattice combined with polarity), but I could not verify the actual value labels (e.g. Full/Partial/Neutral Affirmative/Negative): [unverified].
- MRS: Copestake, Flickinger, Pollard and Sag (2005), "Minimal Recursion Semantics: An Introduction", Research on Language and Computation 3:281-332 [V-abs]. A meta-language for flat semantic structures with an explicit treatment of scope; the object language is predicate calculus with generalized quantifiers. Negation is a scopal predicate (my recollection; the search did not show the negation entry; [unverified]).
- UCCA: Abend and Rappoport (2013), "Universal Conceptual Cognitive Annotation (UCCA)", ACL 2013, pp. 228-238, https://aclanthology.org/P13-1023/ [V-abs]. Multi-layered, builds on Basic Linguistic Theory; the foundational layer's categories for function-word-like material are [unverified].
- **Degree of negation:** none of the systems I verified represents "slightly negated". AMR and UD are binary; UMR has graded epistemic strength; sentiment resources (Zhu 2014) are the only place graded shifting is measured.

**Takeaway.** AMR-style (a flag on a node, :polarity -) corresponds to Option A at its crudest; its known weakness is scope, the same weakness a single NEG dimension would have.

---

## Design options the literature suggests

Notation: nodes carry a point in USVS. "NEG" = a negation dimension in [0,1]. "ADJ axis" = the word's own signed scale dimension (happy-unhappy as one signed axis, plus its other dimensions). "DEG" = a degree dimension (slightly...extremely). "TIME" = time-relation dims. "KNOWN" in [0,1]. "WH" = unknown flag. These dimension names are illustrative, not existing project dimensions.

### Option A: graded feature dimensions on the node
Every function word is just a point whose non-zero dimensions are its meaning; the clause tree is unchanged, and a function word's dimensions are merged (summed/maxed) into the node it modifies.
- Handles well: slightly/very as DEG on the adjective; "not" as NEG on the adjective's node; the/a as KNOWN 1/0; who as someone + KNOWN 0 + WH; matches the shifting finding (Zhu 2014) and the not-cold-near-cold finding (Aina 2018); trivially cross-linguistic (an affix fills the same dimension); no operator machinery; closest to the owner's H1/H2.
- Handles badly: scope ("every student did not fail"); double negation (NEG would saturate or cancel; "not unhappy" must stay different from "happy", Hermann 2013); non-additive interactions (not very happy vs very not happy, Kiritchenko 2016); connectives and relations (if/because need two arguments); no way to say what a word negates.
- Test cases:
  - slightly unhappy: ADJ axis = unhappy-pole, DEG = low.
  - unhappy: ADJ axis = unhappy-pole, DEG = mid (the lexical affix is just a negative pole of the axis).
  - not happy: ADJ axis = happy-pole, NEG = 1. Differs from "unhappy" by representing the pole vs the negation, so contrary vs contradictory is carried by which of the two is set. Reasonable.
  - not unhappy: ADJ axis = unhappy-pole AND NEG = 1; reading ambiguous because the single NEG cannot say it applies to the already-negative adjective only. FAILS to separate "not unhappy" from "not happy" unless the model learns a convention.
  - nobody came: person node with NEG = 1 and QUANT = none merged into one node, and verb with NEG = 0; works only if "no one" is a stored word-point with those values; scope relative to other operators is lost.
  - never: TIME-all-times + NEG = 1 on the clause predicate or on a time node; OK for simple cases.
  - didn't eat: verb node with NEG = 1 and TIME = before-now; the do-support word "did" contributes only the time dimension: good (no special-casing).
  - if: a conditional dimension on the clause node; cannot say WHICH clause is the antecedent: FAILS.
  - the vs a: KNOWN = 1 vs KNOWN = 0 on the noun node: handles.
  - who: someone-point + WH = 1 + KNOWN = 0: handles.

### Option B: operator nodes that transform their argument
Function words are tree nodes with the content as a child; each operator's point defines a transformation (a matrix or learned function) applied to the argument's point. Tree nesting gives scope and order.
- Handles well: scope; double negation as two nested nodes (NEG(NEG(x)) is not x unless the learned map is an involution, and Hermann 2013 argues it should not be exactly identity); connectives as relation nodes over clause children; quantifiers (nobody = NO applied to person); matches Montague/GQ, Coecke et al., Baroni-Zamparelli (adjectives as matrices).
- Handles badly: gradedness (a matrix does not obviously have "slightly" unless degree is a parameter); needs an operator learned per word, which can hide the meaning in the matrix and weakens interpretability ("meaning lives in USVS only"); training signal for matrices is unclear; Rimell 2017 shows such operators need the domain as an extra input; mapping not-A to antonym is empirically wrong (Aina 2018).
- Test cases:
  - slightly unhappy: SLIGHTLY(unhappy), a degree operator on the adjective point.
  - unhappy: lexical point (single node; or UN(happy) if morphology is decomposed).
  - not happy: NOT(happy).
  - not unhappy: NOT(UN(happy)) or NOT(unhappy): different nested structure from "happy"; scope clear. Result point is whatever the matrices give; gradedness depends on matrix design.
  - nobody came: NO(person) as argument of "came", or NOT(SOME(person) came), a scope choice that the tree makes explicit.
  - never: NOT(SOMETIME) or NO(time).
  - didn't eat: NOT(PAST(eat)) vs PAST(NOT(eat)): two scopes, both representable.
  - if: IF(clause1, clause2) as a relation node.
  - the vs a: DEF(noun) vs INDEF(noun) as operators that issue lookup/create against memory.
  - who: WH(someone) as an operator that binds the unknown.

### Option C (hybrid): operator node whose effect is a displacement along dedicated dimensions
Every function word is a tree node (as in B), carrying a point (as in A). One language-neutral composition rule: the parent node's point = child's point plus the operator's displacement on the dimensions the operator has non-zero values in, with clipping (or a learned monotone combination) per dimension; the tree records scope and nesting; the dimension carries degree. No word-specific code: the behaviour "falls out of where the word sits".
- Handles well: scope and nesting (tree), gradedness (dimensions), double negation (two operator nodes; the NEG dimension may saturate but the tree keeps both, so "not unhappy" differs from "happy"), interpretability (the displacement is readable), cross-linguistic fill (affix or particle both create the same operator node), connectives and prepositions (relation nodes with role-bound children whose own point sits on IF/BECAUSE/BEFORE dimensions), numerals (nested trees), definiteness and wh as operator nodes that set KNOWN/WH and trigger memory lookup. It matches the literature's best-supported reading: shifting (Zhu), intermediate meaning (Aina), scope and operators (formal semantics), and operator-with-domain (Rimell).
- Handles badly: requires deciding which dimensions an operator displaces for a given head, i.e. the Rimell finding that the operator needs the domain; the saturation behaviour for repeated negation needs a rule or learning; closed-scale vs open-scale adjectives (Kennedy) may need the operator to know scale type; still no verified account of degree-modifier x negation interaction; non-compositional idioms ("not bad" meaning good) are facts in memory, which fits the project's rule but is not tested here.
- Test cases:
  - slightly unhappy: SLIGHTLY operator point (DEG low) over unhappy: result = unhappy pole + DEG low.
  - unhappy: single word-point (ADJ axis negative pole). If "un-" is morphologically decomposed: UN operator (NEG ~0.7, contrary) over happy; unhappy then differs from "not happy" only by the operator's size/strength value, matching Horn/Krifka's strengthening discussion.
  - not happy: NOT operator (NEG = 1 displacement, weaker pull toward the antonym pole than UN) over happy: result near happy with NEG high, consistent with Aina 2018.
  - not unhappy: NOT over UN over happy: two displacements, NEG saturates but the tree keeps nesting, so it is distinguishable from happy and from not-happy, as Hermann 2013 requires.
  - nobody came: NO-person operator (QUANT none + NEG) on the argument node, or NOT over SOME; both tree shapes are available; the grammar (not code) decides which a language gets.
  - never: operator node on the time role: ALL-times + NEG.
  - didn't eat: NOT operator + PAST operator (time dimensions) over eat; "did" contributes TIME only; order in the tree gives scope.
  - if: IF relation node whose children are two clause trees; its point has the IF dimension (an NSM prime).
  - the vs a: DEF node sets KNOWN = 1 (memory lookup), INDEF sets KNOWN = 0 (memory write); each operator is its position on KNOWN.
  - who: SOMEONE point with WH/unknown set and KNOWN = 0; same word-point as "someone" plus an unknown flag, per Hamblin/Kratzer-Shimoyama.

| Criterion | A: feature dims | B: operator nodes | C: hybrid displacement |
|---|---|---|---|
| Graded negation (slightly/very) | good | needs extra parameter | good |
| Scope / nesting | poor | good | good |
| Double negation (not unhappy) | poor | good | good |
| unhappy vs not happy | partly | partly | good (pole vs displacement) |
| Interpretability (meaning only in USVS) | best | weakest | good |
| No special-casing outside space | yes | needs learned maps | yes (one generic rule) |
| Connectives, prepositions, quantifiers | poor | good | good |
| Numbers > 2 | poor | good (trees) | good (trees) |
| Cross-language fit | good | good | good |
| Main risk | scope failures | opaque matrices | choosing displacement dims, saturation |

Recommendation from the literature: C.

---

## Open questions the literature does not settle

1. Is not-A closer to A, to its antonym, or in between? Aina et al. 2018 find not-A closer to A in contexts of use, but Horn/Krifka's pragmatics say it is often read as the antonym; no verified work settles this as a representation choice, only as a measurement on corpora.
2. Is a single saturating NEG value enough for repeated or nested negation, or must the representation keep the number of negations? Hermann et al. 2013 argue for a representation where not-not-bad is not bad; the proposed form was not verified in detail.
3. How do degree modifiers compose with negation (not very happy vs very unhappy vs slightly unhappy)? No verified formal or vector account; Kiritchenko and Mohammad 2016 only show non-additivity.
4. Do the structures of antonymy (gradable, complementary, converse, reversive) require different representations? The primary-source typology was not verified here.
5. Is there probing evidence that negation is a single linear direction in neural representations? I found direction-style results for truth (Marks and Tegmark) but could not confirm one for negation: [unverified]. If it exists it would support an operator that is a displacement.
6. Can numbers beyond two be represented as points at all, or only as counting trees (Goddard 2009 suggests the latter)? Embedding probes (Wallace 2019) show magnitude is encoded but not reliably extrapolating.
7. Which tense representation is best in a graph: operators, or reference-time variables (Reichenbach/Partee)? Bjorkman's critique shows the pronoun analogy is not settled.
8. Whether closed-class words are cleanly "grammaticalised" instances of a few NSM primes (Goddard 1997 says primes occur in grammatically encoded meaning but I did not verify explications for tense or definiteness).
9. How negative concord and double negation across languages map onto one NEG operator: WALS reports double negation as a type but I did not retrieve a source on its semantic interpretation.
10. Whether universal feature inventories (UD, UniMorph) are semantically motivated or only descriptive conveniences: both are annotation schemas; neither claims to be a semantic space.

## Unverified items collected in one place
Allolexy (NSM, primary source); NSM explications for tense, definiteness, questions; UMR value labels; MRS negation entry; UCCA function-word categories; image schemas (Johnson, Lakoff) and Herskovits; the typology of antonym types (Cruse/Lyons); any verified paper claiming that negation is a linear direction in neural representations; a distinct body of work labelled "conversational negation".
