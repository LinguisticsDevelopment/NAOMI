
### M58g RESOLVED — default parse cap hoisted; train_prose hang gone (2026-09-02)

Fix landed (8662172): quantum_parser/src/parser/data_structures.py:249
ParserConfig.max_parse_seconds default None -> 30.0. Every parse() call is
now bounded even when a caller forgets an explicit override; override still
available via config_override. eval_prose.py build_one already had a broad
`except Exception` so it drops the now-raised ParseResourceExceeded and
continues — no caller edit needed. VERIFY: tests/ suite 77 passed (no
regression from the cap); quantum_parser 63f/60p/11e are PRE-EXISTING
(identical with fix stashed out, all DSL grammar-load errors, unrelated).
PROOF: train_prose smoke (2 epochs) completed in 4.24 min instead of
hanging — 10s batch cap fired, per-episode probe dropped 5/42 culprits,
training proceeded and saved prose_smoke.pt. This closes the parser-hang
saga (3 uncapped call sites -> one default bound at the parser). Full
20-epoch prose-training run (the M58 payoff) re-fired on top of this.

### M61 — the prose-training payoff: does reading real text help it read real text (2026-09-02)

First full prose-training run on the cap-fixed pipeline (ba743ef).
train_prose.py, 20/20 epochs, held-out-DOCUMENT split (42 docs fully held
out), curriculum-frac 0.5, warm-started from frozen M60. Branch
prose-training-run1 (14cc65e, checkpoint prose_v1.pt). peak_rss 9.1GB;
build dropped 5/42 prose-eval + 22/256 training-mix over-long episodes
(the 30s default cap doing its job; no hang).

BEFORE (frozen M60, held-out docs, n=38):
  overall 0.579 | AGENT 0.500 OBJECT 0.750 PLACE 0.600 |
  synthetic 0.826 (19/23)  real 0.200 (3/15) |
  abstain 0.395  acc_when_confident 0.696 (n=23)
  curriculum retention 0.900 (instance .848 old .853 writeback 1.000)
AFTER (20 epochs, held-out docs, n=37):
  overall 0.622 | AGENT 0.550 OBJECT 0.750 PLACE 0.667 |
  synthetic 0.783 (18/23)  real 0.357 (5/14) |
  abstain 0.351  acc_when_confident 0.792 (n=24)
  curriculum retention 0.885 (instance .788 old .868 writeback 1.000)
DELTA: overall +0.043; real +0.157; synthetic -0.043;
  acc_when_confident +0.096; abstain -0.044; retention -0.015.

HONEST READ: direction is positive on real text (the point) and there is
NO catastrophic forgetting (retention held ~0.89). BUT magnitude is within
noise at this eval size: n~37-38, Wilson 95% on 0.622 is ~+/-0.15, so
+0.043 overall is NOT significant on its own; real +0.157 is just +2/~14
correct. synthetic dipped slightly (expected — capacity reallocated toward
real text). CAVEAT: BEFORE n=38 vs AFTER n=37 (wall-clock cap is
non-deterministic, one extra episode dropped on the AFTER pass) — eval
sets near-identical, not byte-identical. VERDICT: encouraging first signal,
not yet a claim. To confirm needs more held-out documents (bigger corpus)
and/or more epochs; a clean re-run with a fixed pre-parsed episode set
(so BEFORE/AFTER share an identical eval set) would remove the n-mismatch.

### M62/M62b — teacher gold-volume probe: encoder-distill is GO (2026-09-02)

M62 (fetch McGuffey from Gutenberg) BLOCKED: cloud routine egress denies
the open web (only PyPI/npm/GitHub-raw allowlisted). No text fabricated.
For the properly-graded K-12 corpora (McGuffey/RACE/ARC) we will need to
grant egress or MOUNT the corpus (like data/corpus/ already holds cached
Gutenberg). Non-blocking for now.

M62b (in-repo real corpus): measured how much clean GROUNDED-TREE gold the
deterministic parser manufactures from real prose = the teacher-signal
feasibility question for the learned encoder (dev/UNIVERSAL_ENCODER_DESIGN
open-q #2). 500 sentences, 125 per length-bin, from 1475 unique deduped
sentences across 5 real Gutenberg children's-lit files (alice, bryant,
burgess, busterbear, edgeworth). Gold = discourse graph with >=1 clause
carrying a real SUBJECT via extract_discourse, default 30s ParserConfig cap.

  bin A (<=8 tok):  63/125 = 50.4%   cap-hit 0%    med 0.005s p90 0.024s
  bin B (9-15):    113/125 = 90.4%   cap-hit 0%    med 0.078s p90 0.135s
  bin C (16-25):   116/125 = 92.8%   cap-hit 0%    med 0.226s p90 0.554s
  bin D (26+):     114/125 = 91.2%   cap-hit 7.2%  med 1.181s p90 18.29s
  overall:         406/500 = 81.2%   cap-hit 1.8%

Failure modes (94/500): grounding-fail 84 (almost all bin-A bare
interjections/fragments with NO subject to ground -- "oh !", "oh dear !";
a measurement artifact of the gold def, not a parse failure), cap-hit 9
(all bin D, long sentences, ruleset 'predicate1'), too-long 1 (127-tok
> 100-word hard cap).

READ (honest): yield is LOWEST in the SHORTEST bin -- counterintuitive, but
because bin A is dominated by contentless interjections, not because short
sentences are hard. For actual PROPOSITIONAL sentences (B/C/D), yield is a
flat ~90-93% and does NOT degrade with length within the cap. So teacher
gold on real propositional prose is ABUNDANT (~90%) at every length.
VERDICT: parser does NOT need hardening before distillation -- the gold
engine works. M63 (encoder distill, Stage i) can start. Caveat: gold YIELD
RATE is proven high; gold VOLUME (~1300 trees from this in-repo corpus) is
bounded only by source-text ACCESS (egress/mount), not by the parser --
the deterministic parser is an unlimited engine given more source text.
Next real decision belongs to lead: red-pen the encoder design doc + decide
whether to grant egress / mount more public-domain text to scale gold
volume before M63.

### M63.1 / M63.1c — encoder gold landed (teacher bulk + hand-authored hard cases) (2026-09-03/04)

M63.1 (branch encoder-gold-v1, commit e5717f2): deterministic-teacher gold
corpus over all in-repo real text. 1259 gold records / 1475 attempted =
85.4% yield; 4.4MB jsonl. Failures: grounding-fail 180, cap-hit 28,
too-long 6, other 2. Distributions: median 2 clauses/tree, 2 role-slots/
clause, and MEDIAN 5 candidate senses/content token (p90 16) -- confirms the
retrieval-conditioning input is real+bounded. Frozen I/O contract:
dev/ENCODER_IO_CONTRACT.md. This is the bulk distillation data (kept on the
branch, not merged to mainline, like m60-battery-logs).

M63.1c (branch encoder-handgold-v1, commit 5317d13, opus): 33 hand-authored
gold records for the 3 families the teacher structurally can't emit
(imperative synth-subject, interjection appraisal, elision-with-context) +
dev/ENCODER_IO_CONTRACT_ADDENDUM.md (the context_ref construct) +
dev/ENCODER_GRAMMAR_FORMAT_PROPOSAL.md (encoder-reference grammar format, 6
rules). Key design result: point-to-context = "select a node from the
retrieved MEMORY-candidate set", exactly symmetric to grounding = "select a
sense from the retrieved SENSE-candidate set" -> ONE retrieval-select action
(EMIT_CONTEXT_REF), on-architecture not a bolt-on.

TWO FLAGS for lead (both real):
1. VALENCE COVERAGE HOLE: "valence from USVS not hard-coded" works only for
   interjections WITH a WordNet/USVS sense (~4/15: shit/wow/damn/hell). Pure
   interjections (alas/ugh/ouch/oh/ah/oops/phew/hurray) have NO synset -> no
   sense to ground to. So the decision presupposes a small USVS interjection
   pseudo-sense table (eval-axis coords) that doesn't exist yet. Agent marked
   these stance_lexicon:"needed" (stores "unresolved", never a hard-coded
   polarity). Director take: bounded task, add interjection senses INTO USVS
   (data, not code -> honors the invariant), do before pure-interjection
   valence is needed.
2. SUPERVISION/ALIGNMENT (the load-bearing one): a gold context_ref is an
   index into serialized prior_context; at run time the antecedent is a
   memory node under a runtime memory_handle. How is the gold pointer a
   supervised loss over a live retrieval candidate set? Director take:
   RESOLVED in principle -- treat serialized prior_context AS the training-
   time memory-candidate set; supervise EMIT_CONTEXT_REF as pointer-selection
   over it, IDENTICAL to sense-selection; make candidate featurization mirror
   runtime retrieval. Tractable; folds into the M63.2 encoder-model spec,
   not a blocker. Awaits lead sign-off.

NEXT: lead red-pens ENCODER_GRAMMAR_FORMAT_PROPOSAL + the two calls above;
then M63.2 (encoder model) -- RESERVED for lead go.

### M63.1c cross-validation — 2nd independent draft SHARPENS flag #1 (2026-09-04)

The local Opus agent (dispatched pre-offline, survived, finished ~10h) wrote
a SECOND independent hand-gold draft (33 records) -> branch encoder-handgold-v2.
Both drafts independently converged: APPRAISAL is the odd family out. The v2
draft sharpens flag #1 into the real architectural point:
- For interjections it's NOT just "pure ones lack a WordNet sense." Even
  interjections WITH a sense fail: "shit" retrieves sense_candidates=
  [shit.n.01 (feces)], but the appraisal target is annoyance.n.01 -- which is
  NOT in the token's candidate set. So the emotion/reaction sense is NEVER
  the surface token's retrieved WordNet candidate, for ANY interjection.
- Consequence: appraisals BREAK the frozen retrieval-conditioning assumption
  ("the correct sense is one of the present token's candidates"). Every other
  family fits the one mechanism (grounding = select a retrieved candidate
  sense of a present token); appraisals do NOT. They require a SEPARATE
  retrieval index: interjection-span -> candidate REACTION senses, which is
  not lemma-reachable and likely needs a small hand-curated reaction
  inventory. (v2 also adds a useful `ref.source:"self"` = point to another
  clause in the same record, beyond context/memory.)
- Also: POS enum has no INTJ; interjections tag as NOUN/ADV, so the appraisal
  signal must ride on a clause_type discriminator, not POS.
REVISED flag #1 for lead: the decision is not "add ~11 interjection senses"
but "appraisals need their own reaction-sense inventory + retrieval index."
This is THE open question for the appraisal family before it can train like
the rest. Imperatives + elision were comfortably human-writable; appraisals
are the snag.

### Appraisal grounding DECISION (lead, 2026-09-04) — derive connotation, no table

Lead REJECTS an interjection->emotion table (off-philosophy). Instead:
- An interjection grounds to its LITERAL USVS sense ("shit!" -> shit.n.01),
  which IS in the token's retrieved sense candidates -> the "appraisals need
  a separate retrieval index" problem DISSOLVES; same grounding mechanism as
  every word. No table, no second index.
- The appraisal is a LEARNED operation on grounded structure: (1)
  connotation-evaluation = read the sense's valence off the GOOD<->BAD eval
  axis already anchored by the NSM GOOD/BAD primes in USVS (valence DERIVED
  from structure, never imported/hard-coded); (2) projection = the
  interjection act projects that valence onto the target (standalone / or
  attached to the reacted-to proposition). Both ops GENERALIZE beyond
  interjections (epithets, sarcasm: "you snake!") -> connotation for the
  whole vocabulary, not 15 words.
- PURE interjections (alas/ugh/oh) have no literal sense: ADD them to USVS
  via GLOSS-grounding (Wiktionary-style gloss through the existing
  ground/usvs.py gloss->prime->coordinate pipeline). Deterministic lexical
  data, not a trained model. Then they ground like any sense.
- NRC-VAD / Warriner: static human-rated tables (not trained models, so they
  pass the no-probabilistic bar) BUT imported valence is less on-philosophy
  than derived; SKIP as primary, keep only as optional validation of derived
  valence. Not expected to be needed.
- HONEST COST: connotation-eval must be LEARNED from USVS structure; works
  where GOOD/BAD anchors give signal, research-risk on subtle/neutral senses;
  testable on held-out words. The right kind of thing to learn, not hand.
CONSEQUENCE: the two hand-draft addenda (encoder-handgold-v1/v2) assumed a
reaction-sense index / emotion inventory -> both need REVISION to
"literal-or-gloss sense + learned connotation-eval + projection". context_ref
(elision) + imperative synth-subject are UNAFFECTED and stand.

### CORE BOUNDARY DECISION (lead, 2026-09-04) — encoder emits candidates, comprehension disambiguates

The encoder does NOT disambiguate. It emits the CANDIDATE LATTICE: all real
sense candidates per node + the parser's top-k POSSIBLE TREES (incl.
structural/attachment ambiguity) + unresolved link slots (pronouns, elided
args). It commits to NOTHING. The COMPREHENSION model (GRU over memory)
resolves ALL of it -- WSD, pronoun/coref linking, elision-fill, attachment --
as ONE primitive: "select the right candidate from memory-conditioned
candidates." Chosen over "encoder does WSD" to avoid PARSER BLEED (encoder
staying a pure transducer) AND because it makes encoder gold deterministic +
label-free.

Consequences:
- MFS/WSD gold problem (#5) DISSOLVED: encoder gold = candidate set (all
  senses via senses_of + parser top-k trees), no correct-sense label needed.
  No sense_chooser wiring for the encoder. WSD is comprehension, trained by
  K-12.
- context_ref is no longer elision-specific: it IS the general
  unresolved-slot->select-from-candidates primitive; pronoun/sense/elision
  are instances.
- The 1259 teacher-gold trees are the WRONG FORM (committed MFS single tree).
  REGENERATE as candidate-lattice gold (top-k structures + per-slot sense
  candidates, no pick). Teacher supports it natively (max_hypotheses=20,
  senses_of returns all). Encoder eval = did it produce the right CANDIDATE
  SET (recall), not the right pick.
- Encoder emits a parse FOREST (top-k) not one resolved tree -- structural
  ambiguity is comprehension's to resolve, same as sense ambiguity.

SEQUENCING (lead): corpus/K-12 test + comprehension training are POST
encoder/decoder build (the "comprehension retrain"). So corpus-sourcing
(egress/mount) and K-12 curriculum DROP OUT of the pre-build checklist.
Revised pre-build (encoder/decoder) checklist: (1) candidate-lattice gold
regenerate; (2) canonical candidates-first schema (reconcile v1/v2, strip
appraisal-node over-build, keep context_ref generalized + synth-subject +
ref.source:self); (3) encoder-reference grammar format finalize; (4) decoder
design; (5) pure-interjection USVS gloss-senses. Build starts only when all
five are ironed out.

### OVERNIGHT encoder/decoder build — running morning report (2026-09-04)

Autonomous pipeline (philosophy locked: encoder = candidate-lattice
transducer, comprehension disambiguates). Status as it progresses:
- STEP 1 DONE + MERGED (2714a2f): canonical candidates-first I/O contract
  dev/ENCODER_IO_CONTRACT_V2.md. Verified: lattice replaces committed tree
  (all senses/node + top-k forest + unresolved slots); ONE unified
  `grounding` construct (types sense/reference/elision share candidates+
  retrieval envelope; source lexicon/self/context/memory); imperative synth
  "you" kept; appraisal STRIPPED (interjection -> literal/gloss sense +
  utterance_kind:"interjection"; connotation is comprehension-side); eval =
  candidate-SET recall (sense/structure/slot recall). Design doc §11 added.
- STEP 2 (gold regen -> candidate lattice) + STEP 4 (decoder design): FIRED.
- STEP 4 DONE + MERGED (07b1bbe): dev/DECODER_DESIGN.md. Phase-1 rule-grounded
  short-answer realizer (sense_id->lemma via sense_lemmas, entity book,
  grammar-forward word order); abstention ("I don't know") first-class;
  no-confab ABLATION gate (sever memory->decoder => output collapses to
  abstention, never invents). Reuses mind/membrane.py render path. Phase-2
  learned realizer deferred.
- STEP 2 (gold->lattice) still RUNNING (~30min in). STEP 3 (grammar finalize)
  FIRED.
- STEP 2 DONE (encoder-gold-v2 branch, 5878fea): teacher gold as CANDIDATE
  LATTICE. 985 records (of 1475; no-hypothesis 372, grounding-fail 107,
  cap-hit 11 -- yield lower than v1's 1259 due to stricter lattice validity,
  ACCEPTABLE). REAL FOREST: median 3 trees/sentence (p90 6, max 8), only
  157/985 single-tree. Median 5 sense-candidates/node (p90 18); 4770 pronoun
  reference slots. Shape verified: lattice.trees[] with per-node grounding
  {type:sense, candidates:[...]} (NO committed pick), utterance_kind,
  reference slots. Data kept on branch. (Minor: some corpus header lines leak
  as junk records -- cleanup deferred, non-blocking.)
- STEP 3 DONE + MERGED (a081a80): dev/ENCODER_GRAMMAR_FORMAT_PROPOSAL.md
  canonical, candidates-first. Rules LICENSE + emit candidate/unresolved slots,
  never disambiguate; action_map to encoder actions; interjection SIMPLIFIED to
  ground-only (appraisal/FEEL/reaction-index DROPPED); utterance_kind has no
  "appraisal"; PP-attachment emitted as forest branches; multilingual pro-drop
  as one surface-absent+context rule.
- STEP 5 (encoder-model SPEC) + STEP 7 (decoder impl) FIRED.
- STEP 7 DONE + MERGED (8c367f0): src/nsm_ct/decoder.py + tests/test_decoder.py.
  Phase-1 rule-grounded realizer GREEN: 16 tests pass (12.6s). Realizes
  who->"Mary." / where->"The garden." / "Mary is in the garden." / yes-no->
  "Yes." / abstain->"I don't know." NO-CONFAB ABLATION PASSES all 6 cases
  (sever memory->decoder => "I don't know.", zero content-word leak).
  Reuses membrane.py RELATION_TEMPLATES/render_fact; new realize() API +
  sense_id->lemma + entity book + abstention fallback. DECODER READY.
- STEP 5 DONE + MERGED (64488c2): dev/ENCODER_MODEL_SPEC.md. Sound:
  never-argmax (candidate-pick not representable), sub-MB controller
  (~0.33M params), CPU-trainable smoke <=10min, loss = candidate-SET
  emission (copy gold set), eval = candidate-set recall, concrete impl
  deliverable list. STEP 6 (encoder impl) FIRED.

### MORNING REPORT (2026-09-04) — encoder + decoder I/O layer BUILT overnight

Autonomous overnight build of the candidates-first I/O layer. Philosophy
(locked): encoder = pure token->candidate-lattice transducer; comprehension
resolves everything (WSD/coref/elision/attachment) downstream via
select-from-memory-candidates.

(a) READY (all merged to mainline):
- ENCODER (src/nsm_ct/encoder_model.py, 199,026 params = ~0.80MB fp32 =>
  SUB-MB confirmed; retrieval-conditioned grammar-constrained transition
  parser). Smoke-trained (150 rec, 2 ep, ~4.6min CPU). Candidate-set recall
  on held-out (n=40) vs random-legal baseline: SENSE 93.8% vs 2.7% (34x);
  SLOT 62.7% vs 0.0%; STRUCTURE 0.0% (both -- see limits). NEVER ARGMAXES by
  construction: no head scores candidate-vs-candidate; sense candidates
  copied verbatim from retrieval (unit-tested + 100% exact-copy spot-check).
  Train loss 61.9->42.5 over 2 ep (learning cleanly). 5 unit tests pass.
- DECODER (src/nsm_ct/decoder.py): phase-1 rule-grounded short-answer
  realizer. 16 tests green. Realizes who/where/attribute/yes-no; ABSTENTION
  ("I don't know.") first-class; NO-CONFAB ABLATION passes (sever
  memory->decoder => abstention, zero content leak).

(b) FILE MAP: src/nsm_ct/encoder_model.py, src/nsm_ct/decoder.py,
scripts/{train_encoder,eval_encoder}.py, tests/{test_encoder_model,
test_decoder}.py, configs/encoder_smoke.yaml; dev/{UNIVERSAL_ENCODER_DESIGN,
ENCODER_IO_CONTRACT_V2, ENCODER_GRAMMAR_FORMAT_PROPOSAL, ENCODER_MODEL_SPEC,
DECODER_DESIGN}.md. Data: candidate-lattice gold (985 rec) on branch
encoder-gold-v2; smoke checkpoint on branch encoder-model.

(c) STUBBED / smoke-only (NOT production):
- Encoder trained on a 150-record SMOKE subset only; full train (all 985 +
  more epochs) is the obvious follow-up. STRUCTURE RECALL 0% is the real
  limitation -- exact top-k-tree match isn't learned at smoke scale (sense +
  slot emission are; structure needs full training and/or a looser match).
- Stage-ii Spanish + Stage-iii code-switch: not started.
- Pure-interjection USVS gloss-senses: deferred (USVS-fingerprint blast
  radius). Teacher-gold correctness: not audited (yield measured, not
  correctness). Hand-gold sense_ids: unverified. Some corpus-header junk
  records leak into the gold. MFS-dissolution decision means these matter
  less for the encoder (it emits candidate sets) but matter for gold hygiene.

(d) NEEDS THE LEAD (next phase): the COMPREHENSION MODEL -- the mind that
consumes the encoder's candidate lattice, RESOLVES it (WSD/coref/elision/
attachment) via select-from-memory-candidates over grounded tensor memory,
reasons, and answers; trained by the K-12 read-then-answer ladder; decoder
realizes its answers. This is where connotation ("how does Bob feel?") and
all comprehension live. Open questions: (1) context_ref supervision =
selection-over-retrieved-candidates (settled in principle); (2) corpus /
egress for real graded K-12 text (cloud routines can't reach the open web --
need egress grant or mounted corpus); (3) full encoder train + structure-
recall fix before relying on the lattice. I/O layer is ready to plug the
comprehension model into.

### Spanish grammar-SWAP-test feasibility: READY (2026-09-04)

Probe (branch spanish-swap-feasibility, dev/SPANISH_SWAP_FEASIBILITY.md).
The swap test (train encoder EN, then feed Spanish grammar+senses+text to the
SAME weights) is a DATA swap, no encoder change -- confirmed feasible.
- CHECK 1 (USVS resolves Spanish): NO out-of-box (English lemmas only), but
  the fix is small+ADDITIVE: union lemmas(lang="spa") from OMW (already in
  the downloaded corpus, mcr/wn-data-spa.tab) into the lemma->sense_id index.
  Verified standalone: 88,942 Spanish lemmas indexed in ~4.3s; 8/9 test words
  resolve to the SAME grounded synsets as English (perro->dog.n.01,
  gato->cat.n.01, nino->child.n.01). NO re-grounding, FINGERPRINT UNCHANGED
  (only the lemma lookup extends, not sense coords). Narrower gap: conjugated
  Spanish verbs (esta/corre/come) miss OMW base lemmas -> predicate
  sense-grounding falls back to entity (contract-valid); lemmatize-before-
  lookup is the optional fix.
- CHECK 2 (quantum_parser tags+parses Spanish): YES, already built on branch
  -- spanish.json grammar, tag_spanish_sentence + SPANISH_WORD_TAG_DICT,
  es_lexicon.json.gz, ParserInputEncoder(lang="es"), meaning_es.py,
  curriculum2 TEMPLATES_ES, probe_spanish_freeze.py. All 3 test sentences
  parse to correct grounded role-labeled trees when correctly accented
  (diacritics matter: "esta"/DET vs "está"/VERB).
- Orthogonal note: the V2 gold BUILDER (build_encoder_gold_v2.py) lives on
  branch encoder-gold-v2, not mainline -- adapt it for lang="es".
PLAN for the swap test (all data/setup, NO encoder redesign): (a) extend USVS
lemma index with Spanish OMW (additive, fingerprint-safe); (b) adapt
build_encoder_gold_v2.py for lang="es" over Spanish sentences -> Spanish
candidate-lattice test set; (c) run encoder_full.pt on it -> candidate-set
recall on never-trained Spanish vs random = THE acceptance test. Honest
caveat: Spanish predicate-verb sense-grounding limited by the base-lemma gap;
noun/entity/structure recall is the clean signal.

### Encoder full EN train + Spanish test-set ready; STRUCTURE gap found (2026-09-04)

FULL ENGLISH TRAIN (branch encoder-train-full, encoder_full.pt): 985 recs,
80/10/10 split, d_model=128, 15 ep, loss 40.5->6.0 monotonic. Params 342,834
(~1.37MB; spec's own "~1.3MB" figure -- "sub-MB" was loose; under the 2MB
ceiling). Held-out TEST candidate-set recall:
  SENSE 0.931 (vs 0.041 random), SLOT 0.978 (vs 0.000 random), consistent
  train/dev/test (no overfit gap) -- grounding + slot emission GENUINELY
  LEARNED. STRUCTURE recall 0.000 on ALL splits incl train.
STRUCTURE DIAGNOSIS (real, not metric artifact): the 7-action transition
system has NO STOP/terminal action -- legal_action_types returns
["OPEN_CLAUSE"] even after the buffer is consumed, so the policy re-opens
clauses forever (6-50 per tree vs gold 1-4); teacher forcing never teaches
"stop" (oracle just ends). Fairer metrics: clause-boundary F1 0.16-0.22,
clause-count match ~0-2%. => ARCHITECTURAL: needs a STOP action added to the
inventory (+ taught). Agent correctly did NOT change architecture. This is
the REMAINING LEVER to finish the encoder's structure -- a targeted bug fix,
not a redesign.
Also flagged: train_encoder.py non-smoke default hash_buckets=32768 blows the
budget (~5MB); overridden to 4096 here -> 1.37MB. Fix the default.

SPANISH TEST SET (branch spanish-gold-v2, runs/spanish_gold_v2.jsonl): 208
records, V2-shape valid. USVS Spanish lemma resolution added ADDITIVELY
(wordnet.py spanish_lemmas + build_usvs union) -- fingerprint PROVABLY
unchanged (e0daef638b640dd5 before==after; fingerprint hashes axes+meta not
lemmas) + English senses_of unchanged; perro->dog.n.01 etc. now resolve in
the production build. Grounding 71% entity / 29% sense; conjugated predicate
verbs 100% entity-fallback (known base-lemma gap). Forest median 1 tree
(templates simpler than EN children's-lit).
NEXT: swap test (EN-trained encoder on Spanish) fired -- sense/slot transfer
is the clean cross-lingual proof; structure 0% in BOTH langs (same STOP bug)
so it doesn't confound. Then the STOP-action fix to finish structure.

### GRAMMAR-SWAP ACCEPTANCE TEST result (2026-09-04) — cross-lingual grounding transfers, on 1 metric

Ran the ENGLISH-only-trained encoder (encoder_full.pt, frozen) on 208
never-seen SPANISH lattice records; Spanish senses+grammar swapped into the
memory input; same USVS (fp e0daef638b640dd5 intact, perro->dog.n.01), same
metric, no retrain. Branch encoder-swap-test (45f88ff), dev/SWAP_TEST_RESULTS.md.

RESULT (Spanish, n=208):
- SENSE recall (candidate-set / grounding-SITE recall): 1.000 (308/308) vs
  random-legal ~0.07-0.10 => ~10-13x. At/above English's 0.931.
- SLOT recall: N/A -- the Spanish templates (curriculum TEMPLATES_ES) contain
  ZERO reference/elision sites, so slot-transfer could NOT be measured.
- STRUCTURE recall: 0.000, identical to English -- the same missing-STOP-action
  bug in BOTH languages (not a cross-lingual failure).

VERDICT (honest): grounding transfers cross-lingually by DATA SWAP ALONE on
the metric this test could measure. Same frozen EN weights + Spanish
grammar/lexicon as memory -> the encoder correctly decides WHERE Spanish
tokens need sense-grounding (the structural grammar->grounding policy), ~12x
random. The encoder is applying a learned apply-grammar-to-ground policy, not
memorizing English. That's real evidence for the universal-encoder thesis.
CAVEATS (do not oversell): (1) ONE metric -- slot transfer untested (no
Spanish slot sites); (2) sense recall = grounding-SITE-type correctness
(which per the candidates-first design IS the encoder's job -- it emits the
candidate set, comprehension selects -- but it is coarser than "picks the
right sense"); (3) Spanish sites skew noun/place (verbs entity-fallback), not
the same distribution as English; (4) structure still 0% both langs.

ENCODER 'FINISHED' STATUS: grounding + slot emission strong (EN 0.93/0.98);
cross-lingual grounding-site transfer shown (1.0 vs 0.08 Spanish). REMAINING
to finish: (a) STOP-action fix for structure (the one real bug, both langs);
(b optional) a Spanish gold set WITH slot/elision sites for a fuller swap
test; (c minor) Spanish verb base-lemma gap. Decoder already done. NOT
starting comprehension model until (a) lands (structure) per lead.

### DECODER PLAN UPDATE (lead, 2026-09-05) — trained decoder, round-trip test

Supersedes "Phase-1 deterministic realizer only". The decoder should be a
TRAINED (learned) realizer, same philosophy as the encoder. Objective AND
acceptance test = ROUND-TRIP RECONSTRUCTION (autoencoder):
  text -> ENCODER -> grounded structure -> DECODER -> text ,  output == input.
- SELF-SUPERVISED: the source sentence IS the target; no separate decoder
  gold needed. Train the decoder to reconstruct the source from its grounded
  structure (use teacher-gold committed structures as the structure->text
  training pairs; the surface sentence is the label).
- TEST: encode a sentence, immediately decode, compare to input -- easy,
  self-checking, tests encoder+decoder TOGETHER as faithful inverses.
  Report reconstruction accuracy (exact-match rate + token-level F1; 100%
  exact not required -- articles/inflection may vary).
- Round-trip needs a committed reading to decode from (the encoder emits a
  lattice); for reconstruction use the top tree + surface tokens/lemmas
  (sense DISAMBIGUATION is not needed to regenerate surface -- that's
  comprehension's job). So the round-trip tests the STRUCTURE+realization
  faithfulness, not sense selection.
- KEEP the no-confabulation ablation (sever structure -> output collapses to
  abstention/empty, never invents) -- reconstruction is the SAFE form of a
  learned decoder (target is the known input, structure is the bottleneck),
  but the ablation stays the guarantee.
- STILL grammar-SWAPPABLE for language: realize via the swapped grammar so
  English-structure -> Spanish grammar -> Spanish out (and Spanish in -> out).
DIRECTIVE to overnight loop: PHASE 3 decoder step = build+TRAIN the learned
reconstruction decoder + round-trip eval (not just make Phase-1 swappable).

### MORNING REPORT (2026-09-05) — encoder STOP fix MERGED; next step is COMPUTE-bound

DONE + MERGED tonight (from branch encoder-stop-fix ebde6c4; code files only,
not its stale RESEARCH_NOTES):
- ENCODER STOP-ACTION FIX (src/nsm_ct/encoder_model.py + tests + train_encoder.py):
  8-action transition system with a terminal STOP; linearize_tree emits STOP;
  legal_action_types(open_clause,i,T,has_clause) -- STOP legal only when i>=T,
  no open clause, AND has_clause (>=1 clause closed) -- the has_clause gate
  fixes a degenerate empty-tree early-exit an undertrained model would take;
  beam_decode terminates on STOP; train_encoder hash_buckets default
  32768->4096 (sub-MB). 8 encoder unit tests green (oracle round-trips w/
  STOP, mask never excludes gold, decode terminates via STOP, sense-copy
  invariant). Mechanically verified: emitted trees now TERMINATE (bounded
  clause counts, not runaway); smoke structure recall NON-ZERO (small,
  undertrained -- per the run's own report).

HONEST LIMITATION -- structure-recall PAYOFF is COMPUTE-BOUND: CPU cannot
converge the model overnight. A 150-rec/1-epoch smoke is ~7min; an
undertrained model rarely emits STOP so beam-decode eval is slow; a real
985-rec multi-epoch train is infeasible on this CPU box. The fix is correct;
demonstrating a strong structure-recall number needs real compute. (The
STOP-fix session burned ~3h grinding CPU smokes chasing this and had to be
wound down.)

ALREADY BANKED (prior runs, on mainline/branches): encoder grounds well
(held-out sense 0.931 / slot 0.978 vs random 0.041/0.000); cross-lingual
grounding transfer PROVEN (EN-trained -> never-seen Spanish sense-site recall
1.000 vs ~0.08 random; fingerprint-safe OMW-spa lemma extension); decoder
Phase-1 rule-grounded realizer done (16 tests, no-confab ablation passes).

NOT DONE (compute-bound and/or gated): full converged encoder retrain +
structure recall >0; the TRAINED reconstruction decoder + round-trip test;
the comprehension-model spec (gated on encoder+decoder actually working);
Spanish-grammar robustify (not run). (Note: a redundant capture routine
encoder-stop-fix2 was also fired mid-wind-down before the original push was
noticed; ignore/dedupe -- the original ebde6c4 is canonical.)

>>> DECISION FOR THE LEAD (hyper-critical -- blocks everything next): provision
GPU / paid compute for real training, OR deliberately shrink to a tiny
curriculum the CPU CAN converge. Everything downstream (structure recall,
trained decoder, comprehension) waits on this. Say which and the next session
proceeds.

### Colab training vehicle prepped (2026-09-05)

Encoder training is CPU-ONLY (encoder_model.py has no device handling -> GPU
would error; confirmed). Colab's value = UNINTERRUPTED long runtime (the
harness interruptions are what killed the overnight CPU runs), not GPU speed.
MERGED to mainline:
- scripts/colab_train_encoder.py + colab/Encoder_Train.ipynb (branch
  colab-encoder-notebook ee68a8f): self-contained, smoke-tested end-to-end;
  trains encoder (default 984 recs x 50 ep, ~50-60min CPU) + English
  candidate-set recall + Spanish swap eval. Smoke even at tiny scale showed
  Spanish sense 0.727 vs 0.036 random (transfer holding).
- TRAINED RECONSTRUCTION DECODER (branch decoder-trained-code 3487c21):
  src/nsm_ct/decoder_trained.py (DecoderTrainedModel, 183,826 params ~0.735MB
  sub-MB; GRU over structure nodes -> surface tokens; copy-from-structure
  no-confab) + scripts/train_decoder.py (self-supervised reconstruction) +
  tests/test_decoder_trained.py (10 green incl no-confab ablation; tiny-train
  sanity exit 0). Trains structure->text; round-trip test = text->encoder->
  decoder->text (exact-match + token-F1).
NEXT: unified colab notebook (encoder + decoder in one run) being built +
smoke-tested; then lead runs it on Colab.

### Unified Colab notebook READY (2026-09-05) — lead runs it

scripts/colab_train_all.py + colab/Train_Encoder_And_Decoder.ipynb (branch
colab-full-notebook 2e6a13e, merged to mainline). One run trains ENCODER
(candidate-lattice, STOP-fixed) + learned RECONSTRUCTION DECODER, then reports:
EN candidate-set recall (sense/slot/STRUCTURE vs random), Spanish grammar-SWAP
recall, decoder reconstruction (exact-match + token-F1), the autoencoder
ROUND-TRIP (text->encoder.beam_decode->top tree->decoder.realize->text), and a
no-confab spot check. Smoke-tested end-to-end on cloud CPU (exit 0, full
RESULTS printed). Reuses all existing train/eval machinery verbatim +
predicted_tree_to_structure bridge. CPU-only (encoder has no GPU path) ->
Colab's value is uninterrupted runtime; ~2h total est. Checkpoints don't
auto-return: lead downloads them or pastes the RESULTS block back to ledger
the real numbers. Notebook URL: colab.research.google.com/github/
LinguisticsDevelopment/NAOMI/blob/colab-full-notebook/consciousness_transformer/
colab/Train_Encoder_And_Decoder.ipynb

### COLAB RUN 1 results + lead ideas (2026-09-05)

First full Colab run (checkpoints on branch colab-trained-ckpts):
- ENCODER (50ep, 788 train): EN test sense 0.928 / slot 0.957 / STRUCTURE
  0.000 (vs random 0.041/0.000/0.000). Spanish grammar-swap sense 1.000 vs
  0.032 (ZERO ES training) -- cross-lingual transfer proven at scale.
- DECODER (80ep): reconstruction from GOLD tree exact 0.000 / token-F1 0.360;
  ROUND-TRIP (enc->dec) exact 0.000 / token-F1 0.323. No-confab 10/10 abstain.
DIAGNOSIS (from numbers): grounding + cross-lingual + no-confab WORK; structure
exact-match 0 and decoder reconstruction weak. Round-trip (0.32) ~= decoder-
from-gold (0.36) => the DECODER is the round-trip bottleneck, not the encoder's
imperfect trees.
LEAD IDEAS (2026-09-05):
1. Decoder: more training + MEMORY-side realization signal -- store each saved
   clause-tree's PARENT RAWTEXT in the memory structures so the realizer learns
   surface/"dialectic" patterns (meaning->how-it-was-said). Retrieval-augmented
   realization; keep it from degenerating into memorizing training sentences.
2. structure recall 0 while sense/slot 0.93/0.96 seems inconsistent -> but it's
   PER-SITE (0.93/0.96) vs WHOLE-TREE-EXACT (0). Lead's key point: there ISN'T
   a single "right" tree -- the encoder emits a candidate FOREST (candidates-
   first); exact-match-to-one-gold-tree is likely the WRONG metric. Right
   metric = gold-tree-in-forest SET recall + soft clause-boundary/attachment F1.
   Don't add a loss that collapses to one tree (breaks candidates-first).
NEXT: diagnosis routine loads the checkpoints -> soft structure metrics + set
recall + example trees + decoder error breakdown, to tell close-vs-broken.

### DIAGNOSIS: encoder structure-0.000 is REAL; 93/96 sense/slot was a MIRAGE (2026-09-06)

Ran scripts/diagnose_colab_ckpts.py on the Colab run-1 checkpoints (98-rec
held-out split reproduced exactly; branch encoder-diag-logs:runs/diag_output.txt).
FINDINGS (decisive):
- [C] edges-off: 98.5% of gold clauses are 4+ edges from the nearest emitted
  clause; 0 exact, 0 one-off. NOT metric strictness -- structure genuinely broken.
- Emitted trees are OVER-ATTACHMENT SOUP: single clause w/ 20-50 role edges,
  same token under SUBJECT+OBJECT+PLACE, predicates = punctuation/<null>/function
  words. [D] predicate token_index agreement = 17%.
- WHY 93/96 looked good: sense/slot are RECALL-ONLY ("token i grounded as sense
  anywhere in forest?"). A dump-everything forest trivially covers every token ->
  ~1.0 recall regardless of correctness. Node-recall WITH relation label = 0.657;
  clause-exact = 0.000. Model beats random (0.04) by DUMPING, not by grounding.
  We skipped CLAUDE.md's cheat-baseline-at-floor law for the encoder.
- ROOT CAUSE (hypothesis): STOP/REDUCE actions never learned to fire (rare in
  oracle -> under-weighted by plain CE) + severe undertraining (50 ep CPU).
- DECODER: reconstruction FROM GOLD tree content=0.386 function=0.404 (both bad,
  equal) + repetition collapse ("let let let", "three three three"). Can't copy
  from a perfect structure yet => memory/retrieval-frame idea PREMATURE.
REVISED PLAN (supersedes "encoder works, move to decoder+memory"):
  1. metric fix: add PRECISION + dump-everything cheat baseline (re-score
     existing ckpt, no retrain) -- non-negotiable gate.
  2. loss/decode fix: make it commit+stop (up-weight terminal/REDUCE / penalize
     over-attach).
  3. THEN long Colab run (fixing 1-2 first; longer training on broken objective
     just overfits dumping).
  4. decoder: more training/capacity vs gold until copy-from-structure works;
     THEN the memory frame.
STATUS: "encoder built+trained (0.93/0.96)" claim RETRACTED -- that was recall
inflated by over-generation. Encoder structure competence is the real blocker.

### ENCODER-FIX-V2 built + merged (2026-09-06, mainline 060550d)

The mirage is now QUANTIFIED (scripts/rescore_encoder.py, runs/rescore_output.txt,
n=98 EN test), model vs baselines:
  MODEL: sense 0.935 slot 0.966 | edge_precision 0.100 edge_recall 0.644 overgen 8.12x
  DUMP : sense 1.000 slot 0.885 | edge_precision 0.016 edge_recall 0.196 overgen 9.77x
  RANDOM: sense 0.031            | edge_precision 0.010 edge_recall 0.010 overgen 0.06x
=> model overgen (8.12) sits right beside the dump cheat baseline (9.77); edge
precision 0.10 (90% of emitted edges spurious). The old 0.93/0.96 was recall
inflated by ~8x over-generation, NOT structure. BUT model edge_recall 0.644 >>
dump 0.196 and random 0.010 => it DID learn which edges are plausible; it just
never learned to stop. Fixable shape.
FIXES SHIPPED (7 commits, 41 tests green, smoke exit 0):
  - edge_precision/edge_recall/overgen_ratio (best-tree EVAL view; forest never
    collapsed -> candidates-first preserved) + policy="dump" cheat baseline.
  - class-weighted CE up-weighting STOP/CLOSE_CLAUSE via --terminal-weight
    (default 4.0; lib default 1.0 = backward-compatible) -- the commit/stop fix.
  - confidence-gated decode --commit-margin (commit top-1 unless 2nd within
    margin) = the lead's "give a set only when unsure".
  - colab_train_all.py + notebook run-2 defaults: --enc-epochs 300, --dec-epochs
    200, --dec-d-model 96, dump baseline in report, git-push cell -> branch
    colab-trained-ckpts-v2. Notebook clones encoder-fix-v2 (kept alive).
GATE for run-2 SUCCESS: model edge_precision must rise WELL above dump's 0.016
and overgen_ratio must fall toward 1.0 (currently 8.12). Recall alone proves
nothing now.
NEXT: lead runs colab/Train_Encoder_And_Decoder.ipynb (run-2). Then evaluate
precision/overgen. If terminal-weight+epochs insufficient, escalate to mask
tightening or GPU-path (encoder is CPU-only; 300 ep may be slow on Colab).

### RUN-2 RESULT + ROOT CAUSE CONFIRMED: it's the MASK, not loss/capacity (2026-09-06)

Colab run-2 (100/100 ep, terminal-weight 4.0, dec d_model 96) FINAL RESULTS:
  encoder model: edge_precision 0.111 (run-1 0.100) overgen 7.50 (run-1 8.12)
                 -> FLAT. terminal-weight loss fix did essentially nothing.
  decoder recon-from-gold token_f1 0.359 (run-1 0.360) -> FLAT despite 2x
                 d_model + more epochs. Decoder ceiling is NOT capacity.
  no-confab 10/10 abstain; Spanish sense 1.000 (transfer holds).
Both "train more/bigger" levers FALSIFIED. ROOT CAUSE found at line level
(src/nsm_ct/encoder_model.py):
  - _apply_action (~L826): GROUND/EMIT_UNRESOLVED_SLOT only advance the buffer
    pointer when state.i < T; at i>=T they produce token_index=None nodes and
    do NOT advance.
  - legal_action_types keeps GROUND legal at i>=T (docstring: for rare
    duplicate-token gold). => once the buffer is consumed, the policy emits
    UNBOUNDED GROUND -> <null>#None phantom nodes (the spam that dominated every
    emitted tree; gold ~6 edges, model best-tree ~45, ~35 are null phantoms).
  - This is why terminal-weight failed: CLOSE can't win when the junk action is
    legal & free every step.
FIX (gold-safe): make GROUND illegal at i>=T. The oracle (linearize_tree) NEVER
GROUNDs content at i>=T (every content node consumes a fresh advancing token),
so this can't mask a gold action -- guarded by the full-corpus oracle-legality
test. Structurally caps tree size ~ token count -> overgen toward 1.0.
IN FLIGHT: no-retrain confirmation routine (trig_018pGHLKSHdcea1eJqH8eJcJ, branch
encoder-mask-fix) re-scores the run-2 ckpt under strict_ground mask + proves
oracle-legality + breaks down null-node spam by gtype (GROUND vs EMIT_SYNTH
prime vs EMIT_UNRESOLVED reference/elision -- the latter two aren't killed by
strict_ground and may need a bound too). If overgen drops toward 1.0, bake the
mask fix in + retrain (training mask must match decode mask).

### CORRECTION: strict_ground fix FAILED + was NOT gold-safe (2026-09-06)

The proposed "GROUND illegal at i>=T" fix (prev entry) is WRONG on both counts
(re-score on run-2 ckpt, branch encoder-mask-fix, runs/mask_fix_output.txt):
  - NO EFFECT: model overgen 8.27 (loose) -> 8.20 (strict), edge_precision
    0.101 -> 0.101. Did nothing.
  - NOT GOLD-SAFE: oracle-legality check FAILED -- 933 gold GROUND actions
    (0.96% of 97530 steps) genuinely land at i>=T via the duplicate-token-index
    collision (e.g. rec6: token_index=59, i=59, T=58). The docstring was right;
    my "oracle never grounds at i>=T" claim was wrong. This fix would mask gold
    actions -> +inf loss. DEAD.
  - Null nodes are only 16.3% of emitted (prime/EMIT_SYNTH_SLOT dominates them
    at 82% of nulls, but nulls overall are minor). Even zero nulls -> overgen
    ~6.9. So the 8x over-generation is REAL-TOKEN over-emission, mechanism NOT
    understood. ~120 emitted nodes/tree vs a handful gold; math doesn't close
    with a monotonic-advance-only GROUND, so the transition dynamics aren't what
    the code reading suggested.
TWO fix hypotheses now falsified (loss-weight; strict_ground). STANCE: stop
theorizing -> AUDIT emitted trees first (trig_01H3Vczy1FvXYj5fAHvH8pHR, branch
encoder-audit-logs): action histogram (does it ever STOP or hit max_clauses=20
cap?), tree shape vs gold, token-index duplication, 8 full example trees. Design
fix #3 from THAT data, not theory.

### ROOT CAUSE FOUND: beam_decode ALIASING BUG (not the model at all) (2026-09-06)

Decode audit (branch encoder-audit-logs:runs/audit_output.txt) revealed the real
mechanism -- a Python object-aliasing bug in beam_decode, NOT model/mask/loss:
  - Winning beams take SHORT, correct action sequences (100% terminate via real
    STOP; e.g. "she answered herself" -> OPEN,EMIT_UNRESOLVED,GROUND,GROUND,
    CLOSE,SHIFT,STOP = a clean parse) BUT emit HUGE trees (that 7-action beam
    produced a 32-node tree). Mathematically impossible without pollution.
  - THE BUG (encoder_model.py ~L739): on beam fork, child BeamState uses
    `cur_clause=b.cur_clause` -- the SAME open-clause dict + roles LIST shared
    across all sibling beams. Every sibling's GROUND/EMIT appends into ONE shared
    roles list. Identity check: sharing factor 2.65, largest shared roles list
    316 nodes. Winning tree = union of many beams' roles.
  - This ALONE produced: overgen ~8x, edge_precision 0.10, structure_recall 0,
    inflated sense recall, token dup up to 16x, 95.6% within-clause collisions.
  - Training is TEACHER-FORCED (single trajectory, no branching) -> NEVER
    affected. That's why the loss curve was healthy and real.
CONSEQUENCE: ALL prior decode metrics are ARTIFACTS. The "encoder over-generates /
is broken" verdict (run-1 AND run-2) is WRONG -- it was beam pollution. The three
"failed fixes" (terminal-weight loss, strict_ground mask, decoder capacity) all
did nothing because none touched the actual bug. The encoder's true quality is
UNMEASURED and may be good.
FIX (decode-only, no retrain): deepcopy cur_clause per fork. Branch
encoder-aliasfix (trig_01CVABFD6npLqmCGb84GnWk4): fix + regression test (tree
node count <= winning-beam emit-action count) + re-score run-2 ckpt + round-trip.
Awaiting: does overgen collapse toward 1.0 and edge_precision/structure jump?
LESSON: validate the eval harness before concluding the model is broken (CLAUDE.md
"perfect-looking results get held-out tests" cuts both ways -- so do terrible ones).

### RESOLVED: aliasing fix -> encoder WORKS. Decoder is the real bottleneck (2026-09-06)

Re-score of the SAME run-2 checkpoint under FIXED beam_decode (deepcopy cur_clause
per fork; mainline 2004059; branch encoder-rescore-logs:runs/rescore_fixed_output.txt):
  ENCODER held-out test (n=98)   CORRUPTED -> FIXED:
    edge_precision   0.101 -> 0.700
    overgen_ratio    8.27  -> 1.082   (near-perfect edge count)
    structure_recall 0.000 -> 0.212   (whole-tree EXACT)
    edge_recall      0.648 -> 0.695
    sense_recall     0.921 -> 0.787   (honest now; 0.92 was over-gen inflation)
    slot_recall      0.963 -> 0.795
  vs DUMP (prec 0.016, overgen 9.77) -> model now crushes the cheat baseline.
  SPANISH grammar-swap (zero ES training, n=208): edge_precision 0.756, overgen
    0.93, structure_recall 0.139, sense 1.000. Cross-lingual grounding transfer
    REAL + strong under clean metric.
VERDICT: the encoder was WORKING the whole time. The "over-generation / structure
broken" verdict (run-1 AND run-2) was 100% the beam_decode aliasing artifact.
Three fix hypotheses (terminal-weight loss, strict_ground mask, decoder capacity)
all no-op'd because none was the bug. The lead's original instinct ("0% can't be
right at 93/96") was correct -- it was a harness bug. ~0.70 precision/recall from
a 343K-param sub-MB parser + zero-shot cross-lingual = the I/O encoder thesis
holds.
REMAINING: the DECODER. Round-trip token_f1 0.35, output garbled even fed the
FIXED (clean) top tree; decoder-from-GOLD-tree 0.36 was never aliasing-affected,
so this limitation is REAL. This is where more decoder training + the lead's
memory-frame idea (store parent rawtext, retrieval-augmented realization with a
copy-from-structure gate) actually apply -- now measurable on clean structure.
NEXT: (1) encoder is essentially done for now; (2) focus shifts to decoder --
retrain/redesign the realizer against clean structure; consider the memory frame.

### TRAINING-PATH AUDITS: encoder CLEAN, decoder has a real bug (2026-09-06)

ENCODER (branch encoder-lossaudit-logs): training/loss path SOUND. Teacher-forced
next-action accuracy 0.888 overall (STOP 1.000, CLOSE_CLAUSE 0.816); every CE
term correct; mask leak-free; metric arithmetic hand-verified (PASS). 0.888 TF
vs 0.70 fixed-decode edge-F1 = normal exposure bias. No loss bug. Encoder DONE.

DECODER (branch decoder-lossaudit-logs): the lead's "logic bug in training"
instinct CONFIRMED, deeper than "too binary":
  - Teacher-forced token acc TRAIN 0.787 vs DEV 0.160 (0.63 gap) = MEMORIZATION.
  - ROOT: build_function_vocab (decoder_trained.py:113) mines EVERY surface token
    not covered by a structure node -> 2047 entries FULL OF CONTENT WORDS
    (bumped, shillings, pope, crown, hungry, inhabited...). The decoder generates
    content from a memorized list instead of copying from structure -> memorizes
    (train) + confabulates (dev: "gingerbread boy"). This is a NO-CONFAB HOLE:
    the function head can emit content absent from the structure.
  - DEEPER CAUSE (the real finding): of 24473 target tokens only 5953 (24%) are
    copyable from structure; 76% are "gaps". The grounded structure is a
    PRED-ARG SKELETON covering ~1/4 of surface words. Exact-surface reconstruction
    by copy is impossible -> function-vocab was papering over it by memorizing.
  - "too binary" exact-position CE is SECONDARY: lemmatized token-F1 (0.352) ~=
    raw (0.345), so output is genuinely wrong content, not synonym/order harshness.
  - Minor: function_vocab built over train+dev combined (vocab leak).
DESIGN FORK (LEAD DECISION, architecture-level): (A) richer structure (ground
more content; more gold) / (B) memory-frame retrieval for the 76% gap (lead's
idea; copy-gated) / (C) honest partial realizer: curate function_vocab to a true
closed class (~100 fn words), decoder realizes only copyable + fn words, stops
confab, measures the HONEST ceiling of what current structure can reconstruct.
RECOMMENDATION: C first (cheap, no new data, fixes confab hole, its ceiling number
decides B vs A). Awaiting lead's pick before building.
