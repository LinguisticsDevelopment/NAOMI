
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
