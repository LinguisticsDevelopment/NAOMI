
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
