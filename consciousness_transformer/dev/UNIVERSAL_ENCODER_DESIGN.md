# Universal Grounded Encoder / Decoder — design v1 (for red-pen)

Status: DRAFT for lead review. Nothing built yet. This supersedes the
"train on converted prose with clause-holdout QA" approach (see postmortem
§0). Author: director. Date: 2026-09-02.

## 0. Postmortem that motivates this (why we pivot)

The first full prose-training run (M61, branch `prose-training-run1`) moved
held-out accuracy 0.579 -> 0.622 (+0.043, inside the noise band at n~38).
Two root causes, and the second is the important one:

1. **Lossy I/O confounded the experiment.** The deterministic parser
   dropped 22/256 training + 5/42 eval episodes (over-length / cap hits),
   and the wall-clock cap made the eval set non-deterministic (BEFORE n=38
   vs AFTER n=37). A wrong answer could be a reasoning failure OR a parse
   failure — unmeasurable.
2. **The training objective was antithetical to the thesis.** "Hold out a
   clause, predict the missing slot" IS masked language modeling — BERT's
   objective. It rewards distributional priors ("what word usually fills
   this slot"), i.e. the transformer/statistical-LM behavior the whole
   project exists to avoid. We were training an anti-confabulation
   architecture to imitate confabulation-by-statistics.

Corrected direction (lead, 2026-09-02):
- **Fix the objective:** read a text -> answer an EXPLICIT question about
  it, with "not stated / I don't know" a first-class, rewarded answer.
- **Fix the data:** real, graded, K-12-style teaching material, bottom-up
  (sentences -> passages -> long reading). Do NOT fabricate a corpus; use
  real graded readers with real comprehension questions. Real prose is the
  TRANSFER EXAM, not the trainer.
- **Fix I/O first** so the cognition layer trains against clean signal.
  This doc specs that I/O layer.

## 1. The invariant (the line that keeps this NAOMI, not an LLM)

The learned pieces are **boundary transducers only**. Knowledge and
cognition stay in the **symbolic grounded middle** — the inspectable
tensor memory, USVS senses, NSM primitives, fixed TPR ops. Concretely:

- Encoder weights hold *transduction policy* (surface -> grounded tree),
  never knowledge. Decoder weights hold *realization policy* (grounded
  answer -> surface), never knowledge.
- **Every answer MUST route through a memory read.** No encoder->decoder
  shortcut. The moment the decoder can answer from encoder activations
  without going through grounded memory, we have built a small transformer
  and lost the thesis. This is a hard architectural gate, testable by
  ablation (sever the memory path -> answers must collapse).
- The encoder's OUTPUT is the same inspectable grounded tree/episode schema
  as today, so auditability is preserved even though the mapping is neural.

## 2. The Universal Grounded Encoder

### 2.1 What it is
ONE neural composer that replaces N brittle per-language deterministic
parsers. It is not "a parser for English." It is a parser conditioned on
the universal symbolic inventories the mind already holds.

### 2.2 Inputs (per sentence)
1. **Surface text** (raw tokens, any language, possibly code-switched).
2. **Retrieved USVS sense candidates** for the tokens present — every
   plotted sense across ALL languages that the tokens could ground to.
   The USVS is already language-agnostic (a sense is a sense regardless of
   the word/language expressing it; "dog"/"perro"/"chien" land in the same
   grounded region), so retrieval spans all languages by construction.
3. **Fired grammar rules** — the grammar-rule inventory across ALL
   languages, filtered to the rules whose triggers actually match the
   input, as candidate operations.

CRUCIAL tractability point: we do NOT feed the entire multilingual
dictionary + full rulebook every forward pass (astronomically large). We
**retrieve** the candidate senses for the tokens present and the rules that
fire, and the model composes over that bounded candidate set. Functionally
"conditioned on all languages"; computationally bounded to the sentence.

### 2.3 Output
A **grounded tree** (the episode structure): nodes = grammar-rule
applications, leaves/slots grounded to specific USVS senses. Same schema
the deterministic parser + input_encoder produce today.

### 2.4 Mechanism
A retrieval-augmented, **grammar-constrained** neural transition/chart
parser. Action space = applying candidate grammar rules; grounding =
selecting candidate USVS senses. Grammar-constrained decoding means the
model *cannot emit ill-formed structure* — the grammar defines the legal
output space; the net only chooses within it. Robustness + well-formedness
together.

### 2.5 Why code-switching ("Spanglish") falls out for free
"Which language" is never a hard gate. Every token grounds into the same
USVS; every language's rules are candidates. A code-switched sentence is
just a tree where some spans applied Spanish rules and some English, all
grounding into one shared meaning space. No monolingual parser can do this;
a universal composer conditioned on the union of inventories can.

## 3. Training plan — monolingual teachers, multilingual student

The elegant part: **the per-language deterministic parsers are the
teachers.** Each manufactures unlimited gold trees on clean monolingual
text (no human annotation). The neural student trains on the UNION of all
teachers, then generalizes to the thing no teacher can produce:
code-switched input.

Staging, each stage checkable except the last:
- **Stage i — distill (hard gold):** neural encoder matches the English
  deterministic parser on clean text. Metric: tree-match / grounding-match
  vs teacher. Gate: high agreement on held-out clean sentences.
- **Stage ii — bilingual (hard gold):** add Spanish rules+senses; match the
  Spanish parser too. Still hard gold, now bilingual. Gate: high agreement
  on both, no English regression.
- **Stage iii — code-switch (no oracle):** Spanglish falls out. NO teacher
  produces gold here, so eval leans on **grounding-consistency / round-trip
  checks** (does the tree ground to a coherent USVS meaning? does it
  back-translate?) plus a small human-checked code-switch set. This is the
  demo no one else can do.

Honest hard parts (flip side of the payoff):
- Sense disambiguation across ALL languages at once is harder (more
  candidate senses per token with no language pre-commit) — but that is
  exactly where a learned model beats the brittle deterministic one, so it
  is where the encoder earns its existence.
- The code-switch case has no gold oracle; verification is by grounding
  consistency, not label match. Named here so it shapes eval from day one.
- Edge-generalization to a NEW language depends on training on a
  **typologically diverse** language set (so the model learns the general
  skill of applying-rules-to-ground-text, not just Indo-European patterns).

## 4. The Decoder (start rule-grounded, learn later)

- **Phase one — rule-grounded realizer:** memory read -> NSM-primitive
  answer structure -> grammar run FORWARD as a deterministic surface
  realizer. Because it can only realize content bound in memory,
  no-confabulation holds by construction. Covers short answers
  (who/what/where/yes-no/"not stated").
- **Phase two — learned realizer (later, its own milestone):** for long-
  form output. This re-opens the confabulation door unless strictly
  constrained (copy-from-memory / grammar-constrained decoding, NOT free
  sampling). Deferred; do not let it gate the encoder or the short-answer
  ladder. Single biggest honesty risk in the whole pivot — prove the
  constraint before making it learned.

## 5. What this unblocks — the K-12 ladder (separate doc to follow)

With clean, faithful grounded I/O, cognition trains against clean signal:
- Objective: read-then-answer comprehension QA, abstention first-class.
- Data: real graded readers, bottom-up. Candidate real sources (public /
  research-available, real questions, real difficulty ordering): McGuffey
  Readers (public domain, First->Sixth Reader, sentence-level up), RACE
  (middle/high-school English-exam RC), ARC (grade 3-9 science). Vet each
  for questions whose answer flows through comprehension, not priors (the
  curriculum design law, applied to found data).
- The parser/encoder competence PACES the ladder: advance to grade N only
  when I/O is clean at grade N's syntax. One ladder paces both policy
  learning and I/O hardening.

Calibration (see §7): clean I/O makes the FOUNDATIONAL grades much easier
than a from-scratch neural QA model (the input is already structured
meaning), AND makes the whole thing measurable — the ladder cleanly reveals
which reasoning skill is present/absent. It does NOT guarantee the upper
grades (coreference, negation/abstention, multi-hop, causal); those remain
the real test of whether a tiny GRU policy acquires compositional
reasoning. Clean I/O turns "we can't tell why it failed" into "we can see
exactly which grade it's stuck at." That legibility is the win.

## 6. ROADMAP ITEM — self-editable grammar / edge language acquisition

Consequence of resource-conditioning (§2.2): because the grammar-rule
inventory is INPUT, not baked-in weights, adding a new language = adding
rules+senses to the inventory; a robust encoder composes with them WITHOUT
retraining. That is near-trivial *for languages whose constructions are
typologically similar to trained ones*. "Self-editable" extends this: when
the machine meets text it cannot ground well, it can HYPOTHESIZE a new
rule, test whether it improves grounding-consistency, and commit it —
runtime grammar induction.

Why this stays honest/on-thesis: grammar rules are symbolic and
inspectable, so a self-edited rule is auditable — you can see exactly what
rule was added and why, and validate it against grounding-consistency
before committing. Not scary black-box self-modification; inspectable,
testable rule proposal gated by a grounding check.

Honest limit: a typologically NOVEL construction (ergativity,
evidentiality, heavy agglutination, non-configurational order) may still
need examples, not just a spec — "trivial" holds strongly for near-neighbor
languages and degrades with typological distance. Maximize the property by
training on a typologically diverse language set.

Placement: AFTER the encoder/decoder are robust (Stages i-ii proven). Added
to roadmap now so §2 is designed with rule-inventory-as-input from day one
(which it already is), keeping the door open.

## 7. Open questions / decisions for the lead

1. Encoder backbone: transition-based (incremental, cheap, streaming) vs
   chart/graph (global, heavier). Lean transition-based for speed +
   streaming, matching the incremental clause reactor.
2. How much teacher gold does grade-1/2 real text actually yield? -> the
   probe firing now answers this (data-volume feasibility).
3. Confirm the decoder stops at short grounded answers for phase one
   (recommended), long-form its own later milestone.
4. Confirm the invariant (§1) as a hard gate we will ablation-test.

## 8. Staged milestones (proposed)

- M62: teacher gold-volume probe on real graded text (IN FLIGHT).
- M63: encoder Stage i — distill English deterministic parser, hard-gold
  agreement gate.
- M64: encoder Stage ii — add Spanish, bilingual gold gate, no EN
  regression.
- M65: rule-grounded short-answer decoder + the §1 memory-bypass ablation.
- M66: code-switch demo (grounding-consistency eval).
- M67+: K-12 read-then-answer ladder on clean I/O; then self-editable
  grammar; then learned long-form decoder.
