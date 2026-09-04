# Encoder-Reference Grammar Format — CANONICAL (candidates-first)

Status: CANONICAL. Author: encoder step 3 pass, 2026-09-04. Reconciles the two
hand-draft proposals (`ENCODER_GRAMMAR_FORMAT_PROPOSAL.md` on branches
`encoder-handgold-v1` and `encoder-handgold-v2`) into one declarative rule
formalism, updated to the **candidates-first world** locked by the CORE
BOUNDARY DECISION and the Appraisal grounding DECISION (both lead, 2026-09-04).

Read `dev/ENCODER_IO_CONTRACT_V2.md` first — this format's only job is to
*license* the structures that contract serializes, and every `licenses`
fragment below is written in its vocabulary. Read `dev/UNIVERSAL_ENCODER_DESIGN.md`
§2.4 (action space) and §11 (the pivot) for how a grammar becomes an encoder
*input* rather than a program.

This document describes a data/interface shape only. It does not modify, and is
not itself, any `src/` model, parser, or grammar-file code.

---

## 0. The one idea (why the format changed twice)

**First shift (both drafts): executable → declarative.** The current grammar
file is an *executable* deterministic-parser format — anchor/before/after
token-pattern rules that fire and rewrite. It is a program the `quantum_parser`
runs to *produce* a parse. The universal encoder (design §2.4) does not run
that program; it is a retrieval-conditioned, grammar-constrained neural
transition parser whose grammar is an **INPUT that conditions** its action
choices. Authoring the hard cases proved three things the executable format
cannot state: **soft/broad licensing** ("a standalone ADV *can* act as an
imperative"), **synthesized constituents** ("an imperative *implies* a
subject with no surface token"), and **pointing at prior context** ("a
fragment *inherits* its predicate from a node in prior discourse"). So the
grammar becomes a **declarative reference inventory**.

**Second shift (this doc): committed → candidates-first.** The prior drafts
still wrote rules that *chose* — a sense-selection bias ("prefer the VERB sense
of `look`"), an appraisal node with a *derived valence*, a reaction-sense
lookup that resolved an interjection to `annoyance.n.01`. Per the CORE BOUNDARY
DECISION the encoder now **emits candidates and commits to nothing** (contract
§0). A grammar rule therefore **licenses structure and emits candidate/unresolved
slots**; it **never disambiguates**. Every place a draft rule picked, a
canonical rule instead emits the whole candidate set and hands the pick to
comprehension. That single constraint drives every change below.

---

## 1. The rule-record schema

**Chosen schema (v1's record, kept):**

```
rule:
  id:         <string>                        # stable identifier
  family:     mood | ellipsis | attachment | core   # (stance is GONE — see §3)
  soft:       <bool>                          # true = broad/pattern license, never forced
  trigger:                                    # declarative match — a predicate over the span/context
    pos_any:        [<POS>, ...]              # coarse POS from the frozen quantum_parser enum
    lexset:         <named lexical class | null>
    position:       utterance_initial | fragment_whole | pre_clause | null
    surface_absent: [<role|PREDICATE>, ...]   # positions with NO surface token in the span
    context_present:<bool | null>             # a prior discourse clause exists
  licenses:                                   # the structure permitted, as a partial grounded-tree
    utterance_kind: proposition | imperative | interjection   # contract §3; NO "appraisal"
    fragment:       <partial clause in ENCODER_IO_CONTRACT_V2 vocabulary>   # the emitted sub-tree
  fill:                                       # per non-surface / unresolved slot: the grounding it emits
    <role|PREDICATE>: prime(<PRIME>)          # -> grounding.type:"prime"      (resolved)
                    | unresolved(source)      # -> grounding.type:"reference"|"elision" (candidate set)
                    | sense                   # -> grounding.type:"sense"      (candidate set)
  action_map: [<encoder-action>, ...]         # §4: which candidates-first actions this conditions
  confidence: high | soft                     # authoring-observed reliability of the license
```

**Why this schema, not v2's `{when, licenses, emits, ground}`.** v1's record is
strictly more expressive for the encoder we are building and needs one fold-in,
not a rewrite:

- v1 already carries the two fields the encoder training loop needs and v2
  dropped: **`soft`/`confidence`** (the whole point of the first shift — a rule
  that is *available*, not *forced*) and an explicit **`action_map`** onto the
  design §2.4 action space (the interface the transition head consumes).
- v2's genuinely better idea is that a rule's licensed structure is a **partial
  grounded-tree fragment written in the I/O-contract's own vocabulary**, so the
  union of fragments *is* the grammar-constrained legal output space (design
  §2.4). That is folded in as `licenses.fragment` — it does not need a separate
  top-level `emits` field.
- v2's `ground` channel is folded into `fill`: each slot names the *grounding
  type* it emits (`sense` / `prime` / `unresolved`), which in candidates-first
  is exactly the contract's unified `grounding` construct (contract §4). v2's
  third channel, `reaction-inventory`, is **deleted** (§3).

Net: **one schema, v1's fields, with v2's contract-vocabulary fragment and the
grounding-type channel folded into `licenses` and `fill`.**

`soft:true` rules are the ones no executable format can express: the net treats
them as *available* actions conditioned on the trigger and learns *when* they
belong from teacher-plus-hand gold — they never force an output, and (§5, §6)
several soft rules may fire on the same span, each contributing a branch to the
candidate forest.

---

## 2. The candidates-first invariant every rule obeys

> **A rule licenses structure and emits slots. It never chooses a filler.**

Concretely, three hard prohibitions that separate this format from both drafts:

1. **No sense pick, no sense bias.** A content position always emits
   `grounding.type:"sense"` with the **full** `senses_of(token)` candidate set
   (contract §4.2). The imperative rule does **not** say "prefer the verb sense
   of `look`" — it emits every sense of `look` and marks the clause imperative;
   comprehension, reading memory, picks the verb sense (contract §8.1 logic).
2. **No antecedent pick.** An elided/pronominal position emits
   `grounding.type ∈ {elision, reference}` with a **candidate antecedent set**
   and a `retrieval` envelope (contract §4.1/§4.3); it binds to none.
3. **No valence, no appraisal, no reaction sense.** Deleted wholesale (§3).

The only *resolved* grounding a rule may emit is `grounding.type:"prime"` — and
only where the rule licenses **exactly one** possible filler (the imperative
addressee = prime `YOU`), so there is nothing to select (contract §5).

---

## 3. What was STRIPPED — the interjection simplification

Per the **Appraisal grounding DECISION**, the appraisal machinery in *both*
drafts is removed. **Deleted:**

- v1's `family:stance`, rules **R2** (`stance.interjection_appraisal`) and
  **R3** (`stance.reaction_attaches_to_prior`), the `clause_kind:appraisal`,
  the `node:appraisal` emit, the `from_prime(I)` experiencer, the
  `GROUND_STANCE` action, and the `from_stance_lexicon` fill channel.
- v2's `clause_type:"appraisal"`, the `predicate:"feel"` / `primes:[I,FEEL]`
  explication, the `EXPERIENCER`/`$SPKR` synthesized speaker, the
  `appraisal:{trigger, reaction_sense_id, valence:"from_sense", scope}`
  sub-object, the **reaction-sense inventory**, and the `reaction-inventory`
  ground channel.

**Replaced by one ground-only rule (R3 below).** An interjection is just a
content utterance that grounds like any other word:

- It emits `grounding.type:"sense"` over its **literal USVS sense candidate(s)**
  — `"shit!"` → `senses_of("shit")`, which already contains the target, so no
  second index is needed.
- **Pure interjections** with no literal synset (`alas`, `ugh`, `oh`, `oops`)
  get **gloss-derived senses added to USVS** by the existing
  `ground/usvs.py` gloss→prime→coordinate pipeline — deterministic lexical
  data, not a trained model — after which they ground like any sense.
- The clause is marked `utterance_kind:"interjection"` (contract §3/§6) so
  downstream code knows the speech-act shape. **That marker plus the grounded
  sense candidates is the entire encoder output for an interjection.**

Connotation — "how does the speaker *feel*?", GOOD↔BAD valence, projecting it
onto a target — is **comprehension-side** (a learned read off the USVS eval
axis, per contract §6/§10), out of scope for the grammar. This also dissolves
v2's biggest open question (its §5): with literal/gloss grounding the target
**is** in the token's candidates, so the "appraisals need a separate retrieval
index" problem no longer exists.

---

## 4. The candidates-first encoder action space (design §2.4)

Design §2.4: "action space = applying candidate grammar rules; grounding =
selecting candidate USVS senses." In candidates-first, "selecting" is replaced
by "emitting the candidate slot." **Four actions, none of which disambiguate:**

| action | what it emits | contract grounding | commits? |
|---|---|---|---|
| `OPEN_CLAUSE(kind)` | starts a clause with `utterance_kind = kind` | — (clause skeleton, §3) | licenses a clause; picks nothing |
| `GROUND(token)` | a **sense-candidate slot** — the full `senses_of(token)` set | `type:"sense"`, `retrieval.source:"lexicon"` | **no** — emits the set, never the pick |
| `EMIT_SYNTH_SLOT(role)` + `GROUND_PRIME(P)` | a filler with **no surface token**, grounded to NSM prime `P` | `type:"prime"`, `candidates:null` | resolved — the rule licenses one filler, nothing to select |
| `EMIT_UNRESOLVED_SLOT(role\|predicate)` | an **unresolved slot** — a candidate antecedent set + a `retrieval` envelope | `type:"reference"` (pronoun/coref) or `"elision"` (elided pred/arg), `retrieval.source ∈ {self, context, memory}` | **no** — emits candidates, binds none |

Deletions from the drafts' action lists, forced by candidates-first:

- **`GROUND_USVS_SENSE` (v1) / grounding-selection bias (v2) → `GROUND`.** The
  old action *picked* one sense from the retrieved set; the new `GROUND` emits
  the whole set as a slot. Same retrieval, no argmax.
- **`GROUND_STANCE` (v1) — deleted** with the whole stance family (§3).
- **`EMIT_CONTEXT_REF` (v1) / `context_ref` pointer (v2) → `EMIT_UNRESOLVED_SLOT`.**
  The drafts modeled context-pointing as an elision/coref-only bolt-on. In
  candidates-first it is the **universal unresolved-slot primitive** (contract
  §4): the *same* action emits a pronoun's candidate antecedents, an elided
  predicate's candidate antecedents, and (with `retrieval.source:"lexicon"`) is
  the shape `GROUND` itself specializes. One construct, three sources.

**Structural / attachment ambiguity is NOT an action.** It lives in the forest
(contract §2.1): when two attachment analyses of a span are both licensed, the
parser emits **two trees**, and the encoder emits their union. No rule and no
action picks between them (§6, R7).

The design symmetry the drafts noted now reads cleanly, because *nothing
resolves*:

> `GROUND` = emit the retrieved **sense**-candidate set as a slot.
> `EMIT_UNRESOLVED_SLOT` = emit the retrieved **memory/context**-candidate set as a slot.

These are the *same* action over different retrieval indices (lexicon vs.
grounded memory) — which is why "every answer routes through a memory read"
(design §1) and "point to prior context" are one mechanism, and why the encoder
can stay a pure transducer: it emits candidate slots; comprehension does the
single `select-from-memory-conditioned-candidates` read over all of them.

---

## 5. The finalized rules (7, candidates-first)

Meta-variables: `$ADDR` = synthesized addressee, grounded to prime `YOU`;
`@unresolved(source, method)` = an `EMIT_UNRESOLVED_SLOT` emitting a
`grounding` with `type ∈ {elision, reference}`, a candidate antecedent set,
and the given `retrieval`. Fragments are partial clauses in
`ENCODER_IO_CONTRACT_V2` vocabulary.

### R1 — imperative: standalone verb licenses addressee subject  (family: mood)
```
id: imperative.synth_subject
family: mood
soft: true
trigger: { pos_any: [VERB], position: utterance_initial, surface_absent: [SUBJECT] }
licenses:
  utterance_kind: imperative
  fragment: { predicate: <tok0>,
              roles: [ { relation: SUBJECT, word: "you", token_index: null, is_entity: true } ] }
fill:
  PREDICATE: sense          # emits senses_of(tok0) — the FULL set; no verb-vs-noun bias
  SUBJECT:   prime(YOU)     # exactly one filler => resolved, no candidate slot
action_map: [ OPEN_CLAUSE(imperative), GROUND(tok0), EMIT_SYNTH_SLOT(SUBJECT), GROUND_PRIME(YOU) ]
confidence: high
```
"Look!", "Sit down!", "Open your books." Note the change from both drafts: the
predicate emits **every** sense of `look` (verb *and* noun), not a
verb-preferring pick — comprehension resolves it against the imperative mood +
memory. The synth subject is the one resolved grounding (contract §5).

### R2 — standalone ADV/ADJ as imperative with elided predicate  (family: mood, soft/broad)
```
id: imperative.bare_adverb
family: mood
soft: true
trigger: { pos_any: [ADV, ADJ], position: fragment_whole, surface_absent: [PREDICATE], context_present: true }
licenses:
  utterance_kind: imperative
  fragment: { predicate: null,
              predicate_grounding: @unresolved(context, elision_inherit_predicate),
              roles: [ { relation: SUBJECT, word: "you", token_index: null, is_entity: true },
                       { relation: <adv_role>, word: <tok0> } ] }
fill:
  PREDICATE:  unresolved(context)   # "an imperative implies a predicate from prior context"
  SUBJECT:    prime(YOU)
  <adv_role>: sense                 # the overt ADV/ADJ grounds normally (full candidate set)
action_map: [ OPEN_CLAUSE(imperative), EMIT_UNRESOLVED_SLOT(predicate),
              EMIT_SYNTH_SLOT(SUBJECT), GROUND_PRIME(YOU), GROUND(tok0) ]
confidence: soft
```
"Quietly!" ("[do it] quietly"), "Again!", "Faster!". This is the soft/broad
licensing the task calls out: a standalone ADV *can* be an imperative, and the
implied predicate is **not invented** — it is emitted as an **elision
unresolved-slot** whose candidate antecedents come from prior context.
`soft:true` means R2 competes with R4 (bare-fragment ellipsis) on the same
span; both may fire → both branches in the forest (§6).

### R3 — interjection: ground-only, mark utterance_kind  (family: core)
```
id: interjection.ground_only
family: core
soft: true
trigger: { lexset: INTERJECTION, position: fragment_whole }
licenses:
  utterance_kind: interjection
  fragment: { predicate: <tok0>, roles: [] }
fill:
  PREDICATE: sense           # senses_of(tok0), literal or gloss-derived (see §3); NOTHING else
action_map: [ OPEN_CLAUSE(interjection), GROUND(tok0) ]
confidence: soft
```
"Shit!", "Wow!", "Alas!". **The entire rule.** No synth subject, no FEEL
explication, no appraisal node, no valence, no reaction sense — those are
stripped (§3). The interjection grounds to its literal/gloss USVS sense
candidate(s) and the clause is marked an interjection utterance. Connotation is
comprehension's.

### R4 — ellipsis: bare fragment inherits predicate from context  (family: ellipsis)
```
id: ellipsis.inherit_predicate
family: ellipsis
soft: true
trigger: { pos_any: [ADV, ADJ, PROPN, ADP, NOUN], position: fragment_whole,
           surface_absent: [PREDICATE], context_present: true }
licenses:
  utterance_kind: proposition
  fragment: { predicate: null,
              predicate_grounding: @unresolved(context, elision_inherit_predicate),
              roles: [ { relation: <frag_role>, word: <tok> } ] }
fill:
  PREDICATE:   unresolved(context)   # elision slot: candidate antecedent predicates from context[]
  <frag_role>: sense                 # overt fragment words ground normally
action_map: [ OPEN_CLAUSE(proposition), EMIT_UNRESOLVED_SLOT(predicate), GROUND(tok) ]
confidence: soft
```
"More!", "In London.", "Under the table." The elided predicate is an
**elision unresolved-slot** — the encoder posits the empty predicate position
and emits the candidate antecedents (contract §8.3); it does **not** fill it.
The overt quantity/PP word grounds to its own full sense set alongside.

### R5 — pro-drop / dropped argument recovered from context  (family: ellipsis, multilingual)
```
id: prodrop.null_argument
family: ellipsis
soft: true
trigger: { surface_absent: [SUBJECT | OBJECT], context_present: true }   # NO language field
licenses:
  utterance_kind: proposition
  fragment: { predicate: <tok0>,
              roles: [ { relation: <absent_role>, word: null, token_index: null,
                         grounding: @unresolved(context, coref_dropped_argument) } ] }
fill:
  <absent_role>: unresolved(context)   # reference/elision slot; candidate antecedents from context[]/memory
  PREDICATE:     sense
action_map: [ OPEN_CLAUSE(proposition), GROUND(tok0), EMIT_UNRESOLVED_SLOT(<absent_role>) ]
confidence: soft
```
English "And found a coin." (dropped subject *Tom*) and Spanish pro-drop
"Corre." (dropped subject recovered from agreement) are **the same rule** — it
is licensed by *surface-absent argument + context*, **not** by any
English-specific condition. This is the multilingual stance made concrete
(design §2.5, §6): rules are per-language entries over a shared action space,
and pro-drop is one such entry keyed on a universal trigger. The candidate
antecedents (including an agreement-compatible referent) are **emitted, not
selected** — comprehension binds one against memory.

### R6 — normal transitive clause: every content node a candidate slot  (family: core)
```
id: clause.transitive
family: core
soft: false
trigger: { pos_any: [VERB], position: pre_clause }   # a finite verb with overt args
licenses:
  utterance_kind: proposition
  fragment: { predicate: <v>,
              roles: [ { relation: SUBJECT, word: <subj> },
                       { relation: OBJECT,  word: <obj> } ] }
fill:
  PREDICATE: sense    # senses_of(v)    -> candidate slot
  SUBJECT:   sense    # senses_of(subj) -> candidate slot
  OBJECT:    sense    # senses_of(obj)  -> candidate slot
action_map: [ OPEN_CLAUSE(proposition), GROUND(v), GROUND(subj), GROUND(obj) ]
confidence: high
```
"The dog wants food." Every content node emits its **full** sense-candidate
set (contract §8.3): `dog` → `[dog.n.01, frump.n.01, cad.n.01]`, `food` →
`[food.n.01, ...]`, `want` → `[want.v.01, ...]`. The one non-soft rule — an
overt finite transitive clause is unambiguously *a* clause — but its **senses**
are all unresolved. This is the base case: candidates-first even where there is
no structural or elision ambiguity, only lexical.

### R7 — PP-attachment: both attachments as forest branches  (family: attachment)
```
id: attachment.pp_ambiguous
family: attachment
soft: true
trigger: { pos_any: [ADP], position: pre_clause, context_present: null }   # a PP after V + NP
licenses:
  # TWO licenses, each a full tree -> two entries in lattice.trees[]
  branch_A (PP attaches to predicate):
    fragment: { predicate: <v>, roles: [ ..., { relation: <PREP>, word: <pp_obj> } ] }
  branch_B (PP modifies the object NP):
    fragment: { predicate: <v>, roles: [ { relation: OBJECT, word: <np>,
                                           modifiers: [ { relation: <PREP>, word: <pp_obj> } ] } ] }
fill:
  <PREP>:   sense
  <pp_obj>: sense
action_map: [ OPEN_CLAUSE(proposition), GROUND(v), GROUND(pp_obj) ]   # x2, one per branch
confidence: soft
```
"I saw the man with the telescope." Branch A = `with the telescope` is a PP
role of `saw`; branch B = it modifies `the man`. **Both** are licensed and both
are emitted — as **two trees in `lattice.trees[]`** (contract §2.1), the parse
*forest*. The encoder commits to neither attachment; the pick is comprehension's,
made by the *same* select-from-candidates primitive as sense and coref
(contract §7). Senses are shared across both branches by `token_index` (contract
§2.2), so the two-tree forest stays compact.

**Coverage check (task point 5):** imperative synth-subject (R1) ✓;
interjection ground-only (R3) ✓; elision predicate-from-context, unresolved
slot (R2, R4) ✓; pro-drop argument, unresolved slot (R5) ✓; normal transitive
with all senses as candidate slots (R6) ✓; PP-attachment as forest branches
(R7) ✓.

### Named lexical classes (declared alongside the rules)
- `INTERJECTION`: the small stance set (`oh`, `ah`, `alas`, `ugh`, `ouch`,
  `oops`, `phew`, `yuck`, `hurray`, `wow`, `shit`, `damn`, `hell`, `oh dear`,
  `oh no`). **No longer split** into usvs-covered vs lexicon-only for grounding
  purposes: literal-synset members ground via `senses_of`; the rest get
  gloss-derived senses added to USVS (§3), after which all ground identically.
  The class is only a **trigger** for R3, never a grounding index.

---

## 6. How the grammar is INPUT to the encoder (retrieval-conditioned)

**Old (executable) model.** The grammar file was a program: the
`quantum_parser` loaded anchor/before/after rules and *ran* them — each rule a
hard trigger that fired and rewrote tokens — to *produce* one parse. Grammar =
code; the parser executed it; the output was a committed tree.

**New (candidates-first, retrieval-conditioned) model.** The grammar is a
passive declarative inventory the encoder *reads*, per design §2.2 item 3 and
§2.4:

1. **Retrieval filter = `trigger`.** At each transition step the encoder is
   offered only the rules whose `trigger` matches the current span/context — a
   bounded candidate action set, not the whole rulebook every forward pass
   (design §2.2 tractability point). `trigger` *is* the retrieval key.
2. **Grammar-constrained decoding = the union of `licenses.fragment`.** The net
   may only add structure that some fired rule licenses, and every fragment is
   in `ENCODER_IO_CONTRACT_V2` vocabulary, so the net *cannot emit ill-formed
   structure* (design §2.4). The grammar defines the legal output space; the net
   composes within it.
3. **Candidates-first twist: firing ≠ resolving.** Where the old parser fired
   *one* rule and rewrote, the encoder emits the **union of all fired rules'
   licensed structure** into the candidate lattice. Two attachment rules on one
   span → two forest trees (R7). A soft ADV-imperative and a bare-fragment
   ellipsis both firing → both branches (R2 vs R4). The grammar **over-generates
   candidates on purpose**; narrowing is comprehension's job, scored there, not
   here (contract §7 — the encoder is scored on candidate-set *recall*, never on
   the pick).
4. **Rule-inventory-as-input ⇒ new languages without retraining.** Because the
   grammar is retrieved input, not baked-in weights, a new language is new rule
   entries (like R5's non-English pro-drop) over the *same* action space; a
   robust encoder composes with them without retraining (design §6). The
   executable model could never do this — its rules were compiled into the
   parser's control flow.

So the format's job is **not** to be executed. It is to (a) key retrieval
(`trigger`), (b) bound the legal output (`licenses.fragment`), and (c) name the
candidates-first actions each rule conditions (`action_map`) — leaving every
disambiguation to the downstream comprehension model.

---

## 7. What this format deliberately does NOT do

- **It does not disambiguate.** No rule picks a sense, an attachment, an
  antecedent, or a filler (§2). Every such choice is emitted as a candidate set
  and left to comprehension's single select-from-memory primitive (contract §7,
  §10).
- **It is not executable.** No rule is guaranteed to fire; it conditions a
  learned chooser, it does not replace one. It cannot be run as a parser.
- **It carries no valence / stance / appraisal / connotation** (§3). Rules route
  to sense candidates; senses carry meaning; feeling is a comprehension-side
  read off the USVS eval axis.
- **It covers only the authored families** (mood, ellipsis, attachment, core).
  Sized to what the examples motivated; broaden it as new hard cases are
  authored, not ahead of them.

---

## 8. Open question for the lead

The v1-draft's `EMIT_CONTEXT_REF` question **partly dissolves and partly
sharpens** under candidates-first. It dissolves on the *authoring* side: gold no
longer needs a single correct antecedent label — the target is the candidate
*set*, and the encoder is scored on set recall (contract §7), so "which
antecedent is right" is no longer a supervised encoder label at all.

What sharpens is the **run-time candidate-set identity**:

> For an `EMIT_UNRESOLVED_SLOT` action, the gold candidate set is authored as
> `retrieval.source:"context"` pointers into serialized `context[]` (dereferenceable,
> checkable). At inference the same slot must emit `retrieval.source:"memory"`
> candidates drawn from a *live retrieved* memory set whose node handles exist
> only at run time (contract §4.3). **What guarantees the run-time memory
> retrieval surfaces the same candidate set the authored context pointers name —
> i.e. how is the encoder's memory-retrieval index trained/aligned so that
> "recall the gold antecedent among the candidates" is a well-posed target when
> the candidate ids are a retrieval result, not fixed labels?** The sense case
> sidesteps this (the USVS lexicon is fixed and shared between authoring and run
> time); the memory index is not. This is the one load-bearing unknown before
> M63 Step 2 trains `EMIT_UNRESOLVED_SLOT` against these targets.
