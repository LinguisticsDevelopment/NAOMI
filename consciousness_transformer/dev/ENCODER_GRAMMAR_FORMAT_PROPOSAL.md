# Encoder-Reference Grammar Format — proposal v1

Status: PROPOSAL for lead. Author: hand-authoring pass (M63.1c), 2026-09-03.
Grounded in what authoring `dev/hand_authored_gold_v1.jsonl` actually
required — not speculative. Read `dev/ENCODER_IO_CONTRACT_ADDENDUM.md` first
(this format's job is to *license* the structures that addendum serializes).

## 0. What changed, and why the executable format can't carry it

The current grammar file is an **executable** deterministic-parser format:
anchor / before / after token-pattern rules that *fire and rewrite*. It is a
program. The universal encoder (design §2.4) does not run that program — it
is a retrieval-conditioned, grammar-constrained neural transition parser
whose grammar is an **INPUT that conditions** the net's action choices, not
code that executes.

Authoring the three hard-case families exposed exactly what the executable
format cannot state:

- **Soft / broad licensing.** "A standalone ADV/ADJ fragment *can* act as an
  imperative or an elliptical predicate-inheritor." The executable format
  has no way to say *can* — every rule is a hard trigger→rewrite.
- **Null / synthesized constituents.** "An imperative *implies* a subject
  (the addressee) with no surface token." The executable format only rewrites
  tokens that exist; it cannot posit an absent one.
- **Point-to-preexisting-context.** "An elliptical fragment *inherits* its
  predicate/argument from a node in prior discourse." No token-local rewrite
  rule can reach across the sentence boundary into memory.

So the reference grammar becomes a **declarative** artifact: it states, per
construction, (a) a soft trigger over POS/lexical-class/position, (b) what
structure it *licenses*, and (c) how each licensed slot is *filled and
grounded* — and it maps each of those onto a concrete class of encoder
action (§4). It is NOT natural language and NOT the anchor/before/after
executable format.

## 1. The formalism (schema of one rule)

A rule is a typed record. Fields:

```
rule:
  id:        <string>                     # stable identifier
  family:    mood | stance | ellipsis     # the hard-case family it serves
  soft:      <bool>                        # true = broad/pattern license, not a hard trigger
  trigger:                                 # declarative match condition (a predicate over the span)
    pos_any:   [<POS>, ...]                # coarse POS from the frozen enum
    lexset:    <named lexical class | null># e.g. INTERJECTION, TRANSFER_VERB
    position:  utterance_initial | fragment_whole | pre_clause | null
    surface_absent: [<role>, ...]          # roles with NO surface token in the span
  licenses:                                # the structure this rule permits (addendum shapes)
    clause_kind: proposition | imperative | appraisal | elliptical
    null_slots:  [{role, fill}]            # slots with no surface token; see fill methods
    node:        appraisal | null          # emit an appraisal node (stance family)
  fill:                                    # per null/non-surface slot: HOW it is grounded
    <role>: from_prime(<PRIME>)            # ground to an NSM prime (synthesized)
          | from_retrieval(scope)          # ground by selecting a context/memory node
          | from_usvs_sense                # ground to a retrieved USVS sense (frozen path)
          | from_stance_lexicon            # ground to an interjection stance sense (proposed)
  action_map: [<encoder-action-class>, ...] # §4: which action(s) this conditions
  confidence: high | soft                  # authoring-observed reliability of the license
```

`soft:true` rules are the ones the executable format cannot express: the net
treats them as *available* actions conditioned on the trigger, and learns
*when* to take them from teacher-plus-hand gold — they never force an output.

## 2. The rules (6, covering the three families)

### R1 — imperative: standalone verb licenses addressee subject  (family: mood)
```
id: imperative.addressee_subject
family: mood
soft: true
trigger: { pos_any: [VERB], position: utterance_initial, surface_absent: [SUBJECT] }
licenses: { clause_kind: imperative, null_slots: [{role: SUBJECT, fill: from_prime(YOU)}] }
fill: { SUBJECT: from_prime(YOU) }
action_map: [OPEN_CLAUSE, EMIT_SYNTH_SLOT(SUBJECT), GROUND_PRIME(YOU)]
confidence: high
```
Motivated by every imperative record ("Look!", "Sit down!", "Open your
books."). The overt object, when present ("Give me the ball"), grounds by the
normal frozen path — this rule only supplies the *missing* subject.

### R2 — stance: interjection licenses an appraisal node  (family: stance)
```
id: stance.interjection_appraisal
family: stance
soft: true
trigger: { pos_any: [PART, NOUN, VERB], lexset: INTERJECTION, position: fragment_whole }
licenses: { clause_kind: appraisal, node: appraisal,
            null_slots: [{role: SUBJECT, fill: from_prime(I)}] }
fill: { SUBJECT: from_prime(I),
        OBJECT:  from_usvs_sense | from_stance_lexicon }   # valence implicit in the chosen sense
action_map: [OPEN_CLAUSE(appraisal), EMIT_SYNTH_SLOT(SUBJECT), GROUND_PRIME(I),
             GROUND_STANCE(OBJECT)]
confidence: soft
```
The explication template is fixed (`I FEEL something`); **valence is never
written by the rule** — it is read off whichever sense `GROUND_STANCE`
selects (`from_usvs_sense` for `shit`/`wow`, else `from_stance_lexicon`).
This keeps "valence from USVS, not hard-coded" while still licensing the
appraisal for interjections with no WordNet synset.

### R3 — stance: reaction attaches appraisal to prior proposition  (family: stance)
```
id: stance.reaction_attaches_to_prior
family: stance
soft: true
trigger: { lexset: INTERJECTION, position: fragment_whole, context_present: true }
licenses: { clause_kind: appraisal,
            stance_target: from_retrieval(prior_context|memory) }
fill: { stance_target: from_retrieval(prior_context|memory) }
action_map: [OPEN_CLAUSE(appraisal), EMIT_CONTEXT_REF(stance_target), GROUND_STANCE(OBJECT)]
confidence: soft
```
Motivated by the reaction records ("...The cat is dead. / Oh no!"). The
appraisal is the same node as R2; this rule adds the pointer to *what is
being reacted to* — the point-to-context action, reused (§4).

### R4 — ellipsis: bare fragment inherits predicate from context  (family: ellipsis)
```
id: ellipsis.inherit_predicate
family: ellipsis
soft: true
trigger: { pos_any: [ADV, ADJ, PROPN, ADP], position: fragment_whole,
           surface_absent: [PREDICATE], context_present: true }
licenses: { clause_kind: elliptical,
            predicate_ref: from_retrieval(prior_context|memory) }
fill: { PREDICATE: from_retrieval(prior_context|memory) }
action_map: [OPEN_CLAUSE(elliptical), EMIT_CONTEXT_REF(predicate)]
confidence: soft
```
Motivated by "More!", "Again!", "In London.", "Under the table.", "John."
The elided predicate is *selected* from the retrieved discourse, not invented.

### R5 — ellipsis: dropped argument inherits from context  (family: ellipsis)
```
id: ellipsis.inherit_argument
family: ellipsis
soft: true
trigger: { surface_absent: [SUBJECT|OBJECT], context_present: true }
licenses: { clause_kind: elliptical,
            null_slots: [{role: SUBJECT|OBJECT, fill: from_retrieval(prior_context|memory)}] }
fill: { SUBJECT|OBJECT: from_retrieval(prior_context|memory) }
action_map: [EMIT_CONTEXT_REF(role)]
confidence: soft
```
Covers the dropped subject/object of "More!" (both), "Again!", the coordinated
"And found a coin." (dropped subject Tom), and — the same rule, no English
special-casing — the Spanish pro-drop "Corre." recovering its subject from
agreement. One rule, licensed by *surface-absent argument + context*, not by
language.

### R6 — ellipsis: fragment answer fills the questioned slot  (family: ellipsis)
```
id: ellipsis.answer_fills_wh_slot
family: ellipsis
soft: true
trigger: { position: fragment_whole, context_present: true, context_is_question: true }
licenses: { clause_kind: elliptical,
            predicate_ref: from_retrieval(prior_context),
            null_slots: [{role: '<the wh-slot>', fill: from_surface}] }
fill: { PREDICATE: from_retrieval(prior_context), '<wh-slot>': from_surface }
action_map: [OPEN_CLAUSE(elliptical), EMIT_CONTEXT_REF(predicate),
             GROUND_USVS_OR_ENTITY('<wh-slot>')]
confidence: soft
```
Motivated by "Who broke the window? / John." and "Where is the cat? / Under
the table." The fragment grounds the *questioned* role by the normal path and
inherits everything else by pointer.

## 3. Named lexical classes referenced (small, declared alongside the rules)

- `INTERJECTION`: the ~15-item stance set (`oh`, `ah`, `alas`, `ugh`, `ouch`,
  `oops`, `phew`, `yuck`, `hurray`, `wow`, `shit`, `damn`, `hell`, `oh dear`,
  `oh no`). Split at grounding time into *usvs-covered* (`shit`, `wow`,
  `damn`, `hell`) and *lexicon-only* (the rest) — see the coverage hole in
  the addendum §7.
- `TRANSFER_VERB`: reuse the existing `nsm_ct.clause._TRANSFER_VERBS` set
  (referenced, not redefined).

## 4. Mapping onto the encoder's retrieval-conditioned action space (design §2.4)

Design §2.4: "action space = applying candidate grammar rules; grounding =
selecting candidate USVS senses." The rules above need exactly **two** new
action classes beyond that, and the second is the crux:

| action class | what it does | conditioned on |
|---|---|---|
| `OPEN_CLAUSE(kind)` | start a clause of the given `clause_kind` | fired rule (existing rule-application action) |
| `GROUND_USVS_SENSE` | pick a sense from the retrieved candidate set | retrieved USVS senses (existing, frozen path) |
| `EMIT_SYNTH_SLOT(role)` + `GROUND_PRIME(P)` | posit a role filler with no surface token, grounded to NSM prime `P` | fired rule; **no retrieval** — the prime is fixed by the rule (R1 YOU, R2 I) |
| `GROUND_STANCE(role)` | ground a stance complement to a USVS sense **or** stance-lexicon entry (valence implicit) | retrieved sense candidates ∪ stance lexicon |
| **`EMIT_CONTEXT_REF(slot)`** | **select an antecedent node from the retrieved discourse/memory candidate set** and bind the slot to it | **retrieved memory/discourse candidates** |

The clean result — and the reason this is on-architecture rather than a
bolt-on — is a **symmetry**:

> grounding = *choose a sense from the retrieved SENSE-candidate set*
> point-to-context = *choose a node from the retrieved MEMORY-candidate set*

`EMIT_CONTEXT_REF` is not a new *kind* of computation; it is the existing
retrieval-select action pointed at a different retrieval index (the discourse
/ grounded-memory nodes instead of the USVS sense inventory). That is why
"every answer routes through a memory read" (design §1) and "point to
preexisting context" are the *same* mechanism: the encoder already selects
from retrieved candidates; ellipsis and reaction-attachment just make the
candidate set be memory nodes. The `synthesized`/`from_prime` actions, by
contrast, take **no** retrieval — they are rule-fixed constants — which is
what makes the imperative "you" and appraisal "I" cheap and unambiguous.

## 5. The open question this raises for the lead

`EMIT_CONTEXT_REF` is authorable and checkable in gold **only** because the
antecedent is serialized in `prior_context`, so the target is a plain array
index (`scope:"prior_context"`). At run time the antecedent is a
grounded-memory node under a `memory_handle` the membrane assigns
(`scope:"memory"`). Nothing here specifies **how the training target — "point
at prior-clause-0's predicate" — becomes a supervised signal over a live
retrieval candidate set whose node ids exist only at inference time.** The
sense-grounding case sidesteps this because the USVS inventory is fixed and
shared between authoring and run time; the memory inventory is not. So the
one design question the lead should resolve before M63 Step 2 trains against
these targets:

> **How is a gold `context_ref` (an index into serialized prior context)
> aligned to the run-time retrieval candidate the encoder must select — i.e.
> what is the supervised loss for `EMIT_CONTEXT_REF` when the correct
> antecedent's identity is a retrieval result, not a fixed label?**

Everything else (imperative synth-subject, appraisal node, stance lexicon)
is comparatively mechanical; this alignment is the load-bearing unknown for
the point-to-context construct.
