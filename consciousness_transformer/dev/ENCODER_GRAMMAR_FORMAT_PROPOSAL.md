# Encoder-Reference Grammar Format — proposal v1

Status: DRAFT for lead. Author: encoder-gold authoring pass. Date: 2026-09-04.
Scope: grounded strictly in what authoring `dev/hand_authored_gold_v1.jsonl` +
`dev/ENCODER_IO_CONTRACT_ADDENDUM.md` actually required. Not speculative.

## 0. What changed, and why the format must change

The grammar file's role is shifting (design §2.2 item 3, §2.4, §6): from
**executable deterministic-parser rules** (the current anchor/before/after
rewrite format that the `quantum_parser` runs to *produce* a parse) to a
**declarative reference inventory** that *conditions* the trained encoder — a
set of candidate operations retrieved and offered to the net, which then
composes over them (design §2.4: "Action space = applying candidate grammar
rules"). Three properties the authoring pass proved are needed and the current
executable format cannot express:

1. **Soft / broad licensing, not deterministic firing.** "A standalone ADV can
   act as an imperative"; "a bare interjection is an appraisal, not a parse
   failure." These are *candidate-generating* statements ("this MAY be
   structure X"), decided by the net, not rewrite rules that must fire.
2. **Synthesizing a constituent with no trigger token.** The imperative "you"
   and appraisal "I" (addendum Ext. A) are posited by a rule, not matched from
   the surface. The executable anchor format keys on present tokens; it has no
   "insert an absent constituent" primitive.
3. **Pointing at pre-existing context.** Elision (addendum Ext. C/D) requires a
   rule whose output *references an antecedent node* in prior discourse or
   memory. No single-sentence rewrite format can express "inherit the predicate
   from whatever the last clause was."

Natural language is rejected (too ambiguous to condition a net or to audit —
design §6 wants inspectable rules). The current executable format is rejected
(can't express 1–3). Proposed below: a concrete **declarative, structured**
formalism — pattern/licenses/emits triples over the SAME grounded-tree
vocabulary the I/O contract already uses.

## 1. The formalism

One rule = a record with four parts:

```
rule <id>:
  when   <trigger pattern over tokens/pos/position, all soft>
  licenses <the clause_type / structure this rule proposes>      # candidate, not forced
  emits  <partial grounded-tree fragment, in I/O-contract vocabulary>
  ground <how leaf senses are obtained: lemma-retrieval | reaction-inventory | inherit>
```

- `when` is a **soft trigger**: matching makes the rule a *candidate* in the
  retrieved action set for that span; it never obligates application. Predicates
  are over the fields the encoder already sees (`tokens`, `pos`, sentence
  position, presence/absence of a finite verb, presence of prior context).
- `emits` is written in the **exact node/role/appraisal/context_ref vocabulary
  of `ENCODER_IO_CONTRACT.md` + its addendum** — so a rule's output fragment is
  a sub-tree of a legal gold record. This is what makes the grammar
  "grammar-constrained decoding" (design §2.4): the union of rule `emits`
  fragments *is* the legal output space.
- `ground` names the retrieval channel for the fragment's leaves — the crucial
  distinction the interjection authoring exposed (see friction #2 below).

Meta-variables: `$SPKR` = synthesized speaker "I"; `$ADDR` = synthesized
addressee "you"; `@ctx[c].slot` = a context_ref pointer (source/clause/slot/
role_index) into the antecedent store.

## 2. Example rules (the three families)

### R-IMP-1 — standalone verb/adv as imperative (Ext. A: synthesized subject)
```
rule imperative.bare_verb:
  when   pos[0] in {VERB, ADV} and no_overt_subject and sentence is not question
  licenses clause_type = "proposition"
  emits  { predicate: token[0],
           roles: [ { relation: SUBJECT, filler: $ADDR,          # synthesized "you"
                      is_entity: true, synthesized: true, synth_kind: "addressee" } ] }
  ground predicate: lemma-retrieval(token[0]), prefer VERB sense over MFS-noun
```
Note the last line: authoring "Look!" showed the correct target is the *verb*
sense while token-MFS is the *noun* (`look.n.01`) — so the rule must state a
sense-selection bias, which the encoder realizes as a non-MFS grounding choice
(design §2.4 grounding-selection; contract §3 final para).

### R-INT-1 — bare interjection as appraisal (Ext. B: valence from sense)
```
rule interjection.standalone:
  when   span is a known interjection lemma  (oh, oh dear, shit, alas, wow, ...)
  licenses clause_type = "appraisal"
  emits  { predicate: "feel", primes: [I, FEEL],
           roles: [ { relation: EXPERIENCER, filler: $SPKR, synthesized: true } ],
           appraisal: { trigger: span, reaction_sense_id: <from ground>,
                        valence: "from_sense" } }
  ground reaction_sense_id: reaction-inventory(span)     # NOT lemma-retrieval(span)
         valence:           eval-axis-of(reaction_sense_id)   # never literal GOOD/BAD
```
Two things authoring forced into the rule: (a) `ground` uses a **separate
reaction-sense inventory**, because the interjection's *lemma* sense
(`shit.n.01`) is not its *reaction* sense (`annoyance.n.01`); (b) valence is a
*derived* read of the reaction sense's eval axis, never written in the rule.

### R-INT-2 — interjection reacting to a proposition (scope = pointer)
```
rule interjection.reaction_scoped:
  when   an appraisal span immediately follows a complete proposition clause C
  licenses clause_type = "appraisal"
  emits  { ...as R-INT-1...,
           appraisal.scope: @ctx[C].clause }     # context_ref at the whole antecedent clause
  ground as R-INT-1
```

### R-ELL-1 — elliptical fragment inherits predicate from antecedent (Ext. C/D)
```
rule ellipsis.inherit_predicate:
  when   fragment has no finite verb and prior context has a clause C
  licenses clause_type = "elliptical"
  emits  { predicate: null,
           predicate_ref: @ctx[C].predicate,               # POINT-TO-CONTEXT
           roles: [ inherited-args-as-context_ref + overt-fragment-words ] }
  ground inherited leaves: inherit (sense comes from the antecedent node, not re-grounded)
         overt words:      lemma-retrieval(token)
```
This is the rule the lead specifically asked the format to express: its `emits`
carries a `context_ref`, and its `ground` channel `inherit` says "do not
re-ground — read the sense off the pointed-at node." "Again!", "Me too.", "The
blue one." are all instances.

### R-ELL-2 — request fragment: synthesized predicate + context object (Ext. A+D)
```
rule ellipsis.request_more:
  when   fragment is a quantifier/NP (more, again, <NP>) with no verb, request context
  licenses clause_type = "elliptical"
  emits  { predicate: "want", predicate_synthesized: true,
           roles: [ { SUBJECT: $SPKR, synthesized: true },
                     { OBJECT: @ctx[C].role[k] | overt-NP },      # pointer OR surface
                     { QUANTITY: token("more") } ] }
  ground predicate: fixed ("want", speech-act posited)
         OBJECT:    inherit if context_ref else lemma-retrieval
```
"More!" (object inherited from context) and "More soup!" (object overt, no
context) are the two branches of this one rule — showing a rule whose slot is
*either* a context pointer *or* a surface filler.

### R-ELL-3 — pro-drop subject recovered from agreement (Ext. A/D, Spanish)
```
rule prodrop.finite_verb:
  when   finite verb with person/number agreement and no overt subject
  licenses clause_type = "elliptical"
  emits  { predicate: token[0],
           roles: [ { SUBJECT: $agreement-referent } ] }
  ground SUBJECT: if antecedent matches agreement -> @ctx[C].role[subj]   # "Comió." -> ella
                  else -> synthesized pronoun from agreement features       # "Comí." -> yo(1sg)
```
Authoring the two Spanish cases showed the SAME rule resolves the subject two
ways: to a context antecedent when one agrees ("Comió." → *ella*), or to a
synthesized agreement-pronoun when none ("Comí." → *yo*). The rule states the
choice; the encoder + memory make it.

## 3. Mapping onto the encoder's retrieval-conditioned action space (§2.4)

The design's encoder is a retrieval-augmented, grammar-constrained transition
parser: action space = applying candidate rules; grounding = selecting candidate
senses. This format drops in cleanly:

- **`when` → the retrieval filter.** Exactly design §2.2 item 3 ("rules whose
  triggers actually match the input"). At each step the encoder is offered the
  rules whose `when` matches the current span/context as its candidate action
  set. Soft triggers over-generate candidates on purpose; the net chooses.
- **`licenses` + `emits` → the transition's output, grammar-constrained.** The
  chosen rule's `emits` fragment is the legal structure the net may add — it
  *cannot* emit ill-formed structure because the fragment is in contract
  vocabulary (design §2.4 "the net only chooses within it").
- **`ground` → the grounding-selection head, with three distinct channels the
  authoring exposed.** (1) `lemma-retrieval` = the frozen
  `token_sense_candidates` path (contract §4). (2) `reaction-inventory` = a NEW
  retrieval channel for appraisals — the interjection target is not in the
  token's candidates, so §2.2's retrieval input must be widened to also retrieve
  candidate *reaction* senses when an interjection rule fires. (3) `inherit` =
  no retrieval; copy the sense off the pointed-at antecedent node. The
  transition head therefore picks not just *a sense* but *which channel*.
- **context_ref = a pointer action, not a grounding action.** `@ctx[...]` in
  `emits` resolves against the same grounded memory the design already routes
  every answer through (design §1, §9.1) — the encoder reads the antecedent
  node's handle rather than composing a new leaf. This is the one action that
  *consumes* memory during encoding, and it is inspectable (you can see exactly
  which node was pointed at), satisfying the §6 auditability requirement.

## 4. What this format deliberately does NOT do

- It is not executable: no rule is guaranteed to fire, so it cannot be run as a
  parser. That is intended — it conditions a learned chooser, it does not
  replace it.
- It does not encode valence, tense, or any content the grounded senses already
  carry. Rules route to senses; senses carry meaning. (Valence lives on the
  reaction sense's eval axis, per addendum Ext. B.)
- It covers only the three authored families. It is sized to what the examples
  motivated; broaden it as new hard cases are authored, not ahead of them.

## 5. The single biggest open question this raises (for the lead)

**Does `ground`'s new `reaction-inventory` channel mean the encoder's retrieval
input (design §2.2) must retrieve candidate senses that are NOT lemma-reachable
from any surface token — and if so, what populates that inventory?** The whole
appraisal family grounds to reaction senses (`annoyance.n.01`, `joy.n.01`) that
the frozen `token_sense_candidates` will never surface for "shit"/"hurray",
because they are not lemmas of those tokens. So either (a) the retrieval step
gains a second index — interjection-span → candidate reaction senses — that must
be built and curated (and several interjections have no clean WordNet emotion
synset, so it may be a small hand-authored inventory), or (b) appraisal
grounding is treated as rule-emitted rather than retrieval-selected, which
breaks the "grounding = selecting retrieved candidate senses" symmetry the
design leans on everywhere else. This is the one place the three families do not
fit the single retrieval-conditioning mechanism, and it needs a decision before
the appraisal family can be trained the same way as the rest.
