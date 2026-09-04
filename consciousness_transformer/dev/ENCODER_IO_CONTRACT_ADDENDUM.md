# Encoder I/O Contract — ADDENDUM (hand-authored hard cases)

Status: DRAFT extension to `dev/ENCODER_IO_CONTRACT.md` (FROZEN v1). Author:
encoder-gold authoring pass. Date: 2026-09-03.

This addendum defines the **minimal** schema extensions needed to express three
families the deterministic teacher structurally cannot produce, and which are
therefore hand-authored in `dev/hand_authored_gold_v1.jsonl`:

1. **Imperatives** — a clause whose SUBJECT is not on the surface ("Look!").
2. **Interjections** — an utterance whose content is an affective *appraisal*
   ("oh dear!", "shit!"), standalone or reacting to a proposition.
3. **Elision / pro-drop** — a fragment whose elided predicate/argument must be
   filled from **prior context** ("More!", "Again!", Spanish "Comió.").

Design rule for the whole addendum: **conform to the frozen contract
everywhere else.** Every extension is either (a) a new *optional* field with a
documented default that reproduces frozen behavior when absent, or (b) a new
enumerated *kind* of an existing object, gated by an explicit discriminator so a
frozen-only consumer can detect and skip it. The frozen top-level record shape
(`text` / `tokens` / `pos` / `gold_tree` / `token_sense_candidates`) is
unchanged; §4 below adds ONE optional sibling top-level field (`context`).

Frozen §3's MFS rule (every `*_sense_id` is `senses_of(word)[0]`) describes what
the *teacher* emits. These records are hand-authored **gold targets**, so a
`*_sense_id` in `gold_tree` MAY be the **context-correct** sense rather than
MFS (e.g. the verb sense of an imperative "look", not its noun MFS). To keep the
retrieval-conditioning input honest, `token_sense_candidates` is still authored
as the MFS-ordered lemma lookup (what retrieval would actually hand the model),
so `token_sense_candidates[i].chosen_sense` (= MFS) and the matching
`gold_tree` `sense_id` (= correct) MAY differ. That divergence is not a bug: it
is precisely the non-MFS choice the encoder is being trained to make (frozen §3,
final paragraph; design §2.4 grounding-selection).

---

## 1. Extension A — SYNTHESIZED role filler (imperative "you", appraisal "I")

**Problem.** An imperative addressee ("you") and an appraisal experiencer ("I")
are real arguments of the grounded clause but have **no surface token** — they
are not in `tokens`, so a frozen role object (which is `{relation, word,
is_entity, sense_id}` read off `arg_node.token`) cannot represent them without
lying about the token stream.

**Extension.** Two new *optional* fields on the frozen `roles[j]` object:

| field | type | nullable | default when absent | notes |
|---|---|---|---|---|
| `synthesized` | bool | no | `false` | `true` = this filler was **posited by a grammar rule**, not read from a surface token. When `true`, `word` is the canonical lemma the rule inserts (e.g. `"you"`, `"I"`) and is **not** expected to occur in `tokens`. |
| `synth_kind` | string \| null | yes | `null` | Why it was synthesized. Vocabulary so far: `"addressee"` (imperative subject), `"speaker"` (appraisal/request experiencer), `"pro_drop_agreement"` (subject recovered from verb morphology). Open string, anchored by these three. |

Coexistence with frozen fields:
- `relation`, `word`, `is_entity`, `sense_id` keep their frozen meaning and
  nullability. For the imperative "you" and appraisal "I", `is_entity` is
  `true` and `sense_id` is `null` — **identical** to how the frozen schema
  already handles any pronoun (frozen §2.2: `sense_id` always `null` when
  `is_entity`). The ONLY new information is `synthesized: true`.
- A frozen-only consumer that ignores `synthesized` sees a well-formed entity
  role filled by "you"/"I"; it simply won't know the token was absent from the
  surface. Nothing breaks.

Example (imperative "Look!" subject):
```json
{ "relation": "SUBJECT", "word": "you", "is_entity": true, "sense_id": null,
  "synthesized": true, "synth_kind": "addressee" }
```

The same two fields also apply, unchanged, to an elided **predicate** via a
clause-level mirror `predicate_synthesized` (§3), used when the request verb
itself is posited (e.g. "More!" → synthesized `want`).

---

## 2. Extension B — APPRAISAL / STANCE clause (interjection grounding)

**Problem.** An interjection carries affective/stance meaning, not a
predicate-argument proposition. Frozen §Provenance returns *grounding-fail* for
"oh dear!" (0 clauses). Design §9.2: ground it to an **appraisal node** ("I feel
something good/bad"), with **valence implicit in a USVS sense**, never
hard-coded.

**Extension.** A new *optional* clause-level discriminator plus one new
sub-object.

New field on `gold_tree.clauses[i]`:

| field | type | nullable | default when absent | notes |
|---|---|---|---|---|
| `clause_type` | string | no | `"proposition"` | Frozen clauses are implicitly `"proposition"`. New values: `"appraisal"` (§2), `"elliptical"` (§3). A frozen-only consumer treats an unknown `clause_type` as its cue to skip the record. |

For a `clause_type: "appraisal"` clause the frozen fields are filled as the NSM
"I feel something" frame and MUST stay well-formed:
- `predicate`: `"feel"` (the FEEL frame's surface anchor).
- `predicate_sense_id`: `null` (FEEL is an NSM prime, not a WordNet sense —
  same nullability the frozen schema already gives primes/copulas).
- `is_question`: `false`.
- `roles`: exactly one experiencer, the synthesized speaker (Extension A):
  `{ "relation": "EXPERIENCER", "word": "I", "is_entity": true,
     "sense_id": null, "synthesized": true, "synth_kind": "speaker" }`.

New sub-object `appraisal` on the clause:

| field | type | nullable | notes |
|---|---|---|---|
| `trigger` | string | no | The interjection surface span, possibly multi-word (`"oh dear"`, `"shit"`). |
| `reaction_sense_id` | string \| null | yes | The **USVS sense the reaction grounds to** — a WordNet *emotion/evaluation* synset (`dismay.n.01`, `annoyance.n.01`, `joy.n.01`, `disgust.n.01`, `relief.n.02`, `pain.n.01`…). **Valence is read from THIS sense's eval / antonym axis** (`ground/polarity.py`, `USVS.antonyms_of`), never authored. `null` only if no reaction sense is available (then the record is flagged for USVS-index follow-up). |
| `valence` | string | no | Literal marker `"from_sense"` — declares that GOOD vs BAD is **resolved from `reaction_sense_id`'s eval axis at grounding time, not stored here.** This field never carries `"GOOD"`/`"BAD"` directly; that would hard-code valence, which the design forbids. |
| `primes` | array[string] | no | The fixed NSM frame `["I", "FEEL"]`. The GOOD/BAD prime is deliberately **absent** — it is supplied by `reaction_sense_id`'s eval axis, not listed here. |
| `scope` | object \| null | yes | `null` = standalone appraisal ("Oh dear!"). Otherwise a **context_ref** (§4) pointing at the proposition the reaction is *about* — for "[news]… shit!" and "We won. Hurray!". Same JSON shape as any context_ref filler. |

Why the reaction sense, not a lemma sense of the interjection: see the friction
note in §5 — the token "shit" grounds (via its lemma index) to `shit.n.01`
(feces), which is *not* the reaction. The appraisal target is a **separate
reaction-sense inventory**, retrieved by appraisal-rule firing, not by
lemma-of-token retrieval. This is the one place the record's grounding target is
NOT in the triggering token's `sense_candidates`.

Example (standalone "Shit!"):
```json
{ "clause_type": "appraisal", "predicate": "feel", "predicate_sense_id": null,
  "is_question": false,
  "roles": [ { "relation": "EXPERIENCER", "word": "I", "is_entity": true,
               "sense_id": null, "synthesized": true, "synth_kind": "speaker" } ],
  "appraisal": { "trigger": "shit", "reaction_sense_id": "annoyance.n.01?",
                 "valence": "from_sense", "primes": ["I", "FEEL"], "scope": null } }
```

---

## 3. Extension C — ELLIPTICAL clause + `predicate_ref`

**Problem.** A fragment ("More!", "Again!", pro-drop "Comió.") has an elided
predicate and/or argument that is only recoverable from **prior discourse**.
The frozen schema parses one sentence in isolation and has no way to leave a
predicate/argument slot empty-but-bound.

**Extension.** `clause_type: "elliptical"` (§2's discriminator) plus:

New *optional* clause-level field:

| field | type | nullable | default when absent | notes |
|---|---|---|---|---|
| `predicate_ref` | object \| null | yes | `null` | A **context_ref** (§4) filling an **elided predicate** from an antecedent — the elided verb of "Again!" is inherited from the prior clause's predicate. When present, `predicate` MAY be `null`. |
| `predicate_synthesized` | bool | no | `false` | Mirror of Extension A for the predicate: `true` when the predicate is **posited by a speech-act rule** rather than inherited or surface (e.g. "More!" → synthesized request verb `want`). Mutually exclusive with a non-null `predicate_ref`. |

Interaction with frozen `predicate` / `predicate_sense_id`:
- If `predicate_ref` is non-null → `predicate` MAY be `null`, `predicate_sense_id`
  MAY be `null` (the real predicate lives at the referenced antecedent).
- If `predicate_synthesized` is `true` → `predicate` carries the posited lemma
  and `predicate_sense_id` its context-correct sense.
- If both are absent/false → frozen behavior (surface predicate), unchanged.

An elliptical clause's **argument** slots may be ordinary surface fillers (the
overt fragment word, e.g. "again", "more"), OR context_ref fillers (§4) for the
inherited arguments. Both can appear in the same `roles` array.

---

## 4. Extension D — POINT-TO-PREEXISTING-CONTEXT (`context_ref`)  ★ the key one

This is the construct the lead asked for: a slot that **references an antecedent
node in prior discourse / memory** instead of grounding a surface token.

### 4.1 The pointer — a `context_ref`

A `context_ref` is a filler (of a role, of `scope`, or of `predicate_ref`)
discriminated by `kind: "context_ref"`. Concrete JSON shape:

```json
{
  "relation": "OBJECT",
  "kind": "context_ref",
  "word": null,
  "is_entity": false,
  "sense_id": null,
  "ref": {
    "source": "context",
    "clause": 0,
    "slot": "role",
    "role_index": 1,
    "handle": null
  }
}
```

Filler-level fields (coexisting with the frozen role fields):

| field | type | nullable | notes |
|---|---|---|---|
| `kind` | string | no | `"context_ref"`. Absent on every frozen/surface filler (implicitly `"surface"`). The discriminator a consumer keys on. |
| `relation` | string | no | Frozen role label, unchanged — the role this antecedent *fills in the fragment* (may differ from the role it played in the antecedent). |
| `word` | string \| null | yes | `null` (no surface token). A consumer that wants a display string reads it from the resolved antecedent. |
| `is_entity` | bool | no | Copied from the antecedent's own `is_entity` after resolution; `false` until resolved is acceptable for authoring. |
| `sense_id` | string \| null | yes | `null` in the pointer. The grounded sense is whatever the antecedent already carries — not re-authored here (single source of truth). |

The `ref` object (the actual pointer):

| field | type | nullable | notes |
|---|---|---|---|
| `source` | string | no | `"context"` = an antecedent inside this record's `context` block (§4.2). `"memory"` = a node in the grounded episodic memory (design §9.1). Open-anchored. |
| `clause` | int \| null | yes | When `source=="context"`: index into `context.clauses[]`. `null` for `source=="memory"`. |
| `slot` | string | no | Which part of the antecedent clause is pointed at: `"predicate"`, `"subject"`, or `"role"` (a specific role, then use `role_index`). |
| `role_index` | int \| null | yes | Index into the antecedent clause's `roles[]` when `slot=="role"`; else `null`. |
| `handle` | string \| null | yes | When `source=="memory"`: an opaque memory-node id (the membrane/LTM handle). `null` for `source=="context"`. Lets the same construct point into persistent memory, not only in-record context, without a schema change. |

### 4.2 The antecedent store — new optional top-level field `context`

A `context_ref` with `source:"context"` needs the antecedent to be *in the
record*. New **optional** top-level sibling of `gold_tree`:

```json
"context": {
  "text": "The dog jumped .",
  "clauses": [ <frozen/appraisal clause objects, same shape as gold_tree.clauses> ]
}
```

| field | type | nullable | notes |
|---|---|---|---|
| `context` | object \| null | yes (absent on all frozen records) | The prior discourse the fragment resolves against. `context.text` is the prior sentence(s); `context.clauses` are their grounded clauses in the **exact** `gold_tree.clauses` shape (so a `ref` into them needs no new addressing scheme). Absent → the record is context-free (all frozen records, all imperatives, all standalone interjections). |

Addressing is deliberately reused: `context.clauses` are the same object type as
`gold_tree.clauses`, so `{clause, slot, role_index}` addresses either. A
`source:"context"` ref MAY also point within the **same** `gold_tree.clauses`
(same-utterance scope, e.g. "shit, [clause]" where the reaction scopes a clause
in the same record) — signaled by `clause` indexing `gold_tree.clauses` when no
`context` block is present. When both exist, `source:"context"` addresses
`context.clauses`; a same-utterance pointer uses `source:"self"`.

`source` vocabulary summary: `"context"` (prior-utterance store), `"self"`
(another clause in this record's `gold_tree`), `"memory"` (persistent grounded
memory by `handle`).

---

## 5. Frictions found while authoring (honest notes for the lead)

1. **POS enum has no INTJ.** Frozen `pos` is `quantum_parser`'s closed `Tag`
   enum (NOUN/VERB/ADJ/…/PART) with **no interjection tag**. Interjection tokens
   are therefore tagged with the nearest available member the tagger would emit
   (NOUN for "shit"/"damn" which have noun lemmas; ADV/PART for "oh"/"alas"/
   "wow"). The `appraisal.trigger` span, not the POS, carries "this is an
   interjection". If the enum is ever extended, add `INTJ`; until then the POS
   field is *not* a reliable interjection signal and consumers must key on
   `clause_type == "appraisal"`.

2. **The appraisal grounding target is NOT among the token's
   `sense_candidates`.** `token_sense_candidates` for "shit" is the lemma lookup
   (`shit.n.01`, …), but the *reaction* grounds to `annoyance.n.01`. So the
   frozen retrieval-conditioning assumption — "the correct sense is one of the
   token's candidates" — **does not hold for interjections.** The appraisal
   sense must come from a **separate reaction-sense retrieval** (fire the
   appraisal rule → retrieve its candidate reaction senses), which the frozen
   `token_sense_candidates` field does not carry. Flagged as a real gap the
   encoder's retrieval input (design §2.2) must widen for appraisals. It is
   representable in the *output* schema (above); it is under-served by the
   *input* schema.

3. **Valence really is resolvable without hard-coding — but only if the
   reaction sense exists in the USVS index.** Grounding to a WordNet emotion
   synset (`joy.n.01` vs `annoyance.n.01`) makes GOOD/BAD fall out of that
   sense's eval/antonym axis, exactly as the design wants. The risk: several
   interjections have **no clean lemma path** to such a synset in WordNet
   (`oops`, `phew`, `ugh`, `oof`), so their `reaction_sense_id` is my best
   guess and **flagged with `?`**. If the USVS index lacks these emotion
   synsets, this family needs a tiny curated reaction-sense inventory (still
   valence-from-axis, just authored senses rather than WordNet-lemma-reachable
   ones). This is the one place the schema is *writable* but the *sense
   inventory* may not yet cover the target.

4. **Everything else is comfortably human-writable.** Imperatives, elided
   arguments, and the context_ref pointer were straightforward to author by hand
   with only the extensions above; no frozen field had to be violated or
   overloaded.
