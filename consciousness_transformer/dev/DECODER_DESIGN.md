# Decoder Design — the OUTPUT half of the I/O layer (grounded answer → surface text)

Status: DRAFT for lead review. Nothing built yet. Author: decoder design pass,
2026-09-04. Companion to `dev/UNIVERSAL_ENCODER_DESIGN.md` (the mind's INPUT half)
and `dev/ENCODER_IO_CONTRACT_V2.md` (the candidates-first grounded schema). This
doc specs the OUTPUT half: how a grounded answer chosen by comprehension becomes
surface text, and why that path **cannot confabulate**.

Architecture, locked (design §1 invariant):

```
  text ──► ENCODER ──► candidate lattice ──► COMPREHENSION ──► grounded answer ──► DECODER ──► text
           (input transducer)               (GRU over grounded          (this doc: output transducer)
                                             tensor memory: resolves,
                                             reasons, answers)
```

The encoder and the decoder are the two **boundary transducers**. Between them
sits the symbolic grounded middle (inspectable tensor memory, USVS senses, NSM
primes, fixed TPR ops). **Knowledge lives only in the middle.** The decoder holds
*realization policy* — how to say a thing — and never *knowledge* — what is true.

---

## 1. Scope + phases

### 1.1 Phase 1 (design concretely here) — rule-grounded short-answer realizer

A **deterministic** pipeline, no learned weights:

```
  grounded answer structure  ──►  sense_id → lemma / handle → surface   ──►  grammar run FORWARD  ──►  text
  (from comprehension)            (USVS sense_lemmas; entity book)           (same grammar files,
                                                                              generation direction)
```

Phase 1 covers exactly the answer types the K-12 short-answer ladder
(design §5) needs:

- **entity / name** ("Who …?" → *Mary*)
- **place** ("Where …?" → *the garden*)
- **attribute value** ("What colour …?" → *red*)
- **yes / no** (polar verdict)
- **abstention** ("not stated / I don't know") — a **first-class** realizable
  answer, per the corrected objective (design §0: abstention is rewarded, not a
  fallback).

Phase 1 is chosen first for one reason: **it makes no-confabulation hold BY
CONSTRUCTION** (§4). It can only emit lemmas/handles that comprehension pulled
out of memory; there is no free generation step anywhere in it.

There is already a working *seed* of this in the repo — `mind/verbalize.py` +
`mind/membrane.py` render `(subject, relation, value)` triples and abstention
(`"I don't know."`) to text over the curriculum's controlled grammar. Phase 1 is
the **generalization** of that seed: (a) replace the hand-written
`RELATION_TEMPLATES` string templates with the grammar run in the generation
direction, and (b) replace curriculum-word passthrough with USVS
`sense_id → lemma` realization, so the decoder covers open vocabulary, not just
the ~dozen curriculum relations. The seed's *contract* (render a grounded triple;
abstain when there is nothing to render) is exactly Phase 1's contract; only the
*coverage* widens.

### 1.2 Phase 2 (DEFERRED — guardrails sketched only) — learned long-form realizer

For multi-sentence / long-form output (narration, explanations longer than one
clause). **Deferred to its own milestone**; it must NOT gate the encoder or the
short-answer ladder (design §4 phase-two note).

Phase 2 **re-opens the confabulation door** — this is the single biggest honesty
risk in the whole pivot (design §4). A learned realizer that free-samples tokens
is, by definition, a small language model producing content that need not be
grounded. Guardrails it MUST satisfy before it may become learned (design
concrete for these is out of scope here; named so the constraint is on record):

- **Copy-from-memory, not free sampling.** Every content lemma emitted must be
  traceable to a specific memory node the realizer was handed — a pointer copy,
  not a distribution over the vocabulary. The learned part chooses *ordering and
  function words*, never *which facts*.
- **Grammar-constrained decoding.** The learned realizer emits within the grammar
  (the same constraint the encoder uses in reverse) so it cannot produce
  ill-formed structure, and within a **content mask** derived from the handed
  memory set so it cannot emit an ungrounded content word.
- **The same ablation gate as Phase 1** (§4.2) applies and must pass: sever the
  memory→decoder path → output collapses to abstention/empty. If a learned
  Phase-2 realizer can still produce fluent content with memory severed, it has
  learned knowledge in its weights and is rejected.

The rest of this doc designs Phase 1.

---

## 2. INPUT to the decoder — the grounded answer structure

The decoder consumes **one object**: the grounded answer structure that
comprehension produces after it has resolved everything. This is the crucial
asymmetry with the encoder:

> The encoder emits a **candidate lattice** (commits to nothing — V2 §0).
> Comprehension **resolves** that lattice against memory. The decoder consumes
> the **RESOLVED** counterpart: every sense chosen, every reference bound, every
> ellipsis filled. There are no `candidates` fields on the way out — a candidate
> set on the decoder's input would mean comprehension had not finished its job.

The answer structure reuses the V2 clause/role/sense vocabulary (so the same
audit tooling reads both directions), but with every `grounding` in a **resolved**
form.

### 2.1 Shape

```jsonc
{
  "answer_kind": "entity" | "place" | "attribute" | "verdict" | "abstain",
  "verdict":     "yes" | "no" | null,          // present iff answer_kind == "verdict"
  "answer_clause": {                            // absent iff answer_kind == "abstain"
    "predicate": str | null,
    "predicate_grounding": <resolved_grounding>,
    "is_question": false,
    "utterance_kind": "proposition" | "imperative" | "interjection",
    "roles": [
      { "relation": str,                        // frozen V2 role vocabulary (§3 of V2)
        "grounding": <resolved_grounding> },
      ...
    ]
  },
  "focus": { "slot": "predicate" | "role", "role_index": int | null } | null,
                                                // which slot the question asked for (the wh-focus);
                                                // null for a full-clause / verdict answer
  "provenance": [ <memory_handle>, ... ]        // the memory node(s) every grounding was read from
}
```

`<resolved_grounding>` is a V2 `grounding` object narrowed to its **resolved
types only** — the two V2 types that carry no candidate set, plus a resolved
sense:

```jsonc
"grounding": {
  "type":     "sense" | "prime" | "entity",
  "sense_id": str | null,      // type == "sense": the ONE chosen USVS sense_id (a member of the
                               //   encoder-emitted candidate set, picked by comprehension)
  "prime":    str | null,      // type == "prime": the NSM prime (e.g. YOU, NOT, MAYBE)
  "handle":   str | null       // the memory node this grounding was read from (REQUIRED for content;
                               //   this is the physical no-confabulation link — see §4)
}
```

Contrast with V2's grounding: V2's unresolved types (`sense` with a `candidates`
list, `reference`, `elision`) **cannot appear** on the decoder input. A `sense`
here is a *committed* `sense_id` (no `candidates`, no `retrieval`); a former
`reference`/`elision` slot arrives already dereferenced to the `entity`/`sense`
it resolved to, carrying the `handle` of the memory node it bound to. If any slot
still carried a candidate set, the decoder **rejects the structure** (a
comprehension bug, not a decoder-realizable input) and abstains.

### 2.2 Why this is the right boundary

- **Committed, because comprehension has chosen.** Every field is a single value.
  The decoder never selects meaning; it only *renders* the meaning it is handed.
- **Every content node carries a `handle`.** The handle is the address in
  grounded memory the value came from. It is the object §4's gate checks and §4.2's
  ablation severs. A content grounding with no handle is not realizable → abstain.
- **`focus` tells the decoder how much to say.** A wh-question resolved to a fact
  triple + a focus on (say) the PLACE role lets the decoder realize either the
  full clause ("Mary is in the garden.") or the short answer ("The garden.")
  without re-deriving anything.

---

## 3. REALIZATION — grounded meaning → surface string

Two independent mappings compose: **word choice** (a grounding → a surface word)
and **word order** (a clause+roles → a token sequence via the grammar forward).

### 3.1 Word choice — a grounded node → a surface word

| grounding | surface word | mechanism |
|---|---|---|
| `type:"sense"`, `sense_id` | a lemma of that sense | `usvs.sense_lemmas[sense_id]` (via the `sense_id → lemmas` index already built in `ground/usvs.py`, `USVS.__post_init__`'s `_lemma_index` inverse). Pick the **canonical lemma** (deterministic: the sense's first lemma, WordNet MFS-lemma order), applying number/tense morphology from the clause's grammar features. |
| `type:"entity"`, `handle` | a name or a pronoun | the **entity book**: a name entity renders to its surface name (`clause.py`'s referent variables — `var:mary` → *Mary*); a discourse-old referent MAY render to a pronoun by the same person/number features comprehension already tracks (V1 interface "resolutions" edge). Phase 1 default: render the **name** (pronoun realization is an §6 open question — safe, never wrong). |
| `type:"prime"`, `prime` | the prime's exponent | `nsm_primes.PRIMES_BY_NAME[prime].exponent` (e.g. `YOU` → *you*, `NOT` → *not*, `MAYBE` → *maybe*). Function-word primes are realized by the grammar directly, not spelled out as content. |

The pivotal property, stated precisely: **word choice is a lookup keyed by a value
comprehension put there.** `sense_lemmas[sense_id]` can only return a surface word
when `sense_id` is a real, grounded sense — and `sense_id` got onto the decoder's
input only because comprehension selected it from the encoder's candidate set,
which came from the lexicon for a token that was actually in memory. There is no
path by which the decoder invents a `sense_id`. (See §4.)

### 3.2 Word order — a clause + roles → a token sequence

Run **the same grammar** used by the encoder, in the **generation (FORWARD)
direction**: instead of `text → structure` (parse), do `structure → text`
(realize). The grammar's production rules are read as generators —
`statement := np predicate '.'` says *emit an NP, then a predicate, then a
period*; `predicate := 'is' 'in' 'the' noun` says *emit "is in the", then the
place noun*. This is the deterministic inverse of `mind/grammar.py`'s
recursive-descent parser, and it supersedes the hand-written string templates in
`membrane.RELATION_TEMPLATES` / `reverse_parser.thought_to_text` (which are the
Phase-1 seed's stand-in for exactly this — see the docstrings in both files, which
explicitly flag themselves as seeds for "the real reverse parser").

Because it is the *grammar* generating (not free text), the output is
well-formed by construction — the mirror image of the encoder's
grammar-constrained decoding (design §2.4). The grammar defines the legal surface
space; the decoder only walks it, filling slots with §3.1's word choices.

Per answer type (the short-answer ladder):

| answer_kind | grammar path (forward) | full-clause form | short form (focus) |
|---|---|---|---|
| **entity** | np(subject-focus) | "Mary saw Sue." | "Mary." |
| **place** | statement, PLACE predicate | "Mary is in the garden." | "The garden." |
| **attribute** | statement, attribute predicate | "The ball is red." | "Red." |
| **verdict** | polar → *Yes.* / *No.* | — | "Yes." / "No." |
| **abstain** | fixed | — | "I don't know." / "Not stated." |

The short form is a projection of the full clause onto the `focus` slot — the
grammar realizes only the focused constituent's NP, dropping the rest. Both are
faithful; the choice is a register/format setting, never a content decision.

### 3.3 Abstention as a first-class realized answer

`answer_kind:"abstain"` has **no `answer_clause`** — there is deliberately nothing
to realize. It renders to a fixed surface form (*"I don't know."* / *"Not
stated."*, register-selectable), exactly as `verbalize.verbalize_answer` already
returns `"I don't know."` for `answer is None` / `ops.ABSTAIN` (`ABSTAIN = "idk"`).

Abstention is not an error path or a decoder failure — it is the correct,
rewarded output whenever comprehension found no grounded answer (design §0's
corrected objective). It is also the **attractor state** the no-confabulation
gate forces the decoder into whenever there is no grounded content to say (§4).

---

## 4. THE NO-CONFABULATION GATE

### 4.1 Why Phase 1 cannot hallucinate — the argument

The claim (design §1 invariant): *the decoder must NOT be able to produce content
that isn't grounded in memory.* For Phase 1 this holds by construction, and here
is the chain, link by link:

1. **The decoder emits content only via §3.1's lookups.** There is no generative
   model, no vocabulary distribution, no sampling anywhere in Phase 1. The only
   way a content word reaches the output is `sense_lemmas[sense_id]`,
   `entity_book[handle]`, or `prime.exponent`.
2. **Every one of those lookups is keyed by a value comprehension supplied.**
   `sense_id`, `handle`, `prime` are fields of the input answer structure (§2).
   The decoder mints none of them.
3. **Comprehension only ever supplies keys it read from memory.** By the design §1
   invariant, *every answer routes through a memory read* — comprehension's one
   primitive is select-from-memory-conditioned-candidates (V2 §0). A `sense_id`
   in the answer is one comprehension *selected* from the encoder's candidate set
   after conditioning on memory; the required `handle` (§2.1) names the exact
   memory node it came from.
4. **Therefore the set of content words the decoder CAN emit = the set of
   grounded values comprehension read from memory.** The grammar (§3.2) can only
   arrange those words and add closed-class function words (which carry no
   knowledge). There is no production that introduces a new content lemma.
5. **When comprehension supplies no grounded content, there is nothing to look
   up.** The input is `answer_kind:"abstain"` (or a structure the decoder rejects
   for a missing `handle`, §2.1), and the decoder realizes abstention (§3.3).

The decoder's realization policy is thus a **pure function of grounded input**:
`realize(answer_structure)`, deterministic, with an image restricted to
{arrangements of the handed grounded words} ∪ {abstention}. It has no channel to
knowledge except the answer structure, and no channel to invent one.

The contrast with an LLM decoder is the whole point: an LLM's decoder emits a
softmax over the entire vocabulary at every step, so any token is reachable and
"grounded in memory" is at best a soft, trained preference. Here, ungrounded
content is **not in the image of the realization function at all**.

### 4.2 The ablation test (the executable gate)

The invariant is a hard architectural gate, testable by ablation (design §1). The
test:

> **Sever the memory→decoder path** — run the exact same query pipeline, but
> replace the grounded answer structure comprehension would hand the decoder with
> one whose content bindings are removed (all `sense_id`/`handle`/`entity`
> resolutions nulled; equivalently, feed the decoder the answer structure built
> from an **empty / zeroed memory**). **The output MUST collapse to
> abstention/empty. It must NEVER invent content.**

Pass condition, made concrete:

- With memory intact: the decoder produces the grounded short answer
  (*"The garden."*).
- With the memory→decoder path severed: for **every** probe in the battery, the
  decoder outputs **only** abstention (*"I don't know."* / *"Not stated."*) or
  empty — and in particular reproduces **zero** of the with-memory content words
  at a rate above chance-collision. Any run where severed-memory output contains a
  grounded content lemma is a **hard failure** of the invariant.

Because Phase 1 is a pure function of the answer structure (§4.1), the severed
case has *nothing to look up* and this pass is guaranteed by construction — which
is exactly why we can run the ablation as a **regression gate** (design §8, M65:
"rule-grounded short-answer decoder + the §1 memory-bypass ablation"): it must
stay green forever. It is also the acceptance test Phase 2 must pass before its
realizer is allowed to become learned (§1.2).

Two supporting checks alongside the ablation:

- **Round-trip / cycle-consistency.** `parse(realize(answer)) == answer` on the
  answer's committed content (the membrane already cycle-tests this direction for
  the curriculum). A realizer that added content would fail to round-trip.
- **Content-mask audit.** Log, per output, the set of content lemmas emitted and
  assert it ⊆ the lemmas of the handed `provenance` nodes. A pure Phase-1
  realizer makes this vacuously true; logging it is what catches a future
  regression (or a Phase-2 leak) early.

---

## 5. INTERFACE with comprehension + the memory-read invariant

The boundary, stated as a contract:

- **The decoder reads exactly one thing: the comprehension-produced grounded
  answer structure (§2).** It does **not** read the encoder's output (the
  candidate lattice), it does **not** read the raw input text, and it does
  **not** read grounded memory directly. Its entire world is the resolved answer
  structure it is handed.
- **Direction of the invariant.** Because the only content in the answer
  structure got there through comprehension's memory read (design §1: every
  answer routes through a memory read), the decoder inherits the invariant for
  free. There is no encoder→decoder shortcut and none is possible: the decoder
  has no reference to the encoder at all. The moment a decoder could answer from
  encoder activations without going through grounded memory, the thesis is lost
  (design §1) — this interface makes that path physically absent.
- **What crosses the boundary, in one line:** comprehension hands the decoder a
  committed clause (or an abstain marker) + a focus + provenance handles; the
  decoder hands back a surface string. Meaning flows one way, already resolved;
  the decoder returns text and nothing else (no resolutions, no memory writes —
  those all happened upstream, V1 interface §2).

```
   COMPREHENSION                                  DECODER
   (GRU over grounded memory)                     (this doc)
        │  resolve lattice against memory              ▲
        │  = every answer routes through a memory read │  realize(answer_structure) → text
        ▼                                              │  (pure function; no memory, no text, no encoder)
   grounded answer structure  ──────────────────►─────┘
   {committed clause | abstain} + focus + provenance
```

---

## 6. Open questions for the lead (short)

1. **Pronoun realization (§3.1).** Phase 1 default renders the **name** always
   (never wrong, sometimes clunky: "Mary saw Sue. Mary waved."). Do we want
   Phase-1 pronoun output ("She waved.") — which needs a small, deterministic
   generation-side referring-expression choice (given/new + person/number, data
   comprehension already tracks) — or defer all pronoun *generation* to Phase 2?
   Recommendation: defer; name-always keeps Phase 1 purely a lookup.
2. **One grammar, both directions, or a generation-annotated copy?** Running
   `mind/grammar.py` / the universal grammar strictly in reverse is cleanest
   (single source of truth, guarantees the mirror property), but some parse rules
   are lossy to invert (positional WSD, article insertion). Do we invert the
   existing grammar files, or maintain generation directives alongside them?
   Recommendation: invert the shared files; treat any non-invertible rule as an
   encoder-side bug to fix, not a decoder-side special case.
3. **Short vs full form default (§3.2).** Is the K-12 ladder graded on the
   **short** answer ("The garden."), the **full** clause ("Mary is in the
   garden."), or either? This sets the default `focus` projection and the eval
   scorer. Recommendation: accept either, score on the focused constituent.
4. **Abstention surface form.** *"I don't know."* vs *"Not stated."* — are these
   register variants of one answer, or does the ladder distinguish "the text
   didn't say" from "I can't tell"? (Comprehension already distinguishes; do we
   surface the distinction?)

---

## 7. Three worked examples

Format: the resolved answer structure comprehension hands the decoder (§2), then
the realization (§3), then what the ablation (§4.2) does to it.

### 7.1 Who / where short answer — "Where is Mary?" → "The garden."

Context in memory (written earlier): `mary —PLACE→ garden` (garden grounded to
`garden.n.01`, read from memory node `h_garden`).

Answer structure from comprehension:

```jsonc
{
  "answer_kind": "place",
  "answer_clause": {
    "predicate": "is", "predicate_grounding": { "type": "prime", "prime": "BE_SOMEWHERE" },
    "is_question": false, "utterance_kind": "proposition",
    "roles": [
      { "relation": "SUBJECT",
        "grounding": { "type": "entity", "handle": "h_mary" } },
      { "relation": "PLACE",
        "grounding": { "type": "sense", "sense_id": "garden.n.01", "handle": "h_garden" } }
    ]
  },
  "focus": { "slot": "role", "role_index": 1 },
  "provenance": ["h_mary", "h_garden"]
}
```

Realization: focus is the PLACE role → §3.1 `sense_lemmas["garden.n.01"][0]` =
*garden*; grammar forward realizes the focused NP with its determiner →
**"The garden."** (full-clause form, if requested: grammar `statement` forward →
*"Mary is in the garden."*).

Ablation: sever memory → `sense_id`/`handle` for the PLACE slot are gone, no
grounded value to look up → **"I don't know."** No content invented. ✓

### 7.2 Yes / no — "Is the ball red?" → "Yes."

Memory holds `ball —ATTRIBUTE→ red` (`red.a.01`, node `h_red`); comprehension
verifies the polar query against it and returns a verdict.

```jsonc
{
  "answer_kind": "verdict",
  "verdict": "yes",
  "answer_clause": {
    "predicate": "is", "predicate_grounding": { "type": "prime", "prime": "BE_SOMEONE_SOMETHING" },
    "is_question": false, "utterance_kind": "proposition",
    "roles": [
      { "relation": "SUBJECT", "grounding": { "type": "sense", "sense_id": "ball.n.01", "handle": "h_ball" } },
      { "relation": "ATTRIBUTE", "grounding": { "type": "sense", "sense_id": "red.a.01", "handle": "h_red" } }
    ]
  },
  "focus": null,
  "provenance": ["h_ball", "h_red"]
}
```

Realization: `answer_kind:"verdict"`, `verdict:"yes"` → the polar path (§3.2) →
**"Yes."** (as `verbalize_verdict` already does for `"true"`). The verdict is a
comprehension output; the decoder realizes it, it does not compute truth.

Ablation: sever memory → comprehension cannot verify → verdict is absent /
`answer_kind:"abstain"` → **"I don't know."** The decoder never fabricates a
*"Yes."* ✓

### 7.3 Honest abstention — "Where is John?" (text never located John) → "Not stated."

Comprehension's memory read for `john —PLACE→ ?` returns nothing grounded (John
was mentioned but never placed). Its one primitive found no candidate to select,
so it produces:

```jsonc
{
  "answer_kind": "abstain",
  "focus": null,
  "provenance": []
}
```

Realization: no `answer_clause`, nothing to look up (§3.3) → fixed abstention
surface → **"Not stated."** / **"I don't know."**

This is the *correct* answer (design §0: abstention is first-class and rewarded),
not a decoder failure — and it is the same output the ablation forces in 7.1/7.2
with memory severed, which is exactly why the ablation is a sound test of the
invariant: **the honest answer and the memory-severed answer are the same
answer.** A decoder that can produce content in 7.3 could produce content under
ablation — the two failures are one, and Phase 1 is immune to both by
construction (§4.1). ✓

---

## 8. Relationship to the milestones + existing code

- **Milestone.** This is design §8's **M65** — "rule-grounded short-answer
  decoder + the §1 memory-bypass ablation." Phase 2 (learned long-form) is the
  deferred "M67+ … then learned long-form decoder."
- **Reuses, in the codebase today:** `mind/verbalize.py` (the answer/verdict/
  abstain rendering contract + `"I don't know."`), `mind/membrane.py`
  (render-fact seed and its cycle-consistency test harness),
  `mind/grammar.py` (the recursive-descent grammar to run forward),
  `reverse_parser.py` (the explicit seed of "structure → text", whose own
  `TODO(reverse-parser)` names this exact generalization),
  `ground/usvs.py` (`sense_lemmas` / the `sense_id → lemmas` index for §3.1
  word choice), `nsm_primes.py` (`PRIMES_BY_NAME[·].exponent` for prime words),
  `clause.py` (the Clause/roles + entity-variable book for §3).
- **Supersedes:** `membrane.RELATION_TEMPLATES` (hand-written per-relation string
  templates) → the grammar run forward; curriculum-word passthrough → USVS
  `sense_id → lemma` realization for open vocabulary.
