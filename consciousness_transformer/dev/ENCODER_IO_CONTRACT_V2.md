# Encoder I/O Contract — CANONICAL v2 (candidates-first)

Status: CANONICAL. Supersedes the FROZEN v1 contract
(`dev/ENCODER_IO_CONTRACT.md`) and reconciles the two divergent hand-draft
addenda (`ENCODER_IO_CONTRACT_ADDENDUM.md` on branches `encoder-handgold-v1`
and `encoder-handgold-v2`) into one schema. Author: encoder step 1 pass,
2026-09-04. Embodies the **CORE BOUNDARY DECISION** and the **Appraisal
grounding DECISION** (both lead, 2026-09-04; `RESEARCH_NOTES.md`).

This document describes data shape only. It does not modify, and is not
itself, any `src/` model or parser code.

---

## 0. The one idea (why v2 exists)

**v1 was a COMMITTED tree.** The teacher looked at each sentence, picked one
parse, picked one (most-frequent) sense per word, and serialized that single
grounded tree as `gold_tree`. Every ambiguity was silently resolved to the
teacher's default — and that default was, provably, often wrong (v1 §6.1:
`"bats"` committed to `balmy.s.01` "crazy", never the animal).

**v2 is a CANDIDATE LATTICE.** The encoder is now defined as *literally just a
token→tree-SET transducer*. It **emits candidates and commits to nothing**:

1. **All real sense candidates per content node** — the full `senses_of`
   list, no chosen sense.
2. **The parser's top-k POSSIBLE trees** — a parse *forest*. Structural and
   attachment ambiguity is *preserved*, not resolved to one tree.
3. **Unresolved link slots** — pronouns, coref, elided predicates/args — each
   carrying a *candidate set*, not a resolution.

The **comprehension model** (downstream — a GRU over grounded memory, NOT the
encoder) resolves *all* of it — word-sense disambiguation, pronoun/coref
linking, elision-fill, attachment — as **one primitive**:

> **select-from-memory-conditioned-candidates.**

Why draw the boundary here:

- **No parser bleed.** The encoder stays a pure, deterministic transducer. It
  never consults memory, never disambiguates, so it can never leak
  comprehension-side inference into the "objective" parse. What it emits is a
  function of the string and the lexicon alone.
- **Gold becomes deterministic + label-free.** v1's gold needed a *correct
  sense* per word (the MFS/WSD gold problem, v1 §3–§4) — a label nobody could
  produce at scale without annotation. v2's gold is the candidate *set*, which
  the teacher already produces natively (`senses_of` returns every sense;
  the parser exposes `max_hypotheses=20` top-k trees). No correct-sense label,
  no `sense_chooser` wiring, on the encoder side. WSD is comprehension's job,
  trained later by the K-12 read-then-answer ladder.

Everything below is the serialization of that lattice.

---

## 1. Top-level record shape

```
{
  "text": str,
  "tokens": [str, ...],
  "pos": [str, ...],
  "lattice": { "trees": [...], "discourse_links_per_tree": [...] },
  "token_sense_candidates": [...],
  "context": [ { "text": str, "lattice": {...} }, ... ]   // OPTIONAL
}
```

| field | type | nullable | notes |
|---|---|---|---|
| `text` | string | no | Raw sentence string as fed to the parser (whitespace-tokenizable; punctuation space-separated — corpus convention). Unchanged from v1. |
| `tokens` | array[string] | no | Surface tokens in order, from the parser's own tagger. Unchanged from v1. |
| `pos` | array[string] | no | Coarse POS per token (`quantum_parser`'s `Tag` enum member name). Same length as `tokens`. Unchanged from v1. Note: this enum has **no `INTJ`** — see §6. |
| `lattice` | object | no | **Replaces v1's `gold_tree`.** The candidate forest. See §2. |
| `token_sense_candidates` | array[object] | no (may be `[]`) | The per-token sense-candidate table (retrieval-conditioning input). Unchanged from v1 §4; it is the shared, single-source-of-truth store the lattice's sense-slots point into (§4.2). See §2.3. |
| `context` | array[object] | **optional** (absent ⇒ `[]`) | Prior discourse the record's reference/elision slots may resolve against, oldest-first. Present only when some slot has `retrieval.source == "context"`. See §4.4. |

`token_sense_candidates[i]` keeps its exact v1 shape:
`{ "index": int, "token": str, "sense_candidates": [str, ...], "chosen_sense": str }`,
sparse over `tokens` (one entry per token the USVS lemma index covers).
**One change of *interpretation*, not of shape:** in v1, `chosen_sense`
(= `sense_candidates[0]`, MFS) was the teacher's committed pick and fed
`gold_tree`. In v2 the encoder commits to no pick, so `chosen_sense` is
**purely informational** — "what MFS *would* have said" — and is *not* a
target. The target is the `sense_candidates` list itself.

---

## 2. `lattice` — the candidate forest

```
"lattice": {
  "trees": [
    { "clauses": [ <clause>, ... ] },
    { "clauses": [ <clause>, ... ] },
    ...
  ],
  "discourse_links_per_tree": [
    [ <discourse_link>, ... ],   // links for trees[0]
    [ <discourse_link>, ... ],   // links for trees[1]
    ...
  ]
}
```

| field | type | nullable | notes |
|---|---|---|---|
| `trees` | array[object] | no (≥1) | The parser's **top-k candidate trees** (a forest), most-plausible-first, capped at the teacher's `max_hypotheses`. Each is a full structural hypothesis: a different attachment/segmentation is a different tree. A fully-unambiguous sentence yields a one-element forest — that is the v1 case as a degenerate v2 lattice. |
| `trees[t].clauses` | array[object] | no | The clauses of tree `t`. Same **structure** as v1's `gold_tree.clauses`, but each node grounds via a *candidate slot* (§4), not a committed `sense_id`. See §3. |
| `discourse_links_per_tree` | array[array] | no (may be `[[]...]`) | Parallel to `trees`: entry `t` is the discourse-link list for `trees[t]` (coordination/negation may itself be structurally ambiguous, so links are per-tree). Same link shape as v1 §2.3. |

### 2.1 Structural ambiguity lives in the forest, not in a node

If "I saw the man with the telescope" has two attachments, the lattice
carries **two trees** — one where `with the telescope` is a PP role of the
predicate, one where it modifies `the man`. The encoder does **not** pick.
Attachment selection is comprehension's job, resolved by the *same*
select-from-candidates primitive as sense and coref (§7).

### 2.2 Per-tree senses are shared, not duplicated

Sense candidates depend only on the token (its lemma), never on attachment.
So a content node in any tree grounds by `token_index` into the single
top-level `token_sense_candidates` table (§4.2); the candidate *list* is not
re-serialized per tree. This keeps a k-tree forest compact. A packed /
shared-subtree forest representation is a permitted future optimization; the
top-k-full-trees form here is the canonical gold serialization.

### 2.3 Where the old single `gold_tree` went

`gold_tree` (v1) ≅ `lattice.trees[0]` **with every node's committed
`sense_id` replaced by that node's candidate slot**. A v1 consumer can
recover a v1-shaped tree by taking `trees[0]` and, at each sense-slot,
reading `token_sense_candidates[token_index].chosen_sense`. v2 is therefore
a strict superset of v1's information.

---

## 3. Clause and node structure (the structural skeleton)

A `clause` keeps v1's skeleton; only *grounding* changes (a `grounding`
object per §4 replaces the scalar `*_sense_id`):

```
{
  "predicate": str | null,
  "predicate_grounding": <grounding>,
  "is_question": bool,
  "utterance_kind": "proposition" | "imperative" | "interjection",   // default "proposition"
  "roles": [
    {
      "relation": str,
      "word": str | null,
      "token_index": int | null,
      "is_entity": bool,
      "grounding": <grounding>
    },
    ...
  ]
}
```

| field | type | nullable | notes |
|---|---|---|---|
| `predicate` | string | **yes** | Surface predicate token, or `null` when the predicate is elided (`grounding.type == "elision"`) or synthesized. Copied for readability; the `grounding` is authoritative. |
| `predicate_grounding` | object | no | The predicate's candidate slot / grounding (§4). Normal verb → `type:"sense"`; elided predicate ("More!") → `type:"elision"`. |
| `is_question` | bool | no | `Clause.is_question`. Unchanged from v1. |
| `utterance_kind` | string | no (default `"proposition"`) | Speech-act shape of the clause. `"proposition"` = v1 default (assertion/question). `"imperative"` = has a synthesized addressee subject (§5). `"interjection"` = an exclamation grounded to its literal/gloss sense (§6). **This is the ONLY appraisal-related marker; there is no FEEL/GOOD/BAD explication** — see §6. |
| `roles[j].relation` | string | no | The role label. **Frozen role vocabulary kept unchanged from v1:** closed core `SUBJECT`, `OBJECT`, `INDIRECT_OBJECT`, `PLACE`, `SOURCE`, `AGENT`, `RECIPIENT`, plus open PP-roles (an uppercased preposition, e.g. `WITH`, `FOR`, `OF`, `ABOUT`). |
| `roles[j].word` | string | **yes** | Surface token filling the role, or `null` for a synthesized/elided filler with no surface token. Copied for readability; `grounding` is authoritative. |
| `roles[j].token_index` | int | **yes** | Index into `tokens`/`pos` for `word`; `null` when there is no surface token (synthesized or elided fillers). Also the key into `token_sense_candidates` for a `sense` slot. |
| `roles[j].is_entity` | bool | no | `is_entity(word)` — true for names/pronouns (this grammar's referent variables). Unchanged from v1. |
| `roles[j].grounding` | object | no | The node's candidate slot / grounding (§4). |

Discourse links (`discourse_links_per_tree[t][k]`) keep v1 §2.3 exactly:
`{ "coordinator": str, "prime": str | null, "clause_i": int, "clause_j": int }`.

---

## 4. THE unified construct — `grounding` (candidate slot / unresolved slot)

This is the heart of v2. **Every** node — predicate or role — grounds through
one object of type `grounding`. Three of its five `type`s are **unresolved
slots**: they carry a *candidate set the encoder does not choose from*. The
lead's core insight is that **sense-ambiguity, pronoun/coref, and elision are
the same construct** — a role/predicate position + a candidate set + how the
candidates are retrieved — differing only in *which source* the candidates
come from. The comprehension model resolves all three with the identical
select-from-candidates operation.

### 4.1 The shape

```
"grounding": {
  "type":       "sense" | "reference" | "elision" | "prime" | "entity",
  "candidates": [str, ...] | null,          // the candidate SET (unresolved slots); null = resolved OR retrieved-at-runtime
  "retrieval":  {                            // present iff type ∈ {sense, reference, elision}
    "source":   "lexicon" | "self" | "context" | "memory",
    "method":   str,                         // how the candidate set is produced
    "ref":      { ... } | null               // addressing into the source (see §4.3); null for lexicon
  },
  "prime":      str | null                   // present iff type == "prime"
}
```

| `type` | resolved? | what it is | `candidates` | `retrieval.source` |
|---|---|---|---|---|
| `sense` | **unresolved slot** | a content word whose SENSE is undetermined | the full `senses_of(token)` list | `lexicon` |
| `reference` | **unresolved slot** | a pronoun / coref mention whose REFERENT is undetermined | candidate antecedent nodes, or `null` if the referent lives in run-time memory | `self` / `context` / `memory` |
| `elision` | **unresolved slot** | an elided predicate/argument whose FILLER is undetermined | candidate antecedent nodes, or `null` | `context` / `memory` / `self` |
| `prime` | resolved | grounds directly to an NSM prime (e.g. synthesized `YOU`) | `null` | — |
| `entity` | resolved | a bare referent variable (name) needing no sense and no link | `null` | — |

The three unresolved `type`s share **exactly** `candidates` + `retrieval`.
That is the whole point: **one shape, three instances.** A comprehension
model can iterate every unresolved slot in the lattice uniformly — collect
every `grounding` with `candidates`/`retrieval` present, condition on memory,
select — without caring whether it is disambiguating a sense, binding a
pronoun, or filling an ellipsis.

### 4.2 `sense` slots point into `token_sense_candidates`

For `type:"sense"`, `candidates` is the token's `senses_of` list. To avoid
duplicating that list across every tree in the forest, the canonical
serialization sets `candidates: null` on the node and relies on
`token_index → token_sense_candidates[·].sense_candidates` as the single
source of truth. The worked examples (§8) inline the list for readability;
either is legal, `token_sense_candidates` is authoritative when they differ.
(A content token the USVS lemma index does not cover has no
`token_sense_candidates` entry and grounds as `type:"entity"` — the v1
"ungrounded content word" case, now explicit rather than a `null sense_id`.)

### 4.3 `ref` — addressing an antecedent (reference / elision)

The `ref` object addresses the antecedent a reference/elision slot may point
at. It **keeps v2-addendum's `source`** vocabulary — with `lexicon` added so
sense-slots use the identical `retrieval` envelope:

```
"ref": {
  "source": "self" | "context" | "memory",
  "clause": int | null,       // clause index: into this record's lattice tree (self) or into context[·].lattice (context)
  "context_index": int | null,// which context[] entry (source == "context"); null otherwise
  "tree_index": int | null,   // which tree the antecedent clause lives in (default 0); null when unambiguous
  "slot": "predicate" | "role" | null,
  "role_index": int | null,   // index into the antecedent clause's roles[] when slot == "role"
  "handle": str | null        // opaque grounded-memory node id (source == "memory")
}
```

- **`source: "self"`** — the antecedent is *another node in this same
  record's lattice* (e.g. a cataphoric/same-utterance coref). `clause` /
  `slot` / `role_index` address it; `context_index` and `handle` are `null`.
- **`source: "context"`** — the antecedent is a node in this record's
  serialized `context[]` store (§4.4). `context_index` picks the entry,
  `clause` / `slot` / `role_index` address within its lattice. This is the
  **authorable / gold** form: a human can write the pointer and a checker can
  dereference it.
- **`source: "memory"`** — the antecedent is a grounded-memory node under an
  opaque `handle`, not serialized here. This is the **run-time** form: at
  inference the encoder emits `candidates` from the *retrieved* memory
  candidate set (it does not mint handles), exactly as a `sense` slot's
  candidates come from the retrieved lexicon set. `source:"context"` at
  authoring time aligns to `source:"memory"` at run time.

### 4.4 `context[]` — the antecedent store

```
"context": [
  { "text": str, "lattice": { "trees": [...], "discourse_links_per_tree": [...] } },
  ...
]
```

Oldest-first prior discourse. Each entry's `lattice` is the **same shape** as
the top-level `lattice`, so a `ref` with `source:"context"` needs no new
addressing scheme — `(context_index, tree_index, clause, slot, role_index)`
lands on a node. An entry MAY carry the full record fields
(`tokens`/`pos`/`token_sense_candidates`) when convenient; only `text` and
`lattice` are required to dereference a pointer.

> **Reconciliation note.** v1-addendum stored antecedents in a `prior_context`
> **array** and used a **flat** `context_ref` (`scope`, `context_index`,
> `clause_index`, `target`, `memory_handle`). v2-addendum stored a **single**
> `context` object and used a **nested** `ref` (`source: self|context|memory`,
> `clause`, `slot`, `handle`). Canonical v2 keeps **v2's nested `ref` and its
> `source` vocabulary** (per the lead directive) but restores v1's **array**
> `context[]` (multi-sentence antecedents are real), and — the substantive
> move — **promotes the pointer out of a role-only add-on into the universal
> `grounding` construct** so that *sense* ambiguity rides the same envelope
> (`retrieval.source: "lexicon"`). What both addenda modeled as an
> elision/coref bolt-on is, in v2, the general unresolved-slot primitive.

---

## 5. Synthesized subject + frozen role vocabulary (kept from the addenda)

The **imperative synthesized subject** is kept: an imperative clause
(`utterance_kind:"imperative"`) has a `SUBJECT` role with **no surface
token** (`word:"you"`, `token_index:null`, `is_entity:true`) grounded to the
prime `YOU`:

```json
{ "relation": "SUBJECT", "word": "you", "token_index": null, "is_entity": true,
  "grounding": { "type": "prime", "prime": "YOU", "candidates": null } }
```

This is a *resolved* grounding (`type:"prime"`), not a candidate slot — the
grammar rule licenses exactly one filler (the addressee = prime `YOU`), so
there is nothing for comprehension to select. The **frozen role vocabulary**
(closed core + open PP-roles, §3) is unchanged. Both addenda's `synthesized`
/ `synth_kind` / `filler_kind:"synthesized"` markers collapse into
`grounding.type:"prime"`.

---

## 6. Interjections — literal/gloss sense, NO appraisal explication

Per the **Appraisal grounding DECISION** (lead, 2026-09-04), the appraisal
over-build in **both** addenda is **STRIPPED**. Specifically removed:

- v1-addendum's `clause_kind:"appraisal"`, `predicate_nsm_prime:"FEEL"`,
  `stance_target`, `stance_lexicon`, and the `I FEEL something (good/bad)`
  explication.
- v2-addendum's `clause_type:"appraisal"`, the `appraisal` sub-object
  (`trigger`, `reaction_sense_id`, `valence:"from_sense"`,
  `primes:["I","FEEL"]`, `scope`), the separate **reaction-sense inventory**,
  and the `EXPERIENCER` synthesized speaker.

**What replaces them.** An interjection is just a content utterance that
grounds like any other word:

- It grounds to its **literal USVS sense candidate(s)** — `"shit!"` →
  `senses_of("shit")` = `[shit.n.01, ...]`, which **is** in the token's
  retrieved candidates. No second index, no emotion table.
- **Pure interjections** with no literal synset (`alas`, `ugh`, `oh`, `oops`)
  are handled by **adding gloss-derived senses to USVS** (the existing
  `ground/usvs.py` gloss→prime→coordinate pipeline) — deterministic lexical
  data, *not* a trained model — after which they ground like any sense.
- The clause is marked `utterance_kind:"interjection"` (an
  interjection/exclamation utterance) so downstream code knows the speech-act
  shape. That marker + the grounded sense candidates is the **entire** encoder
  output for an interjection.

**Explicitly out of scope for the encoder:** connotation / appraisal —
"how does the speaker *feel*?", the good-vs-bad valence, the projection of
that valence onto a target. That is a **comprehension-side inference** (a K-12
"how does Bob feel?" question), a *learned* operation reading valence off the
GOOD↔BAD eval axis already anchored in USVS, generalizing to epithets and
sarcasm — not 15 hard-coded words, and not the encoder's to emit. The encoder
emits the grounded sense candidates; comprehension derives the feeling.

This also removes v2-addendum's friction #2 (the reaction-sense target not
being in the token's candidates): with literal/gloss grounding the target
**is** in the candidates, so the "appraisals need a separate retrieval index"
problem dissolves.

---

## 7. Encoder EVAL contract — candidate-set recall, not the pick

Because the encoder commits to nothing, **it is never scored on choosing the
right sense, attachment, antecedent, or filler.** It is scored on whether the
**gold structure is present among the candidates it emitted** — i.e. recall
of the gold over the lattice:

- **Sense recall.** For each gold content node, the gold sense-id ∈ the
  node's emitted `candidates`. (`senses_of` is exhaustive, so this is a check
  that the encoder attached the right candidate *slot* to the right token, not
  a WSD score.)
- **Structure recall.** The gold tree ∈ `lattice.trees` (the correct
  attachment/segmentation is one of the top-k), and the gold discourse links
  ∈ that tree's link list.
- **Slot recall.** Each gold unresolved slot (pronoun/coref/elision) is
  emitted at the right position with the gold antecedent ∈ its candidate set
  (or, for `source:"memory"`, the gold antecedent ∈ the retrieved memory
  candidate set).

> **Metric.** The encoder is **correct on a record iff every gold element is
> recalled among the candidates** — gold sense ∈ node candidates, gold tree ∈
> forest, gold antecedent ∈ slot candidates — for every node/slot. Report
> per-node/per-slot/per-tree recall and the all-gold-recalled record rate.
> **Precision (extra candidates) is NOT penalized** on the encoder: emitting a
> superset is correct-by-design; narrowing the set is comprehension's job and
> is scored there. A *missing* gold candidate (a sense `senses_of` didn't
> return, a tree outside top-k, an antecedent outside the retrieved set) is the
> only encoder failure.

This is what makes the gold **label-free**: no annotator ever had to mark the
correct sense/attachment. The gold is the candidate *set* the teacher already
produces, and the encoder's job is to reproduce that set.

---

## 8. Three fully-worked examples (v2 lattice format)

Sense candidate lists are inlined on nodes for readability; canonically they
live once in `token_sense_candidates` and nodes reference by `token_index`
(§4.2). Discourse-link lists omitted where empty.

### 8.1 Ambiguous sense — "the bat flew"

`"bat"` is genuinely ambiguous (animal vs. club); the encoder emits **all**
senses and picks none. One tree (no structural ambiguity here).

```json
{
  "text": "the bat flew .",
  "tokens": ["the", "bat", "flew", "."],
  "pos": ["DET", "NOUN", "VERB", "PUNCT"],
  "lattice": {
    "trees": [
      { "clauses": [
        {
          "predicate": "flew", "is_question": false, "utterance_kind": "proposition",
          "predicate_grounding": {
            "type": "sense", "candidates": ["fly.v.01", "fly.v.06", "fly.v.09"],
            "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null }
          },
          "roles": [
            { "relation": "SUBJECT", "word": "bat", "token_index": 1, "is_entity": false,
              "grounding": {
                "type": "sense",
                "candidates": ["bat.n.01", "bat.n.02", "bat.n.05", "cricket_bat.n.01"],
                "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null }
              } }
          ]
        }
      ] }
    ],
    "discourse_links_per_tree": [[]]
  },
  "token_sense_candidates": [
    { "index": 1, "token": "bat",
      "sense_candidates": ["bat.n.01", "bat.n.02", "bat.n.05", "cricket_bat.n.01"],
      "chosen_sense": "bat.n.01" },
    { "index": 2, "token": "flew",
      "sense_candidates": ["fly.v.01", "fly.v.06", "fly.v.09"], "chosen_sense": "fly.v.01" }
  ]
}
```

Contrast with v1: v1 would have committed `bat` to `chosen_sense` and thrown
the rest away. v2 keeps the whole set; whether this `bat` is the animal
(compatible with `flew`) is **comprehension's** call, made against memory.

### 8.2 Pronoun / coref — "Mary saw Sue . She waved ."

`"She"` is a `reference` slot. Its candidate antecedents are the two prior
entities; the encoder emits both and links to neither. Shown with an
in-record (`source:"self"`) candidate set plus the run-time memory form.

```json
{
  "text": "she waved .",
  "tokens": ["she", "waved", "."],
  "pos": ["PRON", "VERB", "PUNCT"],
  "lattice": {
    "trees": [
      { "clauses": [
        {
          "predicate": "waved", "is_question": false, "utterance_kind": "proposition",
          "predicate_grounding": {
            "type": "sense", "candidates": ["wave.v.01", "wave.v.02", "brandish.v.01"],
            "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null }
          },
          "roles": [
            { "relation": "SUBJECT", "word": "she", "token_index": 0, "is_entity": true,
              "grounding": {
                "type": "reference",
                "candidates": ["ctx:0/0/role/0", "ctx:0/0/role/1"],
                "retrieval": {
                  "source": "context", "method": "coref",
                  "ref": { "source": "context", "context_index": 0, "tree_index": 0,
                           "clause": 0, "slot": "role", "role_index": null, "handle": null }
                }
              } }
          ]
        }
      ] }
    ],
    "discourse_links_per_tree": [[]]
  },
  "token_sense_candidates": [
    { "index": 1, "token": "waved",
      "sense_candidates": ["wave.v.01", "wave.v.02", "brandish.v.01"], "chosen_sense": "wave.v.01" }
  ],
  "context": [
    { "text": "mary saw sue .",
      "lattice": { "trees": [ { "clauses": [
        { "predicate": "saw", "is_question": false, "utterance_kind": "proposition",
          "predicate_grounding": { "type": "sense", "candidates": ["see.v.01", "saw.v.01"],
            "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null } },
          "roles": [
            { "relation": "SUBJECT", "word": "mary", "token_index": 0, "is_entity": true,
              "grounding": { "type": "entity", "candidates": null } },
            { "relation": "OBJECT", "word": "sue", "token_index": 2, "is_entity": true,
              "grounding": { "type": "entity", "candidates": null } }
          ] }
      ] } ], "discourse_links_per_tree": [[]] } }
  ]
}
```

The `reference` slot's `candidates` enumerate the two antecedent entities
(here shown as readable `ctx:context/tree/slot/role` handles into `context[0]`
— `Mary` at `role_index 0`, `Sue` at `role_index 1`). The encoder emits both;
comprehension binds one against memory. At **run time** the same slot would
carry `retrieval.source:"memory"` with `candidates` drawn from the retrieved
entity-candidate set and `ref.handle` addressing memory — identical construct,
different source. Note `Mary`/`Sue` ground as `type:"entity"` (referent
variables, no sense, no link needed).

### 8.3 Elision with context — "The dog wants food . More !"

`"More !"` has an **elided predicate** (inherit `want` from the prior clause)
and an **elided argument** (inherit `food`). Both are `elision` slots pointing
into `context[0]`. `"more"` itself is a surface quantity token grounded to its
own sense candidates.

```json
{
  "text": "more !",
  "tokens": ["more", "!"],
  "pos": ["ADJ", "PUNCT"],
  "lattice": {
    "trees": [
      { "clauses": [
        {
          "predicate": null, "is_question": false, "utterance_kind": "proposition",
          "predicate_grounding": {
            "type": "elision", "candidates": ["ctx:0/0/predicate"],
            "retrieval": {
              "source": "context", "method": "elision_inherit_predicate",
              "ref": { "source": "context", "context_index": 0, "tree_index": 0,
                       "clause": 0, "slot": "predicate", "role_index": null, "handle": null }
            }
          },
          "roles": [
            { "relation": "OBJECT", "word": null, "token_index": null, "is_entity": false,
              "grounding": {
                "type": "elision", "candidates": ["ctx:0/0/role/1"],
                "retrieval": {
                  "source": "context", "method": "elision_inherit_arg",
                  "ref": { "source": "context", "context_index": 0, "tree_index": 0,
                           "clause": 0, "slot": "role", "role_index": 1, "handle": null }
                }
              } },
            { "relation": "QUANTITY", "word": "more", "token_index": 0, "is_entity": false,
              "grounding": {
                "type": "sense", "candidates": ["more.a.01", "more.s.02", "more.r.01"],
                "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null }
              } }
          ]
        }
      ] }
    ],
    "discourse_links_per_tree": [[]]
  },
  "token_sense_candidates": [
    { "index": 0, "token": "more",
      "sense_candidates": ["more.a.01", "more.s.02", "more.r.01"], "chosen_sense": "more.a.01" }
  ],
  "context": [
    { "text": "the dog wants food .",
      "lattice": { "trees": [ { "clauses": [
        { "predicate": "wants", "is_question": false, "utterance_kind": "proposition",
          "predicate_grounding": { "type": "sense", "candidates": ["want.v.01", "want.v.03", "want.v.04"],
            "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null } },
          "roles": [
            { "relation": "SUBJECT", "word": "dog", "token_index": 1, "is_entity": false,
              "grounding": { "type": "sense", "candidates": ["dog.n.01", "frump.n.01", "cad.n.01"],
                "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null } } },
            { "relation": "OBJECT", "word": "food", "token_index": 3, "is_entity": false,
              "grounding": { "type": "sense", "candidates": ["food.n.01", "food.n.02", "food.n.03"],
                "retrieval": { "source": "lexicon", "method": "lemma_senses", "ref": null } } }
          ] }
      ] } ], "discourse_links_per_tree": [[]] } }
  ]
}
```

The elided predicate and argument are the **same `grounding` shape** as the
pronoun in §8.2 and the sense slot in §8.1 — only `type` and
`retrieval.source`/`method` differ. The encoder posits the empty slots (a
grammar-licensed capability) and hands comprehension the candidate
antecedents; it does **not** fill them.

---

## 9. What v2 supersedes / reconciles (summary table)

| concern | v1 (frozen) | v1-addendum | v2-addendum | **canonical v2** |
|---|---|---|---|---|
| top-level tree field | `gold_tree` (one tree) | + `prior_context[]` | + single `context` | **`lattice.trees[]`** (forest) + `context[]` |
| sense per node | one `sense_id` (MFS) | same | context-correct `sense_id` | **candidate list** (`grounding.type:"sense"`), no pick |
| candidate table | `token_sense_candidates` (informational) | same | same | **same**, now the sense-slot source of truth |
| context pointer | — | flat `context_ref`, `scope`, `prior_context` array | nested `ref`, `source:self\|context\|memory`, `context` object | **nested `ref`, `source`** kept; `context[]` array restored; folded into `grounding` |
| synthesized subject | — | `filler_kind:"synthesized"`, `nsm_prime` | `synthesized`, `synth_kind` | **`grounding.type:"prime"`** |
| interjection | grounding-fail | `clause_kind:"appraisal"` + FEEL explication | `clause_type:"appraisal"` + reaction-sense inventory | **STRIPPED** → literal/gloss sense + `utterance_kind:"interjection"`; appraisal is comprehension-side |
| unresolved slot | — | elision/coref-specific `context_ref` | elision/coref-specific `ref` | **universal `grounding`** — sense/reference/elision one shape |
| eval | tree/sense agreement (needs correct-sense label) | — | — | **candidate-set recall** (label-free) |

**One committed tree → candidate lattice.** That is the v1→v2 pivot in a line.

---

## 10. What is deliberately NOT in the encoder output

The encoder emits candidates; it resolves nothing. All of the following are
**comprehension-side** (downstream GRU-over-memory), out of scope here:

- **Word-sense disambiguation.** The encoder emits every sense; the pick is
  comprehension's.
- **Attachment / structural disambiguation.** The encoder emits the top-k
  forest; the pick is comprehension's.
- **Pronoun / coref resolution.** The encoder emits candidate antecedents;
  the bind is comprehension's.
- **Elision fill.** The encoder posits the empty slot + candidates; the fill
  is comprehension's.
- **Connotation / appraisal / valence** (§6). "How does the speaker feel?" is
  a learned inference over the grounded structure, not an encoder output.
- **Speaker/time metadata.** Carried by the membrane provenance log (design
  §9.3), not this record.

All five resolutions are the **one** comprehension primitive:
select-from-memory-conditioned-candidates. The encoder's contract is only to
make sure the right candidate is in the set.
