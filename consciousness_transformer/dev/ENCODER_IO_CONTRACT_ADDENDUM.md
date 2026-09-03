# Encoder I/O Contract — ADDENDUM v1 (hard-case families)

Status: PROPOSED extension to `dev/ENCODER_IO_CONTRACT.md` (FROZEN v1). This
document does **not** modify the frozen contract; it only *adds* optional
fields. Author: hand-authoring pass (M63.1c), 2026-09-03. Motivated by, and
tested against, `dev/hand_authored_gold_v1.jsonl` (the ~30 hand-authored
records this addendum is the schema for).

## 0. Why an addendum (scope)

The FROZEN contract serializes exactly what the deterministic teacher can
produce: a fully-surfaced grounded tree over the tokens actually present.
The M62b probe (design §9) showed bin-A yield was dominated not by parser
weakness but by three construction families the teacher *structurally
cannot* emit, because each requires a constituent that is **not a surface
token in the sentence**:

1. **Imperatives** — "Sit down!" has a subject (the addressee) with no
   surface token.
2. **Interjections** — "Alas!" carries stance/appraisal meaning but no
   predicate-argument proposition.
3. **Elision / pro-drop** — "More!" / "Again!" has an elided predicate (and
   often an elided argument) recoverable only from **prior discourse**.

All three need one shared capability the frozen schema lacks: a role/predicate
filler whose grounding is **not** `tokens[i] -> MFS sense`. This addendum
defines the minimal fields for that, and nothing more.

## 1. Design invariants (kept from FROZEN)

- **Additive only.** Every FROZEN top-level, clause, role, and
  discourse-link field is still present in every hand-authored record, with
  its frozen type. A consumer that reads only the FROZEN keys still gets a
  well-formed (if under-grounded) tree — it simply ignores the extension
  keys. All new keys are **optional**: *absent* means *frozen behavior*.
- **One discriminator per level.** A single new field says "this filler /
  clause is an extension case," so a frozen consumer can branch on one key.
- **Extensions are marked in-band.** No extension changes the *meaning* of a
  frozen field; it only relaxes nullability where a non-surface filler makes
  the frozen value undefined, and every such relaxation is listed in §5.

## 2. Record level — `prior_context` (new, optional)

The elision family needs the antecedent to be *addressable*. We serialize
it inline so a record is self-contained (the encoder, at run time, reads the
same antecedent out of grounded memory — see §6).

```
"prior_context": [
  { "text": str, "gold_tree": { "clauses": [...], "discourse_links": [...] } },
  ...
]
```

| field | type | nullable | notes |
|---|---|---|---|
| `prior_context` | array[object] | **optional** (absent ⇒ `[]`) | Antecedent discourse, oldest-first. Present only for records whose gold_tree contains a `context_ref` with `scope:"prior_context"`. |
| `prior_context[k].text` | string | no | Raw antecedent sentence. |
| `prior_context[k].gold_tree` | object | no | Same shape as the FROZEN `gold_tree` (§2 of the contract). This is what makes `context_ref` indices well-defined: a pointer names `(context_index k, clause_index, target, role_index)` into exactly this structure. |

A `prior_context` entry MAY additionally carry the full FROZEN record fields
(`tokens`, `pos`, `token_sense_candidates`) when convenient; only `text` and
`gold_tree` are required to resolve a pointer.

## 3. Clause level — extension fields (all optional)

Added to the FROZEN `clauses[i]` object:

| field | type | nullable | notes |
|---|---|---|---|
| `clause_kind` | string | **optional** (absent ⇒ `"proposition"`) | Family discriminator. Enum: `"proposition"` (frozen), `"imperative"`, `"appraisal"`, `"elliptical"`. |
| `predicate_nsm_prime` | string \| null | **optional** | Canonical NSM prime name (from `nsm_ct.nsm_primes`) when the predicate IS a prime rather than a corpus verb — used by `appraisal` clauses whose predicate is `FEEL`. When set, `predicate` holds the lowercase lemma (`"feel"`) and `predicate_sense_id` is `null`. |
| `predicate_ref` | context_ref \| null | **optional** | Present iff the predicate is **elided** and inherited from context (elliptical clauses). See §4 for the object shape. When set, the FROZEN `predicate` field holds the *resolved antecedent surface token* copied for readability (e.g. `"want"`), and `predicate_sense_id` MAY be null or the copied sense; the **authoritative** grounding is the pointer, not the copied string. |
| `stance_target` | context_ref \| null | **optional** | For an `appraisal` clause that is a REACTION to a preceding proposition ("Oh no! [that clause]") — points at the reacted-to clause. `null`/absent ⇒ standalone appraisal (no target). |

Frozen fields keep their meaning: `is_question` stays `bool`; `roles` is
still the role array (now possibly holding extension fillers, §4).

## 4. Role level — extension fields (all optional)

Added to the FROZEN `clauses[i].roles[j]` object. The `relation` vocabulary
was already declared **open** in the frozen contract ("open string
vocabulary anchored by the closed core"), so appraisal clauses may use
`SUBJECT`/`OBJECT` from the closed core; no new relation labels are required
by the examples (we deliberately reuse `SUBJECT` for the feeler and `OBJECT`
for the stance complement).

| field | type | nullable | notes |
|---|---|---|---|
| `filler_kind` | string | **optional** (absent ⇒ `"surface"`) | Discriminator. `"surface"` = FROZEN (`word` is a real token, `sense_id` is its MFS sense). `"synthesized"` = filler has no surface token (imperative "you", appraisal "I"). `"context_ref"` = filler is a pointer into `prior_context`/memory. |
| `nsm_prime` | string \| null | **optional** | Canonical NSM prime grounding a `synthesized` filler when it IS a prime — `"YOU"` for the imperative addressee, `"I"` for the appraisal experiencer. Coexists with `sense_id` (which stays `null` for these entity-prime fillers). |
| `context_ref` | object \| null | **optional** | Present iff `filler_kind == "context_ref"`. THE point-to-context construct. Shape below. |

### 4.1 `context_ref` object — the point-to-preexisting-context construct

```
{
  "scope":         "prior_context" | "memory",
  "context_index": int | null,   // index into record.prior_context[]; null iff scope=="memory"
  "clause_index":  int | null,   // index into that context record's gold_tree.clauses; null iff scope=="memory"
  "target":        "predicate" | "role",
  "role_index":    int | null,   // index into the antecedent clause's roles[]; set iff target=="role"
  "memory_handle": str | null    // opaque grounded-memory node id; set iff scope=="memory"
}
```

| field | type | nullable | notes |
|---|---|---|---|
| `scope` | string | no | `"prior_context"` = the antecedent is a node in this record's serialized `prior_context` (the authorable, verifiable case). `"memory"` = the antecedent is a grounded-memory node not serialized here (the run-time case; see §6). |
| `context_index` | int | yes | Which `prior_context` entry. `null` when `scope=="memory"`. |
| `clause_index` | int | yes | Which clause within that entry's `gold_tree.clauses`. `null` when `scope=="memory"`. |
| `target` | string | no | `"predicate"` points at that clause's predicate node; `"role"` points at one of its roles. |
| `role_index` | int | yes | Which role, when `target=="role"`. `null` when `target=="predicate"`. |
| `memory_handle` | string | yes | Opaque, stable id of the grounded-memory node, when `scope=="memory"`. `null` for `scope=="prior_context"`. This is the handle the membrane/LTM assigns; the encoder emits a *selection* of it (§6), it does not mint it. |

The **same** `context_ref` object is reused by three sites: a `context_ref`
role filler (elided argument), a clause `predicate_ref` (elided predicate),
and an appraisal `stance_target` (reaction attachment). One construct, three
attachment points.

## 5. Nullability relaxations (the only ones; each marked)

These apply **only** to a role whose `filler_kind` is present and
non-`"surface"`, or a clause carrying a `predicate_ref`. A frozen `surface`
filler is unchanged.

- **`roles[j].word`** — FROZEN: non-nullable (every filler was a surface
  token). RELAXED: MAY be `null` when `filler_kind=="context_ref"` and the
  fragment supplies no surface token for that slot (e.g. the elided argument
  of "More!"). For `filler_kind=="synthesized"` it holds the canonical lemma
  (`"you"`, `"I"`) — never null. For readability, a `context_ref` filler MAY
  instead copy the resolved antecedent surface string into `word`; either is
  legal, and the pointer (not `word`) is authoritative.
- **`clauses[i].predicate`** — FROZEN: non-nullable. Under `predicate_ref`,
  it is still a non-null string, but it is a *copied antecedent surface* for
  readability, not a token of this sentence; the pointer is authoritative.
- **`roles[j].sense_id`** and **`predicate_sense_id`** — already nullable in
  FROZEN; stay `null` for entity-prime and `context_ref` fillers. No change,
  noted for completeness.

Everything else keeps its frozen type and nullability.

## 6. The run-time boundary (why `scope` exists) — and the open edge

`scope:"prior_context"` is the **authorable / gold** form: the antecedent is
serialized, the pointer is a plain array index, a human can write it and a
checker can verify it dereferences. This is what `hand_authored_gold_v1.jsonl`
uses throughout.

`scope:"memory"` is the **run-time** form the design (§1: "every answer routes
through a memory read") actually needs: at inference the antecedent lives in
grounded memory under a `memory_handle`, not in a serialized array. The
encoder does not *invent* a pointer — it **selects** an antecedent from the
retrieved discourse/memory candidate set, exactly as it selects a USVS sense
from the retrieved sense-candidate set (contract §4). That symmetry is the
whole reason this is on-architecture rather than a bolt-on: *grounding =
choose-a-sense-from-retrieved-candidates; context-ref = choose-a-node-from-
retrieved-memory-candidates.*

**Honest gap (flagged for the lead):** the two `scope` values are the same
construct at authoring time and at run time, but nothing in this addendum
specifies how a gold `prior_context` index is *aligned* to the live
`memory_handle` the membrane assigns at inference — i.e. how the training
target ("point at prior-clause-0's predicate") becomes a supervised signal
over a retrieval candidate set whose ids are only known at run time. That
alignment is the point-to-context construct's real cost, and it is the
subject of the grammar-format proposal's headline open question
(`ENCODER_GRAMMAR_FORMAT_PROPOSAL.md` §4).

## 7. Interjection grounding — where valence lives (and a coverage hole)

Per the director decision, **valence is not hard-coded**: an appraisal
clause grounds its stance complement to the interjection's **USVS sense id**
and lets the eval/antonym axis of that sense carry good-vs-bad implicitly.
The canonical explication is the NSM template `I FEEL something (good/bad)`,
authored as: `SUBJECT` = synthesized `I` (`nsm_prime:"I"`), predicate =
`feel` (`predicate_nsm_prime:"FEEL"`), `OBJECT` = the stance complement
grounded to the interjection sense. We do **not** write a literal `GOOD`/`BAD`
prime into the tree — that polarity is read off the grounded sense.

**Coverage hole (flagged, real friction).** This works only for
interjections that HAVE a WordNet/USVS sense — typically those with a
content-word homograph (`shit`, `damn`, `hell`, `wow`). Pure primary
interjections (`ugh`, `alas`, `ouch`, `oh`, `ah`, `oops`, `phew`, `yuck`,
`hurray`) have **no WordNet synset**, so `usvs.senses_of()` returns `[]` and
there is *no sense to ground to*. For those the stance complement is authored
with `filler_kind:"synthesized"`, `sense_id:null`, and a marker
`stance_lexicon:"needed"` (see below). Getting valence "from USVS not
hard-coded" therefore **presupposes a USVS extension** — a small
interjection→pseudo-sense table carrying eval-axis coordinates — that does
not exist today. This is called out as friction in the final report.

To keep the appraisal authorable without hard-coding polarity, an appraisal
clause MAY carry one more optional clause field:

| field | type | nullable | notes |
|---|---|---|---|
| `stance_lexicon` | string \| null | **optional** | `"grounded"` = the stance complement grounded to a real USVS sense (valence from its eval axis). `"needed"` = no USVS sense exists for this interjection; valence requires the proposed interjection stance-lexicon extension. `null`/absent ⇒ not an appraisal clause. |

This field records *where the valence comes from* without ever storing the
valence itself — keeping the director's "valence from USVS, not hard-coded"
invariant even for the coverage-hole cases (it stores "unresolved," not a
polarity).

## 8. Worked shape summary (one of each family)

Imperative "Sit down !" — a `proposition`-shaped clause marked
`clause_kind:"imperative"` with a synthesized `SUBJECT`:
```
{"predicate":"sit","predicate_sense_id":"sit.v.01","is_question":false,
 "clause_kind":"imperative",
 "roles":[{"relation":"SUBJECT","filler_kind":"synthesized","word":"you",
           "is_entity":true,"sense_id":null,"nsm_prime":"YOU"}]}
```

Appraisal "Alas !" (no WN sense) — `I FEEL something`, valence unresolved:
```
{"predicate":"feel","predicate_sense_id":null,"predicate_nsm_prime":"FEEL",
 "is_question":false,"clause_kind":"appraisal","stance_lexicon":"needed",
 "stance_target":null,
 "roles":[{"relation":"SUBJECT","filler_kind":"synthesized","word":"I",
           "is_entity":true,"sense_id":null,"nsm_prime":"I"},
          {"relation":"OBJECT","filler_kind":"synthesized","word":"alas",
           "is_entity":false,"sense_id":null}]}
```

Elliptical "More !" after "The dog wants food ." — elided predicate and
argument both point into `prior_context[0].gold_tree.clauses[0]`:
```
{"predicate":"want","predicate_sense_id":null,"is_question":false,
 "clause_kind":"elliptical",
 "predicate_ref":{"scope":"prior_context","context_index":0,"clause_index":0,
                  "target":"predicate","role_index":null,"memory_handle":null},
 "roles":[
   {"relation":"SUBJECT","filler_kind":"context_ref","word":"dog","is_entity":false,
    "sense_id":null,
    "context_ref":{"scope":"prior_context","context_index":0,"clause_index":0,
                   "target":"role","role_index":0,"memory_handle":null}},
   {"relation":"OBJECT","filler_kind":"context_ref","word":"food","is_entity":false,
    "sense_id":null,
    "context_ref":{"scope":"prior_context","context_index":0,"clause_index":0,
                   "target":"role","role_index":1,"memory_handle":null},
    "quantity":"MORE"}]}
```
(The `quantity:"MORE"` key there is the one intensifier note the "More!"
example needed — an optional role annotation grounding the surface fragment
token to the NSM `MORE` prime; it is local to that example and not a required
extension.)
