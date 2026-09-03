# Encoder I/O Contract — FROZEN v1

Status: FROZEN for M63 Step 1 (dev/UNIVERSAL_ENCODER_DESIGN.md §10). This is
the exact serialization schema of `runs/encoder_gold_v1.jsonl`, produced by
`scripts/build_encoder_gold_v1.py`. It is the learned universal encoder's
**output contract** (§2.3: "same schema the deterministic parser +
input_encoder produce today") plus its **retrieval-conditioning input**
(§2.2 item 2: "retrieved USVS sense candidates for the tokens present").

This document describes data shape only. It does not modify, and is not
itself, any `src/` model or parser code.

## 0. Provenance (how a record is made)

One JSONL record = one sentence for which the deterministic teacher produced
a **full grounded tree**:

1. Sentence text -> `nsm_ct.input_encoder.ParserInputEncoder._tag` (POS
   tagger) -> `tokens`/`pos`.
2. Sentence text -> `ParserInputEncoder._parse_graph_one` (the
   `quantum_parser` deterministic parser, default `ParserConfig` — 30s
   wall-clock cap) -> a flat hypothesis graph
   (`nsm_ct.quantum_adapter.HypGraph`), or `None` on failure.
3. Graph -> `nsm_ct.clause.extract_discourse` -> `(clauses, discourse_links)`.
   "Full grounded tree" = at least one clause with a real (non-punctuation)
   SUBJECT — the same bar `probe_m62_gold_volume.py` (M62b) uses for "clean
   teacher gold".
4. Every clause argument's surface word, and the clause's predicate, is
   looked up in the USVS lemma index
   (`nsm_ct.ground.usvs.USVS.senses_of`, loaded via
   `nsm_ct.usvs_bridge.default_usvs()`) to attach a **chosen sense id** (see
   §3) and, per-token, the **full candidate sense-id list** (see §4).

A sentence that fails at step 2 or 3 (cap-hit / too-long / no-hypothesis /
other-exception / grounding-fail) produces no record — it is skipped and
counted in `dev/ENCODER_GOLD_V1_STATS.md`.

## 1. Top-level record shape

```
{
  "text": str,
  "tokens": [str, ...],
  "pos": [str, ...],
  "gold_tree": { "clauses": [...], "discourse_links": [...] },
  "token_sense_candidates": [...]
}
```

| field | type | nullable | notes |
|---|---|---|---|
| `text` | string | no | Raw sentence string as fed to the parser (whitespace-tokenizable; punctuation is space-separated, matching this codebase's corpus convention). |
| `tokens` | array[string] | no | Surface tokens, in order, from the parser's own tagger (`ParserInputEncoder._tag(text)[i].text`). Same tokenization the parser itself parsed. |
| `pos` | array[string] | no | Same length as `tokens`. Coarse POS tag name per token (`quantum_parser`'s `Tag` enum member name, e.g. `NOUN`, `VERB`, `ADJ`, `ADP`, `PRON`, `PROPN`, `AUX`, `DET`, `PUNCT`, `NUM`, `ADV`, `CCONJ`, `SCONJ`, `PART`). Closed enum, defined by `quantum_parser`'s tagger, not by this schema. |
| `gold_tree` | object | no | See §2. |
| `token_sense_candidates` | array[object] | no (may be `[]`) | See §4. Sparse over `tokens`: only indices the USVS lemma index covers appear. |

## 2. `gold_tree` shape

```
"gold_tree": {
  "clauses": [
    {
      "predicate": str,
      "predicate_sense_id": str | null,
      "is_question": bool,
      "roles": [
        { "relation": str, "word": str, "is_entity": bool, "sense_id": str | null },
        ...
      ]
    },
    ...
  ],
  "discourse_links": [
    { "coordinator": str, "prime": str | null, "clause_i": int, "clause_j": int },
    ...
  ]
}
```

### 2.1 `clauses[i]`

One entry per `nsm_ct.clause.Clause` returned by `extract_discourse` (in the
order that function returns them: primary clause(s) from the single-clause
graph shape, then any independent later-sentence/coordination-orphan
clauses — see `extract_discourse`'s docstring). A sentence can (and often
does) yield more than one clause.

| field | type | nullable | notes |
|---|---|---|---|
| `predicate` | string | no | The clause's predicate surface token (`Clause.predicate`); defaults to `"is"` for copula-less fact clauses the extractor synthesizes (see `_fact_clause`). |
| `predicate_sense_id` | string | **yes** | Chosen (MFS — most-frequent-sense) USVS sense id for `predicate`, via `usvs.senses_of(predicate.lower())[0]`. `null` when the predicate is an entity token (never — predicates are not entities in this grammar) or, in practice, whenever the USVS lemma index does not cover the surface form (inflected forms, closed-class copulas like "is"/"was", hyphenated/compound tokens, OOV). |
| `is_question` | bool | no | `Clause.is_question` — the predicate carries `quantum_parser`'s QUESTION subtype. |
| `roles` | array[object] | no (may be `[]`) | One entry per `(relation, arg_node)` in `Clause.args`, in that order. |

### 2.2 `clauses[i].roles[j]`

| field | type | nullable | notes |
|---|---|---|---|
| `relation` | string | no | The semantic role label. Closed core set from this grammar: `SUBJECT`, `OBJECT`, `INDIRECT_OBJECT`, `PLACE`, `SOURCE`, `AGENT`, `RECIPIENT`. Open fallback: an uppercased preposition token (e.g. `FOR`, `WITH`, `OF`, `ABOUT`) when the PP doesn't map to one of the closed roles — see `nsm_ct.clause._PREP_RELATION` / `_prep_relation`. Not a fixed enum; treat as open string vocabulary anchored by the closed core. |
| `word` | string | no | Surface token filling the role (`arg_node.token`). |
| `is_entity` | bool | no | `nsm_ct.clause.is_entity(word)` — true for names/pronouns (this grammar's referent variables; NSM "someone X", never decomposed into a sense). |
| `sense_id` | string | **yes** | Chosen (MFS) USVS sense id for `word`, via `usvs.senses_of(word.lower())[0]`. Always `null` when `is_entity` is true. Also `null` when the USVS lemma index doesn't cover `word` (function words, proper nouns not treated as entities, inflected/compound/OOV forms — same coverage gap as `predicate_sense_id`). |

### 2.3 `discourse_links[k]`

One entry per `nsm_ct.clause.DiscourseLink` — a coordinator relating two
clauses by index into the `clauses` array above.

| field | type | nullable | notes |
|---|---|---|---|
| `coordinator` | string | no | One of `OR`, `AND`, `BUT`, `NOT` (this pipeline's current coordinator vocabulary — see `_COORDINATOR_LABEL` / the hard-coded `"NOT"` negation link). |
| `prime` | string | **yes** | The NSM atom grounding the coordinator (`MAYBE` for `OR`, `NOT` for `NOT`), or `null` for `AND`/`BUT` (no dedicated atom — see `_COORD_PRIME`). |
| `clause_i` | int | no | Index into `clauses` (the "anchor" side of the link; `0` in every link this pipeline currently emits). |
| `clause_j` | int | no | Index into `clauses` (the related clause). |

Most single-fact sentences have `discourse_links: []` (no coordination /
negation detected).

## 3. "Chosen sense" = MFS, not context-conditioned WSD

Every `*_sense_id` field in `gold_tree` is the candidate at index 0 of
`usvs.senses_of(word)` — i.e. **most-frequent-sense (MFS)**, WordNet's own
lemma-frequency ordering (`nsm_ct.ground.usvs.USVS.__post_init__` sorts each
lemma's sense-id list by `nsm_ct.wordnet.senses(lemma)`'s order, which is
itself MFS-first). This is a fact about what the deterministic teacher
pipeline actually does today, not a design choice made for this dataset:
`nsm_ct.usvs_bridge.usvs_handle` (the function the rest of the pipeline uses
to turn a word into a vector) resolves a non-core word to
`u.senses_of(word)[0]` the exact same way. There is no context-conditioned
word-sense-disambiguation step wired into this path — `nsm_ct.wsd` and
`nsm_ct.sense_chooser` exist in this repo but are separate, explicitly
not-wired-into-the-main-pipeline modules (`wsd.py`'s own docstring: "not
wired yet, by design"; `sense_chooser.py` is a trained scorer exercised only
by its own probe/training scripts). So "chosen sense" here means "the sense
the teacher pipeline would actually hand the rest of the system today," not
"the correct sense disambiguated from context."

## 4. `token_sense_candidates[i]` — the retrieval-conditioning input

```
{ "index": int, "token": str, "sense_candidates": [str, ...], "chosen_sense": str }
```

One entry per token in `tokens` for which `usvs.senses_of(token.lower())` is
non-empty — this is how "content token" is operationalized here: a token is
"content" exactly when the USVS lemma index has at least one candidate sense
for it (in practice this naturally excludes punctuation, most closed-class
function words, and most proper nouns, without any hand-rolled stopword
list). The array is therefore **sparse** over `tokens`/`pos` — indices with
no entry are tokens the USVS lemma index does not cover.

| field | type | nullable | notes |
|---|---|---|---|
| `index` | int | no | Index into the record's `tokens`/`pos` arrays. |
| `token` | string | no | The surface token (`tokens[index]`, un-lowercased). |
| `sense_candidates` | array[string] | no (never empty) | The **full** MFS-ordered candidate sense-id list, `usvs.senses_of(token.lower())` — every USVS sense id whose lemma list contains this token, across every POS WordNet lists it under (a token can carry noun and verb senses in the same list, e.g. "found" -> `found.n.01`, `find.v.01`, ...). This is the actual candidate SET the grounding step has available, not a downsample of it. |
| `chosen_sense` | string | no | `sense_candidates[0]` — the same MFS choice described in §3, repeated here for convenience so a consumer doesn't have to cross-reference `gold_tree` to know what the teacher would have picked for this token. |

This field **did capture the full candidate set**, not just the chosen
sense — see `dev/ENCODER_GOLD_V1_STATS.md` for how many candidates per token
that was in practice. Where a future context-conditioned WSD/retrieval step
would plug in: it would consume `sense_candidates` (plus surrounding
`tokens`/`pos` and, per §2.2's UNIVERSAL_ENCODER_DESIGN.md, fired-grammar-
rule context this dataset does not serialize) and choose an index other than
0 when context disagrees with MFS — exactly the gap `nsm_ct.sense_chooser`
was built to close for a small ambiguity-episode benchmark, not yet
integrated with real-corpus parsing.

## 5. What is deliberately NOT in this schema

- **Coreference resolution.** `roles[j].word` is the raw surface pronoun/name
  as it appears in the sentence (e.g. "she", "it") — no antecedent linking.
  `nsm_ct.clause.EntityTracker` exists and does recency-based coreference,
  but is not exercised by `extract_discourse`/this generation script.
- **NSM-prime explication trees** (`nsm_ct.thought.build_thought`,
  `nsm_ct.meaning.NSMMeaningResolver`). That is a parallel "word -> tree of
  NSM primes" representation used by the tree-view path
  (`ParserInputEncoder._parse_tree` / `encode_structured`), independent of
  the flat-graph/`extract_discourse` path this dataset serializes. Out of
  scope for the grounded-tree contract above.
- **Fired grammar rules** (UNIVERSAL_ENCODER_DESIGN.md §2.2 item 3). Not
  captured by this generation run; the deterministic parser doesn't expose
  a clean per-sentence "which rules fired" trace without parser-source
  changes, which this task is explicitly scoped not to make.

## 6. Two fully-worked real examples

See `runs/encoder_gold_v1.jsonl` for the full dataset (not committed — see
`.gitignore`'s `runs/`; regenerate with `python scripts/build_encoder_gold_v1.py`).
Verbatim records below (pretty-printed; the file itself is one JSON object
per line), copied as-is from the generated data — nothing hand-edited.

### 6.1 Single clause, both content-word roles grounded — and MFS's honest failure mode

`text = "do cats eat bats ? '"`

```json
{
  "text": "do cats eat bats ? '",
  "tokens": ["do", "cats", "eat", "bats", "?", "'"],
  "pos": ["AUX", "NOUN", "VERB", "VERB", "PUNCT", "PUNCT"],
  "gold_tree": {
    "clauses": [
      {
        "predicate": "eat",
        "predicate_sense_id": "eat.v.01",
        "is_question": false,
        "roles": [
          {"relation": "SUBJECT", "word": "cats", "is_entity": false, "sense_id": null},
          {"relation": "OBJECT", "word": "bats", "is_entity": false, "sense_id": "balmy.s.01"}
        ]
      }
    ],
    "discourse_links": []
  },
  "token_sense_candidates": [
    {"index": 0, "token": "do", "sense_candidates": [
        "bash.n.02", "do.n.02", "doctor_of_osteopathy.n.01", "make.v.01", "perform.v.01",
        "do.v.03", "do.v.04", "cause.v.01", "practice.v.01", "suffice.v.01", "do.v.08",
        "act.v.02", "serve.v.09", "do.v.11", "dress.v.16", "do.v.13"],
     "chosen_sense": "bash.n.02"},
    {"index": 2, "token": "eat",
     "sense_candidates": ["eat.v.01", "eat.v.02", "feed.v.06", "eat.v.04", "consume.v.05", "corrode.v.01"],
     "chosen_sense": "eat.v.01"},
    {"index": 3, "token": "bats", "sense_candidates": ["balmy.s.01"], "chosen_sense": "balmy.s.01"}
  ]
}
```

Two honest artifacts of the MFS grounding strategy (§3), visible directly in
this real record:
- `"cats"` gets `sense_id: null` in `gold_tree` **and no entry at all** in
  `token_sense_candidates` — the USVS lemma index is keyed on the singular
  lemma `"cat"`, so the inflected surface form `"cats"` has zero coverage.
  Same reason `"bats"` (below) only resolves via its one adjective sense,
  never its noun ("flying mammal") sense — that lemma entry is simply
  `"bat"`, not `"bats"`, in the index.
- `"bats"` DOES resolve, to `balmy.s.01` — WordNet's most-frequent sense for
  the lemma `"bats"` is the informal adjective ("bats" = "crazy"), not the
  animal. This is MFS doing exactly what MFS does (§3): no context is
  consulted, so a real ambiguity gets the wrong (if "most frequent" by
  WordNet's own count) answer. This is precisely the gap a future
  context-conditioned WSD/retrieval step (§4) exists to close.

### 6.2 Multi-clause with a discourse link (OR / MAYBE)

`text = "they had n't any little girls or any little boys , at all ."`

```json
{
  "text": "they had n't any little girls or any little boys , at all .",
  "tokens": ["they", "had", "n't", "any", "little", "girls", "or", "any", "little", "boys", ",", "at", "all", "."],
  "pos": ["PRON", "AUX", "PART", "ADV", "ADJ", "NOUN", "CCONJ", "ADV", "ADJ", "NOUN", "PUNCT", "ADP", "ADJ", "PUNCT"],
  "gold_tree": {
    "clauses": [
      {
        "predicate": "had", "predicate_sense_id": null, "is_question": false,
        "roles": [
          {"relation": "SUBJECT", "word": "they", "is_entity": true, "sense_id": null},
          {"relation": "PLACE", "word": "girls", "is_entity": false, "sense_id": null}
        ]
      },
      {
        "predicate": "had", "predicate_sense_id": null, "is_question": false,
        "roles": [
          {"relation": "SUBJECT", "word": "they", "is_entity": true, "sense_id": null},
          {"relation": "PLACE", "word": "boys", "is_entity": false, "sense_id": null}
        ]
      }
    ],
    "discourse_links": [
      {"coordinator": "OR", "prime": "MAYBE", "clause_i": 0, "clause_j": 1}
    ]
  },
  "token_sense_candidates": [
    {"index": 3, "token": "any", "sense_candidates": ["any.s.01", "any.r.01"], "chosen_sense": "any.s.01"},
    {"index": 4, "token": "little", "sense_candidates": [
        "little.n.01", "small.a.01", "little.a.02", "little.s.01", "fiddling.s.01",
        "little.s.03", "short.a.03", "little.s.04", "little.s.05", "little.r.01"],
     "chosen_sense": "little.n.01"},
    {"index": 6, "token": "or", "sense_candidates": ["oregon.n.01", "operating_room.n.01"], "chosen_sense": "oregon.n.01"},
    {"index": 7, "token": "any", "sense_candidates": ["any.s.01", "any.r.01"], "chosen_sense": "any.s.01"},
    {"index": 8, "token": "little", "sense_candidates": [
        "little.n.01", "small.a.01", "little.a.02", "little.s.01", "fiddling.s.01",
        "little.s.03", "short.a.03", "little.s.04", "little.s.05", "little.r.01"],
     "chosen_sense": "little.n.01"},
    {"index": 11, "token": "at", "sense_candidates": ["astatine.n.01", "at.n.02"], "chosen_sense": "astatine.n.01"},
    {"index": 12, "token": "all", "sense_candidates": ["all.a.01", "all.s.01", "wholly.r.01"], "chosen_sense": "all.a.01"}
  ]
}
```

Notes visible directly in this real record: the coordinated-value "A or B"
shape (§2.1's ordering) produces one lossless clause per disjunct
(`"girls"`, `"boys"`), both sharing the same predicate/subject, related by
one `discourse_links` entry carrying the `MAYBE` prime (NSM has no OR
exponent — disjunction *is* MAYBE, per `nsm_ct.clause`'s module docstring).
`"they"` is `is_entity: true` (a pronoun/referent variable, never sense-
grounded by design — §2.2). `"girls"`/`"boys"` get `sense_id: null` and no
`token_sense_candidates` entry, the same plural-lemma coverage gap as
`"cats"` in §6.1 — evidently the dominant reason a content word ends up
ungrounded in this dataset (see `dev/ENCODER_GOLD_V1_STATS.md` for how often
that happens in aggregate). Function words that happen to collide with a
WordNet abbreviation lemma get spurious candidates too (`"at"` ->
`astatine.n.01`, the element symbol; `"or"` -> `oregon.n.01`, the state
abbreviation) — real content the schema captures faithfully, exactly as the
teacher pipeline actually grounds them today, however wrong that dominant
sense is.
