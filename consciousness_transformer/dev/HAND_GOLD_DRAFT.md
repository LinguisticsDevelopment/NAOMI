# HAND-AUTHORED GOLD — DRAFT hard cases for lead review

> **STATUS: DRAFT. NOT MERGED GOLD. NOT TRAINED ON.**
> 16 candidate records for the lead to red-pen, on branch `hand-gold-draft`.
> Records: `runs/hand_gold_draft.jsonl`. Helper: `scripts/hand_gold.py`.
> Builder: `scripts/build_hand_gold_draft.py`. Renderer (this doc's example
> section): `scripts/render_hand_gold_draft.py`.

Written against the **two-part gold decision** (lead, 2026-09-07,
`RESEARCH_NOTES.md`): the 16K corpus expansion only scales what the
*deterministic teacher* can parse — declaratives. The structures the *learned*
encoder exists for appear in **zero** teacher gold and must be hand-fabricated.
The gate on that plan was: **is the schema human-writable?**

Schema authority: `dev/ENCODER_IO_CONTRACT_V2.md` (canonical v2). Honored
throughout: the **CORE BOUNDARY** decision (the encoder emits a candidate
lattice and commits to nothing), the **Appraisal grounding** decision (no
emotion node, no reaction-sense inventory — connotation is comprehension-side),
and the generalized `context_ref` (§4: sense / reference / elision are **one**
`grounding` construct).

---

## 1. Schema-writability verdict — **YES**, with one helper

A human can author a v2 gold record. Nothing in the schema is machine-only:
there are no opaque ids, no learned embeddings, no per-token precomputed
tables a person would have to invent. The only fields a human *cannot*
reasonably produce by hand are **mechanically derivable** — the sense-candidate
lists (a lexicon lookup), the token indices (a left-to-right walk), and the
`ctx:…` antecedent handles (an addressing convention). `scripts/hand_gold.py`
fills exactly those.

### The division of labour

| the HUMAN supplies | the HELPER fills |
|---|---|
| the surface string | `tokens`, `pos` — from the **real** parser tagger (`pos_tagger.tag_sentence` / `tag_spanish_sentence`), so hand gold's surface fields have identical provenance to teacher-bulk gold |
| the clause split and each clause's `utterance_kind` (`proposition` / `imperative` / `interjection`) and `is_question` | — |
| the predicate, and each role's **relation label** (`SUBJECT`, `OBJECT`, `PLACE`, `ABOUT`, …) | — |
| for each filler, **one of four things**: a surface word `W("cup")`; a synthesized addressee `PRIME("YOU")`; a context slot `CTX("elision", of="flour")`; a run-time memory slot `MEM("reference")` | `grounding.type` routing (pronoun → `reference`; name → `entity`; covered lemma → `sense`; uncovered → `entity`), which mirrors the teacher's own `ground_word` exactly |
| the antecedent **by content** — the *word* it refers to, or the sentinel `PREDICATE` / `CLAUSE` | `candidates` (the full retrieved antecedent set) and `ref` (`context_index`, `tree_index`, `clause`, `slot`, `role_index`), plus the readable `ctx:ci/ti/slot/j` handles |
| the prior-context sentences, in the same clause spec | each `context[]` entry's full lattice, tokens, pos and sense table |
| — | `token_sense_candidates` (whole table, `senses_of` per surface token, `chosen_sense = [0]`), `candidates` per sense node, `discourse_links_per_tree` scaffolding |

A record is 4–10 lines of spec. `come here !` in full:

```python
hand_gold_record(
    "come here !",
    [C(kind="imperative", predicate=W("come"),
       roles=[("SUBJECT", PRIME("YOU")), ("PLACE", W("here"))])],
    usvs=usvs)
```

### The gate every record passes

`hand_gold.check_record` runs four checks; all 16 drafts pass all four:

1. **Schema validity** (contract §1–§4): field presence/types, `pos`/`tokens`
   parity, `trees`/`discourse_links_per_tree` parity, `chosen_sense ==
   sense_candidates[0]`, resolved groundings carry `candidates: null`, sense
   nodes agree with `token_sense_candidates` (§4.2), `token_index` actually
   points at the role's word, and **every `ctx:` candidate and every `ref`
   dereferences** into a real node of `context[]`.
2. **Oracle linearization**: `encoder_model.linearize_tree` runs on the record.
3. **Action legality — the sacred invariant**: every oracle action is admitted
   by `encoder_model.legal_action_types` at the state it is emitted in. A
   masked-out gold action is a `-inf` logit → `+inf` training loss.
4. **Round-trip**: the action sequence is *de-linearized* back to a clause
   skeleton (`utterance_kind` + ordered `(role, token_index, gtype, source,
   prime)`) and compared with the authored tree. Equal for all 16 — the
   derivation serializes to exactly the tree that was written.

Plus an end-to-end check through the **actual training path**:
`hand_gold.check_trains` builds features and runs the real
`teacher_force_loss` on a freshly-initialized policy for all 16 records.
Finite (mean ≈ 23.3 nats/record on an untrained net), i.e. the legality mask
never excludes a gold action in practice, not just in replay.

The gate has teeth — five deliberate corruptions were all caught:

| injected fault | caught as |
|---|---|
| role `token_index` moved off its word | `token_index does not point at word` |
| a `prime` grounding given a candidate list | `resolved grounding must have candidates=null` |
| `ref.role_index` pointed at a non-existent role | `ref 'ctx:0/0/role/99' does not dereference` |
| `token_sense_candidates` emptied under a sense node | `sense slot at token 0 has no token_sense_candidates entry` |
| `utterance_kind: "appraisal"` (the stripped v1 value) | `bad utterance_kind` |

### The one caveat on "human-writable"

Writability is **not** the same as *fully specified*. Authoring these 16
surfaced nine places where the contract (or the model's feature path) does
not determine the answer, listed in §4. None blocks authoring; each needs a ruling before this becomes gold.

---

## 2. Existing hand-authored gold — what is already covered

Two branches, both **v1-schema and now superseded**:

| branch | file | n | families |
|---|---|---|---|
| `encoder-handgold-v1` (M63.1c, 5317d13) | `dev/hand_authored_gold_v1.jsonl` | 33 | imperative 10, "appraisal" 16, elliptical 7 |
| `encoder-handgold-v2` (independent 2nd draft) | `dev/hand_authored_gold_v1.jsonl` | 33 | proposition 13, "appraisal" 15, elliptical 8 |

Lexical coverage across both (so this batch does not duplicate it):

- **imperatives**: Look, Sit down, Close the door, Stop, Come here, Give me the
  ball, Wait, Listen, Open your books, Run, Be quiet, Do n't move.
- **interjections** (authored as `clause_kind/clause_type: "appraisal"`):
  Oh dear, Shit, Alas, Hurray, Ugh, Wow, Ouch, Oh, Ah, Yuck, Oops, Phew,
  Oh no, Damn, Yay — including three attached to a prior proposition
  ("He failed the test . Damn !").
- **elision / fragments**: More, Again, John, In London, Under the table,
  Me too, More soup, The blue one, And found a coin.
- **code-switch / pro-drop**: Corre, Comió, Comí.

**Neither file is loadable as v2 gold.** They use `gold_tree` (one committed
tree) not `lattice.trees`, `sense_id` (a committed MFS pick, written with a
trailing `?`, e.g. `"drink.v.01?"`) not a candidate set, `prior_context` /
a single `context` object not `context[]`, `synthesized` / `synth_kind` /
`filler_kind` not `grounding.type: "prime"`, a flat `context_ref` not the
nested `ref`, and — the part the lead explicitly **STRIPPED** — an
`"appraisal"` clause kind with `FEEL` explications and a reaction-sense
inventory. `encoder_model.linearize_tree` cannot read any of them.

**So this batch extends in two directions.**

- **10 records are new structures** nothing above covers: three-role
  imperatives (`put the cup on the table .`), a resolved synth slot and an
  unresolved slot in one clause (`tell me about the dog .`, `stir well .`),
  a PP-only imperative (`wait for me .`), an ordinary content noun used as a
  stance exclamation (`nonsense !`), VP-ellipsis with a stranded auxiliary
  (`the dog did .`), multi-entry `context[]` where the antecedent is *not* the
  most recent sentence (`and a hat too .`), English narrative subject-drop
  (`opened the box .`), Spanish pro-drop with a real object
  (`bebió el agua .`), and propositional anaphora (`sounds good .`).
- **6 are canonical items deliberately re-authored** into v2 as the reference
  exemplar for their family: `come here !`, `shit !`, `alas !`, `ugh .`,
  `more !`, `me too .`. Their old records cannot be migrated mechanically —
  the appraisal apparatus they carry no longer exists — and each now also
  carries a finding (see §4: `alas` *does* have a synset; `ugh` does not;
  `more !` is contract §8.3 run through the real retrieval pipeline).

---

## 3. The 16 draft candidates

Each: surface text, the tree in readable form, why it is hard, the design
choice, and anything I was unsure about. Rendered from the actual JSONL by
`scripts/render_hand_gold_draft.py`, so nothing here can drift from the records.

<!-- BEGIN GENERATED EXAMPLES -->

### A. IMPERATIVES — a SUBJECT the string does not contain

#### `come here !`

```
text   : come here !
tokens : ['come', 'here', '!']
pos    : ['VERB', 'ADV', 'PUNCT']
tree:
    clause  utterance_kind=imperative  is_question=False
      PREDICATE  'come'
          sense <- lexicon/lemma_senses  (22 candidates: semen.n.01, come.v.01, arrive.v.01, …)
      SUBJECT          'you' (NO surface token)
          prime:YOU  [RESOLVED]
      PLACE            'here'@1
          sense <- lexicon/lemma_senses  (7 candidates: here.n.01, hera.n.01, here.s.01, …)
```

**Why it is a hard case.** The addressee subject has NO surface token; the teacher's parse has no node to hang it on, so this clause shape appears in zero teacher gold.

**Design choice.** Synthesized SUBJECT as grounding.type:'prime', prime YOU, word='you', token_index=null (contract S5) -- RESOLVED, not a candidate slot: the grammar licenses exactly one filler. utterance_kind:'imperative'. Oracle emits EMIT_SYNTH_SLOT.

**UNSURE — needs the lead:**
- Re-authoring of a v1 hand-gold item ('Come here !', encoder-handgold-v1) into the v2 lattice schema -- kept deliberately as the reference exemplar for the family.

#### `put the cup on the table .`

```
text   : put the cup on the table .
tokens : ['put', 'the', 'cup', 'on', 'the', 'table', '.']
pos    : ['VERB', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'PUNCT']
tree:
    clause  utterance_kind=imperative  is_question=False
      PREDICATE  'put'
          sense <- lexicon/lemma_senses  (10 candidates: put_option.n.02, put.v.01, put.v.02, …)
      SUBJECT          'you' (NO surface token)
          prime:YOU  [RESOLVED]
      OBJECT           'cup'@2
          sense <- lexicon/lemma_senses  (11 candidates: cup.n.01, cup.n.02, cup.n.03, …)
      PLACE            'table'@5
          sense <- lexicon/lemma_senses  (8 candidates: table.n.01, table.n.02, table.n.03, …)
```

**Why it is a hard case.** Three-role imperative (synth SUBJECT + overt OBJECT + PP PLACE). Existing hand-gold tops out at two roles, so nothing yet shows the synth slot coexisting with a full argument frame.

**Design choice.** 'on' -> PLACE via clause.py's _PREP_RELATION (locative), NOT an ON PP-role. The synth SUBJECT sorts LAST in the oracle's node order (nulls last), so the derivation is GROUND(put) GROUND(cup) GROUND(table) EMIT_SYNTH_SLOT(SUBJECT).

#### `tell me about the dog .`

```
text   : tell me about the dog .
tokens : ['tell', 'me', 'about', 'the', 'dog', '.']
pos    : ['VERB', 'PRON', 'ADP', 'DET', 'NOUN', 'PUNCT']
tree:
    clause  utterance_kind=imperative  is_question=False
      PREDICATE  'tell'
          sense <- lexicon/lemma_senses  (9 candidates: tell.n.01, state.v.01, tell.v.02, …)
      SUBJECT          'you' (NO surface token)
          prime:YOU  [RESOLVED]
      INDIRECT_OBJECT  'me'@1
          prime:I  [RESOLVED]
      ABOUT            'dog'@4
          sense <- lexicon/lemma_senses  (8 candidates: dog.n.01, frump.n.01, dog.n.03, …)
```

**Why it is a hard case.** A synthesized slot (SUBJECT=YOU, resolved) and an UNRESOLVED slot (the pronoun 'me', reference/memory) in the same clause -- the encoder must emit one committed node and one candidate slot side by side.

**Design choice.** 'me' is ALWAYS the speaker, so it grounds as the resolved prime I (D3, dev/CURRENT_STATE.md decisions locked) -- symmetric with the imperative's synthesized addressee prime YOU -- rather than an unresolved memory reference. Unlike YOU, 'me' HAS a surface token, so it keeps its real token_index (1); the oracle's EMIT_SYNTH_SLOT now shifts to and consumes it instead of hardcoding token_index=null. 'about' is an open PP-role ABOUT (attested 41x in teacher gold).

#### `wait for me .`

```
text   : wait for me .
tokens : ['wait', 'for', 'me', '.']
pos    : ['VERB', 'ADP', 'PRON', 'PUNCT']
tree:
    clause  utterance_kind=imperative  is_question=False
      PREDICATE  'wait'
          sense <- lexicon/lemma_senses  (6 candidates: delay.n.01, wait.n.02, wait.v.01, …)
      SUBJECT          'you' (NO surface token)
          prime:YOU  [RESOLVED]
      FOR              'me'@2
          prime:I  [RESOLVED]
```

**Why it is a hard case.** Intransitive imperative whose only argument is a PP -- the 'Wait !' already in hand-gold has no arguments at all.

**Design choice.** 'for' -> the open PP-role FOR; the preposition token itself is not grounded (it is flushed by the oracle's terminal SHIFT), matching teacher-gold treatment of function words. 'me' -- the speaker being waited for -- grounds as the resolved prime I (D3), same as 'tell me ...' above.

### B. INTERJECTIONS — literal/gloss sense + utterance_kind, NO appraisal node

#### `shit !`

```
text   : shit !
tokens : ['shit', '!']
pos    : ['NOUN', 'PUNCT']
tree:
    clause  utterance_kind=interjection  is_question=False
      PREDICATE  'shit'
          sense <- lexicon/lemma_senses  (8 candidates: crap.n.01, bullshit.n.01, jack.n.01, …)
```

**Why it is a hard case.** A whole utterance that is one grounded token and no argument structure. The teacher's clause extractor needs a predicate node with edges; a bare exclamation yields none.

**Design choice.** Grounds to its LITERAL USVS candidate set (senses_of('shit') = 8 senses, crap.n.01 first) as an ordinary sense slot. utterance_kind:'interjection' is the ENTIRE extra signal. NO appraisal node, NO valence, no stance_target -- comprehension derives the feeling.

#### `alas !`

```
text   : alas !
tokens : ['alas', '!']
pos    : ['NOUN', 'PUNCT']
tree:
    clause  utterance_kind=interjection  is_question=False
      PREDICATE  'alas'
          sense <- lexicon/lemma_senses  (1 candidates: unfortunately.r.01)
```

**Why it is a hard case.** Listed in RESEARCH_NOTES M63.1c flag #1 as a PURE interjection with no synset -- but USVS actually returns one (unfortunately.r.01). A one-candidate sense slot: no ambiguity for comprehension to resolve, yet still an unresolved-slot shape.

**Design choice.** Literal grounding, same as 'shit !'. This record exists partly to CORRECT the M63.1c flag: 'alas' is lexically covered.

#### `ugh .`

```
text   : ugh .
tokens : ['ugh', '.']
pos    : ['PROPN', 'PUNCT']
tree:
    clause  utterance_kind=interjection  is_question=False
      PREDICATE  'ugh'
          sense <- lexicon/lemma_senses  (1 candidates: interj.ugh.01)
```

**Why it is a hard case.** A GENUINELY pure interjection: senses_of('ugh') == [] in WordNet proper. Contract S4.2 says an uncovered content token grounds as type 'entity' -- so without D2, the utterance would carry no meaning at all beyond its kind.

**Design choice.** D2 (dev/CURRENT_STATE.md decisions locked): 'ugh' is now GLOSS-grounded -- `nsm_ct.ground.usvs.PURE_INTERJECTION_GLOSSES` mints `interj.ugh.01` ('an exclamation expressing disgust or horror') and grounds it through the SAME gloss->coordinate pipeline every real WordNet sense uses, additively and fingerprint-safe. `senses_of('ugh')` now returns ['interj.ugh.01'], so `W('ugh')` grounds as an ordinary sense slot -- no bare 'entity', no record-level change needed here at all. utterance_kind 'interjection' is still the only speech-act marker; still NO appraisal node.

#### `nonsense !`

```
text   : nonsense !
tokens : ['nonsense', '!']
pos    : ['NOUN', 'PUNCT']
tree:
    clause  utterance_kind=interjection  is_question=False
      PREDICATE  'nonsense'
          sense <- lexicon/lemma_senses  (3 candidates: nonsense.n.01, folderal.n.01, nonsense.s.01)
```

**Why it is a hard case.** NEW family member (not in either existing hand-gold draft): an ordinary content noun used as a stance exclamation. This is the case the lead's 'connotation generalizes beyond 15 interjection words' argument needs -- same machinery, non-interjection lexeme.

**Design choice.** Identical treatment to 'shit !': literal senses (nonsense.n.01, folderal.n.01, nonsense.s.01) + utterance_kind 'interjection'. Nothing about the ENCODER record distinguishes an epithet from a swear -- correct by design; the difference is comprehension-side valence.

### C. ELISION / FRAGMENTS — a context_ref slot, antecedent named by CONTENT

#### `more !`

```
text   : more !
tokens : ['more', '!']
pos    : ['ADV', 'PUNCT']
context[0]: the children eat bread .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'eat'
            sense <- lexicon/lemma_senses  (6 candidates: eat.v.01, eat.v.02, feed.v.06, …)
        SUBJECT          'children'@1
            entity  [RESOLVED, no sense]
        OBJECT           'bread'@3
            sense <- lexicon/lemma_senses  (3 candidates: bread.n.01, boodle.n.01, bread.v.01)
tree:
    clause  utterance_kind=proposition  is_question=False
      PREDICATE  (none — elided)
          elision <- context/elision_inherit_predicate
          candidates: ['ctx:0/0/predicate']
          gold ref  : context[0] clause 0 PREDICATE
      OBJECT           (NO surface token)
          elision <- context/elision_inherit_arg
          candidates: ['ctx:0/0/role/0', 'ctx:0/0/role/1']
          gold ref  : context[0] clause 0 role 1
      QUANTITY         'more'@0
          sense <- lexicon/lemma_senses  (5 candidates: more.n.01, more.a.01, more.a.02, …)
```

**Why it is a hard case.** Both the predicate AND an argument are absent from the string. The teacher cannot posit a node for a word that is not there.

**Design choice.** Two elision slots, one shape (contract S4.1): predicate inherits the context predicate, OBJECT inherits the context OBJECT; 'more' is an ordinary surface sense slot under role QUANTITY. predicate=null because predicate_grounding.type is 'elision' (contract S3). This is contract S8.3's own worked example, re-authored through the REAL retrieval pipeline. QUANTITY is now an official structural role label (D4, dev/CURRENT_STATE.md decisions locked) -- a relation label only, never touching USVS sense coordinates -- alongside the new ADDITIVE/FOCUS and the existing DESCRIPTION/SPECIFICATION/COMPLEMENT modifier roles (`nsm_ct.clause.MODIFIER_RELATIONS`).

#### `me too .`

```
text   : me too .
tokens : ['me', 'too', '.']
pos    : ['PRON', 'ADV', 'PUNCT']
context[0]: the boys want a dog .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'want'
            sense <- lexicon/lemma_senses  (9 candidates: privation.n.01, lack.n.01, need.n.02, …)
        SUBJECT          'boys'@1
            entity  [RESOLVED, no sense]
        OBJECT           'dog'@4
            sense <- lexicon/lemma_senses  (8 candidates: dog.n.01, frump.n.01, dog.n.03, …)
tree:
    clause  utterance_kind=proposition  is_question=False
      PREDICATE  (none — elided)
          elision <- context/elision_inherit_predicate
          candidates: ['ctx:0/0/predicate']
          gold ref  : context[0] clause 0 PREDICATE
      SUBJECT          'me'@0
          prime:I  [RESOLVED]
      OBJECT           (NO surface token)
          elision <- context/elision_inherit_arg
          candidates: ['ctx:0/0/role/0', 'ctx:0/0/role/1']
          gold ref  : context[0] clause 0 role 1
      ADDITIVE         'too'@1
          sense <- lexicon/lemma_senses  (2 candidates: excessively.r.01, besides.r.02)
```

**Why it is a hard case.** The ONLY surface content is the subject; predicate and object are both inherited. A fragment whose overt token is an argument, not a head.

**Design choice.** SUBJECT 'me' = the speaker, now the resolved prime I (D3) rather than a reference/memory slot -- symmetric with the imperative addressee prime YOU, and it keeps its real token_index (0). Predicate + OBJECT are context elisions. The additive particle 'too' now takes the new ADDITIVE role (D4) instead of being dropped -- an ordinary structural relation label that never touches USVS grounding.

#### `the dog did .`

```
text   : the dog did .
tokens : ['the', 'dog', 'did', '.']
pos    : ['DET', 'NOUN', 'AUX', 'PUNCT']
context[0]: the boys break the window .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'break'
            sense <- lexicon/lemma_senses  (75 candidates: interruption.n.02, break.n.02, fault.n.04, …)
        SUBJECT          'boys'@1
            entity  [RESOLVED, no sense]
        OBJECT           'window'@4
            sense <- lexicon/lemma_senses  (8 candidates: window.n.01, window.n.02, window.n.03, …)
tree:
    clause  utterance_kind=proposition  is_question=False
      PREDICATE  (none — elided)
          elision <- context/elision_inherit_predicate
          candidates: ['ctx:0/0/predicate']
          gold ref  : context[0] clause 0 PREDICATE
      SUBJECT          'dog'@1
          sense <- lexicon/lemma_senses  (8 candidates: dog.n.01, frump.n.01, dog.n.03, …)
      OBJECT           (NO surface token)
          elision <- context/elision_inherit_arg
          candidates: ['ctx:0/0/role/0', 'ctx:0/0/role/1']
          gold ref  : context[0] clause 0 role 1
```

**Why it is a hard case.** VP-ellipsis with a STRANDED AUXILIARY -- a family neither existing hand-gold draft covers. 'did' is a surface token that stands in for the elided predicate without being it.

**Design choice.** D6 (dev/CURRENT_STATE.md decisions locked): the elided predicate slot now KEEPS 'did's token_index (2) via the new `predicate_token_index` field -- CTX(..., word='did') -- so tense/polarity survive on the stranded auxiliary even though the predicate's MEANING is still inherited (predicate=null, predicate_grounding.type='elision', ref -> context[0] clause 0 PREDICATE 'break'). `encoder_model.clause_node_order` now reads this field for any non-sense/non-entity predicate, and `linearize_tree`'s EMIT_UNRESOLVED_SLOT shifts to and consumes it exactly like any other node with a real surface token. The overt SUBJECT is a normal sense slot, the OBJECT is a context elision.

#### `and a hat too .`

```
text   : and a hat too .
tokens : ['and', 'a', 'hat', 'too', '.']
pos    : ['CCONJ', 'DET', 'NOUN', 'ADV', 'PUNCT']
context[0]: the girls want a doll .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'want'
            sense <- lexicon/lemma_senses  (9 candidates: privation.n.01, lack.n.01, need.n.02, …)
        SUBJECT          'girls'@1
            entity  [RESOLVED, no sense]
        OBJECT           'doll'@4
            sense <- lexicon/lemma_senses  (2 candidates: doll.n.01, dame.n.01)
context[1]: they ask the father .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'ask'
            sense <- lexicon/lemma_senses  (7 candidates: ask.v.01, ask.v.02, ask.v.03, …)
        SUBJECT          'they'@0
            reference <- memory/coref  (candidates retrieved at run time)
        OBJECT           'father'@3
            sense <- lexicon/lemma_senses  (9 candidates: father.n.01, forefather.n.01, father.n.03, …)
tree:
    clause  utterance_kind=proposition  is_question=False
      PREDICATE  (none — elided)
          elision <- context/elision_inherit_predicate
          candidates: ['ctx:0/0/predicate', 'ctx:1/0/predicate']
          gold ref  : context[0] clause 0 PREDICATE
      SUBJECT          (NO surface token)
          reference <- context/coref
          candidates: ['ctx:0/0/role/0', 'ctx:0/0/role/1', 'ctx:1/0/role/0', 'ctx:1/0/role/1']
          gold ref  : context[0] clause 0 role 0
      OBJECT           'hat'@2
          sense <- lexicon/lemma_senses  (4 candidates: hat.n.01, hat.n.02, hat.v.01, …)
      ADDITIVE         'too'@3
          sense <- lexicon/lemma_senses  (2 candidates: excessively.r.01, besides.r.02)
```

**Why it is a hard case.** MULTI-ENTRY context[] where the antecedent is NOT the most recent sentence: 'and a hat too' inherits 'want' from context[0], not 'ask' from context[1]. Nothing in the repo exercises the context[] ARRAY that canonical v2 restored (v2-addendum had a single context object).

**Design choice.** candidates span BOTH entries (2 predicate handles, 4 role handles) -- the encoder emits the whole retrieved set and commits to nothing; ref names the gold antecedent in context[0]. This is the record that makes context_index load-bearing. 'too' now takes the ADDITIVE role (D4) instead of being dropped, same as 'me too .' above.

### D. SYNTHESIZED / DROPPED ARGUMENTS — an absent argument that is not the addressee

#### `opened the box .`

```
text   : opened the box .
tokens : ['opened', 'the', 'box', '.']
pos    : ['ADJ', 'DET', 'NOUN', 'PUNCT']
context[0]: the boys run into the room .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'run'
            sense <- lexicon/lemma_senses  (57 candidates: run.n.01, test.n.05, footrace.n.01, …)
        SUBJECT          'boys'@1
            entity  [RESOLVED, no sense]
        PLACE            'room'@5
            sense <- lexicon/lemma_senses  (5 candidates: room.n.01, room.n.02, room.n.03, …)
tree:
    clause  utterance_kind=proposition  is_question=False
      PREDICATE  'opened'
          sense <- lexicon/lemma_senses  (3 candidates: open.a.05, opened.s.01, open.s.09)
      SUBJECT          (NO surface token)
          reference <- context/coref
          candidates: ['ctx:0/0/role/0', 'ctx:0/0/role/1']
          gold ref  : context[0] clause 0 role 0
      OBJECT           'box'@2
          sense <- lexicon/lemma_senses  (13 candidates: box.n.01, box.n.02, box.n.03, …)
```

**Why it is a hard case.** English narrative subject-drop: a finite clause with a real predicate and object but NO subject token. The teacher emits a subjectless clause or fails; it never posits the empty slot.

**Design choice.** The dropped SUBJECT is an UNRESOLVED slot (type reference, source context, word=null, token_index=null) -- NOT a prime. Primes are for grammar-licensed single fillers (the imperative addressee); a dropped subject has a candidate SET and is comprehension's to bind. Oracle emits EMIT_UNRESOLVED_SLOT with token_index=None.

**UNSURE — needs the lead:**
- The predicate 'opened' retrieves THREE ADJECTIVAL senses (open.a.05, opened.s.01, open.s.09) and no verb sense, so under contract S7 this record's predicate is unrecallable by construction. Symptom of decision D5, not of this record.

#### `bebió el agua .`

```
text   : bebió el agua .
tokens : ['bebió', 'el', 'agua', '.']
pos    : ['NOUN', 'DET', 'NOUN', 'PUNCT']
context[0]: el niño está en la casa .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'está'
            entity  [RESOLVED, no sense]
        SUBJECT          'niño'@1
            sense <- lexicon/lemma_senses  (8 candidates: baby.n.01, chap.n.01, child.n.01, …)
        PLACE            'casa'@5
            sense <- lexicon/lemma_senses  (14 candidates: building.n.01, diggings.n.02, dwelling.n.01, …)
tree:
    clause  utterance_kind=proposition  is_question=False
      PREDICATE  'bebió'
          entity  [RESOLVED, no sense]
      SUBJECT          (NO surface token)
          reference <- context/coref
          candidates: ['ctx:0/0/role/0', 'ctx:0/0/role/1']
          gold ref  : context[0] clause 0 role 0
      OBJECT           'agua'@2
          sense <- lexicon/lemma_senses  (7 candidates: agua.n.01, body_of_water.n.01, rain.n.02, …)
```

**Why it is a hard case.** Spanish pro-drop: the subject is carried by verb morphology and is structurally absent. The code-switch family named in the two-part-gold decision.

**Design choice.** Identical construct to the English subject-drop above -- a reference slot with a context antecedent. FINDING: USVS already indexes OMW Spanish lemmas (senses_of('niño') = 8, 'agua' = 7, 'casa' = 14), so cross-lingual grounding needs NO new machinery; the only gap is inflection ('bebió', 'está' -> []), which is the SAME gap English past tense has. See decision D5.

#### `stir well .`

```
text   : stir well .
tokens : ['stir', 'well', '.']
pos    : ['VERB', 'ADV', 'PUNCT']
context[0]: put the flour in the bowl .
      clause  utterance_kind=imperative  is_question=False
        PREDICATE  'put'
            sense <- lexicon/lemma_senses  (10 candidates: put_option.n.02, put.v.01, put.v.02, …)
        SUBJECT          'you' (NO surface token)
            prime:YOU  [RESOLVED]
        OBJECT           'flour'@2
            sense <- lexicon/lemma_senses  (3 candidates: flour.n.01, flour.v.01, flour.v.02)
        PLACE            'bowl'@5
            sense <- lexicon/lemma_senses  (12 candidates: bowl.n.01, bowl.n.02, bowl.n.03, …)
tree:
    clause  utterance_kind=imperative  is_question=False
      PREDICATE  'stir'
          sense <- lexicon/lemma_senses  (11 candidates: stir.n.01, stir.n.02, bustle.n.01, …)
      SUBJECT          'you' (NO surface token)
          prime:YOU  [RESOLVED]
      OBJECT           (NO surface token)
          elision <- context/elision_inherit_arg
          candidates: ['ctx:0/0/role/0', 'ctx:0/0/role/1', 'ctx:0/0/role/2']
          gold ref  : context[0] clause 0 role 1
```

**Why it is a hard case.** TWO absent arguments of DIFFERENT kinds in one clause: a RESOLVED synthesized subject (prime YOU) and an UNRESOLVED dropped object (elision, several candidates). Recipe/instruction register. Nothing existing combines the two.

**Design choice.** The single most important record in the batch for the core boundary: the encoder emits a committed node AND a candidate set in the same clause, and never chooses between 'flour' and 'bowl' -- the candidate list carries both.

**UNSURE — needs the lead:**
- The context clause's own synthesized SUBJECT (prime YOU, token_index=null) lands in the elision slot's candidate pool. Should resolved prime nodes be filtered out of antecedent candidate sets? See decision D7.
- 'well' (manner adverb) is dropped -- same gap as 'too' above.

#### `sounds good .`

```
text   : sounds good .
tokens : ['sounds', 'good', '.']
pos    : ['NOUN', 'ADJ', 'PUNCT']
context[0]: we go to the park .
      clause  utterance_kind=proposition  is_question=False
        PREDICATE  'go'
            sense <- lexicon/lemma_senses  (35 candidates: go.n.01, adam.n.03, crack.n.09, …)
        SUBJECT          'we'@0
            reference <- memory/coref  (candidates retrieved at run time)
        PLACE            'park'@4
            sense <- lexicon/lemma_senses  (8 candidates: park.n.01, park.n.02, ballpark.n.01, …)
tree:
    clause  utterance_kind=proposition  is_question=False
      PREDICATE  'sounds'
          entity  [RESOLVED, no sense]
      SUBJECT          (NO surface token)
          reference <- context/propositional_anaphora
          candidates: ['ctx:0/0/clause/0']
          gold ref  : context[0] clause 0 (WHOLE CLAUSE)
      COMPLEMENT       'good'@1
          sense <- lexicon/lemma_senses  (27 candidates: good.n.01, good.n.02, good.n.03, …)
```

**Why it is a hard case.** PROPOSITIONAL anaphora: the dropped subject's antecedent is not an entity but the WHOLE prior clause ('that we go to the park'). The ref schema can address a clause (slot:null) but nothing in the repo has ever used it.

**Design choice.** ref = {context_index 0, tree_index 0, clause 0, slot null} -- 'the clause itself', with candidates the clause handle. 'good' takes COMPLEMENT (clause.py MODIFIER_RELATIONS).

**UNSURE — needs the lead:**
- ref.slot:null is OVERLOADED: contract S8.2 uses null role_index to mean 'genuinely ambiguous, not committed', while here null slot means 'the antecedent IS the clause'. These need distinguishing. See decision D1 (HAND_GOLD_DRAFT numbering -- distinct from the locked CURRENT_STATE.md D1, forest top-1).
- COMPLEMENT (and DESCRIPTION/SPECIFICATION/SUBJECT_COMPLEMENT) are usable via clause.py's MODIFIER_RELATIONS but still not listed in the contract's own S3 role-vocabulary table -- a documentation gap D4 (QUANTITY/ADDITIVE/FOCUS) did not itself widen; worth folding in together.

<!-- END GENERATED EXAMPLES -->

---

## 4. Decisions I need the lead to confirm

Ordered by how much downstream work rides on them. Each names the concrete
records affected.

### D1 — How does a `context_ref` name its antecedent? (`ref` is overloaded)

The contract gives one `ref` object and no field for "the gold answer". Three
different meanings are currently expressed through the same fields:

- §8.3 (`More !`): `candidates` has **one** entry, equal to `ref` — a
  deterministic inheritance, ref *is* the answer.
- §8.2 (`She waved .`): `candidates` has **two** entries and `ref.role_index`
  is **null** — genuinely ambiguous, ref deliberately under-specified.
- `sounds good .` (this batch): `ref.slot` is **null** meaning *the antecedent
  is the whole clause* (propositional anaphora) — a different use of null.

**What I did:** `candidates` = the retrieval-time candidate SET (what the
encoder is scored on, contract §7); `ref` = the **gold** antecedent, fully
specified whenever the author knows it. `slot: null` + a `clause` index means
"the clause itself".

**What I need:** confirm that reading, or split the field. My concern is
supervision: contract §7 scores the encoder on set *recall* only, so the gold
pick is not needed encoder-side — but comprehension *will* need it, and `ref`
is the only place it can live. If `ref` must stay "the addressing hint", then
gold hand records need a separate `gold_antecedent` field, and that is a
contract change. This is M63.1c flag #2 (supervision/alignment) coming back in
concrete form.
*(Affects: all 8 records with a `context` source.)*

### D2 — Pure interjections with no synset: `entity`, or hold the family?

`senses_of("ugh") == []`, so `ugh .` grounds as `type: "entity"` — a
contentless node whose only signal is `utterance_kind: "interjection"`. That is
what contract §4.2 prescribes for an uncovered content word, and it is honest,
but it teaches the encoder "interjection ⇒ entity", which the USVS gloss-senses
(pre-build checklist item 5) are meant to make false.

**Options:** (a) ship as `entity` now, re-author after gloss-senses land;
(b) hold pure interjections out of gold until item 5 is done; (c) author now
against the sense ids the gloss pipeline *will* mint (unverifiable today).
**My recommendation: (b)** — one record teaching a wrong regularity is worse
than one fewer record.

**Two corrections to the M63.1c flag #1 list while here:**
- **`alas` is NOT senseless** — `senses_of("alas")` returns
  `["unfortunately.r.01"]`. It grounds literally today, no gloss work needed.
  Only `ugh`, `yuck`, `oops` (of the ones I probed) are genuinely uncovered.
- **The POS enum DOES have `INTJ`** (`quantum_parser/src/parser/enums.py:19`,
  mapped at `pos_tagger.py:1433`), contrary to contract §6's note. The
  *conclusion* still stands, for a different reason: the English tagger
  lexicon assigns no word to it — `ugh`→`PROPN`, `alas`/`shit`/`nonsense`→
  `NOUN`, `more`→`ADV`. So the interjection signal must ride on
  `utterance_kind`, not POS. Contract §6's parenthetical should be corrected.
*(Affects: `ugh .`; and the §6 note.)*

### D3 — Is the speaker a prime, or a memory reference?

`encoder_model.PRIMES == ["YOU", "<UNK_PRIME>"]`. So `YOU` is the only
first-class prime. Where the speaker is the grammatically-licensed filler
(`tell me …`, `me too .`, and any `I FEEL`-shaped reading), I used
`type: "reference"` with `source: "memory"` — identical to the teacher's own
handling of a bare pronoun, and consistent with teacher-bulk gold.

**What I need:** confirm the speaker stays a memory reference, or add `I` to
`PRIMES` and make first-person a resolved prime like the addressee. Symmetry
argues for the prime; consistency with the 1259 teacher records argues against.
*(Affects: `tell me about the dog .`, `wait for me .`, `me too .`.)*

### D4 — The role vocabulary is not actually frozen (three-way divergence)

| label | contract §3 "frozen" list | `clause.py` | teacher gold v2 |
|---|---|---|---|
| `SUBJECT`, `OBJECT`, `INDIRECT_OBJECT`, `PLACE`, `SOURCE`, `AGENT`, `RECIPIENT` | yes | yes | yes |
| PP-roles (`OF`, `FOR`, `WITH`, `ABOUT`, …) | yes (open) | yes | yes (30+ attested) |
| `QUANTITY` | **no** | **no** | **0×** — but contract §8.3's own worked example uses it |
| `COMPLEMENT`, `DESCRIPTION`, `SPECIFICATION`, `SUBJECT_COMPLEMENT` | **no** | yes (`MODIFIER_RELATIONS`, line 442) | 0× in this snapshot; the richer-gold work adds them |
| an additive/focus role (for `too`) or a manner role (for `well`) | no | no | no |

I used `QUANTITY` in `more !` (following §8.3) and `COMPLEMENT` in
`sounds good .` (following `clause.py`), and **dropped** `too` and `well`
entirely — they are flushed as ungrounded tokens, losing the additive and
manner meaning of those utterances.

**What I need:** one reconciled role list, and a ruling on whether particles
like `too` get a role or stay dropped.
*(Affects: `more !`, `me too .`, `stir well .`, `sounds good .`.)*

### D5 — Inflection: `senses_of` is a raw-surface lookup, so most verbs ground as `entity`

This is the highest-volume finding, and it is **not** specific to hand gold —
it is how `scripts/build_encoder_gold_v2.py` already builds all 1259 teacher
records. `senses_of` is a **lemma** index queried with the **raw surface
token**, so every inflected form misses:

```
verbs                        nouns                 (plurals miss too)
come 22   came   0           child 4    children 0
ask   7   asked  0           boy   4    boys     0
want  9   wants  0           girl  5    girls    0
go   35   sounds 0
break 75  broke  1  (broke.s.01 — the adjective "penniless")
open 36   opened 3  (open.a.05, opened.s.01, open.s.09 — all adjectival)
eat   6   ate    2  (ate.n.01 — the Greek goddess; fudge.n.01)
```

Every past-tense and 3sg verb, and every plural noun, therefore grounds as
`type: "entity"` — a *contentless* node. Worse, the forms that *do* hit are
hits on the **wrong lemma**: `opened` returns three adjectives, `broke`
returns "penniless", `ate` returns a Greek goddess. Under contract §7 (gold
sense ∈ emitted candidates) those records are unrecallable by construction —
the correct sense is not in the set. Every `context[]` entry in
this batch has an `entity`-grounded plural subject for exactly this reason. Contract §8.1/§8.3's own
worked examples (`"flew" → fly.v.01…`, `"wants" → want.v.01…`) are **not
reproducible** by the real pipeline.

`hand_gold.W` has a `lemma=` override, **deliberately left off** so hand gold
and teacher-bulk gold stay consistent for the union.

**What I need:** a ruling. (a) leave as-is (hand gold stays consistent, gold
stays impoverished); (b) lemmatize at retrieval in *both* builders (WordNet's
morphy is already available through `nltk`) and regenerate teacher gold;
(c) hand gold only. **My recommendation: (b)** — this looks like a large,
cheap recall win across the whole corpus, independent of the hand-gold work,
and it would make the contract's own examples true. Worth its own probe.
*(Affects: every record with an inflected verb — here `opened the box .`,
`bebió el agua .`, `sounds good .`, `the dog did .`; and all teacher gold.)*

### D6 — May an elided slot name its surface carrier?

In `the dog did .` the auxiliary `did` stands in for the elided predicate. I
gave the elision slot `token_index: null` and let `did` be flushed, so the
record discards tense and (in `did n't`) polarity. Contract §3 says
`token_index` is null "for synthesized/elided fillers with no surface token" —
but `did` *is* a surface token for that slot, just not a contentful one.

**What I need:** may an `elision` grounding carry the `token_index` of its
stranded carrier? It costs nothing structurally (`legal_action_types` already
admits `EMIT_UNRESOLVED_SLOT` with an index) and would preserve tense/polarity.
*(Affects: `the dog did .`; and any `do`/`have`/modal stranding.)*

### D7 — Should resolved nodes be filtered out of antecedent candidate sets?

In `stir well .` the context is itself an imperative, so its synthesized
`SUBJECT` (prime `YOU`, no surface token) sits in the elision slot's candidate
pool alongside `flour` and `bowl`. Emitting a superset is correct-by-design
under contract §7 (precision is not penalized), but a prime is never a
plausible antecedent for a dropped *object*.

**What I need:** confirm "emit everything, let comprehension sort it" — or
specify a filter (by `grounding.type`, by role compatibility). I left the
prime in, per §7.
*(Affects: `stir well .`; any context containing an imperative.)*

### D8 — The `fired_rules` feature does not reach the families it exists for

Not a schema question — a **model-input** finding, but it lands squarely on
this batch, so flagging it here. `encoder_model.compute_fired_rules` is the
encoder's *only* grammar conditioning signal, and the 7 rules are exactly the
7 hard-case rules. Run over these 16 records, it misfires on **6 of them**:

| record | family | rules that fire | problem |
|---|---|---|---|
| `nonsense !` | interjection | **none** | R3's trigger is a hard-coded 13-word `_INTERJECTION_LEXSET` (`encoder_model.py:88`). The record chosen precisely to show the family generalizing beyond a word list gets no signal. |
| `me too .` | elision | **none** | R4 needs `pos[0] ∈ {ADV,ADJ,PROPN,ADP,NOUN}`; this starts `PRON`. |
| `the dog did .` | elision | **none** | starts `DET`. |
| `and a hat too .` | elision | **none** | starts `CCONJ`. |
| `opened the box .` | subject-drop | R4 (`ellipsis.inherit_predicate`) | fires the **wrong** rule; R5 `prodrop.null_argument` does NOT fire, because it requires a token tagged `VERB` and the tagger calls `opened` an `ADJ`. |
| `bebió el agua .`, `sounds good .` | subject-drop | R4 | same: predicates tagged `NOUN`, so R5 cannot fire. |

And the converse: R5 `prodrop.null_argument` **does** fire on `stir well .`,
where the dropped argument is the *object*, not a pro-dropped subject.

So the four imperatives are the only family whose rule fires cleanly. The
`_INTERJECTION_LEXSET` point is the sharper one: a hard-coded 13-word list is
the same off-philosophy shape the lead rejected for the appraisal table, and it
sits in the encoder's feature path today.

**What I need:** confirm this is worth a fix pass (retrigger R3 off
`utterance_kind` context rather than a lexset; loosen R4's `pos[0]` gate;
make R5 fire on a *missing subject* rather than on a `VERB` tag) — or that
`fired_rules` is understood to be advisory and the encoder is expected to learn
these families from the gold alone. Either is defensible; right now the gold
says one thing and the feature says another.
*(Affects: 6 of 16 records; independent of the schema.)*

### D9 (minor) — Forest width for hand gold

Every draft carries a **one-tree** lattice. That matches the margin-pruning
decision (lead, 2026-09-07: gold should reflect real ambiguity, not parser
hypothesis count), and these fragments genuinely have one reading. But a
hand-authored *structurally ambiguous* record is the only way to give the
encoder a positive multi-tree example — the teacher's top-k is exactly the
polluted signal being pruned away. `hand_gold_record` takes a `trees_spec=` for
this; I did not use it. **Worth a follow-up batch?**

---

## 5. Reproducing

```bash
cd consciousness_transformer
pip install torch numpy nltk pytest && pip install -e .
NLTK_ALLOW_PROXIED_URLOPEN=1 python -c \
  "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('omw-2.0')"
python scripts/build_usvs.py                    # ~6 min, CPU
python scripts/build_hand_gold_draft.py         # builds + gates + writes the jsonl
python scripts/render_hand_gold_draft.py        # regenerates §3 of this doc
```

`build_hand_gold_draft.py` exits non-zero and writes nothing if any record
fails the gate.

### One further finding, not a decision

**Cross-lingual grounding already works.** I expected `senses_of` to be
English-only and found that USVS indexes OMW lemmas: `niño` → 8 senses
(`baby.n.01`, `child.n.01`, …), `agua` → 7, `casa` → 14, `beber` → 7. So the
code-switch family named in the two-part-gold decision needs **no new
retrieval machinery** — a Spanish record grounds through the identical path.
The only gap is inflection (`bebió`, `está` → `[]`), which is D5 again, in
Spanish.
