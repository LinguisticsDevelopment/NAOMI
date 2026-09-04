# Encoder Model Spec — the candidate-lattice emitter (implementable)

Status: IMPLEMENTABLE. Author: encoder step 5 pass, 2026-09-04. This is the
architecture spec a Sonnet impl agent builds to next (M63 Step 2, design §10).
It specifies **the model** — the learned universal encoder — that reads a
sentence and **emits the V2 candidate lattice**.

Read first, in order:
- `dev/ENCODER_IO_CONTRACT_V2.md` — the exact OUTPUT the model must emit (the
  candidate lattice). This spec's tensors serialize to that schema; every field
  name below is its vocabulary.
- `dev/ENCODER_GRAMMAR_FORMAT_PROPOSAL.md` — the grammar as retrieval-conditioned
  INPUT and the **action space** (§4: the 4 candidates-first actions; §5: the 7
  rules). The model's action head decodes over exactly that inventory.
- `dev/UNIVERSAL_ENCODER_DESIGN.md` §2 (mechanism), §11 (the pivot), §1 (the
  invariant: learned pieces are *boundary transducers only*, never knowledge).

**The one-line contract of this model.** It is *literally a token→tree-SET
transducer* (contract §0). It **EMITS THE CANDIDATE LATTICE and does NOT
disambiguate**: it never argmaxes a sense, an attachment, an antecedent, or a
filler. Every ambiguity leaves the model as a *candidate set*; the pick is
comprehension's (contract §7, §10). The architecture below is built so that
argmax over candidates is **not even representable** — there is no head that
scores one sense against another.

This document describes a model/interface shape only. It does not modify, and is
not itself, any `src/` code; it is the blueprint the impl writes.

---

## 1. IN / OUT tensor contract

### 1.1 Input (per sentence)

A sentence is presented as a list of `T` tokens, each carrying three retrieved
conditioning channels (design §2.2). Nothing here is a free-text embedding of
world knowledge — every channel is a *retrieval result* off a fixed symbolic
inventory, keeping the invariant (design §1).

| channel | tensor | source | notes |
|---|---|---|---|
| `tok_id` | `LongTensor[T]` | `SimpleTokenizer` | word-level id; **hash-embedded** (§2.2), not a big learned table — knowledge stays in USVS, not the token embedding |
| `pos_id` | `LongTensor[T]` | `ParserInputEncoder._tag` → `quantum_parser` `Tag` enum | coarse POS, ~20 tags (no `INTJ`, contract §6) |
| `sense_cand` | ragged `List[List[sense_id]]` len `T` | `USVS.senses_of(token)` | the per-token sense-candidate table = `token_sense_candidates` (contract §1). Empty ⇒ token is not USVS-covered ⇒ grounds `type:"entity"` (contract §4.2) |
| `sense_feat` | `FloatTensor[T, S_max, d_sense]` | `USVS.sense_dense(sense_id)` | dense USVS vector per candidate sense, mean-pooled per token into `FloatTensor[T, d_sense]` — the *set* summarised, order-free, so no sense is privileged |
| `fired_rules` | `FloatTensor[T, R]` multi-hot | grammar-trigger match (§3.3) | which of the `R` declarative rules (grammar §5) fire at each position; this is the retrieval filter that both *conditions* the controller and *masks* the action head |

`is_entity(token)`, `chosen_sense` (contract §1: MFS, **informational only, never
a target**) travel alongside for serialization but are **not** model inputs to
any scoring head.

### 1.2 Output — the V2 lattice

The model emits, per sentence, exactly the top-level record of contract §1:

```
{ "text", "tokens", "pos",
  "lattice": { "trees": [ {clauses:[…]}, … ],   # the FOREST (§4)
               "discourse_links_per_tree": [ […], … ] },
  "token_sense_candidates": [ … ],               # copied verbatim from sense_cand
  "context": [ … ]  # optional; empty in Stage-i gold (§5) }
```

The load-bearing rule, restated as a tensor invariant:

> **Senses and attachment are EMITTED AS CANDIDATE SETS, never argmax'd.**
> A `grounding.type:"sense"` node's `candidates` is **`sense_cand[token_index]`
> copied whole** — the model chooses *that this node is a sense-grounding site
> and which token it covers*, and the full retrieved set is attached by copy.
> There is no per-sense logit anywhere in the network. Structural ambiguity is
> emitted as **multiple trees in `lattice.trees`** (§4), not a chosen tree.

So the model's entire learned output is: (a) a set of **derivations** (action
sequences) that build the forest, and (b) at each node, a **grounding-type**
(`sense`/`entity`/`reference`/`elision`/`prime`) + the `token_index` / prime it
binds. The candidate *contents* (sense strings, antecedent sets) are copied from
retrieval, never generated — that is what makes the gold label-free (contract §7)
and makes argmax unrepresentable.

---

## 2. Architecture — a retrieval-conditioned, grammar-constrained transition parser

Design §2.4: "a retrieval-augmented, grammar-constrained neural transition
parser. Action space = applying candidate grammar rules; grounding = selecting
candidate USVS senses." In candidates-first, "selecting" becomes "emitting the
candidate slot" (grammar §4). Transition-based over chart-based per design §7.1
(incremental, cheap, streaming, matches the incremental clause reactor).

### 2.1 The transition system (configuration + actions)

Configuration `c = (buffer, stack, partial_lattice, i)`: `buffer` = remaining
tokens, `stack` = open constituents/clauses, `i` = read pointer. A **derivation**
is a sequence of actions from the initial config to a terminal one; a completed
derivation serializes to **one tree** in `lattice.trees`.

The action inventory is the grammar's 4 candidates-first actions (grammar §4)
plus minimal shift/reduce/attach scaffolding. Actions are **factored** — an
action-TYPE head (7-way) then a typed ARGUMENT head conditioned on the type —
which keeps every softmax tiny and masking clean:

| # | action(type, arg) | grammar §4 action | emits into the lattice | commits? |
|---|---|---|---|---|
| 1 | `SHIFT` | — (scaffold) | advance `i`, push token | no |
| 2 | `OPEN_CLAUSE(kind)` | `OPEN_CLAUSE` | new clause frame, `utterance_kind=kind ∈ {proposition,imperative,interjection}` | licenses a clause |
| 3 | `GROUND(role)` | `GROUND` | a node filling `role` (or `PREDICATE`); a **grounding-type sub-head** sets `type ∈ {sense, entity}` — `sense` copies `sense_cand[i]`, binds `token_index=i`, `retrieval.source:"lexicon"` | **no** — emits the SET |
| 4 | `ATTACH(relation)` | — (scaffold) | attach top stack constituent to open clause under `relation` (frozen role vocab, contract §3) | no |
| 5 | `EMIT_SYNTH_SLOT(role)`+`GROUND_PRIME(P)` | `EMIT_SYNTH_SLOT`+`GROUND_PRIME` | a surface-less role, `type:"prime"`, `prime=P` (imperative `YOU`) | resolved — one licensed filler |
| 6 | `EMIT_UNRESOLVED_SLOT(role\|predicate, source)` | `EMIT_UNRESOLVED_SLOT` | empty slot, `type ∈ {reference, elision}`, `retrieval.source ∈ {self,context,memory}`, `candidates` = a *pointer* to the retrieval bucket (never a chosen antecedent) | **no** — emits candidates, binds none |
| 7 | `CLOSE_CLAUSE` / `REDUCE` | — (scaffold) | finish the open clause / pop | no |

Actions 5–6 emit the unresolved/synth constructs uniformly with 3 (grammar §4:
`GROUND` and `EMIT_UNRESOLVED_SLOT` are the same operation over lexicon vs
memory retrieval). The grounding-type sub-head is the ONLY place a node's `type`
is decided; it never ranks candidates *within* a type.

### 2.2 The controller (sub-MB policy)

A small recurrent controller scores the next action from the current config:

```
per-token feature   x_t = concat( hash_emb(tok_id_t),        # d_tok, hashed (no big table)
                                   pos_emb(pos_id_t),         # d_pos
                                   proj_sense(sense_feat_t),  # pooled USVS set → d_sense
                                   rule_proj(fired_rules_t) ) # R-hot → d_rule
enc                = biGRU(x_1..x_T)              # d_model per token, one pass, cached
controller state   h_k = GRU_cell( [ enc[i_k] ; stack_top_repr ; prev_action_emb ], h_{k-1} )
action-type logits = W_type · h_k                 # 7-way
arg logits         = W_arg[type] · h_k            # typed head (role / kind / prime / source)
grounding-type     = W_gt · node_repr             # 5-way, at GROUND/EMIT sites only
```

Recommended dims (Stage-i default): `d_model=128`, `d_tok=64` (hash buckets
`2^15`), `d_pos=16`, `d_sense=32` (USVS `sense_dense` pooled + projected),
`d_rule=32` (`R≈16` rules), GRU hidden `128`.

**Param budget (state param budget, design §9.4 "small model, not an LLM").**

| block | ≈ params |
|---|---|
| hash token emb (`2^15 × 64`, or `2^13 × 64` for smoke) | 2.1 M raw table BUT shared/hashed — count only the *projection* if a hashing-trick bag-of-buckets is used; with `2^13` buckets ≈ 0.5 M, with a 2-hash compositional emb (`2×2^12×32`) ≈ 0.26 M |
| pos + rule + prev-action embs | ≈ 6 K |
| input proj → `d_model` | ≈ 18 K |
| biGRU encoder | ≈ 200 K |
| controller GRU cell | ≈ 99 K |
| action-type + typed-arg + grounding-type heads | ≈ 8 K |
| **learned policy (excl. token table)** | **≈ 0.33 M → ~1.3 MB fp32** |

The **policy state** (encoder + controller + heads, the part that holds
transduction skill) is **sub-MB at ≈0.33 M params**. The token table is the only
thing that can bloat it; the spec mandates a **hashed / compositional token
embedding** (no per-word row) so the table does not carry knowledge and the whole
model stays ≤~2 MB. If even that is too large for the smoke target, drop to
`d_model=64` and `2^12` hash buckets → ≈0.12 M policy params.

### 2.3 CPU-trainable in this cloud env (no GPU)

Everything above is a handful of GRUs + linear heads over sequences of median
length ~10 tokens / ~20–40 actions (gold-v2: median 3 trees/sentence, so ~3
derivations/sentence). No attention over long context, no big matmuls. Targets:

- **Framework:** PyTorch CPU (already the repo's stack). No CUDA calls.
- **Smoke train ≤~10 min CPU:** subset 150 records, `d_model=64`, `2^12`
  buckets, batch 16 derivations, 2 epochs, Adam. ~150 records × ~3 trees ×
  ~30 steps ≈ 13.5 K teacher-forced steps/epoch → seconds/epoch on CPU; the
  10-min budget is dominated by data-load + oracle linearization, not the net.
- **Full Stage-i:** all 985 records, `d_model=128`, batch 32, ~15–30 epochs —
  minutes-to-low-tens-of-minutes on CPU. State the wall-clock in the smoke
  config's header so regressions are visible.

---

## 3. Training

### 3.1 Teacher-forced transition sequences (the oracle)

Gold is `runs/encoder_gold_v2.jsonl` (985 records, built by
`scripts/build_encoder_gold_v2.py` — the SAME `_parse_topk_one` +
`extract_discourse` path this model replaces). For each record, an
**oracle linearizer** converts *each* gold tree in `lattice.trees` into its
canonical action sequence:

```
for tree in record.lattice.trees:                 # ≤ max_hypotheses (8) trees
  for clause in tree.clauses (source order):
    yield OPEN_CLAUSE(clause.utterance_kind)
    # predicate
    if predicate_grounding.type == "sense":  yield GROUND(PREDICATE) + gt=sense
    elif type == "entity":                   yield GROUND(PREDICATE) + gt=entity
    elif type == "elision":                  yield EMIT_UNRESOLVED_SLOT(predicate, source)
    for role in clause.roles (token_index order, synth/elided last):
        if grounding.type == "sense"|"entity":  SHIFT* → GROUND(role) + gt
        elif type == "prime":                   EMIT_SYNTH_SLOT(role)+GROUND_PRIME(prime)
        elif type in {"reference","elision"}:   EMIT_UNRESOLVED_SLOT(role, source)
    yield CLOSE_CLAUSE
```

`token_index` gives the deterministic surface order, so the oracle is
single-valued (the same left-to-right, consume-on-match walk the gold builder
already uses for repeated words). Each of a sentence's ≤8 gold trees is a
separate teacher-forced derivation; the forest is supervised as the *set* of
derivations, not one.

### 3.2 Loss — action CE + candidate-SET emission (why SET, not argmax)

```
L = Σ_steps  CE( action-type )                      # which of the 7 actions
  + Σ_steps  CE( typed-arg | action-type )          # role / kind / prime / source
  + Σ_GROUND CE( grounding-type )                    # sense vs entity per node
  + Σ_EMIT   CE( unresolved-type + source )          # reference vs elision, retrieval source
```

**There is no sense-selection term, by construction.** At a `GROUND` node the
model predicts *that it is a sense site* and *which token* (the action's arg = the
buffer position, supervised by `token_index`); the `candidates` set is then
`sense_cand[token_index]` **copied whole**. So "emit the gold candidate set at
each grounding site" is enforced *architecturally* — the network literally cannot
emit a single sense, because no head ranks senses. This is the contract's core
(contract §0, §7): the encoder is scored on the SET, and picking one is
comprehension's job, trained later by the K-12 ladder — never here.

**Unresolved-slot supervision** is likewise emit-the-slot-plus-its-set, not
select: the model is trained to *posit* the empty slot at the right position with
the right `type`/`source`; the candidate antecedent set is a retrieval result
(runtime memory, or authored `context[]`), never a chosen antecedent. Selection
among candidates is comprehension's (contract §4.1, §10).

### 3.3 Grammar-constrained action masking (illegal actions cannot be emitted)

At every step the action-type + arg logits are masked to the **union of legal
actions**, then softmaxed over survivors (illegal → `-inf` pre-softmax):

- **Transition-system preconditions** (can't `GROUND` an empty buffer, can't
  `CLOSE_CLAUSE` with no open clause, etc.).
- **Fired-rule licensing:** only actions in `⋃ action_map` of the rules whose
  `trigger` matches the current span are legal (grammar §5 rules, §6 item 1 —
  `trigger` IS the retrieval key). `fired_rules_t` (§1.1) precomputes trigger
  matches over POS / `lexset` / `position` / `surface_absent` / `context_present`.

This is design §2.4's guarantee — "the model *cannot emit ill-formed structure*;
the grammar defines the legal output space, the net composes within it." It also
means the net over-generates *candidates* on purpose (grammar §6 item 3): when
two rules fire on a span, both action branches stay legal → both trees reachable
in the forest (§4). Narrowing is comprehension's, scored there, not here.

---

## 4. The forest — top-k emission

At **inference** the model **beam-decodes** action sequences (beam width `B`);
each complete derivation is one tree. `lattice.trees` = the top-`k` **structurally
distinct** completed derivations, score-sorted — mirroring the teacher, which
already exposes `chart.hypotheses[:k]` (score-sorted, structurally deduped;
gold-v2 stats: `k=8`, median forest 3, p90 6, max 8). Dedup uses the same
clause-structure equality the gold builder applies. Senses are shared across
trees by `token_index` into the one `token_sense_candidates` table (contract
§2.2), so a `k`-tree forest stays compact — the model emits the table once.

At **training** decoding is greedy/teacher-forced per gold tree (§3.1); beam is
inference-only. `discourse_links_per_tree[t]` is emitted per tree from the
`ATTACH`/coordination actions in derivation `t` (contract §2, §3).

Set `B ≥ k` (e.g. `B=16, k=8`) so the beam can surface all structural
alternatives the gold forest contains.

---

## 5. Staging + the concrete FIRST deliverable

**Stage-i — English distillation (this deliverable).** Match the teacher
lattices in `encoder_gold_v2.jsonl` on held-out English (design §3 stage i, §11).
Note the honest scope of the *current* gold (from `build_encoder_gold_v2.py`'s
own docstring): it emits only `utterance_kind:"proposition"`, sense/entity
groundings, `reference` slots (`source:"memory"`, `candidates:null`), and the
**forest** (top-k trees). It does **not** yet exercise `EMIT_SYNTH_SLOT`
(imperative `YOU`), elision, or interjection — those live in the hand-authored
gold on the `encoder-handgold-*` branches. So the model *architecture* supports
the full 7-action space (§2.1), but Stage-i *supervision* covers the
propositional + sense + reference + forest subset. The impl must build the full
action head now (imperative/interjection/elision actions present but rarely/never
fired in Stage-i gold), so Stage-ii/iii and the hand-gold fold in without an
architecture change.

**Stage-ii — Spanish (later).** Add Spanish rules + senses to the retrieved
inventories (grammar §5 R5 pro-drop is already language-agnostic); match the
Spanish teacher; no English regression (design §3 stage ii).

**Stage-iii — code-switch (later).** No teacher gold; eval by
grounding-consistency / round-trip (design §3 stage iii). Out of scope now.

### The first deliverable a Sonnet impl produces

1. **`src/nsm_ct/encoder/model.py`** — the transition-parser policy (§2): feature
   embedder, biGRU encoder, controller GRU, factored action-type / typed-arg /
   grounding-type heads, grammar-mask hook. Pure PyTorch CPU. Sub-MB policy.
2. **`src/nsm_ct/encoder/oracle.py`** — the gold-tree → action-sequence linearizer
   (§3.1) + the fired-rule trigger matcher (§3.3), reused for masking at train
   and infer.
3. **`scripts/train_encoder.py`** — teacher-forced training over
   `encoder_gold_v2.jsonl` with the §3.2 loss and §3.3 masking; a seeded,
   forest-width-stratified train/dev/test split (§6); checkpoints + a metrics
   log.
4. **`scripts/eval_encoder.py`** — the §6 candidate-set-recall metric on the
   held-out split; reports per-node / per-slot / per-tree recall + the
   all-gold-recalled record rate; serializes emitted lattices for spot-audit.
5. **`configs/encoder_smoke.yaml`** (or a `--smoke` flag) — the §2.3 smoke config
   (150 records, `d_model=64`, `2^12` buckets, batch 16, 2 epochs) with the
   observed CPU wall-clock in its header, proving ≤~10 min end-to-end.

Ship 1–2 with unit tests (oracle round-trips a gold tree; mask never admits an
illegal action; sense emission copies the full set), then 3–5.

---

## 6. Eval — candidate-SET recall (contract §7)

The model is **never scored on the pick** — only on whether the gold element is
**recalled among the emitted candidates**. Precision (extra candidates) is **not
penalized** (contract §7); a *missing* gold candidate is the only failure.

- **Sense recall.** For each gold `type:"sense"` node, `gold_sense ∈ emitted
  node.candidates`. Since candidates are `senses_of` copied, this scores whether
  the model attached a sense-slot to the right `token_index` (not a WSD score).
- **Structure recall.** `gold_tree ∈ emitted lattice.trees` (clause-structure
  equality — same predicate/role skeleton), and gold discourse links ∈ that
  tree's link list. Report per-tree recall and gold-forest-⊆-emitted-forest rate.
- **Slot recall.** Each gold unresolved slot (`reference`/`elision`) is emitted at
  the right position with the right `type`; for `source:"memory"` the antecedent
  check defers to the retrieved set (contract §4.3 — the run-time index alignment
  is the grammar-doc §8 open question, not a Stage-i target).
- **Record-level metric.** A record is *correct iff every gold element is
  recalled* (sense ∈ node candidates, tree ∈ forest, slot emitted) for every
  node/slot/tree. Report the **all-gold-recalled record rate** as the headline,
  plus the three component recalls.

**Held-out split.** Split the 985 records **by sentence** (dedup already done by
the gold builder, so no leakage), seeded, **stratified by forest width** (1-tree
/ 2–3 / 4+ trees) so wide-forest structural cases appear in dev/test. Suggested
80/10/10. Report train and test recall each epoch; the Stage-i gate (design §3)
is high all-gold-recalled rate on held-out clean English with no train/test gap.

---

## 7. Interface points to existing code (reuse, do not reinvent)

The impl **wires to** these; it does not rebuild the parser or the lexicon.

| need | existing interface | notes |
|---|---|---|
| tokens + POS | `nsm_ct.input_encoder.ParserInputEncoder._tag` (via `nsm_ct.tokenizer.SimpleTokenizer`) | word-level tokens + `quantum_parser` `Tag` POS (design §9.4) |
| per-token **sense candidates** (retrieval input) | `nsm_ct.ground.usvs.USVS.senses_of(word)` → list; `USVS.sense_dense(sense_id)` → dense vec for `sense_feat`; `USVS._lemma_index` coverage decides sense-vs-entity | this IS `token_sense_candidates` (contract §4.2). Copied verbatim into output |
| **structure / top-k** (teacher gold + optional inference feature) | `nsm_ct.input_encoder.ParserInputEncoder._parse_topk_one(sentence, k, max_hypotheses, max_seconds)` → `(graphs, scores, margin)`; `_parse_graph_one` for k=1 | the teacher the model distills; do **not** call it inside the trained model's forward — it is only the gold source (and, if wanted, a training-time comparison) |
| graph → clauses + links | `nsm_ct.clause.extract_discourse(graph)` → `(clauses, links)` | the same extraction the gold builder runs; reuse for oracle verification |
| primes for `GROUND_PRIME` / gloss-added senses | `nsm_ct.ground.usvs` (`ground/usvs.py` gloss→prime→coordinate pipeline, contract §6) | pure interjection / imperative `YOU` grounding data — deterministic, not learned |
| grammar rules + action space | `dev/ENCODER_GRAMMAR_FORMAT_PROPOSAL.md` §4 (4 actions) + §5 (7 rules, `trigger`/`licenses`/`action_map`) | the declarative inventory the fired-rule matcher (§3.3) and mask read. NB: `mind/grammar.py` is the OLD membrane controlled-language parser — **not** this inventory; do not confuse them |
| gold data + its builder | `runs/encoder_gold_v2.jsonl` (985 records) + `scripts/build_encoder_gold_v2.py` + `dev/ENCODER_GOLD_V2_STATS.md` | training set; builder docstring documents which constructs are/ aren't in Stage-i gold (§5) |

New code lives under a new `src/nsm_ct/encoder/` package + `scripts/` +
`configs/`; it **imports** the above, adds no field to any existing module, and
touches no `src/` parser/lexicon internals.

---

## 8. The single biggest impl risk

**The run-time candidate-set identity for unresolved slots** (grammar §8's one
load-bearing unknown). Sense recall is well-posed because the USVS lexicon is
*fixed and shared* between gold-authoring and run time — the candidate set is the
same object both sides. **Memory-sourced antecedent sets are not:** gold-v2 emits
`reference` slots as `source:"memory", candidates:null` because no cross-sentence
context is threaded, so at Stage-i the model can only learn to *posit the slot at
the right position*, and "gold antecedent ∈ candidates" is not yet a checkable
target (the candidate ids are a live retrieval result, not fixed labels). If the
impl tries to supervise antecedent recall now it will train against a null/empty
target and either no-op or hallucinate a set. **Mitigation:** Stage-i trains and
scores unresolved slots on *emission* only (position + `type` + `source`), NOT on
antecedent-set contents; antecedent recall waits on the hand-authored
`context[]` gold (`encoder-handgold-*`) and a trained/aligned memory-retrieval
index — flagged to the lead as the prerequisite before `EMIT_UNRESOLVED_SLOT`
antecedent supervision is meaningful. Second-order risk: the hashed token
embedding under-separating rare words on the small CPU budget — cheap to probe by
ablating it against a POS+sense-only feature set in the smoke run.
