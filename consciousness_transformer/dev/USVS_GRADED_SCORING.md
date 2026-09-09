# USVS-GRADED tree scoring and loss

Status: DESIGN + BUILT (branch `usvs-graded-scoring`, 2026-09-08).
Answers the lead directive of 2026-09-08 (`RESEARCH_NOTES.md`, "LEAD
DIRECTIVE (2026-09-08): score trees in USVS space, not binary").

Implementation: `src/nsm_ct/usvs_graded.py` (metric + role matrix + soft
targets), `src/nsm_ct/encoder_model.py` (`teacher_force_loss(...,
soft_targets=...)`), `src/nsm_ct/encoder_train_util.py`
(`evaluate_full(..., metric=...)`), `scripts/rescore_encoder.py`
(`--metric graded`), `scripts/rescore_graded_arms.py` (the arms table),
`tests/test_usvs_graded.py`.

---

## 0. The problem this fixes

Both the loss and the metric are binary today.

- **Loss.** `encoder_model.teacher_force_loss` is a sum of one-hot
  cross-entropies over five discrete heads (action-type, role, clause-kind,
  grounding-type, source, prime). Predicting `SPECIFICATION` where gold says
  `DESCRIPTION` costs exactly what predicting `SUBJECT` costs.
- **Metric.** `_tree_edge_set` / `_best_tree_overlap` / `score_record` build
  a `frozenset` of `(relation, token_index, gtype)` triples per tree and
  intersect it with gold's. A node attached one token off, or under a
  near-synonymous role, contributes 0 — identically to a phantom node that
  denotes nothing.

So the numbers in `RESEARCH_NOTES.md` ("ARMS-2 COMPLETE": rank-1 edge-F1
0.49 on v4b targets) cannot distinguish *near-miss* from *nonsense*, and the
gradient carries no notion of "you were close".

---

## 1. What the schema actually supports (and what it does not)

Read against `dev/ENCODER_IO_CONTRACT_V2.md` §3–§4:

1. **The encoder never picks a sense.** Contract §0/§4.2 and
   `encoder_model.py`'s module docstring: a `sense` node carries the *whole*
   `senses_of` candidate set, copied verbatim from
   `token_sense_candidates[token_index]`, on the gold side *and* the decode
   side. There is no head anywhere that scores one candidate against another.
   Consequence for the metric: two `sense` nodes at the **same** token index
   necessarily carry the **same** candidate set and therefore the same
   vector; the USVS signal in the metric is entirely about nodes at
   **different** token indices — i.e. *which word did you attach here*, not
   *which sense of it did you choose*. That is the right target: "a modifier
   attached one head off" is exactly the case the lead named.
2. **`prime` / `entity` are resolved, `reference` / `elision` are unresolved
   slots with a `retrieval.source`.** None of the four carry a USVS
   coordinate. They need reserved-axis vectors (§2.2).
3. **Clause structure is a flat list.** A tree is `{"clauses": [...]}`; each
   clause is one predicate node + a list of role nodes + an
   `utterance_kind`. There is no nesting, so "head-clause index" is just the
   clause's position in that list.
4. **No node embedding exists in the model.** `EncoderModel` projects USVS
   features *in* (`sense_proj`) but emits nothing in USVS space; the
   controller hidden state `h` feeds six `nn.Linear` classifier heads and
   nothing else. An auxiliary "cosine(predicted node embedding, gold USVS
   vector)" term therefore has **no hook** without adding a head — that is
   new architecture, so it is not built (§5.4).

---

## 2. Node vectors

`usvs_graded.node_vector(node, record, usvs) -> np.ndarray[D + RESERVED]`.

The space is the USVS axis space (`len(usvs.axes)` = 607 in the current
build) with a **reserved block** appended. Sense meaning lives in the USVS
block; everything else lives in the reserved block, and the two blocks are
disjoint, so a `sense` node and a `prime` node always score cosine 0.

### 2.1 Grounded (`sense`) nodes — the candidate-set mean

The node's candidate set is `grounding.candidates` when present, else
`token_sense_candidates[token_index].sense_candidates` (contract §4.2: the
table is authoritative). The vector is the **L2-normalized mean of
`usvs.sense_dense(sid)` over the candidates that USVS covers**. When the set
has exactly one sense this degenerates to that sense's own (normalized)
vector, as the directive asks.

Why the mean of the *set* and not a pick: the encoder emits sets and the
gold node *is* a set (contract §0). Taking `candidates[0]` would silently
reintroduce the MFS commitment that v2 exists to abolish, and would make the
metric depend on WordNet's frequency ordering.

**Failure modes, stated plainly.**

- *Ambiguity dilution.* A 28-candidate token (`block` in the v4b sample
  record) has a mean vector that is nobody's meaning in particular; its
  cosine against a 2-candidate token is compressed toward the corpus mean.
  Highly-polysemous words therefore get systematically *flatter* graded
  scores, in both directions.
- *Same-token blindness.* Per §1.1 — no credit is ever gained or lost for
  the sense *pick*, because there is none. This metric grades attachment and
  role, not WSD.
- *Anisotropy.* USVS sense vectors are non-negative sparse gloss signatures,
  so the ambient cosine between two unrelated senses is well above 0
  (`sim(dog, justice) = 0.55` on the current build). Absolute graded scores
  are therefore **not** comparable to edge-F1 in level, only in *ordering*.
  The random-sentence control (§4) measures this floor directly, and it is
  the number to read the graded column against — not 0.
- *Uncovered token.* A `sense` node whose candidates are all outside USVS
  (or empty) has no USVS vector; it falls back to the reserved block as
  `("sense", surface token)` — exact-match-only, like an entity.

### 2.2 `entity` / `reference` / `elision` / `prime` — reserved axes

These get a unit vector in the reserved block with two parts:

| component | axes | weight |
|---|---|---|
| **type block** — the `GTYPE_CONFUSION` row of the node's `grounding.type`, L2-normalized | one axis per type (5) | `TYPE_W = 0.5` |
| **identity** — `hash("<gtype>\|<identity_key>") % IDENT_AXES` | 4096 axes | `sqrt(1 - TYPE_W²) ≈ 0.866` |

with `identity_key` = the lowercased surface token for `entity` /
`reference`, the prime name for `prime`, and `None` for `elision` (an elided
filler has no surface identity — its type block alone carries weight 1.0).

The identity axis is keyed **by type as well as identity**, so two different
types never share one; their similarity is decided entirely by the type
block, and the type block reuses the *same* `GTYPE_CONFUSION` table the soft
loss uses (§5.2), so the metric and the loss cannot disagree about which
grounding types are near-misses.

Resulting cosines: same type + same identity **1.0** (they "compare
exactly", as the directive asks); same type + different identity **0.25**;
`reference` vs `elision` (the contract's own "one shape, three instances"
pair) **0.2** with matching identities, **0.4** for an identity-less
`elision`; unrelated types **0.0**; against any `sense` node **0.0**
(disjoint blocks).

*Failure modes.* Two distinct names get 0.25 purely for both being entities,
which is a weak claim, not a semantic one; a `reference` node and the `sense`
node it actually corefers with score 0 even when they denote the same thing
(resolving that is comprehension's job, and the contract deliberately keeps
it out of encoder gold); and the identity block is **hashed**, so two
distinct identities can collide onto one axis and score as identical — at
4096 axes that is ~0.02% of node pairs, small but not zero.

---

## 3. Tree flattening and `usvs_tree_similarity`

`usvs_graded.flatten_tree(record, tree, usvs) -> FlatTree` takes a
**normalized-shape** tree (`beam_decode`'s output shape). Gold-lattice trees
are converted first with `tree_render.normalize_gold_tree` — the same
flattening the parse judge already uses, per the directive.

`FlatTree` = a bag of `FlatNode(role, clause_index, is_predicate,
token_index, vector, weight)` plus `n_clauses` and `kinds`
(`utterance_kind` per clause).

`usvs_tree_similarity(pred_tree, gold_tree, record, usvs) -> dict`.

### 3.1 Pairwise score

    S(p, g) = role_weight(p.role, g.role) * max(0, cosine(p.vector, g.vector))

Negative cosines are clamped to 0 (a node cannot be *worse than unrelated*).
Clause index does **not** enter the pair score — the directive's formula is
role-agreement × cosine, and clause agreement is reported separately as
component (4), so the two are not double-counted.

### 3.2 Role confusion matrix — one place, easy to edit

`usvs_graded.ROLE_CONFUSION` (explicit unordered pairs) plus
`_role_family_weight` (the two rules that need a family, not a pair):

- explicit pairs include `{DESCRIPTION, SPECIFICATION} 0.5`,
  `{COMPLEMENT, SUBJECT_COMPLEMENT} 0.5`, `{ADDITIVE, FOCUS} 0.5`,
  `{SUBJECT, AGENT} 0.5`, `{INDIRECT_OBJECT, RECIPIENT} 0.5`,
  `{OBJECT, INDIRECT_OBJECT} 0.3`, `{OBJECT, RECIPIENT} 0.3`,
  `{SOURCE, PLACE} 0.3`, and — required explicitly by the directive —
  `{SUBJECT, OBJECT} 0.0`.
- family rules: `PLACE` (or `SOURCE`) against an **open PP role** (any role
  label outside the closed core + `clause.MODIFIER_RELATIONS` set — i.e. an
  uppercased preposition, contract §3) scores **0.5**; two *different* open
  PP roles score **0.25**.
- `PREDICATE` matches only `PREDICATE` (1.0); a predicate aligned to a role
  is not partial credit, it is a structural error.
- identical labels 1.0, everything else 0.0.

### 3.3 Node weights

`PREDICATE` **2.0**, core arguments **1.0**, `clause.MODIFIER_RELATIONS`
(`DESCRIPTION`, `SPECIFICATION`, `COMPLEMENT`, `SUBJECT_COMPLEMENT`,
`QUANTITY`, `ADDITIVE`, `FOCUS`) **0.5**. Open PP roles count as core args
(1.0) — they are argument-like attachments, and the PP-attachment case is
the one the lead named.

### 3.4 Alignment

One-to-one, maximizing `Σ S(p,g) · (w(p)+w(g))/2` over the assignment, by an
exact O(n³) Hungarian solver written in-module
(`_hungarian_max`) — no new dependency, fully deterministic. Trees here have
tens of nodes, so cost is irrelevant. A greedy variant
(`_greedy_max`) is kept and is what the solver falls back to if the matrix
is degenerate; the tests pin the Hungarian path.

### 3.5 Graded P / R / F

With `M` the set of aligned pairs:

    graded_P = Σ_{(p,g)∈M} S(p,g)·w(p)  /  Σ_{p∈pred} w(p)
    graded_R = Σ_{(p,g)∈M} S(p,g)·w(g)  /  Σ_{g∈gold} w(g)
    graded_F = harmonic mean(P, R)

so an unmatched (or badly-matched) **predicted** node costs precision and an
unmatched **gold** node costs recall, exactly as the directive specifies.
Empty prediction → P = R = F = 0; empty gold → NaN (skipped in aggregation).

### 3.6 Clause-structure term

    clause_count  = min(n_p, n_g) / max(n_p, n_g)
    clause_kind   = (# i < min(n_p,n_g) with kinds equal) / max(n_p, n_g)
    clause_struct = 0.5·clause_count + 0.5·clause_kind

Reported on its own. The single combined number is
`graded_overall = 0.85·graded_F + 0.15·clause_struct`, and every component
(`graded_p`, `graded_r`, `graded_f`, `clause_count`, `clause_kind`,
`clause_struct`, `n_pred_nodes`, `n_gold_nodes`, `matched_mass`) is returned
in the dict, per the directive's "report the components separately".

Identity (gold vs itself) gives 1.0 on every component.

### 3.7 Forest handling

`usvs_graded.score_record_graded(record, forest)` mirrors
`encoder_model.score_record`'s eval view: for each gold tree, the forest tree
with the **highest `graded_overall`** is selected, and the per-record numbers
are averaged over gold trees. `evaluate_full` reports both the best-of-k
view and the rank-1 (committed) view, matching the existing edge metrics
one-for-one so the two columns are directly comparable.

---

## 4. Sanity controls

`usvs_graded.corrupt_tree` builds the corrupted copy the directive asks for:
swap `SUBJECT`↔`OBJECT`, re-point the predicate at a non-verb token, and add
a phantom role node. Three controls, run in `tests/test_usvs_graded.py` and
printed at the head of `runs/rescore_graded.txt`:

1. gold vs itself → **1.0** exactly.
2. gold vs corrupted copy → strictly less than identity.
3. gold vs a different sentence's tree → near the USVS ambient floor (§2.1),
   *not* 0 — and that floor is what the arm numbers must be read against.

---

## 5. The soft-target loss (`--loss usvs-soft`)

Opt-in. Without the flag `teacher_force_loss` is called exactly as before
(`soft_targets=None`) and is numerically identical — pinned by a test and by
`tests/test_encoder_curve.py`'s byte-identical baseline-commit regression.

### 5.1 ROLE head — soft over the role-confusion matrix

Target row = `softmax(ROW / T)` where `ROW[j] = role_weight(gold_role,
role_vocab[j])` and `T = ROLE_SOFT_TEMPERATURE` (default 0.25, configurable
via `--loss-temperature`). Since `role_weight(r, r) = 1.0` is the row's
unique maximum, the target always puts its maximum mass on the gold class,
and mass on partially-related roles falls off with `T`. Loss becomes
`-(target · log_softmax(logits)).sum()`.

### 5.2 GROUNDING-TYPE head — soft over the contract's own equivalences

`GTYPE_CONFUSION`: `{reference, elision} 0.5` — contract §4 states outright
that these are "one shape, three instances" of the same unresolved-slot
construct — and `{sense, entity} 0.3` (both are resolved-or-lexical content
nodes; `entity` is literally the fallback for a content word USVS does not
cover, contract §4.2). Everything else 0.

### 5.3 SOURCE head — soft over `context ≈ memory`

`SOURCE_CONFUSION`: `{context, memory} 0.5`. Contract §4.3 says it directly:
"`source:"context"` at authoring time aligns to `source:"memory"` at run
time." `{self, context} 0.25`. Everything else 0.

### 5.4 What stays hard, and what could not be done

- **Action-type and the terminal heads stay hard**, as directed — the
  transition system's actions are not semantically graded, and
  `terminal_weight` already shapes them.
- **Clause-kind stays hard.** `proposition`/`imperative`/`interjection` are
  speech-act shapes with no USVS structure to borrow.
- **Prime head stays hard.** `YOU`/`I`/`<UNK_PRIME>` — three referents, no
  graded relation.
- **There is no sense-candidate head to soften.** Per §1.1 the encoder
  never scores one candidate against another; the directive's
  "target proportional to cosine to the gold sense" has no head to attach
  to. **Skipped, by design, not by omission.**
- **The auxiliary USVS-cosine term is not built.** Per §1.4 the model emits
  no node embedding; the nearest hook would be a new
  `Linear(controller_hidden, d_axes)` head trained against the gold node's
  USVS vector. That is new architecture, which the directive explicitly
  forbids inventing. It is the obvious next step if the graded metric proves
  its worth, and is the only place a *sense-space* gradient could enter this
  model at all.

---

## 6. What the graded metric can and cannot settle

It **can** tell you how far a wrong edge sits from the right one — whether
an arm's misses are near-misses (attachment slippage) or nonsense (phantom
nodes), and whether the arm *ranking* under edge-F1 survives.

It **cannot** turn 607 gloss-derived axes into a WSD judgement (§1.1), and
its absolute level is not interpretable without the random-tree floor
(§2.1). Read the graded column as an ordering plus a distance-above-floor,
never as "the encoder is 0.7 correct".
