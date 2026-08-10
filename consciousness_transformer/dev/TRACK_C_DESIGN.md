# Track C research spike: a chained op algebra for the resolver

M56 deliverable (dev/NEXT_ARC_PLAN.md). Research/design only — no source code
changed to produce this document. Charter: single-step routing over op
outputs is just Track B with named features (user's brake, 2026-08-10); a
real instruction-set machine CHAINS, one op's output feeding the next
op's input, and the chain IS the program. This spike formalizes that
algebra against what the codebase actually computes today, surveys why
the chaining idea has failed before, explains the A/B record in terms of
which ops each side had, designs a training regime that treats gold
traces as data we already own (not a workaround), sets numeric go/kill
gates before any prototype exists, and returns a verdict.

Grounding for every claim below: `src/nsm_ct/resolver.py`,
`src/nsm_ct/entity_memory.py`, `src/nsm_ct/membrane.py`,
`src/nsm_ct/sense_chooser.py`, `src/nsm_ct/clause_reactor.py`
(`ClauseReactor._collapse`), `RESEARCH_NOTES.md` M53–M54c,
`dev/MIND_INTERFACE.md`, `dev/ROADMAP_LONG_TERM.md` ("The Advice Test").
One fact below was checked empirically for this spike (small script,
reported inline) because the record didn't already contain it.

---

## 1. The op algebra, formalized

### 1.1 Substrate facts the algebra must respect

Before naming ops, three measured facts about the substrate constrain
what any op can honestly mean:

1. **Memory is a fixed-shape, parameter-free order-3 tensor.**
   `entity_memory.query(memory, entity, relation) = einsum("bijk,bi,bj->bk",
   memory, entity, relation)` — a bilinear read, exact only when keys are
   orthonormal, otherwise interfering. `write` is a gated delta. Both are
   pure math, no learned weights. Any op that "reads memory" is this one
   function; there is no other memory primitive in the codebase.
2. **Candidate identity vectors carry no learned or grounded content.**
   `_ent_vec` grounds a name via `codec.filler_vec("var:" + name)` —
   `TPRCodec.filler_vec` draws a **deterministic, seed-fixed, non-trainable**
   random unit vector per label (`tpr.py:99-105`, confirmed: two independent
   `TPRCodec` instances produce byte-identical `"var:mary"` vectors, and
   `cosine(mary, john) = 0.089` while `cosine(mary, sandra) = -0.043` at
   `dim=32` — i.e. two same-gender names are **no closer** than a
   cross-gender pair; there is no gender geometry in candidate identity
   vectors at all). This matters directly for §1.8 below.
3. **The candidate-set contract has exactly one feature slot, and it is
   the MENTION's, not any candidate's.** `Resolver.forward`'s
   `cand_feature: [B, F]` is broadcast identically across every candidate
   `c` in the set (`resolver.py:97,152`: `cand_feature.unsqueeze(1).expand(b,
   C, -1)`). No per-candidate feature exists in the v1 membrane today.

### 1.2 Types

| type | meaning | shape | example |
|---|---|---|---|
| `Addr` | an entity/handle vector used as a memory KEY | `[d]` | `cand_entity[i]`, `hom_addr`, `subj_addr` |
| `Vec` | a content/meaning vector (memory VALUE, sense vector, context) | `[d]` | `mem_read`, sense candidate vector |
| `Feat` | a small closed-set deterministic feature vector | `[F]` (`F=6` today) | `mention_feature_vector(word)` |
| `Scalar` | a real number: logit, prior, margin | `[]` | `cand_prior[i]`, a score |
| `Dist` | a (masked) probability/one-hot distribution over a candidate axis | `[C]` | softmax/argmax collapse weights |

`Addr` and `Vec` share a representation (both live in `R^d`) but are kept
type-distinct because only `Addr`s are legal first arguments to
`mem_query` — a `Vec` (e.g. a sense candidate, which already **is** a
resolved meaning, per `clause_reactor.py:774-776`) is never a memory key.
This distinction is exactly what M54's SenseHead docstring calls out
("there is no entity address a meaning vector can stand in for") and is
worth making a type rule, not a comment.

### 1.3 The op inventory

Grounded, not invented — every op below is something the codebase
**already computes**, named and typed:

| op | signature | learned params? | where it already exists |
|---|---|---|---|
| `mem_query` | `(Addr, Vec) -> Vec` | none (fixed bilinear) | `entity_memory.query`; `resolver.query_candidates` loops it over `C` |
| `interact` | `(Vec, Vec) -> Vec` (elementwise product) | none | `SenseHead.forward`: `cand_entity * mem_read` — the exact M54c-proven missing prior |
| `combine` | `(Vec, Vec) -> Vec` (elementwise sum) | none | `clause_reactor._collapse` sense branch: `extra_ctx = mem_read + subj_read`, then `+ sense_cand_context` |
| `feature_match` | `(Feat, Feat) -> Scalar` | small (bilinear/MLP head) | **does not exist as a clean op today** — see §1.8, the headline finding |
| `prior` | `(Candidate) -> Scalar` | none | `cand.prior` / `SenseCandidateSet`'s MFS-rank `1/(rank+1)` |
| `score` | `(Vec, Vec, ...) -> Scalar` | small MLP (the one genuinely learned reducer) | `CorefHead.net`, `SenseHead.net`, `SharedScorer.net` — all are this op with different input concatenations |
| `select` | `({Scalar}_C) -> Dist` | none (softmax/argmax) | `ClauseReactor._collapse_weights` — soft in training, hard argmax at eval |
| `emit` | `(Dist, {Vec}_C) -> Vec` | none (weighted sum) | the `(w.unsqueeze(-1) * X).sum(1)` line in both `_collapse` branches — **but which `X`** (candidate `mem_read` vs. candidate vector itself) is hardcoded per branch today, not routed — see §1.9 |
| `compare` | `(Vec, Vec) -> Scalar` (cosine) | none | exists, but at the OUTPUT stage only: `cosine(response_vec, option_vecs)` for the final MC answer — **not currently used inside any resolver**; listed because the M56 charter names it and it IS real, just not yet a collapse-time op |

Seven working ops, one that doesn't clean type-check yet (`feature_match`),
one whose sibling exists but isn't wired into collapse (`compare`). That
gap is itself the point of doing this exercise before committing to an
implementation — see §1.8.

### 1.4 The register model

Minimal, per the charter's ask ("small register file per candidate"):

**Global registers** (one value per episode-step, shared across all `C`
candidates in the set being resolved):

- `G.rel` — the clause's current relation vector (`Addr`-typed key component)
- `G.state` — controller GRU state before this step's update
- `G.mention` — the mention's `Feat` (pronoun/homograph's own feature vector)
- `G.ctx` — an accumulator `Vec`, built by repeated `combine` calls

**Per-candidate registers**, indexed `i = 0..C-1`:

- `P.addr[i]` — the candidate's identity/value vector (`cand_entity[i]`)
- `P.mem[i]` — `mem_query(P.addr[i], G.rel)` — memory content at that candidate's address
- `P.feat[i]` — a per-candidate `Feat` (**the missing register**, §1.8)
- `P.tmp[i]` — scratch `Vec`, written by `interact`/`combine`
- `P.score[i]` — scratch `Scalar`, written by `feature_match`/`score`/`prior`

This is deliberately smaller than a general-purpose register file: no
addressing mode beyond "the current candidate slot `i`" or "global," no
indirection, no loops over anything but the fixed `C` (already small,
already enumerated by perception). That restriction is a design choice,
not an oversight — every graveyard failure in §2 that involved open-ended
addressing (DNC) or unbounded structures (stack RNNs) is exactly the
thing this register model refuses to have.

### 1.5 What chaining means with these types

A **chain** is a sequence of `(op, arg_bindings, dest_register)` triples.
Legal edges (which outputs can feed which inputs) fall out of the type
table directly:

```
mem_query   : (Addr, Vec)      -> Vec     -- consumes P.addr or G-registers holding an Addr
interact    : (Vec, Vec)       -> Vec     -- consumes any Vec register (P.mem, P.addr-as-sense, G.ctx)
combine     : (Vec, Vec)       -> Vec     -- same
feature_match: (Feat, Feat)    -> Scalar  -- consumes G.mention and P.feat ONLY (the only Feat-typed registers)
prior       : (Candidate)      -> Scalar  -- reads Candidate metadata directly, not a register
score       : (Vec, Vec, [Vec])-> Scalar  -- the learned reducer; consumes 2-3 Vec registers
select      : ({Scalar}_C)     -> Dist    -- consumes P.score across the candidate axis (the ONE axis-reducing op)
emit        : (Dist, {Vec}_C)  -> Vec     -- consumes a Dist plus a NAMED per-candidate Vec register (P.mem[*] or P.addr[*] — the routing decision §1.9 flags)
```

The **program** is the sequence of `(op, which registers)` choices a
router makes at each of a small fixed number of steps — that sequence is
the thing routing must learn (the "instruction sequence" in the
instruction-set-machine framing). Everything else (the actual op
execution) is fixed math or a small shared head, exactly like a real ISA
separates "which instruction, which registers" (routing, learned) from
"what add/multiply do" (the ALU, fixed).

### 1.6 Chain length budget / halting

Both gold programs written out below run in **4-5 steps**. Set the
budget accordingly: `K_max = 6` (one slack step over the longest known
program), enforced as a hard cutoff — at `K_max` an implicit `SELECT` +
`EMIT` fires regardless of whether the router asked for it, mirroring the
v2 membrane's `patience` dial (`MIND_INTERFACE.md` §"the dials") rather
than inventing a new halting mechanism. No learned halting probability in
v1 of C — a fixed small budget sidesteps the differentiable-halting
instability that sank Neural GPU and stack RNNs (§2) while costing
nothing at this task scale (both known programs fit in budget with room
to spare, and `C` is a handful of candidates, so `K_max=6` steps × a
handful of ops is enumerable for a legibility check, not just trainable).

### 1.7 The acid test: gold programs for the two solved tasks

**Pronoun program** (reproduces `CorefHead`, mechanism from `resolver.py`
`CorefHead.forward` + `clause_reactor._collapse`'s entity branch, relation
`r = rel:PLACE`, mention feature `G.mention = mention_feature_vector(pronoun)`):

```
for each candidate i in registry:
  1. P.mem[i]   = mem_query(P.addr[i], G.rel)                       # em.query(memory, cand_entity_i, rel:PLACE)
  2. P.feat[i]  = candidate_feature(P.addr[i])                       # *** GAP: no such op/register exists today ***
  3. P.score[i] = feature_match(G.mention, P.feat[i])                # compare mention gender/person vs candidate's own
  4. P.score[i] = combine_scalar(P.score[i], prior(candidate_i))     # prior-mix; INERT for pronouns (uniform prior)
5. w = select({P.score[i]})
6. resolved_v = emit(w, {P.mem[i]})                                  # EMIT reads P.mem, not P.addr — see §1.9
```

Executed by hand on "she found the ball" with registry `{john, mary}`:
step 3 should score `mary` high (FEMALE matches FEMALE) and `john` low;
step 5 selects `mary`; step 6 emits `mary`'s `rel:PLACE` reading — her
earlier place — which is exactly the value `_pronoun_context_step` wants
written, and exactly what `CorefHead` achieves (M53: 1.000/1.000 binding,
including every anti-recency case).

**But step 2 does not exist in the current membrane.** `EntityCandidateSet`
carries a `feature` field for the *mention* only; nothing computes a
per-candidate feature (see §1.3's fact 3). `CorefHead` cannot literally be
running this program — it has no `P.feat[i]` to compare against
`G.mention`. What it must be doing instead, given fact 2 (candidate atoms
carry zero gender geometry: `cosine(mary,sandra) = -0.043`, no closer than
`cosine(mary,john) = 0.089`), is **memorizing, in its shared MLP's first
layer, a lookup from each of the 6 fixed atom identities to how it should
respond to each of `G.mention`'s few gender/person settings** — a
closed-world classifier baked into weights, not a geometric feature-match.
It works, perfectly, because the name pool is fixed and tiny (6 names,
seen thousands of times in training) — but it is not the generalizable
op the algebra wants to name, and it would not transfer to a 7th name
never seen in training. **This is the headline finding of §1**: the op
algebra, as specified by the M56 charter, cannot express the pronoun
program cleanly with today's membrane plumbing. The fix is small and
concrete — extend `EntityCandidateSet`/the resolver contract with a
genuine per-candidate feature register, computed exactly like the mention
one is (`mention_feature_vector(candidate.key)` — candidate keys are
already surface-name strings, `membrane.py:76`, so this is a data-flow
change, not a new deterministic-feature design) — and it is a
**prerequisite for Track C**, not an optional nicety: without it, any
"feature_match" op in the inventory is vacuous, and a chained architecture
that routes through a vacuous op has just re-hidden the memorization
inside a different module.

**Sense program** (reproduces `SenseHead`, mechanism from `resolver.py`
`SenseHead.forward` + `clause_reactor._collapse`'s sense branch and its
M54b addendum):

```
1. G.ctx = mem_query(hom_addr, rel:SENSE)                     # component 1: this step's own address, ~0 (first write)
2. G.ctx = combine(G.ctx, mem_query(subj_addr, rel:PLACE))    # component 2: M54b's decisive addition — the disambiguator
3. G.ctx = combine(G.ctx, ground(context_word))               # component 3: same-clause other-role token (perception-side literal, not a memory op)
for each candidate sense i:
  4. P.tmp[i]   = interact(s_i, G.ctx)                         # elementwise product — the M54c-proven missing prior
  5. P.score[i] = score(s_i, G.ctx, P.tmp[i])                  # the learned MLP reducer (3d -> hidden -> 1)
6. w = select({P.score[i]})
7. resolved_v = emit(w, {s_i})                                 # EMIT reads the candidate vectors THEMSELVES, not P.mem — different register than the pronoun program's step 6!
```

This one type-checks cleanly against the inventory with no missing op —
**but note what it deliberately skips**: `SenseHead`'s own docstring says
it "deliberately ignores `cand_feature`, `cand_prior`, and `state`," so
step 6's `prior_mix` (candidate metadata, no register needed) is **not**
part of Track A's actual sense program; it belongs to a different,
simpler program — the MFS-floor baseline, `w = select({prior(s_i)})`,
skipping steps 1-5 entirely, no memory touched at all. The op algebra
happily expresses *both* programs (A's real one and the floor baseline)
with the same 8-op inventory, which is a good sign for the formalism: it
can represent a spectrum from "ignore memory, just rank by frequency" to
"consult memory, interact, then rank" using the same primitives, and the
66-point floor→ceiling gap A closes (RESEARCH_NOTES M54b) is exactly the
gap between these two programs.

### 1.8 Headline finding, restated plainly

The op algebra **cannot** express Track A's actual pronoun mechanism
faithfully as written, because the membrane doesn't expose a
per-candidate feature and the candidate identity vectors carry no
semantic content to substitute for one (measured: near-zero, ungeometric
cosine similarities between same-gender name atoms). CorefHead's perfect
score is a **closed-set memorization artifact of a 6-name curriculum**,
not evidence that a `feature_match` op already works. Any Track C
prototype MUST add the per-candidate feature register before claiming to
"give the operator explicitly" for coref — otherwise Track C would be
routing through the same memorization CorefHead already does, just with
extra indirection, and the promised legibility ("a human can read the
learned program," §5) would be reading a program whose `feature_match`
step is a no-op standing in for a lookup table.

### 1.9 Second finding: EMIT's source register is hardcoded, not routed

`_collapse`'s two branches differ in exactly one place beyond op
identity: the entity branch emits `sum_i w_i * P.mem[i]` (memory content),
the sense branch emits `sum_i w_i * s_i` (the candidate vector itself).
Today this is a Python `if`/branch choice by the caller, not something
either resolver decides. For genuine chaining (the user's brake: "single-
step routing over op outputs is just B with named features"), **which
register EMIT reads from must become part of the learned program**, or
Track C is still just two hardcoded programs with a shared vocabulary,
not a routed instruction sequence. This is a second, smaller, and
cheaper-to-fix prerequisite than §1.8 (no new data needed — `emit` just
needs to take a register NAME as one of its routed arguments instead of
being baked into the branch).

---

## 2. The graveyard survey

Every entry below is a real system that tried some version of learned
chaining/composition over discrete or quasi-discrete operations, and
either stalled or was later shown to be brittle. For each: the mechanism,
why it failed or stalled, and what specifically differs in our setting.

| system | mechanism | why it failed / stalled | what our setting changes |
|---|---|---|---|
| **NPI** (Reed & de Freitas, 2015) | Controller LSTM + program-embedding table + environment-specific push/pop over a call stack; trained on **full execution traces** including which subprogram is called at each step (addition, sorting, 3D rotation canonicalization) | Demonstrated compositional reuse of learned sub-programs, but needed hand-authored FULL traces per task (expensive to scale beyond toy domains); never showed convincing zero-shot transfer to a genuinely new task type, only recomposition within a hand-designed program hierarchy | We don't need full recursive traces for the NEW task (§4's transfer test explicitly withholds them) — harder in that one respect. But our domain is far smaller (≤6 ops, chain length ≤6, vs. NPI's arbitrary-depth call stack) and perception is deterministic (no vision noise NPI's grid-world tasks didn't have either, but our candidate sets are pre-enumerated, unlike NPI's raw pixel/scalar inputs) |
| **DNC** (Graves et al., 2016) | Controller + external memory with learned content/temporal addressing (usage tracking, temporal link matrix), fully soft, trained end-to-end on graph traversal / bAbI / block puzzles | Soft attention over an open, unconstrained memory blurs over many steps (interference, drift); later work (e.g. Csordás et al. 2021) found published results fragile/hard to reproduce; the addressing subsystem became a huge opaque parametric blob — no legibility, contradicting the promise of "read off what it learned" | Our memory has the SAME interference risk in principle (`entity_memory.query`'s own docstring: exact only for orthonormal keys). But we never ask the network to LEARN where to look — the candidate set is a tiny, closed, perception-enumerated list (a handful of names/senses), so there is no open-ended addressing problem to blur; DNC's core failure mode (learned addressing over unconstrained memory) doesn't apply because addressing is fixed math (`mem_query` on a given key), not learned |
| **Neural Module Networks** (Andreas et al., 2016) | Question parsed into a layout of small reusable modules (attend/combine/describe), assembled per-example, jointly trained | Needed an external parser/layout predictor; end-to-end layout learning (N2NMN) required high-variance REINFORCE; later diagnostics (CLOSURE, CLEVR-CoGenT-style) showed weak module specialization and poor generalization to unseen layouts | Our "layout" is a single collapse decision or a fixed short chain, not an open compositional parse tree; crucially we HAVE gold layouts (traces) for the two solved tasks from curriculum meta — NMN's layout predictor had to be learned or externally parsed over much higher-dimensional linguistic structure than our fixed ~6-op vocabulary over a few candidate slots |
| **TerpreT** (Gaunt et al., 2016) | DSL + head-to-head comparison of gradient-descent program induction (Gumbel-softmax relaxation) vs. discrete search/SMT solvers on IO-example-only program synthesis | **The TerpreT problem**: gradient descent on continuous relaxations of discrete program space reliably underperformed discrete search on the SAME benchmarks — soft relaxation of control flow got stuck in bad optima that solvers found instantly. Became the standard cautionary tale against differentiable program induction from IO pairs alone | We are not searching program TEXT from IO pairs alone — Stage 1 (§4) is DENSE, step-level trace supervision (closer to NPI's regime, which DID work with gradient descent, than to TerpreT's sparse-IO regime, which didn't). Our discrete space per step is tiny (≤8 ops × a handful of register bindings) vs. TerpreT's combinatorial program spaces (loops, branches, arbitrary length). TerpreT's warning is still our sharpest KILL signal (§5): if trace-WEANED soft routing can't sharpen to something legible, that IS the TerpreT failure recurring, and the honest fallback is discrete search over our (small, enumerable) op space, not more gradient pressure |
| **Routing networks / soft MoE collapse** (Rosenbaum et al., 2018; Shazeer et al., 2017's load-balancing losses exist precisely because of this) | A router (REINFORCE or soft/Gumbel) picks which expert/module handles each input, trained jointly with the experts | Rich-get-richer dynamics: a marginally-better expert gets marginally more gradient, monopolizes routing, and specialization collapses; mitigations (load-balancing losses, capacity limits, noise) are real but never fully eliminate the risk — and collapse gets WORSE with fewer distinct task types, which describes our setting | We have exactly 2-3 known collapse KINDS (entity/sense/parse), each already tagged at the membrane boundary by candidate-set provenance — the router can be conditioned on (or even partly gated by) that cheap deterministic signal, unlike open MoE routing over unlabeled inputs. Still a live risk (§6) that needs an explicit anti-collapse term, not assumed away by "we have few tasks" |
| **Neural GPU** (Kaiser & Sutskever, 2015) | Stacked convolutional GRU emulating parallel Turing-machine-style computation; trained on short arithmetic sequences (binary multiplication), tested on longer ones | Notoriously unstable training (needed gradient noise, length curricula, parameter dropout); LENGTH generalization was inconsistent across seeds with no reliable predictor of which seeds would generalize; follow-up analysis found the learned "algorithm" was brittle, not the true one | We are not asking for length/scale extrapolation — chain length is fixed and small by design (§1.6). Our generalization axis is TASK compositionality (new collapse type), a different and, given our closed small op set, more constrained ask than emulate-arbitrary-precision-arithmetic |
| **Stack RNNs / differentiable stacks** (Joulin & Mikolov, 2015; Grefenstette et al., 2015) | RNN augmented with a differentiable stack/queue, push/pop weighted by a soft controller gate, learning bracket-matching/counting/copying | Soft push/pop blurs stack CONTENTS whenever the controller is uncertain — the literal "soft chains blur" problem our own charter names; never scaled past synthetic formal-language benchmarks | No unbounded structure here — the register file is FIXED size and per-candidate, not a growing/shrinking stack; blur risk is confined to `select`'s mixture weights at a few fixed decision points, bounded by the small chain-length budget, not accumulated over an open-ended push/pop history |

**The one thing almost none of the graveyard had, that we do**:
supervised execution traces for the tasks we're starting from. NPI had
traces but paid a heavy authoring cost per new task and every task needed
its own trace design; TerpreT, DNC, NMN's end-to-end variant, and the
routing-network literature all worked from sparse (IO-only or reward-only)
supervision, which is exactly the regime that starved credit assignment
and caused the graveyard outcomes. We already own gold bindings in every
curriculum's `meta` (`gold_antecedent`, `gold_sense`, MFS priors) — the
traces in §1.7 above are DERIVED, not authored, from data the codebase
already logs for scoring. That is the single best argument for optimism
in this survey, and §4 is built around spending it correctly.

---

## 3. Why A and B were each close (data-driven)

Walking `RESEARCH_NOTES.md` M53→M54c op-by-op:

**Pronouns (M53, M54c): B-wide == A exactly, once capacity-matched.**
`B original` (2,841 params) scored 0.807 task / 0.550 pronoun-task /
0.963 binding — badly behind A's 0.913/1.000. `B-wide` (7,248 params,
capacity-matched to A's combined CorefHead+SenseHead) scored **0.913
task / 1.000 pronoun-task / 1.000 binding — bit-for-bit A's number.**
`B-nostate-wide` (state input removed) also hit 1.000/1.000. Given §1.7's
gold program, this is legible: pronoun resolution needs exactly
`{mem_query, feature_match (in whatever memorized form is available),
prior (inert), score, select, emit}` — a plain concatenation MLP CAN
represent that function once it has enough hidden width to memorize the
6-name lookup table §1.8 identified. **Model size, full stop** — the M53
"state entanglement" story is retired by M54c's own text. No missing
operator on the pronoun side; B was starved, not missing a primitive.

**Senses (M54b, M54c): size no, state-removal no, distillation
most-of-the-way.** Reference: A scores 0.863 task / 0.680 binding overall
/ 0.629 flipped (MFS floor 0.700/0.483, gold ceiling 0.953/1.000 —
A closes 66% of the floor-to-ceiling gap). Every capacity-matched B
variant (`B-wide`, `B-nostate`, `B-nostate-wide`) plateaus around
0.727-0.737 task / 0.45-0.53 binding — **no better than B at its
original, starved size**, ruling out "B just needed more width" for this
capability specifically (the pronoun explanation doesn't transfer).
`B-distilled` (trained to imitate A's own logits) reaches 0.787 task /
0.667 binding overall / 0.486 flipped — most of the way to A, but not
all the way, and only by being handed A's answer directly. **B can
roughly REPRESENT the sense solution (distillation finds it) but cannot
FIND it from task+aux reward alone (plain gradient descent on the shared
scorer doesn't discover it).** Per §1.7's gold sense program, the op A's
`SenseHead` has and `SharedScorer` doesn't is `interact` — the
`candidate * context` elementwise product is baked into `SenseHead`'s
input by construction; `SharedScorer` must invent that same nonlinear
cross-term from a flat `[candidate; context]` concatenation via its own
learned weights, and empirically doesn't (0.727-0.737 stuck near the
floor's 0.700, regardless of width or state).

**The minimal op set that explains the entire record**:

| capability | ops actually needed (from §1.7's gold programs) | does a plain concat-MLP find them from task reward alone? |
|---|---|---|
| pronoun (matching-by-lookup) | `mem_query`, (memorized) `feature_match`, inert `prior`, `score`, `select`, `emit(mem)` | **yes**, given enough width (B-wide == A) |
| sense (context-conditioned disambiguation) | `mem_query`×2, `combine`, `interact`, `score`, `select`, `emit(candidate)` | **no** (every width/state variant plateaus near the MFS floor; only distillation — being handed the answer — closes most of the gap) |

This is the precise, data-grounded form of the Track C mandate
(`RESEARCH_NOTES.md` M54c): shared mechanisms scale fine on tasks that
reduce to lookup-and-match; they fail specifically where a multiplicative
interaction is required and isn't handed to the network as an inductive
bias. Distillation "almost working" is the concrete evidence that **the
routing/parameter SPACE containing the answer exists** — B-distilled's
gap to A (0.787 vs 0.863, 0.667/0.486 vs 0.680/0.629) is an
optimization/discovery gap, not a representational one. That gap is
exactly what naming `interact` as an explicit, always-available op (so
it never has to be reinvented from a flat concatenation) is meant to
close — the entire justification for Track C's existence, not an
aspiration layered on top of it.

---

## 4. Training design — the most out of the dataset, without workarounds

### 4.1 What "the dataset" already contains

Every curriculum episode's `meta` carries the gold binding it was
generated from — `gold_antecedent`/`gold_place` (pronoun),
`gold_sense`/MFS rank (sense) — because that's how the accuracy/binding
metrics in every gate table are computed. The gold PROGRAMS in §1.7 are
mechanical rewrites of that same meta into a step sequence: given
`gold_antecedent`, the trace's `select` step is a one-hot at that
candidate's index; given `gold_sense`, likewise. **No new data collection,
no new curriculum, no new labeling effort** — this is data already paid
for and already used for scoring, now also used for supervision. That is
precisely why it is not a workaround (§4.4 makes this contrast explicit).

### 4.2 Stage 1 — trace-supervised

Train the op-execution modules (`mem_query`/`interact`/`combine` are
parameter-free; `feature_match` and `score` carry the only weights) plus
the router, with a per-step cross-entropy loss against the gold
`(op, arg-binding)` sequence from §1.7, **plus** an intermediate-value
loss (cosine or MSE) checking that each register after each op holds the
value hand-execution predicts (this is what makes it a genuine trace,
not just a final-answer label in disguise — NPI's own design). Router
architecture: small (a step-embedding + `G.state` → softmax over the ~8
ops × plausible register bindings) — this is imitation learning over a
tiny discrete action space, the regime gradient descent is good at
(§2's TerpreT contrast).

Prerequisite from §1.8/§1.9: the per-candidate feature register and the
routed `emit` source must exist before Stage 1 can even be posed
correctly — otherwise the "gold trace" for pronouns has a step
(`candidate_feature`) with nothing to supervise.

### 4.3 Stage 2 — trace-weaning

Anneal the per-step trace loss to zero over training (e.g. linear decay
over a fixed fraction of epochs), leaving only the existing task loss +
aux binding loss (the same losses A and B were scored on throughout
M53-M54c) driving the router and op weights. **The gate for Stage 2 is
matching A's standing numbers** (§5) with the trace loss OFF — if
performance craters when the imitation signal is removed, the router
found a program that only works with hand-holding, which is a distinct
and diagnostic failure mode from never finding one at all (worth
recording as such, not conflating with a flat "C failed").

### 4.4 Stage 3 — the decisive compositional transfer test

Train ops + routing on pronoun + sense traces only (Stages 1-2 above).
Then present a **new collapse type with no new ops and few/zero new
traces**: M55's parse-hypothesis collapse (`dev/RESOLVER_BUILD_PLAN.md`
Phase 4 — garden-path sentences, parser emits top-K structural
hypotheses with margins, the mind picks). This candidate set slots into
the SAME typed registers with no new op required:

```
candidates    = parse hypotheses (each a clause-stream reading)         -- Vec-typed, like sense vectors
P.addr[i]     = the hypothesis's own reading vector
prior(i)      = the PARSER'S OWN STRUCTURAL SCORE                       -- a real graded prior, unlike pronoun's uniform one, LIKE sense's MFS rank
G.ctx         = mem_query(relevant entity, relevant relation) accumulated via combine, exactly as in the sense program
P.tmp[i]      = interact(P.addr[i], G.ctx)                              -- does this hypothesis cohere with what memory already holds?
P.score[i]    = score(P.addr[i], G.ctx, P.tmp[i])
w             = select({P.score[i]})
resolved      = emit(w, {P.addr[i]})
```

This is, almost verbatim, the sense program with `s_i` relabeled as
"hypothesis reading" and the disambiguating fact swapped for whichever
memory fact the garden-path sentence needs — which is exactly the
recombination test the M56 charter names as the real bar ("solve a NEW
collapse type by recombining ops with few/no new traces... that transfer
test, not benchmark parity, is what would justify buy-in"). Two
sub-conditions, precisely specified:

- **Zero-trace condition**: freeze the trained op weights and router from
  Stage 2; run on the garden-path battery with NO parse-collapse traces
  at all. Success bar: beat the parser's own top-1-by-structural-score
  baseline (the "no learning" reference — meaningful because garden-path
  sentences are BY CONSTRUCTION the ones where top-1 structural score is
  often wrong) by a margin at least as large, proportionally, as A's own
  66%-of-floor-to-ceiling closure on senses (M54b).
- **Few-trace condition**: fine-tune with ≤10% of M55 episodes
  trace-supervised (a small nudge, not a new Stage 1). Success bar: come
  within noise of Track A's own M55 rank head, once that number exists
  (M55 has not been run yet — this spike predates it per
  `dev/NEXT_ARC_PLAN.md`'s sequencing, so the threshold is stated
  relative to a number that will exist before Track C's own gate run,
  not fabricated here).

This experiment **could literally be M55**, run twice — once for Track A
(as already planned) and once as Track C's transfer test — rather than a
separate build, which keeps the "no downstream work blocks on C" rule
(`NEXT_ARC_PLAN.md`) intact.

### 4.5 What counts as a workaround (explicitly, to avoid)

- **A permanent distillation crutch.** B-distilled (M54c) is diagnostic,
  not a destination — if Track C's only way to reach A's numbers is
  training against A's frozen logits forever (not as a Stage-1-style
  bootstrap that gets annealed away), that is B-distilled wearing a new
  name, not a chained op-learner, and should be scored as a KILL-adjacent
  finding (§5), not a win.
- **Per-task op additions.** Adding a bespoke op the moment a new
  benchmark needs one defeats the entire "hard-code the ABILITY, not the
  cases" framing (`RESEARCH_NOTES.md` M54c's own words). The op inventory
  in §1.3 is meant to be closed for the duration of the spike's gate;
  if M55 genuinely needs a 9th op, that is itself a finding to report
  (§6), not something to quietly patch in before scoring.
- **Benchmark-tuned op sets.** Choosing `interact` specifically because
  it's known (from M54c) to fix the sense gap, then declaring victory
  when it fixes the sense gap, is circular. The transfer test (§4.4) is
  the actual test of whether the op set generalizes BEYOND the two tasks
  it was reverse-engineered from — it is load-bearing precisely because
  §1.7-3 derived the op set from hindsight.

---

## 5. Go/kill gates

Defined here, before any prototype exists, per the charter.

### GO requires ALL of:

| gate | threshold | source |
|---|---|---|
| pronoun task (trace-weaned, Stage 2) | ≥ 0.90 (A: 0.913) | M53/M54c |
| pronoun binding overall / anti-recency | ≥ 0.98 / ≥ 0.98 (A: 1.000/1.000) | M53 |
| sense task (trace-weaned, Stage 2) | ≥ 0.84 (A: 0.863) | M54b |
| sense binding overall / flipped | ≥ 0.65 / ≥ 0.60 (A: 0.680/0.629) | M54b |
| transfer test, zero-trace | beats parser top-1 baseline on the garden-path battery by a margin proportionally comparable to A's own 66% floor-ceiling closure (§4.4) | derived from M54b |
| transfer test, few-trace (≤10%) | within noise of Track A's own M55 result (measured when M55 runs) | to be set concretely once M55 lands |
| routing legibility | a human, reading the router's per-episode op/register choices, can map ≥90% of eval episodes onto the §1.7 gold-program shape (allowing register-binding variation, not op-sequence substitution) | this spike's own §1.7 notation |
| param budget | total ops + router ≤ ~10k params (same order as A's combined 7,194 and B-wide's 7,248 — the point is fewer seams, not a bigger model) | M53/M54c param columns |

### KILL if ANY of:

- Trace-weaned C (Stage 2, trace loss at zero) fails to match A within
  noise on EITHER standing benchmark after a compute budget matched to
  A's own gate runs (1500 eps, 80 epochs) at a capacity-matched param
  count — i.e., C needs MORE resources than A to tie, not just different
  ones.
- Router entropy does not measurably sharpen over Stage 2 training
  (stays diffuse/near-uniform over op choices) — the "soft chains blur"
  graveyard outcome (§2) materializing as predicted.
- Compositional transfer (zero-trace condition) is statistically
  indistinguishable from a naive baseline (top-1 structural score) even
  after weight-tying and full Stage 1-2 training — i.e., the ops
  memorized two tasks but didn't compose.
- The only way to reach GO thresholds is a permanent distillation crutch
  (§4.5) — recorded as a distinct, informative KILL, not folded into "C
  failed" generically (it would mean the REPRESENTATION is there but
  routing can't be trained to find it without a teacher, which is worth
  knowing precisely).

On KILL: Track A remains the standing configuration (already true per
`NEXT_ARC_PLAN.md` — "nothing downstream blocks on C"); the op
formalization in §1 is retained as documentation of what a future
attempt (with, e.g., discrete search over the small op space instead of
gradient-trained routing, per the TerpreT lesson) would need to beat.

### Implementation sketch (cost of a GO decision)

Rough size, scoped to what §1.8/§1.9's prerequisites plus Stages 1-3
require — comparable to one M53b+M54-sized build (each was a half-day to
one-day agent task per `RESOLVER_BUILD_PLAN.md`):

- `membrane.py`: add a per-candidate feature register to
  `EntityCandidateSet` (§1.8) — small, `mention_feature_vector` reused
  as-is on `candidate.key`.
- `resolver.py`: new `ops.py`-style module or additions — `interact`,
  `combine` (both parameter-free, trivial), a shared `score` head
  (reuse `SharedScorer`'s own MLP shape), a router (step-embedding +
  `G.state` → op/register softmax, small — a few thousand params at
  most, reusing `shared_scorer_for_budget`-style capacity matching).
- `clause_reactor.py`: `_collapse` loses its two hardcoded branches,
  replaced by one executor loop over the register model (§1.4) running
  ≤`K_max` steps; `emit`'s source register becomes routed (§1.9) rather
  than branch-selected. This is the highest-risk, highest-value edit —
  it's precisely where "chaining is the crux" lives in code.
- New: trace-generation code reading `ep.meta` into the §1.7 gold
  `(op, register)` sequences, plus the Stage-1 imitation loss and Stage-2
  annealing schedule.
- Estimated total: 400-700 changed/new lines across 4-5 files, plus
  tests mirroring the existing byte-identity-regression discipline
  (`test_no_resolver_byte_identity_regression`-style) so the no-resolver
  and Track-A/B paths stay provably untouched.

---

## 6. Risks & open questions, ranked

1. **§1.8's gap might not be the only one.** The pronoun/sense audit was
   thorough, but M55's parse-hypothesis collapse hasn't been built yet —
   it may reveal a THIRD missing op or type (e.g. a hypothesis "reading"
   may need to bind multiple (entity, relation) pairs simultaneously,
   not one). *Resolving experiment*: attempt to write M55's gold program
   in this notation BEFORE building anything, the same acid test §1.7
   applied prospectively — if it doesn't type-check cleanly, that's a
   finding to fold into this doc before Stage 3, not a discovery mid-build.
2. **Router collapse (§2's routing-networks entry) is a real, not
   hypothetical, risk given only 2-3 task kinds.** *Resolving experiment*:
   Stage 1 should log per-op usage counts across episode kinds from the
   start; if any op goes systematically unused where the gold trace says
   it's needed, that's collapse, not efficiency, and needs an entropy/
   load-balancing term before Stage 2 begins, not after it's observed to
   fail.
3. **The TerpreT problem could recur specifically in Stage 2→3
   (trace-weaned generalization).** Stage 1 imitation is the easy part
   (dense supervision, small discrete space — §2's contrast with NPI);
   the risk concentrates exactly where TerpreT's did, once the dense
   signal is removed. *Resolving experiment*: Stage 2's own gate (§4.3)
   — if task+aux-only training measurably degrades the router's
   sharpness (entropy rising as trace loss anneals out), that IS the
   TerpreT dynamic, caught early rather than discovered at the transfer
   test.
4. **The per-candidate feature fix (§1.8) might not fully explain
   CorefHead's success** — there could be a second signal (e.g. discourse
   order/recency correlated with something else entirely) doing
   uncredited work. *Resolving experiment*: an ablation on CURRENT
   CorefHead — zero out `cand_prior` and shuffle `cand_entity` identities
   across a HELD-OUT 7th name at eval time; if accuracy collapses to
   chance, memorization is confirmed as the sole mechanism (expected,
   given the cosine numbers in §1.1); if it doesn't, there's a
   confound this doc hasn't found yet.
5. **The M55 threshold in the GO table (§5) is explicitly a forward
   reference** — "within noise of Track A's own M55 result" cannot be
   checked today because M55 hasn't run. *Resolving experiment*: none
   needed beyond running M55 itself, per the existing arc plan; flagged
   here so it isn't mistaken for a filled-in number.

---

## 7. Verdict recommendation

**More-research-needed, with a scoped, cheap next step — not proceed-to-
prototype, and not kill.**

Reasoning: the spike found the formalism is *almost* right, but not
free — §1.8's gap is not a footnote, it's a precondition. Building a
routed executor on top of today's candidate-feature plumbing would bake
CorefHead's memorization artifact into Track C's `feature_match` op,
which would make the eventual legibility claim ("a human can read the
learned program") hollow for the pronoun case specifically — it would
read as a clean 6-step program while secretly depending on a closed
6-name lookup table no different from what CorefHead already does. That
is not a reason to kill the idea (§3's data is genuinely encouraging:
`interact` demonstrably explains the entire sense-side gap, and
distillation proves the representational space contains the answer) —
it's a reason to fix the membrane gap FIRST, cheaply, and re-run the
existing A-vs-B pronoun gate with a real per-candidate feature before
committing engineering time to the router/executor build in §5's sketch.

Concretely, before any Track C prototype: (1) add the per-candidate
feature register (§1.8) and re-verify `CorefHead` still hits 1.000/1.000
with it available (it should — this only ADDS a discriminative signal;
if binding degrades, that's itself a finding), and separately confirm
via the §6.4 held-out-name ablation that pre-fix `CorefHead` really is
pure memorization; (2) attempt the M55 gold-program acid test
prospectively (§6.1) so Stage 3's transfer target is verified
expressible before Stage 1 code is written. Both are cheap (hours, not
days — well inside this project's stated planning style) and would
either firm up §5's gates with real numbers or surface a second
formalization gap before the expensive part (the routed executor and
its training stages) is built. Track A remains the working configuration
throughout, per the existing plan; this recommendation changes nothing
about M55 or M57's sequencing — it only adds two cheap, sequenced
checks ahead of the prototype-commitment decision that the charter
already deferred.
