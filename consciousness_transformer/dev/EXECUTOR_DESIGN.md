# The learned EXECUTOR: Track C made concrete

Decision doc, not a survey (2026-08-30). No code touched. For the lead's review before any executor code exists. Read order behind this doc: `dev/TRACK_C_DESIGN.md` (the M56 op-algebra spike this concretizes), `dev/OP_INVENTORY.md` (the built/proven/planned op table reused verbatim), `dev/MIND_INTERFACE.md` (invariants, dials), `dev/LTM_DESIGN_BRIEF.md` §5, `src/nsm_ct/clause_reactor.py` (`ClauseReactor.forward`/`_collapse`), `src/nsm_ct/resolver.py`.

## 0. Framing

Today's per-clause loop is **one fixed program**: `forward` runs `query -> _collapse -> GRU tick -> gate/overwrite/negate -> write -> respond`, one op each, same order every clause step (`clause_reactor.py:3007-3099`). The only "choice" is *which candidate branch is present* — a Python `if`, not a learned decision — and TRACK_C_DESIGN §1.9 already named the concrete place this hides a real routing decision ("EMIT's source register is hardcoded, not routed"). The executor makes op-selection and argument-selection themselves learned and budgeted, so today's sequence becomes one point in a space of programs, not the only program that exists.

**Why now.** TRACK_C_DESIGN's verdict (§7) was "more research," conditioned on three prerequisites: §1.8 (a per-candidate `Feat` register), §1.9 (routed `EMIT`), §1.10 (a per-candidate `Addr` register for `mem_query`). All three are now built and validated — §1.8/§1.9 by M56b/M56c (held-out-name pronoun binding restored to 1.000 via `membrane.EntityCandidateSet.cand_features`), §1.10 by M55a (`query_candidates_per_addr`, `resolver.py:66-81`). M57 then proved the op inventory works at rich scale **and is necessary**: cheat (GRU state-carry, no explicit binding) scores 0.500 vs. normal 0.938 at 8 entities (RESEARCH_NOTES "M57 battery #3") — the strongest evidence yet that this project's discourse cannot be solved by hidden-state shortcuts alone, which is the whole argument for a program-selecting controller over a bigger GRU. Every M53-M57 curriculum's `meta` already carries the gold binding used to score it (TRACK_C_DESIGN §4.1) — §2a's traces are *derived*, not authored.

**What does not change.** MIND_INTERFACE.md's nine invariants hold unmodified: #1 (weights hold policy only — the executor learns *which op, which register*, never a fact), #5 (one resolver contract — the new `SCORE` op generalizes CorefHead/SenseHead/RankHead/SharedScorer, it does not add a fourth head), #6 (dials are explicit named scalars), #8 (compute is budgeted). Track A (today's fixed pipeline) remains the standing tournament baseline and deployed fallback regardless of outcome, per this project's kill-switch convention. No knowledge migrates into weights — the exact failure M56 caught (CorefHead's 1.000 was a six-name lookup table in weights) and §3's honesty machinery exists to keep catching it here.

**The user's framing, verbatim, is this design's premise**: "perception is deterministic and grounded (USVS), structure is explicit (clauses, roles, candidate sets), memory ops are fixed and exact — so the executor learns SHORT programs over clean typed objects, unlike NPI/DNC/NMN which learned perception and programming at once." Transformer-style *training* on text at scale (M58) is in scope; transformer *architecture* inside this loop is not — ROADMAP_LONG_TERM.md's closed door stands: "attention only as a single readable select-over-explicit-objects op." Every attention mechanism below (§2) is exactly that: a pointer over a small, named, typed register file, never over an unbounded token sequence.

## 1. The machine

### 1.1 Registers

Twelve global registers, typed per TRACK_C_DESIGN §1.2 (`Addr`/`Vec`/`Dist`/`Scalar`/`Mem`), plus per-candidate shadow registers sized to `C` (M57's richest curriculum used up to 8 entities):

| register | type | count | pre-loaded per clause from |
|---|---|---|---|
| `A_e`, `A_r` | Addr | 2 | `batch.entity[:,t]`, `batch.relation[:,t]` |
| `A_w` | Addr | 1 (scratch) | defaults to `A_e`; only `EMIT`/`ADDR_REDIRECT` write it |
| `V_v` | Vec | 1 | `batch.value[:,t]` (empty on question steps) |
| `V_read`, `V_ctx` | Vec | 2 (scratch) | last `QUERY`/`REREAD` result; `COMBINE` accumulator |
| `S_gate`,`S_owr`,`S_neg`,`S_tmp` | Scalar | 4 | unset until `GATE`/`OVERWRITE`/`NEGATE`/scratch write them |
| `D_w` | Dist | 1 | last `SELECT` result |
| `M_mem` | Mem handle | 1 | `mem_total(memory, ltm)` (`clause_reactor.py:3017`), read-only, additive STM+LTM |
| `P.addr/feat/prior/mem/tmp/score[i]` | mixed | 6×C | the active candidate set (entity/sense/hyp/LTM), if any |

`C` is pre-enumerated by perception (MIND_INTERFACE.md §1: "perception never guesses") — the executor scores and selects among candidates already handed over, never searches for them. This is the same "no open-ended addressing" restriction TRACK_C_DESIGN §1.4 committed to, now covering the whole clause loop — the thing that keeps this out of the DNC/stack-RNN graveyard (TRACK_C_DESIGN §2). The GRU controller state is **not** a register — it's the program counter `OpSelect` reads and `TICK` updates once per clause step (Decision 2, §6).

### 1.2 The op library

Every row below is already `proven`/`built-unproven`/`planned` in OP_INVENTORY.md's table — this design adds no new fixed math, only routing.

- **Memory**: `QUERY` (bilinear read), `QUERY_ENTITY` (inverse), `QUERY_CAND`/`QUERY_CAND_PER_ADDR` (per-candidate read), `WRITE` (gated bind), `ADDR_REDIRECT`, `REREAD` (M57c.2 post-collapse re-read), `ATTR_WRITE`, `MINT`, `CAND_FOR`/`INVERSE_QUERY` (attribute-match candidate gen — flagged "not yet the live batch-build path" in OP_INVENTORY §5), `PROVENANCE` (audit log).
- **Tiers** (LTM, `planned` — LTM_DESIGN_BRIEF §5): `RECALL` is not a separate op — it's `QUERY`/`QUERY_CAND` against `M_mem`, already the additive `mem_total(STM,LTM)` view; `LINK` (identity linking, reuses entity `SCORE`/`SELECT` with `P.from_ltm[i]` as an extra feature column); `CONSOLIDATE` (substate transition, end-of-passage); `PROMOTE` (tier-generic gated copy, `trust_ltm`/`trust_truth`).
- **Control**: `SCORE` (the one learned reducer — generalizes CorefHead/SenseHead/RankHead/SharedScorer via which registers it reads), `INTERACT`, `COMBINE`, `FEATURE_MATCH`, `PRIOR`, `COMPARE` (cosine, output stage), `SELECT` (softmax/argmax over `P.score[*]`, unchanged `_collapse_weights`), `EMIT` (weighted sum into a named destination — the op TRACK_C_DESIGN §1.9 flagged hardcoded, now routed), `TICK` (GRU update, fixed, not itself selectable — Decision 2), `GATE`/`OVERWRITE`/`NEGATE` (the three existing scalar heads, reframed as `SCORE` ops targeting `S_gate`/`S_owr`/`S_neg`), `RESPOND`/`RESPONSE`, `FORCE` (test-only, never reachable at inference), `HALT` (budget cutoff, §1.3).

A separate, unrelated `src/nsm_ct/mind/ops.py` VM (`PERCEIVE`/`RECALL`/`INFER`/`CONSOLIDATE`/`SUPERSEDE`/`RESPOND`/`HALT`) already exists — M1-M4 prior art over a different substrate (`MeaningGraph` nodes). LTM_DESIGN_BRIEF §0's instruction stands: reusable machinery, not this executor's format (Decision 4, §6).

### 1.3 The per-clause loop

```
op_id    = OpSelect(state, pool(registers))          # categorical, masked to type-legal ops
arg_regs = ArgSelect(op_id, state, registers)         # pointer/attention per argument slot
result   = execute(op_id, arg_regs)                   # fixed math or the SCORE head, never learned routing
registers[dest(op_id)] = result
if op_id == HALT or step_count == K_max: emit_and_advance()
else: step_count += 1; repeat
```

`K_max` is this design's concrete implementation of the **`patience`** dial (named, stubbed in MIND_INTERFACE.md/OP_INVENTORY §3: "not set — no threshold exists in code"). Default `K_max = 12`: §1.4's hardwired trace runs 9-13 ops depending on which candidate branch fires (branches never stack — disjoint per `_collapse`'s own docstring), so 12 is one slack step over the longest existing program, the same rule TRACK_C_DESIGN §1.6 used for `K_max=6` on the collapse-only chain. At budget exhaustion an implicit `SELECT`+`EMIT`+`TICK` fires regardless of what was asked for, mirroring §1.6's hard-cutoff design (no learned halting probability — sidesteps the instability that sank Neural GPU/stack RNNs, TRACK_C_DESIGN §2). The **`caution`** dial (also stubbed) is a natural `HALT` consumer (a low-margin `SELECT` could route to an abstaining `HALT`) but wiring it is no-regret follow-on, not required for GO (§3).

The answer is emitted unchanged: `RESPOND`/`RESPONSE` accumulate per-step logit + content vector, aggregated by softmax-weighted sum, then cosine-compared to option vectors (`clause_reactor.py:3101-3109`). This sits *outside* the op loop — scored once per episode, not once per op.

### 1.4 The bootstrap anchor: today's pipeline as one fixed trace

**An executor with this trace forced must reproduce Track A byte-for-byte (or within float tolerance)** on every existing regression suite. Worst case (candidate families are mutually exclusive per row/step, so a real trace uses at most one bracketed block):

```
QUERY(A_e,A_r) -> V_read                                                          # :3018
[entity]  QUERY_CAND(P.addr[*],A_r|evidence_r)->P.mem[*]                          # :2796
          INTERACT(P.mem[*],evidence_target)->P.score_extra[*]                    # :2811-13, optional
          SCORE(P.addr[*],P.mem[*],mention_feat,P.feat[*],P.prior[*],P.score_extra[*])->P.score[*]  # :2874
          SELECT(P.score[*])->D_w                                                 # :2877, FORCE may override :2890-95
          EMIT(D_w,P.mem[*])->V_v   OR   EMIT(D_w,P.addr[*])->A_w  (never both)   # :2910/2919/2925-26
[sense]   QUERY(A_subj,A_subj_r)->V_tmp; COMBINE(V_read,V_tmp,ctx_word)->V_ctx     # :2935-39
          INTERACT(P.addr[*],V_ctx)->P.tmp[*]; SCORE(P.addr[*],V_ctx,P.tmp[*])->P.score[*]  # :2941
          SELECT(P.score[*])->D_w; EMIT(D_w,P.addr[*])->V_v                       # :2944-46
[hyp]     QUERY_CAND_PER_ADDR(P.query_addr[*],P.query_r[*])->P.mem[*]             # :2954
          INTERACT(P.addr[*],P.mem[*])->P.tmp[*]
          SCORE(P.addr[*],P.mem[*],P.tmp[*],P.prior[*],state)->P.score[*]         # :2956
          SELECT(P.score[*])->D_w; EMIT(D_w,P.mem[*])->V_v                        # :2959-61
[if A_w set]  REREAD(A_w,A_r)->V_read                                             # :3039-40 (M57c.2)
[if inverse]  QUERY_ENTITY(A_r,V_v)->V_read                                       # :3050-53
TICK(A_e|A_w,A_r,V_v,p,c,V_read; state)->state                                    # GRU, :3056
GATE(state)->S_gate; OVERWRITE(state)->S_owr; NEGATE(state,V_v)->S_neg            # :3059-65
WRITE(M_mem,A_e|A_w,A_r,V_v,S_gate-S_neg,overwrite=S_owr)                         # :3066
PROVENANCE(...)                                                                    # :3067-85, if enabled
RESPOND(state)->rl; RESPONSE(state,V_read)->rv                                    # :3086-90
```

Every branch condition is exactly the "hardcoded, not routed" fact TRACK_C_DESIGN §1.9 flagged. Forcing the op/arg-selector to reproduce this trace, step for step, is a mechanical zero-ambiguity regression test — if it doesn't reproduce Track A exactly, the execute-layer (not routing) has a bug, and that's caught in Phase 1 (§5) before any training starts.

## 2. Learning

**Op selection**: a categorical head over the ~20-op vocabulary (§1.2), conditioned on `[state; pool(registers)]` (`pool` = fixed concat of named globals + mean/max over the active `P.*` bank, so input size doesn't grow with `C`). Masked to type-legal next ops given the last op's write and populated register types — TRACK_C_DESIGN §1.5's type table already prunes most of the space, so the head never learns illegal transitions from scratch, only which legal one to take.

**Argument selection**: for each slot the op needs, a small attention head scores every register of the required *type* (an `Addr` slot attends over `{A_e,A_r,A_w,P.addr[0..C-1]}`) and picks by softmax (train) / argmax (eval) — `_collapse_weights` generalized from "which candidate" to "which register," and exactly the "admissible attention" ROADMAP_LONG_TERM.md permits: select-over-explicit-objects, never over an unbounded sequence.

### 2a. Gold programs (trace-supervised targets)

Derived mechanically from existing curricula's `meta` + the code paths that already compute the right answer (TRACK_C_DESIGN §4.1, extended past collapse to the whole clause loop). `[*]` = per-candidate op applied to every `i` in the active set.

**1. Plain fact** ("mary went to the garden . where is mary ?" — no candidates):
```
QUERY(A_e,A_r)->V_read
TICK(A_e,A_r,V_v,p,c,V_read)->state
GATE(state)->S_gate; OVERWRITE(state)->S_owr; NEGATE(state,V_v)->S_neg
WRITE(M_mem,A_e,A_r,V_v,S_gate-S_neg,overwrite=S_owr)
RESPOND(state)->rl; RESPONSE(state,V_read)->rv
HALT
```

**2. Pronoun value-redirect** ("she found the ball" — TRACK_C_DESIGN §1.7, EMIT writes the VALUE):
```
QUERY(A_e,A_r)->V_read
QUERY_CAND(P.addr[*],A_r)->P.mem[*]
FEATURE_MATCH(mention_feat,P.feat[*])->P.score[*]; PRIOR(P.addr[*])-> combine into P.score[*]  (inert, uniform)
SELECT(P.score[*])->D_w
EMIT(D_w,P.mem[*])->V_v
TICK(A_e,A_r,V_v,p,c,V_read)->state
GATE/OVERWRITE/NEGATE(...)->S_gate,S_owr,S_neg
WRITE(M_mem,A_e,A_r,V_v,S_gate-S_neg,overwrite=S_owr)
RESPOND/RESPONSE(state,V_read)
HALT
```

**3. Write-back address-redirect** ("she is tall" — M57b, EMIT writes the ADDRESS):
```
QUERY(A_e,A_r)->V_read
QUERY_CAND(P.addr[*],evidence_r)->P.mem[*]; FEATURE_MATCH/SCORE(...)->P.score[*]
SELECT(P.score[*])->D_w
EMIT(D_w,P.addr[*])->A_w
REREAD(A_w,A_r)->V_read
TICK(A_w,A_r,V_v,p,c,V_read)->state
GATE/OVERWRITE/NEGATE(...)->S_gate,S_owr,S_neg
WRITE(M_mem,A_w,A_r,V_v,S_gate-S_neg,overwrite=S_owr)
PROVENANCE(A_w,A_r,V_v,S_gate,S_owr,S_neg,redirected=True)
HALT
```

**4. Definite-description read** ("what is the doctor like ?" — `candidates_for(kind=doctor)` + M57c.2 read-side redirect, question step, no write):
```
CAND_FOR(M_mem,kind=doctor)->P.addr[*],P.prior[*]
QUERY_CAND(P.addr[*],evidence_r)->P.mem[*]; INTERACT(P.mem[*],evidence_target)->P.score_extra[*]
SCORE(P.addr[*],P.mem[*],P.score_extra[*],state)->P.score[*]
SELECT(P.score[*])->D_w
EMIT(D_w,P.addr[*])->A_w
REREAD(A_w,A_r)->V_read
TICK(A_w,A_r,-,p,c,V_read)->state    # V_v empty: question step
RESPOND/RESPONSE(state,V_read)
HALT
```

**5. Inverse query** ("who is tall ?" — M57c.2 entity-axis unbind):
```
QUERY_ENTITY(A_r,V_v)->V_read
TICK(A_e,A_r,-,p,c,V_read)->state
RESPOND/RESPONSE(state,V_read)
HALT
```

**6. Cross-passage recall+link** (LTM, `planned` — LTM_DESIGN_BRIEF §5; passage 2's "mary" must LINK to passage 1's `mary#1`):
```
CAND_FOR(STM,name=mary)->P.addr_stm[*]; CAND_FOR(LTM,name=mary)->P.addr_ltm[*]; tag P.from_ltm[*]=1
[union into P.addr[*],P.from_ltm[*],P.feat[*]]
QUERY_CAND(P.addr[*],evidence_r)->P.mem[*]           # additive: mem_total already sums STM+LTM
INTERACT(P.mem[*],evidence_target)->P.score_extra[*]
SCORE(P.addr[*],P.mem[*],P.feat[*],P.from_ltm[*],P.score_extra[*],state)->P.score[*]   # LINK op
SELECT(P.score[*])->D_w        # below link_threshold -> selects "new instance"
EMIT(D_w,P.addr[*])->A_w
REREAD(A_w,A_r)->V_read
TICK(A_w,A_r,-,p,c,V_read)->state
RESPOND/RESPONSE(state,V_read)
HALT
```

**Training modes**: **(a) trace-supervised** — teacher-forced (op, arg-register) sequences from above, per-step cross-entropy on both choices, plus an intermediate register-value loss (cosine/MSE against hand-execution) — TRACK_C_DESIGN §4.2 unchanged, what makes it a genuine trace, not a final-answer label in disguise. **(b) straight-through, traces annealed** — `OpSelect`/`ArgSelect` go hard (argmax forward, soft gradient backward) while the trace loss decays to zero (TRACK_C_DESIGN §4.3's Stage 2, extended: selection itself goes hard here, not just collapse weights). Gate: match Stage-(a) numbers with trace loss OFF. **(c) task-loss-only (the cliff)** — zero trace supervision, the TerpreT regime TRACK_C_DESIGN §2 names as the sharpest failure mode. Measured, never assumed: run it and report the number, whatever it is.

**Credit assignment**: at each of ≤12 steps the discrete action space is a handful of ops × a handful of type-legal register bindings — TRACK_C_DESIGN §1.5's type table prunes most of the space before learning starts, the same "tiny discrete space, dense supervision" regime that made NPI's imitation tractable where TerpreT's sparse IO-only search wasn't. What would show blur: (i) op-usage histograms flatten toward uniform across episode kinds (router collapse, TRACK_C_DESIGN §6 risk 2); (ii) intermediate register-value loss stays high while task loss drops (task loss satisfied through a path bypassing nominal register contents — TerpreT recurring); (iii) the forced-WRONG-trace arm (§3) fails to crater — the chosen sequence doesn't determine the outcome, the same diagnostic M57b/M57's forced-wrong arms already use (1.000→0.000 pattern).

## 3. Honesty machinery (mandatory)

- **Forced-trace (gold)**: force §2a's exact trace, train and eval. Must reproduce §1.4's bootstrap numbers.
- **Forced-WRONG-trace**: force a wrong op or wrong arg-register at one decisive step. Must **crater** — the standing proof pattern (M57b/M57: 1.000→0.000, 0.938→0.812).
- **Cheat (no executor)**: Track A's hardwired pipeline, run as-is — the tournament floor, not a disabled-mechanism probe (§4).
- **No-trace eval**: trained executor (Stage b/c), zero trace supervision, zero forcing at eval — must hold at/near normal, same "no-gold gate" every M57 battery is scored on.
- **Held-out atoms**: names/instances never seen in training (M56b/M56c's precedent).
- **Compositional transfer gate**: below.

**Three held-out task types**, each constructible from §1.2's ops with zero new primitives:

- **T1 — inverse + cleanup**: "who is tall in passage 1?" = `QUERY_ENTITY` (inverse read) then `SELECT`+`EMIT` over its own candidate set when more than one entity matches (composes gold programs 5 and 2's tail).
- **T2 — negation-over-LTM**: "mary no longer lives in the kitchen — where does she live?" across two passages = `LINK` (gold program 6) + the existing trained `NEGATE` op + `QUERY`.
- **T3 — two-hop query**: "where is the person who is tall?" = `CAND_FOR`(tall) → `QUERY_ENTITY` → `QUERY`(instance, PLACE) — composes gold programs 4 and 5, the literal recombination test TRACK_C_DESIGN §4.4 names as the real bar.

**GO**: ≥0.8 task accuracy on ≥2 of 3 transfer tasks with ≤10 trace-supervised episodes each, AND the zero-trace condition beats cheat by a margin proportionally comparable to A's own 66% floor-to-ceiling closure on senses (M54b) — TRACK_C_DESIGN §5's yardstick. **KILL** if: no transfer task clears cheat+0.1 after 20 trace-supervised episodes, OR entropy fails to sharpen over Stage-(b) training on any of the three, OR the only way to reach transfer is a per-task op or register addition (TRACK_C_DESIGN §4.5, extended to registers — Decision 3).

Standard parity gates (reused from TRACK_C_DESIGN §5 / M57, "within noise" unless stated): pronoun task ≥0.90, binding ≥0.98/≥0.98 anti-recency (A: 0.913, 1.000/1.000); sense task ≥0.84, binding ≥0.65/≥0.60 flipped (A: 0.863, 0.680/0.629); writeback 1.000/1.000/1.000 with forced-wrong ≤0.000 (A: M57b); rich task ≥0.78, binding ≥0.90 (A: 0.816/0.954); rich@8-entities task ≥0.90 with forced-wrong ≤0.85 (A: 0.938/0.812, the necessity gap); instance task ≥0.68, binding ≥0.78 (A: 0.716/0.819); inverse-query ≥0.60 (A: 0.677-0.710, explicitly "not saturated" — parity means matching A's own headroom).

## 4. The A-vs-C tournament

Compared: Track A's fixed pipeline (§1.4) vs. the executor (§1-2), on the full M53-M57 mix (held-out names/instances by default). "C wins" requires **both**: (1) parity on every solved task at §3's thresholds, under no-trace/held-out/forced-wrong arms, not just forced-gold; (2) clearing the transfer gate (§3) on tasks Track A **cannot do at all** without new specialist-head code (T1-T3 have no existing head to call — the point of choosing them). Parity alone, without transfer, is a legibility/param-count result worth recording honestly, not sufficient to change the tournament outcome — Track A already gets full credit as the simpler, proven "cheat" baseline in that case.

## 5. Plan

| phase | ships | gate |
|---|---|---|
| 0 (no-regret, now) | register-file dataclass, op-legality type table (pure Python), trace-extraction instrumented off `forward`'s own execution (§1.4, mechanical, not hand-authored); wire `candidates_for`/`inverse_query` into the live batch-build path (OP_INVENTORY §5's gap) | unit tests only, zero training-loop risk |
| 1 — bootstrap | executor scaffold + `execute()` dispatch for every §1.2 op | forced §1.4 trace reproduces Track A byte-for-byte / within tolerance on all existing suites |
| 2 — trace-supervised | Stage (a) on §2a's six families, full mix | forced-trace + forced-WRONG + parity thresholds (§3), under forcing |
| 3 — trace-weaning | Stage (b) straight-through + annealing | Stage-2 numbers held with trace loss OFF + held-out atoms + forced-wrong still craters |
| 4 — compositional transfer | T1-T3 curricula, zero-trace/few-trace eval | §3's GO/KILL numbers |
| 5 (GO only) | A-vs-C tournament write-up | §4's two-part bar; Track A ships as fallback regardless |

Estimated agent-scale work: TRACK_C_DESIGN §5's collapse-only sketch was 400-700 lines across 4-5 files, one M53b/M54-sized build. This widens scope from collapse-only to the whole clause pipeline plus transfer curricula: roughly **1,000-1,500 changed/new lines across 6-8 files**, five to six Sonnet-agent-scale milestones (Phases 0-4), each gated and RESEARCH_NOTES-ledgered, win or lose. Track A stays deployed through every phase; it is only ever a *candidate* for replacement after a Phase 5 GO, and stays in the codebase as fallback even then.

## 6. Decisions for the lead

**D1 — Soft vs. hard op/argument selection during training?** Default: soft through Stage (a), straight-through hard only at Stage (b)/(c), mirroring `_collapse_weights`'s existing pattern. Consequence of hard-from-the-start: faster to a legible discrete program, but risks the exact TerpreT gradient-through-discreteness failure the graveyard survey calls this design's sharpest kill signal. **HYPER-CRITICAL.**

**D2 — Does the executor get its own recurrent state, or does the GRU stay the single program counter?** Default: GRU stays the only recurrence; the op-selector reads it, `TICK` still updates it once per clause step. Consequence of a second state: doubles the recurrent parameter surface and reopens the M53 "state entanglement" question M54c already retired as a capacity confound, not architectural — risks reintroducing it before there's evidence it's needed. **HYPER-CRITICAL** — touches invariant #1 and the A/B experiment's own logic.

**D3 — Is the register file (§1.1) locked for the gate's duration, or open to grow per-task?** Default: locked, mirroring TRACK_C_DESIGN §4.5's "no per-task op additions," extended to registers. Consequence of leaving it open: any transfer "success" (§3) becomes unfalsifiable. **ROUTINE** to decide, but the transfer gate's honesty depends on the answer being "locked."

**D4 — Does the (op, register) trace format live in a new module, or reuse `src/nsm_ct/mind/ops.py`'s `Op`/`TraceStep`?** Default: new module — `mind/ops.py` is a coarser 8-op VM over a different substrate, unconnected M1-M4 prior art (LTM_DESIGN_BRIEF §0). Consequence of reusing it: inherits a granularity that doesn't type-check against `Addr`/`Vec`/`Dist`/`Scalar`, forcing a redesign of ITS vocabulary first — more work, and conflates two separate "more research" verdicts. **ROUTINE.**

**D5 — Does the compositional transfer gate (§3) run before or after M58's real-text prose milestone?** Default: before — TRACK_C_DESIGN §4.4 already proposed folding a transfer test into the existing arc, and OP_INVENTORY's closing sequence is "op inventory, then LTM, then M58 prose." Consequence of running it after: M58 gets built entirely on today's fixed pipeline, and a later-successful executor reopens transfer-to-prose as a second wave instead of a built-in property. Consequence of an even earlier slot (ahead of LTM): further delays multi-passage reading, already AURORA_SPRINT priority 2. **HYPER-CRITICAL** — a sequencing call the director owns per CLAUDE.md's orchestration model.
