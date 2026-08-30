# The learned EXECUTOR: Track C made concrete

Decision doc, not a survey (2026-08-30; revised same day to fold in `dev/EXECUTOR_DESIGN_REVIEW.md` — every finding accepted, applied throughout, decisions in §6). For the lead's review before any executor code exists. Read order: `dev/TRACK_C_DESIGN.md` (§1.8/§1.10 — the "acid test" §1.1 now runs on all nine programs), `dev/OP_INVENTORY.md`, `dev/OP_LIBRARY_MAP.md`, `dev/MIND_INTERFACE.md`, `dev/LTM_DESIGN_BRIEF.md` §5, `src/nsm_ct/ops.py` (op signatures + `RegisterFile` trace format, used verbatim below), `src/nsm_ct/ltm.py` (M59a — LTM ops **built**), `src/nsm_ct/clause_reactor.py`, `src/nsm_ct/resolver.py`.

## 0. Framing

Today's per-clause loop is **one fixed program**: `forward` runs `query -> _collapse -> GRU tick -> gate/overwrite/negate -> write -> respond`, same order every clause step (`clause_reactor.py:3007-3099`). The only "choice" is *which candidate branch is present* — a Python `if`, not a learned decision (TRACK_C_DESIGN §1.9: "EMIT's source register is hardcoded, not routed"). The executor makes op- and argument-selection themselves learned and budgeted, so today's sequence becomes one point in a space of programs, not the only program that exists.

**Why now.** TRACK_C_DESIGN §7's verdict was "more research," conditioned on three prerequisites (§1.8 `Feat` register, §1.9 routed `EMIT`, §1.10 per-candidate `Addr`), all now built and validated (M56b/M56c, M55a). M57 then proved the op inventory works at scale **and is necessary**: cheat (GRU state-carry, no explicit binding) scores 0.500 vs. normal 0.938 at 8 entities (M57 battery #3) — the argument for a program-selecting controller over a bigger GRU, and the exact failure §1.3/§6 D2 below now also guards against *inside* the executor's own op loop.

**What does not change.** Invariants hold unmodified: #1 (weights hold policy, never a fact), #4 (every gated write logs, unconditionally, §1.2), #5 (`SCORE` generalizes existing heads, doesn't add a fourth), #6 (dials are named scalars), #8 (compute is budgeted, now with measured kill criteria, §3). Track A remains the tournament baseline and fallback regardless of outcome. No knowledge migrates into weights — §3's honesty machinery, extended here to the router itself, is what keeps catching it.

## 1. The machine

### 1.1 Registers — v0 register file (derived, then locked)

Per D3 (§6): TRACK_C_DESIGN §1.10's acid test — write every gold program out register-by-register, THEN freeze — run here on all **nine** programs this design commits to: six gold-program families plus the three Phase-3 transfer tasks (T1-T3, illustrative only — curricula not authored until Phase 3, after M59b, §3/§6 D5). `C` = candidate-set size, perception-enumerated, ≤8. `A_e/A_r/V_v` and per-candidate `addr/feat/prior` are **pre-loaded** per clause, never op-written; every other cell below is an op's output, verified per program (no read precedes its write). `state` (clause-level GRU) and `ctrl` (op-loop control counter, §1.3/D2) are not registers — neither carries facts.

`X` = the address (`A_e` or `A_w`) a program reads/writes at. **EPILOGUE-W** (ends in a write): `TICK(X,A_r,V_v,V_read,state)→state; GATE/OVERWRITE/NEGATE(state,V_v)→S_gate,S_owr,S_neg; WRITE(M_mem,X,A_r,V_v,S_gate,S_owr,S_neg)→M_mem (**≤1/clause**, triggers unconditional `PROVENANCE`, §1.2/§1.3); RESPOND/RESPONSE(state,V_read)→answer; HALT`. **EPILOGUE-Q** (question, no write): same minus `GATE/OVERWRITE/NEGATE` and `WRITE`.

| program | step | op | reads | writes | type |
|---|---|---|---|---|---|
| **1 plain fact** | 1 | QUERY | A_e,A_r | V_read | Addr,Addr→Vec |
| | 2 | EPILOGUE-W[A_e] | | | GRU/Scalar/Mem/answer |
| **2 pronoun value-redirect** | 1 | QUERY | A_e,A_r | V_read | →Vec |
| | 2 | QUERY_CAND `[*]` | P.addr,A_r | P.mem | per-i→Vec |
| | 3 | SCORE `[*]` (stack: addr,feat,prior,mem; rest 0) | P.*,V_ev | P.score | per-i→Scalar |
| | 4 | SELECT | P.score | D_w | →Dist |
| | 5 | EMIT | D_w,P.mem | V_v | →Vec |
| | 6 | EPILOGUE-W[A_e] | | | |
| **3 write-back addr-redirect** | 1 | QUERY | A_e,A_r | V_read | →Vec |
| | 2 | QUERY_CAND `[*]` | P.addr,A_r | P.mem | per-i→Vec |
| | 3 | SCORE `[*]` (addr,prior,mem; rest 0) | P.*,V_ev | P.score | per-i→Scalar |
| | 4 | SELECT | P.score | D_w | →Dist |
| | 5 | EMIT | D_w,P.addr | A_w | →Addr |
| | 6 | REREAD | A_w,A_r | V_read | →Vec |
| | 7 | EPILOGUE-W[A_w] | | | |
| **4 definite-desc read** (Q) | 1 | CAND_FOR | M_mem,V_desc (="doctor") | P.addr,P.prior | per-i |
| | 2 | QUERY_CAND `[*]` | P.addr,A_r | P.mem | per-i→Vec |
| | 3 | INTERACT `[*]` | P.mem,V_ev | P.score_extra | per-i→Scalar |
| | 4 | SCORE `[*]` (addr,prior,mem,score_extra) | P.*,state | P.score | per-i→Scalar |
| | 5 | SELECT | P.score | D_w | →Dist |
| | 6 | EMIT | D_w,P.addr | A_w | →Addr |
| | 7 | REREAD | A_w,A_r | V_read | →Vec |
| | 8 | EPILOGUE-Q[A_w] | | | |
| **5 inverse query** (Q) | 1 | QUERY_ENTITY | A_r,V_v | V_read | Addr,Vec→Vec |
| | 2 | EPILOGUE-Q[A_e] | | | |
| **6 recall+link** (LTM, M59a, Q) | 1 | CAND_FOR | mem_total(M_mem,M_ltm),V_desc (="mary") | P.addr,P.from_ltm,P.prior | `ltm.py:173-201` |
| | 2 | QUERY_CAND `[*]` | P.addr,A_r | P.mem | per-i→Vec |
| | 3 | INTERACT `[*]` | P.mem,V_ev | P.score_extra | per-i→Scalar |
| | 4 | SCORE `[*]`=`link` (addr,prior,mem,score_extra,from_ltm) | P.*,state | P.score | `ltm.py:322-348` |
| | 5 | SELECT | P.score | D_w | below `LINK_THRESHOLD`→minted NEW, never a 4th atom |
| | 6 | EMIT | D_w,P.addr | A_w | →Addr |
| | 7 | REREAD | A_w,A_r | V_read | →Vec |
| | 8 | EPILOGUE-Q[A_w] | | | |
| **T1 inverse+cleanup** (Q, illustrative, composes 5+2's tail) | 1 | CAND_FOR/INVERSE_QUERY | M_mem,A_r,V_v | P.addr,P.prior | multi-match `QUERY_ENTITY` |
| | 2 | QUERY_CAND `[*]` | P.addr,A_r | P.mem | per-i→Vec |
| | 3 | SCORE `[*]` (addr,prior,mem) | P.*,V_ev | P.score | per-i→Scalar |
| | 4 | SELECT | P.score | D_w | →Dist |
| | 5 | EMIT | D_w,P.addr | A_w | →Addr |
| | 6 | REREAD | A_w,A_r | V_read | →Vec |
| | 7 | EPILOGUE-Q[A_w] | | | |
| **T2 negation-over-LTM** (illustrative, composes 6+NEGATE+QUERY, 2 clauses) | 1-7 | = program 6 steps 1-7 | | A_w,V_read | |
| | 8 | EPILOGUE-W[A_w], NEGATE dominant | state,V_v | S_neg,M_mem | retraction write, ≤1/clause |
| | 9 | QUERY (next clause) | A_w,A_r′ | V_read | new relation, e.g. LOCATION |
| | 10 | EPILOGUE-Q[A_w] | | | |
| **T3 two-hop query** (Q, illustrative, composes 4+direct QUERY) | 1 | CAND_FOR | M_mem,V_desc (="tall") | P.addr,P.prior | per-i |
| | 2 | QUERY_CAND `[*]` | P.addr,A_r | P.mem | per-i→Vec |
| | 3 | SCORE `[*]` (addr,prior,mem) | P.*,V_ev | P.score | per-i→Scalar |
| | 4 | SELECT | P.score | D_w | →Dist |
| | 5 | EMIT | D_w,P.addr | A_w | hop 1 resolved |
| | 6 | QUERY | A_w,A_r=PLACE | V_read | hop 2, direct read |
| | 7 | EPILOGUE-Q[A_w] | | | |

**Derivation.** Union of every reads/writes cell above, type-checked (every read is a pre-load or postdates its own program's write). `V_ctx`, `S_tmp`, `P.tmp[*]`, `P.query_addr/r[*]` (sense/garden-path branches only, not among these nine — RankHead is built-unproven, out of v0 scope) are dropped from the original draft's table, which listed 6 per-candidate slots and was wrong about it:

| register | type | count | source |
|---|---|---|---|
| `A_e`, `A_r` | Addr | 2 | pre-loaded, `batch.entity[:,t]`/`batch.relation[:,t]` |
| `A_w` | Addr | 1 (scratch) | defaults to `A_e`; written only by `EMIT` (3,4,6,T1-T3) |
| `V_v` | Vec | 1 | pre-loaded, `batch.value[:,t]` (empty on question steps) |
| `V_read` | Vec | 1 (scratch) | written by `QUERY`/`QUERY_ENTITY`/`REREAD` |
| `V_ev` | Vec | 1 | pre-loaded, current mention's evidence/feature content (unifies the draft's `mention_feat`/`evidence_target`) |
| `V_desc` | Vec | 1 | pre-loaded, the description/name/attribute KEY fed to `CAND_FOR` (4,6,T2,T3 — closes program 4's missing-key gap) |
| `S_gate`,`S_owr`,`S_neg` | Scalar | 3 | written by `GATE`/`OVERWRITE`/`NEGATE` — three distinct heads, not folded into `SCORE` |
| `D_w` | Dist | 1 | written by `SELECT` |
| `M_mem` | Mem | 1 | STM, read-write, `WRITE` ≤1/clause |
| `M_ltm` | Mem | 1 | LTM (M59a), additive-read-only here (`ltm.mem_total`); consolidated end-of-passage, outside this loop |
| `P.addr/feat/prior/mem/score/score_extra/from_ltm[i]` | mixed | 7×C | active candidate set; `addr/feat/prior` pre-loaded, rest op-written |

**v0 register file — locked at end of Phase 1 (D3, §6).** Any register added after the freeze is a KILL-adjacent finding, ledgered in RESEARCH_NOTES, not a quiet table edit. `C` is pre-enumerated by perception — the executor scores among candidates already handed over, never searches for them (no open-ended addressing, TRACK_C_DESIGN §1.4 — what keeps this out of the DNC graveyard).

### 1.2 The op library

Every row is `proven`/`built`/`built-unproven`/`planned` in OP_INVENTORY.md or concretely implemented in `src/nsm_ct/ops.py` — this design adds routing, no new fixed math.

- **Memory**: QUERY (`ops.unbind_query`), QUERY_ENTITY (`ops.inverse_query_entity`), QUERY_CAND/QUERY_CAND_PER_ADDR, WRITE (`ops.bind_write` — **≤1/clause**, §1.3), ADDR_REDIRECT, REREAD (M57c.2), ATTR_WRITE, MINT (`ops.allocate`), CAND_FOR/INVERSE_QUERY (offline-tested, not yet the live batch-build path, OP_INVENTORY §5).
- **Tiers** (LTM — **built, M59a**, `src/nsm_ct/ltm.py`, re-exported `ops.py:52-54`): RECALL = QUERY/QUERY_CAND against `mem_total(M_mem,M_ltm)` (`ltm.py:173-201`, exact by bilinearity, not approximated); LINK (`ltm.link_decision`, `ltm.py:322-348`, `LINK_THRESHOLD=0.5`) reuses entity SCORE with `P.from_ltm[i]` as a stack column; CONSOLIDATE (substate transition, `DocumentRunner`, end-of-passage); PROMOTE (`ltm.promote`, `ltm.py:218-316`, tier-generic, `TRUST_LTM`/`trust_truth`). M59b (curriculum/loop integration) is open — the ops are not.
- **Control**: SCORE (the one learned reducer, generalizing CorefHead/SenseHead/RankHead/SharedScorer — reads a FIXED-WIDTH stack of the 6 per-candidate columns `[addr,feat,prior,mem,score_extra,from_ltm]`, zero-padded per family not populating one, §1.1 — matches the reactor's existing `cand_feature_per_candidate` register, generalized not replaced; per-family heads were rejected as Track A's heads with a router bolted on, and M54c already measured an under-differentiated shared scorer plateau at 0.727-0.737 vs. 0.863 — the padded stack gives SCORE the per-family column signal that scorer lacked), FEATURE_MATCH/INTERACT/PRIOR/CAND_FOR (write SCORE's stack columns, not reducers themselves), COMPARE, SELECT (`_collapse_weights`), EMIT (routed, §1.9), TICK (fixed, not selectable, §1.3/§6 D2), GATE/OVERWRITE/NEGATE (`write_gate`/`overwrite_gate`/`decide_truth` — three distinct heads, kept separate, **not** folded into SCORE — that breaks §1.4's byte-identity gate and over-stretches invariant #5), RESPOND/RESPONSE, HALT (a selectable op — halting IS a learned choice; `K_max` is the safety net under it, not evidence halting is unlearned).

**Not in the learned vocabulary.** **PROVENANCE** is not selectable — every gated WRITE logs unconditionally (invariant #4); an op the executor could learn to skip would defeat the audit trail. **FORCE** is not selectable — it's the test harness's teacher-forcing mechanism, consumed only by §3's honesty arms, never reachable at inference.

A separate `src/nsm_ct/mind/ops.py` VM already exists — unconnected M1-M4 prior art over a different substrate; reusable machinery, not this executor's format (§6 D4).

### 1.3 The per-clause loop

```
ctrl     = TICK_CTRL(ctrl, prev_op_id, step_idx, type_mask, margin, abstain_flag, K_max-step_count)
op_id    = OpSelect(ctrl, pool(registers))            # categorical, masked to type-legal ops
arg_regs = ArgSelect(op_id, ctrl, registers)           # per argument slot, over FEATURE columns
result   = execute(op_id, arg_regs)                    # fixed math or SCORE, never learned routing
registers[dest(op_id)] = result
assert writes_this_clause_step <= 1                     # §1.3 constraint
if op_id == HALT or step_count == K_max: emit_and_advance()
else: step_count += 1; repeat
```

**Program counter, fixed (D2, DECIDED — lead).** The original `OpSelect(state, pool(registers))` was a defect: `state` updates once per CLAUSE, constant across the ≤12 op-loop steps — the selector could not see which step it was on, and the minimum-loss solution is a ~6-way clause-feature classifier reproducing §1.1's programs from memory, passing every §3 gate without routing at all. Fix: a GRU-shaped counter, `ctrl`, self-loops once per OP-STEP — it IS the program counter — but is strictly Harvard-split from the data path: control signals only (prev op id, step index, type mask, margin/abstain/halt-budget), **never register data**; `OpSelect` no longer reads the main clause-level `state` directly. This is not "a second recurrent state" in the sense the original decision warned against: that risk was a state that SEES register contents and silently computes the answer inside itself, reducing ops to decoration (M57's cheat, relocated inside a clause). A control-only recurrence has nothing to compute an answer from. `state` is untouched, still ticks once per clause via `TICK`, still feeds `GATE`/`OVERWRITE`/`NEGATE`/`RESPOND` — just no longer wired into the router.

**One memory WRITE per clause.** Today's reactor issues at most one `WRITE` per clause step (`clause_reactor.py:3066`); the executor must preserve this, enforced by the assert above, not left emergent — more than one opens a superposition cheat channel (blurred readout hides which bind mattered) and multiplies retained-tensor count against the memory kill criterion (§3). More than one write is a defect, not a discovery.

`K_max` (default 12) is a HYPERPARAMETER baked into the training graph, not the `patience` dial MIND_INTERFACE.md names (a runtime scalar tunable without retraining, still a stub) — this design delivers a fixed cutoff under it, not the dial itself.

The answer is emitted unchanged: `RESPOND`/`RESPONSE` accumulate per-step logit + content, softmax-weighted sum, cosine-compared to option vectors (`clause_reactor.py:3101-3109`) — outside the op loop, scored once per episode.

### 1.4 The bootstrap anchor: today's pipeline as one fixed trace

**An executor with this trace forced must reproduce Track A byte-for-byte** on every existing regression suite (candidate families are mutually exclusive per row/step, so a real trace uses at most one bracket):

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
WRITE(M_mem,A_e|A_w,A_r,V_v,S_gate-S_neg,overwrite=S_owr)                         # :3066, ≤1/clause
PROVENANCE(...)                                                                    # :3067-85, unconditional
RESPOND(state)->rl; RESPONSE(state,V_read)->rv                                    # :3086-90
```

`[sense]`/`[hyp]` are Track A capabilities this trace must still reproduce but aren't among §1.1's nine registers-frozen programs — no v0 register was added for their `V_ctx`/`P.tmp`/`P.query_addr/r` needs. Forcing the op/arg-selector to reproduce this trace step-for-step is a mechanical zero-ambiguity regression test — a mismatch means the execute-layer (not routing) has a bug, caught in Phase 1 before any training starts.

## 2. Learning

**Op selection**: a categorical head over the ~20-op vocabulary (§1.2, minus PROVENANCE/FORCE), conditioned on `[ctrl; pool(registers)]` — `ctrl` is the op-loop's control-only program counter (D2), never `state` directly (Harvard split); `pool` = fixed concat of named globals + mean/max over the active `P.*` bank, size-invariant in `C`. Masked to type-legal next ops given the last op's write and populated register types.

**Argument selection**: for each slot, a small attention head scores registers of the required type, softmax(train)/argmax(eval) per D1's TYPE split (§6) — `_collapse_weights` generalized from "which candidate" to "which register." A per-candidate `Addr` slot attends over `P.feat[0..C-1]` (the FEATURE column), **never** `P.addr[0..C-1]`'s raw identity vectors as the key — that's M56b's memorization surface (a name→slot lookup table in weights, invisible until held-out atoms run); `P.addr[i]` is read only as the VALUE once a slot is chosen.

### 2a. Gold programs (trace-supervised targets)

The six families' programs, and the three Phase-3 transfer tasks (illustrative), are §1.1's table — derived mechanically off `forward`'s own execution, extended past collapse to the whole clause loop; T1-T3's curricula are not authored until Phase 3, after M59b (§3, §6 D5).

**Training modes**: (a) trace-supervised — teacher-forced sequences from §1.1, per-step cross-entropy on both choices, plus intermediate register-value loss. (b) straight-through, traces annealed — selection goes hard per D1's TYPE split (`Addr` hard from step one; `Dist`/`Scalar` soft-to-hard as trace loss decays), not uniformly staged. Gate: match Stage-(a) with trace loss OFF, AND report D1's soft-vs-argmax delta at every gate. (c) task-loss-only (the cliff) — zero trace supervision; measured, never assumed.

**Credit assignment**: ≤12 steps, a handful of ops × type-legal bindings (§1.2's masking prunes most before learning starts). What would show blur: (i) op-usage histograms flatten toward uniform; (ii) intermediate register-value loss stays high while task loss drops; (iii) `H(program | clause-type features)` ≈0 on eval — a dispatch table by measurement, D2's own failure mode if left unfixed; (iv) forced-WRONG-trace (§3) fails to crater.

## 3. Honesty machinery (mandatory)

- **Forced-trace (gold)**: force §1.1's exact program. Must reproduce §1.4's bootstrap numbers.
- **Forced-WRONG-trace**: force a wrong op/arg-register at one decisive step. Must **crater** (1.000→0.000, 0.938→0.812 pattern).
- **Cheat (no executor)**: Track A's pipeline run as-is — the TOURNAMENT floor (§4), **not** the per-family floor arm below — two different quantities, previously conflated.
- **No-trace eval**: trained executor (b/c), zero trace supervision, zero forcing at eval — must hold at/near normal.
- **Held-out atoms**: names/instances unseen in training, run specifically on `ArgSelect` (§2's `P.feat`-not-identity fix).
- **Shuffled-selector arm**: keep `execute()`/SCORE weights, replace trained `OpSelect` with a uniform type-legal sampler — if accuracy holds, routing is decorative (forced-WRONG-trace alone only proves the ops are load-bearing, not the selector).
- **Register/slot permutation invariance**: at eval, permute candidate slot order, rename scratch registers. A composer is invariant; a positional memorizer craters.
- **Departure metrics** (measured every Phase-2/3 report): departure rate (fraction of correct episodes whose op sequence differs from Track A's) and shortcut discovery (dropping a provably-dead op more than a random type-legal walk would) — with no phase rewarding departure from §1.4's program, a GO is indistinguishable from "Track A re-implemented as a 12×-slower interpreter" unless these move.

**Compute kill criteria.** Per-clause wall-clock ≤ `K_max` × Track A's own measured per-step cost; peak training memory ≤ 1.5× Track A's at the same batch (Track A: 6.4GB at T≈50, unfinished — M57 battery #3). **KILL if either is exceeded**, measured first at Phase 1's end — before transfer work, since an unaffordable per-step cost makes every later phase moot under CLAUDE.md's one-run-at-a-time rule.

**Phase 2's FIRST gate — leave-one-family-out (LOFO):** Train with traces from five of §1.1's six gold-program families; measure the sixth with **no traces** (the program-level analogue of M56b's held-out-atom test — the cheapest, most decisive experiment here, runs on existing curricula, no new data). Report per held-out family. GO/KILL are defined relative to two arms measured alongside it, never as an absolute number: (a) a **hand-written-program oracle** — the executor forced to the held-out family's own §1.1 gold trace; (b) a **real floor arm** — the executor with op selection frozen to the plain-fact program (§1.1 #1), **never** Track A (that's the tournament's floor, §4). **GO: ≥0.5 of the oracle-floor gap on ≥4 of 6 held-out families. KILL: <0.2 of the gap on all 6.** An intermediate result is reported honestly, not forced into either bucket, and does not clear Phase 2 alone.

T1-T3's curricula (§1.1) move to **Phase 3, after M59b** — they ride `CAND_FOR` (never run in a training loop) and, for T2, `link`/`consolidate` (built M59a, not curriculum-integrated); gating on them before LOFO risked a false KILL from a gap the executor didn't cause (T1/T3 compose a 0.677-0.710 op with a ~0.90 op, ceiling ~0.63, below a flat 0.8 bar). Phase 3 reuses LOFO's oracle/floor-arm framing, not a second absolute bar, and adds blind curriculum authoring (T1-T3 rewritten by an agent not shown OP_INVENTORY — the tasks as written here were reverse-engineered from the ops).

Standard parity gates (reused from TRACK_C_DESIGN §5/M57, "within noise" unless stated): pronoun ≥0.90, binding ≥0.98/≥0.98 (A: 0.913, 1.000/1.000); sense ≥0.84, binding ≥0.65/≥0.60 (A: 0.863, 0.680/0.629); writeback 1.000/1.000/1.000, forced-wrong ≤0.000 (A: M57b); rich ≥0.78, binding ≥0.90 (A: 0.816/0.954); rich@8-entities ≥0.90, forced-wrong ≤0.85 (A: 0.938/0.812); instance ≥0.68, binding ≥0.78 (A: 0.716/0.819); inverse ≥0.60 (A: 0.677-0.710, "not saturated").

## 4. The A-vs-C tournament

Compared: Track A (§1.4) vs. the executor (§1-2), on the full M53-M57 mix. "C wins" requires **both**: (1) parity on every solved task at §3's thresholds under no-trace/held-out/forced-wrong arms, not just forced-gold; (2) clearing LOFO and, once reached, the Phase-3 transfer bar on tasks Track A **cannot do at all**. Parity alone, without transfer, is a legibility/param-count result worth recording, not sufficient to change the outcome — Track A keeps full credit as the proven "cheat" baseline.

## 5. Plan

| phase | ships | gate |
|---|---|---|
| 0 (no-regret, now) | `RegisterFile` dataclass (`ops.py:711-804`), op-legality table, trace-extraction off `forward`'s own execution; wire `CAND_FOR`/`INVERSE_QUERY` into the live batch-build path | unit tests only |
| 1 — bootstrap | `execute()` dispatch for every §1.2 op; v0 register file FROZEN at phase end (D3) | forced §1.4 trace byte-for-byte; compute kill criteria (§3) measured first |
| 2 — trace-supervised | Stage (a) on §1.1's six families | LOFO FIRST, then forced-trace/forced-WRONG/parity, under D1's per-type split |
| 3 — trace-weaning + transfer | Stage (b) straight-through; T1-T3 (blind-authored), after M59b | Stage-2 held with trace loss OFF + held-out atoms + T1-T3 oracle/floor thresholds |
| 4 (GO only) | A-vs-C tournament write-up | §4's two-part bar; Track A ships as fallback regardless |

Estimated agent-scale work: ~1,000-1,500 changed/new lines across 6-8 files, five Sonnet-agent-scale milestones, each gated and RESEARCH_NOTES-ledgered, win or lose. Track A stays deployed throughout.

## 6. Decisions for the lead

**D1 — Soft vs. hard selection. DECIDED (lead, 2026-08-30): hard (straight-through) for any slot whose value becomes a memory KEY (`Addr`), soft only where mixture is already the semantics (`Dist`/`Scalar`).** Required reporting: a soft-vs-argmax delta at EVERY Phase-2 gate. Rationale kept as the record:
- For: a soft `Addr` mixture is an interference-contaminated bilinear key (exact only for orthonormal keys) — it trains a semantics that doesn't exist at eval, relocating the M57#3 cheat (0.500/0.938) into `ArgSelect`.
- For: trace supervision is cross-entropy on the selector distribution — it needs no soft execution to work at all.
- Against: straight-through-from-the-start is the TerpreT gradient dynamic this design calls its sharpest kill signal; early hard `Addr` may under-explore.
- Against: soft-key/hard-elsewhere has no precedent battery on this codebase.

**D2 — Second recurrent state? DECIDED (lead, 2026-08-30): no second data-carrying recurrent state — the GRU self-loops once per OP-STEP as the program counter (`ctrl`, §1.3), fed CONTROL SIGNALS ONLY.** Inputs: previous op id, step index, register TYPE mask, and scalar summaries (margin/abstain flag/halt budget) — **never register data vectors**; `OpSelect` no longer reads `state` directly (Harvard split: control path vs. data path). Why: a recurrent state that sees register CONTENTS can compute the answer internally and reduce the ops to decoration — the M57 state-carry cheat, relocated inside a clause where no existing crater test catches it. This fixes "the selector can't see which step it's on" (the actual defect) without reopening the M53 entanglement question a second general-purpose, data-carrying recurrent state would.

**D3 — Register file locked? Decided: YES, at end of Phase 1** — only after writing out all nine programs (§1.1) and type-checking the union first (the §1.10 acid test), not before. Any post-freeze register is a KILL-adjacent finding, ledgered in RESEARCH_NOTES, per TRACK_C_DESIGN §4.5's "no per-task op additions" extended to registers.

**D4 — Trace module: new, separate from `mind/ops.py`. Decided: new module** (`ops.py`'s `RegisterFile`, already built) — `mind/ops.py`'s 8-op VM is unconnected M1-M4 prior art over a different substrate (LTM_DESIGN_BRIEF §0); reusing it would force a redesign of ITS vocabulary first. `mind/ops.py` gets an OP_INVENTORY header note that it is not this executor's instruction set.

**D5 — Transfer gate before or after M58? Decided: split.** The paper type-check (§1.1) and LOFO (§3) run NOW, before LTM/M58 — existing data, zero curriculum cost, removes the false-KILL risk of an arithmetic-ceiling bar. The T1-T3 CURRICULUM gate moves to Phase 3, after M59b — T2 rides `link`/`consolidate` (built, not curriculum-integrated), T1/T3 ride `CAND_FOR` (never run in a training loop). Track A ships as fallback through every phase regardless of outcome.
