# The op inventory: the mind's explicit instruction set

M57's closing ask (RESEARCH_NOTES tail, 2026-08-30): before LTM and before
M58 prose, write down every operation the system performs, fixed vs learned,
in one place. Read order behind this doc: `dev/MIND_INTERFACE.md` (dials,
invariants), `dev/TRACK_C_DESIGN.md` §1 (the op algebra + type notation
reused below — `Addr`/`Vec`/`Feat`/`Scalar`/`Dist`), `dev/LTM_DESIGN_BRIEF.md`
§5 (locked LTM decisions, the PLANNED rows' source of truth).

## 0. Thesis

The mind is a small explicit instruction set of fixed operations over an
inspectable memory — an order-3 `entity⊗relation⊗value` tensor
(`entity_memory.py`), an enumeration index (`InstanceRegistry`), and an
append-only audit log (`ProvenanceLog`) — none of which hold a single
trained weight. The learned policy is deliberately tiny (one GRU cell + six
linear heads + a resolver behind one contract, low-thousands to sub-10k
params per head, sub-MB total) and it never decides WHAT the mind can do;
it only decides HOW MUCH (gate/overwrite/negate strengths, respond timing)
and WHICH (collapse: which candidate a pronoun/description/reading binds
to). New capability = a new named op with a deterministic v1 policy FIRST;
a learned gate is added only when a battery shows the deterministic policy
is the actual bottleneck — never by default. The history that earns this
rule: four smoke-scale results have inverted at full scale on this project
(M54/M55/M55b/M57b — CLAUDE.md's ops rules, RESEARCH_NOTES M57b), and M56's
"perfect" 1.000 pronoun binding turned out to be a six-name lookup table
memorized in resolver WEIGHTS, invisible until a held-out-name ablation was
run (M56b) — knowledge silently migrating into weights is the failure mode
this whole design exists to prevent, and only an adversarial test (held-out
atoms, forced-wrong arms, cheat baselines) catches it before it ships.

## 1. The op table

Fixed = deterministic math, zero learned parameters. Learned = at least one
scalar/vector comes from a GRU-state-conditioned head or resolver. Status:
**proven** = full-scale battery with a forced-wrong/no-gold gate passed;
**built-unproven** = code + unit tests exist, no training-loop battery yet;
**offline-tested** = unit-tested standalone, not yet the live batch-build
call path; **test-only** = exists to validate other ops, never runs at
inference; **planned** = designed (LTM_DESIGN_BRIEF §5), not built.

| op | signature (Track C notation) | fixed/learned | learned param source | dial(s) | code | validated by | status |
|---|---|---|---|---|---|---|---|
| **write** (bind) | `(Addr,Addr,Vec,Scalar,Scalar)→Mem` | fixed bilinear + learned gate | `write_gate`/`overwrite_gate` on GRU state | none named (gate is the dial) | `entity_memory.write` (:45); called `clause_reactor.py:2707` | M0h (BUILT); overwrite-vs-vote at M0h+M57 batteries | proven |
| **query** | `(Addr,Addr)→Vec` | fixed bilinear (`einsum bijk,bi,bj→bk`) | none | none | `entity_memory.query` (:21) | M0h (BUILT), every battery since | proven |
| **query_entity** (inverse) | `(Addr,Vec)→Addr` | fixed bilinear (`einsum bijk,bj,bk→bi`) | none | none | `entity_memory.query_entity` (:30); wired in `clause_reactor.forward` via `batch.inverse_mask` (:2691-2694) | M57c battery #1 (below chance, no read existed) → #2 (0.138→0.724) → #3 rich-inverse 0.677-0.710 | proven (mid-60s-70s, not saturated) |
| **collapse — entity** | `({Addr}_C,Feat,{Scalar}_C,{Vec}_C,H)→Dist` | learned (resolver) | `CorefHead`/`SharedScorer` logits | `caution` (planned threshold, not wired — see §4) | `resolver.py` `CorefHead`(:120)/`SharedScorer`(:198); called `clause_reactor._collapse` entity branch (:2463-2589) | M53b→M56b/M56c (held-out 1.000); M57c.3/M57#3 (instance binding 0.486→0.954) | proven |
| **collapse — sense** | `({Vec}_C,Vec,H)→Dist` | learned (resolver) | `SenseHead`/`SharedScorer` logits | none | `resolver.py` `SenseHead`(:336); `_collapse` sense branch (:2591-2608) | M54b (0.680/0.629 binding, closes 66% of floor→ceiling gap) | proven |
| **collapse — hypothesis** (garden-path) | `({Addr}_C\text{ per-cand},Feat,{Scalar}_C,H)→Dist` | learned (resolver) | `RankHead`/`SharedScorer` logits | none | `resolver.py` `RankHead`(:399); `_collapse` hyp branch (:2610-2623) | M55a (plumbing only); M56 spike verdict "more research" before a prototype trains it | built-unproven |
| **address-redirect** (write-back) | `(Dist,{Addr}_C)→Addr` | fixed weighted-sum of the SAME `w` collapse used, applied to candidate atoms not their readout | (reuses entity collapse's `w`) | `cand_addr_mask` (structural, not learned — per-row flag) | `_collapse` entity branch (:2572-2587), `ClauseReactor` docstring M57b paragraph | M57b (forced-wrong craters 1.000→0.000; no-gold 1.000) | proven |
| **post-collapse re-read** | `Addr→Vec` (re-`query` at the resolved node) | fixed | none (gated by `addr_row_t`, itself a byproduct of collapse) | none | `clause_reactor.forward` (:2680-2694) | M57c battery #1 (found the gap: 0.423 forced-gold ceiling) → #2 (0.423→0.856 after this fix) | proven |
| **evidence interaction** (`interact`/`feature_match`) | `(Vec,Vec)→Scalar` (cosine) | fixed | none — a feature INTO the learned resolver, not itself learned | `evidence_prior_beta` (structural-prior mix; off by default) | `resolver.evidence_interaction` (:84); consumed `_collapse` (:2498-2534) | M57c.3/M57 battery #3 (unblocks instance binding 0.486→0.954); `--evidence-prior` measured to add nothing at rich scale | proven (as a resolver input); the prior-mix dial itself: measured null |
| **forcing** (teacher-forced collapse) | `Dist_{forced}` overrides `Dist` | N/A — TEST harness, not a runtime op | none | `cand_forced_index` (-1 = off; per-row/step) | `_collapse` (:2551-2570); consumed nowhere at inference | forced-gold/forced-wrong arms, every M57 battery | **test-only** — never executes outside an honesty-gate eval |
| **mint** (instance) | `()→Addr` (fresh seeded atom) | fixed (seeded RNG, non-trainable) | none | registry `seed` | `instances.InstanceRegistry.mint` (:133) | M57a (near-orthogonality regression-gated, 16/16 tests) | proven (built + regression-gated; not yet a live-batch call path — see gaps) |
| **attribute write** | `(Addr,Addr,Vec,Scalar)→Mem` | fixed (wraps `entity_memory.write`) | gate/overwrite params, same source as **write** | `gate`(default 1.0)/`overwrite` | `instances.write_attribute` (:244) | M57a (roundtrip argmax 100%, 5 instances × 3 attrs) | proven (offline API); live-loop equivalent proven via `_collapse`'s addr-redirect + write |
| **candidates_for / inverse_query** (membrane-side) | `(Mem,\{Addr\}_N,Feat)→\{Addr\}_K` | fixed (enumerate + cosine threshold) | none | `threshold` (default 0.5, named arg, not yet promoted to a module-level dial) | `instances.candidates_for`(:320)/`inverse_query`(:361) | M57a unit tests only | offline-tested — **not** `build_clause_batch`'s live call path today (see gaps) |
| **provenance record** | side-effecting log append | fixed | none | `trust_threshold` (default 0.0, gates whether a write is logged at all) | `instances.ProvenanceLog.append`(:221); reactor wiring `provenance.record_writes`(:48) | M57a (log shape); M57d (wired onto live reactor writes) | proven |
| **respond / emit** | `(H)→Scalar` (timing) ⊕ `(H,Vec)→Vec` (content) ⊕ `(Dist,\{Vec\})→Vec` (weighted sum) | learned timing+content, fixed aggregation | `respond`/`response` linear heads | none | `clause_reactor.__init__` (:2335-2336); aggregated `forward` (:2728-2745) | M0h (BUILT) onward, every battery's `task`/answer column | proven |
| **negate** (truth policy) | `(H,Vec)→Scalar`, subtracted from gate | learned | `decide_truth` linear head | none | `clause_reactor.__init__` (:2334); `forward` (:2706-2707) | §0i (L7/L8 negation: unresolved→MAYBE 1.00, resolved-by-negation 0.86, L8 removal 0.70) | proven (not saturated at L8) |
| **overwrite vs vote** | `(H)→Scalar`, decouples update-vs-accumulate | learned | `overwrite_gate` linear head | none | `clause_reactor.__init__` (:2333); `entity_memory.write`'s `overwrite` arg (:68-70) | §0i (recency/overwrite levels L5/L6, ~0.75-0.94 across runs) | proven |
| **recall** (additive LTM read) | `Mem_{STM}+Mem_{LTM}→Vec` via `query` on each, summed | fixed (design: same `query`, two tensors) | none | none new | none yet — LTM_DESIGN_BRIEF §5.1 | none | planned |
| **link** (LTM identity) | `(\{Addr\}_{STM\cup LTM},Feat,H)→Dist` | learned — reuses existing entity resolver, `from_ltm` feature added to the candidate set | same `CorefHead`/`SharedScorer` contract, new feature column | `link_threshold` | none yet — LTM_DESIGN_BRIEF §5.2 | none | planned |
| **wind-down / consolidate** (substate transition) | `READING→WIND\text{-}DOWN→CONSOLIDATE` | fixed substate machine, fired at end-of-passage | none (transition itself); consolidate's WHAT-to-keep is trust-gated | `trust_ltm` | none yet — LTM_DESIGN_BRIEF §5.3 | none | planned |
| **promote** (tier-generic) | `(Mem_N,ProvenanceLog,Scalar)→Mem_{N+1}` | fixed gated copy (same shape as **write**) | gate = provenance trust vs. dial | `trust_ltm` (tier1→2) / `trust_truth` (tier2→3, strictly higher bar) | none yet — LTM_DESIGN_BRIEF §5.4 | none | planned; L5 corroborate/contradict (`episode.py:_level5`) is the ungated seed |
| **abstain** (idk/MAYBE) | `→\{idk,MAYBE\}\subset\text{options}` | fixed atoms, chosen like any other answer option | `respond`/`response` heads, same as **respond/emit** | none | atoms exist, exercised by §0i's L7 curriculum (`answer=MAYBE`) | §0i: L7-unresolved→MAYBE 1.00 | **partially proven** — works for disjunction-unresolved; NOT wired to `caution`/low collapse-margin anywhere (MIND_INTERFACE.md's 4-sanctioned-forms contract is not yet enforced by any gate) |

## 2. Learned parameters

Every scalar/vector the policy emits, its head, its input, and the arm that
proves it does work (not just that it exists):

| parameter | head | input | proves it works |
|---|---|---|---|
| write gate | `write_gate: Linear(hidden,1)` | GRU state | §0h/§0i baseline; every battery's `old`/`writeback` task column |
| overwrite gate | `overwrite_gate: Linear(hidden,1)` | GRU state | §0i L5/L6 recency levels |
| negate (refutation) strength | `decide_truth: Linear(hidden+d,1)` | GRU state ⊕ value | §0i L7/L8 (MAYBE 1.00, negation-removal 0.70) |
| respond timing | `respond: Linear(hidden,1)` | GRU state, masked to real steps | every battery's `respond_position`/task accuracy (implicit — softmax picks the answer step) |
| response vector | `response: Linear(hidden+d,dim)` | GRU state ⊕ mem_read | same as above — the content half of `respond/emit` |
| entity resolver logits | `CorefHead`/`SharedScorer.forward` | `cand_entity, cand_feature, cand_prior, mem_read, [state]` | M53b→M56c (held-out 1.000); M57#3 (0.954 rich binding) |
| sense resolver logits | `SenseHead`/`SharedScorer.forward` | `cand_entity(=sense vec), mem_read+context, [state]` | M54b (0.680/0.629) |
| hypothesis resolver logits | `RankHead`/`SharedScorer.forward` | `cand_entity, per-addr mem_read, prior, state` | none yet — M56 spike only |
| per-candidate interaction feature | `evidence_interaction` → `CorefHead`'s `cand_feature_extra` column | `cos(cand_mem_read, evidence_target)` | M57c.3/M57#3: instance binding 0.486→0.954 |
| structural-prior mix (unused) | `softmax(s_c·evidence_prior_beta)` × `cand_prior` | same interaction score | M57 battery #3: measured to add NOTHING at rich scale (dial stays off) |

## 3. Dials table

| dial | default | owner module | live/stub |
|---|---|---|---|
| `evidence_prior_beta` | `None` (off); `5.0` when `--evidence-prior` set | `clause_reactor.ClauseReactor`, constant in `scripts/train_instances.py:71` | live, but measured to add nothing (M57#3) — kept off |
| `trust_threshold` (provenance write-gate) | `0.0` (log everything that wrote at all) | `provenance.record_writes` | live |
| `candidates_for` cosine `threshold` | `0.5` | `instances.candidates_for` (function arg, not a named module-level dial yet) | live in the function; not yet reachable from the live training batch-build path |
| `write_gate`/`overwrite_gate`/`decide_truth` | N/A — these are learned heads, not dials | `clause_reactor.ClauseReactor.__init__` | live (learned, not a runtime scalar) |
| `caution` (collapse margin → hard-bind vs hold-open) | not set — no threshold exists in code | none yet (MIND_INTERFACE.md "the dials") | **stub** — margin is computed (`_top2_margin`) but nothing reads it as a gate |
| `patience` (thinking budget) | not set | none in the active v1/v2 track (an unrelated `patience` exists in `mind/drive_rollout.py`, a different, unconnected prior-art subsystem — see LTM_DESIGN_BRIEF §0) | **stub** (name collision with dead code, not the same dial) |
| `yap_emit` / `yap_continue` | not set | none yet | **stub** |
| `trust_ltm` | not set | none yet — LTM_DESIGN_BRIEF §5.3/§5.4 | **stub** (locked design, unbuilt) |
| `trust_truth` | not set (strictly higher bar than `trust_ltm` by design) | none yet — same | **stub** |
| `link_threshold` | not set | none yet — LTM_DESIGN_BRIEF §5.2/§4 "no-regret work" | **stub** |

## 4. Rules for adding an op

- Explicit op + deterministic v1 policy first; a learned gate is added only
  after a full-scale battery (forced-gold/forced-wrong/cheat/no-gold — §5)
  shows the deterministic policy is the actual bottleneck, per this
  project's four recorded smoke→full inversions (CLAUDE.md ops rules).
- Every new op earns a RESEARCH_NOTES ledger entry — win or lose — before
  it's trusted, mirroring how every row above cites the battery that
  proved (or failed to prove) it.
- No new op ships without its four honesty arms: forced-gold (does the
  mechanism work when handed the answer?), forced-wrong (does removing it
  cost accuracy — the craters in the table above are the proof pattern),
  cheat (does a baseline with the op disabled already solve the task by
  a shortcut?), no-gold (does the trained policy alone, zero gold at
  eval, still work?). Only after all four does a op count as **proven**.
- Byte-identity by default: every new optional field (`cand_addr_mask`,
  `cand_forced_index`, `cand_evidence_relation`, `evidence_prior_beta`,
  ...) defaults to `None`/falsy and every table row above documents the
  regression test proving `None` reproduces the pre-change arithmetic
  exactly — no new op is allowed to silently change old behavior.
- Any named scalar is a **dial** in §3, never a magic number in code; any
  op that writes memory logs through `provenance.record_writes` (§1's
  **provenance record** row) — no gated write skips the audit trail; a
  test-only op (**forcing**) is marked as such in the table and never
  reachable from an inference call path.

## 5. Gaps (from RESEARCH_NOTES, mapped to the op they belong to)

| gap | belongs to | note |
|---|---|---|
| no recency feature in the resolver | **collapse — entity** | M57#3: recency-only cases (same-gender pronoun / ambiguous name, n=4-5) stay low; resolver has no recency register at all |
| definite-description questions about an OVERWRITTEN slot cap ~0.64 even under forced-gold | **post-collapse re-read** | M57#3: a read-side limit — the re-read competes with stale GRU recall, not a binding failure |
| inverse-query accuracy stuck mid-60s-70s | **query_entity** | M57#3: "route the entity readout directly to the answer" is the named next fix; not yet a broken mechanism, just unrefined |
| `candidates_for` has no tie-break, and is not the live batch-build path | **candidates_for / inverse_query** | cosine-threshold membership has no margin/caution treatment for near-ties (two instances at ~equal score both clear `threshold` — correct as a candidate SET, but nothing flags the tie as low-confidence the way `_top2_margin` does for resolver logits); `clause_reactor.py` only *references* this function in a comment (:1102) — training curricula build candidate sets some other way, so this op's loop-tested numbers (M57a) haven't been re-validated as the actual live call path |
| footprint at T≈50 is 6.4GB, not finished | **write** (memory tensor growth) | M57 battery #3: `B×d³` with autograd across `T`; AURORA priority-1 engineering item, blocks longer episodes and any LTM tensor that persists across passages |
| `caution`/margin never gates anything | **abstain**, **collapse — entity/sense/hyp** | margins are computed (`_top2_margin`) and recorded, but no code reads them as a hold-open-vs-hard-bind decision — MIND_INTERFACE.md's 4 sanctioned uncertainty forms are only 2/4 live (candidate sets, low-margin-recorded); OPEN-binding and margin-gated abstention are unimplemented |
