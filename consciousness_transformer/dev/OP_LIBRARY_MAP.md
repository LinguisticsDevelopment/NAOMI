# Op library -> reactor integration map

`src/nsm_ct/ops.py` (library) + `tests/test_ops.py` (54 tests). Library-only
milestone -- nothing below is wired into `clause_reactor.py` yet; this is
the plan for the LATER integration milestone. `dev/OP_INVENTORY.md` row
names in the third column.

| op (`ops.py`) | plugs into (file:function) | OP_INVENTORY row / gap closed | battery subset to move on integration |
|---|---|---|---|
| `bind_write` | `clause_reactor.py` forward's `memory = em.write(...)` (:3021) | **write (bind)** | none (already proven; a straight swap) |
| `unbind_query` | `forward`'s `mem_read = query(...)` (pre-write) | **query** | none (already proven) |
| `inverse_query_entity` | `_collapse` inverse-mask branch (:2691-2694) | **query_entity (inverse)** | M57c.3 inverse-query battery (mid-60s-70s, unrefined) |
| `superpose_vote` | `forward`'s write call, `overwrite=owr` path (:3021) | **overwrite vs vote** | §0i L5/L6 recency/overwrite levels |
| `erase` | new: explicit slot clear, no current call site | none yet -- new capability | none built; would gate a future "retraction" curriculum |
| `cleanup` (+ `CLEANUP_MARGIN`) | `_collapse`'s margin computation, `_top2_margin` (:2634) | **caution never gates anything** (Sec.5 gap) | abstain-form batteries: L7/L8 negation, garden-path (RankHead, built-unproven) |
| `similarity` | `resolver.evidence_interaction` (:84-98), final-answer `cosine` (:3060ish) | **evidence interaction** | M57c.3/#3 instance-binding battery, if dot variant is tried against the cosine-only baseline |
| `permute`/`unpermute` | none yet -- no sequence/order encoding exists in the reactor today | new capability (not an OP_INVENTORY row) | would need its own curriculum (ordered multi-arg role binding) |
| `allocate` | `_collapse`'s NEW-candidate path once M59b wires `InstanceRegistry.mint` live (LTM_DESIGN_BRIEF Sec.5.2) | **mint (instance)** | none yet (offline-tested only per OP_INVENTORY) |
| `recency` | `_collapse` entity branch, alongside `cand_feature`/`evidence_interaction` (:2463-2589) | **no recency feature in the resolver** (Sec.5 gap, M57#3) | pronoun/ambiguous-name batteries at n=4-5 (currently stuck low) |
| `temporal_link` | none yet -- no write-order log exists in the reactor (would sit next to `provenance.record_writes`) | new capability (not an OP_INVENTORY row) | would need a "what came before/after X" curriculum |
| `forget_decay` | none yet -- no forgetting mechanism exists | new capability | would need a decay-sensitivity curriculum |
| `recall`/`promote`/`link` | `ltm.py` (re-exports, unchanged) | **recall**, **promote**, **link** (all "planned") | LTM battery (M59b, not yet built) |
| `select` | `_collapse_weights` (:2624) | underlies every **collapse** row | already proven; a straight swap |
| `abstain` (+ `CAUTION`) | new: would sit right after `_top2_margin`, before `respond`/`response` heads | **abstain (idk/MAYBE)** ("partially proven", not margin-gated) | L7 MAYBE battery, extended to margin-triggered (not just disjunction-triggered) abstention |
| `compare` | final-answer cosine (already exists inline); would generalize into `_collapse` if a compare-based branch op is added | **compare** (Sec.1.3, "not currently used inside any resolver") | garden-path battery, if hypothesis collapse routes through an explicit compare/branch step |
| `branch` (+ `BRANCH_THRESHOLD`) | new -- Track C executor's conditional primitive, no reactor call site yet | new capability (Track C algebra, Sec.1.3) | would gate the first executor-routed curriculum |
| `halt` (+ `PATIENCE`) | new -- Track C's `K_max` cutoff, no reactor call site yet | `patience` dial (MIND_INTERFACE.md, currently "stub") | would gate the first multi-step chained-program curriculum |
| `emit` | reference stand-in for `clause_reactor.response`/aggregation (:2602, :3057) | **respond / emit** | none (the learned head stays; this documents the algebra's total function) |
| `RegisterFile` | new -- the executor's program-trace container, no reactor call site yet | Track C's register model (TRACK_C_DESIGN.md Sec.1.4/1.5), gold-trace supervision format | would gate the first trace-supervised executor training run |

Deviations from the brief: `cleanup`/`candidates_for` dot-vs-cosine and
`similarity` are one shared primitive, not two separate implementations.
`abstain` returns python strings (a discrete answer-atom choice), not a
tensor. `emit`/`compare`/`branch`/`halt`/`RegisterFile` have no reactor
call site today by design -- they're the Track C executor's vocabulary,
gated on that later milestone, not this one.
