# LTM design brief: episodic long-term memory + consolidation

Decision doc, not a survey (2026-08-30). No code touched. Read order behind
this brief: `MIND_INTERFACE.md` (v2 locked design + entity-instance
addendum), `ROADMAP_LONG_TERM.md` (Spanish Freeze Test, Advice Test),
`AURORA_SPRINT.md` (2026-08-30 reprioritization), `RESEARCH_NOTES.md`
M55-M57c, `src/nsm_ct/{entity_memory,instances,clause_reactor}.py`.

## 0. Where memory stands today (measured, not aspirational)

`ClauseReactor.forward` (`clause_reactor.py:1964`) does
`memory = em.init_memory(b, d, device)` as its first line — a fresh all-
zero `[B,d,d,d]` tensor **every forward call**. Nothing survives past the
episode. There is no LTM in the active track. `entity_memory.py` is the
whole STM substrate: parameter-free order-3 TPR ops (`query`, `query_entity`
the M57c.2 inverse read, `write` with gate/overwrite/negate). `instances.py`
(M57a) adds an `InstanceRegistry` (enumeration index, zero knowledge) and
attribute facts as ordinary writes (`e_instance ⊗ attr:<name> ⊗ value`),
plus a `ProvenanceLog` — but registry and log are both **per-episode**
objects today; nothing hands them forward.

**Prior art that exists but doesn't plug in**: `src/nsm_ct/mind/knowledge.py`
(`KnowledgeGraph`, M1) + `mind/subconscious_loop.py` (`SubconsciousLoop.
consolidate`, M4) are a full symbolic-graph LTM with STM→LTM consolidation,
forward-chain offline inference, and disk persistence (`mind/persistence.
py`) — built and measured (RESEARCH_NOTES §0q, M4). It runs on a *different*
substrate (`MeaningGraph` codec-handle nodes, not the entity⊗relation⊗value
tensor + `InstanceRegistry` the current v1/v2 track uses) and was never
reconciled with clause_reactor. An even older `lifelong.py` `LongTermMemory`
(RESEARCH_NOTES §4, "two tiers built") is a third, FIFO-pruned generation,
also unconnected. None of the three tracks talk to each other. Treat the
mind/ graph as reusable machinery (forward-chain, Horn rules, persistence),
not a drop-in LTM for the current substrate.

## 1. What LTM must do (binding constraints, cited)

- **Multi-passage reading.** AURORA_SPRINT's 2026-08-30 reprioritization,
  priority 2: "episodic LTM + consolidation (multi-passage reading needs
  it)" is explicit line item, sequenced before the M58 prose test.
- **Cross-episode identity is a decision, not a collision.**
  MIND_INTERFACE.md v2 addendum point 4: "Cross-episode identity ('is this
  mary THAT mary?') becomes an explicit consolidation-time decision (LTM
  linking), not a string collision." Point 1-3 already fixed the
  *within*-episode version (fresh instance atoms, attribute facts,
  attribute-match candidate generation) — LTM is the *across*-episode
  extension of the same mechanism, not a new one.
- **Truth-memory promotion has a named seed.** `episode.py:_level5`
  (`CurriculumGenerator`, docstring lines 128-129) is corroborate/
  contradict today, but it is an STM-only, single-episode, emergent-trust
  curriculum (no LTM, no explicit gate — RESEARCH_NOTES §0: "whom to trust
  demonstrably emerges in memory, but no scalar gate is forced to carry
  it"). MIND_INTERFACE.md's "what exists/what's new" table calls
  truth-memory promotion "new (seed exists: L5 corroborate/contradict)" —
  the gate that seed lacks IS the LTM promotion decision.
- **Provenance/trust dials gate promotion, not just STM writes.**
  `trust_ltm` (STM→episodic LTM) and `trust_truth` (STM/LTM→truth memory,
  "strictly higher bar than trust_ltm") are named dials in the locked
  design (MIND_INTERFACE.md "the dials" table). `instances.py`'s
  `ProvenanceLog` already carries `(source, language, timestamp, trust,
  step)` per write — LTM promotion is a second gated write reusing that
  same record shape, source = "consolidation" instead of "reactor".
- **Spanish Freeze Test / Advice Test must not be foreclosed.** Freeze
  Test: LTM content must be **language-neutral** (synset-keyed sense
  handles, instance atoms, attribute relations — never raw strings) so a
  Spanish passage's consolidated facts are indistinguishable from an
  English passage's (ROADMAP §"Spanish Freeze Test": "transfers for free:
  ... entity atoms (language-neutral by design)"). Advice Test: procedures
  become memory content later ("skills themselves migrate from weights
  into memory... stored as a procedure by the same consolidation
  machinery"); LTM's write/promote contract must not be typed to *facts*
  only in a way that can't extend to *procedures* — trust dials "applied
  to PROCEDURES" is already flagged as a design consequence.
- **Invariants that bind the LTM design** (MIND_INTERFACE.md, locked): #1
  knowledge in structures, weights hold policy only (LTM must be an
  inspectable structure, not a bigger/second GRU); #3 abstain, never
  silently drop uncertainty (an unlinked cross-episode identity must
  surface as a candidate set or "MAYBE", never a silent merge or silent
  duplicate); #4 every write gated/local/auditable (promotion writes need
  the same provenance discipline as STM writes); #5 one resolver contract
  (identity linking should be a resolver problem — candidate set + margin
  — not a bespoke matcher); #6 dials explicit named scalars (trust_ltm/
  trust_truth, not implicit thresholds); #8 compute budgeted (recall
  cost must not scale unboundedly with corpus size); #9 the loop runs on
  structured clauses, not tokens (LTM facts must round-trip through the
  same USVS/instance vector space, not a side string table).

## 2. Candidate designs

| | learned vs deterministic (inv. #1) | identity linking + dial | inference cost | abstention (inv. #3) | provenance (inv. #4) | capacity behavior | first milestone |
|---|---|---|---|---|---|---|---|
| **A. Tensor LTM** | Second `[d,d,d]` tensor (`em.init_memory`), same ops. Consolidation = gated copy of instance-keyed facts (unchanged `em.write`, gate from `trust_ltm`). Zero new learned params — same op algebra as STM. | Attribute-match against the LTM registry via `instances.candidates_for` (already built), threshold = `trust_ltm`-adjacent new dial (`link_threshold`). Below threshold → new instance minted (no merge). | O(1) per query, same einsum as STM; LTM registry enumeration cost grows O(N) per candidate generation (same shape as STM's `candidates_for`, already the mechanism instances.py uses). No inference-time weight growth. | Sub-threshold match → CandidateSet with both "same as X" and "new instance" as options, resolved by the *existing* resolver contract (inv. #5 for free). | `ProvenanceLog` reused verbatim; one record per promoted write, `source="consolidation"`. | Same failure mode as STM: superposition/interference grows with facts-per-instance-slot as the registry accumulates across episodes — this is the one tensor shared by ALL episodes ever seen, so it is the accumulation-stress case the capacity probe (below) must characterize BEFORE this option is trusted past toy scale. | Two-passage curriculum: passage 1 introduces mary#1 (occupation=doctor), passage 2 (new episode) mentions "mary" + a disambiguating clause; question requires the LTM-linked fact. Cheat baseline: no-consolidation arm at chance. Honesty arm: held-out name pairs (M56b-style) + a forced-wrong-link arm. |
| **B. Symbolic graph LTM** | Instance nodes + attribute edges + provenance-per-edge, written by DETERMINISTIC consolidation rules (not learned); tensor demoted to a per-passage working cache, rebuilt fresh each passage from the graph. Cleanly satisfies inv. #1 (nothing about the graph is learned) but pushes the *linking decision itself* toward a learned resolver score over graph edges, same as A. | Attribute-match over graph edges (exact lookup, no interference) + the resolver's margin/threshold — same dial, cleaner substrate (no cosine noise from tensor superposition). | Recall must re-materialize a `[d,d,d]` tensor per passage from the relevant graph slice — an extra deterministic pass per passage, plus a graph query (which nodes are "relevant") that itself needs a policy (relevance is not free either). | Graph naturally represents "no edge exists" as absence, and an ambiguous edge set as a real candidate list — arguably CLEANER abstention than A's cosine threshold. | Provenance IS a graph edge attribute (already the `mind/knowledge.py` pattern) — richer than a side log, queryable. | Exact retrieval, no interference — capacity is bounded by memory/disk, not by tensor rank, so this is the design that scales best raw-passage-count-wise. Cost moves from "recall accuracy degrades" to "which subgraph do I re-materialize" (a new problem, not a smaller one). | Same two-passage curriculum as A, but the interesting first measurement is DIFFERENT: not "does linking work" but "does re-materializing a tensor from a graph slice reproduce the same STM behavior as A's native tensor" — a byte-identity-style regression against A on passage 1 alone. |
| **C. Hybrid (tensor recall + graph audit)** | Tensor is the fast-path LTM (as in A); graph is written IN PARALLEL, off the training critical path, purely for audit/identity-linking evidence — the graph never feeds the loss. | Linking done off the tensor (cosine, as A) but VERIFIED/logged against the graph's exact attribute index — a two-source check (a linking decision the tensor made can be audited exactly, not just trusted). | Same as A on the hot path (tensor ops only); graph write is an O(1) side effect per consolidation event, off the gradient path — no training-time cost added. | Same as A, with the graph available as a tie-breaker when the tensor's cosine margin is itself low (a genuine use for the graph beyond audit). | Best of both: tensor's `ProvenanceLog` (already built) for the fast path, graph edges for a redundant, exactly-queryable second copy. | Same capacity risk as A on the hot path; the graph gives a way to VALIDATE whether tensor capacity is degrading (compare tensor-answer vs graph-ground-truth as corpus size grows) — i.e., C turns "the capacity probe" into a standing regression instrument, not a one-off measurement. | Same curriculum as A, PLUS one extra metric for free: tensor-vs-graph agreement rate as a function of corpus size — this IS the capacity curve, generated as a byproduct instead of a separate probe. |
| **D. (not recommended, recorded for completeness) Reuse `mind/knowledge.py` KnowledgeGraph directly** | Fully deterministic, already built, already measured (M1-M4). | Referent-node dedup by label — i.e., identity IS the string, the EXACT defect M57a's instance atoms were built to fix. | Cheap (graph query). | Truth-tagged clauses (`meta["truth"]`) give a form of abstention but not the four-form contract (candidate sets / margins) inv. #3 requires. | Native (meta dict per clause). | Untested at the entity-instance / attribute-fact schema; built for a 7-relation toy curriculum, not instance-keyed facts. | N/A — would require porting M57a's instance/attribute schema onto `MeaningGraph` nodes first, which is most of option B's build cost anyway, with none of option B's tensor-parity story. |

Capacity probe (dev/CAPACITY_CURVE.md does not exist yet): what it must
measure before A or C is trusted past a two-passage toy — recall accuracy
(or resolver-linking accuracy) vs. (a) number of consolidated instances,
(b) number of attribute facts per instance, (c) number of passages
consolidated, at fixed `d`. AURORA_SPRINT's standing engineering note
applies directly: the order-3 tensor is `B×d³` with autograd history
(~5-8GB per training arm at 1500 episodes, already OOM-prone at 4 parallel
arms on 15GB per RESEARCH_NOTES M57c battery #1) — an LTM tensor that
accumulates across MANY episodes inside one training run is a second,
likely worse instance of the same footprint problem, and the probe must
report memory footprint alongside accuracy, not accuracy alone.

## 3. Recommendation

**Lean C (hybrid), with A as the fallback if the graph side proves to be
schedule risk.** Reasoning: A alone inherits STM's known interference
ceiling with no instrument to detect when it's degrading; B alone adds a
whole new "which subgraph to re-materialize" policy problem before any
recall works at all, and duplicates work the M57a registry+tensor pair
already does. C gets A's zero-new-training-cost hot path plus a free,
exact ground-truth to validate it against — which is exactly the
measurement the project's own "perfect-looking results get held-out
tests" house rule (CLAUDE.md) will demand of an LTM headline number
anyway. But this is a recommendation, not a decision — the questions below
are the actual fork points.

Decisions the lead must make:

1. **Does LTM get its own tensor, or does STM get a "keep across episodes"
   flag?** Default: separate tensor (option A/C as scoped). Consequence of
   the alternative (one tensor, longer-lived): simpler code, but STM's
   per-episode `init_memory` reset (the thing that currently keeps
   training memory bounded) disappears, and the AURORA memory-footprint
   problem (priority 1, already blocking longer episodes) gets strictly
   worse before it gets measured. **Hyper-critical** — this changes the
   sequencing of the 2026-08-30 reprioritization (footprint engineering
   is priority 1 for a REASON).
2. **Is cross-episode identity linking a NEW resolver head, or does it
   reuse the existing entity/instance resolver contract with LTM
   candidates added to the same candidate set?** Default: reuse (inv. #5,
   "one resolver contract"). Consequence of a new head: functionality
   sooner (no risk of LTM candidates confusing the in-episode resolver's
   training signal), but a second specialist head to maintain and a
   second held-out-name gate to run — and it's the exact kind of
   proliferation the Track A/B experiment (MIND_INTERFACE.md §3) exists
   to measure against. **Hyper-critical** — touches invariant #5 directly.
3. **What promotes STM→LTM: an explicit new `trust_ltm`-gated op, or does
   consolidation ride the existing write/overwrite/negate gate machinery
   with different inputs?** Default: new op (consolidation is a distinct
   moment — end of episode/passage — not a per-clause write). Consequence
   of reusing per-clause gates: less new code, but "when does consolidation
   fire" becomes implicit instead of a named event, which weakens the
   audit story invariant #4 wants ("what was written, when, from what
   evidence"). **Routine** — doesn't change any invariant, changes only
   how much new surface area ships.
4. **Does truth-memory promotion (the L5 seed) get built now, alongside
   episodic LTM, or does episodic LTM ship first and truth-memory waits?**
   Default: episodic LTM first, truth-memory promotion deferred — L5's
   trust dial is currently emergent-in-content (RESEARCH_NOTES §0's
   documented gap: "no scalar gate is forced to carry it"), and giving it
   a real gate is a second, separable piece of work with its own honesty
   arms. Consequence of building both at once: one bigger, harder-to-gate
   milestone; the M57c pattern (battery caught a real gap under
   forced-gold) argues for shipping the smaller piece first. **Routine**
   — sequencing only, no invariant or roadmap stage depends on the order.
5. **Does the capacity probe get its own milestone before any consolidation
   code is trained, or does it ride along with the first LTM milestone's
   evaluation?** Default: ride along (option C's design makes the
   tensor-vs-graph agreement curve a byproduct, not a separate probe, so
   the marginal cost of measuring capacity at the same time is small).
   Consequence of a standalone probe first: an honest curve before any
   training investment, at the cost of one more milestone before
   consolidation code exists at all — directly trades off against the
   2026-08-30 "stop requiring MINIMAL episodes" priority, which wants
   forward progress on richer episodes now. **Hyper-critical** if the
   probe is skipped entirely (an LTM headline number without a capacity
   curve is exactly the kind of "perfect-looking result" CLAUDE.md's
   house rules require a held-out test for) — routine if it's merely
   sequenced later vs. rolled in.

## 4. No-regret work (buildable before the decision)

- **Thread `InstanceRegistry` + `ProvenanceLog` across episodes**, i.e.
  make them constructor-injectable into whatever drives multi-episode
  training/eval instead of freshly created per-episode — pure plumbing,
  needed by every option above, decided by none of them.
- **A `link_threshold` dial stub** (named, unused until an option is
  picked) alongside `trust_ltm`/`trust_truth` in whatever module
  eventually owns the dials — keeps dial-naming discipline (inv. #6)
  ahead of the implementation.
- **The capacity probe's harness** (episode generator: N instances × M
  attributes × P passages, at fixed `d`) — useful for measuring options A
  and C identically, and useful standalone even if the recall mechanism
  changes underneath it.
- **A cross-episode identity curriculum generator** (two passages, same
  name, sometimes the same referent / sometimes a different one, with a
  disambiguating clause) — the eval harness for whichever design ships,
  buildable now since it only depends on `episode.py`'s existing
  generator patterns and M57a's instance/attribute schema, not on LTM
  code.
- **A read-only audit of `mind/knowledge.py` + `mind/persistence.py`**
  for salvageable pieces (Horn-rule storage, disk save/load format) BEFORE
  option B/C's graph component is scoped in detail — cheaper to know now
  whether persistence is reusable than to rediscover it mid-build.

## 5. DECISIONS (user, 2026-08-30) — locked

1. **Separate LTM tensor.** STM keeps its per-episode reset; LTM is a second
   order-3 memory that persists across passages of a document/session.
   Reads are ADDITIVE: mem_read = query(STM) + query(LTM) (one vector
   space; a fact in either store is readable). Writes go to STM only.
2. **Identity linking reuses the existing resolver contract** (invariant
   #5): LTM instances join the ENTITY candidate set with a "from_ltm"
   feature and a named `link_threshold` dial; scored by the same head
   that resolves "she"/"the doctor". No new head.
3. **Consolidation is a substate machine**, not a per-clause gate:
   READING → WIND-DOWN → CONSOLIDATE, fired at end of passage (later:
   when the patience budget winds down). A named `consolidate` op pushes
   only facts worth keeping — provenance trust ≥ `trust_ltm` dial —
   into LTM, merging the registry (instance ids persist) and appending
   provenance records tagged with the consolidation event.
4. **Episodic LTM first; truth-memory promotion after** — but the
   consolidate op is TIER-GENERIC from day one (tier N → N+1 with a dial
   and an evidence criterion), so LTM→Truth later reuses the same code
   with `trust_truth` + a corroboration-count criterion.
5. Capacity probe: DONE standalone (dev/CAPACITY_CURVE.md) before any
   consolidation training — resolved.
