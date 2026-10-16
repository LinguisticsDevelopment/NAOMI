# Milestone map (M-numbers -> goal -> outcome)

Every ledger entry before 2026-10-16 was titled with an M-number. New entries
are titled by goal instead (see `GOALS.md` at the repo root). This table is
the lookup so an old ledger entry, commit message, or code comment that still
says "M58" or "M30" can be placed against the current goal list. Pre-M17 work
(the `0a`-`0r` sections of `RESEARCH_NOTES.md`) was never M-numbered; it is
the original engineering substrate (clause reactor, meaning graph, mind/
build) that every later milestone sits on.

| M# | Goal | One-line outcome |
|---|---|---|
| M17 | Read (grounding) | Derived minimal word-meaning basis + clause==word reduction operator. MEASURED. |
| M18 | Read (grounding) | Scaled to 10k corpus; polarity coords, multi-signal basis. MEASURED. |
| M19 | Read (grounding) | Unified meaning space: minimum axes from all relational signal. MEASURED. |
| M20 | Read (grounding) | Honest, normalized meaning space. MEASURED. |
| M21 | Read (grounding) | Null-aware bipolar representation; win on unrelated-pairs, honest negative on per-word antonyms. |
| M22 | Read (grounding) | Sense nodes (WSD by construction) + fused Null+propagation. MEASURED. |
| M23 | Read (grounding) | Denser sense-correct close edges + leakage correction to M22. MEASURED. |
| M24 | Read (grounding) | Full leakage audit across M17-M23 — the standing house rule ("no metric scores its own training pairs") originates here. |
| M25 | Read (grounding) | Retrain placement on all signals: honest negative. |
| M27 | Read (engineering) | Deleted the legacy token stack (paid down the `0h` debt). |
| M28.0 | Read (grounding) | WordNet-only baseline audit. MEASURED. |
| M28.1 | Read (grounding) | Nine-signal sweep (3 parallel agents): 2.5 wins / 6.5 negatives. |
| M29 | Read (grounding) | USVS — the Universal Semantic Vector Space artifact. BUILT + MEASURED. This is the grounding finish line. |
| M30 | Resolve (WSD) | WSD gate on SemCor vs MFS floor. Robust honest negative. |
| M31 | Read (bridge) | USVS as filler/handle vectors in the meaning graph: both extrinsic gates PASS. |
| M32 | Resolve (WSD) | Ambiguity curriculum: sense choice becomes a scored task. Gate PASS. |
| M33 | Read (bridge) | mind-line USVS handle adoption through the graph API. |
| M34 | Resolve (WSD) | First trained WSD sense chooser; 3/4 cross-word transfer. |
| M35 | Read (parser) | Template-generalization audit: no overfitting; parser is the surface bottleneck. |
| M36 | Read (parser) | Parser robustness round 1: 12->18 constructions. |
| M37 | Read (grounding) | 31 homograph families + first scaling curve. |
| M38 | Read (parser) | Parser round 2: passive voice + SUBJECT tie-break; 18->20. |
| M39 | Read (parser) | Parser stress battery round 3 map. |
| M40 | Read (grounding) | Chooser on raw USVS axes: projection exonerated, grounding indicted. |
| M41 | Read (grounding) | WordNet-backed tag lexicon: ends hand-listing (the recipe later reused for Spanish). |
| M42 | Read (grounding) | Grounding-depth diagnosis: why the dead families are dead. |
| M43 | Read (encoder) | First scaling curve: data-limited, not capacity-limited. |
| M44 | Read (grounding) | Gloss content-word enrichment: honest negative; M24 earns its keep. |
| M45 | Read (parser) | Parser round 4 (agent pair): stress battery 23/34 -> 29/34. |
| M46 | Read (grounding) | Senses regrounded in the space: USVS as the metalanguage. |
| M47 | Read (parser) | Parser round 5 (Sonnet agent): stress battery 29/34 -> 32/34. |
| M48 | Read (parser) | Real-text failure map: 89% subject-coverage, 2-arg ceiling exposed. |
| M49 | Read (parser) | Wh-questions land: stress battery saturates at 34/34. |
| M50 | Read (parser) | 2-arg extraction ceiling falls + adapter flags. |
| M51 | Read (parser) | Real-text grammar trio: coverage 89.0% -> 94.5%. |
| M52 | Resolve | Model consumes full argument structure (resolver plan phase 1). |
| M53a | Resolve | The membrane exists: candidate sets + anti-recency pronoun data. |
| M53 | Resolve | First collapse: Track A resolves reference perfectly; Track B binds but interferes. |
| M54 | Resolve | Sense collapse joins the membrane: pipeline works, curriculum doesn't force binding. |
| M54b | Resolve | Memory-context WSD works — the M30 rematch, won honestly. |
| M54c | Resolve | Diagnosis: pronouns were capacity-limited; senses want an interaction prior. |
| M55a | Resolve | Parse-hypothesis membrane + honest ambiguity census. |
| M55 | Resolve | v1 membrane gate: pronouns and senses proven at full scale; readings leak again. |
| M-ES1 | Any language | Spanish Freeze Test stage 1: interlingua claim holds at the stream level. |
| M56 | Reason (executor) | Track C design spike verdict: "more research"; one measured finding against M53's halo. |
| M56b | Resolve | Memorization confirmed, membrane fixed, generalization restored. |
| M56c | Resolve | Re-gate: generalizing pronoun binding restored to 1.000 — closes the M56 arc. |
| M57a | Remember | Entity-instance subsystem built: registry + attribute facts + provenance. |
| M57b | Remember | Resolver-driven write-back proven at full scale (after one leak caught and fixed). |
| M57c (battery #1) | Remember | Write-back replicates; instance episodes fail even under forced-gold — the read path is the gap. |
| M57c (battery #2) | Remember | Read path proven (0.423->0.856); resolver lacks an evidence-target interaction. |
| M57 (battery #3) | Remember | Schema proven on rich episodes; explicit binding shown necessary at 8 entities. M57 closes. |
| M58a | Answer from memory | Corpus->episode converter + prose failure taxonomy: real text parses 5.8%. |
| M58c | Read (parser) | Parser round: real-text parse rate 5.8% -> 18.6% strict / 52.3% usable. |
| M59 | Remember (LTM) | Episodic LTM proven: consolidation carries facts across passages; identity linking 1.000. M59 closes. |
| M60 | Answer from memory | Track A confirmation: recency + abstain proven at scale; frozen checkpoint shipped. |
| M58b | Answer from memory | THE ZERO-SHOT PROSE NUMBER: 0.583 vs 0.250 floor; wild text 0.389; honest abstention. |
| M58d | Answer from memory | Grounding fix + quality filter: eval got more honest, frozen number moved sideways. |
| M58 | Answer from memory | THE BIG-N ZERO-SHOT PROSE NUMBER — the sprint's central measurement. |
| M58f / M58f2 / M58g | Answer from memory (infra) | Prose-training hang chased through three layers (parser caps, quadratic GC, a third uncapped call site) to the real fix: a hoisted default parse-time cap. |
| M61 | Answer from memory | Prose-training payoff: does reading real text help it read real text — measured. |
| M62 / M62b | Read (encoder) | Teacher gold-volume probe: encoder-distillation approach is GO. |
| M63.1 / M63.1c | Read (encoder) | Encoder gold landed (teacher-bulk + hand-authored hard cases). |

## Milestones never folded into this ledger's M-numbering

- The `mind/` build (schema, LTM-as-graph, executor, controller, two loops) used
  its own **M0-M7** table — see `MIND_ARCHITECTURE.md`'s "Build roadmap". That
  table is a design-doc roadmap, not a ledger history; it is unaffected by this
  cleanup.
- The Executor LOFO gates (#1 inconclusive, #2 GO) and everything from the
  2026-09-04 encoder/decoder build onward through the 2026-10-15 audits and
  today's fixes were never given M-numbers at all — the ledger entries speak
  for themselves and are findable by date or by grepping the ledger for their
  own headline text.
