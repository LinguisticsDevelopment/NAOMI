# Integration plan (mid-term) — USVS into the mind

Semantic mapping is frozen (M29 USVS; see `SEMANTIC_MAPPING_PLAN.md`, historical).
This plan covers the integration arc: prove USVS viable at perception time, then
make it the meaning substrate the mind/ line actually consumes. House rules
unchanged: M24 (no metric propagates over its scored pairs), extrinsic validation
(a substrate milestone counts only when a downstream consumer benefits),
hours-not-weeks, compute walls flagged when hit rather than assumed.

## NOW — M30: the WSD gate (USVS viability test #1)

Question: do USVS sense signatures carry enough signal to disambiguate in
context? Test with ZERO training so the answer is about USVS, not about a model:

- **Resolvers compared on SemCor** (nltk download; the standard sense-annotated
  corpus):
  1. **MFS** — WordNet first sense. The floor; famously hard to beat.
  2. **USVS-sim** — training-free: context vector = mean USVS signature of the
     other content words in the sentence; pick the candidate sense whose
     signature is most similar. Pure artifact lookup + cosine.
  3. **Random** — sanity floor.
- **Report:** accuracy on all instances AND on the polysemous-only subset (MFS
  is inflated by monosemous words); per-POS breakdown.
- **Decision rule:** USVS-sim ≥ MFS on polysemous → signatures are viable; wire
  the resolver seam (parser emits lemma+context; WSD module behind a facade,
  MFS as fallback — the parser stays untrained, the resolver may later be).
  USVS-sim well below MFS → documented negative; the grounding is too coarse
  (known: much collapses to SOMETHING) and the fix is grounding depth, not
  resolver machinery.

Deliverable: `scripts/probe_wsd_semcor.py` + M30 RESEARCH_NOTES entry with the
table, either way.

**OUTCOME 2026-08-07 (M30): honest negative, MFS keeps the slot.** On 37,353
polysemous SemCor instances: MFS 0.727, best USVS variant (IDF-weighted) 0.353,
random 0.274. Signatures carry real-but-weak signal (+0.08 over random; verbs
at chance — they ground worst on the noun/adj-heavy axis inventory). Per the
decision rule: the fix is grounding depth (sourced explications, deeper
decomposition), not resolver machinery; the trained coherence resolver stays
unbuilt until signatures show more signal. M31 proceeds with MFS senses.

## M31 — USVS handles in the meaning graph (viability test #2, extrinsic)

Content-word filler/handle vectors in `meaning_graph.py` / mind/ meaning objects
become fixed deterministic projections of USVS signatures (sense chosen by the
M30 winner). Gates: (a) concept handle-dereference margins improve vs §0j
medians (0.034 @ d256); (b) one consumer metric moves (STM read-resolution or
option scoring). Neither moves → negative recorded, USVS stays a standalone
artifact, and the composition work (long-term stage 3) proceeds directly on it.

## M32 — ambiguity-bearing comprehension curriculum

Generated episodes whose answers flip with sense resolution (homographs through
the controlled grammar), so WSD quality shows up in task accuracy — and the seed
of the comprehension-question corpus shift. Gate: measurable accuracy gap
between correct-WSD and MFS-error perception on the same episodes.

## Sequencing

M30 (now, ~hours) → M31 (next session-scale task) → M32 (after M31's gate).
Long-term stages beyond these live in `ROADMAP_LONG_TERM.md`.
