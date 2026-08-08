# Semantic Mapping — the finish line (middle-term plan)

Goal: **finish the meaning-space definition** — a minimum-redundancy, deterministic,
named-axis space with every English WordNet sense placed in it — and ship it as a
versioned artifact plus a sense→coordinate dictionary. This is the substrate the
mind/ line will consume (see `ROADMAP_LONG_TERM.md` for what happens after).

Method constraints (settled by M17–M25, do not relitigate):
- Signals enter only as **typed relational edges** feeding deterministic placement
  (`ground/placement.py`). Never free trained dimensions (M25 closed door).
- Every signal/axis justifies itself **held-out** (M24 rule: a metric that
  propagates over the pairs it scores is leaked by construction — thread
  `train_pairs=` everywhere).
- "Minimum redundancy" = the contribution-ranked minimality curve
  (`ground/honest_minimality.py`), re-run after each signal lands.
- Axes never rotate or mix; every axis keeps its name.

Expected cost per step: **hours of local CPU**, mostly minutes. If a step hits a
real compute wall (full-sense placement is the only candidate), flag it — don't
pre-engineer for it.

## Step A — WordNet-only minimization audit (IMMEDIATE)

Before adding anything: establish exactly where the current signal set
(`RelationGraph`: synonym, antonym, similar, is_a, meronym, derivational,
verb_group + lexname/attribute features) tops out, on one fixed corpus and split.

- Re-run held-out placement + dictionary reconstruction + honest minimality on the
  current code (post-M27), one table, one seed policy: syn AUC, syn>ant, hypernym
  containment, random overlap, minimality peak-K.
- Break results out **by POS region** (noun/verb/adj) — the hypothesis for Step B
  is that the verb region is under-signaled (126 attribute axes are adjectival;
  verbs only get lexname + verb_group).
- Record as the **M28.0 baseline** in RESEARCH_NOTES. Every Step B signal is
  judged as a delta against this table.

Existing probes to reuse: `scripts/probe_placement.py`, `scripts/probe_normalize.py`
(honest minimality path), `scripts/probe_ground_understanding.py`.

## Step B — incremental signals, one at a time, each behind an ablation

Order (cheapest first; each lands as a `wordnet.py` wrapper + a `RelationGraph`
relation + a held-out ablation vs the M28.0 table; keep only what moves a metric;
losers become documented negatives):

1. **WordNet remainder** (same corpus, no new downloads):
   - verb `entailment()` / `causes()` edges (close/directed),
   - `pertainyms()` (adj→noun, cross-POS close),
   - domain links (`topic_domains()`, `region_domains()`, `usage_domains()`) —
     candidate *feature* relations (axis-naming, like lexname),
   - `also_sees()`.
2. **VerbNet classes** (nltk download) — only if Step A confirms the verb region
   is weak: Levin-class co-membership as verb close-edges; class names as
   candidate verb axes.
3. **FrameNet frames** (nltk download) — only if VerbNet moves the needle:
   frame co-membership edges; frame names as candidate event axes.

Out of scope for now (recorded, not forgotten):
- **OMW / multilingual colexification** — future cross-language bridge, not needed
  to finish the English space.
- **Corpus co-occurrence / distributional anything** — thesis violation (opaque
  axes). Permanently excluded as a *placement* signal.
- **ConceptNet** — crowd noise; at most an extra held-out eval set later.

Gate for Step B as a whole: the final signal set's held-out table + a re-run
minimality curve; update the M28.x entries in RESEARCH_NOTES as each signal lands.

## Step C — publish the space + the dictionary

- Scale placement from the ~3k gloss corpus to **all WordNet senses** (~117k
  synsets). `DecompCache` handles decomposition (10k warm ≈ 9s); propagation is
  the part to batch. Measure and report timing — flag if it's genuinely a wall.
- New `ground/space.py`: `build_space()` → a versioned on-disk artifact holding
  (named axes, per-sense value + Null mask, signed antonym-edge store, provenance:
  corpus hash + signal set + code version); `load_space()` → the query API
  (coordinate lookup, masked similarity, antonym-aware closeness).
- **The dictionary**: every English WordNet sense → its placed coordinate, dumped
  in a readable format (one artifact file + a human-browsable export, e.g. JSONL:
  `sense_id, lemmas, gloss, {axis: value}` on applicable axes only). This is the
  "big dictionary mapping English senses onto the space."
- Gates: deterministic rebuild (build twice, byte-identical); held-out metrics at
  full scale reproduce the small-corpus table (no scale artifact); load time
  seconds, not minutes.

## Done means

A frozen, versioned space artifact + sense dictionary, with an honest table saying
exactly what the space captures (synonymy/hypernymy/similarity/unrelatedness) and
what it delegates to structure (antonymy → signed edges; composition → operations).
After that, mapping work stops and integration work starts (`ROADMAP_LONG_TERM.md`).

---

**STATUS 2026-08-07: DONE — all three steps shipped.** Step A = M28.0 baseline;
Step B = M28.1 sweep (2.5 winners → `signal_combined`); Step C = **M29 USVS**
(`ground/usvs.py`, `scripts/build_usvs.py`): 9,946-word placed core on 607 named
axes, 117,659 senses grounded sparse, tiered antonym store (11,317) + directed
genus edges (1,546), dictionary.jsonl.gz, 72s deterministic build, fingerprint
`72b00a67c2b9daca`, scale-validated held-out at 10k (synonym 0.885, similar
0.803, random 0.233 with the winners). Semantic mapping is frozen; work moves to
integration (`ROADMAP_LONG_TERM.md` stages 1–2: USVS handles into the meaning
graph, then the WSD-vs-MFS gate).
