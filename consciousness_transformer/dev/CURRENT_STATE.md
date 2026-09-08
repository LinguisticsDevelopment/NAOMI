# NAOMI — current state & decisions (2026-09-07 handoff)

Read this + `RESEARCH_NOTES.md` (full chronological ledger) to resume. Mainline
branch: `claude/m27-m28-cleanup`. Work happens in worktree
`NAOMI/.claude/worktrees/repo-cleanup`. Local box CANNOT run torch/parser — all
compute runs in cloud routines (RemoteTrigger, env `env_01B5c2PVb7HM6ejZPD5mCtqM`,
CPU) or Colab (the lead runs those). English-only training; Spanish is the
held-out zero-shot grammar-swap exam.

## Component status (one line each)
- **ENCODER** (`src/nsm_ct/encoder_model.py`): WORKS. The old "structure 0 /
  over-generates" verdict was a `beam_decode` **aliasing bug** (shared open-clause
  dict across beam forks) — FIXED + merged (deepcopy per fork). Re-scored run-2
  ckpt: edge_precision 0.70, overgen 1.08, structure 0.21. It is **DATA-LIMITED**
  (scaling curve: edge-F1 0.565→0.573→0.682 for n=200/400/788; tf_acc rising,
  no flattening). Scheduled sampling = ~neutral (the tf-vs-decode gap is error
  compounding, not exposure bias). LEVER = more + cleaner gold.
- **DECODER** (`src/nsm_ct/decoder_trained.py`): the old function_vocab realizer
  is broken (memorizes 2047 content words → confabulates). SUPERSEDED by the
  memory-frame design (`dev/DECODER_MEMORY_DESIGN.md`) — NOT built yet. Metric is
  now **structure-match round-trip** (encode→decode→re-encode→edge-F1), evaluator
  built (`scripts/eval_structure_match.py`, branch `decoder-structmatch`); its
  ceiling is encoder-limited (gold-text control ~0.5-0.8), so a better encoder
  raises the decoder's training signal too. Verbatim token round-trip is RETIRED.
- **GOLD**: two-part = TEACHER-BULK ∪ HAND-AUTHORED hard cases.
  - richer extraction (modifiers DESCRIPTION/SPECIFICATION/COMPLEMENT) MERGED to
    mainline (`421f450`); downstream consumers taught to ignore modifier roles.
  - corpus EXPANDED 11× (1,475 → 16,411 sentences; branch `corpus-expand`) with
    public-domain children's-lit (Grimm/Andersen/Jacobs/McGuffey + more).
  - `colab/Gold_Expand.ipynb` regenerates gold over the corpus → `encoder_gold_v3`
    (Drive-safe). DO NOT run it yet — gate on the decisions below first.

## DECISIONS LOCKED (2026-09-07) — implement these before the gold build
1. **Forest → top-1.** QA showed current gold averages 3.46 trees/sentence (61%
   have 3+); the extra trees are SPURIOUS (trivial modifier-attachment wobble /
   dropped clauses), not genuine ambiguity. `build_encoder_gold_v2` must default
   to keeping only the top-1 parse per sentence (keep 2+ ONLY on a wide score gap;
   ties are spurious). Token-level SENSE candidate-sets stay (candidates-first
   preserved where real). Genuine structural ambiguity = hand-curated, not top-k.
2. **Keep all corpus sources** (Grimm/Jacobs parse ~50% vs Burgess 82%, but a
   failed parse is lost gold not noise; top-1 pruning cleans the rest).
3. **D3 — add prime `I`** to `encoder_model.PRIMES` (canonical NSM set has I+YOU);
   `me` → I. Unblocks the imperative/fragment "me" cases.
4. **D4 — add role labels QUANTITY / ADDITIVE / FOCUS** to the role vocab. These
   are STRUCTURAL relation labels only; they do NOT touch USVS sense coordinates
   (lead's explicit caveat). For "More!" (QUANTITY), "me too"/"and a hat too"
   (ADDITIVE/FOCUS).
5. **D2 — interjections ground to a real sense**, never bare `entity`. Content
   interjections (shit/damn/nonsense) use their WordNet/USVS sense; PURE
   interjections (ugh/alas/oh/ah) get GLOSS-grounded into USVS via the existing
   gloss→coordinate pipeline. utterance_kind = `interjection`. NO appraisal node
   (connotation is comprehension-side, read off GOOD↔BAD).
6. **D6 — elision slots retain their surface carrier.** "the dog did." keeps
   `did`'s token_index so tense/polarity survives; the elided predicate is a
   context_ref to the antecedent verb.

## Branches (all pushed, none but mainline merged unless noted)
- `claude/m27-m28-cleanup` — MAINLINE (has aliasing fix, richer extraction, all metrics/tooling).
- `corpus-expand` — 16,411-sentence corpus + `Gold_Expand.ipynb` + builder.
- `hand-gold-draft` — `dev/HAND_GOLD_DRAFT.md` (16 candidates) + `scripts/hand_gold.py` (human-writable authoring helper, gate-validated). AWAITING lead red-pen.
- `gold-qa` — the QA report (per-source quality + forest analysis).
- `decoder-structmatch` — structure-match evaluator.
- `encoder-schedsamp` (+ `-results`) — scheduled-sampling + scaling-curve.
- `encoder-gold-v2` / `spanish-gold-v2` — the current gold data (gitignored, force-added).
- `colab-trained-ckpts-v2` — run-2 encoder+decoder checkpoints.

## NEXT ACTIONS (execution order)
1. **Implement DECISIONS 1-6** (one routine, on mainline+corpus-expand): forest
   top-1 prune in `build_encoder_gold_v2`; add `I` prime + QUANTITY/ADDITIVE/FOCUS
   roles to `encoder_model`; gloss-ground pure interjections in `build_usvs`;
   elision-carrier retention in `clause.py`/hand_gold. Keep all tests green.
2. **Lead reviews `HAND_GOLD_DRAFT.md`**, red-pens the 16; then finalize the
   hand-authored hard-case gold set.
3. **Run `Gold_Expand.ipynb`** (lead, Colab) → `encoder_gold_v3` (top-1, richer,
   clean). Union with the hand-authored set.
4. **Train encoder** on a ~3-5K subset of the new gold at ~30-50 epochs (CPU
   feasible; full 16K needs a GPU-capable encoder — deferred). Re-score edge-F1;
   expect a jump past 0.68 per the data-limited curve.
5. **Build the memory-frame decoder** (`dev/DECODER_MEMORY_DESIGN.md`), trained
   against structure-match reward + copy-from-structure no-confab gate.
6. (deferred, needs lead) GPU-capable encoder to exploit the full 16K; K-12
   comprehension phase.

## AUDIT ADDENDUM (2026-09-08) — read dev/AUDIT_2026-09-08.md
Independent audit verdict: CONTINUE WITH CHANGES. Changes to the plan above:
- NEXT ACTION 3 (Gold_Expand) is GATED on two measurements now in flight
  (branch `encoder-complement-probe`): (a) rank-1 committed-tree edge-F1 >= 0.55
  (the 0.70 headline is best-of-8 oracle); (b) encoder usable-rate >= 50% on
  sentences the TEACHER FAILS (its only value beyond parser distillation).
- NEXT ACTION 4 (train): never bundle. Arms v2@788 / v3@788 / v3@~3000 at FIXED
  gradient steps, 2 seeds -> separates "more data" from "cleaner data".
- NEXT ACTION 5 (memory-frame decoder): ON HOLD -- its coverage gate failed
  twice; respec toward reduced-realization; confab-rate is the hard gate.
- NEXT ACTION 6: GPU encoder DROPPED (Python-bound, not FLOP-bound).
- Hand-gold 16: treat as held-out TEST set per family, not training data.
- AWAITING LEAD: comprehension-side spike (can a GRU over tensor memory resolve
  a candidate lattice?) -- the thesis risk nothing in this plan touches.
- D1-D6 implementation dispatched 2026-09-08 (branch `decisions-d1-d6`).
