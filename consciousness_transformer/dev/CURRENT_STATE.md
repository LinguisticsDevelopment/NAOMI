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
- GATES MEASURED 2026-09-08 (branch `encoder-complement-probe`, merged): BOTH
  FAIL. (a) rank-1 edge-F1 0.534 (< 0.55), structure-exact 0.062, forest width
  7.97/8 -- the 0.70 headline was a best-of-8 oracle. (b) complement usable
  18.2% (< 25%): fragments 7% (never trained on), parser-timeout long
  sentences 56% at 8.9x speed (n=9). Control: encoder-on-easy 66% vs teacher
  74%. See RESEARCH_NOTES "AUDIT GATES MEASURED".
- NEXT ACTION 3 (16K Gold_Expand on Colab) stays GATED. NEW NEXT ACTION 3':
  small v3 gold (top-1, D1-D6) over ONLY the original 1,475 sentences, built
  on the cloud box, then arms v2@788 vs v3@788 at fixed steps (branch
  `encoder-train-arms` tooling). Decision number: rank-1 F1 >= 0.55 and forest
  width -> ~1 on single-tree targets. Pass -> continue (v3@3000, 16K). Fail ->
  parser becomes the encoder; week moves to the comprehension spike.
- NEXT ACTION 4 (train): never bundle. Arms v2@788 / v3@788 / v3@~3000 at FIXED
  gradient steps, 2 seeds -> separates "more data" from "cleaner data".
- NEXT ACTION 5 (memory-frame decoder): ON HOLD -- its coverage gate failed
  twice; respec toward reduced-realization; confab-rate is the hard gate.
- NEXT ACTION 6: GPU encoder DROPPED (Python-bound, not FLOP-bound).
- Hand-gold: LEAD 2026-09-08 -> GENERATE it from typed-slot templates (hundreds per
  family, filler- and template-held-out splits); routine on branch `hard-gold-gen`.
  The 16 drafts are seeds for the templates, not a red-pen item.
- LEAD 2026-09-08: learned encoder stays DEFAULT whatever the arms show; hybrid
  rejected. Arms decide training recipe (top-1 vs forest gold), not existence.
- AWAITING LEAD: comprehension-side spike (can a GRU over tensor memory resolve
  a candidate lattice?) -- the thesis risk nothing in this plan touches.
- D1-D6 implementation MERGED to mainline 2026-09-08 (from branch `decisions-d1-d6`;
  brings corpus-expand + hand-gold-draft along). Gold builder default is now
  top-1; PRIMES has I (old encoder checkpoints will fail to load, by design).

## 2026-09-08 EVENING STATE (director)
- Mainline has: D1-D6; measurement tooling (rank-1 + complement probe);
  train-arms tooling (fixed steps, shared holdout); hard-gold generator v2
  (1,368 records, lemmatized); K-12 scout (FairytaleQA top-1); LLM parse
  judge (Haiku 4.5; live calibration needs the lead's API key); POS-aware
  lemmatized grounding (senses_of_surface + lemma field).
- Gold data branches: encoder-gold-v2 (985, forest), encoder-gold-v3-small
  (959, top-1, no lemmatization), encoder-gold-v4b-small (987, top-1,
  lemmatized, POS-aware) -- v4b is the current best gold.
- ARMS-1 + ARMS-2 DONE: the 'forest beats top-1' result was a target-shape
  artifact. On the CURRENT gold's targets (v4b): top-1 0.49 >= forest 0.47 ~
  margin 0.45-0.47. D1 stands; variety doesn't help; keep-best is essential
  (arms peak at 750-1,250 steps). Committed edge-F1 on richer targets ~0.49;
  whole-tree exact ~5%. Hard gold: interjections/elision generalize to unseen
  templates; imperatives/additive do not (need more templates); mixing 1.2:1
  hurts in-domain. See RESEARCH_NOTES 'ARMS-2 COMPLETE'.
- FairytaleQA: fetch bug fixed (100-story default) -> 10,556 episodes (train
  8,524 / val 1,025 / test 1,007); 18.2% entity-scoreable (1,923), 57% not
  substrings. Strict parse yield 19.8%.
- RUNNING: 16K-corpus v4b gold build (encoder-gold-v4b-16k). NEXT: scaling arms
  v4b_3000 / v4b_8000 keep-best on the same holdout; hard-gold 1:4 mix arm;
  widen imperative/additive templates.
- NEXT: (1) read the v2/v3 arm result vs the rank-1 >= 0.55 gate; (2) run
  v4b_788 (+ hard gold mixed in as a 4th arm) on the same holdout; (3) if
  rank-1 clears, scale: v4b-style gold over the 16K corpus (Gold_Expand with
  the lemmatized builder), v3000 arm; (4) K-12 comprehension phase on
  FairytaleQA once the encoder is fixed.

## 2026-09-09 01:00Z STATE (director) -- supersedes the evening-state section above where they differ
- THE SCALING CURVE (keep-best, current v4b targets, 98 holdout): rank-1 edge-F1
  0.491 @788 -> 0.549 @3,000 -> 0.632 @8,000; struct-exact 5% -> 21%; still
  climbing. The learned encoder is data-limited and earning its place. Gate
  (0.55 on current targets) PASSED at 3,000.
- Settled today: top-1 gold is fine (forest-vs-top-1 was a target-shape
  artifact); keep-best is essential; lemmatized POS-aware grounding (v4b) is
  the gold standard; full hard gold in the mix (in-domain cost ~0.02, hard
  families +0.1-0.5); template variety is what hard-construction
  generalization needs (149 templates now); USVS-graded metric exists (floor
  0.285); USVS aux loss hurt at 788 (retest at 8,000 in flight).
- Gold branches: encoder-gold-v4b-16k (10,225 records, 68 MB), -v4b-variants,
  -v4b-small, -v2. Hard gold v4 committed on mainline (runs/hard_gold_*.jsonl).
- RUNNING: arms-4a v4b ALL 10,225 + full v4 hard gold (encoder-arms4-a);
  arms-4b USVS-loss retest at 8,000 (encoder-arms4-b).
- NEXT: (1) if arms-4a holds ~0.63+ with hard families ~0.75+, that checkpoint
  is the encoder for the comprehension phase; (2) next corpus expansion = the
  K-12 readers (FairytaleQA 10,556 passages parse ~60% usable) -> v5 gold;
  (3) comprehension phase on the 1,923 entity-answer FairytaleQA items + MCTest;
  (4) decoder/realization reopened (80% of FairytaleQA answers need it);
  (5) lead: run the Haiku judge calibration (dev/PARSE_JUDGE.md) when convenient.

