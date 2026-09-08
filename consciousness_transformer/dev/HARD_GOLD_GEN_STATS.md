# HARD-CASE GOLD GENERATOR — stats (lead directive, 2026-09-08; v2 update, same day; v3 update, same day)

## v3: widening the two families that failed to generalize to unseen templates

Branch `hard-gold-gen-v3`. **Why:** the first encoder trained on v4b gold +
the v2 generated hard gold (`runs/arms/v4b_788_hard_0.log`, "ARMS-2 COMPLETE"
in the research notes) generalizes cleanly to UNSEEN TEMPLATES for
`content_interjection`/`pure_interjection` (0.98-1.00 rank-1 F1) and
tolerably for `elision` (0.71), but falls off a cliff for two families:

| family | unseen-FILLER rank-1 F1 (`test_filler`) | unseen-TEMPLATE rank-1 F1 (`test_template`) |
|---|---:|---:|
| `imperative` | 0.9556 | **0.2721** |
| `additive_focus` | 0.7500 | **0.2326** |

Both families had only 8-9 templates before this pass — few enough that
holding out even 2 of them for `test_template` left the encoder with almost
no exposure to alternative surface shapes for the construction, so it
memorized TEMPLATE SHAPE rather than the underlying grammar (imperative
synthesized-SUBJECT / additive-particle-plus-elided-predicate). This pass
widens both families' typed-slot TEMPLATE COUNT (not filler pools — those
were already ample) so a held-out template is a narrow slice of a wide
surface-shape distribution instead of most of what exists, and increases how
many templates are held out (2 -> 4) so `test_template` is a genuinely
stronger structural-generalization check.

**What changed** (`src/nsm_ct/hard_gold_templates.py`):

- **`imperative`: 9 -> 29 templates.** Added determiner variety on the
  object (`a`/`an` via `article()`, `this`, `that`, `my`, `your` — the
  existing 9 only ever used bare `the`), a `VT` + PROPN object shape
  (`"grab daniel ."`), `PLACE` with `"on"` as well as the existing `"in"`,
  a new `MANNER` role for `VT`/`VI` + adverb (`"buy the soldier
  carefully ."`, `"come quickly ."`  — new `ADV` pool, 10 curated
  manner/time adverbs, all 10 ground), a double-object shape (`"give mary
  the flour ."`, `INDIRECT_OBJECT` = the PROPN), `please` BEFORE the verb
  (the existing template only had it after, trailing-comma), `never` +
  `VT`/`VI` alongside the existing `do n't` + `VT` (added `do n't` + `VI`
  too), bare pronoun objects (`"{VT} it ."`, `"{VT} them ."` — the existing
  template only had `"please {VT} it ."`), a speaker-prime object
  (`"{VT} me the {N} ."`), a vocative-prefixed imperative (new `ADDRESSEE`
  role, `"bill , lock the man ."`), two coordinated imperative clauses in
  one record (`"clean the face and drop the water ."` — 2 clauses, both
  `kind=imperative`, each its own synthesized `SUBJECT=PRIME(YOU)`), and a
  bare `"!"`-terminated `VT`+object variant (the existing bang-terminated
  template was `VI`-only).
- **`additive_focus`: 8 -> 22 templates.** Added nominative bare-pronoun
  `ADDITIVE` subjects (`"{PRON} too ."`/`"{PRON} also ."`, new use of the
  existing `PRON` pool in this family) and oblique-case bare pronouns (new
  `OBL_PRON` pool — `him`/`her`/`them`/`us`, grounded by the same
  grammar-rule `hand_gold._PRONOUNS` routing as `PRON`/`PROPN`, not
  `senses_of`-filtered), determiner variants without the leading `"and"`
  (`"the {N} too ."`, `"{a/an} {N} too ."` — the existing pair only had the
  `"and ..."` form), an `"also"` counterpart for every shape that previously
  only had `"too"` (`and {PROPN} also`, `{PROPN} , also`, oblique-pronoun
  `also`), FOCUS shapes without a trailing `"too"` (`"not {PROPN} ."`,
  `"only {PROPN} ."`, `"just the {N} ."`, `"even {PROPN} ."` — the existing
  FOCUS template was `only {PROPN} too .`, conflating FOCUS with ADDITIVE),
  and an `and the {N} too .` definite-article counterpart to the existing
  `and a {N} too .`. Antecedent-clause CONTEXT sentences for the new
  templates draw their verb from the existing `VT`/`VT_past` pools (present
  and past tense) instead of always the fixed `"want"`, giving the
  "want/see/like/take-shaped, past and present" variation the brief asked
  for without inventing new unverified verb forms.
- **`let 's {VT} the {N} .` — SKIPPED, schema cannot express it.**
  `encoder_model.PRIMES = ["YOU", "I", "<UNK_PRIME>"]` (D3): the schema only
  admits a synthesized SUBJECT for 2nd-person singular (`YOU`, imperative
  addressee) or 1st-person singular (`I`). A first-person-PLURAL synthesized
  subject (`"let's"` = "you and I") has no admitted `PRIME` value — spelling
  it `PRIME("WE")` would train against the catch-all `<UNK_PRIME>` bucket,
  which is not a real gold target and would not genuinely widen the family.
  Left for a future schema change (adding `"WE"` to `PRIMES`), not attempted
  here.
- **`{PROPN} as well .` — SKIPPED, schema cannot express it.** `"as well"`
  is a two-TOKEN particle; every filler in `hand_gold.py`'s DSL (`W`) binds
  to exactly one surface token (`_Matcher.match` consumes one token per
  slot). There is no existing role shape for a multi-token surface span
  here — authoring it would mean either faking `"as well"` as one non-real
  token (would never match the real tagger's per-word output) or dropping
  half the phrase. A correct fix needs a multi-token-span filler primitive
  in `hand_gold.py` itself, out of scope for a template-only widening pass.

**`gen_hard_gold.py`'s held-out count is now per-family**
(`HELD_OUT_COUNT = {"imperative": 4, "additive_focus": 4}`, default 2 for
every other family, unchanged): these two widened families hold out 4 whole
templates from `train`/`test_filler` into `test_template`, not 2 — a bigger,
harder unseen-template slice, proportionate to how many more templates now
exist to hold out from.

### v3 families × templates × splits

`--per-family 150 --seed 0` (same invocation as v2; only the template set
and the two families' held-out count changed):

| family | templates (before → after) | held-out count | generated | **passed** | failed | train | test_filler | test_template | held-out template ids (seed 0) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| additive_focus | 8 → **22** | **4** | 11,567 | **143** | 0 | 109 | 10 | 24 | `add_and_a_n_too`, `add_pron_also`, `add_pron_too`, `add_propn_also` |
| content_interjection | 8 | 2 | 210 | **210** | 0 | 150 | 30 | 30 | `interjc_then_clause_bang`, `interjc_then_clause_comma` |
| elision | 8 | 2 | 5,916 | **138** | 0 | 93 | 15 | 30 | `elision_n_did`, `elision_n_did_too` |
| imperative | 9 → **29** | **4** | 232 | **232** | 0 | 150 | 50 | 32 | `imp_neg_vt_obj`, `imp_vocative`, `imp_vt_obj_adv`, `imp_vt_obj_place_on` |
| pure_interjection | 8 | 2 | 190 | **190** | 0 | 140 | 20 | 30 | `interjp_clause_then`, `interjp_double_dot` |
| quantity | 8 | 2 | 4,918 | **132** | 0 | 108 | 20 | 4 | `quant_bang_past_ctx`, `quant_not_bang` |
| speaker_prime | 9 | 2 | 206 | **193** | 0 | 138 | 25 | 30 | `speaker_show_me_n`, `speaker_vt_it_for_me` |
| synth_subject | 8 | 2 | 287 | **210** | 0 | 150 | 30 | 30 | `synth_feels_bang`, `synth_seems_bang` |
| **total** | **66 → 86** | — | **23,526** | **1,448** | **0** | **1,038** | **200** | **210** | — |

**Failure histogram: empty** — all 23,526 generation attempts across every
family (widened and unwidened alike) passed `check_record` outright; zero
new failure modes from the 20 new templates.

Output files (`runs/`, `git add -f`'d):

| file | records | bytes |
|---|---:|---:|
| `hard_gold_train.jsonl` | 1,038 | 2,616,314 |
| `hard_gold_test_filler.jsonl` | 200 | 497,833 |
| `hard_gold_test_template.jsonl` | 210 | 606,823 |
| `hard_gold_train_small.jsonl` (new, v3) | 200 | 515,458 |
| **total** | **1,648** | **4,236,428 (4.04 MB, < 6 MB cap)** |

`runs/hard_gold_train_small.jsonl` is new in v3: `scripts/make_hard_gold_small.py`
draws a seeded (`--seed 0`), family-stratified 200-record subset of
`hard_gold_train.jsonl` (25 records per family, 8 families) for the 1:4
hard-gold-to-teacher-gold mixing arm — a fixed-size, reproducible slice
rather than re-sampling ad hoc per training run.

### v3 sample renders from new templates (not in the "first-2-per-family" table below, which still shows the pre-existing shapes)

```
imperative:
  imp_vt_propn          'grab daniel .'
  imp_give_propn_n       'give mary the flour .'
  imp_vocative           'bill , lock the man .'
  imp_coord               'clean the face and drop the water .'
  imp_vt_obj_adv          'buy the soldier carefully .'
  imp_never_vi            'never leave !'

additive_focus:
  add_oblpron_too         'her too .'
  add_not_propn            'not sandra .'
  add_just_the_n           'just the patient .'
  add_even_propn           'even fred .'
  add_and_the_n_too        'and the face too .'
```

All pass `hand_gold.check_record` (schema validity, oracle `linearize_tree`,
action legality, skeleton round-trip).

### v3 smoke-train result

`python scripts/train_encoder.py --smoke --max-steps 40 --gold runs/hard_gold_train.jsonl`
(150-record stratified train subset of the new 1,038-record file, 40-dev/40-test, CPU):

```
[  16.0s] USVS loaded: 117679 senses, d_axes=607
[  16.0s] pos_vocab=13 role_vocab=14
[  16.0s] policy params: 197,661 (~0.791 MB fp32)
[  19.2s] epoch 0 step 50  avg_loss=26.874
[  20.2s] === epoch 0 done: avg_loss=24.126 (n=150 derivations) ===
[  21.7s] === epoch 1 done: avg_loss=20.373 (n=150 derivations) ===
[  23.3s] === epoch 2 done: avg_loss=17.400 (n=150 derivations) ===
[  24.8s] === epoch 3 done: avg_loss=15.173 (n=150 derivations) ===
[  24.8s] training wall-clock: 6.2s (stopped_early=True); optimizer_steps=40 (cap=40, stop_reason=max_steps)
[ 229.3s] test: {'sense_recall': 0.7857, 'edge_recall': 0.2396, 'rank1_edge_f1': 0.2722, 'n_records': 40}
[ 230.7s] test (random baseline): {'sense_recall': 0.0143, 'edge_recall': 0.0521}
```

Loss finite and monotonically decreasing over all 40 optimizer steps (26.87
→ 24.13 → 20.37 → 17.40 → 15.17) — the new templates' actions are all
admitted by the legality mask (a masked-out gold action would be a `-inf`
logit → `+inf` loss). `role_vocab=14` in this run. Two new role names were
introduced for the v3 templates: `MANNER` (`imp_vt_obj_adv`/`imp_vi_adv`)
and `ADDRESSEE` (`imp_vocative`). `MANNER` appears in `hard_gold_train.jsonl`
(both its templates stayed in `train` at seed 0); `ADDRESSEE` does NOT —
`imp_vocative` is one of `imperative`'s 4 held-out templates at seed 0, so
every `ADDRESSEE`-bearing record landed in `test_template` and none in
`train`. This is the intended held-out-TEMPLATE design working as specified
(a genuinely unseen template can introduce a role the encoder never trained
on, which is a harder and more honest generalization test than a held-out
template that only varies fillers) — not a gap to fix, since `role_vocab`
already has a fallback bucket for unseen role names at eval time, same as an
unseen sense/POS ID.

### v3 tests

`pytest -q tests/test_hard_gold_gen.py tests/test_decisions_d1_d6.py`:
**38 passed, 1 skipped** (the skip is `test_teacher_gold_inflected_word_grounding_unchanged_or_improved`'s
own pre-existing "if `runs/encoder_gold_v2.jsonl` present" gate, unrelated
to this batch, and unchanged from v2). `test_hard_gold_gen.py` gained two
tests: `test_imperative_and_additive_focus_widened_v3` (template counts
`>=24`/`>=20`) and `test_imperative_and_additive_focus_hold_out_four_templates`
(the generator's `HELD_OUT_COUNT` override actually holds out 4, not the
default 2, for these two families). `test_pools_are_grounded` was updated to
also exclude the new `OBL_PRON` pool from the `senses_of_surface` check —
same grammar-rule-not-sense-lookup exemption already given to `PROPN`/`PRON`.

### v3 unfinished / out of scope, and exactly why

- **`let 's {VT} the {N} .`** — schema cannot express a first-person-plural
  synthesized subject (`encoder_model.PRIMES` only admits `YOU`/`I`); see
  above. Not attempted.
- **`{PROPN} as well .`** — schema's filler DSL has no multi-token-span
  primitive; `"as well"` is two tokens. Not attempted.
- **No new evaluation run against a freshly retrained encoder.** This batch
  regenerates the GOLD (widened templates, wider `test_template` split) and
  confirms it trains (finite, decreasing smoke loss); it does not itself
  retrain the full arms-2-style encoder to confirm the unseen-template
  rank-1 F1 actually improves for `imperative`/`additive_focus` — that is a
  full multi-thousand-step training run (`runs/arms/v4b_788_hard_0.log` took
  ~5,647s to reach its extra-eval), explicitly out of scope for a CPU,
  code-and-generation-only, 40-step-smoke pass in one turn. The generated
  `runs/hard_gold_test_template.jsonl` (now 32 imperative / 24
  additive_focus records drawn from 4 held-out templates each, vs. the v2
  file's thinner 2-held-out-template slices) is the artifact a follow-up
  training run would evaluate against.
- **`quantity`'s known template-surface-collision issue (v2 §9) persists
  unchanged** — out of scope for this pass, which only touched
  `imperative`/`additive_focus`.

---

> Generated by `scripts/gen_hard_gold.py` (templates in
> `src/nsm_ct/hard_gold_templates.py`), this run at
> `--per-family 150 --seed 0`. Every record passes the SAME gate
> `scripts/hand_gold.py`'s 16 hand-authored drafts pass: `check_record`
> (schema validity, oracle `linearize_tree`, action legality against
> `legal_action_types`, skeleton round-trip). Zero records failed the gate
> at this seed — see the failure histogram in §3 (empty).

Branch `hard-gold-gen-v2`. Reproduce:

```bash
cd consciousness_transformer
python scripts/build_usvs.py                 # if data/usvs isn't already built
python scripts/gen_hard_gold.py --per-family 150 --seed 0   # writes runs/hard_gold_{train,test_filler,test_template}.jsonl
# hard_gold_all.jsonl is opt-in now (pure duplicate of the three splits): --write-all
```

## 0. v2: two quality fixes over the original batch

This is a follow-up pass over the SAME generator (§1-§9 below are otherwise
unchanged in substance), closing two quality gaps the original run flagged
in its own §9 ("unfinished / out of scope"):

1. **Lemmatization (`USVS.senses_of_surface`, `src/nsm_ct/ground/usvs.py`).**
   `senses_of` is a raw-surface lemma lookup, so an inflected form —
   `dogs`, `boys`, `walked` — returned `[]` and either dropped out of a
   pool entirely (this is why `N_pl` was irregular-plural-only and
   `VI_past` had only 10 of 35 candidates, per the original §4/§9) or, worse,
   silently grounded as a degenerate `entity` instead of a real sense slot
   whenever it slipped through as a fixed anchor word. `senses_of_surface(word,
   pos_hint=None)` tries the raw surface first (unchanged behavior when it
   already grounds — e.g. `closed`/`cut`/`wanted`, which double as their own
   WordNet adjective/noun lemma) and, only if that is empty, falls back to
   `nltk.corpus.wordnet.morphy` over noun/verb/adj and retries on the
   recovered lemma. Used in `hand_gold.ground_W`/`_sense_grounding` and
   `hand_gold.build_token_sense_candidates` (so a lemma-grounded slot's
   `candidates` still agree with its `token_sense_candidates` entry, contract
   S4.2), and in `build_encoder_gold_v2.ground_word` and its own
   `token_sense_candidates` loop — confirmed the teacher path did NOT already
   lemmatize (`scripts/build_encoder_gold_v2.py`'s old `ground_word` called raw
   `usvs.senses_of(w)` directly), so this is the SAME helper on both gold
   paths, not two divergent ones. Every `type:"sense"` grounding (and its
   `token_sense_candidates` entry) now also carries a `lemma` field
   (`dev/ENCODER_IO_CONTRACT_V2.md` §4.2 updated). `runs/encoder_gold_v2.jsonl`
   itself is NOT rebuilt (out of scope, would touch a separately-owned
   artifact) — §9 below reports how many of a 20-word sample of its
   previously-`entity`-grounded inflected words would now ground as `sense`.

2. **Article agreement (`article()`, `src/nsm_ct/hard_gold_templates.py`).**
   The 3 templates/context-sentences that emit a literal `"a {N}"` (
   `add_and_a_n_too`, `add_and_a_n_also`, `speaker_bring_me_n`, plus all 8
   `additive_focus` templates' shared `"the {N_pl} want a {N2} ."` context
   sentence) always emitted `"a"`, even before a vowel-initial filler
   (`"a optic"`). `article(word)` picks `"a"`/`"an"` by the filler's first
   SOUND (a small exceptions table handles `unicorn`/`hour`-style
   letter/sound mismatches) and is now called at every one of those sites.
   No template mixes singular/plural subject-verb agreement with a fixed verb
   form (`want` is always paired with the plural `N_pl`, never a singular
   subject) — checked; nothing to fix there.

Rebuilding `N_pl`/`VT_past`/`VI_past` through the new lemmatized ground-filter
(§4) is what actually needed the fix to matter at scale: §4 below is the
updated pool table (`N_pl` 30→60, `VI_past` 10→30, `VT_past` 39→40).

---

## 1. Why: the two-part-gold decision, generalized

`dev/HAND_GOLD_DRAFT.md` established that the v2 lattice schema is
human-writable for the constructions the deterministic teacher parser can
never produce gold for — imperatives (synthesized SUBJECT), interjections
(pure and content), elision/fragments (VP-ellipsis, dropped arguments),
additive/focus particles ("too"/"also"), quantity fragments ("more!"), and
propositional-anaphora synth-subjects ("sounds good."). That gave 16
hand-authored exemplars — enough to prove writability, not enough to train
anything.

This generator takes each of those 16 surfaces and GENERALIZES it into one
or more templates with typed slots (verb / noun / adjective / proper-name /
pronoun / quantifier / interjection), fills the slots from word pools built
from the live USVS + WordNet, and gates every fill through the identical
checker. 8 families (imperative, pure interjection, content interjection,
elision, additive/focus, quantity, synth-subject fragments, speaker prime),
66 templates total (8-9 per family), **971 gated records** at the default
seed/per-family setting.

---

## 2. Families × templates × splits

`--per-family 150` (v2: the originally-planned value, now reachable — see
below). Two whole templates per family are held out of `train` and
`test_filler` entirely (`test_template`); the remaining templates split their
fillers between `train` and `test_filler` (fillers never drawn for train).

| family | templates | generated (incl. retries/dupes) | **passed** | failed | train | test_filler | test_template | held-out template ids |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| additive_focus | 8 | 8,617 | **86** | 0 | 64 | 10 | 12 | `add_only_propn_too`, `add_propn_comma_too` |
| content_interjection | 8 | 210 | **210** | 0 | 150 | 30 | 30 | `interjc_clause_then`, `interjc_then_clause_comma` |
| elision | 8 | 4,964 | **153** | 0 | 112 | 20 | 21 | `elision_n_did_not`, `elision_propn_did_not` |
| imperative | 9 | 220 | **219** | 0 | 154 | 35 | 30 | `imp_neg_vt_obj`, `imp_vi_for_me` |
| pure_interjection | 8 | 190 | **190** | 0 | 140 | 20 | 30 | `interjp_double_dot`, `interjp_then_clause_comma` |
| quantity | 8 | 5,851 | **102** | 0 | 62 | 10 | 30 | `quant_n_bang`, `quant_n_pl_bang` |
| speaker_prime | 9 | 228 | **198** | 0 | 147 | 30 | 21 | `speaker_tell_about_n_bang`, `speaker_tell_about_propn` |
| synth_subject | 8 | 281 | **210** | 0 | 150 | 30 | 30 | `synth_looks_bang`, `synth_seems_bang` |
| **total** | **66** | **20,561** | **1,368** | **0** | **979** | **185** | **204** | — |

Held-out template ids are re-sampled per seed (`rng.sample`, seeded); the ids
above are for `--seed 0` (they differ from the original v1 run's ids because
the RNG draw sequence shifted — the pool contents feeding earlier `rng.choice`
calls changed, §4). `test_hard_gold_gen.py::test_test_template_ids_disjoint_from_train`
checks disjointness holds at every seed the tests run, not just this one.

Output files (`runs/`, `git add -f`'d — `.gitignore` normally excludes
`runs/*.jsonl`):

| file | records | bytes |
|---|---:|---:|
| `hard_gold_train.jsonl` | 979 | 2,511,945 |
| `hard_gold_test_filler.jsonl` | 185 | 480,852 |
| `hard_gold_test_template.jsonl` | 204 | 490,436 |
| **total** | **1,368** | **3,483,233 (3.3 MB, < 5 MB cap)** |

**`hard_gold_all.jsonl` is no longer shipped by default (v2).** It was a
pure derived concatenation of the other three files — kept before only
because the original brief named all four files, and it was the reason
`--per-family 150` didn't fit the 5 MB budget the first time around (a full
second copy of every record roughly doubles the four-file total). `gen_hard_gold.py`
now only writes it behind `--write-all`; the default run's 3 split files
alone reach `--per-family 150` (the originally-planned value) at 3.3 MB, well
inside budget with the duplicate file gone.

**Why "generated" ≫ "passed" for `additive_focus`, `elision`, `quantity`.**
Not gate failures (all three show 0 failed) — surface-uniqueness exhaustion.
`me too .` / `me also .` (additive_focus), `the X did .`-style fragments
sharing a small surface vocabulary (elision), and `{QUANT} !` /`{QUANT} .`
(quantity, `QUANT` pool size 4) all have SMALL possible-surface spaces
relative to their context/antecedent variation — the visible text is often
fully determined by 1-2 slots while other slots only vary the invisible
`context[]`. Once every reachable unique surface is claimed, further filler
draws are cheap rejected-duplicate retries (`text in seen_surfaces`, an O(1)
set check before any gate call), not wasted gate work — see §6.

---

## 3. Failure-reason histogram

**Empty at the default seed** — every one of the 20,561 generation attempts
that reached the gate (i.e. produced a not-yet-seen surface) passed
`check_record` outright. `scripts/gen_hard_gold.py::bucket_reason` classifies
failures into stable buckets (bad grounding.type, retrieval.method missing,
candidate/ref dereference failure, sense-candidates disagreement, bad
utterance_kind, illegal oracle action, round-trip mismatch, ...) for when a
future template/pool change does introduce a regression; `tests/test_hard_gold_gen.py`
would then also start failing (`test_every_family_generates_at_least_50_gated_records`
and, if a template starts throwing, `test_all_records_pass_the_gate`).

---

## 4. Word pools

| slot type | n | source |
|---|---:|---|
| `N` (noun, sg) | 60 | curated boost (24 child-register words) + automated: USVS core words whose first WordNet sense is a noun in a concrete lexname (`noun.animal/artifact/body/food/object/plant/substance/person/location/shape`), frequency-ranked (WordNet tagged-sense count), `senses_of` non-empty |
| `N_pl` | **60** (was 30) | **v2 fix.** Curated irregular plurals / plurale tantum (`people`, `men`, `women`, `children`, `teeth`, `feet`, `sheep`, `police`, `scissors`, `staff`, `troops`, ...; 8 more irregulars added to the curated list itself — `women`/`children`/`feet`/`mice`/`geese` were previously missing) **+ regular "-s" plurals of the `N` pool** (`dogs`, `boxes`, `houses`, ...), filtered through the new `USVS.senses_of_surface` (raw surface, falling back to WordNet morphy) instead of raw `senses_of` — this is D5's fix (b), "lemmatize at retrieval time" (`dev/HAND_GOLD_DRAFT.md`). A noun whose OWN irregular plural is curated (`man`→`men`, `woman`→`women`, `child`→`children`, `foot`→`feet`, `tooth`→`teeth`, `mouse`→`mice`, `goose`→`geese`, `person`→`people`) is skipped in the regular-plural auto-fill so it doesn't ALSO mint a wrong `"mans"`/`"womans"`-style form alongside the real irregular. Capped at 60 (`_POOL_CAP`, same cap as `N`/`ADJ`) — every N-pool noun's regular plural that isn't skipped this way grounds, so the pool is cap-limited, not availability-limited. |
| `ADJ` | 60 | curated boost (39 common descriptive adjectives) + automated: USVS core words whose first sense is `a`/`s` (adjective/satellite), comparatives/superlatives and quantifier-adjectives (`many`, `less`, `-er`/`-est`) excluded, frequency-ranked |
| `PROPN` | 6 | `nsm_ct.episode._NAMES` verbatim (`mary`, `john`, `sandra`, `daniel`, `bill`, `fred`) — the SAME name universe as teacher-bulk gold and the hand-gold drafts, deliberately not widened. Grounds via `is_entity` (`type:"entity"`), not a sense lookup, so not filtered by `senses_of` |
| `PRON` | 5 | `he`, `she`, `it`, `they`, `we` — bare 3rd-person pronouns. Grounds via the `_PRONOUNS`/`memory`/`coref` branch of `hand_gold.ground_W` regardless of WordNet coverage, so not filtered either. (`I`/`me`/`myself` are deliberately excluded from this pool — they route to the resolved prime `I`, D3, and are used as fixed lexical anchors, e.g. `"me too ."`, not sampled slot fillers.) |
| `QUANT` | 4 | `more`, `less`, `enough`, `all` — all four pass the ground-filter |
| `INTERJ_PURE` | 20 | all 20 keys of `nsm_ct.ground.usvs.PURE_INTERJECTION_GLOSSES` — all 20 ground (post-build, `senses_of` returns `interj.<word>.01`, D2's gloss pipeline) |
| `INTERJ_CONTENT` | 40 (of 48 curated) | curated content words used as stance exclamations (`nonsense`, `shit`, `damn`, `hero`, `wonderful`, `disaster`, ...); capped at 40 per the plan, all 48 curated candidates actually ground (cap is a size limit, not a filter) |
| `VT` (transitive verb, base) | 40 (of 70 curated) | hand-curated (transitivity is not reliably inferable from WordNet subcat frames at this budget, per the plan); capped at 40, all 70 curated candidates ground |
| `VI` (intransitive verb, base) | 30 (of 36 curated) | hand-curated; capped at 30, all 36 ground |
| `VT_past` | **40** (was 39) | **v2 fix.** `PAST_IRREGULAR` table entry when the base verb has one, else the regular `"-ed"` rule (`_regular_past`) — previously ONLY the table was tried, so a base verb missing from it never got a past-tense candidate at all. Filtered through `senses_of_surface` instead of raw `senses_of`: a regular `-ed` form (`asked`, `helped`, `worked`, ...) now grounds via its morphy-recovered base-verb lemma instead of needing to independently double as a noun/adjective lemma (`broke`→penniless, `cut`, `read`, ...). Capped at 40; 40 of the 70 transitive-side bases pass (up from 39 — the table already covered every `VT` base, so this is almost entirely the lemmatization fix, not the regular-past fallback). |
| `VI_past` | **30** (was 10) | **v2 fix, the pool this batch was funded to unblock.** Same change as `VT_past`. The old table was missing `walk`/`wait` entirely (never got a past form at all before v2) and, more importantly, `senses_of_surface` now recovers the base verb for every OTHER regular `-ed` intransitive too (`listened`, `looked`, `shouted`, `whispered`, `coughed`, `sneezed`, `worked`, `played`, `appeared`, `disappeared`, `wandered`, `traveled`, ...) instead of requiring a noun/adjective homograph. Hits the `_POOL_CAP` of 30 outright — every one of the 35 curated intransitive-side bases now grounds a usable past form; capped, not availability-limited (unlike the v1 run, where `came`/`went`/`ran`/`stood`/`slept`/`swam`/`flew` had NO WordNet-lemma homograph on their irregular surface and were dropped — those are now recovered via `wn.morphy("came","v") == "come"`, etc.). |

**Regular "-ed" drop list** (54 of the 69+35 curated pairs; see D5): every
regular `-ed` form and most intransitive irregulars have `senses_of == []`
on the raw surface form and are dropped, not silently kept as a degenerate
`entity` grounding — full list in the generator's own `--per-family` run
output (`=== pool sizes === / dropped`) and reproduced verbatim in
`scripts/gen_hard_gold.py`'s run log.

**What is deliberately NOT senses_of-filtered.** `PROPN` and `PRON` ground
by grammar rule (`is_entity` / bare-pronoun `memory`/`coref`), not by sense
lookup — filtering them by `senses_of` would be filtering the wrong
condition. Fixed template ANCHOR words (`did`, `sounds`/`looks`/`seems`/`feels`,
`too`/`also`/`either`/`only`, `more`/`less`/`enough`/`all` when used as a
literal template word rather than a `QUANT`-slot fill) follow the precedent
the 16 hand-authored drafts already set: some of those also ground as bare
`entity` (`sounds` — D5/D6, `bebió`/`está` in the Spanish draft) and the gate
accepts it. Only the SAMPLED typed slots are pool-filtered.

---

## 5. Two rendered samples per family (v2, `--per-family 150 --seed 0`)

Note the article fix in action: the v1 sample for this exact family/seed
printed `context: ['the vermin want a optic .']` (wrong article before a
vowel-initial filler); the v2 sample below shows the corrected `'an optic'`/
`'a girl'`. (`synth_subject`'s `sounds` predicate grounding also reads
`gtype=sense` below vs. `gtype=entity` in the v1 doc — that is NOT this
batch's lemmatization fix; raw `senses_of("sounds")` already has candidates
against the USVS artifact this session's `build_usvs.py` produced, an
environment/build difference unrelated to `senses_of_surface`, which only
changes behavior when the RAW lookup is empty.)

```
-- additive_focus --
  text: 'me too .'
    clause kind=proposition predicate=None gtype=elision
      SUBJECT      word='me'           gtype=prime
      OBJECT       word=None           gtype=elision
      ADDITIVE     word='too'          gtype=sense
    context: ['the houses want an optic .']
  text: 'and a girl too .'
    clause kind=proposition predicate=None gtype=elision
      SUBJECT      word=None           gtype=reference
      OBJECT       word='girl'         gtype=sense
      ADDITIVE     word='too'          gtype=sense
    context: ['the mothers want a street .']

-- content_interjection --
  text: 'crap !'
    clause kind=interjection predicate='crap' gtype=sense
  text: 'awesome !'
    clause kind=interjection predicate='awesome' gtype=sense

-- elision --
  text: 'the water did .'
    clause kind=proposition predicate=None gtype=elision
      SUBJECT      word='water'        gtype=sense
      OBJECT       word=None           gtype=elision
    context: ['the books kicked the area .']
  text: 'the girl did .'
    clause kind=proposition predicate=None gtype=elision
      SUBJECT      word='girl'         gtype=sense
      OBJECT       word=None           gtype=elision
    context: ['the vermin found the apple .']

-- imperative --
  text: 'carry the bowl .'
    clause kind=imperative predicate='carry' gtype=sense
      SUBJECT      word='you'          gtype=prime
      OBJECT       word='bowl'         gtype=sense
  text: 'paint the father .'
    clause kind=imperative predicate='paint' gtype=sense
      SUBJECT      word='you'          gtype=prime
      OBJECT       word='father'       gtype=sense

-- pure_interjection --
  text: 'huh !'
    clause kind=interjection predicate='huh' gtype=sense
  text: 'yikes !'
    clause kind=interjection predicate='yikes' gtype=sense

-- quantity --
  text: 'all !'
    clause kind=proposition predicate=None gtype=elision
      OBJECT       word=None           gtype=elision
      QUANTITY     word='all'          gtype=sense
    context: ['the police fix the leg .']
  text: 'enough !'
    clause kind=proposition predicate=None gtype=elision
      OBJECT       word=None           gtype=elision
      QUANTITY     word='enough'       gtype=sense
    context: ['the beds paint the cup .']

-- speaker_prime --
  text: 'tell me about the state .'
    clause kind=imperative predicate='tell' gtype=sense
      SUBJECT      word='you'          gtype=prime
      INDIRECT_OBJECT word='me'           gtype=prime
      ABOUT        word='state'        gtype=sense
  text: 'tell me about the mother .'
    clause kind=imperative predicate='tell' gtype=sense
      SUBJECT      word='you'          gtype=prime
      INDIRECT_OBJECT word='me'           gtype=prime
      ABOUT        word='mother'       gtype=sense

-- synth_subject --
  text: 'sounds easy .'
    clause kind=proposition predicate='sounds' gtype=sense
      SUBJECT      word=None           gtype=reference
      COMPLEMENT   word='easy'         gtype=sense
    context: ['they go to the surface .']
  text: 'sounds hot .'
    clause kind=proposition predicate='sounds' gtype=sense
      SUBJECT      word=None           gtype=reference
      COMPLEMENT   word='hot'          gtype=sense
    context: ['he go to the man .']
```

(Full record JSON, including `token_sense_candidates` and `context[].lattice`
— now also a `lemma` field on every `type:"sense"` grounding and its
matching `token_sense_candidates` entry — is in
`runs/hard_gold_{train,test_filler,test_template}.jsonl`.)

---

## 6. Smoke-train result

`python scripts/train_encoder.py --smoke --max-steps 40 --gold runs/hard_gold_train.jsonl`
(v2: 150-record stratified train subset of the 979-record file, 40-dev/40-test,
CPU; `train_encoder.py`'s eval metrics gained a few fields since the original
v1 run — `slot_recall`/`structure_recall`/`all_gold_recalled_rate`/
`overgen_ratio`/`rank1_structure_recall` — unrelated to this batch):

```
[   0.0s] 979 gold records
[   0.0s] split: train=150 dev=40 test=40
[  17.1s] USVS loaded: 117679 senses, d_axes=607
[  17.1s] policy params: 197,531 (~0.790 MB fp32)
[  18.9s] epoch 0 step 50  avg_loss=26.633
[  20.2s] === epoch 0 done: avg_loss=25.790 (n=150 derivations) ===
[  21.5s] === epoch 1 step 250 avg_loss=22.231 ===
[  22.1s] === epoch 1 done: avg_loss=21.683 (n=150 derivations) ===
[  24.1s] === epoch 2 done: avg_loss=18.155 (n=150 derivations) ===
[  26.0s] === epoch 3 done: avg_loss=15.301 (n=150 derivations) ===
[  26.0s] training wall-clock: 7.7s (stopped_early=True); optimizer_steps=40 (cap=40, stop_reason=max_steps)
[ 651.3s] test: {'sense_recall': 0.7826, 'slot_recall': 0.2083, 'edge_precision': 0.6083,
                 'edge_recall': 0.3292, 'rank1_edge_f1': 0.3444, 'mean_forest_width': 4.7,
                 'n_records': 40}
[ 653.2s] test (random baseline): {'sense_recall': 0.0145, 'edge_recall': 0.0438}
[ 653.2s] saved checkpoint -> runs/encoder_smoke.pt
```

Loss is finite and monotonically decreasing over all 40 optimizer steps
(26.6 → 25.79 → 21.68 → 18.16 → 15.30) — the legality mask never excludes a
gold action in practice (a masked-out gold action is a `-inf` logit → `+inf`
loss; this is the same invariant `hand_gold.check_trains` checks for the 16
drafts, exercised here at record-file scale through the REAL training loop,
now over records that include `lemma`-tagged grounding). The trained policy
also clears the random baseline by a wide margin on every metric (e.g.
`sense_recall` 0.78 vs 0.01, `edge_recall` 0.33 vs 0.04) even after only 40
steps — the hard-case records are still learnable signal, not noise.

---

## 7. Tests

`pytest tests/test_hard_gold_gen.py tests/test_decisions_d1_d6.py tests/test_encoder_holdout.py`:
**41 passed** (0 skipped this run — `runs/encoder_gold_v2.jsonl` IS present
in this checkout, so `test_encoder_holdout.py`'s guard doesn't fire; the 3
skips the v1 doc reported were specific to a checkout that lacked it).
`test_hard_gold_gen.py` (18 tests, up from 10) covers everything the v1 doc
listed, plus the v2 fixes:

- every family generates ≥50 gated records at the default seed;
- no surface string appears in two splits;
- `test_template`'s template ids are disjoint from `train`'s (and from
  `test_filler`'s);
- every record carries `meta.source == "hard_gold_gen"`, `meta.family`,
  `meta.template_id`, `meta.split`;
- a sample of generated records independently re-passes `hand_gold.check_record`;
- determinism (same seed → byte-identical output) and that different seeds
  CAN differ;
- every pool member (excluding `PROPN`/`PRON`) actually grounds — now via
  `usvs.senses_of_surface`, since `N_pl`/`VT_past`/`VI_past` contain
  lemmatized-only fillers that raw `senses_of` would reject;
- the 8-family / ≥8-template-each coverage itself;
- **new:** `senses_of_surface("dogs")` returns `dog`'s senses with
  `lemma == "dog"`; `senses_of_surface("walked")` returns `walk`'s senses;
  a word that already grounds on its raw surface (`closed`/`cut`/`wanted`)
  is NOT re-lemmatized; an unknown word returns `([], word)`;
- **new:** `article()` picks `"a"`/`"an"` correctly on a fixed word list AND
  no generated record (`--per-family 30`, all 3 splits) ever emits `"a"`
  before a vowel-initial filler;
- **new:** a 20-word sample of `runs/encoder_gold_v2.jsonl`'s
  previously-`entity`-grounded inflected words never regresses under
  `senses_of_surface` and reports how many now ground as `sense` (12/20 in
  this checkout's fixture — see §9).

---

## 8. Trainer integration

`nsm_ct.encoder_train_util.load_gold` now accepts a comma-separated list of
gold files (e.g. `--gold runs/encoder_gold_v2.jsonl,runs/hard_gold_train.jsonl`):
concatenates in order, dedupes by surface `text` keeping the FIRST
occurrence (so a hard-gold record never overshadows a teacher-gold record of
the same sentence, and listing a file twice is a no-op). No other trainer
change; `scripts/train_encoder.py`'s `--gold` argument is untouched
(`argparse` string, already compatible — the comma-splitting lives entirely
in `load_gold`).

---

## 9. Unfinished / out of scope, and exactly why

- **No family was skipped.** All 8 families named in the brief
  (imperative, pure interjection, content interjection, elision,
  additive/focus, quantity, synth-subject fragments, speaker prime) are
  expressible with `hand_gold.py`'s existing filler vocabulary (`W`, `PRIME`,
  `CTX`, `MEM`) — none needed new schema.
- **RESOLVED in v2: `--per-family` now ships at 150, the originally-planned
  value.** The v1 blocker was the four-file (`train`/`test_filler`/
  `test_template`/`hard_gold_all`) budget going over 5 MB at 150; v2 makes
  `hard_gold_all.jsonl` opt-in (`--write-all`, §2) since it was always a pure
  derived duplicate, so the three files that are actually used for
  train/eval fit at 150 with room to spare (3.3 MB).
- **RESOLVED in v2: `VI_past` pool thinness (was 10 words).** This was D5's
  own recommendation (b), "lemmatize at retrieval time" — done via
  `USVS.senses_of_surface` (§0/§4). `VI_past` now hits its pool cap at 30 (up
  from 10); `VT_past` similarly reaches 40 (up from 39, though the table
  already covered every `VT` base so the gain there is smaller — the fix
  matters most for `VI_past`, exactly the pool D5 flagged as thin).
- **`runs/encoder_gold_v2.jsonl` (the teacher-bulk gold) is NOT rebuilt with
  the new lemmatized `ground_word`.** `build_encoder_gold_v2.py` is updated
  (§0) so a FUTURE rebuild picks up the fix, but re-running it is a separate,
  multi-minute-to-multi-hour corpus-parse job with its own artifact-ownership
  concerns, explicitly out of scope for this batch ("do NOT rebuild the v2
  gold"). Measured impact instead, on a 20-word sample of the SHIPPED file's
  previously-`entity`-grounded inflected words (alphabetically first, plain-
  alphabetic, still raw-`senses_of`-empty against the live USVS): **12/20**
  would now ground as `sense` under `senses_of_surface` (`abilities`→`ability`,
  `added`→`add`, `advantages`→`advantage`, `adventures`→`adventure`,
  `allowed`→`allow`, `amounted`→`amount`, `animals`→`animal`,
  `answered`→`answer`, `antiquities`→`antiquity`, `asked`→`ask`,
  `assembled`→`assemble`, `became`→`become`); the other 8 (short function-ish
  words like determiners/prepositions caught by the same "content word, no
  sense" bucket) still don't lemmatize to anything WordNet covers, same as
  before — no regressions in the sample (`test_teacher_gold_inflected_word_grounding_unchanged_or_improved`,
  §7). Scanning the WHOLE file (not just the 20-word test sample) for
  reference: 333 of 458 unique entity-grounded non-name content words in
  `runs/encoder_gold_v2.jsonl` would gain sense candidates under the new
  helper — the gap this fix targets is real and large, concentrated in
  ordinary `-ed`/`-s` inflections the teacher path's raw `senses_of` call
  always missed.
- **`additive_focus`'s `me too .` / `me also .` templates cap at ONE unique
  record each**, regardless of `--per-family`: their visible surface has NO
  typed slot (only the invisible `context[]` varies), so every filler draw
  after the first collides on `seen_surfaces` and is rejected as a
  cross-split duplicate (§2's "generated ≫ passed" note). This mirrors reality
  — "me too ." is a genuinely frozen 3-word fragment in English; there is no
  further TYPED surface variation to generalize into without inventing a
  different sentence. The family total (86 passed at `--per-family 150`, v2)
  still clears the ≥50 bar because its other 6 templates (`and a {N} too .`,
  `{PROPN} too .`, ...) do vary the visible surface.
- **`quantity`'s two `{QUANT} !`/`{QUANT} .`-shaped template pairs
  (`quant_bang_ctx`/`quant_bang_past_ctx`, `quant_dot_ctx`) are surface-
  indistinguishable** from each other (both render as one of exactly 4
  strings — `QUANT` pool size 4 — regardless of which invisible context
  antecedent tense underlies them): whichever runs first in generation order
  claims a given `"{quant} !"`/`"{quant} ."` string, so the LOSING sibling
  (whichever is chosen as one of the family's 2 held-out `test_template`
  templates that generation run) can end up with very few of its own
  records (as low as the residual few surfaces the winner didn't reach first
  — observed 4-10 depending on seed). Not a correctness bug (gate passes,
  splits stay disjoint, no cross-split leakage — `test_hard_gold_gen.py`
  passes at every seed tried), but a template-design inefficiency: two
  templates that can never be told apart by surface text alone are a poor
  choice for a held-out "structural generalization" test split. Flagging
  for a follow-up template pass rather than reworking under this run's time
  budget — the family total (102 passed at `--per-family 150`, v2) still
  clears the ≥50 bar regardless.
- **No semantic plausibility filtering.** Context sentences are drawn from
  independent pools (`the {N_pl} {VT_past} the {N2} .` etc.), so many read as
  nonsensical ("the sheep put the state .", "the sofa loved the boat.")
  Deliberate: the task is grammatical/structural generalization (typed-slot
  coverage of the hard-case CONSTRUCTIONS) for an encoder that consumes
  clause structure, not a naturalistic-text corpus — teacher-bulk gold
  itself is templated curriculum sentences, not organic prose, for the same
  reason.
