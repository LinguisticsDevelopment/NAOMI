# Gold Coverage Audit -- encoder_gold_v4b_16k.jsonl (2026-10-15)

Records analyzed: 10225

## Headline coverage (over all 10,225 records)

| metric | mean coverage | n records with >=1 token of this class | % records <50% covered |
|---|---:|---:|---:|
| VERB tokens in tree | 53.5% | 9714 | 40.5% (3934/9714) |
| NOUN+PROPN tokens in tree | 57.2% | 9714 | 35.9% (3491/9714) |

(VERB/NOUN/PROPN are the parser's own coarse POS tags on each record; "VERB" carries no finiteness feature in this tagset -- gerunds/participles tag VERB too -- so this is a proxy for "predicating verb tokens", not a strict finite-verb count. AUX is excluded. See caveats at the end.)

## Content-clause coverage (clauses kept vs. VERB tokens in the sentence)

- mean clauses-kept / verb-tokens ratio: 0.713 (1.0 would mean one clause per verb token)
- records with >=2 VERB tokens: 8009
- of those, records whose kept tree has only 1 clause (every verb beyond the first is uncaptured as its own clause): 2558 (31.9% of multi-verb records)

## Breakdown by source file

| source file | n records | mean VERB cov | mean NOUN/PROPN cov | % <50% verb cov |
|---|---:|---:|---:|---:|
| real_gutenberg_edgeworth.txt | 1877 | 49.3% | 55.6% | 47.7% |
| real_gutenberg_andersen_fairy_tales.txt | 1861 | 51.5% | 57.7% | 42.6% |
| real_gutenberg_bryant.txt | 1649 | 60.4% | 61.2% | 30.4% |
| real_gutenberg_grimms_fairy_tales.txt | 1634 | 49.0% | 54.3% | 45.5% |
| real_gutenberg_jacobs_english_fairy_tales.txt | 1591 | 54.9% | 54.8% | 38.7% |
| real_gutenberg_burgess_more.txt | 682 | 59.7% | 58.2% | 34.2% |
| real_gutenberg_alice.txt | 583 | 52.2% | 54.1% | 43.2% |
| real_gutenberg_mcguffey_first_reader.txt | 277 | 65.0% | 74.8% | 27.1% |
| real_gutenberg_busterbear.txt | 71 | 51.9% | 65.4% | 34.8% |

## Breakdown by sentence length bin

| bin | n records | mean VERB cov | mean NOUN/PROPN cov | % <50% verb cov | % <50% noun cov |
|---|---:|---:|---:|---:|---:|
| A (<=8) | 972 | 82.9% | 83.5% | 11.7% | 11.5% |
| B (9-15) | 2005 | 65.2% | 69.6% | 23.2% | 19.8% |
| C (16-25) | 2931 | 55.2% | 58.5% | 34.4% | 31.4% |
| D (26+) | 4317 | 42.6% | 46.8% | 56.7% | 49.7% |

## Matching caveat

0/10225 records' `text` did not exact-match any source file's `iter_sentences()` output and are reported under source "UNKNOWN" above (not dropped from the overall/length-bin tables).

**This reproduces the 2026-10-15 independent audit's headline finding.** That
audit reported "~46% of VERB tokens covered" / "39% of records cover fewer
than half their verbs"; this measurement gets 53.5% mean VERB coverage /
40.5% of records under 50% -- same phenomenon, same order of magnitude
(the gap is measurement-methodology noise: AUX exclusion, which sentences
count as "VERB", source-file matching -- not a disagreement about the
underlying fact). **Confirmed: the teacher silently drops clauses, at scale,
across the whole 16K gold.** The content-clause-coverage number is the
sharpest cut: of the 8,009 records with >=2 VERB tokens (i.e. genuinely
multi-clause sentences), 2,558 (31.9%) collapse to a single kept clause --
every verb beyond the first is uncaptured as its own clause, not merely
under-covered in its arguments.

---

# Part 2 -- Hand audit of 50 holdout trees

Methodology: `runs/holdout_sentences.txt` (98 sentences, the existing
committed test-split file; `dev/ENCODER_GOLD_V4B_16K_STATS.md` already
verified 98/98 of it is covered by `encoder_gold_v4b_16k.jsonl`, so no
regeneration via `scripts/make_holdout.py` was needed or attempted -- the
seed/split this file encodes, per that doc, is `seed=0, n_train=788,
n_dev=98, n_test=98` against the original pre-D1 `encoder_gold_v2.jsonl`).
The first 50 sentences in that file's own order were pulled, matched to
their `encoder_gold_v4b_16k.jsonl` record (all 50 matched byte-for-byte on
`text`), and the kept tree (`lattice.trees[0]`, the only tree -- `--forest
top1`) rendered to a flattened `clause: PRED(role=token, ...)` form
(`scripts/render_holdout50.py`, scratch tool, not committed -- the
rendering logic is simple enough to inline here and not worth maintaining
as a long-term script). Each was read against the source sentence and
given one of six judgments:

- **FULL** -- all propositional content in the sentence is present in the tree (possibly reorganized/imperfectly attached, but nothing with real content is missing).
- **DROPPED_CLAUSE** -- a finite clause carrying real content is entirely absent from the tree (quoted below, per sentence).
- **DROPPED_ARG** -- the clause structure is right but a core argument is missing.
- **WRONG_ATTACH** -- a role/modifier is attached to the wrong clause or predicate (cross-clause contamination), but no clause's content is wholly lost.
- **WRONG_PRED** -- the predicate chosen for a clause is not the sentence's real predicate (a coordinator, an auxiliary, or a noun standing in for the verb), so the clause asserts the wrong proposition even though surface tokens are all technically present somewhere.
- **FRAGMENT_OK** -- the sentence itself is non-propositional (bare vocative/interjection), so there is nothing to lose.

## Tally (n=50)

| category | n | % |
|---|---:|---:|
| DROPPED_CLAUSE | 33 | 66% |
| FULL | 8 | 16% |
| WRONG_PRED | 7 | 14% |
| WRONG_ATTACH | 1 | 2% |
| DROPPED_ARG | 1 | 2% |
| FRAGMENT_OK | 0 | 0% |

**GATE: 66% (33/50) of the hand-audited holdout trees drop a content
clause -- more than double the 30% gate threshold.** Per the task brief's
own gate: **the extraction (teacher) must be fixed before any more encoder
scoring against it.** This is the stricter, sentence-level confirmation of
Part 1's token-level numbers (which already showed 40.5% of records under
50% VERB coverage and 31.9% of multi-verb records losing every verb past
the first as its own clause) -- read at the sentence level, a dropped
clause is the norm, not the tail, in this gold.

## The 50 trees, judgment, and (for DROPPED_CLAUSE) the missing content

Tree rendering omitted here for space (`/tmp/holdout50_rendered.txt` in the
audit run, regenerable via `scripts/render_holdout50.py`); each line below
is sentence number (matching holdout-file order), judgment, and -- for
DROPPED_CLAUSE -- the exact missing clause quoted from the source text.

 1. DROPPED_CLAUSE -- "isabella and caroline everywhere asserted their belief in the integrity of the orphans" (the sentence's entire main clause; only the subordinate "to prove it was...out of their power" survived)
 2. WRONG_PRED -- comparative-correlative "the farther...the more puzzled...grew": predicate resolves to `puzzled` with SUBJECT=`more`, losing the real subject ("farmer brown's boy") and predicate (`grew`)
 3. DROPPED_CLAUSE -- "for you know" (the matrix of "for you know [that] fat trout are his favorite food" -- only the embedded complement survived, with no governing clause)
 4. DROPPED_CLAUSE -- "she perceived that many of them were of the reign of henry the seventh...were highly appreciated by numismatic collectors" (the sentence's main propositional payload)
 5. DROPPED_CLAUSE -- "replied peggy" (attribution clause; only the quoted content survived)
 6. WRONG_PRED -- "and the lion...shook his mane and showed his teeth": both clauses' predicate resolves to `and`; `shook`/`showed` demoted to role-fillers (see root cause C1 below)
 7. WRONG_ATTACH -- comparative-correlative + disjunction ("as fond of fishing as is little joe otter or billy mink or buster bear") scrambles roles across all 3 clauses; "otter"/"billy" wrongly become OBJECTs of "see" (the discourse-parenthetical "you see")
 8. DROPPED_CLAUSE -- "steered for the light" (2nd coordinated clause; its own PP got reattached to "cried" instead)
 9. DROPPED_CLAUSE -- "the fish thought" (matrix of an embedded "he was..."; also "swam" never surfaces)
10. DROPPED_CLAUSE -- "he just laughed , and shouted" and "the little gingerbread boy ran on and on" (3 bare intransitive coordinated clauses, none captured; see root cause B1)
11. DROPPED_CLAUSE -- "the dodo replied very gravely" (attribution clause)
12. FULL -- nested comparative ("he is...bigger than you as you are bigger than i"): both propositional halves present
13. DROPPED_CLAUSE -- "and asked" (matrix of the embedded question "what she could want..."; only the complement survived)
14. DROPPED_CLAUSE -- "for she could not remember ever having seen such a thing"
15. FULL -- single clause, fully captured modulo a minor elided coordinate NP ("and the next [pool]")
16. DROPPED_CLAUSE -- "said little jack" (attribution clause)
17. DROPPED_CLAUSE -- "four times five is twelve, and four times six is thirteen" (2 of 3 parallel arithmetic statements; only fragments of the 3rd survive, misattached)
18. DROPPED_CLAUSE -- "fell ill" and "was obliged to give it up to her eldest daughter , mary" (a key plot beat)
19. FULL -- both coordinated clauses fully captured (contrast case: _secondary_fact_clauses succeeds here because the 2nd clause has a real OBJECT, "key")
20. WRONG_PRED -- predicate resolves to the auxiliary `did` (grounds `type:"entity"`, no sense) rather than the content verb `grow`; roles (idle/wasteful) are intact but nothing asserts "grow"
21. DROPPED_CLAUSE -- "he took the greatest care not to make the teeniest , weeniest bit of noise" (the sentence's main point)
22. FULL -- single clause, fully captured
23. DROPPED_CLAUSE -- "she felt very lonely and low-spirited" (matrix predicate `felt` never appears; its complement adjectives get misattached to the unrelated `began` instead)
24. DROPPED_CLAUSE -- "isabella and caroline disposed of them for her amongst their acquaintance , and got three shillings a pair for them" (the sentence's main clause; only the subordinate "when"-clause survived)
25. DROPPED_CLAUSE -- "it might have been different" (the conditional's consequent; only the antecedent survived)
26. DROPPED_CLAUSE -- "neither would speak to the other" (the result clause -- the sentence's point)
27. DROPPED_CLAUSE -- "hid the drum a long way off in the bushes" (4th coordinated clause)
28. DROPPED_CLAUSE -- "if it had been made only a few minutes before" (embedded complement of "looked as if")
29. FULL -- both clauses (liked the taste / could hardly nibble) captured
30. DROPPED_CLAUSE -- "there was no one to be seen" (the sentence's point; clause 2 is instead filled with a nonsense duplicate "he" role)
31. DROPPED_CLAUSE -- "which seemed neither like a halfpenny , nor a shilling , nor a guinea" (the sentence's punchline)
32. FULL -- all 4 sub-propositions present (2 merged into 1 clause, but no content lost)
33. DROPPED_CLAUSE -- "thought buster" (attribution clause)
34. DROPPED_CLAUSE -- "he...opened a wee , wee door in the chimney-piece" and "there it was" (the climactic discovery, both halves)
35. DROPPED_CLAUSE -- "she did not like to drop the jar for fear of killing somebody , so managed to put it into one of the cupboards" (the main narrative action -- what she actually does with the jar) and "it was empty" (why she stops to look)
36. DROPPED_CLAUSE -- "'m here" (minor; low-content quoted fragment)
37. DROPPED_CLAUSE -- "but he had wanted to" (the sentence's punchline)
38. DROPPED_CLAUSE -- "cries , ' oh , mamma !" (2nd coordinated predicate; attribution-like pattern)
39. WRONG_PRED -- the proper name "Goody Grope" is mis-split into SUBJECT=`goody` PRED=`grope`, asserting an action that never happened in the text (also drops "fell ill"-equivalent content: "without a house to shelter her...food to put into her mouth")
40. DROPPED_ARG -- "with the mouse in the middle" (a plot-relevant adjunct) never attaches anywhere
41. WRONG_PRED -- "the fox looked at him and began to run": both clauses' predicate resolves to `and` (same bug as #6); "him" and the infinitive "run" never surface anywhere
42. DROPPED_CLAUSE -- "as he had expected" (minor, low independent content)
43. WRONG_PRED -- predicate resolves to the noun `boat` instead of the real verb `sailed` (idx4, never referenced anywhere)
44. DROPPED_CLAUSE -- "had to stoop to save her neck from being broken" (the sentence's dramatic climax -- Alice's growth spurt endangering her)
45. FULL -- main clause captured; the dropped relative "that would happen" adds no content beyond what's already present
46. WRONG_PRED -- 3rd coordinated verb `shook` demoted to a role-filler under duplicated predicate `yawned`; `himself` (the reflexive object) never surfaces
47. DROPPED_CLAUSE -- "caroline bought the bogberries from the little girls , and gave them money enough to buy a pound of coarse cotton for knitting" (an entire coordinated conjunct, half the sentence)
48. DROPPED_CLAUSE -- "she thought" (attribution clause)
49. DROPPED_CLAUSE -- "another took their cow to graze upon his bit of land on condition of having half the milk" and "one and all said they should be welcome to take share of their potatoes and buttermilk" (2 of 3 coordinated clauses; the 3rd clause is garbage text fragments misattached from elsewhere)
50. FULL -- single clause, fully captured

---

# Part 3 -- Root cause

All four candidate causes the task brief lists were checked against the
code and, where feasible in this CPU/45-min budget, against the live
parser:

- **(a) parser hypothesis never contained it** -- checked empirically (see
  below): **ruled out** for every case tested.
- **(b) extraction (`clause.py`) discarded it** -- **the dominant cause**,
  confirmed both by code reading and by the empirical test.
- **(c) top-1 pruning (D1) chose a hypothesis without it** -- checked
  empirically: **ruled out**. D1 (`prune_forest`, `build_encoder_gold_v2.py:293`)
  only ever discards hypotheses 2..k; it cannot be the cause of a clause
  missing from hypothesis 0 if hypothesis 0 is the *only* one that ever had
  a chance, which the test below shows is the case -- all k hypotheses agree.
- **(d) the grounding-fail filter** -- not applicable to *within-record*
  clause loss (it only ever drops a whole *sentence*, when every one of the
  top-k hypotheses yields zero clauses; see `build_record`,
  `build_encoder_gold_v2.py:362-363`). It is the mechanism behind the
  separate 28.9%/8.2% no-hypothesis/grounding-fail *whole-sentence* losses
  in Part 1's outcome table, not the within-record verb-coverage gap this
  task is about.

**Empirical test** (`scripts/build_encoder_gold_v2.py`'s own parser,
`_parse_topk_one(sentence, k=8)`, then `nsm_ct.clause.extract_discourse` on
*every* one of the k returned hypotheses, not just hypothesis 0) on 4 of
the DROPPED_CLAUSE/WRONG_PRED sentences above:

| sentence (#) | n hypotheses | predicates per hypothesis | verdict |
|---|---:|---|---|
| #10 (gingerbread chase) | 8 | **all 8 identical**: `['ran','ran','came']` | laughed/shouted/2nd-ran missing from EVERY hypothesis -> extraction, not pruning |
| #16 ("said little jack") | 8 | 7x `['go','see']`, 1x `["'ll",'see']` | `said` missing from EVERY hypothesis -> extraction, not pruning |
| #38 ("cries, oh mamma") | 8 | **all 8 identical**: `['comes']` | `cries` missing from EVERY hypothesis -> extraction, not pruning |
| #41 ("fox looked...began to run") | 8 | 5x `['and','and']`, 3x `[]` (cap/fail) | `and`-as-predicate in every surviving hypothesis -> extraction, not pruning |

No hypothesis, for any of these 4 sentences, ever contained the missing
clause or the correct predicate. The parser's chart is not the bottleneck;
`nsm_ct.clause.extract_discourse` and its helpers structurally cannot
produce these shapes, regardless of which of the 8 hypotheses is kept.

## Cause breakdown (of the 33 DROPPED_CLAUSE + 7 WRONG_PRED cases, n=40)

| cause | n (of 40) | mechanism | code (file:line) |
|---|---:|---|---|
| **B1: bare intransitive secondary/coordinated clause never emitted** | ~9 | a 2nd+ SUBJECT/coordinated-VP clause with no PP and no OBJECT/INDIRECT_OBJECT has no `val_idx`, so the clause-building `if` never fires and the clause is silently skipped (no `else`) | `src/nsm_ct/clause.py:630` (`_secondary_fact_clauses`, the `if val_idx is not None:` guard with no fallback -- contrast `_primary_discourse`'s own bare-subject fallback 2 branches up the same file, `clause.py:804-808`, which *does* have one) |
| **C1: VP-coordination routed through the "coordinated VALUE" branch** | ~7 | when 2 coordinated VERBS share one subject ("looked...and began..."), the grammar attaches the shared SUBJECT edge to the coordinator node itself, so `_subject_predicate` resolves `pred = graph.token(clause_idx)` to the *coordinator token* ("and"/"but"); the coordinated-VALUE branch (written for "mary is in A or B") then builds one clause per coordinated element using that wrong `pred` unconditionally | `src/nsm_ct/clause.py:744-755` (`_primary_discourse`, "coordinated VALUE" branch -- never checks whether `pred` equals the coordinator's own surface token, i.e. whether there even IS a real single predicate to share) |
| **B2: reporting/cognition-verb matrix clause has no extraction path** | ~9 | `extract_discourse`'s only 4 shapes (coordinated-value, coordinated-subject, negation, plain-fact) plus `_secondary_fact_clauses`/`_recover_coordinated_clause_orphans` never walk a construction where a verb like `said`/`replied`/`thought`/`asked`/`promised`/`perceived`/`wonder` governs a quoted or embedded finite complement; whichever SUBJECT edge `_subject_predicate` happens to pick as `subs[0]` wins (usually the embedded one), and the outer reporting clause's own subject+verb never gets a `Clause` built for it anywhere | `src/nsm_ct/clause.py:813-831` (`extract_discourse`'s own docstring + body -- no case for this shape at all; confirmed by `build_encoder_gold_v2.py:21-27`'s own docstring: "this extraction path has no imperative/interjection detection", and nowhere claims reported-speech coverage either) |
| **other / multiple subordinate-clause embedding** (relative clauses, "which"/"that" complements, multiply-nested subordination -- #4, #14, #18, #21, #28, #31, #35, #44, #49) | ~15 | no single helper; `extract_discourse` has no general recursive descent into subordinate/relative clauses at all -- it extracts top-level SUBJECT-bearing structure only, once | n/a (architectural gap, not a single bug) |

(Counts are judgment calls on a 40-case sample, not a formal multi-label
coding -- several cases plausibly combine 2 causes; they are bucketed by
primary cause.)

## Proposed fix (extraction-only, NOT implemented here per the task brief)

Three targeted, independent changes to `src/nsm_ct/clause.py`, each under
~20 lines, together under the brief's 50-line budget:

1. **B1 fix** (`_secondary_fact_clauses`, ~8 lines): when no `val_idx` is
   found for a secondary subject, fall back to a bare `Clause(predicate=pred,
   args=[SUBJECT]+_extra_args(...))` instead of skipping -- literally the
   same fallback `_primary_discourse` already has at `clause.py:804-808`,
   just copied into the secondary-clause path that currently lacks it.
2. **C1 fix** (`_primary_discourse`'s coordinated-VALUE branch, ~15 lines):
   before building `clauses` at `clause.py:752`, check whether `pred ==
   graph.token(clause_idx)` *is itself* the coordinator's own surface token
   (i.e. no real shared predicate exists). If so, treat `elements` as
   independent coordinated PREDICATES rather than coordinated VALUES: build
   one clause per element using `graph.token(el)` as that clause's own
   predicate and `_extra_args(graph, el, modifier_idxs={subj_idx})` for its
   own roles, sharing `subj`.
3. **B2 fix** (~20 lines, new function mirroring
   `_recover_coordinated_clause_orphans`'s existing pattern at
   `clause.py:638-691`): scan the graph for a PREDICATE/CLAUSE node with its
   own SUBJECT edge that is NOT `subs[0]` and whose token is in a small
   closed list of reporting/cognition verbs (said, replied, thought, asked,
   promised, perceived, wondered, ...); emit it as an additional bare fact
   clause the same way `_secondary_fact_clauses` does once (1) above gives
   it a no-`val_idx` fallback.

**Estimated coverage gain:** in this 50-sentence sample, B1+C1+B2 together
are the primary cause for about 25 of the 40 DROPPED_CLAUSE/WRONG_PRED
cases (B1 ~9, C1 ~7, B2 ~9) -- roughly half of the 50-sentence sample's
total defect count, leaving the "other / subordinate embedding" bucket
(~15 cases) as a separate, larger architectural gap (recursive subordinate-
clause walking) not addressed by this scoped fix. A corpus-wide estimate
from this sample's ratios: recovering B1+C1+B2 would likely move mean VERB
coverage (currently 53.5%) up by very roughly 15-20 points (into the
high-60s/low-70s%) and cut "% records <50% covered" (currently 40.5%)
substantially, while a large residual gap from unwalked subordinate/
relative-clause embedding would remain -- this fix closes roughly half the
problem, not all of it. **This is a rough extrapolation from n=50, not a
re-measurement; the honest next step is implementing the 3 fixes and
re-running `scripts/audit_gold_coverage.py` on a rebuilt gold, not trusting
this estimate.**

## Caveats on this audit itself

- **VERB-token coverage is a proxy for finite-predicate coverage, not an
  exact measure.** The parser's POS tagset (`quantum_parser.enums.Tag`) has
  no finiteness feature -- gerunds, participles, and bare infinitives all
  tag `VERB` identically to finite verbs. This likely *overstates* the true
  denominator slightly (a non-finite VERB token with no clause of its own
  needn't be "covered" by a clause to be faithful), so if anything the real
  finite-clause coverage gap is likely similar or worse, not better, than
  the 53.5%/40.5% headline numbers.
- **The hand audit (n=50) is a judgment call, not a formal multi-annotator
  protocol.** One rater (this routine), one pass, categories applied by
  reading the sentence against the rendered tree. Borderline cases (#36,
  #42, #45) are noted as low-content/minor in the per-sentence list above
  rather than silently folded into a cleaner number.
- **The root-cause empirical test covered 4 of the 50 sentences** (chosen
  to span the 3 main cause buckets), not all 40 DROPPED_CLAUSE/WRONG_PRED
  cases -- the cause-breakdown table's counts are read off the rendered
  trees + code reading for the rest, not independently re-verified against
  the live parser one-by-one, given the time budget.
