# Mind v2b: a better round-trip check, and the full FairytaleQA (2026-11-18)

Status: DONE. The method is unchanged (`dev/design/CONSCIOUSNESS_V2_PROPOSAL.md`); these are measurement and data fixes. Single seed, test looked at once after the val-based choices, one rater for the hand-read.

## Step 1: the round-trip check, reformulated

**Problem (MIND_V2.md).** The old check read the decoded text back and compared it with the thinker's tree cut out of a full sentence. A fragment read alone is not that subtree, so the control with the right words in true order and no decoder already failed 35% (45% on the new test). It measured the reader on fragments, not decoder drift.

**Formulations tried (all in `src/naomi/roundtrip.py`, controls in `scripts/roundtrip_controls.py`, language-neutral: token equality, token positions, tree scorer):**
- **A reread** (primary): tree score of `read(decoded text)` against `read(the SAME words in true source order)`. Both are fragments read by the same reader.
- **B order**: text-level, share of related node pairs (parent/child, siblings) whose order in the decoded text differs from the source (no reader). Setting (a) only (it needs the orderer's node order).
- **C substitution**: the decoded words put back into the source sentence's slots, the sentence re-read, scored against the source sentence's own reading (every third item only, long reads are slow).

**Controls** (built from the v2 main model's own answers, so the same word sets as the real decoder sees): true order (the same words in source order), and negatives: shuffled word order, one adjacent swap, one word replaced by a random word of the story, one word dropped. Honest note: the true-order control compares identical text, so it passes 100% by construction for A, B and C; the information is in how the negatives fail and in what the real decoder does. The old check's problem (true order failing) is gone because the reference is now built the same way as the candidate.

**Threshold rule, declared and fitted on the v2 VAL answers (355 answered, 320 with >= 2 words) before any test number:** the most lenient threshold on a fixed grid whose mean negative-control pass rate is within 0.02 of the minimum, with true-order pass >= 95%. Result: **A: tree score >= 99** (a pass means "reads as the same tree"), C: 100, B: discord share 0. Grids in `runs/mind_v2/roundtrip_val.json`.

Pass rates (A, T = 99; share of items that pass; lower is "caught" for negatives):

| | v2 val | v2 test | v2b test (new model) |
|---|---|---|---|
| true order control | 1.000 | 1.000 | 1.000 |
| negative: shuffled order | 0.050 | 0.053 | 0.061 |
| negative: adjacent swap | 0.131 | 0.130 | 0.112 |
| negative: word -> random story word | 0.037 | 0.024 | 0.014 |
| negative: word dropped | 0.009 | 0.003 | 0.008 |
| **decoder (a) order-only** | 0.614 | 0.668 | **0.578** |
| decoder (b) meaning-only | 0.583 | 0.629 | 0.558 |
| old check, share < 80 (a) / true order | 0.485 / 0.439 | 0.448 / 0.353 | 0.507 / 0.450 |

B (order, tau = 0) on v2 test: true 1.0, shuffle 0.006, adjacent swap 0.239, decoder (a) 0.581. C on v2 test (n=119 decoder items): true 1.0, shuffle 0.0, adjswap 0.018, wordswap 0.009, drop 0.0, decoder (a) 0.715. A and B combined adds nothing over A.

**Is it sound?** (i) The true-order control passes (trivially, see above). (ii) Every negative fails clearly more often (pass 1% to 13% against 100%); the weakest is the adjacent swap (some swaps read the same, 11% to 13% pass). It is strict: T = 99 means "reads identical", and at T = 80 the negatives pass 31% to 37% (word swap, drop, adjacent swap), so it does not tolerate a looser line. **What it says about the decoder: 33% to 42% of answers do NOT read as the thinker's words in source order (v2 test (a): 33%; v2b test: 42%), but the failures are not a private code.** They sit almost entirely in long answers: v2b test pass rate for (a) is 98% on 1-3 word answers (n=246), 84% on 4-6 (139), 33% on 7+ (554). The failing examples (val rows, `runs/mind_v2/roundtrip_val_rows.jsonl`) are the frozen orderer scrambling long, gappy word sets ("brought to them snatches of whatever..." for "snatches of whatever ... brought to them", "he of his lonely life grew weary and who had a daughter also ..."): ordering errors of decoder v1 on non-contiguous subtrees, not the thinker and decoder agreeing on a hidden mapping. Setting (b) behaves the same as (a) (word choice is 98%). The check therefore does what it was asked (separates the controls, passes true order, grades fragments consistently) and the proposal's "> 20% fail" line is still crossed, now for the honest reason: the order decoder is not yet good enough on long kept sets. Conclusion on the proposal rule: the round-trip line is crossed on a sound check; what it points at is orderer quality on gappy trees (training signal for the decoder pair or shorter spans), not drift.

## Step 2: the full FairytaleQA

**Data.** `GEM/FairytaleQA`, revision `b6c76a77359f133f9ee087b65c52a686fada7c15` (`data/fairytaleqa_full/`, regenerate with `scripts/fetch_fairytaleqa_full.py`; read with `python -I`). 10,580 rows (train 8,548 / val 1,025 / test 1,007) from 278 stories (232 / 23 / 23), the dataset's own split by story. **Licence: the dataset card says `license: unknown`, "Copyright Restrictions on the Dataset: research use only", "Language Data: public domain" (the stories). No redistribution licence is stated; the raw files must not be committed.** (`data/fairytaleqa_full/SOURCE.md`.)

**Split overlap with the old split (checked).** The old 79 stories nest exactly: all 62 old train stories are in the new train, the 7 old val in the new val, the 10 old test in the new test; zero old-train or old-val story is in the new test. So there is no leakage and "the old 10 test stories inside test" holds. The old 3,000 items are every item of those 79 stories (so the old 406 test items = all items of the 10 old test stories); the new test adds 13 stories and 601 items. The old-10 numbers are reported separately below.

**Text choices (documented in `scripts/build_fairytaleqa_full_items.py`).** The HF file has ONE answer per question, lowercased and tokenised. A copy of the original release is already in the repo (`data/k12/fairytaleqa`, matched on story + question, 10,556 of 10,580 rows): questions and all annotator answers (1 to 6 per item; 917 of 1,007 test items have more than one) are taken from it so the text has the same form v2 used and the gold list is complete (ROUGE and the tree score take the max over all golds; v2's tree score used only the first). The 24 unmatched rows use the HF strings. Passages: `iter_sentences(HF content)`; for the 3,000 old items the old context is reused (quote spacing in the HF file differs from the repo's tokeniser). The new test also contains answer types the old free-form set may have under-represented (short entity answers). Reading: 36,780 new strings with the v1/v2 reader on 4 cores, about **3,690 s (61 min)**; items build 2,031 s (4 cores). Train: 8,548 normal items (7,536 reachable = some subtree F1 >= 0.5) plus 2,534 wrong-passage items (v2: 2,225 / 1,979 / 647).

**Models (same architecture, features and hyperparameters as the selected v2 run, nothing tuned; selection on val only):** stage 1 `s1_all` 24 epochs lr 2e-3 (750 s; best epoch 22, val 0.3527, true word order); stage 2 `s2b_all` 20 epochs lr 5e-4 seed 1 from `s1_all` (1,084 s; best epoch 15, val decoded 0.3529, vs 0.3343 for stage 1 decoded at stage-2 start); short-only `s1_short` 24 epochs (88 s). Training total about 37 min (budget 2 h). Main model = `s2b_all` (the v2 rule: stage 2 selected by val), curves `thinker_*.curve.tsv`. The v2 checkpoint (`runs/mind_v2/thinker_s2b_all_best.pt`) was run on the same new items (the vocab keeps v2's ids first; the new reads added no relation or tag label, so it loads unchanged).

## Results: held-out test (1,007 items, 23 stories). ROUGE-L F1 (max over all golds), abstain = 0
| system | ROUGE-L (abstain 0) | answered-only | answer rate |
|---|---|---|---|
| **thinker v2b + (a) order-only decoder** | **0.348** | 0.374 | 0.932 |
| thinker v2b + (b) meaning-only decoder | 0.342 | 0.367 | 0.932 |
| thinker v2b, true word order (no decoder) | 0.372 | 0.399 | 0.932 |
| stage 1 only (`s1_all`), (a) / (b) | 0.335 / 0.329 | 0.349 / 0.343 | 0.959 |
| **v2 checkpoint (old, 2,225 items) on this test**, (a) / (b) | **0.301** / 0.297 | 0.326 / 0.321 | 0.926 |
| short-only (`s1_short`), (a) | 0.114 | 0.134 | 0.850 |
| **cheat: best-overlap sentence (IDF), whole** | **0.220** | | 1.0 |
| question echo | 0.116 | | |
| random story sentence | 0.136 | | |
| random-anchor full subtree | 0.078 | | |
| oracle: best full subtree by ROUGE (true order) | 0.757 | | |
| oracle: lemma-F1 pruned subtree | 0.752 | | |

Val (1,025): cheat 0.228, v2 old 0.298, stage 1 0.334, main 0.353, oracle 0.760. Paired bootstrap, main (a) minus cheat: +0.128 (95% over items 0.108 to 0.150; over the 23 stories 0.121 to 0.168). Main minus the v2 checkpoint: **+0.047 (items 0.030 to 0.066; stories 0.029 to 0.070)**. Stage 2 over stage 1: +0.013 test, +0.019 val decoded, within the epoch-to-epoch val noise (+-0.01), as in v2. Reachable items only (874): main (a) 0.387, v2 0.333, cheat 0.242, oracle 0.81.

**Old 10 stories for continuity (the 406 old test items):** main (a) 0.387 (b 0.377), stage 1 0.379, v2 checkpoint 0.325 (v2 reported 0.326, reproduced), cheat 0.243, oracle 0.765. The other 13 stories (601 items): main 0.322, v2 0.285, cheat 0.205, oracle 0.752.

### By answer-sentence length (test; "answer sentence" = story sentence with most lemma overlap with the reference)
| band | n | cheat | v2 old (a) | stage 1 (a) | main (a) | main (b) | oracle | main answer rate |
|---|---|---|---|---|---|---|---|---|
| <=10 words | 35 | 0.104 | 0.188 | 0.156 | 0.168 | 0.168 | 0.815 | 0.91 |
| 11-20 | 202 | 0.205 | 0.371 | 0.417 | 0.410 | 0.400 | 0.802 | 0.97 |
| 21+ | 688 | 0.251 | 0.315 | 0.349 | 0.372 | 0.366 | 0.792 | 0.95 |
| none | 82 | 0.045 | 0.066 | 0.096 | 0.072 | 0.072 | 0.331 | 0.72 |

### Short-to-long (stage-1 models, decoder (a), test)
Short-only model: trained ONLY on items whose answer sentence <= 10 words: **489 reachable items + 165 wrong-passage items** (train has 514 items in that band; v2 had 143 + 41). All-lengths: `s1_all` (7,536 + 2,534). Test band sizes 35 / 202 / 688 / 82.

| band | n | all lengths | <=10 only |
|---|---|---|---|
| <=10 | 35 | 0.156 | **0.276** |
| 11-20 | 202 | 0.417 | 0.193 |
| 21+ | 688 | 0.349 | 0.092 |
| none | 82 | 0.096 | 0.031 |
Val agrees in direction (<=10, n=38: 0.314 vs 0.364). Plain reading: with 3.4 times more short items the short-only model now beats the all-lengths model on the short band (0.276 vs 0.156, 35 items from few stories, so not firm) but it still does not transfer upward: it falls to 0.19 and 0.09 on the longer bands, far below the all-lengths model. Short sentences are rare (35 of 1,007 test items) and the all-lengths model is weak exactly there (0.156 against an oracle of 0.815). The result says short-only training teaches short-sentence span choice that all-lengths training does not, and that what it learns does not compose to long sentences; it does not support "learn the basics, then it extends".

### Needs-new-words items (not tuned on; at least one third of reference content lemmas in neither story nor question)
224 of 1,007 (22%): main (a) 0.149 (b 0.147), stage 1 0.161, v2 0.151, cheat 0.119, oracle 0.508. Unchanged by more data: these need words the span chooser cannot write.

### Controls (test; main model, others in `runs/mind_v2b/eval_test.json`)
| control | main | stage 1 | v2 old |
|---|---|---|---|
| empty memory: abstain rate (must be 100%) | 1.000 | 1.000 | 1.000 |
| question from another story: abstain (>= 0.90) | 0.946 | 0.942 | 0.942 |
| questions shuffled within the story: answer rate / ROUGE-L (floor) | 0.483 / 0.061 | 0.512 / 0.059 | 0.433 / 0.057 |
| memory = the question only: abstain rate (extra) | 0.836 | 0.038 | 0.652 |
Controls hold. The question-only weakness of stage 1 remains (it copies the question's own words 96% of the time); stage 2 fixes it (0.836).

### Scoring with reasons (test, main, 939 answered; tree score = max over ALL golds)
Tree score (a) mean 37.7 over answered (35.2 with abstain = 0), (b) 37.7; the cheat sentence 13.6 (v2 reported 30.7 and 10.7 with the first gold only; the numbers are not comparable). Reason counts (a): node_extra 4,597; node_dropped 1,398; meaning_shift 1,201; head_changed 801; operator_folded_differs 789; subtree_moved 635; name_mismatch 307; relation_changed 162 (answers are too long more than too short). Word-choice accuracy of decoder (b): 98.1% of 10,449 nodes; 98.6% of those with a USVS point; 80.7% of the 300 with no point.

Round trip (test, main (a) / (b), n=939): old check share < 80: 50.7% / 51.1% (true-order control 45.0%); **new check (A, T = 99) pass rate 57.8% / 55.8%**; negatives 1% to 11% (table in Step 1). By answer length (a): 1-3 words 98.4% pass, 4-6 words 84.2%, 7+ words 33.2%.

## Hand-read: 20 test answers (10 low, 10 abstain; seeded; one rater = the implementing agent; `runs/mind_v2b/eval_handread_20{,_judgments}.json`)
| group | finding |
|---|---|
| 10 low (ROUGE <= 0.05, answered; 212 such items) | wrong span 6 (incl. one near miss: "the fourth mikoto" for "Hohodemi", same person; one pronoun node); correct but scored 0 by ROUGE 2 ("astonishment" for "astonished", "fright" for "frightened"); answer not a span 2 (an inference and a "how did they treat" question where a spurious sentence was copied); reading error 0 found |
| 10 abstain (68 abstentions) | answer verbatim in the story sentence 5 (p(abstain) 0.48 to 1.0; "wrote her name across the side", "as smooth as his left", "despair"); answer not a span 5 (all "how did X feel", answer unstated: abstaining is reasonable) |
Same shape as v2: the failure is span choice, then answers that are not spans; reading errors were not the cause in this sample.

## Plain conclusion
**Did more data move span choice toward the oracle?** Yes, modestly. ROUGE-L 0.301 (v2 checkpoint) to 0.348 on the same 1,007 items (+0.047, interval 0.03 to 0.07, positive over stories), and the old 10 stories rose 0.325 to 0.387. Against the cheat the gap is +0.128. But the oracle is 0.757: about a quarter of the cheat-to-oracle distance is closed (v2: 15%). 3.8 times the training items bought +0.05; the curve is not near saturation but it is slow. **Did short-to-long now say anything?** A little: with 489 short items the short-only model beats the all-lengths one on short sentences (0.276 vs 0.156, n=35) and still collapses on longer ones (0.09 on 21+): short training does not extend. **Is the new round-trip check sound?** Yes per its controls (true order 100% by construction, negatives pass 1% to 13%, same threshold held on val, test and the new model), and it replaces a check that failed its own control. It shows the decoder fails to reproduce the thinker's word order on about 42% of answers, almost all of them 7+ word gappy sets, which is orderer error, not a private code. The proposal's 20% line is still crossed on this honest check.

## Three biggest limiting factors (with evidence)
1. **Picking the right span** (the learned part): system 0.348 vs oracle 0.757 (0.387 vs 0.81 on reachable items); the six wrong-span cases of ten low answers, five of ten abstentions on answers verbatim in the sentence (p(abstain) up to 1.0). 3.8 times the training items gave +0.047; stage 2 on the text score adds only noise-level gain.
2. **Answers that are not spans** (feelings, summaries, paraphrase): 224 of 1,007 items (22%) need words in neither story nor question; system 0.149 vs oracle 0.508 there; the "none" band (82 items) is 0.072; ROUGE also misses derivations ("fright"/"frightened", scored 0). Needs decoder v2 (parked); five of ten hand-read abstentions are exactly this.
3. **The order decoder on long, gappy answer sets**: round trip passes 98% on 1-3 word answers and 33% on 7+; the true-word-order text scores 0.372 vs 0.348 decoded (-0.024 ROUGE-L), and the thinker's answers are long (the system's answers are too long more than too short: node_extra 4,597 vs node_dropped 1,398). Smaller: single seed, stage-2 gain within noise, short band n = 35 (13 new stories only add 601 items).

## How firm
Held-out by whole story (official split, 23 test stories, no overlap with training), test looked at once after val-based choices; single seed; the v2b-vs-v2 interval excludes zero (also over stories); stage 1 vs stage 2 and (a) vs (b) differences are within noise; the short-to-long short band has 35 items; hand-read is one rater and 20 items.

## Artifacts and reproduce
Force-add: `runs/mind_v2b/thinker_{s1_all,s2b_all,s1_short}_best.pt` (0.84 MB each; main = `s2b_all`), curves `thinker_*.curve.tsv`, logs `s1_all.log s1_short.log s2b_all.log build_read.log build_read.part1.log build_items.log eval_val.log eval_test.log rt_*.log`, `vocab.json`, `eval_val.json eval_test.json`, `eval_handread_20*.json`, `roundtrip_test.json`, `runs/mind_v2/roundtrip_{val,test}.json` (thresholds declared on val) and the `_rows.jsonl` files (small). NOT: `read_cache/` (158 MB), `stories.pkl` (253 MB), `pairs_*.pkl`, `items_full.jsonl`/`items_meta.json` (regenerable), `eval_rows_*.jsonl` (optional), and the raw dataset (`data/fairytaleqa_full/*.json`, licence is research-use-only: not for redistribution; only `SOURCE.md` and the fetch script).
```
python scripts/fetch_fairytaleqa_full.py
python -I scripts/build_fairytaleqa_full_items.py
python scripts/build_mind_v2_data.py read  --full --workers 4
python scripts/build_mind_v2_data.py items --full --workers 4
python scripts/train_mind_v2.py --dir mind_v2b --stage 1 --variant all   --epochs 24 --tag s1_all
python scripts/train_mind_v2.py --dir mind_v2b --stage 1 --variant short --epochs 24 --tag s1_short
python scripts/train_mind_v2.py --dir mind_v2b --stage 2 --init runs/mind_v2b/thinker_s1_all_best.pt --epochs 20 --lr 5e-4 --seed 1 --tag s2b_all
python scripts/roundtrip_controls.py --dir runs/mind_v2 --split val --model s2b_all --fit        # thresholds, declared on val
python scripts/eval_mind_v2.py --dir mind_v2b --split test --rt_thr runs/mind_v2/roundtrip_val.json --models s1_all=... s2b_all=... s1_short=... v2old=runs/mind_v2/thinker_s2b_all_best.pt --main s2b_all --short s1_short --alllen s1_all
python scripts/roundtrip_controls.py --dir runs/mind_v2 --split test --model s2b_all --thr runs/mind_v2/roundtrip_val.json
python scripts/roundtrip_controls.py --dir runs/mind_v2b --split test --model s2b_all --thr runs/mind_v2/roundtrip_val.json
```
Tests: `pytest tests/test_mind_v1.py tests/test_mind_v2.py tests/test_naomi_no_literals.py tests/test_naomi_memory.py tests/test_roundtrip.py -q`: 23 passed (17 s); no allow-list additions were needed.
