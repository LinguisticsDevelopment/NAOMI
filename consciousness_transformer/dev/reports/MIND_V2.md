# Mind v2: text in / text out (2026-11-17)

Status: DONE (first version). Method = `dev/design/CONSCIOUSNESS_V2_PROPOSAL.md` (approved). Verdict: **beats the word-overlap cheat on held-out stories** (ROUGE-L 0.326 vs 0.243, abstain counted as 0), **but the proposal's round-trip line is crossed** (45% of answers read back below 80, against an abandon line of 20%), so the proposal's own rule says rethink the round-trip check before building further. Single seed, one rater for the hand-read, 10 test stories (see "How firm").

## What was built
Chain: story sentences -> reader (cached, same as v1) -> Memory; question -> question tree; **thinker** (learned) -> answer tree or ABSTAIN; **decoder** -> text; text scored against the reference text.
- Data: FairytaleQA free-form items (3,000 in `runs/fairytaleqa_episodes_samepassage.jsonl`), the file's own split, which is by whole story (79 stories, no story in two splits): train 2,225 / val 369 / test 406 items (test stories = the same FairytaleQA test split v1 used). Story passage per item (about 10 sentences) is the memory. 9,229 new strings (story sentences, questions, first reference answers) read once with v1's reader (984 s on 4 cores); v1's cache reused. Reference answers are read into trees by the same reader (only the first gold answer is read; ROUGE uses the maximum over all gold answers).
- Thinker (`naomi.mind_v2.Thinker`, 0.2 M parameters, CPU, H=64): node inputs are a USVS point through one shared linear projection, relation-label embedding, tag embedding, depth, subtree size, relative position in sentence, function-word bit and no-point bit (lexicon facts), plus per-node bits "same lemma as a question node" (an equality test), the same for content nodes only, IDF-weighted lemma match, max USVS cosine to a question node, and per-sentence coverage / memory-tensor similarity (`Memory.similar`) with its rank. No word identity. Three tree message-passing rounds on the story forest, two on the question tree, one cross-attention from story nodes to question nodes. Outputs: a logit per story node (ANCHOR), one ABSTAIN logit (from pooled node states, item bits and the log-sum-exp of anchor logits), and for an anchor a keep/drop logit for every node of its subtree (a dropped node drops its subtree). Candidates are all nodes of the passage (FIND is a feature, not a filter, so the oracle reach is not capped).
- Stage 1 targets: the reference tree's lemmas are aligned to memory nodes by lemma equality; for every node as anchor the kept set that maximises lemma-multiset F1 is found (two rules: matched nodes connected to the anchor, or also the connectors). The anchors within 0.05 F1 of the best (at most 3) are accepted targets (marginal likelihood); keep labels come from the best set of each. 363 of 406 test items are "reachable" (some subtree F1 >= 0.5); train 1,979 of 2,225 (246 skipped and counted). 647 wrong-passage items (question against a passage of another story, about 22% of the batch) -> target ABSTAIN. Length curriculum: item weight by answer-sentence band: <=10 words always 1; 11-20 ramps 0.3 -> 1 over the first 40% of epochs; 21+ ramps 0.1 -> 1 over the first 70%; wrong-passage items always 1. 24 epochs, AdamW lr 2e-3, batch 16; checkpoint chosen on val ROUGE-L (true word order).
- Stage 2: REINFORCE from the stage-1 checkpoint: 4 samples per item (anchor sampled from the softmax incl. ABSTAIN, keep bits Bernoulli), reward = ROUGE-L F1 of the text decoded by the frozen orderer (setting (a)) against the best gold, baseline = mean reward of the item's other 3 samples. **Abstain reward declared before any test number: 0.10 on a normal item (a wrong answer pays about 0, a good one far more than 0.10), 1.0 on a wrong-passage item (answering there pays 0).** Two runs from the same stage-1 checkpoint: `s2_all` (6 epochs, lr 3e-4) and `s2b_all` (20 epochs, lr 5e-4, best val epoch 13); the main model is `s2b_all` by validation (0.308 vs 0.293 vs 0.289 for stage 1 alone; the val curve moves by about +-0.01 from epoch to epoch, so this choice is partly noise).
- Decoder (a) order-only: the kept nodes keep their words, the frozen orderer (decoder v1, `runs/naomi_orderer/large_best.pt`) orders them. Decoder (b) meaning-only: every answer node is stripped to its USVS point, relation and tags; the word is chosen from the pool of word occurrences of story + question as the one with the nearest USVS point (cosine on the axis-weighted point); ties and morphology go to tag agreement (tag equal + subtype overlap + relation equal), then pool frequency, then the token string; nodes without a point (names, lexicon-less lemmas) can only be matched by kind + tags + frequency. No language rule. Then the same orderer orders them.
- Code: `src/naomi/mind_v2.py`, `scripts/build_mind_v2_data.py`, `scripts/train_mind_v2.py`, `scripts/eval_mind_v2.py`, `tests/test_mind_v2.py` (5 tests, 25 s). `Mind.ask` was NOT switched to v2: v2 beats the cheat on val, but the switch needs the story built from `Mind`'s own memory (v2 builds from the read cache); left as a follow-up.

## Results: held-out test (406 items, 10 stories). ROUGE-L F1, abstain = 0
Main model `s2b_all`. Declared before the test run: headline = mean ROUGE-L, abstain counted 0.

| system | ROUGE-L (abstain 0) | answered-only | answer rate |
|---|---|---|---|
| **thinker v2 + (a) order-only decoder** | **0.326** | 0.341 | 0.956 |
| thinker v2 + (b) meaning-only decoder | 0.322 | 0.337 | 0.956 |
| thinker v2, true word order (no decoder; reference point) | 0.344 | 0.360 | 0.956 |
| stage 1 only (`s1_all`), (a) / (b) | 0.308 / 0.300 | 0.318 / 0.310 | 0.968 |
| stage 2, 6 epochs (`s2_all`), (a) / (b) | 0.323 / 0.315 | 0.341 / 0.334 | 0.946 |
| **cheat: best-overlap sentence (IDF-weighted), whole** | **0.243** | | 1.0 |
| question echo | 0.122 | | |
| random story sentence | 0.141 | | |
| random-anchor full subtree | 0.083 | | |
| oracle ceiling: best full subtree by ROUGE (true order) | 0.767 | | |
| oracle: lemma-F1 pruned subtree (true order) | 0.767 | | |

Difference to the cheat +0.083 (paired bootstrap over items 95% interval 0.053 to 0.116; over the 10 test stories 0.062 to 0.109). Val (369 items): cheat 0.241, stage 1 0.289, stage 2 (selected) 0.308, oracle 0.771. Stage 2 adds +0.015 to +0.018 on test over stage 1 and +0.019 on val for the selected checkpoint; that is within the epoch-to-epoch noise of the val curve, so **stage 2 is not shown to help beyond noise**. On reachable items only (363): thinker 0.353, cheat 0.262, oracle 0.81. The tree score is a second table below.

### By answer-sentence length (test; "answer sentence" = the story sentence with most lemma overlap with the reference; "none" = no overlap at all)
| band | n | cheat | stage 1 (a) | main (a) | main (b) | oracle | main answer rate |
|---|---|---|---|---|---|---|---|
| <=10 words | 15 | 0.103 | 0.224 | 0.247 | 0.247 | 0.749 | 0.80 |
| 11-20 | 83 | 0.262 | 0.379 | 0.393 | 0.391 | 0.802 | 0.98 |
| 21+ | 280 | 0.264 | 0.316 | 0.337 | 0.331 | 0.799 | 0.97 |
| none | 28 | 0.048 | 0.057 | 0.064 | 0.064 | 0.346 | 0.82 |

Only 15 test items have an answer sentence of <=10 words (train 151, val 16): the short band is far too small to say anything about it. Mid and long bands favour the thinker over the cheat by about 0.07 to 0.13.

### Short-to-long (owner condition). Stage-1 models only, decoder (a), ROUGE-L, test
Short-only model: trained ONLY on items whose answer sentence is <=10 words: 143 reachable items + 41 wrong-passage items (of the 151 short items in train). All-lengths: `s1_all` (1,979 + 647).

| band | n | trained on all lengths | trained on <=10 only |
|---|---|---|---|
| <=10 | 15 | 0.224 | 0.134 |
| 11-20 | 83 | 0.379 | 0.209 |
| 21+ | 280 | 0.316 | 0.087 |
| none | 28 | 0.057 | 0.017 |
Plain statement: the short-only model does not transfer and is worse than the all-lengths model even on the short band (0.134 vs 0.224, n=15); with 143 items it learns too little to say whether the mechanism composes. The all-lengths model does not degrade on short sentences, the weaker positive reading. Same finding as v1.

### Needs-new-words items (not tuned on)
Defined before looking: at least one third of the reference's content lemmas are in neither the story nor the question. 70 of 406 test items (17%): thinker 0.175 (b: 0.174), cheat 0.142, oracle 0.49. Answers there are paraphrases, emotions, summaries ("How did X feel?"), which this version cannot write (decoder v2 parked).

## Decoder (a) vs (b), word choice
Setting (b) loses 0.004 ROUGE-L (0.326 -> 0.322): the meaning point, relation and tags are enough to recover the words from the story + question pool. **Word-choice accuracy (chosen surface word == original), test, main model: 98.0% of 3,765 answer nodes; 98.6% of the 3,651 nodes that carry a USVS point; 78.9% of the 114 nodes with no point (names, lexicon-less lemmas), where only kind, tags and frequency are available.** Caveat on what this number measures: the pool contains the node's own word occurrence, so it tests whether, within the words of this story and question, a meaning point + tags identify the word; it is not the harder claim that meaning identifies a word among all words of the language (decoder v2, parked). (Val: 98.1% / 98.5% / 84.2%.)

## Scoring with reasons: tree score and round trip (test, main model, 388 answered)
Tree score (`naomi.score`, decoded answer read by the reader vs the first reference answer read the same way): (a) mean 30.7 over answered items (29.3 with abstain = 0), (b) 30.7; the cheat's sentence scores 10.7 (a whole sentence against a short reference is mostly "node_extra"). Reason counts over all answers (a): node_extra 1,743; node_dropped 775; meaning_shift 489; operator_folded_differs 310; head_changed 284; subtree_moved 215; name_mismatch 89; relation_changed 66 (the answers are too long more than too short).

Round trip (decoded text read back vs the thinker's own tree, tree scorer overall %):
| | mean | share < 80 (proposal line: > 20% = abandon) | share < 50 |
|---|---|---|---|
| (a) order-only | 77.7 | **44.8%** | 16.2% |
| (b) meaning-only | 77.2 | **46.4%** | 15.7% |
| control: the same kept nodes in TRUE word order, no decoder | 81.9 | **35.3%** | 10.6% |
**The line is crossed**: 45% of answers read back below 80. The control says most of that is not the decoder: even the original word order of the same words fails 35%, because answers are fragments ("the king to", "inside a hollow") and the reader / scorer on fragments differs from the tree cut out of a full sentence. The decoder adds about 10 points to the failing share. So the check as specified measures reader and scorer fragility on fragments as much as encoder-decoder drift, and it needs a different formulation (e.g. compare in a form that is robust to fragments) before it can gate anything. Two more measurement flaws seen in the hand-read: the tree score uses only the first gold answer (a correct answer scored 0 because it matched gold 2), and it can be generous ("aglow" vs "disappointment" scored 51).

## Controls (test, 406 items; target in brackets)
| control | main model | stage 1 | stage 2 (6 ep) |
|---|---|---|---|
| empty memory: abstain rate (must be 100%) | 1.000 | 1.000 | 1.000 |
| question from another story: abstain rate (>= 0.90) | 0.946 | 0.926 | 0.961 |
| questions shuffled within the story (another question of the same story, scored against this item's reference): answer rate / ROUGE-L (floor) | 0.438 / 0.054 | 0.488 / 0.052 | 0.404 / 0.044 |
| memory = the question only (v1's severed control): abstain rate (extra, not required) | 0.675 | 0.057 | 0.534 |
Empty memory is satisfied trivially (no node to anchor on) and by the model's own abstain logit. Shuffled questions score at the floor (0.054, below the random-sentence baseline 0.141). The question-only memory is the weak spot: stage 1 answers from the question's own words 94% of the time (v1's policy abstained every time); stage 2 improves it to 0.675 but never trained on such items. Nothing is stored in weights in the sense of facts about the stories (wrong-passage abstain 95%), but the model can be fooled into copying the question.

## Hand-read: 30 test answers (single rater = the implementing agent; seeded draw; files `runs/mind_v2/eval_handread_30.json`, `..._judgments.json`)
| group | right | partly | wrong | notes |
|---|---|---|---|---|
| 10 high (ROUGE >= 0.6) | 7 | 3 | 0 | |
| 10 low (ROUGE <= 0.05, answered) | 1 | 0 | 9 | the one "right" is "troubled" for "worried/upset": ROUGE 0 is wrong, tree score 91 is right |
| 10 abstain | | | | answer stated in the shown sentence: 6; implicit but supported ("thirst" for "thirsty"): 1; unclear: 1; not stated: 2 |
Does the tree-score reason match what is wrong? Mostly yes: in the 12 answers with partial or wrong content the listed reasons (node_extra, node_dropped, meaning_shift, head_changed) describe the real defect (extra subject "the dragon queen and", missing "tree" in "inside a hollow tree", wrong sentence picked). Exceptions: a correct answer scored 0 with head_changed / meaning_shift because the tree score used only the first gold ("the white finger") while the answer was gold 2; "aglow" for "disappointment" got 51. ROUGE misses paraphrases ("troubled"). In the abstain group the story states the answer in 6 of 10 cases with p(abstain) up to 1.0 ("a huldre hat" verbatim, 0.999): the model abstains on answerable items without grounding failures being visible to it. Abstaining is rare on real items (4.4%), so this costs little overall.

## Training wall-clock
Reading 984 s (4 cores), items build 600 s (4 cores), training: stage 1 all-lengths 210 s, stage 1 short 27 s, stage 2 `s2_all` 97 s, `s2b_all` 318 s (about 11 min of training in total; the budget was 2 h). Evaluation about 1.5 min per split. Tests: `pytest tests/test_mind_v1.py tests/test_mind_v2.py tests/test_naomi_no_literals.py tests/test_naomi_memory.py -q` 19 passed (22 s); the no-literals test gained an allow-list entry for non-linguistic identifiers only (dict keys, pair-kind tags, dataset field names), with a comment.

## Plain conclusion
Does it beat the cheat? **Yes on ROUGE-L** (0.326 vs 0.243 test; 0.308 vs 0.241 val; +0.08, interval 0.05 to 0.12 and still positive over the 10 test stories), with controls at target (wrong-passage abstain 94.6%, empty memory 100%, shuffled at floor). Part of the gap is the format: the cheat returns a whole sentence and pays in precision, the thinker returns a span. **It does not pass the round-trip check** as specified (45% < 80, line 20%), and the check is itself unreliable on fragments (35% even for true word order). The stage-2 (text-score) fine-tuning adds a small gain that is not distinguishable from noise. Meaning-only decoding works as far as this test can tell (98%) and costs 0.004 ROUGE-L.

## Three biggest limiting factors (with evidence)
1. **Picking the right span** (the learned part): the oracle subtree reaches 0.767 (0.81 on reachable items) and the system 0.326 (0.353 on reachable items). Hand-read: 9 of 10 low-scoring answers are the wrong span, 6 of 10 abstentions are on answers sitting verbatim in the sentence. Training signal is only ~2,000 reachable items from 62 stories; the short-only run (143 items) learned almost nothing.
2. **Answers that are not spans of the text** (implicit, paraphrase, feelings, summaries): 70 test items (17%) where reference words are missing from story and question; system 0.175 against an oracle of 0.49 on them; in the length table the "none" band (28 items) is 0.064. This needs words the version-1 decoder cannot choose (decoder v2 parked), and ROUGE under-credits paraphrase ("troubled" for "worried").
3. **Measuring through fragments**: reading short answers is unreliable (round trip 35% below 80 even in true word order; the tree score uses one of several gold answers and can be generous), so neither the round-trip gate nor the tree score can currently grade or train this system; ROUGE-L is the only measure trusted here. Smaller: 10 test stories (the length bands are tiny: 15 short items), single seed, stage-2 effect within val noise.

## How firm
Held-out by whole story, test looked at once (after the val-based choices of model and the declared abstain reward); single seed; the headline gap is several times its interval, the stage-1 vs stage-2 and (a) vs (b) differences are not. Hand-read is one rater.

## Artifacts (force-add; caches and pickles are not)
Checkpoints `runs/mind_v2/thinker_{s1_all,s2_all,s2b_all,s1_short}_best.pt` (0.86 MB each, main = `s2b_all`), curves `thinker_*.curve.tsv`, logs `s1_all.log s1_short.log s2_all.log s2b_all.log build_read.log build_items.log eval_val.log eval_test.log`, `vocab.json`, `eval_val.json eval_test.json`, `eval_handread_30.json eval_handread_30_judgments.json`. Not committed: `stories.pkl` (72 MB), `pairs_*.pkl`, `read_cache/`, `eval_rows_*.jsonl` (5 MB each, optional), `items_meta.json` (1 MB, needed to rebuild eval but regenerable).

## Reproduce
```
python scripts/build_mind_v2_data.py read --workers 4      # needs runs/mind_v1/read_cache (v1) plus 9,229 new reads
python scripts/build_mind_v2_data.py items --workers 4
python scripts/train_mind_v2.py --stage 1 --variant all   --epochs 24 --tag s1_all
python scripts/train_mind_v2.py --stage 1 --variant short --epochs 24 --tag s1_short
python scripts/train_mind_v2.py --stage 2 --init runs/mind_v2/thinker_s1_all_best.pt --epochs 6  --lr 3e-4 --tag s2_all
python scripts/train_mind_v2.py --stage 2 --init runs/mind_v2/thinker_s1_all_best.pt --epochs 20 --lr 5e-4 --seed 1 --tag s2b_all
python scripts/eval_mind_v2.py --split test --models s1_all=runs/mind_v2/thinker_s1_all_best.pt s2_all=runs/mind_v2/thinker_s2_all_best.pt s2b_all=runs/mind_v2/thinker_s2b_all_best.pt s1_short=runs/mind_v2/thinker_s1_short_best.pt --main s2b_all --short s1_short --alllen s1_all
```
