# Span diagnosis: where the thinker's gap to the oracle is (2026-11-18)

Read-only analysis of the Mind v2b thinker (`runs/mind_v2b/thinker_s2b_all_best.pt`, code `src/naomi/mind_v2.py`); nothing in the live code, no training. Scripts and raw outputs: `dev/reports/span_diagnosis_scripts/` (`s01` collect decisions, `s02` decomposition / anchor geometry / question types / wh attachment, `s03` feature shift by length, `s04` sentence selection and abstentions, `s05` why sentences are wrong, `s06` length prior; text outputs in `out/*.txt`). Analysis on the **val** split (1,025 items); the test table is the one final confirmation. Single model, single seed.

## Definitions (used in every table)
- Score = ROUGE-L F1 vs the best gold, true word order (no decoder), abstain = 0. Val 0.373, test 0.372 (matches MIND_V2B).
- **Oracle** = the best full subtree by ROUGE over every node as anchor (val 0.760, test 0.757). The gap is oracle minus model per item, so it can be slightly negative (the model prunes better than a full subtree): the "right" row is -2.7% of the loss.
- **Acceptable anchors** = every node whose full-subtree ROUGE is within 0.05 of the best. The "oracle sentence(s)" are the sentences holding an acceptable anchor.
- Categories, applied in this order: (a) abstained; (e) right = model ROUGE >= oracle - 0.05; (d) anchor is acceptable but ROUGE lower (split too much / too little by ROUGE precision vs recall); (c) anchor in an oracle sentence but not acceptable; (b) anchor in another sentence.
- Caveat: "wrong sentence" means a sentence that holds no near-best span. Sentences with partial overlap with the gold count as wrong, so a share of (b) is a near miss, not nonsense.

## Q1. Decomposition of the gap

Val, n=1025, oracle 0.760, model 0.373, gap 0.387. Items % and share of total lost ROUGE; last column converts the share to ROUGE points of the 0.387 gap.

| category | items | share of lost ROUGE | ROUGE points |
|---|---|---|---|
| (a) abstained | 5.4% | 7.0% | 0.027 |
| (b) wrong sentence | 38.2% | 63.9% | 0.247 |
| (c) right sentence, wrong anchor | 29.2% | 31.3% | 0.121 |
| (d) right anchor, too much kept | 0.3% | 0.0% | 0.000 |
| (d) right anchor, too little kept | 1.1% | 0.5% | 0.002 |
| (e) right (within 0.05) | 25.9% | -2.7% | -0.010 |

Test (n=1007, oracle 0.757, model 0.372): abstained 6.8% / 10.4%, wrong sentence 34.4% / 58.4%, right sentence wrong anchor 29.6% / 33.6%, keep too much 0.1% / 0.0%, too little 1.6% / 0.7%, right 27.6% / -3.1% (items / loss share). Same picture.

By answer-sentence length band (val; loss shares in %; gap in points):

| band | n | oracle | model | abstain | wrong sentence | wrong anchor | keep (d) |
|---|---|---|---|---|---|---|---|
| <=10 | 38 | 0.754 | 0.296 | 10.2 | 78.3 | 18.0 | 0.0 |
| 11-20 | 209 | 0.802 | 0.404 | 4.3 | 79.3 | 19.2 | 0.2 |
| 21+ | 708 | 0.801 | 0.404 | 6.6 | 59.1 | 36.1 | 0.6 |
| none (no sentence overlaps the gold) | 70 | 0.220 | 0.017 | 27.0 | 49.0 | 23.9 | 0.0 |

Reading: **the keep/drop head is not the problem (0.5% of the loss).** When the anchor is acceptable the span is right. The gap is choosing WHERE: 64% sentence, 31% anchor inside the sentence, 7% abstaining. Wrong anchors grow with sentence length (18% -> 36% of the loss), wrong sentences shrink in share only because anchors take over.

## Q2. Where the wrong anchor sits

Is the oracle anchor among the candidates? Yes: every node of every story sentence is scored (the anchor head scores all N nodes), so the candidate set always contains it. The question is rank. Items with oracle >= 0.5 (n=925), rank of the best acceptable anchor in the model's anchor scores: **top-1 22.7%, top-3 38.2%, top-5 45.6%, top-10 57.7%, median rank 6** (of ~185 nodes). Sentence level: the sentence holding the oracle anchor is the model's top sentence (by summed anchor probability) in **57.1%** of items.

(c) right sentence, wrong anchor (val n=299; test shape similar): relation of the model's anchor to the nearest acceptable anchor, structure only.

| relation | share | model text precision / recall | share of (c) loss |
|---|---|---|---|
| model anchor is the direct parent of the oracle anchor | 23.1% | 0.41 / 0.78 | 15.1% |
| model anchor is a higher ancestor | 25.4% | 0.30 / 0.70 | 24.0% |
| model anchor is a child | 12.7% | 0.91 / 0.47 | 7.4% |
| model anchor is a deeper descendant | 7.0% | 0.68 / 0.37 | 5.6% |
| sibling | 4.7% | 0.15 / 0.19 | 6.9% |
| same sentence, other branch of the same clause | 3.3% | 0.01 / 0.03 | 6.8% |
| same sentence, other clause (clause head = nearest verb/aux ancestor) | 23.7% | 0.18 / 0.19 | 34.1% |

Tree distance (edges between the two anchors): 1: 107, 2: 60, 3: 47, 4: 30, 5: 14, 6: 10, 7+: 31; mean 2.9. About **48% are on the oracle's own ancestor chain** (too broad: the model takes the clause or sentence that contains the answer, recall 0.7 / precision 0.3), 20% are descendants (too narrow), 24% sit in a different clause of the same sentence. 56% of (c) items are "too broad" overall. (b) wrong sentence: the oracle sentence is the adjacent one (distance 1) in 167 of 392 cases, within 2 in 240; signed offset (oracle minus model sentence): +1: 97, -1: 60, +2: 42, -2: 29, so the oracle sentence tends to come AFTER the chosen one.

Geometry of the target itself, by answer-sentence band (best acceptable anchor, oracle >= 0.5): mean depth 0.85 / 1.87 / 3.34, anchor is the sentence head (depth <= 1) 79% / 51% / 28%, subtree as share of the sentence 0.62 / 0.45 / 0.32 for <=10 / 11-20 / 21+. Long answers are inside the sentence, short answers are the sentence.

## Q3. By question type

Question type from the wh node's lexicon dimensions (node with the UNKNOWN dimension; group = its strongest other dimension in `quantum_parser/lexicons/en.json`; "+MUCH_MANY child" = the wh node has a dependent whose sense carries the MUCH_MANY dimension; no word lists). Val.

| wh dimension | n | oracle | model | gap | abstain / wrong sent / wrong anchor (loss %) |
|---|---|---|---|---|---|
| SOMETHING (what) | 426 | 0.825 | 0.400 | 0.424 | 5.3 / 71.4 / 23.7 |
| BECAUSE (why) | 216 | 0.745 | 0.347 | 0.397 | 5.9 / 63.0 / 32.2 |
| LIKE (how) | 160 | 0.545 | 0.286 | 0.259 | 18.8 / 56.5 / 39.2 |
| SOMEONE (who) | 86 | 0.866 | 0.461 | 0.405 | 9.2 / 38.0 / 55.2 |
| BECAUSE + MUCH_MANY child | 70 | 0.705 | 0.294 | 0.411 | 1.2 / 62.5 / 36.9 |
| WHERE | 38 | 0.888 | 0.546 | 0.342 | 7.7 / 61.2 / 32.1 |
| LIKE + MUCH_MANY child | 18 | 0.670 | 0.232 | 0.438 | 0.0 / 69.1 / 30.9 |

FairytaleQA attribute (val; the dataset's own label):

| attribute | n | oracle | model | gap | abstain / wrong sent / wrong anchor (loss %) |
|---|---|---|---|---|---|
| action | 333 | 0.826 | 0.412 | 0.414 | 5.0 / 64.6 / 30.9 |
| causal relationship | 294 | 0.738 | 0.333 | 0.405 | 4.5 / 63.7 / 32.6 |
| character | 107 | 0.849 | 0.440 | 0.409 | 9.6 / 37.4 / 54.8 |
| outcome resolution | 97 | 0.850 | 0.418 | 0.432 | 5.5 / 82.5 / 12.6 |
| feeling | 94 | 0.411 | 0.229 | 0.182 | 38.5 / 64.9 / 27.3 (abstains 25.5% of items) |
| prediction | 55 | 0.625 | 0.251 | 0.374 | 2.8 / 80.9 / 20.1 |
| setting | 45 | 0.900 | 0.547 | 0.353 | 12.6 / 59.5 / 28.7 |

| split | n | oracle | model | gap |
|---|---|---|---|---|
| explicit | 744 | 0.837 | 0.448 | 0.389 |
| implicit | 281 | 0.555 | 0.175 | 0.380 |
| local | 926 | 0.767 | 0.390 | 0.377 |
| summary | 99 | 0.696 | 0.221 | 0.475 |

Findings: the wh dimension and the attribute are the same grouping (SOMETHING = action / outcome, BECAUSE = causal, SOMEONE = character, WHERE = setting, LIKE = feeling and some actions). The absolute gap is nearly constant (0.38 to 0.44) for what / why / who / where; the failure mode shifts: **what / outcome / prediction are sentence-choice failures (63-83% of the loss)**, **who / character / setting are anchor-within-sentence failures (55% of the loss for who and character on val; character 50% and setting 57% on test)** (the answer is a short noun phrase inside a sentence the model finds), **how / feeling is abstention** (the oracle is lower, 0.41, and the model abstains on 25% of feeling items, 38% of their loss). Implicit and summary items are worse in the model score (0.175, 0.221) but not in the gap (the oracle drops with them). The test table has the same ordering (feeling abstain 51.6% of its loss, character wrong anchor 49.7%, setting 56.6%).

## Q4. Why short-trained models fail on long sentences

First the behaviour (val, items with oracle >= 0.5; test agrees: short-only model by band 0.291 / 0.203 / 0.093 vs main 0.19 / 0.43 / 0.40).

| selector | <=10 | 11-20 | 21+ |
|---|---|---|---|
| short-only model: top sentence = oracle sentence | 54.5% | 27.5% | 13.4% |
| all-lengths stage-1 model | 33.3% | 44.6% | 62.7% |
| main model | 39.4% | 49.5% | 61.0% |
| max question-coverage feature (sb4) | 21.2% | 36.8% | 53.7% |
| IDF best-overlap sentence | 27.3% | 33.8% | 54.5% |
| random sentence | 17.0% | 17.0% | 19.9% |

The short-only model on 21+ is below random at choosing the sentence (13.4% vs 19.9%): 77% of its loss there is wrong sentence, 18% abstain (it abstains on 16% of 21+ items). It is not failing to prune a long tree; it points at the wrong sentence. The reason is a **length prior learned as a feature**:

| model | within-item Spearman(sentence anchor mass, sentence length) | with question coverage (sb4) | with sentence index |
|---|---|---|---|
| short-only | -0.63 | -0.08 | -0.03 |
| all-lengths stage 1 | +0.51 | +0.49 | -0.02 |
| main | +0.34 | +0.46 | -0.02 |

The short model ranks sentences by shortness (chosen sentence length percentile within the story 0.30-0.33; chosen 13 nodes while the oracle sentence is 35). The all-lengths model has the opposite sign (percentile 0.76 in the 21+ band) and over-picks long sentences for short answers (chosen 18.0 vs oracle 10.5 nodes in the <=10 band). So sentence length is a usable input to the scorer and is exploited as a shortcut that depends on the training mix.

Which input features shift with length (reachable train + val items, answer sentence; d = standardised difference 21+ vs <=10; "outside" = share of 21+ values beyond the min/max seen in short-train items):

| feature | <=10 | 11-20 | 21+ | d | 21+ outside short-train range |
|---|---|---|---|---|---|
| max log1p(subtree size) of answer-sentence nodes (scal[1]) | 2.16 | 2.82 | 3.54 | +4.4 | 100% |
| answer-sentence length (nodes; implicit via counts) | 7.9 | 16.1 | 35.2 | +2.1 | 100% |
| max depth of answer-sentence nodes (scal[0] x 8) | 3.1 | 5.2 | 8.3 | +2.0 | 54% (88% above the short p95) |
| mean node depth | 1.59 | 2.60 | 4.16 | +1.9 | 54% |
| number of same-lemma-as-question nodes in the sentence (sb0 sum) | 1.7 | 3.8 | 7.1 | +1.25 | 24% |
| question coverage of the sentence relative to the best sentence (sb4 / max) | 0.32 | 0.57 | 0.71 | +0.97 | 0% |
| target anchor is the sentence head (depth <= 1) | 0.80 | 0.57 | 0.36 | -0.93 | - |
| target anchor log size (keep/anchor input) | 1.70 | 2.09 | 2.46 | +0.85 | 34% |
| target kept set: share of the sentence | 0.51 | 0.39 | 0.26 | -1.10 | 28% |
| FIND rank / FIND similarity of the answer sentence (sb5, sb6) | 0.40 / 0.063 | 0.41 / 0.035 | 0.42 / 0.020 | +0.04 / -0.38 | 0.1% |
| answer sentence is the top FIND sentence | 0.17 | 0.16 | 0.16 | -0.03 | 0% |
| story size (nodes), sentences per story, answer position in story | 169, 11.6, 0.47 | 186, 10.9, 0.45 | 200, 9.2, 0.48 | +0.3, -0.4, 0.0 | ~0% |

Input ranges: anchor-head candidate nodes in the answer sentence have depth/8 up to 0.88 (p95 0.50) in short items, up to 2.75 (p95 1.25) in 21+ items; keep-head relative depth up to 1.20 vs 3.00. Candidate nodes per answer sentence: 8 vs 35.

The features that make the learned function length-dependent: **(1) raw node depth and log subtree size** (`scal[0]`, `scal[1]`, and the keep head's relative depth / log size) which are proxies for "how big is this sentence" and move far outside the short-training range; **(2) sentence-level pooling** (`spn` = mean of node states over the sentence, and the coverage bit sb4) whose scale changes with the number of nodes; **(3) the number of competing candidates** inside the answer sentence (8 vs 35) while the softmax is over all nodes jointly, which lets the model trade sentence length against everything else. The pairwise FIND features (sb5, sb6) and the story-level counts do not shift, i.e. what composes is the question-to-sentence match; what does not compose is everything that describes the shape of the candidate.

## Q5. Question reading: where the wh node attaches

Structure only (parent tag / relation, "main verb" = the shallowest VERB-tagged node of the question tree). Val; test in brackets.

| attachment | n | oracle | model | gap |
|---|---|---|---|---|
| wh under an AUX | 568 (55%) | 0.775 | 0.377 | 0.399 |
| wh is the root, the verb sits below it | 215 (21%) | 0.709 | 0.329 | 0.380 |
| wh directly under the main verb | 215 (21%) | 0.774 | 0.398 | 0.376 |
| wh under a noun / adjective / adverb / other | 27 (3%) | ~0.7 | ~0.4 | n small |

Coarse split: wh under the main verb or root above it: n=430, model 0.363 [test 0.375]; wh under AUX or other (not under the main verb): n=595, model 0.381 [test 0.370]. The oracle anchor rank is the same (top-1 22.4% vs 22.9%, top-3 39.2% vs 37.4%). **Whether the wh node is attached under the main verb makes no measurable difference** (0.018 and -0.005 in opposite directions on val and test); 58% of questions have the wh under an auxiliary or not under the verb, and they score as well as the others. The decomposition is the same in both groups. The question reading is not where the gap lives; reading mistakes in individual questions (e.g. "why did the mother not know what they would do" gets an AUX root with `why` under it) do not separate by wh attachment.

## Q6. Abstentions on answerable items

Answerable = oracle >= 0.5 (925 items); abstained 33 (3.6%); all abstentions on all items 5.4%. The abstained answerable items cost 6.8% of the loss (test 9.9%) = 0.027 ROUGE.

| feature | abstained | answered | AUC (abstained higher) |
|---|---|---|---|
| best sentence's question coverage (item feature itf[1]) | 0.175 | 0.577 | **0.09** |
| max same-lemma content bit in the story (sb1) | 0.879 | 1.000 | 0.06 |
| top anchor logit | -7.2 | +4.5 | 0.18 |
| number of story nodes matching a question lemma | 21 | 30 | 0.30 |
| FIND max, question length, N nodes, n sentences, sentence length, answer sentence length | no difference | | 0.42-0.51 |
| oracle ROUGE (how answerable it truly is) | 0.800 | 0.823 | 0.44 |

Abstentions are decided by **lexical coverage of the question**: when no sentence shares enough question lemmas the model abstains, regardless of how answerable the item is (oracle ROUGE 0.80 vs 0.82, length and story size irrelevant). By group: feeling 19% (25% with oracle <0.5 too), implicit 7% vs explicit 3%, how/LIKE 18%; by band 2/33 (<=10), 5/204, 21/671. Had it been forced to answer, the abstained items would have scored **0.213** on average (45% above 0.10, 18% above 0.30) against 0 for abstaining and 0.10 as the training reward for abstaining, versus 0.80 achievable. So the abstain reward is set slightly low for the items it actually fires on (+0.007 ROUGE if forced), but the real cost is that the same signal that blocks the answer (no lexical match) is what blocks the sentence choice in (b) (see below). Calibration: items with p_abstain below 1e-6 average ROUGE 0.433 when answered, 0.208 for 1e-3 to 0.05, so p_abstain is also a decent (inverse) confidence score.

## Supplement: why the sentence is wrong (val, answered items with oracle >= 0.5)

The wrong sentence is **not mostly a ranking tie**: the oracle sentence has **zero question-lemma coverage in 39.4% of wrong-sentence items** (4.5% of right-sentence items), and is the max-coverage sentence in 18.6% of wrong-sentence items vs 81.8% of right ones. The model's chosen sentence is the max-coverage sentence in 59% of its wrong choices, so it follows the lexical signal and the lexical signal does not lead to the answer. The oracle sentence is within the model's top 2 sentences in 58% and top 4 in 81% of its wrong choices (mean rank 2.9). 15.7% of the gold's content words are absent from the story (7.8% in right-sentence items). The thinker's inputs contain no sentence index and no neighbour information: the tree rounds run inside a sentence (`Thinker._round` over `parent`), the sentence pool `spn` is a per-sentence mean, and the only cross-sentence features are the question-conditioned bits sb4-sb6. Sentences cannot see each other.

## Ranked changes most likely to close the gap

1. **Give the thinker sentence context and make sentence choice its own step (thinker features + framing; targets 0.247 of the 0.387 gap).** Add a sentence layer: one node per sentence carrying its pooled state, sentence index, relative offset to the best-coverage / best-FIND sentences, and the coverage of the previous and next sentence (a few attention or recurrent rounds over the sentence sequence), then pick a sentence (softmax over sentences, with its own loss and a top-k accuracy gate) and score anchors only inside the chosen sentence. Evidence: 64% of the lost ROUGE is wrong sentence; 39% of those have an oracle sentence with zero lexical overlap with the question; the true sentence is more often the one after the matched one (+1: 97 vs -1: 60); the model's sentence choice follows lexical coverage (59% of its errors are max-coverage sentences) and has no way to use neighbours; the coverage feature alone gets 49-54%, the model 58%, so the learned part adds little over the given feature. More data will not fix this (3.5x data gave +0.047) because no input can express "the sentence after the one that matches".
2. **Make the anchor/keep inputs length-invariant, and take length out of the sentence score (thinker features; targets about 0.12 of the gap and the short-to-long collapse).** Replace raw `depth/8` and `log1p(size)/4` with sentence-relative ones (depth below the sentence head, size as a fraction of the sentence, clause-head flag, position among clause heads), score the anchor among clause-level units first (56% of wrong anchors are a too-broad ancestor, 24% are another clause), and keep the sentence-level score free of node counts (normalised pooling, no raw length). Evidence: oracle anchor goes from the sentence head (79%) to a deep node (28% heads, depth 3.3) with length; the node scalars of long sentences are 100% / 54% outside the short-training range; the short model's sentence score correlates -0.63 with length and the all-lengths one +0.51 (a shortcut that flips with the training mix); keep head is already fine (0.5% of loss) so no change there. Test: re-run the short-to-long table; the short-only 21+ score (0.092) should rise toward the all-lengths one. Do not use length curricula as the remedy: they teach the shortcut.
3. **Fix abstention by decoupling it from lexical coverage (thinker framing + data; targets about 0.03 of the gap, mostly feeling / how / implicit items).** Evidence: abstentions on answerable items are explained by question coverage (AUC 0.09) and nothing else; they cost 7% of the loss (10% on test, 52% of the feeling-item loss); forced answers on those items would have averaged 0.21, above the 0.10 abstain reward. Concretely: abstain from the best sentence score after the change in (1), not from the item coverage; raise the answerability bar for the abstain reward (or train abstain only on the wrong-passage items, which is what it is for) and add training items whose answer sentence shares no lemma with the question (currently they are about 40% of the wrong-sentence cases but the target builder only rewards lexical anchors). This is smaller than 1 and 2 and should be done after them, since it shares the cause with (1).

Not worth doing (evidence): question reading / wh attachment (Q5: no effect), the keep/drop head (Q1: 0.5% of the loss), more of the same data (+0.047 for 3.5x), decoder work for this gap (the loss is in the tree choice, measured before any decoding).

## Reproduce
```
cd consciousness_transformer/dev/reports/span_diagnosis_scripts
for m in s2b_all s1_all s1_short; do python s01_collect.py val $m; done
python s01_collect.py test s2b_all; python s01_collect.py test s1_short
python s02_decompose.py val s2b_all; python s02_decompose.py val s1_short; python s02_decompose.py test s2b_all
python s03_features.py; python s04_sentence_and_abstain.py val; python s05_wrong_sentence.py val; python s06_length_prior.py val
```
Each script runs in under 2 minutes (the first load of `stories.pkl` dominates). The `out/*.pkl` intermediates are regenerable and should not be committed.
