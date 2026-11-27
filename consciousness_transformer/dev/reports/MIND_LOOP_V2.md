# Mind loop v2: longer halt prior + counterfactual memory

Status 2026-11-27. Builds on `MIND_LOOP.md` (v1). Same model (`mind_loop.Loop`, 0.79 M params; v1 checkpoints still load); two owner-approved training-side fixes, nothing else changed. Test split evaluated ONCE (seed s1, chosen on val).

## Plain conclusion

- **Halt fix works mechanically, the loop is now trained at every step, but it still does not think.** Halt no longer collapses: the learned halting distribution stays close to the prior (mean halting probability per step on val, s1: 0.31 / 0.22 / 0.15 / 0.10 / 0.07 / 0.14 for steps 1..6, prior 0.33 / 0.22 / 0.15 / 0.10 / 0.07 / 0.13). Per-step val tree loss is flat (3.59, 3.57, 3.56, 3.56, 3.56, 3.56): later steps are as good as step 1 but not better, so the learned halt has nothing to specialise on and simply stays at the prior. Iteration does not refine the answer.
- **Counterfactual memory did NOT make the model read the memory.** On held-out counterfactual items (same question, memory slots aligned to the answer swapped, answer swapped) the swapped word appears in the output text in only 1.8% of test items (val 1.5%); swapping memory changes the decoded text in only 7% of items; ROUGE vs the swapped gold 0.161 equals ROUGE of the same items with original memory vs original gold (0.170). Blocking memory reads changes ROUGE by -0.004 (v1: -0.158). Controls: wrong-story still not abstained (0.147 vs 0.180), shuffled 0.130, question-only 0.059 (v1 0.111), empty memory -> empty tree. The story adds 0.12 over question-only memory, but only 0.03 over a wrong story: the gain is generic shape of the story's vocabulary, not the answer.
- Headline test: ROUGE-L **0.180** vs cheat 0.201 (loop minus cheat -0.021, items [-0.035, -0.005]); v1 0.177, v3_s2 0.346, v5_all 0.298 (copied). Tree score 40.5 (v3_s2 38.3, v5_all 33.6, cheat 11.4; n=300 subsample). Essentially v1 performance. Failure stated plainly; no action head, copy menu or other fix was added.

## What was built

Files: `src/naomi/mind_loop_v2.py`, `scripts/train_mind_loop_v2.py`, `scripts/eval_mind_loop_v2.py` (wraps v1 `eval_mind_loop.main` with outputs redirected to `runs/mind_loop_v2/`, then adds the v2 extras), `tests/test_mind_loop_v2.py`, allow-list edit in `tests/test_naomi_no_literals.py`. v1 files untouched. v1 prep files read only.

1. **Halt schedule (declared exactly).** Cap 6 steps. Prior = geometric lambda_p = 1/3 (p = 0.333, 0.222, 0.148, 0.099, 0.066, last step takes the rest 0.132; mean about 2.7 steps). Epochs 0-3 (warm-up, 4 of 16): the halting distribution is HELD AT the prior, the loss is sum_n prior_n L_n, the halt head receives no gradient. From epoch 4: PonderNet with the learned halt read from the state, loss = sum_n p_n L_n + 0.3 KL(p || prior). (v1: beta 0.05, lambda 0.3 -> collapse.) Evaluation halting is still learned and deterministic: first step where cumulative halting probability reaches tau (tau in {0.3,0.5,0.7} chosen on val: s0 0.5, s1 0.7). Learning rate: 100-step warm-up then cosine to 10% over 16 epochs; AdamW 1e-3; batches of 12 stories; about 38 min per seed (2.4 min/epoch).
2. **Counterfactual memory items** (`CF.build`). For a training item and a randomly chosen reference tree: reference nodes whose identity (sense row if it has a USVS point, else identity key for names/lexicon-less words) occurs in the story's memory, does not occur in the question, and has low document frequency (<= 2% of the split's stories; a data statistic, no word list). Up to 3 such identities are replaced: every memory slot with that identity gets the content (USVS point row, identity key, node) of a slot of the SAME tag (pos) from a different story of the same split, one donor per identity, no donor identity already in the story; the same identity in the reference tree is replaced by the same donor. Relation, pos, parent links, sentence index, question: unchanged. 56% of train items admit a counterfactual (about 1.6 swaps each). Rate: 40% of training items get an extra counterfactual item (in addition to the normal item); v1's wrong-story (12%) and empty-memory (6%) empty-target items kept. Counterfactual val/test items are built the same way from val/test stories (donors from the same split), deterministic per item; only items where the swapped surface token occurs in a gold answer are scoreable (val 456, test 450 of ~1000); gold text = gold with that whole token replaced (case-insensitive).
3. **Seeds and selection.** 2 seeds (s0, s1), 16 epochs each. Checkpoint per seed = best score after the release epoch of 0.5 (val ROUGE + cf ROUGE on 150 val items each) - 0.001 val loss. Seed chosen on the full val set: s1 (val ROUGE 0.187, s0 0.177). Test once for s1 only.

## Test (once): seed s1, thr 0.5, tau 0.7

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop v2 (decoded output tree, setting (b)) | 0.180 | 40.5 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.201 (raw text 0.220) | 11.4 |
| v1 loop (MIND_LOOP.md) | 0.177 | 35.5 |
| v3_s2 (MIND_V5.md, copied) | 0.346 | 38.3 |
| v5_all (MIND_V5.md, copied) | 0.298 | 33.6 |

Loop minus cheat: -0.021 (items [-0.035, -0.005]; stories [-0.032, -0.008]). Mean output nodes 2.99, answer rate 1.0.

| band | n | loop v2 | cheat | mean steps | mean nodes |
|---|---|---|---|---|---|
| <=10 | 35 | 0.151 | 0.098 | 3.91 | 3.09 |
| 11-20 | 202 | 0.186 | 0.196 | 3.84 | 2.89 |
| 21+ | 688 | 0.196 | 0.227 | 3.91 | 3.23 |
| none | 82 | 0.045 | 0.041 | 3.99 | 1.23 |

Short-only model not rerun (time; v1's short-only overfit and lost everywhere).

### Controls (test, ROUGE vs own reference)

| control | abstain rate | mean nodes | ROUGE | v1 test |
|---|---|---|---|---|
| normal | 0.000 | 2.99 | 0.180 | 0.177 |
| empty memory | 1.000 | 0.00 | 0.000 | 0.000 |
| wrong story | 0.000 | 2.97 | 0.147 | 0.144 |
| question-only memory | 0.000 | 1.70 | 0.059 | 0.111 |
| shuffled questions | 0.000 | 3.02 | 0.130 | 0.119 |
| reads blocked | 0.000 | 2.72 | 0.176 | 0.019 |

"Story adds" (normal minus control): vs question-only +0.121 (v1 +0.066), vs wrong story +0.033 (v1 +0.033), vs shuffled +0.050. Empty memory still yields the empty tree (taught by the empty items); wrong story still does not abstain.

### Counterfactual accuracy (held-out, test n=450; val n=456)

| measure | test | val |
|---|---|---|
| swapped (new) token in decoded text | 0.018 | 0.015 |
| original (old) token in decoded text, counterfactual memory | 0.000 | 0.000 |
| ROUGE vs swapped gold, counterfactual memory | 0.161 | 0.162 |
| ROUGE vs original gold, original memory (same items) | 0.170 | 0.162 |
| ROUGE vs original gold, counterfactual memory (stale answer) | 0.158 | 0.157 |
| original token in text, original memory | 0.029 | 0.024 |
| decoded text changed by the swap | 0.071 | 0.064 |

Reading: with original memory the model also rarely emits the answer-specific token (2.9%), so there was little to flip; the answers are generic and memory-independent, and the small cf-ROUGE is the same generic answer shape. Seed s0 (val): cf new-token 0.022, text changed by swap 0.114, story adds vs question-only +0.065: same picture.

### Halting (val, s1; test same within 0.01)

Learned halting probability per step (mean over items): 0.315, 0.223, 0.151, 0.102, 0.068, 0.142 (prior 0.333, 0.222, 0.148, 0.099, 0.066, 0.132). Evaluation rule (first step reaching cumulative tau=0.7): step 3 in 102 items, step 4 in 923 (test 100 / 907); with tau 0.5 (s0) step 2 in 1024. Because the distribution is the prior, a single-step rule is nearly constant per tau: the halt is learned from the state but carries almost no item-specific information. Per-step val tree loss (all 1025 items): [3.5865, 3.5685, 3.5571, 3.556, 3.557, 3.5574].

## Curves

Seed s1 (chosen); warm = halt held at prior; val ROUGE at thr 0.5 on 150 val items, cf on 150 val counterfactual items.

| ep | warm | train | val loss | val ROUGE | cf ROUGE | cf new-token | mean halt step | per-step val loss |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 4.6461 | 4.3054 | 0.0400 | 0.0376 | 0.053 | 2.089 | [4.2741, 4.3008, 4.3309, 4.3464, 4.3149, 4.3218] |
| 1 | 1 | 3.8240 | 4.0317 | 0.0708 | 0.0617 | 0.040 | 2.240 | [3.9879, 4.0511, 4.0174, 4.0528, 4.0473, 4.0654] |
| 2 | 1 | 3.5820 | 3.8902 | 0.1377 | 0.1357 | 0.013 | 2.456 | [3.8819, 3.8625, 3.8735, 3.8986, 3.8912, 3.9012] |
| 3 | 1 | 3.4689 | 3.7496 | 0.1421 | 0.1141 | 0.007 | 2.259 | [3.777, 3.7173, 3.7098, 3.7019, 3.7056, 3.7077] |
| 4 | 0 | 3.4049 | 3.6628 | 0.1510 | 0.1453 | 0.013 | 2.775 | [3.6828, 3.6609, 3.6338, 3.661, 3.6392, 3.6639] |
| 5 | 0 | 3.3461 | 3.5695 | 0.1799 | 0.1554 | 0.020 | 2.807 | [3.5857, 3.5612, 3.5664, 3.5565, 3.5653, 3.561] |
| 6 | 0 | 3.3123 | 3.5586 | 0.1721 | 0.1593 | 0.040 | 2.729 | [3.5745, 3.5453, 3.553, 3.5555, 3.5459, 3.5573] |
| 7 | 0 | 3.2867 | 3.6345 | 0.1569 | 0.1393 | 0.013 | 2.800 | [3.6267, 3.6254, 3.6353, 3.6463, 3.6296, 3.6561] |
| 8 | 0 | 3.2491 | 3.5393 | 0.1634 | 0.1480 | 0.007 | 2.778 | [3.5316, 3.5424, 3.5429, 3.5405, 3.5391, 3.5529] |
| 9 | 0 | 3.2145 | 3.5430 | 0.1631 | 0.1446 | 0.013 | 2.767 | [3.5515, 3.5411, 3.5439, 3.5353, 3.5414, 3.5348] |
| 10 | 0 | 3.2007 | 3.5199 | 0.1720 | 0.1563 | 0.013 | 2.765 | [3.5373, 3.5061, 3.522, 3.5087, 3.5159, 3.5158] |
| 11 | 0 | 3.1604 | 3.5246 | 0.1673 | 0.1432 | 0.027 | 2.756 | [3.5201, 3.5268, 3.5282, 3.5182, 3.5299, 3.5279] |
| 12 | 0 | 3.1499 | 3.5494 | 0.1562 | 0.1434 | 0.020 | 2.730 | [3.5474, 3.5448, 3.5525, 3.5477, 3.5505, 3.5597] |
| 13 | 0 | 3.0967 | 3.5359 | 0.1582 | 0.1462 | 0.020 | 2.754 | [3.537, 3.5202, 3.5461, 3.5371, 3.5485, 3.5442] |
| 14 | 0 | 3.0848 | 3.5772 | 0.1501 | 0.1400 | 0.040 | 2.728 | [3.5578, 3.5704, 3.5852, 3.59, 3.5982, 3.6111] |
| 15 | 0 | 3.0584 | 3.5681 | 0.1545 | 0.1545 | 0.027 | 2.785 | [3.561, 3.5615, 3.5784, 3.5737, 3.5776, 3.575] |

Seed s0:

| ep | warm | train | val loss | val ROUGE | cf ROUGE | cf new-token | mean halt step | per-step val loss |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 4.7230 | 4.8430 | 0.0176 | 0.0219 | 0.020 | 4.161 | [4.7605, 4.8069, 4.7822, 4.8087, 4.9032, 4.8629] |
| 1 | 1 | 3.9600 | 4.2301 | 0.0250 | 0.0234 | 0.027 | 3.919 | [4.2482, 4.2153, 4.2067, 4.2154, 4.1952, 4.226] |
| 2 | 1 | 3.8471 | 4.2968 | 0.0179 | 0.0243 | 0.033 | 3.355 | [4.2499, 4.2622, 4.3242, 4.3205, 4.2908, 4.3349] |
| 3 | 1 | 3.8079 | 4.3135 | 0.0169 | 0.0247 | 0.027 | 3.683 | [4.4022, 4.3065, 4.3023, 4.2458, 4.3007, 4.2771] |
| 4 | 0 | 3.7801 | 4.2758 | 0.0185 | 0.0277 | 0.020 | 2.782 | [4.2123, 4.2826, 4.3384, 4.3363, 4.2539, 4.2983] |
| 5 | 0 | 3.6752 | 3.9004 | 0.1004 | 0.1129 | 0.033 | 2.714 | [3.9041, 3.8714, 3.9017, 3.9418, 3.9304, 3.8996] |
| 6 | 0 | 3.5552 | 3.7552 | 0.1588 | 0.1598 | 0.020 | 2.587 | [3.7374, 3.7498, 3.7854, 3.7756, 3.7674, 3.7659] |
| 7 | 0 | 3.4605 | 3.7297 | 0.1723 | 0.1531 | 0.033 | 2.741 | [3.739, 3.7397, 3.705, 3.7283, 3.7433, 3.7093] |
| 8 | 0 | 3.4236 | 3.7202 | 0.1600 | 0.1353 | 0.027 | 2.607 | [3.7383, 3.6948, 3.714, 3.7186, 3.7297, 3.7134] |
| 9 | 0 | 3.3775 | 3.6779 | 0.1702 | 0.1645 | 0.033 | 2.739 | [3.6779, 3.674, 3.6715, 3.6901, 3.6824, 3.6749] |
| 10 | 0 | 3.3310 | 3.6533 | 0.1493 | 0.1204 | 0.007 | 2.736 | [3.6815, 3.6272, 3.6513, 3.6356, 3.6496, 3.6409] |
| 11 | 0 | 3.3342 | 3.6145 | 0.1485 | 0.1321 | 0.020 | 2.712 | [3.6303, 3.5954, 3.6206, 3.6173, 3.6129, 3.5932] |
| 12 | 0 | 3.3035 | 3.6366 | 0.1492 | 0.1238 | 0.020 | 2.712 | [3.6399, 3.6182, 3.6459, 3.6596, 3.6476, 3.6242] |
| 13 | 0 | 3.3088 | 3.6421 | 0.1365 | 0.1127 | 0.027 | 2.733 | [3.6486, 3.6325, 3.6527, 3.6234, 3.643, 3.6431] |
| 14 | 0 | 3.2694 | 3.6047 | 0.1539 | 0.1250 | 0.007 | 2.705 | [3.616, 3.5907, 3.611, 3.6202, 3.5761, 3.5913] |
| 15 | 0 | 3.2399 | 3.6212 | 0.1442 | 0.1243 | 0.007 | 2.741 | [3.6229, 3.6013, 3.6356, 3.6297, 3.6243, 3.623] |

## Deviations and caveats (declared)

1. Training was started with `nohup ... &` (a foreground tool call is capped at 10 min); one run at a time, `nice`.
2. 2 seeds x 16 epochs (about 38 min each); no hyperparameter search (lambda 1/3, beta 0.3, warm 4, p_cf 0.4, max_df 2% fixed in advance). Warm-up length and beta not ablated. Checkpoint score uses counterfactual val ROUGE as half of the criterion (declared).
3. Val ROUGE during training is noisy (150 items, threshold 0.5); seeds differ by 0.01 on full val.
4. Counterfactual gold is surface-token replacement; swaps whose token never appears in a gold answer are excluded from the cf metrics (about 55% of cf-capable items).
5. Donor words are random same-tag words, so a counterfactual story can read oddly; this is what makes the question unable to predict the answer.
6. Short-only rerun skipped. v3_s2/v5_all numbers copied from MIND_V5.md; cheat recomputed (0.201).
7. Nothing else changed: no action heads, no copy menu; memory pre-stored and never written by T (tests).

## Honest conclusion

The halt prior and warm-up fix the training mechanics (every loop step is trained, the halt does not collapse), and the counterfactual items are well-formed and measurable on held-out stories, yet the model learns to answer without reading memory (reads-blocked ROUGE unchanged, swap flips the output in 7% of items). The Hungarian-matched regression of a 614-d meaning point gives no strong gradient path toward "find the memory slot and reproduce its meaning", and the counterfactual mass (40% of items) did not change that within 16 epochs at this size. Not done (outside the brief): a higher counterfactual rate or a counterfactual-only curriculum, a larger model, a loss on the memory attention itself.

## Files

Commit (force-add `runs/` files): `src/naomi/mind_loop_v2.py`, `scripts/train_mind_loop_v2.py`, `scripts/eval_mind_loop_v2.py`, `tests/test_mind_loop_v2.py`, `tests/test_naomi_no_literals.py`, `dev/reports/MIND_LOOP_V2.md`; in `runs/mind_loop_v2/`: `v2_s1_best.pt` (chosen, 3.1 MB), `v2_s0_best.pt`, `v2_s0.curve.tsv`, `v2_s1.curve.tsv`, `v2_s0.log`, `v2_s1.log`, `eval_val_s0.log`, `eval_val_s1.log`, `eval_test_s1.log`, `eval_val_s0.json`, `eval_val_s1.json`, `eval_test_s1.json`, `extras_val_s0.json`, `extras_val_s1.json`, `extras_test_s1.json`, `tables_val_s0.md`, `tables_val_s1.md`, `tables_test_s1.md`, `handread_*.json`, `val_choice*.json`. Not to add: `eval_rows_*.jsonl`, `*_last.pt`, `read_cache/`.

Commands: `python scripts/train_mind_loop_v2.py train --tag v2_s1 --seed 1 --epochs 16 --warm 4`; `python scripts/eval_mind_loop_v2.py --split val|test --ckpt runs/mind_loop_v2/v2_s1_best.pt --tag _s1` (val first).
