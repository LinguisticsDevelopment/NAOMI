# Mind v4: answering by stepping down the tree

Status 2026-11-19. Test evaluated once, after the checkpoint and the walk settings were fixed on val. Everything below is reproducible from the files listed at the end.

## Plain conclusion

- **v4 does not beat v3.** Test ROUGE-L (decoder (a)) 0.297 against v3_s2 0.345 and v2b 0.348: v4 - v3_s2 = -0.048 (items -0.066 to -0.029, stories -0.066 to -0.035). It is above the cheat baseline (+0.077, interval +0.058 to +0.097) and far below the oracle (0.757). The walk answers 94% of items and abstains correctly on wrong-story items (0.94), so the loss is not abstaining; it is where the walk lands.
- **Short-to-long transfers a little better in relative terms and no better in the thing that matters.** Trained only on items whose answer sentence has at most 10 nodes, v4_short gets 0.144 on the 21+ band (v4_all 0.320): 45% of its all-lengths model. v3_short gets 0.110 (v3_s1 0.361): 30%. But the absolute number is still less than half of the all-lengths model, and when the answer sentence is GIVEN (descent probe) v4_short finds an acceptable anchor in 10% of 21+ items (v3_short 20%; ROUGE 0.274 vs 0.282, a tie). The recursion does extend: the short-trained v4 walks to depth 4.7 on long sentences (oracle 2.8) although none of its training answers was deeper than 6 and the mean was 0.9, and 13% of its answers sit below its deepest training answer. What does not transfer is knowing WHERE to stop and which branch: it overshoots in depth and picks the wrong branch. The same-small-operation-applied-recursively idea is mechanically sound; as trained here it does not learn the stopping rule from short sentences.
- **Long-sentence failure is mostly the policy, not the tree, but the tree caps a real share.** Hand-read of 20 val 21+ failures: 10 policy errors in a tree that held the answer, 4 policy errors where the tree also had no clean subtree for the answer (best subtree ROUGE 0.5 to 0.67), 5 where the reference is not in the story text (inference or new words), 1 inflection mismatch; 0 failures that were the tree alone. Over all test items, the share whose reference equals some subtree at ROUGE >= 0.8 is 0.57 / 0.58 / 0.57 for the <=10 / 11-20 / 21+ bands: the tree is not worse on long sentences. Among 21+ items with a reachable answer, 16% fail in the right sentence with a tree that has no >= 0.8 subtree; 27% fail in the right sentence with a tree that did have one; 33% fail by choosing the wrong sentence.

## What was built

Spec: the owner-approved method change of 2026-11-19 (same small learned operation applied recursively; answer-only signal; chain encoder -> thinker -> decoder as in `dev/design/CONSCIOUSNESS_V2_PROPOSAL.md`). Code: `src/naomi/mind_v4.py`, `scripts/{train,eval,probe,tables}_mind_v4.py`, `tests/test_mind_v4.py`. Data, targets, keep sets, decoders, ROUGE and the tables code are v2/v3's (the v2b full FairytaleQA build, reused through symlinks in `runs/mind_v4/`; `tables_mind_v3.py` gained a `--dir` argument).

**The policy.** The state is the current node. Inputs: the question tree's representation (as v3), the node, its immediate children (relation, USVS point, subtree summary, child order, max match bits over the subtree) and the parent's relation. No absolute depth, no sentence length, no node count, no position in the sentence beyond child order, no word identity. One recursive merge (same weights at every level; depth only schedules the bottom-up order) gives every node a subtree state; two question cross-attentions put the question into it. Actions: HERE, DOWN(k), NEXT, PREV (roots only), ABSTAIN. A step first picks a move (stay / NEXT / PREV) and, when staying, an in-sentence action (HERE / ABSTAIN / DOWN(k)). The logit of DOWN(k) is the log-mass of the HERE scores inside child k's subtree (the same bottom-up merge, lse(here, children)) plus a learned local correction, so a descent is the chain rule of a softmax over the nodes below, applied one step at a time. NEXT / PREV climb a learned sentence value (move only if the neighbour beats the current sentence by a learned positive margin), so the walk cannot ping-pong. After HERE the v3 keep/drop head prunes per descendant. Because the policy is Markov in the node, logits for all nodes are computed at once; training is cross-entropy on the gold states and walking is a loop over those logits.

**Training (answer-only).** Targets come from the reference answer exactly as in v2/v3 (accepted anchors within 0.05 lemma-F1 of the best, at most three, keep sets). States on the gold paths: roots from the start sentence to the answer sentence (NEXT/PREV toward the nearest answer sentence), then the ancestors of every accepted anchor (DOWN toward them, HERE at them; every acceptable action counts); one extra random start sentence per item teaches NEXT/PREV from anywhere; wrong-passage items train ABSTAIN at every root. A softmax over the story's sentence roots toward the answer sentence trains the sentence value. Items with no acceptable anchor are skipped in training (1012 of 8548 train items). 24 epochs, batch 16, same optimiser as v3, no stage 2 (reward fine-tuning was within noise for v3 and was skipped for time). Training wall-clock about 55 minutes in total over five runs (below), run one at a time.

**Deviations from the brief, all declared before the test run.**
1. Start sentence: the walk starts at the root with the best LEARNED sentence value, not the top FIND root. With the FIND start the same checkpoint scores 0.243 on val against 0.317 (val sweep below). The value is the same quantity the moves climb. So sentence selection is one learned softmax over the sentence roots, as in v3; only the descent is stepwise.
2. Step budget 20, not 12 (val: 0.319 vs 0.317, exhaustion 0.9% -> 0); mean steps are 3 to 4.
3. ABSTAIN additionally sees the story's best coverage and best FIND values (item-level numbers, not sentence-wide pooling of node states); roots see signed offsets to the best FIND / coverage sentences (as v3).
4. PREV and the neighbour value are included; "top-k starts" and a decision-time HERE bias were tried on val and rejected (table below).

**Development path on val (kept as checkpoints, not as results).** Only the last row is the reported model.

| run | change | val ROUGE-L (true order), best epoch | why it was dropped |
|---|---|---|---|
| v4a_freenav | free NEXT/PREV logits from node and neighbour states, FIND start | 0.258 (ep 9, killed ep 10) | 24-30% of walks used the whole budget ping-ponging between two sentences |
| v4b_hillclimb | NEXT/PREV = value difference minus margin (no loops), FIND start | 0.238 (ep 6, killed in ep 7) | right sentence in only 40% of items: FIND start + local climbing is a poor sentence selector |
| v4c_localmlp | + sentence-value softmax loss, start at the best value, DOWN from a learned MLP | 0.262 (ep 8, killed in ep 9) | in-sentence anchor mostly too narrow; descent had no view of the mass below |
| **v4_all** | DOWN logit = log-mass of HERE scores below + local correction | **0.317 (ep 23)** | reported model |

Val sweep on the final checkpoint (val_rouge with true word order; answered/abstain not changed by the settings): HERE bias +1/+2/+3: 0.309/0.288/0.266, -1/-2: 0.297/0.266 (so no calibration helps); FIND start 0.243; top-3 starts by summed log-prob 0.306; budget 20: 0.319. Chosen: bias 0, value start, one start, budget 20.

## Test results (n=1007 items, 23 stories; single run; main model v4_all declared on val)

### test: n=1007 items, 23 stories; bands {'<=10': 35, '11-20': 202, '21+': 688, 'none': 82}

### Headline ROUGE-L (max over golds, abstain = 0)

| system | ROUGE-L | answered-only | answer rate |
|---|---|---|---|
| cheat: best-overlap sentence | 0.220 | nan | nan |
| baseline: question echo | 0.116 | nan | nan |
| baseline: random story sentence | 0.136 | nan | nan |
| baseline: random-anchor full subtree | 0.078 | nan | nan |
| oracle: best full subtree (by ROUGE, true order) | 0.757 | nan | nan |
| oracle: lemma-F1 pruned subtree (true order) | 0.752 | nan | nan |
| v4_all: (a) order-only decoder | 0.297 | 0.315 | 0.943 |
| v4_all: (b) meaning-only decoder | 0.293 | 0.311 | 0.943 |
| v4_all: true word order (no decoder) | 0.315 | 0.334 | 0.943 |
| v3_s2: (a) order-only decoder | 0.345 | 0.374 | 0.924 |
| v3_s2: (b) meaning-only decoder | 0.338 | 0.366 | 0.924 |
| v3_s2: true word order (no decoder) | 0.369 | 0.399 | 0.924 |
| v2b: (a) order-only decoder | 0.348 | 0.374 | 0.932 |
| v2b: (b) meaning-only decoder | 0.342 | 0.367 | 0.932 |
| v2b: true word order (no decoder) | 0.372 | 0.399 | 0.932 |
| v3_s1: (a) order-only decoder | 0.339 | 0.365 | 0.929 |
| v3_s1: (b) meaning-only decoder | 0.334 | 0.360 | 0.929 |
| v3_s1: true word order (no decoder) | 0.367 | 0.395 | 0.929 |

### By answer-sentence length band (decoder (a))

| band | n | cheat | v4_all | v3_s2 | v2b | v3_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 35 | 0.104 | 0.150 | 0.228 | 0.168 | 0.177 | 0.815 |
| 11-20 | 202 | 0.205 | 0.347 | 0.405 | 0.410 | 0.406 | 0.802 |
| 21+ | 688 | 0.251 | 0.320 | 0.362 | 0.372 | 0.361 | 0.792 |
| none | 82 | 0.045 | 0.046 | 0.103 | 0.072 | 0.059 | 0.331 |

### Short-to-long (decoder (a))

| pair | band | n | all lengths | <=10 only |
|---|---|---|---|---|
| v4_short vs v4_all | <=10 | 35 | 0.150 | 0.250 |
| v4_short vs v4_all | 11-20 | 202 | 0.347 | 0.163 |
| v4_short vs v4_all | 21+ | 688 | 0.320 | 0.144 |
| v4_short vs v4_all | none | 82 | 0.046 | 0.035 |
| v3_short vs v3_s1 | <=10 | 35 | 0.177 | 0.343 |
| v3_short vs v3_s1 | 11-20 | 202 | 0.406 | 0.176 |
| v3_short vs v3_s1 | 21+ | 688 | 0.361 | 0.110 |
| v3_short vs v3_s1 | none | 82 | 0.059 | 0.022 |

### Paired differences (decoder (a); mean, 95% interval over items, over stories)

- v4_all - v3_s2: -0.048 (items -0.066 to -0.029; stories -0.066 to -0.035)
- v4_all - cheat: +0.077 (items +0.058 to +0.097; stories +0.057 to +0.098)
- v4_all - v2b: -0.051 (items -0.071 to -0.033; stories -0.073 to -0.031)
- v4_all - v3_s1: -0.042 (items -0.061 to -0.024; stories -0.061 to -0.026)

### By FairytaleQA attribute (decoder (a))

| attribute | n | cheat | v4_all | v3_s2 | v2b | v3_s1 | oracle | answer rate (main) |
|---|---|---|---|---|---|---|---|---|
| action | 315 | 0.240 | 0.324 | 0.355 | 0.349 | 0.365 | 0.825 | 0.930 |
| causal relationship | 278 | 0.282 | 0.287 | 0.334 | 0.350 | 0.331 | 0.740 | 0.960 |
| feeling | 106 | 0.053 | 0.248 | 0.234 | 0.269 | 0.241 | 0.462 | 0.877 |
| character | 103 | 0.178 | 0.304 | 0.439 | 0.457 | 0.424 | 0.882 | 0.961 |
| outcome resolution | 78 | 0.285 | 0.348 | 0.404 | 0.328 | 0.373 | 0.813 | 0.962 |
| prediction | 65 | 0.120 | 0.179 | 0.226 | 0.177 | 0.189 | 0.582 | 0.969 |
| setting | 62 | 0.222 | 0.336 | 0.427 | 0.502 | 0.390 | 0.898 | 0.968 |

### Controls

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v4_all | 1.000 | 0.939 | 0.479 / 0.061 | 0.167 |
| v3_s2 | 1.000 | 0.950 | 0.411 / 0.055 | 0.855 |
| v2b | 1.000 | 0.946 | 0.483 / 0.061 | 0.836 |
| v3_s1 | 1.000 | 0.968 | 0.406 / 0.054 | 0.002 |
| v4_short | 1.000 | 0.929 | 0.431 / 0.050 | 0.001 |
| v3_short | 1.000 | 0.913 | 0.416 / 0.044 | 0.000 |

### Tree score and round trip (main = v4_all; threshold 99)

- (a) tree score answered 36.0, abstain=0 34.0; new round-trip pass 0.598; by length {"1-3 words": {"n": 245, "pass_rate": 0.9836734693877551}, "4-6 words": {"n": 166, "pass_rate": 0.7469879518072289}, "7+ words": {"n": 539, "pass_rate": 0.37662337662337664}}; reasons {"node_extra": 3241, "node_dropped": 1763, "meaning_shift": 1343, "operator_folded_differs": 828, "head_changed": 592, "subtree_moved": 450, "name_mismatch": 225, "relation_changed": 161, "opposite_meaning": 2}
- (b) tree score answered 35.9, abstain=0 33.9; new round-trip pass 0.579; by length {"1-3 words": {"n": 245, "pass_rate": 0.963265306122449}, "4-6 words": {"n": 166, "pass_rate": 0.7168674698795181}, "7+ words": {"n": 539, "pass_rate": 0.36178107606679033}}; reasons {"node_extra": 3238, "node_dropped": 1763, "meaning_shift": 1350, "operator_folded_differs": 833, "head_changed": 588, "subtree_moved": 452, "name_mismatch": 228, "relation_changed": 160, "opposite_meaning": 2}
- cheat tree score 13.6

### Needs-new-words items

- v4_all: (a) order-only decoder: 0.144 (n=224)
- v4_all: (b) meaning-only decoder: 0.140 (n=224)
- v4_all: true word order (no decoder): 0.148 (n=224)
- v3_s2: (a) order-only decoder: 0.167 (n=224)
- v3_s2: (b) meaning-only decoder: 0.165 (n=224)
- v3_s2: true word order (no decoder): 0.173 (n=224)
- v2b: (a) order-only decoder: 0.149 (n=224)
- v2b: (b) meaning-only decoder: 0.147 (n=224)
- v2b: true word order (no decoder): 0.157 (n=224)
- v3_s1: (a) order-only decoder: 0.152 (n=224)
- v3_s1: (b) meaning-only decoder: 0.150 (n=224)
- v3_s1: true word order (no decoder): 0.161 (n=224)


## v4 diagnostics (test)

#### Walk diagnostics

Short-trained items: oracle anchor depth max 6, 95th pct 3, mean 0.86.

#### Steps, budget exhaustion, chosen anchor depth vs oracle depth, by answer-sentence band

| model | band | n | answer rate | mean steps | budget exhausted | chosen depth (answered) | oracle depth | chosen deeper than short-train max | chosen anchor is acceptable | chosen sentence is the oracle sentence |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_all | <=10 | 35 | 0.914 | 3.11 | 0.000 | 2.31 | 1.03 | 0.094 | 0.091 | 0.182 |
| v4_all | 11-20 | 202 | 0.965 | 3.01 | 0.000 | 2.08 | 1.53 | 0.031 | 0.260 | 0.505 |
| v4_all | 21+ | 688 | 0.948 | 4.13 | 0.000 | 3.30 | 2.78 | 0.117 | 0.251 | 0.616 |
| v4_all | none | 82 | 0.866 | 3.66 | 0.000 | 3.07 | nan | 0.042 | nan | nan |
| v4_short | <=10 | 35 | 0.914 | 1.46 | 0.000 | 0.50 | 1.03 | 0.000 | 0.152 | 0.303 |
| v4_short | 11-20 | 202 | 0.911 | 2.38 | 0.000 | 1.51 | 1.53 | 0.038 | 0.083 | 0.312 |
| v4_short | 21+ | 688 | 0.887 | 3.58 | 0.000 | 2.89 | 2.78 | 0.131 | 0.031 | 0.254 |
| v4_short | none | 82 | 0.817 | 2.95 | 0.000 | 2.39 | nan | 0.104 | nan | nan |

#### Tree adequacy: share of items whose reference answer equals some subtree at ROUGE-L >= 0.8 (any node as anchor, true word order)

| band | n | share >= 0.8 | share >= 0.5 | mean best-subtree ROUGE |
|---|---|---|---|---|
| <=10 | 35 | 0.571 | 0.943 | 0.815 |
| 11-20 | 202 | 0.579 | 0.936 | 0.802 |
| 21+ | 688 | 0.567 | 0.920 | 0.792 |
| none | 82 | 0.159 | 0.378 | 0.331 |
| all | 1007 | 0.536 | 0.880 | 0.757 |

#### Failure attribution, v4_all (items with a reachable oracle answer; ROUGE-L true order; fail = ROUGE < 0.5 x oracle subtree ROUGE... see columns)

| band | n | right (within 0.05 of oracle subtree) | abstained / budget out | wrong sentence | right sentence, wrong anchor, TREE adequate (>=0.8) | right sentence, wrong anchor, tree NOT adequate |
|---|---|---|---|---|---|---|
| <=10 | 33 | 0.091 (3) | 0.061 (2) | 0.758 (25) | 0.030 (1) | 0.061 (2) |
| 11-20 | 192 | 0.234 (45) | 0.036 (7) | 0.453 (87) | 0.156 (30) | 0.120 (23) |
| 21+ | 649 | 0.185 (120) | 0.046 (30) | 0.331 (215) | 0.274 (178) | 0.163 (106) |


### Descent probe (test): the answer sentence is given

The walk starts at the root of the oracle answer sentence and may not move between sentences; v3 takes the best node of that sentence (its within-sentence softmax) and both use their own keep head. This separates descent from sentence selection. Val gives the same picture (`runs/mind_v4/probe_val.md`).

| model | band | n | anchor acceptable | chosen depth | oracle depth | ROUGE-L (true order, v4 or v3 keep head) |
|---|---|---|---|---|---|---|
| v4_all | <=10 | 33 | 0.636 | 0.77 | 1.03 | 0.566 |
| v4_all | 11-20 | 192 | 0.484 | 1.52 | 1.53 | 0.553 |
| v4_all | 21+ | 649 | 0.350 | 3.34 | 2.78 | 0.441 |
| v4_short | <=10 | 33 | 0.515 | 0.43 | 1.03 | 0.539 |
| v4_short | 11-20 | 192 | 0.240 | 1.41 | 1.53 | 0.394 |
| v4_short | 21+ | 649 | 0.102 | 4.74 | 2.78 | 0.274 |
| v3_s1 | <=10 | 33 | 0.576 | 0.55 | 1.03 | 0.563 |
| v3_s1 | 11-20 | 192 | 0.562 | 1.39 | 1.53 | 0.639 |
| v3_s1 | 21+ | 649 | 0.430 | 2.56 | 2.78 | 0.509 |
| v3_short | <=10 | 33 | 0.636 | 0.88 | 1.03 | 0.576 |
| v3_short | 11-20 | 192 | 0.380 | 1.39 | 1.53 | 0.450 |
| v3_short | 21+ | 649 | 0.202 | 3.14 | 2.78 | 0.282 |


### Val headline (for the record; selection was on val)

v4_all decoder (a) 0.300 (true order 0.319), v3_s2 0.358, v2b 0.353, v3_s1 0.340; short-to-long on val 21+: v4_short 0.149 vs v4_all 0.325, v3_short 0.111 vs v3_s1 0.363. Full val tables: `runs/mind_v4/tables_val.md`, `tables_v4_val.md`.

### Calibration of ABSTAIN

Abstain rate on items with a reachable oracle answer (should be low): v4_all 0.045 test (0.036 val), v3_s2 0.055, v2b 0.046. On items with no reachable answer v4_all abstains 0.135 (v3_s2 0.218): v4 abstains less when it should. Wrong-story abstain 0.939 (v3_s2 0.950). Control weakness: with memory = the question only, v4_all abstains 0.167 (v3_s2 0.855, but v3_s1 0.002): the wrong-passage supervision does not cover a memory that is just the question, and no reward stage fixes it here.

## Reading attribution: is the long-sentence failure the tree or the policy?

Twenty val items (seeded sample of the 407 val items with a 21+ answer sentence and v4_all ROUGE < 0.3; reference vs the oracle sentence's tree and v4's path, in `runs/mind_v4/handread_val_21plus.json`, my verdicts in `handread_val_21plus_judgments.json`):

| verdict | items | meaning |
|---|---|---|
| policy, tree held the answer (best subtree ROUGE >= 0.74) | 10 | wrong sentence in 6 (items 0, 9, 10, 11, 16, 18), abstained 1 (12), right sentence but wrong branch or too deep in 3 (1, 3, 4) |
| policy and tree | 4 | wrong sentence or missed HERE, AND no clean subtree equals the answer (best 0.50 to 0.67): a clause or a conjunct hangs under the wrong node (2, 7, 13, 14) |
| reference not in the text | 5 | the reference is an inference or new words (5, 6, 8, 15, 19); the oracle "anchor" is a pronoun or article |
| metric | 1 | right word, different inflection (17) |
| tree alone | 0 | |

So in 14 of 14 recoverable cases the policy also chose wrong; in 4 of them the tree would have capped the result below 0.7 even with a perfect choice. Of the 14, 10 are sentence-selection errors and 4 are descent errors in the right sentence (two of those pass the anchor without stopping, one takes the wrong conjunct, one takes the wrong branch). Automatic counts over all reachable test items, v4_all (table "Failure attribution" above): in the 21+ band 18.5% right, 4.6% abstained, 33.1% wrong sentence, 27.4% right sentence with a tree that has a >= 0.8 subtree (policy), 16.3% right sentence with no >= 0.8 subtree. The tree-adequacy figures by band (0.57 / 0.58 / 0.57) show the encoder does not build long sentences worse by this measure; its ceiling is the same short or long.

## Three limiting factors

1. **Sentence selection.** One third of reachable 21+ items (45% of 11-20, 76% of <=10, tiny n) fail by choosing the wrong sentence; the chosen sentence is the oracle sentence in 62% of 21+ items. Walking does not change this; it is the learned sentence value, and for reference the v2b thinker picked the oracle sentence 57% of the time on val (SPAN_DIAGNOSIS); v4's 62% on test 21+ items is the same ballpark.
2. **Local descent accuracy.** Given the right sentence, v4 finds an acceptable anchor in 35% of 21+ items against v3's 43%, and its errors are mostly too narrow (passes the answer node) or the wrong branch. Removing top-down context (the node sees only its subtree, children and the parent's relation) costs several points, and a short-trained policy never sees the stopping decision at depth, so it overshoots (depth 4.7 vs oracle 2.8). Teacher forcing on gold paths only also leaves the off-path states untrained.
3. **What the tree and the reference allow.** Only 54% of items have a subtree matching the reference at ROUGE >= 0.8 (88% at >= 0.5; 57% / 58% / 57% in the three length bands); 22% of items need words that are not in the story (needs-new-words ROUGE 0.14). With a perfect choice the ceiling is the oracle 0.757; the rest of the gap is clauses attached under the wrong node, references that are inferences, and inflection mismatches.

## Files

Code: `src/naomi/mind_v4.py`, `scripts/train_mind_v4.py`, `scripts/eval_mind_v4.py`, `scripts/probe_mind_v4.py`, `scripts/tables_mind_v4.py`, `scripts/tables_mind_v3.py` (added `--dir`), `tests/test_mind_v4.py`, `tests/test_naomi_no_literals.py` (allow-list for the new tensor-name keys). Run data in `runs/mind_v4/`: checkpoints `thinker_v4_all_best.pt` (main), `thinker_v4_short_best.pt`, development checkpoints `thinker_v4{a_freenav,b_hillclimb,c_localmlp}_best.pt`; curves `thinker_v4_all.curve.tsv`, `thinker_v4_short.curve.tsv`; logs `v4_all.log`, `v4_short.log`, `v4{a,b,c}_*.log`, `eval_val.log`, `eval_test.log`; `eval_val.json`, `eval_test.json`, `steps_{val,test}.json`, `probe_{val,test}.{json,md}`, `tables_{val,test}.md`, `tables_v4_{val,test}.md`, hand-read dump and judgments. (`eval_rows_{val,test}.jsonl` are 18 MB each: not for the commit.)

Commands: `python scripts/train_mind_v4.py --variant all --epochs 24 --tag v4_all`; `--variant short --tag v4_short`; `python scripts/eval_mind_v4.py --dir mind_v4 --split test --models v4_all=... v3_s2=... v2b=... v3_s1=... v4_short=... v3_short=... --main v4_all --rt_thr runs/mind_v2/roundtrip_val.json`; `python scripts/tables_mind_v3.py test v4_all v3_s2 v2b v3_s1 --short v4_short v4_all --short v3_short v3_s1 --dir mind_v4`; `python scripts/tables_mind_v4.py test v4_all v4_short`; `python scripts/probe_mind_v4.py test v4_all=... v4_short=... v3_s1=... v3_short=...`. Tests: `python -m pytest tests/test_mind_v3.py tests/test_mind_v4.py tests/test_roundtrip.py tests/test_naomi_no_literals.py tests/test_naomi_memory.py -q` -> 22 passed.
