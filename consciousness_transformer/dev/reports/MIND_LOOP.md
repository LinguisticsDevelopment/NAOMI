# Mind loop: one learned transition T(tree, state, memory), looped, learned halt

Status 2026-11-27. Method: `dev/design/CONSCIOUSNESS_LOOP_PROPOSAL.md` (approved) with the owner correction of this session: memory is PRE-STORED by code, the thinker does not read the story through the loop and does not write memory in this version. Test split evaluated once, after every setting was fixed on val.

## Plain conclusion

- The loop trains and fits (overfit check: 20 stories, val=train items ROUGE 0.656, tree loss 0.22), but on held-out stories it does **not beat the cheat**: test ROUGE-L **0.177** vs cheat 0.201 (decoded; raw sentence text 0.220); loop minus cheat -0.024 (items -0.038 to -0.008; stories -0.037 to -0.009). It is far below v3_s2 (0.346) and v5_all (0.298; MIND_V5.md, same 1007 test items). Tree score of the read-back decoded text (300-item subsample): loop 35.5, cheat 11.4 (v3_s2 38.3, v5_all 33.6 from the v5 report, different subsample draw but same seed rule), i.e. the structure is decent while word choice is poor.
- **The learned halt collapsed to one step**: halting distribution on test is 1007/1007 at step 1 (mean 1.00). T used as a single pass; later loop steps are never used (their tree loss is 1-2 higher than step 1, see curve, because nothing trains them once the halt mass sits on step 1). So the "loop" does no iterative thinking here; reported, not rescued.
- **Memory use is shallow.** Empty memory -> empty tree (1.000), but a WRONG story gives a full-size tree (abstain 0.000, ROUGE vs own reference 0.144 vs 0.177 normal), shuffled questions 0.119, question-only memory 0.111. Most of the score is a generic answer shape built from the question plus the pool of words; the story content adds about 0.03-0.06 ROUGE. Blocking memory reads during the question drops ROUGE by 0.158 (to 0.019) but the model then mostly outputs nothing (abstain 0.69), so this shows the reads are needed to produce any tree, not that the answer words come from memory.
- Short-to-long test is uninformative: the short-only model (489 train items with answer sentence <= 10 words) overfits (train loss 1.3, val loss 7) and is worse than the all-lengths model in every band (test <=10: 0.085 vs 0.158).

## Method as built

`src/naomi/mind_loop.py`, `scripts/{train,eval}_mind_loop.py`, `tests/test_mind_loop.py`.

- **Tree object** (same form in and out): K=24 node slots; per slot a USVS point (unit vector in the sense-table space, 614 dims, from the lexicon), a has-point probability, a 24-d identity code (for nodes with no point: names and lexicon-less words; a per-story random code, i.e. a variable with no meaning), a relation distribution, a pos distribution, a soft parent pointer over the K slots + ROOT, an existence probability. Encoder trees are converted to this form by code (no word identity anywhere; pos/rel are the grammar's own labels).
- **Memory (pre-stored)**: every node of every sentence tree of the story, one slot each in the same fields plus parent link (global slot index), sentence index and position in the sentence; cut at 640 slots (1.5% of stories lose their last nodes). Built by `story_memory` / `memory_tensors`; T has no write path (test asserts: memory tensors identical before/after the loop, no write/erase parameters). T reads it by cross-attention in every layer, after ONE learned self-attention layer over the stored slots (computed once per story from the stored tensors; a learned part of T's reading, not a write), plus a learned null slot so an empty memory can be attended.
- **T**: small transformer (H=128, 4 heads, 2 layers, 0.79 M parameters) over [state token; K tree slots], with cross-attention to memory. Output heads are per slot (vec, has, identity code, rel, pos, parent, existence) and the halting logit is a linear read of the new state token. One set of weights; no action head (test lists absent names: pick/keep/drop/replace/stop/abstain/anchor/pointer/copy/sentence). The question tree enters at step 0; each later step's tree_in is the previous tree_out (soft). State starts from a learned vector (the same for every question).
- **Loop and halt (PonderNet)**: up to 6 steps (safety cap). lambda_n = sigmoid(halt logit), p_n = lambda_n prod(1-lambda_k); loss = sum_n p_n L_n + 0.05 KL(p || geometric(0.3)). Evaluation halting is deterministic: first step where the cumulative halting probability reaches tau; tau in {0.3, 0.5, 0.7} chosen on val together with the existence threshold (chosen: tau=0.3, existence threshold 0.4; val grid in `val_choice.json`; all tau values give within 0.001 because everything halts at step 1).
- **Training signal**: only the output tree at each step, weighted by its halting probability, against the reference answer read into a tree by the encoder (a random gold answer among those whose tree reads). Loss L_n = differentiable set matching: Hungarian assignment of the K output slots to reference nodes on 2(1 - USVS cosine) [identity-code cosine for reference nodes with no point] + 0.5 relation CE + 0.5 pos CE + 0.5 has-point BCE + existence; matched slots pay those and existence-to-1, unmatched slots existence-to-0, plus 0.5 x parent-pointer CE of matched slots (target parent mapped through the matching, ROOT stays ROOT); divided by max(#reference nodes, 3). Reference trees longer than 24 nodes are cut (pre-order prefix). Nothing supervises memory, intermediate steps or halting time.
- **Wrong-story / empty-memory items in training** (12% / 6% extra items, target = EMPTY tree): still output-tree-vs-reference only; declared as an addition to the brief to teach "no answer".
- **Decoding (setting (b))**: output slots with existence > 0.4 (val) become nodes; each node's word is the pool entry (words of the story + question) with the nearest USVS point (no-point nodes: nearest identity code), ties by tag agreement then frequency; relation = argmax of the node's relation distribution; structure = argmax parent pointers (cycles broken at ROOT, several roots joined under the most-confident); the frozen orderer (`large_best.pt`) puts the words in order. ROUGE-L is max over gold answers; tree score = the existing scorer on the decoded text read back vs the reference trees (300-item seeded subsample).
- Training: AdamW lr 1e-3, grad-clip 1, batches of 12 stories (all their questions), 15 epochs / 40 min (about 2.7 min/epoch) on CPU; checkpoint = best val ROUGE (150 val items, thr 0.5) - 0.001 x val loss; epoch 8 chosen. Val ROUGE during training is noisy (0.09-0.18) because the existence threshold at 0.5 sits where the output flips between empty and non-empty; the 0.4 threshold chosen on the full val set helps a little.

## Deviations and caveats (declared)

1. Memory pre-stored, no reading through the loop (owner correction); so no BPTT through reading and no state-carry issue from reading. The state at the start of asking is a fixed learned vector.
2. A learned self-attention layer contextualises the stored memory slots (needed so slots know their sentence mates); it only reads the stored tensors.
3. Wrong-story / empty items in training (above). Cap of 640 stored slots; cap of 24 slots per tree.
4. Single seed, 15 epochs; no hyperparameter search (heavy validation deferred by owner directive). The halt prior (lambda 0.3, beta 0.05) was not tuned; a stronger prior might force more steps, not tried.
5. Short-only model: checkpoint chosen on a random val subsample of ALL lengths (not short-only), so its selection is slightly generous; it still loses everywhere.
6. Test numbers for v3_s2 and v5_all are copied from `MIND_V5.md` (same 1007 items), not recomputed; cheat is recomputed here (0.201 decoded (a)/orderer-on-sentence; v5 reported 0.200).
7. Words are chosen in the decoder from meaning only; names are carried by the random identity code (per story), which the model must copy from memory slots.

## Overfit check

20 train stories (38 items), no wrong/empty items, lr 2e-3, batches of 4 stories, 150 epochs (4 min): train tree loss 8.5 -> 0.22; ROUGE-L of the decoded output on those items 0.656 (log `overfit20.log`, `overfit20.curve.tsv`). The model can store-and-retrieve at least on its training items, so the generalisation failure is not a bug in the loop.

## Curve (all-lengths model; val on 150 random val items)

```
epoch	minutes	train_loss	val_loss	val_steps	val_rouge	per_step_loss
0	2.88	4.2852	4.1550	1.000	0.0380	[4.0945, 4.4283, 4.5582, 4.5819, 4.5807, 4.6077]
1	5.48	3.4606	3.9266	1.000	0.0657	[3.8663, 4.215, 4.3526, 4.4737, 4.5764, 4.6633]
2	8.19	3.3354	3.6700	1.000	0.1699	[3.6097, 4.0569, 4.2212, 4.3238, 4.4651, 4.6267]
3	10.90	3.2732	3.6621	1.000	0.1261	[3.6013, 4.1339, 4.2788, 4.3156, 4.362, 4.4615]
4	13.47	3.2265	3.6095	1.000	0.1257	[3.5493, 4.2558, 4.3395, 4.3817, 4.4387, 4.507]
5	16.15	3.2017	3.5806	1.000	0.1627	[3.5203, 4.3803, 4.4339, 4.495, 4.6901, 4.743]
6	18.96	3.1945	3.5916	1.000	0.1779	[3.5314, 4.1993, 4.3432, 4.4796, 4.5736, 4.6841]
7	21.76	3.1594	3.6217	1.000	0.1029	[3.5615, 4.1719, 4.3915, 4.5137, 4.6148, 4.7042]
8	24.54	3.1483	3.5144	1.000	0.1790	[3.4492, 4.0591, 4.4005, 4.5946, 4.7761, 4.8819]
9	27.32	3.1288	3.6114	1.000	0.0932	[3.5512, 4.2865, 4.8502, 5.12, 5.2154, 5.2152]
10	29.87	3.1100	3.6343	1.000	0.1247	[3.5741, 4.1707, 4.4179, 4.7195, 5.018, 5.3576]
11	32.36	3.0911	3.6322	1.000	0.1317	[3.572, 4.5802, 5.2552, 5.6387, 5.7712, 5.8349]
12	35.09	3.0885	3.5902	1.007	0.1654	[3.5149, 4.6631, 5.3516, 5.6385, 5.7985, 5.9914]
13	37.78	3.0585	3.5956	1.000	0.1404	[3.5354, 4.2852, 4.6931, 4.8466, 4.9499, 5.1103]
14	40.05	3.0292	3.6084	1.000	0.1750	[3.5482, 4.7583, 5.7124, 6.2565, 6.6123, 6.867]
```
(columns: epoch, minutes, train loss, val loss, mean halting step, val ROUGE (thr 0.5), per-step val tree loss; the per-step losses show step 1 best and later steps worse.)

## Val (settings chosen here) and test (once)

## val: n=1025 items, thr=0.4, tau=0.3

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.195 | 32.2 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.207 (raw text 0.228) | 13.0 |

answer rate 1.000, mean output nodes 4.51, mean steps 1.00, steps distribution {'1': 1025}

loop minus cheat: {"mean": -0.012626423998783375, "items": [-0.027989361504412454, 0.002304908663306114], "stories": [-0.029221031177525957, 0.0030081934559021543]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 38 | 0.264 | 0.158 | 0.138 | 1.00 | 4.63 |
| 11-20 | 209 | 0.217 | 0.206 | 0.151 | 1.00 | 4.34 |
| 21+ | 708 | 0.201 | 0.229 | 0.136 | 1.00 | 4.89 |
| none | 70 | 0.027 | 0.013 | 0.021 | 1.00 | 1.09 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1025 | 0.000 | 4.51 | 0.154 | 1.00 |
| qonly | 1025 | 0.009 | 2.82 | 0.127 | 1.00 |
| shuffled | 1025 | 0.000 | 4.55 | 0.135 | 1.00 |
| block | 1025 | 0.716 | 0.47 | 0.021 | 1.84 |

block-reads ROUGE delta vs normal: -0.174

tree-score reasons (loop): {'node_dropped': 929, 'meaning_shift': 530, 'node_extra': 324, 'operator_folded_differs': 299, 'subtree_moved': 84, 'relation_changed': 74, 'head_changed': 69, 'name_mismatch': 32}

## test: n=1007 items, thr=0.4, tau=0.3

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.177 | 35.5 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.201 (raw text 0.220) | 11.4 |
| v3_s2 (MIND_V5.md, same items) | 0.346 | 38.3 |
| v5_all (MIND_V5.md) | 0.298 | 33.6 |

answer rate 1.000, mean output nodes 4.29, mean steps 1.00, steps distribution {'1': 1007}

loop minus cheat: {"mean": -0.023945175413818718, "items": [-0.03807145134115974, -0.008068132357251333], "stories": [-0.036652888743827605, -0.008923381301715983]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 35 | 0.158 | 0.098 | 0.085 | 1.00 | 4.43 |
| 11-20 | 202 | 0.174 | 0.196 | 0.131 | 1.00 | 4.06 |
| 21+ | 688 | 0.194 | 0.227 | 0.130 | 1.00 | 4.70 |
| none | 82 | 0.044 | 0.041 | 0.074 | 1.00 | 1.32 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1007 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1007 | 0.000 | 4.29 | 0.144 | 1.00 |
| qonly | 1007 | 0.007 | 2.71 | 0.111 | 1.00 |
| shuffled | 1007 | 0.000 | 4.34 | 0.119 | 1.00 |
| block | 1007 | 0.694 | 0.48 | 0.019 | 1.82 |

block-reads ROUGE delta vs normal: -0.158

tree-score reasons (loop): {'node_dropped': 752, 'meaning_shift': 484, 'node_extra': 293, 'operator_folded_differs': 269, 'subtree_moved': 70, 'relation_changed': 57, 'head_changed': 50, 'name_mismatch': 37, 'opposite_meaning': 1}

Halting-step distribution: test {1: 1007}; val {1: 1025}; controls: 1.00 for all but block-reads (1.82; with reads blocked the halt moves to later steps, the only place any step beyond 1 is used).

## 20 hand-read val answers (seeded sample 7, answered items)

Reading: exactly right in 2 (items 0 and 15), partly right in 1 (14), a few with the right kind of word (items 4, 17, 8 get one meaningful word), the rest is a bag of frequent story words (daughter, princess, wife, was) in a plausible tree shape: the output reproduces the question's subject words rather than answering. Wrong-feeling (opposite value): item 3 ("sad" vs "happy"), as in v5.

| # | question | reference | decoded output | nodes | ROUGE-L |
|---|---|---|---|---|---|
| 0 | Where did the cat take the buck after she captured him? | the castle | the castle | 2 | 1.00 |
| 1 | What did the bird do after the woman cut off its tongue? | It flew away. | drove to clothes it the woman | 6 | 0.25 |
| 2 | What happened after the prince realized how long his nose really was? | Prince Hyacinth's nose was now just like anyone's else. | he the princess was into princess flew | 7 | 0.27 |
| 3 | How did Tephany feel after she spent the evening with Denis? | happy | sad | 1 | 0.00 |
| 4 | Where will the dead man go after he enters the cave of terror? | through the burning forest | the top maid | 3 | 0.29 |
| 5 | What will happen to the church after the giant leaves? | It will never be finished. | wife the wife he were | 5 | 0.37 |
| 6 | How was the six headed troll different from the first one? | The troll with six heads is twice as strong as the firs | the daughter the was headed to | 6 | 0.21 |
| 7 | Why did the eldest daughter not give the old woman food? | It's all too little for herself. | go he the daughter was | 5 | 0.00 |
| 8 | What did the youngest hear when she passed through the dark cavern of  | the most horrible yells | the out fairy | 3 | 0.29 |
| 9 | What did Tschang O do when she reached the moon? | ran into the castle there, and has lived there ever sin | go wife into | 3 | 0.10 |
| 10 | What did the bull have in his left ear? | a cloth | the white servants | 3 | 0.00 |
| 11 | What happened because Charcoal Nils forgot to the sign? | He saw something that nearly robbed him of his wits. | returning he the woman was to woman | 7 | 0.12 |
| 12 | How did the Yellow-Peerie Boy complete the youngest Princess's task? | made the Peerie Folk to his work | princess the were the princess | 5 | 0.31 |
| 13 | Who held an official position in Yangdschou? | Sir We | the old daughter | 3 | 0.00 |
| 14 | Who had been taken prisoner by the Fairies and did not want a like fat | a poor mortal | prisoner | 1 | 0.00 |
| 15 | How will Sir We feel when the match-maker recommends Old Dschang to ma | angry | angry | 1 | 1.00 |
| 16 | Why did the youngest Princess tell the Giant she would come home late? | She wanted to hide herself in the creel. | go the princess he was to princess | 7 | 0.17 |
| 17 | Where did the eldest daughter get lodging? | at a farmer's | the room farmer | 3 | 0.29 |
| 18 | What did the dog and the cat do when their owners had grown poorer? | took counsel together as to how they might restore to t | go owners the cat | 4 | 0.10 |
| 19 | Why was the prince unable to help the king's daughter? | She stepped right into the middle of the tar and leaped | the prince the was | 4 | 0.25 |

## Honest conclusion

A single learned transition over trees with a pre-stored memory is trainable and the pipeline (tree form, set-matching loss, PonderNet halt, meaning-only decoder) works end to end, but within about 40 minutes of CPU training it learns the typical shape and vocabulary of answers and only weakly uses the story: it ties the cheat on short answers (<= 10 words: 0.158 vs 0.098, n=35) and loses to it on long ones, and the learned halt chose one step. The old methods that copy memory nodes (v3, v5) are much better on ROUGE because the loop has to regress a 614-d meaning point and be decoded by nearest word, which blurs the words; the training signal through a Hungarian-matched regression is weak for exactly the word-selection part. Not tried (outside the brief): action/copy heads, longer training, larger model, tuned halt prior, a second seed.

## Files

Code: `src/naomi/mind_loop.py`, `scripts/train_mind_loop.py`, `scripts/eval_mind_loop.py`, `tests/test_mind_loop.py`, allow-list edit in `tests/test_naomi_no_literals.py`. Runs in `runs/mind_loop/`: checkpoints `loop_all_best.pt` (main), `loop_short_best.pt`; curves `loop_all.curve.tsv`, `loop_short.curve.tsv`, `overfit20.curve.tsv`; logs `loop_all.log`, `loop_short.log`, `overfit20.log`, `eval_val.log`, `eval_test.log`; `eval_{val,test}.json`, `tables_{val,test}.md`, `handread_{val,test}.json`, `val_choice.json`, `dims.json`. Not for commit: `items_*.pkl`, `stories_f.pkl`, `sense_matrix.npy` (rebuild with `prep`), `eval_rows_*.jsonl`, `*_last.pt`, `smoke*`, `overfit20_*.pt`, `read_cache/`.

Commands: `python scripts/train_mind_loop.py prep`; `train --tag loop_all --epochs 40 --S 12 --lr 1e-3 --max_minutes 40`; `train --tag loop_short --short --epochs 60 --max_minutes 9`; `train --tag overfit20 --overfit 20 --epochs 150 --S 4 --lr 2e-3`; `eval_mind_loop.py --split val|test --ckpt runs/mind_loop/loop_all_best.pt --short runs/mind_loop/loop_short_best.pt`. Tests: `python -m pytest tests/test_mind_loop.py tests/test_naomi_no_literals.py tests/test_roundtrip.py -q` -> 8 passed.
