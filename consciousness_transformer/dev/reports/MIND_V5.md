# Mind v5: the thinker completes the question

Status 2026-11-27. Method: `dev/design/CONSCIOUSNESS_V4_PROPOSAL.md` option A with the owner amendment (the replace-slot is learned, never labelled) and two later owner changes: (1) the score is ONLY the decoded output text vs the reference text (ROUGE-L, and the tree score of the read-back text); the separate answer-part / frame-part measures were dropped; (2) the thinker also learns keep/drop per question node. Test split evaluated once, after checkpoints and settings were fixed on val.

## Plain conclusion

- **v5 beats the old cheat clearly and does not beat the old method.** Test, full decoded output sentence, order-only decoder (a): v5_all ROUGE-L **0.298**, cheat 0.200, v3_s2 (old method used as the filler) 0.346, oracle (aligned reference nodes) 0.687. v5_all - cheat +0.097 (items +0.079 to +0.117; stories +0.078 to +0.120); v5_all - v3_s2 -0.048 (items -0.065 to -0.031; stories -0.064 to -0.031). Tree score of the read-back decoded text (300-item subsample): v5_all 33.6, cheat 11.2, v3_s2 38.3, oracle 62.7; v5_all - v3_s2 -5.2 (items -8.3 to -2.1). The proposal's abandon criterion "filler does not beat the old cheat" is not met; "does not beat v3_s2" is the honest negative, as with v4.
- **The keep/drop head learned to drop the whole question frame**: average share of question nodes kept in the output is 0.000 in every band for v5_all. Reference answers are mostly short phrases, so any kept frame word costs precision under ROUGE-L; the reward-only head found that. Consequently the output of v5_all IS the filler, and the replace-choice has no effect on the output at all (the replaced node's subtree is dropped and the rest is dropped too). The UNKNOWN diagnostic for v5_all is therefore not evidence of anything (below). A frame-keeping variant (v5_all_s1, keep head untrained at its init, keeps 0.59-0.69) scores 0.245: the frame hurts under this score, as the owner anticipated.
- **Short-to-long does not transfer** (table below): v5_short gets 0.118 full-sentence ROUGE on 21+ vs 0.313 for v5_all (38%). v4 got 45%, v3 30%. Stage-1 imitation from <= 10-word answers (489 train items) is too little for a pointer over all memory nodes.
- References are often short phrases, so a full sentence pays a precision cost under this score. Reported, not compensated.

## What was built

Code: `src/naomi/mind_v5.py`, `scripts/{train,eval}_mind_v5.py`, `tests/test_mind_v5.py`. (`mind_v3.Thinker3.encode` now also returns `hq` and `hsent`: backwards-compatible extra keys.) Data, memory, stories, reads, decoder, ROUGE, cheat, old v3 thinker: v2b/v3's, reused through symlinks in `runs/mind_v5/`.

**Architecture.** The v3 encoder stack (tree rounds, question cross-attention, sentence context) gives every memory node a state; question nodes get their own states (tree rounds, cross-attention to memory). One key matrix over [all memory nodes of the story | question nodes]. Filler = a GRU-driven pointer: each step picks ONE node (a copy) from the whole story (across sentences) or from the question, or STOP (ABSTAIN at step 0); a second pointer picks the earlier filler node it attaches under (feature: the memory parent edge); the copied node keeps its own relation. Max 16 nodes. Provenance (source node) is kept per output node. Replace-choice: a head over question nodes (input: node state, filler summary, question summary) plus a keep/drop head per question node (same input; initialised to keep). No word identity, no slot label. The output tree = question tree with the chosen node's subtree replaced by the filler (taking that node's relation); dropped nodes vanish, their children move up. Decoded by the frozen orderer: (a) order-only, (b) meaning-only.

**Training.** Stage 1 (about 14 min, 12 epochs, warm start from the v3_s1 encoder; short model from v3_short): imitate the filler = reference nodes aligned to memory by lemma (v2-v4 alignment, best accepted anchor, pre-order, cut at 16; any frontier node is accepted at each step), STOP / ABSTAIN, parent pointer; wrong-story items and 10% empty-memory items train ABSTAIN. 1012 of 8548 train items have no acceptable alignment: skipped, counted, not tuned on. Checkpoint chosen on val by lemma-F1 of the filler + wrong-story abstain (0.9/0.1). Stage 2 (REINFORCE, the same score as evaluation): only the replace head and keep head are trained; per item 4 samples of (replaced node, keep mask), reward = ROUGE-L (max over references) of the decoded (a) full output sentence, leave-one-out baseline; filler fixed greedy; 30 epochs x 3000 items; val 300 items. val ROUGE went 0.2575 -> 0.315 within 4 epochs (the head learned to drop the frame) and stayed flat. Short model: stage 2 only over <= 10-word items; no gain (0.142 -> best checkpoint = start).

## Deviations and choices, declared

1. **Stage-2 reward trains only the replace/keep heads, not the filler** (the filler stays at its imitation policy). Time budget. A filler-level reward fine-tune was not run.
2. **Warm start of the encoder from v3** (v3_s1 for v5_all, v3_short for v5_short). The pointer itself (over all nodes) is new. This shares the v3 sentence-level training with the baseline; the output has no sentence-selection step.
3. Filler imitation only copies memory nodes; question-node copies are in the pointer space but never imitated and never used in practice (0 of 7146 output nodes came from the question).
4. **Earlier discarded design, not reported as results:** I first trained the replace-choice with an expected "frame reward" (share of question nodes supported by the answer sentence). The owner dropped that measure; its code, checkpoints and 3 epochs of logs were deleted; nothing reported here uses it. (It would have defined the unknown through the alignment, a de facto label.)
5. Tree score of the read-back decoded sentence is computed on a seeded 300-item subsample per system (reading is slow); ROUGE on all items. Round trip (v2b formulation, threshold 99 declared on val controls) on the same 300 items, filler level.
6. Baselines (cheat, v3_s2, oracle) are plugged into the main model's replace-choice and keep mask, and decoded the same way, so all rows use one score. With the all-dropped keep mask their output is the filler.
7. Controls "shuffled questions" and "memory = question only" are rebuilt in eval (the old shuffled pairs carried no question nodes); wrong-story abstain is the old pairs'.
8. Val/test were evaluated with a bug in `assemble` when the question root is dropped (the filler kept the replaced node's relation instead of ROOT). Fixed afterwards, tests added; val rerun with the fix: 0.303 -> 0.303, tree score 34.6 -> 35.2. The test run was NOT repeated (single-run rule); its numbers carry the bug, whose effect on val was nil to +0.6.
9. Combination formula: none; one score (owner change).

## Test results (n=1007 items, 23 stories)

Bands: <=10: 35, 11-20: 202, 21+: 688, none: 82. Abstain counts 0.

| system | answer rate | full-sentence ROUGE-L (a) | (b) | tree score % (n=~280) | filler ROUGE (a) | question nodes kept |
|---|---|---|---|---|---|---|
| cheat: best-overlap sentence | 1.000 | 0.200 | 0.198 | 11.2 | 0.201 | 0 |
| v3_s2 filler (old method) | 0.924 | 0.346 | 0.337 | 38.3 | 0.345 | 0 |
| **v5_all** | 0.927 | **0.298** | 0.291 | **33.6** | 0.296 | **0.000** |
| v5_all_s1 (stage 1 only, frame kept) | 0.927 | 0.245 | 0.241 | 22.5 | 0.296 | 0.588 |
| v5_short | 0.925 | 0.118 | 0.118 | 12.0 | 0.111 | 0.787 |
| oracle (aligned reference nodes) | 0.919 | 0.687 | 0.665 | 62.7 | 0.689 | 0 |

(Old raw-text cheat ROUGE 0.220 as in v2-v4; for comparison the previous thinkers' filler ROUGE (a) on test: v3_s2 0.345, v4 0.297. v5_all filler 0.296 = v4, below v3_s2.)

By length band, full-sentence ROUGE-L (a) / tree score %:

| band | n | cheat | v3_s2 | v5_all | v5_short | oracle |
|---|---|---|---|---|---|---|
| <=10 | 35 | 0.099 / 11.1 | 0.230 / 36.4 | 0.163 / 28.5 | 0.191 / 14.1 | 0.752 / 71.1 |
| 11-20 | 202 | 0.195 / 9.6 | 0.408 / 44.6 | 0.369 / 41.0 | 0.145 / 12.4 | 0.759 / 64.8 |
| 21+ | 688 | 0.226 / 12.6 | 0.362 / 34.9 | 0.313 / 28.9 | 0.118 / 11.9 | 0.745 / 61.5 |
| none | 82 | 0.041 / 3.7 | 0.104 / 59.7 | 0.055 / 60.0 | 0.024 / 11.0 | 0.000 / - |

Average share of question nodes kept, by band: v5_all 0.000 / 0.000 / 0.000 / 0.000 (<=10, 11-20, 21+, none); v5_short 0.768 / 0.788 / 0.781 / 0.852 (its keep head did not move from its init); v5_all_s1 0.686 / 0.595 / 0.612 / 0.297.

### Short-to-long (full-sentence ROUGE-L (a); tree score %)

| band | n | v5_all | v5_short (<= 10 only) |
|---|---|---|---|
| <=10 | 35 | 0.163 / 28.5 | 0.191 / 14.1 |
| 11-20 | 202 | 0.369 / 41.0 | 0.145 / 12.4 |
| 21+ | 688 | 0.313 / 28.9 | 0.118 / 11.9 |
| none | 82 | 0.055 / 60.0 | 0.024 / 11.0 |

On <= 10 (n=35) v5_short is higher in ROUGE but lower in tree score; n is tiny. On 11-20 and 21+ it keeps 39% and 38% of the all-lengths model.

### Paired differences (full-sentence ROUGE-L (a), v5_all minus ...; mean, 95% over items, over stories)

cheat +0.097 (+0.079 to +0.117; +0.078 to +0.120); v3_s2 -0.048 (-0.065 to -0.031; -0.064 to -0.031); v5_short +0.179; v5_all_s1 +0.053 (+0.043 to +0.063); oracle -0.390. Tree score: cheat +22.3 (+18.8 to +25.9), v3_s2 -5.2 (-8.3 to -2.1).

### Controls (test)

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / filler ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v5_all | 1.000 | 0.958 | 0.413 / 0.043 | 0.006 |
| v5_short | 1.000 | 0.874 | 0.500 / 0.042 | 0.000 |

Empty and wrong-story controls meet the >= 0.90 target for v5_all (short model: wrong-story 0.874, below). Shuffled questions land at the floor (ROUGE 0.043). **Memory = question only fails (abstain 0.006, target: abstain)**: with the question as memory the pointer copies from it; the training had empty-memory items but no question-only-memory items (same weakness as v4).

### Provenance, round trip

Provenance: 7146 of 7146 v5_all output nodes (and 3539 of 3539 for v5_short) are copies of memory nodes, token and lemma equal to the source; 100% traced by construction, verified. Round trip (filler read back vs same words in memory order, threshold 99, 300 items): pass 0.627 (1-3 words 0.973, 4-6 words 0.824, 7+ words 0.376), same pattern as v4 (0.598). The decoder (b) meaning-only setting scores within 0.007 of (a).

Tree-score reasons (v5_all, decoded sentence vs reference, 280 items): node_extra 943, node_dropped 563, meaning_shift 378, operator_folded_differs 255, head_changed 179, subtree_moved 132, name_mismatch 75, relation_changed 45. Extra nodes dominate: fillers are longer than the short reference phrases.

## UNKNOWN-discovery diagnostic (test; diagnostic only, never a training signal)

Share of answered items where the replaced node is the question's highest-UNKNOWN node (ties counted as hits): v5_all 0.239 (chance if uniform 0.118), v5_all_s1 (head untrained by reward) 0.199, v5_short 0.078 (below chance). Val: 0.277 / 0.191 / 0.082. **Read plainly: the thinker did NOT discover the unknown node.** v5_all is above chance, but its replaced node never reaches the output (all frame dropped, replaced subtree removed), so the reward cannot have taught it: any agreement is the head's initial lean and whatever the stage-2 gradient noise left. v5_all_s1 at 0.199 (untrained head) vs 0.239 is within that noise. The question the amendment asks, whether reward alone makes the replace choice discover the UNKNOWN node, is answered here as: not under this score, because the score makes the choice irrelevant once the frame is optional. The root node was replaced in 19% of v5_all items. To test the amendment properly the frame must carry weight (a score that rewards keeping the frame, e.g. references that are full sentences, or a QA2D-like target); not run, as it would change the owner's score.

## 20 hand-read val answers

Seeded sample (seed 7) of answered val items, v5_all with the order-only decoder. v5_all's output is the filler alone, so the "decoded sentence" is the filler. Reading: right or nearly right in 6 (items 2, 4, 7, 8, 12, 19), partly right in 4 (1, 6, 10, 17), wrong in 10 (0, 3, 5, 9, 11, 13, 14, 15, 16, 18). Items 3 and 9 are the typical failure: the right kind of word with the opposite value ("very sad" vs "happy"; "difficult" vs "pleased"), because the memory nodes copied are the emotion words of the sentence found, not the ones answering. Several fillers copy a whole clause chain with function words and stray conjunctions at the start ("and with their spades of gold..."), the pointer copying frontier nodes of a clause above the answer span.

| # | question | reference | decoded output | replaced node | ROUGE-L |
|---|---|---|---|---|---|
| 0 | What did Assipattle's father give to his sister? | rivlins | agree and sent a messenger to the farmer 's house to ask if come to be | did | 0.00 |
| 1 | What will the young man do after Tephany arrived at the castle? | will prepare to hold a wedding | and carried into hall while was sent for to perform marriage ceremony | after | 0.25 |
| 2 | What did the sisters go home with? | spades of gold and shovels of silver | and with their spades of gold and shovels of silver | did | 0.82 |
| 3 | How did the sparrow feel? | pleased | difficult | how | 0.00 |
| 4 | How did the old woman feel when she saw the demons? | frightened | terrified | how | 0.00 |
| 5 | What did the giant want as a reward? | The first boy he had when he was seven years old | reward dost thou ask | did | 0.00 |
| 6 | What did Tschang O do when she reached the moon? | ran into the castle there, and has lived there ever since as | there ran into the castle | what | 0.36 |
| 7 | Why did Charcoal Nils not feel worried about his wife and children? | His wife looked after the children and they gave him no trou | gave no trouble | why | 0.40 |
| 8 | Why couldn't the people on earth endure the ten suns? | The ten suns shone so brightly and burned so fiercely. | so so brightly burned | n't | 0.43 |
| 9 | How did Tephany feel after she spent the evening with Denis? | happy | very sad | how | 0.00 |
| 10 | How will Monedowa dress himself? | as a warrior | dressed warrior embellishing visage with tints to show that prepared for war or peace of | how | 0.48 |
| 11 | To whom did Kari want to take the water to wash in? | the prince | company at the king 's castle | whom | 0.25 |
| 12 | What did the Squire's daughter call the boy? | Little Boy Blue | little boy | what | 0.80 |
| 13 | Why did the old woman not even stop to thank the Lady Sparrow? | She was rude. | and seized hoisted on her back | thank | 0.00 |
| 14 | What did the little page-boy want to do at the Fairy Knowe? | see them, and to visit their abode, just to find out what it | run risk knowe displeasure upon them and bringing down the little folks near the fairy of | did | 0.13 |
| 15 | What will happen after the youngest Princess gives the wool to the Yellow-Haired Peerie Boy? | He will help her complete the task. | ran to the door to see where had made up her mind and went for she | yellow-haired | 0.17 |
| 16 | What did the great blue bull let the king's daughter do? | scratch him | kept and came to the king | what | 0.00 |
| 17 | Why did the wool seem bewitched? | It curled and twisted and coiled itself round her fingers so | and and try so round fingers | the | 0.50 |
| 18 | Why were the king's son and the youngest daughter married again? | The preparations had been made, it seemed a pity to waste th | understood to remind him of what and his memory came back and had forgotten and | the | 0.07 |
| 19 | Why did Tephany believe the little old woman was a fairy? | The little old woman was wise in telling what could happen i | was to her old woman given happen in the days to come | the | 0.57 |

## Limits and next steps

1. The score makes the frame optional, the head drops it, and the replace choice becomes moot; test the amendment under a score where the frame matters.
2. The filler is imitation only; a reward fine-tune of the pointer on the same ROUGE / tree score is the obvious next step (not run).
3. Short-to-long: the pointer over all memory does not generalise from 489 short items; the filler length / stop rule and the frontier choice are what fail (the short model's filler ROUGE on 21+ is 0.091 vs 0.311).
4. Memory = question only does not abstain.

## Files

Code: `src/naomi/mind_v5.py`, `scripts/train_mind_v5.py`, `scripts/eval_mind_v5.py`, `tests/test_mind_v5.py`; edits: `src/naomi/mind_v3.py` (two extra keys in `encode`), `tests/test_naomi_no_literals.py` (allow-list). Run data in `runs/mind_v5/`: checkpoints `thinker_v5_all_s2_best.pt` (main), `thinker_v5_all_s1.pt`, `thinker_v5_short_s2_best.pt`, `thinker_v5_short_s1.pt`; curves `thinker_v5_{all,short}{,_s2}.curve.tsv`; logs `v5_*.log`, `eval_{val,test,val_rootfix}.log`; `eval_{val,test,val_rootfix}.json`, `tables_*.md`, `handread_*.json`, `prep_stats.json`. Not for the commit: `data_*.pkl` (13-77 MB), `eval_rows_*.jsonl`, `*_last.pt`, `*_best.pt` duplicates of s1 (`thinker_v5_all_best.pt`, `thinker_v5_short_best.pt`), the symlinks.

Commands: `python scripts/train_mind_v5.py prep`; `train --variant all --tag v5_all --epochs 12 --init runs/mind_v3/thinker_v3_s1_best.pt --resume --max_minutes 8` (two calls); `train --variant short --tag v5_short --epochs 30 --init runs/mind_v3/thinker_v3_short_best.pt`; `stage2 --init runs/mind_v5/thinker_v5_all_s1.pt --tag v5_all_s2 --epochs 30 --items_per_epoch 3000 --lr 3e-3`; `stage2 --variant short --init ..._short_s1.pt --tag v5_short_s2 --epochs 30 --items_per_epoch 500`; `eval_mind_v5.py --split test --models v5_all=...s2_best.pt v5_short=...s2_best.pt v5_all_s1=...s1.pt --main v5_all --short v5_short --v3 v3_s2=runs/mind_v3/thinker_v3_s2_best.pt --rt_thr runs/mind_v2/roundtrip_val.json --rt_n 300`. Tests: `python -m pytest tests/test_mind_v5.py tests/test_mind_v4.py tests/test_naomi_no_literals.py tests/test_roundtrip.py -q` -> 16 passed.
