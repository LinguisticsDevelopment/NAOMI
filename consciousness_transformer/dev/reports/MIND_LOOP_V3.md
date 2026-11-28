# Mind loop v3: expected-reward grade (owner amendment 2026-11-28)

Status 2026-11-28. Builds on `MIND_LOOP_V2.md`. Same model (`mind_loop.Loop`, 0.79 M params; v1/v2 checkpoints load), same tree form, loop, halt schedule and counterfactual items. Only the training grade changed. Test split evaluated ONCE (config v3b, checkpoint chosen on val).

## Plain conclusion

- **The model went silent. It does not read memory well enough to be allowed to answer.** Under the fixed reward (+1 correct, +0.25 no node, -1 wrong) the reward-optimal policy is "answer only when P(correct | answers) > 0.625". On held-out val the model's top-1 choice among the pool is right for **12.6%** of matched, reachable slots (mean probability on the correct word 0.09); only **4.7%** of slots have P(correct) > 0.625; and the model's confidence barely separates right from wrong (accuracy by confidence quintile, top to bottom: 0.16, 0.22, 0.18, 0.05, 0.03). So silence earns more reward than answering, and both runs converged to the empty tree.
- **Headline test (once): ROUGE-L 0.000, coverage 0.000** at the val-chosen setting (thr 0.05, tau 0.3). Cheat 0.201, loop v2 0.180, v1 0.177, v3_s2 0.346, v5_all 0.298 (copied). This is a worse headline than v2, by construction: silence scores 0. It is not a bug: the empty-memory/wrong-story/question-only controls all give the empty tree and the pool/reward values are covered by tests.
- **The averaging is gone, the memory-reading is not there.** v2's blurred answers (ROUGE 0.18, 100% coverage, text the same with memory blocked) are replaced by no answers. Where the model is forced to answer anyway (existence thresholds 1e-4, see risk-coverage), ROUGE-L is 0.104 on test with memory-independent output (controls within 0.01 of normal on val). No evidence that the thinker uses the story to pick words.
- The reward grade did what it was designed to do (punish averaging); it exposes that the chooser is weak, it does not fix it.

## What was built

Files: `src/naomi/mind_loop_v3.py`, `scripts/train_mind_loop_v3.py`, `scripts/eval_mind_loop_v3.py`, `scripts/diag_mind_loop_v3.py`, `tests/test_mind_loop_v3.py` (5 tests), allow-list edit in `tests/test_naomi_no_literals.py`, one-line change in `scripts/eval_mind_loop.py` (`THR_GRID` module constant, default unchanged). v1/v2 model and checkpoints untouched.

**The grade (declared before training; code in `mind_loop_v3.py`).**
- Candidate pool per item = every distinct identity (USVS sense row, or identity key for names / lexicon-less words) among the story memory slots and the question slots, capped by the memory's 640 slots. A candidate with a USVS point is represented by that point, one without by its identity code (the two representations the decoder uses). Mean pool ~150 candidates.
- Slot choice: s = softmax_c(cos(slot meaning, candidate) / TAU), **TAU = 0.1 fixed, not learned**. Slot meaning is `vec` against point candidates and normalised `idc` against code candidates.
- "No node": P(no node) = 1 - ex exactly (no-node logit = LSE of candidate logits - existence pre-activation); P(candidate c) = ex * s_c. So the decoder's existence threshold is the trained quantity.
- Reward: correct candidate +1, no node +0.25, any wrong candidate -1. Loss = -expected reward summed over the K slots + v1 structural terms of matched slots (relation CE, pos CE, has-point BCE, parent CE; same weights), normalised by max(#reference nodes, 3). No meaning regression.
- Matching: Hungarian on the SAME costs as v1. Matched slot: the correct choice is its reference node's candidate. Unmatched slot, or an item with an empty reference (wrong story / empty memory): the correct choice is "no node".
- **Unreachable reference nodes** (identity not in the pool, so the decoder could not output them): excluded from the reward (their matched slot keeps the structural terms) and counted: **9.9% of reference nodes** (val-subsample; 14 of 181 per batch of val).
- Halting exactly as v2 (warm-up held at the geometric prior, lambda 1/3, 4 epochs; then PonderNet + KL, beta 0.3), expected loss over the halt distribution. Counterfactual items as v2 (p_cf 0.4), wrong-story 0.12 / empty-memory 0.06.

**Deviations needed to train at all (declared).**
1. *Commit warm-up.* In the first epochs (`--commit`) the existence head is bypassed: matched reachable slots get the reward of committing (2 s_correct - 1), no gradient to the existence head from the reward. Reason: with the existence head free, silence (reward 0.25) beats a chooser at chance (1/150).
2. *Config v3 (pure).* After the warm-up the loss is the exact expected reward. Result: existence collapsed to 0 within one epoch (ex on matched slots 0.000), coverage 0, and the choice gradient (proportional to ex) vanished. Run stopped after epoch 6 (it was dead); `v3_s0_best.pt` is its epoch-6 checkpoint.
3. *Config v3b (decoupled, `--decouple`).* To let the choice keep training while the model is silent: choice heads get the expected reward given that the slot answers (the same gradient with the factor ex dropped: a per-slot rescale, same direction), existence head gets the true expected reward with s held fixed. Validation always reports the pure expected-reward loss. Result: same silence (the choice stayed too weak to be worth answering). **Config v3b is the one evaluated on test.**
4. One seed per config (not two seeds of one config): time (both runs 25 + 44 min training). v3 and v3b are different configs, not seeds.
5. Checkpoint = lowest val loss after the release (pure expected-reward loss + structure). Because silence is reward-optimal, that is the epoch-6 checkpoint (val loss 0.690, `v3b_s0_best.pt`); later epochs got worse and noisier (val 0.8 to 2.1), never non-silent.
6. Existence-threshold choice on val: grid (0.05 ... 0.85) x tau (0.3, 0.5, 0.7), best val ROUGE (all tie at 0.000, so thr 0.05, tau 0.3). A first val run with tiny thresholds (1e-4 .. 0.01) in the grid chose thr 1e-4 (ROUGE 0.138): that is an operating point at the noise level of the existence probability (ex ~1e-4 for everything), not the trained policy, so the grid was restricted to meaningful probabilities BEFORE the test run. The tiny thresholds stay in the risk-coverage curve. The first val run is kept as `*_thrgrid_with_tiny*`.

## Test (once): config v3b, checkpoint epoch 6, thr 0.05, tau 0.3

| system | ROUGE-L (full sentence) | tree score % |
|---|---|---|
| loop v3b | 0.000 | 0.0 (n=300) |
| cheat: best-overlap sentence (decoded, recomputed) | 0.201 (raw 0.220) | 11.4 |
| loop v2 (copied) | 0.180 | 40.5 |
| loop v1 (copied) | 0.177 | 35.5 |
| v3_s2 / v5_all (copied) | 0.346 / 0.298 | 38.3 / 33.6 |

Loop minus cheat -0.201 (items [-0.212, -0.190]). Coverage (non-empty answers) **0.000**; ROUGE-L and tree score on answered items: undefined (no answered items). Length bands (n 35 / 202 / 688 / 82 for <=10 / 11-20 / 21+ / none): ROUGE-L 0.000 in all, coverage 0 in all.

### Risk-coverage over the existence threshold (test; selection was on val, val curve in `tables_val_b0.md`)

| thr | coverage | ROUGE-L all | ROUGE-L answered | mean nodes |
|---|---|---|---|---|
| 1e-5 | 1.000 | 0.071 | 0.071 | 23.7 (every slot) |
| 1e-4 | 1.000 | 0.104 | 0.104 | 5.2 |
| 1e-3 | 0.051 | 0.001 | 0.012 | 0.05 |
| 0.01 and above | 0.000 | 0.000 | n/a | 0 |

No threshold gives a regime where the answered subset is better than the whole: the existence probability is not a usable confidence (0.0001 on matched and unmatched slots alike, 0.0002 vs 0.0001).

### Choice accuracy diagnostic (val, 300 items, `diag_v3b_s0_best.json`)

2060 matched reachable slots: top-1 correct 12.6%, mean P(correct) 0.089, share with P(correct) > 0.625: 4.7%; accuracy by confidence quintile 0.155 / 0.216 / 0.180 / 0.051 / 0.029. Training-time val s_correct hovered at 0.04 to 0.10 for all epochs (random is 0.007) and never approached 0.625.

### Controls (test, thr 0.05): all five (empty memory, wrong story, question-only, shuffled, reads blocked) give the empty tree (abstain 1.000, ROUGE 0.000). Uninformative: the model is silent everywhere. At thr 1e-4 on val: normal 0.138, wrong 0.133, question-only 0.127, shuffled 0.136, reads blocked 0.130 (story adds +0.011 vs question-only, +0.005 vs wrong story, +0.002 vs shuffled): memory-independent, as in v2.

### Counterfactual (test n=450, val n=456): swapped (new) token in output 0.000, original token 0.000, output changed by swap 0.000 (silent). At val thr 1e-4: new-token rate 0.000, output changed by swap 0.193, ROUGE vs swapped gold 0.148 vs 0.147 on the original memory and gold: no reading of the swap.

### Halting (val = test within 0.001)
Learned halting probability per step 0.363 / 0.227 / 0.145 / 0.095 / 0.061 / 0.110 (prior 0.333 / 0.222 / 0.148 / 0.099 / 0.066 / 0.132). Eval rule tau 0.3: step 1 for all items (steps distribution {1: 1007}); at tau 0.7 step 3. Per-step pure v3 loss (val): 0.673, 0.731, 0.769, 0.758, 0.753, 0.746; test 0.511, 0.572, 0.610, 0.582, 0.589, 0.587. Almost flat across steps (silence at every step; the first step is slightly lower). The per-step numbers in `extras_*_b0.json` use the old v1 tree loss and are not meaningful for v3.

## Curves (v3b; val 150 items, thr 0.5 for the decoded monitor)

| ep | warm | commit | train | val loss | val cov | val s_correct | ex on matched |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 3.00 | 4.50 | 1.000 | 0.027 | 0.623 |
| 1 | 1 | 1 | 3.08 | 2.57 | 0.027 | 0.066 | 0.174 |
| 3 | 1 | 1 | 3.30 | 2.71 | 0.000 | 0.062 | 0.224 |
| 4 | 0 | 1 | 2.11 | 1.45 | 0.000 | 0.060 | 0.000 |
| 6 | 0 | 0 | 1.19 | **0.69** | 0.000 | 0.087 | 0.000 |
| 9 | 0 | 0 | 1.09 | 0.84 | 0.000 | 0.090 | 0.000 |
| 13 | 0 | 0 | 2.18 | 2.09 | 0.000 | 0.072 | 0.000 |

Pure config v3 (`v3_s0.curve.tsv`): commit epochs 0-5 (coverage 0 to 1, noisy), release at epoch 6: coverage 0.000, ex 0.000, val s_correct 0.037.

## Honest conclusion

The asymmetric reward is the right grade for "do not blur": the blur is gone and nothing else was changed. But with the reward fixed in advance, a chooser that picks the right word from the story for 1 slot in 8 (and cannot tell which 1 in 8) has a reward-optimal policy of staying silent, and the model learns exactly that. So v3 does not read memory either: it is silent instead of wrong. The same failure as v2 shows up from a different side, the thinker's output cannot be traced to the story. It was not tested whether more training, a learned temperature, a different TAU, or more epochs of the choice-only (committed) regime would lift the chooser above 0.625 on a useful subset; in 12 committed-plus-decoupled epochs per run s_correct rose from 0.03 to about 0.09 and plateaued. Per the method, no action head, copy head or attention-read meaning was added. Next step for the owner is a design question about the chooser (what teaches it to find the answer word in memory), not another variation of the grade.

## Files

Force-add: `src/naomi/mind_loop_v3.py`, `scripts/train_mind_loop_v3.py`, `scripts/eval_mind_loop_v3.py`, `scripts/diag_mind_loop_v3.py`, `scripts/eval_mind_loop.py`, `tests/test_mind_loop_v3.py`, `tests/test_naomi_no_literals.py`, `dev/reports/MIND_LOOP_V3.md`; in `runs/mind_loop_v3/`: `v3b_s0_best.pt` (3.2 MB, evaluated), `v3_s0_best.pt` (3.2 MB), `v3b_s0.log`, `v3b_s0.curve.tsv`, `v3_s0.log`, `v3_s0.curve.tsv`, `eval_val_b0.log`, `eval_test_b0.log`, `eval_val_b0.json`, `eval_test_b0.json`, `extras_{val,test}_b0.json`, `v3extras_{val,test}_b0.json`, `tables_{val,test}_b0.md`, `diag_v3b_s0_best.json`, `val_halt_v3loss_b0.json`, `val_choice_b0.json`, `*_thrgrid_with_tiny*` (3 small files).
Do NOT add: `*_last.pt`, `smoke*` (checkpoints and curve), `eval_rows_*_b0.jsonl` (about 0.6 MB each), `read_cache/` (copy of the v1 cache), `diag_v3b_s0_ep6.json` (superseded), `val_choice.json` (working copy).

Commands: `python scripts/train_mind_loop_v3.py train --tag v3b_s0 --seed 0 --epochs 14 --warm 4 --commit 4 --decouple` (pure: omit `--decouple`); `python scripts/eval_mind_loop_v3.py --split val|test --ckpt runs/mind_loop_v3/v3b_s0_best.pt --tag _b0` (val first); `python scripts/diag_mind_loop_v3.py --ckpt ... --n 300 --tag v3b_s0_best`.
