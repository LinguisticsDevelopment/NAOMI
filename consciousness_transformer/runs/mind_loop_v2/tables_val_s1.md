## val: n=1025 items, thr=0.5, tau=0.7

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.187 | 38.3 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.207 (raw text 0.228) | 13.0 |

answer rate 1.000, mean output nodes 3.09, mean steps 3.90, steps distribution {'3': 102, '4': 923}

loop minus cheat: {"mean": -0.020612015676235815, "items": [-0.035870733612290165, -0.005527459579830123], "stories": [-0.037480875380544766, -0.004796798393279389]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 38 | 0.257 | 0.158 |  | 3.76 | 3.11 |
| 11-20 | 209 | 0.189 | 0.206 |  | 3.83 | 3.04 |
| 21+ | 708 | 0.198 | 0.229 |  | 3.92 | 3.30 |
| none | 70 | 0.027 | 0.013 |  | 3.99 | 1.06 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1025 | 0.000 | 3.09 | 0.152 | 3.89 |
| qonly | 1025 | 0.000 | 1.76 | 0.064 | 3.96 |
| shuffled | 1025 | 0.000 | 3.12 | 0.139 | 3.90 |
| block | 1025 | 0.000 | 2.81 | 0.185 | 3.64 |

block-reads ROUGE delta vs normal: -0.002

tree-score reasons (loop): {'node_dropped': 987, 'meaning_shift': 441, 'operator_folded_differs': 266, 'node_extra': 84, 'subtree_moved': 34, 'relation_changed': 31, 'name_mismatch': 19, 'head_changed': 19}
### v2 extras (val, _s1): thr=0.5 tau=0.7

```
{
 "n_cf": 456,
 "n_items": 1025,
 "cf_rouge_vs_swapped_gold": 0.16160050432114914,
 "cf_new_token_rate": 0.015350877192982455,
 "cf_old_token_rate": 0.0,
 "orig_memory_rouge_vs_orig_gold": 0.1622584682577412,
 "cf_memory_rouge_vs_orig_gold": 0.1569989744940923,
 "orig_memory_old_token_rate": 0.02412280701754386,
 "orig_memory_new_token_rate": 0.0,
 "output_changed_by_swap": 0.06359649122807018,
 "per_step_loss": [
  3.5865,
  3.5685,
  3.5571,
  3.556,
  3.557,
  3.5574
 ],
 "mean_halt_prob": [
  0.3148,
  0.223,
  0.1508,
  0.1017,
  0.068,
  0.1418
 ],
 "halt_steps_dist": {
  "3": 102,
  "4": 923
 },
 "story_adds_vs_qonly": 0.12226438229419805,
 "story_adds_vs_wrong": 0.034995745707172204,
 "story_adds_vs_shuffled": 0.047920796699133406
}
```
