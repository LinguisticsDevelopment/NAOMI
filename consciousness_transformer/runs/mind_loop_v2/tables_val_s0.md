## val: n=1025 items, thr=0.5, tau=0.5

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.177 | 38.5 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.207 (raw text 0.228) | 13.0 |

answer rate 1.000, mean output nodes 3.35, mean steps 2.00, steps distribution {'2': 1024, '3': 1}

loop minus cheat: {"mean": -0.029795327854262626, "items": [-0.0447391051574261, -0.015466496537860558], "stories": [-0.04909290091454026, -0.012546054557614254]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 38 | 0.255 | 0.158 |  | 2.00 | 3.45 |
| 11-20 | 209 | 0.179 | 0.206 |  | 2.00 | 3.26 |
| 21+ | 708 | 0.188 | 0.229 |  | 2.00 | 3.59 |
| none | 70 | 0.027 | 0.013 |  | 2.00 | 1.10 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1025 | 0.000 | 3.35 | 0.151 | 2.00 |
| qonly | 1025 | 0.000 | 2.26 | 0.112 | 2.00 |
| shuffled | 1025 | 0.000 | 3.36 | 0.135 | 2.00 |
| block | 1025 | 0.000 | 3.15 | 0.169 | 2.00 |

block-reads ROUGE delta vs normal: -0.008

tree-score reasons (loop): {'node_dropped': 972, 'meaning_shift': 533, 'operator_folded_differs': 303, 'node_extra': 105, 'subtree_moved': 39, 'relation_changed': 36, 'name_mismatch': 20, 'head_changed': 15}
### v2 extras (val, _s0): thr=0.5 tau=0.5

```
{
 "n_cf": 456,
 "n_items": 1025,
 "cf_rouge_vs_swapped_gold": 0.16234643428535522,
 "cf_new_token_rate": 0.021929824561403508,
 "cf_old_token_rate": 0.0,
 "orig_memory_rouge_vs_orig_gold": 0.16161015434015044,
 "cf_memory_rouge_vs_orig_gold": 0.15552408298486162,
 "orig_memory_old_token_rate": 0.02631578947368421,
 "orig_memory_new_token_rate": 0.0,
 "output_changed_by_swap": 0.11403508771929824,
 "per_step_loss": [
  3.6372,
  3.6629,
  3.6535,
  3.6704,
  3.6668,
  3.6597
 ],
 "mean_halt_prob": [
  0.3397,
  0.2195,
  0.1434,
  0.0957,
  0.0643,
  0.1374
 ],
 "halt_steps_dist": {
  "2": 1024,
  "3": 1
 },
 "story_adds_vs_qonly": 0.0651940208529806,
 "story_adds_vs_wrong": 0.02635694918770301,
 "story_adds_vs_shuffled": 0.04280253824091987
}
```
