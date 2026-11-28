## val: n=1025 items, thr=0.05, tau=0.3

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.000 | 0.0 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.207 (raw text 0.228) | 13.0 |

answer rate 0.000, mean output nodes 0.00, mean steps 1.00, steps distribution {'1': 1025}

loop minus cheat: {"mean": -0.20721015648193156, "items": [-0.21850804933351228, -0.19643071211933905], "stories": [-0.2242557388806181, -0.19063124442179555]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 38 | 0.000 | 0.158 |  | 1.00 | 0.00 |
| 11-20 | 209 | 0.000 | 0.206 |  | 1.00 | 0.00 |
| 21+ | 708 | 0.000 | 0.229 |  | 1.00 | 0.00 |
| none | 70 | 0.000 | 0.013 |  | 1.00 | 0.00 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| qonly | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| shuffled | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| block | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |

block-reads ROUGE delta vs normal: +0.000

tree-score reasons (loop): {}
### v2 extras (val, _b0): thr=0.05 tau=0.3

```
{
 "n_cf": 456,
 "n_items": 1025,
 "cf_rouge_vs_swapped_gold": 0.0,
 "cf_new_token_rate": 0.0,
 "cf_old_token_rate": 0.0,
 "orig_memory_rouge_vs_orig_gold": 0.0,
 "cf_memory_rouge_vs_orig_gold": 0.0,
 "orig_memory_old_token_rate": 0.0,
 "orig_memory_new_token_rate": 0.0,
 "output_changed_by_swap": 0.0,
 "per_step_loss": [
  11.3158,
  11.3117,
  11.3152,
  11.2341,
  11.2478,
  11.1735
 ],
 "mean_halt_prob": [
  0.3628,
  0.2269,
  0.1453,
  0.0946,
  0.0606,
  0.1099
 ],
 "halt_steps_dist": {
  "1": 1025
 },
 "story_adds_vs_qonly": 0.0,
 "story_adds_vs_wrong": 0.0,
 "story_adds_vs_shuffled": 0.0
}
```
### v3 additions (val, _b0): thr=0.05 tau=0.3

```
{
 "thr": 0.05,
 "tau": 0.3,
 "n": 1025,
 "coverage": 0.0,
 "rouge_all": 0.0,
 "rouge_answered": null,
 "cheat_on_answered": null,
 "tree_all_300": 0.0,
 "tree_answered_only": null,
 "n_tree": 300,
 "n_tree_answered": 0,
 "bands": {
  "<=10": {
   "n": 38,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  },
  "11-20": {
   "n": 209,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  },
  "21+": {
   "n": 708,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  },
  "none": {
   "n": 70,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  }
 }
}
```

| thr | coverage | ROUGE-L all | ROUGE-L answered | mean nodes |
|---|---|---|---|---|
| 1e-05 | 1.000 | 0.076 | 0.076 | 23.61 |
| 0.0001 | 1.000 | 0.107 | 0.107 | 5.14 |
| 0.001 | 0.061 | 0.003 | 0.047 | 0.06 |
| 0.01 | 0.000 | 0.000 |  | 0.00 |
| 0.02 | 0.000 | 0.000 |  | 0.00 |
| 0.05 | 0.000 | 0.000 |  | 0.00 |
| 0.1 | 0.000 | 0.000 |  | 0.00 |
| 0.15 | 0.000 | 0.000 |  | 0.00 |
| 0.2 | 0.000 | 0.000 |  | 0.00 |
| 0.3 | 0.000 | 0.000 |  | 0.00 |
| 0.4 | 0.000 | 0.000 |  | 0.00 |
| 0.5 | 0.000 | 0.000 |  | 0.00 |
| 0.6 | 0.000 | 0.000 |  | 0.00 |
| 0.7 | 0.000 | 0.000 |  | 0.00 |
| 0.8 | 0.000 | 0.000 |  | 0.00 |
| 0.9 | 0.000 | 0.000 |  | 0.00 |
