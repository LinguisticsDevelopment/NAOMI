## test: n=1007 items, thr=0.05, tau=0.3

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.000 | 0.0 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.201 (raw text 0.220) | 11.4 |
| v3_s2 (MIND_V5.md, same items) | 0.346 | 38.3 |
| v5_all (MIND_V5.md) | 0.298 | 33.6 |

answer rate 0.000, mean output nodes 0.00, mean steps 1.00, steps distribution {'1': 1007}

loop minus cheat: {"mean": -0.20089408123515004, "items": [-0.21178930486073236, -0.18962879409335362], "stories": [-0.21493081216993362, -0.18495821280089225]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 35 | 0.000 | 0.098 |  | 1.00 | 0.00 |
| 11-20 | 202 | 0.000 | 0.196 |  | 1.00 | 0.00 |
| 21+ | 688 | 0.000 | 0.227 |  | 1.00 | 0.00 |
| none | 82 | 0.000 | 0.041 |  | 1.00 | 0.00 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1007 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1007 | 1.000 | 0.00 | 0.000 | 1.00 |
| qonly | 1007 | 1.000 | 0.00 | 0.000 | 1.00 |
| shuffled | 1007 | 1.000 | 0.00 | 0.000 | 1.00 |
| block | 1007 | 1.000 | 0.00 | 0.000 | 1.00 |

block-reads ROUGE delta vs normal: +0.000

tree-score reasons (loop): {}
### v2 extras (test, _b0): thr=0.05 tau=0.3

```
{
 "n_cf": 450,
 "n_items": 1007,
 "cf_rouge_vs_swapped_gold": 0.0,
 "cf_new_token_rate": 0.0,
 "cf_old_token_rate": 0.0,
 "orig_memory_rouge_vs_orig_gold": 0.0,
 "cf_memory_rouge_vs_orig_gold": 0.0,
 "orig_memory_old_token_rate": 0.0,
 "orig_memory_new_token_rate": 0.0,
 "output_changed_by_swap": 0.0,
 "per_step_loss": [
  10.9086,
  10.9065,
  10.9163,
  10.8327,
  10.8582,
  10.7813
 ],
 "mean_halt_prob": [
  0.3627,
  0.227,
  0.1454,
  0.0945,
  0.0606,
  0.1098
 ],
 "halt_steps_dist": {
  "1": 1007
 },
 "story_adds_vs_qonly": 0.0,
 "story_adds_vs_wrong": 0.0,
 "story_adds_vs_shuffled": 0.0
}
```
### v3 additions (test, _b0): thr=0.05 tau=0.3

```
{
 "thr": 0.05,
 "tau": 0.3,
 "n": 1007,
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
   "n": 35,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  },
  "11-20": {
   "n": 202,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  },
  "21+": {
   "n": 688,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  },
  "none": {
   "n": 82,
   "coverage": 0.0,
   "rouge_all": 0.0,
   "rouge_answered": null
  }
 }
}
```

| thr | coverage | ROUGE-L all | ROUGE-L answered | mean nodes |
|---|---|---|---|---|
| 1e-05 | 1.000 | 0.071 | 0.071 | 23.66 |
| 0.0001 | 1.000 | 0.104 | 0.104 | 5.19 |
| 0.001 | 0.051 | 0.001 | 0.012 | 0.05 |
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
