## val: n=1025 items, thr=0.0001, tau=0.7

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.138 | 19.8 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.207 (raw text 0.228) | 13.0 |

answer rate 0.998, mean output nodes 5.48, mean steps 3.00, steps distribution {'3': 1025}

loop minus cheat: {"mean": -0.06964155948427525, "items": [-0.08156352454300579, -0.0577850607361152], "stories": [-0.08706818230818135, -0.052825035100003886]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 38 | 0.142 | 0.158 |  | 3.00 | 5.63 |
| 11-20 | 209 | 0.136 | 0.206 |  | 3.00 | 5.49 |
| 21+ | 708 | 0.150 | 0.229 |  | 3.00 | 5.48 |
| none | 70 | 0.015 | 0.013 |  | 3.00 | 5.39 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1025 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1025 | 0.001 | 5.41 | 0.133 | 3.00 |
| qonly | 1025 | 0.000 | 6.06 | 0.127 | 3.00 |
| shuffled | 1025 | 0.002 | 5.47 | 0.136 | 3.00 |
| block | 1025 | 0.000 | 6.63 | 0.130 | 3.00 |

block-reads ROUGE delta vs normal: -0.007

tree-score reasons (loop): {'node_dropped': 1014, 'node_extra': 462, 'meaning_shift': 415, 'operator_folded_differs': 318, 'relation_changed': 67, 'head_changed': 42, 'subtree_moved': 32, 'name_mismatch': 26}
### v2 extras (val, _b0): thr=0.0001 tau=0.7

```
{
 "n_cf": 456,
 "n_items": 1025,
 "cf_rouge_vs_swapped_gold": 0.147595694272095,
 "cf_new_token_rate": 0.0,
 "cf_old_token_rate": 0.0,
 "orig_memory_rouge_vs_orig_gold": 0.14700725397171155,
 "cf_memory_rouge_vs_orig_gold": 0.14728013117487998,
 "orig_memory_old_token_rate": 0.0,
 "orig_memory_new_token_rate": 0.0,
 "output_changed_by_swap": 0.19298245614035087,
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
  "3": 1025
 },
 "story_adds_vs_qonly": 0.010671889101501963,
 "story_adds_vs_wrong": 0.0045782770880256896,
 "story_adds_vs_shuffled": 0.0020683780723361467
}
```
### v3 additions (val, _b0): thr=0.0001 tau=0.7

```
{
 "thr": 0.0001,
 "tau": 0.7,
 "n": 1025,
 "coverage": 0.9980487804878049,
 "rouge_all": 0.13756859699765633,
 "rouge_answered": 0.13783754831143472,
 "cheat_on_answered": 0.20708460728359993,
 "tree_all_300": 19.765423257115355,
 "tree_answered_only": 19.765423257115355,
 "n_tree": 300,
 "n_tree_answered": 300,
 "bands": {
  "<=10": {
   "n": 38,
   "coverage": 1.0,
   "rouge_all": 0.1420116189853032,
   "rouge_answered": 0.1420116189853032
  },
  "11-20": {
   "n": 209,
   "coverage": 1.0,
   "rouge_all": 0.13616076689542822,
   "rouge_answered": 0.13616076689542822
  },
  "21+": {
   "n": 708,
   "coverage": 0.998587570621469,
   "rouge_all": 0.14990017750786336,
   "rouge_answered": 0.1501122003897698
  },
  "none": {
   "n": 70,
   "coverage": 0.9857142857142858,
   "rouge_all": 0.014634920634920636,
   "rouge_answered": 0.014847020933977457
  }
 }
}
```

| thr | coverage | ROUGE-L all | ROUGE-L answered | mean nodes |
|---|---|---|---|---|
| 1e-05 | 1.000 | 0.066 | 0.066 | 23.95 |
| 0.0001 | 0.998 | 0.138 | 0.138 | 5.48 |
| 0.001 | 0.113 | 0.006 | 0.051 | 0.11 |
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
