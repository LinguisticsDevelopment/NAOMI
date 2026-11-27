## test: n=1007 items, thr=0.5, tau=0.7

| system | full-sentence ROUGE-L | tree score % |
|---|---|---|
| loop (decoded output tree, setting (b)) | 0.180 | 40.5 (n=300) |
| cheat: best-overlap sentence (decoded) | 0.201 (raw text 0.220) | 11.4 |
| v3_s2 (MIND_V5.md, same items) | 0.346 | 38.3 |
| v5_all (MIND_V5.md) | 0.298 | 33.6 |

answer rate 1.000, mean output nodes 2.99, mean steps 3.90, steps distribution {'3': 100, '4': 907}

loop minus cheat: {"mean": -0.020661344683382075, "items": [-0.03539685318418223, -0.0053210838312277116], "stories": [-0.032443815734030705, -0.007837681451837287]}

| band | n | loop | cheat | short-only | mean steps | mean nodes |
|---|---|---|---|---|---|---|
| <=10 | 35 | 0.151 | 0.098 |  | 3.91 | 3.09 |
| 11-20 | 202 | 0.186 | 0.196 |  | 3.84 | 2.89 |
| 21+ | 688 | 0.196 | 0.227 |  | 3.91 | 3.23 |
| none | 82 | 0.045 | 0.041 |  | 3.99 | 1.23 |

### Controls

| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |
|---|---|---|---|---|---|
| empty | 1007 | 1.000 | 0.00 | 0.000 | 1.00 |
| wrong | 1007 | 0.000 | 2.97 | 0.147 | 3.90 |
| qonly | 1007 | 0.000 | 1.70 | 0.059 | 3.97 |
| shuffled | 1007 | 0.000 | 3.02 | 0.130 | 3.89 |
| block | 1007 | 0.000 | 2.72 | 0.176 | 3.62 |

block-reads ROUGE delta vs normal: -0.004

tree-score reasons (loop): {'node_dropped': 823, 'meaning_shift': 404, 'operator_folded_differs': 239, 'node_extra': 83, 'subtree_moved': 33, 'relation_changed': 26, 'name_mismatch': 23, 'head_changed': 16}
### v2 extras (test, _s1): thr=0.5 tau=0.7

```
{
 "n_cf": 450,
 "n_items": 1007,
 "cf_rouge_vs_swapped_gold": 0.16118578634650121,
 "cf_new_token_rate": 0.017777777777777778,
 "cf_old_token_rate": 0.0,
 "orig_memory_rouge_vs_orig_gold": 0.16982747461939152,
 "cf_memory_rouge_vs_orig_gold": 0.15849701281066883,
 "orig_memory_old_token_rate": 0.028888888888888888,
 "orig_memory_new_token_rate": 0.0,
 "output_changed_by_swap": 0.07111111111111111,
 "per_step_loss": [
  3.4942,
  3.4789,
  3.4715,
  3.47,
  3.4731,
  3.4719
 ],
 "mean_halt_prob": [
  0.3148,
  0.2229,
  0.1507,
  0.1016,
  0.068,
  0.1419
 ],
 "halt_steps_dist": {
  "3": 100,
  "4": 907
 },
 "story_adds_vs_qonly": 0.1213344220377793,
 "story_adds_vs_wrong": 0.03312907580806035,
 "story_adds_vs_shuffled": 0.050175831450742725
}
```
