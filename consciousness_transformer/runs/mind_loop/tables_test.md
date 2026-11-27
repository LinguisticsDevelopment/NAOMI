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
