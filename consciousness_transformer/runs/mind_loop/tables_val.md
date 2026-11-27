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
