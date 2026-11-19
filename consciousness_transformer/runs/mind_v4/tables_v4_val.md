## v4 walk diagnostics (val)

Short-trained items: oracle anchor depth max 6, 95th pct 3, mean 0.86.

### Steps, budget exhaustion, chosen anchor depth vs oracle depth, by answer-sentence band

| model | band | n | answer rate | mean steps | budget exhausted | chosen depth (answered) | oracle depth | chosen deeper than short-train max | chosen anchor is acceptable | chosen sentence is the oracle sentence |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_all | <=10 | 38 | 1.000 | 2.71 | 0.000 | 1.71 | 0.53 | 0.053 | 0.219 | 0.281 |
| v4_all | 11-20 | 209 | 0.962 | 3.22 | 0.000 | 2.30 | 1.49 | 0.055 | 0.214 | 0.340 |
| v4_all | 21+ | 708 | 0.963 | 4.35 | 0.000 | 3.47 | 3.02 | 0.147 | 0.231 | 0.588 |
| v4_all | none | 70 | 0.900 | 4.56 | 0.000 | 3.95 | nan | 0.127 | nan | nan |
| v4_short | <=10 | 38 | 0.974 | 1.61 | 0.000 | 0.57 | 0.53 | 0.000 | 0.219 | 0.344 |
| v4_short | 11-20 | 209 | 0.890 | 2.24 | 0.000 | 1.39 | 1.49 | 0.048 | 0.058 | 0.238 |
| v4_short | 21+ | 708 | 0.925 | 3.82 | 0.000 | 3.04 | 3.02 | 0.168 | 0.027 | 0.274 |
| v4_short | none | 70 | 0.857 | 2.40 | 0.000 | 1.63 | nan | 0.067 | nan | nan |

### Tree adequacy: share of items whose reference answer equals some subtree at ROUGE-L >= 0.8 (any node as anchor, true word order)

| band | n | share >= 0.8 | share >= 0.5 | mean best-subtree ROUGE |
|---|---|---|---|---|
| <=10 | 38 | 0.526 | 0.868 | 0.754 |
| 11-20 | 209 | 0.584 | 0.976 | 0.802 |
| 21+ | 708 | 0.568 | 0.948 | 0.801 |
| none | 70 | 0.143 | 0.243 | 0.220 |
| all | 1025 | 0.540 | 0.902 | 0.760 |

### Failure attribution, v4_all (items with a reachable oracle answer; ROUGE-L true order; fail = ROUGE < 0.5 x oracle subtree ROUGE... see columns)

| band | n | right (within 0.05 of oracle subtree) | abstained / budget out | wrong sentence | right sentence, wrong anchor, TREE adequate (>=0.8) | right sentence, wrong anchor, tree NOT adequate |
|---|---|---|---|---|---|---|
| <=10 | 32 | 0.156 (5) | 0.000 (0) | 0.719 (23) | 0.094 (3) | 0.031 (1) |
| 11-20 | 206 | 0.194 (40) | 0.039 (8) | 0.607 (125) | 0.112 (23) | 0.049 (10) |
| 21+ | 668 | 0.168 (112) | 0.037 (25) | 0.368 (246) | 0.274 (183) | 0.153 (102) |

hand-read dump: 20 of 407 failing 21+ items -> runs/mind_v4/handread_val_21plus.json
