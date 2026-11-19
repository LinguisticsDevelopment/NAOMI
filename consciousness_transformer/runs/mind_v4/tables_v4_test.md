## v4 walk diagnostics (test)

Short-trained items: oracle anchor depth max 6, 95th pct 3, mean 0.86.

### Steps, budget exhaustion, chosen anchor depth vs oracle depth, by answer-sentence band

| model | band | n | answer rate | mean steps | budget exhausted | chosen depth (answered) | oracle depth | chosen deeper than short-train max | chosen anchor is acceptable | chosen sentence is the oracle sentence |
|---|---|---|---|---|---|---|---|---|---|---|
| v4_all | <=10 | 35 | 0.914 | 3.11 | 0.000 | 2.31 | 1.03 | 0.094 | 0.091 | 0.182 |
| v4_all | 11-20 | 202 | 0.965 | 3.01 | 0.000 | 2.08 | 1.53 | 0.031 | 0.260 | 0.505 |
| v4_all | 21+ | 688 | 0.948 | 4.13 | 0.000 | 3.30 | 2.78 | 0.117 | 0.251 | 0.616 |
| v4_all | none | 82 | 0.866 | 3.66 | 0.000 | 3.07 | nan | 0.042 | nan | nan |
| v4_short | <=10 | 35 | 0.914 | 1.46 | 0.000 | 0.50 | 1.03 | 0.000 | 0.152 | 0.303 |
| v4_short | 11-20 | 202 | 0.911 | 2.38 | 0.000 | 1.51 | 1.53 | 0.038 | 0.083 | 0.312 |
| v4_short | 21+ | 688 | 0.887 | 3.58 | 0.000 | 2.89 | 2.78 | 0.131 | 0.031 | 0.254 |
| v4_short | none | 82 | 0.817 | 2.95 | 0.000 | 2.39 | nan | 0.104 | nan | nan |

### Tree adequacy: share of items whose reference answer equals some subtree at ROUGE-L >= 0.8 (any node as anchor, true word order)

| band | n | share >= 0.8 | share >= 0.5 | mean best-subtree ROUGE |
|---|---|---|---|---|
| <=10 | 35 | 0.571 | 0.943 | 0.815 |
| 11-20 | 202 | 0.579 | 0.936 | 0.802 |
| 21+ | 688 | 0.567 | 0.920 | 0.792 |
| none | 82 | 0.159 | 0.378 | 0.331 |
| all | 1007 | 0.536 | 0.880 | 0.757 |

### Failure attribution, v4_all (items with a reachable oracle answer; ROUGE-L true order; fail = ROUGE < 0.5 x oracle subtree ROUGE... see columns)

| band | n | right (within 0.05 of oracle subtree) | abstained / budget out | wrong sentence | right sentence, wrong anchor, TREE adequate (>=0.8) | right sentence, wrong anchor, tree NOT adequate |
|---|---|---|---|---|---|---|
| <=10 | 33 | 0.091 (3) | 0.061 (2) | 0.758 (25) | 0.030 (1) | 0.061 (2) |
| 11-20 | 192 | 0.234 (45) | 0.036 (7) | 0.453 (87) | 0.156 (30) | 0.120 (23) |
| 21+ | 649 | 0.185 (120) | 0.046 (30) | 0.331 (215) | 0.274 (178) | 0.163 (106) |
