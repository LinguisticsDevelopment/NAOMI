## test: n=1007 items, 23 stories; bands {'<=10': 35, '11-20': 202, '21+': 688, 'none': 82}

### Headline ROUGE-L (max over golds, abstain = 0)

| system | ROUGE-L | answered-only | answer rate |
|---|---|---|---|
| cheat: best-overlap sentence | 0.220 | nan | nan |
| baseline: question echo | 0.116 | nan | nan |
| baseline: random story sentence | 0.136 | nan | nan |
| baseline: random-anchor full subtree | 0.078 | nan | nan |
| oracle: best full subtree (by ROUGE, true order) | 0.757 | nan | nan |
| oracle: lemma-F1 pruned subtree (true order) | 0.752 | nan | nan |
| v4_all: (a) order-only decoder | 0.297 | 0.315 | 0.943 |
| v4_all: (b) meaning-only decoder | 0.293 | 0.311 | 0.943 |
| v4_all: true word order (no decoder) | 0.315 | 0.334 | 0.943 |
| v3_s2: (a) order-only decoder | 0.345 | 0.374 | 0.924 |
| v3_s2: (b) meaning-only decoder | 0.338 | 0.366 | 0.924 |
| v3_s2: true word order (no decoder) | 0.369 | 0.399 | 0.924 |
| v2b: (a) order-only decoder | 0.348 | 0.374 | 0.932 |
| v2b: (b) meaning-only decoder | 0.342 | 0.367 | 0.932 |
| v2b: true word order (no decoder) | 0.372 | 0.399 | 0.932 |
| v3_s1: (a) order-only decoder | 0.339 | 0.365 | 0.929 |
| v3_s1: (b) meaning-only decoder | 0.334 | 0.360 | 0.929 |
| v3_s1: true word order (no decoder) | 0.367 | 0.395 | 0.929 |

### By answer-sentence length band (decoder (a))

| band | n | cheat | v4_all | v3_s2 | v2b | v3_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 35 | 0.104 | 0.150 | 0.228 | 0.168 | 0.177 | 0.815 |
| 11-20 | 202 | 0.205 | 0.347 | 0.405 | 0.410 | 0.406 | 0.802 |
| 21+ | 688 | 0.251 | 0.320 | 0.362 | 0.372 | 0.361 | 0.792 |
| none | 82 | 0.045 | 0.046 | 0.103 | 0.072 | 0.059 | 0.331 |

### Short-to-long (decoder (a))

| pair | band | n | all lengths | <=10 only |
|---|---|---|---|---|
| v4_short vs v4_all | <=10 | 35 | 0.150 | 0.250 |
| v4_short vs v4_all | 11-20 | 202 | 0.347 | 0.163 |
| v4_short vs v4_all | 21+ | 688 | 0.320 | 0.144 |
| v4_short vs v4_all | none | 82 | 0.046 | 0.035 |
| v3_short vs v3_s1 | <=10 | 35 | 0.177 | 0.343 |
| v3_short vs v3_s1 | 11-20 | 202 | 0.406 | 0.176 |
| v3_short vs v3_s1 | 21+ | 688 | 0.361 | 0.110 |
| v3_short vs v3_s1 | none | 82 | 0.059 | 0.022 |

### Paired differences (decoder (a); mean, 95% interval over items, over stories)

- v4_all - v3_s2: -0.048 (items -0.066 to -0.029; stories -0.066 to -0.035)
- v4_all - cheat: +0.077 (items +0.058 to +0.097; stories +0.057 to +0.098)
- v4_all - v2b: -0.051 (items -0.071 to -0.033; stories -0.073 to -0.031)
- v4_all - v3_s1: -0.042 (items -0.061 to -0.024; stories -0.061 to -0.026)

### By FairytaleQA attribute (decoder (a))

| attribute | n | cheat | v4_all | v3_s2 | v2b | v3_s1 | oracle | answer rate (main) |
|---|---|---|---|---|---|---|---|---|
| action | 315 | 0.240 | 0.324 | 0.355 | 0.349 | 0.365 | 0.825 | 0.930 |
| causal relationship | 278 | 0.282 | 0.287 | 0.334 | 0.350 | 0.331 | 0.740 | 0.960 |
| feeling | 106 | 0.053 | 0.248 | 0.234 | 0.269 | 0.241 | 0.462 | 0.877 |
| character | 103 | 0.178 | 0.304 | 0.439 | 0.457 | 0.424 | 0.882 | 0.961 |
| outcome resolution | 78 | 0.285 | 0.348 | 0.404 | 0.328 | 0.373 | 0.813 | 0.962 |
| prediction | 65 | 0.120 | 0.179 | 0.226 | 0.177 | 0.189 | 0.582 | 0.969 |
| setting | 62 | 0.222 | 0.336 | 0.427 | 0.502 | 0.390 | 0.898 | 0.968 |

### Controls

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v4_all | 1.000 | 0.939 | 0.479 / 0.061 | 0.167 |
| v3_s2 | 1.000 | 0.950 | 0.411 / 0.055 | 0.855 |
| v2b | 1.000 | 0.946 | 0.483 / 0.061 | 0.836 |
| v3_s1 | 1.000 | 0.968 | 0.406 / 0.054 | 0.002 |
| v4_short | 1.000 | 0.929 | 0.431 / 0.050 | 0.001 |
| v3_short | 1.000 | 0.913 | 0.416 / 0.044 | 0.000 |

### Tree score and round trip (main = v4_all; threshold 99)

- (a) tree score answered 36.0, abstain=0 34.0; new round-trip pass 0.598; by length {"1-3 words": {"n": 245, "pass_rate": 0.9836734693877551}, "4-6 words": {"n": 166, "pass_rate": 0.7469879518072289}, "7+ words": {"n": 539, "pass_rate": 0.37662337662337664}}; reasons {"node_extra": 3241, "node_dropped": 1763, "meaning_shift": 1343, "operator_folded_differs": 828, "head_changed": 592, "subtree_moved": 450, "name_mismatch": 225, "relation_changed": 161, "opposite_meaning": 2}
- (b) tree score answered 35.9, abstain=0 33.9; new round-trip pass 0.579; by length {"1-3 words": {"n": 245, "pass_rate": 0.963265306122449}, "4-6 words": {"n": 166, "pass_rate": 0.7168674698795181}, "7+ words": {"n": 539, "pass_rate": 0.36178107606679033}}; reasons {"node_extra": 3238, "node_dropped": 1763, "meaning_shift": 1350, "operator_folded_differs": 833, "head_changed": 588, "subtree_moved": 452, "name_mismatch": 228, "relation_changed": 160, "opposite_meaning": 2}
- cheat tree score 13.6

### Needs-new-words items

- v4_all: (a) order-only decoder: 0.144 (n=224)
- v4_all: (b) meaning-only decoder: 0.140 (n=224)
- v4_all: true word order (no decoder): 0.148 (n=224)
- v3_s2: (a) order-only decoder: 0.167 (n=224)
- v3_s2: (b) meaning-only decoder: 0.165 (n=224)
- v3_s2: true word order (no decoder): 0.173 (n=224)
- v2b: (a) order-only decoder: 0.149 (n=224)
- v2b: (b) meaning-only decoder: 0.147 (n=224)
- v2b: true word order (no decoder): 0.157 (n=224)
- v3_s1: (a) order-only decoder: 0.152 (n=224)
- v3_s1: (b) meaning-only decoder: 0.150 (n=224)
- v3_s1: true word order (no decoder): 0.161 (n=224)
