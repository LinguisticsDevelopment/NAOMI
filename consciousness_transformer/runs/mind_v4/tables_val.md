## val: n=1025 items, 23 stories; bands {'<=10': 38, '11-20': 209, '21+': 708, 'none': 70}

### Headline ROUGE-L (max over golds, abstain = 0)

| system | ROUGE-L | answered-only | answer rate |
|---|---|---|---|
| cheat: best-overlap sentence | 0.228 | nan | nan |
| baseline: question echo | 0.130 | nan | nan |
| baseline: random story sentence | 0.137 | nan | nan |
| baseline: random-anchor full subtree | 0.087 | nan | nan |
| oracle: best full subtree (by ROUGE, true order) | 0.760 | nan | nan |
| oracle: lemma-F1 pruned subtree (true order) | 0.768 | nan | nan |
| v4_all: (a) order-only decoder | 0.300 | 0.312 | 0.960 |
| v4_all: (b) meaning-only decoder | 0.295 | 0.308 | 0.960 |
| v4_all: true word order (no decoder) | 0.319 | 0.332 | 0.960 |
| v3_s2: (a) order-only decoder | 0.358 | 0.382 | 0.936 |
| v3_s2: (b) meaning-only decoder | 0.348 | 0.372 | 0.936 |
| v3_s2: true word order (no decoder) | 0.384 | 0.410 | 0.936 |
| v2b: (a) order-only decoder | 0.353 | 0.373 | 0.946 |
| v2b: (b) meaning-only decoder | 0.346 | 0.365 | 0.946 |
| v2b: true word order (no decoder) | 0.373 | 0.395 | 0.946 |
| v3_s1: (a) order-only decoder | 0.340 | 0.365 | 0.932 |
| v3_s1: (b) meaning-only decoder | 0.334 | 0.359 | 0.932 |
| v3_s1: true word order (no decoder) | 0.365 | 0.392 | 0.932 |

### By answer-sentence length band (decoder (a))

| band | n | cheat | v4_all | v3_s2 | v2b | v3_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 38 | 0.164 | 0.262 | 0.368 | 0.296 | 0.258 | 0.754 |
| 11-20 | 209 | 0.211 | 0.314 | 0.414 | 0.394 | 0.377 | 0.802 |
| 21+ | 708 | 0.258 | 0.325 | 0.374 | 0.377 | 0.363 | 0.801 |
| none | 70 | 0.013 | 0.026 | 0.023 | 0.017 | 0.045 | 0.220 |

### Short-to-long (decoder (a))

| pair | band | n | all lengths | <=10 only |
|---|---|---|---|---|
| v4_short vs v4_all | <=10 | 38 | 0.262 | 0.289 |
| v4_short vs v4_all | 11-20 | 209 | 0.314 | 0.171 |
| v4_short vs v4_all | 21+ | 708 | 0.325 | 0.149 |
| v4_short vs v4_all | none | 70 | 0.026 | 0.005 |
| v3_short vs v3_s1 | <=10 | 38 | 0.258 | 0.386 |
| v3_short vs v3_s1 | 11-20 | 209 | 0.377 | 0.193 |
| v3_short vs v3_s1 | 21+ | 708 | 0.363 | 0.111 |
| v3_short vs v3_s1 | none | 70 | 0.045 | 0.017 |

### Paired differences (decoder (a); mean, 95% interval over items, over stories)

- v4_all - v3_s2: -0.058 (items -0.078 to -0.039; stories -0.079 to -0.039)
- v4_all - cheat: +0.072 (items +0.053 to +0.091; stories +0.053 to +0.090)
- v4_all - v2b: -0.053 (items -0.071 to -0.034; stories -0.071 to -0.031)
- v4_all - v3_s1: -0.040 (items -0.059 to -0.022; stories -0.068 to -0.015)

### By FairytaleQA attribute (decoder (a))

| attribute | n | cheat | v4_all | v3_s2 | v2b | v3_s1 | oracle | answer rate (main) |
|---|---|---|---|---|---|---|---|---|
| action | 333 | 0.266 | 0.325 | 0.383 | 0.383 | 0.364 | 0.826 | 0.964 |
| causal relationship | 294 | 0.264 | 0.283 | 0.323 | 0.313 | 0.311 | 0.738 | 0.966 |
| character | 107 | 0.171 | 0.345 | 0.453 | 0.436 | 0.402 | 0.849 | 0.963 |
| outcome resolution | 97 | 0.312 | 0.327 | 0.433 | 0.375 | 0.390 | 0.850 | 0.959 |
| feeling | 94 | 0.034 | 0.223 | 0.234 | 0.229 | 0.248 | 0.411 | 0.926 |
| prediction | 55 | 0.132 | 0.187 | 0.207 | 0.242 | 0.191 | 0.625 | 0.945 |
| setting | 45 | 0.189 | 0.352 | 0.452 | 0.539 | 0.475 | 0.900 | 0.978 |

### Controls

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v4_all | 1.000 | 0.926 | 0.475 / 0.056 | 0.150 |
| v3_s2 | 1.000 | 0.940 | 0.414 / 0.057 | 0.830 |
| v2b | 1.000 | 0.941 | 0.500 / 0.065 | 0.789 |
| v3_s1 | 1.000 | 0.961 | 0.382 / 0.053 | 0.002 |
| v4_short | 1.000 | 0.926 | 0.380 / 0.047 | 0.000 |
| v3_short | 1.000 | 0.899 | 0.426 / 0.041 | 0.000 |

### Tree score and round trip (main = v4_all; threshold 99)

- (a) tree score answered 72.5, abstain=0 7.6; new round-trip pass 1.000; by length {"1-3 words": {"n": 99, "pass_rate": 1.0}, "4-6 words": {"n": 6, "pass_rate": 1.0}, "7+ words": {"n": 3, "pass_rate": 1.0}}; reasons {"node_dropped": 64, "meaning_shift": 58, "operator_folded_differs": 23, "name_mismatch": 9, "node_extra": 6, "head_changed": 3, "relation_changed": 3}
- (b) tree score answered 72.5, abstain=0 7.4; new round-trip pass 0.990; by length {"1-3 words": {"n": 96, "pass_rate": 0.9895833333333334}, "4-6 words": {"n": 5, "pass_rate": 1.0}, "7+ words": {"n": 3, "pass_rate": 1.0}}; reasons {"node_dropped": 61, "meaning_shift": 57, "operator_folded_differs": 23, "name_mismatch": 8, "node_extra": 6, "head_changed": 3, "relation_changed": 3}
- cheat tree score 14.2

### Needs-new-words items

- v4_all: (a) order-only decoder: 0.136 (n=223)
- v4_all: (b) meaning-only decoder: 0.135 (n=223)
- v4_all: true word order (no decoder): 0.139 (n=223)
- v3_s2: (a) order-only decoder: 0.133 (n=223)
- v3_s2: (b) meaning-only decoder: 0.132 (n=223)
- v3_s2: true word order (no decoder): 0.142 (n=223)
- v2b: (a) order-only decoder: 0.145 (n=223)
- v2b: (b) meaning-only decoder: 0.145 (n=223)
- v2b: true word order (no decoder): 0.156 (n=223)
- v3_s1: (a) order-only decoder: 0.133 (n=223)
- v3_s1: (b) meaning-only decoder: 0.133 (n=223)
- v3_s1: true word order (no decoder): 0.136 (n=223)
