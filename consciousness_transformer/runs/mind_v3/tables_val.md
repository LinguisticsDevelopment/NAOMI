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
| v3_s1: (a) order-only decoder | 0.340 | 0.365 | 0.932 |
| v3_s1: (b) meaning-only decoder | 0.334 | 0.359 | 0.932 |
| v3_s1: true word order (no decoder) | 0.365 | 0.392 | 0.932 |
| v3_s2: (a) order-only decoder | 0.358 | 0.382 | 0.936 |
| v3_s2: (b) meaning-only decoder | 0.348 | 0.372 | 0.936 |
| v3_s2: true word order (no decoder) | 0.384 | 0.410 | 0.936 |
| v3_noctx: (a) order-only decoder | 0.330 | 0.348 | 0.947 |
| v3_noctx: (b) meaning-only decoder | 0.322 | 0.340 | 0.947 |
| v3_noctx: true word order (no decoder) | 0.350 | 0.369 | 0.947 |
| v3_noinv: (a) order-only decoder | 0.351 | 0.364 | 0.965 |
| v3_noinv: (b) meaning-only decoder | 0.343 | 0.355 | 0.965 |
| v3_noinv: true word order (no decoder) | 0.374 | 0.388 | 0.965 |
| v2b: (a) order-only decoder | 0.353 | 0.373 | 0.946 |
| v2b: (b) meaning-only decoder | 0.346 | 0.365 | 0.946 |
| v2b: true word order (no decoder) | 0.373 | 0.395 | 0.946 |
| s1_all: (a) order-only decoder | 0.334 | 0.348 | 0.962 |
| s1_all: (b) meaning-only decoder | 0.330 | 0.343 | 0.962 |
| s1_all: true word order (no decoder) | 0.353 | 0.367 | 0.962 |

### By answer-sentence length band (decoder (a))

| band | n | cheat | v3_s2 | v2b | v3_s1 | v3_noctx | v3_noinv | s1_all | oracle |
|---|---|---|---|---|---|---|---|---|---|
| <=10 | 38 | 0.164 | 0.368 | 0.296 | 0.258 | 0.242 | 0.299 | 0.314 | 0.754 |
| 11-20 | 209 | 0.211 | 0.414 | 0.394 | 0.377 | 0.357 | 0.376 | 0.374 | 0.802 |
| 21+ | 708 | 0.258 | 0.374 | 0.377 | 0.363 | 0.355 | 0.376 | 0.353 | 0.801 |
| none | 70 | 0.013 | 0.023 | 0.017 | 0.045 | 0.035 | 0.055 | 0.041 | 0.220 |

### Short-to-long (decoder (a))

| pair | band | n | all lengths | <=10 only |
|---|---|---|---|---|
| v3_short vs v3_s1 | <=10 | 38 | 0.258 | 0.386 |
| v3_short vs v3_s1 | 11-20 | 209 | 0.377 | 0.193 |
| v3_short vs v3_s1 | 21+ | 708 | 0.363 | 0.111 |
| v3_short vs v3_s1 | none | 70 | 0.045 | 0.017 |
| s1_short vs s1_all | <=10 | 38 | 0.314 | 0.364 |
| s1_short vs s1_all | 11-20 | 209 | 0.374 | 0.201 |
| s1_short vs s1_all | 21+ | 708 | 0.353 | 0.093 |
| s1_short vs s1_all | none | 70 | 0.041 | 0.011 |

### Paired differences (decoder (a); mean, 95% interval over items, over stories)

- v3_s2 - v2b: +0.005 (items -0.014 to +0.023; stories -0.011 to +0.027)
- v3_s2 - cheat: +0.130 (items +0.110 to +0.150; stories +0.109 to +0.157)
- v3_s2 - v3_s1: +0.018 (items +0.003 to +0.032; stories +0.000 to +0.034)
- v3_s2 - v3_noctx: +0.028 (items +0.011 to +0.045; stories +0.008 to +0.051)
- v3_s2 - v3_noinv: +0.007 (items -0.012 to +0.024; stories -0.015 to +0.031)
- v3_s2 - s1_all: +0.024 (items +0.005 to +0.041; stories +0.001 to +0.048)

### By FairytaleQA attribute (decoder (a))

| attribute | n | cheat | v3_s2 | v2b | v3_s1 | v3_noctx | v3_noinv | s1_all | oracle | answer rate (main) |
|---|---|---|---|---|---|---|---|---|---|---|
| action | 333 | 0.266 | 0.383 | 0.383 | 0.364 | 0.352 | 0.379 | 0.365 | 0.826 | 0.964 |
| causal relationship | 294 | 0.264 | 0.323 | 0.313 | 0.311 | 0.305 | 0.298 | 0.277 | 0.738 | 0.952 |
| character | 107 | 0.171 | 0.453 | 0.436 | 0.402 | 0.400 | 0.468 | 0.444 | 0.849 | 0.972 |
| outcome resolution | 97 | 0.312 | 0.433 | 0.375 | 0.390 | 0.343 | 0.399 | 0.378 | 0.850 | 0.959 |
| feeling | 94 | 0.034 | 0.234 | 0.229 | 0.248 | 0.233 | 0.257 | 0.267 | 0.411 | 0.755 |
| prediction | 55 | 0.132 | 0.207 | 0.242 | 0.191 | 0.194 | 0.241 | 0.169 | 0.625 | 0.873 |
| setting | 45 | 0.189 | 0.452 | 0.539 | 0.475 | 0.496 | 0.443 | 0.470 | 0.900 | 0.933 |

### Controls

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v3_s1 | 1.000 | 0.961 | 0.382 / 0.053 | 0.002 |
| v3_s2 | 1.000 | 0.940 | 0.414 / 0.057 | 0.830 |
| v3_noctx | 1.000 | 0.951 | 0.454 / 0.054 | 0.006 |
| v3_noinv | 1.000 | 0.934 | 0.511 / 0.063 | 0.007 |
| v2b | 1.000 | 0.941 | 0.500 / 0.065 | 0.789 |
| s1_all | 1.000 | 0.927 | 0.536 / 0.060 | 0.018 |
| v3_short | 1.000 | 0.899 | 0.426 / 0.041 | 0.000 |
| s1_short | 1.000 | 0.854 | 0.480 / 0.053 | 0.164 |

### Tree score and round trip (main = v3_s2; threshold 99)

- (a) tree score answered 37.6, abstain=0 35.2; new round-trip pass 0.592; by length {"1-3 words": {"n": 219, "pass_rate": 0.9908675799086758}, "4-6 words": {"n": 154, "pass_rate": 0.8896103896103896}, "7+ words": {"n": 586, "pass_rate": 0.3651877133105802}}; reasons {"node_extra": 4931, "node_dropped": 1672, "meaning_shift": 1424, "head_changed": 934, "operator_folded_differs": 928, "subtree_moved": 747, "name_mismatch": 241, "relation_changed": 188, "opposite_meaning": 2}
- (b) tree score answered 37.5, abstain=0 35.1; new round-trip pass 0.569; by length {"1-3 words": {"n": 219, "pass_rate": 0.958904109589041}, "4-6 words": {"n": 154, "pass_rate": 0.8636363636363636}, "7+ words": {"n": 586, "pass_rate": 0.3464163822525597}}; reasons {"node_extra": 4929, "node_dropped": 1672, "meaning_shift": 1424, "head_changed": 936, "operator_folded_differs": 927, "subtree_moved": 744, "name_mismatch": 246, "relation_changed": 186, "opposite_meaning": 2}
- cheat tree score 14.2

### Needs-new-words items

- v3_s1: (a) order-only decoder: 0.133 (n=223)
- v3_s1: (b) meaning-only decoder: 0.133 (n=223)
- v3_s1: true word order (no decoder): 0.136 (n=223)
- v3_s2: (a) order-only decoder: 0.133 (n=223)
- v3_s2: (b) meaning-only decoder: 0.132 (n=223)
- v3_s2: true word order (no decoder): 0.142 (n=223)
- v3_noctx: (a) order-only decoder: 0.132 (n=223)
- v3_noctx: (b) meaning-only decoder: 0.133 (n=223)
- v3_noctx: true word order (no decoder): 0.136 (n=223)
- v3_noinv: (a) order-only decoder: 0.148 (n=223)
- v3_noinv: (b) meaning-only decoder: 0.147 (n=223)
- v3_noinv: true word order (no decoder): 0.151 (n=223)
- v2b: (a) order-only decoder: 0.145 (n=223)
- v2b: (b) meaning-only decoder: 0.145 (n=223)
- v2b: true word order (no decoder): 0.156 (n=223)
- s1_all: (a) order-only decoder: 0.143 (n=223)
- s1_all: (b) meaning-only decoder: 0.143 (n=223)
- s1_all: true word order (no decoder): 0.151 (n=223)
