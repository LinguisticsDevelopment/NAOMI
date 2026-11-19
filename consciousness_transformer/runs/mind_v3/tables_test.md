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
| v3_s1: (a) order-only decoder | 0.339 | 0.365 | 0.929 |
| v3_s1: (b) meaning-only decoder | 0.334 | 0.360 | 0.929 |
| v3_s1: true word order (no decoder) | 0.367 | 0.395 | 0.929 |
| v3_s2: (a) order-only decoder | 0.345 | 0.374 | 0.924 |
| v3_s2: (b) meaning-only decoder | 0.338 | 0.366 | 0.924 |
| v3_s2: true word order (no decoder) | 0.369 | 0.399 | 0.924 |
| v3_noctx: (a) order-only decoder | 0.342 | 0.362 | 0.946 |
| v3_noctx: (b) meaning-only decoder | 0.334 | 0.353 | 0.946 |
| v3_noctx: true word order (no decoder) | 0.362 | 0.383 | 0.946 |
| v3_noinv: (a) order-only decoder | 0.341 | 0.355 | 0.962 |
| v3_noinv: (b) meaning-only decoder | 0.332 | 0.345 | 0.962 |
| v3_noinv: true word order (no decoder) | 0.363 | 0.377 | 0.962 |
| v2b: (a) order-only decoder | 0.348 | 0.374 | 0.932 |
| v2b: (b) meaning-only decoder | 0.342 | 0.367 | 0.932 |
| v2b: true word order (no decoder) | 0.372 | 0.399 | 0.932 |
| s1_all: (a) order-only decoder | 0.278 | 0.294 | 0.946 |
| s1_all: (b) meaning-only decoder | 0.271 | 0.287 | 0.946 |
| s1_all: true word order (no decoder) | 0.295 | 0.312 | 0.946 |

### By answer-sentence length band (decoder (a))

| band | n | cheat | v3_s2 | v2b | v3_s1 | v3_noctx | v3_noinv | s1_all | oracle |
|---|---|---|---|---|---|---|---|---|---|
| <=10 | 35 | 0.104 | 0.228 | 0.168 | 0.177 | 0.146 | 0.154 | 0.174 | 0.815 |
| 11-20 | 202 | 0.205 | 0.405 | 0.410 | 0.406 | 0.399 | 0.409 | 0.334 | 0.802 |
| 21+ | 688 | 0.251 | 0.362 | 0.372 | 0.361 | 0.364 | 0.361 | 0.293 | 0.792 |
| none | 82 | 0.045 | 0.103 | 0.072 | 0.059 | 0.107 | 0.089 | 0.059 | 0.331 |

### Short-to-long (decoder (a))

| pair | band | n | all lengths | <=10 only |
|---|---|---|---|---|
| v3_short vs v3_s1 | <=10 | 35 | 0.177 | 0.343 |
| v3_short vs v3_s1 | 11-20 | 202 | 0.406 | 0.176 |
| v3_short vs v3_s1 | 21+ | 688 | 0.361 | 0.110 |
| v3_short vs v3_s1 | none | 82 | 0.059 | 0.022 |
| s1_short vs s1_all | <=10 | 35 | 0.174 | 0.217 |
| s1_short vs s1_all | 11-20 | 202 | 0.334 | 0.139 |
| s1_short vs s1_all | 21+ | 688 | 0.293 | 0.082 |
| s1_short vs s1_all | none | 82 | 0.059 | 0.012 |
| v3_short_inv2 vs v3_s1 | <=10 | 35 | 0.177 | 0.290 |
| v3_short_inv2 vs v3_s1 | 11-20 | 202 | 0.406 | 0.181 |
| v3_short_inv2 vs v3_s1 | 21+ | 688 | 0.361 | 0.107 |
| v3_short_inv2 vs v3_s1 | none | 82 | 0.059 | 0.027 |
| v3_short_noinv vs v3_s1 | <=10 | 35 | 0.177 | 0.135 |
| v3_short_noinv vs v3_s1 | 11-20 | 202 | 0.406 | 0.196 |
| v3_short_noinv vs v3_s1 | 21+ | 688 | 0.361 | 0.119 |
| v3_short_noinv vs v3_s1 | none | 82 | 0.059 | 0.027 |

### Paired differences (decoder (a); mean, 95% interval over items, over stories)

- v3_s2 - v2b: -0.003 (items -0.021 to +0.014; stories -0.023 to +0.016)
- v3_s2 - cheat: +0.125 (items +0.105 to +0.146; stories +0.107 to +0.149)
- v3_s2 - v3_s1: +0.006 (items -0.010 to +0.021; stories -0.013 to +0.023)
- v3_s2 - v3_noctx: +0.003 (items -0.014 to +0.020; stories -0.017 to +0.022)
- v3_s2 - v3_noinv: +0.004 (items -0.014 to +0.023; stories -0.014 to +0.021)
- v3_s2 - s1_all: +0.067 (items +0.049 to +0.087; stories +0.048 to +0.090)

### By FairytaleQA attribute (decoder (a))

| attribute | n | cheat | v3_s2 | v2b | v3_s1 | v3_noctx | v3_noinv | s1_all | oracle | answer rate (main) |
|---|---|---|---|---|---|---|---|---|---|---|
| action | 315 | 0.240 | 0.355 | 0.349 | 0.365 | 0.348 | 0.343 | 0.287 | 0.825 | 0.917 |
| causal relationship | 278 | 0.282 | 0.334 | 0.350 | 0.331 | 0.308 | 0.299 | 0.269 | 0.740 | 0.957 |
| feeling | 106 | 0.053 | 0.234 | 0.269 | 0.241 | 0.248 | 0.262 | 0.199 | 0.462 | 0.764 |
| character | 103 | 0.178 | 0.439 | 0.457 | 0.424 | 0.492 | 0.489 | 0.379 | 0.882 | 0.942 |
| outcome resolution | 78 | 0.285 | 0.404 | 0.328 | 0.373 | 0.384 | 0.345 | 0.319 | 0.813 | 0.987 |
| prediction | 65 | 0.120 | 0.226 | 0.177 | 0.189 | 0.188 | 0.257 | 0.170 | 0.582 | 0.923 |
| setting | 62 | 0.222 | 0.427 | 0.502 | 0.390 | 0.494 | 0.501 | 0.298 | 0.898 | 0.968 |

### Controls

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v3_s1 | 1.000 | 0.968 | 0.406 / 0.054 | 0.002 |
| v3_s2 | 1.000 | 0.950 | 0.411 / 0.055 | 0.855 |
| v3_noctx | 1.000 | 0.955 | 0.455 / 0.054 | 0.003 |
| v3_noinv | 1.000 | 0.940 | 0.504 / 0.061 | 0.011 |
| v2b | 1.000 | 0.946 | 0.483 / 0.061 | 0.836 |
| s1_all | 1.000 | 0.927 | 0.481 / 0.055 | 0.046 |
| v3_short | 1.000 | 0.913 | 0.416 / 0.044 | 0.000 |
| s1_short | 1.000 | 0.896 | 0.383 / 0.029 | 0.032 |
| v3_short_inv2 | 1.000 | 0.878 | 0.468 / 0.051 | 0.000 |
| v3_short_noinv | 1.000 | 0.907 | 0.408 / 0.049 | 0.039 |

### Tree score and round trip (main = v3_s2; threshold 99)

- (a) tree score answered 37.7, abstain=0 34.8; new round-trip pass 0.560; by length {"1-3 words": {"n": 261, "pass_rate": 0.9885057471264368}, "4-6 words": {"n": 127, "pass_rate": 0.7874015748031497}, "7+ words": {"n": 542, "pass_rate": 0.3007380073800738}}; reasons {"node_extra": 4470, "node_dropped": 1356, "meaning_shift": 1189, "operator_folded_differs": 818, "head_changed": 798, "subtree_moved": 625, "name_mismatch": 299, "relation_changed": 156}
- (b) tree score answered 37.6, abstain=0 34.8; new round-trip pass 0.539; by length {"1-3 words": {"n": 261, "pass_rate": 0.9578544061302682}, "4-6 words": {"n": 127, "pass_rate": 0.7322834645669292}, "7+ words": {"n": 542, "pass_rate": 0.2915129151291513}}; reasons {"node_extra": 4464, "node_dropped": 1358, "meaning_shift": 1198, "operator_folded_differs": 820, "head_changed": 795, "subtree_moved": 631, "name_mismatch": 308, "relation_changed": 156}
- cheat tree score 13.6

### Needs-new-words items

- v3_s1: (a) order-only decoder: 0.152 (n=224)
- v3_s1: (b) meaning-only decoder: 0.150 (n=224)
- v3_s1: true word order (no decoder): 0.161 (n=224)
- v3_s2: (a) order-only decoder: 0.167 (n=224)
- v3_s2: (b) meaning-only decoder: 0.165 (n=224)
- v3_s2: true word order (no decoder): 0.173 (n=224)
- v3_noctx: (a) order-only decoder: 0.157 (n=224)
- v3_noctx: (b) meaning-only decoder: 0.156 (n=224)
- v3_noctx: true word order (no decoder): 0.165 (n=224)
- v3_noinv: (a) order-only decoder: 0.137 (n=224)
- v3_noinv: (b) meaning-only decoder: 0.137 (n=224)
- v3_noinv: true word order (no decoder): 0.141 (n=224)
- v2b: (a) order-only decoder: 0.149 (n=224)
- v2b: (b) meaning-only decoder: 0.147 (n=224)
- v2b: true word order (no decoder): 0.157 (n=224)
- s1_all: (a) order-only decoder: 0.131 (n=224)
- s1_all: (b) meaning-only decoder: 0.130 (n=224)
- s1_all: true word order (no decoder): 0.133 (n=224)
