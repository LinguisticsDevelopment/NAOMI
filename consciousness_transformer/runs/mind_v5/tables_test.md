## test: n=1007 items; bands {'<=10': 35, '11-20': 202, '21+': 688, 'none': 82}; main = v5_all

score = ROUGE-L F1 (max over references) of the decoded output sentence (question with node j replaced by the filler); tree_full = tree score % of that text read back vs the reference tree (subsample, rt_n); references are often short phrases so a full sentence pays a precision cost (reported, not compensated)

### Headline (abstain = 0). Primary: ROUGE-L of the decoded full output sentence; tree score of the read-back sentence on the subsample; secondary: ROUGE-L of the filler alone

| system | answer rate | full-sentence ROUGE (a) | full (b) | tree score % (n) | filler ROUGE (a) | filler (b) | filler true order | question nodes kept |
|---|---|---|---|---|---|---|---|---|
| cheat: best-overlap sentence (as filler) | 1.000 | 0.200 | 0.198 | 11.2 (300) | 0.201 | 0.199 | 0.220 | 0.000 |
| v3_s2: old method (v3 anchor subtree as filler) | 0.924 | 0.346 | 0.337 | 38.3 (272) | 0.345 | 0.338 | 0.369 | 0.000 |
| v5_all: v5 thinker | 0.927 | 0.298 | 0.291 | 33.6 (280) | 0.296 | 0.290 | 0.316 | 0.000 |
| v5_short: v5 thinker | 0.925 | 0.118 | 0.118 | 12.0 (272) | 0.111 | 0.109 | 0.114 | 0.787 |
| v5_all_s1: v5 thinker | 0.927 | 0.245 | 0.241 | 22.5 (280) | 0.296 | 0.290 | 0.316 | 0.588 |
| oracle: aligned reference nodes (as filler) | 0.919 | 0.687 | 0.665 | 62.7 (277) | 0.689 | 0.668 | 0.752 | 0.000 |

Old raw-text cheat ROUGE-L (whole best-overlap sentence text, as v2-v4): 0.220; oracle best full subtree: 0.757

### By answer-sentence length band: full-sentence ROUGE-L (a)

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 35 | 0.099 | 0.230 | 0.163 | 0.191 | 0.148 | 0.752 |
| 11-20 | 202 | 0.195 | 0.408 | 0.369 | 0.145 | 0.268 | 0.759 |
| 21+ | 688 | 0.226 | 0.362 | 0.313 | 0.118 | 0.267 | 0.745 |
| none | 82 | 0.041 | 0.104 | 0.055 | 0.024 | 0.048 | 0.000 |

### By answer-sentence length band: filler ROUGE-L (a)

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 35 | 0.098 | 0.228 | 0.156 | 0.286 | 0.156 | 0.747 |
| 11-20 | 202 | 0.196 | 0.405 | 0.366 | 0.185 | 0.366 | 0.761 |
| 21+ | 688 | 0.227 | 0.362 | 0.311 | 0.091 | 0.311 | 0.747 |
| none | 82 | 0.041 | 0.103 | 0.055 | 0.025 | 0.055 | 0.000 |

### By answer-sentence length band: tree score % of decoded full sentence

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 35 | 11.1 | 36.4 | 28.5 | 14.1 | 13.0 | 71.1 |
| 11-20 | 202 | 9.6 | 44.6 | 41.0 | 12.4 | 25.2 | 64.8 |
| 21+ | 688 | 12.6 | 34.9 | 28.9 | 11.9 | 19.7 | 61.5 |
| none | 82 | 3.7 | 59.7 | 60.0 | 11.0 | 47.7 | nan |

### By answer-sentence length band: average share of question nodes kept in the output

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 35 | 0.000 | 0.000 | 0.000 | 0.768 | 0.686 | 0.000 |
| 11-20 | 202 | 0.000 | 0.000 | 0.000 | 0.788 | 0.595 | 0.000 |
| 21+ | 688 | 0.000 | 0.000 | 0.000 | 0.781 | 0.612 | 0.000 |
| none | 82 | 0.000 | 0.000 | 0.000 | 0.852 | 0.297 | nan |

### Short-to-long (v5_short trained on answer sentences <= 10 words vs v5_all all lengths)

| band | n | full ROUGE all | full ROUGE short | filler ROUGE all | filler ROUGE short | tree % all | tree % short | answer rate all | answer rate short |
|---|---|---|---|---|---|---|---|---|---|
| <=10 | 35 | 0.163 | 0.191 | 0.156 | 0.286 | 28.5 | 14.1 | 0.914 | 0.943 |
| 11-20 | 202 | 0.369 | 0.145 | 0.366 | 0.185 | 41.0 | 12.4 | 0.936 | 0.980 |
| 21+ | 688 | 0.313 | 0.118 | 0.311 | 0.091 | 28.9 | 11.9 | 0.938 | 0.916 |
| none | 82 | 0.055 | 0.024 | 0.055 | 0.025 | 60.0 | 11.0 | 0.817 | 0.854 |

### Paired differences (mean, 95% interval over items, over stories)

- v5_all - cheat [full-sentence ROUGE-L (a)]: +0.097 (items +0.079 to +0.117; stories +0.078 to +0.120)
- v5_all - cheat [tree score % of decoded full sentence]: +22.267 (items +18.775 to +25.867; stories +18.990 to +26.208)
- v5_all - v3_s2 [full-sentence ROUGE-L (a)]: -0.048 (items -0.065 to -0.031; stories -0.064 to -0.031)
- v5_all - v3_s2 [tree score % of decoded full sentence]: -5.198 (items -8.340 to -2.118; stories -7.538 to -2.782)
- v5_all - v5_short [full-sentence ROUGE-L (a)]: +0.179 (items +0.160 to +0.200; stories +0.158 to +0.203)
- v5_all - v5_short [tree score % of decoded full sentence]: +21.076 (items +17.516 to +24.639; stories +17.827 to +24.619)
- v5_all - v5_all_s1 [full-sentence ROUGE-L (a)]: +0.053 (items +0.043 to +0.063; stories +0.043 to +0.065)
- v5_all - v5_all_s1 [tree score % of decoded full sentence]: +11.102 (items +8.912 to +13.522; stories +8.870 to +13.387)
- v5_all - oracle [full-sentence ROUGE-L (a)]: -0.390 (items -0.412 to -0.366; stories -0.408 to -0.369)
- v5_all - oracle [tree score % of decoded full sentence]: -30.813 (items -35.081 to -26.429; stories -35.637 to -25.641)

### Controls

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / filler ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v5_all | 1.000 | 0.958 | 0.413 / 0.043 | 0.006 |
| v5_short | 1.000 | 0.874 | 0.500 / 0.042 | 0.000 |
| v5_all_s1 | 1.000 | 0.958 | 0.413 / 0.043 | 0.006 |

### Provenance

- v5_all: {"nodes": 7146, "from_memory": 7146, "from_question": 0, "traced_share": 1.0, "memory_share": 1.0, "mismatch": 0}
- v5_short: {"nodes": 3539, "from_memory": 3539, "from_question": 0, "traced_share": 1.0, "memory_share": 1.0, "mismatch": 0}

### UNKNOWN-discovery diagnostic (never a training signal)

- v5_all: {"answered": 933, "questions_with_a_node_above_zero": 933, "replaced_is_top_unknown": 0.2390139335476956, "chance_if_uniform": 0.11809239944472343, "root_replaced_share": 0.18756698821007503, "rank_by_unknown_of_replaced": {"0": 0.2357984994640943, "1": 0.12540192926045016, "2": 0.05037513397642015, "3": 0.04608788853161844, "4": 0.10718113612004287, "5": 0.43515541264737406}, "n_has": 933}
- v5_short: {"answered": 931, "questions_with_a_node_above_zero": 931, "replaced_is_top_unknown": 0.07841031149301826, "chance_if_uniform": 0.11744365166550105, "root_replaced_share": 0.03329752953813104, "rank_by_unknown_of_replaced": {"0": 0.06337271750805586, "1": 0.04189044038668099, "2": 0.11170784103114931, "3": 0.13533834586466165, "4": 0.08163265306122448, "5": 0.5660580021482277}, "n_has": 931}
- v5_all_s1: {"answered": 933, "questions_with_a_node_above_zero": 933, "replaced_is_top_unknown": 0.19935691318327975, "chance_if_uniform": 0.11809239944472343, "root_replaced_share": 0.2808145766345123, "rank_by_unknown_of_replaced": {"0": 0.1907824222936763, "1": 0.18435155412647375, "2": 0.05037513397642015, "3": 0.04715969989281887, "4": 0.1157556270096463, "5": 0.4115755627009646}, "n_has": 933}

### Round trip (main, filler, threshold 99)

{"threshold": 99, "n": 300, "pass_rate": 0.6266666666666667, "by_len": {"1-3 words": {"n": 75, "pass_rate": 0.9733333333333334}, "7+ words": {"n": 157, "pass_rate": 0.37579617834394907}, "4-6 words": {"n": 68, "pass_rate": 0.8235294117647058}}}

### Tree-score reasons by system (decoded full sentence vs reference)

{"cheat": {"node_extra": 6197, "head_changed": 527, "meaning_shift": 385, "subtree_moved": 366, "operator_folded_differs": 341, "name_mismatch": 231, "node_dropped": 221, "relation_changed": 45}, "oracle": {"head_changed": 231, "node_extra": 210, "node_dropped": 210, "operator_folded_differs": 200, "subtree_moved": 175, "meaning_shift": 124, "relation_changed": 45, "name_mismatch": 43}, "v5_all": {"node_extra": 943, "node_dropped": 563, "meaning_shift": 378, "operator_folded_differs": 255, "head_changed": 179, "subtree_moved": 132, "name_mismatch": 75, "relation_changed": 45}, "v5_short": {"node_extra": 1858, "node_dropped": 630, "meaning_shift": 462, "operator_folded_differs": 286, "head_changed": 248, "subtree_moved": 160, "name_mismatch": 159, "relation_changed": 46}, "v5_all_s1": {"node_extra": 2132, "node_dropped": 427, "meaning_shift": 417, "head_changed": 317, "operator_folded_differs": 309, "subtree_moved": 219, "name_mismatch": 140, "relation_changed": 49}, "v3_s2": {"node_extra": 1223, "node_dropped": 390, "meaning_shift": 377, "operator_folded_differs": 249, "head_changed": 221, "subtree_moved": 170, "name_mismatch": 85, "relation_changed": 57}}
