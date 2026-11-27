## val: n=1025 items; bands {'<=10': 38, '11-20': 209, '21+': 708, 'none': 70}; main = v5_all

score = ROUGE-L F1 (max over references) of the decoded output sentence (question with node j replaced by the filler); tree_full = tree score % of that text read back vs the reference tree (subsample, rt_n); references are often short phrases so a full sentence pays a precision cost (reported, not compensated)

### Headline (abstain = 0). Primary: ROUGE-L of the decoded full output sentence; tree score of the read-back sentence on the subsample; secondary: ROUGE-L of the filler alone

| system | answer rate | full-sentence ROUGE (a) | full (b) | tree score % (n) | filler ROUGE (a) | filler (b) | filler true order | question nodes kept |
|---|---|---|---|---|---|---|---|---|
| cheat: best-overlap sentence (as filler) | 1.000 | 0.207 | 0.205 | 13.1 (300) | 0.207 | 0.205 | 0.228 | 0.000 |
| v3_s2: old method (v3 anchor subtree as filler) | 0.936 | 0.358 | 0.348 | 37.3 (284) | 0.358 | 0.348 | 0.384 | 0.000 |
| v5_all: v5 thinker | 0.949 | 0.303 | 0.300 | 35.2 (285) | 0.303 | 0.300 | 0.319 | 0.000 |
| v5_short: v5 thinker | 0.925 | 0.129 | 0.129 | 13.1 (277) | 0.118 | 0.117 | 0.120 | 0.793 |
| v5_all_s1: v5 thinker | 0.949 | 0.248 | 0.244 | 20.1 (285) | 0.303 | 0.300 | 0.319 | 0.612 |
| oracle: aligned reference nodes (as filler) | 0.932 | 0.708 | 0.688 | 62.6 (282) | 0.708 | 0.688 | 0.768 | 0.000 |

Old raw-text cheat ROUGE-L (whole best-overlap sentence text, as v2-v4): 0.228; oracle best full subtree: 0.760

### By answer-sentence length band: full-sentence ROUGE-L (a)

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 38 | 0.158 | 0.368 | 0.323 | 0.266 | 0.268 | 0.731 |
| 11-20 | 209 | 0.206 | 0.414 | 0.325 | 0.156 | 0.261 | 0.792 |
| 21+ | 708 | 0.229 | 0.374 | 0.323 | 0.126 | 0.264 | 0.753 |
| none | 70 | 0.013 | 0.023 | 0.026 | 0.008 | 0.026 | 0.000 |

### By answer-sentence length band: filler ROUGE-L (a)

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 38 | 0.158 | 0.368 | 0.323 | 0.319 | 0.323 | 0.731 |
| 11-20 | 209 | 0.206 | 0.414 | 0.325 | 0.177 | 0.325 | 0.792 |
| 21+ | 708 | 0.229 | 0.374 | 0.323 | 0.100 | 0.323 | 0.753 |
| none | 70 | 0.013 | 0.023 | 0.026 | 0.010 | 0.026 | 0.000 |

### By answer-sentence length band: tree score % of decoded full sentence

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 38 | 10.6 | 31.8 | 35.5 | 27.5 | 14.2 | 66.3 |
| 11-20 | 209 | 15.5 | 44.6 | 40.3 | 12.3 | 21.8 | 64.3 |
| 21+ | 708 | 13.2 | 34.1 | 31.1 | 12.5 | 18.2 | 61.9 |
| none | 70 | 4.8 | 61.8 | 76.6 | 11.3 | 50.2 | nan |

### By answer-sentence length band: average share of question nodes kept in the output

| band | n | cheat | v3_s2 | v5_all | v5_short | v5_all_s1 | oracle |
|---|---|---|---|---|---|---|---|
| <=10 | 38 | 0.000 | 0.000 | 0.000 | 0.840 | 0.694 | 0.000 |
| 11-20 | 209 | 0.000 | 0.000 | 0.000 | 0.787 | 0.628 | 0.000 |
| 21+ | 708 | 0.000 | 0.000 | 0.000 | 0.794 | 0.636 | 0.000 |
| none | 70 | 0.000 | 0.000 | 0.000 | 0.765 | 0.231 | nan |

### Short-to-long (v5_short trained on answer sentences <= 10 words vs v5_all all lengths)

| band | n | full ROUGE all | full ROUGE short | filler ROUGE all | filler ROUGE short | tree % all | tree % short | answer rate all | answer rate short |
|---|---|---|---|---|---|---|---|---|---|
| <=10 | 38 | 0.323 | 0.266 | 0.323 | 0.319 | 35.5 | 27.5 | 1.000 | 1.000 |
| 11-20 | 209 | 0.325 | 0.156 | 0.325 | 0.177 | 40.3 | 12.3 | 0.947 | 0.943 |
| 21+ | 708 | 0.323 | 0.126 | 0.323 | 0.100 | 31.1 | 12.5 | 0.958 | 0.921 |
| none | 70 | 0.026 | 0.008 | 0.026 | 0.010 | 76.6 | 11.3 | 0.843 | 0.871 |

### Paired differences (mean, 95% interval over items, over stories)

- v5_all - cheat [full-sentence ROUGE-L (a)]: +0.096 (items +0.077 to +0.114; stories +0.078 to +0.116)
- v5_all - cheat [tree score % of decoded full sentence]: +21.962 (items +18.361 to +25.738; stories +18.317 to +25.040)
- v5_all - v3_s2 [full-sentence ROUGE-L (a)]: -0.055 (items -0.073 to -0.037; stories -0.076 to -0.032)
- v5_all - v3_s2 [tree score % of decoded full sentence]: -2.388 (items -5.142 to +0.520; stories -5.576 to +0.531)
- v5_all - v5_short [full-sentence ROUGE-L (a)]: +0.174 (items +0.154 to +0.194; stories +0.153 to +0.198)
- v5_all - v5_short [tree score % of decoded full sentence]: +21.922 (items +18.256 to +25.636; stories +18.023 to +25.228)
- v5_all - v5_all_s1 [full-sentence ROUGE-L (a)]: +0.056 (items +0.045 to +0.066; stories +0.045 to +0.067)
- v5_all - v5_all_s1 [tree score % of decoded full sentence]: +15.059 (items +12.595 to +17.773; stories +12.179 to +18.111)
- v5_all - oracle [full-sentence ROUGE-L (a)]: -0.405 (items -0.426 to -0.384; stories -0.427 to -0.386)
- v5_all - oracle [tree score % of decoded full sentence]: -29.436 (items -33.657 to -25.399; stories -32.837 to -26.183)

### Controls

| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / filler ROUGE | memory = question only: abstain |
|---|---|---|---|---|
| v5_all | 1.000 | 0.948 | 0.452 / 0.049 | 0.002 |
| v5_short | 1.000 | 0.877 | 0.491 / 0.043 | 0.001 |
| v5_all_s1 | 1.000 | 0.948 | 0.452 / 0.049 | 0.002 |

### Provenance

- v5_all: {"nodes": 7246, "from_memory": 7246, "from_question": 0, "traced_share": 1.0, "memory_share": 1.0, "mismatch": 0}
- v5_short: {"nodes": 3855, "from_memory": 3855, "from_question": 0, "traced_share": 1.0, "memory_share": 1.0, "mismatch": 0}

### UNKNOWN-discovery diagnostic (never a training signal)

- v5_all: {"answered": 973, "questions_with_a_node_above_zero": 973, "replaced_is_top_unknown": 0.2774922918807811, "chance_if_uniform": 0.1138029750384295, "root_replaced_share": 0.18396711202466598, "rank_by_unknown_of_replaced": {"0": 0.2733812949640288, "1": 0.09969167523124357, "2": 0.0421377183967112, "3": 0.052415210688591986, "4": 0.10996916752312436, "5": 0.4224049331963001}, "n_has": 973}
- v5_short: {"answered": 948, "questions_with_a_node_above_zero": 948, "replaced_is_top_unknown": 0.08227848101265822, "chance_if_uniform": 0.11386582056884334, "root_replaced_share": 0.029535864978902954, "rank_by_unknown_of_replaced": {"0": 0.07805907172995781, "1": 0.02531645569620253, "2": 0.12341772151898735, "3": 0.13291139240506328, "4": 0.0770042194092827, "5": 0.5632911392405063}, "n_has": 948}
- v5_all_s1: {"answered": 973, "questions_with_a_node_above_zero": 973, "replaced_is_top_unknown": 0.19116135662898254, "chance_if_uniform": 0.1138029750384295, "root_replaced_share": 0.24871531346351491, "rank_by_unknown_of_replaced": {"0": 0.19116135662898254, "1": 0.15827338129496402, "2": 0.05549845837615622, "3": 0.0565262076053443, "4": 0.1366906474820144, "5": 0.40184994861253853}, "n_has": 973}

### Round trip (main, filler, threshold 99)

{"threshold": 99, "n": 300, "pass_rate": 0.6633333333333333, "by_len": {"7+ words": {"n": 152, "pass_rate": 0.4276315789473684}, "1-3 words": {"n": 74, "pass_rate": 0.972972972972973}, "4-6 words": {"n": 74, "pass_rate": 0.8378378378378378}}}

### Tree-score reasons by system (decoded full sentence vs reference)

{"cheat": {"node_extra": 5908, "head_changed": 600, "meaning_shift": 431, "subtree_moved": 398, "operator_folded_differs": 365, "node_dropped": 297, "name_mismatch": 207, "relation_changed": 61, "opposite_meaning": 1}, "oracle": {"head_changed": 277, "node_dropped": 237, "node_extra": 209, "subtree_moved": 188, "operator_folded_differs": 165, "meaning_shift": 119, "relation_changed": 47, "name_mismatch": 31}, "v5_all": {"node_extra": 790, "node_dropped": 667, "meaning_shift": 405, "operator_folded_differs": 246, "head_changed": 195, "subtree_moved": 105, "relation_changed": 50, "name_mismatch": 48, "opposite_meaning": 1}, "v5_short": {"node_extra": 1893, "node_dropped": 728, "meaning_shift": 526, "operator_folded_differs": 349, "head_changed": 310, "subtree_moved": 195, "name_mismatch": 156, "relation_changed": 53}, "v5_all_s1": {"node_extra": 2099, "node_dropped": 608, "meaning_shift": 437, "head_changed": 414, "operator_folded_differs": 337, "subtree_moved": 230, "name_mismatch": 135, "relation_changed": 55}, "v3_s2": {"node_extra": 1344, "node_dropped": 505, "meaning_shift": 415, "head_changed": 301, "operator_folded_differs": 263, "subtree_moved": 234, "name_mismatch": 80, "relation_changed": 53, "opposite_meaning": 1}}
