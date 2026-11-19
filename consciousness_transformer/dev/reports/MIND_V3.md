# Mind v3: sentence context, length-invariant shape, no length curriculum

Status 2026-11-19. Test evaluated once, after the main model was fixed on val. Everything below is reproducible from the files listed at the end.

## What was built

Spec: `dev/reports/SPAN_DIAGNOSIS.md`, method otherwise as `dev/design/CONSCIOUSNESS_V2_PROPOSAL.md` (the thinker reads memory trees, picks an anchor node, keeps/prunes its subtree, abstains; the orderer decodes the kept nodes; nothing is rule-based).

Code: `src/naomi/mind_v3.py`, `scripts/{train,eval,tables}_mind_v3.py`, `tests/test_mind_v3.py`; diagnosis scripts `dev/reports/span_diagnosis_scripts/s07..s09`.

1. Two-step choice: log p(anchor) = log p(sentence) + log p(node | sentence).
2. Cross-sentence context: sentence position, signed offsets to the top question-matching sentences, two rounds of attention over the sentence sequence (relative-offset bias).
3. Length-invariant shape: depth relative to the sentence's deepest node, size as a share (inv=1) or as a share of the parent (inv=2), clause depth read off tree shape, mean pooling, no node count to the sentence scorer.
4. Abstain read from the chosen scores; no length curriculum.

Variants (stage 1 = supervised choice; stage 2 = reward fine-tune with wrong-passage abstain; all trained once, seed 0):

| name | what |
|---|---|
| v3_s1 | all lengths, full model, stage 1 |
| v3_s2 | all lengths, full model, stage 2. **Main model, declared on val** (best val ROUGE of the full variants; not changed after test) |
| v3_noctx | ablation: sentences scored alone (ctx=0) |
| v3_noinv | ablation: v2 raw depth/size/max+count pooling (inv=0) |
| v3_short | trained only on items whose answer sentence is <=10 nodes (inv=1), stage 1 |
| v3_short_inv2 | same, inv=2 (size = share of the parent's subtree). Added late to test whether the sentence-share size feature leaks length |
| v3_short_noinv | same, inv=0. Added late as the no-fix control for the short-to-long failure |
| v2b | previous best (`runs/mind_v2b/thinker_s2b_all_best.pt`), same items, same decoder |
| s1_all, s1_short | v2 stage-1 all-lengths / short-only |

Round-trip pass uses the threshold declared on val controls (`runs/mind_v2/roundtrip_val.json`, A=99). Decoder (a) = order-only decoder, (b) = meaning-only, "true order" = oracle word order. Headline numbers are (a).

## Test results (n=1007 items, 23 stories; single run)

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


## Gap decomposition (true-order ROUGE, items where the loss falls)

| split | model | ROUGE | oracle | abstained: items / loss | wrong sentence: items / loss | right sentence, wrong anchor: items / loss | right: items |
|---|---|---|---|---|---|---|---|
| test | v2b | 0.372 | 0.757 | 6.8% / 10.4% | 34.4% / 58.4% | 29.6% / 33.6% | 27.6% |
| test | v3_s2 | 0.369 | 0.757 | 7.6% / 12.3% | 32.3% / 53.7% | 32.5% / 36.6% | 26.0% |
| val | v2b | 0.373 | 0.760 | 5.4% / 7.0% | 38.2% / 63.9% | 29.2% / 31.3% | 25.9% |
| val | v3_s2 | 0.384 | 0.760 | 6.4% / 9.0% | 34.5% / 58.3% | 31.1% / 34.8% | 26.9% |

Top-1 sentence hit (oracle>=0.5 items, oracle sentence = argmax sentence mass): test v2b 60.6%, v3_s2 62.3%; val v2b 57.1%, v3_s1 60.1%, v3_s2 59.7%, v3_noctx 57.1%. Oracle sentence in top-2 / top-4 (test): v2b 75.1 / 90.9, v3_s2 79.0 / 91.6. Best acceptable anchor ranked first among all nodes: v2b 25.4%, v3_s2 24.7%. Right-sentence errors (test, v3_s2): model anchor is an ancestor/parent/child/other clause of the oracle anchor; text is too broad (recall > precision) in 56-57% of them, same as v2b.
Files: `dev/reports/span_diagnosis_scripts/out/decomp_{test,val}_v3_s2.txt`, `s02_{test,val}_s2b_all.txt`, `s07_v3.txt`.

## Why short-to-long still fails (val, diagnosis only)

Test confirms the failure: v3_short is 0.343 on <=10 but 0.176 on 11-20 and 0.110 on 21+ (all-length model 0.406 / 0.361); both late variants (inv2: 0.107 on 21+, noinv: 0.119) fail the same way. Length-invariant features did not buy transfer. Evidence from val (`s07`, `s08`, `s09` outputs in `dev/reports/span_diagnosis_scripts/out/`):

**Both steps fail, the sentence step more.** Sentence top-1 hit by band, short vs all-length model: 11-20 27.9% vs 50.5%; 21+ 15.6% vs 64.8%. With the oracle sentence forced (anchor step alone, s08): 21+ acceptable-anchor hit 13.0% (v3_short), 15.6% (inv2), 11.2% (noinv) vs 32.2% (v3_s1). So the anchor step is also broken, at roughly 40% of the all-length level; it is not just the sentence prior.

**1. The sentence score still correlates with length, inversely.** Within-item Spearman between sentence mass and sentence node count: v3_short -0.52, v3_short_noinv -0.42, s1_short -0.62, against +0.42 (v3_s1) and +0.50 (s1_all). The short model picks sentences averaging 18.3 nodes on 21+ items whose gold sentences average 36.0 (v3_s1: 36.9 vs 36.0). Training on <=10 gives a prior "prefer short sentences" which the features cannot cancel because the data never contained a long positive.

**2. The shape features still carry length.** The `inv=1` size share is size/N, and its sentence mean is almost a function of N: Spearman(N, mean size/N) = -0.91 over 42,767 val sentences (mean 0.41 for 2-10 nodes, 0.23 for 11-20, 0.16 for 21+), while mean depth/max-depth is flat (-0.10). Mean pooling hands the sentence scorer that number. The code comment already notes a leaf reads 1/N. inv=2 removes this but did not help (0.107 on 21+), so this leak is real but not the main cause.

**3. The target itself changes with length, so what the anchor scorer learned is length-specific.** Oracle anchor geometry from val (decomp): in <=10 sentences the anchor is the sentence head 79% of the time (depth 0.85, size share 0.62); in 11-20 it is 51% (depth 1.87, share 0.45); in 21+ only 28% (depth 3.34, share 0.32), because long sentences hold several clauses and the answer is a deeper clause. The short model with the oracle sentence forced picks depth 2.59 vs oracle 3.34 on 21+ (v3_s1: also 2.59, but hits 32% vs 13%) and size share 0.54 vs oracle 0.32; the noinv short model picks depth 0.93 and share 0.80 (shallow, large: it picks the sentence head). Relative features make the *coordinates* comparable but the short training set holds no example of "deep, small anchor in a multi-clause sentence", so there is nothing to generalise from.

**4. Inputs that still differ by length** (not removable by normalisation): number of competing nodes and sentences, multi-clause structure (clause depth), and which relation kinds appear in the answer span. Position and context features are length-independent but the short subset is selected by answer length, which probably correlates with attribute (untested here: feeling/character answers look short, causal/action long), so the short model may also have learned a question-type prior that does not transfer.

Table, val, forced-sentence and sentence-choice detail:

```
test s2b_all: oracle>=0.5 items 886; oracle sentence in top-1/2/4 sentences: 60.6% / 75.1% / 90.9%; answered anchor in oracle sentence 61.3%; wrong sentence offset to nearest oracle sentence {'+1': 92, '-1': 47, 'other': 186}
test v3_s2: oracle>=0.5 items 886; oracle sentence in top-1/2/4 sentences: 62.3% / 79.0% / 91.6%; answered anchor in oracle sentence 63.1%; wrong sentence offset to nearest oracle sentence {'+1': 87, '-1': 56, 'other': 164}
===VAL
val s2b_all: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 57.1% / 73.7% / 88.0%; answered anchor in oracle sentence 57.8%; wrong sentence offset to nearest oracle sentence {'+1': 97, '-1': 60, 'other': 219}
val v3_s1: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 60.1% / 77.3% / 91.2%; answered anchor in oracle sentence 62.6%; wrong sentence offset to nearest oracle sentence {'+1': 80, '-1': 58, 'other': 187}
val v3_s2: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 59.7% / 76.8% / 90.3%; answered anchor in oracle sentence 61.3%; wrong sentence offset to nearest oracle sentence {'+1': 96, '-1': 58, 'other': 188}
val v3_noctx: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 57.1% / 72.9% / 88.5%; answered anchor in oracle sentence 58.7%; wrong sentence offset to nearest oracle sentence {'+1': 91, '-1': 52, 'other': 220}
val v3_short: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 19.8% / 35.6% / 63.5%; answered anchor in oracle sentence 18.8%; wrong sentence offset to nearest oracle sentence {'+1': 120, '-1': 164, 'other': 385}
val v3_short_inv2: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 19.0% / 36.5% / 63.7%; answered anchor in oracle sentence 18.3%; wrong sentence offset to nearest oracle sentence {'+1': 132, '-1': 156, 'other': 424}
val v3_short_noinv: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 21.8% / 37.9% / 64.9%; answered anchor in oracle sentence 21.3%; wrong sentence offset to nearest oracle sentence {'+1': 110, '-1': 111, 'other': 431}
val s1_short: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 17.8% / 33.1% / 57.6%; answered anchor in oracle sentence 14.6%; wrong sentence offset to nearest oracle sentence {'+1': 111, '-1': 98, 'other': 480}
val s1_all: oracle>=0.5 items 925; oracle sentence in top-1/2/4 sentences: 57.3% / 72.6% / 88.2%; answered anchor in oracle sentence 56.3%; wrong sentence offset to nearest oracle sentence {'+1': 102, '-1': 52, 'other': 236}


== val v3_short  within-item Spearman(sentence mass, sentence length) mean -0.517 (n=850)
| band | n | sentence top-1 hit | chosen sentence nodes | gold sentence nodes | forced-sentence: model depth / oracle depth | model size share / oracle size share |
| <=10 | 33 | 57.6% | 8.5 | 11.1 | 0.58 / 0.85 | 0.76 / 0.62 |
| 11-20 | 204 | 27.9% | 12.7 | 18.3 | 1.25 / 1.87 | 0.70 / 0.45 |
| 21+ | 671 | 15.6% | 18.3 | 36.0 | 2.59 / 3.34 | 0.54 / 0.32 |

== val v3_s1  within-item Spearman(sentence mass, sentence length) mean +0.421 (n=850)
| band | n | sentence top-1 hit | chosen sentence nodes | gold sentence nodes | forced-sentence: model depth / oracle depth | model size share / oracle size share |
| <=10 | 33 | 33.3% | 19.2 | 11.1 | 0.58 / 0.85 | 0.81 / 0.62 |
| 11-20 | 204 | 50.5% | 22.5 | 18.3 | 1.32 / 1.87 | 0.65 / 0.45 |
| 21+ | 671 | 64.8% | 36.9 | 36.0 | 2.59 / 3.34 | 0.57 / 0.32 |

== val v3_short_noinv  within-item Spearman(sentence mass, sentence length) mean -0.422 (n=850)
| band | n | sentence top-1 hit | chosen sentence nodes | gold sentence nodes | forced-sentence: model depth / oracle depth | model size share / oracle size share |
| <=10 | 33 | 18.2% | 14.5 | 11.1 | 0.33 / 0.85 | 0.86 / 0.62 |
| 11-20 | 204 | 33.8% | 14.7 | 18.3 | 0.62 / 1.87 | 0.83 / 0.45 |
| 21+ | 671 | 18.8% | 19.5 | 36.0 | 0.93 / 3.34 | 0.80 / 0.32 |

== val s1_short  within-item Spearman(sentence mass, sentence length) mean -0.623 (n=850)
| band | n | sentence top-1 hit | chosen sentence nodes | gold sentence nodes | forced-sentence: model depth / oracle depth | model size share / oracle size share |
| <=10 | 33 | 54.5% | 8.8 | 11.1 | 0.64 / 0.85 | 0.80 / 0.62 |
| 11-20 | 204 | 27.5% | 11.7 | 18.3 | 1.64 / 1.87 | 0.59 / 0.45 |
| 21+ | 671 | 13.4% | 16.7 | 36.0 | 3.27 / 3.34 | 0.44 / 0.32 |

== val s1_all  within-item Spearman(sentence mass, sentence length) mean +0.504 (n=850)
| band | n | sentence top-1 hit | chosen sentence nodes | gold sentence nodes | forced-sentence: model depth / oracle depth | model size share / oracle size share |
| <=10 | 33 | 33.3% | 23.5 | 11.1 | 0.45 / 0.85 | 0.82 / 0.62 |
| 11-20 | 204 | 44.6% | 23.6 | 18.3 | 1.36 / 1.87 | 0.64 / 0.45 |
| 21+ | 671 | 62.7% | 37.3 | 36.0 | 2.57 / 3.34 | 0.57 / 0.32 |
```

Anchor step alone, oracle sentence forced (s08, val; top-1 acceptable anchor / node ROUGE):

```
| model | <=10 top1 acceptable / node ROUGE (n) | 11-20 | 21+ |
| s1_short | 36.4% / 0.609 (33) | 30.9% / 0.495 (204) | 13.0% / 0.355 (671) |
| s1_all | 51.5% / 0.667 (33) | 46.6% / 0.625 (204) | 29.2% / 0.534 (671) |
| v3_short | 48.5% / 0.649 (33) | 32.4% / 0.510 (204) | 13.0% / 0.361 (671) |
| v3_short_inv2 | 48.5% / 0.616 (33) | 28.9% / 0.502 (204) | 15.6% / 0.367 (671) |
| v3_short_noinv | 36.4% / 0.577 (33) | 24.5% / 0.481 (204) | 11.2% / 0.385 (671) |
| v3_s1 | 51.5% / 0.672 (33) | 49.5% / 0.615 (204) | 32.2% / 0.531 (671) |
| v3_noinv | 54.5% / 0.661 (33) | 46.6% / 0.617 (204) | 30.6% / 0.531 (671) |
```

Feature leak probe (s09_feature_leak):

```
sentences 42767 Spearman(N, mean size/N) = -0.907 ; Spearman(N, mean depth/maxdepth) = -0.099
2 10 11042 mean(size/N) 0.410  mean(depth/maxd) 0.535
11 20 15007 mean(size/N) 0.233  mean(depth/maxd) 0.510
21 500 16718 mean(size/N) 0.162  mean(depth/maxd) 0.506
```

## Hand-read: 20 test answers (seeded; `runs/mind_v3/eval_handread_20.json`)

10 low-ROUGE answered (of 193 with ROUGE <= 0.05) and 10 abstained (of 77).

Low answers: 4 are feeling questions answered with an emotion word that is not in the evidence (amused vs "excited", very anxious vs "happy", "annoyed" vs thankful, "a love" vs happy). Two are right-sentence fragment errors, i.e. the model selected a sub-span of the right clause that is not the answer (`seek counsel...` question answered `he might escape death but no one could help him`, the next clause of the same sentence; `the foot of some high mountains` answered `a little way off two roads`, the next clause). Two are wrong entity/sentence (`Ryn Jin` answered `the distressed man`; children names answered `a a son`). One is a pruned fragment (`much shorter than`) and one is an anchor too deep (`peasant had two sons named john and`, answer was `the youngest son`).

Abstains: 7 of 10 have an oracle subtree that is a good answer (golden buckets, a song, bread, walked to the window, gave it up and lifting his arm..., the saw-the-name clause, terrified). In ft9887 the cheat sentence (best-overlap) contains the entire reference, yet the model abstained with p=1.00. Abstain probabilities are saturated (1.00 in 9 of 10), so the abstain head is confident, not calibrated. The two feeling abstains (`sad`, `terrified`) have a bare adjective as the answer, with little overlap with the question terms.

## Conclusion

Did sentence context close the sentence-choice gap? Not materially. Top-1 sentence hit goes from 57.1% to 59.7% on val and 60.6% to 62.3% on test, and the wrong-sentence share of the loss falls from 63.9% to 58.3% (val) and 58.4% to 53.7% (test), but ROUGE is flat against v2b: val +0.005 (items -0.014 to +0.023; stories -0.011 to +0.027), test -0.003 (items -0.021 to +0.014; stories -0.023 to +0.016). The ablations on val show a small context effect (v3_s2 minus noctx +0.028, interval excludes zero; v3_s2 minus v3_s1 +0.018), but on test the same contrasts are +0.003 and +0.006 with intervals spanning zero. So context may help a little; it does not show in test. Roughly one third of items still pick the wrong sentence, and wrong sentence still carries more than half of the loss.

Did short-to-long transfer? No: 0.343 on <=10, 0.110 on 21+ (all-length model 0.361), with or without length-invariant features, with any of the three size definitions. All our models beat the cheat baseline (+0.125 test, interval excludes zero) and the v2 stage-1 all-lengths model (+0.067).

Round trip: test 0.560 pass (a), 0.539 (b); 99% of 1-3 word answers pass, 30% of 7+ word answers.

## Three limiting factors (evidence)

1. Sentence choice is still wrong in about 38% of items (oracle sentence top-1 62%; top-4 92%). The correct sentence is nearly always in the top 4, so the ranker cannot separate near-ties, mostly neighbouring or same-passage sentences (wrong-sentence offsets on val: +1/-1 account for 154 of 342 for v3_s2, other offsets 188). The training signal is the final answer, which does not say why a sentence is right.
2. Inside the right sentence the anchor is wrong in about 32% of items, and the miss is structural: ancestor/parent/other clause (decomp table), too broad in 56-57%. The best acceptable anchor is ranked first only 25% of the time among all nodes. The anchor scorer does not know what the question asks for (feeling, cause, time) at the level of clause structure.
3. Training distribution per length: the answer geometry differs by sentence length (head 79% to 28%), the length-restricted model never sees the long cases, and shared features leak length (size/N mean, rho -0.91). The all-length model does not suffer, so the data, not the capacity, limits transfer. Related: abstain is saturated and uncalibrated (p=1.00 in 9 of 10 hand-read abstains, with reachable answers).

## What the evidence suggests next

The part to work on is the span chooser inside a sentence (step 2), made question-conditioned at the clause level rather than the node level: clause segmentation is given by the trees, and the anchor decision should be (which clause, then which node in it) driven by what the question's wh/relation node asks for. In the decomposition, right-sentence errors are 30-33% of items and 34-37% of loss, comparable to the sentence step, and are not reduced by context. Sentence choice needs a different signal rather than more context: a teaching signal about WHY a sentence matches (grounded overlap of question and sentence nodes), not only the final ROUGE of the chosen answer. For length transfer the model needs long positives; a length curriculum (short then long) or mixed training is the only thing the evidence supports, since normalisation did not do it. The abstain head needs calibration against the oracle (reachable vs not) before it is trusted.

## Tests

`python -m pytest tests/test_mind_v2.py tests/test_mind_v3.py tests/test_roundtrip.py tests/test_naomi_no_literals.py tests/test_naomi_memory.py -q`: 22 passed, 1 warning (`runs/mind_v3/pytest_final.log`). (The bare `pytest` on this box uses an interpreter without numpy; use `python -m pytest`.)

## Files

Checkpoints `runs/mind_v3/thinker_v3_{s1,s2,noctx,noinv,short,short_inv2,short_noinv}_best.pt`, curves `.curve.tsv`, logs `v3_*.log`, `eval_{val,test}.json`, `eval_rows_{val,test}.jsonl`, `tables_{val,test}.md`, `eval_handread_20.json`, `eval_test.log`, `pytest_final.log`.

## Val tables (for reference; main model v3_s2)

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
