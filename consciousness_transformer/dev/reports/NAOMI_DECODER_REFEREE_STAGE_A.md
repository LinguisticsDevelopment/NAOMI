# NAOMI decoder v1 + referee, stage A (2026-11-16)

Method: `dev/design/ENCODER_DECODER_PAIR_PROPOSAL.md` (approved). This report covers the DECODER v1 (word order from a tree), the REFEREE,
whether the referee separates good added links from bad ones, and the first round-trip numbers. Not stage B itself.
Encoder = `naomi.encoder.NaomiEncoder` (`runs/naomi_encoder/main_best.pt`); same source split (`runs/naomi_encoder/splits.json`).

## What each part is (method check)
- Decoder v1 = ORDERING ONLY: given a tree (tag, structural classes, USVS meaning vector, name flag, relation to head per node; the nodes carry their words) it learns the linear order.
  Given: the tree and the words. Learned: precedence among a head and its children (pairwise "a before b"). Signal: original relative order in gold v7 (the encoder's source split, TRAIN only). No position is an input: nodes are put in a seeded shuffle, information passing is pooling, so child order cannot leak.
- Referee = three checks, none of them learned: grammar-as-data licence, attestation in parser gold, agreement with the parser; plus a round trip through decoder and encoder.

## Files
`src/naomi/orderer.py`, `src/naomi/referee.py`, `scripts/train_naomi_orderer.py` (train + eval), `scripts/eval_naomi_roundtrip.py` (attested table / part 3 / part 4), `tests/test_naomi_orderer.py`, `tests/test_naomi_referee.py`.

## PRE-DECLARED for Part 3 (written before any Part 3 number was computed)
Small, single-rater, hand-judged set (173 encoder-added links on 40 sentences, my own earlier judgments). Nothing is tuned on it. Fixed settings:
- (a) licensed by the grammar-as-data table. (b) attested in TRAIN gold at least N times, N chosen on DEV by a rule stated before the run: the largest N from {1,2,3,5,10,20,30,40,50,100} that still licenses >= 95% of the true dev parser links (dev links vs. corrupted dev links also reported).
- Confidence thresholds are quantiles of the encoder's top-1 probability `p` and log-prob margin `m` over ALL added links of the 1,098 TEST-source fragmented sentences (judgment-free): q50 and q75.
- Round-trip (RT) = per-sentence `score_pair` overall of encoder tree -> decoder -> encoder, threshold fixed at 80.
- Settings reported: a; b; a&b; b&p>=q50; b&p>=q75; b&m>=q50; a&b&p>=q50; p>=q50; p>=q75; b&RT>=80; b&p>=q50&RT>=80; RT>=80. Sentence-level: accepted iff ALL its judged links pass the setting.
- Success statement asked for: wrong rate among accepted links <= 15% while accepting >= 25% of links. "Wrong rate" = wrong / (links passing); wrong+debatable also given.
- Links whose dependent is the encoder's own root carry no link finding and are excluded from the counts (n shown).

(Results sections follow, written as each stage finishes.)

## 1. Decoder v1 (word order from a tree)

Model (`naomi.orderer`): per node the encoder's own features (POS, 23 structural subtypes, 614-axis USVS vector = mean of top-3 candidate senses, name flag) + relation embedding -> 2-3 rounds of pooling along tree edges (self / mean of children / parent) -> antisymmetric pairwise scorer "a before b" over {head} + children -> items sorted by wins (ties: total score, then token string) -> recursion, each subtree contiguous. No word identity (as the encoder default). Position cannot reach the model (seeded node shuffle; permutation-invariant pooling; tested).
Data: gold v7 TRAIN sources: 26,447 sentences -> every internally connected piece of >= 2 tokens (fully connected sentences are one piece): 65,914 training pieces (463k nodes, 640k precedence pairs); loss = BCE on all within-head precedence pairs. Same source split as the encoder. Runs: `main` (d=128, 2 rounds, 0.30M params; 9.5 min, early stop ep 30, dev heads-right .928) and `large` (d=192, 3 rounds, 0.58M params; 13.8 min, early stop ep 32, dev heads-right .932). `large_best.pt` is THE decoder used everywhere below. Total training wall-clock about 23 min (CPU, 4 threads, nice); well under budget, so no more was spent.

**Ceiling of the formulation.** Share of trees whose true order is reproducible when every subtree must be contiguous: 100% on parser trees (test connected 393/393, test pieces 3,709/3,709, frozen holdout 30/30). That is a property of the parser (its rules are adjacency patterns, so its trees are projective by construction), and a check that feeding the true local orders reproduces every sentence exactly (1.0). The ceiling is NOT 100% for ENCODER trees: 98.3% of encoder trees on parser-connected test sentences, 82.5% on fragmented ones (97% <=10, 88% 11-20, 78% 21-35, 59% 36+) keep every subtree contiguous in the original order. The rest (the encoder attached across another subtree) cannot be re-ordered to the original by this formulation, and that is a check on the tree, not on the decoder.

Results, TEST-source fully connected sentences (393; one-token sentences are not trees), decoder `large` vs the cheat lookup baseline (per (relation, dependent tag, head tag): majority side of the head + mean distance rank, backed off; no model):

| set | model | exact order | heads fully right | pairwise acc | Kendall dist. |
|---|---|---|---|---|---|
| test connected (393) | decoder | **0.718** | 0.923 | 0.938 | 0.076 |
| | cheat baseline | 0.430 | 0.828 | 0.875 | 0.167 |
| test pieces (3,709 incl. fragments' pieces) | decoder | 0.801 | 0.933 | 0.945 | 0.055 |
| | cheat baseline | 0.583 | 0.836 | 0.874 | 0.134 |
| frozen holdout connected (30) | decoder | 0.500 | 0.887 | 0.928 | 0.125 |
| | cheat baseline | 0.200 | 0.824 | 0.879 | 0.207 |
| frozen holdout-dev connected (15) | decoder | 0.600 | 0.940 | 0.942 | 0.152 |
| | cheat baseline | 0.400 | 0.890 | 0.904 | 0.174 |
| train pieces (2,000 sample) | decoder | 0.862 | 0.959 | 0.975 | 0.027 |

By length, test connected (exact / heads / Kendall): <=10 (290): decoder .78 / .93 / .06, baseline .55 / .83 / .14; 11-20 (83): .46 / .91 / .14 vs .11 / .82 / .24; 21-35 (20): .15 / .88 / .15 vs .05 / .83 / .23. (`main` for comparison: exact .674, heads .912.)
Reading: per-head order is about 92% right; exact whole-sentence order falls fast with length because errors compound (about 8% of heads wrong; at 11-20 words only 46% of sentences have every head right). The decoder is far ahead of the lookup (exact +29 points) but is not close to deterministic: word identity is not in the features (adverb and adjunct placement, coordination order), and the train/test gap (.86 vs .72) says some memorisation. Files: `runs/naomi_orderer/eval_orderer_large.json`, `eval_orderer_main.json`.

## 2. Referee (`naomi.referee`, JSON in/out)

`referee(tree, sentence, parser_tree=None, ctx=None)` -> per-link findings (licensed by grammar + reason, attested count + reason), parser agreement (share kept + list of changed links with reason), round trip (decoded text, order distance, score meaning/shape/overall with and without restored final punctuation) and one `accept` with `reasons`. Default accept rule: every link attested >= N, parser agreement >= 0.9 if a parser tree is given, round-trip overall >= 80 if run (these defaults are NOT evaluated in this report; Part 3 evaluates the signals separately).

- (i-a) grammar as data: `derive_order_table` -> 187 (relation, head cat, dep cat, side) entries; tags -> categories through the parser's own tag map; a link is licensed if some entry matches relation, categories (with its unit-lift closure) and side. Rules with subtype conditions are skipped by `derive_order_table`, so (a) is conservative.
- (i-b) attested: 397,519 committed TRAIN links -> 2,596 distinct configurations (head tag + subtypes, relation, dep tag + subtypes, side) with counts, `runs/naomi_orderer/attested_configs.json`. N chosen on DEV by the declared rule (largest N keeping >= 95% of true dev links): **N = 40** (dev true links pass 95.8%; 16.9% of corrupted dev links (random other head or random other relation) pass; for N=1 the figures are 99.9% / 31.7%).
- Agreement of (a) and (b), on 21,109 DEV parser links: licensed by (a) 58.3%, by (b) 95.8%, both 57.6%, (a) only 0.7%, (b) only 38.2%, neither 3.6%. On the corrupted copies: (a) 6.4%, (b) 16.9%, both 4.9%, (a) only 1.5%, (b) only 11.9%, neither 81.7%. So (a) is almost a subset of (b) and is more selective (rejects 94% of corrupted links) but also rejects 42% of links the parser itself commits (subtype-conditioned rules, reversed edges, categories the grammar reaches through lifts). (b) is a permissive, data-driven licence: it can only say "the parser has produced this kind of link", not that this link is right.
- (ii) parser agreement and (iii) round trip as specified. Note on (iii): it uses the encoder twice (once to produce the proposal, once to read the decoded text), so it cannot detect an error the encoder would repeat on its own decoded text, nor can it separate a decoder error from an encoder error; it can detect trees whose words cannot be ordered to something the encoder reads back the same way, and it is blind to wrong links the decoder places contiguously and the encoder re-reads the same way. It also loses information because the tree has no punctuation: the decoder outputs bare words and the encoder uses punctuation as context. The control row in the Part 4 table (the TRUE word order, same treatment) is therefore the upper bound for any round-trip number here.

## 3. Does the referee separate good added links from bad ones? (168 hand-judged links; 5 of the 173 are encoder roots and carry no link)

Overall mix: 54 right / 56 debatable / 58 wrong (wrong 34.5%, wrong+debatable 67.9%). Thresholds (judgment-free, from the 4,500 added links of the 1,098 test fragmented sentences): p q50 = 0.886, q75 = 0.997; margin q50 = 3.60, q75 = 6.87. N = 40. Round-trip variant used for the per-sentence score: final punctuation restored (decided before Part 3 ran).

| signal (declared) | links pass | share | r / d / w among pass | wrong % pass | wrong % fail | sentences accepted (all links pass) | wrong % in them (n links) |
|---|---|---|---|---|---|---|---|
| a (grammar) | 45 | 27% | 14 / 12 / 19 | 42% | 32% | 5/40 | 40% (5) |
| b (attested >= 40) | 113 | 67% | 43 / 36 / 34 | 30% | 44% | 10/40 | 25% (20) |
| a & b | 41 | 24% | 13 / 10 / 18 | 44% | 32% | 3/40 | 67% (3) |
| p >= q50 | 78 | 46% | 38 / 22 / 18 | 23% | 44% | 7/40 | 10% (10) |
| p >= q75 | 37 | 22% | 20 / 10 / 7 | 19% | 39% | 5/40 | 0% (6) |
| b & p >= q50 | 59 | 35% | 29 / 17 / 13 | 22% | 41% | 4/40 | 20% (5) |
| b & p >= q75 | 30 | 18% | 16 / 8 / 6 | 20% | 38% | 3/40 | 0% (4) |
| b & margin >= q50 | 64 | 38% | 27 / 19 / 18 | 28% | 39% | 5/40 | 33% (6) |
| a & b & p >= q50 | 29 | 17% | 11 / 7 / 11 | 38% | 34% | 2/40 | 50% (2) |
| rt >= 80 | 4 | 2% | 1 / 2 / 1 | 25% | 35% | 4/40 | 25% (4) |
| b & rt >= 80 | 2 | 1% | 1 / 0 / 1 | 50% | 34% | 2/40 | 50% (2) |
| b & p >= q50 & rt >= 80 | 2 | 1% | 1 / 0 / 1 | 50% | 34% | 2/40 | 50% (2) |

Descriptive quartiles (not thresholds), 42 links each, right / debatable / wrong: confidence p: Q1 (<=0.21) 5/16/21, Q2 (<=0.85) 10/16/16, Q3 (<=0.99) 18/12/12, Q4 (>0.99) 21/12/9. Margin: 10/16/16, 11/15/16, 16/13/13, 17/12/13. Round-trip score: 7/15/20, 13/20/14, 19/10/12, 15/11/12. Attested count: 7/15/20, 14/14/14, 18/14/10, 15/13/14.

**Plain statement.** No signal or combination reaches <= 15% wrong among accepted links while accepting >= 25% of links. Closest: the encoder's own confidence. p >= q75 gives 18.9% wrong at 22% of links (just short on both), b & p >= q75 20% at 18%. The referee's own checks help little: the grammar licence (a) is not better than chance on these links (42% wrong among passing, 32% among failing: it passes the links on common easy relations, e.g. SUBJECT/SPECIFICATION, but the wrong ones are exactly the plausible-looking ones), attestation (b) is weakly informative (30% vs 44%) and passes two thirds of everything; the round trip says nothing here (only 4 links sit in sentences scoring >= 80, because even a true-order text of a fragmented sentence returns about 63 after losing its internal punctuation, so the declared threshold is uninformative; the quartiles show a weak monotone trend, 20 wrong in Q1 vs 12 in Q4). Sentence-level all-links-pass acceptance accepts 2-10 of 40 sentences with 10-25 links, so counts are too small to read (the p >= q50 rule: 7 sentences, 10 links, 10% wrong, is the only decent-looking row and it is 10 links). What is missing: all three referee checks test whether a link is TYPICAL (configuration seen in the parser's output) or whether the text reads back the same, but not whether the link is the right one among the typical alternatives; the wrong links here are mostly coordination/modification/object attachments of the right category to the wrong clause (see the encoder report). A discriminating check needs the competing candidate heads (that is what the encoder's margin partially encodes) or evidence outside the parser's own habits. Small hand-judged set, single rater (the implementer), 168 links with wide error bars (a 10-point difference in a 30-60 link cell is not significant); nothing was tuned on it beyond the declared settings.
Files: `runs/naomi_orderer/part3_referee_signals.json` (all links with signals), `runs/naomi_orderer_part3.log`.

## 4. First round-trip numbers (TEST-source sentences; scores vs the START tree: meaning / shape / overall)

Text' = decoded words, then re-encoded. "tail" = the original sentence's own trailing punctuation tokens appended (the order-only decoder places none). Control = same words in the TRUE order, same two variants (the ceiling given that punctuation is not in the tree).

| start tree -> decoder -> encoder | n | subtrees contiguous | exact word order | Kendall dist. | decoded, no punct. | decoded + final punct. | control no punct. | control + final punct. |
|---|---|---|---|---|---|---|---|---|
| (1) parser tree, parser-connected | 413 | 100% | 0.690 | 0.080 | 71.7 / 74.0 / 72.8 | 77.5 / 79.4 / **78.5** | 73.4 / 75.7 / 74.5 | 79.7 / 81.3 / 80.5 |
| (2) encoder tree, same sentences | 413 | 98.3% | 0.659 | 0.090 | 72.6 / 74.7 / 73.6 | 81.0 / 82.6 / **81.8** | 74.2 / 76.5 / 75.4 | 84.8 / 86.2 / 85.5 |
| (3) encoder tree, fragmented | 1,098 | 82.5% | 0.136 | 0.189 | 46.0 / 49.7 / 47.9 | 49.8 / 52.9 / **51.4** | 55.0 / 58.1 / 56.6 | 62.4 / 64.5 / 63.4 |

By length band (exact order / overall decoded+final punct. / overall control+final punct.): (1) <=10 (310) .787 / 83.5 / 84.7; 11-20 (83) .458 / 66.6 / 69.7; 21-35 (20) .150 / 49.5 / 60.7. (2) <=10 .748 / 87.1 / 89.2; 11-20 .446 / 68.9 / 76.1; 21-35 .150 / 53.7 / 66.7. (3) <=10 (250) .388 / 56.7 / 69.5; 11-20 (333) .141 / 52.8 / 62.7; 21-35 (353) .014 / 49.3 / 61.8; 36+ (162) .000 / 44.5 / 58.8. Contiguity of the start tree in (3): 97% / 88% / 78% / 59% by band.
Reading: the decoder costs 2-4 points against the control on connected sentences (78.5 vs 80.5; 81.8 vs 85.5) and 12 points on fragmented ones (51.4 vs 63.4), where 17.5% of the encoder's trees are not reproducible by contiguous blocks at all and the trees are often not what an order model can make sense of. The control itself is far below 100: the encoder is not stable under punctuation removal (it gave 88.8 on the original text vs 76.7 without any punctuation in a 20-sentence probe). These are the baseline numbers stage B has to raise. File: `runs/naomi_orderer/eval_roundtrip.json` (+ `roundtrip_per_sentence.jsonl`).

## Tests
`tests/test_naomi_orderer.py` (5: permutation + every subtree contiguous; position removed -- scrambled indices and reversed children give identical output; ceiling test detects a non-projective tree; 40-tree overfit exact > 0.95 and lookup baseline > 0.9; Kendall) and `tests/test_naomi_referee.py` (4: attested vs unattested, grammar licence, accept/reject + JSON-serialisable, parser agreement lists changes) plus the no-literals test with the allow-list extended by non-linguistic identifiers (dict keys, side tags read from the grammar table, artifact paths): 16 passed in 21 s together with the encoder tests.
