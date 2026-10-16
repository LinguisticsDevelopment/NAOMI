# M62b — Teacher Gold-Volume Probe (in-repo real corpus)

Numbers only, no interpretation added beyond what the measurement shows.

- Corpus files (unique sentences after dedup): real_gutenberg_alice.txt=407, real_gutenberg_bryant.txt=410, real_gutenberg_burgess_more.txt=256, real_gutenberg_busterbear.txt=89, real_gutenberg_edgeworth.txt=313
- Total unique sentences available: 1475
- Available per length-bin (A<=8, B 9-15, C 16-25, D 26+): A=227, B=348, C=358, D=542
- Sampled per bin (seed=42, cap=500 total): A=125, B=125, C=125, D=125 = 500 total
- Gold definition: `_parse_graph_one` (default ParserConfig, 30s cap) produces a non-None discourse graph AND `extract_discourse` finds >=1 clause with a real (non-punctuation) SUBJECT.

## Per-bin results

| bin | n | full-tree yield | cap-hit rate | median s | p90 s |
|---|---|---|---|---|---|
| A | 125 | 63/125 = 50.4% | 0/125 = 0.0% | 0.005 | 0.024 |
| B | 125 | 113/125 = 90.4% | 0/125 = 0.0% | 0.078 | 0.135 |
| C | 125 | 116/125 = 92.8% | 0/125 = 0.0% | 0.226 | 0.554 |
| D | 125 | 114/125 = 91.2% | 9/125 = 7.2% | 1.181 | 18.290 |
| **overall** | 500 | 406/500 = 81.2% | 9/500 = 1.8% | 0.127 | 1.469 |

## Yield trend A->D

A:50.4% -> B:90.4% -> C:92.8% -> D:91.2%

## Failure modes (non-full-tree rows only)

94 non-full-tree rows out of 500 total.

| outcome | count | share of failures |
|---|---|---|
| grounding-fail | 84 | 89.4% |
| cap-hit | 9 | 9.6% |
| too-long | 1 | 1.1% |

### grounding-fail — examples

- [A, 2tok, real_gutenberg_alice.txt] `'oh !` (clauses=0)
- [A, 3tok, real_gutenberg_bryant.txt] `more ! "` (clauses=0)
- [A, 3tok, real_gutenberg_alice.txt] `oh dear !` (clauses=0)

### cap-hit — examples

- [D, 68tok, real_gutenberg_alice.txt] `she drew her foot as far down the chimney as she could , and waited till she heard a little animal ( she could n't guess of what sort it was ) scratching and scrambling about in the chimney close above her : then , saying to herself 'this is bill , ' she gave one sharp kick , and waited to see what would happen next .` (parse exceeded 30.0s wall-clock cap (mid-_apply_all, ruleset 'predicate1'))
- [D, 54tok, real_gutenberg_burgess_more.txt] `he may be a bully , because great big people are very apt to be bullies , and though i have n't seen him , i guess buster bear is big enough from all i have heard , but i do n't see how he is a thief , " said grandfather frog .` (parse exceeded 30.0s wall-clock cap (mid-_apply_all, ruleset 'predicate1'))
- [D, 94tok, real_gutenberg_edgeworth.txt] `when he was questioned by gilbert why he did not bring an answer , he did not attempt to make any excuse ; he did not say , " there was no answer , please your honour , " or , " they bid me not to wait , " etc . ; but he told exactly the truth ; and though gilbert scolded him for being so impatient as not to wait , yet his telling the truth was more to the boy 's advantage than any excuse he could have made .` (parse exceeded 30.0s wall-clock cap (mid-_apply_all, ruleset 'predicate1'))

### too-long — examples

- [D, 127tok, real_gutenberg_edgeworth.txt] `mr . harvey , the gentleman on whose estate she lived , was in england , and , in his absence , all was managed by a mr . hopkins , an agent , who was a hard man . * the driver came to mary about a week after her mother 's death , and told her that the rent must be brought in the next day , and that she must leave the cabin , for a new tenant was coming into it ; that she was too young to have a house to herself , and that the only thing she had to do was to get some neighbour to take her and her brother and her sisters in for charity 's sake .` (graph parse failed (Sentence too long (127 > 100)); no discourse structure.)

