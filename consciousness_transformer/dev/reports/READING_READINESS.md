# Reading readiness: can the encoders referee themselves, and how good is reading on simple text (2026-11-17)

Measurement only, no training. Scripts and result JSONs: `dev/reports/reading_readiness_scripts/` (`part1_*.py`, `part2_*.py`, `*_results.json`, blind files). Models: stage A (`naomi_encoder/main_best.pt`), A2 (`naomi_encoder_a2/a2_best.pt`), control (`a2ctrl_best.pt`).

## Settings declared BEFORE computing (Part 1)
Per judged link; "head probability" = the model's arc softmax probability of the link's head; models "choose" a head if it is the head in their decoded single-root tree.
- S1: both stage A and A2 propose the link (same dependent, head, relation).
- S2: A, A2 and control all choose this head.
- S3: S2 and the minimum head probability over the three models >= 0.8.
- S4: S2 and minimum probability >= 0.9.
- S5: S3 and the link's configuration is attested in train gold (referee source (b), count >= 40, the referee's own min_count; ROOT links count as not attested).
- Sentence rule: "accept a sentence if every link the A2 model added in it passes S_k" (A2 links = the up-to-10 per sentence sampled in the A2 hand-check).
- Target: <= 15% wrong among accepted links while accepting >= 25% of links. No tuning beyond these five.

## Part 1: can the encoders referee themselves? (270 judged links, 40 sentences)
Judgments are the existing blind ones (`handcheck_blind_judgments.json`): single rater, rater self-consistency 69%, a handful of links moves any rate by several points. Overall: 87 right / 76 debatable / 107 wrong (39.6% wrong). Margins (top-1 minus top-2 probability) are stored per link and model in `part1_links.json` but no setting uses them (not declared).

| setting | passing (share of 270) | passing r/d/w (wrong%) | failing r/d/w (wrong%) | sentences accepted (of 40) | A2 links in accepted sentences r/d/w (wrong%) |
|---|---|---|---|---|---|
| S1 | 76 (28.1%) | 38/23/15 (19.7%) | 49/53/92 (47.4%) | 8 | 12: 8/4/0 (0%) |
| S2 | 106 (39.3%) | 43/34/29 (27.4%) | 44/42/78 (47.6%) | 11 | 19: 12/6/1 (5.3%) |
| S3 | 40 (14.8%) | 22/12/6 (15.0%) | 65/64/101 (43.9%) | 7 | 9: 5/3/1 (11.1%) |
| S4 | 29 (10.7%) | 18/8/3 (10.3%) | 69/68/104 (43.2%) | 6 | 7: 5/2/0 (0%) |
| S5 | 28 (10.4%) | 16/9/3 (10.7%) | 71/67/104 (43.0%) | 5 | 7: 3/3/1 (14.3%) |

By length band, passing links (right/debatable/wrong, of all links in the band): 
- S1: <=10 8 (5/3/0) of 20; 11-20 7 (3/4/0) of 35; 21-35 30 (16/7/7) of 78; 36+ 31 (14/9/8) of 137.
- S2: 12 (7/5/0); 12 (6/5/1); 36 (17/8/11); 46 (13/16/17).
- S3: 9 (6/3/0); 4 (2/1/1); 14 (8/3/3); 13 (6/5/2).
- S4: 7 (5/2/0); 4 (2/1/1); 6 (5/1/0); 12 (6/4/2).
- S5: 4 (3/1/0); 2 (1/1/0); 9 (6/2/1); 13 (6/5/2).
Accepted sentences by band (<=10 / 11-20 / 21-35 / 36+): S1 5/2/1/0; S2 6/3/2/0; S3 5/1/1/0; S4 5/1/0/0; S5 3/1/1/0. Of 40 sentences, 10 per band: no 36+ sentence is accepted under any setting.
Context: attested (count >= 40) alone: 174 links, 36.2% wrong; A2 head probability >= 0.8 alone (A2-proposed links): 81 links, 17.3% wrong.

**Plain statement on 15%:** no setting reaches <= 15% wrong at >= 25% of links accepted. S1 accepts 28.1% but is 19.7% wrong; S2 accepts 39.3% at 27.4% wrong. The settings under 15% wrong (S3 15.0% = at the line, S4 10.3%, S5 10.7%) accept 10-15% of links, and S3 (15.0%) is already exactly at the limit. All five miss the combined target. The per-sentence rule gives 5-11 accepted sentences of 40 (mostly <= 10 words), 0-5% wrong inside them for S1/S2/S4 but on 7-19 links only.

## Part 2: reading the intended teaching text (not used in training)
Sets: (i) first 400 distinct context sentences of MCTest mc160 (children's stories, none of it in gold v7); (ii) the 399 distinct questions in the file (the file has 399, not 400; questions keep their original capitalisation); (iii) 400 sentences drawn with seed 0 from the distinct context sentences of FairyTaleQA test-split episodes (gold v7 contains FairyTaleQA TRAIN stories only). Length = non-punctuation tokens. Reader: `Reader.read_parser`; A2 then fills the unknown tokens with the parser's committed links forced into the decode, so every sentence ends as one tree (`part2_read.py`). For an added link S1-S5 are computed as in Part 1, with each of the three models decoding under the same forced parser links.

| | stories (i) | questions (ii) | fairytale test (iii) |
|---|---|---|---|
| sentences (assessed; trivial <2 words excluded) | 398 | 399 | 398 |
| length mean / median / max | 10.4 / 10 / 29 | 8.1 / 7 / 30 | 20.3 / 19 / 79 |
| band <=10 / 11-20 / 21-35 / 36+ | 227 / 156 / 17 / 0 | 323 / 75 / 1 / 0 | 96 / 129 / 130 / 45 |
| parser failed (no tree) | 0 | 0 | 22 (all long) |
| parser connects fully | 237 (59.5%) | 210 (52.6%) | 111 (27.9%) |
| fragmented or failed sentences | 161 | 189 | 287 |
| links A2 must add (in those) | 299 (1.9 per sentence) | 270 (1.4) | 1,991 (6.9; 1,031 of them in the 22 failed sentences) |
| added links passing S3 | 67 (22%) | 49 (18%) | 949 (48%); parsed-but-fragmented sentences only: 193 of 960 (20%) |
| added links passing S1 / S2 / S4 / S5 | 135 / 177 / 59 / 53 | 139 / 191 / 39 / 33 | 1,371 / 1,375 / 870 / 874 |

"Trusted tree" = one tree in which every link is parser-committed or passes S3 (sentences the parser failed on are not counted as trusted). Share of sentences, trusted / sentences in band (S3):

| band | stories (i) | questions (ii) | fairytale (iii) |
|---|---|---|---|
| <=10 | 177 / 225 (79%) | 197 / 323 (61%) | 65 / 94 (69%) |
| 11-20 | 76 / 156 (49%) | 31 / 75 (41%) | 45 / 129 (35%) |
| 21-35 | 2 / 17 (12%) | 0 / 1 | 12 / 130 (9%) |
| 36+ | none | none | 0 / 45 |
| all | 255 / 398 (64%) | 228 / 399 (57%) | 122 / 398 (31%) |
Of these, fully parser-connected (no encoder link needed): stories 237, questions 210, fairytale 111; so the encoder-assisted gain is 18 / 18 / 11 sentences. Under S4 / S5 the all-band trusted counts are 252 / 249 (stories), 223 / 222 (questions), 118 / 120 (fairytale).

### Blind hand-judgement of the FINAL tree (parser links + encoder links)
60 sentences (40 stories, 20 questions) drawn from sentences the parser did not fail on, seeded, stratified by length (<=6, 7-10, 11-15, 16+ words; 9/17/18/16 sentences obtained), 654 links. Provenance (parser vs encoder) removed, links and sentences shuffled (`part2_blind.json`), judged right/debatable/wrong by one rater (me, same rubric and conventions as the A2 hand-check: e.g. auxiliary `did` heads the question, infinitive `to` heads the verb), then joined with `part2_blind_key.json` (`part2_blind_result.json`). Single rater, quick judgments.

| | links | right | debatable | wrong |
|---|---|---|---|---|
| stories, all links | 420 | 280 (67%) | 62 (15%) | 78 (19%) |
| stories, parser links | 383 | 267 (70%) | 49 (13%) | 67 (17%) |
| stories, encoder-added | 37 | 13 (35%) | 13 (35%) | 11 (30%) |
| questions, all links | 234 | 154 (66%) | 52 (22%) | 28 (12%) |
| questions, parser links | 208 | 138 (66%) | 46 (22%) | 24 (12%) |
| questions, encoder-added | 26 | 16 (62%) | 6 (23%) | 4 (15%) |
| both, all | 654 | 434 (66%) | 114 (17%) | 106 (16%) |
| both, parser links | 591 | 405 (69%) | 95 (16%) | 91 (15%) |
| both, encoder-added | 63 | 29 (46%) | 19 (30%) | 15 (24%) |

By length stratum (wrong share of links): <=6 words 8% (3 of 38), 7-10 words 5% (6 of 121), 11-15 words 20% (40 of 204), 16+ words 20% (57 of 291). Trees with no wrong link: 16 of 40 stories, 11 of 20 questions (8 and 0 with every link right). Among the 63 encoder-added links judged: those passing S3 were 15 links, 2 wrong (13%), 9 right, 4 debatable; those failing S3 were 48 links, 13 wrong (27%), 20 right, 15 debatable.
Typical wrong parser links: noun-phrase internals of compound nouns (`ice cream truck driver`, `bottle cap collection`, `the` attached to the wrong noun), adverbial / prepositional attachment (`of`, `for` to the wrong head), subject-complement reversals inside quoted or complex questions; encoder links in short sentences are mostly local and fine, the failures are clause-level (`so`, `when` as root, `and`).

## What this supports / does not support
Supports:
- Agreement of the encoders is a real signal but not a referee: links both A and A2 propose are 19.7% wrong against 47.4% for the rest; adding all-three agreement with probability >= 0.9 gets 10% wrong on 11% of links. Neither reaches the declared target.
- On MCTest-style sentences the parser alone fully connects 53-60% of sentences, and 79% of stories sentences of <= 10 words end as trusted trees (S3); on question sentences 61% of <= 10 words.
- In the blind judgement of final trees on short story and question sentences, about two thirds of links are right and about one sixth wrong (5% for <= 10 words, 20% above 10), and the parser's own links carry most of the wrongs (91 of 106).
Does not support:
- That "trusted" means correct: the trusted-tree count only checks the added links, and parser-committed links in the same trees were 15% wrong in the blind sample.
- Any claim for sentences over 20 words: 36+ words yields no trusted tree and 21-35 words 9-12%.
- Precise rates: 63 judged encoder-added links, 654 judged links, one rater with 69% self-consistency; the 60 sentences were stratified by length, so the overall percentages are not population rates. S3 in Part 2 uses forced parser links (Part 1 did not), so the two S3 pass rates (15% vs 18-22%) are not the same quantity. The 22 parser-failed fairytale sentences pass S3 for 73% of their links, which reflects easy local links, not accuracy (not judged).

## Three observations for the decision
1. The encoder's added links are a small part of the work on simple text: 1.4-1.9 added links per fragmented sentence on stories and questions, 7 on fairytale; most of the wrong links the blind pass found (91 of 106) are parser-committed, mostly noun-phrase and attachment errors, so the reading quality of simple lessons is limited about equally by the parser and by the encoder.
2. Sentence length is the dominant variable: wrong share of links is 5-8% up to 10 words and 20% from 11 words; trusted-tree share falls 79% / 49% / 12% across the bands on stories. Only the <= 10-word part of the MCTest-style text (57% of stories sentences, 81% of questions) is in a regime where most trees are fully trusted; fairytale test text is 24% <= 10 words.
3. Questions read differently from statements: 12% wrong (22% debatable) in the judged links, and a lower trusted share at <= 10 words (61% vs 79%), because wh-fronting and auxiliary `did` fragment the parse more often (parser connects 53% vs 60%); judged final questions have 0 of 20 trees with every link right and 11 of 20 with no wrong link.
