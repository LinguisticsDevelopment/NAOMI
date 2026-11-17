# NAOMI encoder, stage A2: treebank links for the unknown tops (2026-11-17)

Method (owner-agreed): the encoder predicts head + relation per word; the parser leaves the tops of unconnected pieces "unknown, not wrong". Stage A showed it then guesses how pieces connect (a third wrong in the hand-check). A2 adds human-annotated sentences (Universal Dependencies English-EWT, CC BY-SA 4.0) as PRE-LABELLED sentences for exactly that gap, under the owner's conditions: the grammar's parser stays the definition of the format and wins wherever it committed; converted links are used only for tops of unconnected pieces / loose tokens; the conversion table is data with no words; the model keeps no word-identity feature.

## Code and data (new)
- `src/naomi/treebank.py`: CoNLL-U reader, table-driven converter (copula swap, adposition / subordinator head swaps, coordination lift), character-stream alignment with masking, `compose` (the supervision rule). `data/treebank/conversion_table.json`: the table (rule names, UD relation names, UPOS tags; no words; `fetch` entry says where the files come from). Every converted link carries `operations+rule` (e.g. `adposition_heads_noun+R29_case_obj_PREPOSITION`).
- Table changes vs the feasibility study's v2 (all in the table): (1) a coordinator before the first conjunct is never chosen (`coord_lift.coordinator_must_follow_first_conjunct`); it stays with its clause; (2) a leftover conjunction is COORDINATION (was MODIFICATION); (3) relative pronouns: three rules labelling by role in the relative clause (SUBJECT / OBJECT / SPECIFICATION via the treebank's PronType=Rel). Evidence for (3) from the parser's own gold v7: `who` is SUBJECT 1060 / MODIFICATION 218 / OBJECT 152, `which` SUBJECT 472 / MODIFICATION 130 / OBJECT 127, so "relative pronoun = MODIFICATION" (the study's note) is the minority label; so the pronoun is labelled by its role in the relative clause; only 28 + 23 + 3 treebank-sourced links in train carry these rule names (most relative pronouns are inside the parser's own pieces).
- `scripts/build_treebank_gold.py`: parse (Reader.read_parser, 10 s cap, 4 workers, resumable shards) -> convert -> compose -> `runs/treebank/ewt_{train,dev,test}.jsonl.gz` (gold v7 schema + `sup_head`, `sup_rel`, `sup_source`, `sup_flags`) + `stats_*.json`. EWT train: seeded random 6,000 of 12.5k sentences (parse 8 min wall); dev and test: all (1,987 / 2,046 sentences with at least one non-punctuation token). Parse + conversion of all three: about 14 min.

## Data counts (node tokens = non-punctuation project tokens)
| split | sentences | parser failed | parser connected / pieces | tokens parser | tokens treebank | tokens unknown | unknown tops before |
|---|---|---|---|---|---|---|---|
| EWT train (6,000) | 5,959 used | 41 | 1,873 / 4,086 | 68,829 | 10,683 | 4,305 | 14,988 |
| EWT dev (1,987) | 1,978 | 9 | 795 / 1,183 | 18,004 | 2,508 | 1,242 | 3,750 |
| EWT test (2,046) | 2,036 | 10 | 842 / 1,194 | 17,775 | 2,489 | 1,317 | 3,806 |
Fate of the unknown tops (train): treebank link accepted 10,683 (71%); dropped because it would make a cycle with the parser's committed links 1,787 (12%); root claim while the parser already committed a root 1,319 (9%); no converted link (masked token, alignment failure, no head) 1,120 (7%); head not a node 79. Dev: 2,508 / 395 / 303 / 530 / 14; test: 2,489 / 395 / 307 / 594 / 21. Alignment: 86% of sentences align exactly, 14% with some masked tokens, none failed on the character stream.
Treebank-sourced links in train by table rule (top): det/predet DESCRIPTION 1,054; case-swap PREPOSITION 1,007; coordination item 946; compound/flat DESCRIPTION 760; case-swap MODIFICATION (obl/nmod) 723; nsubj SUBJECT 609; advmod SPECIFICATION 591; obj OBJECT 576; relative-clause/acl SUBORDINATION 572 + 284; mark-swap MODIFICATION 485; parataxis COORDINATION 412; discourse MODIFICATION 324; appos APPOSITION 294; ccomp OBJECT 280; asyndeton COORDINATION 195; negation MODIFICATION 153; leftover cc COORDINATION 129; (full list per rule and per operation combination in `runs/treebank/stats_train.json`).

## Training (model, features, hyper-parameters identical to stage A's main run; no word identity)
2.64M params, BiLSTM 3 x 192, same vocabularies (taken from stage A's train split only, so extra data cannot change the shape or label set), lr 2e-3, batch 1,800 tokens, patience 5. Train = stage A gold v7 train split (26,447 sentences, parser supervision) + 5,959 EWT train sentences. Keep-best on the mean of (stage-A dev LAS, treebank-dev LAS = LAS on the 2,508 treebank-supervised tokens of EWT dev). 4 CPU threads, nice.
| run | EWT sentences | treebank links | wall | epochs (best) | stage-A dev UAS/LAS at best | treebank-dev UAS/LAS at best |
|---|---|---|---|---|---|---|
| stage A (main) | none | none | 55 min | 37 (32) | .927 / .912 | not measured |
| A2 | 5,959 | yes (10,683 tokens) | 43.0 min | 30 (25) | .924 / .910 | .492 / .400 |
| A2-ctrl | 5,959 | no (parser links only) | 58.6 min (hit the 58-min cap at epoch 40) | 40 (38) | .928 / .913 | .278 / .158 |
Total training wall-clock for the reported runs: 102 min (A2 43 + ctrl 59). A first control run (17.8 min, curve in `runs/naomi_encoder_a2/ctrl_v1_meanscore/`) selected its best epoch with the same mean score; its treebank-dev term cannot improve without the links, so it picked epoch 7 (dev LAS .888). That is a flawed control and was discarded; the control reported here selects on stage-A dev LAS only (`--select-parser-dev`). Stage A's cap was 110 min and not binding; the cap here (58 min) bound only the control, at an epoch where its dev LAS was still creeping (+0.1 point per few epochs).

## Evaluation (identical items; stage A = `runs/naomi_encoder/main_best.pt`)
Files: `runs/naomi_encoder_a2/eval_{A,A2,A2ctrl}.json`.

### (a) Gold v7 test, sentences the parser connected (413)
| | UAS | LAS | exact lab. | scorer overall | <=10 UAS/LAS (310) | 11-20 (83) | 21-35 (20) |
|---|---|---|---|---|---|---|---|
| stage A | .936 | .926 | .765 | 92.9 | .948 / .938 | .934 / .925 | .896 / .881 |
| A2 | .929 | .919 | .746 | 91.9 | .940 / .929 | .936 / .926 | .875 / .863 |
| A2-ctrl | .939 | .930 | .794 | 93.3 | .946 / .936 | .946 / .941 | .894 / .883 |
A2 is 0.7 point UAS / 0.7 LAS below stage A (2 more wrong heads per 300 tokens; most of it in the 21-35 band, 20 sentences, 1 sentence = 5 points of exact match). The control, with the same extra sentences but no treebank links, is slightly ABOVE stage A. So the treebank links cost about 1 point on parser-style trees; I would call that small but not zero (frozen holdout, 30 connected: stage A .940 / .915, A2 .918 / .889, ctrl .949 / .915). Nearest-verb cheat baseline .343 / .292.

### (b) Gold v7 test, fragmented sentences (1,098): agreement on parser-committed links
| | UAS | LAS | main-root agreement | unknown tops placed: inside another piece / ROOT / own piece |
|---|---|---|---|---|
| stage A | .912 | .898 | .773 | 91.0% / 4.5% / 4.4% |
| A2 | .921 | .907 | .862 | 94.1% / 2.4% / 3.5% |
| A2-ctrl | .923 | .909 | .844 | 92.6% / 3.0% / 4.3% |
(by band UAS, A2: <=10 .900, 11-20 .901, 21-35 .920, 36+ .942; stage A .899 / .893 / .911 / .931). On the committed links of long sentences A2 is slightly better, the main-root choice improves by 9 points.

### (c) EWT dev + test sentences the parser leaves in pieces: unknown tops vs the converted treebank link
Items: 1,183 dev / 1,194 test fragmented sentences; 2,508 / 2,489 treebank-supervised tops. Agreement of the model's head (and head + relation) with the converted link.
| | dev head | dev head+rel | test head | test head+rel | test 36+ band head | previous-token baseline (dev / test) |
|---|---|---|---|---|---|---|
| stage A | .335 | .184 | .321 | .168 | .233 | .208 / .221 |
| A2 | .509 | .414 | .489 | .394 | .375 | |
| A2-ctrl | .333 | .177 | .330 | .175 | .264 | |
Test by band, head / head+rel, A2: <=10 .569 / .429; 11-20 .530 / .430; 21-35 .438 / .362; 36+ .375 / .323 (stage A .374 / .156, .332 / .183, .308 / .171, .233 / .139). By relation (A2, test): OBJECT .61 / .48, PREPOSITION .55 / .50, DESCRIPTION .53 / .45, MODIFICATION .46 / .39, SPECIFICATION .46 / .29, COORDINATION .43 / .35, SUBORDINATION .41 / .29, APPOSITION .33 / .23. Parser-committed links on the same sentences: stage A .866 / .867 (dev/test), A2 .881 / .888, ctrl .882 / .890. The EWT dev set was used for keep-best (so the dev column is optimistic by a little); the test column was not.
The control is at stage A's level, so the gain (+17 points of head agreement, +22 head+relation) comes from the treebank links, not from the extra sentences. These are held-out sentences from the treebank's own web-text genre, so this measures how well the model learned the converted links, not how good they are in the project's domain (that is (d)).

### (d) Blind hand-check on the project's own domain
Same 40 fragmented gold-v7 sentences as stage A's hand-check (10 per length band). Links added beyond the parser: stage A = exactly the 173 links judged then (the stage-A model's decode reproduces them: asserted); A2 = up to 10 per sentence (seeded random choice among its added links). The two sets were pooled into 270 distinct (sentence, dependent, head, relation) links, shuffled with seed 0 and given opaque ids (`runs/naomi_encoder_a2/handcheck_blind.json`: tokens, parser pieces, link; no model tree, no model name); I judged every link (`handcheck_blind_judgments.json`, rubric of stage A: right = head and relation as the project's conventions give; debatable = defensible alternative or lossy label; wrong) BEFORE reading the id -> model key (`handcheck_blind_key.json`), then joined (`handcheck_blind_result.json`). Single rater (me), quick judgments, as in stage A.
| | links | right | debatable | wrong |
|---|---|---|---|---|
| stage A (my new blind judgments) | 173 | 60 (35%) | 52 (30%) | 61 (35%) |
| stage A (recorded judgments, earlier) | 173 | 56 (32%) | 58 (34%) | 59 (34%) |
| A2 | 173 | 65 (38%) | 47 (27%) | 61 (35%) |
By band (right / debatable / wrong): stage A <=10 7/7/0, 11-20 6/10/5, 21-35 22/14/18, 36+ 25/21/38; A2 <=10 7/5/2, 11-20 8/9/4, 21-35 22/13/19, 36+ 28/20/36. (both models have 14 / 21 / 54 / 84 links in the four bands.)
Overlap: 76 links proposed by both (38 right / 23 debatable / 15 wrong = 20% wrong); 97 only by stage A (22 / 29 / 46 = 47% wrong); 97 only by A2 (27 / 24 / 46 = 47% wrong). The two models disagree on more than half of their added links and both are wrong about half of the time where they disagree: the links the treebank changed are no better than the ones it replaced.
Rater consistency (same 173 stage-A links, old vs new judgment): identical 120 (69%); of the 53 that differ, 48 are one step (d->r 12, w->d 12, d->w 15, r->d 9) and 5 cross the full range (w->r 3, r->w 2). The new pass was a little more generous (more d->r than r->d) and rated wrong 61 vs 59. So a per-model difference of a few links (right 60 vs 65) is inside the rater noise.
**Target (at most 15% wrong): not reached.** A2 has 35% wrong (61 of 173), the same as stage A; 38% right. By band the long sentences (36+) are 43% wrong (36 of 84), 21-35 is 35% wrong.
What is still wrong (A2 examples; sentence ids S.. are the blind file's):
1. S35, `said` -> `comes` OBJECT: a quotation's reporting verb hung as the object of the previous clause's verb (a long-range clause link, both models).
2. S17, `whether` -> `to` SUBORDINATION and `nor` -> `to` MODIFICATION: clause-level function words attached inside the wrong piece (the piece ends at `to do with it`; the conjunction belongs to `knew`).
3. S33, `from` -> `shaven` MODIFICATION and `paled` -> `sloped` SUBORDINATION: in a 42-word sentence with a relative clause and a nested "which ... though ..." the model connects pieces to the nearest earlier head instead of to the main clause.
4. S08, S28, `said` -> `at` MODIFICATION and `and` -> `at` MODIFICATION: the parser's odd piece `at(PREPOSITION:she..)` / `at(dodo)` (a parser error: the sentence-initial preposition got the subject as its object) is accepted as a head; the model inherits the parser's wrong structure because it is the only thing it can attach to.
5. S40, `answered` -> `if` MODIFICATION, `'a` -> `if` MODIFICATION, `space` -> `if` MODIFICATION: archaic dialogue with fronted objects (`a little space I may allow thee`): everything hangs on the nearest subordinator.
6. S07, S04, `without` -> `and` COORDINATION, `seemed` -> `so` COORDINATION, `was` -> `came` COORDINATION (reversing a correct link the other model also finds): COORDINATION used as a catch-all for "this piece goes near that one" with the wrong head or direction.
Wrong A2 links by relation: OBJECT 15, MODIFICATION 15, COORDINATION 14, DESCRIPTION 5, SPECIFICATION 5, SUBJECT 4, SUBORDINATION 3. Right links are mostly local (determiner/possessor, preposition-object, aux, list conjunction).

### (e) Tree quality: share of output trees on fragmented test sentences whose subtrees are all contiguous
Stage A 82.5%; A2 82.2%; A2-ctrl 81.5% (same definition as the decoder report: rank among node tokens, every subtree one block).

## Tests
`tests/test_naomi_treebank.py` (8 fast tests: copula + preposition swaps and labels; traceability of every link's rule/operation name; leading-conjunction fix and leftover conjunction label; relative pronoun rule; alignment masking; compose: parser wins, cycles dropped, root conflict, masked/no-link tokens; the table holds no lowercase words) and `tests/test_naomi_no_literals.py` (allow-list extended with the new module's non-linguistic identifiers: table field names, split names, source tags, status tags) plus `tests/test_naomi_encoder.py`: 15 passed in 23 s.

## Reading
1. The mechanism works for what it was built to measure: with the links, the model's choice for the tops of pieces agrees with the human-derived tree 49% (head) / 39% (head+relation) on held-out EWT test, vs 32% / 17% without (and 22% for "previous token"); the control proves it is the links, not the extra sentences.
2. It did not make the project-domain links better: blind hand-check 35% wrong for both, 47% wrong on the links where the models differ. Gutenberg/K-12 prose with long, archaic, quote-heavy sentences is not the web-text of EWT; what transferred is the local, in-clause structure, not the clause-to-clause links that matter. The genre/length gap shows in (c) too: the 36+ band is 37.5% head agreement even on EWT.
3. Cost: -0.7 point on parser-connected sentences (control +0.3), no change in tree contiguity; root choice for fragmented sentences +9 points.
4. Hand-check numbers are one rater, 173 links per model, rater self-consistency 69%: differences of a few links are noise; only "not near 15%" is solid.
5. What would shape the next step: (i) the treebank to train on should be in the target genre (GUM has fiction, but is non-commercial; or hand-annotate a small fiction sample, which the 35% rate shows is the real gap); (ii) the converted-link agreement on EWT is only ~50% even when trained on it, so the encoder is not the bottleneck in (c) as much as input features: no word identity means function words (`that`, `and`, `if`, `to`) cannot be told apart beyond POS/subtypes; stage A's hashed-identity ablation gained a point; (iii) a quarter of unknown tops get no usable treebank link (cycles with the parser's own wrong structure 12%, root conflicts 9%): the parser's wrong committed links (e.g. `at(PREPOSITION:she)`) poison both the supervision and the output; (iv) the learned tree can choose between proposals: `propose()` plus the grammar/round-trip referees are the way to use A2's better calibrated "inside another piece" placements.

## Files for the director to force-add (checkpoints, tables, logs, evals; NOT the licensed data)
`runs/naomi_encoder_a2/a2_best.pt`, `runs/naomi_encoder_a2/a2ctrl_best.pt`, `runs/naomi_encoder_a2/a2_curve.tsv`, `runs/naomi_encoder_a2/a2ctrl_curve.tsv`, `runs/naomi_encoder_a2/ctrl_v1_meanscore/a2ctrl_curve.tsv`, `runs/naomi_encoder_a2/ctrl_v1_meanscore/naomi_encoder_a2_ctrl.log`, `runs/naomi_encoder_a2/eval_A.json`, `eval_A2.json`, `eval_A2ctrl.json`, `handcheck_blind.json`, `handcheck_blind_key.json`, `handcheck_blind_judgments.json`, `handcheck_blind_result.json`, `runs/naomi_encoder_a2/splits.json`, `runs/treebank/stats_train.json`, `stats_dev.json`, `stats_test.json`, `runs/naomi_encoder_a2_main.log`, `runs/naomi_encoder_a2_ctrl.log`, `runs/naomi_encoder_a2_eval_A.log`, `..._eval_A2.log`, `..._eval_A2ctrl.log`, `runs/naomi_encoder_a2_blind_build.log`, `runs/treebank_build.log`; code/data: `data/treebank/conversion_table.json`, `src/naomi/treebank.py`, `scripts/build_treebank_gold.py`, `tests/test_naomi_treebank.py`. DO NOT add: `data/ud/**`, `runs/treebank/ewt_*.jsonl.gz`, `runs/treebank/work_*/` (converted/parsed licensed text), `*_last.pt`.
