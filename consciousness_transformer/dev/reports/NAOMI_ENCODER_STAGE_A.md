# NAOMI encoder, stage A (2026-11-16)

Method: `dev/design/ENCODER_DECODER_PAIR_PROPOSAL.md`. Output = for every non-punctuation token a head (token or ROOT) and a relation label; nothing else.
"Unknown, not wrong": tokens the parser left as tops of unconnected pieces or loose (relation UNATTACHED) carry no loss; tokens inside pieces are supervised for their head within the piece.

## Model and features
`src/naomi/encoder.py`. 2.64M params. Per token: tagger POS (embedding), tagger structural subtypes (23 with >=20 train tokens, multi-hot), USVS vector (614 axes, mean of top-3 candidate senses, zeros for names/none) -> linear 96, name flag, punctuation kept as context (not a node, not a head). Optional hashed word identity (off; ablation below). BiLSTM 3 x 192 -> biaffine head scorer with a ROOT slot -> relation MLP on (dependent, head). Decoding: Chu-Liu/Edmonds, single root enforced by trying each unconstrained root child as sole root (tested against brute force). `propose(sentence, k)`: best tree + trees forcing the next-best head for the least-confident tokens + re-rootings, deduplicated, scored by summed log-probs.

## Data (gold v7 rebuilt, 29,458 sentences; 0 token mismatches)
Split by source (whole K-12 story, or 400-sentence contiguous chunk of a book), seed 0: train 26,447 / dev 1,500 / test 1,511 sentences (`runs/naomi_encoder/splits.json`; test = 4 stories + chunks of alice, edgeworth, grimms; dev = 5 stories + chunks of bryant, burgess_more).
Node tokens: train 505,685 (supervised 423,965 = 83.8%, unknown 81,720); dev 26,723 (22,609 / 4,114); test 26,838 (22,338 / 4,500). Frozen holdouts (94 + 98 sentences) never trained on.

## Training
Main run: 37 epochs, 55 min wall (nice, CPU 4 threads; ~75 s/epoch; a stretch of one epoch was slowed to 9 min by my concurrent eval), early stop (patience 5); best dev supervised LAS 0.9121 at epoch 32 (UAS 0.926). Dev LAS curve: ep1 .790, ep5 .868, ep10 .891, ep20 .907, ep32 .912; train UAS reaches .97 while dev loss rises from ep ~10 (mild overfit; dev accuracy plateaus). Ablation (hashed word identity, 4096 buckets): 40 min, best dev LAS 0.9216 (+0.9 point). Total training ~1.6 h.

## (a) Sentences the parser connected fully (test split: 413 sentences, 3,415 tokens)
| | UAS | LAS | exact (unlab./lab.) | scorer meaning/shape/overall |
|---|---|---|---|---|
| encoder | 0.936 | 0.926 | 0.794 / 0.765 | 92.6 / 93.2 / 92.9 |
| nearest-verb cheat baseline | 0.343 | 0.292 | 0.111 / 0.087 | 29.8 / 38.2 / 34.0 |
By length (encoder UAS / LAS / exact-lab.): <=10 (310 sent.) .948/.938/.839; 11-20 (83) .934/.925/.627; 21-35 (20) .896/.881/.200; 36+ none (parser never connects them). Baseline by band UAS .38/.31/.28.
Frozen holdout (30 connected): UAS .940, LAS .915, exact .60, overall 92.0 (baseline .310/.260, 27.2). Frozen holdout-dev (15): UAS .893, LAS .855, overall 84.4 (baseline .237/.204, 19.0).
Word-identity ablation (test connected, no scorer): UAS .944, LAS .935, exact-unlab. .823 (main .936/.926/.794).

## (b) Sentences the parser left in pieces (test: 1,098 sentences; 18,923 supervised, 4,500 unknown tokens)
Agreement on committed links: UAS .912, LAS .898 (by band <=10 .899, 11-20 .893, 21-35 .911, 36+ .931 UAS). Main-root agreement .773.
Unknown tops placed: inside another piece 91.0%, ROOT 4.5%, below itself (own piece) 4.4%. Output is always one single-rooted tree (by decoding), so the connected share is 100% by construction; what matters is link quality below.
Confidence of added links: top-1 prob mean .675 / median .886 (supervised links .952 / 1.0); log-prob margin mean 4.95 / median 3.60 (supervised 15.1 / 13.9). Relations used for added links: MODIFICATION 27.6%, COORDINATION 22.1%, OBJECT 13.7%, SPECIFICATION 8.8%, DESCRIPTION 6.9%, SUBJECT 4.9%, ROOT 4.5%, PREPOSITION 4.1%, SUBORDINATION 3.0%, COMPLEMENT 2.9%, INDIRECT_OBJECT 1.1%, APPOSITION 0.4%.
Frozen holdout: 64 fragmented sentences, UAS .924 / LAS .906, 91.7% inside another piece; holdout-dev 83 sentences, .912 / .904, 89.8%.

## (c) Hand-check (40 held-out fragmented sentences, 10 per length band; all 173 added links I could judge, <=10 per sentence; my judgments, `handcheck_judgments.json`)
Overall right 56 (32%), debatable 58 (34%), wrong 59 (34%). By band: <=10: 7 r / 7 d / 0 w; 11-20: 5 / 8 / 8; 21-35: 19 / 19 / 16; 36+: 25 / 24 / 35.
By relation used (r/d/w): COORDINATION 14/24/12, MODIFICATION 10/15/16, OBJECT 3/6/10, SUBJECT 3/0/7, SPECIFICATION 7/5/2, DESCRIPTION 10/0/3, PREPOSITION 4/0/2, SUBORDINATION 1/3/4, COMPLEMENT 2/3/1, ROOT 2/2/1, INDIRECT_OBJECT 0/0/1.
This misses the proposal's target (<=15% wrong) by a wide margin; short sentences are fine, long ones (36+) are 42% wrong. Judgments are mine, single rater, quick.
Examples (parser pieces -> encoder tree, compact):
- good, v7-000279 "come here directly , and get ready for your walk": parser left `your`, `walk` loose; encoder: `for(PREPOSITION:walk(DESCRIPTION:your))`.
- good, v7-012932 "the howling of the wild dogs was henceforth heard no more ...": parser pieces `the`,`howling`,`of(..)`,`was(..)`; encoder `was(SUBJECT:howling(DESCRIPTION:the) MODIFICATION:of(..) ... COORDINATION:and(..))` (root `was`).
- debatable, v7-013066 "then the night-wind said , 'i will give thee counsel": `give(..)` piece -> `said(... COORDINATION:give(..))` (quoted clause as coordination of the verb).
- debatable, v7-000108 "london is the capital of paris , and paris ... and rome -- no , that 's all wrong": pieces `no`, `'s(..)`, `'m(..)` hung on the coordination `and` with COORDINATION / COMPLEMENT.
- wrong, v7-012993 "'now comes your turn , ' said he to the young tailor ; 'sew the eggs ...": parser `comes(..)`, `said(OBJECT:turn ..)`, `'sew`...; encoder makes `said` an OBJECT of `comes`, `'sew` a SUBJECT of `tailor`, `that` under `together`.
- wrong, v7-008609 "the close shaven green , which sloped down from the hatch-door ... was paled round with a rude paling": encoder `which` OBJECT of `shaven`, `paled` a PREPOSITION child of `of`.

## Tests
`tests/test_naomi_encoder.py` (6 fast tests: features, unknown mask, single-root acyclic decode on random scores, Edmonds = brute force on small cases, `propose` distinct/valid/deterministic, 20-sentence overfit > 0.9 head acc + JSON round-trip) and `tests/test_naomi_no_literals.py` (allow-list extended with non-linguistic identifiers): 7 passed in 21 s.

## Limits observed (most limiting first)
1. Teacher noise: ~16% of tokens are unknown and the supervised links include the parser's reversed-edge/wrong-clause artifacts (e.g. `that`/`has`, `said`/`turn`), so the encoder learns the parser's habits where it did commit (0.93 agreement) and has nothing to correct it on the long, fragmented sentences where it is most needed.
2. Long sentences: the training signal there is pieces only; coordination and clause-attachment choices (which verb a conjunction or quote belongs to) are guesses, 42% wrong in 36+ hand-check, and no fully-connected long sentence exists to test against (0 in the 36+ connected band).
3. No lexical identity by default and shallow meaning vector: the hashed-identity ablation gains ~1 point of dev LAS, showing verb/closed-class behaviour not captured by POS/subtype/USVS features; the relation set also mixes COORDINATION and MODIFICATION for many different uses, so labels for added links are often only "debatable".
Also: eval of the 36+ band against the parser's own committed links is not independent evidence; the hand-check is the only independent signal.

## Files for the director to force-add
runs/naomi_encoder/main_best.pt, runs/naomi_encoder/hash_best.pt, runs/naomi_encoder/splits.json, runs/naomi_encoder/main_curve.tsv, runs/naomi_encoder/hash_curve.tsv, runs/naomi_encoder/eval_main.json, runs/naomi_encoder/eval_hash.json, runs/naomi_encoder/handcheck_dump.json, runs/naomi_encoder/handcheck_judgments.json, runs/naomi_encoder_main.log, runs/naomi_encoder_hash.log, runs/naomi_encoder_eval_main.log. (Also the rebuilt runs/gold_v7.jsonl.gz, runs/holdout_gold_v7.jsonl, runs/holdout_dev_gold_v7.jsonl if not already committed.) Not: *_last.pt, prep_cache.pkl.
