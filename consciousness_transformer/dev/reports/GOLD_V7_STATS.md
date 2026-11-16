# Gold v7 stats (teacher trees in the new tree format)

## Recipe
```
cd consciousness_transformer
OMP_NUM_THREADS=1 nohup nice -n 10 python scripts/build_gold_v7.py --workers 4 --out-dir runs/gold_v7_work > runs/build_gold_v7.log 2>&1 &
python scripts/build_gold_v7.py --merge --workers 4 --out-dir runs/gold_v7_work     # -> runs/gold_v7.jsonl.gz, gold_v7_failures.jsonl.gz
python scripts/build_gold_v7.py --sentences-file runs/holdout_sentences.txt     --out runs/holdout_gold_v7.jsonl
python scripts/build_gold_v7.py --sentences-file runs/holdout_dev_sentences.txt --out runs/holdout_dev_gold_v7.jsonl
```
Population: all `real_gutenberg_*.txt` (sorted) + seeded (0, budget 14000) whole-story K-12 FairytaleQA train subset
(v6's selection algorithm, re-implemented in the script), first-occurrence dedup, both holdout files excluded.
Parser only (no encoder fallback), 10 s cap, no cache, 4 workers (4 cores); sharded i::4, resumable per record.
Pilot: 200 sentences in 58 s wall (4 workers) projected ~2.4 h; actual full build 92 min (5,526 s longest shard).

## Yield (v7 vs v6)
| | v7 | v6 |
|---|---:|---:|
| attempted | 30,037 | 30,232 |
| parsed | 29,421 (97.9%) | 22,990 (76.0%) |
| failed | 616: cap_hit 549, no_hypothesis 67 | no-hyp 4,757, grounding-fail 1,941, cap-hit 544 |
Holdout: test 94/98 (same 4 cap-hits as v6), dev 98/98. (195 fewer attempted than v6; K-12 subset/dedup details differ slightly.)
v7 no longer has grounding failures (no clause extraction); a "cap_hit" is a failure with parse_seconds >= 10 s (the reader does not
distinguish timeout from other raise; 67 fast failures are no-hypothesis/error, e.g. "Sentence too long (>100)").
Failure sentence length (words): median 67, p10 48, p90 98, max 202; 0 below 10 words except 1; 90% are 40+ words. Parsed sentence length: median 20, p99 65.

## Attachment (tokens reachable from the main root, no UNATTACHED on the path)
Overall **30.0%** (167,245 / 557,535 non-punct tokens). By POS: NOUN 31.9, VERB 30.6, DET 33.2, ADP 22.3, PRON 32.6, ADV 29.7,
ADJ 28.1, AUX 34.1, CCONJ 23.1, PROPN 34.4, PART 26.5, NUM 33.7, SCONJ 25.9 (all within 22-34%; ADP/CCONJ lowest).
Per sentence: 100% attached 17.8%; >=90% 19.2%; >=75% 24.4%; <50% 63.3%. Mean `score_coverage` percent 51.2.
Per source (attached share): mcguffey first reader 54%, busterbear 40%, k12 33%, grimms 24%, edgeworth 25%, alice 27%.
Note: every token is still a node with its local structure (e.g. a clause under UNATTACHED keeps its internal edges); the
30% counts only what hangs from the main ROOT.

## Diagnostics (count / sentences with it)
unattached_token 74,232 / 20,188; unattached_component 61,951 / 21,492; multiple_components 21,492 / 21,492;
reversed_edge 75,126 / 22,678; extra_edge 406 / 371; graph_token_mismatch 40 / 9.

## Tree shape
Depth (root=0) p10/25/50/75/90/99/max: 2/3/4/5/6/8/13. Nodes per sentence (incl. synthetic root) 5/10/18/27/37/57/193.
Relation labels (nodes): UNATTACHED 136,183; DESCRIPTION 86,149; MODIFICATION 74,919; SUBJECT 56,670; PREPOSITION 48,981;
OBJECT 42,401; SPECIFICATION 31,678; COORDINATION 29,616; ROOT 29,420; INDIRECT_OBJECT 10,613; COMPLEMENT 6,456;
SUBORDINATION 4,148; APPOSITION 301.
Nodes (real tokens, 557,535): names 17,382 (3.1%); words whose first sense is a lexicon `fn.` sense 293,720 (52.7%);
other content words 246,433 (44.2%). Word nodes with no sense candidate: 8,233 of 540,153 (1.5%).

## What an encoder/decoder must predict in this format (facts)
- 13 distinct relation labels (11 grammar edge types + ROOT + UNATTACHED).
- Max children of one node: 192 (a synthetic ROOT holding many unattached pieces); trees with a synthetic multi-root: 21,493 / 29,421 (73.0%).
- Sentences with a non-main component / dangling token: 21,492 (73%) / 20,188 (69%).
- Reversed edges (modifier pointed at its head, re-attached under it) occur in 77% of sentences.
- 52.7% of nodes are function words; per-node data = token, lemma, pos, kind, candidate sense ids (best-first), label, subtypes.
- Sentence length 1-240 tokens; node count up to 193.

## Files
runs/gold_v7.jsonl.gz 18.3 MB (29,421 records); runs/gold_v7_failures.jsonl.gz 83 KB (616); runs/holdout_gold_v7.jsonl 0.68 MB (94) + `.failures` (4);
runs/holdout_dev_gold_v7.jsonl 0.75 MB (98) + empty `.failures`; runs/build_gold_v7.log.

## Examples of poor trees (compact form)
1. `said grandfather frog in his deepest , gruffest voice .` 3/8 attached: `ROOT(ROOT:said(INDIRECT_OBJECT:grandfather OBJECT:frog) UNATTACHED:in UNATTACHED:his UNATTACHED:deepest UNATTACHED:voice(DESCRIPTION:gruffest))` (inverted "said X" and a prepositional phrase not linked)
2. `" you must not let it grieve you so much , " said the little boy .` 4/13: `ROOT(ROOT:let(SUBJECT:you MODIFICATION:must MODIFICATION:not) UNATTACHED:grieve(...) UNATTACHED:said(INDIRECT_OBJECT:little(DESCRIPTION:the) OBJECT:boy))` (embedded clause and quote attribution split)
3. `then the peasant went his way , well content with his bargain .` 4/11: `ROOT(ROOT:went(SUBJECT:peasant(...)) UNATTACHED:content(MODIFICATION:way(DESCRIPTION:his) SUBJECT:well MODIFICATION:with(...)))` (trailing adjective phrase detached, "way" attached to it instead of "went")
4. `that is the way the tailor killed the third giant .` 3/10: relative clause detached; `killed` gets INDIRECT_OBJECT:third.
5. `said her master , and lamented the fine chickens .` 3/8: `and` unattached; second verb a separate component.

## Rebuild with grammar pass 2 (2026-11-16)

Same script and population, rebuilt after the second grammar pass (clause linking). The
tables above describe the first build; these are the current files' figures, computed by
the director over the whole of `runs/gold_v7.jsonl.gz`.

| | first build | rebuild |
|---|---:|---:|
| sentences parsed | 29,421 | 29,458 (579 failed) |
| tokens connected to the sentence's main root (strict) | 30.0% | 39.6% |
| sentences that are one fully connected tree | 17.8% | 27.9% |
| sentences with several unconnected pieces | 73% | 63.4% |

Fully connected by length (non-punctuation tokens): <=10: 59.6% of 8,420; 11-20: 26.8% of
9,744; 21-35: 6.9% of 8,261; 36+: 0.5% of 3,033. Holdouts: test 94/98, dev 98/98.
