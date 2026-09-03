# M63.1 — Encoder Gold-Tree v1 Stats (numbers only)

No interpretation added beyond what the generation run shows.

## Corpus

- Corpus files (unique sentences after dedup): real_gutenberg_alice.txt=407, real_gutenberg_bryant.txt=410, real_gutenberg_burgess_more.txt=256, real_gutenberg_busterbear.txt=89, real_gutenberg_edgeworth.txt=313
- Total unique sentences attempted: 1475

## Yield

- Gold records emitted: 1259
- Attempted: 1475
- Yield: 1259/1475 = 85.4%
- Failed (all non-full-tree outcomes): 216

### Failure breakdown by outcome

| outcome | count | share of attempted |
|---|---|---|
| full-tree | 1259 | 85.4% |
| grounding-fail | 180 | 12.2% |
| cap-hit | 28 | 1.9% |
| too-long | 6 | 0.4% |
| other-exception | 2 | 0.1% |

## Distributions (over the emitted gold set)

- tokens per sentence: n=1259 median=21.00 p90=45.00 min=3 max=90
- clauses per tree: n=1259 median=2.00 p90=3.00 min=1 max=7
- role-slots per clause: n=2420 median=2.00 p90=4.00 min=1 max=7
- candidate senses per content token: n=15230 median=5.00 p90=16.00 min=1 max=70

## Output artifact

- runs/encoder_gold_v1.jsonl: 1259 records, 4401039 bytes (3496 bytes/record avg)

