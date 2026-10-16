# Memory capacity curve (M57d)

Deterministic probe (no training) of `nsm_ct.entity_memory`'s order-3 `[d,d,d]` entity(x)relation(x)value tensor: how many (instance, attribute-relation) facts fit in ONE memory before recall breaks, using genuinely sequential `nsm_ct.entity_memory.write` calls (gate=1 overwrite) and the matched-filter `query`/`query_entity` reads the reactor uses. Grid: dim in [32, 48, 64, 96, 128], n_instances in [2, 4, 8, 16, 32, 64], n_relations in [1, 2, 4, 8], codebook size V in [8, 32], value source in ['codec', 'random'], 5 seeds each (480 cells, 1 pruned/skipped). Full grid: `runs/capacity_curve.csv` (gitignored). Runtime: 329.7s.

| dim | V | source | max facts @fwd>=0.99 | @fwd>=0.95 | max facts @inv>=0.99 | @inv>=0.95 | overwrite new-value acc @fwd-0.95 cap | stale cosine @fwd-0.95 cap |
|---|---|---|---|---|---|---|---|---|
| 32 | 8 | codec | 32 | 64 | 64 | 64 | 0.99 | -0.037 |
| 32 | 8 | random | 32 | 64 | 64 | 64 | 0.99 | 0.039 |
| 32 | 32 | codec | 64 | 128 | 64 | 128 | 0.97 | 0.057 |
| 32 | 32 | random | 64 | 128 | 64 | 128 | 0.98 | 0.077 |
| 48 | 8 | codec | 64 | 128 | 256 | 256 | 0.98 | 0.065 |
| 48 | 8 | random | 64 | 128 | 512 | 512 | 0.98 | 0.067 |
| 48 | 32 | codec | 128 | 256 | 128 | 256 | 0.98 | 0.050 |
| 48 | 32 | random | 128 | 128 | 128 | 256 | 0.99 | 0.117 |
| 64 | 8 | codec | 128 | 256 | 128 | 256 | 0.97 | 0.044 |
| 64 | 8 | random | 128 | 256 | 512 | 512 | 0.97 | 0.071 |
| 64 | 32 | codec | 128 | 256 | 256 | 256 | 1.00 | 0.046 |
| 64 | 32 | random | 128 | 256 | 256 | 256 | 0.99 | 0.070 |
| 96 | 8 | codec | 128 | 256 | 256 | 256 | 0.99 | 0.076 |
| 96 | 8 | random | 128 | 256 | 512 | 512 | 1.00 | 0.068 |
| 96 | 32 | codec | 256 | 512 | 512 | 512 | 1.00 | 0.061 |
| 96 | 32 | random | 256 | 512 | 512 | 512 | 1.00 | 0.067 |
| 128 | 8 | codec | 256 | 512 | 256 | 256 | 1.00 | 0.057 |
| 128 | 8 | random | 256 | 512 | 512 | 512 | 1.00 | 0.041 |
| 128 | 32 | codec | 512 | 512 | 512 | 512 | 1.00 | 0.048 |
| 128 | 32 | random | 512 | 512 | 512 | 512 | 1.00 | 0.043 |

## So what

Fact-count ceiling at forward recall >= 0.95, V=32, codec-realistic values, per dim: dim=32: 128 facts, dim=48: 256 facts, dim=64: 256 facts, dim=96: 512 facts, dim=128: 512 facts. An episode with 8 entities x 6 facts (48 facts) needs dim >= 32 by this bound. A 50-entity passage at ~3-4 facts/entity (~150-200 facts) needs dim >= 48. Inverse recall ('who holds X?') is read off the SAME tensor via query_entity and is reported separately above because it can diverge from forward recall (see the table) -- if it ceilings lower, 'who is X?' breaks before 'what is X's value?' does at the same dim/fact-count, which matters more for resolver-style candidate generation than for plain attribute lookup.
