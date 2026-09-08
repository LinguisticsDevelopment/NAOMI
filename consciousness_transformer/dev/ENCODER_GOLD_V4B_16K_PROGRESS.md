# encoder-gold-v4b-16k build — interim progress note

Build launched 2026-09-08 18:58:12 UTC (`scripts/build_encoder_gold_v2.py
--workers 4`, `OMP_NUM_THREADS=1`, `CORPUS_MAX_PARSE_SECONDS=30`, forest
`top1` default, full `data/corpus/real_*.txt` population — 16,411 unique
deduped sentences, sharded into 4 contiguous chunks of ~4,102-4,103
sentences each, one fork()ed subprocess per shard).

This snapshot (first ~45-minute checkpoint, ~19:47 UTC, ~49 min wall so
far): **9,559 records** written across the 4 in-progress shard files
(`runs/encoder_gold_v4b_16k.shard{0,1,2,3}.jsonl`), concatenated here in
shard order (0→1→2→3, i.e. original sentence order within what each shard
has processed so far — NOT yet a full-corpus-order file since the shards
are not all at the same sentence offset) as
`runs/encoder_gold_v4b_16k.partial.jsonl` for durability in case the build
is interrupted before completion.

Per-shard progress at snapshot time (from the build log,
`runs/build_encoder_gold_v4b_16k.log`):

| shard | sentences in shard | processed so far | wall clock |
|---|---:|---:|---:|
| 0 | 4,103 | ~3,800-3,900 | ~2,870s |
| 1 | 4,103 | ~3,400+ | ~2,900s+ |
| 2 | 4,103 | ~2,900+ | ~2,900s+ |
| 3 | 4,102 | ~4,100 (near done) | ~2,890s |

This is a **snapshot for durability only** — the final commit on this
branch will replace `runs/encoder_gold_v4b_16k.partial.jsonl` with the
real `runs/encoder_gold_v4b_16k.jsonl` (all shards concatenated in true
original sentence order after all 4 finish) plus the full stats doc
(`dev/ENCODER_GOLD_V4B_16K_STATS.md`).
