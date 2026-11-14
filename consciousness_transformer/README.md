# consciousness_transformer — live code map (2026-11-14)

The reader/grounding core plus a new minimal read -> remember -> recall loop.
Thesis, status and next steps: `../GOALS.md`. Design contract:
`MIND_ARCHITECTURE.md`. Research ledger: `RESEARCH_NOTES.md`. The
curriculum-era code (executor, ops, membrane, the `mind/` package, old
decoder, 90+ scripts, 80+ tests) was moved to `archive/` on 2026-11-14 with
the same relative paths; see `archive/README.md`. The pre-refactor module
map is `archive/README_pre_refactor.md`.

## Live modules

`src/naomi/` — the minimal loop (~1,350 lines with the runner):
- `types.py` — shared contract (Filler, ClauseTree, ReadResult, Answer, LAYERS).
- `reader.py` — hybrid reader: deterministic parser first, encoder v6 fallback; candidate senses kept uncommitted; negation and question holes marked; on-disk read cache.
- `memory.py` — one layered graph of meaning trees on `nsm_ct.meaning_graph`, each clause also a TPR tensor handle; provenance outside the tensor.
- `recall.py` — question tree with a hole -> tensor-similarity candidates -> structural confirmation -> answer or abstain. No learned parts.
- `mind.py` — `Mind.teach / ask / say / sleep / forget`.

`src/nsm_ct/` — reader and grounding core:
- `encoder_model.py`, `encoder_train_util.py`, `input_encoder.py` — the learned encoder (v6 checkpoint: `runs/ARTIFACTS.md`).
- `clause.py`, `corpus.py`, `quantum_adapter.py`, `tokenizer.py`, `hard_gold_templates.py` — clause extraction, corpus/gold building, the bridge to `../quantum_parser`.
- `meaning_graph.py`, `tpr.py`, `data_structures.py`, `serialization.py`, `structure.py`, `thought.py`, `tree_render.py`, `candidates.py` — graph store with vector handles + lossless structure, TPR binding, candidate types.
- `meaning.py`, `nsm_primes.py`, `nsm_molecules.py`, `explications.py`, `wordnet.py`, `usvs_graded.py`, `ground/` — NSM primes, word meaning, and the USVS grounded lexical space.
- `episode.py` — slimmed to the `Episode` dataclass (full original in `archive/src/nsm_ct/episode.py`).
- `mind/` — only `datasets/` remains (SimLex data); the package itself is archived.

`scripts/` (19 files) — `naomi_run.py` (runner, below); encoder training/eval/gold (`train_encoder.py`, `eval_encoder.py`, `rescore_encoder.py`, `build_encoder_gold_v2.py`, `build_holdout_gold.py`, `make_holdout.py`, `gen_hard_gold.py`, `hand_gold.py`, `run_encoder_arms.sh`); `build_usvs.py`; benchmark converters and cheat baselines (`convert_fairytaleqa.py`, `convert_mctest.py`, `build_k12_fairytaleqa_corpus.py`, `fetch_k12.py`, `cheat_baselines_k12.py`, `cheat_encoder_baseline.py`); encoder-vs-parser probes (`encoder_vs_parser_probe.py`, `probe_encoder_complement.py`).

`tests/` (38 files) — `test_naomi_*.py` cover the new loop; the rest cover the reader/grounding core. Default tier is `pytest -m "not slow"`.

Other: `data/` (USVS, corpora, K-12 samples), `runs/` (checkpoints, gold, benchmark outputs; `runs/ARTIFACTS.md` is the index), `dev/` (design docs and reports; `dev/INDEX.md`), `configs/`, `colab/`, `logs/`.

## Run it

```bash
cd consciousness_transformer
python scripts/naomi_run.py teach-ask --text "Mary has a red ball." --ask "Who has a red ball?"
python scripts/naomi_run.py bench --dataset mctest --wrong-passage   # resumable; see --help
python scripts/naomi_run.py report --in runs/naomi_bench_mctest.jsonl
```
