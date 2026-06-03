# NSM Consciousness Transformer

A **research scaffold** — part of the NAOMI project — for a transformer that
reasons over meaning rather than surface tokens. It consumes tokenized text, a
serialized parse tree, a *consciousness* vector, and retrieved memory, and emits
two things: a language-model response and a transition of its internal
consciousness state. The meaning of the model is grounded (eventually) in the
**Natural Semantic Metalanguage (NSM)** prime inventory.

This repository is honest about what it is: the transformer plumbing is real and
trainable, but every genuinely hard component — semantic composition onto NSM
primes, the real NAOMI parser, long-term memory, and the consciousness objective
itself — is **mocked behind a clean interface** so the architecture can be built,
tested, and iterated on before those research problems are solved.

> **Read this first:** [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) lists the open
> problems and the stubbed-vs-real boundary in detail.

## Quick start

```bash
cd consciousness_transformer
pip install -e .            # installs torch (CPU is fine), numpy, pyyaml
pytest                      # ~20 fast tests, should pass cleanly

python scripts/train_phase1.py   # trains the tiny model on the toy dataset
python scripts/eval.py           # prints held-out toy accuracy (~chance; expected)
```

`train_phase1.py` writes a checkpoint to `runs/phase1.pt`; `eval.py` loads it
(or evaluates an untrained model if it is missing).

## Architecture

```
                 ComprehensionExample (passage, question, 4 options, answer)
                                     │
            ┌────────────────────────┼─────────────────────────┐
            ▼                        ▼                          ▼
   MockNaomiParser           MockSemanticMapper           MockMemoryStore
   (text -> ParseTree)       (ParseTree -> NSM            (context -> memory
            │                 activations + causal          vector)
            ▼                 table -> consciousness          │
   serialize_parse_tree       seed vector)                    │
   (flat token stream)            │                           │
            └────────────┬────────┴───────────┬───────────────┘
                         ▼                     ▼
                    SimpleTokenizer       consciousness / memory vectors
                         │                     │
                         └──────► FeatureBuilder ◄──────┘
                                     │  builds 4 causal-LM rows per example:
                                     │  [CONSC][MEM] text [SEP] parse [ANS] option [EOS]
                                     ▼
                         ConsciousnessTransformer
                          ├─ Head 1: language modeling (response + option scoring)
                          └─ Head 2: consciousness state transition
                                     │
                                     ▼
                    weighted loss = w_lm·LM + w_ans·answer + w_consist·consistency
```

### Inputs / outputs

* **Inputs:** tokenized text, a flat-serialized parse tree, a consciousness
  vector, and a retrieved-memory vector. The consciousness and memory vectors
  are injected by adding their linear projections onto the reserved `[CONSC]`
  and `[MEM]` token positions.
* **Output head 1 — language modeling:** standard causal next-token prediction
  over the response region. Multiple-choice answer prediction is derived from
  this same head by scoring each option's length-normalized log-likelihood
  (so there are genuinely two heads, three loss terms).
* **Output head 2 — consciousness transition:** reads the `[CONSC]` slot and
  predicts the next consciousness state.
* **Loss:** `w_lm · LM + w_answer · answer + w_consistency · consistency`, where
  the consistency term is a **placeholder** (L2 between predicted-next and
  current state — see below).

### Configuration

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml) and
are loaded into typed dataclasses by `nsm_ct.config.load_config`. Configurable
knobs include the consciousness dimension, transformer size, learning rate,
batch size, and curriculum phase. (We use a plain YAML loader rather than Hydra
to keep dependencies minimal; swapping in Hydra later is straightforward.)

## What's real vs. what's mocked

| Component | Status | Where to plug in the real thing |
|---|---|---|
| Transformer + two heads | **Real** (tiny) | `model.py` |
| Training / eval loops | **Real** | `scripts/` |
| Loss combination | **Real** | `losses.py` |
| Toy dataset | **Real but synthetic & obviously fake** | `dataset.py::generate_toy_dataset` |
| NSM prime inventory | **Real constants** (~65 primes) | `nsm_primes.py` |
| Parse trees | **Mock** (flat, fake POS) | implement `AbstractParser` (`parser_interface.py`) |
| Semantic mapping → NSM | **Mock** (hash-based) | implement `AbstractSemanticMapper` (`semantic_mapper.py`) |
| Tree serialization | **Real but flat** (lossy) | `serialization.py` (TODO: hierarchical) |
| Retrieved memory | **Mock** (hash-based) | implement `AbstractMemory` (`memory.py`) |
| Consciousness consistency loss | **Placeholder** (L2 inertia) | `losses.py::consciousness_consistency_loss` |
| Real comprehension data | **Hooks only** | `dataset.py::load_commonlit` / `load_race_easy` / `load_jsonl_examples` |

The three mocks all implement the same abstract interfaces the real components
would, so graduating from scaffold to system means writing an adapter and
passing it to `build_default_stack` (or `FeatureBuilder`) — no architectural
surgery.

## Plugging in real components

```python
from nsm_ct import Config, FeatureBuilder
from nsm_ct.dataset import build_tokenizer, generate_toy_dataset

# 1. Implement the interfaces:
#    - AbstractParser           (nsm_ct.parser_interface)
#    - AbstractSemanticMapper   (nsm_ct.semantic_mapper)
#    - AbstractMemory           (nsm_ct.memory)
my_parser = RealNaomiParserAdapter(...)
my_mapper = RealNSMSemanticMapper(...)
my_memory = RealEpisodicMemory(...)

cfg = Config()
examples = generate_toy_dataset(100)          # or load_jsonl_examples("real.jsonl")
tokenizer = build_tokenizer(examples)
fb = FeatureBuilder(tokenizer, my_parser, my_mapper, my_memory, cfg)
# ...everything downstream (dataset, model, training) is unchanged.
```

## Layout

```
consciousness_transformer/
├── README.md                  # this file
├── RESEARCH_NOTES.md          # open problems & research-vs-engineering boundary
├── pyproject.toml             # pip install -e .
├── configs/default.yaml       # all hyperparameters
├── src/nsm_ct/
│   ├── nsm_primes.py          # canonical NSM prime inventory (~65 primes)
│   ├── data_structures.py     # ParseTree, CausalTable, ConsciousnessState, ...
│   ├── tokenizer.py           # simple fixed-vocab tokenizer
│   ├── parser_interface.py    # AbstractParser + MockNaomiParser
│   ├── semantic_mapper.py     # AbstractSemanticMapper + MockSemanticMapper
│   ├── serialization.py       # flat parse-tree serialization
│   ├── memory.py              # AbstractMemory + MockMemoryStore
│   ├── features.py            # FeatureBuilder + Batch + collate (the integration seam)
│   ├── model.py               # ConsciousnessTransformer (two heads)
│   ├── losses.py              # weighted LM + answer + consistency loss
│   ├── dataset.py             # toy data, torch Dataset, real-data hooks
│   └── config.py              # YAML loader + typed config
├── scripts/
│   ├── train_phase1.py        # train on toy data
│   └── eval.py                # multiple-choice accuracy on held-out toy set
└── tests/
    ├── test_nsm_primes.py
    ├── test_data_structures.py
    └── test_integration.py    # one full inference + training step, end-to-end
```

## Open research questions

See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md). In brief: semantic composition onto
NSM primes, tree-structured (vs. flat) serialization, consciousness-dimension
ablation, memory pruning, and coherence checking.
