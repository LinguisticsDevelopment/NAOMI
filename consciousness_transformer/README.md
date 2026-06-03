# NSM Consciousness Transformer

A **research scaffold** — part of the NAOMI project — for a transformer used not
as a text predictor but as a **state-transition function** inside a stateful
reasoning loop. It threads an abstract *consciousness state* across a stream of
input sentences, decides for itself when to commit facts to working memory, and —
when a question arrives — recognizes it and answers from memory.

It is trained "like a kindergartener": on **episodes** (a context stream of
statements followed by a question), with the procedure *absorb → store →
recognize question → respond* emerging from weak supervision plus answer
correctness.

The genuinely hard pieces — what the consciousness state *means*, real semantic
composition onto NSM primes, a consistent parser, long-term memory — are
**mocked or stubbed behind clean interfaces** and clearly marked. See
[`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).

## The flow

```
 episode = [stmt_1, stmt_2, ..., stmt_k, QUESTION]
                         │ per input sentence
   input_obj = InputEncoder(sentence)        (TokenInputEncoder default;
                         │                     ParserInputEncoder optional/unstable)
   mem_read  = Memory.read(state)
   (state, input_obj, mem_read) ─► STEP (transformer) ─► new_state
                                                          ├─ action: {ABSORB, RESPOND, SKIP}
                                                          ├─ if ABSORB: Memory.write(fact)   ← state-gated
                                                          └─ at QUESTION: answer (MC scoring / open-ended)
   new_state ──► carried to the next input  (loop)
```

The state is the spine: its transition output **is** the next step's state and
**drives** the action gate. Memory writes are gated by the ABSORB action, so the
state genuinely controls what gets remembered.

## Quick start

```bash
cd consciousness_transformer
pip install -e .            # torch (CPU is fine), numpy, pyyaml
pytest                      # 29 fast tests

python scripts/train_phase1.py   # trains the loop on reasoning episodes
python scripts/eval.py           # held-out answer + action accuracy
```

With the default config (curriculum source, multiple-choice), training reaches
**~95% held-out answer accuracy and 100% action accuracy** in seconds on CPU —
the model learns to store facts, retrieve the relevant one (including recency on
"X moved to Y"), and answer. Example trace from the trained model:

```
CTX: ['john is in the kitchen .', 'john moved to the garden .'] | Q: where is john ?
   actions: ['ABSORB', 'ABSORB', 'RESPOND']   predicted: garden   gold: garden ✓
```

Use `--source babi` to train on Facebook bAbI (falls back to the curriculum
generator if the data can't be downloaded in your environment).

## Architecture

A single **step** (`model.ConsciousnessTransformer.step`) assembles a short
sequence `[state | memory | input tokens...]`, runs a small Transformer encoder,
and reads the state slot to produce **three heads**:

1. **State transition** → the next consciousness state (the loop's spine).
2. **Action gate** → `{ABSORB, RESPOND, SKIP}` (whether to write to memory / when
   to answer).
3. **Response** → at the question, either multiple-choice option scoring (the
   densest signal, reused from v1) or open-ended answer classification.

The loop itself lives in `agent.Mind`, which unrolls an episode, threads the
state, applies gated `WorkingMemory` writes, and answers at the question.

**Multi-hop reasoning.** Set `model.reasoning_hops > 1` to make the model take
several inference passes over memory at the question — re-reading memory and
updating the state each pass before answering ("reason with states"). `hops = 1`
is the default and reproduces the single-pass loop exactly.

### Two-tier memory & lifelong learning

There are two memories:

* **Local context** — `WorkingMemory`, per-episode, resets each episode.
* **Long-term memory** — `LongTermMemory` (`long_term_memory.py`), **persistent
  across episodes**: a growing store of consolidated entries plus a graph of
  **connections** between them. Reads from it are added to every memory read, so
  accumulated knowledge conditions future reasoning. It can be saved/loaded from
  disk.

The **state controls I/O and retention**: the action head gates input intake
(`ABSORB`/`SKIP`) and output (`RESPOND`), and a `consolidate_gate` head decides
how strongly the local context is committed to long-term memory.

`scripts/lifelong.py` runs the **lifelong loop** — many rounds of fresh episodes
(input→response cycles). Each round both trains the weights (*parametric*) and
consolidates what it absorbed into the long-term repo (*non-parametric*), so the
repository of connections grows round over round ("learn more and more over
tests"):

```
python scripts/lifelong.py --rounds 10 --episodes-per-round 32 --out runs/ltm.pt
# round 1 ... LTM entries=43  connections=15
# round 10 ... LTM entries=520 connections=204   (loss 2.44 -> 1.31)
```

Long-term memory is **off by default** (`model.use_long_term: false`); the
single-episode loop is unchanged unless you enable it.

### Word sense disambiguation (`wsd.py`, draft)

WSD here is a **constrained slice of the (still-mocked) semantic-mapping
problem**: rather than composing meaning from scratch, pick among a *finite,
known* set of candidate senses for a word, where each sense is an **NSM-prime
signature** (a mini-explication). It is **iterative and self-correcting**:
interpret a word with a sense given the current state; a learned **coherence**
head asks "does this make sense?"; if not, the state is updated and *all the
senses are re-evaluated* — so the chosen sense can change across hops. No gold
sense labels: the coherence signal drives the loop.

This is a **standalone module** (`IterativeSenseResolver`) operating on a
context vector built from `(state, memory_read)` plus an inventory's candidates;
it is **not wired into the Mind loop yet** (by design). The inventory and the
sense→prime signatures are **mocked/illustrative** (`MockSenseInventory`); the
real inventory is WordNet (`WordNetSenseInventory`, a hook), and the real
sense→prime mapping is the unsolved semantic-mapping problem.

**Loss** (`losses.compute_losses`):
`w_answer·answer + w_action·action + w_consistency·consistency`, where
`action` is weak supervision (statements→ABSORB, question→RESPOND) and
`consistency` is a **placeholder** auxiliary term (L2 between consecutive states).

### Configuration

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml),
loaded into typed dataclasses by `nsm_ct.config.load_config`. Configurable:
consciousness/memory dims, transformer size, reasoning hops, long-term memory
toggle, learning rate, batch size, loss weights, episode source, answer mode,
curriculum level, and input encoder.

## What's real vs. mocked

| Component | Status | Where |
|---|---|---|
| State-transition loop + memory writes | **Real** | `model.py`, `agent.py`, `memory.py` |
| Local context (per-episode working memory) | **Real** | `memory.py` |
| Long-term memory (persistent repo + connections) | **Real** (config-gated) | `long_term_memory.py` |
| Lifelong loop (grows the repo over tests) | **Real** | `lifelong.py`, `scripts/lifelong.py` |
| State-controlled retention (consolidation gate) | **Real** | `model.py`, `agent.py` |
| Memory pruning policy | **Placeholder** (FIFO cap) | `long_term_memory.py` |
| Multi-hop reasoning over memory | **Real** (config-gated) | `agent.py` |
| Action gating (absorb/respond) | **Real** | `model.py`, `losses.py` |
| WSD scorer + coherence-driven re-evaluation | **Real, standalone** | `wsd.py` |
| WSD sense inventory + sense→prime signatures | **Mocked** (WordNet hook) | `wsd.py` |
| Multiple-choice + open-ended response | **Real** | `model.py` |
| Curriculum reasoning episodes | **Real, synthetic** (memory-required) | `episode.py` |
| bAbI loader | **Real** (offline fallback) | `episode.py` |
| NSM prime inventory (~65) | **Real constants** | `nsm_primes.py` |
| Input encoder (token) | **Real default** | `input_encoder.py` |
| Parser input encoder | **Optional / experimental** (quantum_parser) | `input_encoder.py`, `quantum_adapter.py` |
| Semantic mapping → NSM meaning | **Mocked** | `semantic_mapper.py` |
| Consciousness state *meaning* + real consistency loss | **Placeholder** | `losses.py`, RESEARCH_NOTES |
| Long-term / episodic memory + pruning | **Not built** (per-episode only) | RESEARCH_NOTES |
| Textbook ingestion ("read a textbook, do the homework") | **Stub** (north star) | `episode.py::TextbookSource` |

## Plugging in real components

Everything swaps behind an interface; the loop is unchanged:

```python
from nsm_ct import build_default_stack, load_config
from nsm_ct.episode import make_source

cfg = load_config()
cfg.input_encoder = "parser"          # or implement AbstractInputEncoder
cfg.data.source = "babi"              # or implement AbstractEpisodeSource
episodes = make_source(cfg.data.source).generate(cfg.data.num_episodes)
stack = build_default_stack(cfg, episodes)   # tokenizer + encoder + model + memory + Mind
```

- New input source (incl. the "chained transformers" fallback for messy natural
  language): implement `AbstractInputEncoder` (`input_encoder.py`).
- New data (incl. real textbooks): implement `AbstractEpisodeSource`
  (`episode.py`).
- Real meaning: implement `AbstractSemanticMapper` (`semantic_mapper.py`) — the
  central research problem, still mocked.

## Layout

```
consciousness_transformer/
├── README.md / RESEARCH_NOTES.md
├── pyproject.toml / configs/default.yaml
├── src/nsm_ct/
│   ├── episode.py          # Episode + sources (curriculum, bAbI, textbook stub)
│   ├── input_encoder.py    # sentence -> input object (token default, parser optional)
│   ├── quantum_adapter.py  # experimental quantum_parser -> ParseTree (optional)
│   ├── memory.py           # WorkingMemory (local context): gated write + read
│   ├── long_term_memory.py # LongTermMemory: persistent repo + connection graph
│   ├── lifelong.py         # lifelong loop: grow the repo over rounds of episodes
│   ├── model.py            # ConsciousnessTransformer.step + heads (+ consolidate gate)
│   ├── agent.py            # Mind: unrolls an episode (+ multi-hop, + long-term)
│   ├── wsd.py              # NSM-grounded, coherence-driven WSD (standalone draft)
│   ├── losses.py           # answer + action + placeholder consistency
│   ├── metrics.py          # answer / action accuracy
│   ├── dataset.py          # EpisodeBatch, encoding, collation, tokenizer
│   ├── config.py           # YAML -> typed config
│   ├── nsm_primes.py       # canonical NSM prime inventory (~65)
│   ├── data_structures.py  # ParseTree/ParseNode, CausalTable, ConsciousnessState
│   ├── serialization.py    # flat parse serialization (with relations)
│   ├── parser_interface.py / semantic_mapper.py   # mocks for the research seams
│   └── tokenizer.py
├── scripts/{train_phase1,eval,lifelong}.py
└── tests/{test_nsm_primes,test_data_structures,test_episode,test_memory_rw,test_agent_loop}.py
```

## Open research questions

See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md): what the consciousness state means
and its real objective, semantic composition onto NSM primes, long-term memory +
pruning, tree-structured serialization, consciousness-dim ablation, coherence
checking, the textbook north star, and reconciling NSM primes with NAOMI's
existing 51 anchor dimensions.
