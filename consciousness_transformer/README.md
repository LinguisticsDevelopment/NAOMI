# Psyche — NSM Consciousness Transformer

**Psyche** is the trained consciousness/reasoning entity in this scaffold (part of
the NAOMI project). It is a transformer used not as a text predictor but as a
**state-transition function** inside a stateful reasoning loop: it threads an
abstract *consciousness state* across a stream of input items, and — entirely on
its own — chooses what to remember, **whom to trust**, and when to answer.

Everything is **emergent**: the model is never told which item is the question or
what action to take. From one signal (was the answer right?) it learns to absorb
facts, discount contradicted ones, persist trustworthy knowledge to long-term
memory, and respond when appropriate. It is trained "like a kindergartener" on
**episodes** (a stream of statements/sources, then a question).

The genuinely hard pieces — what the consciousness state *means*, real semantic
composition onto NSM primes, a consistent parser, cross-episode credit assignment
for APPEND — are **mocked or stubbed behind clean interfaces** and clearly marked.
See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md). The Python package is `nsm_ct`; the
loop class is `Psyche`.

## The flow

```
 episode = [item_1, item_2, ..., item_n]   (statements, the question, distractors —
                         │ per item            all uniform; nothing says which is which)
   input_obj = InputEncoder(item)          (TokenInputEncoder default;
                         │                   ParserInputEncoder optional/unstable)
   mem_read  = LocalMemory.read(state) + LongTermMemory.read(state)
   (state, input_obj, mem_read) ─► STEP (transformer) ─► new_state
                                                          └─ action ∈ {ABSORB, APPEND, RESPOND, SKIP}
                                                             (a soft distribution the model CHOOSES)
                                                               ├ ABSORB  → write to local memory
                                                               ├ APPEND  → commit to long-term (world facts)
                                                               └ RESPOND → weight this step's answer
   answer = Σ_t P(RESPOND | item_t) · response_t      ← the model decides WHEN to answer
   new_state ──► carried to the next item  (loop)
```

The state is the spine: its transition output **is** the next step's state and
**drives** the action distribution. Nothing is hard-coded — the model isn't told
which item is the question or what action to take; ABSORB and RESPOND are learned
**only** from whether the final answer is right (see "emergent actions" below).

## Quick start

```bash
cd consciousness_transformer
pip install -e .            # torch (CPU is fine), numpy, pyyaml
pytest                      # 47 fast tests

python scripts/train_phase1.py   # trains the loop on reasoning episodes
python scripts/eval.py           # held-out answer accuracy + response position
```

With the default config (curriculum levels 1–5, multiple-choice), training
reaches **~85% held-out answer accuracy** in seconds on CPU — with **no action
supervision and no trust labels**. The model chooses absorb/append/respond/skip
per item, learns *when* to answer (level 4 puts a corrupting "X moved to Y" after
the question, so it can't just read the final state), and learns *whom to trust*
(level 5 has two sources corroborate and one contradict; the answer is the
corroborated place). `resp_pos` shows where it concentrated its response mass
(0 = first item, 1 = last); `trust_gap` (>0) shows it trusts corroborated items
more than contradicted ones — modest, since the task can also be solved via the
response head, but emergent.

Use `--source babi` to train on Facebook bAbI (falls back to the curriculum
generator if the data can't be downloaded in your environment).

## Architecture

A single **step** (`model.ConsciousnessTransformer.step`) assembles a short
sequence `[state | memory | input tokens...]`, runs a small Transformer encoder,
and reads the state slot to produce **three heads**:

1. **State transition** → the next consciousness state (the loop's spine).
2. **Action repertoire** → a 4-way distribution over
   `{ABSORB, APPEND, RESPOND, SKIP}`. The probabilities are *soft gates*: ABSORB
   gates the local-memory write, APPEND gates the long-term commit, RESPOND
   weights this step's response.
3. **Response** → multiple-choice option scoring or open-ended classification,
   computed at every step.

Plus a **trust** signal (`trust_gate`) that judges each item against what memory
already holds and **scales how strongly it is written** — so corroborated info is
used and contradicted info is discounted. Trust is emergent (no trust labels): it
is learned purely because answering correctly requires discounting contradictions.

**Emergent actions (no hard-coding).** The action choice is **not supervised**.
The episode is a uniform stream of items — the model is never told which is the
question. The answer is the RESPOND-probability-weighted aggregate of the
per-step responses, so the model learns *when* to answer purely from answer
correctness. (One exception: APPEND only pays off in future episodes, so it gets
a small label-free **novelty** signal — append what's new vs. what long-term
memory already knows. See RESEARCH_NOTES.)

The loop lives in `agent.Mind`, which unrolls the stream, threads the state,
applies the soft-gated memory writes, and aggregates the response.

**Multi-hop reasoning.** Set `model.reasoning_hops > 1` to make the model take
several inference passes over memory at the question — re-reading memory and
updating the state each pass before answering ("reason with states"). `hops = 1`
is the default and reproduces the single-pass loop exactly.

### Two-tier memory & lifelong learning

There are two memories:

* **Local context** — `WorkingMemory`, per-episode, resets each episode.
* **Long-term memory** — `LongTermMemory` (`long_term_memory.py`), **persistent
  across episodes**, holding **"facts we know about the world"**: each entry
  carries its fact *text* (provenance). It is **seeded with base world facts**
  and the model's `APPEND` action adds newly-learned facts on top, alongside a
  growing graph of **connections**. Reads from it are added to every memory read.
  It can be saved/loaded from disk and read back as text (`LongTermMemory.facts()`).

The **state controls I/O and retention** through the one action repertoire:
`ABSORB` (intake → local), `APPEND` (retain → long-term), `RESPOND` (output),
`SKIP` — all chosen by the model, none hard-coded.

`scripts/lifelong.py` runs the **lifelong loop** — many rounds of fresh episodes
(input→response cycles). Each round both trains the weights (*parametric*) and
consolidates what it absorbed into the long-term repo (*non-parametric*), so the
repository of connections grows round over round ("learn more and more over
tests"):

```
python scripts/lifelong.py --rounds 8 --episodes-per-round 32 --out runs/ltm.pt
# Seeded 12 base world facts into long-term memory.
# round 8 ... LTM entries=716 connections=706
#   base world facts (sample):   the kitchen is a place . | the garden is a place . | ...
#   learned facts (sample):      bill is in the bathroom . | sandra is in the bedroom . | ...
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
`w_answer·answer + w_novelty·novelty + w_consistency·consistency`. There is **no
action supervision** — `answer` (on the RESPOND-weighted aggregate) is the sole
task signal; `novelty` is the small label-free APPEND auxiliary; `consistency`
is a **placeholder** (L2 between consecutive states).

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
| Emergent action repertoire (absorb/append/respond/skip, no labels) | **Real** | `model.py`, `agent.py`, `losses.py` |
| Model-chosen response timing | **Real** | `agent.py` |
| Emergent trust (corroboration vs contradiction, no labels) | **Real** (modest signal) | `model.py`, `agent.py`, `episode.py` |
| Self-controlled read/think/respond loop | **Not built** (next step) | RESEARCH_NOTES |
| Cross-episode credit assignment for APPEND | **Not built** (next step) | RESEARCH_NOTES |
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
