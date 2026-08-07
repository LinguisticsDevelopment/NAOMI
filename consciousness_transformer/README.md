# NSM Consciousness Transformer — a meaning-space mind (part of NAOMI)

One system you can **talk to**, that **remembers**, **reasons** (derive-or-abstain,
with a faithful trace), and that you can **teach** — built on the thesis that
everything the system holds and manipulates is a **meaning object** over ~65 NSM
primes, never tokens.

Two cleanly separated kinds of learning (the spine):

- **Weights hold zero information.** Gradient descent touches only the cognitive
  policy — how to perceive, recall, infer, respond. No facts, rules, or taxonomy
  are ever baked into weights.
- **All content lives in the graph.** Teaching is a graph write (instant,
  non-destructive, inspectable), never a weight update.

The design contract is [`MIND_ARCHITECTURE.md`](MIND_ARCHITECTURE.md); the full
measured research log (every milestone, including the honest negatives and the
M24 leakage audit) is [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).

> **History note:** the original scaffold here was a token-fed transformer used as
> a state-transition function. The clause/mind line superseded it; the token stack
> was removed (git tag `token-stack-final` holds the last commit that contains it).
> The surviving transformer count in this package: **zero** — the learned parts are
> a small GRU controller + heads and a tiny drive MLP.

## What's here

**The substrate (deterministic, inspectable — never trained):**

- `nsm_primes.py` — the canonical ~65 NSM primes (the only atomic vectors).
- `tpr.py`, `clause.py`, `entity_memory.py` — Tensor Product Representation
  binding: clauses as d×d meaning matrices over primes; an order-3
  `entity⊗relation⊗value` working memory. Zero token embeddings.
- `meaning_graph.py`, `collapse.py`, `serialization.py` — the dual-coded graph
  store: every node carries a lossy vector *handle* + a lossless serialized
  *structure* (correctness always routes through the structure).
- `meaning.py`, `nsm_molecules.py`, `wordnet.py`, `explications.py` — word
  meaning by precedence: prime → cited NSM molecule → WordNet-gloss
  decomposition → SOMEONE/SOMETHING. Nothing is ever hallucinated.
- `ground/` — the **grounded lexical space** (M17–M25): a minimal set of named,
  interpretable axes over which word-senses are *placed* by deterministic
  propagation. Honest held-out numbers: synonym AUC 0.86, syn>ant 0.756,
  hypernym 0.72, random-pair ≈ 0. The measured boundary: antonymy is a signed
  **relation**, composition is an **operation**, reasoning is **dynamics** —
  structure layered on the space, not more axes.
- `reasoning_oracle.py` — the symbolic floor: forward chaining + unification
  (reproduces ProofWriter open-world gold at 0.989). Teacher and validator,
  never the runtime's only path.
- `wsd.py` — sense inventories (incl. `GroundedWordNetSenseInventory`, whose
  sense signatures come from the grounding work) + the coherence-driven
  `IterativeSenseResolver` (standalone; not yet wired into the loop).

**The mind (`src/nsm_ct/mind/` — the learned policy + the two loops):**

- `schema.py`, `knowledge.py`, `persistence.py` — the frozen meaning-object
  contract; LTM as a durable, disk-persistent graph with variable-bearing Horn
  rules *as graph data*.
- `ops.py`, `executor.py` — the cognitive instruction set {PERCEIVE, RECALL,
  INFER, CONSOLIDATE, SUPERSEDE, RESPOND, HALT} over live STM + LTM.
- `controller.py`, `controller_losses.py`, `teacher.py`, `proof_search.py` —
  the learned navigation policy (which rule/relation to follow next; backward
  chaining took ProofWriter depth-2 from 0.48 → ~1.0). The engine derives;
  the controller only navigates.
- `conscious_loop.py`, `subconscious_loop.py` — the one-door clause feed
  (`consume`/`converse`) and the background consolidate/offline-infer/self-train
  loop.
- `grammar.py`, `membrane.py`, `verbalize.py` — the owned, deterministic
  controlled-English membrane (parse ↔ render, round-trip-tested); answers
  verbalize their *actual* derivation chain.
- `coref.py`, `routing.py`, `conversation.py`, `drive*.py` — pronoun
  resolution, the learned absorb-vs-answer routing, stateful multi-turn
  conversation (ask-when-blocked, resolve-later), and the RL-trained
  initiative drive (when to volunteer/ask/stay quiet).

## Quick start

```bash
cd consciousness_transformer
pip install -e '.[dev]'     # torch (CPU is fine), numpy, pyyaml, nltk, pytest
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
pytest                      # the full gate suite

python scripts/talk.py      # talk to it in controlled English
python scripts/train_proofwriter.py --backward   # learned proof navigation
python scripts/train_drive_rl.py                 # the initiative drive (RL)
python scripts/probe_ground_understanding.py     # grounded-space evaluation
```

Example (`talk.py`, no pretraining — knowledge is taught in-session):

```
> a beagle is a dog.
> a dog can bark.
> what can a beagle do?
Because a beagle is a dog and a dog can bark, a beagle can bark. A beagle can bark.
```

## Where the research stands

The short version of `RESEARCH_NOTES.md`:

- **Reasoning:** multi-hop traversal works and generalizes when the loop reads
  its own output (focus-chaining, held-out depth-3 ≈ 0.97 with adequate hops);
  learned *when-to-stop* is still maturing. Backward (goal-directed) navigation
  is the proven direction for rule-chaining.
- **Grounding:** the lexical space is settled (named axes + deterministic
  placement; sense-nodes and trained embeddings both measured and rejected).
  The open work is **composition over grounded primes** — clauses as operations
  (negation/antonymy as signed ops), not points.
- **WSD:** grounded sense signatures exist end to end; the coherence resolver
  runs but has **no correctness measurement yet** (vs. the MFS baseline on gold
  data) — that gate decides what sits in the perception slot.
- **Honesty rules:** any propagation metric must exclude its scored pairs
  (M24); a substrate milestone counts only when a downstream consumer
  measurably benefits.

## Layout

```
consciousness_transformer/
├── README.md / MIND_ARCHITECTURE.md / RESEARCH_NOTES.md
├── pyproject.toml
├── src/nsm_ct/
│   ├── nsm_primes.py  tpr.py  clause.py  entity_memory.py   # the TPR substrate
│   ├── meaning_graph.py  collapse.py  serialization.py      # dual-coded graph store
│   ├── meaning.py  nsm_molecules.py  wordnet.py  wsd.py     # word meaning + senses
│   ├── ground/                                              # the grounded lexical space
│   ├── clause_reactor.py  clause_psyche.py                  # the GRU controller family
│   ├── reasoning_oracle.py  episode.py                      # symbolic floor + curricula
│   ├── input_encoder.py  quantum_adapter.py  tokenizer.py   # parser front-end (text reader)
│   └── mind/                                                # the two loops + membrane + drive
├── scripts/            # talk / train_* / probe_* (each probe is a milestone gate)
└── tests/              # the gate suite
```

## Open problems

See `RESEARCH_NOTES.md` §1–§9 and the post-grounding roadmap: composition over
grounded primes (the central problem), the WSD-vs-MFS gate, antonymy as a
first-class signed-edge layer, what the consciousness state should represent,
cross-episode credit assignment, and the textbook north star
(`episode.py::TextbookSource`).
