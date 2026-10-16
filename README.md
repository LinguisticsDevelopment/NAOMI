# NAOMI

NAOMI is a research project building a system you can **talk to**, that
**remembers**, **reasons** (derive-or-abstain, with a faithful trace), and
that you can **teach** — on the thesis that everything it holds and
manipulates is a **meaning object** over ~65 NSM (Natural Semantic
Metalanguage) primes, never tokens. Gradient descent only ever touches the
cognitive *policy* (how to perceive, recall, infer, respond); all content —
facts, rules, taxonomy — lives in graph/tensor memory and is written there
directly, never baked into weights. The live, substantial part of the
project is `consciousness_transformer/` (the "NSM Consciousness
Transformer", or "the mind"); everything else at the repo root is either
support for it (`quantum_parser/`) or an earlier, now-legacy prototype kept
for history.

The active work is organized by **goal**, not by milestone number — see
**[`GOALS.md`](GOALS.md)** for what's proven, what's open, and what's running
right now. The full measured research log (every milestone, including the
honest negatives) is `consciousness_transformer/RESEARCH_NOTES.md`.

## Repo map

- **`consciousness_transformer/`** — the live project. `src/nsm_ct/` is the
  substrate (meaning graph, TPR binding, USVS grounding, the clause reactor)
  and `src/nsm_ct/mind/` is the learned policy + the two loops (conscious /
  subconscious) built on top of it. `scripts/` holds every training, probe,
  and corpus-building entry point; `tests/` is the gate suite; `dev/` is the
  design docs and research reports (`dev/archive/` for the superseded ones).
  See `consciousness_transformer/README.md` for the module-level map.
- **`quantum_parser/`** — the deterministic, language-agnostic grammar
  parser `consciousness_transformer` reads text through (one grammar file per
  language). Actively used, not legacy, even though its own docs predate the
  mind/ line and talk about an earlier framing of the project.
- **`parser/`, `webber/`, `drill/`, `junk/`, `ref/`** — **legacy.** Early Go
  and scratch prototypes that predate the NSM/meaning-space line entirely.
  Kept for history; nothing here is built or imported by
  `consciousness_transformer` or `quantum_parser`.

## Setup and tests

```bash
cd consciousness_transformer

# CPU-only torch (this is all the cloud jobs and CI use)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'    # numpy, pyyaml, nltk, pytest

python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('omw-2.0')"

python scripts/build_usvs.py   # builds the USVS artifact other scripts/tests expect

pytest                          # the gate suite (slow/full-corpus tests are
                                 # excluded by default; -m slow to include them)
```

Local machines are for unit tests and smokes only. Full-scale training runs
go through a single cloud routine at a time — see `CLAUDE.md`'s ops rules.

## Where to read next

1. **[`GOALS.md`](GOALS.md)** — current status per goal, with evidence and
   next steps. Start here.
2. **[`CLAUDE.md`](CLAUDE.md)** — the working agreement: how agents are
   dispatched, the ops rules, the standing decisions log pointer.
3. **`consciousness_transformer/RESEARCH_NOTES.md`** — the ledger. Entries
   before 2026-10-16 are titled by M-number (see
   `consciousness_transformer/dev/archive/MILESTONE_MAP.md` to place one
   against a goal); entries after are titled by goal.
4. **`consciousness_transformer/MIND_ARCHITECTURE.md`** — the design contract
   the `mind/` build follows.
