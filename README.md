# NAOMI

## Start here (new agent? read in this order)

1. This file (`README.md`) — what NAOMI is, repo map, setup.
2. **[`GOALS.md`](GOALS.md)** — current status per goal, with evidence and
   next steps. The source of truth for "where does the project stand."
3. **[`CLAUDE.md`](CLAUDE.md)** — the working agreement: mainline branch,
   how to merge, cloud-job recipe, artifact discipline, test tiers.
4. **`consciousness_transformer/dev/INDEX.md`** — index of every design doc
   and research report, grouped by what it's for.
5. The tail of **`consciousness_transformer/RESEARCH_NOTES.md`** (the
   ledger) — the last 2-3 entries, for what actually happened most recently;
   `GOALS.md` is the distilled status, this is the raw blow-by-blow.

Five commands to get running (full recipe + why each one, below):

```bash
cd consciousness_transformer
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e '.[dev]'
python -c "import nltk; [nltk.download(x) for x in ('wordnet','omw-1.4','omw-2.0')]"
python scripts/build_usvs.py
pytest -m "not slow" -q    # fast tier, ~4 min — see CLAUDE.md for the full tier
# known pre-existing failure: tests/test_quantum_adapter.py::test_two_pps_both_survive_extraction
```

Try it (no training needed; output from an actual run):

```bash
cd consciousness_transformer
python scripts/naomi_run.py teach-ask --text "Mary has a red ball. Tom does not like cats." --ask "Who has a red ball?"
# taught: {'sentences': 2, 'clauses': 2, 'parser': 2, 'encoder': 0, 'none': 0, 'negated': 1}
# Q: Who has a red ball?
# A: mary  (score=5.41)
#    support: have(SUBJECT=mary DESCRIPTION=red OBJECT=ball) [parser]
```

Benchmarks: `python scripts/naomi_run.py bench --dataset mctest --wrong-passage`
then `report --in runs/naomi_bench_mctest.jsonl` (resumable; see `--help`).

---

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

- **`consciousness_transformer/`** — the live project (module map:
  `consciousness_transformer/README.md`).
  - `src/naomi/` — the new minimal read -> remember -> recall loop
    (`reader.py`, `memory.py`, `recall.py`, `mind.py`, `types.py`).
  - `src/nsm_ct/` — the slimmed reader/grounding core (encoder, clause
    extraction, corpus, USVS grounding, meaning graph, TPR).
  - `scripts/` — `naomi_run.py` plus encoder training/gold/eval, USVS build,
    benchmark converters and cheat baselines. `tests/` — the test suite.
  - `archive/` — the curriculum-era code (executor, ops, membrane, `mind/`,
    old decoder, most scripts and tests), moved 2026-11-14 with the same
    relative paths; see `consciousness_transformer/archive/README.md`. The
    pre-refactor tree is also at branch `integration-2026-11`.
  - `runs/` (checkpoints, gold, benchmark outputs; index in
    `runs/ARTIFACTS.md`), `data/`, `dev/` (design docs and reports; see
    `dev/INDEX.md`).
- **`quantum_parser/`** — the deterministic, language-agnostic grammar
  parser `consciousness_transformer` reads text through (one grammar file per
  language). Actively used, not legacy, even though its own docs predate the
  mind/ line and talk about an earlier framing of the project.
- **`legacy/`** — everything that predates the NSM/meaning-space line
  entirely and that nothing current reads or imports (verified by grep
  against `consciousness_transformer/` and `quantum_parser/` before the
  move; the only hits were documentary mentions in
  `quantum_parser/LEGACY_CONCEPTS.md`/`GRAMMAR_UPDATES.md`, not code).
  Moved here as one batch in the 2026-11 integration cleanup:
  - `legacy/parser/`, `legacy/webber/` — early standalone Go prototypes
    (tree/detree parsing, a translation experiment).
  - `legacy/drill/` — scratch data/scripts (a WordNet-from-Prolog dump, a
    spreadsheet, an sqlite conversion helper) from the same early period.
  - `legacy/ref/` — one reference-notes text file (`possibleRules.txt`).
  - `legacy/junk/` — literally named junk. Two small files (`junk.py`,
    `rules.json`) — a scratch POS-tagging exercise against the early rule
    format; a reasonable candidate for outright deletion, left to the
    owner to decide.
  - `legacy/assignments.json`, `legacy/rules.json`,
    `legacy/windows_install.txt` — leftover grammar-rule/config data and a
    Windows setup note from the `parser/`-era Go prototype.

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

See "Start here" at the top. One more pointer not in that list:
**`consciousness_transformer/MIND_ARCHITECTURE.md`** — the design contract
for the mind (its `mind/` implementation is now under `consciousness_transformer/archive/`). Ledger entries before 2026-10-16 are titled by
M-number (see `consciousness_transformer/dev/archive/MILESTONE_MAP.md` to
place one against a goal); entries after are titled by goal.
