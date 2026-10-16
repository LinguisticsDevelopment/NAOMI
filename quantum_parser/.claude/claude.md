# Quantum Parser — Claude Code configuration

`quantum_parser` is the deterministic, language-agnostic grammar parser that
`consciousness_transformer` reads text through (one grammar file per
language; the parser itself never trains). It is actively used, not a
historical artifact — but its own top-level docs (`README.md`, `STATUS.md`,
`ARCHITECTURE.md`, `RECENT_UPDATES.md`) predate NAOMI's current mind/
meaning-space line and describe an earlier framing of the project (an
"AGI through embedding" vision, a pre-implementation status). Read them for
parser mechanics, not for current project status or goals.

For current status, goals, and the working agreement, see the repo root:
`GOALS.md`, `README.md`, `CLAUDE.md`.

## Key files (parser-internal)
- **`ARCHITECTURE.md`** — technical design of the parser itself.
- **`GRAMMAR_DESIGN.md`** — how to write/extend a grammar file.
- **`GRAMMAR_UPDATES.md`** — changelog for grammar-file changes; update it
  when you change a grammar.
- **`LEGACY_CONCEPTS.md`** — self-documenting history from the original Go
  NAOMI prototype; background only.

## Development guidelines
- Python, type hints, docstrings, max line length 100.
- Grammar files (`grammars/`): document changes in `GRAMMAR_UPDATES.md`,
  test with example sentences, validate against existing parse outputs.
- Parser core (`src/parser/`): maintain hypothesis-tracking integrity,
  keep backward compatibility with existing grammar files, update tests.
- Data pipeline: handle errors gracefully (skip, don't crash); log
  processing steps.
