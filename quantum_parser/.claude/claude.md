# Quantum Parser - Claude Code Configuration

## Project Overview

Universal semantic representation system building AGI through language-agnostic parsing and WordNet-guided embedding learning.

## Key Files

- **ARCHITECTURE.md**: Complete technical design (read this first!)
- **DSL_SPECIFICATION.md**: Grammar DSL reference
- **GRAMMAR_DESIGN.md**: Grammar development guide
- **GRAMMAR_UPDATES.md**: Changelog for grammar modifications

## Development Guidelines

### Code Style
- Python 3.12+
- Type hints for all functions
- Docstrings (Google style)
- Max line length: 100 characters

### Architecture Principles
1. **Dynamic structure**: Support node splitting/merging during training
2. **Fixed anchors**: First 51 embedding dimensions are predefined
3. **Language-agnostic**: Grammar DSL works for any language
4. **CPU-friendly**: No GPU required

### Testing
- Unit tests for all core functions
- Integration tests for pipelines
- Regression tests for consistency

### When Making Changes

**Grammar Files (grammars/):**
- Document all changes in GRAMMAR_UPDATES.md
- Test with example sentences
- Validate against existing parse outputs

**Parser Core (src/parser/):**
- Maintain hypothesis tracking integrity
- Ensure backward compatibility with grammar files
- Update tests

**Embedding Model (src/embedding/):**
- Never modify anchor dimensions (first 51 dims)
- Test on validation set after changes
- Save checkpoints

**Data Pipeline (src/data_pipeline/):**
- Handle errors gracefully (skip, don't crash)
- Log all processing steps
- Use multiprocessing for batch operations

## Current Status

**Phase**: Documentation complete, beginning implementation

**Next**: Implement quantum parser core (src/parser/)

## Common Commands

```bash
# Setup
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download WordNet
python -c "import nltk; nltk.download('wordnet')"

# Run tests
pytest tests/

# Parse a sentence
python -m src.parser.quantum_parser "The dog runs"

# Train embeddings
python -m src.embedding.training --config config.json

# Evaluate
python -m src.utils.evaluation --model checkpoints/best.pkl
```

## File Organization

```
quantum_parser/
├── src/              # Source code
│   ├── parser/       # Quantum parser implementation
│   ├── graph/        # Knowledge graph operations
│   ├── embedding/    # Neural network + training
│   ├── utils/        # Utilities (visualization, evaluation)
│   └── data_pipeline/# Data processing
├── grammars/         # Language grammar files
├── data/             # Generated data (gitignored)
├── tests/            # Unit and integration tests
└── notebooks/        # Jupyter notebooks for exploration
```

## Important Notes

- **Anchor dimensions**: Always preserve first 51 dims in embeddings
- **Grammar changes**: Must update GRAMMAR_UPDATES.md
- **Large files**: Never commit data/ contents (use .gitignore)
- **Checkpoints**: Save every 10 epochs during training
