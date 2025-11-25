# Quick Start Guide

**Get up and running with the quantum parser in 5 minutes.**

---

## Installation

```bash
# Clone and enter directory
git clone <repository-url>
cd quantum_parser

# Create virtual environment (Python 3.11+)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Unix/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run Your First Parse (English)

```python
from src.parser.quantum_parser import QuantumParser
from src.parser.pos_tagger import tag_english_sentence

# Parse a sentence
parser = QuantumParser('grammars/english.json')
words = tag_english_sentence("The big dog runs quickly")
chart = parser.parse(words)

# Get best parse
best = chart.best_hypothesis()
print(f"Score: {best.score:.3f}")
print(f"Unconsumed: {len(best.get_unconsumed())}")

# Show structure
from src.parser.visualizer import print_hypothesis
print_hypothesis(best)
```

**Expected output**:
```
Score: 0.917
Unconsumed: 1
Root: corre (CLAUSE)

Parse tree:
corre (CLAUSE)
  --SUBJECT--> perro (NOMINAL)
    --DESCRIPTION--> El (DESCRIPTOR)
    --DESCRIPTION--> grande (DESCRIPTOR)
  --MODIFICATION--> rápidamente (MODIFIER)
```

---

## Run Your First Parse (Spanish)

```python
from src.parser.chart_parser import ChartParser
from src.parser.pos_tagger import tag_spanish_sentence

# Parse Spanish
parser = ChartParser('grammars/spanish.json')
words = tag_spanish_sentence("El perro grande corre rápidamente")
chart = parser.parse(words)

best = chart.best_hypothesis()
print(f"Score: {best.score:.3f}")

from src.parser.visualizer import print_hypothesis
print_hypothesis(best)
```

---

## Run Comprehensive Tests

```bash
# English tests (40+ cases)
python tests/test_comprehensive.py

# Spanish tests (40+ cases)
python tests/test_spanish_comprehensive.py

# All tests
python -m pytest tests/
```

**Expected**: All tests should pass ✅

---

## Key Concepts (30-Second Version)

**Parse Trees = Meaning**
- Nodes = words or constituents (NOUN, VERB, CLAUSE)
- Edges = relationships (SUBJECT, OBJECT, DESCRIPTION)
- Tree structure encodes who does what to whom

**Quantum Hypothesis Exploration**
- Parser maintains multiple interpretations simultaneously
- Scores by structural coherence + semantic plausibility
- Returns best parse (or top-K alternatives)

**Language-Agnostic**
- Same parser engine for all languages
- Grammar files define language-specific rules
- English and Spanish fully implemented

**Translation Goal**
- Parse English → semantic vector (structure-based)
- Map to Spanish via shared semantic space
- Generate Spanish from vector
- NO transformers - uses graph composition

---

## File Navigation

**Start here**:
- `README.md` - Vision and overview
- `QUICK_START.md` - This file
- `RECENT_UPDATES.md` - What was just completed

**Technical depth**:
- `ARCHITECTURE.md` - Complete design (79KB)
- `TRANSLATION_DESIGN.md` - Translation system (28KB)
- `LEGACY_CONCEPTS.md` - Original NAOMI design evolution

**Implementation**:
- `src/parser/` - Parser code
- `grammars/` - English and Spanish grammar files
- `tests/` - Comprehensive test suites

---

## Common Tasks

### Add a New Word (English)

Edit `src/parser/pos_tagger.py`:

```python
ENGLISH_WORD_TAG_DICT = {
    # ... existing words ...
    "fast": Tag.ADJ,  # Add your word
}

ENGLISH_WORD_SUBTYPES = {
    # ... existing words ...
    "fast": [],  # Add subtypes if needed
}
```

### Add a New Word (Spanish)

Same file, different dictionaries:

```python
SPANISH_WORD_TAG_DICT = {
    # ... existing words ...
    "rápido": Tag.ADJ,
}

SPANISH_WORD_SUBTYPES = {
    # ... existing words ...
    "rápido": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
}
```

### Modify Grammar Rules

Edit `grammars/english.json` or `grammars/spanish.json`:

```json
{
    "result": "NOMINAL",
    "recursive": false,
    "pattern": {
        "anchor": {"type": "NOUN"},
        "before": [{"type": "DESCRIPTOR", "quantifier": "all"}],
        "after": []
    },
    "connections": [{
        "type": "DESCRIPTION",
        "from": "before[*]",
        "to": "anchor"
    }],
    "consume": ["before"],
    "note": "Attach determiners and adjectives before noun"
}
```

See `GRAMMAR_DESIGN.md` for full rule syntax.

### Debug a Parse

```python
from src.parser.visualizer import print_chart

# Show all hypotheses with scores
print_chart(chart)

# Or just the best parse with details
print_hypothesis(best, show_edges=True, show_features=True)
```

### Check Grammar Validity

```python
from src.parser.dsl import load_grammar

try:
    grammar = load_grammar('grammars/english.json')
    print("✓ Grammar is valid")
except Exception as e:
    print(f"✗ Grammar error: {e}")
```

---

## Next Steps

### To Understand the Project
1. Read `README.md` (vision, overview)
2. Skim `ARCHITECTURE.md` (technical design)
3. Run tests to see it working
4. Read `TRANSLATION_DESIGN.md` (next phase)

### To Continue Development
1. Review `TRANSLATION_DESIGN.md` for roadmap
2. Check `RECENT_UPDATES.md` for current state
3. Start with Milestone 1: Encoder (`src/embeddings/encoder.py`)
4. Follow 6-milestone plan in translation design doc

### To Contribute
1. Run tests: `python tests/test_comprehensive.py`
2. Check existing issues (if any)
3. Read `GRAMMAR_DESIGN.md` for grammar conventions
4. Add tests for new features
5. Submit PR with description

---

## Troubleshooting

**"Module not found" error**:
```bash
# Make sure you're in the right directory
cd quantum_parser

# Check virtual environment is activated
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt
```

**"Grammar file not found"**:
```python
# Use absolute path
from pathlib import Path
grammar_path = Path(__file__).parent / 'grammars' / 'english.json'
parser = QuantumParser(str(grammar_path))
```

**"Tests failing"**:
```bash
# Make sure you have latest code
git pull

# Check if there are uncommitted parser fixes
git status

# Run with verbose output
python tests/test_comprehensive.py -v
```

**"Parse produces no output"**:
- Check if sentence words are in lexicon (`pos_tagger.py`)
- Try simpler sentence first ("The dog runs")
- Enable debug output: `print(chart.hypotheses)`

---

## Documentation Index

| Document | Purpose | Size | Read Time |
|----------|---------|------|-----------|
| `README.md` | Vision, overview, quick start | 12KB | 10 min |
| `QUICK_START.md` | Get running fast (this file) | 6KB | 5 min |
| `ARCHITECTURE.md` | Complete technical design | 79KB | 60 min |
| `TRANSLATION_DESIGN.md` | Translation system design | 28KB | 30 min |
| `RECENT_UPDATES.md` | Latest changes (Nov 2025) | 15KB | 15 min |
| `LEGACY_CONCEPTS.md` | Original NAOMI design | 18KB | 20 min |
| `GRAMMAR_DESIGN.md` | How to write grammars | 3KB | 10 min |
| `STATUS.md` | Development status | 7KB | 5 min |

**Recommended reading order**:
1. `README.md` - Understand the vision
2. `QUICK_START.md` - Run the code (this file)
3. `RECENT_UPDATES.md` - See what's new
4. `TRANSLATION_DESIGN.md` - Next phase plan
5. `ARCHITECTURE.md` - Deep dive (when needed)

---

## Getting Help

**Documentation**:
- Check the doc index above
- All major concepts are documented

**Code examples**:
- `tests/` - 80+ working examples
- Test files show expected usage

**Grammar questions**:
- `GRAMMAR_DESIGN.md` - Rule syntax
- `grammars/*.json` - Working examples

**Translation questions**:
- `TRANSLATION_DESIGN.md` - Complete design
- Implementation not yet coded (that's next!)

---

**You're ready! Start parsing. 🚀**

```python
from src.parser import QuantumParser
from src.parser.pos_tagger import tag_english_sentence

parser = QuantumParser('grammars/english.json')
words = tag_english_sentence("The quantum parser works")
chart = parser.parse(words)

print(f"Success! Score: {chart.best_hypothesis().score:.3f}")
```
