# Quantum Parser: Universal Semantic Representation System

A language-agnostic parser with quantum hypothesis exploration and WordNet-guided embedding learning for building universal semantic vector spaces.

## Vision

**Grand Goal:** Create Artificial General Intelligence (AGI) through language-agnostic semantic representation.

**Core Insight:** If we can map all language to a universal semantic vector space and provide Turing-complete logical operations within that space, we enable true reasoning over meaning rather than pattern-matching over tokens.

### How It Works

1. **Universal Semantic Space**: All human languages map to the same continuous vector space representing pure meaning
2. **Language as Sampling**: Different languages are just different lexical samplings of this continuous semantic space
3. **Geometric Semantics**:
   - Words → points in space
   - Clauses → trajectories
   - Paragraphs → manifolds
4. **Fixed Logical Operators**: AND, OR, NOT, IMPLIES serve as anchor points in the space
5. **Direct Meaning Manipulation**: Train AI to manipulate meaning directly and learn logical reasoning
6. **Continuous Learning**: Accepting new truths adjusts the geometric relationships
7. **Translation**: Language A → semantic space → Language B
8. **AGI Achievement**: Turing completeness + direct meaning manipulation + ability to express anything through language + training on logical reasoning + continuous learning

## Current Status: MVP

**Goal:** Prove that sentences can be mapped to and from a continuous vector space while preserving semantic meaning.

**Success Criteria:**
- Parse 10,000 dictionary definitions
- Train embeddings with WordNet-guided constraints
- Achieve 80%+ synonym detection accuracy
- Discover word senses matching WordNet ground truth (60%+ F1)

## Key Features

### Quantum Parser
- **Parallel Hypothesis Exploration**: Maintains multiple parse interpretations simultaneously
- **Semantic Scoring**: Ranks parses by coherence in vector space
- **Language-Agnostic**: Grammar files define language-specific rules, parser engine is universal
- **Chart-Based Structure**: Memory-efficient storage of alternative parse trees

### Universal Semantic Space
- **Predefined Dimensions**: 51 fixed basis vectors (27 semantic + 15 grammatical + 9 logical)
- **Learned Embeddings**: 461 dimensions discovered through training
- **Dual-Source Training**: Combines WordNet expert knowledge with parser-extracted relationships
- **Fuzzy Logical Constraints**: Synonyms (~), Antonyms (!~), Hypernyms (→), Hyponyms (←)

### Automatic Sense Discovery
- **Contradiction Detection**: Identifies words with conflicting semantic relationships
- **Node Splitting**: Automatically discovers distinct word senses
- **WordNet Validation**: Compares discovered senses to expert-labeled ground truth

## Quick Start

### Installation

```bash
# Clone the repository
cd quantum_parser

# Create virtual environment (Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download WordNet data
python -c "import nltk; nltk.download('wordnet')"
```

### Basic Usage

#### Parse a Sentence

```python
from src.parser import QuantumParser
from src.parser.dsl import load_grammar

# Load grammar
grammar = load_grammar("grammars/english.json")

# Initialize parser
parser = QuantumParser(
    max_hypotheses=20,
    prune_threshold=0.4,
    score_continuously=True
)

# Parse
sentence = "The big red dog runs quickly"
chart = parser.parse(sentence, grammar)

# Get best parse
best = chart.best_hypothesis()

# Visualize
chart.visualize("output.png")
```

#### Extract Semantic Triples

```python
from src.graph import extract_triples

# Extract relationships from parse tree
triples = extract_triples(best)

for triple in triples:
    print(f"{triple.subject} --[{triple.relation}]--> {triple.object}")

# Output:
# dog --[has-property]--> big
# dog --[has-property]--> red
# dog --[is-subject-of]--> runs
# runs --[has-manner]--> quickly
```

#### Train Embeddings

```python
from src.graph import import_wordnet, build_dual_graph
from src.embedding import train_embeddings

# Load WordNet
wordnet_graph = import_wordnet()

# Parse dictionary definitions
parsed_graph = parse_definitions("data/selected_words.json")

# Build dual graph
dual_graph = build_dual_graph(wordnet_graph, parsed_graph)

# Train embeddings
model = train_embeddings(
    dual_graph,
    embedding_dim=512,
    num_anchors=51,
    num_epochs=200,
    batch_size=256
)

# Save
model.save("data/embeddings/model.pkl")
```

#### Discover Word Senses

```python
from src.graph import detect_contradictions
from src.embedding import split_node

# Load trained model
model = load_model("data/embeddings/model.pkl")

# Detect contradictions
contradictions = detect_contradictions(
    dual_graph,
    model.embeddings,
    threshold=0.5
)

# Split contradictory nodes
for node_info in contradictions:
    new_senses = split_node(
        dual_graph,
        model.embeddings,
        node_info['node'],
        node_info['clusters']
    )
    print(f"Split {node_info['node']} into {len(new_senses)} senses")
```

## Directory Structure

```
quantum_parser/
├── README.md                        # This file
├── ARCHITECTURE.md                  # Complete technical design
├── DSL_SPECIFICATION.md            # Grammar DSL reference
├── GRAMMAR_DESIGN.md               # Grammar development guide
├── GRAMMAR_UPDATES.md              # Changelog for grammar files
├── requirements.txt                # Python dependencies
├── .claude/                        # Claude Code configuration
│   └── claude.md
├── src/
│   ├── parser/                     # Quantum parser implementation
│   │   ├── quantum_parser.py       # Main parsing engine
│   │   ├── dsl.py                  # Grammar DSL parser
│   │   ├── hypothesis.py           # Hypothesis tracking
│   │   └── scorer.py               # Semantic scoring
│   ├── graph/                      # Knowledge graph operations
│   │   ├── knowledge_graph.py      # Graph data structure
│   │   ├── wordnet_import.py       # WordNet loader
│   │   ├── triple_extractor.py     # Parse tree → triples
│   │   └── contradiction_detector.py
│   ├── embedding/                  # Neural network + training
│   │   ├── model.py                # Embedding model
│   │   ├── anchors.py              # Predefined dimensions
│   │   ├── training.py             # Training loop
│   │   ├── constraints.py          # Fuzzy logical constraints
│   │   └── splitting.py            # Sense splitting algorithm
│   ├── utils/                      # Utilities
│   │   ├── visualization.py        # t-SNE, graph plots
│   │   ├── evaluation.py           # Metrics, accuracy tests
│   │   └── logging.py              # Logging configuration
│   └── data_pipeline/              # Data processing
│       ├── wiktionary_parser.py    # Parse Wiktionary XML
│       ├── word_selection.py       # Select top 10K words
│       └── batch_processor.py      # Batch definition parsing
├── grammars/                       # Language grammar files
│   ├── dsl_spec.md                 # DSL documentation
│   ├── english.json                # English grammar rules
│   └── spanish.json                # Spanish grammar rules
├── data/                           # Generated data (gitignored)
│   ├── wordnet/
│   ├── wiktionary/
│   ├── parsed_definitions/
│   └── embeddings/
├── tests/                          # Unit tests
│   ├── test_parser.py
│   ├── test_embedding.py
│   ├── test_splitting.py
│   └── fixtures/
└── notebooks/                      # Jupyter notebooks
    ├── 01_wordnet_exploration.ipynb
    ├── 02_parser_validation.ipynb
    ├── 03_embedding_visualization.ipynb
    └── 04_sense_splitting_analysis.ipynb
```

## Documentation

### For Users
- **README.md** (this file): Overview and quick start
- **grammars/dsl_spec.md**: How to write grammar files

### For Developers
- **ARCHITECTURE.md**: Complete technical design
- **DSL_SPECIFICATION.md**: Grammar DSL reference
- **GRAMMAR_DESIGN.md**: Grammar development principles
- **GRAMMAR_UPDATES.md**: Changes to existing grammars

### Interactive Exploration
- **notebooks/**: Jupyter notebooks for hands-on experimentation

## Architecture Highlights

### Predefined Semantic Dimensions (51 total)

**Semantic Dimensions (27):**
- **Nominals (6)**: determinatory, personal, living, permanence, embodiment, magnitude
- **Scopes (11)**: temporal, frequency, location, manner, extent, reason, attitude, relative, direction, spacialExtent, beneficiary
- **Roles (10)**: fundemental, subject, subjectComp, objects, results, instruments, sources, goals, experiencer, nominal

**Grammatical Dimensions (15):**
- tense, aspect, mood, voice, person, number, gender, case, definiteness, polarity, animacy, countability, degree, transitivity, evidentiality

**Logical Operators (9):**
- AND, OR, XOR, NAND, IF, XIF, NOT, NOR, XNOR

### Quantum Parser Algorithm

1. **Initialize**: Create hypothesis chart with all words as unconsumed nodes
2. **Apply Rules**: For each ruleset (in order of importance):
   - Try all rules on all unconsumed nodes in all hypotheses
   - Create new hypotheses for each successful match
   - Score hypotheses by semantic coherence
   - Prune to top-K (configurable, default: 20)
3. **Return**: Best-scoring hypothesis as primary parse, alternatives available

### Dual-Source Training

```
Loss = 0.6 * WordNet_loss + 0.4 * Parsed_loss + 0.1 * Regularization + 0.05 * Anchor_preservation

Where:
- WordNet_loss: How well embeddings satisfy expert-labeled relationships
- Parsed_loss: How well embeddings satisfy parser-extracted relationships
- Regularization: Prevent overfitting, encourage smooth embeddings
- Anchor_preservation: Keep first 51 dimensions fixed as basis vectors
```

This dual approach teaches the model the PATTERN of how syntax encodes semantics, allowing it to extend beyond WordNet's coverage.

## Development Roadmap

### Phase 1: Foundation (Weeks 1-2) ✓
- [x] Documentation (ARCHITECTURE, DSL_SPEC, GRAMMAR_DESIGN)
- [ ] DSL parser implementation
- [ ] Hypothesis tracking system
- [ ] Quantum parser engine
- [ ] Semantic scorer

### Phase 2: Grammars (Weeks 3-4)
- [ ] English grammar validation and updates
- [ ] Spanish grammar complete rewrite
- [ ] Test suite (80+ example sentences)

### Phase 3: Data Pipeline (Weeks 4-5)
- [ ] WordNet importer
- [ ] Word selection (top 10K)
- [ ] Wiktionary parser
- [ ] Batch definition processor
- [ ] Semantic triple extractor

### Phase 4: Neural Network (Weeks 5-7)
- [ ] Anchor system (51 fixed dimensions)
- [ ] Embedding model
- [ ] Fuzzy logical constraints
- [ ] Dual-source training loop
- [ ] Dimension analysis

### Phase 5: Sense Discovery (Weeks 7-8)
- [ ] Contradiction detection
- [ ] Node splitting algorithm
- [ ] WordNet validation
- [ ] Comprehensive evaluation suite

### Phase 6: Documentation + Polish (Week 8)
- [ ] Jupyter notebooks
- [ ] API documentation
- [ ] Usage examples
- [ ] Results summary

## Contributing

This is a research project. Contributions are welcome, especially:
- Grammar files for new languages
- Improvements to parsing algorithm
- Better semantic triple extraction heuristics
- Novel embedding constraints
- Evaluation metrics

## License

[To be determined]

## Citation

If you use this work, please cite:

```
[Citation to be added after publication]
```

## Contact

[Your contact information]

## Acknowledgments

- **WordNet**: Princeton University's WordNet project for lexical database
- **Wiktionary**: Wikimedia Foundation for dictionary data
- **Inspiration**: This project builds on ideas from Universal Dependencies, FrameNet, PropBank, and semantic role labeling research

---

**Status:** Active development - MVP in progress (as of Week 1)

**Next Milestone:** Complete documentation and begin parser implementation (Week 1-2)
