# Legacy Concepts from Original NAOMI Design

This document preserves important design concepts from the original NAOMI project (Go implementation) that influenced the quantum parser.

---

## Original Vision (from parent README.md)

**Phase Progress**: Currently at Phase 9 (Multi-Language Support), working toward Phase 3 (Tree→Vector→Sentence)

### The 10 Phases
1. ✅ Tree/DeTree - Parse sentences to trees and back
2. ⚠️ Determine Vectors and Database - Define semantic axes
3. 🔄 Tree→Vector→Sentence - Translation via semantic space (CURRENT)
4. ❌ Analogies, Yes/No - Logical reasoning
5. ❌ New Word Implementation - Learn new vocabulary
6. ❌ Working (Contextual) Memory
7. ❌ Long-Term Memory (stories, ideas, concepts, people)
8. ❌ Conversations affecting Lexicon and Long-Term Memory
9. 🔄 Multi-Language Support (English ✅, Spanish ✅)
10. ❌ Natural Language (more lenient parsing)

### Key Original Insight
> "Neural Net to generate the 'meaning-axises' (need a new name) for the vectors in the starter Lexicon. The general idea is to do this by making it attempt to make all words unique using as few axises as possible."

This became the **51 fixed anchor dimensions** + **461 learned dimensions** in the quantum parser.

---

## Original Parser Design (Go Implementation)

### File Structure (for reference)
```
legacy/parser/                 (Old Go implementation)
├── parser.go                  Main entry point
├── cores/
│   ├── parser.go              Core parsing logic
│   ├── masher.go              Tree combination logic
│   ├── grammarGen.go          Grammar generation
│   ├── frontEnd.go            User interface
│   └── words/                 Word representations
├── assignments.json           Relationship assignments
├── rules.json                 Grammar rules (old format)
└── en.lmm                     Lemmatizer model
```

### assignments.json - Relationship Types

The original design defined relationships between word types using a **before/after assignment table**:

**Relationship Categories** (from assignments.json):
- **Specifiers**: Words that specify/quantify (adverbs modifying verbs)
- **Descriptors**: Words that describe (adjectives before nouns)
- **Complements**: Coordinated elements (X and Y)
- **Connections**: Prepositional and subordinate connections
- **Connected**: Prepositional phrase attachment
- **Relater**: Relative clause markers
- **Subject**: Subject-verb relationship
- **Objects**: Verb-object relationship

**Example relationships**:
```json
"adj": {
    "before": {
        "noun": "Descriptors",      // adj before noun = descriptor
        "verb": "Subject"            // adj before verb = subject
    },
    "after": {
        "verb": "Objects",           // verb after adj = adj is object
        "coord": "Complements"       // adj after coord = coordinated
    }
}
```

This concept evolved into the **ConnectionType enum** in quantum parser:
- Descriptors → DESCRIPTION
- Specifiers → SPECIFICATION
- Subject → SUBJECT
- Objects → OBJECT
- Connections → PREPOSITION
- Complements → COORDINATION

---

## Bucket Parser Concept (Original Go Implementation)

### Philosophy
The original parser used a **"bucket" approach** where words were iteratively combined based on adjacency rules:

1. **Start**: All words in separate buckets
2. **Iterate**: For each pair of adjacent buckets, check if they can combine
3. **Combine**: Merge buckets according to grammar rules
4. **Repeat**: Until only one bucket remains (complete parse)

### Rule Notation (from possibleRules.txt)

Original rule syntax:
```
Order is Important
+ = new (create new node)
all parts right of = become consumed
" = skip
. = subclass
\ = Overwriting
@ = index of
> = if next word is a
>> = next unconsumed word is
< = last word is
<< = last uncomsumed word is
! = not is
*... = get/check *Type until param
```

**Part of Speech Codes**:
```
D = Adv (Descriptor/Modifier)
J = Adj (adJective)
C = Coord (Conjunction)
S = Suboord (Subordinator)
N = Noun
V = Verb
```

### Example Rules from Original Design

```
noun: noun
    | noun* + coordinate + *noun       // RR = recursive right
    | adj + noun                        // RR
    | noun + (normPrep | relatePrep | advPrep) + noun
    | noun + adjPrep + (noun | adj)
    | noun + suboord

verb: verb
    | verb* + coordinate + *verb
    | verb + adv
    | adv + verb
    | verb + advPrep + (noun | verb | adv)
    | noun + verb + noun                 // SVO
    | noun + verb + noun + noun          // SVOO
    | noun + verb + adj                  // SVC
    | noun + verb + noun + normPrep

adj: adj
    | adj* + coordinate + *adv
    | adj + adv

adv: adv
    | adv* + coordinate + *adv
    | adv + adv
```

These evolved into the **ruleset system** in quantum parser with explicit patterns, connections, and consumption rules.

---

## Evolution: Go Parser → Quantum Parser

### What Was Kept

1. **Core Philosophy**: Structure-based parsing, grammatical relationships as first-class citizens
2. **Relationship Types**: Specifiers, Descriptors, Subjects, Objects, Coordination
3. **Recursive Combining**: Continue applying rules until convergence
4. **Consumed Tracking**: Mark words as incorporated into tree
5. **Multi-hypothesis exploration**: Original "masher" tried different combinations

### What Changed

1. **Language**: Go → Python (for ML integration)
2. **Rule Format**: Custom syntax → JSON DSL with explicit schemas
3. **Hypothesis Management**: Implicit → Explicit parallel tracking (quantum superposition)
4. **Scoring**: None → Structural + semantic scoring
5. **Feature System**: Implicit → Explicit SubTypes with agreement checking
6. **Multi-language**: English only → Language-agnostic DSL

### Key Innovation: Quantum Hypothesis Exploration

The original parser tried combinations sequentially. The quantum parser maintains **all viable parses simultaneously** and scores them:

**Original (Go)**:
```go
// Try combination, if fails, backtrack
if canCombine(bucket1, bucket2) {
    newBucket = combine(bucket1, bucket2)
    if parseSucceeds(newBucket) {
        return success
    }
    // else backtrack and try different combination
}
```

**Quantum Parser (Python)**:
```python
# Maintain all hypotheses in parallel
for rule_match in all_possible_matches:
    new_hypothesis = apply_rule(current_hyp, rule_match)
    new_hypothesis.score = score_hypothesis(new_hypothesis)
    hypotheses.append(new_hypothesis)

# Keep top-K, prune low-scoring
hypotheses = prune_to_top_k(hypotheses, k=20)
```

---

## Predefined Semantic Dimensions (Original Concept)

### The "Meaning-Axes" Idea

From original README:
> "Neural Net to generate the 'meaning-axises' (need a new name) for the vectors in the starter Lexicon [The general idea is to do this by making it attempt to make all words unique using as few axises as possible]"

This insight led to the **51 fixed anchor dimensions**:

**Why 51?**
- Hypothesis: There are a small number of fundamental semantic/grammatical distinctions
- Goal: Find the minimal set of axes that uniquely represent all word meanings
- Method: Fix these as anchors, learn the rest

**Categories** (from ARCHITECTURE.md):
1. **Semantic (27 dims)**: Nominal properties, scopes, semantic roles
2. **Grammatical (15 dims)**: Tense, aspect, mood, gender, number, person, case, etc.
3. **Logical (9 dims)**: AND, OR, NOT, XOR, NAND, IF, XIF, NOR, XNOR

### Evolution from Original Concept

**Original idea**: Neural net discovers axes automatically by making words maximally distinct

**Quantum parser approach**: Hybrid system
- **51 dimensions**: Predefined based on linguistic theory (fixed anchors)
- **461 dimensions**: Learned from data (flexible representation)
- **Training**: Dual-source (WordNet + parsed definitions) guides learning

**Why the hybrid approach?**
1. **Interpretability**: Fixed dimensions have clear semantic meaning
2. **Efficiency**: Don't need to rediscover basic grammatical distinctions
3. **Stability**: Anchors provide consistent coordinate system across training
4. **Flexibility**: Learned dimensions capture nuances beyond predefined categories

---

## The "Masher" Concept

### Original Design (from cores/masher.go)

The "masher" was the original combining mechanism that tried to merge parse trees:

**Purpose**: Given multiple parse trees, find the best way to combine them into a single coherent structure.

**Approach**:
1. Try all possible merge points
2. Check grammatical validity at each merge
3. Score resulting tree (originally: prefer fewer nodes)
4. Recursively mash until one tree remains

### Evolution: Masher → Hypothesis Scorer

**Quantum parser equivalent**: The `scorer.py` module

**Improvements**:
1. **Explicit scoring criteria**: Coverage, connectivity, projectivity, balance, semantics
2. **Vector-based semantic scoring**: Use embeddings to assess plausibility
3. **Principled metrics**: Based on parsing theory (projectivity, tree properties)
4. **Continuous scores**: 0.0-1.0 instead of binary valid/invalid

---

## Privacy & Encryption Note

From original parser.go header:
> "If i get it to maximum high level understanding level - make a unique encryption key for every time a new 'mind' is created, and then destroy the key - allows for computer to have a memory without having it's mind read, and for privacy between user and computer to be maintained"

**Concept**: Each AI instance should have private internal representations that can't be read by external parties.

**Relevance to quantum parser**: The semantic vector space could be encrypted with instance-specific keys, preventing:
- External inspection of learned representations
- Memory extraction from deployed models
- Privacy violation through embedding analysis

**Implementation consideration** (future):
- Generate random rotation matrix for each instance
- Apply rotation to all embeddings: `E' = R × E`
- Destroy rotation matrix after initialization
- Model still functions (relationships preserved under rotation)
- But external parties can't interpret vectors without key

---

## Lemmatization (Original Implementation)

**Original approach** (parser.go):
```go
import "github.com/smileart/lemmingo"

lem, _ := lemmingo.New("en.lmm", "", "", false, false, false)
lemma, _, _ := lem.Lemma(token.Text, token.Tag)
```

**Quantum parser approach**:
- Currently: Direct lemma storage in POS tagger dictionaries
- Future: Could integrate lemmatization library for OOV words

**Trade-off**:
- **Original**: Handle any word via lemmatizer
- **Quantum**: Explicit dictionaries (more control, less coverage)
- **Hybrid future**: Dictionary for common words, lemmatizer for OOV

---

## Key Design Principles Preserved

1. **Structure IS Meaning**: Parse trees encode semantic relationships
2. **Minimal Representation**: Use fewest axes possible (51 anchors)
3. **Grammatical Relationships**: Subject, Object, Modifier as first-class
4. **Recursive Combining**: Apply rules until convergence
5. **Language-Agnostic Core**: Grammar files define language-specific rules
6. **Tree ↔ Sentence**: Bidirectional (parse and generate)
7. **Privacy Conscious**: Internal representations can be private
8. **Multi-Language**: Same semantic space for all languages

---

## What's Not Preserved (Intentionally)

1. **Sequential backtracking**: Replaced with parallel hypothesis exploration
2. **Go implementation**: Switched to Python for ML ecosystem
3. **Implicit relationships**: Made explicit with ConnectionType enum
4. **Limited scoring**: Added comprehensive structural + semantic metrics
5. **Single parse**: Now maintain top-K hypotheses

---

## References to Original Files

If you need to understand the original implementation:

**Core logic** (moved to `legacy/` in the 2026-11 integration cleanup):
- `legacy/parser/parser.go` - Main entry point
- `legacy/parser/cores/parser.go` - Core parsing algorithm
- `legacy/parser/cores/masher.go` - Tree combination logic

**Grammar/Rules**:
- `legacy/assignments.json` - Relationship type definitions
- `legacy/rules.json` - Original grammar rules
- `legacy/ref/possibleRules.txt` - Rule notation reference

**Data**:
- `en.lmm` - Lemmatization model
- Various JSON files - Word assignments

---

## Connection to Current Work

The quantum parser is a **direct evolution** of the original NAOMI design:

**Original vision** → **Quantum parser implementation**
- Meaning-axes → 51 anchor dimensions + 461 learned
- Bucket parser → Quantum hypothesis exploration
- Assignments.json → ConnectionType + grammar DSL
- Masher → Hypothesis scorer
- Go → Python
- English only → Multi-language (EN + ES)
- Phase 1-2 → Phase 3 (Tree→Vector→Sentence)

**Next step**: Implement translation (Phase 3) using structure-based semantic vectors as originally envisioned in Phase 2.

---

**Summary**: The quantum parser preserves the core philosophy of the original NAOMI design (structure-based parsing, minimal semantic axes, grammatical relationships as first-class) while adding:
- Explicit parallel hypothesis exploration
- Comprehensive scoring metrics
- Multi-language support with feature agreement
- Language-agnostic grammar DSL
- Path toward semantic vector space (Phase 3)

All the important concepts are either implemented or documented in the roadmap.
