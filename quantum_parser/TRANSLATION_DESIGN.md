# Structure-Based Machine Translation Design

**NAOMI Quantum Parser - Phase 3: Tree → Vector → Sentence**

Version: 1.0
Date: 2025-11-25
Status: Design Document - Ready for Implementation

---

## Core Philosophy: Structure IS Meaning

**Reject**: Token-based transformers (BERT, LaBSE, etc.) that ignore syntactic structure
**Embrace**: Compositional semantics where meaning emerges from grammatical relationships

### Why Not Transformers?

1. **Loss of Structure**: Transformers treat sentences as token bags with attention. Our parser builds rich CLAUSE → PREDICATE → SUBJECT/OBJECT trees that encode WHO does WHAT to WHOM.

2. **Language-Specific Biases**: Pre-trained models learn English-specific patterns. We need language-agnostic semantic representation.

3. **Opaque Meaning**: 768-dim BERT vectors are uninterpretable. We need vectors where dimensions correspond to semantic/grammatical features.

4. **No Compositionality**: Transformers can't explain how "big red dog" means [big ∧ red ∧ dog]. Our parse trees explicitly show DESCRIPTION edges.

### Our Approach: Graph-Based Semantic Vectors

Parse trees are **labeled directed graphs** where:
- **Nodes** = Semantic constituents (nominals, predicates, clauses)
- **Edges** = Grammatical relationships (SUBJECT, OBJECT, DESCRIPTION, etc.)
- **Features** = Morphological properties (gender, number, person, tense)

**Translation pipeline**:
```
English text → English parse tree → Semantic vector
                                         ↓
Spanish text ← Spanish parse tree ← Semantic vector
```

The semantic vector encodes:
1. **Predicate**: What action/state (RUN, EAT, BE)
2. **Arguments**: Semantic roles (AGENT, PATIENT, THEME)
3. **Modifiers**: Properties (SIZE, COLOR, MANNER)
4. **Grammatical**: Tense, aspect, mood, polarity
5. **Logical**: Coordination, subordination, negation

---

## Translation Architecture

### Phase 1: Parse Tree to Semantic Vector (Encoder)

#### Input: Hypothesis (English Parse Tree)
```python
# Example: "The big dog runs quickly"
Hypothesis(
    nodes=[
        Node(type=CLAUSE, value=None),
        Node(type=PREDICATE, value=Word("runs")),
        Node(type=NOMINAL, value=None),
        Node(type=NOUN, value=Word("dog")),
        Node(type=DESCRIPTOR, value=Word("big")),
        Node(type=DESCRIPTOR, value=Word("the")),
        Node(type=MODIFIER, value=Word("quickly"))
    ],
    edges=[
        Edge(SUBJECT, parent=1, child=2),      # runs --SUBJECT--> [nominal]
        Edge(MODIFICATION, parent=1, child=6), # runs --MODIFICATION--> quickly
        Edge(DESCRIPTION, parent=3, child=4),  # dog --DESCRIPTION--> big
        Edge(DESCRIPTION, parent=3, child=5),  # dog --DESCRIPTION--> the
    ]
)
```

#### Output: Semantic Vector (512-dim)

**Vector Structure** (matching existing architecture from ARCHITECTURE.md):
```python
[
    # Fixed Anchor Dimensions (51 dims) - lines 0-50
    # Semantic anchors (27)
    [0-5]:   Nominal properties (determinatory, personal, living, permanence, embodiment, magnitude)
    [6-16]:  Scopes (temporal, frequency, location, manner, extent, reason, attitude, etc.)
    [17-26]: Roles (subject, object, instrument, source, goal, experiencer, etc.)

    # Grammatical anchors (15)
    [27-41]: Grammar (tense, aspect, mood, voice, person, number, gender, case, etc.)

    # Logical anchors (9)
    [42-50]: Logic (AND, OR, XOR, NAND, IF, XIF, NOT, NOR, XNOR)

    # Learned Dimensions (461 dims) - lines 51-511
    [51-511]: Compositional semantics learned from dual-graph training
]
```

#### Encoding Algorithm

```python
def encode_hypothesis(hypothesis: Hypothesis, word_embeddings: Dict) -> np.ndarray:
    """
    Convert parse tree to semantic vector using graph composition.

    Strategy: Compose word embeddings using edge relationships
    """
    # Initialize semantic vector
    semantic_vec = np.zeros(512)

    # Step 1: Find root (highest-level unconsumed node, usually CLAUSE)
    root_idx = find_root(hypothesis)

    # Step 2: Traverse tree from root, composing vectors
    semantic_vec = compose_subtree(hypothesis, root_idx, word_embeddings)

    # Step 3: Apply normalization
    semantic_vec = normalize(semantic_vec)

    return semantic_vec

def compose_subtree(hyp: Hypothesis, node_idx: int, word_embs: Dict) -> np.ndarray:
    """
    Recursively compose semantic vector for subtree.

    Uses edge types to determine composition function:
    - SUBJECT: Add to agent role dimensions
    - OBJECT: Add to patient role dimensions
    - DESCRIPTION: Conjoin (element-wise multiplication for shared features)
    - MODIFICATION: Add to scope dimensions
    - COORDINATION: Average (equal contribution)
    """
    node = hyp.nodes[node_idx]

    # Base case: Leaf node with word
    if node.value:
        vec = word_embs.get(node.value.text.lower(), np.random.randn(512) * 0.01)
        # Copy morphological features to anchor dimensions
        vec = add_morphological_features(vec, node.flags)
        return vec

    # Recursive case: Internal node, compose from children
    child_edges = [e for e in hyp.edges if e.parent == node_idx]

    if not child_edges:
        # No children, use zero vector
        return np.zeros(512)

    # Compose based on edge types
    composed = np.zeros(512)

    for edge in child_edges:
        child_vec = compose_subtree(hyp, edge.child, word_embs)

        if edge.type == ConnectionType.SUBJECT:
            # Subject fills AGENT role
            composed[17:19] += child_vec[17:19]  # fundemental, subject roles
            composed[51:] += 0.5 * child_vec[51:]  # blend learned dims

        elif edge.type == ConnectionType.OBJECT:
            # Object fills PATIENT role
            composed[20:22] += child_vec[20:22]  # objects, results roles
            composed[51:] += 0.5 * child_vec[51:]

        elif edge.type == ConnectionType.DESCRIPTION:
            # Description conjoins (shared features intensify)
            composed = conjoin(composed, child_vec)

        elif edge.type == ConnectionType.MODIFICATION:
            # Modification adds to scope dimensions
            composed[6:17] += child_vec[6:17]  # scope anchors
            composed[51:] += 0.3 * child_vec[51:]

        elif edge.type == ConnectionType.COORDINATION:
            # Coordination averages (equal weight)
            composed[51:] += child_vec[51:] / len(child_edges)

        elif edge.type == ConnectionType.PREPOSITION:
            # Preposition adds to location/direction scopes
            composed[8:11] += child_vec[8:11]  # location, manner, direction
            composed[51:] += 0.4 * child_vec[51:]

        else:
            # Generic: just add
            composed += 0.5 * child_vec

    return composed

def conjoin(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Conjoin two vectors (logical AND operation).

    Semantic interpretation: "big red dog" = features shared by [big ∧ red ∧ dog]
    Implementation: Element-wise multiplication (fuzzy intersection)
    """
    return vec1 * vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def add_morphological_features(vec: np.ndarray, flags: List[SubType]) -> np.ndarray:
    """
    Copy morphological features to anchor dimensions.

    Maps SubTypes to anchor dimension indices:
    - MASCULINE/FEMININE → gender dimension [33]
    - SINGULAR/PLURAL → number dimension [32]
    - FIRST/SECOND/THIRD_PERSON → person dimension [31]
    - PAST/PRESENT/FUTURE → tense dimension [27]
    - etc.
    """
    from src.parser.enums import SubType

    # Gender
    if SubType.MASCULINE in flags:
        vec[33] = 1.0
    elif SubType.FEMININE in flags:
        vec[33] = -1.0

    # Number
    if SubType.SINGULAR in flags:
        vec[32] = 1.0
    elif SubType.PLURAL in flags:
        vec[32] = -1.0

    # Person
    if SubType.FIRST_PERSON in flags:
        vec[31] = 1.0
    elif SubType.SECOND_PERSON in flags:
        vec[31] = 0.0
    elif SubType.THIRD_PERSON in flags:
        vec[31] = -1.0

    # Tense
    if SubType.PAST in flags:
        vec[27] = -1.0
    elif SubType.PRESENT in flags:
        vec[27] = 0.0
    elif SubType.FUTURE in flags:
        vec[27] = 1.0

    return vec
```

### Phase 2: Shared Semantic Space (Bilingual Word Embeddings)

#### Training Strategy: Dual-Graph Alignment

We already have English and Spanish parsing. Now we need:

**1. Monolingual Word Embeddings**
- Train English word embeddings from English parse trees + WordNet
- Train Spanish word embeddings from Spanish parse trees + equivalent Spanish resource
- Use TransE approach from existing architecture (h + r ≈ t)

**2. Cross-Lingual Alignment**
- Use parallel translation pairs (English sentence ↔ Spanish sentence)
- Parse both, extract semantic vectors
- Learn mapping matrix M such that: `M * english_vec ≈ spanish_vec`
- Alternatively: Train with shared concept nodes (e.g., "dog" and "perro" map to same CONCEPT_DOG node)

**3. Implementation**

```python
def train_bilingual_embeddings(
    english_graph: KnowledgeGraph,
    spanish_graph: KnowledgeGraph,
    parallel_pairs: List[Tuple[str, str]],
    config: TrainingConfig
) -> BilingualEmbeddings:
    """
    Train bilingual word embeddings with alignment.

    Args:
        english_graph: English WordNet + parsed definitions
        spanish_graph: Spanish equivalent
        parallel_pairs: [(english_sentence, spanish_sentence), ...]

    Returns:
        BilingualEmbeddings with shared semantic space
    """
    # Step 1: Merge graphs at concept level
    # "dog" (EN) and "perro" (ES) → same CONCEPT node
    unified_graph = create_concept_aligned_graph(english_graph, spanish_graph)

    # Step 2: Train embeddings on unified graph
    # Each word gets vector in shared 512-dim space
    model = GraphEmbeddingModel(
        num_nodes=len(unified_graph.nodes),
        embedding_dim=512,
        num_anchors=51
    )

    # Step 3: Add alignment loss from parallel data
    for eng_sent, spa_sent in parallel_pairs:
        # Parse both
        eng_hyp = parse(eng_sent, english_grammar)
        spa_hyp = parse(spa_sent, spanish_grammar)

        # Encode to vectors
        eng_vec = encode_hypothesis(eng_hyp, model.embeddings)
        spa_vec = encode_hypothesis(spa_hyp, model.embeddings)

        # Add alignment loss: vectors should be similar
        loss = np.linalg.norm(eng_vec - spa_vec) ** 2
        # ... compute gradients and update

    return BilingualEmbeddings(model)
```

**Key Insight**: Since grammatical relationships (SUBJECT, OBJECT, etc.) are language-universal, parse trees with equivalent structure should yield similar semantic vectors even if word order differs.

Example:
- English: "The dog runs" → CLAUSE(PREDICATE(runs) --SUBJECT--> NOMINAL(dog))
- Spanish: "El perro corre" → CLAUSE(PREDICATE(corre) --SUBJECT--> NOMINAL(perro))

Both encode: AGENT=DOG, ACTION=RUN, TENSE=PRESENT

### Phase 3: Semantic Vector to Spanish Parse Tree (Decoder)

This is the **hardest** part. Given a semantic vector, generate a valid Spanish parse tree.

#### Approach 1: Template Retrieval (Initial Implementation)

**Strategy**: Maintain database of Spanish parse templates, retrieve most similar.

```python
class SpanishTemplateDatabase:
    """
    Database of pre-parsed Spanish sentences with semantic vectors.
    """
    def __init__(self):
        self.templates = []  # List of (spanish_hypothesis, semantic_vector, surface_form)

    def add_template(self, spanish_sent: str, spanish_grammar: Grammar):
        """Parse Spanish sentence and store template."""
        words = tag_spanish_sentence(spanish_sent)
        chart = parse(words, spanish_grammar)
        hyp = chart.best_hypothesis()

        vec = encode_hypothesis(hyp, word_embeddings)

        self.templates.append((hyp, vec, spanish_sent))

    def retrieve(self, target_vec: np.ndarray, top_k=5) -> List[Hypothesis]:
        """Find most similar Spanish templates."""
        similarities = [
            (hyp, cosine_similarity(target_vec, template_vec))
            for hyp, template_vec, _ in self.templates
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [hyp for hyp, _ in similarities[:top_k]]

def translate_via_retrieval(
    english_sent: str,
    english_grammar: Grammar,
    spanish_db: SpanishTemplateDatabase,
    word_embeddings: BilingualEmbeddings
) -> str:
    """
    Translate English to Spanish via template retrieval.
    """
    # Step 1: Parse English
    eng_words = tag_english_sentence(english_sent)
    eng_chart = parse(eng_words, english_grammar)
    eng_hyp = eng_chart.best_hypothesis()

    # Step 2: Encode to semantic vector
    semantic_vec = encode_hypothesis(eng_hyp, word_embeddings)

    # Step 3: Retrieve similar Spanish templates
    spanish_candidates = spanish_db.retrieve(semantic_vec, top_k=5)

    # Step 4: Select best candidate (for now, just take top)
    best_spanish_hyp = spanish_candidates[0]

    # Step 5: Realize to surface form
    spanish_sent = realize_sentence(best_spanish_hyp, language="spanish")

    return spanish_sent
```

**Limitations**:
- Only works for sentences structurally similar to templates
- Can't handle novel constructions
- No lexical choice optimization

**Advantages**:
- Simple to implement
- Grammatically correct by construction (uses real parses)
- Interpretable (can see which template was used)

#### Approach 2: Grammar-Guided Generation (Future)

**Strategy**: Use Spanish grammar rules to construct parse tree from semantic vector.

```python
def generate_spanish_hypothesis(
    semantic_vec: np.ndarray,
    spanish_grammar: Grammar,
    word_embeddings: BilingualEmbeddings,
    max_depth=10
) -> Hypothesis:
    """
    Generate Spanish parse tree constrained by grammar rules.

    Algorithm: Greedy search through parse tree space
    1. Start with empty hypothesis
    2. At each step, try applying grammar rules in reverse
    3. Score candidate by vector similarity to target
    4. Select best, repeat until complete
    """
    # Initialize with root CLAUSE node
    current_hyp = Hypothesis(
        nodes=[Node(type=NodeType.CLAUSE, value=None)],
        edges=[],
        consumed=set()
    )

    for depth in range(max_depth):
        # Check if complete
        current_vec = encode_hypothesis(current_hyp, word_embeddings)
        if vector_similarity(current_vec, semantic_vec) > 0.95:
            break

        # Generate candidate expansions
        candidates = []
        for node_idx in range(len(current_hyp.nodes)):
            # Try expanding this node
            for ruleset_name in reversed(spanish_grammar.order):
                # Apply rules in reverse (expand CLAUSE → PREDICATE+NOMINAL, etc.)
                expansions = expand_node(current_hyp, node_idx, spanish_grammar.rulesets[ruleset_name])
                candidates.extend(expansions)

        # Score candidates
        scored = [
            (cand, vector_similarity(encode_hypothesis(cand, word_embeddings), semantic_vec))
            for cand in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select best
        current_hyp = scored[0][0]

    return current_hyp
```

**Challenges**:
- Search space explosion (exponential in tree depth)
- Need heuristics to prune bad paths early
- Lexical choice: which Spanish word for a concept?

**Solutions**:
- Beam search (keep top-K candidates at each step)
- Use grammar rules as hard constraints (only generate valid trees)
- Lexical choice from bilingual embeddings (find Spanish word closest to target concept)

### Phase 4: Surface Realization (Spanish)

Convert Spanish parse tree to surface string.

#### Algorithm: Depth-First Traversal with Language-Specific Ordering

```python
def realize_sentence(hypothesis: Hypothesis, language: str) -> str:
    """
    Convert parse tree to surface sentence.

    Handles:
    - Word order (Spanish post-nominal adjectives)
    - Morphological agreement (gender, number)
    - Clitics (Spanish pronoun placement)
    - Punctuation
    """
    root_idx = find_root(hypothesis)
    tokens = []
    realize_subtree(hypothesis, root_idx, tokens, language)
    return " ".join(tokens)

def realize_subtree(
    hyp: Hypothesis,
    node_idx: int,
    tokens: List[str],
    language: str
):
    """
    Recursively realize subtree in language-specific order.
    """
    node = hyp.nodes[node_idx]

    # Get children by edge type
    children_by_type = defaultdict(list)
    for edge in hyp.edges:
        if edge.parent == node_idx:
            children_by_type[edge.type].append(edge.child)

    # Spanish-specific ordering
    if language == "spanish":
        # Order: SUBJECT < PREDICATE < OBJECT < MODIFICATION
        # But: DESCRIPTION (determiners/adjectives) around noun

        if node.type == NodeType.CLAUSE:
            # Subject-Verb-Object order
            for subj_idx in children_by_type[ConnectionType.SUBJECT]:
                realize_subtree(hyp, subj_idx, tokens, language)

            # Verb (predicate)
            realize_node(node, tokens)

            for obj_idx in children_by_type[ConnectionType.OBJECT]:
                realize_subtree(hyp, obj_idx, tokens, language)

            for mod_idx in children_by_type[ConnectionType.MODIFICATION]:
                realize_subtree(hyp, mod_idx, tokens, language)

        elif node.type == NodeType.NOMINAL:
            # Determiners before noun
            pre_nominal = [
                child_idx for child_idx in children_by_type[ConnectionType.DESCRIPTION]
                if SubType.PRE_NOMINAL in hyp.nodes[child_idx].flags
            ]
            for idx in pre_nominal:
                realize_subtree(hyp, idx, tokens, language)

            # Noun head
            realize_node(node, tokens)

            # Adjectives after noun
            post_nominal = [
                child_idx for child_idx in children_by_type[ConnectionType.DESCRIPTION]
                if SubType.POST_NOMINAL in hyp.nodes[child_idx].flags
            ]
            for idx in post_nominal:
                realize_subtree(hyp, idx, tokens, language)

        else:
            # Default: realize node then children
            realize_node(node, tokens)
            for child_idx in sum(children_by_type.values(), []):
                realize_subtree(hyp, child_idx, tokens, language)

    elif language == "english":
        # English: mostly head-initial
        # Similar logic but different ordering...
        pass

def realize_node(node: Node, tokens: List[str]):
    """
    Realize single node to surface form.

    Handles:
    - Morphological inflection (verb conjugation, noun pluralization)
    - Agreement features from flags
    """
    if node.value:
        # Leaf node with word
        # TODO: Apply inflection based on node.flags
        tokens.append(node.value.text)
    # Internal nodes don't contribute surface form
```

#### Morphological Generation

For full translation, need to inflect words:

```python
class SpanishMorphology:
    """Handle Spanish word inflection."""

    def inflect_verb(self, lemma: str, flags: List[SubType]) -> str:
        """Conjugate Spanish verb based on person, number, tense."""
        # Extract features
        person = get_person(flags)  # FIRST/SECOND/THIRD
        number = get_number(flags)  # SINGULAR/PLURAL
        tense = get_tense(flags)    # PRESENT/PAST/FUTURE

        # Look up in conjugation table or apply rules
        if lemma == "correr" and person == FIRST_PERSON and number == SINGULAR and tense == PRESENT:
            return "corro"
        # ... full conjugation tables for common verbs

    def inflect_adjective(self, lemma: str, flags: List[SubType]) -> str:
        """Inflect Spanish adjective for gender and number."""
        gender = get_gender(flags)
        number = get_number(flags)

        # Simple rule-based (real implementation needs more cases)
        if gender == MASCULINE and number == SINGULAR:
            return lemma  # "grande"
        elif gender == MASCULINE and number == PLURAL:
            return lemma + "s"  # "grandes"
        elif gender == FEMININE and number == SINGULAR:
            if lemma.endswith("o"):
                return lemma[:-1] + "a"  # "rojo" → "roja"
            return lemma
        elif gender == FEMININE and number == PLURAL:
            if lemma.endswith("o"):
                return lemma[:-1] + "as"  # "rojo" → "rojas"
            return lemma + "s"
```

---

## Implementation Roadmap

### Milestone 1: Encoder (2-3 days)
**Goal**: English parse tree → semantic vector

**Files to create**:
- `src/embeddings/encoder.py` - `encode_hypothesis()` function
- `src/embeddings/composition.py` - Tree composition functions
- `tests/test_encoder.py` - Verify encoding consistency

**Success criteria**:
- Same English sentence → same vector (±0.01)
- Similar sentences → similar vectors (cosine > 0.7)
- Different sentences → different vectors (cosine < 0.5)

### Milestone 2: Bilingual Word Embeddings (1 week)
**Goal**: English and Spanish words in shared semantic space

**Files to create**:
- `src/embeddings/bilingual_trainer.py` - Train aligned embeddings
- `src/embeddings/concept_graph.py` - Unified graph with concept nodes
- `data/parallel_corpus.txt` - English-Spanish parallel sentences
- `tests/test_bilingual.py` - Verify alignment

**Data needed**:
- 1000+ English-Spanish parallel sentence pairs
- Can use: Tatoeba, OpenSubtitles, Bible translations

**Success criteria**:
- Translation pairs have cosine similarity > 0.8
- "perro" is nearest neighbor to "dog" (and vice versa)
- Morphological features preserved across languages

### Milestone 3: Template Database (2-3 days)
**Goal**: Spanish sentence → parse → store in database

**Files to create**:
- `src/translator/template_db.py` - SpanishTemplateDatabase class
- `scripts/build_spanish_db.py` - Parse 1000+ Spanish sentences
- `data/spanish_sentences.txt` - Spanish sentence corpus

**Success criteria**:
- Parse 1000+ Spanish sentences without errors
- Store (hypothesis, vector, surface_form) triples
- Retrieve by vector similarity in <1ms

### Milestone 4: Surface Realization (2-3 days)
**Goal**: Spanish parse tree → Spanish sentence

**Files to create**:
- `src/generator/realizer.py` - `realize_sentence()` function
- `src/generator/morphology.py` - Spanish inflection tables
- `tests/test_realizer.py` - Verify correct surface forms

**Success criteria**:
- Spanish parse tree → grammatically correct sentence
- Post-nominal adjectives in correct position
- Gender/number agreement on determiners and adjectives
- Verb conjugation matches subject person/number

### Milestone 5: End-to-End Translation (1-2 days)
**Goal**: English sentence → Spanish sentence

**Files to create**:
- `src/translator/translator.py` - Main `translate()` function
- `scripts/demo_translation.py` - Interactive demo
- `tests/test_translation.py` - Translation quality tests

**Success criteria**:
- Translate 20+ test sentences with reasonable quality
- Preserve meaning (no hallucinations)
- Grammatically correct Spanish
- Handle coordination, prepositional phrases, adjectives

### Milestone 6: Evaluation & Iteration (ongoing)
**Metrics**:
- **Structural preservation**: Does Spanish parse have same shape as English parse?
- **Semantic similarity**: Do English and Spanish vectors match?
- **Grammar correctness**: Is Spanish grammatical?
- **Meaning preservation**: Human evaluation (does translation convey same meaning?)

**Improvements**:
- Expand template database (more sentences)
- Add grammar-guided generation for novel structures
- Optimize lexical choice
- Handle more complex constructions (subordinate clauses, relative clauses)

---

## Expected Translation Quality

### What Will Work Well (Initial Implementation)

✅ **Simple SVO sentences**
- "The dog runs" → "El perro corre"
- "The cat eats food" → "El gato come comida"

✅ **Adjective modification**
- "The big dog" → "El perro grande"
- "The white house" → "La casa blanca"

✅ **Prepositional phrases**
- "The dog in the house" → "El perro en la casa"
- "The book on the table" → "El libro en la mesa"

✅ **Coordination**
- "dogs and cats" → "perros y gatos"
- "big and black" → "grande y negro"

✅ **Adverbial modification**
- "runs quickly" → "corre rápidamente"

### What Will Be Challenging

⚠️ **Lexical ambiguity**
- English "bank" (financial vs. river) → need context to choose Spanish word
- Solution: Use template with best semantic match

⚠️ **Idiomatic expressions**
- "It's raining cats and dogs" → literal translation won't work
- Solution: Add idiom templates to database

⚠️ **Pro-drop vs. explicit subjects**
- Spanish: "Corro" (I run) vs. "Yo corro" (I run - explicit)
- English requires "I run" (subject mandatory)
- Solution: Spanish generation can drop subject if verb conjugation is clear

⚠️ **Tense/aspect differences**
- Spanish has preterite vs. imperfect past tenses
- English simple past maps to both
- Solution: Default mapping, improve with context analysis later

⚠️ **Novel structures not in database**
- If English sentence structure not similar to any Spanish template
- Solution: Fall back to grammar-guided generation (Milestone 7+)

---

## Files to Create (Summary)

```
quantum_parser/
├── src/
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── encoder.py              # encode_hypothesis()
│   │   ├── composition.py          # Tree composition functions
│   │   ├── bilingual_trainer.py    # Train aligned embeddings
│   │   └── concept_graph.py        # Unified concept graph
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── realizer.py             # realize_sentence()
│   │   └── morphology.py           # Spanish inflection
│   └── translator/
│       ├── __init__.py
│       ├── template_db.py          # Template database
│       └── translator.py           # Main translate()
├── data/
│   ├── parallel_corpus.txt         # EN-ES parallel sentences
│   ├── spanish_sentences.txt       # Spanish corpus for templates
│   └── spanish_template_db.pkl     # Serialized template database
├── scripts/
│   ├── build_spanish_db.py         # Build template database
│   ├── train_bilingual_embs.py     # Train embeddings
│   └── demo_translation.py         # Interactive demo
├── tests/
│   ├── test_encoder.py
│   ├── test_bilingual.py
│   ├── test_realizer.py
│   └── test_translation.py
└── TRANSLATION_DESIGN.md           # This document
```

---

## Comparison with Transformer Approach

| Aspect | Transformers (LaBSE, etc.) | Our Approach |
|--------|---------------------------|--------------|
| **Input** | Token sequence | Parse tree (graph) |
| **Encoding** | Attention over tokens | Composition over edges |
| **Vector semantics** | Opaque 768-dim | Interpretable 512-dim with anchors |
| **Structure preservation** | None (bag of tokens) | Explicit (tree isomorphism) |
| **Grammar awareness** | Implicit (learned) | Explicit (grammar rules) |
| **Multilingual** | Pre-trained black box | Concept-aligned graph |
| **Morphology** | Subword tokens | Explicit features (flags) |
| **Training data needed** | Millions of sentences | Thousands (+ grammar rules) |
| **Interpretability** | Low | High (can inspect parse and composition) |
| **Novel structures** | Good (interpolation) | Limited (need templates or generation) |

---

## Next Steps After This Design

1. **Review this document** with user for approval/modifications
2. **Update requirements.txt** with dependencies (numpy, scipy, sklearn for embeddings)
3. **Create stub files** for all modules
4. **Implement Milestone 1** (encoder)
5. **Test encoding** on existing English comprehensive tests
6. **Proceed to Milestone 2** (bilingual embeddings)

---

**End of Translation Design Document**

Ready for implementation once approved.
