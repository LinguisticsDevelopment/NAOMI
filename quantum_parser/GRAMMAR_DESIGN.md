# Grammar Design Guide

**Principles for Developing Language Grammars**

Version 1.0
Last Updated: 2025-11-22

---

## Philosophy

### Order of Importance Principle

**Core Insight:** Strip decorations until you find Subject-Verb-Object structure.

Grammar rules should be applied from **largest to smallest constituents**:

1. **Clauses** (subordinate, relative, independent)
2. **Verbal Predicates** (verb + complements)
3. **Subjects** (who/what performs action)
4. **Objects** (direct, indirect, prepositional)
5. **Prepositional Phrases**
6. **Adjectives**
7. **Adverbs**
8. **Determiners**
9. **Coordination**
10. **Specifiers**

**Rationale:** Larger structures constrain smaller ones. Finding the main clause helps identify its components.

---

## Rule Categories

### 1. Clauses

**Purpose:** Identify sentence-level and embedded clause structures

**Patterns:**
- Subordinate clauses (because, although, if, when, that)
- Relative clauses (who, which, that)
- Independent clauses (main sentence)

**English Example:**
```
"I know that the dog runs"
→ CLAUSE(independent): I know
  └─ SUBORDINATION_TO: CLAUSE(subordinate): the dog runs
```

**Spanish Considerations:**
- Subjunctive mood in subordinate clauses
- Indicative/subjunctive distinction affects meaning

---

## Language-Specific Guidelines

### English

**Characteristics:**
- Rigid SVO word order
- Minimal agreement (subject-verb number only)
- Heavy use of word order for meaning

**Rule Priorities:**
1. Clause structure
2. Verbal phrases (modal + main verb)
3. Subject-verb-object identification
4. Prepositional phrases (common)
5. Adjectives (pre-nominal)
6. Determiners

**Special Cases:**
- Phrasal verbs: "look up", "give in"
- Auxiliaries: "have been walking"
- Passive voice: "was eaten by"

### Spanish

**Characteristics:**
- Flexible word order (SVO default, but VSO, OVS possible)
- Rich agreement (gender, number across NPs)
- Pro-drop (subject often omitted)
- Clitic pronouns

**Rule Priorities:**
1. Clause structure (indicative vs. subjunctive)
2. Verbal complexes (auxiliary + main, reflexive)
3. Identify subjects (explicit or pro-drop)
4. Objects (including clitic pronouns)
5. Prepositional phrases
6. Adjectives (POST-nominal, gender/number agreement)
7. Determiners (articles, demonstratives with agreement)

**Special Cases:**
- Ser vs. estar (permanent vs. temporary state)
- Clitic pronoun ordering: "se lo doy" = give-him-it
- Gender/number agreement chains
- Subjunctive triggers: desire, doubt, emotion, impersonal expressions

---

## Testing Strategy

For each rule, create test cases:

### Positive Tests (should match)
```
Rule: Adjective-noun (English)
✓ "red car"
✓ "big house"
✓ "very happy dog"
```

### Negative Tests (should NOT match)
```
Rule: Adjective-noun (English)
✗ "runs quickly" (verb-adverb)
✗ "the car" (determiner-noun)
```

### Edge Cases
```
Rule: Coordination
⚠ "apples, oranges, and bananas" (serial comma)
⚠ "either...or" (correlative conjunction)
```

---

## Common Pitfalls

1. **Over-specification:** Making rules too specific limits coverage
2. **Under-specification:** Too general rules create spurious parses
3. **Word order bias:** Assuming fixed order in flexible languages
4. **Missing agreement:** Forgetting subcategory constraints
5. **Ignoring recursion:** Not using recursive rules for lists

---

_See ARCHITECTURE.md and DSL_SPECIFICATION.md for implementation details_
