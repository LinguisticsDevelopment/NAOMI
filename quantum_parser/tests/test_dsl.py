"""Test DSL parser."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser.dsl import load_grammar, DSLParseError
from src.parser.enums import NodeType, ConnectionType


def test_load_test_grammar():
    """Test loading the test grammar file."""
    grammar = load_grammar("grammars/test_grammar.json")

    assert grammar.language == "test"
    assert grammar.version == "1.0"
    assert len(grammar.order) == 2
    assert "adj1" in grammar.order
    assert "noun1" in grammar.order

    print("[OK] Grammar file loaded")


def test_ruleset_parsing():
    """Test that rulesets are parsed correctly."""
    grammar = load_grammar("grammars/test_grammar.json")

    # Check adj1 ruleset
    adj1 = grammar.rulesets["adj1"]
    assert adj1.name == "adj1"
    assert adj1.result == NodeType.DESCRIPTOR
    assert len(adj1.rules) == 1

    # Check rule details
    rule = adj1.rules[0]
    assert rule.result == NodeType.DESCRIPTOR
    assert rule.recursive == False
    assert rule.anchor.type == NodeType.DESCRIPTOR
    assert len(rule.before) == 1
    assert len(rule.after) == 0

    print("[OK] Rulesets parsed correctly")


def test_pattern_elements():
    """Test pattern element parsing."""
    grammar = load_grammar("grammars/test_grammar.json")

    adj1_rule = grammar.rulesets["adj1"].rules[0]

    # Check anchor
    assert adj1_rule.anchor.type == NodeType.DESCRIPTOR
    assert adj1_rule.anchor.quantifier == "one"

    # Check before element
    before_elem = adj1_rule.before[0]
    assert before_elem.type == NodeType.SPECIFIER
    assert before_elem.quantifier == "one"

    print("[OK] Pattern elements parsed correctly")


def test_connections():
    """Test connection specifications."""
    grammar = load_grammar("grammars/test_grammar.json")

    adj1_rule = grammar.rulesets["adj1"].rules[0]

    assert len(adj1_rule.connections) == 1

    conn = adj1_rule.connections[0]
    assert conn.type == ConnectionType.SPECIFICATION
    assert conn.from_ref == "before[0]"
    assert conn.to_ref == "anchor"

    print("[OK] Connections parsed correctly")


def test_quantifiers():
    """Test different quantifier types."""
    grammar = load_grammar("grammars/test_grammar.json")

    noun1_rule = grammar.rulesets["noun1"].rules[0]

    # This rule has quantifier "all"
    before_elem = noun1_rule.before[0]
    assert before_elem.quantifier == "all"

    print("[OK] Quantifiers parsed correctly")


def test_consume_list():
    """Test consume specifications."""
    grammar = load_grammar("grammars/test_grammar.json")

    adj1_rule = grammar.rulesets["adj1"].rules[0]

    assert "before" in adj1_rule.consume
    assert len(adj1_rule.consume) == 1

    print("[OK] Consume list parsed correctly")


def test_invalid_grammar():
    """Test that invalid grammar raises errors."""
    try:
        grammar = load_grammar("grammars/nonexistent.json")
        assert False, "Should have raised DSLParseError"
    except DSLParseError as e:
        assert "not found" in str(e)
        print("[OK] Invalid grammar raises error")


if __name__ == "__main__":
    print("Testing DSL parser...")
    print()

    test_load_test_grammar()
    test_ruleset_parsing()
    test_pattern_elements()
    test_connections()
    test_quantifiers()
    test_consume_list()
    test_invalid_grammar()

    print()
    print("All DSL tests passed! [OK]")
