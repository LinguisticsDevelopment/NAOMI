"""
DSL Parser - Load and validate grammar JSON files.

Parses the new verbose JSON grammar format into executable Rule objects.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from .enums import NodeType, ConnectionType, SubType, SubCat


@dataclass
class PatternElement:
    """
    Specification for matching a node in a pattern.

    Attributes:
        type: NodeType to match
        quantifier: "one", "all", or "one_or_more"
        subtypes: Required SubTypes (AND logic)
        subcategories: SubCategories that must match anchor
        original_type: If specified, check node's OG field instead
    """
    type: NodeType
    quantifier: str = "one"  # "one", "all", "one_or_more"
    subtypes: List[SubType] = field(default_factory=list)
    subcategories: List[SubCat] = field(default_factory=list)
    original_type: Optional[NodeType] = None

    def __repr__(self) -> str:
        return f"PatternElement({self.type.name}, quantifier={self.quantifier})"


@dataclass
class ConnectionSpec:
    """
    Specification for creating an edge.

    Attributes:
        type: ConnectionType (SUBJECT, OBJECT, etc.)
        from_ref: Reference to source node ("anchor", "before[0]", etc.)
        to_ref: Reference to target node
    """
    type: ConnectionType
    from_ref: str
    to_ref: str

    def __repr__(self) -> str:
        return f"ConnectionSpec({self.type.name}: {self.from_ref} -> {self.to_ref})"


@dataclass
class Rule:
    """
    A single grammar rule.

    Attributes:
        result: NodeType to transform anchor into
        recursive: Restart ruleset after match?
        anchor: Pattern for root node (marked with ? in old DSL)
        before: Patterns for elements left of anchor
        after: Patterns for elements right of anchor
        connections: Edge specifications
        consume: Which elements to mark as consumed
        pull_categories: SubCategories to propagate to anchor
        pop_categories: SubCategories to remove from anchor
        note: Optional comment for documentation
    """
    result: NodeType
    recursive: bool
    anchor: PatternElement
    before: List[PatternElement] = field(default_factory=list)
    after: List[PatternElement] = field(default_factory=list)
    connections: List[ConnectionSpec] = field(default_factory=list)
    consume: List[str] = field(default_factory=list)  # ["before", "after", "anchor"]
    pull_categories: List[SubCat] = field(default_factory=list)
    pop_categories: List[SubCat] = field(default_factory=list)
    note: str = ""

    def __repr__(self) -> str:
        return f"Rule(result={self.result.name}, recursive={self.recursive}, before={len(self.before)}, after={len(self.after)})"


@dataclass
class Ruleset:
    """
    A group of related rules applied together.

    Attributes:
        name: Ruleset identifier
        description: Human-readable explanation
        result: Default result type
        rules: List of rules to try
    """
    name: str
    description: str
    result: NodeType
    rules: List[Rule] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Ruleset({self.name}, {len(self.rules)} rules)"


@dataclass
class Grammar:
    """
    Complete grammar for a language.

    Attributes:
        language: Language identifier
        version: Grammar version
        order: List of ruleset names in application order
        rulesets: Dict mapping name to Ruleset
    """
    language: str
    version: str
    order: List[str]
    rulesets: Dict[str, Ruleset] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Grammar({self.language} v{self.version}, {len(self.rulesets)} rulesets)"


class DSLParseError(Exception):
    """Error parsing grammar file."""
    pass


def load_grammar(filepath: str) -> Grammar:
    """
    Load grammar from JSON file.

    Args:
        filepath: Path to grammar JSON file

    Returns:
        Parsed Grammar object

    Raises:
        DSLParseError: If grammar file is invalid
    """
    path = Path(filepath)

    if not path.exists():
        raise DSLParseError(f"Grammar file not found: {filepath}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DSLParseError(f"Invalid JSON in {filepath}: {e}")

    # Validate top-level structure
    if "order" not in data:
        raise DSLParseError("Grammar must have 'order' field")
    if "rulesets" not in data:
        raise DSLParseError("Grammar must have 'rulesets' field")

    # Parse metadata
    metadata = data.get("metadata", {})
    language = metadata.get("language", path.stem)
    version = metadata.get("version", "1.0")

    # Parse order
    order = data["order"]
    if not isinstance(order, list):
        raise DSLParseError("'order' must be a list of ruleset names")

    # Parse rulesets
    rulesets = {}
    for name, ruleset_data in data["rulesets"].items():
        try:
            ruleset = parse_ruleset(name, ruleset_data)
            rulesets[name] = ruleset
        except Exception as e:
            raise DSLParseError(f"Error parsing ruleset '{name}': {e}")

    # Validate order references
    for ruleset_name in order:
        if ruleset_name not in rulesets:
            raise DSLParseError(f"Order references undefined ruleset: '{ruleset_name}'")

    grammar = Grammar(
        language=language,
        version=version,
        order=order,
        rulesets=rulesets
    )

    return grammar


def parse_ruleset(name: str, data: Dict[str, Any]) -> Ruleset:
    """Parse a ruleset from JSON data."""

    if "description" not in data:
        raise DSLParseError(f"Ruleset '{name}' must have 'description'")
    if "result" not in data:
        raise DSLParseError(f"Ruleset '{name}' must have 'result'")
    if "rules" not in data:
        raise DSLParseError(f"Ruleset '{name}' must have 'rules'")

    description = data["description"]
    result = parse_node_type(data["result"])

    rules = []
    for i, rule_data in enumerate(data["rules"]):
        try:
            rule = parse_rule(rule_data, result)
            rules.append(rule)
        except Exception as e:
            raise DSLParseError(f"Error parsing rule {i} in ruleset '{name}': {e}")

    return Ruleset(
        name=name,
        description=description,
        result=result,
        rules=rules
    )


def parse_rule(data: Dict[str, Any], default_result: NodeType) -> Rule:
    """Parse a rule from JSON data."""

    # Required fields
    if "pattern" not in data:
        raise DSLParseError("Rule must have 'pattern' field")
    if "connections" not in data:
        raise DSLParseError("Rule must have 'connections' field")
    if "consume" not in data:
        raise DSLParseError("Rule must have 'consume' field")

    # Result (can override ruleset default)
    result = parse_node_type(data.get("result", default_result.name))

    # Recursive flag (default: false)
    recursive = data.get("recursive", False)
    if not isinstance(recursive, bool):
        raise DSLParseError("'recursive' must be boolean")

    # Parse pattern
    pattern = data["pattern"]
    if "anchor" not in pattern:
        raise DSLParseError("Pattern must have 'anchor' field")

    anchor = parse_pattern_element(pattern["anchor"])
    before = [parse_pattern_element(elem) for elem in pattern.get("before", [])]
    after = [parse_pattern_element(elem) for elem in pattern.get("after", [])]

    # Parse connections
    connections = [parse_connection_spec(conn) for conn in data["connections"]]

    # Consume list
    consume = data["consume"]
    if not isinstance(consume, list):
        raise DSLParseError("'consume' must be a list")

    valid_consume = ["before", "after", "anchor"]
    for item in consume:
        if item not in valid_consume:
            raise DSLParseError(f"Invalid consume value: '{item}' (must be one of {valid_consume})")

    # Flags (optional)
    flags = data.get("flags", {})
    pull_categories = [parse_subcat(sc) for sc in flags.get("pull_categories", [])]
    pop_categories = [parse_subcat(sc) for sc in flags.get("pop_categories", [])]

    # Note (optional)
    note = data.get("note", "")

    return Rule(
        result=result,
        recursive=recursive,
        anchor=anchor,
        before=before,
        after=after,
        connections=connections,
        consume=consume,
        pull_categories=pull_categories,
        pop_categories=pop_categories,
        note=note
    )


def parse_pattern_element(data: Dict[str, Any]) -> PatternElement:
    """Parse a pattern element from JSON data."""

    if "type" not in data:
        raise DSLParseError("Pattern element must have 'type' field")

    node_type = parse_node_type(data["type"])
    quantifier = data.get("quantifier", "one")

    valid_quantifiers = ["one", "all", "one_or_more"]
    if quantifier not in valid_quantifiers:
        raise DSLParseError(f"Invalid quantifier: '{quantifier}' (must be one of {valid_quantifiers})")

    subtypes = [parse_subtype(st) for st in data.get("subtypes", [])]
    subcategories = [parse_subcat(sc) for sc in data.get("subcategories", [])]

    original_type = None
    if "original_type" in data and data["original_type"] is not None:
        original_type = parse_node_type(data["original_type"])

    return PatternElement(
        type=node_type,
        quantifier=quantifier,
        subtypes=subtypes,
        subcategories=subcategories,
        original_type=original_type
    )


def parse_connection_spec(data: Dict[str, Any]) -> ConnectionSpec:
    """Parse a connection specification from JSON data."""

    if "type" not in data:
        raise DSLParseError("Connection spec must have 'type' field")
    if "from" not in data:
        raise DSLParseError("Connection spec must have 'from' field")
    if "to" not in data:
        raise DSLParseError("Connection spec must have 'to' field")

    conn_type = parse_connection_type(data["type"])
    from_ref = data["from"]
    to_ref = data["to"]

    # Validate references
    valid_refs = ["anchor"]
    for ref in [from_ref, to_ref]:
        if ref != "anchor" and not (ref.startswith("before[") or ref.startswith("after[")):
            raise DSLParseError(f"Invalid node reference: '{ref}' (must be 'anchor', 'before[N]', or 'after[N]')")

    return ConnectionSpec(
        type=conn_type,
        from_ref=from_ref,
        to_ref=to_ref
    )


def parse_node_type(name: str) -> NodeType:
    """Convert string to NodeType enum."""
    try:
        return NodeType[name.upper()]
    except KeyError:
        raise DSLParseError(f"Unknown NodeType: '{name}'")


def parse_connection_type(name: str) -> ConnectionType:
    """Convert string to ConnectionType enum."""
    try:
        return ConnectionType[name.upper()]
    except KeyError:
        raise DSLParseError(f"Unknown ConnectionType: '{name}'")


def parse_subtype(name: str) -> SubType:
    """Convert string to SubType enum."""
    try:
        return SubType[name.upper()]
    except KeyError:
        raise DSLParseError(f"Unknown SubType: '{name}'")


def parse_subcat(name: str) -> SubCat:
    """Convert string to SubCat enum."""
    try:
        return SubCat[name.upper()]
    except KeyError:
        raise DSLParseError(f"Unknown SubCat: '{name}'")
