"""Owner rule: no hand-written language patterns in code. Greps the naomi package source for quoted
lowercase alphabetic literals (outside docstrings and comments) that are not on a small allow-list of
non-linguistic strings (dict keys, internal tags, file names, library argument values)."""
import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "naomi"
# decoder.py and score.py are being replaced by later jobs and are not part of this contract.
SKIP_FILES = {"decoder.py", "score.py"}
ALLOWED = {
    # internal structure tags / dict keys / dataclass field names
    "word", "name", "parser", "none", "fail", "nodes", "edges", "roots", "tree", "sentence", "source",
    "diagnostics", "root", "rel", "token", "lemma", "pos", "kind", "senses", "sense", "label", "subtypes",
    "children", "index", "type", "parent", "parent_token", "size", "count", "current", "short_term",
    "long_term", "fact", "layer", "provenance", "handle", "sense_rules", "reader", "trees", "by_layer",
    "node_kinds", "sentences", "token_index",
    # diagnostic kinds
    "reversed_edge", "extra_edge", "multiple_components", "unattached_component", "unattached_token",
    "graph_token_mismatch",
    # memory sense-choice rule names (internal)
    "no_candidates", "reuse_lemma_sense", "same_synset_in_layer", "hypernym_hyponym_neighbour",
    "first_candidate",
    # file / library vocabulary: wordnet corpus + resource names, checkpoint and data layout, cache keys
    "runs", "arms", "data", "usvs", "ascii", "model", "model_state", "pos_vocab", "role_vocab", "d_axes",
    "hash_buckets", "d_model", "cpu", "torch", "clauses", "roles", "predicate", "relation", "text",
    "sense_candidates", "token_sense_candidates", "tokens", "sentence_index", "policy", "en", "utf",
    # cache-key prefixes and the signal module's timer name
    "tp", "tl", "setitimer",
}
_WORDISH = re.compile(r"^[a-z][a-z_]+$")


def _docstring_nodes(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def test_no_english_literals_in_naomi_source():
    bad = []
    for f in sorted(PKG.glob("*.py")):
        if f.name in SKIP_FILES:
            continue
        tree = ast.parse(f.read_text())
        doc = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc:
                v = node.value
                if _WORDISH.match(v) and v not in ALLOWED:
                    bad.append(f"{f.name}:{node.lineno}: {v!r}")
    assert not bad, "unexpected lowercase word literals:\n" + "\n".join(bad)
