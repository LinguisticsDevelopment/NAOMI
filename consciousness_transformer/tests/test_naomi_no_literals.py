"""Owner rule: no hand-written language patterns in code. Greps the naomi package source for quoted
lowercase alphabetic literals (outside docstrings and comments) that are not on a small allow-list of
non-linguistic strings (dict keys, internal tags, file names, library argument values)."""
import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "naomi"
# decoder.py is being replaced by a later job and is not part of this contract.
SKIP_FILES = {"decoder.py"}
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
    # scorer reason kinds / side tags (internal vocabulary of naomi.score)
    "node_dropped", "node_extra", "meaning_shift", "relation_changed", "head_changed", "name_mismatch",
    "subtree_moved", "ungrounded_word", "opposite_meaning", "direct", "satellite_head", "operator_folded_differs", "reference", "candidate",
    # memory sense-choice rule names (internal)
    "no_candidates", "reuse_lemma_sense", "same_synset_in_layer", "hypernym_hyponym_neighbour",
    "first_candidate",
    # file / library vocabulary: wordnet corpus + resource names, checkpoint and data layout, cache keys
    "runs", "arms", "data", "usvs", "ascii", "model", "model_state", "pos_vocab", "role_vocab", "d_axes",
    "hash_buckets", "d_model", "cpu", "torch", "clauses", "roles", "predicate", "relation", "text",
    "sense_candidates", "token_sense_candidates", "tokens", "sentence_index", "policy", "en", "utf",
    # cache-key prefixes and the signal module's timer name
    "tp", "tl", "setitimer",
    # naomi.encoder: tree source tag and checkpoint dict keys (non-linguistic identifiers)
    "encoder", "config", "state", "sub_vocab", "rel_labels", "subs", "meaning", "hashed", "lengths", "node", "stable",
    # naomi.orderer / naomi.referee: result-dict keys, batch keys, side tags read from the grammar table, artifact paths
    'accept', 'after', 'all', 'attested_count', 'attested_count_pos_only', 'attested_reason', 'before', 'by_band', 'changed', 'classes', 'columns', 'decoded_equals_original_order', 'decoded_text', 'dep', 'dep_classes', 'dep_pos', 'dep_token', 'exact_order', 'grammar_reason', 'head', 'head_classes', 'head_pos', 'head_token', 'heads', 'heads_fully_right', 'infos', 'kendall_distance', 'kept_same_head', 'kept_same_head_and_relation', 'licensed_attested', 'licensed_grammar', 'links', 'min_count', 'n_links', 'n_unattested', 'n_unlicensed_grammar', 'naomi_encoder', 'naomi_orderer', 'no_punctuation', 'order_kendall_distance', 'overall', 'pa', 'pairwise_acc', 'parser_agreement', 'parser_head', 'parser_links', 'parser_rel', 'pb', 'ph', 'proposal_head', 'proposal_rel', 'proposal_root', 'reason', 'reasons', 'roundtrip', 'row', 'rows', 'shape', 'share_kept', 'share_kept_head_only', 'side', 'sv', 'with_final_punctuation',
    # naomi.treebank: conversion-table schema keys (the table is data, these are only its field names), result counters,
    # alignment status tags, split names and supervision-source tags; no word of any language
    'accepted', 'base_url', 'bring_up_conj_if_conjunct_has_marker', 'bring_up_rels', 'cc_rel', 'conj_rel', 'coord_lift', 'coordinator_must_follow_first_conjunct', 'copula_swap', 'cycle_dropped', 'default_label', 'dep_feat', 'dep_upos', 'dev', 'drop_punct', 'feats', 'fetch', 'files', 'head_not_node', 'head_upos', 'id', 'kids', 'marker_rel', 'move_conj_if_conjunct_has_rel', 'move_to_copula_rels', 'no_link', 'ok', 'ok_masked', 'op_order', 'ops', 'orig', 'par', 'phrase_head_swap', 'preconj_rel', 'pseudo_asyndeton', 'pseudo_item', 'pseudo_rel', 'root_conflict', 'root_label', 'root_rel', 'rule', 'rules', 'sent_id', 'stream', 'test', 'train', 'treebank', 'ud', 'unknown', 'upos',
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
