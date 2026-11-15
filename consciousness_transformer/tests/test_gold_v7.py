"""Fast tests for scripts/build_gold_v7.py's record function (parser-only, no cache)."""
import importlib.util
import json
from pathlib import Path

import pytest

from naomi.reader import _PUNCT_POS
from naomi.tree import tree_from_dict, tree_to_json, walk

_spec = importlib.util.spec_from_file_location(
    "build_gold_v7", Path(__file__).resolve().parents[1] / "scripts" / "build_gold_v7.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

SENTS = ["the dog ran home .", "mary is in the garden .", "the old man did not see the big bird .",
         "she gave him a red apple .", "a girl and a boy played in the field ."]
KEYS = {"id", "text", "source", "tokens", "tree", "coverage", "diagnostics", "n_tokens", "n_attached",
        "parse_seconds"}


@pytest.fixture(scope="module")
def records():
    r = g.make_reader()
    out = []
    for i, s in enumerate(SENTS):
        st, rec = g.build_record(r, f"t-{i}", s, "test")
        assert st == "ok", s
        out.append(rec)
    return out


def test_schema(records):
    for rec in records:
        assert KEYS <= set(rec)
        assert {"percent", "attached", "attached_by_pos"} <= set(rec["coverage"])
        assert 0 <= rec["n_attached"] <= rec["n_tokens"]


def test_tree_roundtrip(records):
    for rec in records:
        t = tree_from_dict(rec["tree"])
        assert json.loads(tree_to_json(t)) == json.loads(json.dumps(rec["tree"], sort_keys=True))


def test_every_token_once_and_senses_uncommitted(records):
    for rec in records:
        t = tree_from_dict(rec["tree"])
        idx = sorted(n.token_index for n, _, _ in walk(t.root) if n.token_index is not None)
        want = [i for i, (_, p) in enumerate(rec["tokens"]) if p != _PUNCT_POS]
        assert idx == want
        assert all(n.sense is None for n, _, _ in walk(t.root))


def test_holdout_excluded():
    r = g.make_reader()
    st, rec = g.build_record(r, "t-x", SENTS[0], "test", exclude={SENTS[0]})
    assert st == "skipped" and rec is None
