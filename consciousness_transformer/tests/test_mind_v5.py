"""Mind v5 (question completion): fast tests. The central one: no code path of the thinker / its training reads the lexicon's
unknown axis, a wh index or any slot label to choose the node to replace."""
import ast
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from naomi import mind_v5 as m5
from naomi.types import REL_ROOT, Node, Tree

CT = Path(__file__).resolve().parents[1]


def _code_names(path):
    """Every identifier, attribute and non-docstring string constant of a source file."""
    tree = ast.parse(Path(path).read_text())
    doc = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            doc.add(id(n.body[0].value))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.append(n.id)
        elif isinstance(n, ast.Attribute):
            out.append(n.attr)
        elif isinstance(n, ast.arg):
            out.append(n.arg)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc:
            out.append(n.value)
    return out


@pytest.mark.parametrize("f", ["src/naomi/mind_v5.py", "scripts/train_mind_v5.py"])
def test_no_code_path_reads_the_unknown_axis_or_a_slot_label(f):
    names = [x.lower() for x in _code_names(CT / f)]
    names_raw = names
    bad = [x for x in names if "unknown" in x or x in ("wh_found", "unknown_idx", "unknown_axis")]
    assert not bad, bad
    # the only place a wh index appears is the constant -1 handed to the shared batch builder, and the model asserts it
    if f.startswith("src"):
        assert "q_wh" in names                                # the assertion in Thinker5.pack
    else:
        assert "q_wh" not in names
    src = (CT / f).read_text()
    assert ".unknown_idx" not in src


def _node(tok, idx, rel="X"):
    return Node(rel=rel, token=tok, token_index=idx, lemma=tok, pos="P", kind="word")


def _qtree():
    root = _node("was", 0, REL_ROOT)
    root.children = [_node("why", 1, "ADV"), _node("angry", 2, "DESC")]
    root.children[1].children = [_node("very", 3, "MOD")]
    return Tree(sentence="why was angry very", root=root)


def test_keep_drop_and_removed_subtree():
    qt = _qtree()
    from naomi import mind_v2 as m2
    qn, par, _ = m2.flat_tree(qt)
    assert m5.removed_by(np.array(par), 2) == {2, 3}
    assert m5.kept_count(np.array(par), 2, [True, True, True, False]) == 2
    tr = Tree(sentence="", root=_node("lied", 0, REL_ROOT))
    out = m5.assemble(qt, qn, 2, tr, [False, False, True, True])      # drop "was" (root) and "why": the filler becomes the root
    assert out.root.token == "lied" and out.root.rel == REL_ROOT and out.root.children == []


def test_filler_and_assemble_keep_provenance_and_replace_subtree():
    qt = _qtree()
    from naomi import mind_v2 as m2
    qn = m2.flat_tree(qt)[0]
    arr = {"nodes": [_node("king", 0, "SUBJ"), _node("lied", 1, "PRED"), _node("the", 2, "DET")], "sent": np.array([0, 0, 0]), "tidx": np.array([2, 1, 0])}
    tr, pv = m5.filler_tree(arr, qn, [("m", 1), ("m", 0), ("m", 2)], [-1, 0, 1])
    assert pv == [("m", 1), ("m", 0), ("m", 2)]
    assert tr.root.token == "lied" and tr.root.rel == REL_ROOT
    assert [c.token for c in tr.root.children] == ["king"] and tr.root.children[0].children[0].token == "the"
    out = m5.assemble(qt, qn, 2, tr)                          # replace "angry" and its subtree ("very")
    toks = []
    stack = [out.root]
    while stack:
        n = stack.pop()
        toks.append(n.token)
        stack += n.children
    assert sorted(toks) == sorted(["was", "why", "lied", "king", "the"])
    angry_slot = [c for c in out.root.children if c.token == "lied"][0]
    assert angry_slot.rel == "DESC"                           # the filler takes the replaced node's relation


def test_slot_index_is_refused_by_the_model():
    model = m5.Thinker5(8, 3, 3)
    b = {"q_wh": torch.tensor([0]), "B": 1}
    with pytest.raises(AssertionError):
        model.pack(b, {})


DATA = CT / "runs" / "mind_v5"


@pytest.mark.skipif(not (DATA / "data_val.pkl").exists(), reason="v5 data not built")
def test_loss_and_decode_run_on_real_pairs():
    from naomi import mind_v1 as m1
    from naomi import mind_v2 as m2
    import json
    torch.manual_seed(0)
    stories = pickle.load(open(DATA / "stories.pkl", "rb"))
    pairs = pickle.load(open(DATA / "data_val.pkl", "rb"))
    ch = [p for p in pairs if p["kind"] == "normal" and p.get("tg")][:3] + [p for p in pairs if p["kind"] == "wrong"][:1]
    for p in ch:
        assert p["wh"] == -1 and float(np.abs(p["qb"][:, 2]).sum()) == 0.0        # the old unknown-value column is zero
    chl = [dict(p, tg=None) if p["kind"] == "wrong" else p for p in ch]
    vocab = json.load(open(DATA / "vocab.json"))
    model = m5.Thinker5(m1.Ctx.get().D, len(vocab["rels"]), len(vocab["poss"]))
    tab = m2.SenseTable()
    b = m2.collate(chl, stories, tab, with_targets=False)
    lp, lpar = model.tf_loss(b, chl)
    assert torch.isfinite(lp) and torch.isfinite(lpar)
    (lp + lpar).backward()
    res = model.decode(b)
    assert len(res) == 4
    for r, p in zip(res, chl):
        assert 0 <= r["j"] < len(p["q"]["parent"])
        assert len(r["pos"]) == len(r["par"]) <= m5.TMAX and len(r["keep"]) == len(p["q"]["parent"])
