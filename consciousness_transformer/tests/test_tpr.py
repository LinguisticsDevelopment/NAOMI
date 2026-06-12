"""Tests for the TPR meaning-vector prototype (deterministic, unique, recursive)."""

import numpy as np

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.tpr import TPRCodec


def _tree(label, children=()):
    n = ParseNode(label=label)
    for c in children:
        n.children.append(c)
    return ParseTree(root=n)


def _kill_like():
    """Shape of the gold 'kill' explication: a root with ~9 prime children."""
    primes = ["SOMEONE", "SOMETHING", "OTHER", "BECAUSE", "HAPPEN",
              "BODY", "NOT", "LIVE", "AFTER"]
    return _tree("EXPLICATION", [ParseNode(label=p) for p in primes])


def test_deterministic():
    """Same tree → identical vector, across codec instances."""
    a = TPRCodec(dim=64).encode_tree(_kill_like())
    b = TPRCodec(dim=64).encode_tree(_kill_like())
    assert np.allclose(a, b)


def test_unique_and_noncommutative():
    """Different trees → different vectors; swapped roles ≠ same vector."""
    c = TPRCodec(dim=64)
    t1 = _tree("EXPLICATION", [ParseNode(label="GOOD"), ParseNode(label="BAD")])
    t2 = _tree("EXPLICATION", [ParseNode(label="BAD"), ParseNode(label="GOOD")])
    t3 = _tree("EXPLICATION", [ParseNode(label="GOOD"), ParseNode(label="SMALL")])
    v1, v2, v3 = (c.encode_tree(t) for t in (t1, t2, t3))
    assert not np.allclose(v1, v2)  # order matters (role binding)
    assert not np.allclose(v1, v3)  # content matters


def test_single_level_unbind_recovers_every_child():
    """Same-relation roles are exactly orthonormal → every child label recovers
    from the matrix. (Cosines ~0.98-1.0: the small slack is the documented
    cross-relation quasi-orthogonality from the self-label binding.)"""
    c = TPRCodec(dim=64)
    t = _kill_like()
    m = c.encode_matrix(t.root)
    for i, child in enumerate(t.root.children):
        got, score = c.cleanup(c.unbind(m, c.role_vec(i, child.relation)))
        assert got == child.label and score > 0.95


def test_real_width_tree_matrix_roundtrip_perfect():
    """A realistic wide explication (30 children) round-trips perfectly at d=128
    in matrix form (measured: all real gold+DeepNSM trees hit 1.00)."""
    c = TPRCodec(dim=128)
    from nsm_ct.nsm_primes import PRIME_NAMES
    t = _tree("EXPLICATION", [ParseNode(label=PRIME_NAMES[i % len(PRIME_NAMES)])
                              for i in range(30)])
    correct, total = c.decode_guided(c.encode_matrix(t.root), t.root)
    assert correct == total


def test_depth2_recursive_roundtrip_with_cleanup():
    """Depth-2 (one contraction layer) recovers well at low branching (d=128)."""
    c = TPRCodec(dim=128)
    inner = ParseNode(label="LONG", children=[ParseNode(label="BIG"),
                                              ParseNode(label="SIDE")])
    t = _tree("EXPLICATION", [ParseNode(label="SOMEONE"), inner])
    correct, total = c.decode_guided(c.encode_matrix(t.root), t.root)
    assert correct / total >= 0.8  # measured ≈1.0 at branch 2; margin for noise


def test_contracted_vector_is_lossy_but_nontrivial():
    """The fully contracted fixed-d vector keeps partial info (honest limit)."""
    c = TPRCodec(dim=128)
    t = _kill_like()
    m = c.lift(c.encode_tree(t))
    correct, total = c.decode_guided(m, t.root)
    assert 0.4 <= correct / total <= 1.0  # measured ≈0.7 at d=128


def test_exact_growing_tpr_refuses_depth_blowup():
    c = TPRCodec(dim=16)
    deep = _tree("A", [ParseNode(label="B", children=[
        ParseNode(label="C", children=[ParseNode(label="D", children=[
            ParseNode(label="E")])])])])
    try:
        c.encode_exact(deep.root, max_depth=3)
        raised = False
    except MemoryError:
        raised = True
    assert raised
