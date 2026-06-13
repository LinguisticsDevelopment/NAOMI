"""Stage 3 gate — operator-nodes deconvolve (the thing flat flags could not)."""

import numpy as np

from nsm_ct.collapse import collapse, expand
from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.meaning_graph import (
    OPERATES_ON,
    MeaningGraph,
    NodeKind,
    apply_operator,
    read_operator,
)
from nsm_ct.tpr import TPRCodec


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def _clause(person, place):
    root = ParseNode(label="is", token="is")
    root.children.append(ParseNode(label="SOMEONE", token=person, relation="SUBJECT"))
    root.children.append(ParseNode(label="SOMEWHERE", token=place, relation="PLACE"))
    return ParseTree(root=root, text=f"{person} is in the {place}")


def _graph(dim=256):
    g = MeaningGraph(TPRCodec(dim=dim))
    return g


def test_not_operator_recovers_clause_and_reads_label():
    g = _graph()
    office = collapse(g, _clause("mary", "office"), g.codec, kind=NodeKind.CLAUSE)
    op = apply_operator(g, "NOT", office, g.codec)

    label, score, args = read_operator(g, op, g.codec)
    assert label == "NOT" and score > 0.5            # operator label readable
    assert _cos(args[0], g.node(office).handle) > 0.9  # clause argument recovered

    # the exact clause structure is reachable via the OPERATES_ON edge (the truth)
    (edge,) = g.out(op, OPERATES_ON)
    assert edge.dst == office
    assert expand(g, office).root.token == "is"
    # NOT flipped the target clause's truth tag (lossless), for L8 read filtering
    assert g.node(office).meta["truth"] == "FALSE"


def test_maybe_over_two_clauses_recovers_both_arguments():
    g = _graph()
    a = collapse(g, _clause("mary", "kitchen"), g.codec, kind=NodeKind.CLAUSE)
    b = collapse(g, _clause("mary", "office"), g.codec, kind=NodeKind.CLAUSE)
    op = apply_operator(g, "MAYBE", [a, b], g.codec)  # disjunction = MAYBE over both

    label, score, args = read_operator(g, op, g.codec)
    assert label == "MAYBE" and score > 0.5
    assert _cos(args[0], g.node(a).handle) > 0.85
    assert _cos(args[1], g.node(b).handle) > 0.85
    assert {e.dst for e in g.out(op, OPERATES_ON)} == {a, b}
    # MAYBE does not assert falsity
    assert g.node(a).meta.get("truth") != "FALSE"


def test_operator_node_is_distinct_from_a_flag():
    # A flag would live inside the clause vector with no role → unrecoverable.
    # Here the operator is its OWN node bound on a reserved role: removing it
    # leaves the clause untouched, and it decodes back out.
    g = _graph()
    c = collapse(g, _clause("john", "garden"), g.codec, kind=NodeKind.CLAUSE)
    before = g.node(c).handle.copy()
    apply_operator(g, "NOT", c, g.codec)
    assert np.allclose(g.node(c).handle, before)  # clause vector itself is untouched
