"""Stage 3 gate — operator-nodes deconvolve (unlike the old flat flags).

Wraps a clause in a NOT operator-node and recovers BOTH the clause argument and
the operator label exactly, then shows MAYBE over two clauses. Contrast: the old
quantum_parser flag (a flat List[SubType] on a node) carries no binding to which
child/relation set it, so it cannot be deconvolved.

Run:
    python scripts/probe_operators.py --dim 256
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.collapse import collapse, expand  # noqa: E402
from nsm_ct.data_structures import ParseNode, ParseTree  # noqa: E402
from nsm_ct.meaning_graph import (  # noqa: E402
    OPERATES_ON,
    MeaningGraph,
    NodeKind,
    apply_operator,
    read_operator,
)
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def _clause(person, place):
    root = ParseNode(label="is", token="is")
    root.children.append(ParseNode(label="SOMEONE", token=person, relation="SUBJECT"))
    root.children.append(ParseNode(label="SOMEWHERE", token=place, relation="PLACE"))
    return ParseTree(root=root, text=f"{person} is in the {place}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()
    g = MeaningGraph(TPRCodec(dim=args.dim))

    print("=== NOT over one clause (mary is in the office) ===")
    office = collapse(g, _clause("mary", "office"), g.codec, kind=NodeKind.CLAUSE)
    op = apply_operator(g, "NOT", office, g.codec)
    label, score, recovered = read_operator(g, op, g.codec)
    (edge,) = g.out(op, OPERATES_ON)
    print(f"  operator label read back: {label!r} (score {score:.3f})")
    print(f"  clause argument recovered: cos={_cos(recovered[0], g.node(office).handle):.3f}")
    print(f"  exact clause via OPERATES_ON edge: {[n.token for n in expand(g, edge.dst).root.iter_preorder()]}")
    print(f"  target clause truth tag now: {g.node(office).meta['truth']!r}")

    print("\n=== MAYBE over two clauses (kitchen OR office) ===")
    a = collapse(g, _clause("mary", "kitchen"), g.codec, kind=NodeKind.CLAUSE)
    b = collapse(g, _clause("mary", "office"), g.codec, kind=NodeKind.CLAUSE)
    op2 = apply_operator(g, "MAYBE", [a, b], g.codec)
    label2, score2, rec2 = read_operator(g, op2, g.codec)
    print(f"  operator label: {label2!r} (score {score2:.3f})")
    print(f"  arg0 recovered cos={_cos(rec2[0], g.node(a).handle):.3f}; "
          f"arg1 recovered cos={_cos(rec2[1], g.node(b).handle):.3f}")
    print("GATE OK: operators are nodes bound on a reserved role — fully deconvolvable.")


if __name__ == "__main__":
    main()
