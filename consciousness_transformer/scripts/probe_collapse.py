"""Stage 2 gate — collapse/expand losslessness + handle dereferencing margin.

Collapses real meaning trees (place concepts + a couple of L8 clauses), then:
  * asserts EXACT expand round-trip = 100% (losslessness lives in the structure);
  * reports vector-dereference top-1 accuracy + margin median (the number that
    tells Stage 5 whether vector addressing is trustworthy).

Run:
    python scripts/probe_collapse.py --dim 256
    python scripts/probe_collapse.py --dim 512
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.collapse import collapse, dereference_by_vector, expand  # noqa: E402
from nsm_ct.data_structures import ParseNode, ParseTree  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.meaning_graph import MeaningGraph, NodeKind  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

PLACES = ["kitchen", "office", "garden", "bedroom", "hallway", "bathroom", "cellar"]
PEOPLE = ["mary", "john", "sandra", "daniel"]


def _same(a: ParseNode, b: ParseNode) -> bool:
    return (
        a.label == b.label and a.relation == b.relation and a.token == b.token
        and len(a.children) == len(b.children)
        and all(_same(x, y) for x, y in zip(a.children, b.children))
    )


def _clause_tree(person: str, place: str, resolver) -> ParseTree:
    root = ParseNode(label="is", token="is")
    root.children.append(ParseNode(label="SOMEONE", token=person, relation="SUBJECT"))
    place_node = copy.deepcopy(resolver.resolve(place).root)  # full subtree; copy (resolver caches)
    place_node.token, place_node.relation = place, "PLACE"
    root.children.append(place_node)
    return ParseTree(root=root, text=f"{person} is in the {place}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()

    codec = TPRCodec(dim=args.dim)
    g = MeaningGraph(codec)
    resolver = NSMMeaningResolver()

    concepts = []  # (label, nid, original_root)
    for w in PLACES:
        t = resolver.resolve(w)
        concepts.append((w, collapse(g, t, codec, label=w), t.root))
    clauses = []
    for p in PEOPLE[:2]:
        for pl in PLACES[:2]:
            t = _clause_tree(p, pl, resolver)
            clauses.append((f"{p}/{pl}", collapse(g, t, codec, kind=NodeKind.CLAUSE), t.root))
    items = concepts + clauses

    # 1) exact round-trip (the hard gate — correctness lives here, in the structure)
    exact = sum(_same(expand(g, nid).root, root) for _label, nid, root in items)
    print(f"EXACT round-trip: {exact}/{len(items)} "
          f"({100.0 * exact / len(items):.0f}%)  [dim={args.dim}]")

    # 2) vector dereference — split CONCEPT (kept apart by structure/geometry, the
    #    handles vector addressing relies on) from CLAUSE (entity identity is a
    #    *variable token* in the structure, label-collapsed in the handle → expected
    #    to alias; clauses are addressed by graph edges + the exact path, not handles).
    def _deref(group, kind):
        correct, margins = 0, []
        for _label, nid, _root in group:
            found, margin = dereference_by_vector(g, g.node(nid).handle, kind_filter=kind)
            correct += int(found == nid)
            margins.append(margin)
        return correct, np.array(margins)

    c_ok, c_m = _deref(concepts, NodeKind.CONCEPT)
    print(f"CONCEPT deref: {c_ok}/{len(concepts)} top-1; "
          f"margin median={np.median(c_m):.3f} min={c_m.min():.3f} max={c_m.max():.3f}")
    cl_ok, cl_m = _deref(clauses, NodeKind.CLAUSE)
    print(f"CLAUSE  deref: {cl_ok}/{len(clauses)} top-1; "
          f"margin median={np.median(cl_m):.3f} (aliasing on shared SOMEONE label is expected)")

    # 3) CONCEPT dereference under 5% noise (the real vector-addressing test)
    rng = np.random.default_rng(0)
    noisy_ok = 0
    for _label, nid, _root in concepts:
        h = g.node(nid).handle
        noisy = h + 0.05 * float(np.linalg.norm(h)) * rng.standard_normal(h.shape).astype(np.float32)
        found, _m = dereference_by_vector(g, noisy, kind_filter=NodeKind.CONCEPT)
        noisy_ok += int(found == nid)
    print(f"CONCEPT deref (+5% noise): {noisy_ok}/{len(concepts)} top-1")

    assert exact == len(items), "losslessness gate FAILED — structure round-trip must be 100%"
    assert c_ok == len(concepts), "CONCEPT vector addressing should separate distinct concepts"
    print("GATE OK: lossless structure round-trip; concept handles separate; "
          "clause identity routes through the exact path.")


if __name__ == "__main__":
    main()
