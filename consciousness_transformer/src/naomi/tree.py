"""Tree helpers: traversal, JSON, human renderings, and the language-neutral graph -> tree attachment.

``attach_graph`` is pure graph traversal over the parser's typed edges. It knows no word, tag or
relation name: edge types are copied through as relation labels. Anything the graph does not attach
is attached to the root as UNATTACHED and written to the diagnostics (never repaired).
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .types import KIND_WORD, REL_ROOT, REL_UNATTACHED, Node, Tree

ROOT_TOKEN = "ROOT"


# ----------------------------------------------------------------------------- traversal
def walk(node: Node, depth: int = 0) -> Iterator[Tuple[Node, int, Optional[Node]]]:
    """Pre-order (node, depth, parent)."""
    stack = [(node, depth, None)]
    while stack:
        n, d, parent = stack.pop()
        yield n, d, parent
        for c in reversed(n.children):
            stack.append((c, d + 1, n))


def token_nodes(tree: Tree) -> List[Node]:
    """All real (non-synthetic) nodes in token order."""
    return sorted((n for n, _, _ in walk(tree.root) if n.token_index is not None), key=lambda n: n.token_index)


def max_depth(tree: Tree) -> int:
    return max(d for _, d, _ in walk(tree.root))


# ----------------------------------------------------------------------------- JSON
def node_to_dict(n: Node) -> dict:
    return {"rel": n.rel, "token": n.token, "token_index": n.token_index, "lemma": n.lemma, "pos": n.pos,
            "kind": n.kind, "senses": list(n.senses), "sense": n.sense, "label": n.label,
            "subtypes": list(n.subtypes), "children": [node_to_dict(c) for c in n.children]}


def node_from_dict(d: dict) -> Node:
    return Node(rel=d["rel"], token=d["token"], token_index=d["token_index"], lemma=d["lemma"], pos=d["pos"],
                kind=d["kind"], senses=list(d["senses"]), sense=d["sense"], label=d["label"],
                subtypes=list(d["subtypes"]), children=[node_from_dict(c) for c in d["children"]])


def tree_to_dict(t: Tree) -> dict:
    return {"sentence": t.sentence, "source": t.source, "diagnostics": [dict(x) for x in t.diagnostics],
            "root": node_to_dict(t.root)}


def tree_from_dict(d: dict) -> Tree:
    return Tree(sentence=d["sentence"], root=node_from_dict(d["root"]), source=d["source"],
                diagnostics=[dict(x) for x in d["diagnostics"]])


def tree_to_json(t: Tree) -> str:
    return json.dumps(tree_to_dict(t), sort_keys=True, ensure_ascii=False)


def tree_from_json(s: str) -> Tree:
    return tree_from_dict(json.loads(s))


# ----------------------------------------------------------------------------- renderings
def render(tree: Tree) -> str:
    """Indented, one node per line: relation, token#index, lemma, pos, kind."""
    lines = []
    for n, d, _ in walk(tree.root):
        idx = "" if n.token_index is None else f"#{n.token_index}"
        lines.append(f"{'  ' * d}{n.rel}  {n.token}{idx}  {n.lemma}  {n.pos}  {n.kind}")
    return "\n".join(lines)


def _compact(n: Node) -> str:
    if not n.children:
        return n.token
    return n.token + "(" + " ".join(f"{c.rel}:{_compact(c)}" for c in n.children) + ")"


def compact(tree: Tree) -> str:
    """One line: ``head(REL:dep REL:dep(REL:dep))``."""
    return _compact(tree.root)


# ----------------------------------------------------------------------------- graph -> tree
def attach_graph(tokens: Sequence[str], pos: Sequence[str], edges: Sequence[Tuple[str, int, int]],
                 roots: Sequence[int], make_node: Callable[[int], Node],
                 skip: Callable[[int], bool]) -> Tuple[Node, List[dict]]:
    """Build the nested tree from the parser's typed edges ``(type, parent, child)``.

    * every token ``i`` with ``skip(i)`` False becomes exactly one node (``make_node(i)``);
    * connected components of the edge set are grown from a top node: edges are followed in the
      parser's own order, first in their given direction (parent -> child), and only when no such
      edge remains against it (child -> parent), so a modifier the parser points *at* its head is
      still attached under that head. The edge type is the relation label either way;
    * an edge that would give a token a second head (or close a cycle) is dropped and recorded;
    * tokens in no edge, and component tops other than the main one, hang from the root as
      UNATTACHED (recorded); several multi-token components get a synthetic ROOT holding them.
    """
    diags: List[dict] = []
    toks = [i for i in range(len(tokens)) if not skip(i)]
    keep = set(toks)
    es = [(t, p, c) for t, p, c in edges if p in keep and c in keep and p != c]
    nbr: Dict[int, set] = {i: set() for i in toks}
    for _t, p, c in es:
        nbr[p].add(c)
        nbr[c].add(p)
    comp_of: Dict[int, int] = {}
    comps: List[List[int]] = []
    for i in toks:
        if i in comp_of:
            continue
        stack, cur = [i], []
        comp_of[i] = len(comps)
        while stack:
            x = stack.pop()
            cur.append(x)
            for y in nbr[x]:
                if y not in comp_of:
                    comp_of[y] = len(comps)
                    stack.append(y)
        comps.append(sorted(cur))
    has_parent = {c for _t, _p, c in es}

    def top_of(comp: List[int]) -> int:
        for r in roots:
            if r in comp_of and comp_of[r] == comp_of[comp[0]]:
                return r
        free = [i for i in comp if i not in has_parent]
        return free[0] if free else comp[0]

    multi = [c for c in comps if len(c) > 1]
    single = [c[0] for c in comps if len(c) == 1]
    main = None
    for r in roots:
        if r in comp_of and len(comps[comp_of[r]]) > 1:
            main = comps[comp_of[r]]
            break
    if main is None:
        main = multi[0] if multi else None
    nodes: Dict[int, Node] = {i: make_node(i) for i in toks}

    def grow(top: int, comp: List[int]) -> None:
        inside = set(comp)
        used: set = set()
        in_tree = {top}
        nodes[top].rel = REL_ROOT
        while True:
            step = None
            for k, (t, p, c) in enumerate(es):
                if k not in used and p in in_tree and c not in in_tree and c in inside:
                    step = (k, t, p, c, False)
                    break
            if step is None:
                for k, (t, p, c) in enumerate(es):
                    if k not in used and c in in_tree and p not in in_tree and p in inside:
                        step = (k, t, c, p, True)
                        break
            if step is None:
                break
            k, t, head, dep, rev = step
            used.add(k)
            nodes[dep].rel = t
            nodes[head].children.append(nodes[dep])
            in_tree.add(dep)
            if rev:
                diags.append({"kind": "reversed_edge", "type": t, "index": dep, "token": tokens[dep]})
        for k, (t, p, c) in enumerate(es):
            if k not in used and p in inside:
                diags.append({"kind": "extra_edge", "type": t, "index": c, "token": tokens[c],
                              "parent": p, "parent_token": tokens[p]})

    if not toks:
        return Node(rel=REL_ROOT, token=ROOT_TOKEN, token_index=None, lemma=ROOT_TOKEN, pos=ROOT_TOKEN,
                    kind=KIND_WORD), diags
    main_top = top_of(main) if main else single[0]
    if main:
        grow(main_top, main)
    else:
        nodes[main_top].rel = REL_ROOT
    others = [c for c in multi if c is not main]
    if others:
        diags.append({"kind": "multiple_components", "count": len(others) + 1})
        root = Node(rel=REL_ROOT, token=ROOT_TOKEN, token_index=None, lemma=ROOT_TOKEN, pos=ROOT_TOKEN,
                    kind=KIND_WORD)
        tops = [(main_top, main)] + [(top_of(c), c) for c in others]
        tops.sort(key=lambda tc: tc[1][0])
        for top, comp in tops:
            if comp is not main:
                grow(top, comp)
                nodes[top].rel = REL_UNATTACHED
                diags.append({"kind": "unattached_component", "index": top, "token": tokens[top],
                              "size": len(comp)})
            root.children.append(nodes[top])
        anchor = root
    else:
        root = anchor = nodes[main_top]
    for i in single:
        if i == main_top and not main:
            continue
        nodes[i].rel = REL_UNATTACHED
        anchor.children.append(nodes[i])
        diags.append({"kind": "unattached_token", "index": i, "token": tokens[i], "pos": pos[i]})
    for n, _d, _p in walk(root):
        n.children.sort(key=lambda c: (c.token_index is None, c.token_index))
    return root, diags
