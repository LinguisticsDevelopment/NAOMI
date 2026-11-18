"""Round-trip check, reformulated (Mind v2b).  Purpose: the thinker and the decoder must not drift into a private code.

The first formulation read the decoded answer back and compared it with the thinker's tree CUT OUT OF A FULL SENTENCE; that
measures the reader on fragments (a fragment read alone is not the subtree of its sentence), and the control with the right
words in true order already failed 35%.  Here both sides are fragments read by the same reader:

    reread check   = tree score of  read(decoded text)  against  read(the SAME words in true source order)
    order check    = share of related node pairs (parent/child, siblings) whose order in the decoded text differs from
                     their order in the source (text level, no reader)
    substitution   = the decoded words put back into the source sentence's slots, the sentence re-read, scored against
                     the source sentence's own reading

Everything is language-neutral: token equality, token positions and the tree scorer only.  Thresholds are set from the
controls in ``scripts/roundtrip_controls.py`` and declared in ``dev/reports/MIND_V2B.md`` before they touch the model.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from .tree import walk
from .types import Node, Tree


def source_nodes(tree: Tree) -> List[Node]:
    """Token nodes of an answer tree in source order (the order of the sentence it was cut from)."""
    return sorted((n for n, _d, _p in walk(tree.root) if n.token_index is not None), key=lambda n: n.token_index)


def source_text(tree: Tree) -> str:
    return " ".join(n.token for n in source_nodes(tree))


def related_pairs(tree: Tree) -> List[Tuple[int, int]]:
    """Pairs of source positions whose relative order is part of the tree: parent-child and sibling-sibling."""
    out = []
    for n, _d, _p in walk(tree.root):
        if n.token_index is None:
            continue
        kids = [c.token_index for c in n.children if c.token_index is not None]
        out += [(n.token_index, k) for k in kids]
        out += [(a, b) for i, a in enumerate(kids) for b in kids[i + 1:]]
    return out


def order_discord(tree: Tree, decoded_nodes: Sequence[Node]) -> Optional[float]:
    """Share of related pairs (see ``related_pairs``) whose order in ``decoded_nodes`` differs from the source order;
    None when the tree has no related pair (one word).  ``decoded_nodes`` are the orderer's output nodes (they keep
    their source ``token_index``, also in the meaning-only setting)."""
    pos = {n.token_index: i for i, n in enumerate(decoded_nodes) if n.token_index is not None}
    pairs = [(a, b) for a, b in related_pairs(tree) if a in pos and b in pos]
    if not pairs:
        return None
    bad = sum(1 for a, b in pairs if (pos[a] < pos[b]) != (a < b))
    return bad / len(pairs)


def reread_score(read: Callable[[str], Optional[Tree]], scorer, usvs, decoded: str, reference_text: str) -> Optional[float]:
    """Tree-score overall (0..100) of the re-read of ``decoded`` against the re-read of ``reference_text`` (the same
    words in true source order, or the source sentence).  None when either cannot be read."""
    a, b = read(reference_text), read(decoded)
    if a is None or b is None:
        return None
    return float(scorer.score_pair(a, b, usvs).overall)


def substitute(sentence: str, positions: Sequence[int], words: Sequence[str]) -> str:
    """Put ``words`` (decoded order) into the sentence slots ``positions`` (source token indices of the kept nodes,
    sorted); every other token of the sentence stays."""
    toks = sentence.split()
    for p, w in zip(sorted(positions), words):
        if p < len(toks):
            toks[p] = w
    return " ".join(toks)


def passes(score: Optional[float], threshold: float) -> bool:
    return score is not None and score >= threshold
