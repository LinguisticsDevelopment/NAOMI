"""Tree-aware structure: align a parse onto the model's token stream.

The flat :mod:`~nsm_ct.serialization` scheme linearizes a tree and throws away
hierarchy (and drops any words the parser failed to attach). Instead, this module
keeps **every** token and annotates each with the structure the parser found:

* its **role** — the relation linking its node to the parent (SUBJECT, OBJECT,
  PREPOSITION, …), or the node label, and
* its **depth** in the tree.

These become parallel id streams the model adds as structural embeddings on top
of the word embeddings (see :meth:`ConsciousnessTransformer.step`). Because it is
*additive* over the full token stream, a lossy or noisy parse can only help or be
ignored — no content is ever lost.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .data_structures import ParseTree
from .tokenizer import basic_tokenize

NOROLE = "NOROLE"   # role for positions the parser didn't annotate (id 0)
MAX_DEPTH = 8       # tree depth is clamped to [0, MAX_DEPTH-1]

# Dedicated structure-role vocabulary (NodeType + ConnectionType labels), kept
# separate from the token vocab so role embeddings can be zero-initialized into a
# no-op. NOROLE is id 0. Unknown labels fall back to NOROLE.
_NODE_TYPES = [
    "NIL", "NOUN", "VERBAL", "PREDICATE", "NOMINAL", "CLAUSE", "SPECIFIER",
    "DESCRIPTOR", "MODIFIER", "COORD", "SUBOORD", "PREP", "PREP_SPEC", "PREP_DESC",
    "PREP_MIX", "PP_NOUN", "PP_MIX", "PP_DESC", "PP_SPEC", "PP_VERB", "INTJ", "ROOT",
]
_CONNECTION_TYPES = [
    "SUBJECT", "OBJECT", "INDIRECT_OBJECT", "SUBJECT_COMPLEMENT", "DESCRIPTION",
    "SPECIFICATION", "MODIFICATION", "COMPLEMENT", "COORDINATION", "PREPOSITION",
    "PREPOSITION_FROM", "PREPOSITION_TO", "SUBORDINATION", "SUBORDINATION_FROM",
    "SUBORDINATION_TO", "APPOSITION", "REL",
]
STRUCTURE_LABELS: List[str] = [NOROLE] + _NODE_TYPES + _CONNECTION_TYPES
_ROLE_TO_ID = {name: i for i, name in enumerate(STRUCTURE_LABELS)}
NUM_ROLES = len(STRUCTURE_LABELS)


def role_id(name: str) -> int:
    """Map a structure-role label to its id (unknown / None -> NOROLE = 0)."""
    return _ROLE_TO_ID.get(name, 0)


def _token_annotations(tree: ParseTree) -> List[Tuple[int, str, str, int, List[int]]]:
    """``(source_index, token, role, depth, meaning_prime_ids)`` per token node.

    Sorted by source index (surface order). Nodes without a source index are
    skipped (they can't be aligned to a token). ``meaning_prime_ids`` is the word's
    meaning tree as a (capped) bag of prime ids, empty when no meaning is attached.
    """
    from .thought import meaning_prime_ids  # local import (avoids import cycle)

    out: List[Tuple[int, str, str, int, List[int]]] = []

    def walk(node, depth: int) -> None:
        if node.token is not None and node.index is not None:
            role = node.relation or node.label or NOROLE
            out.append((int(node.index), str(node.token).lower(), role,
                        min(depth, MAX_DEPTH - 1), meaning_prime_ids(node.meaning)))
        for child in node.children:
            walk(child, depth + 1)

    walk(tree.root, 0)
    out.sort(key=lambda x: x[0])
    return out


def align_structure(
    sentence: str, tree: Optional[ParseTree]
) -> Tuple[List[str], List[str], List[int], List[List[int]]]:
    """Align a parse (with optional meaning) onto ``basic_tokenize(sentence)``.

    Returns ``(tokens, roles, depths, meanings)`` of equal length: the surface
    tokens (what the model embeds), per-token role/depth, and per-token meaning
    (a bag of NSM prime ids from the word's meaning tree, ``[]`` when absent).
    Tokens the parser didn't cover get ``NOROLE`` / depth 0 / no meaning. Alignment
    is a greedy surface-text match in source order, so a parser that drops or
    reorders a few words degrades gracefully rather than corrupting the stream.
    """
    tokens = basic_tokenize(sentence)
    roles = [NOROLE] * len(tokens)
    depths = [0] * len(tokens)
    meanings: List[List[int]] = [[] for _ in tokens]
    if tree is None:
        return tokens, roles, depths, meanings

    anns = _token_annotations(tree)
    j = 0  # cursor into annotations
    for i, tok in enumerate(tokens):
        k = j
        while k < len(anns) and anns[k][1] != tok:
            k += 1
        if k < len(anns):
            roles[i] = anns[k][2]
            depths[i] = anns[k][3]
            meanings[i] = anns[k][4]
            j = k + 1
    return tokens, roles, depths, meanings
