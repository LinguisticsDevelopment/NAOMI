"""The symbolic teacher — gold supervision for the learned controller (M3).

For each reasoning episode it produces, deterministically:

* the **gold op-trace** (``list[mind.ops.Op]``) — replayable through the M2
  :class:`~nsm_ct.mind.executor.Executor` to the oracle answer (the correctness
  anchor / teacher-correctness gate);
* the **gold focus-chaining relation sequence** — the relations to follow from the
  query entity to the answer, found by **BFS over the streamed memory edges** (rule
  antecedents + consequents + facts), generic across L9-L13;
* the **gold depth** (= path length = the hop at which the answer is reached) and
  the **answerable** flag (unanswerable → abstain / "never produce").

This is the ``gold_chain`` aux-supervision that ``RESEARCH_NOTES`` §0k/§0n/§0p
flagged as "the fallback to switch on next" — now switched on.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import reasoning_oracle as ro
from ..tpr import TPRCodec
from . import ops
from .executor import Executor
from .knowledge import KnowledgeGraph

# Canonical reasoning-relation codebook (fixed index order). Matches how
# ``clause_reactor`` encodes relation atoms: ``codec.filler_vec("rel:" + R)``.
REASONING_RELS: Tuple[str, ...] = (
    "IS_A", "CAN", "CAN_SEE", "PLACE", "CAN_HOLD", "CAN_OPEN", "CAN_REACH", "KIND", "HAS",
)
_REL_INDEX = {r: i for i, r in enumerate(REASONING_RELS)}


def relation_codebook(codec: TPRCodec) -> np.ndarray:
    """``[R, d]`` matrix of the relation atoms, in :data:`REASONING_RELS` order."""
    return np.stack([codec.filler_vec("rel:" + r) for r in REASONING_RELS]).astype(np.float32)


def rel_index(relation: str) -> int:
    """Index of a relation in the codebook (``-1`` if unknown)."""
    return _REL_INDEX.get(relation, -1)


def rules_for_episode(ep) -> List[ro.Rule]:
    """The rules an episode reasons over (reconstructed from its level + meta)."""
    lvl = getattr(ep, "level", 0)
    if lvl in (9, 11):
        (_, _, a), (_, _, x) = ep.meta["rule"]
        return [ro.conditional_rule(a, x)]
    if lvl == 10:
        return [ro.INHERITANCE]
    if lvl == 12:
        return [ro.IS_A_TRANS, ro.INHERITANCE]
    if lvl == 13:
        return [ro.Rule((a,), c, name="mp") for a, c in ep.meta["rules"]]
    return []


def _streamed_edges(ep) -> List[Tuple[str, str, str]]:
    """Every (entity, relation, value) written to memory during ingest.

    Mirrors ``clause_reactor._reasoning_steps``: rule antecedents + consequents
    (all streamed) plus the plain facts. These are the edges focus-chaining walks.
    """
    edges: List[Tuple[str, str, str]] = []
    rule_list = ep.meta.get("rules") or ([ep.meta["rule"]] if ep.meta.get("rule") else [])
    for rule in rule_list:
        for (e, r, v) in rule:
            edges.append((e, r, v))
    for (e, r, v) in ep.meta.get("facts", []):
        edges.append((e, r, v))
    return edges


def gold_relation_path(ep) -> Tuple[List[str], bool]:
    """The relations to follow from the query entity to the gold answer.

    BFS over the streamed edges; returns ``(relations, answerable)``. Unanswerable
    (abstain) episodes return ``([], False)``.
    """
    answerable = bool(getattr(ep, "answerable", True))
    if not answerable:
        return [], False
    qent, _qrel = ep.meta["query"]
    answer = ep.answer_text
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for (e, r, v) in _streamed_edges(ep):
        adj.setdefault(e, []).append((r, v))
    # BFS for the shortest relation path qent -> answer.
    queue = deque([(qent, [])])
    seen = {qent}
    while queue:
        node, rels = queue.popleft()
        for (r, v) in adj.get(node, []):
            if v == answer:
                return rels + [r], True
            if v not in seen:
                seen.add(v)
                queue.append((v, rels + [r]))
    return [], answerable  # answerable but no path found (shouldn't happen on the curriculum)


def gold_op_trace(ep) -> List[ops.Op]:
    """A gold op-trace: PERCEIVE every fact → INFER → RESPOND the query."""
    trace = [ops.Op(ops.PERCEIVE, {"subject": s, "relation": r, "value": v})
             for (s, r, v) in ep.meta.get("facts", [])]
    trace.append(ops.Op(ops.INFER, {}))
    qs, qr = ep.meta["query"]
    trace.append(ops.Op(ops.RESPOND, {"subject": qs, "relation": qr}))
    return trace


def replay(ep, *, dim: int = 64) -> Dict[str, object]:
    """Replay the gold op-trace through the M2 executor (teacher-correctness check)."""
    ltm = KnowledgeGraph(dim=dim)
    ltm.add_rules(rules_for_episode(ep))
    return Executor(ltm).run_trace(gold_op_trace(ep))


def build_supervision(episodes, hops: int) -> Dict[str, "np.ndarray"]:
    """Per-episode supervision tensors aligned to ``hops`` inference steps.

    Returns numpy arrays (caller converts to torch):
      * ``rel_targets`` ``[B, hops]`` long — relation codebook index per hop
        (``-1`` = pad / no supervision past the gold depth or for abstain episodes);
      * ``depth`` ``[B]`` long — gold path length (0 for abstain);
      * ``answerable`` ``[B]`` float — 1 derivable, 0 should-abstain.
    """
    b = len(episodes)
    rel_targets = np.full((b, hops), -1, dtype=np.int64)
    depth = np.zeros(b, dtype=np.int64)
    answerable = np.zeros(b, dtype=np.float32)
    for i, ep in enumerate(episodes):
        rels, ok = gold_relation_path(ep)
        answerable[i] = 1.0 if ok else 0.0
        depth[i] = min(len(rels), hops)
        for k, r in enumerate(rels[:hops]):
            rel_targets[i, k] = rel_index(r)
    return {"rel_targets": rel_targets, "depth": depth, "answerable": answerable}


__all__ = [
    "REASONING_RELS", "relation_codebook", "rel_index", "rules_for_episode",
    "gold_relation_path", "gold_op_trace", "replay", "build_supervision",
]
