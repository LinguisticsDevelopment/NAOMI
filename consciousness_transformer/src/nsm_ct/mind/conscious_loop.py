"""The conscious loop — stimulus-driven reactive reasoning (M4).

`text/episode → RECALL from LTM → run the learned controller → answer + a faithful
op-trace → write-back to STM`. The learned vector loop (the M3 ``MindController``)
is the runtime reasoner; the M2 symbolic ``Executor`` is available as a *validator*
(the architecture's "symbolic engine as validator" cross-check) and as the
deterministic durable-recall path for directly-taught facts.

Knowledge enters only via the graph (RECALL from / write-back to the STM/LTM
meaning graphs); the controller's weights are never touched here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..clause_reactor import build_clause_batch
from ..tpr import TPRCodec
from . import ops, teacher
from .controller import MindController, emit_op_trace
from .executor import Executor, _TrivialResolver
from .knowledge import KnowledgeGraph


class ConsciousLoop:
    """Reactive loop over one STM episode + a durable LTM knowledge graph.

    Args:
        ltm: The durable :class:`KnowledgeGraph` (recall source / write-back target).
        controller: The learned :class:`MindController` (runtime reasoner). Optional —
            the deterministic ``recall`` path works without it.
        codec: TPR codec (defaults to the LTM's).
        resolver: Word→meaning resolver for encoding (trivial, dependency-free default).
    """

    def __init__(self, ltm: KnowledgeGraph, *, controller: Optional[MindController] = None,
                 codec: Optional[TPRCodec] = None, resolver=None) -> None:
        self.ltm = ltm
        self.codec = codec or ltm.codec
        self.controller = controller
        self.resolver = resolver or _TrivialResolver()

    # -- durable recall (graph read; no weights) -----------------------------
    def recall(self, subject: str, relation: str):
        """Answer ``(subject, relation)`` from durable LTM — a graph read, no weights.

        Returns ``(value | None, chain)``; ``None`` ⇒ not derivable (abstain).
        """
        return self.ltm.derive(subject, relation)

    # -- learned reactive answer ---------------------------------------------
    def respond(self, episode) -> Dict[str, object]:
        """Run the learned controller on an episode → answer + faithful op-trace."""
        if self.controller is None:
            raise ValueError("respond() needs a learned controller; use recall() otherwise")
        batch = build_clause_batch([episode], None, self.resolver, self.codec)
        self.controller.eval()
        import torch
        with torch.no_grad():
            out = self.controller(batch)
        abstain = bool(out["abstain_prob"][0] >= 0.5)
        idx = int(out["answer_logits"].argmax(-1)[0])
        answer = ops.ABSTAIN if abstain else episode.options[idx]
        query = tuple(episode.meta.get("query", (None, None)))
        trace = emit_op_trace(out, self.controller.relation_codebook, 0, query)
        return {"answer": answer, "abstain": abstain, "trace": trace}

    # -- symbolic validator (cross-check) ------------------------------------
    def validate(self, episode) -> Dict[str, object]:
        """Symbolic ground-truth answer for an episode (the validator cross-check)."""
        return teacher.replay(episode, dim=self.codec.dim)


__all__ = ["ConsciousLoop"]
