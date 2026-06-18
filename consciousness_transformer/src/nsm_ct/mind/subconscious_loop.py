"""The subconscious loop — stimulus-independent background processing (M4).

The same controller family, run *without* external input, to do what cannot happen
while reacting — and, crucially, the engine that **matures M3**:

* :meth:`consolidate` — promote settled STM facts into durable LTM.
* :meth:`offline_infer` — forward-chain over LTM and **materialize derived facts**
  (derive-before-asked), so the conscious loop later answers multi-hop queries in
  *fewer* hops (sidestepping the fixed-hop overshoot that hurt M3).
* :meth:`self_train` — replay + freshly-generated episodes train the controller with
  the M3 teacher supervision, **accumulating iterations across rounds** (the scale M3
  lacked), anchored by the symbolic oracle (no self-train collapse).

"The consolidation loop and the learning loop are the same loop."
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..clause_psyche import clause_decode_accuracy
from ..clause_reactor import build_clause_batch
from ..episode import CurriculumGenerator
from ..reasoning_oracle import forward_chain
from ..tpr import TPRCodec
from . import teacher
from .controller import MindController, relation_match
from .controller_losses import combined_loss
from .executor import Executor, _TrivialResolver
from .knowledge import KnowledgeGraph

_REASONING_LEVELS = (9, 10, 11, 12, 13)


class SubconsciousLoop:
    """Background consolidation + offline inference + self-training over one LTM."""

    def __init__(self, ltm: KnowledgeGraph, controller: Optional[MindController] = None, *,
                 codec: Optional[TPRCodec] = None, resolver=None, seed: int = 0,
                 lr: float = 3e-3, total_rounds: int = 10) -> None:
        self.ltm = ltm
        self.controller = controller
        self.codec = codec or ltm.codec
        self.resolver = resolver or _TrivialResolver()
        self.gen = CurriculumGenerator(max_level=13, seed=seed)
        self.replay: List = []                       # experience buffer (episodes)
        self.opt = (torch.optim.AdamW(controller.parameters(), lr=lr)
                    if controller is not None else None)
        self.total_rounds = max(total_rounds, 1)
        self._round = 0

    # -- consolidation (STM -> LTM) ------------------------------------------
    def consolidate(self, stm) -> int:
        """Promote a finished episode's resolved STM facts into durable LTM."""
        ex = Executor(self.ltm, codec=self.codec)
        ex.stm = stm
        return int(ex.consolidate().result)

    # -- offline inference (derive-before-asked) -----------------------------
    def offline_infer(self) -> int:
        """Forward-chain over LTM and materialize derived facts as direct LTM facts.

        After this, a previously multi-hop query (e.g. ``(robin, CAN)`` via an is-a
        chain) is a **direct** fact — the conscious loop answers it in one hop.
        """
        existing = set(self.ltm.facts())
        known, _chain = forward_chain(list(existing), self.ltm.rules())
        added = 0
        for (e, r, v) in known:
            if (e, r, v) not in existing and r != "SUBJECT":
                self.ltm.add_fact(e, r, v)
                added += 1
        return added

    def forget_supersede(self) -> int:
        """Recency/contradiction maintenance in LTM (FALSE-tagged facts already drop
        at read time; reserved for richer policies)."""
        return 0

    # -- self-training / replay (the M3-maturation engine) -------------------
    def self_train(self, episodes_per_round: int = 120, steps: int = 40,
                   replay_keep: int = 200) -> Dict[str, float]:
        """One round of self-training: generate + replay reasoning episodes, train the
        controller with teacher supervision under the cross-round anneal schedule."""
        if self.controller is None:
            return {"loss": 0.0, "rel_match": 0.0, "episodes": 0}
        fresh = [e for e in self.gen.generate(episodes_per_round) if e.level in _REASONING_LEVELS]
        episodes = fresh + self.replay[-replay_keep:]
        if not episodes:
            return {"loss": 0.0, "rel_match": 0.0, "episodes": 0}
        batch = build_clause_batch(episodes, None, self.resolver, self.codec)
        sup_np = teacher.build_supervision(episodes, self.controller.hops)
        sup = {k: torch.from_numpy(v) for k, v in sup_np.items()}
        codebook = self.controller.relation_codebook
        halting = self.controller.halting

        frac = self._round / max(self.total_rounds - 1, 1)
        temperature = 2.0 * (1 - frac) + 0.3 * frac          # soft -> sharp across rounds
        w_rel = 2.0 * (1 - frac) + 0.5 * frac
        w_halt = 1.0 if halting else 0.0

        self.controller.train()
        last = {"loss": 0.0, "rel_match": 0.0}
        for _ in range(steps):
            out = self.controller(batch)
            loss = combined_loss(out, batch, self.controller, sup, codebook,
                                 temperature=temperature, w_rel=w_rel, w_halt=w_halt,
                                 w_prior=0.3 if halting else 0.0)
            self.opt.zero_grad(); loss["total"].backward(); self.opt.step()
            last = {"loss": float(loss["total"].detach()),
                    "rel_match": relation_match(out, sup["rel_targets"], codebook)}
        self.replay = (self.replay + fresh)[-replay_keep:]
        self._round += 1
        return {**last, "episodes": len(episodes)}

    # -- orchestration -------------------------------------------------------
    def run(self, rounds: int, *, episodes_per_round: int = 120, steps: int = 40,
            val=None, verbose: bool = True) -> List[Dict[str, float]]:
        """Run ``rounds`` of self-train → offline-infer, reporting per round."""
        history: List[Dict[str, float]] = []
        val_batch = val_sup = None
        if val is not None and self.controller is not None:
            val_batch = build_clause_batch(val, None, self.resolver, self.codec)
            vs = teacher.build_supervision(val, self.controller.hops)
            val_sup = {k: torch.from_numpy(v) for k, v in vs.items()}
        for _ in range(rounds):
            m = self.self_train(episodes_per_round, steps)
            inferred = self.offline_infer()
            rec = {"round": self._round, "loss": m["loss"], "train_rel_match": m["rel_match"],
                   "ltm_facts": len(self.ltm.facts()), "offline_inferred": inferred}
            if val_batch is not None:
                self.controller.eval()
                with torch.no_grad():
                    vo = self.controller(val_batch)
                rec["val_decode"] = clause_decode_accuracy(vo, val_batch)
                rec["val_optrace_match"] = relation_match(vo, val_sup["rel_targets"],
                                                          self.controller.relation_codebook)
            history.append(rec)
            if verbose:
                extra = (f" val_decode={rec.get('val_decode', float('nan')):.2f}"
                         f" val_optrace={rec.get('val_optrace_match', float('nan')):.2f}"
                         if val_batch is not None else "")
                print(f"round {rec['round']:>3} | loss={rec['loss']:.3f} "
                      f"train_rel_match={rec['train_rel_match']:.2f} "
                      f"| LTM facts={rec['ltm_facts']} +infer {rec['offline_inferred']}{extra}",
                      flush=True)
        return history


__all__ = ["SubconsciousLoop"]
