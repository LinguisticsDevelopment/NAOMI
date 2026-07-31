"""``nsm_ct.mind`` — the unified meaning-space consciousness build.

This subpackage realizes the architecture in ``MIND_ARCHITECTURE.md``: a learned
consciousness (state machine) riding on the deterministic meaning-graph substrate.
It *imports* the existing, gate-tested primitives (``meaning_graph``, ``tpr``,
``collapse``, ``serialization``, ``clause_psyche_graph``, ``reasoning_oracle``,
``meaning``) rather than re-implementing them — "not three legacy stacks wired
together", one cohesive layer built on the parts that already work.

Build milestones (see ``MIND_ARCHITECTURE.md``):

* **M0** — schema freeze (:mod:`nsm_ct.mind.schema`) + scaffold.
* **M1** — knowledge layer: durable, disk-persistent LTM graph
  (:mod:`nsm_ct.mind.persistence`), variable-bearing rules + live unification
  (:mod:`nsm_ct.mind.knowledge`).
* **M2+** — the deterministic executor, the learned controller, the two loops.
"""

from __future__ import annotations

from . import schema

__all__ = ["schema"]
