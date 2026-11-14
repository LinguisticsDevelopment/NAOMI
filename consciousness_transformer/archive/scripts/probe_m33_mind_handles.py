"""M33 gate — the meaning graph's optional USVS handle path, measured THROUGH
the graph API (not the raw handle arrays probe_m31_handles.py compares).

Mirrors ``scripts/probe_m31_handles.py``'s measurement (top-1 accuracy + margin
= top1-top2 cosine, under +5% gaussian noise on the query) but files concepts
into a real ``MeaningGraph`` via ``collapse()`` for both arms:

  (a) DEFAULT  — ``collapse(g, tree, codec, label=w)`` with no ``handle_fn``:
      the pre-M33 label/TPR handle, byte-identical to today's behavior.
  (b) USVS HOOK — ``collapse(g, tree, codec, label=w, handle_fn=...)`` where
      the hook is ``nsm_ct.usvs_bridge.usvs_handle`` closed over ``d``: the
      CONCEPT handle becomes the word's USVS-coordinate projection instead.

Words where USVS has no handle are dropped from the pool before sampling
N=300 (report says how many) — both arms are built over the identical word
set so the comparison isolates the handle-building change, not vocabulary
coverage.

Run:
    python scripts/probe_m33_mind_handles.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.collapse import collapse, dereference_by_vector  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.meaning_graph import MeaningGraph, NodeKind  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402
from nsm_ct.usvs_bridge import usvs_handle  # noqa: E402

POOL_SIZE = 600
N_WORDS = 300
DIM = 256
NOISE_FRAC = 0.05
SEED = 0


def build_graph(words, codec: TPRCodec, resolver: NSMMeaningResolver, *, handle_fn=None):
    """File ``words`` as CONCEPT nodes into a fresh MeaningGraph via collapse()."""
    g = MeaningGraph(codec)
    nids = {}
    for w in words:
        tree = resolver.resolve(w)
        nids[w] = collapse(g, tree, codec, label=w, handle_fn=handle_fn)
    return g, nids


def measure_arm(g: MeaningGraph, nids: dict) -> dict:
    """Top-1 accuracy + margin quartiles under +5% noise, queried via the
    graph's own ``dereference_by_vector`` (CONCEPT-only)."""
    rng = np.random.default_rng(SEED)
    correct = 0
    margins = []
    for w, nid in nids.items():
        h = g.node(nid).handle
        noisy = h + NOISE_FRAC * float(np.linalg.norm(h)) * rng.standard_normal(h.shape).astype(np.float32)
        found, margin = dereference_by_vector(g, noisy, kind_filter=NodeKind.CONCEPT)
        correct += int(found == nid)
        margins.append(margin)
    margins = np.array(margins)
    return {
        "top1_acc": correct / len(nids),
        "median": float(np.median(margins)),
        "q1": float(np.percentile(margins, 25)),
        "q3": float(np.percentile(margins, 75)),
    }


def main() -> None:
    pool = gloss_vocabulary(POOL_SIZE)
    if not pool:
        print("gloss_vocabulary() returned no words (WordNet unavailable?) — aborting.")
        sys.exit(1)

    codec = TPRCodec(dim=DIM)
    resolver = NSMMeaningResolver()

    available = [w for w in pool if usvs_handle(w, DIM) is not None]
    skipped = len(pool) - len(available)
    words = available[:N_WORDS]
    print(f"pool={len(pool)} available={len(available)} skipped={skipped} used={len(words)}")
    if len(words) < N_WORDS:
        print(f"WARNING: only {len(words)} words available, fewer than requested N={N_WORDS}")

    def usvs_fn(word: str):
        return usvs_handle(word, DIM)

    g_default, nids_default = build_graph(words, codec, resolver, handle_fn=None)
    g_hooked, nids_hooked = build_graph(words, codec, resolver, handle_fn=usvs_fn)

    default_stats = measure_arm(g_default, nids_default)
    hooked_stats = measure_arm(g_hooked, nids_hooked)

    rows = [
        ("DEFAULT (label/TPR)", DIM, default_stats),
        ("USVS HOOK", DIM, hooked_stats),
    ]

    header = f"{'arm':<22}{'d':>6}{'top1_acc':>12}{'margin_q1':>12}{'margin_med':>12}{'margin_q3':>12}"
    print(header)
    print("-" * len(header))
    for arm, d, s in rows:
        print(f"{arm:<22}{d:>6}{s['top1_acc']:>12.3f}{s['q1']:>12.4f}{s['median']:>12.4f}{s['q3']:>12.4f}")


if __name__ == "__main__":
    main()
