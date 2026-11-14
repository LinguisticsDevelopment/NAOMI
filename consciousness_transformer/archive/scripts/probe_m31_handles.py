"""M31 dereference gate — status-quo (label/TPR) handles vs. USVS handles.

Mirrors the CONCEPT dereference measurement in ``scripts/probe_collapse.py``
(top-1 accuracy + margin = top1-top2 cosine, under +5% gaussian noise on the
query) but runs it over N=500 content words drawn from the gloss vocabulary,
comparing two ways of building a CONCEPT node's lossy handle:

  (a) STATUS QUO — the label-based TPR handle ``probe_collapse`` uses today:
      ``codec.contract(codec.encode_matrix(resolver.resolve(word).root))``.
  (b) USVS — ``nsm_ct.usvs_bridge.usvs_handle(word, d)``, a deterministic
      projection of the word's USVS coordinate.

Words where either handle is unavailable are dropped from the pool before
sampling N=500 (report says how many).

Run:
    python scripts/probe_m31_handles.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.collapse import dereference_by_vector  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.meaning_graph import MeaningGraph, NodeKind  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402
from nsm_ct.usvs_bridge import usvs_handle  # noqa: E402

POOL_SIZE = 600
N_WORDS = 500
DIMS = (256, 512)
NOISE_FRAC = 0.05
SEED = 0


def status_quo_handle(word: str, codec: TPRCodec, resolver: NSMMeaningResolver):
    """The same label/TPR handle ``probe_collapse`` uses today."""
    tree = resolver.resolve(word)
    return codec.contract(codec.encode_matrix(tree.root))


def build_index(words, handles, codec: TPRCodec):
    """Build a CONCEPT-only MeaningGraph indexed by precomputed ``handles``."""
    g = MeaningGraph(codec)
    nids = []
    for w, h in zip(words, handles):
        nid = g.add_node(NodeKind.CONCEPT, h, label=w)
        nids.append(nid)
    return g, nids


def measure_arm(words, handles, codec: TPRCodec):
    """Top-1 accuracy + margin quartiles under +5% noise, CONCEPT-only index."""
    g, nids = build_index(words, handles, codec)
    rng = np.random.default_rng(SEED)
    correct = 0
    margins = []
    for nid in nids:
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

    # One TPRCodec per dim, built once and reused everywhere (QR at __post_init__
    # is the expensive part — never rebuild it per word or per arm).
    codecs = {d: TPRCodec(dim=d) for d in DIMS}
    resolver = NSMMeaningResolver()
    probe_dim = DIMS[0]

    available = []
    skipped = 0
    for w in pool:
        sq = status_quo_handle(w, codecs[probe_dim], resolver)
        uv = usvs_handle(w, probe_dim)
        if sq is None or uv is None:
            skipped += 1
            continue
        available.append(w)

    words = available[:N_WORDS]
    print(f"pool={len(pool)} available={len(available)} skipped={skipped} used={len(words)}")
    if len(words) < N_WORDS:
        print(f"WARNING: only {len(words)} words available, fewer than requested N={N_WORDS}")

    rows = []
    for d in DIMS:
        codec = codecs[d]
        sq_handles = [status_quo_handle(w, codec, resolver) for w in words]
        uv_handles = [usvs_handle(w, d) for w in words]
        sq_stats = measure_arm(words, sq_handles, codec)
        uv_stats = measure_arm(words, uv_handles, codec)
        rows.append(("STATUS QUO", d, sq_stats))
        rows.append(("USVS", d, uv_stats))

    header = f"{'arm':<11}{'d':>6}{'top1_acc':>12}{'margin_q1':>12}{'margin_med':>12}{'margin_q3':>12}"
    print(header)
    print("-" * len(header))
    for arm, d, s in rows:
        print(f"{arm:<11}{d:>6}{s['top1_acc']:>12.3f}{s['q1']:>12.4f}{s['median']:>12.4f}{s['q3']:>12.4f}")


if __name__ == "__main__":
    main()
