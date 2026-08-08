"""Probe: M30 — the WSD gate on SemCor (USVS viability test #1).

Training-free by design: the USVS resolver is pure artifact lookup + cosine, so
the number measures USVS itself, not a model. Resolvers:

- MFS      — WordNet first sense (the famously-hard floor).
- USVS-sim — context = mean USVS signature of the sentence's OTHER annotated
             content words (their MFS senses — gold labels never enter the
             context); prediction = candidate sense with the most similar
             signature; zero-signal falls back to MFS.
- random   — sanity floor.

Reports all-instances AND polysemous-only accuracy (MFS is inflated by
monosemous words), plus per-POS. Decision rule (dev/INTEGRATION_PLAN.md):
USVS-sim >= MFS on polysemous -> signatures are viable for the perception slot;
well below -> documented negative, fix is grounding depth not resolver machinery.

Usage:
    python scripts/probe_wsd_semcor.py [--sents 4000] [--usvs data/usvs]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.meaning_value import cosine  # noqa: E402
from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct.wordnet import _wn  # noqa: E402


def sentence_instances(sent, wn):
    """(word, gold_synset_name, pos, candidates) for each annotated chunk."""
    out = []
    for chunk in sent:
        try:
            lab = chunk.label()
        except AttributeError:
            continue
        if not hasattr(lab, "synset"):          # some labels are plain strings
            continue
        gold = lab.synset().name()
        word = lab.name().lower()
        pos = lab.synset().pos()
        cands = [s.name() for s in wn.synsets(word, "as" if pos in ("a", "s") else pos)]
        if not cands or gold not in cands:
            continue
        out.append((word, gold, "a" if pos == "s" else pos, cands))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sents", type=int, default=4000)
    ap.add_argument("--usvs", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    u = load_usvs(args.usvs)
    wn = _wn()
    from nltk.corpus import semcor
    sents = semcor.tagged_sents(tag="sem")[: args.sents]
    rng = random.Random(args.seed)
    print(f"USVS {u.fingerprint} loaded; scoring {len(sents)} SemCor sentences")

    dense_cache: dict = {}

    def dense(sid):
        if sid not in dense_cache:
            dense_cache[sid] = u.sense_dense(sid)
        return dense_cache[sid]

    # Axis IDF over the sense layer (M21 distinctiveness weighting): ubiquitous
    # axes (SOMETHING, lex:*) otherwise dominate every cosine.
    n_senses = len(u.sense_ids)
    df = np.bincount(u.sense_axis_idx, minlength=len(u.axes)).astype(np.float64)
    idf = np.log((n_senses + 1.0) / (df + 1.0)).astype(np.float32)

    def core_vec(word):
        return u.word_coord(word)

    # v2 sense vector: mean PLACED-CORE coordinate of the sense's gloss content
    # words (the artifact's held-out-validated layer), not their prime
    # decompositions. Tests whether the sense-layer *definition* was the
    # bottleneck (M30 diagnostic: sense signatures separate fine — intra-word
    # top-2 cosine 0.41 — so the weak link is what we match them against).
    from nsm_ct.ground.definition_graph import content_words
    v2_cache: dict = {}

    def dense_v2(sid):
        if sid not in v2_cache:
            gloss = wn.synset(sid).definition() or ""
            vs = [u.word_coord(w) for w in content_words(gloss)]
            vs = [v for v in vs if v is not None]
            v2_cache[sid] = np.mean(vs, axis=0) if vs else None
        return v2_cache[sid]

    hits = defaultdict(int)     # (resolver, subset) -> correct
    tot = defaultdict(int)
    pos_hits = defaultdict(int)  # (resolver, pos) -> correct on polysemous
    pos_tot = defaultdict(int)
    skipped = 0

    for sent in sents:
        inst = sentence_instances(sent, wn)
        if not inst:
            skipped += 1
            continue
        # contexts per sentence, leave-one-out per target:
        #   sig  — sum of MFS sense signatures of the other instances
        #   core — sum of placed core coords of the other instances' words
        mfs_vecs = []
        core_vecs = []
        for word, _gold, _pos, cands in inst:
            v = dense(cands[0])
            mfs_vecs.append(v if v is not None else None)
            core_vecs.append(core_vec(word))
        sig_valid = [v for v in mfs_vecs if v is not None]
        sig_sum = np.sum(sig_valid, axis=0) if sig_valid else None
        core_valid = [v for v in core_vecs if v is not None]
        core_sum = np.sum(core_valid, axis=0) if core_valid else None

        def pick_v2(cands, k):
            # context: the sentence's other words' own core coords (surface
            # words, no sense commitment); candidates scored by their gloss-
            # core vector. Both sides live in the placed-core space.
            if core_sum is None:
                return cands[0]
            ctx = core_sum - (core_vecs[k] if core_vecs[k] is not None else 0.0)
            if not np.any(ctx):
                return cands[0]
            scored = []
            for c in cands:
                v = dense_v2(c)
                scored.append((cosine(v, ctx) if v is not None else -2.0, c))
            best = max(scored)
            return best[1] if best[0] > -2.0 else cands[0]

        def pick(cands, ctx, weight=None):
            if ctx is None or not np.any(ctx):
                return cands[0]
            c_w = ctx * weight if weight is not None else ctx
            scored = []
            for c in cands:
                v = dense(c)
                if v is None or not np.any(v):
                    scored.append((-2.0, c))
                    continue
                v_w = v * weight if weight is not None else v
                scored.append((cosine(v_w, c_w), c))
            best = max(scored)
            return best[1] if best[0] > -2.0 else cands[0]

        for k, (word, gold, pos, cands) in enumerate(inst):
            mfs = cands[0]
            rnd = rng.choice(cands)
            if len(cands) == 1:
                usim = uidf = ucore = uv2 = mfs
            else:
                uv2 = pick_v2(cands, k)
                sig_ctx = None if sig_sum is None else \
                    sig_sum - (mfs_vecs[k] if mfs_vecs[k] is not None else 0.0)
                core_ctx = None if core_sum is None else \
                    core_sum - (core_vecs[k] if core_vecs[k] is not None else 0.0)
                usim = pick(cands, sig_ctx)
                uidf = pick(cands, sig_ctx, weight=idf)
                ucore = pick(cands, core_ctx, weight=idf)
            poly = len(cands) > 1
            # MFS-confidence stratum: how sure is the frequency prior itself?
            # counts from WordNet lemma.count(); "unsure" = no data or the top
            # two senses are close — the exact slice a fallback would own.
            if poly:
                counts = []
                for c in cands:
                    syn_c = wn.synset(c)
                    counts.append(sum(l.count() for l in syn_c.lemmas()
                                      if l.name().lower() == word))
                c_sorted = sorted(counts, reverse=True)
                unsure = (c_sorted[0] == 0
                          or c_sorted[0] - c_sorted[1] <= 1
                          or c_sorted[1] / c_sorted[0] > 0.6)
                stratum = "unsure" if unsure else "sure"
            for name, pred in (("mfs", mfs), ("usvs", usim), ("usvs-idf", uidf), ("usvs-v2", uv2),
                               ("usvs-core", ucore), ("random", rnd)):
                ok = pred == gold
                hits[(name, "all")] += ok
                tot[(name, "all")] += 1
                if poly:
                    hits[(name, "poly")] += ok
                    tot[(name, "poly")] += 1
                    hits[(name, stratum)] += ok
                    tot[(name, stratum)] += 1
                    pos_hits[(name, pos)] += ok
                    pos_tot[(name, pos)] += 1

    print(f"\ninstances: all={tot[('mfs','all')]}  polysemous={tot[('mfs','poly')]}  "
          f"({time.time()-t0:.0f}s)")
    print(f"{'resolver':9} {'all':>7} {'poly':>7} {'sure':>7} {'unsure':>7}   " +
          " ".join(f"{p:>7}" for p in "nvar"))
    for name in ("mfs", "usvs", "usvs-idf", "usvs-v2", "usvs-core", "random"):
        row = [hits[(name, s)] / max(tot[(name, s)], 1)
               for s in ("all", "poly", "sure", "unsure")]
        pos_row = [pos_hits[(name, p)] / max(pos_tot[(name, p)], 1) for p in "nvar"]
        print(f"{name:9} " + " ".join(f"{v:7.3f}" for v in row) + "   " +
              " ".join(f"{v:7.3f}" for v in pos_row))
    print(f"strata: sure={tot[('mfs','sure')]} unsure={tot[('mfs','unsure')]}")
    print("poly counts per POS: " +
          " ".join(f"{p}={pos_tot[('mfs', p)]}" for p in "nvar"))


if __name__ == "__main__":
    main()
