"""M43 probe — does opt-in gloss content-word enrichment fix the M42 shallow-
signature failures (ball/court/hood/mole/yard)?

M42 (`probe_grounding_depth.py`) diagnosed the mechanism: sense signatures are
grounded ONLY through `naive_decompose`'s prime-only filter, which keeps a
content word's decomposition ONLY if it bottoms out on one of the ~65 literal
NSM primes — discarding named MOLECULE axes (HEAD, FACE, ...) and anything
that doesn't get lucky, so most signatures collapse to 2-6 generic axes
(SOMETHING appears in 57% of all 117,659 sense signatures) and cosine against
an answer word becomes a genericity match. `nsm_ct.ground.explication` adds an
opt-in `enriched_sense_dense(usvs, sense_id, alpha)`: blend the base signature
with the sense's OWN gloss content words' placed-CORE coordinates (the deeper,
non-prime-filtered layer), damped by alpha, renormalized. Nothing here is
wired into `build_usvs` or `data/usvs/` — this is pure evaluation-time probing.

Three parts:
  (a) The M42 sense-level ranking table (all 31 ambiguity families, 62
      sense-answer rankings from `nsm_ct.episode._AMBIGUITY_FAMILIES`):
      plain vs IDF vs enriched vs enriched+IDF, with casualty-family detail.
  (c) An alpha sweep over {0.2, 0.35, 0.5} for (a); the best alpha carries
      into (b). Also reports (a) WITH and WITHOUT the M24 exclusion (drop an
      evaluated family's own answer words from the gloss-word enrichment set
      before lookup — otherwise a gloss that happens to literally mention the
      scored answer word would leak it straight into the sense's vector).
  (b) A SemCor WSD spot check reusing `probe_wsd_semcor.py`'s leave-one-out
      machinery, subsampled to ~2000 polysemous instances (fixed seed) so it
      runs in about a minute: MFS vs usvs-idf (M30's best training-free
      variant) vs enriched vs enriched-idf on that subsample.

Usage: python scripts/probe_explication_depth.py [--usvs data/usvs]
                                                   [--semcor-poly 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import FrozenSet

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.episode import _AMBIGUITY_FAMILIES  # noqa: E402
from nsm_ct.ground.explication import enriched_sense_dense  # noqa: E402
from nsm_ct.ground.meaning_value import cosine  # noqa: E402
from nsm_ct.ground.usvs import load_usvs  # noqa: E402

DEFAULT_USVS = str(Path(__file__).resolve().parent.parent / "data" / "usvs")
CASUALTIES = ("ball", "court", "racket", "yard", "cell", "mole", "hood")
VARIANTS = ("plain", "idf", "enriched", "enriched-idf")


def word_vec(u, word):
    v = u.word_coord(word)
    if v is not None:
        return v
    sids = u.senses_of(word)
    return u.sense_dense(sids[0]) if sids else None


def axis_idf(u) -> np.ndarray:
    """M21/M42 distinctiveness weighting: log(N/(df+1)) over the sense layer."""
    n = len(u.sense_ids)
    df = np.bincount(u.sense_axis_idx, minlength=len(u.axes)).astype(np.float64)
    return np.log((n + 1.0) / (df + 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# (a) + (c) — 31-family / 62-sense ranking table, per variant, per alpha
# ---------------------------------------------------------------------------
def rank_families(u, idf: np.ndarray, alpha: float, *, m24_exclude: bool):
    correct = {v: 0 for v in VARIANTS}
    total = 0
    casualty_detail = defaultdict(list)  # variant -> [(family, sense_key, ...)]

    for fam, spec in sorted(_AMBIGUITY_FAMILIES.items()):
        senses = spec["senses"]
        answer_words = {k: s["answer"] for k, s in senses.items()}
        ans_vecs = {k: word_vec(u, w) for k, w in answer_words.items()}
        excl: FrozenSet[str] = (
            frozenset(w.lower() for w in answer_words.values()) if m24_exclude else frozenset()
        )
        for key, s in senses.items():
            sid = s["synset"]
            other = [k for k in senses if k != key][0]
            cv, wv = ans_vecs.get(key), ans_vecs.get(other)
            base = u.sense_dense(sid)
            if cv is None or wv is None or base is None or not base.any():
                continue
            enr = enriched_sense_dense(u, sid, alpha, exclude=excl)
            total += 1
            for v in VARIANTS:
                if v == "plain":
                    sv, a_c, a_w = base, cv, wv
                elif v == "idf":
                    sv, a_c, a_w = base * idf, cv * idf, wv * idf
                elif v == "enriched":
                    sv, a_c, a_w = enr, cv, wv
                else:
                    sv, a_c, a_w = enr * idf, cv * idf, wv * idf
                c_cos, w_cos = cosine(sv, a_c), cosine(sv, a_w)
                ok = c_cos > w_cos
                correct[v] += int(ok)
                if not ok:
                    casualty_detail[v].append(
                        (fam, key, sid, answer_words[key], answer_words[other], c_cos, w_cos)
                    )
    return correct, total, casualty_detail


def print_ranking_table(label: str, correct, total) -> None:
    print(f"  {label}: " + "  ".join(f"{v}={correct[v]}/{total}" for v in VARIANTS))


def print_casualty_detail(casualty_detail, total) -> None:
    for v in VARIANTS:
        fams = sorted({fam for fam, *_ in casualty_detail[v]})
        hit_casualties = [f for f in fams if f in CASUALTIES]
        print(f"    {v:14} misses={len(casualty_detail[v])}/{total}  "
              f"casualty families still missed: {hit_casualties}")


# ---------------------------------------------------------------------------
# (b) — SemCor subsample spot check (reuses probe_wsd_semcor's LOO machinery)
# ---------------------------------------------------------------------------
def sentence_instances(sent, wn):
    out = []
    for chunk in sent:
        try:
            lab = chunk.label()
        except AttributeError:
            continue
        if not hasattr(lab, "synset"):
            continue
        gold = lab.synset().name()
        word = lab.name().lower()
        pos = lab.synset().pos()
        cands = [s.name() for s in wn.synsets(word, "as" if pos in ("a", "s") else pos)]
        if not cands or gold not in cands:
            continue
        out.append((word, gold, "a" if pos == "s" else pos, cands))
    return out


def semcor_subsample_check(u, idf: np.ndarray, alpha: float, *, poly_target: int, pool: int, seed: int):
    from nltk.corpus import semcor
    from nsm_ct.wordnet import _wn

    wn = _wn()
    sents = list(semcor.tagged_sents(tag="sem")[:pool])
    order = list(range(len(sents)))
    random.Random(seed).shuffle(order)

    dense_cache: dict = {}
    enr_cache: dict = {}

    def dense(sid):
        if sid not in dense_cache:
            dense_cache[sid] = u.sense_dense(sid)
        return dense_cache[sid]

    def enr(sid):
        if sid not in enr_cache:
            enr_cache[sid] = enriched_sense_dense(u, sid, alpha)
        return enr_cache[sid]

    def pick(cands, ctx, weight=None, get=dense):
        if ctx is None or not np.any(ctx):
            return cands[0]
        c_w = ctx * weight if weight is not None else ctx
        scored = []
        for c in cands:
            v = get(c)
            if v is None or not np.any(v):
                scored.append((-2.0, c))
                continue
            v_w = v * weight if weight is not None else v
            scored.append((cosine(v_w, c_w), c))
        best = max(scored)
        return best[1] if best[0] > -2.0 else cands[0]

    rng = random.Random(seed + 1)
    hits = defaultdict(int)
    tot = defaultdict(int)
    n_poly = 0
    n_sent_used = 0

    for si in order:
        if n_poly >= poly_target:
            break
        inst = sentence_instances(sents[si], wn)
        if not inst:
            continue
        n_sent_used += 1
        mfs_vecs = [dense(cands[0]) for _, _, _, cands in inst]
        enr_vecs = [enr(cands[0]) for _, _, _, cands in inst]
        sig_valid = [v for v in mfs_vecs if v is not None]
        sig_sum = np.sum(sig_valid, axis=0) if sig_valid else None
        enr_valid = [v for v in enr_vecs if v is not None]
        enr_sum = np.sum(enr_valid, axis=0) if enr_valid else None

        for k, (word, gold, pos, cands) in enumerate(inst):
            mfs = cands[0]
            rnd = rng.choice(cands)
            poly = len(cands) > 1
            if not poly:
                usvs_idf = enriched = enriched_idf = mfs
            else:
                sig_ctx = None if sig_sum is None else \
                    sig_sum - (mfs_vecs[k] if mfs_vecs[k] is not None else 0.0)
                enr_ctx = None if enr_sum is None else \
                    enr_sum - (enr_vecs[k] if enr_vecs[k] is not None else 0.0)
                usvs_idf = pick(cands, sig_ctx, weight=idf, get=dense)
                enriched = pick(cands, enr_ctx, get=enr)
                enriched_idf = pick(cands, enr_ctx, weight=idf, get=enr)
                n_poly += 1
            for name, pred in (("mfs", mfs), ("usvs-idf", usvs_idf), ("enriched", enriched),
                               ("enriched-idf", enriched_idf), ("random", rnd)):
                ok = pred == gold
                hits[(name, "all")] += ok
                tot[(name, "all")] += 1
                if poly:
                    hits[(name, "poly")] += ok
                    tot[(name, "poly")] += 1
    return hits, tot, n_sent_used


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usvs", default=DEFAULT_USVS)
    ap.add_argument("--semcor-poly", type=int, default=2000)
    ap.add_argument("--semcor-pool", type=int, default=4000,
                     help="sentence pool to shuffle-sample from (M30's original pool)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    u = load_usvs(args.usvs)
    idf = axis_idf(u)
    print(f"USVS {u.fingerprint} loaded ({len(u.axes)} axes, {len(u.sense_ids)} senses); "
          f"{len(_AMBIGUITY_FAMILIES)} families")

    # ---- (a) + (c): alpha sweep, M24-honest (exclusion ON) ----------------
    print("\n=== (a)+(c) 31-family / 62-sense ranking, M24 exclusion ON ===")
    best_alpha, best_score = None, -1
    sweep_results = {}
    for alpha in (0.2, 0.35, 0.5):
        correct, total, casualty_detail = rank_families(u, idf, alpha, m24_exclude=True)
        sweep_results[alpha] = (correct, total, casualty_detail)
        print_ranking_table(f"alpha={alpha}", correct, total)
        score = correct["enriched-idf"]
        if score > best_score:
            best_alpha, best_score = alpha, score

    print(f"\nbest alpha by enriched-idf score: {best_alpha} ({best_score}/{sweep_results[best_alpha][1]})")
    print("\ncasualty-family detail at best alpha (M24 ON):")
    print_casualty_detail(sweep_results[best_alpha][2], sweep_results[best_alpha][1])

    # ---- M24 exclusion OFF at the same best alpha, for the honest contrast ----
    print(f"\n=== same ranking at alpha={best_alpha}, M24 exclusion OFF (answer word left in "
          f"gloss enrichment set — for comparison only, NOT the honest number) ===")
    correct_off, total_off, casualty_off = rank_families(u, idf, best_alpha, m24_exclude=False)
    print_ranking_table(f"alpha={best_alpha} (no M24 exclusion)", correct_off, total_off)
    print_casualty_detail(casualty_off, total_off)

    # ---- (b): SemCor subsample spot check at best alpha --------------------
    print(f"\n=== (b) SemCor subsample spot check (~{args.semcor_poly} polysemous instances, "
          f"seed={args.seed}, alpha={best_alpha}) ===")
    hits, tot, n_sent = semcor_subsample_check(
        u, idf, best_alpha, poly_target=args.semcor_poly, pool=args.semcor_pool, seed=args.seed)
    print(f"sentences sampled: {n_sent}  instances: all={tot[('mfs','all')]} "
          f"polysemous={tot[('mfs','poly')]}")
    print(f"{'resolver':14} {'all':>7} {'poly':>7}")
    for name in ("mfs", "usvs-idf", "enriched", "enriched-idf", "random"):
        row = [hits[(name, s)] / max(tot[(name, s)], 1) for s in ("all", "poly")]
        print(f"{name:14} " + " ".join(f"{v:7.3f}" for v in row))

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
