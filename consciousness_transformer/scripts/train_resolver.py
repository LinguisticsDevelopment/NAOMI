"""Resolver training script: M53's pronoun collapse (Track A CorefHead vs
Track B SharedScorer vs --gold-binding, the M53a placeholder ceiling) EXTENDED
for M54's sense collapse (dev/RESOLVER_BUILD_PLAN.md Phase 3): the SAME two
tracks now ALSO resolve homograph senses in the same forward pass (Track A
gets a SECOND, sense-specialist head -- SenseHead; Track B reuses its ONE
SharedScorer instance for both candidate kinds, ``resolver is sense_resolver``
literally), plus a new --mfs-floor arm (the M30 baseline: always bind the
most-frequent sense, no resolver) alongside the pre-existing --gold-binding
ceiling. Follows scripts/probe_m52_transfer.py's arm pattern.

Curriculum mix:
    --mix m53 (M53's original): 1/2 old L1-6 + 1/4 transfer + 1/4 pronoun.
    --mix m54 (default): 40% old L1-6 + 20% transfer + 20% pronoun + 20%
        ambiguity (episode.generate_ambiguity_episodes, M32).

Reports: val accuracy overall + per curriculum kind (old/transfer_*/
pronoun_binding/ambiguity), RESOLVER BINDING ACCURACY (pronoun antecedent,
M53's own metric) overall + anti-recency half vs the scripted nearest-entity
baseline, SENSE BINDING ACCURACY (M54, new) overall + the sense-FLIPPED half
(gold_sense != mfs_sense, the M32 metric) vs the MFS floor (0.000 on the
flipped half BY CONSTRUCTION -- MFS is always candidate index 0, and
"flipped" means gold != mfs), both resolvers' param counts, and both
margin distributions.

Usage:
    python scripts/train_resolver.py --track A
    python scripts/train_resolver.py --track B
    python scripts/train_resolver.py --gold-binding      # ceiling: gold sense + gold antecedent, no resolver
    python scripts/train_resolver.py --mfs-floor         # floor: MFS sense (+ gold antecedent), no resolver
    python scripts/train_resolver.py --track A --mix m53 # M53 reproduction (no ambiguity episodes)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

torch.set_num_threads(1)

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import (  # noqa: E402
    association_only_baseline,
    generate_pronoun_episodes,
    generate_sense_binding_episodes,
    generate_transfer_episodes,
    nearest_entity_baseline,
)
from nsm_ct.episode import CurriculumGenerator, generate_ambiguity_episodes, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.resolver import make_resolver, make_sense_resolver  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

AUX_WEIGHT = 0.5   # resolver cross-entropy weight added to the answer loss (training-script side, not the model)


def build_mixed_curriculum(n_episodes: int, seed: int):
    """M53's original mix: 1/2 old L1-6 + 1/4 transfer + 1/4 pronoun,
    deterministic given (n_episodes, seed). Kept for --mix m53 (exact M53
    reproduction, no ambiguity episodes at all)."""
    n_pronoun = n_episodes // 4
    n_transfer = n_episodes // 4
    n_old = n_episodes - n_pronoun - n_transfer
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    transfer = generate_transfer_episodes(n_transfer, seed=seed + 1)
    pronoun = generate_pronoun_episodes(n_pronoun, seed=seed + 2)
    episodes = old + transfer + pronoun
    order = np.random.RandomState(seed + 3).permutation(len(episodes))
    return [episodes[i] for i in order]


def build_m54_curriculum(n_episodes: int, seed: int):
    """M54's mix (RESOLVER_BUILD_PLAN.md Phase 3, "Curriculum mix"): 40% old
    L1-6 + 20% transfer + 20% pronoun + 20% ambiguity, deterministic given
    (n_episodes, seed). Ambiguity episodes get ``meta["kind"] = "ambiguity"``
    set here (episode.generate_ambiguity_episodes doesn't set "kind" itself
    -- every other generator does) purely so the per-kind task-accuracy
    report below groups them under a real label instead of the "old"
    fallback; this never touches episode.py."""
    n_ambiguity = n_episodes // 5
    n_transfer = n_episodes // 5
    n_pronoun = n_episodes // 5
    n_old = n_episodes - n_ambiguity - n_transfer - n_pronoun
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    transfer = generate_transfer_episodes(n_transfer, seed=seed + 1)
    pronoun = generate_pronoun_episodes(n_pronoun, seed=seed + 2)
    ambiguity = generate_ambiguity_episodes(n_ambiguity, seed=seed + 3)
    for e in ambiguity:
        e.meta.setdefault("kind", "ambiguity")
    episodes = old + transfer + pronoun + ambiguity
    order = np.random.RandomState(seed + 4).permutation(len(episodes))
    return [episodes[i] for i in order]


def build_m54b_curriculum(n_episodes: int, seed: int):
    """M54b's mix (RESEARCH_NOTES M54b): 50% old L1-6 + 50% the NEW
    entity-keyed, binding-critical sense curriculum
    (curriculum2.generate_sense_binding_episodes) -- deliberately heavy in
    the new kind (unlike --mix m54's 20% old-M32-ambiguity slice) so the
    gold-ceiling-vs-MFS-floor gap probe has enough of the new signal to
    measure cleanly. No transfer/pronoun episodes here on purpose -- this
    mix isolates the ONE new capability M54b adds; --mix m54 already covers
    the full curriculum-mix regression. Deterministic given
    (n_episodes, seed). Episodes get ``meta["kind"] = "sense_binding"``
    already (the generator sets it itself, unlike the M32 generator)."""
    n_new = n_episodes // 2
    n_old = n_episodes - n_new
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    new = generate_sense_binding_episodes(n_new, seed=seed + 1)
    episodes = old + new
    order = np.random.RandomState(seed + 2).permutation(len(episodes))
    return [episodes[i] for i in order]


def resolver_binding_stats(out, batch, va_eps):
    """RESOLVER BINDING ACCURACY overall + anti-recency half, and the raw margin
    list, read off one resolver-carrying forward pass + the episodes' meta."""
    if "resolver_logits" not in out:
        return None
    cand_gold = batch.cand_gold
    has_cand = cand_gold >= 0
    pred_idx = out["resolver_logits"].argmax(-1)
    correct = (pred_idx == cand_gold) & has_cand
    margins = []
    hits = anti_hits = anti_total = total = 0
    for i, e in enumerate(va_eps):
        row_mask = has_cand[i]
        if not bool(row_mask.any()):
            continue
        t = int(row_mask.nonzero()[0, 0])
        total += 1
        is_hit = bool(correct[i, t])
        hits += int(is_hit)
        margins.append(float(out["resolver_margin"][i, t]))
        if e.meta.get("antecedent_recency") == "old":
            anti_total += 1
            anti_hits += int(is_hit)
    if total == 0:
        return None
    return {
        "overall_acc": hits / total, "n": total,
        "anti_recency_acc": (anti_hits / anti_total) if anti_total else float("nan"),
        "n_anti_recency": anti_total,
        "margins": margins,
    }


def sense_binding_stats(out, batch, va_eps):
    """M54's SENSE BINDING ACCURACY: overall + the sense-FLIPPED half (M32's
    own metric -- ``gold_sense != mfs_sense``, the half where the MFS floor
    is WRONG by construction), and the raw margin list. Mirrors
    :func:`resolver_binding_stats` exactly, just keyed on
    ``sense_resolver_logits``/``sense_cand_gold`` and the flip criterion
    instead of pronoun anti-recency."""
    if "sense_resolver_logits" not in out:
        return None
    cand_gold = batch.sense_cand_gold
    has_cand = cand_gold >= 0
    pred_idx = out["sense_resolver_logits"].argmax(-1)
    correct = (pred_idx == cand_gold) & has_cand
    margins = []
    hits = flip_hits = flip_total = total = 0
    for i, e in enumerate(va_eps):
        row_mask = has_cand[i]
        if not bool(row_mask.any()):
            continue
        # an ambiguity episode's homograph steps all carry the SAME gold
        # sense -- take the FIRST real step (mirrors resolver_binding_stats).
        t = int(row_mask.nonzero()[0, 0])
        total += 1
        is_hit = bool(correct[i, t])
        hits += int(is_hit)
        margins.append(float(out["sense_resolver_margin"][i, t]))
        if e.meta.get("gold_sense") != e.meta.get("mfs_sense"):
            flip_total += 1
            flip_hits += int(is_hit)
    if total == 0:
        return None
    return {
        "overall_acc": hits / total, "n": total,
        "flipped_acc": (flip_hits / flip_total) if flip_total else float("nan"),
        "n_flipped": flip_total,
        "margins": margins,
    }


def run_arm(name: str, track, episodes, dim: int, epochs: int, seed: int, hidden: int = 128,
            sense_bind: str = "gold"):
    """``track``: "A" | "B" | None (None = --gold-binding / --mfs-floor, no resolver).

    M54: when ``track`` is set, installs BOTH a pronoun resolver (``resolver=``,
    unchanged from M53) AND a sense resolver (``sense_resolver=``, new) on the
    SAME model -- Track A gets two INDEPENDENT specialist heads (CorefHead +
    SenseHead); Track B gets ONE SharedScorer instance passed to both slots
    (``resolver is sense_resolver`` -- the literal "one shared scorer for
    everything" experiment). When ``track`` is None, neither slot is
    installed and ``sense_bind`` controls what :func:`build_clause_batch`
    placeholder-binds homograph steps to (``"gold"`` = ceiling, ``"mfs"`` =
    floor); pronoun steps always placeholder-bind to the gold antecedent
    regardless (there is no MFS-equivalent floor for coreference).
    """
    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return None
    meaning_resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, meaning_resolver, codec, sense_bind=sense_bind)
    va = build_clause_batch(va_eps, parser, meaning_resolver, codec, sense_bind=sense_bind)

    torch.manual_seed(seed)
    if track:
        t = track.strip().upper()
        resolver = make_resolver(t, dim, hidden)
        sense_resolver = resolver if t == "B" else make_sense_resolver(t, dim, hidden)
    else:
        resolver = sense_resolver = None
    model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver, sense_resolver=sense_resolver)

    def _params(m):
        return sum(p.numel() for p in m.parameters()) if m is not None else 0

    n_resolver_params = _params(resolver)
    n_sense_params = _params(sense_resolver)
    shared = resolver is not None and resolver is sense_resolver
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])
    gold_va = torch.tensor([e.answer_idx for e in va_eps])

    t0 = time.time()
    model.train()
    for i in range(epochs):
        out = model(tr)
        loss = F.cross_entropy(out["answer_logits"], gold_tr)
        if resolver is not None and "resolver_logits" in out:
            cg = tr.cand_gold
            has_cand = cg >= 0
            if bool(has_cand.any()):
                aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
                loss = loss + AUX_WEIGHT * aux
        if sense_resolver is not None and "sense_resolver_logits" in out:
            scg = tr.sense_cand_gold
            has_scand = scg >= 0
            if bool(has_scand.any()):
                saux = F.cross_entropy(out["sense_resolver_logits"][has_scand], scg[has_scand])
                loss = loss + AUX_WEIGHT * saux
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 20 == 0 or i == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(va)["answer_logits"].argmax(-1) == gold_va).float().mean()
            model.train()
            print(f"  [{name}] epoch {i+1:3d} loss={loss.item():.3f} val={acc:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        out_va = model(va)
    pred = out_va["answer_logits"].argmax(-1)
    total_acc = float((pred == gold_va).float().mean())
    print(f"  [{name}] val total={total_acc:.3f}  resolver_params={n_resolver_params} "
          f"sense_resolver_params={n_sense_params}{' (SHARED, same instance)' if shared else ''}  "
          f"time={(time.time()-t0)/60:.2f} min", flush=True)

    per_kind = {}
    for i, e in enumerate(va_eps):
        per_kind.setdefault(str(e.meta.get("kind", "old")), []).append(bool(pred[i] == gold_va[i]))
    for k in sorted(per_kind):
        w = per_kind[k]
        print(f"    kind {k}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")

    binding = resolver_binding_stats(out_va, va, va_eps)
    if binding is not None:
        m = np.array(binding["margins"]) if binding["margins"] else np.array([0.0])
        print(f"  [{name}] RESOLVER BINDING ACCURACY overall={binding['overall_acc']:.3f} "
              f"(n={binding['n']}) anti-recency={binding['anti_recency_acc']:.3f} "
              f"(n={binding['n_anti_recency']}) vs baseline 0.500/0.000", flush=True)
        print(f"  [{name}] pronoun margin distribution: min={m.min():.3f} p25={np.percentile(m, 25):.3f} "
              f"median={np.median(m):.3f} p75={np.percentile(m, 75):.3f} max={m.max():.3f}", flush=True)

    sense_binding = sense_binding_stats(out_va, va, va_eps)
    if sense_binding is not None:
        sm = np.array(sense_binding["margins"]) if sense_binding["margins"] else np.array([0.0])
        print(f"  [{name}] SENSE BINDING ACCURACY overall={sense_binding['overall_acc']:.3f} "
              f"(n={sense_binding['n']}) flipped-half={sense_binding['flipped_acc']:.3f} "
              f"(n={sense_binding['n_flipped']}) vs MFS floor 0.000 (by construction)", flush=True)
        print(f"  [{name}] sense margin distribution: min={sm.min():.3f} p25={np.percentile(sm, 25):.3f} "
              f"median={np.median(sm):.3f} p75={np.percentile(sm, 75):.3f} max={sm.max():.3f}", flush=True)

    return {"total_acc": total_acc, "n_resolver_params": n_resolver_params,
            "n_sense_params": n_sense_params, "binding": binding, "sense_binding": sense_binding}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B"], default=None)
    ap.add_argument("--gold-binding", action="store_true",
                     help="ceiling arm: no resolver, gold sense + gold antecedent placeholder binding")
    ap.add_argument("--mfs-floor", action="store_true",
                     help="floor arm: no resolver, MFS sense placeholder binding "
                          "(gold antecedent still bound) -- the M30 baseline")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", choices=["m53", "m54", "m54b"], default="m54",
                     help="m53 = the original 1/2 old + 1/4 transfer + 1/4 pronoun mix "
                          "(no ambiguity episodes, exact M53 reproduction); "
                          "m54 (default) = 40%% old / 20%% transfer / 20%% pronoun / 20%% ambiguity; "
                          "m54b = 50%% old / 50%% the NEW entity-keyed sense-binding curriculum "
                          "(RESEARCH_NOTES M54b's gold/MFS gap probe)")
    args = ap.parse_args()

    n_arms = sum([args.gold_binding, args.mfs_floor, args.track is not None])
    if n_arms != 1:
        raise SystemExit("pass exactly one of --track A|B, --gold-binding, --mfs-floor")

    if args.mix == "m54":
        episodes = build_m54_curriculum(args.episodes, args.seed)
    elif args.mix == "m54b":
        episodes = build_m54b_curriculum(args.episodes, args.seed)
    else:
        episodes = build_mixed_curriculum(args.episodes, args.seed)
    baseline = nearest_entity_baseline(episodes)
    print(f"nearest-entity baseline: overall={baseline['accuracy']:.3f} (n={baseline['n']}) "
          f"anti-recency={baseline['anti_recency_accuracy']:.3f} (n={baseline['n_anti_recency']})",
          flush=True)
    assoc = association_only_baseline(episodes)
    if assoc["n"]:
        print(f"association-only baseline (M54b sense-binding kind): "
              f"accuracy={assoc['accuracy']:.3f} (n={assoc['n']}) vs chance 0.500", flush=True)

    if args.gold_binding:
        print(f"=== gold-binding ceiling: no resolver, {args.episodes} eps, dim={args.dim}, "
              f"mix={args.mix} ===", flush=True)
        run_arm("gold-binding", None, episodes, args.dim, args.epochs, args.seed, args.hidden,
                 sense_bind="gold")
    elif args.mfs_floor:
        print(f"=== mfs-floor: no resolver, {args.episodes} eps, dim={args.dim}, "
              f"mix={args.mix} ===", flush=True)
        run_arm("mfs-floor", None, episodes, args.dim, args.epochs, args.seed, args.hidden,
                 sense_bind="mfs")
    else:
        print(f"=== track {args.track}: {args.episodes} eps, dim={args.dim}, mix={args.mix} ===",
              flush=True)
        run_arm(f"track-{args.track}", args.track, episodes, args.dim, args.epochs, args.seed,
                 args.hidden)


if __name__ == "__main__":
    main()
