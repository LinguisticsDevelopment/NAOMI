"""M57c training script: instance atoms + definite-description referring
expressions -- the two-Marys curriculum. Integrates M57a's InstanceRegistry
(nsm_ct.instances) with M57b's proven resolver-driven write-back mechanism
(nsm_ct.clause_reactor._writeback_steps / ClauseReactor's address-redirect
collapse). Sibling to scripts/train_writeback.py -- SAME arm/report pattern
(track dispatch, aux loss, per-kind task accuracy, a resolver-binding-
accuracy metric, margin distribution), extended with a THIRD curriculum kind
(``ep.meta["kind"] == "instance"``, nsm_ct.curriculum2.
InstanceCurriculumGenerator via nsm_ct.clause_reactor._instance_steps) mixed
in alongside old L1-6 and writeback (M57b).

Curriculum mix (:func:`build_instance_curriculum`): ~1/3 old L1-6, ~1/3
write-back (M57b), ~1/3 instance (M57c -- itself a mix of "target"-question
and "who is X ?" inverse-query episodes, controlled by ``--inverse-frac``).

Arms (identical five to scripts/train_writeback.py -- ``writeback_cheat``/
``writeback_no_gold``/``writeback_force`` are REUSED verbatim for instance
episodes too, see nsm_ct.clause_reactor.build_clause_batch's M57c docstring
paragraph: one arm setting applies across the whole mixed batch):
    python scripts/train_instances.py --track A                        # normal arm
    python scripts/train_instances.py --track A --force-binding gold   # forced-gold eval (ceiling)
    python scripts/train_instances.py --track A --force-binding wrong  # forced-wrong eval (floor) --
                                                                          THE full-scale validity check
                                                                          (CLAUDE.md: smoke-scale never
                                                                          gates curriculum validity).
    python scripts/train_instances.py --track A --cheat                # cheat-baseline floor check
    python scripts/train_instances.py --track A --no-gold-eval         # eval batches built with NO
                                                                          gold grounding anywhere
                                                                          (candidates + the TRAINED
                                                                          resolver only)
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

from _train_common import add_footprint_args, apply_threads, epoch_minibatches, eval_minibatched, peak_rss_mb  # noqa: E402
from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import (  # noqa: E402
    generate_instance_episodes,
    generate_rich_episodes,
    generate_writeback_episodes,
)
from nsm_ct.episode import CurriculumGenerator, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.resolver import make_resolver  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

AUX_WEIGHT = 0.5   # resolver cross-entropy weight added to the answer loss -- same constant train_writeback.py uses


def build_instance_curriculum(n_episodes: int, seed: int, inverse_frac: float = 0.3,
                               rich_frac: float = 0.0, rich_inverse_frac: float = 0.3):
    """~1/3 old L1-6 + ~1/3 write-back (M57b) + ~1/3 instance (M57c,
    target + inverse-query mixed via ``inverse_frac``) -- deterministic
    given ``(n_episodes, seed, inverse_frac, rich_frac, rich_inverse_frac)``.
    Mirrors scripts/train_writeback.py's ``build_writeback_curriculum``
    exactly (isolate the new capability; old L1-6 supplies the
    non-candidate-bearing bulk of the mix), extended with the third kind.

    ``rich_frac`` (RICH-EPISODE curriculum, CLAUDE.md's 2026-08-30
    reprioritization "stop requiring minimal episodes"): a FOURTH slice --
    ``round(n_episodes * rich_frac)`` episodes from
    ``nsm_ct.curriculum2.generate_rich_episodes`` -- added on top of the
    old/writeback/instance thirds, which still split the REMAINING
    episodes evenly (so ``rich_frac=0`` reproduces every pre-rich call of
    this function exactly). ``rich_inverse_frac`` forwards to
    ``generate_rich_episodes``'s own ``inverse_frac``."""
    n_rich = int(round(n_episodes * rich_frac))
    n_rest = n_episodes - n_rich
    n_each = n_rest // 3
    n_old = n_rest - 2 * n_each
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    wb = generate_writeback_episodes(n_each, seed=seed + 1)
    inst = generate_instance_episodes(n_each, seed=seed + 2, inverse_frac=inverse_frac)
    rich = (generate_rich_episodes(n_rich, seed=seed + 4, inverse_frac=rich_inverse_frac)
            if n_rich else [])
    episodes = old + wb + inst + rich
    order = np.random.RandomState(seed + 3).permutation(len(episodes))
    return [episodes[i] for i in order]


def binding_stats(out, batch, eps):
    """Resolver-binding accuracy over EVERY gold-bearing candidate step in
    the batch -- NOT just the first per row (unlike
    scripts/train_writeback.py's own ``writeback_binding_stats``, which
    could get away with that since a writeback row carries exactly one
    resolvable step): an instance episode may carry TWO -- the overwrite
    step AND, when the question's own referring expression is non-unique
    (a/b targets), the question step itself, exercising the SAME
    resolution mechanism at read time. Returns overall accuracy + a
    per-episode-kind breakdown; ``None`` if the batch carries no candidate
    data at all (e.g. the cheat arm)."""
    if "resolver_logits" not in out:
        return None
    cand_gold = batch.cand_gold
    has_cand = cand_gold >= 0
    pred_idx = out["resolver_logits"].argmax(-1)
    correct = (pred_idx == cand_gold) & has_cand
    per_kind: dict = {}
    margins = []
    hits = total = 0
    for i, e in enumerate(eps):
        ts = has_cand[i].nonzero().flatten().tolist()
        if not ts:
            continue
        kind = str(e.meta.get("kind", "old"))
        for t in ts:
            total += 1
            is_hit = bool(correct[i, t])
            hits += int(is_hit)
            margins.append(float(out["resolver_margin"][i, t]))
            per_kind.setdefault(kind, [0, 0])
            per_kind[kind][0] += int(is_hit)
            per_kind[kind][1] += 1
    if total == 0:
        return None
    return {
        "overall_acc": hits / total, "n": total, "margins": margins,
        "per_kind": {k: (v[0] / v[1], v[1]) for k, v in per_kind.items()},
    }


def run_arm(name: str, track, episodes, dim: int, epochs: int, seed: int, hidden: int = 128,
            cheat: bool = False, no_gold_eval: bool = False, force_binding: str = None,
            batch_size: int = 64) -> dict:
    """``track``: "A" | "B" | None (None = no resolver installed -- both
    writeback's address-redirect AND instance's evidence-relation
    resolution never fire at all, a genuine floor arm).

    ``cheat``/``no_gold_eval``/``force_binding`` forward to
    ``build_clause_batch``'s ``writeback_cheat``/``writeback_no_gold``/
    ``writeback_force`` -- see nsm_ct.clause_reactor.build_clause_batch's
    M57c docstring paragraph: these flags are REUSED verbatim for instance
    episodes (one arm setting applies across the whole mixed batch, same
    as scripts/train_writeback.py's own contract for writeback alone).

    ``batch_size`` (M57c footprint fix, same contract as
    scripts/train_writeback.py's ``run_arm``): each epoch shuffles
    ``tr_eps``' indices (``_train_common.epoch_minibatches``, seeded by
    ``seed`` and the epoch number) and steps the optimizer once per
    minibatch instead of once over the whole training set; ``0`` =
    full-batch (the pre-fix behavior, kept reachable for before/after
    measurement). Evaluation (periodic val print + final held-out eval)
    is minibatched too, via ``_train_common.eval_minibatched`` (no-grad,
    row order preserved -- value-for-value equivalent of a full-batch
    eval)."""
    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return {}
    meaning_resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, meaning_resolver, codec, writeback_cheat=cheat)
    va = build_clause_batch(va_eps, parser, meaning_resolver, codec, writeback_cheat=cheat,
                             writeback_no_gold=no_gold_eval, writeback_force=force_binding)

    torch.manual_seed(seed)
    resolver = make_resolver(track, dim, hidden) if track else None
    model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver)
    n_resolver_params = sum(p.numel() for p in resolver.parameters()) if resolver is not None else 0

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])
    gold_va = torch.tensor([e.answer_idx for e in va_eps])

    n_tr = len(tr_eps)
    t0 = time.time()
    model.train()
    losses = []
    for i in range(epochs):
        epoch_losses = []
        for mb_idx in epoch_minibatches(n_tr, batch_size, seed, i):
            idx_t = torch.from_numpy(mb_idx)
            sub = tr.subset(idx_t)
            sub_gold = gold_tr[idx_t]
            out = model(sub)
            loss = F.cross_entropy(out["answer_logits"], sub_gold)
            if resolver is not None and "resolver_logits" in out:
                cg = sub.cand_gold
                has_cand = cg >= 0
                if bool(has_cand.any()):
                    aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
                    loss = loss + AUX_WEIGHT * aux
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(float(loss.item()))
        last_loss = epoch_losses[-1]
        losses.append(last_loss)
        if (i + 1) % 20 == 0 or i == 0:
            model.eval()
            acc = (eval_minibatched(model, va, batch_size)["answer_logits"].argmax(-1) == gold_va).float().mean()
            model.train()
            print(f"  [{name}] epoch {i+1:3d} loss={last_loss:.3f} val={acc:.3f}", flush=True)
    elapsed_min = (time.time() - t0) / 60

    model.eval()
    out_va = eval_minibatched(model, va, batch_size)
    pred = out_va["answer_logits"].argmax(-1)
    total_acc = float((pred == gold_va).float().mean())
    print(f"  [{name}] val total={total_acc:.3f}  resolver_params={n_resolver_params}"
          f"  time={elapsed_min:.2f} min  peak_rss_mb={peak_rss_mb():.1f}", flush=True)

    per_kind: dict = {}
    for i, e in enumerate(va_eps):
        per_kind.setdefault(str(e.meta.get("kind", "old")), []).append(bool(pred[i] == gold_va[i]))
    for k in sorted(per_kind):
        w = per_kind[k]
        print(f"    kind {k}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")

    # writeback per-subset (referent vs other), unchanged from train_writeback.py.
    wb_subsets = {"referent_targeted": [], "other_targeted": []}
    for i, e in enumerate(va_eps):
        if e.meta.get("kind") != "writeback":
            continue
        key = "referent_targeted" if e.meta.get("question_targets_referent") else "other_targeted"
        wb_subsets[key].append(bool(pred[i] == gold_va[i]))
    for k in ("referent_targeted", "other_targeted"):
        w = wb_subsets[k]
        if w:
            print(f"    writeback/{k}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")

    # instance per-(referring_device, referent/other) subset (target-mode
    # only) + inverse-query accuracy on its own.
    inst_device: dict = {}
    inst_inverse = []
    for i, e in enumerate(va_eps):
        if e.meta.get("kind") != "instance":
            continue
        hit = bool(pred[i] == gold_va[i])
        if e.meta.get("question_mode") == "inverse":
            inst_inverse.append(hit)
            continue
        device = e.meta.get("referring_device")
        subset = "referent_targeted" if e.meta.get("question_targets_referent") else "other_targeted"
        inst_device.setdefault((device, subset), []).append(hit)
    for (device, subset), w in sorted(inst_device.items()):
        print(f"    instance/{device}/{subset}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")
    if inst_inverse:
        print(f"    instance/inverse_query: {sum(inst_inverse)}/{len(inst_inverse)} "
              f"= {sum(inst_inverse)/len(inst_inverse):.3f}")

    # RICH-EPISODE curriculum (kind=rich): per-(overwriting-device,
    # overwritten/baseline) subset, per-n_entities accuracy, and
    # inverse-query accuracy -- see nsm_ct.curriculum2.RichEpisodeGenerator's
    # meta schema. "device" here is the device of the referring statement
    # that overwrote the TARGETED slot (None/"none" when the question
    # targets a never-overwritten, baseline slot -- there is no single
    # "referent" in a rich episode the way there is in a writeback/instance
    # episode, since K statements may target K different entities).
    rich_device: dict = {}
    rich_by_n: dict = {}
    rich_inverse = []
    for i, e in enumerate(va_eps):
        if e.meta.get("kind") != "rich":
            continue
        hit = bool(pred[i] == gold_va[i])
        rich_by_n.setdefault(e.meta["n_entities"], []).append(hit)
        if e.meta.get("question_mode") == "inverse":
            rich_inverse.append(hit)
            continue
        subset = "overwritten" if e.meta.get("question_targets_overwritten") else "baseline"
        device = e.meta.get("question_device") or "none"
        rich_device.setdefault((device, subset), []).append(hit)
    for (device, subset), w in sorted(rich_device.items()):
        print(f"    rich/{device}/{subset}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")
    for n_e in sorted(rich_by_n):
        w = rich_by_n[n_e]
        print(f"    rich/by_n_entities/{n_e}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")
    if rich_inverse:
        print(f"    rich/inverse_query: {sum(rich_inverse)}/{len(rich_inverse)} "
              f"= {sum(rich_inverse)/len(rich_inverse):.3f}")

    binding = binding_stats(out_va, va, va_eps)
    if binding is not None:
        m = np.array(binding["margins"]) if binding["margins"] else np.array([0.0])
        print(f"  [{name}] BINDING ACCURACY overall={binding['overall_acc']:.3f} "
              f"(n={binding['n']}) vs baseline chance", flush=True)
        for k, (acc, n) in sorted(binding["per_kind"].items()):
            print(f"    binding/{k}: acc={acc:.3f} (n={n})")
        print(f"  [{name}] margin distribution: min={m.min():.3f} p25={np.percentile(m, 25):.3f} "
              f"median={np.median(m):.3f} p75={np.percentile(m, 75):.3f} max={m.max():.3f}", flush=True)

    return {"losses": losses, "total_acc": total_acc, "n_resolver_params": n_resolver_params,
            "binding": binding, "peak_rss_mb": peak_rss_mb(), "elapsed_min": elapsed_min}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B"], default=None,
                     help="A = CorefHead, B = SharedScorer. Omit (None) for a no-resolver floor arm.")
    ap.add_argument("--force-binding", choices=["gold", "wrong"], default=None,
                     help="Training is ALWAYS normal (resolver + aux loss train against the TRUE "
                          "gold_index); only the HELD-OUT EVAL batch is built with the collapse "
                          "TEACHER-FORCED to the gold or the wrong candidate. FULL-SCALE ONLY "
                          "(CLAUDE.md: smoke-scale results never gate curriculum validity): forced-gold "
                          "must sit near ceiling and forced-wrong near floor -- that GAP is the proof "
                          "the answer flows through the redirect.")
    ap.add_argument("--cheat", action="store_true",
                     help="Cheat-baseline arm: candidate sets stripped entirely for every writeback/"
                          "instance episode -- no resolver data at all, even an installed resolver "
                          "never fires. Must sit at floor if the task genuinely requires binding.")
    ap.add_argument("--no-gold-eval", action="store_true",
                     help="THE GATE: the EVAL batch only is built with no gold grounding anywhere "
                          "(candidates + the TRAINED resolver only). Training is unaffected.")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--inverse-frac", type=float, default=0.3,
                     help="Fraction of instance episodes that are inverse-query (\"who is X ?\") "
                          "rather than target-question (\"what is X like ?\").")
    ap.add_argument("--rich-frac", type=float, default=0.0,
                     help="Fraction of the mix that is RICH-EPISODE curriculum episodes "
                          "(nsm_ct.curriculum2.RichEpisodeGenerator: N entities 3-8, K "
                          "referring/overwrite statements 1-4) -- added on top of the "
                          "old/writeback/instance thirds, which still split the remainder evenly. "
                          "Default 0.0 = byte-identical to every pre-rich mix.")
    ap.add_argument("--rich-inverse-frac", type=float, default=0.3,
                     help="Fraction of rich episodes that are inverse-query (\"who is X ?\") "
                          "rather than target-question, forwarded to generate_rich_episodes.")
    ap.add_argument("--seed", type=int, default=0)
    add_footprint_args(ap)
    args = ap.parse_args()
    apply_threads(args)

    episodes = build_instance_curriculum(args.episodes, args.seed, inverse_frac=args.inverse_frac,
                                          rich_frac=args.rich_frac, rich_inverse_frac=args.rich_inverse_frac)
    n_wb = sum(1 for e in episodes if e.meta.get("kind") == "writeback")
    n_inst = sum(1 for e in episodes if e.meta.get("kind") == "instance")
    n_rich = sum(1 for e in episodes if e.meta.get("kind") == "rich")
    print(f"=== instance-mix "
          f"{f'(force-binding={args.force_binding}) ' if args.force_binding else ''}"
          f"{'(cheat) ' if args.cheat else ''}{'(no-gold-eval) ' if args.no_gold_eval else ''}"
          f"track={args.track}: {args.episodes} eps ({n_wb} writeback, {n_inst} instance, "
          f"{n_rich} rich, inverse_frac={args.inverse_frac}, rich_frac={args.rich_frac}), "
          f"dim={args.dim}, epochs={args.epochs}, batch_size={args.batch_size} ===", flush=True)
    run_arm(f"track-{args.track}", args.track, episodes, args.dim, args.epochs, args.seed, args.hidden,
             cheat=args.cheat, no_gold_eval=args.no_gold_eval, force_binding=args.force_binding,
             batch_size=args.batch_size)


if __name__ == "__main__":
    main()
