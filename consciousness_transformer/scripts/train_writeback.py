"""M57b training script: resolver-driven WRITE-BACK -- the resolver's
collapsed choice redirects the write ADDRESS ("she is tall ." must land on
mary's own node), not just the value (M53b's pronoun-antecedent-for-a-value
capability, unchanged and untouched by this script). Sibling to
scripts/train_resolver.py (same arm/report pattern -- track dispatch, aux
loss, per-kind task accuracy, a resolver-binding-accuracy metric, margin
distribution) rather than an extension of that already-982-line file, since
M57b only needs ONE collapse kind (the SAME ``resolver=`` slot M53b already
installs on ClauseReactor -- write-back candidate sets are still
:class:`nsm_ct.membrane.EntityCandidateSet`, just with ``addr_redirect=True``
-- no new sense_resolver/hyp_resolver plumbing at all).

Curriculum mix (:func:`build_writeback_curriculum`): 50% old L1-6 + 50%
the NEW write-back curriculum (nsm_ct.curriculum2.generate_writeback_episodes)
-- mirrors scripts/train_resolver.py's build_m54b_curriculum/
build_m55a_curriculum "isolate the one new capability" reasoning.

Arms (the M57b v2 milestone spec asks for five):
    python scripts/train_writeback.py --track A                        # normal arm
    python scripts/train_writeback.py --track A --force-binding gold   # forced-gold eval (ceiling)
    python scripts/train_writeback.py --track A --force-binding wrong  # forced-wrong eval (floor) --
                                                                          THE full-scale validity
                                                                          check (CLAUDE.md:
                                                                          smoke-scale NEVER gates
                                                                          curriculum validity, only
                                                                          this gold-vs-wrong GAP
                                                                          does): forced-gold must
                                                                          sit near ceiling,
                                                                          forced-wrong near floor.
    python scripts/train_writeback.py --track A --cheat                # cheat-baseline floor check
    python scripts/train_writeback.py --track A --no-gold-eval         # eval batches built with NO
                                                                          gold grounding of the
                                                                          pronoun address anywhere
                                                                          (candidates + the TRAINED
                                                                          resolver only)

``--force-binding {gold,wrong}`` (v2, replacing the old ``--wrong-binding``
curriculum-level aux-gold-corruption arm -- see
nsm_ct.clause_reactor.ClauseReactor._collapse's teacher-forcing paragraph):
training is ALWAYS normal (the resolver + its aux loss train against the
TRUE gold_index, regardless of this flag) -- only the HELD-OUT EVAL batch is
built with ``writeback_force=`` set, so the reported accuracy reflects the
collapse being forced to the gold or the wrong candidate REGARDLESS of the
trained resolver's own prediction. ``--cheat`` affects TRAINING data too
(the curriculum's own episodes still generate normally in v2 -- only
``build_clause_batch``'s ``writeback_cheat=`` strips candidate data, for
both train and eval); ``--no-gold-eval`` affects ONLY the held-out eval
batch (training still sees gold_index for its aux loss) -- see
:func:`run_arm`'s docstring for exactly which flag touches which batch.
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
from nsm_ct.curriculum2 import generate_writeback_episodes  # noqa: E402
from nsm_ct.episode import CurriculumGenerator, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.resolver import make_resolver  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

AUX_WEIGHT = 0.5   # resolver cross-entropy weight added to the answer loss -- same constant train_resolver.py uses


def build_writeback_curriculum(n_episodes: int, seed: int):
    """50% old L1-6 + 50% write-back episodes, deterministic given
    ``(n_episodes, seed)``. Mirrors ``scripts/train_resolver.py``'s
    ``build_m54b_curriculum``/``build_m55a_curriculum`` exactly (isolate the
    one new capability; old L1-6 supplies the non-candidate-bearing bulk of
    the mix so the reactor still has to hold ordinary facts too, not just
    write-back ones). v2: no ``wrong_binding`` flag anymore -- the
    curriculum itself never corrupts its own gold_index; validity forcing
    happens at collapse time instead (``--force-binding``, see module
    docstring)."""
    n_new = n_episodes // 2
    n_old = n_episodes - n_new
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    new = generate_writeback_episodes(n_new, seed=seed + 1)
    episodes = old + new
    order = np.random.RandomState(seed + 2).permutation(len(episodes))
    return [episodes[i] for i in order]


def writeback_binding_stats(out, batch, va_eps):
    """WRITE-BACK BINDING ACCURACY: overall + anti-recency half (the SAME
    metric shape ``scripts/train_resolver.py``'s ``resolver_binding_stats``
    reports for M53's pronoun-antecedent-for-a-value capability --
    WriteBackCurriculumGenerator sets the SAME ``antecedent_recency`` meta
    key). Reads off ONE resolver-carrying forward pass + the episodes' own
    meta; ``None`` if the batch carries no candidate data at all (e.g. the
    cheat arm)."""
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


def run_arm(name: str, track, episodes, dim: int, epochs: int, seed: int, hidden: int = 128,
            cheat: bool = False, no_gold_eval: bool = False, force_binding: str = None) -> dict:
    """``track``: "A" | "B" | None (None = no resolver installed at all --
    with no resolver, the write-back step's entity NEVER redirects, so this
    is a genuine floor arm too, distinct from ``--cheat`` -- see the README
    at the top of this file for which flag does what).

    ``cheat`` forwards to ``build_clause_batch``'s ``writeback_cheat=`` for
    BOTH the train and eval batch (the cheat-baseline arm: no candidate data
    anywhere, so the resolver -- even if installed -- never fires; the task
    must therefore sit at floor if the curriculum genuinely requires
    binding). ``no_gold_eval`` forwards to ``writeback_no_gold=`` for the
    EVAL batch ONLY (training still sees gold_index for its aux loss) --
    candidates + the trained resolver only, no gold grounding of the
    pronoun address anywhere at eval time. ``force_binding`` (``"gold"`` |
    ``"wrong"`` | ``None``, v2's honest validity machinery) forwards to
    ``writeback_force=`` for the EVAL batch ONLY -- training is always
    normal (resolver + aux loss against TRUE gold_index); THE M57b GATE is
    the forced-gold vs forced-wrong accuracy GAP this produces (see module
    docstring).
    """
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

    t0 = time.time()
    model.train()
    losses = []
    for i in range(epochs):
        out = model(tr)
        loss = F.cross_entropy(out["answer_logits"], gold_tr)
        if resolver is not None and "resolver_logits" in out:
            cg = tr.cand_gold
            has_cand = cg >= 0
            if bool(has_cand.any()):
                aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
                loss = loss + AUX_WEIGHT * aux
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (i + 1) % 20 == 0 or i == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(va)["answer_logits"].argmax(-1) == gold_va).float().mean()
            model.train()
            print(f"  [{name}] epoch {i+1:3d} loss={loss.item():.3f} val={acc:.3f}", flush=True)
    elapsed_min = (time.time() - t0) / 60

    model.eval()
    with torch.no_grad():
        out_va = model(va)
    pred = out_va["answer_logits"].argmax(-1)
    total_acc = float((pred == gold_va).float().mean())
    print(f"  [{name}] val total={total_acc:.3f}  resolver_params={n_resolver_params}"
          f"  time={elapsed_min:.2f} min", flush=True)

    per_kind = {}
    for i, e in enumerate(va_eps):
        per_kind.setdefault(str(e.meta.get("kind", "old")), []).append(bool(pred[i] == gold_va[i]))
    for k in sorted(per_kind):
        w = per_kind[k]
        print(f"    kind {k}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")

    # v2 per-subset breakdown (in EVERY arm): referent-targeted questions
    # ("what is {referent} like ?", answerable only via the redirect+
    # overwrite) vs other-targeted questions (the redirect-free control
    # condition, answerable from the other entity's own named attribute
    # alone) -- see curriculum2.WriteBackCurriculumGenerator's v2 docstring.
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

    binding = writeback_binding_stats(out_va, va, va_eps)
    if binding is not None:
        m = np.array(binding["margins"]) if binding["margins"] else np.array([0.0])
        print(f"  [{name}] WRITE-BACK BINDING ACCURACY overall={binding['overall_acc']:.3f} "
              f"(n={binding['n']}) anti-recency={binding['anti_recency_acc']:.3f} "
              f"(n={binding['n_anti_recency']}) vs baseline 0.500/0.000", flush=True)
        print(f"  [{name}] write-back margin distribution: min={m.min():.3f} p25={np.percentile(m, 25):.3f} "
              f"median={np.median(m):.3f} p75={np.percentile(m, 75):.3f} max={m.max():.3f}", flush=True)

    return {"losses": losses, "total_acc": total_acc, "n_resolver_params": n_resolver_params, "binding": binding,
            "writeback_subsets": {k: (sum(v) / len(v) if v else float("nan")) for k, v in wb_subsets.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B"], default=None,
                     help="A = CorefHead, B = SharedScorer. Omit (None) for a no-resolver floor arm "
                          "(the write-back address never redirects at all).")
    ap.add_argument("--force-binding", choices=["gold", "wrong"], default=None,
                     help="M57b v2 honest validity machinery (replaces the old --wrong-binding "
                          "curriculum-level aux-gold-corruption arm): training is ALWAYS normal (the "
                          "resolver + its aux loss train against the TRUE gold_index); only the "
                          "HELD-OUT EVAL batch is built with the collapse TEACHER-FORCED to the gold "
                          "or the wrong candidate, regardless of the trained resolver's own logits. "
                          "FULL-SCALE ONLY (CLAUDE.md: smoke-scale results never gate curriculum "
                          "validity): forced-gold must sit near ceiling and forced-wrong near floor -- "
                          "that GAP is the proof the answer flows through the redirect. Run both "
                          "'gold' and 'wrong' as separate invocations to see the gap.")
    ap.add_argument("--cheat", action="store_true",
                     help="M57b cheat-baseline arm: candidate sets stripped entirely for every write-back "
                          "episode (writeback_cheat=True on both train and eval batches) -- no resolver "
                          "data at all, so even an installed resolver never fires. Must sit at floor if the "
                          "task genuinely requires binding.")
    ap.add_argument("--no-gold-eval", action="store_true",
                     help="THE M57b GATE: the EVAL batch only is built with writeback_no_gold=True -- "
                          "no gold grounding of the pronoun address anywhere (candidates + the TRAINED "
                          "resolver only). Training is unaffected (still sees gold_index for its aux loss).")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    episodes = build_writeback_curriculum(args.episodes, args.seed)
    n_wb = sum(1 for e in episodes if e.meta.get("kind") == "writeback")
    print(f"=== write-back "
          f"{f'(force-binding={args.force_binding}) ' if args.force_binding else ''}"
          f"{'(cheat) ' if args.cheat else ''}{'(no-gold-eval) ' if args.no_gold_eval else ''}"
          f"track={args.track}: {args.episodes} eps ({n_wb} write-back), dim={args.dim}, "
          f"epochs={args.epochs} ===", flush=True)
    run_arm(f"track-{args.track}", args.track, episodes, args.dim, args.epochs, args.seed, args.hidden,
             cheat=args.cheat, no_gold_eval=args.no_gold_eval, force_binding=args.force_binding)


if __name__ == "__main__":
    main()
