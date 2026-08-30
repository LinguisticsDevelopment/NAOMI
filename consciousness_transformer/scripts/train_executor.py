"""EXECUTOR PHASE 2 training script (dev/EXECUTOR_DESIGN.md Sec.2/Sec.3,
CLAUDE.md's executor D1/D2 decisions): trains :class:`nsm_ct.op_select.
OpSelect`/``ArgSelect`` alongside a :class:`~nsm_ct.clause_reactor.
ClauseReactor` via :meth:`nsm_ct.executor.Executor.run_learned`, on a mixed
corpus covering all SIX gold-program families (:data:`nsm_ct.programs.
FAMILY_NAMES`) -- old L1-6 + M53a ``PronounCurriculumGenerator`` +
``WriteBackCurriculumGenerator`` + ``InstanceCurriculumGenerator`` +
``RichEpisodeGenerator`` + ``DocumentGenerator`` (via
``_train_common.DocumentRunner`` for passage-0 LTM consolidation, see
:func:`build_document_items`) -- and reports the LOFO gate's per-family
numbers (dev/EXECUTOR_DESIGN.md Sec.3).

Scope note (read src/nsm_ct/op_select.py's own module docstring first):
this milestone's ``run_learned`` learns exactly TWO decision points per
clause step (the op-loop's ENTRY op, and EMIT's destination register) --
the frozen v0 register table (D3) leaves no further op/arg ambiguity once
those are fixed. ``op_acc``/``arg_acc`` below report accuracy on those two
decisions, keyed by :data:`nsm_ct.programs.FAMILY_NAMES` (the precise
six-way program taxonomy); ``learned_acc``/``oracle_acc``/``floor_acc``
report per-corpus-source TASK accuracy instead (the six CURRICULA this
script mixes -- ``instance``/``rich`` each straddle TWO program families,
``definite_desc_read`` and ``inverse_query``, since both generators mix
addr-redirect and inverse-query episodes in one corpus; this is a
documented simplification, not a taxonomy error -- the precise per-program
breakdown lives in ``op_acc``/``arg_acc``).

Usage:
    python scripts/train_executor.py --episodes-per-family 60 --dim 24 \\
        --epochs 15 --batch-size 16 --threads 4
    python scripts/train_executor.py --episodes-per-family 60 --dim 24 \\
        --epochs 15 --batch-size 16 --threads 4 \\
        --lofo writeback_addr_redirect

Flags: ``--lofo FAMILY`` (drop that family's trace supervision entirely,
Sec.3's LOFO gate), ``--oracle``/``--floor`` (report-only ceiling/floor
arms, always computed and printed regardless -- these flags additionally
select which arm's accuracy the script exits nonzero against nothing; see
``--gate`` for smoke-scale opt-in gating, default OFF per CLAUDE.md's
"smoke-scale results NEVER gate curriculum validity"), ``--soft-report``
(D1's soft-vs-argmax delta on the EMIT-destination decision),
``--save``/``--load`` (checkpoint the model + heads; NOT
``nsm_ct.checkpoint``'s schema -- that module's config contract has no
slot for ``op_select``/``arg_select``, so this script uses its own small
``torch.save`` dict instead, documented as a deviation in this milestone's
report).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from _train_common import apply_threads, epoch_minibatches, peak_rss_mb, DocumentRunner  # noqa: E402
from nsm_ct import op_select as osl  # noqa: E402
from nsm_ct import programs  # noqa: E402
from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import (  # noqa: E402
    InstanceCurriculumGenerator,
    PronounCurriculumGenerator,
    RichEpisodeGenerator,
    WriteBackCurriculumGenerator,
    generate_document_episodes,
)
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.executor import Executor  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.instances import InstanceRegistry, ProvenanceLog  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.resolver import make_resolver  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

FLAT_FAMILIES = ("old_l1_6", "writeback", "instance", "rich", "pronoun")


# ---------------------------------------------------------------------------
# Corpus construction.
# ---------------------------------------------------------------------------
def build_pronoun_batch(n: int, seed: int, meaning, codec: TPRCodec):
    """PronounCurriculumGenerator batches classify as
    ``"pronoun_value_redirect"`` ONLY with a REAL parser
    (:func:`nsm_ct.clause_reactor._pronoun_context_step` gates on
    ``hasattr(parser, "_parse_graph")``) -- see programs.py's module
    docstring note on this. Returns ``None`` if ``quantum_parser`` is
    unavailable in this environment (mirrors every other script's own
    ``getattr(parser, "_parser", None) is None`` skip contract)."""
    eps = PronounCurriculumGenerator(seed=seed).generate(n)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        return None
    return build_clause_batch(eps, parser, meaning, codec)


def build_document_items(n: int, seed: int, dim: int, meaning, codec: TPRCodec) -> List[dict]:
    """One dict per document: ``{"pre_batches": [...], "final_batch": ...,
    "registry": InstanceRegistry}``. ``pre_batches`` (passage 0, and any
    filler passage) are run through ``_train_common.DocumentRunner`` by
    the CALLER (train/eval loop below) each time they're needed -- rerun
    every call rather than cached once, since passage 0's own forward pass
    uses the model's CURRENT (being-trained) weights, and its consolidated
    LTM tensor is what the final passage's ``recall_link`` collapse reads.
    ``final_batch`` is the ONE passage this script actually runs through
    ``Executor.run_learned`` (its mention step is the family-6
    ``recall_link`` program, dev/EXECUTOR_DESIGN.md Sec.1.1)."""
    episodes = generate_document_episodes(n, seed=seed)
    docs: Dict[str, list] = {}
    for ep in episodes:
        docs.setdefault(ep.meta["doc_id"], []).append(ep)
    items = []
    for doc_id, passages in docs.items():
        passages.sort(key=lambda e: e.meta["passage_index"])
        registry = InstanceRegistry(dim=dim, seed=passages[0].meta["instance_seed"])
        pre_batches = [build_clause_batch([ep], None, meaning, codec, document_registry=registry)
                       for ep in passages[:-1]]
        final_batch = build_clause_batch([passages[-1]], None, meaning, codec, document_registry=registry)
        items.append({"pre_batches": pre_batches, "final_batch": final_batch, "registry": registry})
    return items


def build_corpus(episodes_per_family: int, dim: int, hidden: int, seed: int):
    """Returns ``(model, codec, meaning, corpus)``. ``corpus`` maps each of
    the SIX curriculum names to ``{"train": batch, "val": batch}``
    (``"document"`` maps to ``{"train": [items...], "val": [items...]}``
    instead -- see :func:`build_document_items`). ``"pronoun"`` is omitted
    entirely (not even an empty entry) when ``quantum_parser`` is
    unavailable -- callers must handle a five-family corpus gracefully in
    that environment, same as every other script in this repo does."""
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)
    # use_cand_feature=False (deviation from train_ltm.py's own
    # use_cand_feature=True/cand_feature_extra=2 document-only config,
    # documented here): ClauseReactor._collapse's (and Executor.run()'s
    # verbatim copy of it -- both READ-ONLY for this milestone, pinned by
    # the Phase 1 anchor tests) extra-column zero-padding is only applied
    # when at least one optional extra column (evidence-interaction/
    # from_ltm/recency) is ACTUALLY present that step -- WriteBackCurriculumGenerator's
    # candidate sets populate `cand_features` (the per-candidate slot)
    # but supply NONE of those extra columns, so a resolver built with
    # `cand_feature_extra > 0` and shared across this mixed corpus hits a
    # width mismatch the moment a writeback step reaches `ClauseReactor.
    # forward`/`Executor.run` (Executor.run_learned's OWN copy of this
    # block is fixed, see executor.py; forward()/run() are not, by
    # design, this milestone). Dropping `use_cand_feature` entirely here
    # sidesteps the whole code path for every family -- a documented
    # simplification (this milestone's report), not a workaround for a
    # bug this milestone's files are responsible for.
    resolver = make_resolver("A", dim, hidden)
    model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver)

    n_val = max(1, episodes_per_family // 5)
    n_tr = max(1, episodes_per_family - n_val)

    corpus: Dict[str, dict] = {
        "old_l1_6": {
            "train": build_clause_batch(CurriculumGenerator(max_level=6, seed=seed).generate(n_tr),
                                         None, meaning, codec),
            "val": build_clause_batch(CurriculumGenerator(max_level=6, seed=seed + 1).generate(n_val),
                                       None, meaning, codec),
        },
        "writeback": {
            "train": build_clause_batch(WriteBackCurriculumGenerator(seed=seed).generate(n_tr),
                                         None, meaning, codec),
            "val": build_clause_batch(WriteBackCurriculumGenerator(seed=seed + 1).generate(n_val),
                                       None, meaning, codec),
        },
        "instance": {
            "train": build_clause_batch(InstanceCurriculumGenerator(seed=seed, inverse_frac=0.3).generate(n_tr),
                                         None, meaning, codec),
            "val": build_clause_batch(InstanceCurriculumGenerator(seed=seed + 1, inverse_frac=0.3).generate(n_val),
                                       None, meaning, codec),
        },
        "rich": {
            "train": build_clause_batch(RichEpisodeGenerator(seed=seed, inverse_frac=0.3).generate(n_tr),
                                         None, meaning, codec),
            "val": build_clause_batch(RichEpisodeGenerator(seed=seed + 1, inverse_frac=0.3).generate(n_val),
                                       None, meaning, codec),
        },
        "document": {
            "train": build_document_items(n_tr, seed, dim, meaning, codec),
            "val": build_document_items(n_val, seed + 1, dim, meaning, codec),
        },
    }
    pron_tr = build_pronoun_batch(n_tr, seed, meaning, codec)
    pron_val = build_pronoun_batch(n_val, seed + 1, meaning, codec)
    if pron_tr is not None and pron_val is not None:
        corpus["pronoun"] = {"train": pron_tr, "val": pron_val}
    else:
        print("quantum_parser unavailable -- 'pronoun' (pronoun_value_redirect) family "
              "SKIPPED (see build_pronoun_batch); reports below cover 5 curricula, not 6.")
    return model, codec, meaning, corpus


def family_histogram(corpus: dict, *, split: str = "train") -> Counter:
    """:func:`nsm_ct.programs.family_of_step` counts across every (row,
    step) of ``corpus[*][split]`` -- the six-family corpus-coverage check
    tests/test_executor_phase2.py and this script's own startup report
    both use."""
    hist: Counter = Counter()
    for name, entry in corpus.items():
        if name == "document":
            for item in entry[split]:
                b = item["final_batch"]
                for t in range(b.entity.shape[1]):
                    hist.update(programs.family_of_step(b, t))
            continue
        b = entry[split]
        for t in range(b.entity.shape[1]):
            hist.update(programs.family_of_step(b, t))
    return hist


# ---------------------------------------------------------------------------
# Train / eval loops.
# ---------------------------------------------------------------------------
def _document_ltm(model, item: dict, codec: TPRCodec, *, train: bool):
    if not item["pre_batches"]:
        return None
    runner = DocumentRunner(model)
    reports, _ = runner.run_document(item["pre_batches"], item["registry"], ProvenanceLog(), codec, train=train)
    return reports[-1]["ltm"]


def train_epoch(model, ex: Executor, op_sel, arg_sel, opt, corpus: dict, codec: TPRCodec, *,
                 batch_size: int, seed: int, epoch: int, lofo_family: Optional[str],
                 trace_weight: float) -> float:
    model.train()
    total_loss, n_steps = 0.0, 0
    for name in FLAT_FAMILIES:
        if name not in corpus:
            continue
        batch = corpus[name]["train"]
        n = batch.entity.shape[0]
        for idx in epoch_minibatches(n, batch_size, seed, epoch):
            sub = batch.subset(torch.as_tensor(idx))
            opt.zero_grad()
            out = ex.run_learned(sub, op_sel, arg_sel, lofo_family=lofo_family,
                                  teacher_force=True, trace_weight=trace_weight)
            task_loss = F.cross_entropy(out["answer_logits"], sub.answer)
            loss = task_loss + out["trace_loss"]
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_steps += 1

    doc_items = corpus["document"]["train"]
    step = max(batch_size, 1)
    for start in range(0, len(doc_items), step):
        chunk = doc_items[start:start + step]
        if not chunk:
            continue
        opt.zero_grad()
        chunk_loss = 0.0
        for item in chunk:
            ltm_tensor = _document_ltm(model, item, codec, train=True)
            out = ex.run_learned(item["final_batch"], op_sel, arg_sel, ltm=ltm_tensor,
                                  lofo_family=lofo_family, teacher_force=True, trace_weight=trace_weight)
            task_loss = F.cross_entropy(out["answer_logits"], item["final_batch"].answer)
            doc_loss = task_loss + out["trace_loss"]
            doc_loss.backward()
            chunk_loss += float(doc_loss.detach())
        opt.step()
        total_loss += chunk_loss
        n_steps += len(chunk)
    return total_loss / max(n_steps, 1)


def _acc(out, gold) -> float:
    return (out["answer_logits"].argmax(-1) == gold).float().mean().item()


def evaluate(model, ex: Executor, op_sel, arg_sel, corpus: dict, codec: TPRCodec, *,
             dest_mode: str = "hard") -> Dict[str, dict]:
    """Per-corpus-source ``learned_acc`` (Executor.run_learned,
    ``teacher_force=False`` -- the "no-trace eval" honesty arm: execution
    driven entirely by the model's own predictions), ``oracle_acc``
    (``Executor.run`` -- forced-gold, Sec.3's oracle arm), ``floor_acc``
    (``Executor.run(force_plain_fact=True)`` -- Sec.3's floor arm), plus
    the precise per-program-family ``op_acc``/``arg_acc`` (accumulated
    across whichever curricula exercise each family)."""
    model.eval()
    results: Dict[str, dict] = {}
    with torch.no_grad():
        for name in FLAT_FAMILIES:
            if name not in corpus:
                continue
            batch = corpus[name]["val"]
            learned = ex.run_learned(batch, op_sel, arg_sel, teacher_force=False, dest_mode=dest_mode)
            oracle = ex.run(batch)
            floor = ex.run(batch, force_plain_fact=True)
            results[name] = {
                "learned_acc": _acc(learned, batch.answer),
                "oracle_acc": _acc(oracle, batch.answer),
                "floor_acc": _acc(floor, batch.answer),
                "op_acc": learned["op_acc"], "arg_acc": learned["arg_acc"],
                "write_violations": learned["write_violations"],
            }

        doc_items = corpus["document"]["val"]
        lc = oc = fc = total = 0
        op_c: Counter = Counter(); op_n: Counter = Counter()
        arg_c: Counter = Counter(); arg_n: Counter = Counter()
        for item in doc_items:
            ltm_tensor = _document_ltm(model, item, codec, train=False)
            fb = item["final_batch"]
            learned = ex.run_learned(fb, op_sel, arg_sel, ltm=ltm_tensor, teacher_force=False, dest_mode=dest_mode)
            oracle = ex.run(fb, ltm=ltm_tensor)
            floor = ex.run(fb, ltm=ltm_tensor, force_plain_fact=True)
            gold = fb.answer
            lc += int((learned["answer_logits"].argmax(-1) == gold).sum())
            oc += int((oracle["answer_logits"].argmax(-1) == gold).sum())
            fc += int((floor["answer_logits"].argmax(-1) == gold).sum())
            total += fb.entity.shape[0]
            for fam, (c, n) in learned["op_acc"].items():
                op_c[fam] += c; op_n[fam] += n
            for fam, (c, n) in learned["arg_acc"].items():
                arg_c[fam] += c; arg_n[fam] += n
        if total:
            results["document"] = {
                "learned_acc": lc / total, "oracle_acc": oc / total, "floor_acc": fc / total,
                "op_acc": {f: (op_c[f], op_n[f]) for f in op_c},
                "arg_acc": {f: (arg_c[f], arg_n[f]) for f in arg_c},
                "write_violations": 0,
            }
    return results


def print_report(results: Dict[str, dict], *, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'corpus':<12} {'learned':>8} {'oracle':>8} {'floor':>8}")
    for name, r in results.items():
        print(f"{name:<12} {r['learned_acc']:>8.3f} {r['oracle_acc']:>8.3f} {r['floor_acc']:>8.3f}")
    print("-- per-program-family op/arg-selection accuracy (vs gold trace) --")
    fam_op: Dict[str, list] = {}
    fam_arg: Dict[str, list] = {}
    for r in results.values():
        for fam, (c, n) in r["op_acc"].items():
            fam_op.setdefault(fam, [0, 0])
            fam_op[fam][0] += c; fam_op[fam][1] += n
        for fam, (c, n) in r["arg_acc"].items():
            fam_arg.setdefault(fam, [0, 0])
            fam_arg[fam][0] += c; fam_arg[fam][1] += n
    for fam in programs.FAMILY_NAMES:
        c, n = fam_op.get(fam, [0, 0])
        ac, an = fam_arg.get(fam, [0, 0])
        op_str = f"{c / n:.3f}" if n else "n/a"
        arg_str = f"{ac / an:.3f}" if an else "n/a"
        print(f"  {fam:<26} op_acc={op_str:>6} (n={n:<4}) arg_acc={arg_str:>6} (n={an})")
    write_violations = sum(r["write_violations"] for r in results.values())
    print(f"write_violations (total)={write_violations}")


# ---------------------------------------------------------------------------
# Kill-criteria instrumentation (dev/EXECUTOR_DESIGN.md Sec.3): wall-clock
# ratio vs forward(), peak RSS -- mirrors tests/test_executor_phase1.py's
# own kill-criteria test, extended to run_learned().
# ---------------------------------------------------------------------------
def report_kill_criteria(model, ex: Executor, op_sel, arg_sel, batch, *, n_reps: int = 5) -> None:
    model.eval()
    T = batch.entity.shape[1]
    with torch.no_grad():
        model(batch)
        ex.run_learned(batch, op_sel, arg_sel, teacher_force=False)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(batch)
        t_forward = (time.perf_counter() - t0) / n_reps
        t0 = time.perf_counter()
        for _ in range(n_reps):
            ex.run_learned(batch, op_sel, arg_sel, teacher_force=False)
        t_learned = (time.perf_counter() - t0) / n_reps
    ratio = (t_learned / T) / (t_forward / T) if t_forward > 0 else float("inf")
    print(f"\n[kill-criteria] forward/clause={t_forward / T * 1e3:.4f}ms "
          f"run_learned/clause={t_learned / T * 1e3:.4f}ms ratio={ratio:.2f}x "
          f"k_max={ex.k_max} peak_rss_mb={peak_rss_mb():.1f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-per-family", type=int, default=60)
    ap.add_argument("--dim", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lofo", type=str, default=None, choices=list(programs.FAMILY_NAMES),
                     help="Drop this family's trace supervision entirely (Sec.3 LOFO gate). "
                          "Its EMIT-destination decision is still shaped by TASK loss alone "
                          "(straight-through), per D1.")
    ap.add_argument("--trace-weight", type=float, default=1.0, help="Initial trace-loss weight.")
    ap.add_argument("--trace-weight-final", type=float, default=None,
                     help="Final trace-loss weight (linear anneal over epochs); default = "
                          "--trace-weight (no anneal).")
    ap.add_argument("--soft-report", action="store_true",
                     help="Also evaluate with dest_mode='soft' and report the delta vs the "
                          "default hard (D1 straight-through) arm.")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--load", type=str, default=None)
    args = ap.parse_args()

    apply_threads(args)
    torch.manual_seed(args.seed)

    model, codec, meaning, corpus = build_corpus(args.episodes_per_family, args.dim, args.hidden, args.seed)
    ex = Executor(model)
    op_sel = osl.OpSelect(k_max=ex.k_max)
    arg_sel = osl.ArgSelect(op_sel.op_embed, ctrl_dim=op_sel.ctrl_dim)
    print(f"OpSelect params={sum(p.numel() for p in op_sel.parameters())} "
          f"ArgSelect params={sum(p.numel() for p in arg_sel.parameters())} "
          f"combined={osl.count_params(op_sel, arg_sel)} (budget: <=10000)")

    hist = family_histogram(corpus, split="train")
    print("family_of_step coverage (train split):", dict(hist))
    for fam in programs.FAMILY_NAMES:
        if hist.get(fam, 0) == 0:
            print(f"  WARNING: family {fam!r} has ZERO coverage in this corpus.")

    if args.load:
        ckpt = torch.load(args.load, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        op_sel.load_state_dict(ckpt["op_select"])
        arg_sel.load_state_dict(ckpt["arg_select"])
        print(f"Loaded checkpoint from {args.load}; skipping training.")
    else:
        params = list(model.parameters()) + list(op_sel.parameters()) + list(arg_sel.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        tw_final = args.trace_weight_final if args.trace_weight_final is not None else args.trace_weight
        t0 = time.time()
        for epoch in range(args.epochs):
            tw = args.trace_weight + (tw_final - args.trace_weight) * (epoch / max(args.epochs - 1, 1))
            mean_loss = train_epoch(model, ex, op_sel, arg_sel, opt, corpus, codec,
                                     batch_size=args.batch_size, seed=args.seed, epoch=epoch,
                                     lofo_family=args.lofo, trace_weight=tw)
            print(f"epoch {epoch + 1:3d}/{args.epochs} mean_loss={mean_loss:.4f} trace_weight={tw:.3f}")
        print(f"training wall-clock: {(time.time() - t0) / 60:.2f} min")

    results = evaluate(model, ex, op_sel, arg_sel, corpus, codec)
    title = "EVAL" if not args.lofo else f"EVAL (LOFO held-out family: {args.lofo})"
    print_report(results, title=title)

    if args.lofo and args.lofo in dict(hist) and hist.get(args.lofo, 0) > 0:
        # Locate which corpus source(s) actually exercise the held-out
        # family, for the short LOFO-vs-oracle-vs-floor callout the report
        # asks for -- op_acc/arg_acc already isolate it precisely; for
        # learned/oracle/floor TASK accuracy we report the corpus source
        # whose PRIMARY family matches (documented per-corpus-source
        # simplification, see module docstring).
        primary = {"plain_fact": "old_l1_6", "pronoun_value_redirect": "pronoun",
                   "writeback_addr_redirect": "writeback", "definite_desc_read": "instance",
                   "inverse_query": "instance", "recall_link": "document"}.get(args.lofo)
        if primary and primary in results:
            r = results[primary]
            print(f"\n[LOFO SMOKE NUMBERS -- do NOT gate on these] held-out family={args.lofo!r} "
                  f"(primary corpus source={primary!r}): "
                  f"learned_acc={r['learned_acc']:.3f} oracle_acc={r['oracle_acc']:.3f} "
                  f"floor_acc={r['floor_acc']:.3f}")

    if args.soft_report:
        soft_results = evaluate(model, ex, op_sel, arg_sel, corpus, codec, dest_mode="soft")
        print("\n=== --soft-report: soft vs argmax (D1) delta, per corpus source ===")
        for name in results:
            hard_acc = results[name]["learned_acc"]
            soft_acc = soft_results[name]["learned_acc"]
            print(f"  {name:<12} hard={hard_acc:.3f} soft={soft_acc:.3f} delta={soft_acc - hard_acc:+.3f}")

    kc_batch = corpus.get("rich", corpus.get("instance", corpus.get("old_l1_6")))
    if kc_batch is not None:
        report_kill_criteria(model, ex, op_sel, arg_sel, kc_batch["val"])

    if args.save:
        torch.save({
            "model": model.state_dict(), "op_select": op_sel.state_dict(), "arg_select": arg_sel.state_dict(),
            "config": {"dim": args.dim, "hidden": args.hidden, "k_max": ex.k_max},
        }, args.save)
        print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
