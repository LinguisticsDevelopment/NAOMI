"""M58b: THE ZERO-SHOT PROSE NUMBER.

Evaluates a FROZEN checkpoint (``nsm_ct.checkpoint.save_checkpoint`` --
today, M60's ``runs/m60_checkpoint.pt``, trained entirely on synthetic
curricula) on real-prose episodes produced by scripts/convert_corpus.py
(RESEARCH_NOTES M58a/M58c) -- PURE EVALUATION, no training step, no gradient,
no optimizer. This is the "once checkpointing lands" seam
scripts/eval_checkpoint.py's own tail names: that tool only ever built
batches from ``train_instances.py``'s synthetic curriculum generator; this
one builds them from a JSONL file of converted :class:`~nsm_ct.episode.Episode`
objects instead, through the SAME ``build_clause_batch`` (the "old",
parser-based path -- see ``nsm_ct.corpus``'s own module docstring: a prose
episode's ``kind == "prose"`` meta key routes it there with no new branch
needed in ``build_clause_batch`` at all).

Report reuses the SAME abstain/cleanup computation
``scripts/train_instances.py``'s ``run_arm`` already established (``model.
cleanup = True`` at eval, ``out["cleanup_index"]``/``out["cleanup_abstain"]``
from ``ClauseReactor.forward``'s ``ops.cleanup`` block) -- one report
implementation, not a second hand-rolled one.

Sanity guard (a real-text episode's parse can fail differently than it did
at CONVERSION time, if the parser/lexicon changed in between --
dev/PROSE_FAILURE_TAXONOMY.md's own worklist keeps moving): every episode is
first built ALONE, in a try/except; anything that raises, or silently drops
its own row (``build_clause_batch``'s own "no question entity resolved ->
skip the episode" contract -- an empty ``rows`` list makes its own ``max()``
call raise ``ValueError``, so this is really the same case), is counted as a
BUILD FAILURE and excluded from the scored batch -- never a crash.

Usage:
    python scripts/eval_prose.py --ckpt runs/m60_checkpoint.pt \\
        --episodes runs/prose_episodes.jsonl --verbose
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from _train_common import eval_minibatched  # noqa: E402  (scripts/ already on sys.path -- this is a sibling module)
from nsm_ct.checkpoint import load_checkpoint  # noqa: E402
from nsm_ct.clause_reactor import ClauseBatch, build_clause_batch  # noqa: E402
from nsm_ct.episode import Episode  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

_EPISODE_FIELDS = {f.name for f in dataclasses.fields(Episode)}


# ---------------------------------------------------------------------------
# 1. JSONL round-trip loader
# ---------------------------------------------------------------------------
def load_episodes(path: str) -> List[Episode]:
    """Reads one :class:`Episode` per line, written by
    ``scripts/convert_corpus.py`` as ``json.dumps(dataclasses.asdict(ep))``.
    Keys not on :class:`Episode` are dropped defensively (forward-compat);
    every key ``dataclasses.asdict`` emits IS a real ``Episode`` field, so
    this is a no-op filter in practice, not a silent-data-loss path.
    """
    episodes: List[Episode] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kwargs = {k: v for k, v in row.items() if k in _EPISODE_FIELDS}
            episodes.append(Episode(**kwargs))
    return episodes


# ---------------------------------------------------------------------------
# 2. per-episode defensive batch build (the sanity guard)
# ---------------------------------------------------------------------------
def build_one(ep: Episode, parser, meaning_resolver, codec: TPRCodec) -> Tuple[Optional[ClauseBatch], Optional[str]]:
    """Builds ONE episode into a 1-row :class:`ClauseBatch`. Returns
    ``(batch, None)`` on success or ``(None, reason)`` on any failure --
    never raises."""
    try:
        batch = build_clause_batch([ep], parser, meaning_resolver, codec)
    except Exception as exc:  # noqa: BLE001 -- deliberate catch-all, see module docstring
        return None, f"{type(exc).__name__}: {exc}"
    if batch.entity.shape[0] != 1:
        return None, "dropped by build_clause_batch (no row produced)"
    return batch, None


def partition_buildable(episodes: List[Episode], parser, meaning_resolver, codec: TPRCodec
                         ) -> Tuple[List[Episode], List[Tuple[Episode, str]]]:
    """Probes every episode ALONE (see module docstring) and splits into
    (buildable, [(episode, reason), ...] for failures). The buildable list
    preserves input order -- it is what the final, batched, scored
    ``build_clause_batch`` call below is built from."""
    ok: List[Episode] = []
    failed: List[Tuple[Episode, str]] = []
    for ep in episodes:
        _batch, reason = build_one(ep, parser, meaning_resolver, codec)
        if reason is None:
            ok.append(ep)
        else:
            failed.append((ep, reason))
    return ok, failed


# ---------------------------------------------------------------------------
# 3. baselines
# ---------------------------------------------------------------------------
def random_guess_floor(episodes: List[Episode]) -> float:
    """Mean of 1/n_options over the evaluated episodes -- the expected
    accuracy of picking uniformly at random among each episode's own MC
    options."""
    if not episodes:
        return float("nan")
    return sum(1.0 / len(e.options) for e in episodes) / len(episodes)


def per_group_accuracy(episodes: List[Episode], correct, group_fn) -> Dict[str, Tuple[int, int]]:
    """Buckets ``episodes`` by ``group_fn(episode)`` and returns
    ``{group: (hits, n)}``. ``correct`` is a same-length bool sequence
    (index-aligned with ``episodes``). The groups' ``n`` values always sum
    to ``len(episodes)`` by construction -- every episode falls into
    exactly one bucket -- which is exactly the "per-relation/per-source
    counts sum to total" invariant the eval report's own asserts check."""
    counts: Dict[str, List[bool]] = defaultdict(list)
    for i, e in enumerate(episodes):
        counts[group_fn(e)].append(bool(correct[i]))
    return {g: (sum(w), len(w)) for g, w in counts.items()}


def majority_baseline(episodes: List[Episode]) -> Tuple[float, str]:
    """The "always guess the single most common gold answer text" floor --
    the standard majority-class baseline for a varying-label-set MC task
    (each episode's own option SET differs, but ``answer_text`` is directly
    comparable across episodes). Returns ``(accuracy, modal_text)``."""
    if not episodes:
        return float("nan"), ""
    counts = Counter(e.answer_text for e in episodes)
    modal_text, modal_count = counts.most_common(1)[0]
    return modal_count / len(episodes), modal_text


# ---------------------------------------------------------------------------
# 4. report
# ---------------------------------------------------------------------------
def _group_of(ep: Episode) -> str:
    return "synthetic" if ep.meta.get("source_doc", "").startswith("synthetic_") else "real"


def run(args) -> None:
    episodes = load_episodes(args.episodes)
    print(f"=== eval_prose: loaded {len(episodes)} episodes from {args.episodes} ===", flush=True)
    if not episodes:
        print("no episodes to evaluate -- nothing to do.")
        return

    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        sys.exit(1)
    meaning_resolver = NSMMeaningResolver()

    model, ckpt_config, ckpt_extra = load_checkpoint(args.ckpt)
    dim = ckpt_config.get("dim", 48)
    codec = TPRCodec(dim=dim, max_pos=ckpt_config.get("codec_max_pos", 64))
    print(f"=== loaded checkpoint {args.ckpt}: dim={dim} hidden={ckpt_config.get('hidden')} "
          f"track={ckpt_config.get('track')} git_commit={ckpt_config.get('git_commit')} "
          f"trained_total_acc={ckpt_extra.get('total_acc')} ===", flush=True)

    buildable, failures = partition_buildable(episodes, parser, meaning_resolver, codec)
    print(f"=== build: {len(buildable)}/{len(episodes)} episodes built ok, "
          f"{len(failures)} build-failure(s) ===", flush=True)
    for ep, reason in failures[:10]:
        print(f"  BUILD-FAILURE [{ep.meta.get('source_doc')}]: {reason}  question={ep.question!r}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more build-failure(s)")

    if not buildable:
        print("no buildable episodes -- nothing to score.")
        return

    batch = build_clause_batch(buildable, parser, meaning_resolver, codec)
    gold = torch.tensor([e.answer_idx for e in buildable])

    # M60 CLEANUP wiring (reused verbatim from scripts/train_instances.py's
    # run_arm): a freshly-loaded checkpoint's model.cleanup defaults to
    # False (build_model/the ClauseReactor constructor default) -- set it
    # explicitly here so the abstain/margin report below actually runs.
    model.cleanup = True
    model.eval()
    out = eval_minibatched(model, batch, args.batch_size)
    pred = out["answer_logits"].argmax(-1)
    correct = (pred == gold)
    n = len(buildable)
    overall_acc = float(correct.float().mean())

    floor = random_guess_floor(buildable)
    maj_acc, maj_text = majority_baseline(buildable)

    print(f"\n=== THE ZERO-SHOT PROSE NUMBER ===")
    print(f"n episodes evaluated: {n}  (loaded={len(episodes)}, build-failures={len(failures)})")
    print(f"overall accuracy: {overall_acc:.3f}")
    print(f"random-guess floor (mean 1/n_options): {floor:.3f}")
    print(f"majority baseline (always {maj_text!r}): {maj_acc:.3f}")

    # per-relation
    print("\nper-relation:")
    rel_acc = per_group_accuracy(buildable, correct, lambda e: e.meta.get("relation", "?"))
    total_rel = sum(cnt for _hits, cnt in rel_acc.values())
    for rel in sorted(rel_acc):
        hits, cnt = rel_acc[rel]
        print(f"  {rel:<10} {hits}/{cnt} = {hits / cnt:.3f}")
    assert total_rel == n, "per-relation counts must sum to n evaluated episodes"

    # per-source (synthetic_prose vs real)
    print("\nper-source:")
    src_acc = per_group_accuracy(buildable, correct, _group_of)
    total_src = sum(cnt for _hits, cnt in src_acc.values())
    for grp in ("synthetic", "real"):
        if grp not in src_acc:
            continue
        hits, cnt = src_acc[grp]
        print(f"  {grp:<10} {hits}/{cnt} = {hits / cnt:.3f}")
    assert total_src == n, "per-source counts must sum to n evaluated episodes"

    # abstain / cleanup (reused from train_instances.py's run_arm). train_instances.py
    # hard-asserts cleanup_index == pred (ops.cleanup's own docstring invariant: the
    # SAME cosine(r, options) computation as answer_logits, just per-row instead of
    # batched) -- kept a SOFT check here instead, because real prose text can produce
    # a genuine EXACT top1==top2 cosine tie (margin == 0.0) where torch.topk's and
    # torch.einsum-argmax's tie-break conventions can pick different indices; such a
    # row is already flagged abstain regardless of which index "wins", so this is
    # cosmetic, not a scoring bug -- but it's a real edge case a synthetic-only
    # curriculum (always well-separated option meanings) never exercised, worth
    # surfacing rather than crashing the whole eval over one tied row.
    if "cleanup_index" in out:
        mismatch = out["cleanup_index"] != pred
        n_mismatch = int(mismatch.sum())
        if n_mismatch:
            tied_margins = out["cleanup_margin"][mismatch]
            print(f"\nNOTE: cleanup_index disagreed with answer_logits.argmax on {n_mismatch}/{n} "
                  f"row(s) -- margins there: {[round(float(m), 4) for m in tied_margins]} "
                  f"(0.0 == an exact top1/top2 tie, already abstain-flagged either way).")
        cleanup_abstain = out["cleanup_abstain"].bool()
        abstain_rate = float(cleanup_abstain.float().mean())
        confident = ~cleanup_abstain
        acc_confident = (float((pred[confident] == gold[confident]).float().mean())
                         if bool(confident.any()) else float("nan"))
        margins = out["cleanup_margin"]
        print(f"\nCLEANUP: abstain_rate={abstain_rate:.3f} "
              f"acc_when_confident={acc_confident:.3f} (n={int(confident.sum())}) "
              f"vs overall={overall_acc:.3f} (n={n})")
        print(f"  margin: min={float(margins.min()):.3f} mean={float(margins.mean()):.3f} "
              f"max={float(margins.max()):.3f}")
    else:
        print("\nCLEANUP: no cleanup output in forward pass (unexpected -- model.cleanup was set True).")

    if args.verbose:
        print("\n=== per-episode dump ===")
        for i, e in enumerate(buildable):
            mark = "OK  " if bool(correct[i]) else "WRONG"
            print(f"[{mark}] source={e.meta.get('source_doc')} relation={e.meta.get('relation')}")
            print(f"       question: {e.question}")
            print(f"       options:  {e.options}")
            print(f"       predicted: {e.options[int(pred[i])]!r}   gold: {e.answer_text!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="Frozen checkpoint path (nsm_ct.checkpoint.save_checkpoint).")
    ap.add_argument("--episodes", type=str, default="runs/prose_episodes.jsonl",
                     help="JSONL file of converted prose episodes (scripts/convert_corpus.py --out).")
    ap.add_argument("--batch-size", type=int, default=32,
                     help="Minibatch size for the scored forward pass (0 = full-batch).")
    ap.add_argument("--verbose", action="store_true", help="Dump every evaluated episode's question/options/prediction/gold.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
