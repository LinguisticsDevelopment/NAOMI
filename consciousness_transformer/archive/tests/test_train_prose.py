"""Tests for M58e's scripts/train_prose.py -- the first prose-training
script (RESEARCH_NOTES M58a-d's own "Next" line: retrain on the corrected
PLACE grounding, re-measure zero-shot on held-out documents).

Tiers, matching the module's own shape (kept fast -- run ONLY this file,
``pytest tests/test_train_prose.py -q``, target <=60s total):
  1. ``split_by_document`` -- no document leaks between train/eval, seeded
     determinism. Parser-free (fake Episode objects).
  2. ``load_episodes`` reuse -- 3 hand-made episodes, and that it really is
     the SAME function object as ``scripts.eval_prose.load_episodes`` (not
     a duplicate implementation). Parser-free.
  3. ``build_mixed_training_set`` -- the training mix hits the target
     curriculum fraction and both kinds are present; ``curriculum_frac=0``
     is prose-only; deterministic given the seed. Parser-free (curriculum
     generation is pure Python, no parser needed).
  4. A tiny end-to-end run (5 real converted prose episodes across >=2
     documents, dim=16, 2 epochs) -- completes and emits every report key/
     header, accuracy value irrelevant. Needs quantum_parser; skipped
     cleanly (not failed) when unavailable, matching every other
     parser-dependent test in this suite.
  5. ``--frozen-eval`` -- runs against a freshly-saved (untrained)
     checkpoint with NO optimizer ever constructed (training truly
     skipped), and ``--check-args`` validation for the flag combination.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, _SRC)
sys.path.insert(0, _SCRIPTS)

from nsm_ct.episode import Episode  # noqa: E402

import eval_prose  # noqa: E402
import train_prose  # noqa: E402


def _fake_ep(source_doc: str, question: str = "q", answer: str = "a") -> Episode:
    """A parser-free stand-in Episode -- only ``meta['source_doc']`` and
    the MC-answer shape matter for tiers 1 and 3 below."""
    options = [answer, "b", "c", "d"]
    return Episode(
        context=["mary is in the garden ."],
        question=question,
        answer_text=answer,
        options=options,
        answer_idx=0,
        meta={"kind": "prose", "source_doc": source_doc, "relation": "PLACE"},
    )


# ---------------------------------------------------------------------------
# 1. split_by_document
# ---------------------------------------------------------------------------
def test_split_by_document_no_leaks_and_covers_every_episode():
    eps = []
    for d in range(6):
        for _ in range(3):
            eps.append(_fake_ep(f"doc{d}"))
    tr, ev = train_prose.split_by_document(eps, holdout_docs=2, seed=0)

    tr_docs = {e.meta["source_doc"] for e in tr}
    ev_docs = {e.meta["source_doc"] for e in ev}
    assert tr_docs.isdisjoint(ev_docs), "a document leaked into both train and eval"
    assert len(ev_docs) == 2
    assert len(tr) + len(ev) == len(eps)
    # every episode from a held-out document is in eval, none of its siblings in train
    for d in ev_docs:
        assert all(e.meta["source_doc"] != d for e in tr)


def test_split_by_document_holdout_frac():
    eps = [_fake_ep(f"doc{d}") for d in range(10) for _ in range(2)]
    tr, ev = train_prose.split_by_document(eps, holdout_frac=0.3, seed=1)
    ev_docs = {e.meta["source_doc"] for e in ev}
    assert len(ev_docs) == 3  # round(10 * 0.3)


def test_split_by_document_default_frac_when_neither_given():
    eps = [_fake_ep(f"doc{d}") for d in range(10)]
    tr, ev = train_prose.split_by_document(eps, seed=0)
    ev_docs = {e.meta["source_doc"] for e in ev}
    assert len(ev_docs) == round(10 * train_prose._DEFAULT_HOLDOUT_FRAC)


def test_split_by_document_seeded_determinism():
    eps = [_fake_ep(f"doc{d}") for d in range(12) for _ in range(2)]
    tr1, ev1 = train_prose.split_by_document(eps, holdout_docs=4, seed=7)
    tr2, ev2 = train_prose.split_by_document(eps, holdout_docs=4, seed=7)
    assert [e.question for e in tr1] == [e.question for e in tr2]
    assert {e.meta["source_doc"] for e in ev1} == {e.meta["source_doc"] for e in ev2}


def test_split_by_document_different_seed_can_differ():
    eps = [_fake_ep(f"doc{d}") for d in range(20)]
    _, ev_a = train_prose.split_by_document(eps, holdout_docs=5, seed=0)
    _, ev_b = train_prose.split_by_document(eps, holdout_docs=5, seed=1)
    docs_a = {e.meta["source_doc"] for e in ev_a}
    docs_b = {e.meta["source_doc"] for e in ev_b}
    assert docs_a != docs_b, "different seeds landed on the identical held-out doc set (extremely unlikely, check RNG wiring)"


def test_split_by_document_zero_holdout():
    eps = [_fake_ep(f"doc{d}") for d in range(5)]
    tr, ev = train_prose.split_by_document(eps, holdout_docs=0, seed=0)
    assert len(ev) == 0
    assert len(tr) == len(eps)


# ---------------------------------------------------------------------------
# 2. load_episodes reuse (not duplication)
# ---------------------------------------------------------------------------
def test_load_episodes_is_the_same_function_as_eval_prose():
    assert train_prose.load_episodes is eval_prose.load_episodes


def test_load_episodes_round_trip_three_hand_made_episodes(tmp_path):
    eps = [
        _fake_ep("doc_a#0", "where is mary ?", "garden"),
        _fake_ep("doc_a#0", "who has the book ?", "john"),
        _fake_ep("doc_b#1", "who gave the book ?", "sandra"),
    ]
    path = str(tmp_path / "eps.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ep in eps:
            f.write(json.dumps(dataclasses.asdict(ep)) + "\n")

    loaded = train_prose.load_episodes(path)
    assert len(loaded) == 3
    for orig, got in zip(eps, loaded):
        assert isinstance(got, Episode)
        assert got.question == orig.question
        assert got.answer_text == orig.answer_text
        assert got.meta == orig.meta


# ---------------------------------------------------------------------------
# 3. build_mixed_training_set
# ---------------------------------------------------------------------------
def test_build_mixed_training_set_hits_target_fraction():
    prose_train = [_fake_ep(f"doc{d}") for d in range(40)]
    mixed, curr_tr, curr_va = train_prose.build_mixed_training_set(
        prose_train, curriculum_frac=0.5, seed=0, curriculum_val_episodes=8)

    assert len(mixed) == len(prose_train) + len(curr_tr)
    frac = len(curr_tr) / len(mixed)
    assert frac == pytest.approx(0.5, abs=0.05)
    assert len(curr_va) == 8
    # both kinds actually present in the mix (interleaved, not two blocks)
    mixed_ids = {id(e) for e in mixed}
    assert any(id(e) in mixed_ids for e in prose_train)
    assert any(id(e) in mixed_ids for e in curr_tr)


def test_build_mixed_training_set_curriculum_frac_zero_is_prose_only():
    prose_train = [_fake_ep(f"doc{d}") for d in range(10)]
    mixed, curr_tr, curr_va = train_prose.build_mixed_training_set(
        prose_train, curriculum_frac=0.0, seed=0, curriculum_val_episodes=5)
    assert curr_tr == []
    assert len(mixed) == len(prose_train)
    assert {id(e) for e in mixed} == {id(e) for e in prose_train}
    # curriculum_val is independent of curriculum_frac -- still generated
    assert len(curr_va) == 5


def test_build_mixed_training_set_deterministic():
    prose_train = [_fake_ep(f"doc{d}") for d in range(15)]
    mixed1, curr_tr1, curr_va1 = train_prose.build_mixed_training_set(
        prose_train, curriculum_frac=0.4, seed=3, curriculum_val_episodes=4)
    mixed2, curr_tr2, curr_va2 = train_prose.build_mixed_training_set(
        prose_train, curriculum_frac=0.4, seed=3, curriculum_val_episodes=4)
    assert [e.question for e in mixed1] == [e.question for e in mixed2]
    assert len(curr_tr1) == len(curr_tr2)
    assert [e.question for e in curr_va1] == [e.question for e in curr_va2]


def test_build_mixed_training_set_higher_frac_yields_more_curriculum():
    prose_train = [_fake_ep(f"doc{d}") for d in range(30)]
    _, curr_lo, _ = train_prose.build_mixed_training_set(
        prose_train, curriculum_frac=0.2, seed=0, curriculum_val_episodes=2)
    _, curr_hi, _ = train_prose.build_mixed_training_set(
        prose_train, curriculum_frac=0.7, seed=0, curriculum_val_episodes=2)
    assert len(curr_hi) > len(curr_lo)


# ---------------------------------------------------------------------------
# 4 + 5. parser-dependent: tiny end-to-end run, and --frozen-eval
# ---------------------------------------------------------------------------
def _quantum_parser_available() -> bool:
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    tok = SimpleTokenizer.build(["mary is in the garden ."], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    return getattr(parser, "_parser", None) is not None


def _make_prose_episodes_from_fixture_corpus(n_min: int = 5):
    """Mirrors tests/test_eval_prose.py's own end-to-end fixture: real
    converted episodes from data/corpus/synthetic_*.txt via
    nsm_ct.corpus's converter path, so this test exercises train_prose.py
    against genuinely parseable prose, not hand-rolled text that might not
    parse. Spreads episodes across >=2 distinct doc_ids (one passage per
    doc_id) so the document-holdout split has something real to hold out."""
    import glob

    from nsm_ct.corpus import ParsedClause, iter_sentences, make_episodes, parse_passage
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
    files = sorted(glob.glob(os.path.join(data_dir, "synthetic_*.txt")))
    assert files, "expected data/corpus/synthetic_*.txt to exist"

    passages = []
    all_sentences = []
    for fpath in files:
        text = "\n".join(ln for ln in open(fpath, encoding="utf-8") if not ln.strip().startswith("#"))
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            sents = iter_sentences(block)
            if sents:
                passages.append(sents)
                all_sentences.extend(sents)

    tok = SimpleTokenizer.build(all_sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    if getattr(parser, "_parser", None) is None:
        return None

    all_results = [parse_passage(s, parser) for s in passages]
    all_clauses = [c for res in all_results for c in res if isinstance(c, ParsedClause)]

    episodes = []
    for i, res in enumerate(all_results):
        own = {id(r) for r in res}
        pool = [c for c in all_clauses if id(c) not in own]
        episodes.extend(make_episodes(res, holdout="last", seed=i, doc_id=f"synthetic_prose_test#{i}",
                                       distractor_pool=pool))
        if len(episodes) >= n_min:
            break
    assert len({e.meta["source_doc"] for e in episodes}) >= 2, \
        "need episodes from >=2 documents for the holdout split to mean anything"
    return episodes


def _write_jsonl(path, episodes):
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(dataclasses.asdict(ep)) + "\n")


def _base_args(episodes_path, **overrides):
    args = argparse.Namespace(
        episodes=episodes_path, holdout_docs=1, holdout_frac=None,
        curriculum_frac=0.5, curriculum_val_episodes=6,
        track="A", dim=16, hidden=16, epochs=2, log_interval=1, seed=0,
        cheat=False, no_gold_eval=False, frozen_eval=False,
        load=None, save=None, batch_size=0, threads=None,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_end_to_end_five_episodes_two_epochs_dim16(tmp_path, capsys):
    if not _quantum_parser_available():
        pytest.skip("quantum_parser unavailable in this environment")
    episodes = _make_prose_episodes_from_fixture_corpus(n_min=5)
    if episodes is None:
        pytest.skip("quantum_parser unavailable in this environment")

    ep_path = str(tmp_path / "prose5.jsonl")
    _write_jsonl(ep_path, episodes)

    args = _base_args(ep_path)
    result = train_prose.run(args)

    assert result, "run() returned an empty dict -- likely the parser-unavailable early-return"
    for key in ("prose_eval", "curriculum_retention", "peak_rss_mb", "elapsed_min",
                "losses", "n_resolver_params", "model", "model_config"):
        assert key in result, f"missing report key {key!r}"
    assert isinstance(result["peak_rss_mb"], float)
    assert len(result["losses"]) == args.epochs

    captured = capsys.readouterr().out
    assert "document split:" in captured
    assert "training mix:" in captured
    assert "HELD-OUT-DOCUMENT ZERO-SHOT" in captured
    assert "CURRICULUM RETENTION" in captured
    assert "peak_rss_mb:" in captured


def test_frozen_eval_runs_without_optimizer(tmp_path, monkeypatch, capsys):
    if not _quantum_parser_available():
        pytest.skip("quantum_parser unavailable in this environment")
    episodes = _make_prose_episodes_from_fixture_corpus(n_min=5)
    if episodes is None:
        pytest.skip("quantum_parser unavailable in this environment")

    ep_path = str(tmp_path / "prose5.jsonl")
    _write_jsonl(ep_path, episodes)

    # A FRESH (untrained) checkpoint -- build_model + save_checkpoint, no training step,
    # same pattern tests/test_eval_prose.py's own end-to-end tier uses.
    import torch

    from _train_common import build_model
    from nsm_ct.checkpoint import save_checkpoint

    config = {"dim": 16, "hidden": 16, "track": "A",
              "use_cand_feature": True, "cand_feature_extra": 1, "evidence_prior_beta": None,
              "codec_max_pos": 64}
    torch.manual_seed(0)
    model = build_model(config)
    ckpt_path = str(tmp_path / "fresh.pt")
    save_checkpoint(ckpt_path, model, config=config, extra={})

    def _no_optimizer(*a, **kw):
        raise AssertionError("--frozen-eval must not construct an optimizer -- training was supposed to be skipped")
    monkeypatch.setattr(torch.optim, "Adam", _no_optimizer)

    args = _base_args(ep_path, frozen_eval=True, load=ckpt_path)
    result = train_prose.run(args)

    assert result
    assert result["losses"] == []
    assert result["elapsed_min"] == 0.0

    captured = capsys.readouterr().out
    assert "--frozen-eval" in captured
    assert "HELD-OUT-DOCUMENT ZERO-SHOT" in captured


def test_check_args_frozen_eval_requires_load():
    args = argparse.Namespace(frozen_eval=True, load=None, save=None)
    with pytest.raises(ValueError, match="requires --load"):
        train_prose._check_args(args)


def test_check_args_save_with_frozen_eval_rejected():
    args = argparse.Namespace(frozen_eval=True, load="ckpt.pt", save="out.pt")
    with pytest.raises(ValueError, match="frozen-eval"):
        train_prose._check_args(args)


def test_check_args_ok_combinations_pass():
    train_prose._check_args(argparse.Namespace(frozen_eval=False, load=None, save=None))
    train_prose._check_args(argparse.Namespace(frozen_eval=True, load="ckpt.pt", save=None))
    train_prose._check_args(argparse.Namespace(frozen_eval=False, load="ckpt.pt", save="out.pt"))
