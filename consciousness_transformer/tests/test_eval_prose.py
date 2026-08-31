"""Tests for M58b's scripts/eval_prose.py -- THE ZERO-SHOT PROSE NUMBER.

Three tiers, matching the module's own shape:
  1. JSONL round-trip loader correctness (``load_episodes``) on 3
     hand-made episodes -- no parser/torch dependency, always runs.
  2. The script's aggregation math -- ``per_group_accuracy`` (per-relation/
     per-source counts sum to the total) and the two baselines
     (``random_guess_floor``, ``majority_baseline``) -- also parser-free.
  3. A tiny end-to-end: a FRESH (untrained) checkpoint evaluated on 5
     converted synthetic-prose episodes, run through ``scripts.eval_prose.
     run`` exactly as the CLI would. Skipped cleanly (not failed) when
     quantum_parser is unavailable, matching every other parser-dependent
     test in this suite (see tests/test_corpus.py's own skip contract).
     Accuracy near the random floor is EXPECTED at this scale/an untrained
     model -- this tier only asserts the report's shape (keys exist,
     counts match), not that the number is any good.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 1. JSONL round-trip loader
# ---------------------------------------------------------------------------
def _make_episode(question, answer, options, relation, source_doc) -> Episode:
    idx = options.index(answer)
    return Episode(
        context=["mary is in the garden .", "john gave mary the book ."],
        question=question,
        answer_text=answer,
        options=options,
        answer_idx=idx,
        level=0,
        meta={"kind": "prose", "source_doc": source_doc, "relation": relation,
              "held_out_entity": "mary", "held_out_value": answer},
    )


def _write_jsonl(path, episodes):
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(dataclasses.asdict(ep)) + "\n")


def test_load_episodes_round_trip(tmp_path):
    eps = [
        _make_episode("where is mary ?", "garden", ["garden", "kitchen", "office", "hallway"],
                       "PLACE", "synthetic_prose_x#0"),
        _make_episode("who has the book ?", "mary", ["mary", "john", "sandra", "bill"],
                       "RECIPIENT", "real_gutenberg_y#3"),
        _make_episode("who gave the book ?", "john", ["john", "fred", "daniel", "mary"],
                       "AGENT", "real_gutenberg_y#3"),
    ]
    path = str(tmp_path / "eps.jsonl")
    _write_jsonl(path, eps)

    loaded = eval_prose.load_episodes(path)

    assert len(loaded) == 3
    for orig, got in zip(eps, loaded):
        assert isinstance(got, Episode)
        assert got.context == orig.context
        assert got.question == orig.question
        assert got.answer_text == orig.answer_text
        assert got.options == orig.options
        assert got.answer_idx == orig.answer_idx
        assert got.meta == orig.meta


def test_load_episodes_skips_blank_lines(tmp_path):
    eps = [_make_episode("where is mary ?", "garden", ["garden", "kitchen", "office", "hallway"],
                          "PLACE", "synthetic_prose_x#0")]
    path = str(tmp_path / "eps.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n")
        f.write(json.dumps(dataclasses.asdict(eps[0])) + "\n")
        f.write("   \n")
    loaded = eval_prose.load_episodes(path)
    assert len(loaded) == 1
    assert loaded[0].question == "where is mary ?"


def test_load_episodes_ignores_unknown_keys(tmp_path):
    ep = _make_episode("where is mary ?", "garden", ["garden", "kitchen", "office", "hallway"],
                        "PLACE", "synthetic_prose_x#0")
    row = dataclasses.asdict(ep)
    row["some_future_field_not_on_episode"] = "whatever"
    path = str(tmp_path / "eps.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    loaded = eval_prose.load_episodes(path)
    assert len(loaded) == 1
    assert loaded[0].question == "where is mary ?"


# ---------------------------------------------------------------------------
# 2. aggregation math
# ---------------------------------------------------------------------------
def test_per_group_accuracy_counts_sum_to_total_and_hits_correct():
    eps = [
        _make_episode("q1", "a", ["a", "b", "c", "d"], "PLACE", "d1"),
        _make_episode("q2", "a", ["a", "b", "c", "d"], "PLACE", "d1"),
        _make_episode("q3", "a", ["a", "b", "c", "d"], "AGENT", "d1"),
        _make_episode("q4", "a", ["a", "b", "c", "d"], "RECIPIENT", "d1"),
        _make_episode("q5", "a", ["a", "b", "c", "d"], "RECIPIENT", "d1"),
    ]
    correct = [True, False, True, False, False]
    acc = eval_prose.per_group_accuracy(eps, correct, lambda e: e.meta["relation"])

    assert set(acc) == {"PLACE", "AGENT", "RECIPIENT"}
    assert acc["PLACE"] == (1, 2)
    assert acc["AGENT"] == (1, 1)
    assert acc["RECIPIENT"] == (0, 2)
    total_n = sum(n for _hits, n in acc.values())
    total_hits = sum(hits for hits, _n in acc.values())
    assert total_n == len(eps)
    assert total_hits == sum(correct)


def test_per_group_accuracy_empty_input():
    assert eval_prose.per_group_accuracy([], [], lambda e: "x") == {}


def test_random_guess_floor_mean_of_reciprocals():
    eps = [
        _make_episode("q1", "a", ["a", "b", "c", "d"], "PLACE", "d1"),        # 4 options -> 0.25
        _make_episode("q2", "a", ["a", "b", "c"], "PLACE", "d1"),             # 3 options -> 1/3
    ]
    floor = eval_prose.random_guess_floor(eps)
    assert floor == pytest.approx((0.25 + 1.0 / 3.0) / 2.0)


def test_random_guess_floor_empty_is_nan():
    import math
    assert math.isnan(eval_prose.random_guess_floor([]))


def test_majority_baseline_picks_modal_answer_text():
    eps = [
        _make_episode("q1", "garden", ["garden", "kitchen", "office", "hallway"], "PLACE", "d1"),
        _make_episode("q2", "garden", ["garden", "kitchen", "office", "hallway"], "PLACE", "d1"),
        _make_episode("q3", "kitchen", ["garden", "kitchen", "office", "hallway"], "PLACE", "d1"),
    ]
    acc, text = eval_prose.majority_baseline(eps)
    assert text == "garden"
    assert acc == pytest.approx(2.0 / 3.0)


def test_group_of_synthetic_vs_real():
    synth = _make_episode("q", "a", ["a", "b", "c", "d"], "PLACE", "synthetic_prose_01#0")
    real = _make_episode("q", "a", ["a", "b", "c", "d"], "PLACE", "real_gutenberg_busterbear#2")
    assert eval_prose._group_of(synth) == "synthetic"
    assert eval_prose._group_of(real) == "real"


# ---------------------------------------------------------------------------
# 3. tiny end-to-end: fresh (untrained) checkpoint on 5 synthetic-prose episodes
# ---------------------------------------------------------------------------
def _quantum_parser_available() -> bool:
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    tok = SimpleTokenizer.build(["mary is in the garden ."], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    return getattr(parser, "_parser", None) is not None


def test_end_to_end_fresh_checkpoint_on_five_prose_episodes(tmp_path, capsys):
    if not _quantum_parser_available():
        pytest.skip("quantum_parser unavailable in this environment")

    import glob

    from nsm_ct.corpus import make_episodes, parse_passage
    from nsm_ct.corpus import iter_sentences
    from nsm_ct.checkpoint import save_checkpoint
    from _train_common import build_model
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
        pytest.skip("quantum_parser unavailable in this environment")

    all_results = [parse_passage(s, parser) for s in passages]
    from nsm_ct.corpus import ParsedClause
    all_clauses = [c for res in all_results for c in res if isinstance(c, ParsedClause)]

    episodes = []
    for i, res in enumerate(all_results):
        own = {id(r) for r in res}
        pool = [c for c in all_clauses if id(c) not in own]
        episodes.extend(make_episodes(res, holdout="last", seed=i, doc_id=f"synthetic_prose_test#{i}",
                                       distractor_pool=pool))
        if len(episodes) >= 5:
            break
    episodes = episodes[:5]
    assert len(episodes) >= 3, "expected at least a few converted synthetic-prose episodes from the fixture corpus"

    ep_path = str(tmp_path / "prose5.jsonl")
    eval_prose_dump = [dataclasses.asdict(ep) for ep in episodes]
    with open(ep_path, "w", encoding="utf-8") as f:
        for row in eval_prose_dump:
            f.write(json.dumps(row) + "\n")

    # A FRESH (untrained) checkpoint: build_model + save_checkpoint, no training step.
    dim = 24
    config = {"dim": dim, "hidden": 32, "track": "A",
              "use_cand_feature": True, "cand_feature_extra": 1,
              "evidence_prior_beta": None, "codec_max_pos": 64}
    import torch
    torch.manual_seed(0)
    model = build_model(config)
    ckpt_path = str(tmp_path / "fresh.pt")
    save_checkpoint(ckpt_path, model, config=config, extra={})

    import argparse
    args = argparse.Namespace(ckpt=ckpt_path, episodes=ep_path, batch_size=0, verbose=True)
    eval_prose.run(args)

    captured = capsys.readouterr().out
    assert "THE ZERO-SHOT PROSE NUMBER" in captured
    assert "n episodes evaluated:" in captured
    assert "random-guess floor" in captured
    assert "majority baseline" in captured
    assert "per-relation:" in captured
    assert "per-source:" in captured
    # the per-episode dump (--verbose) prints each buildable episode's question
    assert "predicted:" in captured and "gold:" in captured
