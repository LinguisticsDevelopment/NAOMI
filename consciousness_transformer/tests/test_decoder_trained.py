"""Tests for the TRAINED reconstruction decoder (RESEARCH_NOTES "DECODER PLAN
UPDATE", 2026-09-05): a learned realizer trained self-supervised by
reconstruction (autoencoder: text -> encoder structure -> decoder -> text),
and gated the same way Phase-1's rule decoder is (`test_decoder.py`) -- sever
the structure's content and the output MUST collapse to empty, never invent.

Tiny, CPU-only, no training: this file only exercises model construction,
one forward/loss step, greedy `realize`, the round-trip + accuracy helpers,
and the no-confab ablation.
"""

from __future__ import annotations

import re

import pytest
import torch

from nsm_ct import decoder_trained as dt
from nsm_ct import encoder_model as em


# --------------------------------------------------------------------- fixtures ---

def _mary_saw_sue():
    """encoder_gold_v2 / ENCODER_IO_CONTRACT_V2 §8.2's own worked example."""
    return {
        "text": "mary saw sue .",
        "tokens": ["mary", "saw", "sue", "."],
        "pos": ["PROPN", "VERB", "PROPN", "PUNCT"],
        "lattice": {
            "trees": [
                {"clauses": [
                    {
                        "predicate": "saw",
                        "predicate_grounding": {
                            "type": "sense", "candidates": ["see.v.01", "saw.v.01"],
                            "retrieval": {"source": "lexicon", "method": "lemma_senses", "ref": None},
                        },
                        "is_question": False, "utterance_kind": "proposition",
                        "roles": [
                            {"relation": "SUBJECT", "word": "mary", "token_index": 0, "is_entity": True,
                             "grounding": {"type": "entity", "candidates": None}},
                            {"relation": "OBJECT", "word": "sue", "token_index": 2, "is_entity": True,
                             "grounding": {"type": "entity", "candidates": None}},
                        ],
                    },
                ]},
            ],
            "discourse_links_per_tree": [[]],
        },
        "token_sense_candidates": [],
    }


def _dog_wants_food():
    """ENCODER_IO_CONTRACT_V2 §8.3's context sentence: "the dog wants food ."."""
    return {
        "text": "the dog wants food .",
        "tokens": ["the", "dog", "wants", "food", "."],
        "pos": ["DET", "NOUN", "VERB", "NOUN", "PUNCT"],
        "lattice": {
            "trees": [
                {"clauses": [
                    {
                        "predicate": "wants",
                        "predicate_grounding": {
                            "type": "sense", "candidates": ["want.v.01", "want.v.03", "want.v.04"],
                            "retrieval": {"source": "lexicon", "method": "lemma_senses", "ref": None},
                        },
                        "is_question": False, "utterance_kind": "proposition",
                        "roles": [
                            {"relation": "SUBJECT", "word": "dog", "token_index": 1, "is_entity": False,
                             "grounding": {"type": "sense", "candidates": ["dog.n.01", "frump.n.01", "cad.n.01"],
                                           "retrieval": {"source": "lexicon", "method": "lemma_senses",
                                                         "ref": None}}},
                            {"relation": "OBJECT", "word": "food", "token_index": 3, "is_entity": False,
                             "grounding": {"type": "sense", "candidates": ["food.n.01", "food.n.02", "food.n.03"],
                                           "retrieval": {"source": "lexicon", "method": "lemma_senses",
                                                         "ref": None}}},
                        ],
                    },
                ]},
            ],
            "discourse_links_per_tree": [[]],
        },
        "token_sense_candidates": [],
    }


HAND_RECORDS = [_mary_saw_sue(), _dog_wants_food()]


@pytest.fixture(scope="module")
def relation_vocab():
    return em.build_role_vocab(HAND_RECORDS)


@pytest.fixture(scope="module")
def function_vocab():
    return dt.build_function_vocab(HAND_RECORDS)


@pytest.fixture(scope="module")
def model(relation_vocab, function_vocab):
    torch.manual_seed(0)
    return dt.DecoderTrainedModel(relation_vocab, function_vocab)


# --------------------------------------------------------------------- (a) model ---

def test_relation_and_function_vocab_are_closed_and_small(relation_vocab, function_vocab):
    assert "PREDICATE" in relation_vocab and "SUBJECT" in relation_vocab and "OBJECT" in relation_vocab
    # gaps: "." (mary saw sue) and "the", "." (the dog wants food) -> {'.', 'the'} + <unk>
    assert function_vocab[0] == dt.UNK_FUNC
    assert set(function_vocab) == {dt.UNK_FUNC, ".", "the"}


def test_model_builds_and_is_sub_mb(model):
    n_params = model.num_params()
    n_bytes = n_params * 4
    assert n_params > 0
    assert n_bytes < 1_000_000, f"decoder_trained must be sub-MB, got {n_bytes} bytes ({n_params} params)"


# --------------------------------------------------------------------- (b) features + loss ---

def test_build_decoder_features_aligns_copy_targets_to_structure_nodes(relation_vocab, function_vocab):
    record = _mary_saw_sue()
    tree = record["lattice"]["trees"][0]
    feats = dt.build_decoder_features(record, tree, function_vocab, relation_vocab)

    assert feats.node_words == ["mary", "saw", "sue"]          # SUBJECT, PREDICATE, OBJECT in token order
    assert feats.target_tokens == ["mary", "saw", "sue", "."]
    # "mary"/"saw"/"sue" copy from nodes 0/1/2; "." is a function-vocab generation; final step is EOS.
    M = len(feats.node_words)
    func_index = {w: i for i, w in enumerate(function_vocab)}
    expected = [0, 1, 2, M + func_index["."], M + len(function_vocab)]
    assert feats.target_labels.tolist() == expected


def test_forward_and_loss_run_on_two_hand_records(model, relation_vocab, function_vocab):
    for record in HAND_RECORDS:
        tree = record["lattice"]["trees"][0]
        feats = dt.build_decoder_features(record, tree, function_vocab, relation_vocab)
        loss = dt.reconstruction_loss(model, feats)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad
        loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# --------------------------------------------------------------------- (d) realize ---

def test_realize_returns_only_tokens_derivable_from_structure(model, function_vocab):
    record = _mary_saw_sue()
    tree = record["lattice"]["trees"][0]
    structure = dt.build_structure(record, tree)

    out = dt.realize(model, structure)
    assert isinstance(out, list)
    assert out                                                   # untrained but non-empty (has content nodes)

    allowed = {n.word.lower() for n in structure.nodes} | set(function_vocab)
    leaked = {w.lower() for w in out} - allowed
    assert not leaked, f"realize emitted tokens outside copy-source/closed-vocab: {leaked!r}"


# --------------------------------------------------------------------- (e) round_trip + accuracy ---

def test_round_trip_and_reconstruction_accuracy_compute(model):
    record = _mary_saw_sue()
    tree = record["lattice"]["trees"][0]

    pred_text = dt.round_trip(record, tree, model)
    assert isinstance(pred_text, str)

    metrics = dt.reconstruction_accuracy(pred_text, record["text"])
    assert set(metrics) == {"exact_match", "token_f1"}
    assert metrics["exact_match"] in (0.0, 1.0)
    assert 0.0 <= metrics["token_f1"] <= 1.0

    # identical sequences must score perfectly (sanity on the metric itself).
    perfect = dt.reconstruction_accuracy("mary saw sue .", "mary saw sue .")
    assert perfect == {"exact_match": 1.0, "token_f1": 1.0}

    disjoint = dt.reconstruction_accuracy("x y z", "mary saw sue .")
    assert disjoint == {"exact_match": 0.0, "token_f1": 0.0}


# --------------------------------------------------------------------- no-confab ablation ---

ALL_CONTENT_WORDS = {"mary", "saw", "sue", "dog", "wants", "food"}


@pytest.mark.parametrize("build_record", [_mary_saw_sue, _dog_wants_food])
def test_no_confab_ablation_collapses_to_empty(model, build_record):
    """Sever the structure's content (design's ablation, applied to the
    learned decoder, module docstring): the realized output MUST be empty
    and MUST NOT leak any content word from the intact structure."""
    record = build_record()
    tree = record["lattice"]["trees"][0]
    structure = dt.build_structure(record, tree)
    assert structure.nodes, "sanity: the intact structure must have content nodes"

    severed = dt.sever_structure_content(structure)
    assert all(n.word is None for n in severed.nodes)

    out = dt.realize(model, severed)
    assert out == []

    out_text = " ".join(out)
    leaked_words = set(re.findall(r"[a-z']+", out_text.lower())) & ALL_CONTENT_WORDS
    assert not leaked_words


def test_no_confab_ablation_gate_short_circuits_before_the_network(model, monkeypatch):
    """The empty-structure gate must fire without ever calling the decoder
    network -- not merely produce an empty result after running it."""
    record = _mary_saw_sue()
    tree = record["lattice"]["trees"][0]
    severed = dt.sever_structure_content(dt.build_structure(record, tree))

    def _boom(*args, **kwargs):
        raise AssertionError("decode_step must not run when the structure has no content")

    monkeypatch.setattr(model, "decode_step", _boom)
    assert dt.realize(model, severed) == []


def test_empty_structure_realizes_to_empty(model):
    empty = dt.CommittedStructure(nodes=[], tokens=["irrelevant"])
    assert dt.realize(model, empty) == []
