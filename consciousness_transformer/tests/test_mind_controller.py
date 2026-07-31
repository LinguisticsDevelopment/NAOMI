"""M3 gates (fast): teacher correctness, the read-encoder, the controller forward,
and op-trace (relation-to-follow) imitation learnability.

The heavy research numbers (held-out multi-hop, abstain calibration) live in
``scripts/train_mind_controller.py`` / ``scripts/probe_mind_controller.py``; these
unit gates keep dims/episodes tiny so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import torch

from nsm_ct.clause_reactor import build_clause_batch
from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.mind import teacher
from nsm_ct.mind.controller import MindController, relation_match
from nsm_ct.mind.controller_losses import supervision_loss
from nsm_ct.mind.read_encoder import ReadEncoder
from nsm_ct import entity_memory as em
from nsm_ct.tpr import TPRCodec


class _StubResolver:
    def resolve(self, word: str) -> ParseTree:
        return ParseTree(root=ParseNode(label=word, token=word))


def _reasoning_episodes(n=120, seed=0):
    eps = CurriculumGenerator(max_level=13, seed=seed).generate(n)
    return [e for e in eps if e.level in (9, 10, 11, 12, 13)]


# ----------------------------------------------------------------- teacher ---
def test_teacher_gold_traces_replay_to_oracle_answer():
    """Every gold op-trace, replayed through the M2 executor, yields the oracle answer."""
    for ep in _reasoning_episodes(150):
        out = teacher.replay(ep)
        assert out["answer"] == ep.answer_text, (ep.level, ep.answer_text, out["answer"])


def test_gold_relation_path_inheritance():
    """L10 relation-to-follow is IS_A then CAN; abstain episodes yield no path."""
    eps = _reasoning_episodes(200)
    l10 = next(e for e in eps if e.level == 10)
    rels, ok = teacher.gold_relation_path(l10)
    assert ok and rels == ["IS_A", "CAN"]
    l11 = next(e for e in eps if e.level == 11)
    rels, ok = teacher.gold_relation_path(l11)
    assert not ok and rels == []


def test_gold_relation_path_deep_chain_len():
    """L12 follows k IS_A edges then CAN (path length = chain depth + 1)."""
    for ep in _reasoning_episodes(300):
        if ep.level == 12 and ep.answerable:
            rels, ok = teacher.gold_relation_path(ep)
            assert ok and rels[-1] == "CAN" and all(r == "IS_A" for r in rels[:-1])
            assert len(rels) == ep.meta["chain_len"] + 1
            break


# ------------------------------------------------------------- read-encoder ---
def test_read_encoder_recovers_binding():
    codec = TPRCodec(dim=64)
    enc = ReadEncoder(64)
    e = torch.from_numpy(codec.filler_vec("var:mary")).unsqueeze(0)
    r = torch.from_numpy(codec.filler_vec("rel:PLACE")).unsqueeze(0)
    v = torch.from_numpy(codec.filler_vec("kitchen")).unsqueeze(0)
    memory = em.init_memory(1, 64, torch.device("cpu"))
    memory = em.write(memory, e, r, v, gate=torch.ones(1))
    got = enc.read(memory, e, r)
    cos = torch.cosine_similarity(got, v).item()
    assert cos > 0.99, cos


# -------------------------------------------------------------- controller ---
def test_controller_forward_exposes_hop_rels():
    codec = TPRCodec(dim=32)
    eps = _reasoning_episodes(40)[:8]
    batch = build_clause_batch(eps, None, _StubResolver(), codec)
    model = MindController(codec, hidden=32, hops=5, halting=False)
    out = model(batch)
    assert out["hop_rels"].shape == (len(eps), 5, 32)
    assert out["hop_rels"].requires_grad


def test_relation_to_follow_imitation_overfits():
    """The controller learns to emit the teacher's gold relation-to-follow sequence."""
    torch.manual_seed(0)
    codec = TPRCodec(dim=48)
    eps = [e for e in _reasoning_episodes(120) if e.level in (10, 12)][:16]
    batch = build_clause_batch(eps, None, _StubResolver(), codec)
    sup_np = teacher.build_supervision(eps, hops=5)
    sup = {"rel_targets": torch.from_numpy(sup_np["rel_targets"]),
           "depth": torch.from_numpy(sup_np["depth"]),
           "answerable": torch.from_numpy(sup_np["answerable"])}
    model = MindController(codec, hidden=64, hops=5, halting=False)
    codebook = model.relation_codebook
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    start = relation_match(model(batch), sup["rel_targets"], codebook)
    for _ in range(200):
        out = model(batch)
        loss = supervision_loss(out, sup["rel_targets"], sup["depth"],
                                sup["answerable"], codebook, temperature=0.5)["supervision"]
        opt.zero_grad(); loss.backward(); opt.step()
    final = relation_match(model(batch), sup["rel_targets"], codebook)
    assert final > 0.9, (start, final)
    assert final > start


if __name__ == "__main__":  # pragma: no cover
    import pytest
    pytest.main([__file__, "-v"])
