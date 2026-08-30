"""Tests for M57c.2: nsm_ct.entity_memory.query_entity, the entity-axis
inverse read ("who is tall?") -- unbinding the ENTITY dimension from a
KNOWN (relation, value) pair, instead of :func:`nsm_ct.entity_memory.query`'s
ordinary read (a KNOWN (entity, relation) address, an unknown value).

See RESEARCH_NOTES "M57c battery #1" (inverse_query measured BELOW chance
at full scale because no entity-axis read existed at all) and CLAUDE.md's
M57 memory-schema decision. No parser dependency -- synthetic-tensor tests,
mirroring tests/test_instances.py's own isolation discipline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.instances import InstanceRegistry, ProvenanceLog, write_attribute
from nsm_ct.tpr import TPRCodec

DIM = 48


def _codec() -> TPRCodec:
    return TPRCodec(dim=DIM)


# ---------------------------------------------------------------------------
# 1. query_entity matches the einsum it documents, and roundtrips a single
#    binding written via the ordinary entity_memory.write path.
# ---------------------------------------------------------------------------
def test_query_entity_matches_manual_einsum():
    b, d = 4, 16
    memory = torch.randn(b, d, d, d)
    relation = F.normalize(torch.randn(b, d), dim=-1)
    value = F.normalize(torch.randn(b, d), dim=-1)
    got = em.query_entity(memory, relation, value)
    expected = torch.einsum("bijk,bj,bk->bi", memory, relation, value)
    assert torch.equal(got, expected)
    assert got.shape == (b, d)


def test_query_entity_roundtrip_single_binding():
    d = 24
    memory = em.init_memory(1, d, "cpu")
    entity = F.normalize(torch.randn(1, d), dim=-1)
    relation = F.normalize(torch.randn(1, d), dim=-1)
    value = F.normalize(torch.randn(1, d), dim=-1)
    memory = em.write(memory, entity, relation, value, torch.ones(1))
    recovered = em.query_entity(memory, relation, value)
    assert F.cosine_similarity(recovered, entity).item() > 0.9


# ---------------------------------------------------------------------------
# 2. The milestone's own scenario: 3 instances x attr facts superposed in
#    ONE dim-48 memory; query_entity's argmax-cosine over the REGISTRY'S
#    OWN atoms picks the instance holding the queried trait, for each of
#    3 distinct traits (not just one lucky draw).
# ---------------------------------------------------------------------------
def test_query_entity_picks_correct_instance_for_three_distinct_traits():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=0)
    log = ProvenanceLog()
    id_a, atom_a = reg.mint("mary")
    id_b, atom_b = reg.mint("mary")
    id_c, atom_c = reg.mint("fred")
    ids = {"a": id_a, "b": id_b, "c": id_c}
    atoms = torch.stack([atom_a, atom_b, atom_c], dim=0)   # [3, d]

    traits = {"a": "tall", "b": "quiet", "c": "kind"}
    memory = torch.zeros(DIM, DIM, DIM)
    for role, trait in traits.items():
        value_vec = torch.from_numpy(codec.filler_vec("val:" + trait))
        memory = write_attribute(
            memory, reg, ids[role], "trait", value_vec, codec, gate=1.0,
            log=log, source="test", language="en", timestamp=0.0, trust=1.0,
            value_label=trait, step=None,
        )

    relation = torch.from_numpy(codec.filler_vec("attr:trait")).unsqueeze(0)   # [1, d]
    for role, trait in traits.items():
        value_vec = torch.from_numpy(codec.filler_vec("val:" + trait)).unsqueeze(0)
        readout = em.query_entity(memory.unsqueeze(0), relation, value_vec)   # [1, d]
        scores = F.cosine_similarity(readout.expand(3, -1), atoms, dim=-1)   # [3]
        winner_role = ["a", "b", "c"][int(scores.argmax())]
        assert winner_role == role, (role, trait, scores.tolist())
