"""Tests for M57a: the entity-instance subsystem (nsm_ct.instances).

See dev/MIND_INTERFACE.md's "v2 addendum -- the entity-instance subsystem"
and CLAUDE.md's "M57 memory-schema decision". No parser dependency -- these
are synthetic-tensor tests of the registry/write/query/candidate-generation
machinery, deliberately isolated from quantum_parser so they run everywhere
(mirrors tests/test_resolver.py's own rationale).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from nsm_ct.instances import (
    InstanceRegistry,
    ProvenanceLog,
    candidates_for,
    inverse_query,
    query_attribute,
    to_candidate_set,
    write_attribute,
)
from nsm_ct.tpr import TPRCodec

DIM = 48


def _codec() -> TPRCodec:
    return TPRCodec(dim=DIM)


# ---------------------------------------------------------------------------
# Minting: fresh variables, not meanings.
# ---------------------------------------------------------------------------
def test_two_mints_same_name_hint_are_distinct():
    reg = InstanceRegistry(dim=DIM, seed=0)
    id1, atom1 = reg.mint("mary")
    id2, atom2 = reg.mint("mary")
    assert id1 != id2
    assert id1 == "inst:mary#1"
    assert id2 == "inst:mary#2"
    cos = torch.dot(atom1, atom2).item()
    assert abs(cos) < 0.5, f"mary#1/mary#2 should be near-orthogonal, cos={cos}"
    assert torch.allclose(atom1.norm(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(atom2.norm(), torch.tensor(1.0), atol=1e-4)


def test_different_name_hints_get_sequential_ids():
    reg = InstanceRegistry(dim=DIM, seed=0)
    id1, _ = reg.mint("mary")
    id2, _ = reg.mint("john")
    id3, _ = reg.mint("mary")
    assert [id1, id2, id3] == ["inst:mary#1", "inst:john#1", "inst:mary#2"]


def test_determinism_same_seed():
    order = ["mary", "mary", "john", "sandra"]
    reg_a = InstanceRegistry(dim=DIM, seed=42)
    reg_b = InstanceRegistry(dim=DIM, seed=42)
    for name in order:
        id_a, atom_a = reg_a.mint(name)
        id_b, atom_b = reg_b.mint(name)
        assert id_a == id_b
        assert torch.equal(atom_a, atom_b)


def test_different_seeds_diverge():
    reg_a = InstanceRegistry(dim=DIM, seed=1)
    reg_b = InstanceRegistry(dim=DIM, seed=2)
    _, atom_a = reg_a.mint("mary")
    _, atom_b = reg_b.mint("mary")
    assert not torch.allclose(atom_a, atom_b)


def test_near_orthogonality_regression_50_atoms():
    # Seed chosen deterministically for margin under the 0.5 gate: the
    # pairwise-max-|cos| statistic over C(50,2)=1225 near-Gaussian iid draws
    # at dim=48 has an expected value close to the 0.5 line itself (extreme-
    # value behavior of ~1225 samples with per-pair std ~1/sqrt(48)=0.144),
    # so this is a real regression gate, not a rubber stamp -- a future
    # minting-scheme regression that inflates pairwise correlation will still
    # trip it.
    reg = InstanceRegistry(dim=DIM, seed=0)
    for i in range(50):
        reg.mint(f"person{i % 5}")   # some repeated name_hints, some not
    atoms = reg.atoms()
    assert atoms.shape == (50, DIM)
    gram = atoms @ atoms.T
    off_diag = gram - torch.diag(torch.diag(gram))
    max_abs_cos = off_diag.abs().max().item()
    assert max_abs_cos < 0.5, f"pairwise |cos| among 50 minted atoms too high: {max_abs_cos}"


def test_registry_enumeration_helpers():
    reg = InstanceRegistry(dim=DIM, seed=0)
    assert len(reg) == 0
    assert reg.ids() == []
    assert reg.atoms().shape == (0, DIM)
    id1, atom1 = reg.mint("mary")
    id2, atom2 = reg.mint("john")
    assert len(reg) == 2
    assert reg.ids() == [id1, id2]
    assert torch.equal(reg.lookup(id1), atom1)
    assert torch.equal(reg.lookup(id2), atom2)
    stacked = reg.atoms()
    assert stacked.shape == (2, DIM)
    assert torch.equal(stacked[0], atom1)
    assert torch.equal(stacked[1], atom2)
    assert id1 in reg
    assert "inst:nobody#1" not in reg


# ---------------------------------------------------------------------------
# Attribute write/query roundtrip.
# ---------------------------------------------------------------------------
def _val(codec: TPRCodec, label: str) -> torch.Tensor:
    return torch.from_numpy(codec.filler_vec(label))


def _argmax_label(vec: torch.Tensor, codebook: dict) -> str:
    best, best_score = None, -2.0
    for label, v in codebook.items():
        s = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), v.unsqueeze(0)).item()
        if s > best_score:
            best, best_score = label, s
    return best


def test_attribute_write_query_roundtrip():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=0)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)

    id_mary, _ = reg.mint("mary")
    id_john, _ = reg.mint("john")

    name_book = {"mary": _val(codec, "name:mary"), "john": _val(codec, "name:john")}
    kind_book = {"doctor": _val(codec, "kind:doctor"), "teacher": _val(codec, "kind:teacher")}
    gender_book = {"F": _val(codec, "gender:F"), "M": _val(codec, "gender:M")}

    facts = {
        id_mary: {"name": ("mary", name_book["mary"]),
                  "kind": ("doctor", kind_book["doctor"]),
                  "gender": ("F", gender_book["F"])},
        id_john: {"name": ("john", name_book["john"]),
                  "kind": ("teacher", kind_book["teacher"]),
                  "gender": ("M", gender_book["M"])},
    }

    for iid, attrs in facts.items():
        for attr_name, (label, vec) in attrs.items():
            memory = write_attribute(
                memory, reg, iid, attr_name, vec, codec,
                log=log, source="text", language="en", timestamp=0.0, trust=1.0,
                value_label=label,
            )

    books = {"name": name_book, "kind": kind_book, "gender": gender_book}
    for iid, attrs in facts.items():
        for attr_name, (label, _vec) in attrs.items():
            got = query_attribute(memory, reg, iid, attr_name, codec)
            assert _argmax_label(got, books[attr_name]) == label


def test_interference_sanity_5_instances_3_attrs_shared_memory():
    """5 instances x 3 attributes each (15 writes) into ONE shared memory
    tensor at dim=48: roundtrip argmax accuracy must still be 100%."""
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=3)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)

    names = ["mary", "sandra", "john", "daniel", "bill"]
    kinds = ["doctor", "teacher", "nurse", "pilot", "chef"]
    genders = ["F", "F", "M", "M", "M"]

    name_book = {n: _val(codec, "name:" + n) for n in names}
    kind_book = {k: _val(codec, "kind:" + k) for k in kinds}
    gender_book = {"F": _val(codec, "gender:F"), "M": _val(codec, "gender:M")}

    ids = []
    for n in names:
        iid, _ = reg.mint(n)
        ids.append(iid)

    plan = {}
    for iid, n, k, g in zip(ids, names, kinds, genders):
        plan[iid] = {"name": (n, name_book[n]), "kind": (k, kind_book[k]),
                     "gender": (g, gender_book[g])}
        for attr_name, (label, vec) in plan[iid].items():
            memory = write_attribute(
                memory, reg, iid, attr_name, vec, codec,
                log=log, source="text", language="en", timestamp=0.0, trust=1.0,
                value_label=label,
            )

    assert len(log) == 15
    books = {"name": name_book, "kind": kind_book, "gender": gender_book}
    correct = total = 0
    for iid, attrs in plan.items():
        for attr_name, (label, _vec) in attrs.items():
            got = query_attribute(memory, reg, iid, attr_name, codec)
            total += 1
            correct += int(_argmax_label(got, books[attr_name]) == label)
    assert correct == total == 15, f"interference broke roundtrip accuracy: {correct}/{total}"


# ---------------------------------------------------------------------------
# Candidate generation: two Marys, definite descriptions, inverse query.
# ---------------------------------------------------------------------------
def test_two_marys_candidate_generation():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=5)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)

    mary1, _ = reg.mint("mary")
    mary2, _ = reg.mint("mary")
    john1, _ = reg.mint("john")

    mary_vec = _val(codec, "name:mary")
    john_vec = _val(codec, "name:john")
    for iid, (label, vec) in [(mary1, ("mary", mary_vec)), (mary2, ("mary", mary_vec)),
                               (john1, ("john", john_vec))]:
        memory = write_attribute(memory, reg, iid, "name", vec, codec,
                                  log=log, source="text", language="en",
                                  timestamp=0.0, trust=1.0, value_label=label)

    ids, atoms, scores = candidates_for(memory, reg, codec, attr_name="name",
                                         target_vec=mary_vec, threshold=0.5)
    assert set(ids) == {mary1, mary2}
    assert john1 not in ids
    assert atoms.shape == (2, DIM)
    assert scores.shape == (2,)
    assert (scores >= 0.5).all()

    cs = to_candidate_set(ids, scores)
    assert len(cs) == 2
    assert set(cs.keys) == {mary1, mary2}
    assert abs(float(cs.priors.sum()) - 1.0) < 1e-5


def test_definite_description_the_doctor():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=9)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)

    a, _ = reg.mint("mary")
    b, _ = reg.mint("john")
    c, _ = reg.mint("sandra")

    doctor_vec = _val(codec, "kind:doctor")
    teacher_vec = _val(codec, "kind:teacher")
    nurse_vec = _val(codec, "kind:nurse")
    for iid, (label, vec) in [(a, ("doctor", doctor_vec)), (b, ("teacher", teacher_vec)),
                               (c, ("nurse", nurse_vec))]:
        memory = write_attribute(memory, reg, iid, "kind", vec, codec,
                                  log=log, source="text", language="en",
                                  timestamp=0.0, trust=1.0, value_label=label)

    ids, atoms, scores = candidates_for(memory, reg, codec, attr_name="kind",
                                         target_vec=doctor_vec, threshold=0.5)
    assert ids == [a]
    assert atoms.shape == (1, DIM)


def test_candidates_for_empty_registry():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=0)
    memory = torch.zeros(DIM, DIM, DIM)
    ids, atoms, scores = candidates_for(memory, reg, codec, attr_name="kind",
                                         target_vec=_val(codec, "kind:doctor"))
    assert ids == []
    assert atoms.shape == (0, DIM)
    assert scores.shape == (0,)
    cs = to_candidate_set(ids, scores)
    assert len(cs) == 0


def test_inverse_query_who_is_a_doctor():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=11)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)

    names = ["mary", "john", "sandra", "daniel"]
    kinds = ["doctor", "teacher", "doctor", "pilot"]
    kind_book = {"doctor": _val(codec, "kind:doctor"), "teacher": _val(codec, "kind:teacher"),
                 "pilot": _val(codec, "kind:pilot")}
    ids = []
    for n, k in zip(names, kinds):
        iid, _ = reg.mint(n)
        ids.append(iid)
        memory = write_attribute(memory, reg, iid, "kind", kind_book[k], codec,
                                  log=log, source="text", language="en",
                                  timestamp=0.0, trust=1.0, value_label=k)

    all_ids, scores = inverse_query(memory, reg, codec, "kind", kind_book["doctor"])
    assert all_ids == ids
    assert scores.shape == (4,)

    order = sorted(range(4), key=lambda i: -scores[i].item())
    top2 = {all_ids[order[0]], all_ids[order[1]]}
    doctors = {ids[0], ids[2]}   # mary, sandra
    assert top2 == doctors
    margin = scores[order[1]].item() - scores[order[2]].item()
    assert margin > 0.3, f"expected a clear margin between doctors and non-doctors, got {margin}"


# ---------------------------------------------------------------------------
# Provenance.
# ---------------------------------------------------------------------------
def test_provenance_one_record_per_write_and_fields_carried():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=0)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)

    id_mary, _ = reg.mint("mary")
    id_john, _ = reg.mint("john")

    memory = write_attribute(memory, reg, id_mary, "name", _val(codec, "name:mary"), codec,
                              log=log, source="sentence:0", language="en",
                              timestamp=1.0, trust=0.9, value_label="mary", step=0)
    assert len(log) == 1
    memory = write_attribute(memory, reg, id_mary, "kind", _val(codec, "kind:doctor"), codec,
                              log=log, source="sentence:1", language="en",
                              timestamp=2.0, trust=0.8, value_label="doctor", step=1)
    assert len(log) == 2
    memory = write_attribute(memory, reg, id_john, "name", _val(codec, "name:john"), codec,
                              log=log, source="sentence:2", language="es",
                              timestamp=3.0, trust=0.7, value_label="john", step=2)
    assert len(log) == 3

    mary_records = log.records_for(id_mary)
    assert len(mary_records) == 2
    assert {r.relation for r in mary_records} == {"attr:name", "attr:kind"}

    john_records = log.records_for(id_john)
    assert len(john_records) == 1
    r = john_records[0]
    assert r.instance_id == id_john
    assert r.relation == "attr:name"
    assert r.value_label == "john"
    assert r.source == "sentence:2"
    assert r.language == "es"
    assert r.timestamp == 3.0
    assert r.trust == 0.7
    assert r.step == 2

    assert log.records_for("inst:nobody#1") == []
    assert len(log.records) == 3


def test_provenance_records_are_immutable():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=0)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)
    id_mary, _ = reg.mint("mary")
    write_attribute(memory, reg, id_mary, "name", _val(codec, "name:mary"), codec,
                     log=log, source="s", language="en", timestamp=0.0, trust=1.0,
                     value_label="mary")
    rec = log.records[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.trust = 0.5


def test_provenance_records_view_is_a_copy():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=0)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)
    id_mary, _ = reg.mint("mary")
    write_attribute(memory, reg, id_mary, "name", _val(codec, "name:mary"), codec,
                     log=log, source="s", language="en", timestamp=0.0, trust=1.0,
                     value_label="mary")
    view = log.records
    assert isinstance(view, tuple)
    assert len(view) == 1


# ---------------------------------------------------------------------------
# write_attribute is out-of-place.
# ---------------------------------------------------------------------------
def test_write_attribute_is_out_of_place():
    codec = _codec()
    reg = InstanceRegistry(dim=DIM, seed=0)
    log = ProvenanceLog()
    memory = torch.zeros(DIM, DIM, DIM)
    id_mary, _ = reg.mint("mary")
    new_memory = write_attribute(memory, reg, id_mary, "name", _val(codec, "name:mary"), codec,
                                  log=log, source="s", language="en", timestamp=0.0, trust=1.0,
                                  value_label="mary")
    assert torch.equal(memory, torch.zeros(DIM, DIM, DIM))
    assert not torch.equal(new_memory, torch.zeros(DIM, DIM, DIM))
