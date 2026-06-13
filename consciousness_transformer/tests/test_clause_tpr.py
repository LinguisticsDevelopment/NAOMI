"""Tests for token-free clause TPRs + cross-clause entity memory (prototype)."""

import hashlib

import numpy as np

from nsm_ct.clause import (
    EntityMemory,
    EntityTracker,
    clause_tpr,
    decode_clause,
    extract_clauses,
    is_entity,
)
from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.tpr import TPRCodec


class StubResolver:
    """Deterministic word → small prime tree (no WordNet; keeps tests fast)."""

    def resolve(self, word, context=None):
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        root = ParseNode(label=PRIME_NAMES[h % len(PRIME_NAMES)])
        root.children = [ParseNode(label=PRIME_NAMES[(h // 7) % len(PRIME_NAMES)])]
        return ParseTree(root=root)


def _mary_kitchen_tree():
    """The parse shape of 'mary is in the kitchen' (verified against the parser)."""
    root = ParseNode(label="CLAUSE", token="is")
    pp = ParseNode(label="PP_NOUN", token="in", relation="MODIFICATION")
    pp.children.append(ParseNode(label="NOUN", token="kitchen", relation="PREPOSITION"))
    root.children.append(pp)
    root.children.append(ParseNode(label="NOMINAL", token="mary", relation="SUBJECT"))
    return ParseTree(root=root)


def test_extract_clauses_finds_subject_and_place():
    clause = extract_clauses(_mary_kitchen_tree())[0]
    assert clause.predicate == "is"
    rels = dict((r, a.token) for r, a in clause.args)
    assert rels.get("SUBJECT") == "mary"
    assert rels.get("PLACE") == "kitchen"          # preposition "in" -> PLACE


def test_is_entity():
    assert is_entity("mary") and is_entity("she")  # name + pronoun = variables
    assert not is_entity("kitchen")                # content word = meaning


def test_clause_tpr_is_token_free_and_decodes():
    codec = TPRCodec(dim=256)
    clause = extract_clauses(_mary_kitchen_tree())[0]
    m, triples = clause_tpr(clause, codec, StubResolver())
    assert m.shape == (256, 256)
    out = decode_clause(m, clause, codec, StubResolver())
    # subject entity recovered as the mary-variable with high confidence
    name, score = out["SUBJECT"]
    assert name == "mary" and score > 0.9
    # the discourse triple is keyed on the subject entity
    assert triples and triples[0][0] == "mary" and triples[0][1] == "PLACE"


def test_entity_tracker_resolves_pronoun_by_recency():
    t = EntityTracker()
    assert t.resolve("mary") == "mary"
    assert t.resolve("she") == "mary"              # nearest antecedent
    assert t.resolve("john") == "john"
    assert t.resolve("she") == "john"              # recency moved on


def test_bind3_unbind3_exact_on_orthonormal_keys():
    codec = TPRCodec(dim=128)
    a, b = codec._Q[:, 1], codec._Q[:, 2]          # orthonormal
    c = codec.filler_vec("VALUE")
    rec = codec.unbind3(codec.bind3(a, b, c), a, b)
    assert float(rec @ c / (np.linalg.norm(rec) + 1e-8)) > 0.99


def test_entity_memory_updates_across_writes():
    """A later write to the same (entity, relation) wins — cross-clause update."""
    codec = TPRCodec(dim=256)
    mem = EntityMemory(codec)
    v1 = codec.filler_vec("kitchen-val")
    v2 = codec.filler_vec("office-val")
    mem.write("mary", "PLACE", v1)
    q1 = mem.query("mary", "PLACE")
    assert (q1 @ v1) > (q1 @ v2)                    # first fact retrievable
    mem.write("mary", "PLACE", v2)                  # update (recency)
    q2 = mem.query("mary", "PLACE")
    assert (q2 @ v2) > (q2 @ v1)                    # latest fact dominates
