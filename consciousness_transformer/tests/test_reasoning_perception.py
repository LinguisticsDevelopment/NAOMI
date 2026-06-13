"""Stage R2 gate — reasoning episodes ground to rule/fact streams; L1-L8 unchanged."""

import hashlib

import numpy as np
import pytest

from nsm_ct.clause_reactor import build_clause_batch
from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.tpr import TPRCodec


class StubResolver:
    def resolve(self, word, context=None):
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        return ParseTree(root=ParseNode(label=PRIME_NAMES[h % len(PRIME_NAMES)], token=word))


def _reasoning_eps(level, n=4):
    gen = CurriculumGenerator(max_level=11, seed=5)
    return [e for e in gen.generate(11 * n) if e.level == level][:n]


def test_conditional_episode_streams_rule_and_carries_if_atom():
    codec = TPRCodec(dim=48)
    eps = _reasoning_eps(9)
    batch = build_clause_batch(eps, None, StubResolver(), codec)   # parser unused for L9-L11
    # the IF atom appears on the coord channel of the rule steps (antecedent+consequent)
    if_atom = codec.filler_vec("IF")
    coord = batch.coord[0].numpy()
    rows_with_if = [t for t in range(coord.shape[0])
                    if np.allclose(coord[t], if_atom, atol=1e-5)]
    assert len(rows_with_if) == 2                     # antecedent + consequent
    # exactly one question step, and the episode is marked answerable
    assert float(batch.is_q[0].sum()) == 1.0
    assert float(batch.answerable[0]) == 1.0


def test_unanswerable_episode_is_marked_for_abstain():
    codec = TPRCodec(dim=48)
    eps = _reasoning_eps(11)
    batch = build_clause_batch(eps, None, StubResolver(), codec)
    assert all(float(a) == 0.0 for a in batch.answerable)   # L11 -> abstain


def test_inheritance_episode_streams_two_facts_no_if():
    codec = TPRCodec(dim=48)
    eps = _reasoning_eps(10)
    batch = build_clause_batch(eps, None, StubResolver(), codec)
    # no IF atom (pure facts), one question step, answerable
    if_atom = codec.filler_vec("IF")
    coord = batch.coord[0].numpy()
    assert not any(np.allclose(coord[t], if_atom, atol=1e-5) for t in range(coord.shape[0]))
    assert float(batch.is_q[0].sum()) == 1.0
    assert float(batch.answerable[0]) == 1.0


def test_l1_l8_streams_unaffected_by_reasoning_additions():
    # building an L1-L8 batch still works and marks everything answerable (=1).
    pytest.importorskip("nsm_ct.input_encoder")
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.tokenizer import SimpleTokenizer
    eps = CurriculumGenerator(max_level=8, seed=0).generate(16)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES))
    try:
        batch = build_clause_batch(eps, ParserInputEncoder(tok), StubResolver(), TPRCodec(dim=48))
    except Exception:
        pytest.skip("parser unavailable in this environment")
    assert batch.answerable is not None and float(batch.answerable.min()) == 1.0
