"""Regression tests for the B1/C1/B2 clause-extraction fixes.

See ``dev/GOLD_COVERAGE_AUDIT.md`` ("Part 3 -- Root cause" / "Proposed fix")
and ``dev/CLAUSE_EXTRACTION_FIX.md`` for the full writeup. Each sentence
below is quoted verbatim from ``runs/holdout_sentences.txt`` (the 50-tree
hand audit's own holdout file), so these are real prose failures, not
synthetic cases.

- B1 (``_secondary_fact_clauses``, clause.py): a bare intransitive
  secondary/coordinated clause with no PP and no OBJECT/INDIRECT_OBJECT had
  no ``val_idx`` fallback, so the clause was silently dropped.
- C1 (``_primary_discourse``'s coordinated-VALUE branch): VP-coordination
  ("did A and did B") misrouted into the coordinated-VALUE branch, giving
  both clauses the coordinator's own token ("and") as their predicate.
- B2 (``_recover_reporting_matrix_clauses``, new): a reporting/cognition
  verb with a POSTPOSED subject ("said little jack", "replied peggy") has
  no SUBJECT edge at all (the subject is misattached as an OBJECT), so no
  existing path ever built a clause for it.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def parser():
    tok = SimpleTokenizer.build([], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok)
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p


def _by_pred(clauses, pred):
    return [c for c in clauses if c.predicate == pred]


def _subj(c):
    return next((a.token for r, a in c.args if r == "SUBJECT"), None)


# -- B1: bare intransitive secondary/coordinated clause ------------------

def test_b1_bare_intransitive_coordinated_clause_recovered(parser):
    """Holdout #10: 'he just laughed , and shouted' / 'the ... boy ran on
    and on' -- the bare intransitive secondary clauses ('laughed', second
    'ran') used to be silently dropped (no val_idx fallback)."""
    clauses, _links = extract_discourse(parser._parse_graph(
        "the little old woman and the little old man ran after him as fast "
        "as they could , but he just laughed , and shouted , -- the little "
        "gingerbread boy ran on and on , until he came to a cow , by the "
        "roadside ."
    ))
    laughed = _by_pred(clauses, "laughed")
    assert laughed and _subj(laughed[0]) == "he"
    ran = _by_pred(clauses, "ran")
    assert any(_subj(c) == "boy" for c in ran)


def test_b1_bare_intransitive_matrix_clause_recovered(parser):
    """Holdout #48: ''they must go by the carrier ,' she thought' -- the
    bare 'she thought' clause (no PP, no object) used to be dropped even
    though it has its own SUBJECT edge."""
    clauses, _links = extract_discourse(parser._parse_graph(
        "'they must go by the carrier , ' she thought ; 'and how funny it "
        "'ll seem , sending presents to one 's own feet !"
    ))
    thought = _by_pred(clauses, "thought")
    assert thought and _subj(thought[0]) == "she"


# -- C1: VP-coordination misrouted as coordinated VALUE -------------------

def test_c1_vp_coordination_shares_subject_own_predicates(parser):
    """Holdout #6: 'the lion ... shook his mane and showed his teeth' --
    both clauses used to collapse to predicate='and'; each coordinated verb
    must now be its own clause's predicate, sharing the subject."""
    clauses, _links = extract_discourse(parser._parse_graph(
        "and the lion in the water shook his mane and showed his teeth ."
    ))
    assert not _by_pred(clauses, "and")
    shook = _by_pred(clauses, "shook")
    showed = _by_pred(clauses, "showed")
    assert shook and _subj(shook[0]) == "lion"
    assert showed and _subj(showed[0]) == "lion"
    assert any(a.token == "mane" for r, a in shook[0].args if r == "OBJECT")
    assert any(a.token == "teeth" for r, a in showed[0].args if r == "OBJECT")


def test_c1_vp_coordination_fox_looked_and_began(parser):
    """Holdout #41: 'the fox looked at him and began to run' -- both clauses
    used to collapse to predicate='and'."""
    clauses, _links = extract_discourse(parser._parse_graph(
        "the fox looked at him and began to run ."
    ))
    assert not _by_pred(clauses, "and")
    looked = _by_pred(clauses, "looked")
    began = _by_pred(clauses, "began")
    assert looked and _subj(looked[0]) == "fox"
    assert began and _subj(began[0]) == "fox"


# -- B2: reporting/cognition matrix clause, postposed subject --------------

def test_b2_reported_speech_postposed_subject_replied(parser):
    """Holdout #5: '"your poor goat is dead ," replied peggy .' -- 'replied'
    has no SUBJECT edge at all (peggy attaches as its OBJECT instead), so it
    used to vanish entirely."""
    clauses, _links = extract_discourse(parser._parse_graph(
        '" your poor goat is dead , " replied peggy .'
    ))
    replied = _by_pred(clauses, "replied")
    assert replied and _subj(replied[0]) == "peggy"


def test_b2_reported_speech_postposed_subject_said(parser):
    """Holdout #16: '"... ," said little jack .' -- 'said' has no SUBJECT
    edge (the postposed subject is split across an INDIRECT_OBJECT/OBJECT
    pair); the clause used to be dropped entirely."""
    clauses, _links = extract_discourse(parser._parse_graph(
        "\" then i 'll go to the woods , so that the animals may see me , \" "
        "said little jack ."
    ))
    said = _by_pred(clauses, "said")
    assert said and _subj(said[0]) == "jack"


def test_b2_reported_speech_postposed_subject_thought(parser):
    """Holdout #33: '" i wonder , " thought buster , " if ...' -- the
    cognition-verb matrix clause 'thought buster' used to be dropped."""
    clauses, _links = extract_discourse(parser._parse_graph(
        "\" i wonder , \" thought buster , \" if some one has stolen sammy "
        "'s breakfast , or if he has stolen the breakfast of some one else ."
    ))
    thought = _by_pred(clauses, "thought")
    assert thought and _subj(thought[0]) == "buster"
