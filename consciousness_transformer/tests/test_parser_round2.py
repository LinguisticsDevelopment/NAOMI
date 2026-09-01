"""Parser round 2 (dev/AURORA_SPRINT.md / RESEARCH_NOTES M58c's own "next
round" pointer): routes prose pronouns through the mind's membrane resolver
instead of the parser's own deterministic guess, strips attribution frames
around a quoted core clause, and adds a plain-OBJECT question template so
more "no-relation-extracted" clauses become episodes.

The three LOCKED-design changes, one section each:

1. PROSE PRONOUNS TO THE RESOLVER (nsm_ct.corpus's registry broadening +
   gender-compatible candidate routing; nsm_ct.clause_reactor._prose_steps'
   batch-grounding half).
2. ATTRIBUTION-WRAPPED NARRATION (nsm_ct.clause.strip_attribution +
   nsm_ct.corpus._attribution_fallback).
3. PLAIN-OBJECT QUESTIONS (nsm_ct.corpus._extract_triples's generic
   catch-all + "OBJECT" question template).

Regression anchors this file does NOT duplicate: tests/test_parser_round.py
(M58c's own byte-identical curriculum gate) and tests/test_corpus.py (the
M58a/M58d converter contract) are run alongside this file per the task's
own VERIFY step -- this file only covers what round 2 actually changed.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
_QP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "quantum_parser")
sys.path.insert(0, _QP_ROOT)

from nsm_ct import corpus  # noqa: E402
from nsm_ct.clause import strip_attribution  # noqa: E402
from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _parser_for(*extra_sentences):
    sentences = list(extra_sentences)
    tok = SimpleTokenizer.build(sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok, lang="en")
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p


# ---------------------------------------------------------------------------
# item 1: PROSE PRONOUNS TO THE RESOLVER
# ---------------------------------------------------------------------------

def test_registry_recognizes_real_text_names_not_just_curriculum_names():
    # "sammy" is not one of episode._NAMES (the old is_entity-only registry
    # never saw it -- see test_parser_round.py's own pronoun tests, both of
    # which had to use "mary"). The M58c-measured root cause: real-text
    # antecedents almost never register at all without this widening.
    assert not corpus.is_entity("sammy")
    assert corpus._is_registrable_entity("sammy")


def test_pronoun_resolves_against_a_real_text_name():
    sentences = corpus.iter_sentences("Sammy is in the garden. He went to the kitchen.")
    p = _parser_for(*sentences, "sammy", "garden", "kitchen")
    results = corpus.parse_passage(sentences, p)
    outcomes = corpus.taxonomy_counts(results)
    assert outcomes.get("pronoun-unresolvable", 0) == 0, outcomes
    assert outcomes.get("parsed-pronoun-resolved", 0) == 1, outcomes
    resolved = [r for r in results if isinstance(r, corpus.ParsedClause) and r.pronoun_candidates]
    assert len(resolved) == 1
    assert resolved[0].entity == "sammy"
    assert resolved[0].pronoun_candidates == ("sammy",)


def test_pronoun_candidates_are_gender_filtered_most_recent_first():
    # "john" (M) is the MOST RECENT mention, but "she" is F -- only "mary"
    # (also F) is gender-compatible, so it must win despite being older.
    sentences = corpus.iter_sentences(
        "Mary is in the garden. John is in the kitchen. She went to the office.")
    p = _parser_for(*sentences, "mary", "john", "garden", "kitchen", "office")
    results = corpus.parse_passage(sentences, p)
    resolved = [r for r in results if isinstance(r, corpus.ParsedClause) and r.pronoun_candidates]
    assert len(resolved) == 1, results
    assert resolved[0].entity == "mary"
    assert resolved[0].pronoun_candidates == ("mary",)


def test_gender_incompatible_pronoun_stays_unresolvable():
    # The only antecedent ("john", M) conflicts with "she" (F) -- perception
    # must not guess past a genuine gender contradiction (the LOCKED
    # design's "gender-compatible" qualifier); this stays the pre-existing
    # pronoun-unresolvable outcome, never a wrong resolution.
    sentences = corpus.iter_sentences("John is in the garden. She went to the kitchen.")
    p = _parser_for(*sentences, "john", "garden", "kitchen")
    results = corpus.parse_passage(sentences, p)
    outcomes = corpus.taxonomy_counts(results)
    assert outcomes.get("pronoun-unresolvable", 0) == 1, outcomes
    assert outcomes.get("parsed-pronoun-resolved", 0) == 0, outcomes


def test_zero_compatible_antecedent_keeps_pronoun_unresolvable():
    # No antecedent at all -- unchanged from the pre-round-2 contract
    # (tests/test_parser_round.py's own test_pronoun_with_no_antecedent_
    # still_unresolved pins the same thing for the OLD deterministic path).
    sentences = corpus.iter_sentences("She went to the kitchen.")
    p = _parser_for(*sentences, "kitchen")
    results = corpus.parse_passage(sentences, p)
    outcomes = corpus.taxonomy_counts(results)
    assert outcomes.get("pronoun-unresolvable", 0) == 1, outcomes


def test_gender_compatible_helper_unknown_never_excludes():
    # Unknown-gender pronoun ("it"/"they") or unknown-gender name (any
    # real-text name outside the 6 curriculum names) never rules a
    # candidate out -- perception "never guesses" a contradiction it can't
    # actually support.
    assert corpus._gender_compatible("it", "sammy")
    assert corpus._gender_compatible("they", "mary")
    assert corpus._gender_compatible("she", "sammy")
    # A genuine, KNOWN conflict is the only thing that excludes.
    assert not corpus._gender_compatible("she", "john")
    assert corpus._gender_compatible("she", "mary")


def test_old_curriculum_pronoun_test_still_holds():
    # tests/test_parser_round.py's own regression anchor, re-asserted here
    # to pin that round 2 didn't change its observable contract (the
    # ENTITY value, not just the taxonomy code -- see that file's own
    # test_pronoun_with_earlier_antecedent_resolves).
    sentences = corpus.iter_sentences("Mary is in the garden. She went to the kitchen.")
    p = _parser_for(*sentences, "mary", "garden", "kitchen")
    results = corpus.parse_passage(sentences, p)
    values = {(r.entity, r.relation, r.value) for r in results if isinstance(r, corpus.ParsedClause)}
    assert ("mary", "PLACE", "kitchen") in values, values


def test_prose_pronoun_step_grounds_through_the_resolver_contract():
    """End-to-end: a converted prose episode whose context carries a
    resolved pronoun clause builds a ClauseBatch with the SAME addr_redirect
    / evidence_relation / recency contract curriculum pronoun devices use
    (_instance_steps/_rich_steps), and a real forward pass runs clean."""
    text = ("Mary is in the garden. She went to the kitchen. She moved to the office. "
            "John is in the library.")
    sentences = corpus.iter_sentences(text)
    p = _parser_for(*sentences, "mary", "john", "garden", "kitchen", "office", "library")
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=16)

    results = corpus.parse_passage(sentences, p)
    eps = corpus.make_episodes(results, holdout="last", seed=0, doc_id="t1")
    assert len(eps) == 1
    ep = eps[0]
    # the held-out clause names a real entity, never a bare pronoun (see
    # make_episodes' own "no gold leakage" note).
    assert ep.meta["held_out_entity"] not in ("she", "he", "it", "they")

    batch = build_clause_batch(eps, p, resolver, codec)
    assert batch.cand_entity is not None, "expected a pronoun candidate set in this batch"
    assert batch.cand_addr_mask is not None
    assert batch.cand_addr_mask.sum() >= 1, "expected at least one addr_redirect step"
    assert batch.cand_evidence_relation is not None
    assert batch.cand_recency is not None

    model = ClauseReactor(dim=16)
    with torch.no_grad():
        out = model(batch)
    assert out["answer_logits"].shape == (1, batch.options.shape[1])
    assert torch.isfinite(out["answer_logits"]).all()


def test_prose_pronoun_gold_index_always_none_no_gold_leakage():
    """The LOCKED design's own requirement: prose pronoun candidate sets
    NEVER carry a gold_index, unlike every curriculum pronoun device
    (_instance_steps/_rich_steps/_pronoun_context_step), which always know
    the true referent at generation time."""
    from nsm_ct import clause_reactor
    from nsm_ct.episode import Episode

    text = "Mary is in the garden. She went to the kitchen."
    sentences = corpus.iter_sentences(text)
    p = _parser_for(*sentences, "mary", "garden", "kitchen")
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=16)

    ep = Episode(context=sentences, question="where is the mary ?", answer_text="kitchen",
                 options=["garden", "kitchen"], answer_idx=1, level=0,
                 meta={"kind": "prose", "source_doc": "t", "relation": "PLACE"})
    steps, cand_sets = clause_reactor._prose_steps(ep, p, resolver, codec, {}, "usvs")
    assert cand_sets, "expected the pronoun sentence to yield a candidate set"
    for cs in cand_sets.values():
        assert cs.gold_index is None
        assert cs.addr_redirect is True
        assert cs.evidence_relation == "gender"


# ---------------------------------------------------------------------------
# item 2: ATTRIBUTION-WRAPPED NARRATION
# ---------------------------------------------------------------------------

def test_strip_attribution_narration_after_quote():
    toks = '" the dog is in the garden , " said mary .'.split()
    core, speaker = strip_attribution(toks)
    assert core == ["the", "dog", "is", "in", "the", "garden", ","]
    assert speaker == "mary"


def test_strip_attribution_narration_before_quote():
    toks = 'mary said , " the dog is in the garden . "'.split()
    core, speaker = strip_attribution(toks)
    assert core == ["the", "dog", "is", "in", "the", "garden", "."]
    assert speaker == "mary"


def test_strip_attribution_speaker_can_be_a_pronoun():
    toks = '" the dog is in the garden , " said he .'.split()
    core, speaker = strip_attribution(toks)
    assert speaker == "he"


def test_strip_attribution_returns_none_without_a_recognized_verb():
    # A quote with no attribution verb at all -- out of scope, callers fall
    # through to the pre-existing quoted-span fallback instead.
    toks = '" the dog is in the garden . "'.split()
    assert strip_attribution(toks) is None


def test_strip_attribution_returns_none_without_any_quote():
    assert strip_attribution("mary is in the garden .".split()) is None


def test_attribution_fallback_tags_source_with_the_speaker():
    sentences = corpus.iter_sentences('"The dog is in the garden," said Mary.')
    p = _parser_for(*sentences, "dog", "garden")
    fallback = corpus._attribution_fallback(0, sentences[0], sentences[0].split(), p,
                                             corpus._PassageRegistry())
    assert fallback is not None
    assert len(fallback) == 1
    assert fallback[0].entity == "dog"
    assert fallback[0].relation == "PLACE"
    assert fallback[0].value == "garden"
    assert fallback[0].source == "quoted:mary"


def test_attribution_fallback_narration_before_quote_also_tags_speaker():
    sentences = corpus.iter_sentences('Mary said, "The dog is in the garden."')
    p = _parser_for(*sentences, "dog", "garden")
    fallback = corpus._attribution_fallback(0, sentences[0], sentences[0].split(), p,
                                             corpus._PassageRegistry())
    assert fallback is not None
    assert fallback[0].source == "quoted:mary"


def test_attribution_fallback_none_when_core_itself_fails():
    # A quoted QUESTION asserts nothing -- there is no fact to extract
    # regardless of the attribution fix; the fallback must return None
    # (not crash), leaving the sentence to its normal failure classification.
    sentences = corpus.iter_sentences('Mary asked, "Where is the dog?"')
    p = _parser_for(*sentences, "dog")
    fallback = corpus._attribution_fallback(0, sentences[0], sentences[0].split(), p,
                                             corpus._PassageRegistry())
    assert fallback is None


def test_failed_attribution_core_falls_back_to_existing_codes():
    # No quote mark at all -> strip_attribution (and therefore
    # _attribution_fallback) returns None immediately, and the sentence
    # still gets a valid, pre-existing taxonomy code (never a crash, never
    # a new bespoke code) -- a real dev/PROSE_FAILURE_TAXONOMY.md example
    # ("and grinned .", coordination) that round 2 leaves exactly as-is.
    sentences = corpus.iter_sentences("And grinned.")
    p = _parser_for(*sentences)
    results = corpus.parse_passage(sentences, p)
    assert len(results) == 1
    assert isinstance(results[0], corpus.ParseFailure)
    assert results[0].reason in corpus.FAILURE_REASONS


# ---------------------------------------------------------------------------
# item 3: PLAIN-OBJECT QUESTIONS
# ---------------------------------------------------------------------------

def test_object_relation_template_registered():
    assert corpus._RELATION_QUESTION_TEMPLATE["OBJECT"] == "what does the {e} have ?"


def test_clean_transitive_clause_yields_both_directions():
    sentences = corpus.iter_sentences("The man ate the apple.")
    p = _parser_for(*sentences, "man", "apple")
    results = corpus.parse_passage(sentences, p)
    triples = {(r.entity, r.relation, r.value) for r in results if isinstance(r, corpus.ParsedClause)}
    assert ("apple", "AGENT", "man") in triples, triples
    assert ("man", "OBJECT", "apple") in triples, triples


def test_generic_pp_clause_becomes_an_object_fact_not_a_failure():
    # "steam rises FROM the pot" -- SOURCE is not PLACE, and was never
    # captured by the old subj+place-only branch at all (a real, measured
    # no-relation-extracted case -- dev/PROSE_FAILURE_TAXONOMY.md's own
    # corpus). Round 2's catch-all asks it via the OBJECT template instead.
    sentences = corpus.iter_sentences("Steam rises from the pot.")
    p = _parser_for(*sentences, "steam", "pot")
    results = corpus.parse_passage(sentences, p)
    outcomes = corpus.taxonomy_counts(results)
    assert outcomes.get("no-relation-extracted", 0) == 0, outcomes
    triples = {(r.entity, r.relation, r.value) for r in results if isinstance(r, corpus.ParsedClause)}
    assert ("steam", "OBJECT", "pot") in triples, triples


def test_object_question_becomes_a_real_episode():
    sentences = corpus.iter_sentences(
        "Steam rises from the pot. Smoke rises from the fire. Mist rises from the lake.")
    p = _parser_for(*sentences, "steam", "pot", "smoke", "fire", "mist", "lake")
    results = corpus.parse_passage(sentences, p)
    eps = corpus.make_episodes(results, holdout="last", seed=0, doc_id="t2")
    assert len(eps) == 1
    ep = eps[0]
    assert ep.meta["relation"] == "OBJECT"
    assert ep.question == "what does the mist have ?"
    assert ep.answer_text == "lake"


def test_clean_transitive_pronoun_subject_not_routed_to_object_direction():
    # Item 1's addr_redirect scope stops at the subj+single-other-role
    # shape (see _extract_triples's own docstring on the address-vs-value
    # asymmetry): a pronoun SUBJECT of a clean transitive clause keeps the
    # OLD deterministic AGENT-direction resolution only -- no OBJECT-
    # direction triple is added for it (there is no candidate-set-based
    # value redirect in this codebase to route it through).
    sentences = corpus.iter_sentences("Mary is in the garden. She ate the apple.")
    p = _parser_for(*sentences, "mary", "garden", "apple")
    results = corpus.parse_passage(sentences, p)
    triples = [(r.entity, r.relation, r.value) for r in results if isinstance(r, corpus.ParsedClause)]
    assert ("apple", "AGENT", "mary") in triples, triples
    assert not any(rel == "OBJECT" and val == "apple" for _e, rel, val in triples), triples
