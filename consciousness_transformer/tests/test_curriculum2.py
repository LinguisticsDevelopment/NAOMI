"""Tests for curriculum2: the template-varied curriculum used to probe template
overfitting (scripts/probe_template_overfit.py).
"""

from __future__ import annotations

import pytest

from nsm_ct.curriculum2 import (
    TEMPLATES,
    VOCAB_SCALE_PLACES,
    generate_varied_episodes,
    verify_templates,
)
from nsm_ct.episode import _PLACES


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_deterministic_given_seed():
    a = generate_varied_episodes(40, seed=7, template_set="A")
    b = generate_varied_episodes(40, seed=7, template_set="A")
    assert [e.context for e in a] == [e.context for e in b]
    assert [e.question for e in a] == [e.question for e in b]
    assert [e.answer_text for e in a] == [e.answer_text for e in b]
    assert [e.options for e in a] == [e.options for e in b]


def test_different_seeds_differ():
    a = generate_varied_episodes(40, seed=1, template_set="A")
    b = generate_varied_episodes(40, seed=2, template_set="A")
    assert [e.context for e in a] != [e.context for e in b]


# ---------------------------------------------------------------------------
# A/B disjointness
# ---------------------------------------------------------------------------
def test_template_sets_disjoint():
    a_templates = {t for action in TEMPLATES["A"].values() for t in action}
    b_templates = {t for action in TEMPLATES["B"].values() for t in action}
    assert a_templates & b_templates == set(), "A and B must share no template string"


def test_template_sets_disjoint_verb_phrases():
    # A stronger disjointness check than raw string equality: no shared
    # (verb, particle) bigram between the two sets, so B can't be "A with a
    # single word swapped in" (both sets do share the bare copula "is" —
    # "is in"/"is now in" vs "is at"/"is currently in"/"is standing in" — so
    # the check is on the two-word phrase, not the first word alone).
    def verb_phrases(s):
        out = set()
        for action in TEMPLATES[s].values():
            for t in action:
                words = t.replace("{n}", "").split()
                out.add(" ".join(words[:2]))
        return out

    assert verb_phrases("A") & verb_phrases("B") == set()


def test_generated_episodes_use_only_their_set():
    """Every context sentence in a template_set='A' run must come from set A's
    templates (with names/places substituted in), and never from set B, and
    vice versa — otherwise the two arms of the overfit probe aren't clean."""
    for which in ("A", "B"):
        eps = generate_varied_episodes(60, seed=3, template_set=which, max_level=6)
        other = "B" if which == "A" else "A"
        own_templates = {t for action in TEMPLATES[which].values() for t in action}
        other_templates = {t for action in TEMPLATES[other].values() for t in action}
        for ep in eps:
            for sent in ep.context + (ep.post_context or []):
                # Recover the template shape by substituting back "{n}"/"{p}" is
                # lossy w.r.t. place, but the fixed skeleton (words minus the two
                # slots) is what identifies the template, so compare skeletons.
                matched_own = any(
                    _matches_template(sent, t) for t in own_templates
                )
                matched_other = any(
                    _matches_template(sent, t) for t in other_templates
                )
                assert matched_own, f"{sent!r} did not match any {which} template"
                assert not matched_other, f"{sent!r} leaked a {other} template"


def _matches_template(sentence: str, template: str) -> bool:
    """True if ``sentence`` could have been produced by ``template.format(n=.., p=..)``."""
    prefix, _, rest = template.partition("{n}")
    mid, _, suffix = rest.partition("{p}")
    return sentence.startswith(prefix) and sentence.endswith(suffix) and mid in sentence


# ---------------------------------------------------------------------------
# Parse-success rate (the real quantum_parser, not a mock)
# ---------------------------------------------------------------------------
def test_kept_templates_parse_successfully():
    results = verify_templates(("A", "B"))
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    rate = sum(results.values()) / len(results)
    assert rate >= 0.9, f"kept templates should mostly parse; got {rate:.2f}: {results}"


def test_dropped_templates_are_actually_bad():
    """Sanity check on the negative result: the templates we chose to DROP
    really do fail verification (guards against silently fixing one and
    forgetting to move it back into TEMPLATES)."""
    from nsm_ct.curriculum2 import DROPPED_TEMPLATES, verify_templates
    from nsm_ct.clause import extract_discourse
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    name, place = "mary", "garden"
    texts = [t.format(n=name, p=place) for t in DROPPED_TEMPLATES] + [name, place]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")

    n_bad = 0
    for t in DROPPED_TEMPLATES:
        sent = t.format(n=name, p=place)
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        ok = False
        for cl in clauses:
            subj = pl = None
            for rel, arg in cl.args:
                if rel == "SUBJECT":
                    subj = (arg.token or "").lower()
                elif rel == "PLACE":
                    pl = (arg.token or "").lower()
            if subj == name and pl == place:
                ok = True
        n_bad += not ok
    assert n_bad == len(DROPPED_TEMPLATES), "a 'dropped' template actually parses now"


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------
def test_episodes_structurally_valid():
    for which in ("A", "B", "AB"):
        eps = generate_varied_episodes(60, seed=11, template_set=which, max_level=6)
        assert len(eps) == 60
        assert {e.level for e in eps} == {1, 2, 3, 4, 5, 6}
        for e in eps:
            assert e.is_multiple_choice
            assert e.answer_text in e.options
            assert e.options[e.answer_idx] == e.answer_text
            assert e.context
            assert e.question.strip().endswith("?")
            assert e.meta.get("src") == "curriculum2"
            assert e.meta.get("template_set") == which


def test_invalid_template_set_rejected():
    with pytest.raises(ValueError):
        generate_varied_episodes(5, seed=0, template_set="C")


def test_level4_post_context_after_question():
    eps = [e for e in generate_varied_episodes(60, seed=5, template_set="A", max_level=4)
           if e.level == 4]
    assert eps
    for e in eps:
        assert e.post_context, "level 4 needs a post-question distractor"


def test_level5_trust_labels_present():
    eps = [e for e in generate_varied_episodes(60, seed=5, template_set="B", max_level=5)
           if e.level == 5]
    assert eps
    for e in eps:
        assert e.trust_labels is not None
        assert len(e.trust_labels) == len(e.context)
        # the corroborated (1.0) place must be the answer
        assert e.answer_text in e.context[e.trust_labels.index(1.0)]


# ---------------------------------------------------------------------------
# vocab_scale
# ---------------------------------------------------------------------------
def test_vocab_scale_grows_distinct_nouns():
    small = generate_varied_episodes(200, seed=9, template_set="A", vocab_scale=False)
    big = generate_varied_episodes(200, seed=9, template_set="A", vocab_scale=True)

    def distinct_places(eps):
        return {e.answer_text for e in eps} | {o for e in eps for o in e.options}

    small_nouns = distinct_places(small)
    big_nouns = distinct_places(big)
    assert small_nouns <= set(_PLACES)
    assert len(big_nouns) > len(small_nouns)
    assert len(VOCAB_SCALE_PLACES) >= 50


def test_vocab_scale_pool_disjoint_ok_but_all_distinct():
    assert len(VOCAB_SCALE_PLACES) == len(set(VOCAB_SCALE_PLACES)), "no duplicate nouns in the pool"
