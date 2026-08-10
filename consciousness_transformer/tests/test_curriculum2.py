"""Tests for curriculum2: the template-varied curriculum used to probe template
overfitting (scripts/probe_template_overfit.py).
"""

from __future__ import annotations

import pytest

from nsm_ct.curriculum2 import (
    TEMPLATES,
    VOCAB_SCALE_PLACES,
    _FEMALE_NAMES,
    _MALE_NAMES,
    _SENSE_BINDING_ELIGIBLE,
    _SENSE_BINDING_EXCLUDE_FAMILY,
    _SENSE_BINDING_NAMES,
    PronounCurriculumGenerator,
    association_only_baseline,
    generate_pronoun_episodes,
    generate_scaled_episodes,
    generate_sense_binding_episodes,
    generate_varied_episodes,
    nearest_entity_baseline,
    verify_pronoun_templates,
    verify_sense_binding_templates,
    verify_templates,
)
from nsm_ct.episode import _AMBIGUITY_FAMILIES, _NAMES, _PLACES


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


# ---------------------------------------------------------------------------
# "scaled" mode (M-scaling experiment's data axis, scripts/probe_scaled_training.py)
# ---------------------------------------------------------------------------
def _count_facts(ep) -> int:
    return len(ep.context) + len(ep.post_context or [])


def _count_distinct_entities(ep) -> int:
    text = " " + " ".join(ep.context + (ep.post_context or []) + [ep.question]) + " "
    return sum(1 for nm in _NAMES if f" {nm} " in text)


def test_scaled_deterministic_given_seed():
    a = generate_scaled_episodes(80, seed=7)
    b = generate_scaled_episodes(80, seed=7)
    assert [e.context for e in a] == [e.context for e in b]
    assert [e.post_context for e in a] == [e.post_context for e in b]
    assert [e.answer_text for e in a] == [e.answer_text for e in b]
    assert [e.options for e in a] == [e.options for e in b]
    assert [e.trust_labels for e in a] == [e.trust_labels for e in b]


def test_scaled_different_seeds_differ():
    a = generate_scaled_episodes(80, seed=1)
    b = generate_scaled_episodes(80, seed=2)
    assert [e.context for e in a] != [e.context for e in b]


def test_scaled_defaults_to_mixed_templates_and_vocab_scale():
    """generate_scaled_episodes' whole point is the scaling arm: mixed A+B
    templates and the 61-noun pool, on by default (unlike generate_varied_episodes,
    whose defaults are the single-template, 6-noun status quo)."""
    eps = generate_scaled_episodes(40, seed=0)
    assert all(e.meta.get("template_set") == "AB" for e in eps)
    assert all(e.meta.get("vocab_scale") is True for e in eps)
    assert all(e.meta.get("scaled") is True for e in eps)
    noun_pool = {o for e in eps for o in e.options}
    assert not noun_pool <= set(_PLACES), "vocab_scale should pull in the 61-noun pool, not just the 6 base places"


def test_scaled_episode_density_in_range():
    """Every scaled episode packs min_facts..max_facts context sentences
    (context + post_context together), and never claims more entities than
    max_entities -- the density knobs the scaling probe actually varies."""
    min_facts, max_facts, min_entities, max_entities = 4, 8, 2, 4
    eps = generate_scaled_episodes(300, seed=3, min_facts=min_facts, max_facts=max_facts,
                                    min_entities=min_entities, max_entities=max_entities)
    assert len(eps) == 300
    for e in eps:
        n_facts = _count_facts(e)
        assert min_facts <= n_facts <= max_facts, f"L{e.level} had {n_facts} facts: {e.context}"
        n_ent = _count_distinct_entities(e)
        assert 1 <= n_ent <= max_entities, f"L{e.level} had {n_ent} distinct entities: {e.context}"
        assert min_entities <= e.meta["n_entities"] <= max_entities

    # Aggregate: the density knobs should actually widen variety, not just be
    # present in metadata -- most episodes should see more than one entity.
    avg_entities = sum(_count_distinct_entities(e) for e in eps) / len(eps)
    assert avg_entities > 1.5, f"scaled episodes barely more varied than plain ones: avg={avg_entities:.2f}"


def test_scaled_density_knobs_are_respected():
    """A narrower min/max range actually narrows what gets generated."""
    eps = generate_scaled_episodes(150, seed=4, min_facts=5, max_facts=6,
                                    min_entities=3, max_entities=3)
    for e in eps:
        assert 5 <= _count_facts(e) <= 6
        assert e.meta["n_entities"] == 3


def test_scaled_min_greater_than_max_rejected():
    with pytest.raises(ValueError):
        generate_scaled_episodes(5, seed=0, min_facts=8, max_facts=4)
    with pytest.raises(ValueError):
        generate_scaled_episodes(5, seed=0, min_entities=4, max_entities=2)


def test_scaled_episodes_structurally_valid():
    eps = generate_scaled_episodes(120, seed=11)
    assert {e.level for e in eps} == {1, 2, 3, 4, 5, 6}
    for e in eps:
        assert e.is_multiple_choice
        assert e.answer_text in e.options
        assert e.options[e.answer_idx] == e.answer_text
        assert e.context
        assert e.question.strip().endswith("?")
        assert e.meta.get("src") == "curriculum2"


def test_scaled_level4_post_context_present():
    eps = [e for e in generate_scaled_episodes(120, seed=5) if e.level == 4]
    assert eps
    for e in eps:
        assert e.post_context, "level 4 needs at least one post-question distractor"


def test_scaled_level5_corroboration_labels_are_meaningful():
    """Every scaled level-5 episode still needs at least one trustworthy
    (label 1.0) statement that actually states the answer, and at least one
    contradicting (label 0.0) statement -- the corroboration-vs-contradiction
    signal from the plain curriculum, preserved under padding."""
    eps = [e for e in generate_scaled_episodes(200, seed=5) if e.level == 5]
    assert eps
    for e in eps:
        assert e.trust_labels is not None
        assert len(e.trust_labels) == len(e.context)
        assert any(lab == 1.0 and e.answer_text in ctx
                   for ctx, lab in zip(e.context, e.trust_labels)), e.context
        assert any(lab == 0.0 for lab in e.trust_labels), "no contradiction present"


# -- parse-verification gate: scaled mode introduces no new sentence shapes --
def test_scaled_mode_introduces_no_new_sentence_shapes():
    """The scaling experiment's whole premise is "more of the SAME verified
    sentences" -- density must come from generating more TEMPLATES-derived
    sentences, never from new, unverified surface forms. Every context/
    post_context sentence in scaled mode must match one of the already
    parser-verified templates in TEMPLATES (see verify_templates() /
    test_kept_templates_parse_successfully, which gates TEMPLATES itself)."""
    all_templates = [t for s in ("A", "B") for action in TEMPLATES[s].values() for t in action]
    eps = generate_scaled_episodes(150, seed=6)
    for e in eps:
        for sent in e.context + (e.post_context or []):
            assert any(_matches_template(sent, t) for t in all_templates), \
                f"{sent!r} does not match any parser-verified template"


# ---------------------------------------------------------------------------
# M53a -- pronoun-binding curriculum (dev/RESOLVER_BUILD_PLAN.md Phase 2)
# ---------------------------------------------------------------------------
def test_pronoun_templates_all_roles_correct():
    results = verify_pronoun_templates()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    assert results["CONTEXT_MOVE"]["ok"]
    for pr in ("she", "he", "it", "they"):
        assert results[f"PRONOUN_FIND[{pr}]"]["ok"], results[f"PRONOUN_FIND[{pr}]"]


def test_pronoun_deterministic_given_seed():
    a = generate_pronoun_episodes(40, seed=7)
    b = generate_pronoun_episodes(40, seed=7)
    assert [e.context for e in a] == [e.context for e in b]
    assert [e.meta for e in a] == [e.meta for e in b]
    assert [e.options for e in a] == [e.options for e in b]


def test_pronoun_different_seeds_differ():
    a = generate_pronoun_episodes(40, seed=1)
    b = generate_pronoun_episodes(40, seed=2)
    assert [e.context for e in a] != [e.context for e in b]


def test_pronoun_episodes_structurally_valid():
    eps = generate_pronoun_episodes(80, seed=3)
    assert len(eps) == 80
    for e in eps:
        assert e.level == 14
        assert e.is_multiple_choice
        assert e.answer_text in e.options
        assert e.options[e.answer_idx] == e.answer_text
        assert len(e.context) == 3
        assert e.question.strip().endswith("?")
        assert e.meta["kind"] == "pronoun_binding"
        assert e.meta["pronoun"] in ("she", "he")
        assert e.meta["pronoun_sentence_index"] == 2
        assert e.context[2].split()[0] == e.meta["pronoun"]
        gold, other = e.meta["gold_antecedent"], e.meta["other_entity"]
        assert gold != other
        assert {gold, other} <= set(_NAMES)
        assert gold in e.meta["registry_order"] and other in e.meta["registry_order"]
        assert e.meta["gold_place"] != e.meta["other_place"]
        assert e.answer_text == e.meta["gold_place"]
        assert "fred" not in (gold, other), "known parser landmine, see curriculum2.py"


def test_pronoun_gender_uniquely_identifies_antecedent():
    """The whole design hinges on gender disambiguating: the pronoun's
    gender must match the gold antecedent's gender and NOT the other
    entity's (else "she"/"he" wouldn't carry any information)."""
    from nsm_ct.membrane import NAME_GENDER

    eps = generate_pronoun_episodes(80, seed=4)
    for e in eps:
        want_gender = "F" if e.meta["pronoun"] == "she" else "M"
        assert NAME_GENDER[e.meta["gold_antecedent"]] == want_gender
        assert NAME_GENDER[e.meta["other_entity"]] != want_gender


# ---------------------------------------------------------------------------
# M56b: splittable name pools (dev/TRACK_C_DESIGN.md §1.8/Risk #4,
# RESEARCH_NOTES M56/M56b's held-out-name ablation) -- PronounCurriculumGenerator's
# female_names/male_names constructor override.
# ---------------------------------------------------------------------------
def test_pronoun_generator_default_pools_match_module_globals():
    """Default (no override) must reproduce the exact pre-M56b generator --
    same globals, same rng draws, for any seed."""
    a = PronounCurriculumGenerator(seed=5).generate(20)
    b = PronounCurriculumGenerator(seed=5, female_names=_FEMALE_NAMES, male_names=_MALE_NAMES).generate(20)
    assert [e.meta["gold_antecedent"] for e in a] == [e.meta["gold_antecedent"] for e in b]
    assert [e.meta["other_entity"] for e in a] == [e.meta["other_entity"] for e in b]
    assert [e.context for e in a] == [e.context for e in b]


def test_generate_pronoun_episodes_default_matches_no_kwargs():
    a = generate_pronoun_episodes(20, seed=6)
    b = generate_pronoun_episodes(20, seed=6, female_names=None, male_names=None)
    assert [e.context for e in a] == [e.context for e in b]


def test_pronoun_generator_name_split_deterministic():
    """Same custom pools + same seed -> identical episodes (determinism
    survives the new constructor args, not just the default path)."""
    kwargs = dict(seed=7, female_names=["mary"], male_names=["john", "daniel"])
    a = PronounCurriculumGenerator(**kwargs).generate(30)
    b = PronounCurriculumGenerator(**kwargs).generate(30)
    assert [e.meta["gold_antecedent"] for e in a] == [e.meta["gold_antecedent"] for e in b]
    assert [e.meta["other_entity"] for e in a] == [e.meta["other_entity"] for e in b]


def test_pronoun_generator_name_split_restricts_names_used():
    """A generator built from a restricted pool must NEVER sample a name
    outside it -- the property the held-out-name ablation depends on for
    both its train side and its held-out side."""
    train_female, train_male = ["mary"], ["john", "daniel"]
    eps = PronounCurriculumGenerator(seed=8, female_names=train_female, male_names=train_male).generate(60)
    used_names = {e.meta["gold_antecedent"] for e in eps} | {e.meta["other_entity"] for e in eps}
    assert used_names <= set(train_female) | set(train_male)
    # and the restriction is non-trivial: names OUTSIDE the pool never appear,
    # even though they're in the module-level pools generally available.
    assert used_names.isdisjoint({"sandra", "bill"})


def test_pronoun_generator_name_split_no_leak_between_train_and_held_out():
    """The ablation's core precondition: a TRAIN-pool generator and a
    HELD-OUT-pool generator (disjoint name sets by construction) never
    produce episodes referencing each other's names."""
    held_out_female, held_out_male = "sandra", "bill"
    train_female = [n for n in _FEMALE_NAMES if n != held_out_female]
    train_male = [n for n in _MALE_NAMES if n != held_out_male]

    train_eps = PronounCurriculumGenerator(seed=9, female_names=train_female,
                                            male_names=train_male).generate(80)
    held_out_eps = PronounCurriculumGenerator(seed=10, female_names=[held_out_female],
                                               male_names=[held_out_male]).generate(80)

    train_names = {e.meta["gold_antecedent"] for e in train_eps} | {e.meta["other_entity"] for e in train_eps}
    held_out_names = {e.meta["gold_antecedent"] for e in held_out_eps} | {e.meta["other_entity"] for e in held_out_eps}

    assert train_names.isdisjoint(held_out_names)
    assert held_out_names == {held_out_female, held_out_male}
    assert train_names == set(train_female) | set(train_male)


# ---------------------------------------------------------------------------
# Anti-recency data-design gate
# ---------------------------------------------------------------------------
def test_pronoun_anti_recency_is_at_least_half():
    eps = generate_pronoun_episodes(201, seed=5)   # odd n: parity must still hold
    n_anti = sum(1 for e in eps if e.meta["antecedent_recency"] == "old")
    assert n_anti / len(eps) >= 0.5


def test_nearest_entity_baseline_sits_at_or_below_chance_floor():
    """The scripted nearest-entity baseline (RESOLVER_BUILD_PLAN.md Phase 2's
    data-design check): on the anti-recency half it must be wrong 100% of
    the time (0.0 accuracy -- strictly below the 1/num_options chance
    floor), and overall it must be no better than the "recency" fraction
    (<=50%, since anti-recency is >=50% by design). If this test starts
    failing because the baseline scores high, the CURRICULUM's data design
    is broken -- fix the data, don't raise the bound."""
    eps = generate_pronoun_episodes(400, seed=8)
    result = nearest_entity_baseline(eps)
    assert result["n"] == 400
    assert result["n_anti_recency"] >= 200
    assert result["anti_recency_accuracy"] == 0.0
    assert result["accuracy"] <= 0.5 + 1e-9


def test_nearest_entity_baseline_ignores_non_pronoun_episodes():
    eps = generate_varied_episodes(20, seed=0) + generate_pronoun_episodes(10, seed=0)
    result = nearest_entity_baseline(eps)
    assert result["n"] == 10


# ---------------------------------------------------------------------------
# M54b: entity-keyed, binding-critical sense-ambiguity curriculum
# ---------------------------------------------------------------------------
def test_sense_binding_deterministic_given_seed():
    a = generate_sense_binding_episodes(60, seed=7)
    b = generate_sense_binding_episodes(60, seed=7)
    assert [e.context for e in a] == [e.context for e in b]
    assert [e.meta for e in a] == [e.meta for e in b]
    assert [e.options for e in a] == [e.options for e in b]


def test_sense_binding_different_seeds_differ():
    a = generate_sense_binding_episodes(60, seed=1)
    b = generate_sense_binding_episodes(60, seed=2)
    assert [e.context for e in a] != [e.context for e in b]


def test_sense_binding_episodes_structurally_valid():
    eps = generate_sense_binding_episodes(120, seed=3)
    assert len(eps) == 120
    for e in eps:
        assert e.level == 15
        assert e.is_multiple_choice
        assert e.answer_text in e.options
        assert e.options[e.answer_idx] == e.answer_text
        assert len(e.context) == 3
        assert e.question == f"what kind of {e.meta['homograph']} is it ?"
        assert e.meta["kind"] == "sense_binding"
        gold, other = e.meta["entity"], e.meta["other_entity"]
        assert gold != other
        assert {gold, other} <= set(_SENSE_BINDING_NAMES)
        assert "fred" not in (gold, other), "known parser landmine, see curriculum2.py"
        assert "bill" not in (gold, other), "collides with the 'bill' homograph family"
        # the anchor sentence (always context[-1]) is the gold entity's own
        # -- it must be the ONLY sentence carrying gold_name, and both cue
        # sentences (context[0:2]) must open with a distinct entity name.
        assert e.context[-1].startswith(gold)
        cue_subjects = {e.context[0].split()[0], e.context[1].split()[0]}
        assert cue_subjects == {gold, other}


def test_sense_binding_meta_completeness():
    eps = generate_sense_binding_episodes(80, seed=9)
    required = {"src", "kind", "family", "homograph", "gold_sense", "mfs_sense",
                "sense_key", "flipped", "entity", "other_entity", "gold_answer", "other_answer"}
    for e in eps:
        assert required <= set(e.meta)
        fam = _AMBIGUITY_FAMILIES[e.meta["family"]]
        assert fam["word"] == e.meta["homograph"]
        assert e.meta["gold_sense"] == fam["senses"][e.meta["sense_key"]]["synset"]
        assert e.meta["gold_answer"] == fam["senses"][e.meta["sense_key"]]["answer"]
        assert e.meta["flipped"] == (e.meta["gold_sense"] != e.meta["mfs_sense"])
        assert e.meta["gold_answer"] == e.answer_text


def test_sense_binding_flip_balance_is_exact_alternation():
    """Mirrors the pronoun curriculum's anti-recency counter discipline: a
    COUNTER, not RNG, so the flipped/unflipped split is exact (to within one
    episode of parity), not merely statistically likely."""
    eps = generate_sense_binding_episodes(300, seed=11)
    n_flipped = sum(1 for e in eps if e.meta["flipped"])
    assert abs(n_flipped - len(eps) / 2) <= 1


def test_sense_binding_excludes_landmine_family():
    eps = generate_sense_binding_episodes(400, seed=13)
    assert all(e.meta["family"] not in _SENSE_BINDING_EXCLUDE_FAMILY for e in eps)
    assert all("swim" not in (e.meta["gold_answer"], e.meta["other_answer"]) for e in eps)


def test_sense_binding_eligible_pairs_cover_both_flip_directions():
    flipped = [f for (_w, _k, f) in _SENSE_BINDING_ELIGIBLE if f]
    unflipped = [f for (_w, _k, f) in _SENSE_BINDING_ELIGIBLE if not f]
    assert len(flipped) > 10 and len(unflipped) > 10


# -- decoy-balance property (the association-only baseline's precondition) --
def test_sense_binding_decoy_balance_both_senses_present_distinct_entities():
    """Both senses' associated content word must be present in the episode,
    each attached to a DIFFERENT entity -- the property that makes
    bag-of-words association uninformative by construction."""
    eps = generate_sense_binding_episodes(150, seed=17)
    for e in eps:
        gold, other = e.meta["entity"], e.meta["other_entity"]
        gold_answer, other_answer = e.meta["gold_answer"], e.meta["other_answer"]
        assert gold_answer != other_answer
        gold_cue = next(s for s in e.context[:2] if s.startswith(gold))
        other_cue = next(s for s in e.context[:2] if s.startswith(other))
        assert gold_answer in gold_cue.split()
        assert other_answer in other_cue.split()
        assert set(e.options) == {gold_answer, other_answer}


# -- association-only baseline: must sit at chance BY CONSTRUCTION ----------
def test_association_only_baseline_at_chance():
    eps = generate_sense_binding_episodes(500, seed=21)
    result = association_only_baseline(eps)
    assert result["n"] >= 400
    assert abs(result["accuracy"] - 0.5) < 0.08, (
        "association-only baseline drifted from chance -- fix the curriculum's "
        "decoy balance, don't loosen this bound: " + str(result)
    )


def test_association_only_baseline_ignores_non_sense_binding_episodes():
    eps = generate_varied_episodes(20, seed=0) + generate_sense_binding_episodes(10, seed=0)
    result = association_only_baseline(eps)
    assert result["n"] == 10


# -- parser verification (verify_* pattern) ----------------------------------
def test_sense_binding_templates_all_verified():
    results = verify_sense_binding_templates()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    bad = {k: v for k, v in results.items() if not v["ok"]}
    assert not bad, bad


def test_sense_binding_anchor_clauses_never_carry_a_context_word():
    """The M54 same-clause leak, directly: every anchor this generator can
    pick must show SUBJECT == the sample name and NO other content word in
    the SAME clause besides the homograph (verify_sense_binding_templates's
    ANCHOR[...] checks assert this already; this test pins the specific
    families this generator is eligible to draw the anchor from, so a future
    _AMBIGUITY_FAMILIES edit that reintroduces a leaky anchor is caught even
    if quantum_parser is unavailable in some other environment)."""
    results = verify_sense_binding_templates()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    for (word, key, _flipped) in _SENSE_BINDING_ELIGIBLE:
        r = results[f"ANCHOR[{word}.{key}]"]
        assert r["ok"], r
        assert not r["context_leak"], r
