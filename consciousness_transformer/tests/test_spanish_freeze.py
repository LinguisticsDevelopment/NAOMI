"""Tests for the Spanish Freeze Test's perception-side deliverables (see
dev/ROADMAP_LONG_TERM.md "The Spanish Freeze Test" and
scripts/probe_spanish_freeze.py).

Covers: es_lexicon.json.gz build determinism, Spanish template verification
(exact-role gate, mirroring test_curriculum2.py's English contract),
generate_freeze_pairs determinism, and stream-equivalence sanity (entity/
relation exact match, value cosine above a floor) on a small sample.
"""

from __future__ import annotations

import importlib.util
import os

import pytest


def _load_builder():
    """Load scripts/build_parser_lexicon.py by path (scripts/ isn't a package)."""
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "build_parser_lexicon.py")
    spec = importlib.util.spec_from_file_location("build_parser_lexicon", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    try:
        return _load_builder()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"nltk/wordnet unavailable: {exc}")


# ---------------------------------------------------------------------------
# 1. Lexicon build determinism
# ---------------------------------------------------------------------------

def test_es_lexicon_build_is_deterministic(builder):
    lex1 = builder.build("spa")
    lex2 = builder.build("spa")
    assert lex1 == lex2


def test_es_lexicon_covers_curriculum_core_nouns(builder):
    lex = builder.build("spa")
    for w in ("jardín", "cocina", "oficina", "pelota", "caja", "libro"):
        assert w in lex, w
        assert lex[w][0][0] == "NOUN", (w, lex[w])


def test_en_lexicon_build_unchanged_fingerprint(builder):
    """The Spanish --lang flag must not have touched the English path (M41
    contract, committed data/en_lexicon.json.gz fingerprint 80fa20bf6b16c13f)."""
    import hashlib
    import json

    lex = builder.build("eng")
    payload = json.dumps(lex, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    fp = hashlib.sha256(payload).hexdigest()[:16]
    assert fp == "80fa20bf6b16c13f"
    assert len(lex) == 159993


# ---------------------------------------------------------------------------
# 2. Grammar file
# ---------------------------------------------------------------------------

def test_spanish_grammar_rulesets_match_english():
    """spanish.json is a deliberate clone of english.json (quantum_parser's
    rules match on abstract Tag/NodeType/SubType categories, not literal
    words -- see the file's own metadata.description)."""
    import json

    root = os.path.join(os.path.dirname(__file__), "..", "..", "quantum_parser", "grammars")
    with open(os.path.join(root, "english.json"), encoding="utf-8") as f:
        en = json.load(f)
    with open(os.path.join(root, "spanish.json"), encoding="utf-8") as f:
        es = json.load(f)
    assert es["order"] == en["order"]
    assert es["rulesets"] == en["rulesets"]
    assert es["metadata"]["language"] == "spanish"


# ---------------------------------------------------------------------------
# 3. Spanish template verification (exact-role gate)
# ---------------------------------------------------------------------------

def test_spanish_place_move_templates_all_pass():
    from nsm_ct.curriculum2 import verify_templates_es

    results = verify_templates_es()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    assert all(results.values()), results


def test_spanish_transfer_take_template_all_roles_correct():
    from nsm_ct.curriculum2 import verify_transfer_templates_es

    results = verify_transfer_templates_es()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    for key, v in results.items():
        assert v["ok"], (key, v)


def test_spanish_pronoun_subject_edge_lands_on_pronoun():
    """Parse-layer only (see curriculum2.py's module note): the SUBJECT edge
    is structurally correct even though clause.py's English-only _PRONOUNS
    means it is not yet recognized as an entity downstream."""
    from nsm_ct.curriculum2 import verify_pronoun_templates_es

    results = verify_pronoun_templates_es()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    for pr, v in results.items():
        assert v["subject_ok"], (pr, v)


def test_spanish_question_template_documented_limitation():
    """Locks the CURRENT, measured limitation (mirrors
    test_dropped_templates_are_actually_bad's pattern): wh-question
    subject-verb inversion does not produce a clause, because Spanish
    "está" is tagged VERB-only (see curriculum2.py's module note for why
    making it AUX/VERB-ambiguous, like English "is", was not done). If this
    ever starts passing, this test should be updated, not silently left
    green on outdated reasoning.
    """
    from nsm_ct.clause import extract_discourse
    from nsm_ct.curriculum2 import QUESTION_TEMPLATE_ES, _PLACES_ES
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    place = _PLACES_ES["garden"]
    sent = QUESTION_TEMPLATE_ES.format(p=place["det"])
    tok = SimpleTokenizer.build([sent], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="es")
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    graph = parser._parse_graph(sent)
    clauses, _links = extract_discourse(graph)
    assert clauses == []


# ---------------------------------------------------------------------------
# 4. generate_freeze_pairs determinism
# ---------------------------------------------------------------------------

def test_freeze_pairs_deterministic_given_seed():
    from nsm_ct.curriculum2 import generate_freeze_pairs

    a = generate_freeze_pairs(30, seed=1)
    b = generate_freeze_pairs(30, seed=1)
    assert a == b


def test_freeze_pairs_different_seeds_differ():
    from nsm_ct.curriculum2 import generate_freeze_pairs

    a = generate_freeze_pairs(30, seed=1)
    b = generate_freeze_pairs(30, seed=2)
    assert a != b


def test_freeze_pairs_excludes_fred():
    """fred mistags as VERB under English's suffix heuristic (pre-existing,
    see curriculum2.py's _MALE_NAMES exclusion) -- generate_freeze_pairs
    must not draw it, or the English side of the pair silently loses its
    clause."""
    from nsm_ct.curriculum2 import generate_freeze_pairs

    pairs = generate_freeze_pairs(200, seed=0)
    drawn = {p.get("name") for p in pairs} | {p.get("taker") for p in pairs} | {p.get("source") for p in pairs}
    drawn.discard(None)
    assert "fred" not in drawn


# ---------------------------------------------------------------------------
# 5. Stream equivalence (smoke test on a small sample; the full N>=200 run
#    lives in scripts/probe_spanish_freeze.py and is reported, not re-run
#    here to keep the suite fast)
# ---------------------------------------------------------------------------

def test_stream_equivalence_smoke():
    from nsm_ct.clause import EntityTracker, clause_tpr, extract_discourse
    from nsm_ct.curriculum2 import generate_freeze_pairs
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.meaning_es import SpanishMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.tpr import TPRCodec

    pairs = generate_freeze_pairs(60, seed=0)
    tok_en = SimpleTokenizer.build([s for p in pairs for s in p["en"]],
                                    extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    tok_es = SimpleTokenizer.build([s for p in pairs for s in p["es"]],
                                    extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser_en = ParserInputEncoder(tok_en, lang="en")
    parser_es = ParserInputEncoder(tok_es, lang="es")
    if getattr(parser_en, "_parser", None) is None or getattr(parser_es, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")

    codec = TPRCodec(dim=64)
    en_resolver = NSMMeaningResolver()
    es_resolver = SpanishMeaningResolver()

    entity_total = entity_match = 0
    relation_total = relation_match = 0
    cosines = []
    for p in pairs:
        en_graph = parser_en._parse_graph(p["en"][0])
        es_graph = parser_es._parse_graph(p["es"][0])
        en_clauses, _ = extract_discourse(en_graph)
        es_clauses, _ = extract_discourse(es_graph)
        if not en_clauses or not es_clauses:
            continue
        en_m, en_triples = clause_tpr(en_clauses[0], codec, en_resolver, EntityTracker())
        es_m, es_triples = clause_tpr(es_clauses[0], codec, es_resolver, EntityTracker())
        en_subj = en_clauses[0].args[0][1].token if en_clauses[0].args else None
        es_subj = es_clauses[0].args[0][1].token if es_clauses[0].args else None
        entity_total += 1
        entity_match += int((en_subj or "").lower() == (es_subj or "").lower())

        en_by_rel = {rel: val for _s, rel, val in en_triples}
        es_by_rel = {rel: val for _s, rel, val in es_triples}
        for rel in set(en_by_rel) | set(es_by_rel):
            relation_total += 1
            if rel in en_by_rel and rel in es_by_rel:
                relation_match += 1
                a, b = en_by_rel[rel], es_by_rel[rel]
                na, nb = float((a**2).sum() ** 0.5), float((b**2).sum() ** 0.5)
                if na > 1e-8 and nb > 1e-8:
                    cosines.append(float(a @ b / (na * nb)))

    assert entity_total >= 40, "too many pairs failed to parse at all"
    assert entity_match / entity_total == 1.0, "entities are language-neutral atoms (same name strings)"
    assert relation_match / relation_total >= 0.95, "relation labels should match after the prep-relation seam"
    assert cosines, "no matched-relation value pairs to compare"
    mean_cos = sum(cosines) / len(cosines)
    assert mean_cos > 0.6, f"mean value cosine too low, mechanism likely broken: {mean_cos:.3f}"


# ---------------------------------------------------------------------------
# 6. SpanishMeaningResolver: synset-keyed mechanism + the documented leak
# ---------------------------------------------------------------------------

def test_spanish_resolver_matches_english_for_mfs_aligned_word():
    """"jardín"/"garden" share the SAME MFS synset (garden.n.01 both sides,
    verified by scripts/probe_spanish_freeze.py's coverage table) -- the
    resolved meaning trees should therefore be IDENTICAL, not just similar
    (this is the freeze test's central claim, checked at unit scale)."""
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.meaning_es import SpanishMeaningResolver

    en = NSMMeaningResolver()
    es = SpanishMeaningResolver()

    def flatten(node):
        out = [node.label]
        for c in node.children:
            out.extend(flatten(c))
        return out

    assert flatten(en.resolve("garden").root) == flatten(es.resolve("jardín").root)


def test_spanish_resolver_documents_mfs_ordering_leak():
    """"pelota"'s OMW-es sense-0 synset is NOT ball.n.01 (an MFS-ordering
    artifact, not a bug in this resolver -- see meaning_es.py's docstring
    and scripts/probe_spanish_freeze.py's leak report). Locked here so a
    future OMW data update that fixes the ordering is noticed, not silently
    assumed still-broken."""
    from nltk.corpus import wordnet as wn

    if not wn.synsets("bank"):
        pytest.skip("wordnet unavailable in this environment")
    es_synsets = wn.synsets("pelota", lang="spa")
    if not es_synsets:
        pytest.skip("OMW-es unavailable in this environment")
    assert es_synsets[0].name() != "ball.n.01", (
        "OMW-es ordering for 'pelota' changed -- update scripts/probe_spanish_freeze.py's "
        "reported leak list and this test")
