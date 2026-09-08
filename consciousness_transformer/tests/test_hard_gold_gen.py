"""Gates for the generated hard-case gold (lead directive, 2026-09-08):
`scripts/gen_hard_gold.py` + `src/nsm_ct/hard_gold_templates.py` generalize
the 16 hand-authored drafts (`dev/HAND_GOLD_DRAFT.md`) into typed-slot
templates filled at scale. Every generated record must pass the same gate
the drafts pass (`hand_gold.check_record`); this file checks the GENERATOR's
own invariants -- family coverage, split disjointness, determinism.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

_USVS_DIR = _ROOT / "data" / "usvs"


def _skip_if_no_usvs():
    if not _USVS_DIR.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")


@pytest.fixture(scope="module")
def generated():
    _skip_if_no_usvs()
    import gen_hard_gold as ghg
    out, report = ghg.generate(per_family=100, seed=0)
    return out, report


def test_every_family_generates_at_least_50_gated_records(generated):
    out, report = generated
    for family, stats in report["families"].items():
        assert stats["passed"] >= 50, f"{family}: only {stats['passed']} passed"


def test_no_surface_appears_in_two_splits(generated):
    out, _ = generated
    by_split = {s: {r["text"] for r in out[s]} for s in out}
    splits = list(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            assert not overlap, f"{splits[i]} / {splits[j]} share surfaces: {overlap}"


def test_test_template_ids_disjoint_from_train(generated):
    out, _ = generated
    train_ids = {r["meta"]["template_id"] for r in out["train"]}
    test_template_ids = {r["meta"]["template_id"] for r in out["test_template"]}
    assert not (train_ids & test_template_ids), (
        f"held-out templates leaked into train: {train_ids & test_template_ids}")
    # every held-out template id should also be absent from test_filler
    filler_ids = {r["meta"]["template_id"] for r in out["test_filler"]}
    assert not (filler_ids & test_template_ids), (
        f"held-out templates leaked into test_filler: {filler_ids & test_template_ids}")


def test_records_carry_required_meta(generated):
    out, _ = generated
    for split, records in out.items():
        for r in records:
            assert r["meta"]["source"] == "hard_gold_gen"
            assert r["meta"]["split"] == split
            assert r["meta"]["family"]
            assert r["meta"]["template_id"]


def test_all_records_pass_the_gate(generated):
    """Belt-and-suspenders: `generate()` already gates every record before
    keeping it, so this re-checks a sample through `hand_gold.check_record`
    directly to catch any drift between the generator's own pass/fail
    bookkeeping and the real gate."""
    import hand_gold as hg
    out, _ = generated
    usvs = hg.load_usvs_default()
    sample = (out["train"][:30] + out["test_filler"][:15] + out["test_template"][:15])
    for rec in sample:
        report = hg.check_record(rec, usvs, strict=False)
        assert report["ok"], f"{rec['text']!r} failed the gate: {report['problems']}"


def test_determinism_same_seed_same_output():
    _skip_if_no_usvs()
    import gen_hard_gold as ghg
    out1, _ = ghg.generate(per_family=20, seed=42)
    out2, _ = ghg.generate(per_family=20, seed=42)
    for split in out1:
        texts1 = [r["text"] for r in out1[split]]
        texts2 = [r["text"] for r in out2[split]]
        assert texts1 == texts2, f"split {split!r} not deterministic"


def test_different_seeds_can_differ():
    _skip_if_no_usvs()
    import gen_hard_gold as ghg
    out1, _ = ghg.generate(per_family=20, seed=1)
    out2, _ = ghg.generate(per_family=20, seed=2)
    texts1 = {r["text"] for r in out1["train"]}
    texts2 = {r["text"] for r in out2["train"]}
    assert texts1 != texts2


def test_pools_are_grounded():
    """Every sampled slot type (excluding PROPN/PRON, which ground by
    grammar rule rather than sense lookup) must have `usvs.senses_of_surface`
    non-empty for every pool member. `N_pl`/`VT_past`/`VI_past` now contain
    regular inflections (`"dogs"`, `"walked"`) that only ground through the
    morphy fallback, not raw `senses_of` -- this must use the SAME helper
    the pool builder and the actual grounding path use."""
    _skip_if_no_usvs()
    from nsm_ct.ground.usvs import load_usvs
    from nsm_ct.hard_gold_templates import build_pools
    usvs = load_usvs(str(_USVS_DIR))
    pools, _ = build_pools(usvs)
    for key, words in pools.items():
        if key in ("PROPN", "PRON"):
            continue
        for w in words:
            cands, _ = usvs.senses_of_surface(w)
            assert cands, f"pool {key!r} filler {w!r} does not ground"


# ---------------------------------------------------------------------------
# senses_of_surface / article agreement (hard-gold-gen v2 quality fixes)
# ---------------------------------------------------------------------------

def test_senses_of_surface_dogs_lemmatizes_to_dog():
    _skip_if_no_usvs()
    from nsm_ct.ground.usvs import load_usvs
    usvs = load_usvs(str(_USVS_DIR))
    assert usvs.senses_of("dogs") == []  # raw surface: not its own WordNet lemma
    cands, lemma = usvs.senses_of_surface("dogs")
    assert lemma == "dog"
    assert cands
    assert cands == usvs.senses_of("dog")


def test_senses_of_surface_walked_lemmatizes_to_walk():
    _skip_if_no_usvs()
    from nsm_ct.ground.usvs import load_usvs
    usvs = load_usvs(str(_USVS_DIR))
    assert usvs.senses_of("walked") == []
    cands, lemma = usvs.senses_of_surface("walked")
    assert lemma == "walk"
    assert cands
    assert cands == usvs.senses_of("walk")


def test_senses_of_surface_prefers_raw_surface_when_it_grounds():
    """A word that already grounds on its own raw surface (e.g. an irregular
    past that doubles as a WordNet adjective/noun lemma, "closed"/"cut")
    must NOT be silently re-lemmatized -- the raw candidate set + lemma ==
    the surface word itself."""
    _skip_if_no_usvs()
    from nsm_ct.ground.usvs import load_usvs
    usvs = load_usvs(str(_USVS_DIR))
    for w in ("closed", "cut", "wanted"):
        raw = usvs.senses_of(w)
        assert raw
        cands, lemma = usvs.senses_of_surface(w)
        assert lemma == w
        assert cands == raw


def test_senses_of_surface_unknown_word_returns_empty():
    _skip_if_no_usvs()
    from nsm_ct.ground.usvs import load_usvs
    usvs = load_usvs(str(_USVS_DIR))
    cands, lemma = usvs.senses_of_surface("xyzzynotarealword")
    assert cands == []
    assert lemma == "xyzzynotarealword"


def test_article_agreement_never_emits_a_before_vowel_initial_filler():
    """Every generated `"...a {N}..."`-shaped record must have picked "an"
    when the filler is vowel-initial (article exceptions aside)."""
    _skip_if_no_usvs()
    import re
    from nsm_ct.hard_gold_templates import article, _ARTICLE_A_EXCEPTIONS
    for w in ("apple", "eye", "optic", "area", "elephant", "umbrella"):
        assert w not in _ARTICLE_A_EXCEPTIONS
        assert article(w) == "an", f"{w!r} should take 'an'"
    for w in ("dog", "cat", "unicorn", "unit", "university"):
        assert article(w) == "a", f"{w!r} should take 'a'"

    import gen_hard_gold as ghg
    out, _ = ghg.generate(per_family=30, seed=0)
    bad = []
    for split in out:
        for rec in out[split]:
            for m in re.finditer(r"\ba ([A-Za-z]\w*)", rec["text"]):
                w = m.group(1).lower()
                if w[:1] in "aeiou" and w not in _ARTICLE_A_EXCEPTIONS:
                    bad.append(rec["text"])
    assert not bad, f"'a' before a vowel-initial filler: {bad[:10]}"


def test_teacher_gold_inflected_word_grounding_unchanged_or_improved():
    """`build_encoder_gold_v2.ground_word` now also uses
    `senses_of_surface`. This does NOT rebuild `runs/encoder_gold_v2.jsonl`
    (out of scope for this batch) -- it only checks that, for a sample of
    words the SHIPPED v2 gold grounded as bare `entity` (raw `senses_of`
    empty), the new helper never makes things worse (a strict superset
    relationship: `senses_of_surface`'s raw-first branch is exactly the old
    behavior) and counts how many of them would now ground as `sense`."""
    _skip_if_no_usvs()
    gold_path = _ROOT / "runs" / "encoder_gold_v2.jsonl"
    if not gold_path.exists():
        pytest.skip("runs/encoder_gold_v2.jsonl not present in this checkout")
    import json
    from nsm_ct.ground.usvs import load_usvs
    usvs = load_usvs(str(_USVS_DIR))

    entity_words = set()
    with gold_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            for tree in rec["lattice"]["trees"]:
                for clause in tree["clauses"]:
                    pg = clause.get("predicate_grounding", {})
                    if pg.get("type") == "entity" and clause.get("predicate"):
                        entity_words.add(clause["predicate"].lower())
                    for role in clause["roles"]:
                        g = role.get("grounding", {})
                        if (g.get("type") == "entity" and role.get("word")
                                and not role.get("is_entity", False)):
                            entity_words.add(role["word"].lower())

    # Restrict to plain alphabetic tokens -- quoted-dialogue fragments like
    # "'ah"/"'ll" are punctuation-glued artifacts, not inflected content
    # words, and would never lemmatize under any scheme. Also re-check raw
    # `senses_of` against the LIVE usvs (not just trust the file's stored
    # grounding): the USVS artifact can gain coverage between the gold
    # file's build and this checkout's `build_usvs.py` run (e.g. gloss-
    # grounded interjection senses), so a handful of stored `entity`
    # groundings are already raw-gettable today and are not the "inflected,
    # still needs lemmatization" case this test targets.
    still_entity = sorted(w for w in entity_words
                          if w.isalpha() and not usvs.senses_of(w))
    sample = still_entity[:20]
    assert len(sample) == 20, "fixture too small to sample 20 entity-grounded words"

    improved, unchanged = [], []
    for w in sample:
        cands, lemma = usvs.senses_of_surface(w)
        if cands:
            improved.append((w, lemma, len(cands)))
        else:
            unchanged.append(w)

    print(f"\nteacher-gold sample: {len(improved)}/20 previously-entity words "
          f"now ground as sense via senses_of_surface: {improved}")
    assert len(improved) >= 1, "expected at least one inflected word to improve"


def test_families_cover_the_eight_hard_case_kinds():
    from nsm_ct.hard_gold_templates import FAMILIES
    expected = {
        "imperative", "pure_interjection", "content_interjection", "elision",
        "additive_focus", "quantity", "synth_subject", "speaker_prime",
    }
    assert set(FAMILIES) == expected


def test_each_family_has_at_least_eight_templates():
    from nsm_ct.hard_gold_templates import templates_by_family
    by_family = templates_by_family()
    for family, templates in by_family.items():
        assert len(templates) >= 8, f"{family} has only {len(templates)} templates"
