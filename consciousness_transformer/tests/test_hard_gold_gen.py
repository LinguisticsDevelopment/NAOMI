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
    grammar rule rather than sense lookup) must have `usvs.senses_of`
    non-empty for every pool member."""
    _skip_if_no_usvs()
    from nsm_ct.ground.usvs import load_usvs
    from nsm_ct.hard_gold_templates import build_pools
    usvs = load_usvs(str(_USVS_DIR))
    pools, _ = build_pools(usvs)
    for key, words in pools.items():
        if key in ("PROPN", "PRON"):
            continue
        for w in words:
            assert usvs.senses_of(w), f"pool {key!r} filler {w!r} does not ground"


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
