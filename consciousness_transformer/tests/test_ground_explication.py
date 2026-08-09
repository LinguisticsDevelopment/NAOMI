"""Gates for the M43 opt-in enrichment prototype (nsm_ct.ground.explication).

House rules under test: (1) default-off — alpha=0 must be byte-identical to
the untouched base signature, so nothing changes unless a caller explicitly
asks; (2) determinism — no RNG anywhere in the blend; (3) the M24 `exclude`
knob is load-bearing (it must actually remove a word's contribution), not a
silent no-op.
"""

import numpy as np
import pytest

from nsm_ct.ground.definition_graph import content_words
from nsm_ct.ground.explication import enriched_sense_dense, gloss_of
from nsm_ct.ground.usvs import build_usvs
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(not wordnet_available(), reason="needs WordNet")


@pytest.fixture(scope="module")
def small_usvs():
    return build_usvs(n_core=400, max_senses=800, depth=2)


def test_alpha_zero_is_identity(small_usvs):
    """Default (alpha=0) changes nothing — the build path never calls this
    function, and callers who pass alpha=0 get the exact base signature."""
    checked = 0
    for sid in small_usvs.sense_ids[:50]:
        base = small_usvs.sense_dense(sid)
        if base is None:
            continue
        enr = enriched_sense_dense(small_usvs, sid, alpha=0.0)
        assert np.array_equal(base, enr)
        checked += 1
    assert checked > 0


def test_enrichment_is_deterministic(small_usvs):
    """Same inputs -> byte-identical output, repeatedly, for several senses."""
    checked = 0
    for sid in small_usvs.sense_ids[:100]:
        a = enriched_sense_dense(small_usvs, sid, alpha=0.35)
        b = enriched_sense_dense(small_usvs, sid, alpha=0.35)
        if a is None:
            assert b is None
            continue
        assert np.array_equal(a, b)
        checked += 1
    assert checked > 0


def test_enrichment_renormalizes_to_unit_length(small_usvs):
    """Whenever a blend actually happens (content words found), the result is
    L2-unit (matching the convention `usvs_bridge` queries expect)."""
    found_blend = False
    for sid in small_usvs.sense_ids:
        gloss = gloss_of(sid)
        if not gloss:
            continue
        base = small_usvs.sense_dense(sid)
        if base is None or not base.any():
            continue
        v = enriched_sense_dense(small_usvs, sid, alpha=0.35)
        assert v is not None
        n = float(np.linalg.norm(v))
        # either untouched (no content vecs found -> falls back to base, which
        # may not be unit-norm) or genuinely blended -> unit-norm.
        if not np.array_equal(v, base):
            assert n == pytest.approx(1.0, abs=1e-4)
            found_blend = True
    assert found_blend


def test_exclude_removes_a_word_that_was_contributing(small_usvs):
    """M24 house rule: excluding a gloss word that actually supplies content
    must change the enriched vector, not silently ignore the argument."""
    core = set(small_usvs.core_words)
    for sid in small_usvs.sense_ids:
        gloss = gloss_of(sid)
        cw = [w for w in content_words(gloss) if w in core]
        if len(cw) < 2:
            continue
        with_all = enriched_sense_dense(small_usvs, sid, alpha=0.5)
        excl_one = enriched_sense_dense(small_usvs, sid, alpha=0.5, exclude=frozenset({cw[0]}))
        if with_all is None or excl_one is None:
            continue
        if np.array_equal(with_all, excl_one):
            continue  # this particular word happened not to move the mean; try another
        assert not np.array_equal(with_all, excl_one)
        return
    pytest.fail("no sense in the small fixture exercised the exclude path")


def test_default_build_path_untouched(small_usvs):
    """Sanity: explication.py must not be imported by usvs.py's build path —
    the artifact-facing surface is unchanged by this module's existence."""
    import nsm_ct.ground.usvs as usvs_mod
    assert "explication" not in usvs_mod.__dict__
