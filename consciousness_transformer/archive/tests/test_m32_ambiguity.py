"""Gates for the M32 ambiguity-bearing comprehension curriculum.

Generator determinism and sense-id sanity need no external data; the probe
end-to-end run needs WordNet + a built ``data/usvs`` artifact (same guard as
``tests/test_usvs_bridge.py``).
"""

import os
import sys

import pytest

from nsm_ct.episode import generate_ambiguity_episodes
from nsm_ct.usvs_bridge import _DEFAULT_DIR
from nsm_ct.wordnet import wordnet_available

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")


def test_generator_deterministic():
    a = generate_ambiguity_episodes(50, seed=7)
    b = generate_ambiguity_episodes(50, seed=7)
    assert len(a) == len(b) == 50
    for ea, eb in zip(a, b):
        assert ea.context == eb.context
        assert ea.question == eb.question
        assert ea.options == eb.options
        assert ea.answer_idx == eb.answer_idx
        assert ea.meta == eb.meta


def test_generator_different_seed_differs():
    a = generate_ambiguity_episodes(50, seed=1)
    b = generate_ambiguity_episodes(50, seed=2)
    assert [e.context for e in a] != [e.context for e in b]


def test_episode_shape():
    eps = generate_ambiguity_episodes(60, seed=3)
    for ep in eps:
        assert 2 <= len(ep.context) <= 4
        assert ep.is_multiple_choice
        assert len(ep.options) == 2
        assert ep.answer_text in ep.options
        assert ep.options[ep.answer_idx] == ep.answer_text
        for key in ("family", "homograph", "gold_sense", "mfs_sense", "sense_key"):
            assert key in ep.meta
        # the bare homograph word appears in at least one context sentence
        assert any(ep.meta["homograph"] in sent.split() for sent in ep.context)


def test_at_least_three_families():
    eps = generate_ambiguity_episodes(200, seed=0)
    families = {e.meta["family"] for e in eps}
    assert len(families) >= 3


@pytest.mark.skipif(not wordnet_available(), reason="needs WordNet")
def test_sense_ids_are_real_synsets():
    from nltk.corpus import wordnet as wn

    eps = generate_ambiguity_episodes(200, seed=0)
    for ep in eps:
        # raises WordNetError if the name isn't a real synset
        wn.synset(ep.meta["gold_sense"])
        wn.synset(ep.meta["mfs_sense"])


@pytest.mark.skipif(not wordnet_available(), reason="needs WordNet")
def test_mfs_matches_live_wordnet_lookup():
    """The hardcoded per-family MFS constant must match a live wn.synsets(word)[0]."""
    from nltk.corpus import wordnet as wn

    eps = generate_ambiguity_episodes(200, seed=0)
    seen = {}
    for ep in eps:
        seen[ep.meta["homograph"]] = ep.meta["mfs_sense"]
    for word, mfs in seen.items():
        live = wn.synsets(word)[0].name()
        assert mfs == live, f"{word}: recorded mfs {mfs!r} != live wn.synsets(word)[0] {live!r}"


def test_at_least_40pct_sense_flipped():
    eps = generate_ambiguity_episodes(400, seed=0)
    flipped = sum(1 for e in eps if e.meta["mfs_sense"] != e.meta["gold_sense"])
    assert flipped / len(eps) >= 0.40, f"only {flipped}/{len(eps)} flipped"


pytestmark_probe = pytest.mark.skipif(
    not wordnet_available() or not _DEFAULT_DIR.exists(),
    reason="needs WordNet + a built data/usvs artifact",
)


@pytestmark_probe
def test_probe_runs_end_to_end_on_20_episodes():
    sys.path.insert(0, _SCRIPTS_DIR)
    from probe_m32_ambiguity import run  # noqa: E402

    result = run(n_episodes=20, d=64, seed=0)
    assert len(result["episodes"]) == 20
    assert len(result["families"]) >= 1
    for (_label, _subset), a in result["acc"].items():
        assert a != a or 0.0 <= a <= 1.0  # nan (empty subset) or a valid fraction
    assert isinstance(result["gap"], float)


@pytestmark_probe
def test_probe_gold_beats_mfs_on_flipped_subset():
    sys.path.insert(0, _SCRIPTS_DIR)
    from probe_m32_ambiguity import run  # noqa: E402

    result = run(n_episodes=200, d=128, seed=0)
    gold_flipped = result["acc"][("GOLD", "flipped (mfs!=gold)")]
    mfs_flipped = result["acc"][("MFS", "flipped (mfs!=gold)")]
    assert gold_flipped - mfs_flipped > 0.10
