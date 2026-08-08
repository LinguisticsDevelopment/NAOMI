"""Gates for the M32.1 trained sense chooser (nsm_ct.sense_chooser)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nsm_ct.episode import generate_ambiguity_episodes
from nsm_ct.sense_chooser import (
    D_DEFAULT,
    Example,
    SenseChooser,
    build_example,
    collate,
    predicted_sense_ids,
)
from nsm_ct.usvs_bridge import _DEFAULT_DIR
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(
    not wordnet_available() or not _DEFAULT_DIR.exists(),
    reason="needs WordNet + a built data/usvs artifact",
)


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_forward_shape():
    eps = generate_ambiguity_episodes(12, seed=0)
    examples = [build_example(e, D_DEFAULT) for e in eps]
    batch = collate(examples)
    k = batch["candidates"].shape[1]
    model = SenseChooser(d=D_DEFAULT)
    logits = model(batch["candidates"], batch["mask"], batch["context"])
    assert logits.shape == (len(examples), k)
    assert batch["mask"].shape == (len(examples), k)
    assert batch["context"].shape == (len(examples), D_DEFAULT)


def test_param_count_under_50k():
    model = SenseChooser(d=D_DEFAULT)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 50_000, f"chooser has {n_params} params, expected <50k"


# ---------------------------------------------------------------------------
# Overfit-tiny-batch sanity
# ---------------------------------------------------------------------------


def test_overfits_tiny_batch():
    eps = generate_ambiguity_episodes(20, seed=1)
    examples = [build_example(e, D_DEFAULT) for e in eps]
    # every M32 family episode's gold sense must be among USVS-grounded candidates
    assert all(e.gold_idx != -100 for e in examples)

    torch.manual_seed(0)
    model = SenseChooser(d=D_DEFAULT)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch = collate(examples)

    acc = 0.0
    for _ in range(500):
        logits = model(batch["candidates"], batch["mask"], batch["context"])
        loss = F.cross_entropy(logits, batch["gold_idx"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        pred = logits.argmax(dim=-1)
        acc = (pred == batch["gold_idx"]).float().mean().item()

    assert acc == pytest.approx(1.0), f"failed to overfit 20 episodes, final acc={acc}"


# ---------------------------------------------------------------------------
# Determinism given seed
# ---------------------------------------------------------------------------


def test_deterministic_given_seed():
    eps = generate_ambiguity_episodes(16, seed=2)
    examples = [build_example(e, D_DEFAULT) for e in eps]
    batch = collate(examples)

    torch.manual_seed(42)
    model_a = SenseChooser(d=D_DEFAULT)
    with torch.no_grad():
        logits_a = model_a(batch["candidates"], batch["mask"], batch["context"])

    torch.manual_seed(42)
    model_b = SenseChooser(d=D_DEFAULT)
    with torch.no_grad():
        logits_b = model_b(batch["candidates"], batch["mask"], batch["context"])

    assert torch.equal(logits_a, logits_b)
    assert predicted_sense_ids(examples, logits_a) == predicted_sense_ids(examples, logits_b)


def test_context_vector_deterministic():
    from nsm_ct.sense_chooser import context_vector

    eps = generate_ambiguity_episodes(5, seed=3)
    for ep in eps:
        a = context_vector(ep, D_DEFAULT)
        b = context_vector(ep, D_DEFAULT)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Unknown-sense fallback (None vectors handled, never crashes)
# ---------------------------------------------------------------------------


def test_unknown_sense_fallback_all_masked_row():
    """A row where every candidate is USVS-ungrounded (mask all 0) must not
    produce NaN/crash, and must fall back to a valid (in-range) index."""
    d = D_DEFAULT
    b, k = 3, 4
    candidates = torch.zeros(b, k, d)
    mask = torch.zeros(b, k)  # nothing grounded for any example
    context = torch.randn(b, d)

    model = SenseChooser(d=d)
    logits = model(candidates, mask, context)

    assert torch.isfinite(logits).all(), "all-masked row produced non-finite logits"
    idx = logits.argmax(dim=-1)
    assert (idx >= 0).all() and (idx < k).all()


def test_unknown_sense_fallback_mixed_row():
    """One example has a real candidate, another has none: the masked one must
    not crash softmax/CE and the real one must still be scored normally."""
    d = D_DEFAULT
    k = 3
    candidates = torch.randn(2, k, d)
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    context = torch.randn(2, d)

    model = SenseChooser(d=d)
    logits = model(candidates, mask, context)
    # masked-out real candidate in row 0 must not be chosen
    assert logits[0, 2] == float("-inf")
    assert torch.isfinite(logits[0, :2]).all()
    # row 1 is fully unmasked (fallback), so no -inf entries
    assert torch.isfinite(logits[1]).all()

    probs = F.softmax(logits, dim=-1)
    assert torch.isfinite(probs).all()


def test_candidate_vectors_masks_ungrounded_sense(monkeypatch):
    """If usvs_sense_handle can't ground a real WordNet sense, candidate_vectors
    must zero it out and mask it (not crash, not fabricate a vector)."""
    import nsm_ct.sense_chooser as sc

    sc.candidate_sense_ids.cache_clear()
    sc.candidate_vectors.cache_clear()
    monkeypatch.setattr(sc, "usvs_sense_handle", lambda sense_id, d, **kw: None)

    ids, vecs, mask = sc.candidate_vectors("bank", 32)
    assert len(ids) > 1
    assert vecs.shape == (len(ids), 32)
    assert np.all(mask == 0.0)
    assert not vecs.any()

    sc.candidate_sense_ids.cache_clear()
    sc.candidate_vectors.cache_clear()


def test_build_example_unknown_word_does_not_crash():
    from nsm_ct.episode import Episode
    from nsm_ct.sense_chooser import build_example

    ep = Episode(
        context=["the zzznotaword sat there ."],
        question="what kind of zzznotaword is it ?",
        answer_text="foo",
        options=["foo", "bar"],
        answer_idx=0,
        meta={
            "family": "fake",
            "homograph": "zzznotaword",
            "gold_sense": "zzznotaword.n.01",
            "mfs_sense": "zzznotaword.n.01",
            "sense_key": "A",
        },
    )
    ex = build_example(ep, D_DEFAULT)
    assert isinstance(ex, Example)
    assert ex.gold_idx == -100  # unknown word has no real candidates to match
    assert ex.candidates.shape[1] == D_DEFAULT
