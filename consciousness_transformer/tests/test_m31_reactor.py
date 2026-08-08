"""M31 consumer-gate tests: content-word ``meaning_source`` switch on the clause
perception path (see RESEARCH_NOTES §0h). Covers three guarantees:

1. Default ("explication") is a zero-behavior-change no-op — a batch built with
   the default matches one built by explicitly passing "explication", and the
   pre-existing ``test_clause_reactor.py`` suite (which never passes the new
   arg) stays green.
2. "usvs" mode changes CONTENT-WORD (relation/value) vectors to unit-norm USVS
   handles, while ENTITY-VARIABLE atoms (bound into ``entity`` and any
   ``var:`` filler used as a value) are IDENTICAL between modes — entities are
   atomic referents, never routed through content-word grounding.
3. A word USVS doesn't know falls back to the explication path exactly.
"""

import numpy as np
import pytest
import torch

from nsm_ct.clause_reactor import _content_vec, build_clause_batch
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.input_encoder import ParserInputEncoder
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.structure import PARSE_LABELS
from nsm_ct.tokenizer import SimpleTokenizer
from nsm_ct.tpr import TPRCodec
from nsm_ct.usvs_bridge import usvs_handle


def _env(n=24, seed=0, dim=48, max_level=8):
    eps = CurriculumGenerator(max_level=max_level, seed=seed).generate(n)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return eps, parser, NSMMeaningResolver(), TPRCodec(dim=dim)


def _batches_equal(a, b) -> bool:
    for field in ("entity", "relation", "value", "pred", "is_q", "mask", "options", "answer"):
        ta, tb = getattr(a, field), getattr(b, field)
        if not torch.equal(ta, tb):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. default == explication (zero behavior change)
# ---------------------------------------------------------------------------
def test_default_meaning_source_is_explication():
    eps, parser, resolver, codec = _env()
    default_batch = build_clause_batch(eps, parser, resolver, codec)
    explicit_batch = build_clause_batch(eps, parser, resolver, codec, "explication")
    assert _batches_equal(default_batch, explicit_batch)


def test_default_meaning_source_matches_positional_call_without_arg():
    """Calling build_clause_batch with the old 4-positional-arg signature (as every
    other caller in the repo does) is untouched by the new kwarg's existence."""
    eps, parser, resolver, codec = _env(n=12, seed=1)
    a = build_clause_batch(eps, parser, resolver, codec)
    b = build_clause_batch(eps, parser, resolver, codec)
    assert _batches_equal(a, b)


# ---------------------------------------------------------------------------
# 2. usvs mode: content vectors change + are unit-norm; entity vectors identical
# ---------------------------------------------------------------------------
def test_usvs_mode_produces_unit_norm_content_vectors_different_from_explication():
    eps, parser, resolver, codec = _env()
    expl = build_clause_batch(eps, parser, resolver, codec, "explication")
    usvs = build_clause_batch(eps, parser, resolver, codec, "usvs")

    # value vectors differ between modes wherever a real (non-zero) value exists
    real_value_mask = usvs.mask.bool() & (usvs.value.norm(dim=-1) > 1e-6)
    assert real_value_mask.any(), "fixture produced no real content-word value steps"
    diffs = (usvs.value[real_value_mask] - expl.value[real_value_mask]).norm(dim=-1)
    assert (diffs > 1e-4).any()

    # usvs content vectors are unit-norm (usvs_handle always returns a unit vector)
    norms = usvs.value[real_value_mask].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # option vectors (content words, e.g. the MC place options) likewise change
    opt_diffs = (usvs.options - expl.options).norm(dim=-1)
    assert (opt_diffs > 1e-4).any()


def test_usvs_mode_leaves_entity_variable_atoms_untouched():
    """Entities are atomic var: fillers (mary, john, ...) — never content-word
    grounded — so switching meaning_source must not move them at all."""
    eps, parser, resolver, codec = _env()
    expl = build_clause_batch(eps, parser, resolver, codec, "explication")
    usvs = build_clause_batch(eps, parser, resolver, codec, "usvs")
    # the `entity` channel is built exclusively from var: atoms / zeros — identical
    assert torch.equal(expl.entity, usvs.entity)
    # the reserved MAYBE/idk option atoms (non-content options) are also untouched
    assert torch.equal(expl.coord, usvs.coord) if expl.coord is not None else True


def test_usvs_handle_is_unit_norm_for_known_content_words():
    codec = TPRCodec(dim=48)
    for w in ("kitchen", "office", "garden", "bedroom", "hallway", "bathroom"):
        v = usvs_handle(w, codec.dim)
        assert v is not None
        assert v.dtype == np.float32
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 3. unknown-to-USVS words fall back to the explication path
# ---------------------------------------------------------------------------
def test_unknown_word_falls_back_to_explication():
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=48)
    word = "xyzzyzzynotarealword"
    assert usvs_handle(word, codec.dim) is None      # precondition: USVS doesn't know it

    cache_a: dict = {}
    cache_b: dict = {}
    expl = _content_vec(word, resolver, codec, cache_a, "explication")
    usvs_fallback = _content_vec(word, resolver, codec, cache_b, "usvs")
    assert np.array_equal(expl, usvs_fallback)


def test_known_word_does_not_fall_back():
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=48)
    word = "kitchen"
    assert usvs_handle(word, codec.dim) is not None  # precondition: USVS knows it

    expl = _content_vec(word, resolver, codec, {}, "explication")
    usvs_vec = _content_vec(word, resolver, codec, {}, "usvs")
    assert not np.array_equal(expl, usvs_vec)
    assert abs(float(np.linalg.norm(usvs_vec)) - 1.0) < 1e-4
