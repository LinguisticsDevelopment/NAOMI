"""M58d: the PLACE-grounding fix (RESEARCH_NOTES "M58b"'s first diagnosed
defect -- measured cosine(write,read) = -0.28 for a non-curriculum subject).

``_context_steps``'s bare-PLACE branch used to write the clause's entity
unconditionally as ``codec.filler_vec("var:" + subj)`` while the question
step (and every OTHER grounding path in this module) reads an entity via
``_ent_vec`` (``var:`` only for the six curriculum names in ``_NAMESET``,
else the content-word meaning vector). For a curriculum subject the two
formulas happen to coincide (the subject IS in ``_NAMESET``), so the write
and read sides silently agreed by ACCIDENT; for a prose subject ("bear")
they diverged, making the question structurally unanswerable. The fix
(clause_reactor.py's ``_context_steps``) makes the write side call
``_ent_vec`` too -- ONE code path, no flag -- so write and read agree BY
CONSTRUCTION for every subject, curriculum or prose.

Two things this file proves:

1. **The gate**: not one existing curriculum episode's grounded batch may
   change. There is no git available in this environment (same discipline
   tests/test_parser_round.py's own docstring documents for M58c), so
   ``test_curriculum_batches_byte_identical_to_pre_fix`` reconstructs the
   PRE-fix ``_context_steps`` locally (a verbatim copy with only the one
   buggy line restored) and monkeypatches it in for one batch build,
   comparing every tensor field against the post-fix batch -- the same
   "reconstruct the old behavior via a scoped monkeypatch, diff the whole
   battery" technique that file's ``test_curriculum_byte_identical_
   regression`` uses for the parser round.
2. **The fix**: cosine(write_vec, read_vec) == 1.0 for a non-curriculum
   subject ("bear"), where write/read used to measure -0.28.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct import clause_reactor  # noqa: E402
from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.clause_reactor import (  # noqa: E402
    _NAMESET,
    _TRANSFER_ROLE_MAP,
    _content_vec,
    _context_steps,
    _ent_vec,
    build_clause_batch,
)
from nsm_ct.curriculum2 import (  # noqa: E402
    generate_instance_episodes,
    generate_rich_episodes,
    generate_writeback_episodes,
)
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

_PROSE_PROBE_SENTENCES = [
    "the bear is in the forest .",
    "bear is in the forest .",
]


def _curriculum_battery():
    """~100+ episodes across old/writeback/instance/rich generators (M58d's
    locked test plan), mirroring test_parser_round.py's own battery."""
    episodes = []
    for lvl in range(1, 4):
        episodes.extend(CurriculumGenerator(max_level=lvl, seed=lvl).generate(10))
    episodes.extend(generate_writeback_episodes(30, seed=1))
    episodes.extend(generate_instance_episodes(30, seed=2))
    episodes.extend(generate_rich_episodes(30, seed=3))
    return episodes


@pytest.fixture(scope="module")
def env():
    episodes = _curriculum_battery()
    texts = list(_PROSE_PROBE_SENTENCES)
    for ep in episodes:
        texts.extend(ep.context)
        texts.append(ep.question)
        texts.extend(getattr(ep, "post_context", []) or [])
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=32)
    return episodes, parser, resolver, codec


# ---------------------------------------------------------------------------
# 1. THE GATE: byte-identical curriculum regression
# ---------------------------------------------------------------------------

def _old_context_steps(sent, parser, resolver, codec, cache, meaning_source="usvs"):
    """Verbatim pre-M58d ``_context_steps`` -- the ONLY change is the bare-
    PLACE branch's entity, restored to the buggy unconditional
    ``codec.filler_vec("var:" + subj)`` this module's own fix replaced with
    ``_ent_vec(subj, ...)``. Kept local to this test file (no flag in
    production code, per the locked design) purely to reconstruct the old
    tensor values for the byte-identity diff below.
    """
    d = codec.dim
    graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
    clauses, links = extract_discourse(graph)
    if not clauses:
        return []
    prime = links[0].prime if links else None
    coordv = codec.filler_vec(prime) if prime else np.zeros(d, np.float32)
    steps = []
    for cl in clauses:
        pred_vec = codec.filler_vec("pred:" + (cl.predicate or "").lower())
        obj_tok = next(((arg.token or "").lower() for rel, arg in cl.args if rel == "OBJECT"), None)
        if obj_tok:
            entity_vec = _ent_vec(obj_tok, resolver, codec, cache, meaning_source)
            for rel, arg in cl.args:
                if rel == "OBJECT":
                    continue
                tok = (arg.token or "").lower()
                if not tok:
                    continue
                mapped = _TRANSFER_ROLE_MAP.get(rel, rel)
                steps.append((entity_vec, codec.filler_vec("rel:" + mapped),
                              _ent_vec(tok, resolver, codec, cache, meaning_source),
                              pred_vec, coordv, 0))
            continue
        subj = place = None
        for rel, arg in cl.args:
            if rel == "SUBJECT":
                subj = (arg.token or "").lower()
            elif rel == "PLACE":
                place = (arg.token or "").lower()
        if not (subj and place):
            continue
        # THE PRE-FIX LINE (the bug): unconditional "var:"+subj, ignoring
        # whether subj is actually a curriculum name.
        steps.append((codec.filler_vec("var:" + subj), codec.filler_vec("rel:PLACE"),
                      _content_vec(place, resolver, codec, cache, meaning_source),
                      pred_vec, coordv, 0))
    return steps


_TENSOR_FIELDS = ("entity", "relation", "value", "pred", "is_q", "mask", "options", "answer")
# Optional (None for a candidate-free batch) -- diffed only when populated.
_OPTIONAL_TENSOR_FIELDS = ("coord", "cand_entity", "cand_mask", "cand_prior", "cand_gold",
                            "cand_feature", "cand_feature_per_candidate")


def test_curriculum_batches_byte_identical_to_pre_fix(env):
    episodes, parser, resolver, codec = env

    b_new = build_clause_batch(episodes, parser, resolver, codec)

    prior = clause_reactor._context_steps
    clause_reactor._context_steps = _old_context_steps
    try:
        b_old = build_clause_batch(episodes, parser, resolver, codec)
    finally:
        clause_reactor._context_steps = prior

    for field in _TENSOR_FIELDS:
        new_t, old_t = getattr(b_new, field), getattr(b_old, field)
        assert torch.equal(new_t, old_t), f"field {field!r} changed for the curriculum battery"
    for field in _OPTIONAL_TENSOR_FIELDS:
        new_t, old_t = getattr(b_new, field), getattr(b_old, field)
        assert (new_t is None) == (old_t is None), f"field {field!r} None-ness changed"
        if new_t is not None:
            assert torch.equal(new_t, old_t), f"field {field!r} changed for the curriculum battery"


def test_every_curriculum_bare_place_subject_is_in_nameset(env):
    """The MECHANISM behind the gate above, made explicit: the fix is a
    no-op for curriculum episodes exactly because every bare-PLACE clause's
    SUBJECT they ever generate is one of the six names in _NAMESET, where
    _ent_vec(name) == codec.filler_vec("var:" + name) by construction (see
    _ent_vec's own one-line definition). This is what makes the write and
    read sides "agree by accident" pre-fix and "agree by construction"
    post-fix, for curriculum episodes specifically.
    """
    episodes, parser, _resolver, _codec = env
    seen_subjects = set()
    for ep in episodes:
        for sent in list(ep.context) + list(getattr(ep, "post_context", []) or []):
            graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
            clauses, _links = extract_discourse(graph)
            for cl in clauses:
                roles = {rel: (arg.token or "").lower() for rel, arg in cl.args}
                if "OBJECT" not in roles and "SUBJECT" in roles and "PLACE" in roles:
                    seen_subjects.add(roles["SUBJECT"])
    assert seen_subjects, "expected at least one bare-PLACE clause in the curriculum battery"
    assert seen_subjects <= _NAMESET, seen_subjects - _NAMESET


# ---------------------------------------------------------------------------
# 2. THE FIX: write/read agree for a non-curriculum (prose) subject
# ---------------------------------------------------------------------------

def test_bare_place_write_read_cosine_one_for_prose_subject(env):
    """The measured defect (RESEARCH_NOTES M58b: cosine(write,read) = -0.28
    for non-curriculum subjects) is fixed: for "bear" (not in _NAMESET),
    the write-side entity vector _context_steps grounds a bare-PLACE clause
    with and the read-side vector the question step grounds via _ent_vec
    now cosine-agree exactly.
    """
    _episodes, parser, resolver, codec = env
    assert "bear" not in _NAMESET
    cache = {}

    steps = _context_steps("the bear is in the forest .", parser, resolver, codec, cache)
    assert len(steps) == 1
    write_vec = steps[0][0]
    read_vec = _ent_vec("bear", resolver, codec, cache)

    cos = float(np.dot(write_vec, read_vec) /
                (np.linalg.norm(write_vec) * np.linalg.norm(read_vec) + 1e-12))
    assert cos == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(write_vec, read_vec)

    # And the old (buggy) formula is a GENUINE regression fixture, not a
    # vacuous check: it used to disagree sharply with the read side.
    old_write_vec = codec.filler_vec("var:bear")
    old_cos = float(np.dot(old_write_vec, read_vec) /
                     (np.linalg.norm(old_write_vec) * np.linalg.norm(read_vec) + 1e-12))
    assert old_cos < 0.5, "the pre-fix formula should NOT agree with the read side"


def test_ent_vec_matches_var_filler_for_every_curriculum_name(env):
    """Locks the invariant the gate above relies on: _ent_vec(name) ==
    codec.filler_vec("var:" + name) for every one of the six curriculum
    names -- if this ever stopped being true, the byte-identity gate would
    need to catch it (and does, independently, via the tensor diff)."""
    _episodes, _parser, resolver, codec = env
    cache = {}
    for name in sorted(_NAMESET):
        assert np.array_equal(_ent_vec(name, resolver, codec, cache),
                               codec.filler_vec("var:" + name))
