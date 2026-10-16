"""E2E QA SPIKE minimal adapter (dev/E2E_QA_SPIKE.md): a free-form MC
question (FairytaleQA/MCTest real text, via scripts/convert_fairytaleqa.py /
scripts/convert_mctest.py) usually has no NAME and no "the NOUN" for
nsm_ct.clause_reactor._question_entity to anchor on -- pre-M61,
build_clause_batch's own ``if qent is None: continue`` silently dropped
EVERY such episode for a ``kind == "prose"`` batch, which is why the
benchmark needed a minimal adapter at all (not a parser/clause.py fix).
This file is the regression anchor for that one change.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
_QP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "quantum_parser")
sys.path.insert(0, _QP_ROOT)

from nsm_ct import corpus  # noqa: E402
from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.episode import Episode  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _parser_for(*extra_sentences):
    tok = SimpleTokenizer.build(list(extra_sentences), extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok, lang="en")
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p


def _prose_episode(question, options, answer_idx):
    text = "Guddhu was skilled in magic. Thuggai asked guddhu to light the fire."
    sentences = corpus.iter_sentences(text)
    return sentences, Episode(
        context=sentences, question=question, answer_text=options[answer_idx],
        options=options, answer_idx=answer_idx, level=0,
        meta={"kind": "prose", "source_doc": "t", "answer_type": "entity"})


def test_free_form_question_with_no_qent_is_not_dropped():
    """Pre-M61: a question with no NAMESET word and no "the NOUN" made
    ``_question_entity`` return ``None``, and build_clause_batch's prose
    branch ``continue``d past the whole episode -- a FairytaleQA-shaped
    question ("who was skilled in magic ?") would score ZERO rows, not a
    bad row. Post-M61: the generic question-step fallback keeps it."""
    sentences, ep = _prose_episode(
        "who was skilled in magic ?", ["guddhu", "thuggai", "fire"], 0)
    p = _parser_for(*sentences, "guddhu", "thuggai", "fire", "magic")
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=16)

    from nsm_ct.clause_reactor import _question_entity
    assert _question_entity(ep.question) is None, "fixture question must exercise the no-qent fallback"

    batch = build_clause_batch([ep], p, resolver, codec)
    assert batch.entity.shape[0] == 1, "the episode must survive, not be dropped"
    assert bool(batch.is_q[0].any()), "a question step must have been appended"

    model = ClauseReactor(dim=16)
    with torch.no_grad():
        out = model(batch)
    assert out["answer_logits"].shape == (1, batch.options.shape[1])
    assert torch.isfinite(out["answer_logits"]).all()


def test_templated_question_still_uses_the_real_address_unchanged():
    """Byte-identical-behavior regression: a corpus.py-shaped templated
    question ("where is the X ?") must still resolve ``qent``/``qrel`` and
    take the OLD branch, not the new generic fallback -- the new branch is
    additive, not a replacement."""
    sentences, ep = _prose_episode("where is the fire ?", ["there", "nowhere"], 0)
    p = _parser_for(*sentences, "fire", "there", "nowhere")

    from nsm_ct.clause_reactor import _question_entity, _queried_role
    assert _question_entity(ep.question) == "fire"
    assert _queried_role(ep.question) == "PLACE"

    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=16)
    batch = build_clause_batch([ep], p, resolver, codec)
    assert batch.entity.shape[0] == 1
