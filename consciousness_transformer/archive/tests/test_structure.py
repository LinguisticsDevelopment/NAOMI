"""Tests for tree-aware structured input (parser structure alignment)."""

from nsm_ct.episode import CurriculumGenerator
from nsm_ct.input_encoder import ParserInputEncoder, TokenInputEncoder
from nsm_ct.structure import NUM_ROLES, role_id
from nsm_ct.tokenizer import SimpleTokenizer, basic_tokenize


def _tok():
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.nsm_primes import PRIME_NAMES
    eps = CurriculumGenerator(max_level=6, seed=0).generate(12)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    return SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)


def test_align_structure_is_lossless_and_aligned():
    """Every token is preserved; roles/depths/meaning align 1:1, even if the
    parser drops a word (that token just gets NOROLE/0/no-meaning)."""
    tok = _tok()
    enc = ParserInputEncoder(tok)
    sent = "mary is in the kitchen ."
    ids, roles, depths, meanings = enc.encode_structured(sent)
    words = basic_tokenize(sent)
    assert len(ids) == len(roles) == len(depths) == len(meanings) == len(words)
    assert ids == tok.encode(sent)                               # exact same tokens as plain
    assert all(0 <= r < NUM_ROLES for r in roles)
    assert any(r != role_id("NOROLE") for r in roles)           # real syntactic role found
    # content words carry a meaning bag (prime ids); the parse covers some of them
    assert any(len(m) > 0 for m in meanings)


def test_token_encoder_emits_empty_structure():
    tok = _tok()
    ids, roles, depths, meanings = TokenInputEncoder(tok).encode_structured("mary is in the kitchen .")
    assert roles == [0] * len(ids) and depths == [0] * len(ids)  # NOROLE / depth 0
    assert meanings == [[] for _ in ids]                         # no meaning
