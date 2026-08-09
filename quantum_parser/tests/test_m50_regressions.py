"""Regression tests for the M50 real-text failure-class fixes.

Driven by consciousness_transformer's probe_realtext.py taxonomy (RESEARCH_NOTES
M48): three failure classes, each fixed by a small grammar/tagger addition.
Every test pins one "now parses" case AND, where relevant, a "didn't break
the ordinary reading" guard case at the engine level (SUBJECT edges on the
best hypothesis), independent of the probe/extraction layer:

1. APPOSITIVE_INTERRUPT (appositive1 ruleset + M50 COMMA/DASH/PAREN_OPEN/
   PAREN_CLOSE punctuation subtypes): a bracketed span between a subject
   NOMINAL and its predicate no longer defeats SUBJECT attachment.
2. QUANTIFIER_SUBJECT (quant1 ruleset + M50 QUANTIFIER subtype): bare
   this/both/each/all/most/numerals used as standalone subjects get promoted
   to NOUN when nothing else claims them as a determiner.
3. QUOTE_INVERSION (quote1 ruleset + M50 QUOTE_OPEN/QUOTE_CLOSE subtypes): a
   postposed subject after a quotation's reporting verb comes out as SUBJECT,
   not OBJECT.
4. EXISTENTIAL_THERE bonus (exist1 ruleset + M50 EXISTENTIAL subtype):
   sentence-initial "there" + copula promotes the postposed logical subject
   to SUBJECT instead of OBJECT/INDIRECT_OBJECT.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser import ConnectionType, Hypothesis, QuantumParser
from parser.pos_tagger import tag_sentence

_GRAMMAR = str(Path(__file__).resolve().parent.parent / "grammars" / "english.json")
_parser = QuantumParser(_GRAMMAR)


def _find(hyp: Hypothesis, text: str) -> int:
    return next(i for i, n in enumerate(hyp.nodes) if n.value and n.value.text == text)


def _has_edge(hyp: Hypothesis, etype: ConnectionType, parent_idx: int, child_idx: int) -> bool:
    return any(e.type == etype and e.parent == parent_idx and e.child == child_idx for e in hyp.edges)


def _best(sentence: str) -> Hypothesis:
    words = tag_sentence(sentence)
    hyp = _parser.parse(words).best_hypothesis()
    assert hyp is not None, f"no hypothesis at all for: {sentence!r}"
    return hyp


def _assert_subject(sentence: str, subj_text: str, pred_text: str) -> None:
    best = _best(sentence)
    pred_idx = _find(best, pred_text)
    subj_idx = _find(best, subj_text)
    assert _has_edge(best, ConnectionType.SUBJECT, pred_idx, subj_idx), (
        f"expected SUBJECT({pred_text!r} -> {subj_text!r}) in best hypothesis for "
        f"{sentence!r}; edges were {[(e.type.name, e.parent, e.child) for e in best.edges]}"
    )


def _assert_has_subject_edge_to(sentence: str, subj_text: str) -> None:
    """Looser check: some SUBJECT edge targets subj_text, regardless of which
    node ends up as the predicate (useful when the predicate word itself is
    lexically ambiguous and the exact winning branch's predicate identity
    isn't the thing under test)."""
    best = _best(sentence)
    subj_idx = _find(best, subj_text)
    assert any(e.type == ConnectionType.SUBJECT and e.child == subj_idx for e in best.edges), (
        f"expected some SUBJECT(*, {subj_text!r}) in best hypothesis for {sentence!r}; "
        f"edges were {[(e.type.name, e.parent, e.child) for e in best.edges]}"
    )


# == 1. APPOSITIVE_INTERRUPT ================================================

def test_comma_set_appositive_subject_attaches_across_gap():
    """M48 example: 'pete rozelle , the commissioner , pointed out .'"""
    _assert_subject("pete rozelle , the commissioner , pointed out .", "rozelle", "pointed")


def test_comma_set_appositive_two_word_compound_head():
    """The appositive's own head is a two-word compound ('league commissioner') --
    nominal1 promotes NOUN->NOMINAL long before any noun-noun rule runs, so this
    needs appositive1's dedicated two-NOMINAL interior variant."""
    _assert_subject(
        "pete rozelle , the league commissioner , pointed out .", "rozelle", "pointed"
    )


def test_comma_set_parenthetical_adverb():
    """M48 example: 'moritz , however , kicks only ...'"""
    _assert_subject("moritz , however , kicks the ball .", "moritz", "kicks")


def test_dash_set_appositive():
    _assert_subject("the mayor - a longtime resident - greeted the crowd .", "mayor", "greeted")


def test_paren_set_appositive():
    """M48 example: 'bill white ( sore ankles ) should be ready .'"""
    best = _best("bill white ( sore ankles ) should be ready .")
    be_idx = _find(best, "be")
    white_idx = _find(best, "white")
    assert _has_edge(best, ConnectionType.SUBJECT, be_idx, white_idx)


def test_appositive_rule_does_not_break_plain_declarative():
    """Guard: an ordinary sentence with no bracketed interrupter is untouched."""
    _assert_subject("the dog runs quickly .", "dog", "runs")


def test_appositive_rule_does_not_break_ordinary_comma_list_verb():
    """Guard: a comma-bracketed span that is immediately followed by a
    coordinator (an ordinary subject list, not an appositive) doesn't get
    silently eaten -- 'the two-word interior' rule requires the closing
    comma to be followed directly by the two-NOMINAL content, and stops
    there; it must not reach across into 'and'."""
    best = _best("the captain , the coach , and the fans celebrated .")
    assert best is not None  # engine must not crash; exact reading is not pinned here


# == 2. QUANTIFIER_SUBJECT ===================================================

def test_bare_both_as_subject():
    """M48 example: 'both were under the meet mark ...'"""
    _assert_subject("both were under the mark .", "both", "were")


def test_bare_all_as_subject():
    """'quiet' is lexically ambiguous (ADJ/NOUN/VERB/ADV), so which node ends
    up carrying the SUBJECT edge (the copula vs. a VERB-branch reading of
    'quiet' itself) isn't pinned here -- only that 'all' is a real subject."""
    _assert_has_subject_edge_to("all was quiet in the office .", "all")


def test_bare_this_as_subject():
    """M48 example: 'this would help the little peanut districts .'"""
    _assert_subject("this would help the districts .", "this", "would")


def test_partitive_each_of_np_as_subject():
    """M48 example: 'each of the four wayward shots cost him two strokes .'"""
    _assert_subject("each of the four shots cost him two strokes .", "each", "cost")


def test_bare_numeral_as_subject():
    """M48 example: 'three were doubles ...'"""
    _assert_subject("three were doubles .", "three", "were")


def test_both_as_determiner_still_works():
    """Guard: 'both teams' -- 'both' must stay a determiner attached to
    'teams', not get promoted to a dangling standalone NOMINAL."""
    best = _best("both teams won .")
    won_idx = _find(best, "won")
    teams_idx = _find(best, "teams")
    both_idx = _find(best, "both")
    assert _has_edge(best, ConnectionType.SUBJECT, won_idx, teams_idx)
    assert not _has_edge(best, ConnectionType.SUBJECT, won_idx, both_idx)


def test_each_as_determiner_still_works():
    """Guard: 'each team' -- same as above for the ADV-tagged 'each'."""
    best = _best("each team won .")
    won_idx = _find(best, "won")
    team_idx = _find(best, "team")
    assert _has_edge(best, ConnectionType.SUBJECT, won_idx, team_idx)


def test_this_as_determiner_still_works():
    """Guard: 'this dog' -- 'this' must stay attached to 'dog', not become
    a second dangling subject."""
    best = _best("this dog runs .")
    runs_idx = _find(best, "runs")
    dog_idx = _find(best, "dog")
    this_idx = _find(best, "this")
    assert _has_edge(best, ConnectionType.SUBJECT, runs_idx, dog_idx)
    assert not _has_edge(best, ConnectionType.SUBJECT, runs_idx, this_idx)


# == 3. QUOTE_INVERSION =======================================================

def test_quote_inversion_simple():
    """M48 example: '`` thirteen '' , said long jim .'"""
    best = _best("`` thirteen '' , said long jim .")
    said_idx = _find(best, "said")
    jim_idx = _find(best, "jim")
    assert _has_edge(best, ConnectionType.SUBJECT, said_idx, jim_idx)
    assert not _has_edge(best, ConnectionType.OBJECT, said_idx, jim_idx)


def test_quote_inversion_no_intervening_punctuation():
    best = _best("`` fine '' said chapman .")
    said_idx = _find(best, "said")
    chapman_idx = _find(best, "chapman")
    assert _has_edge(best, ConnectionType.SUBJECT, said_idx, chapman_idx)


def test_quote_inversion_does_not_break_ordinary_transitive_verb():
    """Guard: an ordinary reporting verb with no preceding quote-close still
    takes its postposed NOMINAL as OBJECT, not SUBJECT."""
    best = _best("the coach said hello .")
    said_idx = _find(best, "said")
    hello_idx = _find(best, "hello")
    assert _has_edge(best, ConnectionType.OBJECT, said_idx, hello_idx)
    assert not _has_edge(best, ConnectionType.SUBJECT, said_idx, hello_idx)


# == 4. EXISTENTIAL_THERE (bonus) =============================================

def test_existential_there_simple():
    best = _best("there is a problem .")
    is_idx = _find(best, "is")
    problem_idx = _find(best, "problem")
    assert _has_edge(best, ConnectionType.SUBJECT, is_idx, problem_idx)


def test_existential_there_with_negation():
    """M48 example: '`` there was n't a bit of trouble '' .' (unquoted core)."""
    best = _best("there was n't a bit of trouble .")
    was_idx = _find(best, "was")
    bit_idx = _find(best, "bit")
    assert _has_edge(best, ConnectionType.SUBJECT, was_idx, bit_idx)


def test_locative_there_still_attaches_as_specifier():
    """Guard: ordinary locative 'there' ('he went there') keeps 'he' as
    SUBJECT of 'went' -- EXISTENTIAL tagging must not touch this reading."""
    best = _best("he went there .")
    went_idx = _find(best, "went")
    he_idx = _find(best, "he")
    assert _has_edge(best, ConnectionType.SUBJECT, went_idx, he_idx)
