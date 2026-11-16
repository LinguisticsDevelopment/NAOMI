"""Fast tests for the referee (naomi.referee): licensing by grammar and by attestation, parser agreement, JSON."""
import json

from naomi import referee as R
from naomi.types import REL_ROOT, REL_UNATTACHED, Node, Tree


def node(i, tok, rel, pos, kids=(), subs=()):
    return Node(rel=rel, token=tok, token_index=i, lemma=tok, pos=pos, subtypes=list(subs), children=list(kids))


def tree(dep_rel="SUBJECT", dep_pos="NOUN", before=True):
    d = node(0 if before else 2, "dog", dep_rel, dep_pos)
    v = node(1, "ran", REL_ROOT, "VERB", [d])
    return Tree("the dog ran", v)


def attested():
    row = R.link_row("VERB", [], "SUBJECT", "NOUN", [], True, None)
    return R.Attested({row: 500}, [], min_count=5)


def test_attested_and_unattested_links():
    ref = R.Referee(attested())
    good = ref.check_links(tree())
    assert good[0]["licensed_attested"] and good[0]["attested_count"] == 500
    bad = ref.check_links(tree(dep_rel="OBJECT", dep_pos="ADV"))
    assert not bad[0]["licensed_attested"] and "attested 0x" in bad[0]["attested_reason"]
    assert not bad[0]["licensed_grammar"] and bad[0]["grammar_reason"]


def test_grammar_licenses_a_rule_it_has_and_rejects_nonsense():
    g = R.GrammarLicense()
    assert g.check("VERB", "SUBJECT", "NOUN", "before")[0] is True
    assert g.check("NOUN", "SUBJECT", "ADP", "after")[0] is False
    assert g.check("NOT_A_TAG", "SUBJECT", "NOUN", "before")[0] is False


def test_accept_reject_and_json():
    ref = R.Referee(attested())
    ok = ref.referee(tree(), "the dog ran", roundtrip=False)
    assert ok["accept"] and ok["reasons"] == []
    rej = ref.referee(tree(dep_rel="OBJECT", dep_pos="ADV"), "x", roundtrip=False)
    assert not rej["accept"] and rej["reasons"]
    json.dumps(ok), json.dumps(rej)                        # JSON-serialisable


def test_parser_agreement_lists_changes():
    parser = Tree("s", node(1, "ran", REL_ROOT, "VERB", [node(0, "dog", "SUBJECT", "NOUN"),
                                                          node(2, "home", REL_UNATTACHED, "NOUN")]))
    same = tree()
    r = R.Referee.agree_parser(same, parser)
    assert r["parser_links"] == 1 and r["share_kept"] == 1.0 and r["changed"] == []
    other = tree(dep_rel="OBJECT")
    r2 = R.Referee.agree_parser(other, parser)
    assert r2["share_kept"] == 0.0 and r2["share_kept_head_only"] == 1.0 and r2["changed"][0]["reason"] == "relation changed"
    json.dumps(r2)
