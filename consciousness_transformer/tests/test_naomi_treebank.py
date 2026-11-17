"""Treebank teacher: conversion, alignment and supervision composition (tiny inline snippets, no network)."""
from pathlib import Path

from naomi import treebank as TB

TABLE = TB.load_table()


def snippet(rows, sid="s1", text="x"):
    head = f"# sent_id = {sid}\n# text = {text}\n"
    return TB.parse_conllu((head + "\n".join("\t".join(r) for r in rows) + "\n").splitlines())[0]


def row(i, form, upos, head, rel, feats="_"):
    return [str(i), form, form, upos, "_", feats, str(head), rel, "_", "_"]


def by_form(s, c):
    f = {t.id: t.form for t in s.toks}
    return {f[i]: (f.get(h, "ROOT"), c.labels[i], c.flags[i]) for i, h in c.heads.items()}


def test_copula_and_preposition_swaps():
    s = snippet([row(1, "she", "PRON", 3, "nsubj"), row(2, "is", "AUX", 3, "cop"), row(3, "happy", "ADJ", 0, "root"),
                 row(4, "in", "ADP", 5, "case"), row(5, "bed", "NOUN", 3, "obl"), row(6, ".", "PUNCT", 3, "punct")])
    c = TB.convert(s, TABLE)
    assert c.well_formed
    d = by_form(s, c)
    assert d["is"][0] == "ROOT" and d["is"][1] == "ROOT"
    assert d["she"][:2] == ("is", "SUBJECT") and d["happy"][:2] == ("is", "COMPLEMENT")
    assert d["in"][:2] == ("happy", "MODIFICATION") and d["bed"][:2] == ("in", "PREPOSITION")
    assert "." not in d


def test_every_link_names_its_rule_and_operation():
    s = snippet([row(1, "a", "NOUN", 3, "nsubj"), row(2, "b", "ADP", 3, "case"), row(3, "c", "VERB", 0, "root")])
    c = TB.convert(s, TABLE)
    names = {r["name"] for r in TABLE["rules"]} | {"root"}
    for i, f in c.flags.items():
        assert f.split("+")[-1] in names
    assert "adposition_heads_noun" in c.flags[3] or "adposition_heads_noun" in c.flags[2]


def test_leading_conjunction_is_not_the_coordinator_and_leftover_is_coordination():
    s = snippet([row(1, "and", "CCONJ", 2, "cc"), row(2, "i", "PRON", 3, "nsubj"), row(3, "left", "VERB", 0, "root"),
                 row(4, "and", "CCONJ", 6, "cc"), row(5, "he", "PRON", 6, "nsubj"), row(6, "stayed", "VERB", 3, "conj")])
    c = TB.convert(s, TABLE)
    assert c.well_formed
    assert c.heads[4] == 0                       # the later conjunction heads the coordination
    assert c.heads[3] == 4 and c.heads[6] == 4   # both clauses hang under it
    assert c.heads[1] == 2 or c.heads[1] == 3    # the leading one stays with its clause
    assert c.labels[1] == "COORDINATION"


def test_relative_pronoun_label():
    s = snippet([row(1, "man", "NOUN", 0, "root"), row(2, "who", "PRON", 3, "nsubj", "PronType=Rel"),
                 row(3, "sleeps", "VERB", 1, "acl:relcl")])
    c = TB.convert(s, TABLE)
    assert c.labels[2] == "SUBJECT" and c.flags[2].endswith("rel_pron_subject_SUBJECT")


def test_alignment_masks_split_words():
    s = snippet([row(1, "cannot", "VERB", 0, "root"), row(2, "go", "VERB", 1, "xcomp")])
    g, st = TB.align(s, ["can", "not", "go"])
    assert st == TB.ALIGN_MASKED and g == [[], [], [2]]
    g, st = TB.align(s, ["cannot", "go"])
    assert st == TB.ALIGN_OK and g == [[1], [2]]
    assert TB.align(s, ["other", "go"])[1] == TB.ALIGN_FAIL


def test_compose_parser_wins_and_cycles_dropped():
    # tokens 0..3; parser: 1 -> 0 committed (head index 1), 2 and 3 unknown tops, 0 unknown top of piece {0,1}
    sv_head, sv_rel = [-1, 1, -1, -1], [None, "SUBJECT", None, None]
    sup, unk = [False, True, False, False], [True, False, True, True]
    L = TB.Link
    links = {0: L(1, "OBJECT", "obj", "r"),          # head 1 is inside token 0's own piece -> cycle
             1: L(3, "SPECIFICATION", "advmod", "parser_must_win"),
             2: L(3, "OBJECT", "obj", "R_a"), 3: L(-1, "ROOT", "root", "root")}
    c = TB.compose(sv_head, sv_rel, sup, unk, links, [True] * 4)
    assert c.source == [TB.SRC_UNKNOWN, TB.SRC_PARSER, TB.SRC_TREEBANK, TB.SRC_TREEBANK]
    assert c.head[1] == 1 and c.rel[1] == "SUBJECT"          # parser link untouched
    assert c.head[2] == 4 and c.head[3] == 0 and c.rel[3] == "ROOT"
    assert c.counts["cycle_dropped"] == 1 and c.counts["accepted"] == 2


def test_compose_root_conflict_and_masked():
    sv_head, sv_rel = [0, -1, -1], ["ROOT", None, None]
    c = TB.compose(sv_head, sv_rel, [True, False, False], [False, True, True],
                   {1: TB.Link(-1, "ROOT", "root", "root")}, [True] * 3)
    assert c.source[1] == TB.SRC_UNKNOWN and c.source[2] == TB.SRC_UNKNOWN
    assert c.counts["root_conflict"] == 1 and c.counts["no_link"] == 1


def test_table_has_no_words():
    import json
    t = json.loads(Path(TB.TABLE_PATH).read_text())
    vocab = set()
    for r in t["rules"]:
        vocab |= {x for k in ("dep_upos", "head_upos") for x in r.get(k, [])}
    assert all(v.isupper() for v in vocab)      # only tag names, nothing lowercase / lexical
