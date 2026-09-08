"""DRAFT hard-case hand-gold candidates for lead review -- NOT merged gold.

Writes `runs/hand_gold_draft.jsonl` (records) and the readable companion
`dev/HAND_GOLD_DRAFT.md`. Every record goes through `hand_gold.check_record`
(schema + oracle linearize + action legality + skeleton round-trip) and
`hand_gold.check_trains` (a real teacher-forced loss on a fresh policy).

4 categories x 4 = 16 candidates, per the two-part-gold decision (lead,
2026-09-07): these are the structures the DETERMINISTIC teacher cannot emit,
so they must be hand-fabricated.

Run:  python scripts/build_hand_gold_draft.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from hand_gold import (  # noqa: E402
    C, W, PRIME, CTX, MEM, CLAUSE, PREDICATE,
    hand_gold_record, context_entry, check_record, check_trains,
    load_usvs_default,
)

OUT_JSONL = _HERE.parent / "runs" / "hand_gold_draft.jsonl"


def build(usvs):
    """Returns [(record, note_dict), ...]. `note` is the human-facing
    rationale rendered into dev/HAND_GOLD_DRAFT.md."""
    D = []

    def add(rec, *, cat, why, choice, unsure=None):
        rec["meta"] = {"status": "DRAFT -- for lead review, NOT merged gold",
                       "category": cat, "why_hard": why, "design_choice": choice,
                       "unsure": unsure or []}
        D.append(rec)

    # ---------------------------------------------------------------- A
    # IMPERATIVES -- a SUBJECT the string does not contain. The teacher
    # cannot emit this: it reads roles off parse edges, and there is no
    # edge to an absent word.
    # ----------------------------------------------------------------
    add(hand_gold_record(
        "come here !",
        [C(kind="imperative", predicate=W("come"),
           roles=[("SUBJECT", PRIME("YOU")), ("PLACE", W("here"))])],
        usvs=usvs),
        cat="imperative",
        why="The addressee subject has NO surface token; the teacher's parse "
            "has no node to hang it on, so this clause shape appears in zero "
            "teacher gold.",
        choice="Synthesized SUBJECT as grounding.type:'prime', prime YOU, "
               "word='you', token_index=null (contract S5) -- RESOLVED, not a "
               "candidate slot: the grammar licenses exactly one filler. "
               "utterance_kind:'imperative'. Oracle emits EMIT_SYNTH_SLOT.",
        unsure=["Re-authoring of a v1 hand-gold item ('Come here !', "
                "encoder-handgold-v1) into the v2 lattice schema -- kept "
                "deliberately as the reference exemplar for the family."])

    add(hand_gold_record(
        "put the cup on the table .",
        [C(kind="imperative", predicate=W("put"),
           roles=[("SUBJECT", PRIME("YOU")), ("OBJECT", W("cup")),
                  ("PLACE", W("table"))])],
        usvs=usvs),
        cat="imperative",
        why="Three-role imperative (synth SUBJECT + overt OBJECT + PP PLACE). "
            "Existing hand-gold tops out at two roles, so nothing yet shows "
            "the synth slot coexisting with a full argument frame.",
        choice="'on' -> PLACE via clause.py's _PREP_RELATION (locative), NOT "
               "an ON PP-role. The synth SUBJECT sorts LAST in the oracle's "
               "node order (nulls last), so the derivation is "
               "GROUND(put) GROUND(cup) GROUND(table) EMIT_SYNTH_SLOT(SUBJECT).")

    add(hand_gold_record(
        "tell me about the dog .",
        [C(kind="imperative", predicate=W("tell"),
           roles=[("SUBJECT", PRIME("YOU")),
                  ("INDIRECT_OBJECT", W("me")), ("ABOUT", W("dog"))])],
        usvs=usvs),
        cat="imperative",
        why="A synthesized slot (SUBJECT=YOU, resolved) and an UNRESOLVED "
            "slot (the pronoun 'me', reference/memory) in the same clause -- "
            "the encoder must emit one committed node and one candidate slot "
            "side by side.",
        choice="'me' is ALWAYS the speaker, so it grounds as the resolved "
               "prime I (D3, dev/CURRENT_STATE.md decisions locked) -- "
               "symmetric with the imperative's synthesized addressee prime "
               "YOU -- rather than an unresolved memory reference. Unlike "
               "YOU, 'me' HAS a surface token, so it keeps its real "
               "token_index (1); the oracle's EMIT_SYNTH_SLOT now shifts to "
               "and consumes it instead of hardcoding token_index=null. "
               "'about' is an open PP-role ABOUT (attested 41x in teacher "
               "gold).")

    add(hand_gold_record(
        "wait for me .",
        [C(kind="imperative", predicate=W("wait"),
           roles=[("SUBJECT", PRIME("YOU")), ("FOR", W("me"))])],
        usvs=usvs),
        cat="imperative",
        why="Intransitive imperative whose only argument is a PP -- the "
            "'Wait !' already in hand-gold has no arguments at all.",
        choice="'for' -> the open PP-role FOR; the preposition token itself is "
               "not grounded (it is flushed by the oracle's terminal SHIFT), "
               "matching teacher-gold treatment of function words. 'me' -- "
               "the speaker being waited for -- grounds as the resolved "
               "prime I (D3), same as 'tell me ...' above.")

    # ---------------------------------------------------------------- B
    # INTERJECTIONS -- literal/gloss sense + utterance_kind only.
    # Per the Appraisal grounding DECISION (lead, 2026-09-04) there is NO
    # emotion node, NO FEEL explication, NO reaction-sense inventory.
    # Connotation is read off GOOD<->BAD comprehension-side.
    # ----------------------------------------------------------------
    add(hand_gold_record(
        "shit !",
        [C(kind="interjection", predicate=W("shit"), roles=[])],
        usvs=usvs),
        cat="interjection",
        why="A whole utterance that is one grounded token and no argument "
            "structure. The teacher's clause extractor needs a predicate node "
            "with edges; a bare exclamation yields none.",
        choice="Grounds to its LITERAL USVS candidate set "
               "(senses_of('shit') = 8 senses, crap.n.01 first) as an "
               "ordinary sense slot. utterance_kind:'interjection' is the "
               "ENTIRE extra signal. NO appraisal node, NO valence, no "
               "stance_target -- comprehension derives the feeling.")

    add(hand_gold_record(
        "alas !",
        [C(kind="interjection", predicate=W("alas"), roles=[])],
        usvs=usvs),
        cat="interjection",
        why="Listed in RESEARCH_NOTES M63.1c flag #1 as a PURE interjection "
            "with no synset -- but USVS actually returns one "
            "(unfortunately.r.01). A one-candidate sense slot: no ambiguity "
            "for comprehension to resolve, yet still an unresolved-slot shape.",
        choice="Literal grounding, same as 'shit !'. This record exists partly "
               "to CORRECT the M63.1c flag: 'alas' is lexically covered.")

    add(hand_gold_record(
        "ugh .",
        [C(kind="interjection", predicate=W("ugh"), roles=[])],
        usvs=usvs),
        cat="interjection",
        why="A GENUINELY pure interjection: senses_of('ugh') == [] in "
            "WordNet proper. Contract S4.2 says an uncovered content token "
            "grounds as type 'entity' -- so without D2, the utterance would "
            "carry no meaning at all beyond its kind.",
        choice="D2 (dev/CURRENT_STATE.md decisions locked): 'ugh' is now "
               "GLOSS-grounded -- `nsm_ct.ground.usvs.PURE_INTERJECTION_GLOSSES` "
               "mints `interj.ugh.01` ('an exclamation expressing disgust or "
               "horror') and grounds it through the SAME gloss->coordinate "
               "pipeline every real WordNet sense uses, additively and "
               "fingerprint-safe. `senses_of('ugh')` now returns "
               "['interj.ugh.01'], so `W('ugh')` grounds as an ordinary "
               "sense slot -- no bare 'entity', no record-level change "
               "needed here at all. utterance_kind 'interjection' is still "
               "the only speech-act marker; still NO appraisal node.")

    add(hand_gold_record(
        "nonsense !",
        [C(kind="interjection", predicate=W("nonsense"), roles=[])],
        usvs=usvs),
        cat="interjection",
        why="NEW family member (not in either existing hand-gold draft): an "
            "ordinary content noun used as a stance exclamation. This is the "
            "case the lead's 'connotation generalizes beyond 15 interjection "
            "words' argument needs -- same machinery, non-interjection lexeme.",
        choice="Identical treatment to 'shit !': literal senses "
               "(nonsense.n.01, folderal.n.01, nonsense.s.01) + "
               "utterance_kind 'interjection'. Nothing about the ENCODER "
               "record distinguishes an epithet from a swear -- correct by "
               "design; the difference is comprehension-side valence.")

    # ---------------------------------------------------------------- C
    # ELISION / FRAGMENTS -- a context_ref slot whose antecedent is named
    # BY CONTENT and resolved against a serialized context[].
    # ----------------------------------------------------------------
    ctx_eat = [context_entry(usvs, "the children eat bread .",
                             [C(predicate=W("eat"),
                                roles=[("SUBJECT", W("children")),
                                       ("OBJECT", W("bread"))])])]
    add(hand_gold_record(
        "more !",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("OBJECT", CTX("elision", of="bread", scope="roles")),
                  ("QUANTITY", W("more"))])],
        usvs=usvs, context=ctx_eat),
        cat="elision",
        why="Both the predicate AND an argument are absent from the string. "
            "The teacher cannot posit a node for a word that is not there.",
        choice="Two elision slots, one shape (contract S4.1): predicate "
               "inherits the context predicate, OBJECT inherits the context "
               "OBJECT; 'more' is an ordinary surface sense slot under role "
               "QUANTITY. predicate=null because predicate_grounding.type is "
               "'elision' (contract S3). This is contract S8.3's own worked "
               "example, re-authored through the REAL retrieval pipeline. "
               "QUANTITY is now an official structural role label (D4, "
               "dev/CURRENT_STATE.md decisions locked) -- a relation label "
               "only, never touching USVS sense coordinates -- alongside the "
               "new ADDITIVE/FOCUS and the existing DESCRIPTION/"
               "SPECIFICATION/COMPLEMENT modifier roles "
               "(`nsm_ct.clause.MODIFIER_RELATIONS`).")

    ctx_want_dog = [context_entry(usvs, "the boys want a dog .",
                                  [C(predicate=W("want"),
                                     roles=[("SUBJECT", W("boys")),
                                            ("OBJECT", W("dog"))])])]
    add(hand_gold_record(
        "me too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", W("me")),
                  ("OBJECT", CTX("elision", of="dog", scope="roles")),
                  ("ADDITIVE", W("too"))])],
        usvs=usvs, context=ctx_want_dog),
        cat="elision",
        why="The ONLY surface content is the subject; predicate and object are "
            "both inherited. A fragment whose overt token is an argument, not "
            "a head.",
        choice="SUBJECT 'me' = the speaker, now the resolved prime I (D3) "
               "rather than a reference/memory slot -- symmetric with the "
               "imperative addressee prime YOU, and it keeps its real "
               "token_index (0). Predicate + OBJECT are context elisions. "
               "The additive particle 'too' now takes the new ADDITIVE role "
               "(D4) instead of being dropped -- an ordinary structural "
               "relation label that never touches USVS grounding.")

    ctx_break = [context_entry(usvs, "the boys break the window .",
                               [C(predicate=W("break"),
                                  roles=[("SUBJECT", W("boys")),
                                         ("OBJECT", W("window"))])])]
    add(hand_gold_record(
        "the dog did .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W("dog")),
                  ("OBJECT", CTX("elision", of="window", scope="roles"))])],
        usvs=usvs, context=ctx_break),
        cat="elision",
        why="VP-ellipsis with a STRANDED AUXILIARY -- a family neither "
            "existing hand-gold draft covers. 'did' is a surface token that "
            "stands in for the elided predicate without being it.",
        choice="D6 (dev/CURRENT_STATE.md decisions locked): the elided "
               "predicate slot now KEEPS 'did's token_index (2) via the new "
               "`predicate_token_index` field -- CTX(..., word='did') -- so "
               "tense/polarity survive on the stranded auxiliary even though "
               "the predicate's MEANING is still inherited (predicate=null, "
               "predicate_grounding.type='elision', ref -> context[0] "
               "clause 0 PREDICATE 'break'). `encoder_model.clause_node_order` "
               "now reads this field for any non-sense/non-entity predicate, "
               "and `linearize_tree`'s EMIT_UNRESOLVED_SLOT shifts to and "
               "consumes it exactly like any other node with a real surface "
               "token. The overt SUBJECT is a normal sense slot, the OBJECT "
               "is a context elision.")

    ctx_two = [
        context_entry(usvs, "the girls want a doll .",
                      [C(predicate=W("want"),
                         roles=[("SUBJECT", W("girls")), ("OBJECT", W("doll"))])]),
        context_entry(usvs, "they ask the father .",
                      [C(predicate=W("ask"),
                         roles=[("SUBJECT", W("they")), ("OBJECT", W("father"))])]),
    ]
    add(hand_gold_record(
        "and a hat too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates",
                         context_index=None),
           roles=[("SUBJECT", CTX("reference", of="girls", scope="roles")),
                  ("OBJECT", W("hat")),
                  ("ADDITIVE", W("too"))])],
        usvs=usvs, context=ctx_two),
        cat="elision",
        why="MULTI-ENTRY context[] where the antecedent is NOT the most recent "
            "sentence: 'and a hat too' inherits 'want' from context[0], not "
            "'ask' from context[1]. Nothing in the repo exercises the context[] "
            "ARRAY that canonical v2 restored (v2-addendum had a single "
            "context object).",
        choice="candidates span BOTH entries (2 predicate handles, 4 role "
               "handles) -- the encoder emits the whole retrieved set and "
               "commits to nothing; ref names the gold antecedent in "
               "context[0]. This is the record that makes context_index "
               "load-bearing. 'too' now takes the ADDITIVE role (D4) instead "
               "of being dropped, same as 'me too .' above.")

    # ---------------------------------------------------------------- D
    # SYNTHESIZED / DROPPED ARGUMENTS -- an argument position with no
    # surface token that is NOT the imperative addressee.
    # ----------------------------------------------------------------
    ctx_run = [context_entry(usvs, "the boys run into the room .",
                             [C(predicate=W("run"),
                                roles=[("SUBJECT", W("boys")),
                                       ("PLACE", W("room"))])])]
    add(hand_gold_record(
        "opened the box .",
        [C(predicate=W("opened"),
           roles=[("SUBJECT", CTX("reference", of="boys", scope="roles")),
                  ("OBJECT", W("box"))])],
        usvs=usvs, context=ctx_run),
        cat="synth_argument",
        why="English narrative subject-drop: a finite clause with a real "
            "predicate and object but NO subject token. The teacher emits a "
            "subjectless clause or fails; it never posits the empty slot.",
        choice="The dropped SUBJECT is an UNRESOLVED slot "
               "(type reference, source context, word=null, token_index=null) "
               "-- NOT a prime. Primes are for grammar-licensed single "
               "fillers (the imperative addressee); a dropped subject has a "
               "candidate SET and is comprehension's to bind. Oracle emits "
               "EMIT_UNRESOLVED_SLOT with token_index=None.",
        unsure=["The predicate 'opened' retrieves THREE ADJECTIVAL senses "
                "(open.a.05, opened.s.01, open.s.09) and no verb sense, so "
                "under contract S7 this record's predicate is unrecallable by "
                "construction. Symptom of decision D5, not of this record."])

    ctx_es = [context_entry(usvs, "el niño está en la casa .",
                            [C(predicate=W("está"),
                               roles=[("SUBJECT", W("niño")),
                                      ("PLACE", W("casa"))])], lang="es")]
    add(hand_gold_record(
        "bebió el agua .",
        [C(predicate=W("bebió"),
           roles=[("SUBJECT", CTX("reference", of="niño", scope="roles")),
                  ("OBJECT", W("agua"))])],
        usvs=usvs, context=ctx_es, lang="es"),
        cat="synth_argument",
        why="Spanish pro-drop: the subject is carried by verb morphology and "
            "is structurally absent. The code-switch family named in the "
            "two-part-gold decision.",
        choice="Identical construct to the English subject-drop above -- a "
               "reference slot with a context antecedent. FINDING: USVS "
               "already indexes OMW Spanish lemmas (senses_of('niño') = 8, "
               "'agua' = 7, 'casa' = 14), so cross-lingual grounding needs NO "
               "new machinery; the only gap is inflection ('bebió', 'está' -> "
               "[]), which is the SAME gap English past tense has. See "
               "decision D5.")

    ctx_put = [context_entry(usvs, "put the flour in the bowl .",
                             [C(kind="imperative", predicate=W("put"),
                                roles=[("SUBJECT", PRIME("YOU")),
                                       ("OBJECT", W("flour")),
                                       ("PLACE", W("bowl"))])])]
    add(hand_gold_record(
        "stir well .",
        [C(kind="imperative", predicate=W("stir"),
           roles=[("SUBJECT", PRIME("YOU")),
                  ("OBJECT", CTX("elision", of="flour", scope="roles"))])],
        usvs=usvs, context=ctx_put),
        cat="synth_argument",
        why="TWO absent arguments of DIFFERENT kinds in one clause: a "
            "RESOLVED synthesized subject (prime YOU) and an UNRESOLVED "
            "dropped object (elision, several candidates). Recipe/instruction "
            "register. Nothing existing combines the two.",
        choice="The single most important record in the batch for the core "
               "boundary: the encoder emits a committed node AND a candidate "
               "set in the same clause, and never chooses between 'flour' and "
               "'bowl' -- the candidate list carries both.",
        unsure=["The context clause's own synthesized SUBJECT (prime YOU, "
                "token_index=null) lands in the elision slot's candidate pool. "
                "Should resolved prime nodes be filtered out of antecedent "
                "candidate sets? See decision D7.",
                "'well' (manner adverb) is dropped -- same gap as 'too' above."])

    ctx_park = [context_entry(usvs, "we go to the park .",
                              [C(predicate=W("go"),
                                 roles=[("SUBJECT", W("we")),
                                        ("PLACE", W("park"))])])]
    add(hand_gold_record(
        "sounds good .",
        [C(predicate=W("sounds"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W("good"))])],
        usvs=usvs, context=ctx_park),
        cat="synth_argument",
        why="PROPOSITIONAL anaphora: the dropped subject's antecedent is not "
            "an entity but the WHOLE prior clause ('that we go to the park'). "
            "The ref schema can address a clause (slot:null) but nothing in "
            "the repo has ever used it.",
        choice="ref = {context_index 0, tree_index 0, clause 0, slot null} -- "
               "'the clause itself', with candidates the clause handle. "
               "'good' takes COMPLEMENT (clause.py MODIFIER_RELATIONS).",
        unsure=["ref.slot:null is OVERLOADED: contract S8.2 uses null "
                "role_index to mean 'genuinely ambiguous, not committed', "
                "while here null slot means 'the antecedent IS the clause'. "
                "These need distinguishing. See decision D1 (HAND_GOLD_DRAFT "
                "numbering -- distinct from the locked CURRENT_STATE.md D1, "
                "forest top-1).",
                "COMPLEMENT (and DESCRIPTION/SPECIFICATION/SUBJECT_COMPLEMENT) "
                "are usable via clause.py's MODIFIER_RELATIONS but still not "
                "listed in the contract's own S3 role-vocabulary table -- a "
                "documentation gap D4 (QUANTITY/ADDITIVE/FOCUS) did not "
                "itself widen; worth folding in together."])

    return D


def main() -> int:
    usvs = load_usvs_default()
    records = build(usvs)

    print(f"built {len(records)} draft records; checking...\n")
    failures = 0
    for rec in records:
        try:
            rep = check_record(rec, usvs, strict=True)
            acts = rep["trees"][0]["actions"]
            print(f"  OK  {rec['text']!r:32s} steps={len(acts):2d}  {' '.join(acts)}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {rec['text']!r}\n      {exc}")
    if failures:
        print(f"\n{failures} record(s) FAILED the gate -- not writing.")
        return 1

    print("\nteacher-forced loss on a fresh policy (must be finite):")
    losses = check_trains(records, usvs)
    print(f"  mean loss = {losses['__mean__']:.3f} over {len(records)} records")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote {OUT_JSONL} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
