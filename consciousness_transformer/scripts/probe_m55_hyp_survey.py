"""M55a survey (step 2, "PARSER TOP-K EXPOSURE"): which sentence SHAPES in our
grammar actually produce multiple non-equivalent, close-margin top hypotheses
TODAY -- BEFORE any input_encoder/membrane code is written. Read-only against
quantum_parser (no code changes here); the result table is the load-bearing
deliverable for M55a's report (dev/NEXT_ARC_PLAN.md M55, RESEARCH_NOTES M56/
M56b/M56c, dev/TRACK_C_DESIGN.md Sec 1.10).

Method: parser.parse(words) already deduplicates structurally-EQUIVALENT
hypotheses at every grammar-rule application step (QuantumParser.parse's
"DEDUPLICATION" block, is_equivalent) and returns chart.hypotheses sorted
descending by (score, completeness_key) with only genuinely complete parses
kept when any exist. So "K>1 real readings" reduces to: after parsing, does
chart.hypotheses have length > 1? If so every entry beyond the first is, by
construction, NOT structurally equivalent to the first (dedup already ran) --
the margin is simply hyp[0].score - hyp[1].score.

Candidates tried: (a) every existing curriculum sentence SHAPE (TEMPLATES A/
B/C, transfer templates, sense-binding cue template) with a homograph word
(quantum_parser.pos_tagger.AMBIGUOUS_WORDS: book/run/duck/watch/train/fly/
fish/saw/park/... -- the M41 WordNet-lexicon-backed lattice) substituted into
each open slot; (b) a set of hand-written classic-garden-path-flavored
sentences built from the SAME vocabulary, since the toy grammar may not
support real reduced-relative-clause structures at all -- this is checked
empirically, not assumed.

Usage: python scripts/probe_m55_hyp_survey.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                 "quantum_parser"))

from src.parser.pos_tagger import tag_sentence, AMBIGUOUS_WORDS  # noqa: E402
from src.parser.quantum_parser import QuantumParser  # noqa: E402
from src.parser.enums import Tag  # noqa: E402

from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.quantum_adapter import hypothesis_to_graph  # noqa: E402

GRAMMAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "quantum_parser", "grammars", "english.json")

NAMES = ["mary", "john", "sandra", "daniel", "bill", "fred"]
PLACES = ["garden", "kitchen", "library", "station", "harbor", "museum"]

# Verb/noun (or verb/aux, verb/adj) homographs surviving in AMBIGUOUS_WORDS
# that are plausible content words for our sentence shapes (drop pure
# function-word ambiguity: to/that/her/up/down/out -- those aren't the
# "noun/verb homograph lattice" the task names).
HOMOGRAPHS = [w for w, tags in AMBIGUOUS_WORDS.items()
              if Tag.NOUN in tags and (Tag.VERB in tags or Tag.AUX in tags)
              and w not in ("be", "am", "is", "are", "was", "were", "been", "being",
                            "have", "has", "had", "do", "does", "did")]

# Curriculum sentence SHAPES (verbatim from curriculum2.TEMPLATES / episode.py
# transfer verbs / the sense-binding cue template), generalized to a
# {slot}-format so a homograph can be dropped into whichever open content
# slot the shape has (subject-place "MOVE"/"PLACE" shapes only take a place
# noun; transfer shapes take an object noun).
PLACE_SHAPES = [
    "{n} is in the {p} .", "{n} is now in the {p} .", "{n} stayed in the {p} .",
    "{n} is at the {p} .", "{n} is currently in the {p} .", "{n} is standing in the {p} .",
    "{n} is inside the {p} .", "{n} is by the {p} .", "{n} is near the {p} .",
    "{n} sat in the {p} .", "{n} is located in the {p} .", "{n} can be found in the {p} .",
]
MOVE_SHAPES = [
    "{n} moved to the {p} .", "{n} walked to the {p} .", "{n} moved into the {p} .",
    "{n} went to the {p} .", "{n} returned to the {p} .", "{n} headed to the {p} .",
    "{n} came to the {p} .", "{n} entered the {p} .",
]
TRANSFER_SHAPES = [
    "{giver} gave the {obj} to {receiver} .", "{giver} handed the {obj} to {receiver} .",
    "{giver} passed the {obj} to {receiver} .", "{giver} took the {obj} to {receiver} .",
]
FIND_SHAPE = "{pronoun} found the {obj} ."
CUE_SHAPE = "{n} went to the {p} ."

# Hand-written candidate shapes probing genuine STRUCTURAL (not just lexical)
# ambiguity: a homograph placed where BOTH its noun and verb/aux readings
# could plausibly complete a legal clause differently. Built only from
# words already in the tagger's vocabulary (WORD_TAG_DICT / AMBIGUOUS_WORDS)
# so failures are about grammar coverage, not missing-word parse failures.
HAND_SHAPES = [
    "{n} can {v} .",                       # can: AUX(modal) vs NOUN
    "{n} will {v} .",                       # will: AUX(modal) vs NOUN
    "{n} saw the {noun} .",                 # saw: VERB(irregular) vs NOUN(tool)
    "{n} watched the {h} .",                # watched(VERB) + h: NOUN vs VERB slot
    "the {h} is in the {p} .",              # h as SUBJECT: NOUN vs VERB
    "the {h} can {v} .",                    # h SUBJECT + can + verb
    "{n} left the {p} .",                   # left: VERB(irregular) vs ADJ/NOUN
    "{n} can {h} .",                        # can (AUX) + h (VERB or NOUN?)
]


def _clause_signature(hyp) -> frozenset:
    """The ROLE-LEVEL reading a hypothesis asserts: {(predicate, relation,
    arg_token)} across every clause -- exactly the "which (entity, relation)
    is even worth querying" content dev/TRACK_C_DESIGN.md Sec 1.10 says is
    PART OF what a hypothesis asserts. Two hypotheses with different node
    objects/edge sets but the SAME signature are semantically the same
    reading for our purposes (e.g. a stray NIL-node variant) -- not the
    "genuinely different reading" K>1 is supposed to mean."""
    graph = hypothesis_to_graph(hyp)
    clauses, _links = extract_discourse(graph)
    sig = set()
    for cl in clauses:
        for rel, arg in cl.args:
            sig.add(((cl.predicate or "").lower(), rel, (arg.token or "").lower()))
    return frozenset(sig)


def _survey_one(parser: QuantumParser, sentence: str):
    try:
        words = tag_sentence(sentence)
        chart = parser.parse(words)
    except Exception as exc:  # pragma: no cover -- diagnostic only
        return {"sentence": sentence, "error": str(exc)}
    n = len(chart.hypotheses)
    top = chart.hypotheses[0].score if n else None
    second = chart.hypotheses[1].score if n > 1 else None
    margin = (top - second) if (top is not None and second is not None) else None
    genuine = False
    sig0 = sig1 = None
    if n > 1:
        try:
            sig0 = _clause_signature(chart.hypotheses[0])
            sig1 = _clause_signature(chart.hypotheses[1])
            genuine = sig0 != sig1 and bool(sig0) and bool(sig1)
        except Exception:  # pragma: no cover -- diagnostic only
            genuine = False
    return {"sentence": sentence, "n_hyp": n, "top_score": top,
            "second_score": second, "margin": margin, "genuine": genuine,
            "sig0": sig0, "sig1": sig1}


def _has_subject(sig) -> bool:
    return any(rel == "SUBJECT" for _p, rel, _a in sig)


def main() -> None:
    parser = QuantumParser(GRAMMAR)

    candidates = []   # [(sentence, shape_label)]
    for shape in PLACE_SHAPES + MOVE_SHAPES:
        for n in NAMES[:2]:
            for p in [*PLACES[:2], *HOMOGRAPHS]:
                candidates.append((shape.format(n=n, p=p), f"CTX[{shape}]"))
    for shape in TRANSFER_SHAPES:
        for obj in HOMOGRAPHS:
            candidates.append((shape.format(giver="mary", obj=obj, receiver="john"),
                                f"TRANSFER[{shape}]"))
    for pr in ("she", "he", "it", "they"):
        for obj in HOMOGRAPHS:
            candidates.append((FIND_SHAPE.format(pronoun=pr, obj=obj), "PRONOUN_FIND"))
    for hand in HAND_SHAPES:
        for h in HOMOGRAPHS:
            for extra in HOMOGRAPHS[:6]:
                try:
                    s = hand.format(n="mary", p="garden", h=h, v=extra, noun="shed")
                    candidates.append((s, f"HAND[{hand}]"))
                except (KeyError, IndexError):
                    pass

    seen = set()
    dedup_candidates = []
    for s, shape in candidates:
        if s in seen:
            continue
        seen.add(s)
        dedup_candidates.append((s, shape))
    print(f"probing {len(dedup_candidates)} candidate sentences across "
          f"{len(set(shape for _s, shape in dedup_candidates))} shapes ...")

    non_equiv = []       # n_hyp > 1 (structurally non-equivalent per is_equivalent)
    genuine = []          # ALSO differ in extracted (predicate, relation, arg) signature
    both_complete = []    # genuine AND both readings have a real SUBJECT (not a broken parse)
    errors = 0
    single = 0
    for sent, shape in dedup_candidates:
        r = _survey_one(parser, sent)
        r["shape"] = shape
        if "error" in r:
            errors += 1
            continue
        if r["n_hyp"] > 1 and r["margin"] is not None:
            non_equiv.append(r)
            if r["genuine"]:
                genuine.append(r)
                if _has_subject(r["sig0"]) and _has_subject(r["sig1"]):
                    both_complete.append(r)
        else:
            single += 1

    print(f"\n{len(dedup_candidates)} candidates probed: {single} single-reading, "
          f"{errors} parse errors, {len(non_equiv)} structurally-non-equivalent "
          f"top-2 (is_equivalent dedup already applied), {len(genuine)} GENUINE "
          f"(role-signature differs), {len(both_complete)} BOTH-COMPLETE (genuine "
          f"AND both readings carry their own SUBJECT -- neither is a broken/"
          f"degenerate parse) -- this last number is the real survey headline.\n")

    # --- Per-shape summary: one row per sentence SHAPE (not per instantiated
    # sentence), the "probe a dozen shapes" deliverable. ---
    by_shape: Dict[str, List[dict]] = {}
    for r in both_complete:
        by_shape.setdefault(r["shape"], []).append(r)
    print(f"{'shape':<28} {'n hits':>6} {'min margin':>11} {'max margin':>11}  example")
    for shape, rows in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
        margins = [r["margin"] for r in rows]
        ex = min(rows, key=lambda r: r["margin"])
        print(f"{shape:<28} {len(rows):>6} {min(margins):>11.4f} {max(margins):>11.4f}  "
              f"{ex['sentence']!r}")
        d0 = sorted(ex["sig0"] - ex["sig1"])
        d1 = sorted(ex["sig1"] - ex["sig0"])
        print(f"    reading1-only: {d0}")
        print(f"    reading2-only: {d1}")

    # Margin histogram (both-complete cases only) -- does margin correlate
    # with anything, or is it a flat sea of ties (M55a survey question)?
    from collections import Counter
    buckets = Counter()
    for r in both_complete:
        m = r["margin"]
        buckets["0.000 (tie)" if m == 0 else "0.000-0.05" if m < 0.05 else
                "0.05-0.15" if m < 0.15 else ">0.15"] += 1
    print("\nmargin histogram (both-complete cases):", dict(buckets))


if __name__ == "__main__":
    main()
