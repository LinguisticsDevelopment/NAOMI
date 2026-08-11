"""Probe: the Spanish Freeze Test (dev/ROADMAP_LONG_TERM.md), perception side.

Charter (the roadmap's own words): sense signatures are keyed to Princeton
SYNSET IDs, not English strings (OMW links Spanish lemmas to the same
synsets -- "gato" -> cat.n.01 -> the SAME vector). A translated episode
should therefore produce a near-identical clause stream at the membrane. If
perception does its job, Spanish is invisible to a frozen mind. This script
does NOT touch the reactor (no training, no checkpoint load -- see its
"Track A checkpoint" section for why) -- it measures the mechanism directly:
does the (entity, relation, value) clause stream quantum_parser + clause.py
extract from a Spanish sentence match the stream extracted from its English
counterpart?

Three parts, run in order (each prints its own report section):

1. **Coverage** -- OMW-es lemma coverage for the curriculum vocabulary (the
   base 6 places, the 60-noun VOCAB_SCALE_PLACES pool, the 6 transfer
   objects, the 31 ambiguity-family synsets/62 senses). Reported per word;
   words OMW has no Spanish lemma for are reported MISSING, never faked.
2. **Templates** -- every Spanish surface template (curriculum2.TEMPLATES_ES,
   TRANSFER_TEMPLATES_ES, PRONOUN_FIND_TEMPLATE_ES, QUESTION_TEMPLATE_ES)
   parse-checked through the real grammar/tagger, exact-role gate (same
   contract as verify_templates()).
3. **Stream equivalence** -- THE deliverable: >=200 parallel English/Spanish
   episode pairs (curriculum2.generate_freeze_pairs), each parsed through its
   own language's full ParserInputEncoder -> clause.extract_discourse ->
   clause.clause_tpr, comparing entity/relation/value per triple. Reports
   entity exact-match rate, relation exact-match rate, and value cosine
   statistics (mean/median/quartiles), broken out by shape (PLACE/MOVE/
   TRANSFER) and overall, plus a leak audit: which pairs' values diverge and
   why (MFS-ordering mismatch between English WordNet and OMW-es, quantified
   via the coverage section's per-word synset comparison).

Track A checkpoint check (task-mandated): scanned `runs/` (only `.log` files,
no `.pt`/`.ckpt` anywhere in the repo) and the M53/M55-gate training scripts
`scripts/train_resolver.py` / `scripts/train_clause.py` -- neither calls
`torch.save`/`.state_dict()` anywhere; they train and report metrics in-
process only, nothing is persisted. (A few UNRELATED scripts elsewhere --
`train_drive.py`, `train_clause_psyche.py`, `run_mind_loops.py` -- do save
checkpoints for their own separate experiments, but none of them is the
Track A resolver/clause-reactor run the M55 gate refers to.) So there is no
trained Track A checkpoint to freeze-load. This is reported, not worked
around: the frozen-reactor comparison this task's step 5 asks for (same
weights, English vs Spanish input) needs a checkpoint-save flag added to
train_resolver.py/train_clause.py first -- out of scope here (would touch
training code, and the task is explicit that no training may run in this
worktree). The next-best comparison -- stream equivalence, which is the
freeze test's actual MECHANISM (if the streams match, a frozen reactor of
any weights provably cannot tell the difference) -- is what this script
measures.

Run:  python scripts/probe_spanish_freeze.py [--n 240] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from nsm_ct.episode import _PLACES, _AMBIGUITY_FAMILIES  # noqa: E402
from nsm_ct.curriculum2 import (  # noqa: E402
    VOCAB_SCALE_PLACES, _TRANSFER_OBJECTS, _PLACES_ES, _TRANSFER_OBJECTS_ES,
    TEMPLATES_ES, TRANSFER_TEMPLATES_ES, QUESTION_TEMPLATE_ES,
    verify_templates_es, verify_transfer_templates_es, verify_pronoun_templates_es,
    generate_freeze_pairs,
)
from nsm_ct.clause import extract_discourse, clause_tpr, EntityTracker  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.meaning_es import SpanishMeaningResolver  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402


# ===========================================================================
# 1. Coverage
# ===========================================================================

def _es_lemmas_for_english_word(word: str) -> Tuple[Optional[str], List[str]]:
    """(english_mfs_synset_name_or_None, [spanish lemma names for that synset])."""
    from nltk.corpus import wordnet as wn

    syns = wn.synsets(word)
    if not syns:
        return None, []
    mfs = syns[0]
    es = [lem.name() for lem in mfs.lemmas(lang="spa")]
    return mfs.name(), es


def report_coverage() -> Dict[str, object]:
    print("\n" + "=" * 78)
    print("1. SPANISH SENSE MAPPING COVERAGE (OMW-es, wn.lemmas(lang='spa'))")
    print("=" * 78)

    groups = {
        "base _PLACES (6)": list(_PLACES),
        "VOCAB_SCALE_PLACES (%d)" % len(VOCAB_SCALE_PLACES): list(VOCAB_SCALE_PLACES),
        "_TRANSFER_OBJECTS (%d)" % len(_TRANSFER_OBJECTS): list(_TRANSFER_OBJECTS),
    }
    summary: Dict[str, Dict[str, object]] = {}
    for gname, words in groups.items():
        missing = []
        rows = []
        for w in words:
            mfs, es = _es_lemmas_for_english_word(w)
            rows.append((w, mfs, es))
            if not es:
                missing.append(w)
        covered = len(words) - len(missing)
        print(f"\n-- {gname}: {covered}/{len(words)} covered --")
        for w, mfs, es in rows:
            tag = "OK " if es else "MISS"
            print(f"  [{tag}] {w:12s} mfs={mfs or '(none)':20s} es={es[:3]}")
        summary[gname] = {"covered": covered, "total": len(words), "missing": missing}

    # Ambiguity families: per-sense coverage (62 synsets across 31 families).
    print(f"\n-- _AMBIGUITY_FAMILIES ({len(_AMBIGUITY_FAMILIES)} families, "
          f"{2*len(_AMBIGUITY_FAMILIES)} senses) --")
    from nltk.corpus import wordnet as wn
    fam_covered = 0
    fam_total = 0
    fam_rows = []
    for word, fam in _AMBIGUITY_FAMILIES.items():
        for skey, sdata in fam["senses"].items():
            sid = sdata["synset"]
            try:
                syn = wn.synset(sid)
                es = [lem.name() for lem in syn.lemmas(lang="spa")]
            except Exception:
                es = []
            fam_total += 1
            fam_covered += int(bool(es))
            fam_rows.append((word, skey, sid, es))
    for word, skey, sid, es in fam_rows:
        tag = "OK " if es else "MISS"
        print(f"  [{tag}] {word:10s} {skey} {sid:32s} es={es[:2]}")
    print(f"\n  ambiguity-family sense coverage: {fam_covered}/{fam_total}")
    summary["_AMBIGUITY_FAMILIES senses"] = {"covered": fam_covered, "total": fam_total}

    return summary


# ===========================================================================
# 2. Templates
# ===========================================================================

def report_templates() -> Dict[str, object]:
    print("\n" + "=" * 78)
    print("2. SPANISH TEMPLATE VERIFICATION (real grammar + tagger, exact-role gate)")
    print("=" * 78)

    place_move = verify_templates_es()
    transfer = verify_transfer_templates_es()
    pronoun = verify_pronoun_templates_es()

    print("\n-- PLACE / MOVE (TEMPLATES_ES) --")
    for sent, ok in place_move.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {sent}")

    print("\n-- TRANSFER (TAKE/SOURCE variant; GIVE/dative excluded, see curriculum2.py) --")
    for k, v in transfer.items():
        print(f"  [{'OK' if v['ok'] else 'FAIL'}] {k}: {v['sentence']}  roles={v['roles']}")

    print("\n-- PRONOUN (parse-layer SUBJECT check only; entity-extraction BLOCKED, see below) --")
    for pr, v in pronoun.items():
        print(f"  [{'OK' if v['subject_ok'] else 'FAIL'}] {v['sentence']}  subject_ok={v['subject_ok']}")

    print("\n-- QUESTION (wh-inversion; expected to fail, see curriculum2.py note) --")
    place = _PLACES_ES["garden"]
    q_sent = QUESTION_TEMPLATE_ES.format(p=place["det"])
    tok = SimpleTokenizer.build([q_sent], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="es")
    graph = parser._parse_graph(q_sent)
    clauses, _links = extract_discourse(graph)
    q_ok = len(clauses) > 0
    print(f"  [{'OK' if q_ok else 'FAIL'}] {q_sent}  clauses_extracted={len(clauses)}")
    if not q_ok:
        print("    (measured, not assumed: 0 clauses -- 'está' has only the VERB tag, so "
              "question1's MODIFIER+PROGRESSIVE anchor never fires; 'donde' attaches as a "
              "plain SPECIFIER and 'el jardin' is mis-consumed as a bare OBJECT, no SUBJECT "
              "edge at all. See curriculum2.py's module note for the fix and why it wasn't done.)")

    n_ok = sum(place_move.values()) + sum(v["ok"] for v in transfer.values())
    n_total = len(place_move) + len(transfer)
    print(f"\n  role-exact pass rate (PLACE/MOVE + TRANSFER, the templates used by the "
          f"stream-equivalence gate below): {n_ok}/{n_total}")

    return {"place_move": place_move, "transfer": transfer, "pronoun": pronoun, "question_ok": q_ok}


# ===========================================================================
# 3. Stream equivalence
# ===========================================================================

def _clause_streams(sentences: List[str], parser: ParserInputEncoder, codec: TPRCodec,
                     resolver) -> List[Tuple[Optional[str], Dict[str, Tuple[str, np.ndarray]]]]:
    """Parse ``sentences`` (already-merged single-sentence context) and return
    one ``(subject, {relation: (arg_token, filler_vec)})`` per clause. Filler
    vectors are the SAME ones clause_tpr binds into the TPR matrix (entity
    var: atoms for entity args, content vectors for everything else) -- the
    actual thing a frozen reactor would see.
    """
    out = []
    for sent in sentences:
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        for cl in clauses:
            tracker = EntityTracker()
            _m, triples = clause_tpr(cl, codec, resolver, tracker)
            per_rel: Dict[str, Tuple[str, np.ndarray]] = {}
            subject = None
            for rel, arg in cl.args:
                if rel == "SUBJECT":
                    subject = (arg.token or "").lower()
            for subj, rel, val in triples:
                # find the matching arg token for this relation (first match;
                # templates never repeat a relation within one clause)
                tok = next((a.token for r, a in cl.args if r == rel), None)
                per_rel[rel] = ((tok or "").lower(), val)
            out.append((subject, per_rel))
    return out


def report_stream_equivalence(n: int, seed: int) -> Dict[str, object]:
    print("\n" + "=" * 78)
    print(f"3. STREAM EQUIVALENCE ({n} parallel English/Spanish pairs, seed={seed})")
    print("=" * 78)

    pairs = generate_freeze_pairs(n, seed=seed)
    by_shape = Counter(p["shape"] for p in pairs)
    print(f"\n  shapes drawn: {dict(by_shape)}")

    codec = TPRCodec(dim=96)
    en_resolver = NSMMeaningResolver()
    es_resolver = SpanishMeaningResolver()

    tok_en = SimpleTokenizer.build(
        [s for p in pairs for s in p["en"]],
        extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    tok_es = SimpleTokenizer.build(
        [s for p in pairs for s in p["es"]],
        extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser_en = ParserInputEncoder(tok_en, lang="en")
    parser_es = ParserInputEncoder(tok_es, lang="es")

    entity_match = 0
    entity_total = 0
    relation_match = 0
    relation_total = 0
    value_cos: List[float] = []
    value_cos_by_shape: Dict[str, List[float]] = defaultdict(list)
    leaks: List[Dict[str, object]] = []
    no_clause_en = no_clause_es = 0

    for p in pairs:
        en_streams = _clause_streams(p["en"], parser_en, codec, en_resolver)
        es_streams = _clause_streams(p["es"], parser_es, codec, es_resolver)
        if not en_streams:
            no_clause_en += 1
            continue
        if not es_streams:
            no_clause_es += 1
            continue
        en_subj, en_rels = en_streams[0]
        es_subj, es_rels = es_streams[0]

        entity_total += 1
        if en_subj is not None and en_subj == es_subj:
            entity_match += 1

        common_rels = set(en_rels) | set(es_rels)
        for rel in common_rels:
            relation_total += 1
            if rel in en_rels and rel in es_rels:
                relation_match += 1
                en_tok, en_val = en_rels[rel]
                es_tok, es_val = es_rels[rel]
                # SOURCE/etc. entity args: compare the var: atom directly
                # (should be EXACTLY 1.0, same string -> same filler_vec).
                en_n = np.linalg.norm(en_val)
                es_n = np.linalg.norm(es_val)
                if en_n < 1e-8 or es_n < 1e-8:
                    continue
                cos = float(en_val @ es_val / (en_n * es_n))
                value_cos.append(cos)
                value_cos_by_shape[p["shape"]].append(cos)
                if cos < 0.7:
                    leaks.append({
                        "shape": p["shape"], "relation": rel,
                        "en": p["en"], "es": p["es"],
                        "en_token": en_tok, "es_token": es_tok, "cosine": round(cos, 3),
                    })

    print(f"\n  parses with >=1 clause: en {n - no_clause_en}/{n}, es {n - no_clause_es}/{n}")
    print(f"  entity exact-match: {entity_match}/{entity_total} "
          f"({entity_match / entity_total:.3f})" if entity_total else "  entity: n/a")
    print(f"  relation exact-match: {relation_match}/{relation_total} "
          f"({relation_match / relation_total:.3f})" if relation_total else "  relation: n/a")

    if value_cos:
        print(f"\n  value cosine (matched relations, n={len(value_cos)}):")
        print(f"    mean={statistics.mean(value_cos):.3f}  median={statistics.median(value_cos):.3f}  "
              f"min={min(value_cos):.3f}  max={max(value_cos):.3f}")
        quart = statistics.quantiles(value_cos, n=4) if len(value_cos) >= 4 else []
        if quart:
            print(f"    quartiles: {[round(q, 3) for q in quart]}")
        near_1 = sum(1 for c in value_cos if c > 0.999)
        print(f"    exact-match (cos>0.999, i.e. SAME synset -> SAME vector): "
              f"{near_1}/{len(value_cos)} ({near_1/len(value_cos):.3f})")

    print("\n  value cosine BY SHAPE:")
    for shape, cs in value_cos_by_shape.items():
        if cs:
            print(f"    {shape:10s} n={len(cs):4d} mean={statistics.mean(cs):.3f} "
                  f"median={statistics.median(cs):.3f} exact={sum(1 for c in cs if c>0.999)}/{len(cs)}")

    print(f"\n  LEAKS (matched-relation pairs with value cosine < 0.7): {len(leaks)}")
    for leak in leaks[:20]:
        print(f"    [{leak['shape']}/{leak['relation']}] en={leak['en_token']!r} "
              f"es={leak['es_token']!r} cos={leak['cosine']}")
        # root-cause: was this an MFS-ordering mismatch or an OOV word?
        en_mfs, _ = _es_lemmas_for_english_word(leak["en_token"])
        es_syn, _es_lemmas = None, None
        try:
            from nltk.corpus import wordnet as wn
            es_syns = wn.synsets(leak["es_token"], lang="spa")
            es_syn = es_syns[0].name() if es_syns else None
        except Exception:
            pass
        cause = "OOV (no OMW-es entry)" if es_syn is None else (
            "MFS-ordering mismatch" if es_syn != en_mfs else "same synset but low cosine (unexpected)")
        print(f"        en_mfs={en_mfs}  es_mfs={es_syn}  cause={cause}")
    leak_words = sorted({(leak["en_token"], leak["es_token"]) for leak in leaks})
    if len(leaks) > 20:
        print(f"    ... and {len(leaks) - 20} more instances (same small set of leaking "
              f"word pairs recurring across draws): {leak_words}")

    return {
        "n": n, "seed": seed, "shapes": dict(by_shape),
        "no_clause_en": no_clause_en, "no_clause_es": no_clause_es,
        "entity_match": entity_match, "entity_total": entity_total,
        "relation_match": relation_match, "relation_total": relation_total,
        "value_cos_n": len(value_cos),
        "value_cos_mean": statistics.mean(value_cos) if value_cos else None,
        "value_cos_median": statistics.median(value_cos) if value_cos else None,
        "value_cos_exact": sum(1 for c in value_cos if c > 0.999),
        "leaks_n": len(leaks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-coverage", action="store_true")
    args = ap.parse_args()

    if not args.skip_coverage:
        report_coverage()
    report_templates()
    report_stream_equivalence(args.n, args.seed)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
