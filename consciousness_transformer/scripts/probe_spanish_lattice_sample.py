"""Probe: build a small SAMPLE of V2-shaped candidate-lattice records from
Spanish sentences (dev/SPANISH_SWAP_FEASIBILITY.md, Check 1 + Check 2).

This is NOT the production encoder-gold builder (none exists on this branch
for either language -- there is no committed script that turns raw text into
`runs/encoder_gold_v2.jsonl`; only the *consumers*, `scripts/train_encoder.py`
/ `scripts/eval_encoder.py`, and the *format reader*,
`nsm_ct.encoder_model._gold_sites`, exist). It is a minimal, honest
demonstration that the two data prerequisites for a Spanish grammar-swap test
(USVS Spanish sense resolution + quantum_parser Spanish tag+parse) combine
into records matching `dev/ENCODER_IO_CONTRACT_V2.md`'s shape, using the SAME
parser path (`ParserInputEncoder(lang="es")` + `clause.extract_discourse`)
the real gold builder would use.

Check 1's gap (`USVS.senses_of` only indexes ENGLISH lemmas -- see
dev/SPANISH_SWAP_FEASIBILITY.md) is bridged here WITHOUT touching
`nsm_ct/ground/usvs.py` or `nsm_ct/wordnet.py`: `_spanish_lemma_index` below
builds a supplementary Spanish lemma -> sense_id map by calling
`wn.synset(sid).lemmas(lang="spa")` for every sense_id ALREADY in the loaded
USVS artifact (117,659 senses, ~4s). This works because sense signatures are
grounded per SYNSET ID, which is language-independent (that's the whole
Freeze Test premise) -- no re-grounding needed, purely an additive lemma
index. Flagged per the task's "enabling tweak" allowance; it is prototype
code in this probe script, not applied to `src/`.

Sentences are drawn from the ALREADY-VERIFIED `nsm_ct.curriculum2` Spanish
templates (`TEMPLATES_ES`, `TRANSFER_TEMPLATES_ES` -- both pass
`verify_templates_es`/`verify_transfer_templates_es`'s exact-role gate), so
every sentence here is known-parseable before this script even runs it.

Run:  python scripts/probe_spanish_lattice_sample.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nltk.corpus import wordnet as wn  # noqa: E402

from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct.clause import extract_discourse, is_entity  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.curriculum2 import (  # noqa: E402
    _PLACES_ES, _TRANSFER_OBJECTS_ES, TEMPLATES_ES, TRANSFER_TEMPLATES_ES,
)

_NAMES = ["mary", "john", "sandra", "daniel", "bill", "fred"]


# ---------------------------------------------------------------------------
# Check-1 enabling tweak, standalone (see module docstring): additive
# Spanish lemma index over the sense_ids the USVS artifact already grounds.
# ---------------------------------------------------------------------------
def build_spanish_lemma_index(usvs) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for sid in usvs.sense_ids:
        try:
            syn = wn.synset(sid)
        except Exception:
            continue
        for lem in syn.lemmas(lang="spa"):
            idx.setdefault(lem.name().lower(), []).append(sid)
    return idx


def spanish_senses_of(word: str, spa_index: Dict[str, List[str]]) -> List[str]:
    return spa_index.get(word.lower(), [])


# ---------------------------------------------------------------------------
# Sentence generation (from already-verified curriculum2 Spanish templates)
# ---------------------------------------------------------------------------
def gen_sentences(n: int) -> List[str]:
    sents: List[str] = []
    places = list(_PLACES_ES.keys())
    objs = list(_TRANSFER_OBJECTS_ES.keys())
    i = 0
    while len(sents) < n:
        name = _NAMES[i % len(_NAMES)]
        place = _PLACES_ES[places[i % len(places)]]
        kind = ("PLACE", "MOVE", "TRANSFER")[i % 3]
        if kind == "PLACE":
            t = TEMPLATES_ES["A"]["PLACE"][i % len(TEMPLATES_ES["A"]["PLACE"])]
            sents.append(t.format(n=name, p=place["det"]))
        elif kind == "MOVE":
            t = TEMPLATES_ES["A"]["MOVE"][i % len(TEMPLATES_ES["A"]["MOVE"])]
            sents.append(t.format(n=name, p=place["a_det"]))
        else:
            taker = name
            source = _NAMES[(i + 1) % len(_NAMES)]
            obj = _TRANSFER_OBJECTS_ES[objs[i % len(objs)]]
            sents.append(TRANSFER_TEMPLATES_ES["TAKE"].format(
                taker=taker, obj=obj["det"], source=source, place=place["det"]))
        i += 1
    return sents


# ---------------------------------------------------------------------------
# V2 record construction
# ---------------------------------------------------------------------------
def _token_indices(tokens: List[str]) -> Dict[str, List[int]]:
    pos: Dict[str, List[int]] = {}
    for i, t in enumerate(tokens):
        pos.setdefault(t.lower(), []).append(i)
    return pos


def _take_index(avail: Dict[str, List[int]], word: str) -> Optional[int]:
    lst = avail.get((word or "").lower())
    if not lst:
        return None
    return lst.pop(0)


def _grounding_for(word: str, spa_index: Dict[str, List[str]],
                    tsc: Dict[int, dict], tidx: Optional[int]) -> dict:
    if is_entity(word or ""):
        return {"type": "entity", "candidates": None, "retrieval": None, "prime": None}
    cands = spanish_senses_of(word or "", spa_index)
    if cands and tidx is not None:
        tsc[tidx] = {
            "index": tidx, "token": word, "sense_candidates": cands,
            "chosen_sense": cands[0],
        }
        return {"type": "sense", "candidates": None,
                "retrieval": {"source": "lexicon", "method": "senses_of", "ref": None},
                "prime": None}
    # content token the (Spanish) lemma index does not cover -> entity fallback
    # (contract's documented behavior for an uncovered content word).
    return {"type": "entity", "candidates": None, "retrieval": None, "prime": None}


def build_record(sentence: str, parser: ParserInputEncoder, spa_index: Dict[str, List[str]]
                  ) -> Optional[dict]:
    words = parser._tag(sentence)
    tokens = [w.text for w in words]
    pos = [w.pos.name for w in words]
    graph = parser._parse_graph(sentence)
    if graph is None:
        return None
    clauses, links = extract_discourse(graph)
    if not clauses:
        return None

    avail = _token_indices(tokens)
    tsc: Dict[int, dict] = {}
    out_clauses = []
    for cl in clauses:
        pred_tidx = _take_index(avail, cl.predicate)
        pred_g = _grounding_for(cl.predicate, spa_index, tsc, pred_tidx)
        roles = []
        for rel, node in cl.args:
            word = node.token
            tidx = _take_index(avail, word)
            g = _grounding_for(word, spa_index, tsc, tidx)
            roles.append({
                "relation": rel, "word": word, "token_index": tidx,
                "is_entity": is_entity(word or ""), "grounding": g,
            })
        out_clauses.append({
            "predicate": cl.predicate, "predicate_grounding": pred_g,
            "is_question": cl.is_question, "utterance_kind": "proposition",
            "roles": roles,
        })

    dlinks = [{"coordinator": l.coordinator, "prime": l.prime,
               "clause_i": l.clause_i, "clause_j": l.clause_j} for l in links]

    record = {
        "text": sentence,
        "tokens": tokens,
        "pos": pos,
        "lattice": {"trees": [{"clauses": out_clauses}],
                    "discourse_links_per_tree": [dlinks]},
        "token_sense_candidates": [tsc[i] for i in sorted(tsc)],
    }
    return record


def main() -> None:
    t0 = time.time()
    usvs = load_usvs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "usvs"))
    spa_index = build_spanish_lemma_index(usvs)
    print(f"[{time.time()-t0:.1f}s] spanish lemma index: {len(spa_index)} lemmas "
          f"over {len(usvs.sense_ids)} USVS senses")

    sentences = gen_sentences(20)
    tok = SimpleTokenizer.build(sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="es")
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable -- aborting sample generation.")
        return

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs",
                             "spanish_gold_probe.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_ok = 0
    n_sense_slots = 0
    n_entity_slots = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            rec = build_record(s, parser, spa_index)
            if rec is None:
                print(f"  [FAIL] {s!r} -- no clauses extracted")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            for cl in rec["lattice"]["trees"][0]["clauses"]:
                gtypes = [cl["predicate_grounding"]["type"]] + [r["grounding"]["type"] for r in cl["roles"]]
                n_sense_slots += gtypes.count("sense")
                n_entity_slots += gtypes.count("entity")

    print(f"[{time.time()-t0:.1f}s] wrote {n_ok}/{len(sentences)} records -> {out_path}")
    print(f"  sense-grounded slots: {n_sense_slots}  entity-grounded slots: {n_entity_slots}")

    # shape sanity: run the real reader (nsm_ct.encoder_model._gold_sites) over
    # every record written, exactly like the trainer would.
    from nsm_ct import encoder_model as em
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sense_sites, slot_sites, trees = em._gold_sites(rec)
            assert len(trees) == 1 and len(trees[0]) == len(rec["lattice"]["trees"][0]["clauses"])
    print("  shape check: every record parses through encoder_model._gold_sites() OK")


if __name__ == "__main__":
    main()
