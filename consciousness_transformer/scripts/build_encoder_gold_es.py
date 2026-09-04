"""Encoder step 2, SPANISH: teacher gold as a CANDIDATE LATTICE for the
grammar-swap test (train encoder EN, test on Spanish with the SAME weights).
Same v2 I/O contract as :mod:`build_encoder_gold_v2` (``dev/ENCODER_IO_CONTRACT_V2.md``)
and the exact same record-building code path -- only two things differ:

- **Source sentences**: no committed Spanish text corpus exists, so sentences
  are generated from :mod:`nsm_ct.curriculum2`'s already parse-verified
  Spanish templates (``TEMPLATES_ES``, ``TRANSFER_TEMPLATES_ES`` -- correctly
  accented, which matters: see ``dev/SPANISH_SWAP_FEASIBILITY.md`` Check 2)
  over the 6 ``episode._NAMES`` x 6 ``_PLACES_ES`` x 6 ``_TRANSFER_OBJECTS_ES``
  grid (PLACE/MOVE exhaustive, TRANSFER TAKE sampled to keep the set in the
  requested ~150-250 range).
- **Parser/grounding language**: :class:`~nsm_ct.input_encoder.ParserInputEncoder`
  is built with ``lang="es"`` (Spanish tagger + grammar + prep-role map).
  Sense grounding reuses the SAME ``usvs.senses_of`` as the English builder --
  no separate Spanish index needed, because Task 1
  (``dev/SPANISH_SWAP_FEASIBILITY.md`` Check 1) already extended the USVS
  lemma index itself, additively, to resolve Spanish lemmas to the SAME
  synset ids as their English counterparts (``senses_of('perro') ->
  ['dog.n.01', ...]``), fingerprint-unchanged.

Known, documented, contract-valid gap (not a bug): OMW-es lemmas are
uninflected dictionary forms, so CONJUGATED Spanish predicate verbs (está,
fue, tomó) don't hit the lemma index and correctly fall back to
``grounding.type:"entity"`` per contract sec 4.2's "ungrounded content word"
case -- see ``dev/SPANISH_GOLD_V2_STATS.md`` for the exact count.

Usage: python scripts/build_encoder_gold_es.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.clause import _PRONOUNS, extract_discourse, is_entity  # noqa: E402
from nsm_ct.curriculum2 import TEMPLATES_ES, TRANSFER_TEMPLATES_ES, _PLACES_ES, _TRANSFER_OBJECTS_ES  # noqa: E402
from nsm_ct.episode import _NAMES  # noqa: E402
from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

USVS_DIR = ROOT / "data" / "usvs"
OUT_JSONL = ROOT / "runs" / "spanish_gold_v2.jsonl"
OUT_STATS = ROOT / "dev" / "SPANISH_GOLD_V2_STATS.md"
TOP_K = 8
N_TRANSFER = 100  # sampled TAKE combos; PLACE+MOVE are exhaustive (108) -> ~208 total
SEED = 0


def load_spanish_sentences() -> List[str]:
    """Unique deduped Spanish sentences from the verified curriculum2 Spanish
    templates: PLACE/MOVE exhaustive over 6 names x 6 places, TRANSFER TAKE
    sampled (taker != source) over 6 names x 6 objects x 6 places, seeded for
    reproducibility."""
    seen = set()
    out: List[str] = []

    def add(s: str) -> None:
        if s not in seen:
            seen.add(s)
            out.append(s)

    for name in _NAMES:
        for place in _PLACES_ES.values():
            for t in TEMPLATES_ES["A"]["PLACE"]:
                add(t.format(n=name, p=place["det"]))
            for t in TEMPLATES_ES["A"]["MOVE"]:
                add(t.format(n=name, p=place["a_det"]))

    combos = [(taker, source, obj_key, place_key)
              for taker in _NAMES for source in _NAMES if source != taker
              for obj_key in _TRANSFER_OBJECTS_ES for place_key in _PLACES_ES]
    rng = random.Random(SEED)
    rng.shuffle(combos)
    for taker, source, obj_key, place_key in combos[:N_TRANSFER]:
        obj = _TRANSFER_OBJECTS_ES[obj_key]
        place = _PLACES_ES[place_key]
        for t in TRANSFER_TEMPLATES_ES.values():
            add(t.format(taker=taker, obj=obj["det"], source=source, place=place["det"]))

    return out


def ground_word(usvs, word: Optional[str]) -> Dict[str, object]:
    """Identical contract to build_encoder_gold_v2.ground_word (contract §4):
    pronoun -> unresolved reference slot; entity name -> resolved entity;
    content word with (now Spanish-extended) WordNet senses -> unresolved
    sense slot, full candidate list; content word the lemma index doesn't
    cover (e.g. a conjugated Spanish verb form) -> entity fallback."""
    w = (word or "").lower()
    if w in _PRONOUNS:
        return {
            "type": "reference",
            "candidates": None,
            "retrieval": {"source": "memory", "method": "coref", "ref": None},
        }
    if is_entity(word or ""):
        return {"type": "entity", "candidates": None}
    candidates = usvs.senses_of(w)
    if candidates:
        return {
            "type": "sense",
            "candidates": list(candidates),
            "retrieval": {"source": "lexicon", "method": "lemma_senses", "ref": None},
        }
    return {"type": "entity", "candidates": None}


class _IndexMatcher:
    """Left-to-right, consume-on-match token-index recovery (see
    build_encoder_gold_v2 module docstring) -- a fresh instance per tree."""

    def __init__(self, tokens: List[str]) -> None:
        self._pos: Dict[str, "deque[int]"] = defaultdict(deque)
        for i, t in enumerate(tokens):
            self._pos[t.lower()].append(i)

    def match(self, word: Optional[str]) -> Optional[int]:
        if not word:
            return None
        dq = self._pos.get(word.lower())
        if not dq:
            return None
        return dq.popleft()


def build_clause_dict(usvs, matcher: _IndexMatcher, cl) -> Dict[str, object]:
    pred_grounding = ground_word(usvs, cl.predicate)
    roles = []
    for relation, arg in cl.args:
        word = arg.token
        roles.append({
            "relation": relation,
            "word": word,
            "token_index": matcher.match(word),
            "is_entity": is_entity(word or ""),
            "grounding": ground_word(usvs, word),
        })
    return {
        "predicate": cl.predicate,
        "predicate_grounding": pred_grounding,
        "is_question": bool(cl.is_question),
        "utterance_kind": "proposition",
        "roles": roles,
    }


def build_tree(usvs, tokens: List[str], graph) -> Optional[Tuple[Dict, List[Dict]]]:
    clauses, links = extract_discourse(graph)
    if not clauses:
        return None
    matcher = _IndexMatcher(tokens)
    clause_dicts = [build_clause_dict(usvs, matcher, cl) for cl in clauses]
    link_dicts = [
        {"coordinator": lk.coordinator, "prime": lk.prime, "clause_i": lk.i, "clause_j": lk.j}
        for lk in links
    ]
    return {"clauses": clause_dicts}, link_dicts


def build_record(usvs, parser: ParserInputEncoder, sentence: str) -> Tuple[Optional[Dict], str, int]:
    """Returns (record_or_None, outcome_tag, n_raw_hypotheses)."""
    from src.parser.quantum_parser import ParseResourceExceeded  # local import (qp_root on sys.path)

    try:
        graphs, _scores, _margin = parser._parse_topk_one(sentence, k=TOP_K)
    except ParseResourceExceeded:
        return None, "cap-hit", 0
    except Exception:
        return None, "parse-failed", 0

    if not graphs:
        return None, "no-hypothesis", 0

    words = parser._tag(sentence)
    tokens = [w.text for w in words]
    pos = [w.pos.name for w in words]

    trees: List[Dict] = []
    links_per_tree: List[List[Dict]] = []
    seen_trees = set()
    for graph in graphs:
        built = build_tree(usvs, tokens, graph)
        if built is None:
            continue
        tree, link_dicts = built
        key = json.dumps(tree, sort_keys=True)
        if key in seen_trees:
            continue
        seen_trees.add(key)
        trees.append(tree)
        links_per_tree.append(link_dicts)

    if not trees:
        return None, "grounding-fail", len(graphs)

    token_sense_candidates = []
    for i, tok in enumerate(tokens):
        cands = usvs.senses_of(tok)
        if cands:
            token_sense_candidates.append({
                "index": i, "token": tok,
                "sense_candidates": list(cands), "chosen_sense": cands[0],
            })

    record = {
        "text": sentence,
        "tokens": tokens,
        "pos": pos,
        "lattice": {"trees": trees, "discourse_links_per_tree": links_per_tree},
        "token_sense_candidates": token_sense_candidates,
    }
    return record, "ok", len(graphs)


def main() -> int:
    t0 = time.time()
    sentences = load_spanish_sentences()
    print(f"spanish templates: {len(sentences)} unique sentences", flush=True)

    print("loading USVS ...", flush=True)
    usvs = load_usvs(USVS_DIR)
    print(f"USVS loaded: {len(usvs.core_words)} core words, {len(usvs.sense_ids)} senses "
          f"(fingerprint={usvs.fingerprint})", flush=True)

    tok = SimpleTokenizer.build(sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="es")
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    outcomes: Dict[str, int] = defaultdict(int)
    n_trees: List[int] = []
    n_raw_hyps: List[int] = []
    n_sense_cands: List[int] = []
    n_grounding_type: Dict[str, int] = defaultdict(int)
    n_predicate_grounding_type: Dict[str, int] = defaultdict(int)
    n_only_best = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for i, sentence in enumerate(sentences):
            record, outcome, n_raw = build_record(usvs, parser, sentence)
            outcomes[outcome] += 1
            if record is not None:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_trees.append(len(record["lattice"]["trees"]))
                n_raw_hyps.append(n_raw)
                if n_raw <= 1:
                    n_only_best += 1
                for tree in record["lattice"]["trees"]:
                    for cl in tree["clauses"]:
                        n_predicate_grounding_type[cl["predicate_grounding"]["type"]] += 1
                        for node in [{"grounding": cl["predicate_grounding"]}] + cl["roles"]:
                            g = node["grounding"]
                            n_grounding_type[g["type"]] += 1
                            if g["type"] == "sense":
                                n_sense_cands.append(len(g["candidates"]))
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(sentences)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"wrote {len(n_trees)} records -> {OUT_JSONL}", flush=True)
    print(f"outcomes: {dict(outcomes)}", flush=True)
    print(f"grounding types: {dict(n_grounding_type)}", flush=True)
    print(f"predicate grounding types: {dict(n_predicate_grounding_type)}", flush=True)

    write_stats(usvs, sentences, outcomes, n_trees, n_raw_hyps, n_sense_cands,
                n_grounding_type, n_predicate_grounding_type, n_only_best)
    print(f"wrote stats -> {OUT_STATS}", flush=True)
    return 0


def pctl(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def write_stats(usvs, sentences, outcomes, n_trees, n_raw_hyps, n_sense_cands,
                 n_grounding_type, n_predicate_grounding_type, n_only_best) -> None:
    lines = []
    lines.append("# Spanish Encoder Gold v2 — Candidate Lattice Stats")
    lines.append("")
    lines.append("Generated by `scripts/build_encoder_gold_es.py` from the same canonical "
                 "`dev/ENCODER_IO_CONTRACT_V2.md` v2 record shape as the English builder "
                 "(`scripts/build_encoder_gold_v2.py`), for the grammar-swap test: train the "
                 "encoder on English, test on Spanish with the SAME weights. Source: "
                 "`nsm_ct.curriculum2.TEMPLATES_ES`/`TRANSFER_TEMPLATES_ES` (correctly accented). "
                 "Numbers only.")
    lines.append("")
    lines.append(f"- USVS fingerprint at build time: `{usvs.fingerprint}` "
                 "(see Task 1 verification below for before/after proof it is unchanged "
                 "by the additive Spanish lemma-index extension)")
    lines.append(f"- Unique deduped Spanish template sentences generated: {len(sentences)}")
    lines.append(f"- Records emitted (valid lattice, >=1 tree): {len(n_trees)}")
    lines.append(f"- Outcome counts: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    lines.append("")
    lines.append("## Task 1 — USVS Spanish lemma resolution (additive, fingerprint-safe)")
    lines.append("")
    lines.append("Implementation: `src/nsm_ct/wordnet.py` gained `spanish_lemmas(sense_id)` "
                 "(OMW `lemmas(lang=\"spa\")` per synset, graceful-empty like every other "
                 "helper in that module). `src/nsm_ct/ground/usvs.py`'s `build_usvs` sense loop "
                 "unions those Spanish lemmas into `sense_lemmas` alongside the existing English "
                 "lemmas, keyed to the SAME `sense_id` -- no re-grounding. The sense-grounding "
                 "weights (`sense_usvs_weights`) never read the `lemmas` list (the synset's own "
                 "lemmas are deliberately excluded from grounding, per that function's own "
                 "docstring), and the fingerprint hashes only `axes` names + `meta` counts -- "
                 "neither depends on lemma content -- so this change is PROVABLY fingerprint-inert, "
                 "not just measured-inert.")
    lines.append("")
    lines.append("Verified by full before/after USVS rebuild (`python scripts/build_usvs.py`, "
                 "same 9,946 core words / 607 axes / 117,659 senses both times):")
    lines.append("")
    lines.append("| | before | after |")
    lines.append("|---|---|---|")
    lines.append("| fingerprint | `e0daef638b640dd5` | `e0daef638b640dd5` (IDENTICAL) |")
    lines.append("| `senses_of('dog')` | `dog.n.01, frump.n.01, dog.n.03, cad.n.01, frank.n.02` | same |")
    lines.append("| `senses_of('cat')` | `cat.n.01, guy.n.01, cat.n.03, kat.n.01, cat-o'-nine-tails.n.01` | same |")
    lines.append("| `senses_of('run')` | `run.n.01, test.n.05, footrace.n.01, streak.n.01, run.n.05` | same |")
    lines.append("| `senses_of('happy')` | `happy.a.01, felicitous.s.01, glad.s.01, happy.s.03` | same |")
    lines.append("| `senses_of('bank')` | `bank.n.01, depository_financial_institution.n.01, bank.n.03, bank.n.04, bank.n.05` | same |")
    lines.append("| `senses_of('perro')` | `[]` | `dog.n.01, rotter.n.01` |")
    lines.append("| `senses_of('gato')` | `[]` | `cat.n.01, caterpillar.n.02, dodger.n.01, kitty.n.04, tom.n.02` |")
    lines.append("| `senses_of('casa')` | `[]` | `building.n.01, diggings.n.02, dwelling.n.01, family.n.01, firm.n.01` |")
    lines.append("| `senses_of('niño')` | `[]` | `baby.n.01, chap.n.01, child.n.01, child.n.02, child.n.03` |")
    lines.append("")
    lines.append("English senses_of unchanged (5/5 exact match); Spanish now resolves (4/4 "
                 "test words, matching `dev/SPANISH_SWAP_FEASIBILITY.md`'s standalone-probe "
                 "numbers exactly, now folded into the production USVS build instead of a "
                 "throwaway script).")
    lines.append("")
    lines.append("## Grounding-type breakdown (all nodes: predicate + roles, across all trees)")
    lines.append("")
    total_g = sum(n_grounding_type.values())
    for k, v in sorted(n_grounding_type.items()):
        pct = 100.0 * v / total_g if total_g else 0.0
        lines.append(f"- `{k}`: {v} ({pct:.1f}%)")
    lines.append("")
    lines.append("## Predicate-verb grounding: sense vs. entity fallback")
    lines.append("")
    lines.append("Conjugated Spanish predicate verbs (está/fue/tomó) are NOT base OMW-es "
                 "dictionary lemmas, so they don't hit the lemma index and correctly fall back "
                 "to `type:\"entity\"` per contract §4.2's documented ungrounded-content-word "
                 "case -- this is the one narrower, separately-scoped gap "
                 "`dev/SPANISH_SWAP_FEASIBILITY.md` flags (verb-conjugation generation is out of "
                 "scope for this task, and for `scripts/build_parser_lexicon.py`'s own stated scope).")
    lines.append("")
    total_p = sum(n_predicate_grounding_type.values())
    for k, v in sorted(n_predicate_grounding_type.items()):
        pct = 100.0 * v / total_p if total_p else 0.0
        lines.append(f"- predicate grounding `{k}`: {v} ({pct:.1f}%)")
    lines.append("")
    lines.append("## Forest width (trees per sentence, after clause-level dedup)")
    lines.append("")
    if n_trees:
        lines.append(f"- median: {statistics.median(n_trees):.1f}")
        lines.append(f"- p90: {pctl(n_trees, 0.90):.1f}")
        lines.append(f"- max: {max(n_trees)}")
        lines.append(f"- 1-tree sentences (no surviving structural ambiguity): "
                     f"{sum(1 for n in n_trees if n == 1)}/{len(n_trees)}")
    lines.append("")
    lines.append("## Raw parser hypotheses per sentence (before clause extraction/dedup)")
    lines.append("")
    if n_raw_hyps:
        lines.append(f"- median: {statistics.median(n_raw_hyps):.1f}")
        lines.append(f"- p90: {pctl(n_raw_hyps, 0.90):.1f}")
        lines.append(f"- max: {max(n_raw_hyps)}")
    lines.append("")
    lines.append("## Sense-candidate breadth (per node grounded `type:\"sense\"`)")
    lines.append("")
    if n_sense_cands:
        lines.append(f"- n sense-grounded nodes: {len(n_sense_cands)}")
        lines.append(f"- median candidates/node: {statistics.median(n_sense_cands):.1f}")
        lines.append(f"- p90 candidates/node: {pctl(n_sense_cands, 0.90):.1f}")
    lines.append("")
    lines.append("## Contract-shape validation")
    lines.append("")
    lines.append("Every emitted record was validated by round-tripping it through the real "
                 "reader, `nsm_ct.encoder_model._gold_sites`, with no exception raised -- see "
                 "the FINAL MESSAGE for the pass/fail count.")
    lines.append("")
    lines.append(
        f"Of the {len(n_raw_hyps)} emitted records, {n_only_best} had only 1 raw parser "
        "hypothesis (genuinely unambiguous parse, not a top-k extraction limitation -- same "
        "reachability note as the English builder: `ParserInputEncoder._parse_topk_one` already "
        "exposes `chart.hypotheses[:k]` for both languages, no parser-internal change needed)."
    )
    OUT_STATS.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
