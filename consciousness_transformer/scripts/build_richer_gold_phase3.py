"""richer-gold PHASE 3 driver: regenerate the first 200 `text` sentences from
runs/encoder_gold_v2.jsonl through the phase-3 clause.py (phase1 modifier
extraction + phase3 SUBORDINATION recursion), reusing build_encoder_gold_v2's
build_record/build_clause_dict/ground_word machinery verbatim (that script
itself lives on the encoder-gold-v2 branch, not this one).

20s/sentence parse cap (ParserInputEncoder._parse_topk_one's own
max_seconds), same as phase1's driver.

Usage: python scripts/build_richer_gold_phase3.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.clause import _PRONOUNS, extract_discourse, is_entity  # noqa: E402
from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

USVS_DIR = ROOT / "data" / "usvs"
IN_JSONL = ROOT / "runs" / "encoder_gold_v2.jsonl"
OUT_JSONL = ROOT / "runs" / "richer_gold_phase3_sample.jsonl"
N_SAMPLE = 200
MAX_SECONDS = 20.0


def ground_word(usvs, word: Optional[str]) -> Dict[str, object]:
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
    def __init__(self, tokens: List[str]) -> None:
        from collections import deque
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
    from src.parser.quantum_parser import ParseResourceExceeded  # local import (qp_root on sys.path)

    try:
        graphs, _scores, _margin = parser._parse_topk_one(sentence, k=8, max_seconds=MAX_SECONDS)
    except ParseResourceExceeded:
        return None, "cap-hit", 0
    except TypeError:
        # _parse_topk_one may not accept max_seconds in this branch's signature -- fall back.
        try:
            graphs, _scores, _margin = parser._parse_topk_one(sentence, k=8)
        except ParseResourceExceeded:
            return None, "cap-hit", 0
        except Exception:
            return None, "parse-failed", 0
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

    record = {
        "text": sentence,
        "tokens": tokens,
        "pos": pos,
        "lattice": {"trees": trees, "discourse_links_per_tree": links_per_tree},
    }
    return record, "ok", len(graphs)


def main() -> int:
    t0 = time.time()
    with open(IN_JSONL, encoding="utf-8") as f:
        source_records = [json.loads(line) for line in f]
    sentences = [r["text"] for r in source_records[:N_SAMPLE]]
    print(f"sample: {len(sentences)} sentences (first {N_SAMPLE} from {IN_JSONL})", flush=True)

    print("loading USVS ...", flush=True)
    usvs = load_usvs(USVS_DIR)
    print(f"USVS loaded: {len(usvs.core_words)} core words, {len(usvs.sense_ids)} senses", flush=True)

    tok = SimpleTokenizer.build(sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    outcomes: Dict[str, int] = defaultdict(int)
    n_ok = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for i, sentence in enumerate(sentences):
            record, outcome, _n_raw = build_record(usvs, parser, sentence)
            outcomes[outcome] += 1
            if record is not None:
                out_f.write(json.dumps(record) + "\n")
                n_ok += 1
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(sentences)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"wrote {n_ok} records -> {OUT_JSONL}", flush=True)
    print(f"outcomes: {dict(outcomes)}", flush=True)
    print(f"total time: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
