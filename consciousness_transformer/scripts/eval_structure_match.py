"""STRUCTURE-MATCH round-trip evaluator (cycle-consistency) for the trained
decoder -- a MEANING-LEVEL replacement for the old verbatim-text round-trip
(`scripts/_roundtrip_aliasfix.py`'s `token_f1`, ~0.32).

Old metric: text -> encoder.beam_decode -> decoder.realize -> text', scored
by literal token overlap against the ORIGINAL text. That punishes a
perfectly faithful paraphrase that drops a modifier the committed structure
never carried in the first place -- it is a text metric standing in for a
meaning metric.

This metric instead asks: does the REALIZED text carry the same STRUCTURE
the decoder was handed? For a held-out gold record:

    1. tree_A  = record["lattice"]["trees"][0]           (the committed
       structure the decoder is given -- GOLD shape: clause_node_order).
    2. text'   = decoder.realize(structure built from tree_A).
    3. record' = re-parse text' through the SAME pipeline the gold builder
       uses (quantum_parser chart parse for a pipeline-fidelity gate, the
       same POS tagger, the same USVS sense grounding) -- tokens/pos/
       token_sense_candidates, nothing else (no tree extraction: tree_B
       comes from the trained encoder, not from re-running gold extraction).
    4. tree_B  = encoder.beam_decode(features(record'))[0]  (FOREST shape:
       clause["predicate"/"roles"][i]["token_index"/"grounding"]).
    5. score edge_precision/edge_recall/F1(tree_A, tree_B).

Edge KEY choice -- the one deliberate departure from reusing
`encoder_model.score_record` byte-for-byte: that function's edge tuples are
keyed by `token_index`, which is only comparable when both trees index the
SAME token sequence (its normal use: one gold record's own trees vs its own
beam_decode forest). Here tree_A indexes into the ORIGINAL sentence's tokens
and tree_B indexes into text''s (a different, usually different-length,
re-tokenized) token sequence -- comparing raw positions across the two would
mostly measure coincidence, not structure. So each edge here is
`(relation, surface_word.lower() or None, grounding_type)` -- literally
`encoder_model._clause_skeleton`'s own tuple shape (relation, key, gtype),
just with the key resolved through each tree's OWN `tokens` list to the word
it actually points at, keeping tree_A and tree_B comparable regardless of
how text' re-tokenizes. `gold_tree_edges` reuses
`encoder_model.clause_node_order` verbatim (the same node walk the encoder
oracle and the decoder both already use) to enumerate tree_A's nodes;
`forest_tree_edges` walks tree_B's already-flat predicted shape the same way
`encoder_model._clause_skeleton`/`colab_train_all.predicted_tree_to_structure`
do. Multiple nodes resolving to the same (relation, word, type) collapse to
one edge, same as `_clause_skeleton`'s own frozenset-of-clause design already
tolerates.

A faithful paraphrase that never had a modifier scores high: tree_A and
tree_B are compared at the same (relation, word, grounding-type) abstraction
level, not as raw text.

Usage: python scripts/eval_structure_match.py [--n 40] [--control-n 15]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from nsm_ct import decoder_trained as dt  # noqa: E402
from nsm_ct import encoder_model as em  # noqa: E402
from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

from train_encoder import load_gold, stratified_split  # noqa: E402
from train_decoder import load_records as load_decoder_records, split_records as split_decoder_records  # noqa: E402

RUNS = ROOT / "runs"
DEFAULT_N = 40
DEFAULT_CONTROL_N = 15
PARSE_CAP_SECONDS = 20.0
OLD_VERBATIM_TOKEN_F1 = 0.32  # scripts/_roundtrip_aliasfix.py's reference number


# ---------------------------------------------------------------------------
# Checkpoint / data loading (mirrors scripts/_roundtrip_aliasfix.py exactly,
# so this reproduces the SAME encoder test split / decoder train split the
# ckpts were themselves evaluated against)
# ---------------------------------------------------------------------------

def load_checkpoints():
    enc_ckpt = torch.load(RUNS / "encoder_colab.pt", map_location="cpu", weights_only=False)
    encoder = em.EncoderModel(enc_ckpt["pos_vocab"], enc_ckpt["role_vocab"], d_axes=enc_ckpt["d_axes"],
                               hash_buckets=enc_ckpt["hash_buckets"], d_model=enc_ckpt["d_model"],
                               controller_hidden=enc_ckpt["d_model"])
    encoder.load_state_dict(enc_ckpt["model_state"])
    encoder.eval()

    dec_ckpt = torch.load(RUNS / "decoder_colab.pt", map_location="cpu", weights_only=False)
    decoder = dt.DecoderTrainedModel(dec_ckpt["relation_vocab"], dec_ckpt["function_vocab"],
                                      hash_buckets=dec_ckpt["hash_buckets"], d_model=dec_ckpt["d_model"])
    decoder.load_state_dict(dec_ckpt["model_state"])
    decoder.eval()
    return enc_ckpt, encoder, dec_ckpt, decoder


def held_out_test_records(enc_ckpt: dict, dec_ckpt: dict) -> List[dict]:
    records = load_gold(str(RUNS / "encoder_gold_v2.jsonl"))
    enc_cfg = enc_ckpt["config"]
    _enc_train, _enc_dev, enc_test = stratified_split(
        records, enc_cfg["seed"], enc_cfg["n_train"], enc_cfg["n_dev"], enc_cfg["n_test"])

    dec_cfg = dec_ckpt["config"]
    dec_records_all = load_decoder_records(str(RUNS / "encoder_gold_v2.jsonl"))
    import random
    n_dec = dec_cfg["n_train"] + dec_cfg["n_dev"]
    dec_pool = dec_records_all[:]
    random.Random(dec_cfg["seed"]).shuffle(dec_pool)
    dec_pool = dec_pool[:n_dec]
    dec_train, _dec_dev = split_decoder_records(dec_pool, dec_cfg["seed"], dec_cfg["n_train"], dec_cfg["n_dev"])
    dec_train_texts = {r["text"] for r in dec_train}

    pool = [r for r in enc_test if r["text"] not in dec_train_texts] or enc_test
    print(f"reproduced encoder test split: {len(enc_test)}; held out from decoder training: {len(pool)}",
          flush=True)
    return records, pool


# ---------------------------------------------------------------------------
# The re-parse pipeline (parser -> ground -> record), reused verbatim in
# shape from scripts/build_encoder_gold_v2.py's build_record/ground_word,
# minus tree/lattice extraction (encoder_model.build_features never reads
# "lattice" -- only tokens/pos/token_sense_candidates/context).
# ---------------------------------------------------------------------------

def build_reparse_pipeline(seed_texts: List[str]) -> ParserInputEncoder:
    tok = SimpleTokenizer.build(seed_texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        raise RuntimeError("quantum_parser unavailable; cannot re-parse text' for structure-match")
    return parser


def reparse_to_record(usvs, parser: ParserInputEncoder, text: str,
                       max_seconds: float = PARSE_CAP_SECONDS) -> Tuple[Optional[dict], str]:
    """text -> {tokens, pos, token_sense_candidates}, or (None, reason)."""
    from src.parser.quantum_parser import ParseResourceExceeded  # local import (qp_root on sys.path)

    try:
        graphs, _scores, _margin = parser._parse_topk_one(text, k=1, max_seconds=max_seconds)
    except ParseResourceExceeded:
        return None, "cap-hit"
    except Exception:
        return None, "parse-failed"
    if not graphs:
        return None, "no-hypothesis"

    words = parser._tag(text)
    tokens = [w.text for w in words]
    pos = [w.pos.name for w in words]
    if not tokens:
        return None, "empty-tokens"

    token_sense_candidates = []
    for i, tok in enumerate(tokens):
        cands = usvs.senses_of(tok)
        if cands:
            token_sense_candidates.append({"index": i, "token": tok,
                                            "sense_candidates": list(cands), "chosen_sense": cands[0]})
    record = {"text": text, "tokens": tokens, "pos": pos, "token_sense_candidates": token_sense_candidates}
    return record, "ok"


# ---------------------------------------------------------------------------
# Edge extraction (see module docstring for why word, not token_index, is
# the key) + precision/recall/F1
# ---------------------------------------------------------------------------

def gold_tree_edges(record: dict, tree: dict) -> frozenset:
    edges = set()
    for clause in tree.get("clauses", []):
        for relation, grounding, tidx in em.clause_node_order(record, clause):
            word = record["tokens"][tidx].lower() if tidx is not None else None
            edges.add((relation, word, grounding.get("type")))
    return frozenset(edges)


def forest_tree_edges(record: dict, tree: dict) -> frozenset:
    edges = set()
    for clause in tree.get("clauses", []):
        pred = clause.get("predicate") or {}
        tidx = pred.get("token_index")
        word = record["tokens"][tidx].lower() if tidx is not None else None
        edges.add(("PREDICATE", word, (pred.get("grounding") or {}).get("type")))
        for role in clause.get("roles", []):
            tidx = role.get("token_index")
            word = record["tokens"][tidx].lower() if tidx is not None else None
            edges.add((role.get("relation", "PREDICATE"), word, (role.get("grounding") or {}).get("type")))
    return frozenset(edges)


def edge_prf(edges_a: frozenset, edges_b: frozenset) -> Dict[str, float]:
    overlap = len(edges_a & edges_b)
    precision = overlap / len(edges_b) if edges_b else 0.0
    recall = overlap / len(edges_a) if edges_a else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


ABSTAIN_PRF = {"edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0}


# ---------------------------------------------------------------------------
# One record's structure-match cycle
# ---------------------------------------------------------------------------

def structure_match_one(usvs, parser, encoder, decoder, pos_vocab: dict, hash_buckets: int,
                         record: dict, beam_width: int, k: int,
                         override_text_prime: Optional[str] = None) -> dict:
    tree_a = record["lattice"]["trees"][0]
    edges_a = gold_tree_edges(record, tree_a)

    if override_text_prime is not None:
        text_prime = override_text_prime
    else:
        structure = dt.build_structure(record, tree_a)
        realized = dt.realize(decoder, structure)
        if not realized:
            return {"abstain": True, "reason": "empty-realize", "text": record["text"],
                     "text_prime": "", **ABSTAIN_PRF}
        text_prime = " ".join(realized)

    new_record, outcome = reparse_to_record(usvs, parser, text_prime)
    if new_record is None:
        return {"abstain": True, "reason": f"reparse-{outcome}", "text": record["text"],
                 "text_prime": text_prime, **ABSTAIN_PRF}

    feats = em.build_features(new_record, usvs, pos_vocab, hash_buckets)
    forest = em.beam_decode(encoder, feats, beam_width=beam_width, k=k, policy="model")
    if not forest:
        return {"abstain": True, "reason": "empty-forest", "text": record["text"],
                 "text_prime": text_prime, **ABSTAIN_PRF}

    tree_b = forest[0]
    edges_b = forest_tree_edges(new_record, tree_b)
    prf = edge_prf(edges_a, edges_b)
    return {"abstain": False, "reason": "ok", "text": record["text"], "text_prime": text_prime, **prf}


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def run_pass(usvs, parser, encoder, decoder, pos_vocab, hash_buckets, records: List[dict],
             beam_width: int, k: int, label: str, control: bool = False) -> Tuple[List[dict], dict]:
    results = []
    t0 = time.time()
    for i, record in enumerate(records):
        override = record["text"] if control else None
        res = structure_match_one(usvs, parser, encoder, decoder, pos_vocab, hash_buckets,
                                   record, beam_width, k, override_text_prime=override)
        results.append(res)
        print(f"  [{label} {i + 1}/{len(records)}] ({time.time() - t0:.0f}s) abstain={res['abstain']} "
              f"({res['reason']}) edge_f1={res['edge_f1']:.3f}", flush=True)
    n = max(len(results), 1)
    agg = {
        "n": len(results),
        "abstain_rate": sum(1 for r in results if r["abstain"]) / n,
        "edge_precision": _mean([r["edge_precision"] for r in results]),
        "edge_recall": _mean([r["edge_recall"] for r in results]),
        "edge_f1": _mean([r["edge_f1"] for r in results]),
    }
    return results, agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="held-out test sentences to score")
    ap.add_argument("--control-n", type=int, default=DEFAULT_CONTROL_N,
                     help="sanity-control sentences (re-encode the GOLD text itself, skip the decoder)")
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    usvs = load_usvs(str(ROOT / "data" / "usvs"))
    enc_ckpt, encoder, dec_ckpt, decoder = load_checkpoints()
    all_gold_records, pool = held_out_test_records(enc_ckpt, dec_ckpt)

    test_records = pool[:args.n]
    control_records = test_records[:args.control_n]
    print(f"scoring {len(test_records)} held-out sentences (structure-match), "
          f"{len(control_records)} of them also as a sanity control", flush=True)

    parser = build_reparse_pipeline([r["text"] for r in all_gold_records])

    pos_vocab = enc_ckpt["pos_vocab"]
    hash_buckets = enc_ckpt["hash_buckets"]

    print("\n--- MAIN PASS: text -> decoder.realize -> re-parse -> beam_decode ---", flush=True)
    main_results, main_agg = run_pass(usvs, parser, encoder, decoder, pos_vocab, hash_buckets,
                                       test_records, args.beam_width, args.k, label="main", control=False)

    print("\n--- SANITY CONTROL: GOLD text -> re-parse -> beam_decode (decoder bypassed) ---", flush=True)
    control_results, control_agg = run_pass(usvs, parser, encoder, decoder, pos_vocab, hash_buckets,
                                             control_records, args.beam_width, args.k,
                                             label="control", control=True)

    print("\n" + "=" * 78)
    print("STRUCTURE-MATCH RESULT (round-trip cycle-consistency, edge-level)")
    print("=" * 78)
    print(f"  n = {main_agg['n']}")
    print(f"  edge_precision = {main_agg['edge_precision']:.3f}")
    print(f"  edge_recall    = {main_agg['edge_recall']:.3f}")
    print(f"  edge_F1        = {main_agg['edge_f1']:.3f}")
    print(f"  abstain_rate   = {main_agg['abstain_rate']:.3f}")
    print()
    print(f"  vs. OLD verbatim-text round-trip token_f1 = {OLD_VERBATIM_TOKEN_F1:.3f} "
          f"(scripts/_roundtrip_aliasfix.py)")
    print()
    print(f"  SANITY CONTROL (n={control_agg['n']}, GOLD text re-encoded directly, decoder bypassed):")
    print(f"    edge_precision = {control_agg['edge_precision']:.3f}")
    print(f"    edge_recall    = {control_agg['edge_recall']:.3f}")
    print(f"    edge_F1        = {control_agg['edge_f1']:.3f}")
    print(f"    abstain_rate   = {control_agg['abstain_rate']:.3f}")
    sane = control_agg["edge_f1"] > main_agg["edge_f1"]
    print(f"    re-encode cycle behaving sanely (control F1 > decoder-realized F1): {sane}")

    print("\n" + "-" * 78)
    print("10 EXAMPLES (gold text / realized text\\' / edge-F1(tree_A, tree_B))")
    print("-" * 78)
    for r in main_results[:10]:
        print(f"  gold:  {r['text']!r}")
        print(f"  text\\': {r['text_prime']!r}")
        print(f"  edge_F1={r['edge_f1']:.3f}  (P={r['edge_precision']:.3f} R={r['edge_recall']:.3f} "
              f"abstain={r['abstain']} reason={r['reason']})")
        print()

    print("-" * 78)
    print("NOTE: the decoder scored here (runs/decoder_colab.pt) is the OLD buggy")
    print("function_vocab decoder -- this run's deliverable is a WORKING structure-")
    print("match evaluator + a baseline number + confirmation the re-encode path")
    print("works end-to-end (see sanity control above), NOT a claim of decoder quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
