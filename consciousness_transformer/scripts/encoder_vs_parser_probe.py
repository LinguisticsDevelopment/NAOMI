"""Encoder vs teacher, head to head, on prose NEITHER was trained on
(dev/reports/ENCODER_VS_PARSER.md; branch encoder-vs-parser).

Samples 1,500 FairytaleQA VAL+TEST sentences (never in any gold -- a
held-out corpus by construction, encoder-gold-v5-k12) + 500 Gutenberg
sentences from data/corpus/real_*.txt that do NOT appear in
runs/encoder_gold_v6.jsonl.gz or runs/hard_gold_train.jsonl (the two files
the reference checkpoint runs/arms/v6_all_hard_seed0.pt was trained on),
runs the deterministic TEACHER parser (same settings as the gold-v6 build:
build_encoder_gold_v2.build_tree/prune_forest, TOP_K=8,
CORPUS_MAX_HYPOTHESES/CORPUS_MAX_PARSE_SECONDS) and the ENCODER (rank-1,
beam_width=8/k=8, same as eval) on every sentence, then compares them:

  * on teacher-OK sentences: encoder-vs-teacher rank-1 edge-F1 / struct-exact
    / clause-predicate recall (OOD check of the committed 0.708 number).
  * on the COMPLEMENT (teacher failed: no-hypothesis/cap-hit/grounding-fail/
    too-long): automatic structural metrics for the encoder's rank-1 tree,
    plus the inputs for a hand-judged USABLE/PARTIAL/WRONG/EMPTY pass
    (done separately by the operator, not by this script -- see
    `--stage render-judge`).
  * the POS-cheat heuristic (scripts/cheat_encoder_baseline.py) on the same
    complement sentences, as a floor.

Stages (run in order; each stage reads the previous stage's output under
runs/evp_*):
    sample      -> runs/evp_sample.csv
    teacher     -> runs/evp_teacher.jsonl   (sharded across --workers)
    encoder     -> runs/evp_encoder.jsonl   (sharded across --workers)
    metrics     -> runs/evp_metrics.json + prints the outcome/OOD/complement tables
    judgeset    -> runs/evp_judgeset.jsonl  (the 80 complement + 40 calibration +
                   80-cheat-on-complement items to hand-judge, rendered for reading)

Usage:
    python scripts/encoder_vs_parser_probe.py --stage sample
    python scripts/encoder_vs_parser_probe.py --stage teacher --workers 4
    python scripts/encoder_vs_parser_probe.py --stage encoder --workers 4
    python scripts/encoder_vs_parser_probe.py --stage metrics
    python scripts/encoder_vs_parser_probe.py --stage judgeset
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 20261114
N_FAIRYTALE = 1500
N_GUTENBERG = 500
BINS = ("A", "B", "C", "D")

CORPUS_DIR = ROOT / "data" / "corpus"
GOLD_V6_GZ = ROOT / "runs" / "encoder_gold_v6.jsonl.gz"
HARD_GOLD_TRAIN = ROOT / "runs" / "hard_gold_train.jsonl"
CKPT_PATH = ROOT / "runs" / "arms" / "v6_all_hard_seed0.pt"
USVS_DIR = ROOT / "data" / "usvs"

OUT_SAMPLE_CSV = ROOT / "runs" / "evp_sample.csv"
OUT_TEACHER_JSONL = ROOT / "runs" / "evp_teacher.jsonl"
OUT_ENCODER_JSONL = ROOT / "runs" / "evp_encoder.jsonl"
OUT_METRICS_JSON = ROOT / "runs" / "evp_metrics.json"
OUT_JUDGESET_JSONL = ROOT / "runs" / "evp_judgeset.jsonl"
OUT_JUDGESET_RENDER = ROOT / "runs" / "evp_judgeset_render.txt"


def bin_of(n_tokens: int) -> str:
    if n_tokens <= 8:
        return "A"
    if n_tokens <= 15:
        return "B"
    if n_tokens <= 25:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# stage: sample
# ---------------------------------------------------------------------------

def _strip_comment_header(text: str) -> str:
    """These corpus files open with a `#`-prefixed provenance line (or two)
    followed by a blank line before the real prose starts; drop that block
    so it doesn't get split into junk pseudo-sentences by `iter_sentences`."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
        i += 1
    return "\n".join(lines[i:])


def load_fairytale_sentences() -> list:
    from nsm_ct.corpus import iter_sentences

    seen = set()
    out = []
    for name in ("k12_fairytaleqa_val.txt", "k12_fairytaleqa_test.txt"):
        text = _strip_comment_header((CORPUS_DIR / name).read_text(encoding="utf-8"))
        for s in iter_sentences(text):
            s = s.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append((name, s))
    return out


def load_gutenberg_sentences() -> list:
    import glob

    from nsm_ct.corpus import iter_sentences

    seen = set()
    out = []
    for f in sorted(glob.glob(str(CORPUS_DIR / "real_gutenberg_*.txt"))):
        text = _strip_comment_header(Path(f).read_text(encoding="utf-8"))
        for s in iter_sentences(text):
            s = s.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append((Path(f).name, s))
    return out


def load_gold_v6_texts() -> set:
    texts = set()
    with gzip.open(GOLD_V6_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("text", "").strip()
            if t:
                texts.add(t)
    if HARD_GOLD_TRAIN.exists():
        with open(HARD_GOLD_TRAIN, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                t = r.get("text", "").strip()
                if t:
                    texts.add(t)
    return texts


def stage_sample() -> None:
    rng = random.Random(SEED)

    fairytale = load_fairytale_sentences()
    print(f"fairytale VAL+TEST unique sentences available: {len(fairytale)}")
    rng.shuffle(fairytale)
    fairytale_sample = fairytale[:N_FAIRYTALE]

    gutenberg = load_gutenberg_sentences()
    print(f"gutenberg unique sentences available: {len(gutenberg)}")
    gold_texts = load_gold_v6_texts()
    print(f"gold-v6 + hard-gold-train texts loaded for exclusion: {len(gold_texts)}")
    gutenberg_unseen = [(f, s) for f, s in gutenberg if s not in gold_texts]
    print(f"gutenberg sentences NOT in encoder_gold_v6/hard_gold_train: {len(gutenberg_unseen)}")
    rng.shuffle(gutenberg_unseen)
    gutenberg_sample = gutenberg_unseen[:N_GUTENBERG]

    rows = []
    for fname, s in fairytale_sample:
        n = len(s.split())
        rows.append({"source": "fairytaleqa", "file": fname, "sentence": s, "n_tokens": n, "bin": bin_of(n)})
    for fname, s in gutenberg_sample:
        n = len(s.split())
        rows.append({"source": "gutenberg_unseen", "file": fname, "sentence": s, "n_tokens": n, "bin": bin_of(n)})

    with open(OUT_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "file", "sentence", "n_tokens", "bin"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} sampled sentences -> {OUT_SAMPLE_CSV}")
    print("by source:", Counter(r["source"] for r in rows))
    print("by source x bin:")
    for src in ("fairytaleqa", "gutenberg_unseen"):
        c = Counter(r["bin"] for r in rows if r["source"] == src)
        print(f"  {src}: " + ", ".join(f"{b}={c.get(b, 0)}" for b in BINS))


# ---------------------------------------------------------------------------
# stage: teacher (deterministic parser, same settings as gold v6)
# ---------------------------------------------------------------------------

def _load_sample_rows() -> list:
    with open(OUT_SAMPLE_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_and_build(usvs, parser, sentence: str, forest_mode: str, forest_margin: float):
    """Same per-sentence code path as build_encoder_gold_v2.build_record
    (TOP_K / CORPUS_MAX_HYPOTHESES / CORPUS_MAX_PARSE_SECONDS, build_tree,
    prune_forest -- the gold-v6 settings), but with finer-grained outcome
    tagging (too-long split out of the generic exception branch) and timing.
    Returns (record_or_None, outcome, elapsed_seconds)."""
    from build_encoder_gold_v2 import TOP_K, build_tree, prune_forest
    from nsm_ct.corpus import CORPUS_MAX_HYPOTHESES, CORPUS_MAX_PARSE_SECONDS
    from src.parser.quantum_parser import ParseResourceExceeded

    t0 = time.perf_counter()
    try:
        graphs, scores, _margin = parser._parse_topk_one(
            sentence, k=TOP_K, max_hypotheses=CORPUS_MAX_HYPOTHESES, max_seconds=CORPUS_MAX_PARSE_SECONDS,
        )
    except ParseResourceExceeded:
        return None, "cap-hit", time.perf_counter() - t0
    except ValueError as exc:
        if "too long" in str(exc).lower():
            return None, "too-long", time.perf_counter() - t0
        return None, "other-exception", time.perf_counter() - t0
    except Exception:
        return None, "other-exception", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0

    if not graphs:
        return None, "no-hypothesis", elapsed

    words = parser._tag(sentence)
    tokens = [w.text for w in words]
    pos = [w.pos.name for w in words]

    trees, links_per_tree, tree_scores, seen = [], [], [], set()
    for graph, score in zip(graphs, scores):
        built = build_tree(usvs, tokens, pos, graph)
        if built is None:
            continue
        tree, link_dicts = built
        key = json.dumps(tree, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trees.append(tree)
        links_per_tree.append(link_dicts)
        tree_scores.append(score)

    if not trees:
        return None, "grounding-fail", elapsed

    trees, links_per_tree = prune_forest(trees, links_per_tree, tree_scores, forest_mode, forest_margin)

    token_sense_candidates = []
    for i, tok in enumerate(tokens):
        cands, lemma = usvs.senses_of_surface(tok, pos_hint=pos[i])
        cands = list(cands)
        if cands:
            token_sense_candidates.append({
                "index": i, "token": tok, "lemma": lemma,
                "sense_candidates": cands, "chosen_sense": cands[0],
            })

    record = {
        "text": sentence, "tokens": tokens, "pos": pos,
        "lattice": {"trees": trees, "discourse_links_per_tree": links_per_tree},
        "token_sense_candidates": token_sense_candidates,
    }
    return record, "ok", elapsed


def _teacher_worker(shard_idx: int, rows: list, out_path: Path) -> None:
    from build_encoder_gold_v2 import FOREST_MARGIN_DEFAULT, FOREST_MODE_DEFAULT, _new_parser
    from nsm_ct.ground.usvs import load_usvs

    usvs = load_usvs(str(USVS_DIR))
    parser = _new_parser()
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            record, outcome, elapsed = classify_and_build(
                usvs, parser, row["sentence"], FOREST_MODE_DEFAULT, FOREST_MARGIN_DEFAULT)
            out_row = {
                "source": row["source"], "file": row["file"], "sentence": row["sentence"],
                "n_tokens": int(row["n_tokens"]), "bin": row["bin"],
                "outcome": outcome, "parse_seconds": round(elapsed, 4),
                "record": record,
            }
            out_f.write(json.dumps(out_row) + "\n")
            out_f.flush()
            if (i + 1) % 25 == 0:
                print(f"[shard {shard_idx}] {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[shard {shard_idx}] done: {len(rows)} sentences in {time.time() - t0:.0f}s", flush=True)


def stage_teacher(workers: int, limit: int = None) -> None:
    import multiprocessing as mp

    rows = _load_sample_rows()
    if limit:
        rows = rows[:limit]
    print(f"teacher stage: {len(rows)} sentences, {workers} worker(s)")

    if workers <= 1:
        shard_path = OUT_TEACHER_JSONL
        _teacher_worker(0, rows, shard_path)
        return

    n = len(rows)
    base, extra = divmod(n, workers)
    chunks, start = [], 0
    for i in range(workers):
        size = base + (1 if i < extra else 0)
        chunks.append(rows[start:start + size])
        start += size

    shard_paths = [OUT_TEACHER_JSONL.with_suffix(f".shard{i}.jsonl") for i in range(workers)]
    procs = []
    for i, (chunk, sp) in enumerate(zip(chunks, shard_paths)):
        p = mp.Process(target=_teacher_worker, args=(i, chunk, sp))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        raise SystemExit(f"a teacher worker failed: exit codes {[p.exitcode for p in procs]}")

    with open(OUT_TEACHER_JSONL, "w", encoding="utf-8") as out_f:
        for sp in shard_paths:
            out_f.write(sp.read_text(encoding="utf-8"))
            sp.unlink()
    print(f"merged {workers} shards -> {OUT_TEACHER_JSONL}")


# ---------------------------------------------------------------------------
# stage: encoder (rank-1 beam decode, same settings as eval)
# ---------------------------------------------------------------------------

def _encoder_worker(shard_idx: int, rows: list, out_path: Path) -> None:
    import torch

    from nsm_ct import encoder_model as em
    from nsm_ct.ground.usvs import load_usvs
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from probe_encoder_complement import build_bare_record, judge_tree

    torch.set_num_threads(1)  # avoid N-workers x all-cores contention
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    pos_vocab = ckpt["pos_vocab"]
    role_vocab = ckpt["role_vocab"]
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=ckpt["d_axes"],
                             hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                             controller_hidden=ckpt["d_model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    usvs = load_usvs(str(USVS_DIR))
    tok = SimpleTokenizer.build([], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            sentence = row["sentence"]
            rec = build_bare_record(parser, usvs, sentence)
            feats = em.build_features(rec, usvs, pos_vocab, ckpt["hash_buckets"])
            t1 = time.perf_counter()
            forest = em.beam_decode(model, feats, beam_width=8, k=8, policy="model")
            dt = time.perf_counter() - t1
            top1 = forest[0] if forest else {"clauses": []}
            judge = judge_tree(top1, rec["pos"], rec["tokens"])
            out_row = {
                "source": row["source"], "file": row["file"], "sentence": sentence,
                "n_tokens": int(row["n_tokens"]), "bin": row["bin"],
                "tokens": rec["tokens"], "pos": rec["pos"], "tree": top1,
                "judge": judge, "decode_seconds": round(dt, 4),
            }
            out_f.write(json.dumps(out_row) + "\n")
            out_f.flush()
            if (i + 1) % 25 == 0:
                print(f"[shard {shard_idx}] {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[shard {shard_idx}] done: {len(rows)} sentences in {time.time() - t0:.0f}s", flush=True)


def stage_encoder(workers: int, limit: int = None) -> None:
    import multiprocessing as mp

    rows = _load_sample_rows()
    if limit:
        rows = rows[:limit]
    print(f"encoder stage: {len(rows)} sentences, {workers} worker(s)")

    if workers <= 1:
        _encoder_worker(0, rows, OUT_ENCODER_JSONL)
        return

    n = len(rows)
    base, extra = divmod(n, workers)
    chunks, start = [], 0
    for i in range(workers):
        size = base + (1 if i < extra else 0)
        chunks.append(rows[start:start + size])
        start += size

    shard_paths = [OUT_ENCODER_JSONL.with_suffix(f".shard{i}.jsonl") for i in range(workers)]
    procs = []
    for i, (chunk, sp) in enumerate(zip(chunks, shard_paths)):
        p = mp.Process(target=_encoder_worker, args=(i, chunk, sp))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        raise SystemExit(f"an encoder worker failed: exit codes {[p.exitcode for p in procs]}")

    with open(OUT_ENCODER_JSONL, "w", encoding="utf-8") as out_f:
        for sp in shard_paths:
            out_f.write(sp.read_text(encoding="utf-8"))
            sp.unlink()
    print(f"merged {workers} shards -> {OUT_ENCODER_JSONL}")


# ---------------------------------------------------------------------------
# stage: metrics
# ---------------------------------------------------------------------------

FAIL_OUTCOMES = ("no-hypothesis", "cap-hit", "grounding-fail", "too-long", "other-exception")
VERB_LIKE = {"VERB", "AUX"}
NOMINAL_LIKE = {"NOUN", "PROPN", "PRON"}


def _load_jsonl(path: Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _nanmean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _f1(p: float, r: float) -> float:
    if p != p or r != r or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


def clause_predicate_indices(tree: dict) -> set:
    """For an ENCODER-emitted tree, where `clause["predicate"]` is already a
    grounding-node dict carrying `token_index` (the beam_decode/judge_tree
    shape -- see probe_encoder_complement.py)."""
    idxs = set()
    for cl in (tree or {}).get("clauses", []):
        pred = cl.get("predicate", {})
        idx = pred.get("token_index") if isinstance(pred, dict) else None
        if idx is not None:
            idxs.add(idx)
    return idxs


def gold_clause_predicate_indices(record: dict, tree: dict) -> set:
    """For a GOLD (teacher) tree, where `clause["predicate"]` is instead the
    bare surface string per the IO contract (dev/ENCODER_IO_CONTRACT_V2.md
    S3) -- recover its token_index the same way `clause_node_order` does
    (`em._predicate_token_index`'s consume-on-match walk)."""
    from nsm_ct import encoder_model as em

    idxs = set()
    for cl in (tree or {}).get("clauses", []):
        idx = em._predicate_token_index(record, cl)
        if idx is not None:
            idxs.add(idx)
    return idxs


def gold_top1_tree(record: dict) -> dict:
    trees = record.get("lattice", {}).get("trees", [])
    return trees[0] if trees else {"clauses": []}


def complement_auto_metrics(tree: dict, pos: list, judge: dict) -> dict:
    clauses = (tree or {}).get("clauses", [])
    produced_tree = len(clauses) > 0
    structurally_valid = produced_tree and judge["j3"]

    verb_idx = {i for i, p in enumerate(pos) if p in VERB_LIKE}
    noun_idx = {i for i, p in enumerate(pos) if p in NOMINAL_LIKE}

    pred_idx = clause_predicate_indices(tree)
    role_idx = set()
    writeable_clauses = 0
    for cl in clauses:
        pred = cl.get("predicate", {})
        p_idx = pred.get("token_index") if isinstance(pred, dict) else None
        p_pos = pos[p_idx] if p_idx is not None and 0 <= p_idx < len(pos) else None
        grounded_roles = 0
        for r in cl.get("roles", []):
            idx = r.get("token_index")
            if idx is not None:
                role_idx.add(idx)
                grounded_roles += 1
        if p_pos in VERB_LIKE and grounded_roles >= 1:
            writeable_clauses += 1

    verb_coverage = len(pred_idx & verb_idx) / len(verb_idx) if verb_idx else float("nan")
    noun_coverage = len(role_idx & noun_idx) / len(noun_idx) if noun_idx else float("nan")

    return {
        "produced_tree": produced_tree,
        "structurally_valid": structurally_valid,
        "n_clauses": len(clauses),
        "n_tagger_verbs": len(verb_idx),
        "verb_coverage": verb_coverage,
        "noun_coverage": noun_coverage,
        "n_writeable_clauses": writeable_clauses,
    }


def stage_metrics() -> None:
    from nsm_ct import encoder_model as em

    teacher_rows = _load_jsonl(OUT_TEACHER_JSONL)
    encoder_rows = _load_jsonl(OUT_ENCODER_JSONL)
    assert len(teacher_rows) == len(encoder_rows), (len(teacher_rows), len(encoder_rows))
    n = len(teacher_rows)
    print(f"metrics: {n} sentences (teacher+encoder joined by row order)")

    for t, e in zip(teacher_rows, encoder_rows):
        assert t["sentence"] == e["sentence"], "row-order mismatch between teacher/encoder outputs"

    out = {"n_total": n}

    # ---- outcome table by source x bin --------------------------------
    outcome_table = defaultdict(lambda: defaultdict(int))
    for t in teacher_rows:
        outcome_table[(t["source"], t["bin"])][t["outcome"]] += 1
    out["outcome_table"] = {f"{src}|{b}": dict(v) for (src, b), v in outcome_table.items()}

    overall_outcomes = Counter(t["outcome"] for t in teacher_rows)
    out["overall_outcomes"] = dict(overall_outcomes)
    by_source_outcomes = defaultdict(Counter)
    for t in teacher_rows:
        by_source_outcomes[t["source"]][t["outcome"]] += 1
    out["by_source_outcomes"] = {k: dict(v) for k, v in by_source_outcomes.items()}

    print("\noverall outcomes:", dict(overall_outcomes))
    for src, c in by_source_outcomes.items():
        print(f"  {src}: {dict(c)}")

    # ---- OOD agreement on teacher-OK sentences -------------------------
    ok_pairs = [(t, e) for t, e in zip(teacher_rows, encoder_rows) if t["outcome"] == "ok"]
    print(f"\nteacher-OK sentences: {len(ok_pairs)}/{n}")

    scores = []
    clause_recalls = []
    for t, e in ok_pairs:
        record = t["record"]
        top1 = e["tree"]
        scores.append(em.score_record(record, [top1]))
        gold_pred_idx = gold_clause_predicate_indices(record, gold_top1_tree(record))
        enc_pred_idx = clause_predicate_indices(top1)
        if gold_pred_idx:
            clause_recalls.append(len(gold_pred_idx & enc_pred_idx) / len(gold_pred_idx))

    agg = em.aggregate_recall(scores)
    rank1_f1 = _f1(agg["edge_precision"], agg["edge_recall"])
    clause_recall = _nanmean(clause_recalls)
    out["ood_agreement"] = {
        "n": len(ok_pairs), "edge_precision": agg["edge_precision"], "edge_recall": agg["edge_recall"],
        "edge_f1": rank1_f1, "struct_exact": agg["structure_recall"], "clause_predicate_recall": clause_recall,
    }
    print(f"\nOOD agreement (encoder rank-1 vs teacher, n={len(ok_pairs)}):")
    print(f"  edge_P={agg['edge_precision']:.3f}  edge_R={agg['edge_recall']:.3f}  edge_F1={rank1_f1:.3f}  "
          f"struct-exact={agg['structure_recall']:.3f}  clause_predicate_recall={clause_recall:.3f}")
    print(f"  (reference branch encoder-v6-seed0 reports edge_F1=0.708 on the teacher's own in-distribution holdout)")

    by_source_scores = defaultdict(list)
    for (t, e), sc in zip(ok_pairs, scores):
        by_source_scores[t["source"]].append(sc)
    out["ood_agreement_by_source"] = {}
    for src, sc_list in by_source_scores.items():
        a = em.aggregate_recall(sc_list)
        f1 = _f1(a["edge_precision"], a["edge_recall"])
        out["ood_agreement_by_source"][src] = {
            "n": len(sc_list), "edge_precision": a["edge_precision"], "edge_recall": a["edge_recall"],
            "edge_f1": f1, "struct_exact": a["structure_recall"],
        }
        print(f"  [{src}] n={len(sc_list)} edge_F1={f1:.3f} struct-exact={a['structure_recall']:.3f}")

    # ---- complement automatic metrics ----------------------------------
    complement_pairs = [(t, e) for t, e in zip(teacher_rows, encoder_rows) if t["outcome"] in FAIL_OUTCOMES]
    print(f"\ncomplement (teacher-failed) sentences: {len(complement_pairs)}/{n}")

    auto_metrics = []
    for t, e in complement_pairs:
        m = complement_auto_metrics(e["tree"], e["pos"], e["judge"])
        m["outcome"] = t["outcome"]
        m["bin"] = t["bin"]
        m["source"] = t["source"]
        auto_metrics.append(m)

    n_c = len(auto_metrics)
    out["complement_auto"] = {
        "n": n_c,
        "produced_tree_rate": sum(m["produced_tree"] for m in auto_metrics) / n_c if n_c else float("nan"),
        "structurally_valid_rate": sum(m["structurally_valid"] for m in auto_metrics) / n_c if n_c else float("nan"),
        "mean_verb_coverage": _nanmean([m["verb_coverage"] for m in auto_metrics]),
        "mean_noun_coverage": _nanmean([m["noun_coverage"] for m in auto_metrics]),
        "mean_n_clauses": _nanmean([m["n_clauses"] for m in auto_metrics]),
        "mean_n_tagger_verbs": _nanmean([m["n_tagger_verbs"] for m in auto_metrics]),
        "mean_n_writeable_clauses": _nanmean([m["n_writeable_clauses"] for m in auto_metrics]),
        "any_writeable_rate": sum(1 for m in auto_metrics if m["n_writeable_clauses"] >= 1) / n_c if n_c else float("nan"),
    }
    print(f"\ncomplement automatic metrics (n={n_c}):")
    for k, v in out["complement_auto"].items():
        print(f"  {k}: {v}")

    by_outcome = defaultdict(list)
    for m in auto_metrics:
        by_outcome[m["outcome"]].append(m)
    out["complement_auto_by_outcome"] = {}
    for outc, ms in by_outcome.items():
        nn = len(ms)
        out["complement_auto_by_outcome"][outc] = {
            "n": nn,
            "produced_tree_rate": sum(mm["produced_tree"] for mm in ms) / nn,
            "any_writeable_rate": sum(1 for mm in ms if mm["n_writeable_clauses"] >= 1) / nn,
        }
    print("\nby outcome:")
    for outc, d in out["complement_auto_by_outcome"].items():
        print(f"  {outc}: n={d['n']} produced_tree={d['produced_tree_rate']:.1%} any_writeable={d['any_writeable_rate']:.1%}")

    with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {OUT_METRICS_JSON}")


# ---------------------------------------------------------------------------
# stage: judgeset -- build the hand-judge sample (80 complement + 40
# teacher-OK calibration) and render both PLUS the POS-cheat heuristic's
# output on the same 80 complement sentences, for the operator to read and
# label USABLE / PARTIAL / WRONG / EMPTY.
# ---------------------------------------------------------------------------

N_JUDGE_COMPLEMENT = 80
N_JUDGE_CALIBRATION = 40


def _stratified_sample(pool: list, n_target: int, seed: int, key_fn) -> list:
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in pool:
        cells[key_fn(r)].append(r)
    for c in cells.values():
        rng.shuffle(c)
    nonempty = [k for k, v in cells.items() if v]
    if not nonempty:
        return []
    quota = max(1, n_target // len(nonempty))
    sampled = []
    for k in nonempty:
        take = cells[k][:quota]
        sampled.extend(take)
        cells[k] = cells[k][len(take):]
    shortfall = n_target - len(sampled)
    if shortfall > 0:
        leftover = [r for c in cells.values() for r in c]
        rng.shuffle(leftover)
        sampled.extend(leftover[:shortfall])
    rng.shuffle(sampled)
    return sampled[:n_target]


def render_node(role: str, node: dict, tokens: list) -> str:
    idx = node.get("token_index")
    tok = tokens[idx] if idx is not None and 0 <= idx < len(tokens) else "<none>"
    gtype = (node.get("grounding") or {}).get("type")
    return f"    {role}: {tok} [{gtype}]"


def render_tree_block(tree: dict, tokens: list, pos: list) -> str:
    lines = ["  POS: " + " ".join(f"{t}/{p}" for t, p in zip(tokens, pos))]
    clauses = (tree or {}).get("clauses", [])
    if not clauses:
        lines.append("  <no clauses emitted>")
    for ci, clause in enumerate(clauses):
        kind = clause.get("utterance_kind", "proposition")
        lines.append(f"  clause {ci} ({kind}):")
        lines.append(render_node("PREDICATE", clause.get("predicate", {}), tokens))
        for r in clause.get("roles", []):
            lines.append(render_node(r.get("relation", "?"), r, tokens))
    return "\n".join(lines)


def stage_judgeset() -> None:
    from cheat_encoder_baseline import heuristic_forest

    teacher_rows = _load_jsonl(OUT_TEACHER_JSONL)
    encoder_rows = _load_jsonl(OUT_ENCODER_JSONL)
    for t, e in zip(teacher_rows, encoder_rows):
        assert t["sentence"] == e["sentence"]

    complement_pool = [(t, e) for t, e in zip(teacher_rows, encoder_rows) if t["outcome"] in FAIL_OUTCOMES]
    ok_pool = [(t, e) for t, e in zip(teacher_rows, encoder_rows) if t["outcome"] == "ok"]

    complement_sample = _stratified_sample(
        complement_pool, N_JUDGE_COMPLEMENT, SEED + 1,
        key_fn=lambda te: (te[0]["outcome"], te[0]["bin"]))
    calibration_sample = _stratified_sample(
        ok_pool, N_JUDGE_CALIBRATION, SEED + 2,
        key_fn=lambda te: te[0]["bin"])

    items = []
    render_lines = []

    def pemit(s=""):
        render_lines.append(s)

    pemit("ENCODER VS PARSER -- hand-judge set")
    pemit(f"complement (teacher-failed) n={len(complement_sample)}, "
          f"calibration (teacher-OK) n={len(calibration_sample)}")
    pemit("Label each item USABLE / PARTIAL / WRONG / EMPTY in the judgeset jsonl.")
    pemit("=" * 78)

    pemit("\n### SECTION 1: COMPLEMENT -- ENCODER rank-1 tree (n=%d) ###\n" % len(complement_sample))
    for i, (t, e) in enumerate(complement_sample):
        tokens, pos, tree = e["tokens"], e["pos"], e["tree"]
        pemit(f"[C{i}] ({t['outcome']}, bin {t['bin']}, {t['source']}) {t['sentence']}")
        pemit(render_tree_block(tree, tokens, pos))
        pemit("")
        items.append({
            "item_id": f"C{i}", "section": "complement_encoder", "outcome": t["outcome"], "bin": t["bin"],
            "source": t["source"], "sentence": t["sentence"], "tokens": tokens, "pos": pos, "tree": tree,
            "label": None,
        })

    pemit("\n### SECTION 2: COMPLEMENT -- POS-CHEAT heuristic tree (SAME %d sentences) ###\n" % len(complement_sample))
    for i, (t, e) in enumerate(complement_sample):
        tokens, pos = e["tokens"], e["pos"]
        fake_record = {"tokens": tokens, "pos": pos, "token_sense_candidates": []}
        cheat_tree = heuristic_forest(fake_record)[0]
        pemit(f"[H{i}] ({t['outcome']}, bin {t['bin']}, {t['source']}) {t['sentence']}")
        pemit(render_tree_block(cheat_tree, tokens, pos))
        pemit("")
        items.append({
            "item_id": f"H{i}", "section": "complement_cheat", "outcome": t["outcome"], "bin": t["bin"],
            "source": t["source"], "sentence": t["sentence"], "tokens": tokens, "pos": pos, "tree": cheat_tree,
            "label": None,
        })

    pemit("\n### SECTION 3: CALIBRATION -- teacher-OK, ENCODER rank-1 tree (n=%d) ###\n" % len(calibration_sample))
    for i, (t, e) in enumerate(calibration_sample):
        tokens, pos, tree = e["tokens"], e["pos"], e["tree"]
        pemit(f"[K{i}] (bin {t['bin']}, {t['source']}) {t['sentence']}")
        pemit(render_tree_block(tree, tokens, pos))
        pemit("")
        items.append({
            "item_id": f"K{i}", "section": "calibration_encoder", "outcome": "ok", "bin": t["bin"],
            "source": t["source"], "sentence": t["sentence"], "tokens": tokens, "pos": pos, "tree": tree,
            "label": None,
        })

    with open(OUT_JUDGESET_JSONL, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    OUT_JUDGESET_RENDER.write_text("\n".join(render_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} items -> {OUT_JUDGESET_JSONL}")
    print(f"wrote rendered form -> {OUT_JUDGESET_RENDER}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                     choices=("sample", "teacher", "encoder", "metrics", "judgeset"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap n sentences processed")
    args = ap.parse_args()

    if args.stage == "sample":
        stage_sample()
    elif args.stage == "teacher":
        stage_teacher(args.workers, args.limit)
    elif args.stage == "encoder":
        stage_encoder(args.workers, args.limit)
    elif args.stage == "metrics":
        stage_metrics()
    elif args.stage == "judgeset":
        stage_judgeset()
    else:
        print(f"stage {args.stage!r} not implemented yet in this cut", file=sys.stderr)
        raise SystemExit(1)
