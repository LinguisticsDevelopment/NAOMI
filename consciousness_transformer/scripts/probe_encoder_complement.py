"""PART B -- teacher-complement probe (dev/AUDIT_2026-09-08.md action 1).

The encoder is trained to imitate the deterministic teacher parser; on
teacher-parseable text it is, by construction, never better than the
teacher (it is parser distillation). Its only possible independent value is
on sentences the teacher CANNOT parse -- and nobody has measured that. This
script does:

  1. Build the complement set: 100 sentences sampled from `data/corpus/
     real_*.txt` (the 1,475 unique sentences) where the teacher parser
     FAILS (`runs/m62_full_probe.csv`, built by
     `probe_teacher_complement_build.py`), stratified across outcome
     (grounding-fail / cap-hit / no-hypothesis) x length bin (A/B/C/D),
     excluding fragments under 2 content tokens.
  2. Run the run-2 encoder checkpoint (`beam_decode`, RANK-1 tree only --
     the tree the model would actually commit to) on each, timing it, next
     to the teacher's own recorded wall-clock on the same sentence (the
     teacher's attempt that failed still took time -- that IS the teacher's
     cost on this input).
  3. Auto-judge each rank-1 tree on three binary criteria against the real
     POS tagger's output: J1 predicate-ok, J2 subject-ok, J3 no-phantoms
     (see `judge_tree` below). usable = J1 and J2 and J3.
  4. CONTROL: run the identical judge on 50 teacher-PARSEABLE held-out
     TEST-split sentences (`runs/encoder_gold_v2.jsonl`, same split as
     `rescore_encoder.py`), scoring both the encoder's rank-1 tree and the
     teacher's own (top-1) gold tree -- the gold tree's usable-rate is the
     judge's ceiling; encoder-on-easy-text is the comparison point for the
     complement rate.
  5. Render all 100 complement outputs to `runs/complement_probe.txt`
     (5 best-looking / 5 worst-looking first), and write the numeric
     summary + gate verdict to `runs/complement_summary.txt`.

NO training. NO changes to `encoder_model.py`'s decode behavior -- this only
adds a reference-free auto-judge over its existing `beam_decode` output.

Usage: python scripts/probe_encoder_complement.py
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct import encoder_model as em  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from train_encoder import load_gold, stratified_split  # noqa: E402
from nsm_ct.tree_render import node_detail_compact, normalize_gold_tree  # noqa: E402

SEED = 20260908
N_COMPLEMENT = 100
N_CONTROL = 50
FAIL_OUTCOMES = ("grounding-fail", "cap-hit", "no-hypothesis")
BINS = ("A", "B", "C", "D")
CKPT_PATH = ROOT / "runs" / "encoder_colab.pt"
GOLD_PATH = ROOT / "runs" / "encoder_gold_v2.jsonl"
FULL_PROBE_CSV = ROOT / "runs" / "m62_full_probe.csv"
USVS_DIR = ROOT / "data" / "usvs"
OUT_PROBE_TXT = ROOT / "runs" / "complement_probe.txt"
OUT_SUMMARY_TXT = ROOT / "runs" / "complement_summary.txt"

VERB_LIKE = {"VERB", "AUX"}
NOMINAL_LIKE = {"NOUN", "PROPN", "PRON"}


# ---------------------------------------------------------------------------
# length bin (shared with probe_m62_gold_volume.py's A<=8/B9-15/C16-25/D26+)
# ---------------------------------------------------------------------------

def bin_of(n_tokens: int) -> str:
    if n_tokens <= 8:
        return "A"
    if n_tokens <= 15:
        return "B"
    if n_tokens <= 25:
        return "C"
    return "D"


def n_content_tokens(tokens) -> int:
    return sum(1 for t in tokens if any(c.isalnum() for c in t))


# ---------------------------------------------------------------------------
# 1. complement-set sampling
# ---------------------------------------------------------------------------

def load_failure_pool() -> list:
    rows = []
    with open(FULL_PROBE_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["outcome"] not in FAIL_OUTCOMES:
                continue
            sent = r["sentence"]
            if n_content_tokens(sent.split()) < 2:
                continue
            rows.append({
                "sentence": sent, "outcome": r["outcome"], "bin": r["bin"],
                "n_tokens": int(r["n_tokens"]), "parse_seconds": float(r["parse_seconds"]),
                "file": r["file"],
            })
    return rows


def stratified_sample(pool: list, n_target: int, seed: int) -> list:
    """Greedy proportional-with-cap stratification across (outcome, bin)
    cells -- mirrors `probe_m62_gold_volume.sample_bins`'s approach, but over
    a 2D stratum since real cap-hit sentences only occur in the D bin here
    (no forcing empty cells)."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for r in pool:
        cells[(r["outcome"], r["bin"])].append(r)
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


# ---------------------------------------------------------------------------
# build an encoder-ready record for a raw (untagged, no gold) sentence
# ---------------------------------------------------------------------------

def build_bare_record(parser: ParserInputEncoder, usvs, sentence: str) -> dict:
    words = parser._tag(sentence)
    tokens = [w.text for w in words]
    pos = [w.pos.name for w in words]
    token_sense_candidates = []
    for i, tok in enumerate(tokens):
        cands = usvs.senses_of(tok)
        if cands:
            token_sense_candidates.append({"index": i, "token": tok, "sense_candidates": list(cands)})
    return {"text": sentence, "tokens": tokens, "pos": pos, "token_sense_candidates": token_sense_candidates}


# ---------------------------------------------------------------------------
# 3. auto-judge (reference-free: checks the tree against the sentence's own
#    POS tags, not against any gold tree)
# ---------------------------------------------------------------------------

def judge_tree(tree: dict, pos_list: list, tokens: list) -> dict:
    clauses = tree.get("clauses", []) if tree else []
    reasons = []
    if not clauses:
        return {"j1": False, "j2": False, "j3": False, "usable": False,
                "reasons": ["no clauses emitted"], "n_nodes": 0}

    def pos_at(idx):
        if idx is None or not (0 <= idx < len(pos_list)):
            return None
        return pos_list[idx]

    j1, j2, j3 = True, True, True
    total_nodes = 0
    for ci, clause in enumerate(clauses):
        pred = clause.get("predicate", {})
        pred_idx = pred.get("token_index")
        if pos_at(pred_idx) not in VERB_LIKE:
            j1 = False
            reasons.append(f"clause {ci}: J1 fail (predicate token_index={pred_idx} pos={pos_at(pred_idx)})")

        kind = clause.get("utterance_kind", "proposition")
        roles = clause.get("roles", [])
        subj_roles = [r for r in roles if r.get("relation") == "SUBJECT"]
        if kind == "proposition":
            ok = any(pos_at(r.get("token_index")) in NOMINAL_LIKE for r in subj_roles)
            if not ok:
                j2 = False
                reasons.append(f"clause {ci}: J2 fail (no NOUN/PROPN/PRON SUBJECT)")
        elif kind == "imperative":
            ok = bool(subj_roles) and all(
                r.get("token_index") is None or pos_at(r.get("token_index")) in NOMINAL_LIKE
                for r in subj_roles
            )
            if not ok:
                j2 = False
                reasons.append(f"clause {ci}: J2 fail (imperative: no real/synth SUBJECT)")
        # interjection clauses carry no SUBJECT requirement

        node_list = [pred] + roles
        total_nodes += len(node_list)
        seen = set()
        for n in node_list:
            idx = n.get("token_index")
            if idx is None:
                continue
            if not (0 <= idx < len(tokens)):
                j3 = False
                reasons.append(f"clause {ci}: J3 fail (token_index {idx} out of range)")
            elif idx in seen:
                j3 = False
                reasons.append(f"clause {ci}: J3 fail (duplicate token_index {idx})")
            else:
                seen.add(idx)

    content_n = n_content_tokens(tokens)
    if total_nodes > content_n:
        j3 = False
        reasons.append(f"J3 fail (total nodes {total_nodes} > content tokens {content_n})")

    usable = j1 and j2 and j3
    return {"j1": j1, "j2": j2, "j3": j3, "usable": usable, "reasons": reasons, "n_nodes": total_nodes}


# ---------------------------------------------------------------------------
# gold (teacher) tree -> the SAME node/clause dict shape beam_decode uses,
# so `judge_tree` applies unmodified (top-1 tree only, per the ledger's
# "gold -> top-1" convention).
# ---------------------------------------------------------------------------

def gold_top1_tree(record: dict) -> dict:
    trees = record.get("lattice", {}).get("trees", [])
    if not trees:
        return {"clauses": []}
    return normalize_gold_tree(record, trees[0])


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_node(role: str, node: dict, tokens: list) -> str:
    idx = node.get("token_index")
    tok = tokens[idx] if idx is not None and 0 <= idx < len(tokens) else "<none>"
    detail = node_detail_compact(node, {"tokens": tokens})
    return f"    {role}: {tok}({detail})"


def render_block(idx: int, sentence: str, tokens: list, pos: list, tree: dict, judge: dict,
                  note: str = "") -> str:
    lines = [f"[{idx}] {sentence}"]
    if note:
        lines.append(f"    NOTE: {note}")
    lines.append("  POS: " + " ".join(f"{t}/{p}" for t, p in zip(tokens, pos)))
    clauses = tree.get("clauses", []) if tree else []
    if not clauses:
        lines.append("  <no clauses emitted>")
    for ci, clause in enumerate(clauses):
        kind = clause.get("utterance_kind", "proposition")
        lines.append(f"  clause {ci} ({kind}):")
        lines.append(render_node("PREDICATE", clause.get("predicate", {}), tokens))
        for r in clause.get("roles", []):
            lines.append(render_node(r.get("relation", "?"), r, tokens))
    lines.append(f"  J1(predicate-ok)={judge['j1']}  J2(subject-ok)={judge['j2']}  "
                 f"J3(no-phantoms)={judge['j3']}  USABLE={judge['usable']}")
    if judge["reasons"]:
        for reason in judge["reasons"]:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


def nanmean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    t_start = time.time()
    print("loading checkpoint / usvs / gold ...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    pos_vocab = ckpt["pos_vocab"]
    role_vocab = ckpt["role_vocab"]
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=ckpt["d_axes"],
                             hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                             controller_hidden=ckpt["d_model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    usvs = load_usvs(str(USVS_DIR))

    if not FULL_PROBE_CSV.exists():
        print(f"ERROR: {FULL_PROBE_CSV} missing -- run probe_teacher_complement_build.py first")
        return 1
    pool = load_failure_pool()
    print(f"teacher-failed pool (outcomes={FAIL_OUTCOMES}, >=2 content tokens): {len(pool)}")
    print("pool by (outcome,bin):", Counter((r["outcome"], r["bin"]) for r in pool))

    complement = stratified_sample(pool, N_COMPLEMENT, SEED)
    print(f"sampled complement set: {len(complement)}  "
          f"by outcome={Counter(r['outcome'] for r in complement)}  "
          f"by bin={Counter(r['bin'] for r in complement)}")

    tok = SimpleTokenizer.build([r["sentence"] for r in complement],
                                 extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)

    # --- run encoder rank-1 on the complement set ---------------------------
    complement_results = []
    for i, row in enumerate(complement):
        rec = build_bare_record(parser, usvs, row["sentence"])
        feats = em.build_features(rec, usvs, pos_vocab, ckpt["hash_buckets"])
        t0 = time.perf_counter()
        forest = em.beam_decode(model, feats, beam_width=8, k=8, policy="model")
        enc_seconds = time.perf_counter() - t0
        top1 = forest[0] if forest else {"clauses": []}
        judge = judge_tree(top1, rec["pos"], rec["tokens"])
        complement_results.append({
            **row, "tokens": rec["tokens"], "pos": rec["pos"], "tree": top1,
            "judge": judge, "encoder_seconds": enc_seconds,
        })
        if (i + 1) % 20 == 0:
            print(f"  complement: {i+1}/{len(complement)} decoded")

    # --- control set: 50 teacher-parseable TEST-split sentences -------------
    records = load_gold(str(GOLD_PATH))
    train_recs, dev_recs, test_recs = stratified_split(records, cfg["seed"], cfg["n_train"],
                                                         cfg["n_dev"], cfg["n_test"])
    rng = random.Random(SEED)
    control_recs = list(test_recs)
    rng.shuffle(control_recs)
    control_recs = control_recs[:N_CONTROL]

    control_results = []
    for rec in control_recs:
        feats = em.build_features(rec, usvs, pos_vocab, ckpt["hash_buckets"])
        t0 = time.perf_counter()
        forest = em.beam_decode(model, feats, beam_width=8, k=8, policy="model")
        enc_seconds = time.perf_counter() - t0
        enc_top1 = forest[0] if forest else {"clauses": []}
        enc_judge = judge_tree(enc_top1, rec["pos"], rec["tokens"])
        gold_tree = gold_top1_tree(rec)
        gold_judge = judge_tree(gold_tree, rec["pos"], rec["tokens"])
        control_results.append({
            "sentence": rec["text"], "tokens": rec["tokens"], "pos": rec["pos"],
            "enc_tree": enc_top1, "enc_judge": enc_judge, "enc_seconds": enc_seconds,
            "gold_tree": gold_tree, "gold_judge": gold_judge,
        })

    # =========================================================================
    # tables
    # =========================================================================
    summary_lines = []

    def emit(s: str = "") -> None:
        print(s)
        summary_lines.append(s)

    emit("PART B -- teacher-complement probe: summary")
    emit("=" * 78)
    emit(f"complement set n={len(complement_results)}  control set n={len(control_results)}  "
         f"seed={SEED}")

    def usable_rate(results, judge_key="judge"):
        n = len(results)
        if n == 0:
            return float("nan"), 0, 0
        u = sum(1 for r in results if r[judge_key]["usable"])
        return u / n, u, n

    overall_rate, overall_u, overall_n = usable_rate(complement_results)
    emit(f"\nCOMPLEMENT usable-rate overall: {overall_u}/{overall_n} = {overall_rate:.1%}")

    emit("\nusable-rate by outcome bucket:")
    for outc in FAIL_OUTCOMES:
        sub = [r for r in complement_results if r["outcome"] == outc]
        rate, u, n = usable_rate(sub)
        emit(f"  {outc:<15} n={n:<3} usable={u:<3} rate={rate:.1%}" if n else f"  {outc:<15} n=0")

    emit("\nusable-rate by length bin:")
    for b in BINS:
        sub = [r for r in complement_results if r["bin"] == b]
        rate, u, n = usable_rate(sub)
        emit(f"  {b} n={n:<3} usable={u:<3} rate={rate:.1%}" if n else f"  {b} n=0")

    emit("\nusable-rate by outcome x length bin:")
    emit("  outcome         " + "  ".join(f"{b:>10}" for b in BINS))
    for outc in FAIL_OUTCOMES:
        cells = []
        for b in BINS:
            sub = [r for r in complement_results if r["outcome"] == outc and r["bin"] == b]
            rate, u, n = usable_rate(sub)
            cells.append(f"{u}/{n}={rate:.0%}" if n else "n=0")
        emit(f"  {outc:<15} " + "  ".join(f"{c:>10}" for c in cells))

    emit("\ncriterion pass rates (complement set):")
    for k, name in (("j1", "J1 predicate-ok"), ("j2", "J2 subject-ok"), ("j3", "J3 no-phantoms")):
        n = len(complement_results)
        p = sum(1 for r in complement_results if r["judge"][k]) / n if n else float("nan")
        emit(f"  {name:<18} {p:.1%}")

    # --- control ---
    gold_rate, gold_u, gold_n = usable_rate(control_results, "gold_judge")
    enc_easy_rate, enc_easy_u, enc_easy_n = usable_rate(control_results, "enc_judge")
    emit(f"\nCONTROL (n={len(control_results)}, teacher-parseable TEST-split):")
    emit(f"  gold/teacher tree usable-rate (judge ceiling): {gold_u}/{gold_n} = {gold_rate:.1%}")
    emit(f"  encoder rank-1 usable-rate on EASY text:       {enc_easy_u}/{enc_easy_n} = {enc_easy_rate:.1%}")
    for k, name in (("j1", "J1"), ("j2", "J2"), ("j3", "J3")):
        n = len(control_results)
        pg = sum(1 for r in control_results if r["gold_judge"][k]) / n if n else float("nan")
        pe = sum(1 for r in control_results if r["enc_judge"][k]) / n if n else float("nan")
        emit(f"    {name}: gold={pg:.1%}  encoder-on-easy={pe:.1%}")

    # --- timing: encoder vs teacher, by length bin (complement set only;
    #     teacher's recorded wall-clock IS its cost on this input, whether
    #     or not the attempt produced usable output) ---
    emit("\nencoder-vs-teacher wall-clock by length bin (complement set; teacher time is its "
         "OWN recorded attempt on the same failed sentence):")
    emit("  bin   n   enc_mean_s  enc_median_s   teacher_mean_s  teacher_median_s   speedup(median)")
    for b in BINS:
        sub = [r for r in complement_results if r["bin"] == b]
        if not sub:
            emit(f"  {b}     0")
            continue
        enc_s = [r["encoder_seconds"] for r in sub]
        tea_s = [r["parse_seconds"] for r in sub]
        em_, ed_ = statistics.mean(enc_s), statistics.median(enc_s)
        tm_, td_ = statistics.mean(tea_s), statistics.median(tea_s)
        speedup = (td_ / ed_) if ed_ else float("nan")
        emit(f"  {b}   {len(sub):<3} {em_:.4f}     {ed_:.4f}        {tm_:.4f}          {td_:.4f}"
             f"           {speedup:.1f}x")

    # =========================================================================
    # gate verdict
    # =========================================================================
    emit("\n" + "=" * 78)
    emit("GATE (audit): complement usable-rate >= 50% (and not far below the "
         "encoder-on-easy control) -> encoder earns its existence beyond parser "
         "distillation. < 25% -> parser distillation on this checkpoint. "
         "In between -> inconclusive.")
    if overall_rate != overall_rate:
        emit("  INCONCLUSIVE: complement usable-rate is NaN (empty sample).")
    elif overall_rate >= 0.50:
        gap = enc_easy_rate - overall_rate if enc_easy_rate == enc_easy_rate else float("nan")
        note = "" if (gap != gap or gap <= 0.15) else f"  (NOTE: {gap:.1%} below the easy-text control)"
        emit(f"  complement usable-rate {overall_rate:.1%} >= 50% -> PASSES.{note}")
    elif overall_rate < 0.25:
        emit(f"  complement usable-rate {overall_rate:.1%} < 25% -> FAILS: parser distillation "
             "on this checkpoint.")
    else:
        emit(f"  complement usable-rate {overall_rate:.1%} is between 25% and 50% -> INCONCLUSIVE.")

    OUT_SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nwrote summary -> {OUT_SUMMARY_TXT}")

    # =========================================================================
    # render all 100 + best5/worst5
    # =========================================================================
    def score_key(r):
        j = r["judge"]
        return j["j1"] + j["j2"] + j["j3"]

    ordered = sorted(range(len(complement_results)), key=lambda i: -score_key(complement_results[i]))
    best5 = ordered[:5]
    worst5 = ordered[-5:]

    probe_lines = []

    def pemit(s: str = "") -> None:
        probe_lines.append(s)

    pemit(f"PART B -- complement probe: all {len(complement_results)} rendered outputs")
    pemit(f"(5 best-looking / 5 worst-looking first, by index; seed={SEED})")
    pemit("=" * 78)

    pemit("\n----- 5 BEST-LOOKING -----\n")
    for rank, i in enumerate(best5):
        r = complement_results[i]
        note = f"rank {rank+1} best: {score_key(r)}/3 criteria pass ({r['outcome']}, bin {r['bin']})"
        pemit(render_block(i, r["sentence"], r["tokens"], r["pos"], r["tree"], r["judge"], note))
        pemit("")

    pemit("\n----- 5 WORST-LOOKING -----\n")
    for rank, i in enumerate(worst5):
        r = complement_results[i]
        note = f"rank {rank+1} worst: {score_key(r)}/3 criteria pass ({r['outcome']}, bin {r['bin']})"
        pemit(render_block(i, r["sentence"], r["tokens"], r["pos"], r["tree"], r["judge"], note))
        pemit("")

    pemit("\n----- ALL 100 (index order) -----\n")
    for i, r in enumerate(complement_results):
        note = f"{r['outcome']}, bin {r['bin']}, teacher_s={r['parse_seconds']:.3f}, enc_s={r['encoder_seconds']:.4f}"
        pemit(render_block(i, r["sentence"], r["tokens"], r["pos"], r["tree"], r["judge"], note))
        pemit("")

    OUT_PROBE_TXT.write_text("\n".join(probe_lines) + "\n", encoding="utf-8")
    print(f"wrote rendered probe -> {OUT_PROBE_TXT}")
    print(f"\ntotal wall-clock: {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
