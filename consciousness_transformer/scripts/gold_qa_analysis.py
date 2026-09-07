"""Gold-QA analysis (read-only): per-source parse quality + forest-width /
margin-pruning characterization for the corpus-expand gold builder.

This does NOT modify scripts/build_encoder_gold_v2.py or its default
behavior. It reuses that module's own helper functions (``build_tree``,
``ground_word``, ``_IndexMatcher``, ``load_corpus``) and the same parser
entrypoint (``ParserInputEncoder._parse_topk_one``) so every number here
reflects exactly what the shipped builder would do -- the pruning
experiment (Part B.3) is a pure re-filter of the SAME per-hypothesis
scores the builder already computes, done in this script's own process,
never by editing the shipped file.

PART A -- per source (4 new fairy-tale sources + 2 control sources),
sample ~40 sentences (fixed seed) and report:
  1. outcome yield % (ok / no-hypothesis / grounding-fail / cap-hit / parse-failed)
  2. median / p90 sentence length in tokens
  3. predicate POS distribution (top-ranked tree only) -- VERB good, AUX/
     CCONJ/other likely junk
  4. role function-word rate excluding PRON (top-ranked tree only) -- role
     fillers whose POS is a function-word class {DET, ADP, CCONJ, SCONJ,
     PART, AUX}, out of all roles whose POS != PRON. NOTE: no pre-existing
     "gold-audit" script/metric with this exact definition was found
     anywhere in git history (checked via `git log --all` + `git grep`
     across all commits for "gold_audit", "function.word", "suspect role",
     etc.) -- this metric is defined fresh in this script, matching the
     task's specification as closely as possible. This is flagged again
     in the written report.

PART B -- forest width, sampled from the FULL real_*.txt corpus (all
sources, i.e. exactly ``build_encoder_gold_v2.load_corpus()``'s scope --
noted explicitly since Part A samples per-source files individually):
  1. current trees-per-sentence distribution (post clause-level dedup,
     exactly as build_encoder_gold_v2 would store them)
  2. margin-based pruning re-filter at several M thresholds, re-using the
     SAME per-hypothesis structural scores (no parser change, no re-parse)
  3. dumped examples (unambiguous / genuinely-ambiguous / spurious-pruned)

Usage: python scripts/gold_qa_analysis.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_encoder_gold_v2 as G  # noqa: E402
from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
# NOTE: `src.parser.quantum_parser` (quantum_parser's own package) is only
# importable once nsm_ct.input_encoder.ParserInputEncoder._init_adapter has
# inserted quantum_parser's repo root onto sys.path (same as
# build_encoder_gold_v2.build_record's own local import) -- so this import
# is deferred to inside parse_sentence(), after a ParserInputEncoder exists.

SEED = 20260907
SAMPLE_N_PART_A = 40
SAMPLE_N_PART_B = 200
MARGINS = [0.0, 0.02, 0.05, 0.10]

# Function-word POS classes for the role-filler metric. PRON is
# deliberately excluded from BOTH the numerator and denominator (a
# pronoun filling a role is normal and not what this metric is trying to
# catch) -- see module docstring.
FUNC_WORD_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "AUX"}

SOURCES: Dict[str, List[str]] = {
    "grimms_fairy_tales (NEW)": ["real_gutenberg_grimms_fairy_tales.txt"],
    "andersen_fairy_tales (NEW)": ["real_gutenberg_andersen_fairy_tales.txt"],
    "jacobs_english_fairy_tales (NEW)": ["real_gutenberg_jacobs_english_fairy_tales.txt"],
    "mcguffey_first_reader (NEW)": ["real_gutenberg_mcguffey_first_reader.txt"],
    "busterbear+burgess_more (CONTROL)": [
        "real_gutenberg_busterbear.txt", "real_gutenberg_burgess_more.txt",
    ],
    "alice (CONTROL)": ["real_gutenberg_alice.txt"],
}

CORPUS_DIR = ROOT / "data" / "corpus"


def load_source_sentences(files: List[str]) -> List[str]:
    from nsm_ct.corpus import iter_sentences
    seen, out = set(), []
    for fname in files:
        text = (CORPUS_DIR / fname).read_text(encoding="utf-8")
        for s in iter_sentences(text):
            s = s.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def pctl(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def clause_pos_info(tokens: List[str], pos: List[str], clauses) -> List[Dict]:
    """POS-tag the predicate + each role filler of each clause, using the
    SAME left-to-right consume-on-match heuristic build_encoder_gold_v2
    uses for role token_index (nsm_ct.clause._IndexMatcher-equivalent, via
    G._IndexMatcher), extended to also index the predicate token (which
    the shipped builder does not need since it never emits a predicate
    token_index). Analysis-only; does not affect any emitted gold record.
    """
    matcher = G._IndexMatcher(tokens)
    info = []
    for cl in clauses:
        p_idx = matcher.match(cl.predicate)
        p_pos = pos[p_idx] if p_idx is not None else None
        roles = []
        for relation, arg in cl.args:
            r_idx = matcher.match(arg.token)
            r_pos = pos[r_idx] if r_idx is not None else None
            roles.append({"relation": relation, "word": arg.token, "pos": r_pos})
        info.append({"predicate": cl.predicate, "predicate_pos": p_pos, "roles": roles})
    return info


class ParsedSentence:
    """One sentence's full parse-and-extract result, computed once and
    reused by both Part A and Part B (they can overlap)."""

    __slots__ = ("sentence", "outcome", "n_raw", "tokens", "pos",
                 "tree_dicts", "tree_scores", "tree_pos_info")

    def __init__(self, sentence):
        self.sentence = sentence
        self.outcome = None
        self.n_raw = 0
        self.tokens: List[str] = []
        self.pos: List[str] = []
        self.tree_dicts: List[Dict] = []
        self.tree_scores: List[float] = []
        self.tree_pos_info: List[List[Dict]] = []  # per tree, per clause


def parse_sentence(usvs, parser: ParserInputEncoder, sentence: str) -> ParsedSentence:
    """Mirrors build_encoder_gold_v2.build_record's exact control flow
    (same outcome tags, same TOP_K/caps, same dedup-by-JSON), but ALSO
    keeps the per-tree structural score (first-occurrence score among the
    raw hypotheses that deduped into it -- the highest, since
    chart.hypotheses is score-sorted) and a POS breakdown, needed for
    Parts A and B. No behavior of build_encoder_gold_v2.py is changed;
    this is an independent read of the same parser API."""
    from src.parser.quantum_parser import ParseResourceExceeded  # local import (qp_root on sys.path)
    ps = ParsedSentence(sentence)
    try:
        graphs, scores, _margin = parser._parse_topk_one(
            sentence, k=G.TOP_K,
            max_hypotheses=G.CORPUS_MAX_HYPOTHESES, max_seconds=G.CORPUS_MAX_PARSE_SECONDS,
        )
    except ParseResourceExceeded:
        ps.outcome = "cap-hit"
        return ps
    except Exception:
        ps.outcome = "parse-failed"
        return ps

    ps.n_raw = len(graphs)
    if not graphs:
        ps.outcome = "no-hypothesis"
        return ps

    words = parser._tag(sentence)
    ps.tokens = [w.text for w in words]
    ps.pos = [w.pos.name for w in words]

    seen = set()
    for graph, score in zip(graphs, scores):
        built = G.build_tree(usvs, ps.tokens, graph)
        if built is None:
            continue
        tree, _links = built
        key = json.dumps(tree, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        clauses, _ = extract_discourse(graph)
        ps.tree_dicts.append(tree)
        ps.tree_scores.append(score)
        ps.tree_pos_info.append(clause_pos_info(ps.tokens, ps.pos, clauses))

    if not ps.tree_dicts:
        ps.outcome = "grounding-fail"
        return ps

    ps.outcome = "ok"
    return ps


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------

def run_part_a(usvs, parser, cache: Dict[str, ParsedSentence]) -> Tuple[str, Dict[str, ParsedSentence]]:
    lines = []
    lines.append("=" * 78)
    lines.append("PART A -- per-source parse quality (n=%d sampled per source, seed=%d)" % (SAMPLE_N_PART_A, SEED))
    lines.append("=" * 78)

    per_source_samples: Dict[str, List[str]] = {}
    rng = random.Random(SEED)
    for name, files in SOURCES.items():
        pool = load_source_sentences(files)
        n = min(SAMPLE_N_PART_A, len(pool))
        sample = rng.sample(pool, n)
        per_source_samples[name] = sample
        lines.append(f"\n-- {name}: pool={len(pool)} unique sentences, sampled={n} --")

    table_rows = []
    for name, sample in per_source_samples.items():
        outcomes = Counter()
        lengths = []
        pred_pos = Counter()
        role_pos_func = 0
        role_pos_total = 0
        for si, s in enumerate(sample):
            if s not in cache:
                cache[s] = parse_sentence(usvs, parser, s)
            if (len(cache)) % 20 == 0:
                print(f"  ... [part A] {len(cache)} sentences parsed so far ({time.time()-_T0:.0f}s)", flush=True)
            ps = cache[s]
            outcomes[ps.outcome] += 1
            lengths.append(len(s.split()))
            if ps.outcome == "ok":
                # top-ranked tree only
                top = ps.tree_pos_info[0]
                for cl in top:
                    pred_pos[cl["predicate_pos"] or "UNK"] += 1
                    for r in cl["roles"]:
                        if r["pos"] == "PRON":
                            continue
                        role_pos_total += 1
                        if r["pos"] in FUNC_WORD_POS:
                            role_pos_func += 1

        n = len(sample)
        ok = outcomes.get("ok", 0)
        yield_pct = 100.0 * ok / n if n else 0.0
        med_len = statistics.median(lengths) if lengths else float("nan")
        p90_len = pctl(lengths, 0.90)
        pred_total = sum(pred_pos.values())
        verb_pct = 100.0 * pred_pos.get("VERB", 0) / pred_total if pred_total else float("nan")
        aux_pct = 100.0 * pred_pos.get("AUX", 0) / pred_total if pred_total else float("nan")
        junk_pct = 100.0 * sum(v for k, v in pred_pos.items() if k not in ("VERB",)) / pred_total if pred_total else float("nan")
        func_rate = 100.0 * role_pos_func / role_pos_total if role_pos_total else float("nan")

        table_rows.append({
            "name": name, "n": n, "outcomes": dict(outcomes), "yield_pct": yield_pct,
            "med_len": med_len, "p90_len": p90_len,
            "pred_pos": dict(pred_pos), "verb_pct": verb_pct, "aux_pct": aux_pct,
            "non_verb_pred_pct": junk_pct,
            "role_func_rate_pct": func_rate, "role_pos_total": role_pos_total,
        })

    lines.append("\n" + "-" * 78)
    lines.append("PER-SOURCE TABLE")
    lines.append("-" * 78)
    hdr = f"{'source':38s} {'yield%':>7s} {'med_len':>8s} {'p90_len':>8s} {'VERB%':>7s} {'AUX%':>6s} {'nonVERB%':>9s} {'roleFunc%':>10s}"
    lines.append(hdr)
    for row in table_rows:
        lines.append(
            f"{row['name']:38s} {row['yield_pct']:7.1f} {row['med_len']:8.1f} {row['p90_len']:8.1f} "
            f"{row['verb_pct']:7.1f} {row['aux_pct']:6.1f} {row['non_verb_pred_pct']:9.1f} "
            f"{row['role_func_rate_pct']:10.1f}"
        )
    lines.append("")
    lines.append("Outcome breakdown per source (raw counts):")
    for row in table_rows:
        lines.append(f"  {row['name']:38s} {row['outcomes']}")
    lines.append("")
    lines.append("Predicate POS breakdown per source (top tree only, raw counts):")
    for row in table_rows:
        lines.append(f"  {row['name']:38s} {row['pred_pos']}")

    return "\n".join(lines), table_rows


# ---------------------------------------------------------------------------
# PART B
# ---------------------------------------------------------------------------

def tree_summary(tree: Dict) -> str:
    parts = []
    for cl in tree["clauses"]:
        roles = ", ".join(f"{r['relation']}={r['word']}" for r in cl["roles"])
        parts.append(f"{cl['predicate']}({roles})")
    return " | ".join(parts)


def run_part_b(usvs, parser, cache: Dict[str, ParsedSentence]) -> str:
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append(f"PART B -- forest width (n={SAMPLE_N_PART_B} sampled from FULL real_*.txt corpus, "
                 f"i.e. build_encoder_gold_v2.load_corpus() scope, seed={SEED})")
    lines.append("=" * 78)

    all_sentences = G.load_corpus()
    rng = random.Random(SEED + 1)
    sample = rng.sample(all_sentences, min(SAMPLE_N_PART_B, len(all_sentences)))

    outcomes = Counter()
    current_dist = Counter()  # n_trees -> count of sentences ("ok" sentences only)
    pruned_dist = {M: Counter() for M in MARGINS}
    all_margins_top12 = []  # top1-top2 margin, for sentences with >=2 trees
    ok_records = []  # (sentence, ParsedSentence) for "ok" outcomes, for example-mining

    for s in sample:
        if s not in cache:
            cache[s] = parse_sentence(usvs, parser, s)
        if (len(cache)) % 20 == 0:
            print(f"  ... [part B] {len(cache)} sentences parsed so far ({time.time()-_T0:.0f}s)", flush=True)
        ps = cache[s]
        outcomes[ps.outcome] += 1
        if ps.outcome != "ok":
            continue
        n = len(ps.tree_scores)
        current_dist[n] += 1
        ok_records.append((s, ps))
        if n > 1:
            all_margins_top12.append(ps.tree_scores[0] - ps.tree_scores[1])
        for M in MARGINS:
            kept = 1
            for i in range(1, n):
                if ps.tree_scores[0] - ps.tree_scores[i] <= M:
                    kept += 1
                else:
                    break  # scores are descending; once margin exceeds M it won't come back under
            pruned_dist[M][kept] += 1

    n_ok = sum(current_dist.values())
    lines.append(f"\nOutcome counts over the {len(sample)}-sentence sample: {dict(outcomes)}")
    lines.append(f"'ok' (>=1 tree emitted) records: {n_ok}")

    def dist_str(dist: Counter) -> str:
        one = dist.get(1, 0)
        two = dist.get(2, 0)
        three_plus = sum(v for k, v in dist.items() if k >= 3)
        total = sum(dist.values())
        mean = sum(k * v for k, v in dist.items()) / total if total else float("nan")
        return (f"1-tree={one} ({100.0*one/total:.1f}%)  2-tree={two} ({100.0*two/total:.1f}%)  "
                f"3+-tree={three_plus} ({100.0*three_plus/total:.1f}%)  mean_trees={mean:.2f}  "
                f"full_hist={dict(sorted(dist.items()))}")

    lines.append("\n-- CURRENT build (build_encoder_gold_v2's own clause-level dedup, no extra pruning) --")
    lines.append(dist_str(current_dist))

    lines.append(f"\nTop1-top2 structural-score scale, over the {len(all_margins_top12)} multi-tree "
                 f"'ok' sentences in this sample:")
    if all_margins_top12:
        lines.append(f"  min={min(all_margins_top12):.4f}  median={statistics.median(all_margins_top12):.4f}  "
                     f"p90={pctl(all_margins_top12, 0.90):.4f}  max={max(all_margins_top12):.4f}")
        n_exact_tie = sum(1 for m in all_margins_top12 if m <= 1e-9)
        lines.append(f"  exact ties (margin<=1e-9): {n_exact_tie}/{len(all_margins_top12)} "
                     "(cf. nsm_ct.corpus._AMBIGUITY_MARGIN=1e-6, the codebase's own precedent "
                     "for a 'genuine tie' threshold on this same [0,1] structural score)")
    lines.append(f"\nMargin thresholds tested (chosen after inspecting the above scale): {MARGINS}")

    for M in MARGINS:
        lines.append(f"\n-- pruned @ M={M} (keep hyp i>=1 only while cumulative top1-hypI margin <= M) --")
        lines.append(dist_str(pruned_dist[M]))

    # -------------------------------------------------------------
    # Example mining
    # -------------------------------------------------------------
    RECO_M = 0.02  # picked after seeing the distributions above; see report prose
    lines.append(f"\n{'-'*78}\nEXAMPLES (recommended M={RECO_M} used for the 'pruning fixes it' cases)\n{'-'*78}")

    def simplify(tree):
        return tuple((cl["predicate"], tuple((r["relation"], r["word"]) for r in cl["roles"]))
                     for cl in tree["clauses"])

    unambiguous = [(s, ps) for s, ps in ok_records if len(ps.tree_scores) == 1]
    ambiguous_keep = []   # still >=2 trees after RECO_M pruning
    spurious_fixed = []   # >=2 currently, ==1 after RECO_M pruning
    for s, ps in ok_records:
        n = len(ps.tree_scores)
        if n < 2:
            continue
        kept = 1
        for i in range(1, n):
            if ps.tree_scores[0] - ps.tree_scores[i] <= RECO_M:
                kept += 1
            else:
                break
        if kept >= 2:
            ambiguous_keep.append((s, ps, kept))
        else:
            spurious_fixed.append((s, ps))

    def fmt_example(tag, s, ps, extra=""):
        out = [f"\n[{tag}] {extra}", f"  sentence: {s}", f"  n_tokens={len(ps.tokens)}  current_n_trees={len(ps.tree_scores)}",
               f"  scores={['%.4f' % x for x in ps.tree_scores]}"]
        for i, tree in enumerate(ps.tree_dicts):
            out.append(f"    tree[{i}] score={ps.tree_scores[i]:.4f}: {tree_summary(tree)}")
        return "\n".join(out)

    if unambiguous:
        s, ps = unambiguous[0]
        lines.append(fmt_example("(i) UNAMBIGUOUS -- correctly 1 tree", s, ps))

    if ambiguous_keep:
        # prefer one where the two top trees look structurally distinct (not just role-order)
        ambiguous_keep.sort(key=lambda t: -(t[1].tree_scores[0] - t[1].tree_scores[min(1, len(t[1].tree_scores)-1)]))
        s, ps, kept = ambiguous_keep[len(ambiguous_keep)//2] if len(ambiguous_keep) > 1 else ambiguous_keep[0]
        margin = ps.tree_scores[0] - ps.tree_scores[1]
        lines.append(fmt_example("(ii) GENUINELY AMBIGUOUS -- keeps 2+ even after pruning",
                                  s, ps, extra=f"margin(top1,top2)={margin:.4f} > M={RECO_M}"))

    for j, (s, ps) in enumerate(spurious_fixed[:4], start=3):
        margin = ps.tree_scores[0] - ps.tree_scores[1]
        rel = "<=" if margin <= RECO_M else ">"
        why = ("near-tie noise (margin within M, but pruning still drops it)" if margin <= RECO_M
               else "clearly lower-scoring extra hypothesis (margin exceeds M -> dropped as noise, "
                    "not a competing genuine reading)")
        lines.append(fmt_example(f"(iii+{j-3}) SPURIOUS EXTRAS -- current={len(ps.tree_scores)} trees, "
                                  f"pruning@M={RECO_M} -> 1",
                                  s, ps, extra=f"margin(top1,top2)={margin:.4f} {rel} M={RECO_M} ({why})"))

    lines.append(f"\nCounts feeding the example mining: unambiguous(1-tree)={len(unambiguous)}, "
                 f"genuinely-ambiguous-after-pruning={len(ambiguous_keep)}, "
                 f"spurious-collapsed-by-pruning={len(spurious_fixed)}, "
                 f"(out of {n_ok} 'ok' sample sentences)")

    return "\n".join(lines)


_T0 = time.time()


def main():
    global _T0
    t0 = time.time()
    _T0 = t0
    print(f"loading USVS from {G.USVS_DIR} ...", flush=True)
    usvs = load_usvs(G.USVS_DIR)
    print(f"USVS loaded ({time.time()-t0:.0f}s)", flush=True)

    all_sentences = G.load_corpus()
    tok = SimpleTokenizer.build(all_sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1
    print(f"parser ready ({time.time()-t0:.0f}s); full corpus = {len(all_sentences)} unique sentences", flush=True)

    cache: Dict[str, ParsedSentence] = {}

    part_a_text, _rows = run_part_a(usvs, parser, cache)
    print(part_a_text, flush=True)
    print(f"\n[{time.time()-t0:.0f}s] Part A done, {len(cache)} sentences parsed so far", flush=True)

    part_b_text = run_part_b(usvs, parser, cache)
    print(part_b_text, flush=True)
    print(f"\n[{time.time()-t0:.0f}s] Part B done, {len(cache)} sentences parsed total (cumulative)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
