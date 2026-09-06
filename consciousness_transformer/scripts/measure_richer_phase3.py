"""richer-gold PHASE 3 measurement: coverage (orig/phase1/phase3) AND
connective cleanliness (derivation-ii, reusing decoder-memory-step1's
methodology), on the same 198 matched sentences (the 200-sentence sample
minus the 2 pre-existing 20s-cap-hit sentences, identical across phase1 and
phase3 -- verified: same 2 sentences drop out both times, same order for
the other 198).

Usage: python scripts/measure_richer_phase3.py | tee runs/richer_phase3_report.txt
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct import decoder_trained as dt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ORIG_PATH = ROOT / "runs" / "encoder_gold_v2.jsonl"
PHASE1_PATH = ROOT / "runs" / "richer_gold_sample.jsonl"
PHASE3_PATH = ROOT / "runs" / "richer_gold_phase3_sample.jsonl"
N_SAMPLE = 200

FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "AUX", "PRON", "PART", "PUNCT"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}


def pos_class(tag: str) -> str:
    if tag in FUNCTION_POS:
        return "function"
    if tag in CONTENT_POS:
        return "content"
    return "other"


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def covered_indices(record: dict, tree: dict) -> set:
    return {n.token_index for n in dt.extract_nodes(record, tree)}


def role_relations(record: dict, tree: dict) -> Counter:
    c = Counter()
    for clause in tree.get("clauses", []):
        for role in clause["roles"]:
            c[role["relation"]] += 1
    return c


# ---------------------------------------------------------------------------
# Derivation (ii): single-token node + connectives (decoder-memory-step1)
# ---------------------------------------------------------------------------

def sorted_nodes(record: dict, tree: dict):
    nodes = dt.extract_nodes(record, tree)
    by_idx = defaultdict(list)
    for n in nodes:
        if n.relation not in by_idx[n.token_index]:
            by_idx[n.token_index].append(n.relation)
    Merged = dt.Node
    return [Merged(token_index=i, word=record["tokens"][i], relation="+".join(by_idx[i]), gtype=None)
            for i in sorted(by_idx)]


def derivation_connectives(record: dict, nodes: list) -> dict:
    tokens = record["tokens"]
    connectives = []
    if not nodes:
        if tokens:
            connectives.append((("BOS", "EOS"), list(range(len(tokens)))))
        return {"connectives": connectives}
    first = nodes[0]
    if first.token_index > 0:
        connectives.append((("BOS", first.relation), list(range(0, first.token_index))))
    for a, b in zip(nodes, nodes[1:]):
        gap = list(range(a.token_index + 1, b.token_index))
        if gap:
            connectives.append(((a.relation, b.relation), gap))
    last = nodes[-1]
    if last.token_index < len(tokens) - 1:
        connectives.append(((last.relation, "EOS"), list(range(last.token_index + 1, len(tokens)))))
    return {"connectives": connectives}


def connective_cleanliness(records: list) -> dict:
    conn_pos_counts = Counter()
    key_freq = Counter()
    key_strings = defaultdict(Counter)
    total_node_owned = 0
    total_conn_tokens = 0
    total_tokens = 0
    for r, tree in records:
        nodes = sorted_nodes(r, tree)
        res = derivation_connectives(r, nodes)
        total_node_owned += len(nodes)
        total_tokens += len(r["tokens"])
        pos = r["pos"]
        for key, idxs in res["connectives"]:
            total_conn_tokens += len(idxs)
            key_freq[key] += 1
            conn_str = " ".join(r["tokens"][i] for i in idxs)
            key_strings[key][conn_str] += 1
            for i in idxs:
                conn_pos_counts[pos_class(pos[i])] += 1
    return {
        "conn_pos_counts": conn_pos_counts,
        "key_freq": key_freq,
        "key_strings": key_strings,
        "total_node_owned": total_node_owned,
        "total_conn_tokens": total_conn_tokens,
        "total_tokens": total_tokens,
    }


def main() -> None:
    def p(s: str = "") -> None:
        print(s)

    orig_all = load_jsonl(ORIG_PATH)[:N_SAMPLE]
    phase1_all = load_jsonl(PHASE1_PATH)
    phase3_all = load_jsonl(PHASE3_PATH)

    phase1_texts = [r["text"] for r in phase1_all]
    phase3_texts = [r["text"] for r in phase3_all]
    assert phase1_texts == phase3_texts, "phase1/phase3 sample sentence sets/order differ -- not directly comparable"
    matched_set = set(phase1_texts)

    orig_by_text = {r["text"]: r for r in orig_all if r["text"] in matched_set}
    assert len(orig_by_text) == len(phase1_all), "orig gold missing a matched sentence"

    orig_recs = [(orig_by_text[t], orig_by_text[t]["lattice"]["trees"][0]) for t in phase1_texts]
    phase1_recs = [(r, r["lattice"]["trees"][0]) for r in phase1_all]
    phase3_recs = [(r, r["lattice"]["trees"][0]) for r in phase3_all]

    p("=" * 80)
    p("NAOMI richer-structure PHASE 3 -- coverage + connective-cleanliness report")
    p(f"Branch: richer-gold-phase3 (base: richer-gold-phase1)")
    p(f"Matched sentences (200-sample minus 2 pre-existing 20s cap-hits, identical set/order "
      f"across phase1 and phase3): {len(phase1_texts)}")
    p("=" * 80)
    p()

    # -----------------------------------------------------------------
    # (a) grounded-token coverage: orig vs phase1 vs phase3
    # -----------------------------------------------------------------
    p("-" * 80)
    p("STEP 3a -- GROUNDED-TOKEN COVERAGE (predicate-inclusive: dt.extract_nodes, best tree)")
    p("-" * 80)

    def coverage_stats(recs):
        total_tokens = sum(len(r["tokens"]) for r, _ in recs)
        total_covered = sum(len(covered_indices(r, t)) for r, t in recs)
        return total_covered, total_tokens

    orig_cov, total_tok = coverage_stats(orig_recs)
    p1_cov, _ = coverage_stats(phase1_recs)
    p3_cov, _ = coverage_stats(phase3_recs)

    p(f"ORIGINAL gold  : {orig_cov}/{total_tok} = {pct(orig_cov, total_tok):.2f}%")
    p(f"PHASE1 gold    : {p1_cov}/{total_tok} = {pct(p1_cov, total_tok):.2f}%  "
      f"(delta vs orig: {pct(p1_cov, total_tok) - pct(orig_cov, total_tok):+.2f}pp)")
    p(f"PHASE3 gold    : {p3_cov}/{total_tok} = {pct(p3_cov, total_tok):.2f}%  "
      f"(delta vs orig: {pct(p3_cov, total_tok) - pct(orig_cov, total_tok):+.2f}pp, "
      f"delta vs phase1: {pct(p3_cov, total_tok) - pct(p1_cov, total_tok):+.2f}pp)")
    p()

    # -----------------------------------------------------------------
    # (c) subordination clause count + POS composition of newly-covered
    #     tokens (phase3 minus phase1) -- reported before (b) since (b)'s
    #     example dump wants the subordination counts alongside it.
    # -----------------------------------------------------------------
    p("-" * 80)
    p("STEP 3c -- NEW SUBORDINATION CLAUSES + POS OF NEWLY-COVERED TOKENS (phase3 minus phase1)")
    p("-" * 80)

    n_subordinate_clauses_forest = 0
    for r in phase3_all:
        for links in r["lattice"]["discourse_links_per_tree"]:
            n_subordinate_clauses_forest += sum(1 for lk in links if lk["coordinator"] == "SUBORDINATE")

    n_sent_best_tree_subordinate = 0
    for r in phase3_all:
        best_links = r["lattice"]["discourse_links_per_tree"][0]
        n_sent_best_tree_subordinate += sum(1 for lk in best_links if lk["coordinator"] == "SUBORDINATE")

    p(f"SUBORDINATE-linked clauses across the WHOLE forest (all {sum(len(r['lattice']['trees']) for r in phase3_all)} "
      f"trees over all {len(phase3_all)} records -- a sentence can contribute several trees to the k=8 lattice): "
      f"{n_subordinate_clauses_forest}")
    p(f"SUBORDINATE-linked clauses in the BEST tree only (the tree coverage/connective numbers above actually use): "
      f"{n_sent_best_tree_subordinate}  (SUBORDINATION is genuinely rare in this successfully-parsed corpus -- "
      f"a separate top-1-hypothesis check found SUBORDINATION edges present in only 17/200 raw parses, of which "
      f"12 became a clause; the rest hit either no MODIFICATION edge to the real nested predicate/gap, or a "
      f"nested predicate whose graph-node index collided with the (separately, pre-existing) heuristic that "
      f"picks the primary clause -- see VERDICT)")

    new_tokens_pos = Counter()
    n_new_tokens = 0
    for (r1, t1), (r3, t3) in zip(phase1_recs, phase3_recs):
        assert r1["text"] == r3["text"]
        cov1 = covered_indices(r1, t1)
        cov3 = covered_indices(r3, t3)
        newly = cov3 - cov1
        for i in newly:
            new_tokens_pos[r3["pos"][i]] += 1
            n_new_tokens += 1
    p(f"newly-covered tokens (phase3 vs phase1, best tree): {n_new_tokens}")
    for tag, n in new_tokens_pos.most_common():
        p(f"  {tag:8s} {n:5d} ({pct(n, n_new_tokens):.1f}%)")
    p()

    # -----------------------------------------------------------------
    # (b) connective cleanliness: derivation (ii), orig vs phase1 vs phase3
    # -----------------------------------------------------------------
    p("-" * 80)
    p("STEP 3b -- CONNECTIVE CLEANLINESS (derivation-ii, decoder-memory-step1 methodology) *** THE GATE ***")
    p("-" * 80)
    p("step-1 baseline (decoder-memory-step1, 98-record TEST split of full encoder_gold_v2, "
      "ORIGINAL gold, no modifier/subordination extraction): "
      "function=61.4% content=38.6% other=0.0% of the raw connective gap (n=1845 connective tokens)")
    p()

    for label, recs in (("ORIGINAL", orig_recs), ("PHASE1", phase1_recs), ("PHASE3", phase3_recs)):
        res = connective_cleanliness(recs)
        total_conn_pos = sum(res["conn_pos_counts"].values())
        fn = pct(res["conn_pos_counts"]["function"], total_conn_pos)
        ct = pct(res["conn_pos_counts"]["content"], total_conn_pos)
        ot = pct(res["conn_pos_counts"]["other"], total_conn_pos)
        p(f"{label:10s} connective-token POS composition (n={total_conn_pos}): "
          f"function={fn:.1f}% content={ct:.1f}% other={ot:.1f}%   "
          f"(node-owned={res['total_node_owned']}/{res['total_tokens']} = "
          f"{pct(res['total_node_owned'], res['total_tokens']):.1f}%, "
          f"connective={res['total_conn_tokens']}/{res['total_tokens']} = "
          f"{pct(res['total_conn_tokens'], res['total_tokens']):.1f}%)")
        if label == "PHASE3":
            phase3_conn = res
    p()
    p(f"GATE CHECK: step-1 baseline content fraction = 38.6%. Is phase3's content fraction materially DOWN? "
      f"phase3 = {pct(phase3_conn['conn_pos_counts']['content'], sum(phase3_conn['conn_pos_counts'].values())):.1f}%")
    p()

    p("connective diversity, phase3 gold, top 10 most common (relation_a, relation_b) keys:")
    for key, freq in phase3_conn["key_freq"].most_common(10):
        strings = phase3_conn["key_strings"][key]
        top5 = strings.most_common(5)
        top5_str = ", ".join(f"{s!r}x{c}" for s, c in top5)
        p(f"  {key!s:40s} freq={freq:4d}  distinct_strings={len(strings):4d}  top5=[{top5_str}]")
    p()

    # -----------------------------------------------------------------
    # 8 before(orig)->phase1->phase3 examples
    # -----------------------------------------------------------------
    p("-" * 80)
    p("8 BEFORE(orig) -> PHASE1 -> PHASE3 EXAMPLES")
    p("-" * 80)

    def render_roles(record, tree):
        lines = []
        for cl in tree["clauses"]:
            roles_str = ", ".join(f"{r['relation']}={r['word']!r}" for r in cl["roles"])
            lines.append(f"    pred={cl['predicate']!r}  {roles_str}")
        return "\n".join(lines)

    shown = 0
    for (ro, to), (r1, t1), (r3, t3) in zip(orig_recs, phase1_recs, phase3_recs):
        cov3 = covered_indices(r3, t3)
        cov1 = covered_indices(r1, t1)
        if len(cov3) <= len(cov1):
            continue  # prefer examples where phase3 actually added something
        p(f"TEXT: {ro['text']!r}")
        p("  ORIG:")
        p(render_roles(ro, to))
        p("  PHASE1:")
        p(render_roles(r1, t1))
        p("  PHASE3:")
        p(render_roles(r3, t3))
        p()
        shown += 1
        if shown >= 8:
            break
    if shown < 8:
        p(f"(only {shown} examples where phase3 strictly added coverage beyond phase1 in this sample)")
    p()

    p("=" * 80)
    p("VERDICT")
    p("=" * 80)
    p("Coverage: orig 22.70% -> phase1 29.90% -> phase3 30.17% (+0.27pp over phase1, "
      "+7.47pp over orig). SUBORDINATION recursion adds only a THIN slice on top of phase1, "
      "because most transitive complementizer clauses in this corpus (\"mary believes that the "
      "cat sees the dog\") were ALREADY being recovered -- unlinked, mislabeled as an unrelated "
      "top-level fact -- by the pre-existing _secondary_fact_clauses fallback (any extra SUBJECT "
      "edge with a PP/OBJECT becomes its own fact clause, subordination-blind). The genuinely NEW "
      "coverage this phase adds is narrow: intransitive embedded clauses that fallback used to drop "
      "entirely (no PP/OBJECT to qualify), plus subject-relative clauses (whose predicate never had "
      "a SUBJECT edge in the first place, so the fallback never saw them at all). SUBORDINATION "
      "itself is also rare in this corpus's successfully-parsed structures (17/200 top-1 parses).")
    p()
    p("Connective cleanliness (THE gate): content fraction of the raw connective gap is "
      "UNCHANGED within noise -- orig 39.4%, phase1 38.9%, phase3 38.9%, vs the step-1 baseline's "
      "38.6%. Not materially down. The new subordinate clauses pull a handful of content words "
      "(verbs, a noun) out of the gap, but not enough to move the aggregate needle: the gap is "
      "still ~39% content words, same order of magnitude as before any richer-structure work.")


if __name__ == "__main__":
    main()
