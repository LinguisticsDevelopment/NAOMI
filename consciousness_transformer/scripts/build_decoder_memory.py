"""Decoder-v2 STEP 1 (dev/DECODER_MEMORY_DESIGN.md's "Build order" item 1):
span-alignment data layer + coverage/viability report. NO training, NO model.

For every TEST-split gold record's top tree, derive node/span alignments two
ways and measure whether the resulting CONNECTIVE gaps are a small, mostly
function-word closed set (derivation-v2 viable) or still content-laden (the
76% uncovered-surface gap just relocated, approach needs rethink):

  (i)  NEAREST-NODE ATTACH -- each node absorbs a token-block (itself + every
       uncovered token nearer to it than to any other node).
  (ii) SINGLE-TOKEN NODE + CONNECTIVES -- each node owns only its own token;
       every maximal uncovered run between two nodes is a connective keyed on
       (left.relation, right.relation) (BOS/EOS at the sentence edges).

Usage:
    python scripts/build_decoder_memory.py | tee runs/decoder_memory_step1.txt
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct import decoder_trained as dt  # noqa: E402

FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "AUX", "PRON", "PART", "PUNCT"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

GOLD_PATH = Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"


def load_gold(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_test_split(records: list) -> list:
    """Reproduce train_encoder.py's default full-run stratified split
    (seed=0, n_train=788/n_dev=98/n_test=98) if the gold file is large
    enough; otherwise fall back to using every record (labeled as such)."""
    n_train, n_dev, n_test = 788, 98, 98
    if len(records) >= n_train + n_dev + n_test:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_encoder import stratified_split  # noqa: E402

        _, _, test = stratified_split(records, seed=0, n_train=n_train, n_dev=n_dev, n_test=n_test)
        return test, True
    return records, False


def pos_class(tag: str) -> str:
    if tag in FUNCTION_POS:
        return "function"
    if tag in CONTENT_POS:
        return "content"
    return "other"


MergedNode = namedtuple("MergedNode", ["token_index", "relation"])


def sorted_nodes(record: dict, tree: dict):
    """extract_nodes only sorts WITHIN a clause; a multi-clause forest needs
    one global left-to-right walk over the sentence for span/connective
    derivation. A multi-clause tree can also share one physical token across
    two clause roles (e.g. an elided-coordination predicate) -- extract_nodes
    then yields two Node objects with the same token_index. Since both
    derivations are a walk over PHYSICAL token positions (not node objects),
    such duplicates are merged into a single position, whose relation is the
    ordered union (e.g. "PREDICATE+PREDICATE"/"SUBJECT+OBJECT"), so each
    surface token is counted exactly once."""
    nodes = dt.extract_nodes(record, tree)
    by_idx = defaultdict(list)
    for n in nodes:
        if n.relation not in by_idx[n.token_index]:
            by_idx[n.token_index].append(n.relation)
    return [MergedNode(token_index=i, relation="+".join(by_idx[i])) for i in sorted(by_idx)]


def derivation_nearest(record: dict, nodes: list) -> dict:
    """(i) NEAREST-NODE ATTACH: each uncovered token joins the nearest node's
    span by token distance; ties go to the FOLLOWING node."""
    tokens = record["tokens"]
    pos = record["pos"]
    covered = {n.token_index for n in nodes}
    node_idxs = [n.token_index for n in nodes]

    # span[i] = list of absorbed (non-head) token indices for node i
    spans = {n.token_index: [] for n in nodes}
    for i in range(len(tokens)):
        if i in covered:
            continue
        best_node = None
        best_dist = None
        for ni in node_idxs:
            d = abs(i - ni)
            if best_dist is None or d < best_dist or (d == best_dist and ni > best_node):
                # d < best_dist: strictly closer wins.
                # d == best_dist and ni > best_node: tie -> prefer the FOLLOWING node
                #   (larger token_index among the two equidistant candidates).
                best_dist = d
                best_node = ni
        spans[best_node].append(i)

    span_lengths = [1 + len(spans[ni]) for ni in node_idxs]  # head token + absorbed
    absorbed_pos = []
    bleed_count = 0
    for n in nodes:
        for i in spans[n.token_index]:
            cls = pos_class(pos[i])
            absorbed_pos.append(cls)
            if cls == "content" and abs(i - n.token_index) > 1:
                bleed_count += 1

    return {
        "span_lengths": span_lengths,
        "absorbed_pos": absorbed_pos,
        "bleed_count": bleed_count,
        "n_absorbed": len(absorbed_pos),
        "spans_by_head": spans,
    }


def derivation_connectives(record: dict, nodes: list) -> dict:
    """(ii) SINGLE-TOKEN NODE + CONNECTIVES: each node owns only its own
    token; maximal uncovered runs between consecutive nodes (or sentence
    edges) become relation-pair-keyed connectives."""
    tokens = record["tokens"]
    pos = record["pos"]
    connectives = []  # (key, [token indices])

    if not nodes:
        if tokens:
            connectives.append((("BOS", "EOS"), list(range(len(tokens)))))
        return {"connectives": connectives, "n_node_owned": 0}

    # leading run
    first = nodes[0]
    if first.token_index > 0:
        connectives.append((("BOS", first.relation), list(range(0, first.token_index))))

    # between consecutive nodes
    for a, b in zip(nodes, nodes[1:]):
        gap = list(range(a.token_index + 1, b.token_index))
        if gap:
            connectives.append(((a.relation, b.relation), gap))

    # trailing run
    last = nodes[-1]
    if last.token_index < len(tokens) - 1:
        connectives.append(((last.relation, "EOS"), list(range(last.token_index + 1, len(tokens)))))

    return {"connectives": connectives, "n_node_owned": len(nodes)}


def render_example_nearest(record: dict, nodes: list, result: dict) -> str:
    tokens = record["tokens"]
    lines = [f"  TEXT: {record['text']!r}"]
    for n in nodes:
        block_idxs = sorted([n.token_index] + result["spans_by_head"][n.token_index])
        block = " ".join(tokens[i] for i in block_idxs)
        lines.append(f"    [{n.relation:10s}] head={tokens[n.token_index]!r:20s} block={block!r}")
    return "\n".join(lines)


def render_example_connectives(record: dict, nodes: list, result: dict) -> str:
    tokens = record["tokens"]
    lines = [f"  TEXT: {record['text']!r}"]
    seq = []
    for n in nodes:
        seq.append(f"[{n.relation}:{tokens[n.token_index]!r}]")
    conn_by_pos = {}
    for key, idxs in result["connectives"]:
        conn_str = " ".join(tokens[i] for i in idxs)
        conn_by_pos[key] = conn_str
    lines.append(f"    nodes: {' '.join(seq)}")
    for key, txt in conn_by_pos.items():
        lines.append(f"    connective {key}: {txt!r}")
    return "\n".join(lines)


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def main() -> None:
    def p(s: str = "") -> None:
        print(s)

    records = load_gold(GOLD_PATH)
    p(f"loaded {len(records)} gold records from {GOLD_PATH}")
    test_recs, reproduced = get_test_split(records)
    p(f"test split: {len(test_recs)} records (reproduced via train_encoder.stratified_split: {reproduced})")
    p()

    # ------------------------------------------------------------------
    # Original (v1) coverage gap, as a ceiling reference.
    # ------------------------------------------------------------------
    orig_uncovered_pos = []
    total_tokens = 0
    total_covered = 0
    usable_recs = []
    for r in test_recs:
        trees = r.get("lattice", {}).get("trees") or []
        if not trees:
            continue
        tree = trees[0]
        nodes = sorted_nodes(r, tree)
        usable_recs.append((r, tree, nodes))
        covered = {n.token_index for n in nodes}
        total_tokens += len(r["tokens"])
        total_covered += len(covered)
        for i, tag in enumerate(r["pos"]):
            if i not in covered:
                orig_uncovered_pos.append(pos_class(tag))

    p(f"usable test records (have >=1 tree): {len(usable_recs)} / {len(test_recs)}")
    p(f"ORIGINAL grounded-structure coverage: {total_covered}/{total_tokens} tokens "
      f"({pct(total_covered, total_tokens):.1f}%) -- uncovered gap = {pct(total_tokens - total_covered, total_tokens):.1f}%")
    orig_content = sum(1 for c in orig_uncovered_pos if c == "content")
    orig_function = sum(1 for c in orig_uncovered_pos if c == "function")
    orig_other = sum(1 for c in orig_uncovered_pos if c == "other")
    p(f"ORIGINAL uncovered-token POS composition (the raw 76% gap, ceiling reference): "
      f"content={pct(orig_content, len(orig_uncovered_pos)):.1f}% "
      f"function={pct(orig_function, len(orig_uncovered_pos)):.1f}% "
      f"other={pct(orig_other, len(orig_uncovered_pos)):.1f}% "
      f"(n={len(orig_uncovered_pos)})")
    p()

    # ------------------------------------------------------------------
    # Derivation (i): nearest-node attach
    # ------------------------------------------------------------------
    all_span_lengths = []
    all_absorbed_pos = []
    total_bleed = 0
    total_absorbed = 0
    nearest_examples = []
    nearest_assigned_tokens = 0

    for r, tree, nodes in usable_recs:
        if not nodes:
            continue
        res = derivation_nearest(r, nodes)
        all_span_lengths.extend(res["span_lengths"])
        all_absorbed_pos.extend(res["absorbed_pos"])
        total_bleed += res["bleed_count"]
        total_absorbed += res["n_absorbed"]
        nearest_assigned_tokens += sum(res["span_lengths"])
        if len(nearest_examples) < 10:
            nearest_examples.append(render_example_nearest(r, nodes, res))

    p("=" * 78)
    p("DERIVATION (i): NEAREST-NODE ATTACH (node owns a block)")
    p("=" * 78)
    if all_span_lengths:
        p(f"node-span length distribution (n={len(all_span_lengths)} spans): "
          f"min={min(all_span_lengths)} "
          f"median={statistics.median(all_span_lengths):.1f} "
          f"mean={statistics.mean(all_span_lengths):.2f} "
          f"max={max(all_span_lengths)}")
    absorbed_content = sum(1 for c in all_absorbed_pos if c == "content")
    absorbed_function = sum(1 for c in all_absorbed_pos if c == "function")
    absorbed_other = sum(1 for c in all_absorbed_pos if c == "other")
    p(f"ABSORBED (non-head) token POS composition (n={total_absorbed}): "
      f"function={pct(absorbed_function, total_absorbed):.1f}% "
      f"content={pct(absorbed_content, total_absorbed):.1f}% "
      f"other={pct(absorbed_other, total_absorbed):.1f}%")
    p(f"BLEED RATE (absorbed content-POS token whose nearest node is >1 away / all absorbed): "
      f"{pct(total_bleed, total_absorbed):.1f}%  ({total_bleed}/{total_absorbed})")
    p()
    p("-- 10 example sentences (derivation i) --")
    for ex in nearest_examples:
        p(ex)
        p()

    # ------------------------------------------------------------------
    # Derivation (ii): single-token node + connectives
    # ------------------------------------------------------------------
    conn_pos_counts = Counter()  # 'function'/'content'/'other' -> token count
    key_freq = Counter()  # key -> number of connective RUNS with that key
    key_strings = defaultdict(Counter)  # key -> Counter(connective string -> count)
    conn_examples = []
    total_node_owned_tokens = 0
    total_connective_tokens = 0
    total_all_tokens = 0

    for r, tree, nodes in usable_recs:
        res = derivation_connectives(r, nodes)
        total_node_owned_tokens += res["n_node_owned"]
        total_all_tokens += len(r["tokens"])
        pos = r["pos"]
        for key, idxs in res["connectives"]:
            total_connective_tokens += len(idxs)
            key_freq[key] += 1
            conn_str = " ".join(r["tokens"][i] for i in idxs)
            key_strings[key][conn_str] += 1
            for i in idxs:
                conn_pos_counts[pos_class(pos[i])] += 1
        if len(conn_examples) < 10:
            conn_examples.append(render_example_connectives(r, nodes, res))

    p("=" * 78)
    p("DERIVATION (ii): SINGLE-TOKEN NODE + CONNECTIVES")
    p("=" * 78)
    total_conn_pos = sum(conn_pos_counts.values())
    p(f"*** THE KEY NUMBER *** connective-token POS composition (n={total_conn_pos} connective tokens): "
      f"function={pct(conn_pos_counts['function'], total_conn_pos):.1f}% "
      f"content={pct(conn_pos_counts['content'], total_conn_pos):.1f}% "
      f"other={pct(conn_pos_counts['other'], total_conn_pos):.1f}%")
    p()
    p(f"overall surface-token assignment: node-owned={pct(total_node_owned_tokens, total_all_tokens):.1f}% "
      f"connective={pct(total_connective_tokens, total_all_tokens):.1f}%  "
      f"(node-owned={total_node_owned_tokens}, connective={total_connective_tokens}, total={total_all_tokens})")
    p()
    p("connective diversity, top 15 most common (relation_a, relation_b) keys:")
    for key, freq in key_freq.most_common(15):
        strings = key_strings[key]
        top5 = strings.most_common(5)
        top5_str = ", ".join(f"{s!r}x{c}" for s, c in top5)
        p(f"  {key!s:40s} freq={freq:4d}  distinct_strings={len(strings):4d}  top5=[{top5_str}]")
    p()
    p("-- 10 example sentences (derivation ii) --")
    for ex in conn_examples:
        p(ex)
        p()

    # ------------------------------------------------------------------
    # Sanity: 100% token coverage in both derivations
    # ------------------------------------------------------------------
    p("=" * 78)
    p("SANITY CHECKS")
    p("=" * 78)
    total_tokens_all = sum(len(r["tokens"]) for r, _, _ in usable_recs)
    p(f"derivation (i) total assigned tokens vs total surface tokens: "
      f"{nearest_assigned_tokens} / {total_tokens_all}  "
      f"({'OK: 100%' if nearest_assigned_tokens == total_tokens_all else 'MISMATCH'})")
    ii_total = total_node_owned_tokens + total_connective_tokens
    p(f"derivation (ii) total assigned tokens vs total surface tokens: "
      f"{ii_total} / {total_tokens_all}  "
      f"({'OK: 100%' if ii_total == total_tokens_all else 'MISMATCH'})")


if __name__ == "__main__":
    main()
