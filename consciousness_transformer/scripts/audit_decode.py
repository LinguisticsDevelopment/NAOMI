"""READ-ONLY audit of what `beam_decode` actually emits for the run-2
checkpoint (dev/RESEARCH_NOTES.md 2026-09-06 CORRECTION entry: two fix
hypotheses -- loss-weight terminal-weight, strict_ground mask -- both
FALSIFIED; the ~8x real-token over-generation mechanism is not understood).

This script does NOT train, does NOT edit the model or metrics code, and
does NOT change `beam_decode`'s default behavior. It:

  (A) replicates `encoder_model.beam_decode`'s loop verbatim (same actions,
      same mask, same branching, same beam_width/k/max_steps/max_clauses)
      in `instrumented_beam_decode` below, ONLY adding bookkeeping (the
      winning beam's action-type sequence + why it stopped: a real STOP
      action vs the max_steps/max_clauses safety-net cutoff);
  (B) reports tree-shape distributions for the "best tree per gold tree"
      (same best-overlap selection `rescore_encoder.py`/`score_record` use,
      generalized only to always resolve to a concrete tree -- see
      `select_best_emitted_tree` docstring) vs the matching gold shape;
  (C) reports token-index duplication within a best tree and within a
      single clause -- does the monotonic buffer pointer actually forbid
      re-grounding a token, or are tokens reused;
  (D) dumps 8 full example trees (source, gold, winning emitted tree +
      its action sequence) spanning short/medium/long sentences.

Usage:
    python scripts/audit_decode.py --checkpoint runs/encoder_colab.pt \
        --gold runs/encoder_gold_v2.jsonl --usvs-dir ../data/usvs \
        | tee runs/audit_output.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from train_encoder import load_gold, stratified_split  # noqa: E402


# ---------------------------------------------------------------------------
# (A) instrumented beam_decode -- a faithful replica of
# `encoder_model.beam_decode`'s control flow (policy="model" only; that is
# the only policy this audit needs). Every line that is not pure
# bookkeeping (the `.actions` / `.term_reason` additions) is copied
# verbatim from the production function so the audit measures the SAME
# decode `beam_decode` performs, not an approximation of it.
# ---------------------------------------------------------------------------

@dataclass
class IBeamState:
    h: object
    i: int
    open_clause: bool
    open_kind_id: int
    prev_action_id: int
    logprob: float
    clauses: list = field(default_factory=list)
    cur_clause: Optional[dict] = None
    steps_taken: int = 0
    done: bool = False
    actions: list = field(default_factory=list)
    term_reason: Optional[str] = None


def instrumented_beam_decode(model, feats, beam_width: int = 8, k: int = 8,
                              max_steps: int = 400, max_clauses: int = 20):
    """Returns (forest, forest_states, winning) where `forest`/`forest_states`
    mirror `beam_decode`'s (tree, originating-beam) pairs (up to k, dedup by
    skeleton) and `winning` is the single top-logprob finished beam (== the
    beam behind `forest[0]`, since `finished` is logprob-sorted and the
    first-seen skeleton is always kept)."""
    with torch.no_grad():
        enc = model.encode(feats)
        T = len(feats.tokens)

        beams = [IBeamState(h=model.init_controller_state(), i=0, open_clause=False,
                             open_kind_id=model._none_clause_id,
                             prev_action_id=model._start_action_id, logprob=0.0)]
        finished: List[IBeamState] = []

        for _ in range(max_steps):
            if not beams:
                break
            candidates: List[IBeamState] = []
            for b in beams:
                if b.done:
                    finished.append(b)
                    continue
                legal = em.legal_action_types(b.open_clause, b.i, T, has_clause=bool(b.clauses))
                i_clamped = min(b.i, T - 1) if T > 0 else 0
                enc_i = enc[i_clamped] if T > 0 else torch.zeros(model.d_model)
                h = model.controller_step(enc_i, b.open_kind_id, b.prev_action_id, b.h)
                logits = model.action_type_head(h).squeeze(0) + em._mask_vector(legal)
                logp = F.log_softmax(logits, dim=-1)
                order = torch.argsort(logp, descending=True).tolist()
                legal_ranked = [em.ACTION_TYPES[idx] for idx in order if em.ACTION_TYPES[idx] in legal]
                top = em._select_branch_actions(legal_ranked, logp, 0.0)

                for action in top:
                    nb = IBeamState(h=h, i=b.i, open_clause=b.open_clause, open_kind_id=b.open_kind_id,
                                     prev_action_id=b.prev_action_id, logprob=b.logprob,
                                     clauses=list(b.clauses), cur_clause=b.cur_clause,
                                     steps_taken=b.steps_taken + 1,
                                     actions=b.actions + [action], term_reason=b.term_reason)
                    nb.logprob = b.logprob + float(logp[em.ACTION_INDEX[action]])
                    em._apply_action(nb, action, model, feats, h)
                    if action == "STOP":
                        nb.term_reason = "STOP"
                    if len(nb.clauses) >= max_clauses or nb.steps_taken >= max_steps:
                        if nb.term_reason is None:
                            nb.term_reason = "MAX_CLAUSES" if len(nb.clauses) >= max_clauses else "MAX_STEPS"
                        nb.done = True
                    candidates.append(nb)
            candidates.sort(key=lambda s: s.logprob, reverse=True)
            beams = candidates[:beam_width]
            still_going = [b for b in beams if not b.done]
            if not still_going:
                finished.extend(beams)
                break
            beams = still_going + [b for b in beams if b.done]
            finished.extend([b for b in beams if b.done])
            beams = [b for b in beams if not b.done]

        finished.sort(key=lambda s: s.logprob, reverse=True)
        seen = set()
        forest: List[dict] = []
        forest_states: List[IBeamState] = []
        for b in finished:
            tree = {"clauses": b.clauses}
            sk = em._tree_skeleton(tree)
            if sk in seen:
                continue
            seen.add(sk)
            forest.append(tree)
            forest_states.append(b)
            if len(forest) >= k:
                break
        if not forest and beams:
            forest = [{"clauses": beams[0].clauses}]
            forest_states = [beams[0]]
        winning = finished[0] if finished else (beams[0] if beams else None)
        return forest, forest_states, winning


# ---------------------------------------------------------------------------
# (B)/(C) shape + duplication helpers
# ---------------------------------------------------------------------------

def select_best_emitted_tree(gold_edges: frozenset, forest: List[dict],
                              forest_edge_sets: List[frozenset]) -> Tuple[int, int, Optional[int]]:
    """Same selection rule `_best_tree_overlap` uses (max overlap, ties to
    smaller size), generalized ONLY so it always resolves to a concrete
    tree index even when every forest tree has zero overlap with this gold
    tree (`_best_tree_overlap` itself reports (0, 0) in that case without
    naming a tree, which is fine for a ratio but not for shape reporting).
    Returns (overlap, size, idx); idx is None only if the forest is empty."""
    best_overlap, best_size, best_idx = -1, float("inf"), None
    for idx, edges in enumerate(forest_edge_sets):
        overlap = len(gold_edges & edges)
        size = len(edges)
        if overlap > best_overlap or (overlap == best_overlap and size < best_size):
            best_overlap, best_size, best_idx = overlap, size, idx
    if best_idx is None:
        return 0, 0, None
    return best_overlap, best_size, best_idx


def tree_node_list(tree: dict) -> List[dict]:
    nodes = []
    for clause in tree["clauses"]:
        nodes.append(clause["predicate"])
        nodes.extend(clause["roles"])
    return nodes


def tree_real_null_counts(tree: dict) -> Tuple[int, int]:
    nodes = tree_node_list(tree)
    real = sum(1 for n in nodes if n["token_index"] is not None)
    null = sum(1 for n in nodes if n["token_index"] is None)
    return real, null


def gold_clause_node_count(record: dict, clause: dict) -> int:
    return len(em.clause_node_order(record, clause))


def fmt_dist(vals: List[float]) -> str:
    if not vals:
        return "n=0"
    return (f"n={len(vals)} min={min(vals):.1f} median={statistics.median(vals):.1f} "
            f"mean={statistics.mean(vals):.2f} max={max(vals):.1f}")


# ---------------------------------------------------------------------------
# (D) example-tree dumping
# ---------------------------------------------------------------------------

def fmt_gold_tree(record: dict, tree: dict) -> str:
    tokens = record["tokens"]
    lines = []
    for ci, clause in enumerate(tree["clauses"]):
        lines.append(f"    clause[{ci}] kind={clause.get('utterance_kind')} "
                      f"predicate_word={clause.get('predicate')!r}")
        for role, g, tidx in em.clause_node_order(record, clause):
            tok = tokens[tidx] if tidx is not None else "<null>"
            lines.append(f"      {role}: {tok!r} #idx={tidx} gtype={g['type']}")
    return "\n".join(lines)


def fmt_emitted_tree(record: dict, tree: dict) -> str:
    tokens = record["tokens"]
    lines = []
    for ci, clause in enumerate(tree["clauses"]):
        lines.append(f"    clause[{ci}] kind={clause.get('utterance_kind')}")
        nodes = [("PREDICATE", clause["predicate"])] + [(r["relation"], r) for r in clause["roles"]]
        for role, node in nodes:
            tidx = node["token_index"]
            tok = tokens[tidx] if tidx is not None else "<null>"
            gtype = node["grounding"]["type"]
            lines.append(f"      {role}:{tok}#idx={tidx}:{gtype}")
    return "\n".join(lines)


def fmt_actions(actions: List[str]) -> str:
    return " ".join(actions)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_colab.pt"))
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", choices=("train", "dev", "test"), default="test")
    ap.add_argument("--n-examples", type=int, default=8)
    args = ap.parse_args()

    print(f"loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    pos_vocab = ckpt["pos_vocab"]
    role_vocab = ckpt["role_vocab"]

    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=ckpt["d_axes"],
                             hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                             controller_hidden=ckpt["d_model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"policy params: {model.num_policy_params():,}")
    print(f"checkpoint config: {cfg}")
    print(f"checkpoint's reported metrics: {ckpt.get('metrics', {})}")

    records = load_gold(args.gold)
    train_recs, dev_recs, test_recs = stratified_split(records, cfg["seed"], cfg["n_train"],
                                                         cfg["n_dev"], cfg["n_test"])
    splits = {"train": train_recs, "dev": dev_recs, "test": test_recs}
    recs = splits[args.split]
    print(f"reproduced split: train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)} (seed={cfg['seed']})")
    print(f"decoding split={args.split} (n={len(recs)}) beam_width={args.beam_width} k={args.k}")

    usvs = load_usvs(args.usvs_dir)

    # ---- run instrumented beam_decode once per record; keep everything ----
    per_record = []  # list of dicts: record, feats, forest, forest_states, winning
    for ri, record in enumerate(recs):
        feats = em.build_features(record, usvs, pos_vocab, ckpt["hash_buckets"])
        forest, forest_states, winning = instrumented_beam_decode(
            model, feats, beam_width=args.beam_width, k=args.k)
        per_record.append({"record": record, "feats": feats, "forest": forest,
                            "forest_states": forest_states, "winning": winning})
        if (ri + 1) % 20 == 0 or ri == len(recs) - 1:
            print(f"  decoded {ri + 1}/{len(recs)} ...", file=sys.stderr)

    # =====================================================================
    print("\n" + "=" * 78)
    print("(A) ACTION-SEQUENCE HISTOGRAM (winning = top-logprob beam)")
    print("=" * 78)
    action_totals = Counter()
    per_tree_action_counts = {a: [] for a in em.ACTION_TYPES}
    term_reason_counts = Counter()
    seq_lens = []
    for pr in per_record:
        w = pr["winning"]
        if w is None:
            term_reason_counts["NO_WINNING_BEAM"] += 1
            continue
        counts = Counter(w.actions)
        for a in em.ACTION_TYPES:
            per_tree_action_counts[a].append(counts.get(a, 0))
        action_totals.update(counts)
        term_reason_counts[w.term_reason or "UNKNOWN"] += 1
        seq_lens.append(len(w.actions))

    n = len(per_record)
    print(f"n records = {n}")
    print(f"winning-beam action-sequence length: {fmt_dist(seq_lens)}")
    print("\nper action-type: total / per-tree mean")
    for a in em.ACTION_TYPES:
        total = action_totals.get(a, 0)
        mean = total / n if n else float("nan")
        print(f"  {a:22s} total={total:6d}  mean/tree={mean:7.3f}")

    print("\nTERMINATION REASON (winning beam):")
    for reason, cnt in term_reason_counts.most_common():
        print(f"  {reason:15s} {cnt:4d}  ({cnt / n * 100:.1f}%)")
    stop_frac = term_reason_counts.get("STOP", 0) / n if n else float("nan")
    cutoff_frac = (term_reason_counts.get("MAX_CLAUSES", 0) + term_reason_counts.get("MAX_STEPS", 0)) / n if n else float("nan")
    print(f"\n  => real-STOP fraction = {stop_frac:.3f}   safety-net-cutoff fraction = {cutoff_frac:.3f}")

    # =====================================================================
    print("\n" + "=" * 78)
    print("(B) TREE SHAPE: best-emitted-tree-per-gold-tree vs GOLD")
    print("=" * 78)
    emitted_clauses_per_tree = []
    emitted_nodes_per_clause = []
    emitted_real_nodes_per_tree = []
    emitted_null_nodes_per_tree = []
    gold_clauses_per_tree = []
    gold_nodes_per_clause = []
    no_overlap_count = 0
    n_gold_trees_total = 0

    best_tree_cache: Dict[int, List[Tuple[dict, int]]] = {}  # record idx -> [(best_tree, overlap)]

    for pr in per_record:
        record = pr["record"]
        forest = pr["forest"]
        gold_sense, gold_slots, gold_trees_sk = em._gold_sites(record)
        forest_sk = [em._tree_skeleton(t) for t in forest]
        forest_edge_sets = [em._tree_edge_set(sk) for sk in forest_sk]

        entries = []
        for tree in record["lattice"]["trees"]:
            gold_clauses_per_tree.append(len(tree["clauses"]))
            for clause in tree["clauses"]:
                gold_nodes_per_clause.append(gold_clause_node_count(record, clause))

        for gold_sk in gold_trees_sk:
            gold_edges = em._tree_edge_set(gold_sk)
            if not gold_edges:
                continue
            n_gold_trees_total += 1
            overlap, size, idx = select_best_emitted_tree(gold_edges, forest, forest_edge_sets)
            if overlap == 0:
                no_overlap_count += 1
            if idx is None:
                continue
            best_tree = forest[idx]
            entries.append((best_tree, overlap))
            emitted_clauses_per_tree.append(len(best_tree["clauses"]))
            for clause in best_tree["clauses"]:
                emitted_nodes_per_clause.append(1 + len(clause["roles"]))
            real, nul = tree_real_null_counts(best_tree)
            emitted_real_nodes_per_tree.append(real)
            emitted_null_nodes_per_tree.append(nul)
        best_tree_cache[id(record)] = entries

    print(f"gold trees scored (non-empty edge set): {n_gold_trees_total}")
    print(f"  of which the best-overlap emitted tree had ZERO overlap: {no_overlap_count} "
          f"({no_overlap_count / n_gold_trees_total * 100:.1f}%)" if n_gold_trees_total else "")
    print()
    print("EMITTED (best tree per gold tree):")
    print(f"  clauses/tree:          {fmt_dist(emitted_clauses_per_tree)}")
    print(f"  nodes/clause:          {fmt_dist(emitted_nodes_per_clause)}")
    print(f"  real-token nodes/tree: {fmt_dist(emitted_real_nodes_per_tree)}")
    print(f"  null nodes/tree:       {fmt_dist(emitted_null_nodes_per_tree)}")
    print()
    print("GOLD (for comparison):")
    print(f"  clauses/tree:  {fmt_dist(gold_clauses_per_tree)}")
    print(f"  nodes/clause:  {fmt_dist(gold_nodes_per_clause)}")

    # =====================================================================
    print("\n" + "=" * 78)
    print("(C0) OBJECT-IDENTITY ALIASING CHECK (candidate mechanism)")
    print("=" * 78)
    print("Python object identity of clause dicts / roles-lists across each record's")
    print("full k=8 forest -- beam branching does `clauses=list(b.clauses)` (a new")
    print("outer list) and `cur_clause=b.cur_clause` (the SAME inner dict/list, not")
    print("copied). If many sibling beams share one still-open clause before it is")
    print("closed, every one of THEIR GROUND/EMIT actions appends into the SAME")
    print("`roles` list object -- not just the eventual winning trajectory's own.")
    total_clause_slots = 0
    total_distinct_clause_objs = 0
    total_distinct_roles_objs = 0
    max_roles_list_len = 0
    per_record_ratios = []
    for pr in per_record:
        forest = pr["forest"]
        slots = sum(len(t["clauses"]) for t in forest)
        distinct_clause = len(set(id(c) for t in forest for c in t["clauses"]))
        distinct_roles = len(set(id(c["roles"]) for t in forest for c in t["clauses"]))
        for t in forest:
            for c in t["clauses"]:
                max_roles_list_len = max(max_roles_list_len, len(c["roles"]))
        total_clause_slots += slots
        total_distinct_clause_objs += distinct_clause
        total_distinct_roles_objs += distinct_roles
        if distinct_clause:
            per_record_ratios.append(slots / distinct_clause)
    print(f"\ntotal clause-slots across all forests (sum over records of sum over "
          f"forest trees of len(clauses)): {total_clause_slots}")
    print(f"total DISTINCT clause dict objects (by id()):                          {total_distinct_clause_objs}")
    print(f"total DISTINCT roles-list objects (by id()):                           {total_distinct_roles_objs}")
    print(f"=> mean clause-slots-per-distinct-object (sharing factor): "
          f"{fmt_dist(per_record_ratios)}")
    print(f"largest single roles-list length observed (one shared clause's role "
          f"count, any tree, any record): {max_roles_list_len}")

    print("\n" + "=" * 78)
    print("(C) TOKEN-INDEX DUPLICATION (within the best-emitted trees from B)")
    print("=" * 78)
    per_token_repeat_hist = Counter()   # repeat-count -> number of (tree, token_index) pairs
    max_repeat = 0
    clauses_with_dup = 0
    clauses_total = 0
    clause_dup_role_examples = []

    seen_trees = set()
    for pr in per_record:
        record = pr["record"]
        for tree, _overlap in best_tree_cache.get(id(record), []):
            tid = id(tree)
            if tid in seen_trees:
                continue
            seen_trees.add(tid)
            whole_tree_counter = Counter()
            for clause in tree["clauses"]:
                clauses_total += 1
                nodes = [("PREDICATE", clause["predicate"])] + [(r["relation"], r) for r in clause["roles"]]
                clause_counter = Counter(n["token_index"] for _role, n in nodes if n["token_index"] is not None)
                whole_tree_counter.update(clause_counter)
                dup_tidx = [t for t, c in clause_counter.items() if c >= 2]
                if dup_tidx:
                    clauses_with_dup += 1
                    if len(clause_dup_role_examples) < 5:
                        roles_at = [(role, n["token_index"]) for role, n in nodes if n["token_index"] in dup_tidx]
                        clause_dup_role_examples.append((record["tokens"], roles_at))
            for tidx, cnt in whole_tree_counter.items():
                per_token_repeat_hist[cnt] += 1
                max_repeat = max(max_repeat, cnt)

    print(f"distinct best-emitted trees examined: {len(seen_trees)}")
    print(f"per-token repeat-count histogram (across all best-emitted trees' real-token nodes):")
    for cnt in sorted(per_token_repeat_hist):
        print(f"  repeat={cnt}: {per_token_repeat_hist[cnt]} (token_index, tree) pairs")
    print(f"max repeat count observed: {max_repeat}")
    print()
    print(f"clauses with a same-token_index-under-multiple-roles collision: "
          f"{clauses_with_dup}/{clauses_total} ({clauses_with_dup / clauses_total * 100:.2f}%)" if clauses_total else "n=0")
    if clause_dup_role_examples:
        print("example within-clause collisions (tokens truncated):")
        for tokens, roles_at in clause_dup_role_examples:
            print(f"  tokens={tokens[:12]}{'...' if len(tokens) > 12 else ''}  roles_at_dup_index={roles_at}")

    # =====================================================================
    print("\n" + "=" * 78)
    print(f"(D) {args.n_examples} FULL EXAMPLE TREES (spread of short/medium/long)")
    print("=" * 78)
    order = sorted(range(len(per_record)), key=lambda idx: len(per_record[idx]["record"]["tokens"]))
    n_ex = min(args.n_examples, len(order))
    if n_ex > 0:
        pick_positions = [round(p * (len(order) - 1) / (n_ex - 1)) if n_ex > 1 else 0 for p in range(n_ex)]
        picks = sorted(set(order[p] for p in pick_positions))
        # top up if dedup collapsed positions
        i = 0
        while len(picks) < n_ex and i < len(order):
            if order[i] not in picks:
                picks.append(order[i])
            i += 1
        picks = sorted(picks, key=lambda idx: len(per_record[idx]["record"]["tokens"]))[:n_ex]
    else:
        picks = []

    for ex_num, ridx in enumerate(picks):
        pr = per_record[ridx]
        record = pr["record"]
        winning = pr["winning"]
        T = len(record["tokens"])
        print(f"\n--- EXAMPLE {ex_num + 1}/{n_ex}  (test-split idx={ridx}, T={T} tokens) ---")
        print(f"SOURCE: {record['text']!r}")
        print(f"TOKENS: {record['tokens']}")
        print(f"\nGOLD ({len(record['lattice']['trees'])} tree(s)):")
        for ti, tree in enumerate(record["lattice"]["trees"]):
            print(f"  gold_tree[{ti}]:")
            print(fmt_gold_tree(record, tree))
        print(f"\nWINNING EMITTED TREE (top-logprob, term_reason={winning.term_reason if winning else None}):")
        if winning is not None:
            print(fmt_emitted_tree(record, {"clauses": winning.clauses}))
            print(f"\n  action sequence ({len(winning.actions)} steps):")
            print(f"  {fmt_actions(winning.actions)}")
        else:
            print("  <no winning beam produced>")

    print("\n" + "=" * 78)
    print("AUDIT COMPLETE -- measurement only, no fixes proposed or applied.")
    print("=" * 78)


if __name__ == "__main__":
    main()
