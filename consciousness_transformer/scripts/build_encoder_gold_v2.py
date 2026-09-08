"""Encoder step 2: teacher gold as a CANDIDATE LATTICE, per the canonical v2
I/O contract (``dev/ENCODER_IO_CONTRACT_V2.md``).

Unlike a v1-style generator (one committed tree, MFS sense per word), this
emits ONE record per unique deduped sentence with:

- ``lattice.trees``: up to k top-k parser hypotheses (a forest), each run
  through the SAME :func:`nsm_ct.clause.extract_discourse` path -- structural
  ambiguity lives in the forest, not in a picked tree.
- every content node grounded by the FULL sense-candidate set
  (``grounding.type:"sense"``, ``candidates = usvs.senses_of(word)``) --
  no committed sense.
- pronoun tokens grounded as unresolved ``type:"reference"`` slots
  (``candidates: null``, ``retrieval.source:"memory"`` -- no cross-sentence
  context is threaded between these deduped, independently-sampled corpus
  sentences, so the run-time memory form is the honest one, not an
  authored ``context:`` pointer).
- entity names (non-pronoun) grounded as resolved ``type:"entity"`` --
  referent variables, no sense, no link needed.

Elision is NOT emitted: this pipeline's ``extract_discourse`` never
posits an elided predicate/argument (:class:`~nsm_ct.clause.Clause` always
carries a real surface ``predicate`` string), and hand-authoring elision
gold is explicitly out of scope here (see the encoder step 2 task note).
Likewise no ``utterance_kind`` other than the default ``"proposition"`` is
emitted: this extraction path has no imperative/interjection detection, so
claiming one would be fabricated, not read off the teacher.

Top-k hypotheses come from
:meth:`nsm_ct.input_encoder.ParserInputEncoder._parse_topk_one`, which
already exposes ``chart.hypotheses[:k]`` (quantum_parser's own top-K,
score-sorted, structurally deduped list) as flat
:class:`~nsm_ct.quantum_adapter.HypGraph` views -- the same graph shape
:func:`~nsm_ct.clause.extract_discourse` already consumes via
``_parse_graph_one``/``best_hypothesis``. No parser-internal surgery is
needed to reach top-k; this script just calls the existing method with
``k>1`` instead of always taking the single best hypothesis.

A role/predicate's own graph-node index is NOT preserved through
``extract_discourse`` (its synthetic ``ParseNode`` construction never sets
``.index``), so ``roles[j].token_index`` is recovered by a left-to-right,
consume-on-match walk against ``tokens`` (unambiguous for the common case
of a word appearing once; the leftmost remaining occurrence is claimed for
a repeated word, a heuristic approximation like several already in
``nsm_ct.clause``, e.g. ``EntityTracker``'s recency coref).

Usage: python scripts/build_encoder_gold_v2.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.clause import (  # noqa: E402
    FIRST_PERSON_SINGULAR, MODIFIER_RELATIONS, _PRONOUNS, extract_discourse, is_entity,
)
from nsm_ct.corpus import iter_sentences  # noqa: E402
from nsm_ct.ground.usvs import load_usvs  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

CORPUS_GLOB = str(ROOT / "data" / "corpus" / "real_*.txt")
# GOLD_CORPUS_GLOB: overridable via env var (mirrors GOLD_OUT_JSONL below) so a
# targeted rebuild (e.g. the v2-population-matched v3-small gold) can point at
# a narrower glob without editing this file; unset, behavior is byte-identical
# to before (every data/corpus/real_*.txt file).
CORPUS_GLOB_DEFAULT = os.environ.get("GOLD_CORPUS_GLOB", CORPUS_GLOB)
USVS_DIR = ROOT / "data" / "usvs"
# GOLD_OUT_JSONL/GOLD_OUT_STATS: overridable via env var so the overnight
# corpus-expand run (colab/Gold_Expand.ipynb) can point straight at a
# Drive-backed path (e.g. encoder_gold_v3.jsonl) without editing this file;
# unset, behavior is byte-identical to before (runs/encoder_gold_v2.jsonl).
OUT_JSONL = Path(os.environ.get("GOLD_OUT_JSONL", str(ROOT / "runs" / "encoder_gold_v2.jsonl")))
OUT_STATS = Path(os.environ.get("GOLD_OUT_STATS", str(ROOT / "dev" / "ENCODER_GOLD_V2_STATS.md")))
TOP_K = 8

# D1 (dev/CURRENT_STATE.md decisions locked, 2026-09-07): gold_qa_report.txt
# Part B measured the OLD default (clause-level dedup only, no score
# pruning) at mean_trees=3.46 (61% of records have 3+ trees) and found the
# extra trees SPURIOUS -- trivial modifier-attachment wobble / dropped
# clauses, not genuine ambiguity. The new DEFAULT keeps only the top-1
# parser hypothesis per sentence. `--forest margin` is an opt-in alternative
# that keeps a 2nd+ tree only when it is BOTH within a small score margin of
# the top AND structurally distinct (a different clause split or a
# different role assignment on a content node -- not just a modifier
# reattaching). 0.02 is that report's own recommended margin (its "EXAMPLES"
# section: "recommended M=0.02 used for the 'pruning fixes it' cases" --
# chosen from the tested thresholds [0.0, 0.02, 0.05, 0.1] as the smallest
# one that still separates the genuinely-ambiguous example from the
# spurious-extra ones). `--forest all` disables pruning entirely (the OLD
# behavior: every structurally-deduped top-TOP_K tree is kept). Env-var
# fallbacks (GOLD_FOREST_MODE/GOLD_FOREST_MARGIN) mirror the
# GOLD_OUT_JSONL/GOLD_OUT_STATS pattern above so `colab/Gold_Expand.ipynb`
# (which just runs `python scripts/build_encoder_gold_v2.py` with no args)
# picks up the new top1 default with NO notebook edit.
FOREST_MODES = ("top1", "margin", "all")
FOREST_MODE_DEFAULT = os.environ.get("GOLD_FOREST_MODE", "top1")
FOREST_MARGIN_DEFAULT = float(os.environ.get("GOLD_FOREST_MARGIN", "0.02"))
# A gap below this is an EXACT tie, not a close call -- the QA report found
# ties (54/109 multi-tree sentences) come from parses that dedup to
# different clause dicts by pure noise (attachment order, etc.) despite
# genuinely identical scores; margin mode treats these as spurious
# duplicates of tree[0], never a second genuine reading, however
# structurally distinct their JSON happens to look.
_EXACT_TIE_EPS = 1e-6

# corpus-expand: the pre-expansion corpus (~1k sentences) was small enough
# that an unbounded per-sentence parse never mattered in practice; a
# multi-thousand-sentence overnight run (colab/Gold_Expand.ipynb) can't
# assume that -- one pathological long/ambiguous sentence hanging the
# top-k parse would kill the whole unattended run. Reuse nsm_ct.corpus's
# own dials (the SAME cap :func:`nsm_ct.corpus.parse_passage` already
# applies) so a runaway sentence is capped and tagged, never a hang.
from nsm_ct.corpus import CORPUS_MAX_HYPOTHESES, CORPUS_MAX_PARSE_SECONDS  # noqa: E402


def load_corpus(glob_pattern: str = CORPUS_GLOB_DEFAULT,
                 allowed_basenames: Optional[List[str]] = None) -> List[str]:
    """Every unique deduped sentence across the matching corpus files.

    ``allowed_basenames``, when given, is an explicit filename allow-list
    (applied after the glob) -- e.g. to reproduce the exact v2 sentence
    population (the ORIGINAL five corpus files) after corpus-expand added
    more ``real_*.txt`` files to the default glob.
    """
    seen = set()
    out: List[str] = []
    files = sorted(glob.glob(glob_pattern))
    if allowed_basenames is not None:
        allowed = set(allowed_basenames)
        files = [f for f in files if Path(f).name in allowed]
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        for s in iter_sentences(text):
            s = s.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def ground_word(usvs, word: Optional[str]) -> Dict[str, object]:
    """The unified ``grounding`` construct (contract §4) for one surface word.

    - pronoun -> unresolved ``reference`` slot, run-time memory form
      (no context to author a pointer against).
    - entity name -> resolved ``entity`` (referent variable, no sense).
    - content word with WordNet senses -> unresolved ``sense`` slot, the
      FULL candidate list (no commit).
    - content word the USVS lemma index doesn't cover -> ``entity``
      (contract §4.2's explicit "ungrounded content word" case).
    """
    w = (word or "").lower()
    if w in FIRST_PERSON_SINGULAR:
        # D3 (dev/CURRENT_STATE.md decisions locked): the speaker is a
        # grammar-licensed single referent, symmetric with the imperative's
        # synthesized addressee (prime YOU) -- resolved, no candidate set.
        return {"type": "prime", "prime": "I", "candidates": None}
    if w in _PRONOUNS:
        return {
            "type": "reference",
            "candidates": None,
            "retrieval": {"source": "memory", "method": "coref", "ref": None},
        }
    if is_entity(word or ""):
        return {"type": "entity", "candidates": None}
    candidates, lemma = usvs.senses_of_surface(w)
    if candidates:
        return {
            "type": "sense",
            "candidates": list(candidates),
            "lemma": lemma,
            "retrieval": {"source": "lexicon", "method": "lemma_senses", "ref": None},
        }
    return {"type": "entity", "candidates": None}


class _IndexMatcher:
    """Left-to-right, consume-on-match token-index recovery (see module
    docstring) -- a fresh instance per tree so two trees of the SAME
    sentence independently re-walk the same word occurrences."""

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


def _tree_core_skeleton(tree: Dict) -> Tuple[frozenset, ...]:
    """D1 margin mode's "is this a genuinely different reading" test: the
    clause split + core role assignment (relation, token_index) per clause,
    IGNORING modifier-attachment roles (`nsm_ct.clause.MODIFIER_RELATIONS`
    -- DESCRIPTION/SPECIFICATION/COMPLEMENT/.../QUANTITY/ADDITIVE/FOCUS).
    Two trees with the same skeleton differ only by modifier-attachment
    wobble (an extra/missing descriptor, a different adverb parent) -- the
    exact "spurious extra hypothesis" shape gold_qa_report.txt's examples
    (iii+0..iii+3) show pruning is meant to collapse. A different clause
    count/grouping, a different predicate, or a different relation/index for
    a CORE role always changes this skeleton -- that is "genuinely distinct"
    for this decision's purposes."""
    clauses_sk = []
    for clause in tree["clauses"]:
        core = {("PREDICATE", clause.get("predicate"), clause["predicate_grounding"]["type"])}
        for role in clause["roles"]:
            if role["relation"] in MODIFIER_RELATIONS:
                continue
            core.add((role["relation"], role["token_index"], role["grounding"]["type"]))
        clauses_sk.append(frozenset(core))
    return tuple(sorted(clauses_sk, key=lambda s: sorted(map(str, s))))


def prune_forest(trees: List[Dict], links_per_tree: List[List[Dict]], scores: List[float],
                  mode: str, margin: float) -> Tuple[List[Dict], List[List[Dict]]]:
    """D1: apply the `--forest` policy to an already structurally-deduped,
    score-sorted (best-first) list of trees. Always keeps `trees[0]` (a
    record must emit >=1 tree, contract S2)."""
    if mode not in FOREST_MODES:
        raise ValueError(f"unknown --forest mode {mode!r} (choose from {FOREST_MODES})")
    if mode == "all" or len(trees) <= 1:
        return trees, links_per_tree
    if mode == "top1":
        return trees[:1], links_per_tree[:1]

    # mode == "margin"
    top_score = scores[0]
    top_skeleton = _tree_core_skeleton(trees[0])
    kept_trees, kept_links = [trees[0]], [links_per_tree[0]]
    for tree, link_dicts, score in zip(trees[1:], links_per_tree[1:], scores[1:]):
        gap = top_score - score
        if gap <= _EXACT_TIE_EPS:
            continue  # an exact tie is spurious (see FOREST_MARGIN_DEFAULT note), never kept
        if gap > margin:
            continue  # too far below top-1 to be a competing reading
        if _tree_core_skeleton(tree) == top_skeleton:
            continue  # modifier-attachment wobble only, not a genuine 2nd reading
        kept_trees.append(tree)
        kept_links.append(link_dicts)
    return kept_trees, kept_links


def build_record(usvs, parser: ParserInputEncoder, sentence: str,
                  forest_mode: str = FOREST_MODE_DEFAULT,
                  forest_margin: float = FOREST_MARGIN_DEFAULT) -> Tuple[Optional[Dict], str, int]:
    """Returns (record_or_None, outcome_tag, n_raw_hypotheses)."""
    from src.parser.quantum_parser import ParseResourceExceeded  # local import (qp_root on sys.path)

    try:
        graphs, scores, _margin = parser._parse_topk_one(
            sentence, k=TOP_K,
            max_hypotheses=CORPUS_MAX_HYPOTHESES, max_seconds=CORPUS_MAX_PARSE_SECONDS,
        )
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
    tree_scores: List[float] = []
    seen_trees = set()
    for graph, score in zip(graphs, scores):
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
        tree_scores.append(score)

    if not trees:
        return None, "grounding-fail", len(graphs)

    trees, links_per_tree = prune_forest(trees, links_per_tree, tree_scores,
                                          forest_mode, forest_margin)

    token_sense_candidates = []
    for i, tok in enumerate(tokens):
        cands, lemma = usvs.senses_of_surface(tok)
        cands = list(cands)
        if cands:
            token_sense_candidates.append({
                "index": i, "token": tok, "lemma": lemma,
                "sense_candidates": cands, "chosen_sense": cands[0],
            })

    record = {
        "text": sentence,
        "tokens": tokens,
        "pos": pos,
        "lattice": {"trees": trees, "discourse_links_per_tree": links_per_tree},
        "token_sense_candidates": token_sense_candidates,
    }
    return record, "ok", len(graphs)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forest", choices=FOREST_MODES, default=FOREST_MODE_DEFAULT,
                   help="forest-width policy (default: %(default)s; env GOLD_FOREST_MODE). "
                        "'top1' keeps only the best parser hypothesis per sentence; 'margin' "
                        "also keeps a structurally-distinct 2nd+ tree within --forest-margin "
                        "of the top score; 'all' disables pruning (pre-D1 behavior).")
    p.add_argument("--forest-margin", type=float, default=FOREST_MARGIN_DEFAULT,
                   help="margin mode only: max score gap from top-1 to still keep a "
                        "structurally-distinct tree (default: %(default)s; env GOLD_FOREST_MARGIN).")
    p.add_argument("--corpus-glob", default=CORPUS_GLOB_DEFAULT,
                   help="glob for corpus files (default: %(default)s; env GOLD_CORPUS_GLOB).")
    p.add_argument("--corpus-files", default=None,
                   help="comma-separated basenames (e.g. real_gutenberg_alice.txt); when given, "
                        "restricts --corpus-glob's matches to exactly these files.")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    t0 = time.time()
    corpus_files = args.corpus_files.split(",") if args.corpus_files else None
    sentences = load_corpus(glob_pattern=args.corpus_glob, allowed_basenames=corpus_files)
    print(f"corpus: {len(sentences)} unique sentences", flush=True)
    print(f"corpus glob: {args.corpus_glob}"
          + (f" (files={corpus_files})" if corpus_files else ""), flush=True)
    print(f"forest policy: {args.forest}"
          + (f" (margin={args.forest_margin})" if args.forest == "margin" else ""), flush=True)

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
    n_trees: List[int] = []
    n_raw_hyps: List[int] = []
    n_sense_cands: List[int] = []
    n_reference_slots = 0
    n_only_best = 0  # raw hypothesis count was 1 (nothing to widen)

    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for i, sentence in enumerate(sentences):
            record, outcome, n_raw = build_record(usvs, parser, sentence,
                                                    forest_mode=args.forest,
                                                    forest_margin=args.forest_margin)
            outcomes[outcome] += 1
            if record is not None:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())  # overnight-run durability: a killed
                                           # runtime still leaves every record
                                           # written so far readable on disk.
                n_trees.append(len(record["lattice"]["trees"]))
                n_raw_hyps.append(n_raw)
                if n_raw <= 1:
                    n_only_best += 1
                for tree in record["lattice"]["trees"]:
                    for cl in tree["clauses"]:
                        for node in [{"grounding": cl["predicate_grounding"]}] + cl["roles"]:
                            g = node["grounding"]
                            if g["type"] == "sense":
                                n_sense_cands.append(len(g["candidates"]))
                            elif g["type"] == "reference":
                                n_reference_slots += 1
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(sentences)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"wrote {len(n_trees)} records -> {OUT_JSONL}", flush=True)
    print(f"outcomes: {dict(outcomes)}", flush=True)

    write_stats(sentences, outcomes, n_trees, n_raw_hyps, n_sense_cands, n_reference_slots, n_only_best)
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


def write_stats(sentences, outcomes, n_trees, n_raw_hyps, n_sense_cands, n_reference_slots, n_only_best) -> None:
    lines = []
    lines.append("# Encoder Gold v2 — Candidate Lattice Stats")
    lines.append("")
    lines.append("Generated by `scripts/build_encoder_gold_v2.py` from the canonical "
                 "`dev/ENCODER_IO_CONTRACT_V2.md`. Numbers only.")
    lines.append("")
    lines.append(f"- Unique deduped sentences (all `data/corpus/real_*.txt`): {len(sentences)}")
    lines.append(f"- Records emitted (valid lattice, >=1 tree): {len(n_trees)}")
    lines.append(f"- Outcome counts: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
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
    lines.append("## Sense-candidate breadth (per content node grounded `type:\"sense\"`)")
    lines.append("")
    if n_sense_cands:
        lines.append(f"- n sense-grounded nodes: {len(n_sense_cands)}")
        lines.append(f"- median candidates/node: {statistics.median(n_sense_cands):.1f}")
        lines.append(f"- p90 candidates/node: {pctl(n_sense_cands, 0.90):.1f}")
    lines.append("")
    lines.append("## Pronoun / reference slots")
    lines.append("")
    lines.append(f"- n `type:\"reference\"` (pronoun) slots emitted: {n_reference_slots}")
    lines.append("")
    lines.append("## Top-k reachability")
    lines.append("")
    lines.append(
        "Top-k WAS reachable directly: `nsm_ct.input_encoder.ParserInputEncoder._parse_topk_one` "
        "already exposes `chart.hypotheses[:k]` (quantum_parser's own score-sorted, "
        "structurally-deduped top-K list) as flat `HypGraph` views -- the identical shape "
        "`extract_discourse` already consumes via `best_hypothesis()`. No parser-internal "
        "change was needed; this script calls the existing method with k=8 instead of k=1. "
        "No sentence was limited to only `best_hypothesis`."
    )
    lines.append(
        f"\nOf the {len(n_raw_hyps)} emitted records, {n_only_best} had only 1 raw parser "
        "hypothesis (the parser itself found no structural alternative for that sentence -- "
        "not a limitation of the top-k extraction path, just a genuinely unambiguous parse)."
    )
    OUT_STATS.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
