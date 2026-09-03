"""M63.1 — encoder gold-tree distillation corpus (dev/UNIVERSAL_ENCODER_DESIGN.md
§10, Step 1). Data-generation only: no model/parser source is touched.

Runs the SAME deterministic-teacher parse path as
``scripts/probe_m62_gold_volume.py``
(``nsm_ct.input_encoder.ParserInputEncoder._parse_graph_one``, default
``ParserConfig`` -- 30s wall-clock cap) over EVERY unique deduped sentence in
ALL ``data/corpus/real_*.txt`` files, and for every sentence that yields a FULL
grounded tree (a non-None discourse graph whose ``extract_discourse`` produces
>=1 clause with a real, non-punctuation SUBJECT -- exactly M62b's "gold"
definition), serializes one JSONL record to ``runs/encoder_gold_v1.jsonl``.

Per record:
  text                     -- raw sentence string fed to the parser.
  tokens / pos             -- the parser's own tagger output
                              (``nsm_ct.input_encoder.ParserInputEncoder._tag``),
                              aligned 1:1.
  gold_tree.clauses        -- ``nsm_ct.clause.extract_discourse``'s clauses:
                              predicate (+ its MFS USVS sense id, if any) and
                              role slots (subject/object/place/...), each with
                              its surface word and MFS USVS sense id (``null``
                              for entity variables -- names/pronouns, per
                              ``nsm_ct.clause.is_entity`` -- and for words the
                              USVS lemma index doesn't cover).
  gold_tree.discourse_links -- ``nsm_ct.clause.DiscourseLink``s between clauses
                              (coordination/negation).
  token_sense_candidates   -- for every token the USVS lemma index covers
                              (``nsm_ct.ground.usvs.USVS.senses_of``), the FULL
                              MFS-ordered candidate sense-id list plus which
                              one is "chosen" (index 0 -- see
                              dev/ENCODER_IO_CONTRACT.md for why this pipeline
                              has no context-conditioned WSD step to plug in
                              instead: ``nsm_ct.wsd``/``nsm_ct.sense_chooser``
                              are separate, not-wired-into-this-path modules).

Sentences that don't yield a full tree are skipped and counted by failure
mode (mirrors ``probe_m62_gold_volume.py``'s ``classify()`` taxonomy exactly:
too-long / cap-hit / no-hypothesis / grounding-fail / other-exception).

Usage: python scripts/build_encoder_gold_v1.py
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.clause import _PUNCT, extract_discourse, is_entity  # noqa: E402
from nsm_ct.corpus import iter_sentences  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.usvs_bridge import default_usvs  # noqa: E402

CORPUS_GLOB = str(ROOT / "data" / "corpus" / "real_*.txt")
OUT_JSONL = ROOT / "runs" / "encoder_gold_v1.jsonl"
OUT_STATS_MD = ROOT / "dev" / "ENCODER_GOLD_V1_STATS.md"


def load_corpus():
    seen = set()
    rows = []  # (filename, sentence)
    for f in sorted(glob.glob(CORPUS_GLOB)):
        text = Path(f).read_text(encoding="utf-8")
        for s in iter_sentences(text):
            s = s.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            rows.append((Path(f).name, s))
    return rows


def has_real_subject(cl) -> bool:
    for rel, arg in cl.args:
        if rel == "SUBJECT":
            tok = (arg.token or "").lower()
            if tok and tok not in _PUNCT:
                return True
    return False


def classify(parser: ParserInputEncoder, sentence: str):
    """Same taxonomy as probe_m62_gold_volume.py's classify(), plus returns
    the graph/clauses/links on success so the caller can serialize them
    without re-parsing."""
    from src.parser.quantum_parser import ParseResourceExceeded  # local import; qp_root already on sys.path

    captured = {}
    parser._warned = False
    parser._note = lambda msg: captured.__setitem__("msg", msg)  # instance-level override, not a source edit

    t0 = time.perf_counter()
    try:
        graph = parser._parse_graph_one(sentence)
    except ParseResourceExceeded as exc:
        return "cap-hit", time.perf_counter() - t0, str(exc), None, None, None
    except Exception as exc:  # pragma: no cover - defensive, mirrors _parse_graph_one's own catch
        return "other-exception", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}", None, None, None
    elapsed = time.perf_counter() - t0

    if graph is None:
        msg = captured.get("msg", "")
        if "too long" in msg.lower():
            return "too-long", elapsed, msg, None, None, None
        if msg:
            return "other-exception", elapsed, msg, None, None, None
        return "no-hypothesis", elapsed, "", None, None, None

    clauses, links = extract_discourse(graph)
    if clauses and any(has_real_subject(c) for c in clauses):
        return "full-tree", elapsed, "", graph, clauses, links
    return "grounding-fail", elapsed, f"clauses={len(clauses)}", None, None, None


def _mfs_sense(usvs, word: str):
    """MFS-ordered candidate sense ids for `word`, or [] if the USVS lemma
    index doesn't cover it (function word, proper noun, punctuation, ...)."""
    if not word:
        return []
    return usvs.senses_of(word.lower())


def _role_dict(usvs, relation: str, arg_node) -> dict:
    word = arg_node.token
    ent = is_entity(word)
    cands = [] if ent else _mfs_sense(usvs, word)
    return {
        "relation": relation,
        "word": word,
        "is_entity": ent,
        "sense_id": cands[0] if cands else None,
    }


def serialize_clause(usvs, cl) -> dict:
    pred_cands = [] if is_entity(cl.predicate) else _mfs_sense(usvs, cl.predicate)
    return {
        "predicate": cl.predicate,
        "predicate_sense_id": pred_cands[0] if pred_cands else None,
        "is_question": cl.is_question,
        "roles": [_role_dict(usvs, rel, arg) for rel, arg in cl.args],
    }


def serialize_links(links) -> list:
    return [
        {"coordinator": lk.coordinator, "prime": lk.prime, "clause_i": lk.i, "clause_j": lk.j}
        for lk in links
    ]


def token_sense_candidates(usvs, tokens) -> list:
    out = []
    for i, tok in enumerate(tokens):
        cands = _mfs_sense(usvs, tok)
        if not cands:
            continue
        out.append({
            "index": i,
            "token": tok,
            "sense_candidates": cands,
            "chosen_sense": cands[0],
        })
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


def main() -> int:
    rows = load_corpus()
    file_counts = Counter(f for f, _ in rows)
    print(f"corpus: {len(rows)} unique sentences from {len(file_counts)} files: {dict(file_counts)}", flush=True)

    all_texts = [s for _f, s in rows]
    tok = SimpleTokenizer.build(all_texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1
    usvs = default_usvs()
    print(f"usvs loaded: core_words={len(usvs.core_words)} senses={len(usvs.sense_ids)}", flush=True)

    outcome_counts = Counter()
    n_tokens_gold = []
    n_clauses_gold = []
    n_roles_per_clause = []
    n_cands_per_token = []
    n_records = 0
    t_start = time.perf_counter()

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as out_f:
        for i, (fname, sent) in enumerate(rows):
            outcome, elapsed, detail, graph, clauses, links = classify(parser, sent)
            outcome_counts[outcome] += 1

            if outcome == "full-tree":
                words = parser._tag(sent)
                tokens = [w.text for w in words]
                pos = [getattr(w.pos, "name", str(w.pos)) for w in words]

                gold_clauses = [serialize_clause(usvs, cl) for cl in clauses]
                gold_links = serialize_links(links)
                tsc = token_sense_candidates(usvs, tokens)

                record = {
                    "text": sent,
                    "tokens": tokens,
                    "pos": pos,
                    "gold_tree": {
                        "clauses": gold_clauses,
                        "discourse_links": gold_links,
                    },
                    "token_sense_candidates": tsc,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1

                n_tokens_gold.append(len(tokens))
                n_clauses_gold.append(len(gold_clauses))
                n_roles_per_clause.extend(len(c["roles"]) for c in gold_clauses)
                n_cands_per_token.extend(len(t["sense_candidates"]) for t in tsc)

            if (i + 1) % 50 == 0:
                dt = time.perf_counter() - t_start
                print(f"  ... {i + 1}/{len(rows)} attempted, {n_records} gold so far, "
                      f"{dt:.1f}s elapsed, last='{sent[:40]}' -> {outcome}", flush=True)

    dt_total = time.perf_counter() - t_start
    print(f"done: {n_records} gold / {len(rows)} attempted in {dt_total:.1f}s", flush=True)
    print(f"outcome counts: {dict(outcome_counts)}", flush=True)
    print(f"wrote -> {OUT_JSONL} ({OUT_JSONL.stat().st_size} bytes)", flush=True)

    write_stats(rows, file_counts, outcome_counts, n_records,
                n_tokens_gold, n_clauses_gold, n_roles_per_clause, n_cands_per_token)
    print(f"wrote stats -> {OUT_STATS_MD}", flush=True)
    return 0


def write_stats(rows, file_counts, outcome_counts, n_records,
                 n_tokens_gold, n_clauses_gold, n_roles_per_clause, n_cands_per_token):
    n_attempted = len(rows)
    n_failed = n_attempted - n_records

    def stat_line(name, values):
        if not values:
            return f"- {name}: n=0"
        return (f"- {name}: n={len(values)} median={statistics.median(values):.2f} "
                f"p90={pctl(values, 0.90):.2f} min={min(values)} max={max(values)}")

    lines = []
    lines.append("# M63.1 — Encoder Gold-Tree v1 Stats (numbers only)")
    lines.append("")
    lines.append("No interpretation added beyond what the generation run shows.")
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append("- Corpus files (unique sentences after dedup): "
                  + ", ".join(f"{f}={n}" for f, n in sorted(file_counts.items())))
    lines.append(f"- Total unique sentences attempted: {n_attempted}")
    lines.append("")
    lines.append("## Yield")
    lines.append("")
    lines.append(f"- Gold records emitted: {n_records}")
    lines.append(f"- Attempted: {n_attempted}")
    lines.append(f"- Yield: {n_records}/{n_attempted} = {n_records / n_attempted:.1%}" if n_attempted else "- Yield: n/a")
    lines.append(f"- Failed (all non-full-tree outcomes): {n_failed}")
    lines.append("")
    lines.append("### Failure breakdown by outcome")
    lines.append("")
    lines.append("| outcome | count | share of attempted |")
    lines.append("|---|---|---|")
    for name, c in outcome_counts.most_common():
        lines.append(f"| {name} | {c} | {c / n_attempted:.1%} |" if n_attempted else f"| {name} | {c} | - |")
    lines.append("")
    lines.append("## Distributions (over the emitted gold set)")
    lines.append("")
    lines.append(stat_line("tokens per sentence", n_tokens_gold))
    lines.append(stat_line("clauses per tree", n_clauses_gold))
    lines.append(stat_line("role-slots per clause", n_roles_per_clause))
    lines.append(stat_line("candidate senses per content token", n_cands_per_token))
    lines.append("")
    lines.append("## Output artifact")
    lines.append("")
    size = OUT_JSONL.stat().st_size if OUT_JSONL.exists() else 0
    lines.append(f"- {OUT_JSONL.relative_to(ROOT)}: {n_records} records, {size} bytes "
                 f"({size / n_records:.0f} bytes/record avg)" if n_records else
                 f"- {OUT_JSONL.relative_to(ROOT)}: 0 records")
    lines.append("")

    OUT_STATS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
