"""Real-text coverage probe — round-5+ regression gate for the parser pipeline.

The hand-built stress battery (``probe_parser_stress.py``) is saturated
(32/34, M47). Per RESEARCH_NOTES M47, the next parser work must be driven
by real-text failures, not invented cases. This script draws a deterministic
sample of SemCor sentences, runs them through the SAME path the curriculum
uses (``ParserInputEncoder._parse_graph`` -> ``nsm_ct.clause.extract_discourse``),
and classifies each sentence's outcome:

  NO_PARSE     quantum_parser produced no hypothesis at all (graph is None)
  NO_CLAUSE    parsed, but extract_discourse produced zero clauses
  CLAUSE_WEAK  clauses exist, but none has a real (non-punctuation) SUBJECT
  CLAUSE_OK    at least one clause has a non-punctuation SUBJECT

There is no gold structure here -- this is coverage triage (does *anything*
usable come out), not accuracy scoring. Rerunnable: prints the coverage
table and, with --taxonomy, a starter breakdown of NO_CLAUSE/CLAUSE_WEAK
sentences bucketed by cheap surface signals (quotes, possessives, appositive
commas, etc.) to speed up manual failure-class triage.

Usage:  python scripts/probe_realtext.py [--n 400] [--min-len 5] [--max-len 25]
                                          [--taxonomy] [--dump-failures PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.clause import extract_discourse, _PUNCT
from nsm_ct.input_encoder import ParserInputEncoder
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.structure import PARSE_LABELS
from nsm_ct.tokenizer import SimpleTokenizer


def load_semcor_sample(n: int, min_len: int, max_len: int):
    """First ``n`` SemCor sentences with min_len <= len(tokens) <= max_len.

    Deterministic: iterates ``semcor.sents()`` in corpus order, lowercases,
    joins with spaces (final period stays a separate token, matching the
    pipeline's tokenization convention -- SemCor already tokenizes ``.`` as
    its own item). Returns (sentences, n_skipped_long, n_scanned).
    """
    from nltk.corpus import semcor

    sentences = []
    n_skipped_long = 0
    n_scanned = 0
    for toks in semcor.sents():
        n_scanned += 1
        L = len(toks)
        if L < min_len:
            continue
        if L > max_len:
            n_skipped_long += 1
            continue
        text = " ".join(t.lower() for t in toks)
        sentences.append(text)
        if len(sentences) >= n:
            break
    return sentences, n_skipped_long, n_scanned


def clause_roles(cl) -> dict:
    roles = {"PRED": (cl.predicate or "").lower()}
    for rel, arg in cl.args:
        roles.setdefault(rel, (arg.token or "").lower())
    return roles


def has_real_subject(cl) -> bool:
    for rel, arg in cl.args:
        if rel == "SUBJECT":
            tok = (arg.token or "").lower()
            if tok and tok not in _PUNCT:
                return True
    return False


def classify(parser, sentence: str):
    graph = parser._parse_graph(sentence)
    if graph is None:
        return "NO_PARSE", [], 0
    clauses, _links = extract_discourse(graph)
    seen = [clause_roles(c) for c in clauses]
    max_args = max((len(c.args) for c in clauses), default=0)
    if not clauses:
        return "NO_CLAUSE", seen, max_args
    if any(has_real_subject(c) for c in clauses):
        return "CLAUSE_OK", seen, max_args
    return "CLAUSE_WEAK", seen, max_args


# -- cheap surface-signal buckets for --taxonomy (starter triage only; the
# real taxonomy in the round-5 writeup is graph-level, done by hand) --------
_SURFACE_SIGNALS = [
    ("quoted_speech", lambda s: "``" in s or "''" in s or '"' in s),
    ("possessive_s", lambda s: " 's " in s or s.endswith(" 's") or "'s " in s),
    ("appositive_comma", lambda s: s.count(",") >= 1),
    ("semicolon_colon", lambda s: ";" in s or ":" in s),
    ("conjunction_and", lambda s: " and " in s),
    ("conjunction_but_or", lambda s: " but " in s or " or " in s),
    ("relative_wh", lambda s: any(f" {w} " in s for w in ("who", "which", "that", "whom", "whose"))),
    ("numbers_dates", lambda s: any(ch.isdigit() for ch in s)),
    ("dash", lambda s: "--" in s or " - " in s),
    ("multi_sentence", lambda s: s.rstrip(" .?!").count(" . ") > 0 or
        sum(s.count(p) for p in (".", "?", "!")) > 1),
]


def surface_buckets(sentence: str):
    return [name for name, fn in _SURFACE_SIGNALS if fn(sentence)] or ["plain"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of sentences to sample")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--taxonomy", action="store_true", help="print surface-signal breakdown of failures")
    ap.add_argument("--dump-failures", type=str, default=None, help="write every NO_CLAUSE/CLAUSE_WEAK sentence + clauses to this path")
    ap.add_argument("--verbose", action="store_true", help="print every sentence's outcome")
    args = ap.parse_args()

    sentences, n_skipped_long, n_scanned = load_semcor_sample(args.n, args.min_len, args.max_len)
    print(f"SemCor scan: {n_scanned} sentences scanned, {n_skipped_long} skipped "
          f"(len > {args.max_len}), {len(sentences)} sampled "
          f"(target {args.n}, len {args.min_len}-{args.max_len}).")

    tok = SimpleTokenizer.build(sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1

    counts = Counter()
    failures = []  # (outcome, sentence, clauses)
    all_rows = []
    arg_dist = Counter()  # richness of the best clause, CLAUSE_OK sentences only
    for sent in sentences:
        outcome, seen, max_args = classify(parser, sent)
        counts[outcome] += 1
        all_rows.append((outcome, sent, seen))
        if outcome == "CLAUSE_OK":
            arg_dist[max_args] += 1
        if outcome in ("NO_PARSE", "NO_CLAUSE", "CLAUSE_WEAK"):
            failures.append((outcome, sent, seen))
        if args.verbose:
            mark = {"CLAUSE_OK": ".", "CLAUSE_WEAK": "w", "NO_CLAUSE": "c", "NO_PARSE": "x"}[outcome]
            print(f"  [{mark}] {sent}")
            if outcome != "CLAUSE_OK":
                for roles in seen:
                    print(f"        clause: {roles}")

    total = len(sentences)
    print("\n=== coverage table ===")
    for name in ("CLAUSE_OK", "CLAUSE_WEAK", "NO_CLAUSE", "NO_PARSE"):
        c = counts[name]
        print(f"{name:<12} {c:>4} / {total}  ({c/total:.1%})")
    print(f"{'TOTAL':<12} {total:>4}")
    print(f"\nCLAUSE_OK rate: {counts['CLAUSE_OK']}/{total} = {counts['CLAUSE_OK']/total:.1%}")

    if counts["CLAUSE_OK"]:
        print("\n=== richness of CLAUSE_OK's best clause (arg count = SUBJECT + others) ===")
        print("(extraction is curriculum-scoped to at most one non-SUBJECT role today; "
              "this tracks whether that ceiling is still 2 as the corpus/extraction change)")
        n_ok = counts["CLAUSE_OK"]
        for n_args, c in sorted(arg_dist.items()):
            tag = " (SUBJECT-only stump)" if n_args <= 1 else ""
            print(f"  {n_args} arg(s): {c:>4} / {n_ok}  ({c/n_ok:.1%}){tag}")

    if args.taxonomy:
        print("\n=== surface-signal breakdown of failures (NO_PARSE + NO_CLAUSE + CLAUSE_WEAK) ===")
        bucket_counts = defaultdict(int)
        for outcome, sent, _seen in failures:
            for b in surface_buckets(sent):
                bucket_counts[b] += 1
        n_fail = len(failures)
        print(f"({n_fail} failing sentences; a sentence may hit multiple buckets)")
        for name, c in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<18} {c:>4}  ({c/max(n_fail,1):.1%} of failures)")

    if args.dump_failures:
        with open(args.dump_failures, "w") as f:
            for outcome, sent, seen in failures:
                f.write(f"[{outcome}] {sent}\n")
                for roles in seen:
                    f.write(f"    clause: {roles}\n")
        print(f"\nWrote {len(failures)} failure records to {args.dump_failures}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
