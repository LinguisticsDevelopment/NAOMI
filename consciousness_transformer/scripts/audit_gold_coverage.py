"""Gold-coverage audit (2026-10-15): measure how much propositional content
the teacher gold (``runs/encoder_gold_v4b_16k.jsonl``) actually keeps per
sentence, vs. how much it silently drops.

Deliberately torch-free / dependency-free (stdlib only) so it can run
without the full nsm_ct/torch stack. The sentence-splitting/tokenization
helpers below are copied (not imported) from ``nsm_ct.corpus`` --
``_split_raw_sentences``/``_tokenize_words`` -- purely so this script can
rebuild the source-file -> sentence membership map without importing a
module chain that pulls in numpy/torch.

For each gold record this measures:
  - VERB-token coverage: fraction of POS=="VERB" token indices in `tokens`
    that appear as *some* node's `token_index` anywhere in `lattice.trees[0]`
    (predicate or role). The v4b-16k pos tagset distinguishes VERB from AUX,
    but carries no finiteness feature (VBD/VBG/VBN are all "VERB") -- so
    "VERB tokens" is a proxy for "predicating verbs", not strictly "finite
    verbs"; flagged as a caveat, not silently assumed away.
  - NOUN+PROPN-token coverage: same, for POS in {"NOUN","PROPN"}.
  - content-clause coverage: (# VERB tokens in the sentence) vs
    (# clauses in the kept tree) -- the same proxy limitation applies.

Breakdowns: by source corpus file (data/corpus/real_*.txt, via the same
first-file-wins dedup `build_encoder_gold_v2.load_corpus` uses) and by
sentence length bin (<=8 / 9-15 / 16-25 / 26+ tokens).

Usage:
    python scripts/audit_gold_coverage.py \
        --gold runs/encoder_gold_v4b_16k.jsonl \
        --corpus-glob "data/corpus/real_*.txt" \
        --out dev/GOLD_COVERAGE_AUDIT.md
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# sentence splitting / tokenization, copied from nsm_ct.corpus (see module
# docstring) so this script needs no nsm_ct/numpy/torch import at all.
# ---------------------------------------------------------------------------
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "no", "prof", "capt", "gen", "col"}
_SENT_END_CHARS = ".!?"
_CONTRACTION_SUFFIXES = ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m")


def _split_raw_sentences(text: str) -> List[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(normalized)
    while i < n:
        ch = normalized[i]
        buf.append(ch)
        if ch in _SENT_END_CHARS:
            j = i + 1
            while j < n and normalized[j] in _SENT_END_CHARS + '"\'':
                buf.append(normalized[j])
                j += 1
            i = j
            word = "".join(buf).strip()
            last_word = word.rstrip(_SENT_END_CHARS + '"\'').split(" ")[-1].lower()
            at_end = i >= n
            next_is_space = i < n and normalized[i] == " "
            if last_word in _ABBREV and not at_end:
                continue
            if at_end or next_is_space:
                parts.append(word)
                buf = []
            continue
        i += 1
    if buf:
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
    return [p for p in parts if p]


def _tokenize_words(raw_sentence: str) -> List[str]:
    text = raw_sentence.lower()
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("--", " -- ")
    for suf in _CONTRACTION_SUFFIXES:
        text = text.replace(suf, " " + suf)
    for ch in ".,!?;:()\"":
        text = text.replace(ch, f" {ch} ")
    return text.split()


def iter_sentences(text: str) -> List[str]:
    out = []
    for raw in _split_raw_sentences(text):
        toks = _tokenize_words(raw)
        if toks:
            out.append(" ".join(toks))
    return out


def load_source_map(glob_pattern: str) -> Dict[str, str]:
    """sentence text -> basename of the FIRST source file it appears in
    (mirrors build_encoder_gold_v2.load_corpus's global first-file-wins
    dedup, so this reproduces the same per-file membership the v4b-16k
    build's own per-source yield table used)."""
    seen: Dict[str, str] = {}
    for f in sorted(glob.glob(glob_pattern)):
        name = Path(f).name
        text = Path(f).read_text(encoding="utf-8")
        for s in iter_sentences(text):
            s = s.strip()
            if not s or s in seen:
                continue
            seen[s] = name
    return seen


def len_bin(n: int) -> str:
    if n <= 8:
        return "A (<=8)"
    if n <= 15:
        return "B (9-15)"
    if n <= 25:
        return "C (16-25)"
    return "D (26+)"


def tree_node_indices(tree: Dict, tokens: List[str]) -> set:
    """Every `token_index` referenced by any predicate or role in the tree's
    (single, top-1) clause list. The predicate itself carries no
    `token_index` field in this schema (only roles do; `predicate_token_index`
    is set only for elision/reference carriers, which this extraction path
    never emits per build_encoder_gold_v2's own docstring) -- so a predicate's
    surface index is recovered by a left-to-right, first-unclaimed-occurrence
    string match against `tokens`, the same heuristic build_encoder_gold_v2.py
    itself documents using for roles."""
    claimed: set = set()
    for clause in tree.get("clauses", []):
        for role in clause.get("roles", []):
            ti = role.get("token_index")
            if ti is not None:
                claimed.add(ti)
    for clause in tree.get("clauses", []):
        pred = clause.get("predicate")
        pti = clause.get("predicate_token_index")
        if pti is not None:
            claimed.add(pti)
            continue
        if pred is None:
            continue
        pred_l = pred.lower()
        for i, tok in enumerate(tokens):
            if i in claimed:
                continue
            if tok.lower() == pred_l:
                claimed.add(i)
                break
    return claimed


def analyze(records: List[Dict], source_map: Dict[str, str]) -> Dict:
    per_record = []
    for rec in records:
        tokens = rec["tokens"]
        pos = rec["pos"]
        n_tok = len(tokens)
        trees = rec.get("lattice", {}).get("trees", [])
        tree0 = trees[0] if trees else {"clauses": []}
        n_clauses = len(tree0.get("clauses", []))
        node_idxs = tree_node_indices(tree0, tokens)

        verb_idxs = {i for i, p in enumerate(pos) if p == "VERB"}
        noun_idxs = {i for i, p in enumerate(pos) if p in ("NOUN", "PROPN")}

        verb_cov = (len(verb_idxs & node_idxs) / len(verb_idxs)) if verb_idxs else None
        noun_cov = (len(noun_idxs & node_idxs) / len(noun_idxs)) if noun_idxs else None

        src = source_map.get(rec["text"].strip(), "UNKNOWN")
        per_record.append({
            "text": rec["text"],
            "n_tok": n_tok,
            "bin": len_bin(n_tok),
            "source": src,
            "n_verb": len(verb_idxs),
            "verb_cov": verb_cov,
            "n_noun": len(noun_idxs),
            "noun_cov": noun_cov,
            "n_clauses": n_clauses,
            "clause_verb_ratio": (n_clauses / len(verb_idxs)) if verb_idxs else None,
        })
    return per_record


def agg_coverage(rows: List[Dict], key: str) -> Tuple[Optional[float], int]:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None, 0
    return statistics.mean(vals), len(vals)


def pct_under(rows: List[Dict], key: str, thresh: float = 0.5) -> Tuple[Optional[float], int, int]:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None, 0, 0
    n_under = sum(1 for v in vals if v < thresh)
    return 100.0 * n_under / len(vals), n_under, len(vals)


def fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100*x:.1f}%"


def build_report(rows: List[Dict]) -> str:
    lines = []
    lines.append("# Gold Coverage Audit -- encoder_gold_v4b_16k.jsonl (2026-10-15)\n")
    lines.append(f"Records analyzed: {len(rows)}\n")

    verb_mean, verb_n = agg_coverage(rows, "verb_cov")
    noun_mean, noun_n = agg_coverage(rows, "noun_cov")
    verb_u50, verb_u50_n, verb_u50_d = pct_under(rows, "verb_cov")
    noun_u50, noun_u50_n, noun_u50_d = pct_under(rows, "noun_cov")

    lines.append("## Headline coverage (over all 10,225 records)\n")
    lines.append("| metric | mean coverage | n records with >=1 token of this class | % records <50% covered |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| VERB tokens in tree | {fmt_pct(verb_mean)} | {verb_n} | "
                  f"{verb_u50:.1f}% ({verb_u50_n}/{verb_u50_d}) |")
    lines.append(f"| NOUN+PROPN tokens in tree | {fmt_pct(noun_mean)} | {noun_n} | "
                  f"{noun_u50:.1f}% ({noun_u50_n}/{noun_u50_d}) |")
    lines.append("")
    lines.append("(VERB/NOUN/PROPN are the parser's own coarse POS tags on each record; "
                  "\"VERB\" carries no finiteness feature in this tagset -- gerunds/participles "
                  "tag VERB too -- so this is a proxy for \"predicating verb tokens\", not a "
                  "strict finite-verb count. AUX is excluded. See caveats at the end.)\n")

    # content-clause coverage: clauses in tree vs verb count
    ratios = [r["clause_verb_ratio"] for r in rows if r["clause_verb_ratio"] is not None]
    lines.append("## Content-clause coverage (clauses kept vs. VERB tokens in the sentence)\n")
    if ratios:
        lines.append(f"- mean clauses-kept / verb-tokens ratio: {statistics.mean(ratios):.3f} "
                      f"(1.0 would mean one clause per verb token)")
        n_single_clause = sum(1 for r in rows if r["n_clauses"] <= 1)
        n_multi_verb = sum(1 for r in rows if r["n_verb"] >= 2)
        n_single_clause_multi_verb = sum(1 for r in rows if r["n_clauses"] <= 1 and r["n_verb"] >= 2)
        lines.append(f"- records with >=2 VERB tokens: {n_multi_verb}")
        lines.append(f"- of those, records whose kept tree has only 1 clause "
                      f"(every verb beyond the first is uncaptured as its own clause): "
                      f"{n_single_clause_multi_verb} "
                      f"({100*n_single_clause_multi_verb/n_multi_verb:.1f}% of multi-verb records)")
    lines.append("")

    # by source file
    lines.append("## Breakdown by source file\n")
    lines.append("| source file | n records | mean VERB cov | mean NOUN/PROPN cov | % <50% verb cov |")
    lines.append("|---|---:|---:|---:|---:|")
    by_src: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for src in sorted(by_src, key=lambda s: -len(by_src[s])):
        sub = by_src[src]
        vmean, _ = agg_coverage(sub, "verb_cov")
        nmean, _ = agg_coverage(sub, "noun_cov")
        vu50, vu50_n, vu50_d = pct_under(sub, "verb_cov")
        vu50_s = f"{vu50:.1f}%" if vu50 is not None else "n/a"
        lines.append(f"| {src} | {len(sub)} | {fmt_pct(vmean)} | {fmt_pct(nmean)} | {vu50_s} |")
    lines.append("")

    # by length bin
    lines.append("## Breakdown by sentence length bin\n")
    lines.append("| bin | n records | mean VERB cov | mean NOUN/PROPN cov | % <50% verb cov | % <50% noun cov |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    by_bin: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_bin[r["bin"]].append(r)
    for b in ["A (<=8)", "B (9-15)", "C (16-25)", "D (26+)"]:
        sub = by_bin.get(b, [])
        if not sub:
            continue
        vmean, _ = agg_coverage(sub, "verb_cov")
        nmean, _ = agg_coverage(sub, "noun_cov")
        vu50, vu50_n, vu50_d = pct_under(sub, "verb_cov")
        nu50, nu50_n, nu50_d = pct_under(sub, "noun_cov")
        vu50_s = f"{vu50:.1f}%" if vu50 is not None else "n/a"
        nu50_s = f"{nu50:.1f}%" if nu50 is not None else "n/a"
        lines.append(f"| {b} | {len(sub)} | {fmt_pct(vmean)} | {fmt_pct(nmean)} | {vu50_s} | {nu50_s} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=str(ROOT / "runs" / "encoder_gold_v4b_16k.jsonl"))
    ap.add_argument("--corpus-glob", default=str(ROOT / "data" / "corpus" / "real_*.txt"))
    ap.add_argument("--out", default=str(ROOT / "dev" / "GOLD_COVERAGE_AUDIT.md"))
    ap.add_argument("--dump-rows", default=None, help="optional path to dump per-record rows as jsonl")
    args = ap.parse_args()

    print(f"loading corpus source map from {args.corpus_glob} ...", file=sys.stderr)
    source_map = load_source_map(args.corpus_glob)
    print(f"  {len(source_map)} unique sentences mapped to a source file", file=sys.stderr)

    print(f"loading gold records from {args.gold} ...", file=sys.stderr)
    records = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  {len(records)} records loaded", file=sys.stderr)

    rows = analyze(records, source_map)
    n_unknown_src = sum(1 for r in rows if r["source"] == "UNKNOWN")
    print(f"  {n_unknown_src} records could not be matched to a source file "
          f"(whitespace/tokenization drift between the corpus-split sentence and the gold "
          f"record's own `text`)", file=sys.stderr)

    report = build_report(rows)
    report += (f"\n## Matching caveat\n\n{n_unknown_src}/{len(rows)} records' `text` did not "
               f"exact-match any source file's `iter_sentences()` output and are reported under "
               f"source \"UNKNOWN\" above (not dropped from the overall/length-bin tables).\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)

    if args.dump_rows:
        with open(args.dump_rows, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote per-record rows -> {args.dump_rows}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
