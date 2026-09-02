"""M62b: teacher gold-volume probe on the in-repo real corpus, binned by
sentence length (dev/UNIVERSAL_ENCODER_DESIGN.md open question #2: "how
much teacher gold does real text actually yield?"). Measurement only --
no model code, no source changes.

Uses the SAME parse path the curriculum uses:
``nsm_ct.input_encoder.ParserInputEncoder._parse_graph_one`` (the
quantum_parser deterministic parser, default ``ParserConfig`` -- 30s
wall-clock cap, no hypotheses cap). A sentence's "gold signal" (FULL
grounded tree) is: the parse produces a non-None discourse graph AND
``nsm_ct.clause.extract_discourse`` on that graph yields at least one
clause with a real (non-punctuation) SUBJECT -- the same CLAUSE_OK bar
``probe_realtext.py`` uses for "something usable came out", here read as
"clean teacher gold" rather than mere coverage triage.

Corpus: every ``data/corpus/real_*.txt`` file (real, already-committed
Gutenberg children's-literature prose; no network fetch). Sentences are
split/tokenized via ``nsm_ct.corpus.iter_sentences`` (the pipeline's own
raw-prose splitter, built for exactly these files), deduped, and binned by
token count: A<=8, B 9-15, C 16-25, D 26+. Capped at ~500 total, sampled
evenly across bins (seeded shuffle) so the run finishes in a few minutes.

Per sentence we record: full-tree yes/no, parse wall-clock seconds, and a
failure-mode tag when not full-tree:
  too-long        quantum_parser's own ``max_sentence_length`` (100 words)
                  hard cap fired (ValueError inside parse()).
  cap-hit         ``ParseResourceExceeded`` -- the 30s wall-clock cap fired.
  no-hypothesis   parser ran to completion with zero exceptions but
                  ``best_hypothesis()`` was None (no reading found).
  grounding-fail  a discourse graph WAS produced, but extract_discourse
                  found no clause, or no clause has a real SUBJECT.
  other-exception any other exception (tagging, adapter, etc.).

``_parse_graph_one`` itself swallows every non-ParseResourceExceeded
exception into a single printed note (see input_encoder.py) -- to recover
the real failure mode without touching that source, this script
monkeypatches the ONE instance's ``_note`` method (a runtime attribute
override on our own encoder object, not a source edit) to capture the
message text per call instead of print-once.

Usage: python scripts/probe_m62_gold_volume.py
"""

from __future__ import annotations

import csv
import glob
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.clause import _PUNCT, extract_discourse  # noqa: E402
from nsm_ct.corpus import iter_sentences  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

CORPUS_GLOB = str(ROOT / "data" / "corpus" / "real_*.txt")
CAP_TOTAL = 500
BINS = ("A", "B", "C", "D")
SEED = 42
OUT_CSV = ROOT / "dev" / "M62_gold_volume_probe.csv"
OUT_MD = ROOT / "dev" / "M62_GOLD_VOLUME_PROBE.md"


def bin_of(n_tokens: int) -> str:
    if n_tokens <= 8:
        return "A"
    if n_tokens <= 15:
        return "B"
    if n_tokens <= 25:
        return "C"
    return "D"


def has_real_subject(cl) -> bool:
    for rel, arg in cl.args:
        if rel == "SUBJECT":
            tok = (arg.token or "").lower()
            if tok and tok not in _PUNCT:
                return True
    return False


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


def sample_bins(rows, cap_total: int, seed: int):
    by_bin = defaultdict(list)
    for fname, s in rows:
        n = len(s.split())
        by_bin[bin_of(n)].append((fname, s, n))
    rng = random.Random(seed)
    for b in BINS:
        rng.shuffle(by_bin[b])
    target = cap_total // len(BINS)
    sampled = {b: by_bin[b][:target] for b in BINS}
    used = sum(len(v) for v in sampled.values())
    shortfall = cap_total - used
    if shortfall > 0:
        leftover = []
        for b in BINS:
            leftover.extend(by_bin[b][target:])
        rng.shuffle(leftover)
        for x in leftover[:shortfall]:
            sampled[bin_of(x[2])].append(x)
    return sampled, {b: len(by_bin[b]) for b in BINS}


def classify(parser: ParserInputEncoder, sentence: str):
    from src.parser.quantum_parser import ParseResourceExceeded  # local import; qp_root already on sys.path

    captured = {}
    parser._warned = False
    parser._note = lambda msg: captured.__setitem__("msg", msg)  # instance-level override, not a source edit

    t0 = time.perf_counter()
    try:
        graph = parser._parse_graph_one(sentence)
    except ParseResourceExceeded as exc:
        return "cap-hit", time.perf_counter() - t0, str(exc)
    except Exception as exc:  # pragma: no cover - defensive, mirrors _parse_graph_one's own catch
        return "other-exception", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0

    if graph is None:
        msg = captured.get("msg", "")
        if "too long" in msg.lower():
            return "too-long", elapsed, msg
        if msg:
            return "other-exception", elapsed, msg
        return "no-hypothesis", elapsed, ""

    clauses, _links = extract_discourse(graph)
    if clauses and any(has_real_subject(c) for c in clauses):
        return "full-tree", elapsed, ""
    return "grounding-fail", elapsed, f"clauses={len(clauses)}"


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
    print(f"corpus: {len(rows)} unique sentences from {len(file_counts)} files: "
          f"{dict(file_counts)}")

    sampled, avail = sample_bins(rows, CAP_TOTAL, SEED)
    total_sampled = sum(len(v) for v in sampled.values())
    print(f"available per bin: {avail}")
    print(f"sampled per bin (cap {CAP_TOTAL}): {{ {', '.join(f'{b}: {len(sampled[b])}' for b in BINS)} }} "
          f"= {total_sampled} total")

    tok_texts = [s for b in BINS for _f, s, _n in sampled[b]]
    tok = SimpleTokenizer.build(tok_texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1

    all_rows = []  # dicts for csv
    n_done = 0
    for b in BINS:
        for fname, sent, n_tok in sampled[b]:
            outcome, elapsed, detail = classify(parser, sent)
            all_rows.append({
                "bin": b, "file": fname, "n_tokens": n_tok,
                "outcome": outcome, "full_tree": outcome == "full-tree",
                "parse_seconds": round(elapsed, 4), "detail": detail,
                "sentence": sent,
            })
            n_done += 1
            if n_done % 50 == 0:
                print(f"  ... {n_done}/{total_sampled} parsed")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bin", "file", "n_tokens", "outcome", "full_tree",
                                          "parse_seconds", "detail", "sentence"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {OUT_CSV}")

    write_report(all_rows, file_counts, avail, sampled)
    print(f"wrote report -> {OUT_MD}")
    return 0


def write_report(all_rows, file_counts, avail, sampled):
    by_bin = defaultdict(list)
    for r in all_rows:
        by_bin[r["bin"]].append(r)

    def stats(rows):
        n = len(rows)
        n_full = sum(1 for r in rows if r["full_tree"])
        n_cap = sum(1 for r in rows if r["outcome"] == "cap-hit")
        secs = [r["parse_seconds"] for r in rows]
        return {
            "n": n,
            "full_tree": n_full,
            "yield": n_full / n if n else float("nan"),
            "cap_hit": n_cap,
            "cap_hit_rate": n_cap / n if n else float("nan"),
            "median_s": statistics.median(secs) if secs else float("nan"),
            "p90_s": pctl(secs, 0.90) if secs else float("nan"),
        }

    overall = stats(all_rows)
    bin_stats = {b: stats(by_bin[b]) for b in BINS}

    failure_counter = Counter(r["outcome"] for r in all_rows if not r["full_tree"])
    examples = defaultdict(list)
    for r in all_rows:
        if not r["full_tree"] and len(examples[r["outcome"]]) < 3:
            examples[r["outcome"]].append(r)

    lines = []
    lines.append("# M62b — Teacher Gold-Volume Probe (in-repo real corpus)")
    lines.append("")
    lines.append("Numbers only, no interpretation added beyond what the measurement shows.")
    lines.append("")
    lines.append(f"- Corpus files (unique sentences after dedup): "
                 + ", ".join(f"{f}={n}" for f, n in sorted(file_counts.items())))
    lines.append(f"- Total unique sentences available: {sum(file_counts.values())}")
    lines.append(f"- Available per length-bin (A<=8, B 9-15, C 16-25, D 26+): "
                 + ", ".join(f"{b}={avail[b]}" for b in BINS))
    lines.append(f"- Sampled per bin (seed={SEED}, cap={CAP_TOTAL} total): "
                 + ", ".join(f"{b}={len(sampled[b])}" for b in BINS)
                 + f" = {sum(len(sampled[b]) for b in BINS)} total")
    lines.append("- Gold definition: `_parse_graph_one` (default ParserConfig, 30s cap) "
                 "produces a non-None discourse graph AND `extract_discourse` finds >=1 "
                 "clause with a real (non-punctuation) SUBJECT.")
    lines.append("")
    lines.append("## Per-bin results")
    lines.append("")
    lines.append("| bin | n | full-tree yield | cap-hit rate | median s | p90 s |")
    lines.append("|---|---|---|---|---|---|")
    for b in BINS:
        st = bin_stats[b]
        lines.append(f"| {b} | {st['n']} | {st['full_tree']}/{st['n']} = {st['yield']:.1%} "
                     f"| {st['cap_hit']}/{st['n']} = {st['cap_hit_rate']:.1%} "
                     f"| {st['median_s']:.3f} | {st['p90_s']:.3f} |")
    lines.append(f"| **overall** | {overall['n']} | {overall['full_tree']}/{overall['n']} "
                 f"= {overall['yield']:.1%} | {overall['cap_hit']}/{overall['n']} "
                 f"= {overall['cap_hit_rate']:.1%} | {overall['median_s']:.3f} | {overall['p90_s']:.3f} |")
    lines.append("")
    lines.append("## Yield trend A->D")
    lines.append("")
    trend = " -> ".join(f"{b}:{bin_stats[b]['yield']:.1%}" for b in BINS)
    lines.append(trend)
    lines.append("")
    lines.append("## Failure modes (non-full-tree rows only)")
    lines.append("")
    n_fail = sum(failure_counter.values())
    lines.append(f"{n_fail} non-full-tree rows out of {overall['n']} total.")
    lines.append("")
    lines.append("| outcome | count | share of failures |")
    lines.append("|---|---|---|")
    for name, c in failure_counter.most_common():
        lines.append(f"| {name} | {c} | {c/n_fail:.1%} |" if n_fail else f"| {name} | {c} | - |")
    lines.append("")
    for name, _c in failure_counter.most_common():
        lines.append(f"### {name} — examples")
        lines.append("")
        for r in examples[name]:
            det = f" ({r['detail']})" if r["detail"] else ""
            lines.append(f"- [{r['bin']}, {r['n_tokens']}tok, {r['file']}] `{r['sentence']}`{det}")
        lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
