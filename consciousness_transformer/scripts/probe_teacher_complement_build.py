"""Build FULL per-sentence teacher-parse outcomes over the entire in-repo real
corpus (`data/corpus/real_*.txt`, 1,475 unique sentences) -- the teacher-
complement audit (dev/AUDIT_2026-09-08.md action 1) needs the complete failure
population (grounding-fail / cap-hit / no-hypothesis / too-long / full-tree)
to stratify a 100-sentence sample by outcome x length bin; `runs/m62_probe.csv`
only covers a seeded 500-sentence SAMPLE of the 1,475 (probe_m62_gold_volume.py's
own CAP_TOTAL), and that sample happens to contain zero `no-hypothesis` rows.

Re-uses `probe_m62_gold_volume.py`'s own `classify`/`bin_of`/`load_corpus`
(same parse path, same gold definition, same 30s-per-sentence cap; NO changes
to that module or to any decode/parse behavior) -- just runs it over every
unique sentence instead of a capped sample, skipping any sentence already
classified in the existing `runs/m62_probe.csv` (reused verbatim: outcome is a
pure function of the sentence text + the deterministic parser, not of the
scratch tokenizer object `classify` is handed).

Usage: python scripts/probe_teacher_complement_build.py
Writes: runs/m62_full_probe.csv (bin, file, n_tokens, outcome, full_tree,
        parse_seconds, detail, sentence) for all 1,475 unique sentences.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_m62_gold_volume import classify, bin_of, load_corpus  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402

OUT_CSV = ROOT / "runs" / "m62_full_probe.csv"
KNOWN_CSV = ROOT / "runs" / "m62_probe.csv"


def load_known() -> dict:
    known = {}
    if KNOWN_CSV.exists():
        with open(KNOWN_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                known[row["sentence"]] = row
    return known


def main() -> int:
    rows = load_corpus()
    print(f"corpus: {len(rows)} unique sentences")
    known = load_known()
    print(f"reused (already classified in {KNOWN_CSV.name}): {len(known)}")

    all_texts = [s for _f, s in rows]
    tok = SimpleTokenizer.build(all_texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1

    out_rows = []
    n_new = 0
    t0 = time.time()
    for i, (fname, sent) in enumerate(rows):
        n_tok = len(sent.split())
        b = bin_of(n_tok)
        if sent in known:
            k = known[sent]
            out_rows.append({
                "bin": k["bin"], "file": k["file"], "n_tokens": k["n_tokens"],
                "outcome": k["outcome"], "full_tree": k["full_tree"],
                "parse_seconds": k["parse_seconds"], "detail": k["detail"],
                "sentence": sent,
            })
            continue
        outcome, elapsed, detail = classify(parser, sent)
        out_rows.append({
            "bin": b, "file": fname, "n_tokens": n_tok,
            "outcome": outcome, "full_tree": outcome == "full-tree",
            "parse_seconds": round(elapsed, 4), "detail": detail,
            "sentence": sent,
        })
        n_new += 1
        if n_new % 25 == 0:
            print(f"  ... {n_new} newly classified, {i+1}/{len(rows)} scanned, "
                  f"{time.time()-t0:.0f}s elapsed")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bin", "file", "n_tokens", "outcome", "full_tree",
                                          "parse_seconds", "detail", "sentence"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows ({n_new} newly classified) -> {OUT_CSV} "
          f"in {time.time()-t0:.0f}s")

    from collections import Counter
    print("outcome counts:", Counter(r["outcome"] for r in out_rows))
    print("bin counts:", Counter(r["bin"] for r in out_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
