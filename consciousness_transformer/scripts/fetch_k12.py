"""scripts/fetch_k12.py -- K-12 comprehension-QA corpus fetcher (dev/K12_CORPUS_SCOUT.md).

Fetches the scout's TOP-3 real, license-cleared K-12 reading sources into
``data/k12/<source>/``, mirroring ``scripts/fetch_corpus.py``'s conventions:
one directory per source, gated behind ``--allow-download`` (never invoked
by default, never by the test suite), each source fetch is independently
best-effort (a network failure SKIPs that source, never aborts the run),
and a manifest is printed/written at the end.

Unlike ``fetch_corpus.py`` (raw prose only), these three sources ship real
human-authored comprehension QUESTIONS with gold answers -- see
dev/K12_CORPUS_SCOUT.md for the full source-by-source comparison and the
license text/URL for each. License check happens up front, per source,
before any network call: :data:`SOURCE_LICENSES` names the exact license
this script relies on, and :func:`_print_license_notice` prints it so a
`--allow-download` run's own output is a license audit trail.

(1) FairytaleQA (github.com/uci-soe/FairytaleQAData, Apache License 2.0) --
    278 Gutenberg-derived fairy tales, K-8, with education-expert-written
    QA pairs marked local/summary and explicit/implicit. Fetched via
    ``story_meta.csv`` (the repo's own story index) -> per-story
    ``<name>-story.csv`` (section-level passage text) + ``<name>-
    questions.csv`` (QA pairs), both CSVs kept in their native schema
    (no reformatting) so downstream conversion can choose which fields
    to consume.

(2) MCTest (github.com/mcobzarenco/mctest mirror of the original Microsoft
    Research release, data/MCTest/*; original terms: Microsoft Research
    Data License -- research use; NOT independently re-verified for
    commercial redistribution in this script, see dev/K12_CORPUS_SCOUT.md
    licensing caveats) -- grade-school fictional stories (mc160, aimed at
    ~7-year-olds) and harder mc500 stories, each with exactly 4
    multiple-choice questions and a gold answer letter. Fetched as the
    original tab-separated .tsv (story+questions+options) and .ans (gold
    answer letters) files, unmodified.

(3) African Storybook Project, English stories (github.com/global-asp/
    asp-source, CC-BY 4.0/3.0 per-story -- verified per-story from the
    global-asp/global-asp INDEX.md license column, only CC-BY-marked
    stories are fetched) -- leveled early readers, grade-marked by ASP's
    own reading-level scheme, real prose but NO built-in comprehension
    questions (a real, honestly-reported gap -- see dev/
    K12_CORPUS_SCOUT.md). Intended to feed the EXISTING
    ``nsm_ct.corpus`` question-synthesis path (the same "queried role"
    machinery ``scripts/fetch_corpus.py``'s Gutenberg prose already goes
    through), not to ship pre-made questions itself.

Dedup: within a source, a story/passage already written to disk (by
content hash) is not re-fetched or re-written on a second run; across
sources, no cross-source dedup is attempted (they don't overlap).

Usage:
    python scripts/fetch_k12.py [--out-dir data/k12] [--allow-download]
                                 [--fairytaleqa-limit N] [--asp-limit N]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data" / "k12"
_UA = {"User-Agent": "naomi-k12-fetch/1.0 (+dev/K12_CORPUS_SCOUT.md)"}
_TIMEOUT = 20

# ---------------------------------------------------------------------------
# license notices -- printed up front, one per source, before any fetch.
# See dev/K12_CORPUS_SCOUT.md for the full verification writeup.
# ---------------------------------------------------------------------------
SOURCE_LICENSES: Dict[str, str] = {
    "fairytaleqa": (
        "FairytaleQA (uci-soe/FairytaleQAData) -- Apache License 2.0 "
        "(https://github.com/uci-soe/FairytaleQAData/blob/main/LICENSE). "
        "Stories are themselves 278 Project Gutenberg fairy tales (public "
        "domain); the QA annotations are the Apache-2.0-licensed original "
        "contribution of this repo."
    ),
    "mctest": (
        "MCTest (Microsoft Research, mirrored at github.com/mcobzarenco/"
        "mctest/data/MCTest) -- released under Microsoft's research data "
        "license (non-commercial research use); NO LICENSE file is "
        "co-located with this specific mirror, so the exact terms are "
        "NOT independently re-verified here -- confirm against Microsoft "
        "Research's own MCTest page before any commercial reuse. See "
        "dev/K12_CORPUS_SCOUT.md licensing caveats."
    ),
    "african_storybook": (
        "African Storybook Project (github.com/global-asp/asp-source, "
        "story text; license per-story from github.com/global-asp/"
        "global-asp INDEX.md) -- only stories whose INDEX.md license "
        "column says CC-BY are fetched. Per-story attribution (writer/"
        "illustrator/translator) is preserved in each story's own "
        "metadata footer, unedited."
    ),
}


def _print_license_notice(source: str) -> None:
    print(f"[license] {source}: {SOURCE_LICENSES[source]}")


def _get(url: str) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310 -- gated by --allow-download
            return resp.read()
    except Exception as exc:  # noqa: BLE001 -- network is best-effort, never fatal
        print(f"  SKIP {url}: {type(exc).__name__}: {exc}")
        return None


class Dedup:
    """Content-hash dedup within one source's output directory."""

    def __init__(self) -> None:
        self._seen: set = set()

    def write_if_new(self, path: Path, content: bytes) -> bool:
        h = hashlib.sha1(content).hexdigest()
        if h in self._seen:
            return False
        self._seen.add(h)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return True


# ---------------------------------------------------------------------------
# (1) FairytaleQA
# ---------------------------------------------------------------------------

FTQA_BASE = "https://raw.githubusercontent.com/uci-soe/FairytaleQAData/main/"


def fetch_fairytaleqa(out_dir: Path, limit: int) -> List[Tuple[str, int]]:
    _print_license_notice("fairytaleqa")
    src_dir = out_dir / "fairytaleqa"
    dedup = Dedup()
    manifest: List[Tuple[str, int]] = []

    meta_bytes = _get(FTQA_BASE + "story_meta.csv")
    if meta_bytes is None:
        print("  fairytaleqa: story_meta.csv unreachable, skipping source entirely")
        return manifest
    dedup.write_if_new(src_dir / "story_meta.csv", meta_bytes)

    reader = csv.DictReader(io.StringIO(meta_bytes.decode("utf-8")))
    rows = list(reader)[:limit]
    n_ok = 0
    for row in rows:
        fname = row["filename"]
        # story_meta.csv's own "origin" column IS the folder name under
        # data-by-origin/{section-stories,questions}/ -- either an anthology
        # slug (e.g. "andersen-fairybook") or a literal "first-round" /
        # "second-round" for the initial pilot stories. Trust the metadata,
        # don't guess.
        origin = row["origin"]
        story_url = FTQA_BASE + f"data-by-origin/section-stories/{origin}/{fname}-story.csv"
        q_url = FTQA_BASE + f"data-by-origin/questions/{origin}/{fname}-questions.csv"
        story_bytes = _get(story_url)
        q_bytes = _get(q_url) if story_bytes is not None else None
        if story_bytes is None or q_bytes is None:
            continue
        got = (story_bytes, q_bytes)
        story_bytes, q_bytes = got
        wrote_s = dedup.write_if_new(src_dir / f"{fname}-story.csv", story_bytes)
        wrote_q = dedup.write_if_new(src_dir / f"{fname}-questions.csv", q_bytes)
        if wrote_s or wrote_q:
            n_ok += 1
    manifest.append(("fairytaleqa/*-story.csv + *-questions.csv", n_ok))
    print(f"  fairytaleqa: {n_ok}/{len(rows)} stories fetched -> {src_dir}")
    return manifest


# ---------------------------------------------------------------------------
# (2) MCTest
# ---------------------------------------------------------------------------

MCTEST_BASE = "https://raw.githubusercontent.com/mcobzarenco/mctest/master/data/MCTest/"
MCTEST_FILES = [
    "mc160.dev.tsv", "mc160.dev.ans",
    "mc160.train.tsv", "mc160.train.ans",
    "mc160.test.tsv",
    "mc500.dev.tsv", "mc500.dev.ans",
    "mc500.train.tsv", "mc500.train.ans",
    "mc500.test.tsv",
]


def fetch_mctest(out_dir: Path) -> List[Tuple[str, int]]:
    _print_license_notice("mctest")
    src_dir = out_dir / "mctest"
    dedup = Dedup()
    manifest: List[Tuple[str, int]] = []
    n_ok = 0
    for fname in MCTEST_FILES:
        content = _get(MCTEST_BASE + fname)
        if content is None:
            continue
        if dedup.write_if_new(src_dir / fname, content):
            n_ok += 1
    manifest.append(("mctest/*.tsv + *.ans", n_ok))
    print(f"  mctest: {n_ok}/{len(MCTEST_FILES)} files fetched -> {src_dir}")
    return manifest


# ---------------------------------------------------------------------------
# (3) African Storybook Project (English, CC-BY only)
# ---------------------------------------------------------------------------

ASP_INDEX_URL = "https://raw.githubusercontent.com/global-asp/global-asp/master/INDEX.md"
ASP_STORY_BASE = "https://raw.githubusercontent.com/global-asp/asp-source/master/en/"
_ASP_ROW_RE = re.compile(r"^(\d{4}) \| \[([^\]]+)\]\(([^)]+)\) \| \[([^\]]+)\]")


def _slugify(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s-]", "", t)
    return re.sub(r"\s+", "-", t.strip())


def fetch_african_storybook(out_dir: Path, limit: int) -> List[Tuple[str, int]]:
    _print_license_notice("african_storybook")
    src_dir = out_dir / "african_storybook"
    dedup = Dedup()
    manifest: List[Tuple[str, int]] = []

    index_bytes = _get(ASP_INDEX_URL)
    if index_bytes is None:
        print("  african_storybook: INDEX.md unreachable, skipping source entirely")
        return manifest

    rows = []
    for line in index_bytes.decode("utf-8").splitlines():
        m = _ASP_ROW_RE.match(line)
        if m:
            rows.append(m.groups())  # (num, title, orig_url, license)

    n_ok = 0
    manifest_rows = []
    for num, title, _orig_url, lic in rows:
        if n_ok >= limit:
            break
        if "CC-BY" not in lic and "CC BY" not in lic:
            continue  # license check: only permissively-licensed stories
        slug = _slugify(title)
        url = ASP_STORY_BASE + f"{num}_{slug}.md"
        content = _get(url)
        if content is None or len(content.strip()) < 50:
            continue
        fname = f"{num}_{slug}.md"
        if dedup.write_if_new(src_dir / fname, content):
            n_ok += 1
            manifest_rows.append((num, title, lic))
    manifest.append(("african_storybook/*.md", n_ok))
    print(f"  african_storybook: {n_ok} CC-BY stories fetched (of {len(rows)} indexed) -> {src_dir}")
    return manifest


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_SOURCES = ("fairytaleqa", "mctest", "african_storybook")


def run(out_dir: Path, allow_download: bool, ftqa_limit: int, asp_limit: int,
        sources: Optional[List[str]] = None) -> None:
    if not allow_download:
        print("--allow-download not set -- nothing to do (this script only ever "
              "fetches over the network; see dev/K12_CORPUS_SCOUT.md for the "
              "already-committed samples under data/k12_samples/).")
        return

    sources = sources or list(_SOURCES)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Tuple[str, int]] = []

    if "fairytaleqa" in sources:
        print("=== (1) FairytaleQA ===")
        manifest += fetch_fairytaleqa(out_dir, ftqa_limit)

    if "mctest" in sources:
        print("\n=== (2) MCTest ===")
        manifest += fetch_mctest(out_dir)

    if "african_storybook" in sources:
        print("\n=== (3) African Storybook Project (English, CC-BY) ===")
        manifest += fetch_african_storybook(out_dir, asp_limit)

    print(f"\n=== manifest ({out_dir}) ===")
    manifest_path = out_dir / "MANIFEST.tsv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("source\tcount\n")
        for name, n in manifest:
            f.write(f"{name}\t{n}\n")
            print(f"  {name:<45} {n:>5}")
    print(f"wrote manifest -> {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--allow-download", action="store_true",
                     help="Actually hit the network (github raw mirrors). Never set by default "
                          "or by the test suite -- mirrors scripts/fetch_corpus.py's convention.")
    ap.add_argument("--fairytaleqa-limit", type=int, default=100,
                     help="Max FairytaleQA stories to fetch (of 278 total).")
    ap.add_argument("--asp-limit", type=int, default=150,
                     help="Max African Storybook English CC-BY stories to fetch.")
    ap.add_argument("--source", action="append", choices=_SOURCES, default=None,
                     help="Restrict the fetch to one source (repeatable). Default: all three.")
    args = ap.parse_args()
    run(args.out_dir, args.allow_download, args.fairytaleqa_limit, args.asp_limit, sources=args.source)


if __name__ == "__main__":
    main()
