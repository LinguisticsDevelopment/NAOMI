"""scripts/fetch_corpus.py -- corpus scale-up fetcher (RESEARCH_NOTES M58b/M58d
tail: "corpus scale-up (statistical power)").

Assembles a LARGER public-domain prose corpus into ``data/corpus/`` so
``scripts/convert_corpus.py`` + ``scripts/eval_prose.py`` have hundreds of
episodes instead of ``n=43`` (2 source documents). Three pieces, matching
the existing ``data/corpus/`` convention (one ``.txt`` file per source, a
``#``-comment header, blank-line-separated blocks = passages, ``synthetic_*``
filenames vs. everything else -- see ``scripts/convert_corpus.py``'s
``corpus_group``):

(a) **nltk.corpus.gutenberg, real prose.** Of the 18 fileids nltk ships,
    four are graded-reader-adjacent (children's / simple third-person
    narrative prose); the rest are verse (``blake-poems``, ``milton-
    paradise``, ``whitman-leaves``), drama (``shakespeare-*``), scripture
    (``bible-kjv``), adult-register literary prose (``austen-*``,
    ``melville-moby_dick``, ``chesterton-*``) -- out of scope here. See
    :data:`NLTK_BOOKS`. Each book's raw text is cleaned (bracket title
    line dropped, heading/verse paragraphs filtered by
    :func:`_is_prose_paragraph`), then chunked into ~20-sentence passages
    by :func:`_chunk_paragraphs` (whole paragraphs only -- a passage never
    splits a paragraph, so it can never cross a document boundary
    either). Burgess is deduplicated against the existing
    ``data/corpus/real_gutenberg_busterbear.txt`` (M58a) by substring
    match, so "more Burgess" never repeats the M58a excerpt.

(b) **60 more hand-authored synthetic-prose paragraphs**, in the exact
    style of ``data/corpus/synthetic_prose_01.txt`` (M58a) -- simple
    declarative English about people/places/objects/attributes, pronouns
    mixed in on purpose -- split across two new files,
    ``synthetic_prose_02.txt`` and ``synthetic_prose_03.txt`` (30 each,
    entirely new entities/places/objects vs. file 01). Hardcoded (not
    templated/randomized): this keeps the fetcher's determinism trivial
    and the prose genuinely varied rather than combinatorially generated.

(c) **Direct Project Gutenberg download fallback**, only used if (a)+(b)
    fall short of a "big enough" corpus. Gated behind ``--allow-download``
    (never invoked by default, and never by the test suite) so the
    fetcher stays reproducible offline. In THIS run nltk alone supplies
    ~1350 real-prose sentences, comfortably over the fallback threshold,
    so (c) does not fire -- see the module docstring's own run log in
    RESEARCH_NOTES.md for the numbers actually assembled.

Determinism: every step here is a pure function of (a) nltk's bundled
corpus data (already on disk, no network) and (b) this file's own
hardcoded text -- no randomness, no timestamps in output, no dict-order
dependence (source lists are literal Python lists). Running this script
twice byte-for-byte reproduces every file it writes (see
tests/test_corpus_scale.py::test_fetch_is_deterministic).

Usage:
    python scripts/fetch_corpus.py [--out-dir data/corpus] [--allow-download]
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nsm_ct.corpus import iter_sentences  # noqa: E402 -- the SAME splitter the converter
                                           # uses, imported here ONLY to COUNT sentences
                                           # while chunking (never to reformat file content:
                                           # files on disk keep natural casing/punctuation,
                                           # exactly like the existing data/corpus/*.txt).

DATA_DIR = _ROOT / "data" / "corpus"

# ---------------------------------------------------------------------------
# (a) nltk.corpus.gutenberg book list.
#
# (fileid, out slug, title, author, year, sentence budget, dedup-against)
# ---------------------------------------------------------------------------
NLTK_BOOKS: List[Tuple[str, str, str, str, int, int, Optional[str]]] = [
    ("burgess-busterbrown.txt", "burgess_more", "The Adventures of Buster Bear",
     "Thornton W. Burgess", 1920, 250, "real_gutenberg_busterbear.txt"),
    ("carroll-alice.txt", "alice", "Alice's Adventures in Wonderland",
     "Lewis Carroll", 1865, 400, None),
    ("bryant-stories.txt", "bryant", "Stories to Tell to Children",
     "Sara Cone Bryant", 1918, 400, None),
    ("edgeworth-parents.txt", "edgeworth", "The Parent's Assistant",
     "Maria Edgeworth", 1796, 300, None),
]

# ---------------------------------------------------------------------------
# (c) direct-download fallback candidates (gutenberg.org plain text,
# cache.epub mirrors -- short public-domain children's books). Only
# consulted if --allow-download is passed AND (a)+(b) fall short of
# _MIN_REAL_SENTENCES.
# ---------------------------------------------------------------------------
_MIN_REAL_SENTENCES = 500

DIRECT_BOOKS: List[Tuple[str, str, str, str, int, int]] = [
    # (url, out slug, title, author, year, sentence budget)
    ("https://www.gutenberg.org/cache/epub/14838/pg14838.txt", "peter_rabbit",
     "The Tale of Peter Rabbit", "Beatrix Potter", 1902, 150),
    ("https://www.gutenberg.org/cache/epub/15517/pg15517.txt", "squirrel_nutkin",
     "The Tale of Squirrel Nutkin", "Beatrix Potter", 1903, 150),
    ("https://www.gutenberg.org/cache/epub/11339/pg11339.txt", "aesop_fables",
     "Aesop's Fables (V. S. Vernon Jones translation)", "Aesop", 1912, 200),
]


# ---------------------------------------------------------------------------
# paragraph cleaning / filtering
# ---------------------------------------------------------------------------

def _strip_bracket_title(text: str) -> str:
    """Drops a leading ``[Title by Author Year]`` line (nltk's own gutenberg
    convention -- see every fileid's ``raw()[:0]`` in the corpus README)."""
    stripped = text.lstrip()
    if stripped.startswith("["):
        end = stripped.find("]")
        if end != -1:
            return stripped[end + 1:]
    return text


def _raw_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_bracket_title(text)
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _is_prose_paragraph(para: str) -> bool:
    """True for an ordinary narrative-prose paragraph; False for a
    chapter/section heading, a roman-numeral marker, or a verse/poem block
    (short, ragged lines -- Gutenberg children's-story collections like
    ``bryant-stories.txt`` interleave rhymes between the prose tales).

    A heuristic, not a parser: false negatives (a real prose paragraph
    dropped) just cost a little corpus volume, which the sentence budgets
    below have headroom for; false positives (verse kept as "prose") would
    pollute the corpus, so this errs toward rejecting anything ambiguous.
    """
    lines = [ln for ln in para.splitlines() if ln.strip()]
    if not lines:
        return False
    joined = " ".join(ln.strip() for ln in lines)
    words = joined.split()
    if len(words) < 15:
        return False  # too short to be a real narrative paragraph
    if len(lines) >= 3:
        avg_line_words = sum(len(ln.split()) for ln in lines) / len(lines)
        if avg_line_words < 8:
            return False  # ragged short lines -> verse/poem formatting
    return True


def _normalize_paragraph(para: str) -> str:
    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
    return " ".join(lines)


def _load_dedup_text(fname: Optional[str]) -> str:
    if fname is None:
        return ""
    path = DATA_DIR / fname
    if not path.exists():
        return ""
    lines = [ln for ln in path.read_text(encoding="utf-8").split("\n")
             if not ln.strip().startswith("#")]
    return " ".join(" ".join(lines).split())


def _is_duplicate(para: str, dedup_text: str) -> bool:
    if not dedup_text:
        return False
    probe = " ".join(para.split())[:80]
    return probe in dedup_text


# ---------------------------------------------------------------------------
# chunking: whole paragraphs only accumulate into a ~target-sentence block,
# so a block/passage NEVER splits a paragraph and (by construction, since
# this is called once per book) never crosses a document boundary.
# ---------------------------------------------------------------------------

def _chunk_paragraphs(paragraphs: List[str], target_sentences: int = 20) -> List[str]:
    blocks: List[str] = []
    cur: List[str] = []
    cur_n = 0
    for para in paragraphs:
        n = len(iter_sentences(para))
        if n == 0:
            continue
        cur.append(para)
        cur_n += n
        if cur_n >= target_sentences:
            blocks.append(" ".join(cur))
            cur = []
            cur_n = 0
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def _budget_blocks(blocks: List[str], sentence_budget: int) -> List[str]:
    out: List[str] = []
    total = 0
    for b in blocks:
        n = len(iter_sentences(b))
        if total >= sentence_budget:
            break
        out.append(b)
        total += n
    return out


# ---------------------------------------------------------------------------
# (a) assemble one nltk book
# ---------------------------------------------------------------------------

def assemble_nltk_book(fileid: str, slug: str, title: str, author: str, year: int,
                        sentence_budget: int, dedup_against: Optional[str]) -> Tuple[str, int]:
    from nltk.corpus import gutenberg  # local import: only needed for this path

    try:
        gutenberg.raw(fileid)
    except LookupError:
        import nltk
        nltk.download("gutenberg", quiet=True)

    raw = gutenberg.raw(fileid)
    dedup_text = _load_dedup_text(dedup_against)

    paragraphs = []
    for para in _raw_paragraphs(raw):
        if not _is_prose_paragraph(para):
            continue
        norm = _normalize_paragraph(para)
        if _is_duplicate(norm, dedup_text):
            continue
        paragraphs.append(norm)

    blocks = _chunk_paragraphs(paragraphs, target_sentences=20)
    blocks = _budget_blocks(blocks, sentence_budget)
    n_sentences = sum(len(iter_sentences(b)) for b in blocks)

    dedup_note = (f" Paragraphs already used in data/corpus/{dedup_against} "
                  f"(M58a) are excluded by substring match." if dedup_against else "")
    header = (
        f"# REAL PROSE (corpus scale-up) -- {title} by {author} ({year}), public domain, "
        f"via nltk.corpus.gutenberg fileid '{fileid}' (cached here so the converter/tests "
        f"need no network access). Heading/verse paragraphs filtered "
        f"(scripts/fetch_corpus.py::_is_prose_paragraph); surviving paragraphs chunked into "
        f"~20-sentence passages, whitespace-normalized, unedited otherwise.{dedup_note} "
        f"One blank-line-separated block = one passage. {n_sentences} sentences, "
        f"{len(blocks)} passages."
    )
    content = header + "\n\n" + "\n\n".join(blocks) + "\n"
    return content, n_sentences


# ---------------------------------------------------------------------------
# (b) 60 more hand-authored synthetic-prose paragraphs (30 + 30), style-
# matched to data/corpus/synthetic_prose_01.txt: simple declarative English,
# a person/role in a place or a three-party transfer, pronouns and definite
# descriptions mixed in, varied entities/places/objects vs. file 01.
# ---------------------------------------------------------------------------

SYNTHETIC_PROSE_02 = """\
Ben is in the workshop. He repairs old clocks. The workshop smells of oil and wood. Sunlight falls through a dusty window.

Carl gave Nina the lantern. Nina carried it down the stairs. The lantern flickered in the draft. She set it on the windowsill.

The old sailor is on the dock. He coils a length of rope. The dock creaks under his boots. Gulls circle above the harbor.

Dana and Leo are in the orchard. They are picking apples. Dana filled a wooden crate. Leo climbed the tallest tree.

Ruth handed Sam the parcel. Sam untied the string carefully. The parcel held a small clock. He wound it and listened.

The blacksmith is in the forge. He hammers a glowing bar. The forge glows with orange light. Sparks scatter across the floor.

Ivy is in the meadow. She watches the grazing sheep. The meadow stretches to the hills. A lark sings somewhere overhead.

Max passed Zoe the ribbon. Zoe tied it around the basket. The ribbon was a deep red. She hung the basket by the door.

The old potter is in the studio. He shapes a lump of clay. The studio is lined with shelves. Finished bowls dry in the sun.

Owen is in the stable. He brushes a grey mare. The stable smells of hay and leather. A cat naps in the corner.

Rita gave Jack the compass. Jack studied the spinning needle. The compass pointed toward the north. He tucked it into his coat.

Fiona is in the lighthouse. She climbs the narrow stairs. The lighthouse stands on a rocky point. Waves crash against the base below.

Ray and Nora are in the vineyard. They are trimming the vines. Ray carried the wooden ladder. Nora gathered the fallen leaves.

Eli handed Wendy the scarf. Wendy wrapped it around her neck. The scarf was soft grey wool. She thanked him with a smile.

The weaver is in the cottage. She works at a tall loom. The cottage is warm and quiet. Thread spools line the far wall.

Hugo is in the courtyard. He sweeps the fallen leaves. The courtyard has a stone fountain. Pigeons gather near the water.

June gave Otto the medallion. Otto held it up to the light. The medallion bore an old crest. He slipped it into his pocket.

The miner is in the quarry. He loads stone onto a cart. The quarry echoes with distant hammering. Dust rises in the still air.

Vera is in the greenhouse. She tends rows of young herbs. The greenhouse traps the morning heat. Bees drift among the blossoms.

Dean passed Lola the kettle. Lola set it on the stove. The kettle began to hiss softly. Steam curled toward the ceiling.

The cobbler is in the workshop. He stitches a worn boot. The workshop is cluttered with leather. A bell hangs above the door.

Gus is in the pasture. He counts the grazing cattle. The pasture runs along the river. A hawk circles high above.

Mia gave Ted the quill. Ted dipped it in dark ink. The quill scratched across the page. He signed his name at the bottom.

The innkeeper is in the tavern. He wipes down the long counter. The tavern is loud in the evening. A fire crackles in the hearth.

Freya is in the attic. She sorts through an old trunk. The attic is thick with dust. Light slips through a small window.

Kim handed Arlo the coin. Arlo turned it over twice. The coin was worn nearly smooth. He dropped it in his purse.

The shepherd is in the pasture. He watches the newborn lambs. The pasture is fenced with grey stone. Clouds drift low over the hills.

Nadia is in the cellar. She stacks the wine barrels. The cellar is cool and dark. A single lamp hangs from a hook.

Silas gave Priya the satchel. Priya slung it over her shoulder. The satchel was heavy with books. She set off down the road.

The watchman is on the tower. He scans the road below. The tower rises above the town wall. Torches burn along the ramparts.
"""

SYNTHETIC_PROSE_03 = """\
Otto is in the printshop. He sets type by hand. The printshop smells of fresh ink. Presses clatter through the afternoon.

Priya gave Silas the brooch. Silas pinned it to his coat. The brooch caught the lamplight. He admired it in the mirror.

The herbalist is in the garden. She gathers sprigs of mint. The garden is bordered by low hedges. Bees hum among the flowers.

Nadia and Otto are in the cellar. They are counting the barrels. Nadia marked each one with chalk. Otto rolled the last barrel aside.

Wendy handed Eli the ledger. Eli opened it on the desk. The ledger listed every sale. He checked the numbers twice.

The ferryman is on the river. He guides the boat across. The river runs fast and cold. Mist hangs low over the water.

Zoe is in the depot. She stacks crates by the door. The depot echoes with rolling wheels. A whistle sounds from the yard.

Jack gave Rita the feather. Rita placed it in a book. The feather was pale and long. She pressed the pages shut.

The ranger is in the forest. He marks a trail with paint. The forest is thick with pine. A deer watches from the shade.

Sam and Ruth are in the orchard. They are stacking the crates. Sam counted the apples twice. Ruth swept the fallen leaves.

Lola handed Dean the spool. Dean threaded it through the loom. The spool held fine blue thread. He began to weave a pattern.

The cartographer is in the study. She traces a line on the map. The study is stacked with old charts. Ink stains cover her fingers.

Arlo is in the hangar. He checks the engine gauges. The hangar is cold before dawn. A single bulb lights the workbench.

Ted gave Mia the token. Mia turned it in her palm. The token bore a small star. She kept it in a pouch.

The glassblower is in the workshop. He shapes a glowing bulb. The workshop hums with the furnace. Colored glass lines the shelves.

Nora and Ray are on the terrace. They are watering the plants. Nora filled the tin cans. Ray trimmed the drooping stems.

Vera handed Gus the jar. Gus sealed it with wax. The jar held dark honey. He set it on the shelf.

The falconer is in the field. She releases a waiting hawk. The field stretches beneath open sky. The hawk climbs in wide circles.

Fiona is in the cove. She searches the wet sand. The cove is quiet at low tide. Shells scatter along the shoreline.

Leo gave Dana the bell. Dana hung it above the gate. The bell rang softly in the wind. She smiled at the sound.

The midwife is in the cottage. She checks on the sleeping baby. The cottage is warm by the fire. A kettle simmers on the stove.

Owen and Fiona are in the harbor. They are loading the crates. Owen counted every box. Fiona checked the ship's manifest.

Nina gave Carl the cloak. Carl wrapped it around his shoulders. The cloak was heavy grey wool. He stepped out into the rain.

The beekeeper is in the meadow. He lifts a frame from the hive. The meadow buzzes with a steady sound. Golden wax drips from his gloves.

Otto handed Vera the spool. Vera set it beside the loom. The spool was wound with red thread. She began the morning's weaving.

The observatory keeper is on the tower. He adjusts the great telescope. The tower stands above the quiet town. Stars gather thick overhead.

Hugo and June are in the plaza. They are hanging paper lanterns. Hugo climbed the wooden ladder. June handed up the string of lights.

Dean gave Nadia the hammer. Nadia struck the loose nail. The hammer rang against the wood. She checked the shelf for balance.

The carpenter is in the workshop. He planes a long plank smooth. The workshop is filled with sawdust. Sunlight falls across the bench.

Priya and Ted are in the den. They are sorting old letters. Priya read one aloud. Ted filed the rest away.
"""

_SYNTHETIC_HEADER_02 = (
    "# SYNTHETIC PROSE (corpus scale-up) -- hand-authored graded-reader-style paragraphs,\n"
    "# NOT drawn from any real corpus, matching the style of synthetic_prose_01.txt (M58a)\n"
    "# but with entirely new entities/places/objects. One blank-line-separated block = one\n"
    "# passage.\n\n"
)
_SYNTHETIC_HEADER_03 = (
    "# SYNTHETIC PROSE (corpus scale-up) -- hand-authored graded-reader-style paragraphs,\n"
    "# NOT drawn from any real corpus, matching the style of synthetic_prose_01.txt (M58a),\n"
    "# continuing synthetic_prose_02.txt with further new/reused entities (people move\n"
    "# between rooms and hand each other objects across both files, same as real narrative\n"
    "# text would). One blank-line-separated block = one passage.\n\n"
)


def synthetic_files() -> List[Tuple[str, str]]:
    """Returns [(filename, content), ...] for the two new synthetic files."""
    return [
        ("synthetic_prose_02.txt", _SYNTHETIC_HEADER_02 + SYNTHETIC_PROSE_02),
        ("synthetic_prose_03.txt", _SYNTHETIC_HEADER_03 + SYNTHETIC_PROSE_03),
    ]


# ---------------------------------------------------------------------------
# (c) direct-download fallback (gated behind --allow-download)
# ---------------------------------------------------------------------------

_PG_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_PG_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)


def _strip_pg_boilerplate(text: str) -> str:
    start_m = _PG_START_RE.search(text)
    if start_m:
        text = text[start_m.end():]
    end_m = _PG_END_RE.search(text)
    if end_m:
        text = text[:end_m.start()]
    return text


def fetch_direct_book(url: str, title: str, author: str, year: int, sentence_budget: int) -> Tuple[str, int]:
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 -- deliberate, gated by --allow-download
        raw_bytes = resp.read()
    raw = raw_bytes.decode("utf-8", errors="replace")
    raw = _strip_pg_boilerplate(raw)

    paragraphs = []
    for para in _raw_paragraphs(raw):
        if not _is_prose_paragraph(para):
            continue
        paragraphs.append(_normalize_paragraph(para))

    blocks = _chunk_paragraphs(paragraphs, target_sentences=20)
    blocks = _budget_blocks(blocks, sentence_budget)
    n_sentences = sum(len(iter_sentences(b)) for b in blocks)

    header = (
        f"# REAL PROSE (corpus scale-up, direct download) -- {title} by {author} ({year}), "
        f"public domain, downloaded from {url} and cached here (--allow-download fallback, "
        f"only used when nltk.corpus.gutenberg alone did not reach the minimum real-sentence "
        f"target). Heading/verse paragraphs filtered, whitespace-normalized, unedited "
        f"otherwise. One blank-line-separated block = one passage. {n_sentences} sentences, "
        f"{len(blocks)} passages."
    )
    content = header + "\n\n" + "\n\n".join(blocks) + "\n"
    return content, n_sentences


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(out_dir: Path, allow_download: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Tuple[str, int]] = []

    print("=== (a) nltk.corpus.gutenberg books ===")
    real_total = 0
    for fileid, slug, title, author, year, budget, dedup_against in NLTK_BOOKS:
        content, n = assemble_nltk_book(fileid, slug, title, author, year, budget, dedup_against)
        out_name = f"real_gutenberg_{slug}.txt"
        (out_dir / out_name).write_text(content, encoding="utf-8")
        manifest.append((out_name, n))
        real_total += n
        print(f"  {fileid:<28} -> {out_name:<32} {n:>5} sentences  ({title}, {author}, {year})")

    if real_total < _MIN_REAL_SENTENCES and allow_download:
        print(f"\n=== (c) real-sentence total {real_total} < {_MIN_REAL_SENTENCES} -- "
              f"topping up via direct download ===")
        for url, slug, title, author, year, budget in DIRECT_BOOKS:
            if real_total >= _MIN_REAL_SENTENCES:
                break
            try:
                content, n = fetch_direct_book(url, title, author, year, budget)
            except Exception as exc:  # noqa: BLE001 -- network is best-effort, never fatal
                print(f"  SKIP {url}: {type(exc).__name__}: {exc}")
                continue
            out_name = f"real_gutenberg_{slug}.txt"
            (out_dir / out_name).write_text(content, encoding="utf-8")
            manifest.append((out_name, n))
            real_total += n
            print(f"  {url} -> {out_name:<32} {n:>5} sentences  ({title}, {author}, {year})")
    elif real_total < _MIN_REAL_SENTENCES:
        print(f"\n(c) real-sentence total {real_total} < {_MIN_REAL_SENTENCES}, but "
              f"--allow-download not set -- skipping the direct-download fallback.")

    print("\n=== (b) hand-authored synthetic prose ===")
    synth_total = 0
    for fname, content in synthetic_files():
        (out_dir / fname).write_text(content, encoding="utf-8")
        n = sum(len(iter_sentences(b)) for b in content.split("\n\n") if b.strip() and not b.strip().startswith("#"))
        manifest.append((fname, n))
        synth_total += n
        print(f"  {fname:<32} {n:>5} sentences")

    grand_total = sum(n for _f, n in manifest)
    print(f"\n=== manifest ({out_dir}) ===")
    for fname, n in manifest:
        print(f"  {fname:<32} {n:>5} sentences")
    print(f"  {'TOTAL (this run)':<32} {grand_total:>5} sentences")

    all_files_total = 0
    for path in sorted(glob.glob(str(out_dir / "*.txt"))):
        text = Path(path).read_text(encoding="utf-8")
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("#")]
        text = "\n".join(lines)
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        all_files_total += sum(len(iter_sentences(b)) for b in blocks)
    print(f"\n=== grand total across ALL data/corpus/*.txt: {all_files_total} sentences ===")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    ap.add_argument("--allow-download", action="store_true",
                     help="Permit the Project Gutenberg direct-download fallback (part c). "
                          "Off by default -- the fetcher stays reproducible offline.")
    args = ap.parse_args()
    run(Path(args.out_dir), args.allow_download)


if __name__ == "__main__":
    main()
