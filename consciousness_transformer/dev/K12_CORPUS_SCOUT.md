# K-12 Comprehension-QA Corpus Scout

Status: SCOUT REPORT, nothing trained. Author: k12-scout session, 2026-09-08.
Motivated by `dev/UNIVERSAL_ENCODER_DESIGN.md` §0/§5 (the pivot away from
fabricated clause-holdout QA toward real graded reading material with real
comprehension questions, "not stated" a first-class answer, bottom-up
sentences -> passages -> long reading) and `dev/ROADMAP_LONG_TERM.md`.
Nothing has been sourced before this scout; this is the first pass.

**Network reality check (read this before the table):** this sandbox's
egress proxy allows `raw.githubusercontent.com`, `api.github.com` (though
`api.github.com` is itself intercepted and scoped to this repo only, so it
could not be used to browse *other* repos), and general PyPI, but **blocks**
`huggingface.co`, `gutenberg.org`, `microsoft.com`, `cs.cmu.edu`,
`storyweaver.org.in`, `africanstorybook.org`, `digitallibrary.io`,
`bloomlibrary.org`, `coreknowledge.org`, `engageny.org`, `wikipedia.org` and
`wikimedia.org` outright (`CONNECT tunnel failed, response 403` -- an
organization egress policy, not a transient failure; confirmed by repeated
probes to multiple hosts, some of which succeeded and some of which did not
in a consistent pattern). Practically: **any source that is GitHub-mirrored
could be evaluated AND fetched for real; any source that lives only on its
own domain could only be evaluated by search/citation, not fetched, in this
run.** That constraint, not source quality, is why the TOP-3 below skews
toward GitHub-hosted sources -- it is an environment limitation to flag to
whoever runs `scripts/fetch_k12.py` next from a less restricted network.

## Comparison table

| Source | License (verify link) | Grades | Ships Qs? | Q type | Size (as evaluated) | Fetch method | Parse yield on sample |
|---|---|---|---|---|---|---|---|
| **FairytaleQA** | Apache 2.0 — [LICENSE](https://github.com/uci-soe/FairytaleQAData/blob/main/LICENSE) | K-8 (corpus-level, not per-story) | **Yes** | free-text, gold answer(s), `local`/`summary` + `explicit`/`implicit` marked. No "not stated" items. | 278 stories, 10,580 QA pairs, ~3,300 sections | GitHub raw (`story_meta.csv` index -> per-story CSV pair) | **94.0%** (47/50) |
| **MCTest** | Microsoft Research data terms (research use) — **not independently re-verified**, see caveats | ~7yo (mc160), slightly harder (mc500); not finer-marked | **Yes** | 4-way MC, 1 gold letter/question. No "not stated" option. | mc160: 660 stories (2,640 Qs); mc500: 500 stories (2,000 Qs) | GitHub mirror raw (`.tsv`+`.ans`) | **88.0%** (44/50) |
| **African Storybook Project** (English) | CC-BY (per-story, verified per-story via `INDEX.md`) — [example story](https://github.com/global-asp/asp-source/blob/master/en/0001_a-very-tall-man.md) | Leveled 1-4 on ASP's own site; level not present in the fetched Markdown | **No** — real gap | n/a (would need NAOMI's own question synthesis) | 105 English titles indexed here (ASP's full catalog is far larger; this index is only the cross-language-translation subset) | GitHub raw (`INDEX.md` -> per-story `.md`) | **94.0%** (47/50) |
| Children's Book Test (CBT) | Unverified — original host (`thespermwhale.com`, Facebook bAbI) blocked by this sandbox's egress policy | Not marked (Gutenberg children's-book vocabulary) | Partial — cloze-style: 1 blanked word/passage, not an explicit question | cloze (fill-the-blank), 10 candidate words incl. gold | ~669K instances (V/P/NE/CN splits) reported in the paper | Blocked in this env; would need `curl` from an unrestricted network | not run (unreachable) |
| RACE | Research/educational use only per CMU's stated terms (not independently re-fetched — `cs.cmu.edu` blocked) | Middle (RACE-M) / High (RACE-H) school, explicitly split | **Yes** | 4-way MC | ~28K passages, ~100K questions | Blocked in this env (`cs.cmu.edu`); a `race-c` GitHub companion repo exists for a smaller add-on set | not run (unreachable) |
| DREAM | **Non-commercial research only** — confirmed via fetched `license.txt` | EFL-exam level (Chinese learners of English), not native K-12 grade bands | **Yes** | 4-way MC (84% non-extractive — requires inference, not lookup) | 10,197 questions / 6,444 dialogues | GitHub raw (`nlpdata/dream`), fetchable | not run (deprioritized: dialogue-format, non-commercial, weak K-12-native fit) |
| NarrativeQA | Apache 2.0 for the QA/index files (confirmed via fetched `LICENSE`); full story text is fetched separately per-document from third-party `story_url`s (Gutenberg, script sites — mixed licenses) | Wide (children's books through adult novels/movies), not grade-marked | **Yes** | free-text, 2 human answers/question | ~1,567 documents, ~46K QA pairs | GitHub raw (`deepmind/narrativeqa`) for `qaps.csv`/`documents.csv`; full story text needs a second per-document fetch this scout did not run | not run (answers are grounded in Wikipedia PLOT SUMMARIES, not the full passage — a real mismatch with NAOMI's Episode contract, see caveats) |
| CoQA (children's-lit subset) | Mixed by domain: literature passages CC BY-SA 4.0, but the **children's-story** slice is drawn from MCTest (MSR-LA) | Domain-mixed, not K-12-graded | **Yes** | free-text, conversational (multi-turn, prior-turn-dependent) | 8k conversations / 127k Qs total, across 7 domains | `stanfordnlp/coqa` GitHub repo has code/README only — the actual `.json` data files are hosted on `stanfordnlp.github.io`, which is blocked in this sandbox | not run (data files unreachable) |
| CommonLit Ease-of-Readability (CLEAR) | Corpus/metadata: CC BY-NC-SA 4.0 (non-commercial); individual excerpts carry their OWN licenses (mostly public domain, some CC BY / CC BY-SA) | **3rd-12th grade, explicit numeric readability score per excerpt** — best grade-granularity of anything evaluated | **No** | n/a — a readability-difficulty benchmark, not a QA dataset | ~5,000 excerpts | GitHub raw (`scrosseye/CLEAR-Corpus`), fetchable | not run (no questions; best used as a DIFFICULTY-ORDERING signal for the curriculum, not a training source) |
| Core Knowledge (CKLA) readers | Free to download/print/adapt; **may not be sold** without CKF permission; some grade-5 units are separately CC-licensed | K-5, explicit by curriculum design | **Yes** (per-chapter comprehension questions in the Teacher's Guide) | short-answer, per-chapter | Multi-unit curriculum, PDF-only | `coreknowledge.org` blocked in this sandbox; PDFs, not machine-readable without OCR/parsing work either way | not run (unreachable + PDF-only) |
| EngageNY ELA modules | NY State Education Dept — public domain / CC0-style (state government work) per general knowledge, not re-verified here | K-12, explicit by module | **Yes** (embedded assessments) | mixed (short-answer, essay prompts) | Large (full state ELA curriculum) | `engageny.org` blocked in this sandbox | not run (unreachable) |
| StoryWeaver (Pratham) | CC BY (platform-wide) | Leveled 1-4, explicit | **No** (leveled readers, no comprehension Qs) | n/a | 53K+ storybooks, ~330 languages | Primary API host (`storyweaver.org.in`) blocked; GitHub org (`PrathamBooks/StoryWeaverOpen`) is platform CODE, not story content; a third-party `sushi-chef-pratham-books-storyweaver` scraper exists but targets the same blocked host | not run (unreachable) |
| Global Digital Library | Aggregator: mostly CC-BY/CC-BY-SA, some CC-BY-NC-SA (per-book) | Leveled, explicit (aggregates StoryWeaver/ASP/Bookdash/etc.) | **No** | n/a | Aggregates the sources above | `book-api` code is on GitHub (`GlobalDigitalLibraryio/book-api`) but the live API (`digitallibrary.io`) is blocked in this sandbox | not run (unreachable) |
| Bloom Library | Per-book, imports licenses from its source libraries (mostly CC-BY family) | Leveled, explicit | **No** | n/a | Aggregates ASP/StoryWeaver/Let's Read/etc. | `bloomlibrary.org` blocked in this sandbox | not run (unreachable) |
| Bookbot | Proprietary app; leveled readers behind a commercial product | Leveled, explicit | Unclear (app-embedded, not a downloadable dataset) | n/a | n/a | No open dataset/API found; likely not reusable as a training corpus without a licensing agreement | not evaluated further |
| CK-12 | CC BY-NC (FlexBooks) | Grades 6-12, subject textbooks (not leveled *readers*) | Some (embedded review questions per section) | mixed | Large (full K-12 STEM+ELA library) | `ck12.org` not directly probed (out of scope given non-commercial license + subject-textbook rather than narrative-reading shape) | not run |
| McGuffey Readers | Public domain | Grades 1-6 (First - Sixth Eclectic Reader), explicit by book | **No** (period readers, no embedded comprehension Qs in the plain-text editions) | n/a | Already partially in `data/corpus/real_gutenberg_mcguffey_first_reader.txt` (grade 1 only, via `scripts/fetch_corpus.py`'s `--allow-download` GITenberg mirror) | Same GITenberg pattern as the existing fetcher; grades 2-6 not yet fetched | not re-run (already measured as part of the existing Gutenberg corpus) |
| Simple English Wikipedia / Wikijunior | CC BY-SA / GFDL | Not graded (simplified register only, not leveled) | **No** | n/a | Large | `wikipedia.org`/`wikimedia.org` blocked in this sandbox (dumps and API both) | not run (unreachable) |
| ReadWorks | Free-with-account; terms restrict bulk/automated redistribution | K-12, explicit grade bands, curated comprehension sets | **Yes** | mixed (MC + short-answer) | Large curated library | Requires a login-gated account; terms of service likely prohibit the kind of bulk scripted fetch this scout does elsewhere | not evaluated further (terms preclude it) |
| Project Gutenberg children's series w/ embedded exercises | Public domain (text); varies for any modern added exercises | Not systematically graded | Rare | n/a | n/a | `gutenberg.org` itself blocked in this sandbox; GITenberg GitHub mirrors work (already the exact pattern `scripts/fetch_corpus.py` uses) | Already covered by the existing Gutenberg corpus + M62 probe (81.2% overall, see below) |

**Parse-yield methodology and baseline.** "Parse yield" reproduces
`scripts/probe_m62_gold_volume.py`'s bar for teacher gold (the deterministic
`quantum_parser` teacher, default `ParserConfig`, must produce a discourse
graph with >=1 clause carrying a real non-punctuation SUBJECT) on 50
sentences per source, pulled from the samples actually fetched into
`data/k12_samples/`. The existing in-repo Gutenberg baseline
(`dev/M62_GOLD_VOLUME_PROBE.md`, 500 sentences across `real_*.txt`) is
**81.2% overall**, dragged down hard in the shortest-sentence bin (A<=8
tokens: 50.4%, dominated by bare interjections/imperatives per
`UNIVERSAL_ENCODER_DESIGN.md` §9). All three K-12 candidates measured here
(FairytaleQA 94.0%, African Storybook 94.0%, MCTest 88.0%) beat that overall
baseline outright, confirming §0's prediction that graded children's
material parses cleaner than 19th-century Gutenberg prose -- with no
special-casing of short sentences (these are ordinary run-of-the-mill
grades-1-4 declarative narrative, not filtered for parse-friendliness).

## TOP-3 recommendation

**1. FairytaleQA** -- the strongest fit by a wide margin: it is the ONE
source in this list that already ships exactly what §0 asks for --
human-expert-written explicit questions over real passages, WITH a
machine-readable explicit/implicit distinction (`ex-or-im1`/`ex-or-im2`)
that maps directly onto NAOMI's read-then-answer objective (`explicit` =
answer traceable to the passage; `implicit` = requires inference, a genuine
stepping-stone toward "not stated" without literally being it). Apache 2.0
means no license friction. 94.0% parse yield. The one gap: no
"not-answerable" items -- every question has a supported gold answer, so
"not stated" as a first-class answer still needs to come from elsewhere
(synthetic distractor questions, or a later source).

**2. MCTest** -- the closest thing in this list to a pure grade-school
multiple-choice comprehension benchmark (mc160 explicitly targets ~7-year-old
readers), with a clean fixed schema (4 questions/story, 4 options/question,
1 gold letter) that is trivial to turn into NAOMI Episodes with MC
distractors already provided (no need to synthesize same-relation
distractors the way `nsm_ct.corpus.make_episodes` does for converted prose).
88.0% parse yield. Caveat: license is Microsoft Research's own data terms,
not independently re-verified here (host blocked) -- confirm before any
non-research use.

**3. African Storybook Project (English, CC-BY)** -- included NOT because it
ships questions (it doesn't -- a real, stated gap) but because it is the
only evaluated source that is BOTH a genuine leveled reader (ASP's own
grading, unlike Gutenberg-era children's lit which is graded only by proxy)
AND fully open (CC-BY, no non-commercial restriction, no research-only
clause) AND has the highest measured parse yield of the three (94.0%,
tied with FairytaleQA). It slots into the EXISTING
`nsm_ct.corpus` conversion pipeline exactly like `scripts/fetch_corpus.py`'s
Gutenberg prose does today (auto-synthesized "queried role" questions), so
adopting it costs zero new question-generation engineering -- it costs one
new prose source with dramatically simpler syntax and non-Western,
non-19th-century vocabulary and settings, which is a real diversification
win for the bottom rung of the ladder.

*Runner-up worth a second look once network access is less restricted:*
**RACE** (real exam-grade MC questions, explicit middle/high-school split --
exactly the material to extend the ladder upward once grade-1-4 is solid)
and **NarrativeQA** (real expert QA, Apache 2.0 -- but its answers are
grounded in Wikipedia plot summaries rather than the full passage, which is
a genuine architectural mismatch with NAOMI's Episode contract, "answer
comes from a memory read over the CONTEXT the sentences built" -- would need
either the full story text substituted as context, with unclear answer
support, or treatment as its own separate summarization-QA task type).

## Proposed graded curriculum order

1. **Sentence-level (grade K-1):** African Storybook's simplest titles
   (very short declarative sentences, concrete nouns) + the existing
   `synthetic_prose_*` + McGuffey First Reader (already in `data/corpus/`).
   No real questions yet at this rung -- use NAOMI's existing auto-synthesized
   "queried role" questions (PLACE/RECIPIENT/AGENT), same objective shape as
   the eventual grade-2+ rungs, just simpler syntax.
2. **Short passage, explicit-only (grade 1-3):** FairytaleQA questions
   filtered to `ex-or-im1 == explicit` and `local-or-sum == local` (one
   section, one traceable fact) + MCTest mc160 (4-way MC, ~7yo level).
   First rung with REAL human-authored questions and REAL gold answers.
3. **Short passage, with inference (grade 3-5):** FairytaleQA
   `implicit` questions and `summary`-level questions (span multiple
   sections) + MCTest mc500. Introduces "the answer isn't a single sentence
   away" without yet requiring "not stated."
4. **Long reading (grade 6+, later milestone):** RACE (once license
   re-verified and network access allows fetching it) for real exam-grade
   multi-hop MC over longer passages; CLEAR-Corpus excerpts (no questions,
   but its per-excerpt readability score is a ready-made DIFFICULTY LADDER
   to pace this rung, independent of any one Q&A source) to select passages
   at the right difficulty as the ladder climbs.
5. **"Not stated" abstention (cross-cutting, needs new work):** none of the
   evaluated sources ship native unanswerable items. The cheapest path is
   synthetic: for every real passage+question pair above, generate a
   same-topic DISTRACTOR question the passage does not actually answer
   (e.g. swap the queried relation to one never mentioned) and label its
   gold answer "not stated" -- mirrors how `make_episodes` already builds MC
   distractors from same-relation values elsewhere in the passage, just
   applied to whole questions instead of answer options.

## Licensing caveats (one paragraph)

Of the TOP-3, only FairytaleQA (Apache 2.0) and African Storybook
(CC-BY, verified per-story) are unambiguously clear for both research and
commercial reuse with attribution; MCTest's Microsoft Research data terms
were NOT independently re-verified in this environment (the mirror this
scout fetched from does not carry Microsoft's own LICENSE file, and
microsoft.com was unreachable through this sandbox's egress policy), so
treat it as research-use-only until the canonical terms are pulled from
Microsoft Research directly. Several sources evaluated but not adopted
carry harder restrictions worth remembering if picked up later: DREAM and
CoQA's children's-story slice are non-commercial/MSR-derived; CommonLit's
CLEAR corpus wrapper is CC BY-NC-SA even though many of its individual
excerpts are public domain; Core Knowledge's CKLA readers are free to use
but may not be sold or repackaged for sale without the Foundation's
permission; and ReadWorks' account-gated terms likely preclude the kind of
scripted bulk fetch this scout otherwise favors. Every fetch this scout
actually ran writes attribution + license into a `SOURCE.md` next to the
data (`data/k12_samples/<source>/SOURCE.md`), mirroring this repo's
existing `NOTICE` file convention -- extend that same pattern for any new
source added later.

## What could not be verified

- **MCTest's exact license text** -- Microsoft Research's own MCTest page
  (`microsoft.com`) is blocked by this sandbox's egress policy; only
  secondary/search-engine characterizations were available.
- **RACE's license and content** -- `cs.cmu.edu` (the canonical host) is
  blocked; not independently fetched or re-verified, characterized from
  search results only.
- **CBT (Children's Book Test)** -- `thespermwhale.com` (the bAbI/CBT host)
  is blocked; license and exact format not independently verified.
- **CoQA's actual data files** -- `stanfordnlp.github.io` (where the real
  `.json` files live; the GitHub repo itself is code+docs only) is blocked.
- **CKLA, EngageNY, StoryWeaver's live API, Global Digital Library's live
  API, Bloom Library, CK-12, Simple Wikipedia/Wikijunian, ReadWorks** --
  all live on hosts this sandbox's egress policy blocks outright
  (`coreknowledge.org`, `engageny.org`, `storyweaver.org.in`,
  `digitallibrary.io`, `bloomlibrary.org`, `ck12.org`, `wikipedia.org`,
  `wikimedia.org`, `readworks.org`) -- evaluated by search/citation only,
  not fetched. Re-run this scout (or just `scripts/fetch_k12.py` plus manual
  follow-up for these specific hosts) from a network without that
  restriction to actually pull samples.
- **African Storybook's per-story reading LEVEL** -- present on ASP's own
  website per-story page, not captured in the `asp-source` Markdown mirror
  this scout fetched from; would need a second scrape of the live ASP site
  (blocked here) or GDL's aggregated metadata (also blocked here).
