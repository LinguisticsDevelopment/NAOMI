# African Storybook Project (English) -- sample

30 CC-BY English stories (of 105 indexed as having translations; ASP's full
catalog is much larger -- ~105 is only the subset covered by the
`global-asp/global-asp` cross-language index used to find slugs), fetched
2026-09-08 for the K-12 corpus scout (dev/K12_CORPUS_SCOUT.md).

- **Source**: story text from https://github.com/global-asp/asp-source
  (a Markdown mirror of story text extracted from the original African
  Storybook Project PDFs, https://africanstorybook.org); per-story license
  looked up from https://github.com/global-asp/global-asp `INDEX.md`.
- **License**: CC-BY (per-story; this fetch ONLY keeps stories whose
  `INDEX.md` license column says CC-BY -- some ASP stories carry
  CC-BY-SA or CC-BY-NC instead, and those are skipped). Each story's own
  metadata footer (kept verbatim in the `.md` file) names the license,
  writer, illustrator and translator for that specific story.
- **Grade level**: African Storybook levels its readers (roughly level 1-4,
  first-words through fluent-reader complexity); the level is NOT captured
  in these Markdown files' text (it lives on the ASP website's per-story
  page, linked from `INDEX.md`'s "Original ASP Title" column) -- a gap to
  fix before large-scale ingestion (scrape the ASP web page's level badge,
  or accept level-unmarked text for the bottom curriculum rung where any
  short/simple text works).
- **Format**: Markdown. Line 1 = title (`# Title`); page breaks marked
  `##`; a trailing metadata block (`* License: ...`, `* Text: ...`, etc.)
  that this repo's own fetch/probe code strips before treating the rest as
  passage prose.
- **Comprehension questions: NONE.** This is a real, load-bearing gap --
  ASP ships story text only. It is included in the TOP-3 anyway because it
  fits directly into NAOMI's EXISTING auto-question-synthesis path
  (`nsm_ct.corpus`'s "queried role" machinery, the same one
  `scripts/fetch_corpus.py`'s Gutenberg prose already goes through) rather
  than requiring new engineering, and its prose is dramatically simpler
  than Gutenberg's (see the parse-yield numbers in
  dev/K12_CORPUS_SCOUT.md).
- **Fetch method**: GET `global-asp/global-asp`'s `INDEX.md`, parse the
  `Story # | Original ASP Title | License | ...` table, slugify each
  CC-BY-marked title, GET
  `raw.githubusercontent.com/global-asp/asp-source/master/en/<num>_<slug>.md`.
