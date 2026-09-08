# MCTest -- sample

mc160 (dev + train, 100 stories) and mc500 dev (50 stories), fetched
2026-09-08 for the K-12 corpus scout (dev/K12_CORPUS_SCOUT.md).

- **Source**: https://github.com/mcobzarenco/mctest (a code mirror that
  bundles the original Microsoft Research MCTest release under
  `data/MCTest/`); original publication page:
  https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/
- **License**: Microsoft Research's own data-release terms (research use).
  **NOT independently re-verified in this environment** -- no LICENSE file
  is co-located with this specific mirror's `data/MCTest/` directory, and
  Microsoft's canonical MCTest license text was not directly fetchable
  (microsoft.com blocked by this sandbox's egress policy). Treat as
  research-use-only until confirmed against Microsoft's own page. See the
  licensing caveats paragraph in dev/K12_CORPUS_SCOUT.md.
- **Paper**: Richardson, Burges & Renshaw, "MCTest: A Challenge Dataset for
  the Open-Domain Machine Comprehension of Text," EMNLP 2013.
- **Grade level**: mc160 stories are explicitly written at a ~7-year-old
  reading level (simple declarative fiction); mc500 is a larger, slightly
  harder companion set at a similar target age. No finer per-story grade
  marking.
- **Format**: `.tsv` (one story per line: id, author/worktime metadata,
  story text with literal `\newline` markers, then 4 questions each with 4
  answer-option columns) + a matching `.ans` file (one line per story, 4
  gold answer letters A-D, one per question). Every story has EXACTLY 4
  multiple-choice questions with a single gold letter each; no
  "not stated" option.
- **Fetch method**: direct GET of each `.tsv`/`.ans` file from
  `raw.githubusercontent.com/mcobzarenco/mctest/master/data/MCTest/`.
