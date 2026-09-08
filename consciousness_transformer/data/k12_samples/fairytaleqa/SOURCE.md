# FairytaleQA -- sample

15 story+question pairs (of 278 total), fetched 2026-09-08 for the K-12
corpus scout (dev/K12_CORPUS_SCOUT.md).

- **Source**: https://github.com/uci-soe/FairytaleQAData
- **License**: Apache License 2.0
  (https://github.com/uci-soe/FairytaleQAData/blob/main/LICENSE). The
  underlying stories are 278 children's tales drawn from Project Gutenberg
  (public domain); the QA annotations are the Apache-2.0-licensed original
  contribution of this repository. Commercial and research reuse both
  permitted, attribution required (see the LICENSE file's NOTICE terms).
- **Paper**: Xu et al., "Fantastic Questions and Where to Find Them:
  FairytaleQA -- An Authentic Dataset for Narrative Comprehension," ACL 2022
  (https://arxiv.org/abs/2203.13947).
- **Grade level**: kindergarten - 8th grade (dataset's own stated target
  population; not marked per-story, only at the corpus level).
- **Format**: two CSVs per story.
  - `<name>-story.csv`: columns `section,text` -- one row per human-coded
    section (multiple paragraphs), the passage text.
  - `<name>-questions.csv`: columns `question_id, local-or-sum, cor_section,
    attribute1, attribute2, question, ex-or-im1, answer1..answer6,
    ex-or-im2` -- one row per QA pair. `local-or-sum` marks whether the
    question targets a single section ("local") or the whole story
    ("summary"); `ex-or-im{1,2}` marks each annotator's judgment of
    whether the answer is "explicit" (verbatim/paraphrase of the text) or
    "implicit" (requires inference) -- gold FREE-TEXT answers, not
    multiple-choice. No "not answerable from the passage" items are
    present; every question has a supported gold answer.
- **Fetch method**: `story_meta.csv` at the repo root lists every story's
  filename + origin folder + split; per story, GET
  `data-by-origin/section-stories/<origin>/<name>-story.csv` and
  `data-by-origin/questions/<origin>/<name>-questions.csv` (raw.githubusercontent.com).
