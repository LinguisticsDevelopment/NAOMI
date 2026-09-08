# FairytaleQA -- full-fetch manifest + sample

MANIFEST.json indexes all 278 stories from the full `scripts/fetch_k12.py --source fairytaleqa --allow-download` fetch (277 fetched successfully -- see dev/FAIRYTALEQA_STATS.md for why any are missing) with per-story split/section/question counts and sha256 of both CSVs. The raw fetch itself lives in `data/k12/fairytaleqa/` (gitignored -- re-fetch with the command above). The CSVs alongside this file are a deterministic 10-story slice (first fetched stories in story_meta.csv order), copied verbatim, distinct from the pre-existing hand-picked `data/k12_samples/fairytaleqa/`.

- **License**: FairytaleQA (uci-soe/FairytaleQAData) -- Apache License 2.0 (https://github.com/uci-soe/FairytaleQAData/blob/main/LICENSE). Stories are 278 Project Gutenberg fairy tales (public domain); the QA annotations are the Apache-2.0-licensed original contribution of that repo.
