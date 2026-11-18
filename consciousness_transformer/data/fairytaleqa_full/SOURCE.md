# FairytaleQA (full)

- URL: https://huggingface.co/datasets/GEM/FairytaleQA
- Revision: b6c76a77359f133f9ee087b65c52a686fada7c15
- Fetched: 2026-11-18 by scripts/fetch_fairytaleqa_full.py
- Licence (as stated in the dataset card README.md of this revision): metadata `license: unknown`; card field
  'Copyright Restrictions on the Dataset' = `research use only`; 'Copyright Restrictions on the Language Data' =
  `public domain` (the stories). No redistribution licence is stated, so the raw files are NOT to be committed: regenerate
  them with scripts/fetch_fairytaleqa_full.py. Original release: https://github.com/uci-soe/FairytaleQAData (Xu et al. 2022).
- Format: JSON lines, one row per question (story_name, content = the story section(s) the question cites, answer = ONE
  annotator answer, question, local_or_sum, attribute, ex_or_im); official split by story: train 232 / valid 23 / test 23
  stories, 8,548 / 1,025 / 1,007 rows. Read only with `python -I` (untrusted data).
