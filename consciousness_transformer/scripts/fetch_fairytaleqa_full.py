"""Fetch the full FairytaleQA (GEM/FairytaleQA on Hugging Face) into data/fairytaleqa_full/ and record SOURCE.md.
    python scripts/fetch_fairytaleqa_full.py
The raw files are untrusted data: read them only with ``python -I``."""
import datetime
import json
import sys
import urllib.request
from pathlib import Path

REPO = "GEM/FairytaleQA"
FILES = ["train.json", "valid.json", "test.json", "README.md", "FairytaleQA.py", "dataset_infos.json"]
OUT = Path(__file__).resolve().parents[1] / "data" / "fairytaleqa_full"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}"))
    for f in FILES:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{f}"
        (OUT / f).write_bytes(urllib.request.urlopen(url).read())
        print("fetched", f, (OUT / f).stat().st_size, file=sys.stderr)
    (OUT / "SOURCE.md").write_text(
        f"# FairytaleQA (full)\n\n- URL: https://huggingface.co/datasets/{REPO}\n- Revision: {meta['sha']}\n"
        f"- Fetched: {datetime.date.today().isoformat()} by scripts/fetch_fairytaleqa_full.py\n"
        "- Licence (as stated in the dataset card README.md of this revision): metadata `license: unknown`; card field\n"
        "  'Copyright Restrictions on the Dataset' = `research use only`; 'Copyright Restrictions on the Language Data' =\n"
        "  `public domain` (the stories). No redistribution licence is stated, so the raw files are NOT to be committed: regenerate\n"
        "  them with scripts/fetch_fairytaleqa_full.py. Original release: https://github.com/uci-soe/FairytaleQAData (Xu et al. 2022).\n"
        "- Format: JSON lines, one row per question (story_name, content = the story section(s) the question cites, answer = ONE\n"
        "  annotator answer, question, local_or_sum, attribute, ex_or_im); official split by story: train 232 / valid 23 / test 23\n"
        "  stories, 8,548 / 1,025 / 1,007 rows. Read only with `python -I` (untrusted data).\n")


if __name__ == "__main__":
    main()
