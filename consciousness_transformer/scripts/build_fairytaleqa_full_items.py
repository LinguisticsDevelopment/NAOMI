"""Turn data/fairytaleqa_full (HF GEM/FairytaleQA, official story split) into episode-format items for Mind v2b.
    python -I scripts/build_fairytaleqa_full_items.py
Items = the HF rows (10,580, split train/valid/test by story).  Text choices (all documented in dev/reports/MIND_V2B.md):
  * context: iter_sentences(HF content); for the 3,000 items already in runs/fairytaleqa_episodes_samepassage.jsonl the old
    context is reused verbatim (so Mind v2's read cache applies).
  * question / gold answers: the original-case text and ALL annotator answers from the local copy of the original release
    (data/k12/fairytaleqa) when the (story, question) matches (10,556 of 10,580); else the HF (lowercased) strings.
Writes runs/mind_v2b/items_full.jsonl (same schema as the v1/v2 episodes file)."""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
from nsm_ct.corpus import iter_sentences  # noqa: E402

D = CT / "data" / "fairytaleqa_full"
LOC = CT / "data" / "k12" / "fairytaleqa"
OLD = CT / "runs" / "fairytaleqa_episodes_samepassage.jsonl"
OUT = CT / "runs" / "mind_v2b" / "items_full.jsonl"


def norm(q):
    return re.sub(r"\W", "", q.lower())


def main():
    old = {}
    for line in open(OLD, encoding="utf-8"):
        r = json.loads(line)
        old[(r["meta"]["story_id"], norm(r["question"]))] = r
    loc = {}
    for p in glob.glob(str(LOC / "*-questions.csv")):
        s = os.path.basename(p)[: -len("-questions.csv")]
        for r in csv.DictReader(open(p, encoding="utf-8")):
            loc.setdefault((s, norm(r["question"])), r)
    n = Counter()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for fn, split in (("train", "train"), ("valid", "val"), ("test", "test")):
            for line in open(D / f"{fn}.json", encoding="utf-8"):
                r = json.loads(line)
                k = (r["story_name"], norm(r["question"]))
                o, l = old.get(k), loc.get(k)
                question, answer, golds = r["question"], r["answer"], [r["answer"]]
                if l is not None:
                    golds = list(dict.fromkeys(l[c].strip() for c in ("answer1", "answer2", "answer3", "answer4", "answer5", "answer6") if l[c].strip()))
                    question, answer = l["question"], golds[0]
                    n["orig_text"] += 1
                ctx = o["context"] if o is not None else iter_sentences(r["content"])
                n["old_item"] += o is not None
                if not ctx or not golds:
                    n["dropped"] += 1
                    continue
                f.write(json.dumps({"context": ctx, "question": question, "answer_text": answer,
                                    "meta": {"story_id": r["story_name"], "split": split, "local_or_sum": r["local_or_sum"], "ex_or_im": r["ex_or_im"],
                                             "attribute": r["attribute"], "all_gold_answers": golds, "in_old_3000": o is not None}}, ensure_ascii=False) + "\n")
                n[split] += 1
    print(dict(n))


if __name__ == "__main__":
    main()
