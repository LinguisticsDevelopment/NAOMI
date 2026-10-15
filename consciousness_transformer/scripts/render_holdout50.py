"""Render the first 50 holdout sentences' gold trees in a compact, readable
form for the hand audit (dev/GOLD_COVERAGE_AUDIT.md section 2). Scratch tool,
not part of the committed audit script -- stdlib only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render(rec, idx):
    tokens = rec["tokens"]
    pos = rec["pos"]
    trees = rec["lattice"]["trees"]
    tree = trees[0]
    out = [f"### [{idx}] {rec['text']}"]
    verb_idxs = [i for i, p in enumerate(pos) if p == "VERB"]
    noun_idxs = [i for i, p in enumerate(pos) if p in ("NOUN", "PROPN")]
    out.append(f"    tokens: {' '.join(tokens)}")
    out.append(f"    VERB tokens: {[(i, tokens[i]) for i in verb_idxs]}")
    for ci, clause in enumerate(tree["clauses"]):
        pred = clause.get("predicate")
        pg = clause["predicate_grounding"]
        out.append(f"    clause[{ci}] PRED={pred!r} type={pg['type']}")
        for role in clause["roles"]:
            g = role["grounding"]
            extra = ""
            if g["type"] == "sense":
                extra = f" top={g.get('candidates', ['?'])[0] if g.get('candidates') else '(see tsc)'}"
            out.append(f"        {role['relation']:<16} word={role['word']!r:<18} idx={role['token_index']} type={g['type']}{extra}")
    return "\n".join(out)


def main():
    holdout = [l.rstrip("\n") for l in open(ROOT / "runs" / "holdout_sentences.txt", encoding="utf-8")]
    holdout = [h for h in holdout if h.strip()]
    first50 = holdout[:50]

    gold = {}
    with open(ROOT / "runs" / "encoder_gold_v4b_16k.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold[rec["text"].strip()] = rec

    for i, text in enumerate(first50, 1):
        rec = gold[text]
        print(render(rec, i))
        print()


if __name__ == "__main__":
    main()
