"""Shared helpers for the span diagnosis (read-only analysis; imports the live code, changes nothing)."""
import json, pickle, sys
from pathlib import Path
import numpy as np

CT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
import train_mind_v2 as T  # noqa: E402

RUN = CT / "runs" / "mind_v2b"
OUTD = Path(__file__).resolve().parent / "out"
OUTD.mkdir(exist_ok=True)


def load_split(split):
    stories = pickle.load(open(RUN / "stories.pkl", "rb"))
    pairs = pickle.load(open(RUN / f"pairs_{split}.pkl", "rb"))
    meta = json.load(open(RUN / "items_meta.json"))
    return stories, pairs, meta


def attributes():
    """item id -> FairytaleQA attribute label, same order/skips as m2.load_items."""
    out = {}
    n = 0
    for line in open(RUN / "items_full.jsonl", encoding="utf-8"):
        r = json.loads(line)
        m = r["meta"]
        golds = [g for g in (m.get("all_gold_answers") or [r.get("answer_text")]) if g]
        if not r.get("answer_text") or not golds:
            continue
        out[f"ft{n}"] = m.get("attribute")
        n += 1
    return out


def rouge_prf(pred, refs):
    """Best-gold ROUGE-L (f, p, r)."""
    best = (0.0, 0.0, 0.0)
    a = m2.toks(pred)
    for ref in refs:
        b = m2.toks(ref)
        if not a or not b:
            continue
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        l = prev[-1]
        if l == 0:
            continue
        p, r = l / len(a), l / len(b)
        f = 2 * p * r / (p + r)
        if f > best[0]:
            best = (f, p, r)
    return best


# ----------------------------------------------------------------------------- decomposition
SLACK = 0.05          # an anchor / answer within this ROUGE of the oracle counts as right (same slack idea as ANCHOR_SLACK)
CATS = ["abstained", "wrong_sentence", "right_sentence_wrong_anchor", "right_anchor_too_much", "right_anchor_too_little", "right"]


def oracle_of(rec):
    nr = rec["node_rouge"]
    best = float(nr.max()) if len(nr) else 0.0
    acc = np.where(nr >= best - SLACK)[0] if len(nr) else np.zeros(0, dtype=int)
    return best, acc


def categorize(rec, arr):
    best, acc = oracle_of(rec)
    a = rec["anchor"]
    f = rec["f"] if a >= 0 else 0.0
    loss = best - f
    if a < 0:
        return "abstained", loss, best, f, acc
    if f >= best - SLACK:
        return "right", loss, best, f, acc
    if a in set(acc.tolist()):
        return ("right_anchor_too_much" if rec["prec"] < rec["rec"] else "right_anchor_too_little"), loss, best, f, acc
    sents = {int(arr["sent"][g]) for g in acc}
    if int(arr["sent"][a]) in sents:
        return "right_sentence_wrong_anchor", loss, best, f, acc
    return "wrong_sentence", loss, best, f, acc


def mean(x):
    x = list(x)
    return float(np.mean(x)) if x else float("nan")
