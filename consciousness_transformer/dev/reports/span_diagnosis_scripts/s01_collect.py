"""Run the trained thinker on a split and store, per item: decision, anchor scores, oracle per-node full-subtree ROUGE,
question wh info. Output out/collect_<split>_<model>.pkl.   python s01_collect.py val s2b_all"""
import sys, time, pickle
import numpy as np, torch
from common import *

split, name = sys.argv[1], sys.argv[2]
lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
torch.set_num_threads(4)
stories, pairs, meta = load_split(split)
vj = json.load(open(RUN / "vocab.json"))
tab = m2.SenseTable()
model = T.load_model(str(CKPT / f"thinker_{name}_best.pt"))[0]
normal = [p for p in pairs if p["kind"] == "normal"]
if lim:
    normal = normal[:lim]
attr = attributes()
ctx = m1.Ctx.get()
recs = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(normal), 48):
        ch = normal[i:i + 48]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e, anch, res = m2.decide(model, b)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        for j, (p, a, k, r) in enumerate(zip(ch, anch, kept, res)):
            o0, o1 = int(b["s_off"][j]), int(b["s_off"][j + 1])
            alog = e["alog"][o0:o1].numpy().copy()
            arr = stories[p["skey"]]
            m = meta[p["item"]]
            golds = m["golds"]
            N = len(arr["parent"])
            # oracle: every node as anchor, full subtree, true order, ROUGE max over golds
            nr = np.zeros(N, dtype=np.float32)
            for g in range(N):
                kk = {g} | set(arr["desc"][g].tolist())
                nr[g] = m2.rouge_multi(m2.true_order_text(arr, kk), golds)
            f = p_ = r_ = 0.0
            txt = ""
            if a >= 0:
                txt = m2.true_order_text(arr, k)
                f, p_, r_ = rouge_prf(txt, golds)
            # wh info
            qs = p["q"]
            wh = p["wh"]
            rec = {"id": p["item"], "skey": p["skey"], "N": N, "anchor": a, "kept": sorted(k) if k else None, "p_abstain": r["p_abstain"],
                   "p_anchor": r["p_anchor"], "alog": alog, "text": txt, "f": f, "prec": p_, "rec": r_, "node_rouge": nr,
                   "best_f1": p["tgt"]["best_f1"], "ans_sent": p["tgt"]["ans_sent"], "ans_len": p["tgt"]["ans_len"],
                   "n_sent": arr["n_sent"], "attribute": attr.get(p["item"]), "local": m["local"], "explicit": m["explicit"],
                   "question": m["question"], "q_parent": qs["parent"].tolist(), "q_rel": qs["rel"].tolist(), "q_pos": qs["pos"].tolist(),
                   "q_sid": qs["sid"], "wh": wh}
            recs.append(rec)
        print(i, time.time() - t0, flush=True)
pickle.dump(recs, open(OUTD / f"collect_{split}_{name}{'_lim' if lim else ''}.pkl", "wb"))
print("done", len(recs), time.time() - t0)
