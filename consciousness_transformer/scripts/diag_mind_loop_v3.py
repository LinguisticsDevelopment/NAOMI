"""Diagnostic: how well does a v3 checkpoint's choice distribution identify the reference node? Matched reachable slots on val (Hungarian as in training), step = last.
    python scripts/diag_mind_loop_v3.py --ckpt runs/mind_loop_v3/v3b_s0_last.pt --n 300 --tag v3b_s0"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src")); sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_loop as ML  # noqa: E402
from naomi import mind_loop_v3 as V3  # noqa: E402
import train_mind_loop as TL  # noqa: E402
import train_mind_loop_v3 as T3  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--tag", default="")
ap.add_argument("--split", default="val"); a = ap.parse_args()
torch.set_num_threads(2)
D = TL.Data((a.split,)); items = random.Random(0).sample(D.items[a.split], min(a.n, len(D.items[a.split])))
model, _ = TL.load_model(a.ckpt)
rec = []
with torch.no_grad():
    for i in range(0, len(items), 24):
        ch = items[i:i + 24]
        reqs = [{"mem": x["skey"], "q": x["q"], "t": x["refs"][0], "qid": x["item"]} for x in ch]
        outs, hl, tg, pool = T3.forward_batch(model, D, reqs)
        o = outs[-1]
        s = torch.softmax(V3.choice_logits(o, pool), -1)
        for b in range(len(ch)):
            T = int(tg["ex"][b].sum())
            if T == 0: continue
            cos = o["vec"][b] @ tg["vec"][b, :T].T
            idc = torch.nn.functional.normalize(o["idc"][b], dim=-1) @ torch.nn.functional.normalize(tg["idc"][b, :T], dim=-1).T
            cost = torch.where(tg["has"][b, :T][None] > 0.5, 1 - cos, tg["idm"][b, :T][None] * (1 - idc)).numpy()
            r, c = linear_sum_assignment(cost)
            for ri, cj in zip(r, c):
                t = int(pool["tidx"][b, cj])
                if t < 0: continue
                rec.append((float(s[b, ri, t]), float(s[b, ri].max()), int(s[b, ri].argmax()) == t, float(o["ex"][b, ri])))
A = np.array(rec)
res = {"n_matched_reachable": len(A), "mean_s_corr": float(A[:, 0].mean()), "top1_acc": float(A[:, 2].mean()), "frac_s_corr_gt_0.625": float((A[:, 0] > 0.625).mean()),
       "mean_pool_size_proxy": float(np.mean([int(p) for p in [1]]))}
order = np.argsort(-A[:, 1])
res["acc_by_confidence_quintile(top->bottom)"] = [float(A[order[k * len(A) // 5:(k + 1) * len(A) // 5], 2].mean()) for k in range(5)]
res["mean_ex"] = float(A[:, 3].mean())
print(json.dumps(res, indent=1)); json.dump(res, open(CT / "runs" / "mind_loop_v3" / f"diag_{a.tag}.json", "w"), indent=1)
