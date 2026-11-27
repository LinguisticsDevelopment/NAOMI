"""Evaluate looped thinker v2 on one split (val first: picks thr/tau; test ONCE). Reuses eval_mind_loop.main (same tables as v1) with outputs
redirected to runs/mind_loop_v2/, then adds: counterfactual accuracy, per-step loss, halting distribution, 'story adds'.
    python scripts/eval_mind_loop_v2.py --split val --ckpt runs/mind_loop_v2/v2_s0_best.pt --tag _s0
    python scripts/eval_mind_loop_v2.py --split test --ckpt ... --tag _s0        (reads val_choice.json written by the val run: copy per tag)"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_loop as ML  # noqa: E402
from naomi import mind_loop_v2 as V2  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
import eval_mind_loop as EL  # noqa: E402
import train_mind_loop as TL  # noqa: E402
import train_mind_loop_v2 as T2  # noqa: E402

OUT = T2.OUT


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


@torch.no_grad()
def extras(split, ckpt, tag):
    D = TL.Data((split,))
    items = D.items[split]
    stories = pickle.load(open(TL.V2B / "stories.pkl", "rb"))
    dc = TL.Decode(D, stories)
    model, _ = TL.load_model(ckpt)
    ch = json.load(open(OUT / f"val_choice{tag}.json"))["chosen"]
    thr, tau = ch["thr"], ch["tau"]
    cf = V2.CF(D.sf, stories, {x["skey"] for x in items})
    cfi = T2.cf_items(cf, items, stories, D)
    # counterfactual accuracy: memory swapped, question fixed
    res = {"n_cf": len(cfi), "n_items": len(items)}
    cr, cn, co = T2.cf_val(model, D, dc, cfi, thr, tau)
    res.update(cf_rouge_vs_swapped_gold=cr, cf_new_token_rate=cn, cf_old_token_rate=co)
    # same items, ORIGINAL memory: rouge vs original gold, old/new-token rates (what the answer would be without the swap)
    sc, so, new0, old0, stale, changed = [], [], [], [], [], []
    for i in range(0, len(cfi), 24):
        b = cfi[i:i + 24]
        for mk, gold_key in ((None, "golds"),):
            pass
        reqs = [{"mem": x["skey"], "q": x["q"], "t": None, "qid": x["item"]} for x in b]
        outs, hl, _ = TL.forward_batch(model, D, reqs)
        t0, _ = dc.texts(model, outs, ML.halting(hl), b, [x["skey"] for x in b], thr, tau, [TL.zlib_salt(x["skey"]) for x in b])
        reqs = [{"mem": x["cf_key"], "q": x["q"], "t": None, "qid": x["item"]} for x in b]
        outs, hl, _ = TL.forward_batch(model, D, reqs)
        t1, _ = dc.texts(model, outs, ML.halting(hl), b, [x["cf_key"] for x in b], thr, tau, [TL.zlib_salt(x["cf_key"]) for x in b])
        for x, a, c in zip(b, t0, t1):
            wa = {w.strip(".,;:!?\"'()").lower() for w in a.split()}
            sc.append(m2.rouge_multi(a, x["golds"]) if a else 0.0)
            so.append(m2.rouge_multi(c, x["golds"]) if c else 0.0)            # cf-memory output scored against the ORIGINAL gold (stale answer)
            new0.append(float(any(d.lower() in wa for _o, d in x["swaps"])))
            old0.append(float(any(o.lower() in wa for o, _d in x["swaps"])))
            changed.append(float(a != c))
    res.update(orig_memory_rouge_vs_orig_gold=float(np.mean(sc)), cf_memory_rouge_vs_orig_gold=float(np.mean(so)), orig_memory_old_token_rate=float(np.mean(old0)),
               orig_memory_new_token_rate=float(np.mean(new0)), output_changed_by_swap=float(np.mean(changed)))
    # per-step loss and halting distribution on the whole split
    pl, pp, ms = T2.val_halt(model, D, items)
    res.update(per_step_loss=[round(x, 4) for x in pl], mean_halt_prob=[round(x, 4) for x in pp])
    # halting-step distribution at the chosen tau
    steps = []
    for i in range(0, len(items), 24):
        b = items[i:i + 24]
        outs, hl, _ = TL.forward_batch(model, D, [{"mem": x["skey"], "q": x["q"], "t": None, "qid": x["item"]} for x in b])
        steps += (ML.halt_step(ML.halting(hl), tau) + 1).tolist()
    res["halt_steps_dist"] = {str(k): steps.count(k) for k in sorted(set(steps))}
    # story adds: normal vs question-only / wrong-story memory (from the controls), recomputed on rows
    ev = json.load(open(OUT / f"eval_{split}{tag}.json"))
    res["story_adds_vs_qonly"] = ev["rouge_loop"] - ev["controls"]["qonly"]["rouge_vs_own_reference"]
    res["story_adds_vs_wrong"] = ev["rouge_loop"] - ev["controls"]["wrong"]["rouge_vs_own_reference"]
    res["story_adds_vs_shuffled"] = ev["rouge_loop"] - ev["controls"]["shuffled"]["rouge_vs_own_reference"]
    json.dump(res, open(OUT / f"extras_{split}{tag}.json", "w"), indent=1)
    L = [f"### v2 extras ({split}, {tag or 'run'}): thr={thr} tau={tau}", "", "```", json.dumps(res, indent=1), "```"]
    open(OUT / f"tables_{split}{tag}.md", "a").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    tag = arg("--tag", "")
    EL.OUT = OUT
    rb = OUT / "read_cache" / "readback.jsonl"
    if not rb.exists():
        rb.parent.mkdir(exist_ok=True)
        shutil.copy(TL.OUT / "read_cache" / "readback.jsonl", rb)         # copy of the v1 read-back cache (cheat texts identical); v1 file untouched
    # per-tag choice file so test reads the val choice of the same model
    cp = OUT / f"val_choice{tag}.json"
    import eval_mind_loop as _e
    orig = _e.main
    # eval_mind_loop.main uses OUT/'val_choice.json'; point it at the tag-specific file via a symlink-free swap
    plain = OUT / "val_choice.json"
    if arg("--split") == "test" and cp.exists():
        shutil.copy(cp, plain)
    EL.main()
    if arg("--split") == "val":
        shutil.copy(plain, cp)
    extras(arg("--split"), arg("--ckpt"), tag)
