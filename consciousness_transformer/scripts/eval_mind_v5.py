"""Evaluate Mind v5 (question completion) on one split. Run on test ONCE, after every choice is made on val.
    python scripts/eval_mind_v5.py --split val --models v5_all=runs/mind_v5/thinker_v5_all_best.pt v5_short=... --main v5_all --short v5_short \
        --v3 v3_s2=runs/mind_v3/thinker_v3_s2_best.pt --rt_thr runs/mind_v2/roundtrip_val.json
Writes runs/mind_v5/eval_<split>.json, eval_rows_<split>.jsonl (large: not for the commit), tables_<split>.md and handread_<split>.json.
This script (and only this script) reads the lexicon's unknown axis, for the DIAGNOSTIC of which question node was replaced."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
from naomi import mind_v3 as m3  # noqa: E402
from naomi import mind_v5 as m5  # noqa: E402
from naomi.roundtrip import source_text  # noqa: E402
from naomi.tree import tree_from_dict, tree_to_dict, walk  # noqa: E402
import build_mind_data as B  # noqa: E402
import eval_mind_v2 as E2  # noqa: E402
import train_mind_v3 as T3  # noqa: E402
import train_mind_v5 as T5  # noqa: E402

OUT = CT / "runs" / "mind_v5"
BANDS = ["<=10", "11-20", "21+", "none"]
mean = E2.mean


def sent_tree(arr, k):
    """The whole of story sentence k as an answer tree (the cheat's filler)."""
    rows = [g for g in range(len(arr["parent"])) if int(arr["sent"][g]) == k and arr["parent"][g] < 0]
    r = max(rows, key=lambda g: len(arr["desc"][g]))
    return m2.answer_tree(arr, r, {r} | set(arr["desc"][r].tolist()))


def tree_toks(t):
    return [n for n, _d, _p in walk(t.root) if n.token_index is not None]


@torch.no_grad()
def run_v5(model, stories, pairs, tab, bs=48):
    out = {}
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        for p, r in zip(ch, model.decode(b)):
            out[p["id"]] = r
    return out


@torch.no_grad()
def run_v3(model, stories, pairs, tab, bs=48):
    out = {}
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e, anch, _ = m3.decide(model, b)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        for p, a, k in zip(ch, anch, kept):
            out[p["id"]] = (a, k)
    return out


def boot(rows, f, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.array([f(r) - g(r) for r in rows])
    st = defaultdict(list)
    for r, x in zip(rows, d):
        st[r["story"]].append(x)
    sv = [np.array(v) for v in st.values()]
    bi = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(sv), len(sv))
        bs.append(np.concatenate([sv[i] for i in pick]).mean())
    return float(d.mean()), np.percentile(bi, [2.5, 97.5]), np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--models", nargs="+", required=True, help="name=path of v5 models")
    ap.add_argument("--main", required=True)
    ap.add_argument("--short", default=None, help="name of the short-trained v5 model")
    ap.add_argument("--v3", nargs="*", default=[], help="name=path of v3 thinkers whose output tree is used as a filler (old method)")
    ap.add_argument("--rt_thr", default=None)
    ap.add_argument("--no_readback", action="store_true")
    ap.add_argument("--rt_n", type=int, default=0, help="round-trip subsample size (0 = all answered items)")
    ap.add_argument("--orderer", default="runs/naomi_orderer/large_best.pt")
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: random sample of this many items")
    args = ap.parse_args()
    torch.set_num_threads(4)
    rng = random.Random(0)
    DATA = CT / "runs" / "mind_v2b"
    E2.OUT = OUT
    E2.RB = DATA / "read_cache" / "readback.jsonl"
    stories = pickle.load(open(DATA / "stories.pkl", "rb"))
    data = pickle.load(open(OUT / f"data_{args.split}.pkl", "rb"))
    old_pairs = {p["id"]: p for p in pickle.load(open(DATA / f"pairs_{args.split}.pkl", "rb"))}
    meta = json.load(open(DATA / "items_meta.json"))
    reads = T5.reads_all()
    vj = json.load(open(DATA / "vocab.json"))
    vocab = m2.Vocab(vj["rels"], vj["poss"])
    ctx = m1.Ctx.get()
    tab = m2.SenseTable()
    dec = m2.Decoder(str(CT / args.orderer))
    rt_thr = json.load(open(args.rt_thr))["thresholds"]["A"] if args.rt_thr else 99
    normal = [p for p in data if p["kind"] == "normal"]
    wrong = [p for p in data if p["kind"] == "wrong"]
    if args.limit:
        keep = {p["item"] for p in random.Random(0).sample(normal, args.limit)}
        normal = [p for p in normal if p["item"] in keep]
        wrong = [p for p in wrong if p["item"] in keep]
    models = {}
    for spec in args.models:
        nm, path = spec.split("=")
        models[nm] = T5.load_model(path)[0]
    v3models = {}
    for spec in args.v3:
        nm, path = spec.split("=")
        v3models[nm] = T3.load_model(path)[0]
    M = args.main

    # ---------------------------------------------------------------- static per-item info
    rows, Q = {}, {}
    for p in normal:
        m = meta[p["item"]]
        t0 = p["tgt0"]
        arr = stories[p["skey"]]
        qt = m1.tree_of(reads, m["question"])
        qn = m2.flat_tree(qt)[0]
        qpar = p["q"]["parent"]
        supp = m5.support_vector(arr, t0["ans_sent"], qn)
        Q[p["item"]] = (qt, qn, qpar, supp)
        rows[p["item"]] = {"id": p["item"], "question": m["question"], "reference": m["answer"], "golds": m["golds"], "story": m["story"],
                           "band": m2.band_of(t0["ans_len"]) if t0["ans_len"] is not None else "none", "reachable": t0["best_f1"] >= m2.ACCEPT_F1,
                           "needs_new": t0["ref_missing"] >= 1.0 / 3.0, "n_q": len(qn)}
    ref_trees = {i: [t for t in (m1.tree_of(reads, g) for g in r["golds"]) if t is not None] for i, r in rows.items()}
    by_id = {p["item"]: p for p in normal}

    # ---------------------------------------------------------------- fillers of every system
    fill = {}            # system -> item -> Tree or None (abstain)
    prov = {}            # main-style models: item -> provenance list
    jch = {}             # model -> item -> chosen question node (None if abstained)
    for p in normal:
        i = p["item"]
        arr = stories[p["skey"]]
        r = rows[i]
        sents = arr["sent_text"]
        k = E2.idf_pick(r["question"], sents)
        r["cheat_text"] = sents[k] if k >= 0 else ""
        fill.setdefault("cheat", {})[i] = sent_tree(arr, k) if k >= 0 else None
        t0 = p["tgt0"]
        fill.setdefault("oracle", {})[i] = m2.answer_tree(arr, t0["anchors"][0][0], set(t0["anchors"][0][1])) if t0["anchors"] else None
        r["oracle_subtree_text"] = E2.best_full_subtree_text(arr, r["golds"]) if len(arr["parent"]) else ""
    dm = {}
    for nm, model in models.items():
        res = run_v5(model, stories, normal, tab)
        dm[nm] = res
        for p in normal:
            i = p["item"]
            rr = res[p["id"]]
            arr = stories[p["skey"]]
            qt, qn, qpar, supp = Q[i]
            if rr["abstain"] or not rr["pos"]:
                fill.setdefault(nm, {})[i] = None
                jch.setdefault(nm, {})[i] = None
                continue
            srcs = [m5.source_of(x, rr["Nmax"]) for x in rr["pos"]]
            tr, pv = m5.filler_tree(arr, qn, srcs, rr["par"])
            fill.setdefault(nm, {})[i] = tr
            jch.setdefault(nm, {})[i] = rr["j"]
            if nm == M or nm == args.short:
                prov.setdefault(nm, {})[i] = pv
    for nm, model in v3models.items():
        res = run_v3(model, stories, [old_pairs[p["id"]] for p in normal], tab)
        for p in normal:
            a, k = res[p["id"]]
            arr = stories[p["skey"]]
            fill.setdefault(nm, {})[p["item"]] = m2.answer_tree(arr, a, k) if a >= 0 else None
    systems = list(fill)
    print("systems:", systems, flush=True)

    # ---------------------------------------------------------------- scoring: answer part, frame part, decoded filler, ROUGE
    score = ctx.score
    for s in systems:
        jsrc = s if s in jch else (args.main)
        for p in normal:
            i = p["item"]
            r = rows[i]
            tr = fill[s][i]
            r[f"{s}_ans"] = tr is not None
            if tr is None:
                continue
            qt, qn, qpar, supp = Q[i]
            j = jch[jsrc][i]
            if j is None:                      # a baseline filler where the main model abstained: plug into the model's would-be choice
                lg = dm[jsrc][p["id"]]["j_logits"]
                j = int(np.argmax(lg)) if lg else 0
            r[f"{s}_j"] = j
            fa, fs = m5.frame_parts(qpar, j, supp)
            r[f"{s}_fa"], r[f"{s}_fs"] = fa, (fs if fs is not None else fa)
            if ref_trees[i]:
                ps = max((score.score_pair(g, tr, ctx.usvs) for g in ref_trees[i]), key=lambda x: x.overall)
                r[f"{s}_A"] = ps.overall
                r[f"{s}_reasons"] = [x["kind"] for x in ps.reasons]
            r[f"{s}_true_text"] = source_text(tr)
    print("scored answer parts", flush=True)

    def decode_system(s, which):
        ids = [i for i in rows if fill[s].get(i) is not None]
        trees = []
        for i in ids:
            tr = fill[s][i]
            if which == "b":
                arr = stories[by_id[i]["skey"]]
                tr, _ = m2.meaning_only(tr, m2.Pool(arr["nodes"], Q[i][1]))
            trees.append(tr)
        for i, t in zip(ids, dec.texts(trees)):
            rows[i][f"{s}_text_{which}"] = t
    for s in systems:
        decode_system(s, "a")
        decode_system(s, "b")
        print("decoded", s, flush=True)

    # full output sentences for the v5 models
    for nm in models:
        ids, trees_a, trees_b = [], [], []
        for p in normal:
            i = p["item"]
            tr = fill[nm][i]
            if tr is None:
                continue
            qt, qn, qpar, supp = Q[i]
            out = m5.assemble(qt, qn, jch[nm][i], tr)
            ids.append(i)
            trees_a.append(out)
            arr = stories[p["skey"]]
            trees_b.append(m2.meaning_only(out, m2.Pool(arr["nodes"], qn))[0])
        for i, ta, tb in zip(ids, dec.texts(trees_a), dec.texts(trees_b)):
            rows[i][f"{nm}_full_a"], rows[i][f"{nm}_full_b"] = ta, tb
            rows[i][f"{nm}_out_tree"] = tree_to_dict(m5.assemble(Q[i][0], Q[i][1], jch[nm][i], fill[nm][i]))
        print("full sentences", nm, flush=True)

    # provenance (main): every output node is a copy of a memory or question node; verify token + lemma equality
    pv_stats = {}
    for nm in prov:
        n_all = n_mem = n_q = n_bad = 0
        for i, pv in prov[nm].items():
            arr = stories[by_id[i]["skey"]]
            qn = Q[i][1]
            toks = sorted(tree_toks(fill[nm][i]), key=lambda n: n.token_index)
            # filler tree nodes are renumbered in (memory first, question last) order: rebuild expected order
            keyed = sorted(pv, key=lambda s: ((0, int(arr["sent"][s[1]]), int(arr["tidx"][s[1]])) if s[0] == "m" else (1, 0, int(qn[s[1]].token_index))))
            for n, (kind, ix) in zip(toks, keyed):
                src = arr["nodes"][ix] if kind == "m" else qn[ix]
                n_all += 1
                n_mem += kind == "m"
                n_q += kind == "q"
                n_bad += (n.token, n.lemma) != (src.token, src.lemma)
        pv_stats[nm] = {"nodes": n_all, "from_memory": n_mem, "from_question": n_q, "traced_share": (n_mem + n_q - n_bad) / max(n_all, 1), "memory_share": n_mem / max(n_all, 1), "mismatch": n_bad}

    # ---------------------------------------------------------------- numbers per item
    for r in rows.values():
        for s in systems:
            if not r.get(f"{s}_ans"):
                for k in ("A", "fa", "fs", "comb", "comb_all", "rouge_a", "rouge_b", "rouge_t"):
                    r[f"{s}_{k}"] = 0.0
                continue
            r.setdefault(f"{s}_A", 0.0)
            a = r[f"{s}_A"]
            r[f"{s}_comb"] = math.sqrt(max(a, 0.0) * 100.0 * r[f"{s}_fs"])            # geometric mean of answer % and frame % (declared)
            r[f"{s}_comb_all"] = math.sqrt(max(a, 0.0) * 100.0 * r[f"{s}_fa"])
            r[f"{s}_rouge_a"] = m2.rouge_multi(r.get(f"{s}_text_a") or "", r["golds"])
            r[f"{s}_rouge_b"] = m2.rouge_multi(r.get(f"{s}_text_b") or "", r["golds"])
            r[f"{s}_rouge_t"] = m2.rouge_multi(r.get(f"{s}_true_text") or "", r["golds"])
        r["cheat_raw_rouge"] = m2.rouge_multi(r["cheat_text"], r["golds"])
        r["oracle_subtree_rouge"] = m2.rouge_multi(r["oracle_subtree_text"], r["golds"])
        for nm in models:
            if r.get(f"{nm}_full_a"):
                r[f"{nm}_full_rouge_a"] = m2.rouge_multi(r[f"{nm}_full_a"], r["golds"])
    R = list(rows.values())

    # ---------------------------------------------------------------- UNKNOWN diagnostic (reads the axis; diagnostic only)
    diag = {}
    for nm in models:
        n_has = agree = n_tie_any = n_root = n_leaf_unsup = n_ans = 0
        chance = []
        rk = Counter()
        for p in normal:
            i = p["item"]
            j = jch[nm][i]
            if j is None:
                continue
            n_ans += 1
            qt, qn, qpar, supp = Q[i]
            vals = []
            for n in qn:
                v = ctx.raw(ctx.chosen(n))
                vals.append(float(v[ctx.unknown_idx]) if (v is not None and ctx.unknown_idx >= 0) else 0.0)
            mx = max(vals)
            if qpar[j] < 0:
                n_root += 1
            if mx <= 0:
                continue
            n_has += 1
            top = [k for k, v in enumerate(vals) if v >= mx - 1e-9]
            agree += j in top
            n_tie_any += len(top) > 1
            chance.append(len(top) / len(qn))
            order = sorted(range(len(qn)), key=lambda k: -vals[k])
            rk[min(order.index(j), 5)] += 1
        diag[nm] = {"answered": n_ans, "questions_with_a_node_above_zero": n_has, "replaced_is_top_unknown": agree / max(n_has, 1),
                    "chance_if_uniform": float(np.mean(chance)) if chance else float("nan"), "root_replaced_share": n_root / max(n_ans, 1),
                    "rank_by_unknown_of_replaced": {str(k): v / max(n_has, 1) for k, v in sorted(rk.items())}, "n_has": n_has}
    # frame reference points: what a random / root / best choice of node gets
    ref_pts = {"random_node_fs": mean([mean([r_ for r_ in m5.frame_rewards(Q[i][2], Q[i][3])]) for i in rows if Q[i][3] is not None and Q[i][3].sum() > 0]),
               "best_node_fs": mean([float(m5.frame_rewards(Q[i][2], Q[i][3]).max()) for i in rows if Q[i][3] is not None and Q[i][3].sum() > 0]),
               "random_node_fa": mean([mean([m5.frame_parts(Q[i][2], j, None)[0] for j in range(len(Q[i][1]))]) for i in rows])}

    # ---------------------------------------------------------------- controls
    stats = {}
    ctxs = {it["id"]: it for it in m2.load_items(DATA / "items_full.jsonl")}
    story_obj = {}
    for p in normal:
        k = p["skey"]
        if k not in story_obj:
            story_obj[k] = m2.Story(ctxs[p["item"]]["context"], reads, vocab)
    for nm, model in models.items():
        dw = run_v5(model, stories, wrong, tab)
        shuf_pairs, shuf_gold, emp_pairs, sev_pairs, sev_st = [], {}, [], [], {}
        by_story = defaultdict(list)
        for p in normal:
            by_story[meta[p["item"]]["story"]].append(p["item"])
        r2 = random.Random(1)
        for j_, p in enumerate(normal):
            i = p["item"]
            m = meta[i]
            others = [o for o in by_story[m["story"]] if meta[o]["question"] != m["question"]]
            if others:
                o = r2.choice(others)
                qt2 = m1.tree_of(reads, meta[o]["question"])
                if qt2 is not None:
                    story = story_obj[p["skey"]]
                    pf = m2.pair_features(story, qt2, vocab)
                    if pf is not None:
                        pf = m5.refeature(pf, stories[p["skey"]], m2.flat_tree(qt2)[0], ctx)
                        pf.update(id=f"{i}|shuf", kind="shuf", item=i, skey=p["skey"])
                        shuf_pairs.append(pf)
            q = m["question"]
            qt = Q[i][0]
            for tag, ctxlist, store in (("sev", [q], sev_pairs), ("emp", [], emp_pairs)):
                s1 = m2.Story(ctxlist, reads, vocab)
                sk = (tag, j_)
                sev_st[sk] = s1.arrays()
                pf = m2.pair_features(s1, qt, vocab)
                if pf is not None:
                    pf = m5.refeature(pf, sev_st[sk], Q[i][1], ctx)
                    pf.update(id=f"{i}|{tag}", kind=tag, item=i, skey=sk)
                    store.append(pf)
        allst = dict(stories)
        allst.update(sev_st)
        ds = run_v5(model, allst, shuf_pairs, tab)
        dsv = run_v5(model, allst, sev_pairs, tab)
        dem = run_v5(model, allst, emp_pairs, tab)
        sh = []
        for p in shuf_pairs:
            rr = ds[p["id"]]
            txt = ""
            if not rr["abstain"] and rr["pos"]:
                arr = stories[p["skey"]]
                srcs = [m5.source_of(x, rr["Nmax"]) for x in rr["pos"]]
                srcs = [s_ for s_ in srcs if s_[0] == "m"]
                if srcs:
                    tr, _ = m5.filler_tree(arr, [], srcs, [-1] + [0] * (len(srcs) - 1))
                    txt = dec.texts([tr])[0]
            sh.append((not rr["abstain"] and bool(rr["pos"]), m2.rouge_multi(txt, meta[p["item"]]["golds"])))
        stats[nm] = {"wrong_abstain": mean([float(dw[p["id"]]["abstain"]) for p in wrong]), "n_wrong": len(wrong),
                     "shuf_answer_rate": mean([a for a, _ in sh]), "shuf_rouge": mean([s_ for _, s_ in sh]), "n_shuf": len(sh),
                     "severed_abstain": mean([float(v["abstain"]) for v in dsv.values()]), "empty_abstain": mean([float(v["abstain"]) for v in dem.values()]),
                     "n_severed": len(dsv), "n_empty": len(dem)}
        print("controls", nm, stats[nm], flush=True)

    # ---------------------------------------------------------------- round trip (main filler, as v2b)
    rtres = {}
    ans_ids = [i for i in rows if rows[i].get(f"{M}_ans")]
    if not args.no_readback and ans_ids:
        sel = ans_ids if not args.rt_n or args.rt_n >= len(ans_ids) else rng.sample(ans_ids, args.rt_n)
        texts = []
        for i in sel:
            texts += [rows[i][f"{M}_text_a"], rows[i][f"{M}_true_text"]]
        rb = E2._read_texts(texts)

        def rt(s):
            x = rb.get(s)
            return None if x is None or x["tree"] is None else tree_from_dict(x["tree"])
        vals, bylen = [], defaultdict(list)
        for i in sel:
            a, b_ = rt(rows[i][f"{M}_true_text"]), rt(rows[i][f"{M}_text_a"])
            if a is None or b_ is None:
                continue
            v = float(score.score_pair(a, b_, ctx.usvs).overall)
            rows[i]["rt2"] = v
            vals.append(v >= rt_thr)
            n = len(rows[i][f"{M}_true_text"].split())
            bylen["1-3 words" if n <= 3 else ("4-6 words" if n <= 6 else "7+ words")].append(v >= rt_thr)
        rtres = {"threshold": rt_thr, "n": len(vals), "pass_rate": mean(vals), "by_len": {k: {"n": len(v), "pass_rate": mean(v)} for k, v in bylen.items()}}

    # ---------------------------------------------------------------- tables
    names = {"cheat": "cheat: best-overlap sentence (as filler)", "oracle": "oracle: aligned reference nodes (as filler)"}
    names.update({k: f"{k}: old method (v3 anchor subtree as filler)" for k in v3models})
    names.update({k: f"{k}: v5 thinker" for k in models})
    sysorder = [s for s in ["cheat"] + list(v3models) + list(models) + ["oracle"] if s in systems]

    def tab_rows(sel, cols=("A", "fs", "fa", "comb", "comb_all", "rouge_a", "rouge_b", "rouge_t")):
        out = {}
        for s in sysorder:
            out[s] = {"n": len(sel), "answer_rate": mean([float(bool(r.get(f"{s}_ans"))) for r in sel]), **{c: mean([r[f"{s}_{c}"] for r in sel]) for c in cols}}
        return out
    summary = {"split": args.split, "n_items": len(R), "main": M, "short": args.short, "systems": sysorder, "band_counts": {b: sum(r["band"] == b for r in R) for b in BANDS},
               "overall": tab_rows(R), "by_band": {b: tab_rows([r for r in R if r["band"] == b]) for b in BANDS},
               "needs_new_words": tab_rows([r for r in R if r["needs_new"]]), "reachable_only": tab_rows([r for r in R if r["reachable"]]),
               "old_cheat_raw_text_rouge": mean([r["cheat_raw_rouge"] for r in R]), "oracle_subtree_rouge": mean([r["oracle_subtree_rouge"] for r in R]),
               "full_sentence_rouge_a": {nm: mean([r.get(f"{nm}_full_rouge_a", 0.0) for r in R]) for nm in models},
               "controls": stats, "provenance": pv_stats, "unknown_diagnostic": diag, "frame_reference_points": ref_pts, "round_trip_main": rtres,
               "reasons_main": dict(Counter(x for r in R for x in r.get(f"{M}_reasons", [])).most_common()),
               "decoder_note": "(a) order-only decoder on the filler tree; (b) meaning-only decoder; rouge_t = filler words in memory order, no decoder"}
    # paired differences
    pd = {}
    for base in [s for s in sysorder if s != M]:
        for k in ("A", "rouge_a", "comb"):
            d, ci, cs = boot(R, lambda r, k=k: r[f"{M}_{k}"], lambda r, k=k, base=base: r[f"{base}_{k}"])
            pd[f"{M} - {base} [{k}]"] = {"mean": d, "items_95": ci.tolist(), "stories_95": cs.tolist()}
    summary["paired"] = pd
    if args.short and args.short in models:
        summary["short_to_long"] = {b: {"all": tab_rows([r for r in R if r["band"] == b])[M], "short": tab_rows([r for r in R if r["band"] == b])[args.short]} for b in BANDS}
    tag = args.out_tag
    json.dump(summary, open(OUT / f"eval_{args.split}{tag}.json", "w"), indent=1, default=float)
    with open(OUT / f"eval_rows_{args.split}{tag}.jsonl", "w") as f:
        for r in R:
            f.write(json.dumps(r, default=float) + "\n")
    # hand-read dump (seeded sample of answered main items)
    hr = random.Random(7)
    pool = [r for r in R if r.get(f"{M}_ans")]
    pick = hr.sample(pool, min(20, len(pool)))
    json.dump([{"id": r["id"], "question": r["question"], "golds": r["golds"], "band": r["band"], "filler_true_order": r[f"{M}_true_text"], "filler_decoded_a": r.get(f"{M}_text_a"),
                "full_sentence_a": r.get(f"{M}_full_a"), "replaced_node": Q[r["id"]][1][r[f"{M}_j"]].token, "question_nodes": [n.token for n in Q[r["id"]][1]],
                "A": r.get(f"{M}_A"), "fs": r.get(f"{M}_fs"), "provenance": [list(x) for x in prov[M][r["id"]]]} for r in pick],
              open(OUT / f"handread_{args.split}{tag}.json", "w"), indent=1, default=float)
    write_tables(summary, R, OUT / f"tables_{args.split}{tag}.md", sysorder, names)
    print("done", flush=True)


def write_tables(S, R, path, sysorder, names):
    L = [f"## {S['split']}: n={S['n_items']} items; bands {S['band_counts']}; main = {S['main']}\n"]
    L.append("### Headline (abstain = 0). A = answer part (tree score % of the inserted nodes vs the best reference tree), Fr_sup / Fr_all = frame part, combined = sqrt(A * Fr_sup), ROUGE-L of the decoded filler (max over golds)\n")
    L.append("| system | answer rate | A % | Fr_sup | Fr_all | combined (sup) | combined (all) | ROUGE (a) | ROUGE (b) | ROUGE true order |\n|---|---|---|---|---|---|---|---|---|---|")
    for s in sysorder:
        v = S["overall"][s]
        L.append(f"| {names[s]} | {v['answer_rate']:.3f} | {v['A']:.1f} | {v['fs']:.3f} | {v['fa']:.3f} | {v['comb']:.1f} | {v['comb_all']:.1f} | {v['rouge_a']:.3f} | {v['rouge_b']:.3f} | {v['rouge_t']:.3f} |")
    L.append(f"\nOld raw-text cheat ROUGE-L (whole best-overlap sentence text): {S['old_cheat_raw_text_rouge']:.3f}; oracle best full subtree: {S['oracle_subtree_rouge']:.3f}; full output sentence ROUGE (a): {json.dumps({k: round(v, 3) for k, v in S['full_sentence_rouge_a'].items()})}\n")
    for metric, label in (("A", "A % (answer part)"), ("rouge_a", "ROUGE-L (a)"), ("comb", "combined")):
        L.append(f"### By answer-sentence length band: {label}\n\n| band | n | " + " | ".join(sysorder) + " |\n|---|---|" + "---|" * len(sysorder))
        for b in BANDS:
            v = S["by_band"][b]
            if v[sysorder[0]]["n"]:
                L.append(f"| {b} | {v[sysorder[0]]['n']} | " + " | ".join(f"{v[s][metric]:.3f}" if metric == 'rouge_a' else f"{v[s][metric]:.1f}" for s in sysorder) + " |")
        L.append("")
    if "short_to_long" in S:
        L.append(f"### Short-to-long ({S['short']} trained on answer sentences <= 10 words vs {S['main']} all lengths)\n\n| band | n | A all | A short | ROUGE(a) all | ROUGE(a) short | combined all | combined short | answer rate all | answer rate short |\n|---|---|---|---|---|---|---|---|---|---|")
        for b in BANDS:
            x = S["short_to_long"][b]
            if x["all"]["n"]:
                L.append(f"| {b} | {x['all']['n']} | {x['all']['A']:.1f} | {x['short']['A']:.1f} | {x['all']['rouge_a']:.3f} | {x['short']['rouge_a']:.3f} | {x['all']['comb']:.1f} | {x['short']['comb']:.1f} | {x['all']['answer_rate']:.3f} | {x['short']['answer_rate']:.3f} |")
    L.append("\n### Paired differences (mean, 95% interval over items, over stories)\n")
    for k, v in S["paired"].items():
        L.append(f"- {k}: {v['mean']:+.3f} (items {v['items_95'][0]:+.3f} to {v['items_95'][1]:+.3f}; stories {v['stories_95'][0]:+.3f} to {v['stories_95'][1]:+.3f})")
    L.append("\n### Controls\n\n| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / ROUGE | memory = question only: abstain |\n|---|---|---|---|---|")
    for k, v in S["controls"].items():
        L.append(f"| {k} | {v['empty_abstain']:.3f} | {v['wrong_abstain']:.3f} | {v['shuf_answer_rate']:.3f} / {v['shuf_rouge']:.3f} | {v['severed_abstain']:.3f} |")
    L.append("\n### Provenance\n\n" + "\n".join(f"- {k}: {json.dumps(v)}" for k, v in S["provenance"].items()))
    L.append("\n### UNKNOWN-discovery diagnostic (never a training signal)\n\n" + "\n".join(f"- {k}: {json.dumps(v)}" for k, v in S["unknown_diagnostic"].items()))
    L.append(f"\nFrame reference points: {json.dumps(S['frame_reference_points'])}")
    L.append(f"\n### Round trip (main, filler, threshold {S['round_trip_main'].get('threshold')})\n\n{json.dumps(S['round_trip_main'])}")
    L.append(f"\n### Scorer reasons (main filler vs best reference)\n\n{json.dumps(S['reasons_main'])}")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
