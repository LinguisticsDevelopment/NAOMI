"""Evaluate the stage-A NAOMI encoder: connected sentences (a), fragmented sentences (b), frozen holdouts (d),
a cheat baseline, and a hand-check dump / tally (c).

  python scripts/eval_naomi_encoder.py --ckpt runs/naomi_encoder/main_best.pt --out runs/naomi_encoder/eval_main.json
  python scripts/eval_naomi_encoder.py --handcheck-tally runs/naomi_encoder/handcheck_judgments.json   (after judging the dump)
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from naomi import encoder as E  # noqa: E402
from naomi.reader import Reader, _sense_space  # noqa: E402
from naomi.score import score_pair  # noqa: E402
from naomi.tree import compact, tree_from_dict  # noqa: E402
from naomi.types import REL_ROOT  # noqa: E402
from train_naomi_encoder import GOLD, OUT, load_jsonl, prep, prep_extra  # noqa: E402

BANDS = [("<=10", 0, 10), ("11-20", 11, 20), ("21-35", 21, 35), ("36+", 36, 10 ** 9)]
VERB_TAG = "VERB"


def band_of(n):
    return next(name for name, lo, hi in BANDS if lo <= n <= hi)


# ----------------------------------------------------------------------------- items
def items_from_records(recs, reader):
    items, _ = prep(recs, None, reader)
    by_id = {r["id"]: r for r in recs}
    for it in items:
        it["tree"] = tree_from_dict(by_id[it["id"]]["tree"])
        it["text"] = by_id[it["id"]]["text"]
    return items


def piece_top(sv, i):
    while sv.sup[i] and sv.head[i] != 0:
        i = sv.head[i] - 1
    return i


# ----------------------------------------------------------------------------- baseline
def baseline_tables(train_items):
    cnt = defaultdict(Counter)
    allc = Counter()
    for it in train_items:
        sv, infos = it["sv"], it["infos"]
        for i, ok in enumerate(sv.sup):
            if ok and sv.head[i] != 0:
                j = sv.head[i] - 1
                cnt[(infos[i].pos, infos[j].pos, j < i)][sv.rel[i]] += 1
                allc[sv.rel[i]] += 1
    table = {k: c.most_common(1)[0][0] for k, c in cnt.items()}
    fallback = next(r for r, _ in allc.most_common() if r != REL_ROOT)
    return table, fallback


def baseline_parse(infos, table, fallback):
    nodes = [i for i, t in enumerate(infos) if t.is_node]
    verbs = [i for i in nodes if infos[i].pos == VERB_TAG]
    heads, rels = {}, {}
    if not verbs:
        verbs = nodes[:1]
    first = verbs[0]
    for i in nodes:
        if i == first:
            heads[i], rels[i] = 0, REL_ROOT
            continue
        if infos[i].pos == VERB_TAG:
            cand = [v for v in verbs if v < i] or [first]
            h = cand[-1]
        else:
            h = min((v for v in verbs if v != i), key=lambda v: (abs(v - i), v), default=first)
        heads[i] = h + 1
        rels[i] = table.get((infos[i].pos, infos[h].pos, h < i), fallback)
    return heads, rels


# ----------------------------------------------------------------------------- scoring one sentence
def eval_sentence(it, heads, rels):
    sv = it["sv"]
    sup = [i for i, ok in enumerate(sv.sup) if ok]
    uas = sum(heads[i] == sv.head[i] for i in sup)
    las = sum(heads[i] == sv.head[i] and rels[i] == sv.rel[i] for i in sup)
    return len(sup), uas, las


class Agg:
    def __init__(self):
        self.n = self.uas = self.las = self.sents = self.exact_u = self.exact_l = 0
        self.mean, self.shape, self.overall = [], [], []

    def add(self, n, uas, las, scores=None):
        self.n, self.uas, self.las, self.sents = self.n + n, self.uas + uas, self.las + las, self.sents + 1
        self.exact_u += int(uas == n)
        self.exact_l += int(las == n)
        if scores:
            self.mean.append(scores[0]), self.shape.append(scores[1]), self.overall.append(scores[2])

    def out(self):
        m = lambda v: round(float(np.mean(v)), 2) if v else None  # noqa: E731
        return {"sentences": self.sents, "tokens": self.n, "uas": round(self.uas / max(self.n, 1), 4),
                "las": round(self.las / max(self.n, 1), 4), "exact_unlabelled": round(self.exact_u / max(self.sents, 1), 4),
                "exact_labelled": round(self.exact_l / max(self.sents, 1), 4),
                "scorer_meaning": m(self.mean), "scorer_shape": m(self.shape), "scorer_overall": m(self.overall)}


def pair_scores(usvs, ref, cand):
    try:
        p = score_pair(ref, cand, usvs)
        return p.meaning, p.shape, p.overall
    except Exception:
        return None


def evaluate_set(enc, items, table, fallback, usvs, with_scores=True):
    conn = {"model": defaultdict(Agg), "baseline": defaultdict(Agg)}
    pieces = {"sup": Agg(), "sup_by_band": defaultdict(Agg)}
    place, root_agree, n_pieces = Counter(), [0, 0], 0
    marg_add, marg_sup, p_add, p_sup = [], [], [], []
    rel_added = Counter()
    n_sup = n_unk = 0
    for it in items:
        infos, sv = it["infos"], it["sv"]
        prop, logp = enc.decode_infos(it["text"], infos)
        heads, rels = prop.heads, prop.rels
        nn = len(heads)
        band = band_of(nn)
        if sv.connected:
            n, u, l = eval_sentence(it, heads, rels)
            sc = pair_scores(usvs, it["tree"], prop.tree) if with_scores else None
            conn["model"][band].add(n, u, l, sc), conn["model"]["all"].add(n, u, l, sc)
            bh, br = baseline_parse(infos, table, fallback)
            bt = E.build_tree(it["text"], infos, bh, br)
            n, u, l = eval_sentence(it, bh, br)
            sc = pair_scores(usvs, it["tree"], bt) if with_scores else None
            conn["baseline"][band].add(n, u, l, sc), conn["baseline"]["all"].add(n, u, l, sc)
            continue
        n, u, l = eval_sentence(it, heads, rels)
        pieces["sup"].add(n, u, l), pieces["sup_by_band"][band].add(n, u, l)
        n_pieces += 1
        main = next((i for i, h in enumerate(sv.head) if sv.sup[i] and h == 0), None)
        if main is not None:
            root_agree[0] += int(heads[main] == 0)
            root_agree[1] += 1
        for i in heads:
            row = np.sort(logp[i][logp[i] > E.NEG / 2])[::-1]
            mg = float(row[0] - row[1]) if len(row) > 1 else float(row[0])
            pc = float(np.exp(logp[i, heads[i]]))
            if sv.unknown[i]:
                n_unk += 1
                marg_add.append(mg), p_add.append(pc)
                if heads[i] == 0:
                    place["ROOT"] += 1
                elif piece_top(sv, heads[i] - 1) == i:        # a token of the piece this top heads
                    place["own piece"] += 1
                else:
                    place["inside another piece"] += 1
                rel_added[rels[i]] += 1
            elif sv.sup[i]:
                n_sup += 1
                marg_sup.append(mg), p_sup.append(pc)
    tot = max(sum(place.values()), 1)
    q = lambda v: {"mean": round(float(np.mean(v)), 3), "median": round(float(np.median(v)), 3)} if v else None  # noqa: E731
    return {
        "connected_sentences": {"model": {k: v.out() for k, v in sorted(conn["model"].items())},
                                "baseline_nearest_verb": {k: v.out() for k, v in sorted(conn["baseline"].items())}},
        "fragmented_sentences": {
            "sentences": n_pieces, "supervised_agreement": pieces["sup"].out(),
            "by_band": {k: v.out() for k, v in sorted(pieces["sup_by_band"].items())},
            "unknown_tokens": n_unk, "supervised_tokens": n_sup,
            "unknown_placement_share": {k: round(c / tot, 4) for k, c in place.most_common()},
            "output_single_rooted_connected_share": 1.0,
            "main_root_agreement": round(root_agree[0] / max(root_agree[1], 1), 4),
            "margin_added_links": q(marg_add), "margin_supervised_links": q(marg_sup),
            "top1_prob_added": q(p_add), "top1_prob_supervised": q(p_sup),
            "relations_used_for_added_links": {k: round(c / max(n_unk, 1), 4) for k, c in rel_added.most_common()}}}


# ----------------------------------------------------------------------------- hand-check
def handcheck_dump(enc, items, path, per_band=10, max_links=10, seed=0):
    rng = random.Random(seed)
    frag = [it for it in items if not it["sv"].connected]
    out = []
    for name, lo, hi in BANDS:
        pool = [it for it in frag if lo <= sum(t.is_node for t in it["infos"]) <= hi]
        rng.shuffle(pool)
        for it in pool[:per_band]:
            infos, sv = it["infos"], it["sv"]
            prop, logp = enc.decode_infos(it["text"], infos)
            added = [i for i in prop.heads if sv.unknown[i]]
            rng.shuffle(added)
            links = []
            for i in sorted(added[:max_links]):
                h = prop.heads[i]
                links.append({"dep": i, "dep_tok": infos[i].token, "head": h - 1 if h else "ROOT",
                              "head_tok": infos[h - 1].token if h else "ROOT", "rel": prop.rels[i],
                              "p": round(float(np.exp(logp[i, h])), 3)})
            out.append({"id": it["id"], "band": name, "n_added_total": len(added),
                        "tokens": " ".join(f"{i}:{t.token}" for i, t in enumerate(infos)),
                        "parser": compact(it["tree"]), "encoder": compact(prop.tree), "links": links})
    json.dump(out, open(path, "w"), indent=1)
    return out


def handcheck_tally(path, dump_path):
    judg = json.load(open(path))["judgments"]            # {id: {dep_index: "r"|"d"|"w"}}
    dump = {d["id"]: d for d in json.load(open(dump_path))}
    tot, by_rel = Counter(), defaultdict(Counter)
    for sid, js in judg.items():
        rel = {str(l["dep"]): l["rel"] for l in dump[sid]["links"]}
        for dep, v in js.items():
            tot[v] += 1
            by_rel[rel[dep]][v] += 1
    return {"overall": dict(tot), "by_relation": {k: dict(v) for k, v in sorted(by_rel.items())}}


# ----------------------------------------------------------------------------- stage A2 additions
TB_DIR = ROOT / "runs" / "treebank"


def contiguous_share(trees_heads):
    """Share of decoded trees whose every subtree is a contiguous block of node tokens (rank among node tokens)."""
    ok = 0
    for heads, nodes in trees_heads:
        rank = {i: r for r, i in enumerate(sorted(nodes))}
        kids = defaultdict(list)
        for i, h in heads.items():
            kids[h - 1 if h else -1].append(i)
        good = True

        def span(i):
            nonlocal good
            lo = hi = rank[i]
            cnt = 1
            for c in kids.get(i, []):
                l, h, n = span(c)
                lo, hi, cnt = min(lo, l), max(hi, h), cnt + n
            good = good and (hi - lo + 1 == cnt)
            return lo, hi, cnt
        for r_ in kids.get(-1, []):
            span(r_)
        ok += good
    return round(ok / max(len(trees_heads), 1), 4)


def evaluate_ewt(enc, items):
    """(c) EWT sentences the parser leaves in pieces: agreement of the model's head / head+relation with the converted
    treebank link on tokens whose supervision came from the treebank, by length band, plus the previous-token baseline."""
    T, band, relc = Counter(), defaultdict(Counter), defaultdict(Counter)
    committed = Counter()
    for it in items:
        if it["sv_parser"].connected or not any(t.is_node for t in it["infos"]):
            continue
        infos, sv = it["infos"], it["sv"]
        prop, _ = enc.decode_infos(it["text"], infos)
        b = band_of(len(prop.heads))
        nodes = sorted(prop.heads)
        for i in nodes:
            if it["src"][i] == "treebank":
                prev = max((j for j in nodes if j < i), default=-1)
                hit = prop.heads[i] == sv.head[i]
                hl = hit and prop.rels[i] == sv.rel[i]
                for d in (T, band[b], relc[sv.rel[i]]):
                    d["n"] += 1
                    d["head"] += hit
                    d["headrel"] += hl
                    d["prev_token_head"] += (prev + 1 == sv.head[i])
            elif it["src"][i] == "parser":
                committed["n"] += 1
                committed["head"] += prop.heads[i] == sv.head[i]
    f = lambda d: {"n": d["n"], **{k: round(d[k] / max(d["n"], 1), 4) for k in ("head", "headrel", "prev_token_head")}}  # noqa: E731
    return {"fragmented_sentences": sum(not it["sv_parser"].connected for it in items), "treebank_links": f(T),
            "by_band": {k: f(v) for k, v in sorted(band.items())},
            "by_relation": {k: f(v) for k, v in sorted(relc.items(), key=lambda x: -x[1]["n"])},
            "parser_committed_agreement": {"n": committed["n"], "head": round(committed["head"] / max(committed["n"], 1), 4)}}


def added_links(enc, it, cap, seed):
    """Links a model adds beyond the parser (tokens the parser left unknown), at most `cap` per sentence, seeded."""
    infos, sv = it["infos"], it["sv"]
    prop, logp = enc.decode_infos(it["text"], infos)
    added = [i for i in prop.heads if sv.unknown[i]]
    random.Random(f"{seed}-{it['id']}").shuffle(added)
    return {i: (prop.heads[i], prop.rels[i]) for i in sorted(added[:cap])}, prop


def blind_build(args, test_items):
    """Pool the links stage A and the other model(s) add on the stage-A hand-check sentences; shuffle; no model names."""
    dump = json.load(open(args.handcheck_dump))
    byid = {it["id"]: it for it in test_items}
    usvs = _sense_space()
    models = {name: E.NaomiEncoder.load(path, usvs) for name, path in (kv.split("=") for kv in args.models.split(","))}
    pool, who = {}, defaultdict(dict)                          # key -> link ; key -> {model: True}
    for d in dump:
        it = byid[d["id"]]
        for name, enc in models.items():
            if name == "A":                                    # stage A: exactly the links recorded and judged then
                links = {l["dep"]: ((l["head"] + 1) if l["head"] != "ROOT" else 0, l["rel"]) for l in d["links"]}
                chk, _ = added_links(enc, it, 10 ** 6, 0)
                assert all(chk.get(i) == v for i, v in links.items()), "stage A decode differs from the recorded dump"
            else:
                links, _ = added_links(enc, it, 10, args.seed)
            for i, (h, r) in links.items():
                key = f"{d['id']}|{i}|{h}|{r}"
                pool[key] = {"sid": d["id"], "dep": i, "head": h, "rel": r}
                who[key][name] = True
    sents = defaultdict(list)
    for key, l in pool.items():
        sents[l["sid"]].append(key)
    rng = random.Random(args.seed)
    order = sorted(sents)
    rng.shuffle(order)
    blind, secret, n = [], {"sentences": {}, "links": {}}, 0
    for si, sid in enumerate(order):
        it, d = byid[sid], next(x for x in dump if x["id"] == sid)
        keys = sorted(sents[sid])
        rng.shuffle(keys)
        links = []
        for key in keys:
            l = pool[key]
            n += 1
            lid = f"L{n:03d}"
            secret["links"][lid] = {"key": key, "models": sorted(who[key])}
            links.append({"link": lid, "dependent": f"{l['dep']}:{it['infos'][l['dep']].token}",
                          "head": "ROOT" if l["head"] == 0 else f"{l['head'] - 1}:{it['infos'][l['head'] - 1].token}",
                          "relation": l["rel"]})
        secret["sentences"][f"S{si + 1:02d}"] = {"id": sid, "band": d["band"]}
        blind.append({"sentence": f"S{si + 1:02d}", "tokens": d["tokens"], "parser_pieces": d["parser"], "links": links})
    json.dump(blind, open(args.blind_out, "w"), indent=1)
    json.dump(secret, open(args.blind_key, "w"), indent=1)
    print(f"{len(blind)} sentences, {n} distinct links -> {args.blind_out} (key in {args.blind_key}, not to be read before judging)")


def blind_join(args):
    j = json.load(open(args.blind_judgments))["judgments"]
    sec = json.load(open(args.blind_key))
    old = json.load(open(args.stage_a_judgments))["judgments"]
    tot = defaultdict(Counter)
    band = defaultdict(lambda: defaultdict(Counter))
    both = Counter()
    confusion = Counter()
    rows = []
    for lid, v in j.items():
        meta = sec["links"][lid]
        sid, dep, head, rel = meta["key"].split("|")
        b = next(x["band"] for x in sec["sentences"].values() if x["id"] == sid)
        ms = meta["models"]
        both["both" if len(ms) > 1 else "only_" + ms[0]] += 1
        for m in ms:
            tot[m][v] += 1
            band[m][b][v] += 1
        if "A" in ms and sid in old and dep in old[sid]:
            confusion[(old[sid][dep], v)] += 1
        rows.append({"link": lid, "key": meta["key"], "models": ms, "judgment": v})
    by_set = defaultdict(Counter)
    for r in rows:
        by_set["both" if len(r["models"]) > 1 else "only_" + r["models"][0]][r["judgment"]] += 1
    n_same = sum(c for (a, b_), c in confusion.items() if a == b_)
    res = {"per_model": {m: dict(c) for m, c in tot.items()},
           "per_model_by_band": {m: {b: dict(c) for b, c in bb.items()} for m, bb in band.items()},
           "link_sets": {k: dict(c) for k, c in by_set.items()},
           "rater_consistency_stage_A_links": {"n": sum(confusion.values()), "same_judgment": n_same,
                                               "old_new_counts": {f"{a}->{b_}": c for (a, b_), c in sorted(confusion.items())}}}
    json.dump({"summary": res, "rows": rows}, open(args.blind_result, "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "main_best.pt"))
    ap.add_argument("--out", default=str(OUT / "eval_main.json"))
    ap.add_argument("--handcheck-dump", default=None, help="write a stage-A style dump here (never overwritten by default)")
    ap.add_argument("--ewt", action="store_true", help="(c) EWT dev+test pieces vs converted treebank links; (e) contiguity")
    ap.add_argument("--blind-build", action="store_true")
    ap.add_argument("--blind-join", action="store_true")
    ap.add_argument("--models", default="A=runs/naomi_encoder/main_best.pt,A2=runs/naomi_encoder_a2/a2_best.pt")
    ap.add_argument("--stage-a-dump", default=str(OUT / "handcheck_dump.json"))
    ap.add_argument("--stage-a-judgments", default=str(OUT / "handcheck_judgments.json"))
    ap.add_argument("--seed", type=int, default=0)
    A2D = ROOT / "runs" / "naomi_encoder_a2"
    ap.add_argument("--blind-out", default=str(A2D / "handcheck_blind.json"))
    ap.add_argument("--blind-key", default=str(A2D / "handcheck_blind_key.json"))
    ap.add_argument("--blind-judgments", default=str(A2D / "handcheck_blind_judgments.json"))
    ap.add_argument("--blind-result", default=str(A2D / "handcheck_blind_result.json"))
    ap.add_argument("--handcheck-tally", default=None)
    ap.add_argument("--max-sents", type=int, default=0, help="quick check on the first N test sentences")
    ap.add_argument("--no-scores", action="store_true")
    a = ap.parse_args()
    a.handcheck_dump = a.handcheck_dump or a.stage_a_dump
    if a.blind_join:
        blind_join(a)
        return
    if a.handcheck_tally:
        res = handcheck_tally(a.handcheck_tally, a.handcheck_dump)
        print(json.dumps(res, indent=1))
        return
    t0 = time.time()
    usvs = _sense_space()
    enc = E.NaomiEncoder.load(a.ckpt, usvs)
    cache = pickle.load(open(OUT / "prep_cache.pkl", "rb"))
    splits = json.load(open(OUT / "splits.json"))["split_of_source"]
    items, keys = cache["items"], cache["keys"]
    train = [it for it, k in zip(items, keys) if splits[k] == "train"]
    test = [it for it, k in zip(items, keys) if splits[k] == "test"]
    dev = [it for it, k in zip(items, keys) if splits[k] == "dev"]
    if a.max_sents:
        test = test[:a.max_sents]
    table, fallback = baseline_tables(train)
    gold = {r["id"]: r for r in load_jsonl(GOLD) if r["id"] in {it["id"] for it in test + dev}}
    for it in test + dev:
        it["tree"], it["text"] = tree_from_dict(gold[it["id"]]["tree"]), gold[it["id"]]["text"]
    print(f"loaded in {time.time() - t0:.0f}s; test {len(test)}", flush=True)
    if a.blind_build:
        blind_build(a, test)
        return
    res = {"ckpt": a.ckpt, "meta": E.torch.load(a.ckpt, map_location="cpu", weights_only=False)["meta"]}
    sets = {"test_split": test}
    rd = Reader(use_encoder=False, cache_path=None)
    for name, p in (("frozen_holdout", ROOT / "runs" / "holdout_gold_v7.jsonl"),
                    ("frozen_holdout_dev", ROOT / "runs" / "holdout_dev_gold_v7.jsonl")):
        if p.exists():
            sets[name] = items_from_records(load_jsonl(p), rd)
    for name, its in sets.items():
        res[name] = evaluate_set(enc, its, table, fallback, usvs, not a.no_scores)
        print(f"{name}: done {time.time() - t0:.0f}s", flush=True)
    frag = [it for it in test if not it["sv"].connected]
    res["test_fragmented_contiguous_share"] = contiguous_share(
        [(p.heads, set(p.heads)) for p in (enc.decode_infos(it["text"], it["infos"])[0] for it in frag)])
    if a.ewt:
        for sp in ("dev", "test"):
            its = prep_extra(TB_DIR / f"ewt_{sp}.jsonl.gz", rd, True)[0]
            res["ewt_" + sp] = evaluate_ewt(enc, its)
            print(f"ewt_{sp}: done {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if "--handcheck-dump" in sys.argv:
        handcheck_dump(enc, test, a.handcheck_dump)
    print(json.dumps(res["test_split"]["connected_sentences"]["model"]["all"], indent=1))
    print(json.dumps(res["test_split"]["fragmented_sentences"], indent=1)[:2500])


if __name__ == "__main__":
    main()
