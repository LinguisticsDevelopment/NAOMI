#!/usr/bin/env python
"""NAOMI runner: `read` prints the trees of a text; `coverage` measures how much of each sentence the tree attaches."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))

from naomi import tree as T  # noqa: E402
from naomi.mind import split_sentences  # noqa: E402
from naomi.reader import Reader, normalize  # noqa: E402
from naomi.types import REL_UNATTACHED, SOURCE_PARSER  # noqa: E402


def _p(path):
    p = Path(path)
    return p if p.is_absolute() else CT / p


def cmd_read(a):
    r = Reader(cache_path=a.cache)
    text = _p(a.text).read_text() if _p(a.text).exists() else a.text
    for i, s in enumerate(split_sentences(text)):
        rr = r.read(s, i)
        print(f"[{rr.source}] {s}")
        if rr.tree is None:
            continue
        print(T.render(rr.tree))
        print("  compact:", T.compact(rr.tree))
        for d in rr.tree.diagnostics:
            print("  diagnostic:", json.dumps(d))
    r.save_cache()


def load_sentences(path, limit):
    """One sentence per line, or episodes JSONL (key `context` = list of sentences)."""
    out, seen = [], set()
    for line in open(_p(path), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sents = [line]
        if line.startswith("{"):
            sents = json.loads(line).get("context", [])
        for s in sents:
            if s not in seen:
                seen.add(s)
                out.append(s)
        if len(out) >= limit:
            break
    return out[:limit]


def cmd_coverage(a):
    r = Reader(cache_path=a.cache, use_encoder=a.with_encoder)
    sents = load_sentences(a.sentences, a.limit)
    t0, done, ok = time.time(), 0, 0
    att, tot = Counter(), Counter()          # by pos, parser-success sentences only
    fail_tok = Counter()                     # tokens of sentences with no parser tree
    legacy_att, legacy_tot = 0, 0
    diags, depth, pats = Counter(), Counter(), Counter()
    dump = open(_p(a.dump), "w") if a.dump else None
    for s in sents:
        if time.time() - t0 > a.time_limit:
            print(f"time limit hit after {done} sentences (rerun to continue; results are cached)")
            break
        rr = r.read(s)
        done += 1
        if rr.tree is None or rr.source != SOURCE_PARSER:
            _, pos = r.tag(normalize(s))
            for p in pos:
                if p != "PUNCT":
                    fail_tok[p] += 1
            if rr.tree is not None:
                ns = T.token_nodes(rr.tree)
                legacy_tot += len(ns)
                legacy_att += sum(n.rel != REL_UNATTACHED for n in ns)
            continue
        ok += 1
        for n in T.token_nodes(rr.tree):
            tot[n.pos] += 1
            if n.rel != REL_UNATTACHED:
                att[n.pos] += 1
            else:
                pats[(n.pos, n.label, tuple(n.subtypes))] += 1
                if dump:
                    dump.write(json.dumps({"token": n.token, "pos": n.pos, "label": n.label,
                                           "subtypes": n.subtypes, "sentence": s}) + "\n")
        depth[T.max_depth(rr.tree)] += 1
        diags.update(d["kind"] for d in rr.tree.diagnostics)
    n_all = sum(tot.values())
    pct = lambda x, y: 100.0 * x / y if y else float("nan")  # noqa: E731
    print(f"== {done} sentences; parser success {ok} ({pct(ok, done):.1f}%)")
    print(f"attached tokens, parser-success sentences: {sum(att.values())}/{n_all} = {pct(sum(att.values()), n_all):.1f}%")
    nf = sum(fail_tok.values()) + legacy_tot
    print(f"attached tokens, all sentences (parser failures count as unattached except converted legacy trees): "
          f"{pct(sum(att.values()) + legacy_att, n_all + nf):.1f}%  (failed-sentence tokens {nf}, of which legacy-attached {legacy_att})")
    print("by part of speech (attached/total):")
    for p, t in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"   {p:6s} {att[p]:6d}/{t:6d} = {pct(att[p], t):5.1f}%")
    print("diagnostics:", dict(diags.most_common()))
    print("tree depth distribution:", dict(sorted(depth.items())))
    print("top 30 (pos, parser label, tagger subtypes) among UNATTACHED tokens:")
    for (p, lab, st), c in pats.most_common(30):
        print(f"   {c:6d}  {p:6s} {str(lab):12s} {list(st)}")
    if dump:
        dump.close()
    r.save_cache()


def _print_pair(ta, tb, usvs, label_a="A", label_b="B"):
    from naomi.score import score_coverage, score_pair, tree_similarity
    for lab, t in ((label_a, ta), (label_b, tb)):
        print(f"{lab}: {T.compact(t)}")
        c = score_coverage(t)
        print(f"   coverage {c.percent:.1f}%  attached {100 * c.attached:.1f}%  grounded {100 * c.grounded:.1f}%")
    ps = score_pair(ta, tb, usvs)
    fmt = lambda f: "n/a" if f is None else (  # noqa: E731
        f"recall {f.node_recall:.1f}  precision {f.node_precision:.1f}  F1 {f.node_f1:.1f}  "
        f"attachment {f.attachment_f1:.1f}  overall {f.overall:.1f}")
    print(f"pair (A as reference): MEANING {ps.meaning:.1f}%   SHAPE {ps.shape:.1f}%   overall {ps.overall:.1f}%   "
          f"symmetric overall {tree_similarity(ta, tb, usvs):.1f}%")
    print("  skeleton (operators folded):", fmt(ps.figures))
    print("  function:", fmt(ps.function_words))
    print("  content :", fmt(ps.content_words))
    for r in ps.reasons:
        print("  reason:", json.dumps(r))


def cmd_score(a):
    from nsm_ct.ground.usvs import load_usvs
    usvs = load_usvs(str(CT / "data" / "usvs"))
    if a.tree_json:
        ta, tb = (T.tree_from_json(_p(f).read_text()) for f in a.tree_json)
    else:
        if not (a.a and a.b):
            raise SystemExit("score needs --a and --b sentences, or --tree-json FILE_A FILE_B")
        r = Reader(cache_path=a.cache)
        ta, tb = (r.read(s).tree for s in (a.a, a.b))
        r.save_cache()
        if ta is None or tb is None:
            raise SystemExit("could not read one of the sentences")
    _print_pair(ta, tb, usvs)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("read", help="print the trees of a text (string or file, one sentence per line)")
    t.add_argument("--text", required=True)
    t.add_argument("--cache", default=None)
    s = sp.add_parser("coverage", help="share of tokens the parser tree attaches, by part of speech")
    s.add_argument("--sentences", required=True, help="text file (one per line) or episodes .jsonl (key `context`)")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--cache", default="runs/naomi_tree_cache.jsonl")
    s.add_argument("--time-limit", type=float, default=110.0)
    s.add_argument("--with-encoder", action="store_true", help="fall back to the legacy v6 encoder on parser failure")
    s.add_argument("--dump", default=None, help="write one JSON line per UNATTACHED token here")
    c = sp.add_parser("score", help="grade tree B against tree A node by node in USVS, with reasons")
    c.add_argument("--a", default=None, help="reference sentence")
    c.add_argument("--b", default=None, help="candidate sentence")
    c.add_argument("--tree-json", nargs=2, metavar=("FILE_A", "FILE_B"), default=None, help="stored trees instead")
    c.add_argument("--cache", default=None)
    a = ap.parse_args()
    {"read": cmd_read, "coverage": cmd_coverage, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
