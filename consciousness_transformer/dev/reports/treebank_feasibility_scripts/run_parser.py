"""Run the project's parser (Reader.read_parser, 10 s cap) on sampled UD sentences. Usage: run_parser.py SAMPLE.json OUT_PREFIX SHARD NSHARD [ud_file_glob_dir]"""
import sys, os, json, time, glob
sys.path.insert(0, os.path.dirname(__file__))
import conllu
from naomi.reader import Reader, normalize
from naomi.tree import tree_to_dict
sample, outp, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = sys.argv[5:]
ids = {x["id"]: x for x in json.load(open(sample))}
texts = {}
for f in files:
    for s in conllu.read(f):
        i = s["meta"]["sent_id"]
        if i in ids:
            texts[i] = s["meta"]["text"]
order = sorted(ids)
rd = Reader(use_encoder=False, cache_path=None, parser_seconds=10)
done = set()
fn = f"{outp}.{shard}.jsonl"
if os.path.exists(fn):
    done = {json.loads(l)["id"] for l in open(fn)}
with open(fn, "a") as out:
    for k, i in enumerate(order):
        if k % nsh != shard or i in done:
            continue
        txt = texts[i]
        t0 = time.time()
        toks, pos = rd.tag(normalize(txt))
        rr = rd.read_parser(txt)
        out.write(json.dumps({"id": i, "text": txt, "tokens": toks, "pos": pos, "secs": round(time.time() - t0, 2),
                              "tree": tree_to_dict(rr.tree) if rr else None}) + "\n")
        out.flush()
