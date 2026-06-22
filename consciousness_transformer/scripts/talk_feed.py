"""M11 demo + gate: ONE door (`ConsciousLoop.consume`) over a clause feed.

Trains a small backward navigation policy, then (1) routes ProofWriter examples
through `consume` as clause feeds — verifying the one-door path answers questions at
the backward-reasoner's accuracy by depth — and (2) shows a hand-typed teach-then-ask
feed. No `is_q` flag, no answer options: each clause is self-routed by its meaning-type.
"""
import collections
import sys

sys.path.insert(0, "scripts")
import torch

from nsm_ct.tpr import TPRCodec
from nsm_ct.mind.conscious_loop import ConsciousLoop
from nsm_ct.mind.controller import MindController
from nsm_ct.mind.datasets import proofwriter as pw
from nsm_ct.mind.knowledge import KnowledgeGraph
from nsm_ct.clause_psyche import compute_clause_psyche_losses
from train_proofwriter import _load_items

torch.manual_seed(0)
DEPTHS = ["0", "1", "2"]
codec = TPRCodec(dim=32)
train_items = _load_items("train", DEPTHS, 70)
test_items = _load_items("test", DEPTHS, 40)

# --- train the backward navigation policy (the door's reasoner) ---------------
pool = pw.backward_examples(train_items)
model = MindController(codec, hidden=96, hops=3, halting=False)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print(f"training backward policy on {len(pool)} subgoal steps ...", flush=True)
for epoch in range(35):
    model.train(); perm = torch.randperm(len(pool)).tolist()
    for i in range(0, len(pool), 64):
        sub = [pool[j] for j in perm[i:i + 64]]
        b = pw.build_proofsearch_batch(sub, codec)
        out = model(b); loss = compute_clause_psyche_losses(out, b, model)
        opt.zero_grad(); loss["total"].backward(); opt.step()

# --- gate: answer ProofWriter through the ONE door (consume) ------------------
GOLDS = (pw.TRUE, pw.FALSE, pw.UNKNOWN)
by_depth = collections.defaultdict(lambda: [0, 0])


def _by_example(items):
    """Group flattened (facts,rules,query,ans,depth) items back per shared theory."""
    groups = collections.OrderedDict()
    for (facts, rules, query, ans, depth) in items:
        key = id(rules)
        groups.setdefault(key, (facts, rules, []))[2].append((query, ans, depth))
    return groups.values()


loop = ConsciousLoop(KnowledgeGraph(codec=codec), controller=model)
for (facts, rules, qs) in _by_example(test_items):
    feed = [("fact", *f) for f in facts] + [("rule", r) for r in rules]
    feed += [("query", *q[0]) for q in qs]
    resp = loop.consume(feed)
    for (r, (query, ans, depth)) in zip(resp, qs):
        hit = GOLDS.index(r["answer"]) == ans
        by_depth[depth][0] += hit; by_depth[depth][1] += 1

tot = [sum(v[0] for v in by_depth.values()), sum(v[1] for v in by_depth.values())]
line = "  ".join(f"d{d}={by_depth[d][0]/max(by_depth[d][1],1):.2f}" for d in sorted(by_depth))
print(f"\n[ONE-DOOR consume() on ProofWriter] acc={tot[0]/max(tot[1],1):.3f} | {line}", flush=True)

# --- demo: a hand-typed teach-then-ask feed -----------------------------------
print("\n--- demo: teach three things, then ask (one feed, self-routed) ---")
demo = [
    ("fact", "alice", "is", "furry", "+"),
    ("rule", (("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+")),
    ("rule", (("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+")),
    ("query", "alice", "is", "smart", "+"),
    ("query", "alice", "is", "green", "+"),
]
for r in ConsciousLoop(KnowledgeGraph(codec=codec), controller=model).consume(demo):
    print(f"  is {r['query'][0]} {r['query'][2]}?  ->  {r['answer']}  ({r.get('steps','?')} steps)")
