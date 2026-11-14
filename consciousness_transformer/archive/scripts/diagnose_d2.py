"""One-off diagnostic (not a gate): for depth-2 PROVABLE items, does the learned
rollout fail at the 1st rule selection or the 2nd? Isolates exposure-bias (2nd-step)
from a 1st-step / representation problem. Trains a small policy then inspects."""
import sys, collections
sys.path.insert(0, "scripts")
import torch
from nsm_ct.tpr import TPRCodec
from nsm_ct.mind.controller import MindController
from nsm_ct.mind.datasets import proofwriter as pw
from nsm_ct.mind.executor import Executor
from nsm_ct.clause_psyche import compute_clause_psyche_losses
from train_proofwriter import _load_items

torch.manual_seed(0)
codec = TPRCodec(dim=32)
train_items = _load_items("train", ["0", "1", "2"], 70)
test_items = _load_items("test", ["0", "1", "2"], 50)

# Rebalanced pool: oversample the FIRST move of multi-step proofs (the rare,
# non-goal-matching "intermediate" choice the myopic heuristic drowns out).
OVERSAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 1 = baseline
pool = []
for (facts, rules, query, _a, _d) in train_items:
    steps, _label = pw.proof_rule_steps(facts, rules, query)
    for k, (cur, gold) in enumerate(steps):
        reps = OVERSAMPLE if (len(steps) >= 2 and k == 0) else 1
        pool.extend([(cur, query, rules, gold)] * reps)
print(f"pool={len(pool)} (oversample multi-step first-move x{OVERSAMPLE})", flush=True)
model = MindController(codec, hidden=96, hops=3, halting=False)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for epoch in range(30):
    model.train(); perm = torch.randperm(len(pool)).tolist()
    for i in range(0, len(pool), 64):
        sub = [pool[j] for j in perm[i:i + 64]]
        b = pw.build_proofsearch_batch(sub, codec)
        out = model(b); loss = compute_clause_psyche_losses(out, b, model)
        opt.zero_grad(); loss["total"].backward(); opt.step()
model.eval()

# Inspect depth-2 provable items: gold vs policy rollout rule choices.
first_ok = second_ok = solved = n = 0
seen = 0
for (facts, rules, query, ans, d) in test_items:
    if d != 2 or ans == 2:                      # depth-2 provable only
        continue
    if seen >= 40:
        break
    seen += 1
    needed, rule_of, label = pw.gold_plan(facts, rules, query)
    gold_seq = [rule_of[lit] for lit in needed]
    if len(gold_seq) < 2:                        # want genuine 2-step proofs
        continue
    n += 1
    ex = Executor(codec=codec); ex.load_theory(facts, rules)
    s, p, o, qpol = query
    picks = []
    for step in range(6):
        if (s, p, o, qpol) in ex.pw_closure or (s, p, o, "-" if qpol == "+" else "+") in ex.pw_closure:
            break
        bt = pw.build_proofsearch_batch([(sorted(ex.pw_closure), query, rules, 0)], codec)
        with torch.no_grad():
            idx = int(model(bt)["answer_logits"].argmax(-1)[0])
        picks.append(idx); ex.apply_rule(rules[idx])
    first_ok += (len(picks) >= 1 and picks[0] == gold_seq[0])
    # 2nd-step: among rollouts that got step 1 right, did step 2 match the expert at that state?
    if len(picks) >= 1 and picks[0] == gold_seq[0]:
        # expert at the post-step-1 state
        ex2 = Executor(codec=codec); ex2.load_theory(facts, rules); ex2.apply_rule(rules[gold_seq[0]])
        exp2 = pw.expert_action(ex2.pw_closure, needed, rule_of)
        second_ok += (len(picks) >= 2 and picks[1] == exp2)
    solved += (s, p, o, qpol) in ex.pw_closure

print(f"depth-2 genuine 2-step provable items: n={n}")
print(f"  1st rule correct: {first_ok}/{n} = {first_ok/max(n,1):.2f}")
print(f"  2nd rule correct | 1st correct: {second_ok}/{max(first_ok,1)} = {second_ok/max(first_ok,1):.2f}")
print(f"  fully solved (goal in closure): {solved}/{n} = {solved/max(n,1):.2f}")
