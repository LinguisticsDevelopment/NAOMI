"""M53a smoke gate -- does the placeholder-bound pronoun pipeline learn?

Tiny run only (<=200 episodes, <=2 min): proves the data plumbing (membrane
candidate sets + PLACEHOLDER gold-binding in clause_reactor.build_clause_batch)
is exercised end to end -- the REAL resolver training run is M53b's job, not
this one. Also reports the scripted nearest-entity baseline (data-design
check: must sit at/below chance on the anti-recency half).

Usage: python scripts/probe_pronoun_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

torch.set_num_threads(1)

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import generate_pronoun_episodes, nearest_entity_baseline  # noqa: E402
from nsm_ct.episode import CurriculumGenerator, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def main() -> None:
    t0 = time.time()
    n_pronoun, n_old = 100, 100
    dim, epochs, num_options = 32, 60, 4

    pronoun_eps = generate_pronoun_episodes(n_pronoun, seed=42, num_options=num_options)
    old_eps = CurriculumGenerator(max_level=6, seed=42, num_options=num_options).generate(n_old)
    episodes = old_eps + pronoun_eps

    baseline = nearest_entity_baseline(pronoun_eps)
    print(f"nearest-entity baseline: overall={baseline['accuracy']:.3f} "
          f"(n={baseline['n']}), anti-recency={baseline['anti_recency_accuracy']:.3f} "
          f"(n={baseline['n_anti_recency']})", flush=True)

    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping smoke train.")
        return
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, resolver, codec)
    va = build_clause_batch(va_eps, parser, resolver, codec)

    torch.manual_seed(0)
    model = ClauseReactor(dim=dim)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])
    gold_va = torch.tensor([e.answer_idx for e in va_eps])

    for i in range(epochs):
        out = model(tr)
        loss = F.cross_entropy(out["answer_logits"], gold_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        pred = model(va)["answer_logits"].argmax(-1)
    total = float((pred == gold_va).float().mean())
    print(f"val total={total:.3f}  time={(time.time()-t0):.1f}s", flush=True)

    per_kind = {}
    for i, e in enumerate(va_eps):
        kind = e.meta.get("kind", "old")
        per_kind.setdefault(kind, []).append(bool(pred[i] == e.answer_idx))
    for k in sorted(per_kind):
        w = per_kind[k]
        print(f"  {k}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")


if __name__ == "__main__":
    main()
