"""M55a smoke gate -- does the placeholder-bound garden-path pipeline train
like any other level?

Tiny run only (<=2.5 min): proves the data plumbing (membrane
HypothesisCandidateSet + per-candidate query addresses + PLACEHOLDER
gold-binding in clause_reactor.build_clause_batch / _garden_path_steps) is
exercised end to end -- NO resolver is trained here (M55a is plumbing only,
mirrors scripts/probe_pronoun_smoke.py before M53b's resolver existed).
Also reports the two M55a honesty-gate baselines: PARSER-TOP-1 (always
trust the parser's own best_hypothesis(), no memory) and ASSOCIATION-ONLY
(bag-of-words over context vs options) -- both must sit near/at floor
(~0.5, chance) by curriculum construction (see
dev/TRACK_C_DESIGN.md Sec 1.10, RESEARCH_NOTES M55a,
scripts/probe_m55_hyp_survey.py's survey).

Usage: python scripts/probe_garden_path_smoke.py
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
from nsm_ct.curriculum2 import (  # noqa: E402
    garden_path_association_baseline,
    garden_path_parser_top1_baseline,
    generate_garden_path_episodes,
)
from nsm_ct.episode import CurriculumGenerator, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def main() -> None:
    t0 = time.time()
    n_gp, n_old = 150, 150
    dim, epochs, num_options = 32, 60, 2

    gp_eps = generate_garden_path_episodes(n_gp, seed=42, num_options=num_options)
    old_eps = CurriculumGenerator(max_level=6, seed=42, num_options=4).generate(n_old)
    episodes = old_eps + gp_eps

    assoc = garden_path_association_baseline(gp_eps)
    print(f"association-only baseline: accuracy={assoc['accuracy']:.3f} (n={assoc['n']})",
          flush=True)

    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping smoke train.")
        return

    top1 = garden_path_parser_top1_baseline(gp_eps, parser)
    print(f"parser-top1 baseline: accuracy={top1['accuracy']:.3f} (n={top1['n']})", flush=True)

    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, resolver, codec)
    va = build_clause_batch(va_eps, parser, resolver, codec)

    torch.manual_seed(0)
    model = ClauseReactor(dim=dim)   # no resolver installed -- M55a is plumbing only
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
