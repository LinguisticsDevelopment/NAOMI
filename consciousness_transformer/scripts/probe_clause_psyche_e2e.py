"""Stage 6 — the minimal end-to-end increment on the L8 scenario.

Runs the vertical slice:
  (1) collapse the L8 clauses + exact-expand        (Stage 2, always)
  (2) NOT operator-node + deconvolve                (Stage 3, always)
  (3) SYMBOLIC shared-mary read -> kitchen          (Stage 4, always green)
  (4) NEURAL ClausePsyche generates the answer      (Stage 5, if --ckpt given)

Run:
    python scripts/probe_clause_psyche_e2e.py
    python scripts/probe_clause_psyche_e2e.py --ckpt /tmp/psyche.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_psyche import ClausePsyche  # noqa: E402
from nsm_ct.clause_psyche_graph import STM  # noqa: E402
from nsm_ct.clause_reactor import build_clause_batch  # noqa: E402
from nsm_ct.collapse import expand  # noqa: E402
from nsm_ct.dataset import PARSE_LABELS  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.meaning_graph import OPERATES_ON, read_operator  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="")
    args = ap.parse_args()
    res = NSMMeaningResolver()

    print("=== (1-3) SYMBOLIC: L8 kitchen / office / not office ===")
    stm = STM(TPRCodec(dim=128), res)
    stm.add_clause("mary", "PLACE", "kitchen")
    office = stm.add_clause("mary", "PLACE", "office")
    print(f"  two distinct clauses, one shared mary referent "
          f"(clauses={len(stm.graph.clauses_about(stm.graph.referent_index['mary']))}, "
          f"referents={len(stm.graph.referent_index)})")
    op = stm.negate(office)                                    # NOT operator-node
    label, score, _args = read_operator(stm.graph, op, stm.codec)
    (edge,) = stm.graph.out(op, OPERATES_ON)
    print(f"  negation: operator {label!r} (score {score:.2f}) over clause "
          f"{[n.token for n in expand(stm.graph, edge.dst).root.iter_preorder()]} "
          f"-> truth={stm.graph.node(office).meta['truth']!r}")
    status, val = stm.read("mary", "PLACE")
    print(f"  read(mary, PLACE) -> {status} {stm.value_label(val)!r}")

    if not args.ckpt or not os.path.exists(args.ckpt):
        print("\n(4) NEURAL step skipped (pass --ckpt from train_clause_psyche.py --save).")
        return

    print("\n=== (4) NEURAL: ClausePsyche generates the answer meaning-object ===")
    ck = torch.load(args.ckpt, weights_only=False)
    codec = TPRCodec(dim=ck["dim"])
    eps = [e for e in CurriculumGenerator(max_level=8, seed=1).generate(200) if e.level == 8][:8]
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    batch = build_clause_batch(eps, ParserInputEncoder(tok), res, codec)
    model = ClausePsyche(codec)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        out = model(batch)
    place = torch.einsum("d,bde->be", model.place_role, out["matrix"])   # unbind PLACE from the generated matrix
    pick = torch.einsum("bd,bkd->bk", F.normalize(place, dim=-1),
                        F.normalize(batch.options, dim=-1)).argmax(-1)
    acc = float((pick == batch.answer).float().mean())
    print(f"  generated-matrix decode on {len(eps)} L8 episodes: {acc:.2f} exact place")


if __name__ == "__main__":
    main()
