"""The token-free clause reactor: perception fixed, only the REACTION learned.

Perception is deterministic and grounded — each clause becomes a
``(entity, relation, value)`` triple of TPR/prime vectors (no token embedding). The
**only learned parameters** are a small GRU controller + heads that decide, per
clause, how to REACT: a write *gate* into the order-3 entity memory and a *respond*
weight; on responding it **generates** a response meaning-vector, scored
contrastively against the (fixed) option meaning-vectors. See plan / RESEARCH_NOTES §0h.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import entity_memory as em
from .clause import extract_clauses
from .episode import _NAMES
from .tpr import TPRCodec

_NAMESET = {n.lower() for n in _NAMES}


# ---------------------------------------------------------------------------
# Fixed perception: curriculum episode -> stream of grounded clause triples
# ---------------------------------------------------------------------------
def _content_vec(word: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray]) -> np.ndarray:
    if word not in cache:
        tree = resolver.resolve(word)
        cache[word] = codec.contract(codec.encode_matrix(tree.root))
    return cache[word]


def _sentence_triple(sent: str, parser) -> Optional[Tuple[str, str, str, str]]:
    """(subject_entity, 'PLACE', place_word, predicate) for a statement, or None."""
    tree = parser._parse_tree(sent)
    if tree is None:
        return None
    clauses = extract_clauses(tree)
    if not clauses:
        return None
    subj = place = None
    pred = (clauses[0].predicate or "").lower()
    for rel, arg in clauses[0].args:
        if rel == "SUBJECT":
            subj = (arg.token or "").lower()
        elif rel == "PLACE":
            place = (arg.token or "").lower()
    return (subj, "PLACE", place, pred) if subj and place else None


def _question_entity(question: str) -> Optional[str]:
    for w in question.lower().replace("?", " ").split():
        if w in _NAMESET:
            return w
    return None


@dataclass
class ClauseBatch:
    entity: torch.Tensor    # [B, T, d]
    relation: torch.Tensor  # [B, T, d]
    value: torch.Tensor     # [B, T, d]
    pred: torch.Tensor      # [B, T, d]  predicate filler (the verb signal)
    is_q: torch.Tensor      # [B, T]  1 = question (respond) step
    mask: torch.Tensor      # [B, T]  1 = real step
    options: torch.Tensor   # [B, K, d]
    answer: torch.Tensor    # [B]

    def to(self, device):
        return ClauseBatch(*(t.to(device) for t in
                             (self.entity, self.relation, self.value, self.pred,
                              self.is_q, self.mask, self.options, self.answer)))

    def subset(self, idx) -> "ClauseBatch":
        """A minibatch over the leading (episode) dimension."""
        return ClauseBatch(self.entity[idx], self.relation[idx], self.value[idx],
                           self.pred[idx], self.is_q[idx], self.mask[idx],
                           self.options[idx], self.answer[idx])


def build_clause_batch(episodes, parser, resolver, codec: TPRCodec) -> ClauseBatch:
    """Encode curriculum episodes into grounded clause-triple streams (fixed)."""
    cache: Dict[str, np.ndarray] = {}
    d = codec.dim
    q_pred = codec.filler_vec("pred:?")             # the question's (unknown) predicate
    rows = []
    for ep in episodes:
        steps: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
        for sent in ep.context:
            tri = _sentence_triple(sent, parser)
            if tri:
                e, r, v, p = tri
                steps.append((codec.filler_vec("var:" + e), codec.filler_vec("rel:" + r),
                              _content_vec(v, resolver, codec, cache),
                              codec.filler_vec("pred:" + p), 0))
        qent = _question_entity(ep.question)
        if qent is None:
            continue
        steps.append((codec.filler_vec("var:" + qent), codec.filler_vec("rel:PLACE"),
                      np.zeros(d, np.float32), q_pred, 1))
        for sent in getattr(ep, "post_context", []) or []:
            tri = _sentence_triple(sent, parser)
            if tri:
                e, r, v, p = tri
                steps.append((codec.filler_vec("var:" + e), codec.filler_vec("rel:" + r),
                              _content_vec(v, resolver, codec, cache),
                              codec.filler_vec("pred:" + p), 0))
        opt = [_content_vec(o, resolver, codec, cache) for o in ep.options]
        rows.append((steps, opt, ep.answer_idx))

    b = len(rows)
    T = max(len(s) for s, _, _ in rows)
    K = max(len(o) for _, o, _ in rows)
    ent = torch.zeros(b, T, d); rel = torch.zeros(b, T, d); val = torch.zeros(b, T, d)
    prd = torch.zeros(b, T, d)
    is_q = torch.zeros(b, T); mask = torch.zeros(b, T)
    opts = torch.zeros(b, K, d); ans = torch.zeros(b, dtype=torch.long)
    for i, (steps, opt, a) in enumerate(rows):
        for t, (e, r, v, p, q) in enumerate(steps):
            ent[i, t] = torch.from_numpy(e); rel[i, t] = torch.from_numpy(r)
            val[i, t] = torch.from_numpy(v); prd[i, t] = torch.from_numpy(p)
            is_q[i, t] = q; mask[i, t] = 1.0
        for k, ov in enumerate(opt):
            opts[i, k] = torch.from_numpy(ov)
        ans[i] = a
    return ClauseBatch(ent, rel, val, prd, is_q, mask, opts, ans)


# ---------------------------------------------------------------------------
# Learned reaction policy (the ONLY parameters)
# ---------------------------------------------------------------------------
class ClauseReactor(nn.Module):
    """GRU controller over grounded clause triples + the order-3 entity memory.

    Learns: a write gate (commit/overwrite/trust), a respond weight (timing), and a
    generated response meaning-vector. No embeddings — input is fixed grounded vectors.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.dim = dim
        # (entity, relation, value, predicate, mem_read)
        self.gru = nn.GRUCell(5 * dim, hidden)
        self.write_gate = nn.Linear(hidden, 1)            # how strongly to commit value
        self.overwrite_gate = nn.Linear(hidden, 1)        # replace (update) vs add (vote)
        self.respond = nn.Linear(hidden, 1)
        self.response = nn.Linear(hidden + dim, dim)      # generate the response meaning-vector

    def forward(self, batch: ClauseBatch) -> Dict[str, torch.Tensor]:
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, self.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)

        resp_logits, resp_vecs = [], []
        for t in range(T):
            e, r, v = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p = batch.pred[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]
            mem_read = em.query(memory, e, r)                          # [B, d]
            state = self.gru(torch.cat([e, r, v, p, mem_read], dim=-1), state)
            # write gate: statement steps only (questions carry no value)
            gate = torch.sigmoid(self.write_gate(state)).squeeze(-1) * real * (1.0 - isq)
            # overwrite gate: replace the old binding (update) vs accumulate (vote)
            owr = torch.sigmoid(self.overwrite_gate(state)).squeeze(-1) * gate
            memory = em.write(memory, e, r, v, gate, overwrite=owr)
            # respond weight (timing) + generated response meaning-vector
            rl = self.respond(state).squeeze(-1)
            rl = rl.masked_fill(real <= 0, float("-inf"))
            resp_logits.append(rl)
            resp_vecs.append(self.response(torch.cat([state, mem_read], dim=-1)))  # [B, d]

        RL = torch.stack(resp_logits, dim=1)               # [B, T]
        RV = torch.stack(resp_vecs, dim=1)                 # [B, T, d]
        w = torch.softmax(RL, dim=1)                        # respond distribution over steps
        r = (w.unsqueeze(-1) * RV).sum(dim=1)              # [B, d] aggregated response

        # contrastive answer: cosine(generated r, option meaning-vectors)
        rn = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
        on = batch.options / (batch.options.norm(dim=-1, keepdim=True) + 1e-8)
        answer_logits = torch.einsum("bd,bkd->bk", rn, on) * 10.0   # temperature

        return {"answer_logits": answer_logits, "response": r,
                "respond_gates": w, "respond_position": (w * batch.is_q).sum(1)}
