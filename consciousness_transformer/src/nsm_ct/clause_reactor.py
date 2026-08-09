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
from .clause import extract_discourse
from .episode import _NAMES
from .tpr import TPRCodec
from .usvs_bridge import usvs_handle

_NAMESET = {n.lower() for n in _NAMES}

# Content-word meaning source: "usvs" (default since M31.1: the word's USVS
# handle, falling back to explication when the word isn't known to USVS) or
# "explication" (the depth-bounded subtree TPR — still exercised as the
# fallback path, and available explicitly via --meaning-source explication).
# Entity/variable atoms are NEVER affected — they never go through _content_vec.
MeaningSource = str  # "explication" | "usvs"


# ---------------------------------------------------------------------------
# Fixed perception: curriculum episode -> stream of grounded clause triples
# ---------------------------------------------------------------------------
def _content_vec(word: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                  meaning_source: MeaningSource = "usvs") -> np.ndarray:
    key = f"{meaning_source}:{word}"
    if key not in cache:
        vec = None
        if meaning_source == "usvs":
            vec = usvs_handle(word, codec.dim)
        if vec is None:                            # explication path (default / fallback)
            tree = resolver.resolve(word)
            vec = codec.contract(codec.encode_matrix(tree.root))
        cache[key] = vec
    return cache[key]


def _option_vec(word: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                 meaning_source: MeaningSource = "usvs") -> np.ndarray:
    """An MC option's meaning-vector — MAYBE/idk atoms for those, else content."""
    w = (word or "").lower()
    if w == "maybe":
        return codec.filler_vec("MAYBE")
    if w == "idk":
        return codec.filler_vec("idk")            # the abstain atom
    return _content_vec(word, resolver, codec, cache, meaning_source)


def _ent_vec(name: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
             meaning_source: MeaningSource = "usvs") -> np.ndarray:
    """Ground an entity/value: a person → its variable atom; a concept → its meaning."""
    return (codec.filler_vec("var:" + name) if name in _NAMESET
            else _content_vec(name, resolver, codec, cache, meaning_source))


def _reasoning_steps(ep, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                      meaning_source: MeaningSource = "usvs"):
    """Grounded stream for a reasoning episode (L9-L11).

    Perception is a deterministic grounding of the input's *meaning* (the oracle's
    structured premises — the same content the sentence carries; the derived answer
    is never input). A conditional rule streams its antecedent + consequent tagged
    with the IF atom on the ``coord`` channel; facts stream plainly; the question
    step carries the query (entity, relation). The model must CHAIN these to answer.
    Returns ``[(entity, relation, value, pred, coord, is_q)]`` like ``_context_steps``.
    """
    d = codec.dim
    z = np.zeros(d, np.float32)
    pred_is, pred_if = codec.filler_vec("pred:is"), codec.filler_vec("pred:if")
    q_pred, ifv = codec.filler_vec("pred:?"), codec.filler_vec("IF")
    steps = []
    # one conditional rule (meta["rule"]) or a cascade of them (meta["rules"]); each rule
    # streams its antecedent then consequent, tagged with the IF atom on the coord channel.
    rule_list = ep.meta.get("rules") or ([ep.meta["rule"]] if ep.meta.get("rule") else [])
    for rule in rule_list:
        for (e, r, v) in rule:                          # antecedent then consequent
            steps.append((_ent_vec(e, resolver, codec, cache, meaning_source), codec.filler_vec("rel:" + r),
                          _ent_vec(v, resolver, codec, cache, meaning_source), pred_if, ifv, 0))
    for (e, r, v) in ep.meta["facts"]:
        steps.append((_ent_vec(e, resolver, codec, cache, meaning_source), codec.filler_vec("rel:" + r),
                      _ent_vec(v, resolver, codec, cache, meaning_source), pred_is, z, 0))
    qe, qr = ep.meta["query"]
    steps.append((_ent_vec(qe, resolver, codec, cache, meaning_source), codec.filler_vec("rel:" + qr),
                  z, q_pred, z, 1))
    return steps


# M52: SUBJECT->AGENT, INDIRECT_OBJECT->RECIPIENT for OBJECT-bearing (transfer)
# clauses -- the roles the transferred OBJECT itself bears w.r.t. the event
# (see _context_steps). SOURCE/PLACE already read fine as relation names and
# pass through unchanged.
_TRANSFER_ROLE_MAP = {"SUBJECT": "AGENT", "INDIRECT_OBJECT": "RECIPIENT"}


def _context_steps(sent: str, parser, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                    meaning_source: MeaningSource = "usvs"):
    """Grounded steps for a context sentence: one per clause ROLE (M52).

    A disjunction ("A or B") yields one step per disjunct, each carrying the OR
    atom on the coord channel (so the controller can VOTE/superpose them); a
    negation ("not in A") yields one step carrying the NOT atom (so it can
    SUBTRACT). Plain facts carry a zero coord — identical to the old single-triple
    behaviour.

    M52 (multi-arg clauses, RESOLVER_BUILD_PLAN Phase 1): a clause carrying an
    OBJECT argument (a give/hand/pass/take transfer) is unrolled into ONE STEP
    PER OTHER ROLE, all sharing the transferred OBJECT as the entity -- so a
    later question can address the object directly ("where is the ball",
    "who has the ball") without chaining through the giver. SUBJECT maps to
    AGENT and INDIRECT_OBJECT to RECIPIENT for these steps (see
    ``_TRANSFER_ROLE_MAP``); SOURCE/PLACE pass through unchanged. A clause
    with NO OBJECT argument keeps the old single-step shape exactly
    (SUBJECT-is-the-entity, PLACE-is-the-value) -- byte-identical to pre-M52
    for every existing curriculum episode, none of which ever emits an
    OBJECT-labeled arg (see tests/test_multi_arg_transfer.py's regression
    test). Returns ``[(entity, relation, value, pred, coord, is_q=0)]``.
    """
    d = codec.dim
    graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
    clauses, links = extract_discourse(graph)
    if not clauses:
        return []
    prime = links[0].prime if links else None
    coordv = codec.filler_vec(prime) if prime else np.zeros(d, np.float32)
    steps = []
    for cl in clauses:
        pred_vec = codec.filler_vec("pred:" + (cl.predicate or "").lower())
        obj_tok = next(((arg.token or "").lower() for rel, arg in cl.args if rel == "OBJECT"), None)
        if obj_tok:
            entity_vec = _ent_vec(obj_tok, resolver, codec, cache, meaning_source)
            for rel, arg in cl.args:
                if rel == "OBJECT":
                    continue
                tok = (arg.token or "").lower()
                if not tok:
                    continue
                mapped = _TRANSFER_ROLE_MAP.get(rel, rel)
                steps.append((entity_vec, codec.filler_vec("rel:" + mapped),
                              _ent_vec(tok, resolver, codec, cache, meaning_source),
                              pred_vec, coordv, 0))
            continue
        subj = place = None
        for rel, arg in cl.args:
            if rel == "SUBJECT":
                subj = (arg.token or "").lower()
            elif rel == "PLACE":
                place = (arg.token or "").lower()
        if not (subj and place):
            continue
        steps.append((codec.filler_vec("var:" + subj), codec.filler_vec("rel:PLACE"),
                      _content_vec(place, resolver, codec, cache, meaning_source),
                      pred_vec, coordv, 0))
    return steps


# M52 v1 IN table ("queried role"): the relation a question is actually
# asking about. "has" -> RECIPIENT (who now holds it -- the transfer_who
# level); "where" -> PLACE, which is ALSO the fallback/default -- every
# question asked before M52 was "where is {name} ?", so this default
# preserves byte-identical behavior for all of them. "who"/"what" are the
# v1 IN table's other named cases (who-subject -> AGENT/SUBJECT, what ->
# OBJECT), not yet exercised by any curriculum level but resolvable here so
# a future one doesn't need to touch this function.
_QUESTION_ROLE_KEYWORDS = (("has", "RECIPIENT"), ("where", "PLACE"),
                           ("who", "AGENT"), ("what", "OBJECT"))


def _queried_role(question: str) -> str:
    words = set(question.lower().replace("?", " ").split())
    for kw, role in _QUESTION_ROLE_KEYWORDS:
        if kw in words:
            return role
    return "PLACE"


def _question_entity(question: str) -> Optional[str]:
    """The entity a question is about: a NAME if one appears (unchanged), else
    (M52) the noun immediately following "the" -- "where is the ball ?" /
    "who has the ball ?" ask about an OBJECT, not a person. Old-curriculum
    questions never contain "the" (always "where is {name} ?"), so this
    fallback is inert for them: behavior is byte-identical.
    """
    words = question.lower().replace("?", " ").split()
    for w in words:
        if w in _NAMESET:
            return w
    if "the" in words:
        idx = words.index("the")
        if idx + 1 < len(words):
            return words[idx + 1]
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
    coord: Optional[torch.Tensor] = None  # [B, T, d]  OR/NOT/IF atom (else zeros): the logical signal
    answerable: Optional[torch.Tensor] = None  # [B]  1 = derivable, 0 = should abstain (reasoning levels)

    def _coord(self) -> torch.Tensor:
        return self.coord if self.coord is not None else torch.zeros_like(self.entity)

    def to(self, device):
        coord = self.coord.to(device) if self.coord is not None else None
        ans_ok = self.answerable.to(device) if self.answerable is not None else None
        return ClauseBatch(self.entity.to(device), self.relation.to(device),
                           self.value.to(device), self.pred.to(device),
                           self.is_q.to(device), self.mask.to(device),
                           self.options.to(device), self.answer.to(device), coord, ans_ok)

    def subset(self, idx) -> "ClauseBatch":
        """A minibatch over the leading (episode) dimension."""
        coord = self.coord[idx] if self.coord is not None else None
        ans_ok = self.answerable[idx] if self.answerable is not None else None
        return ClauseBatch(self.entity[idx], self.relation[idx], self.value[idx],
                           self.pred[idx], self.is_q[idx], self.mask[idx],
                           self.options[idx], self.answer[idx], coord, ans_ok)


def build_clause_batch(episodes, parser, resolver, codec: TPRCodec,
                        meaning_source: MeaningSource = "usvs") -> ClauseBatch:
    """Encode curriculum episodes into grounded clause-triple streams (fixed).

    Each step is ``(entity, relation, value, pred, coord, is_q)``; ``coord`` carries
    the OR/NOT atom for disjunction/negation steps (zeros otherwise) — the logical
    signal the controller reacts to. Options ground to content vectors, except
    "maybe" → the NSM MAYBE atom (so a disjunction can be answered "maybe").

    ``meaning_source`` (M31 consumer gate) selects how CONTENT-WORD meaning
    vectors are built: ``"usvs"`` (default since M31.1 — beats explication at
    both quick and full training budgets, see RESEARCH_NOTES) uses the word's
    USVS handle (:func:`nsm_ct.usvs_bridge.usvs_handle`), falling back to
    ``"explication"`` (the depth-bounded explication subtree TPR) for words
    USVS doesn't know. Entity/variable atoms (``var:<name>``) are never
    affected by this switch either way.

    M52 (RESOLVER_BUILD_PLAN Phase 1): multi-arg clauses (an OBJECT-bearing
    give/hand/pass/take transfer) unroll into one step per role instead of
    being dropped down to a single (SUBJECT, PLACE) pair — see
    :func:`_context_steps`. The question step's relation is now the
    QUERIED ROLE the question text actually names (:func:`_queried_role`)
    rather than a hardcoded PLACE; the default is still PLACE, so every
    pre-M52 "where is X ?" question is unaffected.
    """
    cache: Dict[str, np.ndarray] = {}
    d = codec.dim
    q_pred = codec.filler_vec("pred:?")             # the question's (unknown) predicate
    z = np.zeros(d, np.float32)
    rows = []
    for ep in episodes:
        if getattr(ep, "level", 0) >= 9 and ep.meta.get("query"):
            steps = _reasoning_steps(ep, resolver, codec, cache, meaning_source)   # L9-L11 reasoning stream
        else:
            steps = []
            for sent in ep.context:
                steps += _context_steps(sent, parser, resolver, codec, cache, meaning_source)
            qent = _question_entity(ep.question)
            if qent is None:
                continue
            qrel = _queried_role(ep.question)
            steps.append((_ent_vec(qent, resolver, codec, cache, meaning_source),
                          codec.filler_vec("rel:" + qrel), z, q_pred, z, 1))
            for sent in getattr(ep, "post_context", []) or []:
                steps += _context_steps(sent, parser, resolver, codec, cache, meaning_source)
        opt = [_option_vec(o, resolver, codec, cache, meaning_source) for o in ep.options]
        rows.append((steps, opt, ep.answer_idx, 1.0 if getattr(ep, "answerable", True) else 0.0))

    b = len(rows)
    T = max(len(s) for s, _, _, _ in rows)
    K = max(len(o) for _, o, _, _ in rows)
    ent = torch.zeros(b, T, d); rel = torch.zeros(b, T, d); val = torch.zeros(b, T, d)
    prd = torch.zeros(b, T, d); crd = torch.zeros(b, T, d)
    is_q = torch.zeros(b, T); mask = torch.zeros(b, T)
    opts = torch.zeros(b, K, d); ans = torch.zeros(b, dtype=torch.long)
    ans_ok = torch.zeros(b)
    for i, (steps, opt, a, ok) in enumerate(rows):
        for t, (e, r, v, p, c, q) in enumerate(steps):
            ent[i, t] = torch.from_numpy(e); rel[i, t] = torch.from_numpy(r)
            val[i, t] = torch.from_numpy(v); prd[i, t] = torch.from_numpy(p)
            crd[i, t] = torch.from_numpy(c); is_q[i, t] = q; mask[i, t] = 1.0
        for k, ov in enumerate(opt):
            opts[i, k] = torch.from_numpy(ov)
        ans[i] = a; ans_ok[i] = ok
    return ClauseBatch(ent, rel, val, prd, is_q, mask, opts, ans, crd, ans_ok)


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
        # (entity, relation, value, predicate, coord, mem_read)
        self.gru = nn.GRUCell(6 * dim, hidden)
        self.write_gate = nn.Linear(hidden, 1)            # how strongly to commit value
        self.overwrite_gate = nn.Linear(hidden, 1)        # replace (update) vs add (vote)
        self.decide_truth = nn.Linear(hidden + dim, 1)    # refute (subtract) this value? (the truth policy)
        self.respond = nn.Linear(hidden, 1)
        self.response = nn.Linear(hidden + dim, dim)      # generate the response meaning-vector

    def forward(self, batch: ClauseBatch) -> Dict[str, torch.Tensor]:
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, self.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)

        coord = batch._coord()
        resp_logits, resp_vecs = [], []
        for t in range(T):
            e, r, v = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p, c = batch.pred[:, t], coord[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]
            mem_read = em.query(memory, e, r)                          # [B, d]
            state = self.gru(torch.cat([e, r, v, p, c, mem_read], dim=-1), state)
            stmt = real * (1.0 - isq)                                  # statement (write) step
            # write gate: statement steps only (questions carry no value)
            gate = torch.sigmoid(self.write_gate(state)).squeeze(-1) * stmt
            # overwrite gate: replace the old binding (update) vs accumulate (vote)
            owr = torch.sigmoid(self.overwrite_gate(state)).squeeze(-1) * gate
            # decide-truth: refute (SUBTRACT) this value — the learned NOT policy. Driven
            # by the coord=NOT signal but supervised only by the answer. value_gate may go
            # negative, so a negated value cancels a previously-voted one (A or B, not A → B).
            neg = torch.sigmoid(self.decide_truth(torch.cat([state, v], dim=-1))).squeeze(-1) * stmt
            memory = em.write(memory, e, r, v, gate - neg, overwrite=owr)
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
