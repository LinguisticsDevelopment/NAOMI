"""The learned universal encoder: a candidate-lattice emitter.

Builds to `dev/ENCODER_MODEL_SPEC.md` (+ `dev/ENCODER_IO_CONTRACT_V2.md`,
`dev/ENCODER_GRAMMAR_FORMAT_PROPOSAL.md`). This module is a *token -> tree-SET
transducer*: a small retrieval-conditioned, grammar-constrained neural
transition parser. It NEVER argmaxes a sense, an attachment, or an antecedent
-- there is no head anywhere in this file that scores one sense/candidate
against another. Every ambiguity is emitted as a candidate SET, copied whole
from retrieval (`token_sense_candidates`), never generated or ranked.

Contents:
  - the 7-action transition system + a grammar-conditioned legality mask
    (structural preconditions only; see the module docstring note on scope)
  - the oracle: gold tree -> teacher-forced action sequence (spec S3.1)
  - the policy network (spec S2.2): hash token embedding + POS + pooled USVS
    sense features + fired-rule features -> biGRU encoder -> a GRU-cell
    transition controller -> factored action-type / typed-arg /
    grounding-type / source heads
  - greedy + beam decoding (inference), producing a top-k candidate forest
  - candidate-set recall evaluation (spec S6)

Scope note (the one deliberate simplification, beyond spec S8's own flagged
memory-retrieval-identity gap): `fired_rules` is a real, computed multi-hot
feature over the 7 named grammar rules, POS/lexset/position-triggered, and
conditions the controller -- but the hard action-legality MASK enforced below
implements only the transition-system STRUCTURAL preconditions (spec S3.3
bullet 1: can't GROUND an empty buffer, can't CLOSE_CLAUSE with no open
clause, etc.), not a per-rule action_map gate (S3.3 bullet 2). The structural
mask is what makes illegal transitions unrepresentable; the finer per-rule
gate is left as a learned conditioning signal in this Stage-i smoke build.
This never excludes a gold oracle action (verified by construction), so
teacher forcing is exact either way.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# The action / label inventories (spec S2.1, S2.2)
# ---------------------------------------------------------------------------

ACTION_TYPES: List[str] = [
    "SHIFT", "OPEN_CLAUSE", "GROUND", "ATTACH",
    "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT", "CLOSE_CLAUSE",
]
ACTION_INDEX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_TYPES)}

CLAUSE_KINDS: List[str] = ["proposition", "imperative", "interjection"]
KIND_INDEX: Dict[str, int] = {k: i for i, k in enumerate(CLAUSE_KINDS)}

# The unified `grounding.type` vocabulary (contract S4.1) -- the ONLY place
# a node's type is decided; it never ranks candidates within a type.
GROUNDING_TYPES: List[str] = ["sense", "entity", "reference", "elision", "prime"]
GTYPE_INDEX: Dict[str, int] = {g: i for i, g in enumerate(GROUNDING_TYPES)}

SOURCES: List[str] = ["lexicon", "self", "context", "memory"]
SOURCE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(SOURCES)}

PRIMES: List[str] = ["YOU", "<UNK_PRIME>"]
PRIME_INDEX: Dict[str, int] = {p: i for i, p in enumerate(PRIMES)}

UNK = "<UNK>"

# The 7 declarative grammar rules (grammar-format-proposal S5), by id -- the
# fired-rule feature's column order.
RULE_IDS: List[str] = [
    "imperative.synth_subject",       # R1
    "imperative.bare_adverb",         # R2
    "interjection.ground_only",       # R3
    "ellipsis.inherit_predicate",     # R4
    "prodrop.null_argument",          # R5
    "clause.transitive",              # R6
    "attachment.pp_ambiguous",        # R7
]
N_RULES = len(RULE_IDS)

_INTERJECTION_LEXSET = {
    "oh", "ah", "alas", "ugh", "ouch", "oops", "phew", "yuck", "hurray",
    "wow", "shit", "damn", "hell",
}


# ---------------------------------------------------------------------------
# Deterministic hashing (spec S2.2: hash-embedded token id, no big table;
# independent of PYTHONHASHSEED so runs are reproducible)
# ---------------------------------------------------------------------------

def hash_bucket(token: str, n_buckets: int) -> int:
    h = 2166136261
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h % n_buckets


# ---------------------------------------------------------------------------
# Vocabularies over the gold corpus (POS tags, role labels -- closed/open
# linguistic label spaces, not evaluation targets; content candidates are
# never drawn from a learned vocabulary, only ever copied from retrieval)
# ---------------------------------------------------------------------------

def build_pos_vocab(records: Sequence[dict]) -> Dict[str, int]:
    tags = sorted({p for r in records for p in r["pos"]})
    return {t: i for i, t in enumerate([UNK] + tags)}


def build_role_vocab(records: Sequence[dict]) -> Dict[str, int]:
    roles = {"PREDICATE"}
    for r in records:
        for tree in r["lattice"]["trees"]:
            for clause in tree["clauses"]:
                for role in clause["roles"]:
                    roles.add(role["relation"])
    return {t: i for i, t in enumerate([UNK] + sorted(roles))}


# ---------------------------------------------------------------------------
# Fired-rule matcher (spec S3.3 / grammar S5) -- a real, declared-trigger
# multi-hot feature. See module docstring for the masking-scope note.
# ---------------------------------------------------------------------------

def compute_fired_rules(tokens: Sequence[str], pos: Sequence[str],
                         context_present: bool) -> np.ndarray:
    T = len(tokens)
    out = np.zeros((T, N_RULES), dtype=np.float32)
    if T == 0:
        return out
    has_verb = any(p == "VERB" for p in pos)

    # R1 imperative.synth_subject: {pos_any:[VERB], position:utterance_initial,
    # surface_absent:[SUBJECT]} -- approximated by "sentence opens on a VERB".
    if pos[0] == "VERB":
        out[0, 0] = 1.0

    # R2 imperative.bare_adverb: {pos_any:[ADV,ADJ], position:fragment_whole,
    # surface_absent:[PREDICATE], context_present:true}.
    if pos[0] in ("ADV", "ADJ") and context_present:
        out[0, 1] = 1.0

    # R3 interjection.ground_only: {lexset:INTERJECTION, position:fragment_whole}.
    for t, tok in enumerate(tokens):
        if tok.lower() in _INTERJECTION_LEXSET:
            out[t, 2] = 1.0

    # R4 ellipsis.inherit_predicate: {pos_any:[ADV,ADJ,PROPN,ADP,NOUN],
    # position:fragment_whole, surface_absent:[PREDICATE], context_present:true}
    # -- "no finite verb anywhere in the span" stands in for surface_absent.
    if pos[0] in ("ADV", "ADJ", "PROPN", "ADP", "NOUN") and not has_verb and context_present:
        out[0, 3] = 1.0

    # R5 prodrop.null_argument: {surface_absent:[SUBJECT|OBJECT],
    # context_present:true} -- "no nominal precedes the first verb" stands in
    # for a missing overt subject.
    first_verb = next((i for i, p in enumerate(pos) if p == "VERB"), None)
    if context_present and first_verb is not None:
        if not any(p in ("PRON", "NOUN", "PROPN") for p in pos[:first_verb]):
            out[first_verb, 4] = 1.0

    # R6 clause.transitive: {pos_any:[VERB], position:pre_clause} -- the
    # non-soft base case; fires at every finite verb.
    for t, p in enumerate(pos):
        if p == "VERB":
            out[t, 5] = 1.0

    # R7 attachment.pp_ambiguous: {pos_any:[ADP], position:pre_clause} --
    # fires at every preposition (a candidate PP-attachment site).
    for t, p in enumerate(pos):
        if p == "ADP":
            out[t, 6] = 1.0

    return out


# ---------------------------------------------------------------------------
# Per-sentence features (spec S1.1) -- every channel a retrieval result, never
# a free-text embedding of world knowledge (design S1 invariant).
# ---------------------------------------------------------------------------

@dataclass
class SentenceFeatures:
    tokens: List[str]
    pos: List[str]
    tok_hash: torch.Tensor       # LongTensor[T]
    pos_id: torch.Tensor         # LongTensor[T]
    sense_feat: torch.Tensor     # FloatTensor[T, d_axes] (pooled, unprojected)
    fired_rules: torch.Tensor    # FloatTensor[T, N_RULES]
    sense_cand: List[List[str]]  # sense_cand[t] -- possibly empty


def build_features(record: dict, usvs, pos_vocab: Dict[str, int],
                    hash_buckets: int) -> SentenceFeatures:
    tokens = record["tokens"]
    pos = record["pos"]
    T = len(tokens)
    d_axes = len(usvs.axes)

    sense_cand: List[List[str]] = [[] for _ in range(T)]
    for entry in record.get("token_sense_candidates", []):
        idx = entry["index"]
        if 0 <= idx < T:
            sense_cand[idx] = list(entry["sense_candidates"])

    tok_hash = torch.tensor([hash_bucket(tok.lower(), hash_buckets) for tok in tokens],
                             dtype=torch.long)
    pos_id = torch.tensor([pos_vocab.get(p, pos_vocab[UNK]) for p in pos], dtype=torch.long)

    sense_feat = np.zeros((T, d_axes), dtype=np.float32)
    for t, cands in enumerate(sense_cand):
        if not cands:
            continue
        vecs = [usvs.sense_dense(sid) for sid in cands]
        vecs = [v for v in vecs if v is not None]
        if vecs:
            sense_feat[t] = np.mean(vecs, axis=0)
    sense_feat_t = torch.from_numpy(sense_feat)

    context_present = bool(record.get("context"))
    fired = compute_fired_rules(tokens, pos, context_present)
    fired_t = torch.from_numpy(fired)

    return SentenceFeatures(tokens=tokens, pos=pos, tok_hash=tok_hash, pos_id=pos_id,
                             sense_feat=sense_feat_t, fired_rules=fired_t, sense_cand=sense_cand)


# ---------------------------------------------------------------------------
# The oracle: gold tree -> teacher-forced transition sequence (spec S3.1)
# ---------------------------------------------------------------------------

@dataclass
class Step:
    action: str
    token_index: Optional[int] = None
    role: Optional[str] = None          # arg for GROUND / ATTACH / EMIT_*_SLOT
    kind: Optional[str] = None          # arg for OPEN_CLAUSE
    gtype: Optional[str] = None         # grounding-type, at GROUND/EMIT sites
    source: Optional[str] = None        # retrieval.source, where present
    prime: Optional[str] = None         # arg for GROUND_PRIME (folded into EMIT_SYNTH_SLOT step)


def _predicate_token_index(record: dict, clause: dict) -> Optional[int]:
    """Recover the predicate's surface token_index (not stored directly on
    `predicate_grounding`) via the first occurrence of `predicate` not
    already claimed by one of this clause's roles -- the same deterministic,
    consume-on-match left-to-right walk the gold builder itself uses for
    repeated words (spec S3.1's closing note)."""
    pred_tok = clause.get("predicate")
    if pred_tok is None:
        return None
    claimed = {r["token_index"] for r in clause["roles"] if r["token_index"] is not None}
    for t, tok in enumerate(record["tokens"]):
        if tok == pred_tok and t not in claimed:
            return t
    return None


def clause_node_order(record: dict, clause: dict) -> List[Tuple[str, dict, Optional[int]]]:
    """The clause's nodes (predicate + roles) in ONE unified left-to-right
    token_index walk -- nulls (synthesized/elided fillers) last. This is what
    keeps the oracle's buffer pointer strictly monotonic (spec S3.1's closing
    note: "the same left-to-right, consume-on-match walk"), including
    clauses whose predicate is not the textually-first content word.
    """
    pg = clause["predicate_grounding"]
    pred_idx = _predicate_token_index(record, clause) if pg["type"] in ("sense", "entity") else None
    nodes: List[Tuple[str, dict, Optional[int]]] = [("PREDICATE", pg, pred_idx)]
    for role in clause["roles"]:
        nodes.append((role["relation"], role["grounding"], role["token_index"]))
    nodes.sort(key=lambda n: (n[2] is None, n[2] if n[2] is not None else 0))
    return nodes


def linearize_tree(record: dict, tree: dict) -> List[Step]:
    """One gold tree -> its canonical teacher-forced action sequence.

    A completed derivation serializes to exactly this one tree. The buffer
    pointer `i` is monotonic across the whole tree (all its clauses share one
    continuous left-to-right walk over `tokens`), matching spec S3.1/S2.1.
    """
    tokens = record["tokens"]
    T = len(tokens)
    steps: List[Step] = []
    i = 0

    def shift_to(t: int) -> None:
        nonlocal i
        while i < t:
            steps.append(Step(action="SHIFT", token_index=i))
            i += 1

    for clause in tree["clauses"]:
        kind = clause.get("utterance_kind", "proposition")
        steps.append(Step(action="OPEN_CLAUSE", kind=kind))

        for role, g, tidx in clause_node_order(record, clause):
            gtype = g["type"]
            source = (g.get("retrieval") or {}).get("source")
            # nodes arrive in ascending token_index order (nulls last), so
            # `i` is always <= tidx here; clamp defensively against a
            # duplicate-token collision rather than ever shifting backward.
            eff_tidx = max(tidx, i) if tidx is not None else None
            if gtype in ("sense", "entity"):
                if eff_tidx is not None:
                    shift_to(eff_tidx)
                steps.append(Step(action="GROUND", token_index=eff_tidx, role=role,
                                   gtype=gtype, source=source))
                if eff_tidx is not None:
                    i = eff_tidx + 1
            elif gtype == "prime":
                steps.append(Step(action="EMIT_SYNTH_SLOT", token_index=None, role=role,
                                   gtype="prime", prime=g.get("prime")))
            elif gtype in ("reference", "elision"):
                if eff_tidx is not None:
                    shift_to(eff_tidx)
                steps.append(Step(action="EMIT_UNRESOLVED_SLOT", token_index=eff_tidx, role=role,
                                   gtype=gtype, source=source))
                if eff_tidx is not None:
                    i = eff_tidx + 1

        steps.append(Step(action="CLOSE_CLAUSE"))

    return steps


# ---------------------------------------------------------------------------
# Grammar-constrained legality mask (spec S3.3, structural preconditions)
# ---------------------------------------------------------------------------

def legal_action_types(open_clause: bool, i: int, T: int) -> List[str]:
    """Structural preconditions only (spec S3.3 bullet 1). Note GROUND /
    EMIT_UNRESOLVED_SLOT / EMIT_SYNTH_SLOT stay legal even once `i>=T`:
    real gold occasionally has two nodes address an overlapping/duplicate
    token_index (e.g. a coref slot re-pointing at an already-consumed
    position), and the oracle's buffer pointer only ever advances -- so a
    node's grounding action must remain legal regardless of exactly where
    the monotonic pointer has gotten to. Only SHIFT needs `i<T` (there is
    nothing left to advance onto). This guarantees the mask can never
    exclude a gold oracle action -- a masked-out gold target would make its
    cross-entropy target -inf-logit -> +inf loss, which is exactly the
    training-time invariant this function exists to prevent.
    """
    if not open_clause:
        return ["OPEN_CLAUSE"]
    types = ["GROUND", "EMIT_UNRESOLVED_SLOT", "EMIT_SYNTH_SLOT", "CLOSE_CLAUSE"]
    if i < T:
        types = ["SHIFT"] + types
    return types


def _mask_vector(legal: Sequence[str]) -> torch.Tensor:
    m = torch.full((len(ACTION_TYPES),), float("-inf"))
    for a in legal:
        m[ACTION_INDEX[a]] = 0.0
    return m


# ---------------------------------------------------------------------------
# The policy (spec S2.2) -- sub-MB, PyTorch CPU only
# ---------------------------------------------------------------------------

class EncoderModel(nn.Module):
    def __init__(self, pos_vocab: Dict[str, int], role_vocab: Dict[str, int],
                 d_axes: int, hash_buckets: int = 4096,
                 d_tok: int = 32, d_pos: int = 8, d_sense: int = 16, d_rule: int = 8,
                 d_model: int = 64, controller_hidden: int = 64,
                 stack_dim: int = 16, action_emb_dim: int = 16):
        super().__init__()
        self.pos_vocab = pos_vocab
        self.role_vocab = role_vocab
        self.hash_buckets = hash_buckets
        self.d_model = d_model
        self.controller_hidden = controller_hidden

        # -- input embedder (S2.2's x_t) --------------------------------
        self.tok_emb = nn.Embedding(hash_buckets, d_tok)
        self.pos_emb = nn.Embedding(len(pos_vocab), d_pos)
        self.sense_proj = nn.Linear(d_axes, d_sense)
        self.rule_proj = nn.Linear(N_RULES, d_rule)
        self.input_proj = nn.Linear(d_tok + d_pos + d_sense + d_rule, d_model)

        # -- encoder: one biGRU pass, cached (S2.2) ----------------------
        assert d_model % 2 == 0
        self.encoder = nn.GRU(d_model, d_model // 2, bidirectional=True, batch_first=True)

        # -- controller (S2.2) -------------------------------------------
        self.clause_kind_emb = nn.Embedding(len(CLAUSE_KINDS) + 1, stack_dim)  # +1 = "no open clause"
        self.action_emb = nn.Embedding(len(ACTION_TYPES) + 1, action_emb_dim)  # +1 = "<start>"
        ctrl_in = d_model + stack_dim + action_emb_dim
        self.controller = nn.GRUCell(ctrl_in, controller_hidden)

        # -- factored heads (S2.1/S2.2) ------------------------------------
        self.action_type_head = nn.Linear(controller_hidden, len(ACTION_TYPES))
        self.kind_head = nn.Linear(controller_hidden, len(CLAUSE_KINDS))
        self.role_head = nn.Linear(controller_hidden, len(role_vocab))
        self.gtype_head = nn.Linear(controller_hidden, len(GROUNDING_TYPES))
        self.source_head = nn.Linear(controller_hidden, len(SOURCES))
        self.prime_head = nn.Linear(controller_hidden, len(PRIMES))

        self._none_clause_id = len(CLAUSE_KINDS)
        self._start_action_id = len(ACTION_TYPES)

    def num_policy_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- feature -> per-token encoding --------------------------------------
    def encode(self, feats: SentenceFeatures) -> torch.Tensor:
        tok = self.tok_emb(feats.tok_hash)
        pos = self.pos_emb(feats.pos_id)
        sense = self.sense_proj(feats.sense_feat)
        rule = self.rule_proj(feats.fired_rules)
        x = torch.cat([tok, pos, sense, rule], dim=-1)
        x = self.input_proj(x).unsqueeze(0)          # [1, T, d_model]
        enc, _ = self.encoder(x)
        return enc.squeeze(0)                         # [T, d_model]

    def init_controller_state(self) -> torch.Tensor:
        return torch.zeros(1, self.controller_hidden)

    def controller_step(self, enc_i: torch.Tensor, open_kind_id: int, prev_action_id: int,
                         h_prev: torch.Tensor) -> torch.Tensor:
        stack_repr = self.clause_kind_emb(torch.tensor([open_kind_id]))
        prev_emb = self.action_emb(torch.tensor([prev_action_id]))
        ctrl_in = torch.cat([enc_i.unsqueeze(0), stack_repr, prev_emb], dim=-1)
        return self.controller(ctrl_in, h_prev)

    def role_id(self, role: Optional[str]) -> int:
        return self.role_vocab.get(role, self.role_vocab[UNK])


# ---------------------------------------------------------------------------
# Teacher-forced loss (spec S3.2) -- action CE + candidate-SET emission.
# There is NO sense-selection term anywhere: the candidate set at a GROUND /
# EMIT_UNRESOLVED_SLOT node is *always* `sense_cand[token_index]` copied
# verbatim by the caller (see `decode_*` / `emit_node` below), never produced
# by this network. The network is only ever asked to predict the action
# TYPE, the role/kind ARG, and the grounding TYPE -- never a candidate.
# ---------------------------------------------------------------------------

def teacher_force_loss(model: EncoderModel, feats: SentenceFeatures, steps: List[Step]) -> torch.Tensor:
    enc = model.encode(feats)
    T = enc.shape[0]
    h = model.init_controller_state()
    open_clause = False
    open_kind_id = model._none_clause_id
    prev_action_id = model._start_action_id
    i = 0

    losses = []
    for step in steps:
        i_clamped = min(i, T - 1) if T > 0 else 0
        enc_i = enc[i_clamped] if T > 0 else torch.zeros(model.d_model)
        h = model.controller_step(enc_i, open_kind_id, prev_action_id, h)

        legal = legal_action_types(open_clause, i, T)
        mask = _mask_vector(legal)
        type_logits = model.action_type_head(h).squeeze(0) + mask
        target_type = torch.tensor(ACTION_INDEX[step.action])
        losses.append(F.cross_entropy(type_logits.unsqueeze(0), target_type.unsqueeze(0)))

        if step.action == "OPEN_CLAUSE":
            kind_logits = model.kind_head(h).squeeze(0)
            losses.append(F.cross_entropy(kind_logits.unsqueeze(0),
                                           torch.tensor([KIND_INDEX.get(step.kind, 0)])))
            open_clause = True
            open_kind_id = KIND_INDEX.get(step.kind, 0)
        elif step.action == "CLOSE_CLAUSE":
            open_clause = False
            open_kind_id = model._none_clause_id
        elif step.action in ("GROUND", "ATTACH", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"):
            role_logits = model.role_head(h).squeeze(0)
            losses.append(F.cross_entropy(role_logits.unsqueeze(0),
                                           torch.tensor([model.role_id(step.role)])))
        if step.action in ("GROUND", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT") and step.gtype is not None:
            gtype_logits = model.gtype_head(h).squeeze(0)
            losses.append(F.cross_entropy(gtype_logits.unsqueeze(0),
                                           torch.tensor([GTYPE_INDEX[step.gtype]])))
        if step.gtype in ("sense", "reference", "elision") and step.source is not None:
            source_logits = model.source_head(h).squeeze(0)
            losses.append(F.cross_entropy(source_logits.unsqueeze(0),
                                           torch.tensor([SOURCE_INDEX.get(step.source, 0)])))
        if step.action == "EMIT_SYNTH_SLOT" and step.prime is not None:
            prime_logits = model.prime_head(h).squeeze(0)
            losses.append(F.cross_entropy(prime_logits.unsqueeze(0),
                                           torch.tensor([PRIME_INDEX.get(step.prime, PRIME_INDEX["<UNK_PRIME>"])])))

        if step.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and step.token_index is not None:
            i = step.token_index + 1
        prev_action_id = ACTION_INDEX[step.action]

    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Decoding (inference) -- beam search over the SAME masked action space.
# Emitted "candidates" are copied verbatim from `sense_cand`, exactly like
# the gold does; the network never ranks them (S0 of the model spec).
# ---------------------------------------------------------------------------

@dataclass
class BeamState:
    h: torch.Tensor
    i: int
    open_clause: bool
    open_kind_id: int
    prev_action_id: int
    logprob: float
    clauses: List[dict] = field(default_factory=list)
    cur_clause: Optional[dict] = None
    steps_taken: int = 0
    done: bool = False


def _emit_node(role: str, token_index: Optional[int], gtype: str,
                source: Optional[str], prime: Optional[str],
                feats: SentenceFeatures) -> dict:
    cands = None
    if gtype == "sense" and token_index is not None:
        cands = list(feats.sense_cand[token_index])
    return {"relation": role, "token_index": token_index, "grounding":
            {"type": gtype, "source": source, "prime": prime}}


def _clause_skeleton(clause: dict) -> frozenset:
    keys = [("PREDICATE", clause["predicate"]["token_index"], clause["predicate"]["grounding"]["type"])]
    for r in clause["roles"]:
        keys.append((r["relation"], r["token_index"], r["grounding"]["type"]))
    return frozenset(keys)


def _tree_skeleton(tree: dict) -> Tuple[frozenset, ...]:
    return tuple(sorted((_clause_skeleton(c) for c in tree["clauses"]), key=lambda s: sorted(map(str, s))))


def beam_decode(model: EncoderModel, feats: SentenceFeatures, beam_width: int = 8,
                 k: int = 8, max_steps: int = 400, max_clauses: int = 6,
                 policy: str = "model", rng: Optional[random.Random] = None) -> List[dict]:
    """Returns up to `k` structurally-distinct trees (a candidate forest).

    `policy="model"` uses the learned action-type distribution (masked);
    `policy="random"` samples uniformly among legal actions instead -- the
    random baseline used for comparison in eval (never used for training).
    """
    with torch.no_grad():
        enc = model.encode(feats) if policy == "model" else None
        T = len(feats.tokens)
        rng = rng or random.Random(0)

        beams = [BeamState(h=model.init_controller_state() if policy == "model" else None,
                            i=0, open_clause=False, open_kind_id=model._none_clause_id,
                            prev_action_id=model._start_action_id, logprob=0.0)]
        finished: List[BeamState] = []

        for _ in range(max_steps):
            if not beams:
                break
            candidates: List[BeamState] = []
            for b in beams:
                if b.done:
                    finished.append(b)
                    continue
                legal = legal_action_types(b.open_clause, b.i, T)
                if policy == "model":
                    i_clamped = min(b.i, T - 1) if T > 0 else 0
                    enc_i = enc[i_clamped] if T > 0 else torch.zeros(model.d_model)
                    h = model.controller_step(enc_i, b.open_kind_id, b.prev_action_id, b.h)
                    logits = model.action_type_head(h).squeeze(0) + _mask_vector(legal)
                    logp = F.log_softmax(logits, dim=-1)
                    order = torch.argsort(logp, descending=True).tolist()
                    top = [ACTION_TYPES[idx] for idx in order if ACTION_TYPES[idx] in legal][:min(3, len(legal))]
                else:
                    h = None
                    top = list(legal)
                    rng.shuffle(top)
                    top = top[:min(2, len(legal))]

                for action in top:
                    nb = BeamState(h=h, i=b.i, open_clause=b.open_clause, open_kind_id=b.open_kind_id,
                                    prev_action_id=b.prev_action_id, logprob=b.logprob,
                                    clauses=list(b.clauses), cur_clause=b.cur_clause,
                                    steps_taken=b.steps_taken + 1)
                    if policy == "model":
                        nb.logprob = b.logprob + float(logp[ACTION_INDEX[action]])
                    else:
                        nb.logprob = b.logprob - math.log(max(len(top), 1))
                    _apply_action(nb, action, model, feats, h)
                    if len(nb.clauses) >= max_clauses or nb.steps_taken >= max_steps:
                        nb.done = True
                    candidates.append(nb)
            candidates.sort(key=lambda s: s.logprob, reverse=True)
            beams = candidates[:beam_width]
            beams = [b for b in beams]
            still_going = [b for b in beams if not b.done]
            if not still_going:
                finished.extend(beams)
                break
            beams = still_going + [b for b in beams if b.done]
            finished.extend([b for b in beams if b.done])
            beams = [b for b in beams if not b.done]

        finished.sort(key=lambda s: s.logprob, reverse=True)
        seen = set()
        forest: List[dict] = []
        for b in finished:
            tree = {"clauses": b.clauses}
            sk = _tree_skeleton(tree)
            if sk in seen:
                continue
            seen.add(sk)
            forest.append(tree)
            if len(forest) >= k:
                break
        if not forest and beams:
            forest = [{"clauses": beams[0].clauses}]
        return forest


def _apply_action(state: BeamState, action: str, model: EncoderModel,
                   feats: SentenceFeatures, h: Optional[torch.Tensor]) -> None:
    T = len(feats.tokens)
    state.prev_action_id = ACTION_INDEX[action]
    if action == "SHIFT":
        state.i = min(state.i + 1, T)
        return
    if action == "OPEN_CLAUSE":
        kind = "proposition"
        if h is not None:
            kind = CLAUSE_KINDS[int(torch.argmax(model.kind_head(h).squeeze(0)))]
        state.open_clause = True
        state.open_kind_id = KIND_INDEX.get(kind, 0)
        state.cur_clause = {"predicate": {"token_index": None, "grounding": {"type": "entity"}},
                             "roles": [], "utterance_kind": kind}
        return
    if action == "CLOSE_CLAUSE":
        state.open_clause = False
        state.open_kind_id = model._none_clause_id
        if state.cur_clause is not None:
            state.clauses.append(state.cur_clause)
            state.cur_clause = None
        return

    role = "PREDICATE"
    if h is not None and action in ("GROUND", "ATTACH", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"):
        role_idx = int(torch.argmax(model.role_head(h).squeeze(0)))
        inv = {v: k for k, v in model.role_vocab.items()}
        role = inv.get(role_idx, "PREDICATE")

    gtype, source, prime = None, None, None
    if action in ("GROUND", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"):
        if h is not None:
            gtype = GROUNDING_TYPES[int(torch.argmax(model.gtype_head(h).squeeze(0)))]
        else:
            gtype = random.choice(GROUNDING_TYPES)
        if gtype in ("sense", "reference", "elision"):
            source = SOURCES[int(torch.argmax(model.source_head(h).squeeze(0)))] if h is not None else random.choice(SOURCES)
        if action == "EMIT_SYNTH_SLOT":
            gtype = "prime"
            prime = PRIMES[int(torch.argmax(model.prime_head(h).squeeze(0)))] if h is not None else random.choice(PRIMES)

    token_index = None
    if action in ("GROUND", "EMIT_UNRESOLVED_SLOT") and state.i < T:
        token_index = state.i
        state.i = min(state.i + 1, T)

    # The candidate SET, when present, is always copied verbatim from
    # retrieval (`sense_cand`) -- there is no head anywhere that ranks or
    # selects among them; this is what makes argmax-over-candidates
    # unrepresentable (model spec S0/S3.2).
    candidates = None
    if gtype == "sense" and token_index is not None:
        candidates = list(feats.sense_cand[token_index])
    node = {"relation": role, "token_index": token_index,
            "grounding": {"type": gtype or "entity", "source": source, "prime": prime,
                          "candidates": candidates}}
    if state.cur_clause is None:
        return
    if role == "PREDICATE":
        state.cur_clause["predicate"] = node
    else:
        state.cur_clause["roles"].append(node)


# ---------------------------------------------------------------------------
# Candidate-set recall (spec S6 / contract S7) -- the encoder is NEVER scored
# on the pick, only on whether the gold element is recalled among what it
# emitted. Precision (extra candidates) is not penalized.
# ---------------------------------------------------------------------------

def _gold_sites(record: dict) -> Tuple[Dict[int, str], List[Tuple[Optional[int], str]], List[Tuple[frozenset, ...]]]:
    """-> (sense_sites: token_index->'sense', slot_sites: [(token_index,type)],
          tree_skeletons: one skeleton per gold tree)."""
    sense_sites: Dict[int, str] = {}
    slot_sites: List[Tuple[Optional[int], str]] = []
    trees: List[Tuple[frozenset, ...]] = []
    for tree in record["lattice"]["trees"]:
        clauses_sk = []
        for clause in tree["clauses"]:
            keys = []
            for role, g, tidx in clause_node_order(record, clause):
                if g["type"] == "sense" and tidx is not None:
                    sense_sites[tidx] = "sense"
                elif g["type"] in ("reference", "elision"):
                    slot_sites.append((tidx, g["type"]))
                keys.append((role, tidx, g["type"]))
            clauses_sk.append(frozenset(keys))
        trees.append(tuple(sorted(clauses_sk, key=lambda s: sorted(map(str, s)))))
    return sense_sites, slot_sites, trees


def _emitted_sites(forest: List[dict]) -> Tuple[Dict[int, str], List[Tuple[Optional[int], str]]]:
    sense_sites: Dict[int, str] = {}
    slot_sites: List[Tuple[Optional[int], str]] = []
    for tree in forest:
        for clause in tree["clauses"]:
            pred = clause["predicate"]
            if pred["grounding"]["type"] == "sense" and pred["token_index"] is not None:
                sense_sites[pred["token_index"]] = "sense"
            for role in clause["roles"]:
                g = role["grounding"]
                if g["type"] == "sense" and role["token_index"] is not None:
                    sense_sites[role["token_index"]] = "sense"
                elif g["type"] in ("reference", "elision"):
                    slot_sites.append((role["token_index"], g["type"]))
    return sense_sites, slot_sites


@dataclass
class RecordRecall:
    sense_hits: int
    sense_total: int
    slot_hits: int
    slot_total: int
    tree_hits: int
    tree_total: int
    all_recalled: bool


def score_record(record: dict, forest: List[dict]) -> RecordRecall:
    gold_sense, gold_slots, gold_trees = _gold_sites(record)
    emit_sense, emit_slots = _emitted_sites(forest)
    emit_forest_sk = [_tree_skeleton(t) for t in forest]

    sense_hits = sum(1 for idx in gold_sense if idx in emit_sense)
    sense_total = len(gold_sense)

    slot_hits = 0
    for tidx, gtype in gold_slots:
        if any(t == tidx and g == gtype for t, g in emit_slots):
            slot_hits += 1
    slot_total = len(gold_slots)

    tree_hits = sum(1 for gt in gold_trees if gt in emit_forest_sk)
    tree_total = len(gold_trees)

    all_recalled = (sense_hits == sense_total and slot_hits == slot_total and tree_hits == tree_total)
    return RecordRecall(sense_hits, sense_total, slot_hits, slot_total, tree_hits, tree_total, all_recalled)


def aggregate_recall(scores: List[RecordRecall]) -> Dict[str, float]:
    sh = sum(s.sense_hits for s in scores); st = sum(s.sense_total for s in scores)
    lh = sum(s.slot_hits for s in scores); lt = sum(s.slot_total for s in scores)
    th = sum(s.tree_hits for s in scores); tt = sum(s.tree_total for s in scores)
    ar = sum(1 for s in scores if s.all_recalled)
    n = len(scores)
    return {
        "sense_recall": sh / st if st else float("nan"),
        "slot_recall": lh / lt if lt else float("nan"),
        "structure_recall": th / tt if tt else float("nan"),
        "all_gold_recalled_rate": ar / n if n else float("nan"),
        "n_records": n,
    }


def evaluate(model: EncoderModel, records: Sequence[dict], usvs, pos_vocab: Dict[str, int],
             hash_buckets: int, beam_width: int = 8, k: int = 8,
             policy: str = "model", rng: Optional[random.Random] = None) -> Dict[str, float]:
    scores = []
    for record in records:
        feats = build_features(record, usvs, pos_vocab, hash_buckets)
        forest = beam_decode(model, feats, beam_width=beam_width, k=k, policy=policy, rng=rng)
        scores.append(score_record(record, forest))
    return aggregate_recall(scores)
