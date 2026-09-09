"""USVS-GRADED tree scoring + soft targets (dev/USVS_GRADED_SCORING.md).

The lead directive of 2026-09-08 (`RESEARCH_NOTES.md`): the loss and the
evaluation must score the FLATTENED output tree against the gold tree in
USVS space -- graded similarity -- not binary match / no-match. Today
`encoder_model.teacher_force_loss` is one-hot CE over five discrete heads
and the metric is edge-F1 over exact `(relation, token_index, gtype)`
triples, so a node attached one head off scores 0 exactly like a phantom.

This module supplies the graded half:

  - `node_vector` -- a node's coordinate in USVS-space-plus-reserved-axes
    (S2 of the design doc): the normalized mean of a `sense` node's
    CANDIDATE SET (candidates-first -- the encoder emits sets and the gold
    node is a set, so neither side is ever collapsed to a pick), and
    reserved two-hot vectors for `entity`/`reference`/`elision`/`prime`.
  - `flatten_tree` / `usvs_tree_similarity` -- the bag-of-nodes flattening
    and the aligned, role-weighted graded P/R/F (S3).
  - `score_record_graded` / `aggregate_graded` -- the forest-level eval view,
    mirroring `encoder_model.score_record`/`aggregate_recall` field for field.
  - `ROLE_CONFUSION` + `role_weight` -- the role-confusion matrix, in ONE
    place, deliberately easy to edit (S3.2).
  - `soft_role_target` / `soft_gtype_target` / `soft_source_target` -- the
    opt-in soft CE targets for `--loss usvs-soft` (S5).
  - `corrupt_tree` -- the sanity control (S4).

Nothing here is imported by the default training or eval path unless a
caller opts in; `encoder_model`'s existing behavior is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clause import MODIFIER_RELATIONS
from .tree_render import normalize_gold_tree

# ---------------------------------------------------------------------------
# S3.2 -- THE ROLE CONFUSION MATRIX. One place. Edit here.
# ---------------------------------------------------------------------------

#: Explicit unordered role pairs -> partial credit in [0, 1]. Identical
#: labels are always 1.0 and are not listed. Anything unlisted (and not
#: covered by a family rule below) is 0.0.
ROLE_CONFUSION: Dict[frozenset, float] = {
    # modifier family -- the labels the richer-gold extraction most often
    # trades between (dev/ENCODER_IO_CONTRACT_V2.md S3)
    frozenset({"DESCRIPTION", "SPECIFICATION"}): 0.5,
    frozenset({"COMPLEMENT", "SUBJECT_COMPLEMENT"}): 0.5,
    frozenset({"DESCRIPTION", "COMPLEMENT"}): 0.3,
    frozenset({"SPECIFICATION", "COMPLEMENT"}): 0.3,
    frozenset({"ADDITIVE", "FOCUS"}): 0.5,           # D4, the "too"/"also" particle
    # core-argument confusions
    frozenset({"SUBJECT", "AGENT"}): 0.5,
    frozenset({"INDIRECT_OBJECT", "RECIPIENT"}): 0.5,
    frozenset({"OBJECT", "INDIRECT_OBJECT"}): 0.3,
    frozenset({"OBJECT", "RECIPIENT"}): 0.3,
    frozenset({"SOURCE", "PLACE"}): 0.3,
    # named explicitly by the directive: a subject/object swap is NOT a
    # near miss, it inverts the proposition.
    frozenset({"SUBJECT", "OBJECT"}): 0.0,
}

#: The closed core relation labels (contract S3). Everything that is neither
#: one of these, nor `PREDICATE`, nor in `MODIFIER_RELATIONS`, is an OPEN
#: PP-role (an uppercased preposition) -- see `is_pp_role`.
CORE_RELATIONS = frozenset({
    "SUBJECT", "OBJECT", "INDIRECT_OBJECT", "PLACE", "SOURCE", "AGENT", "RECIPIENT",
})

#: PLACE / SOURCE against an open PP role: the same attachment, differently
#: labelled (the directive's "PLACE vs a preposition role 0.5").
PLACE_LIKE_VS_PP_ROLE = 0.5
#: Two DIFFERENT open PP roles: same construction, different preposition.
PP_ROLE_VS_PP_ROLE = 0.25

#: Node weights (S3.3).
WEIGHT_PREDICATE = 2.0
WEIGHT_CORE = 1.0
WEIGHT_MODIFIER = 0.5

#: The combined single number (S3.6).
CLAUSE_STRUCT_WEIGHT = 0.15


def is_pp_role(relation: str) -> bool:
    """An OPEN PP role -- contract S3's "an uppercased preposition, e.g.
    `WITH`, `FOR`, `OF`, `ABOUT`". Defined negatively (anything outside the
    closed sets) exactly as the contract defines it, so a new preposition
    needs no table entry here."""
    return (relation not in CORE_RELATIONS
            and relation not in MODIFIER_RELATIONS
            and relation != "PREDICATE")


def role_weight(a: Optional[str], b: Optional[str]) -> float:
    """Role-agreement weight in [0, 1] for a (predicted, gold) role pair --
    symmetric. See dev/USVS_GRADED_SCORING.md S3.2."""
    if a is None or b is None:
        return 0.0
    if a == b:
        return 1.0
    # PREDICATE is structural: a predicate aligned to a role is an error,
    # not a near miss.
    if a == "PREDICATE" or b == "PREDICATE":
        return 0.0
    explicit = ROLE_CONFUSION.get(frozenset({a, b}))
    if explicit is not None:
        return explicit
    a_pp, b_pp = is_pp_role(a), is_pp_role(b)
    if a_pp and b_pp:
        return PP_ROLE_VS_PP_ROLE
    if (a_pp and b in ("PLACE", "SOURCE")) or (b_pp and a in ("PLACE", "SOURCE")):
        return PLACE_LIKE_VS_PP_ROLE
    return 0.0


def node_weight(relation: Optional[str], is_predicate: bool) -> float:
    if is_predicate or relation == "PREDICATE":
        return WEIGHT_PREDICATE
    if relation in MODIFIER_RELATIONS:
        return WEIGHT_MODIFIER
    return WEIGHT_CORE


# ---------------------------------------------------------------------------
# S2 -- node vectors
# ---------------------------------------------------------------------------

#: S5.2 -- the grounding-type confusions the CONTRACT itself states.
#: `{reference, elision}`: contract S4 -- "one shape, three instances" of the
#: same unresolved-slot construct. `{sense, entity}`: `entity` is literally
#: the fallback for a content word USVS does not cover (contract S4.2).
GTYPE_CONFUSION: Dict[frozenset, float] = {
    frozenset({"reference", "elision"}): 0.5,
    frozenset({"sense", "entity"}): 0.3,
}

#: S5.3 -- contract S4.3: "`source:"context"` at authoring time aligns to
#: `source:"memory"` at run time."
SOURCE_CONFUSION: Dict[frozenset, float] = {
    frozenset({"context", "memory"}): 0.5,
    frozenset({"self", "context"}): 0.25,
}


def _confusion_weight(table: Dict[frozenset, float], a: str, b: str) -> float:
    if a == b:
        return 1.0
    return table.get(frozenset({a, b}), 0.0)



#: Reserved-block layout, appended after the `len(usvs.axes)` USVS axes.
_RESERVED_TYPES: Tuple[str, ...] = ("sense", "entity", "reference", "elision", "prime")
#: Width of the hashed identity block. Two distinct identities colliding on
#: one axis would score as identical, so this is sized to make that
#: negligible (~0.02% per node pair) rather than merely unlikely; it is the
#: one place the reserved block is approximate, and it is a stated failure
#: mode in dev/USVS_GRADED_SCORING.md S2.2.
IDENT_AXES = 4096
#: Weight on the TYPE block; the identity axis carries the rest of the unit
#: norm. Same type + same identity -> cosine 1.0 (they "compare exactly");
#: same type + different identity -> TYPE_W**2 = 0.25; different types ->
#: TYPE_W**2 times their `GTYPE_CONFUSION` row cosine (0.0 for unrelated
#: types); against any USVS-block `sense` vector -> 0.0 (disjoint blocks).
TYPE_W = 0.5


def reserved_size() -> int:
    return len(_RESERVED_TYPES) + IDENT_AXES


def _ident_axis(key: str) -> int:
    h = 2166136261
    for byte in key.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h % IDENT_AXES


def _type_block(gtype: str) -> np.ndarray:
    """The TYPE half of a reserved vector: the `GTYPE_CONFUSION` row of
    `gtype`, L2-normalized. Using the same table the soft loss uses (S5.2)
    is what gives a `reference` mistaken for an `elision` partial credit --
    contract S4 calls them "one shape, three instances" of the same
    unresolved-slot construct -- while keeping unrelated types orthogonal."""
    row = np.array([_confusion_weight(GTYPE_CONFUSION, gtype, t) for t in _RESERVED_TYPES],
                   dtype=np.float64)
    n = float(np.linalg.norm(row))
    return row / n if n > 0 else row


def _reserved_vector(d_axes: int, gtype: str, ident: Optional[str]) -> np.ndarray:
    """A non-`sense` node's coordinate: a TYPE block (`_type_block`) plus,
    when the node has a surface/named identity, ONE identity axis hashed
    over `"<gtype>|<ident>"` -- keyed by the type on purpose, so two
    different types never share an identity axis and their similarity is
    decided entirely by the type block."""
    v = np.zeros(d_axes + reserved_size(), dtype=np.float64)
    base = d_axes
    type_scale = TYPE_W if ident is not None else 1.0
    v[base:base + len(_RESERVED_TYPES)] = type_scale * _type_block(gtype)
    if ident is not None:
        v[base + len(_RESERVED_TYPES) + _ident_axis(f"{gtype}|{ident}")] = \
            math.sqrt(1.0 - TYPE_W ** 2)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def node_candidates(node: dict, record: dict) -> List[str]:
    """A node's sense-candidate SET. `grounding.candidates` when the node
    carries it inline, else the record's `token_sense_candidates` table --
    which contract S4.2 makes authoritative when the two disagree."""
    g = node.get("grounding") or {}
    cands = g.get("candidates")
    if cands:
        return list(cands)
    idx = node.get("token_index")
    if idx is None:
        return []
    for entry in record.get("token_sense_candidates", []):
        if entry.get("index") == idx:
            return list(entry.get("sense_candidates") or [])
    return []


def _surface(node: dict, record: dict) -> Optional[str]:
    idx = node.get("token_index")
    tokens = record.get("tokens", [])
    if idx is None or not (0 <= idx < len(tokens)):
        return None
    return str(tokens[idx]).lower()


def node_vector(node: dict, record: dict, usvs) -> np.ndarray:
    """The node's coordinate: `[USVS axes | reserved axes]`, L2-normalized.

    `sense` -> the normalized MEAN of its candidate senses' USVS vectors
    (candidates-first: never a pick; a single-candidate set degenerates to
    that sense's own vector). Everything else -> a reserved two-hot
    (type marker + hashed identity). A `sense` node USVS does not cover
    falls back to the reserved block keyed by its surface token."""
    d_axes = len(usvs.axes)
    g = node.get("grounding") or {}
    gtype = g.get("type")

    if gtype == "sense":
        vecs = []
        for sid in node_candidates(node, record):
            dv = usvs.sense_dense(sid)
            if dv is not None:
                vecs.append(dv)
        if vecs:
            mean = np.mean(np.asarray(vecs, dtype=np.float64), axis=0)
            n = float(np.linalg.norm(mean))
            if n > 0:
                out = np.zeros(d_axes + reserved_size(), dtype=np.float64)
                out[:d_axes] = mean / n
                return out
        # no USVS coverage -- exact-match-only fallback (design S2.1)
        return _reserved_vector(d_axes, "sense", _surface(node, record))

    if gtype == "prime":
        return _reserved_vector(d_axes, "prime", str(g.get("prime")))
    if gtype == "elision":
        return _reserved_vector(d_axes, "elision", None)
    if gtype in ("entity", "reference"):
        return _reserved_vector(d_axes, gtype, _surface(node, record))
    # unknown/absent type (a decode-time node before any gtype was chosen)
    return _reserved_vector(d_axes, "entity", _surface(node, record))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# S3 -- flattening
# ---------------------------------------------------------------------------

@dataclass
class FlatNode:
    role: str
    clause_index: int
    is_predicate: bool
    token_index: Optional[int]
    vector: np.ndarray
    weight: float


@dataclass
class FlatTree:
    nodes: List[FlatNode] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)

    @property
    def n_clauses(self) -> int:
        return len(self.kinds)


def flatten_tree(record: dict, tree: Optional[dict], usvs) -> FlatTree:
    """A NORMALIZED-shape tree (what `beam_decode` emits; run a gold-lattice
    tree through `tree_render.normalize_gold_tree` first -- see
    `flatten_gold_tree`) -> the bag of nodes + clause-level structure the
    similarity consumes."""
    flat = FlatTree()
    for ci, clause in enumerate((tree or {}).get("clauses", []) or []):
        flat.kinds.append(clause.get("utterance_kind", "proposition"))
        pred = clause.get("predicate") or {}
        if pred:
            flat.nodes.append(FlatNode(
                role="PREDICATE", clause_index=ci, is_predicate=True,
                token_index=pred.get("token_index"),
                vector=node_vector(pred, record, usvs),
                weight=node_weight("PREDICATE", True)))
        for r in clause.get("roles", []) or []:
            rel = r.get("relation") or "?"
            flat.nodes.append(FlatNode(
                role=rel, clause_index=ci, is_predicate=False,
                token_index=r.get("token_index"),
                vector=node_vector(r, record, usvs),
                weight=node_weight(rel, False)))
    return flat


def flatten_gold_tree(record: dict, gold_tree: dict, usvs) -> FlatTree:
    """One entry of `record["lattice"]["trees"]` -> `FlatTree`, via the same
    `tree_render.normalize_gold_tree` flattening the parse judge uses."""
    return flatten_tree(record, normalize_gold_tree(record, gold_tree), usvs)


# ---------------------------------------------------------------------------
# S3.4 -- assignment
# ---------------------------------------------------------------------------

def _greedy_max(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Deterministic greedy fallback: repeatedly take the largest remaining
    cell whose row and column are both free."""
    pairs: List[Tuple[int, int]] = []
    if matrix.size == 0:
        return pairs
    rows_free = set(range(matrix.shape[0]))
    cols_free = set(range(matrix.shape[1]))
    order = sorted(((float(matrix[i, j]), i, j)
                    for i in range(matrix.shape[0]) for j in range(matrix.shape[1])),
                   key=lambda t: (-t[0], t[1], t[2]))
    for val, i, j in order:
        if val <= 0.0:
            break
        if i in rows_free and j in cols_free:
            rows_free.discard(i)
            cols_free.discard(j)
            pairs.append((i, j))
    return pairs


def _hungarian_max(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Exact maximum-weight one-to-one assignment (O(n^3) Hungarian /
    Jonker-Volgenant shortest-augmenting-path form, on the negated matrix so
    it maximizes). Written in-module on purpose: scipy is NOT a declared
    dependency of this project (pyproject.toml), and a metric must not
    silently change with the environment. Trees have tens of nodes, so the
    cubic cost is irrelevant."""
    if matrix.size == 0:
        return []
    n, m = matrix.shape
    transposed = n > m
    cost = (-matrix.T if transposed else -matrix).astype(np.float64)
    n_rows, n_cols = cost.shape          # n_rows <= n_cols

    INF = float("inf")
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    p = np.zeros(n_cols + 1, dtype=int)   # p[j] = row matched to column j
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, INF)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    pairs = []
    for j in range(1, n_cols + 1):
        i = p[j]
        if i == 0:
            continue
        r, c = (j - 1, i - 1) if transposed else (i - 1, j - 1)
        if matrix[r, c] > 0.0:
            pairs.append((r, c))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# S3.5/S3.6 -- the similarity itself
# ---------------------------------------------------------------------------

def _empty_result(n_pred: int, n_gold: int) -> Dict[str, float]:
    nan = float("nan")
    return {"graded_p": nan, "graded_r": nan, "graded_f": nan, "graded_overall": nan,
            "clause_count": nan, "clause_kind": nan, "clause_struct": nan,
            "matched_mass": 0.0, "n_matched": 0,
            "n_pred_nodes": n_pred, "n_gold_nodes": n_gold}


def usvs_tree_similarity(pred_tree: Optional[dict], gold_tree: Optional[dict],
                          record: dict, usvs, *, gold_is_lattice: bool = False,
                          use_hungarian: bool = True,
                          pred_record: Optional[dict] = None) -> Dict[str, float]:
    """Graded similarity of a predicted tree against a gold tree in USVS
    space (dev/USVS_GRADED_SCORING.md S3). Both trees are NORMALIZED-shape
    unless `gold_is_lattice`, in which case `gold_tree` is one entry of
    `record["lattice"]["trees"]` and is normalized here.

    `pred_record` (default: `record`) is the record whose `tokens` /
    `token_sense_candidates` the PREDICTED tree's nodes address. It differs
    from `record` only for the "gold vs a different sentence's tree" sanity
    control (design S4), where the two trees genuinely belong to different
    sentences; every real eval call leaves it `None`.

    Returns every component separately, per the directive -- not just the
    final number. `graded_overall` is the one combined number:
    `0.85*graded_f + 0.15*clause_struct`."""
    pred = flatten_tree(pred_record if pred_record is not None else record, pred_tree, usvs)
    gold = (flatten_gold_tree(record, gold_tree, usvs) if gold_is_lattice
            else flatten_tree(record, gold_tree, usvs))

    n_p, n_g = len(pred.nodes), len(gold.nodes)
    if n_g == 0:
        return _empty_result(n_p, n_g)

    # -- clause-structure term (S3.6) ---------------------------------------
    cp, cg = pred.n_clauses, gold.n_clauses
    denom = max(cp, cg)
    if denom == 0:
        clause_count = clause_kind = 0.0
    else:
        clause_count = min(cp, cg) / denom
        agree = sum(1 for i in range(min(cp, cg)) if pred.kinds[i] == gold.kinds[i])
        clause_kind = agree / denom
    clause_struct = 0.5 * clause_count + 0.5 * clause_kind

    if n_p == 0:
        out = _empty_result(n_p, n_g)
        out.update({"graded_p": 0.0, "graded_r": 0.0, "graded_f": 0.0,
                    "clause_count": clause_count, "clause_kind": clause_kind,
                    "clause_struct": clause_struct,
                    "graded_overall": CLAUSE_STRUCT_WEIGHT * clause_struct})
        return out

    # -- pairwise scores + alignment (S3.1/S3.4) ----------------------------
    S = np.zeros((n_p, n_g), dtype=np.float64)
    for i, pnode in enumerate(pred.nodes):
        for j, gnode in enumerate(gold.nodes):
            rw = role_weight(pnode.role, gnode.role)
            if rw <= 0.0:
                continue
            S[i, j] = rw * max(0.0, _cosine(pnode.vector, gnode.vector))

    # the assignment objective weights each pair by the mean of the two
    # nodes' importances, so a predicate match is preferred over a modifier
    # match at equal similarity.
    obj = np.zeros_like(S)
    for i, pnode in enumerate(pred.nodes):
        for j, gnode in enumerate(gold.nodes):
            obj[i, j] = S[i, j] * 0.5 * (pnode.weight + gnode.weight)

    pairs = _hungarian_max(obj) if use_hungarian else _greedy_max(obj)

    matched_p = sum(S[i, j] * pred.nodes[i].weight for i, j in pairs)
    matched_g = sum(S[i, j] * gold.nodes[j].weight for i, j in pairs)
    total_p = sum(nd.weight for nd in pred.nodes)
    total_g = sum(nd.weight for nd in gold.nodes)

    graded_p = matched_p / total_p if total_p else 0.0
    graded_r = matched_g / total_g if total_g else 0.0
    graded_f = (2 * graded_p * graded_r / (graded_p + graded_r)
                if (graded_p + graded_r) > 0 else 0.0)
    overall = (1.0 - CLAUSE_STRUCT_WEIGHT) * graded_f + CLAUSE_STRUCT_WEIGHT * clause_struct

    return {"graded_p": graded_p, "graded_r": graded_r, "graded_f": graded_f,
            "graded_overall": overall,
            "clause_count": clause_count, "clause_kind": clause_kind,
            "clause_struct": clause_struct,
            "matched_mass": float(sum(S[i, j] for i, j in pairs)), "n_matched": len(pairs),
            "n_pred_nodes": n_p, "n_gold_nodes": n_g}


# ---------------------------------------------------------------------------
# S3.7 -- record / corpus level (mirrors encoder_model.score_record)
# ---------------------------------------------------------------------------

_GRADED_FIELDS = ("graded_p", "graded_r", "graded_f", "graded_overall",
                  "clause_count", "clause_kind", "clause_struct")


def score_record_graded(record: dict, forest: Sequence[dict], usvs) -> Dict[str, float]:
    """For each GOLD tree pick the forest tree with the highest
    `graded_overall` (the same best-tree EVAL VIEW `_best_tree_overlap`
    takes for edge-F1), then average over the record's gold trees."""
    gold_trees = record.get("lattice", {}).get("trees", []) or []
    per_gold: List[Dict[str, float]] = []
    for gt in gold_trees:
        best = None
        for pt in forest:
            r = usvs_tree_similarity(pt, gt, record, usvs, gold_is_lattice=True)
            if r["graded_overall"] != r["graded_overall"]:
                continue
            if best is None or r["graded_overall"] > best["graded_overall"]:
                best = r
        if best is None:
            best = usvs_tree_similarity(None, gt, record, usvs, gold_is_lattice=True)
        per_gold.append(best)

    out: Dict[str, float] = {}
    for key in _GRADED_FIELDS:
        vals = [d[key] for d in per_gold if d.get(key) == d.get(key)]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def aggregate_graded(scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in _GRADED_FIELDS:
        vals = [s[key] for s in scores if s.get(key) == s.get(key)]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    out["n_records"] = len(scores)
    return out


# ---------------------------------------------------------------------------
# S4 -- the sanity control
# ---------------------------------------------------------------------------

def corrupt_tree(record: dict, tree: dict, *, gold_is_lattice: bool = True) -> dict:
    """The directive's corrupted copy of a gold tree, as a NORMALIZED-shape
    tree: swap SUBJECT<->OBJECT, re-point the predicate at a non-verb token,
    and add one phantom role node. Deterministic."""
    norm = normalize_gold_tree(record, tree) if gold_is_lattice else tree
    pos = record.get("pos", [])
    non_verb = next((i for i, p in enumerate(pos) if p not in ("VERB", "AUX")), 0)

    clauses = []
    for clause in norm.get("clauses", []):
        pred = dict(clause.get("predicate") or {})
        pred["token_index"] = non_verb
        pred["grounding"] = dict(pred.get("grounding") or {})
        pred["grounding"]["candidates"] = None
        roles = []
        for r in clause.get("roles", []):
            r = dict(r)
            if r.get("relation") == "SUBJECT":
                r["relation"] = "OBJECT"
            elif r.get("relation") == "OBJECT":
                r["relation"] = "SUBJECT"
            roles.append(r)
        roles.append({"relation": "PLACE", "token_index": None,
                       "grounding": {"type": "elision", "candidates": None,
                                      "retrieval": {"source": "memory"}}})
        clauses.append({"predicate": pred, "roles": roles,
                         "utterance_kind": clause.get("utterance_kind", "proposition")})
    return {"clauses": clauses}


# ---------------------------------------------------------------------------
# S5 -- soft CE targets (opt-in; `--loss usvs-soft`)
# ---------------------------------------------------------------------------

#: Default softmax temperature for the soft ROLE target. Smaller = closer to
#: one-hot. Overridable per-call / via `--loss-temperature`.
ROLE_SOFT_TEMPERATURE = 0.25

def _softmax_row(row: Sequence[float], temperature: float) -> List[float]:
    t = max(float(temperature), 1e-6)
    scaled = [v / t for v in row]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def soft_role_target(gold_role: Optional[str], role_vocab: Dict[str, int],
                      temperature: float = ROLE_SOFT_TEMPERATURE) -> List[float]:
    """`softmax(ROLE_CONFUSION row of `gold_role` / T)` over `role_vocab`'s
    index order. `role_weight(r, r) == 1.0` is the row's unique maximum, so
    the target always puts its maximum mass on the gold class."""
    inv = [None] * len(role_vocab)
    for label, idx in role_vocab.items():
        inv[idx] = label
    row = [role_weight(gold_role, label) for label in inv]
    if gold_role in role_vocab:
        row[role_vocab[gold_role]] = 1.0
    return _softmax_row(row, temperature)


def soft_gtype_target(gold_gtype: str, gtypes: Sequence[str],
                       temperature: float = ROLE_SOFT_TEMPERATURE) -> List[float]:
    return _softmax_row([_confusion_weight(GTYPE_CONFUSION, gold_gtype, g) for g in gtypes],
                        temperature)


def soft_source_target(gold_source: str, sources: Sequence[str],
                        temperature: float = ROLE_SOFT_TEMPERATURE) -> List[float]:
    return _softmax_row([_confusion_weight(SOURCE_CONFUSION, gold_source, s) for s in sources],
                        temperature)
