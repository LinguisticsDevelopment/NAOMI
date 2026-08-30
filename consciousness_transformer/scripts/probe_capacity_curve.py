"""M57d: memory CAPACITY CURVE for the order-3 entity(x)relation(x)value
memory tensor (nsm_ct.entity_memory). No training -- deterministic probe.

THE QUESTION: how many instances x attribute relations can ONE [d,d,d]
memory hold before recall breaks, under the EXACT write/read ops the
reactor uses (nsm_ct.entity_memory.write / .query / .query_entity, gate=1
overwrite semantics, argmax-cosine cleanup against a value codebook --
mirrors tests/test_instances.py's interference-sanity test, scaled up)?
This bounds episode length/richness and the dim the scale campaign needs.

WRITES ARE GENUINELY SEQUENTIAL (an earlier draft of this probe assumed a
fill of never-before-written slots collapses to a batched sum of outer
products -- FALSE: entity_memory.write's `old = query(memory, entity,
relation)` is only exactly zero for the very first write into an empty
memory; every later write already sees interference from earlier writes
via `old`, so its `-overwrite*old` term is a real, order-dependent
self-correction, not a no-op. Verified by diffing a naive batched-sum
build against real sequential nsm_ct.entity_memory.write calls on a tiny
case -- they disagree well past float noise.). So :func:`_fill_memory`
below calls :func:`nsm_ct.entity_memory.write` once per (instance,
relation) fact, in instance-major order (all of one instance's attributes,
then the next -- the order tests/test_instances.py and
nsm_ct.instances.write_attribute callers use), batched only over the 5
SEEDS (independent memories) via entity_memory's existing batch dimension.
The OVERWRITE sub-probe continues the same sequential loop with 25% of
slots re-written to a new value, so its `old` is a real read of the
already-interfered-with memory, exactly like a live "recency" update.

Reads (forward :func:`nsm_ct.entity_memory.query` / inverse
:func:`nsm_ct.entity_memory.query_entity`) do NOT mutate memory, so once a
memory tensor is built, every (instance, relation) pair can be queried in
one shot: :func:`_batched_reads` applies the SAME einsums as the real ops,
factored to unbind the (large) relation axis once and reuse it for both
the forward and inverse read, instead of looping.

Grid: dim x n_instances x n_relations x codebook-size V x {codec, random}
value source x 5 seeds. Skips cells whose product exceeds 4x the largest
already-failed product at that dim (capacity is ~monotone decreasing in
fact count, so once a small product has broken, much larger ones are
assumed broken too -- keeps total runtime near budget; sequential writes
make the naive full grid too slow otherwise).

Outputs: runs/capacity_curve.csv (gitignored, full grid) and
dev/CAPACITY_CURVE.md (compact per-dim summary + "so what").
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nsm_ct import entity_memory as em
from nsm_ct.tpr import TPRCodec

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
DEV_DIR = REPO_ROOT / "dev"

DIMS = [32, 48, 64, 96, 128]
N_INSTANCES_GRID = [2, 4, 8, 16, 32, 64]
N_RELATIONS_GRID = [1, 2, 4, 8]
V_GRID = [8, 32]
SOURCES = ["codec", "random"]  # codec = TPRCodec filler vecs (realistic); random = iid unit vectors (theory baseline)
SEEDS = [0, 1, 2, 3, 4]
OVERWRITE_FRAC = 0.25
PRUNE_MULT = 4.0
FAIL_THRESHOLD = 0.5  # forward accuracy below this at a cell => "failed" for pruning purposes
TIME_BUDGET_S = 480.0  # soft cap; remaining cells are skipped (flagged) past this


@dataclass
class CellResult:
    dim: int
    n_instances: int
    n_relations: int
    V: int
    source: str
    n_seeds: int
    product: int
    forward_acc: float
    forward_margin: float
    inverse_acc: float
    inverse_coverage: float  # fraction of (instance, relation) cells where the value was unique in its relation column
    overwrite_new_acc: float
    overwrite_stale_cosine: float
    skipped: bool
    runtime_s: float


# ---------------------------------------------------------------------------
# Seed material.
# ---------------------------------------------------------------------------
def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / (np.linalg.norm(rows, axis=-1, keepdims=True) + 1e-8)


def _draw_seed_material(
    dim: int, n_instances: int, n_relations: int, V: int, source: str, seed: int, codec: Optional[TPRCodec],
):
    """One seed's (entities, relations, codebook, value-index-assignment).

    Entities are fresh iid unit Gaussian atoms -- exactly
    :meth:`nsm_ct.instances.InstanceRegistry.mint`'s distribution (a fresh
    registry per seed). Relations and the value codebook are either
    TPRCodec filler vectors (``source="codec"``, deterministic labels
    ``"attr:rel<r>"`` / ``"val<v>"`` -- the same minting nsm_ct.instances
    uses for attribute relations) or fresh iid unit Gaussian vectors
    (``source="random"``, the near-orthogonal theory baseline).
    """
    rng = np.random.default_rng(seed)
    entities = _unit(rng.standard_normal((n_instances, dim)).astype(np.float32))

    if source == "codec":
        assert codec is not None and codec.dim == dim
        relations = np.stack([codec.filler_vec(f"attr:rel{r}") for r in range(n_relations)])
        codebook = np.stack([codec.filler_vec(f"val{v}") for v in range(V)])
    else:
        relations = _unit(rng.standard_normal((n_relations, dim)).astype(np.float32))
        codebook = _unit(rng.standard_normal((V, dim)).astype(np.float32))

    val_idx = rng.integers(0, V, size=(n_instances, n_relations))
    return entities, relations, codebook, val_idx


# ---------------------------------------------------------------------------
# Sequential fill / overwrite (real nsm_ct.entity_memory.write calls,
# batched only over seeds) and vectorized batched reads.
# ---------------------------------------------------------------------------
def _fill_memory(
    entities: torch.Tensor, relations: torch.Tensor, values: torch.Tensor, n_instances: int, n_relations: int,
) -> torch.Tensor:
    """Write every (instance, relation) fact into an initially-empty memory,
    ONE em.write call per fact (instance-major order), batched over seeds.
    ``entities``: [S, ni, d], ``relations``: [S, nr, d], ``values``:
    [S, ni, nr, d]. Returns the filled ``[S, d, d, d]`` memory.
    """
    S, d = entities.shape[0], entities.shape[-1]
    memory = torch.zeros(S, d, d, d, dtype=entities.dtype)
    gate = torch.ones(S, dtype=entities.dtype)
    for n in range(n_instances):
        e = entities[:, n, :]
        for r in range(n_relations):
            memory = em.write(memory, e, relations[:, r, :], values[:, n, r, :], gate)
    return memory


def _overwrite_slots(
    memory: torch.Tensor,
    entities: torch.Tensor,
    relations: torch.Tensor,
    new_values: torch.Tensor,
    n_sel: torch.Tensor,
    r_sel: torch.Tensor,
) -> torch.Tensor:
    """Continue the SAME sequential-write process with real "recency"
    updates at the selected (n_sel[m], r_sel[m]) slots (one em.write call
    per slot, in selection order), so `old` is a genuine read of the
    already-interfered-with memory, matching entity_memory.write's own
    "overwrite=gate=1 -> the slot becomes value (an UPDATE / recency)"
    semantics. Does not mutate ``memory`` (write is out-of-place).
    """
    S = memory.shape[0]
    gate = torch.ones(S, dtype=memory.dtype)
    out = memory
    for m in range(n_sel.shape[0]):
        n, r = int(n_sel[m]), int(r_sel[m])
        out = em.write(out, entities[:, n, :], relations[:, r, :], new_values[:, m, :], gate)
    return out


def _batched_reads(
    memory: torch.Tensor, entities: torch.Tensor, relations: torch.Tensor, values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward (:func:`nsm_ct.entity_memory.query`) and inverse
    (:func:`nsm_ct.entity_memory.query_entity`) predictions for EVERY
    (instance, relation) pair at once. Reads don't mutate memory, so this
    is safe to batch even though the memory itself was built sequentially.

    ``tmp[s,r,p,q] = sum_j memory[s,p,j,q] * relations[s,r,j]`` unbinds the
    relation axis once; contracting it against entities reproduces
    ``entity_memory.query``'s ``einsum("bijk,bi,bj->bk")`` per (n, r) pair,
    and contracting it against values reproduces
    ``entity_memory.query_entity``'s ``einsum("bijk,bj,bk->bi")``.
    """
    tmp = torch.einsum("spjq,srj->srpq", memory, relations)
    pred = torch.einsum("srpq,snp->snrq", tmp, entities)          # ~ query(memory, entity_n, relation_r)
    entity_pred = torch.einsum("srpq,snrq->snrp", tmp, values)    # ~ query_entity(memory, relation_r, value_{n,r})
    return pred, entity_pred


def measure_cell(
    dim: int,
    n_instances: int,
    n_relations: int,
    V: int,
    source: str,
    seeds: List[int],
    codec: Optional[TPRCodec] = None,
    overwrite_frac: float = OVERWRITE_FRAC,
) -> Dict[str, float]:
    """Build ONE memory per seed, fill it, and measure forward/inverse
    recall + the overwrite sub-probe. Returns plain-float metrics (mean
    over seeds where applicable). Deterministic: same ``seeds`` (and same
    grid params) -> byte-identical output every call.
    """
    S = len(seeds)
    entities = torch.empty(S, n_instances, dim)
    relations = torch.empty(S, n_relations, dim)
    codebook = torch.empty(S, V, dim)
    val_idx = torch.empty(S, n_instances, n_relations, dtype=torch.long)

    for si, seed in enumerate(seeds):
        e, r, cb, vi = _draw_seed_material(dim, n_instances, n_relations, V, source, seed, codec)
        entities[si] = torch.from_numpy(e)
        relations[si] = torch.from_numpy(r)
        codebook[si] = torch.from_numpy(cb)
        val_idx[si] = torch.from_numpy(vi)

    s_range = torch.arange(S).view(S, 1, 1).expand(S, n_instances, n_relations)
    values = codebook[s_range, val_idx]  # [S, ni, nr, d] -- gather each slot's assigned value vector

    memory = _fill_memory(entities, relations, values, n_instances, n_relations)
    pred, entity_pred = _batched_reads(memory, entities, relations, values)

    # --- forward recall ---
    pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    cb_n = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
    fwd_sims = torch.einsum("snrd,svd->snrv", pred_n, cb_n)      # [S, ni, nr, V]
    top2 = fwd_sims.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    fwd_pred_idx = fwd_sims.argmax(dim=-1)
    forward_correct = fwd_pred_idx == val_idx
    forward_acc = forward_correct.float().mean().item()
    forward_margin = margin.mean().item()

    # --- inverse recall (only where the (relation, value) pair is unique among instances at that relation) ---
    ent_n = entity_pred / (entity_pred.norm(dim=-1, keepdim=True) + 1e-8)
    ent_cb_n = entities / (entities.norm(dim=-1, keepdim=True) + 1e-8)
    inv_sims = torch.einsum("snrd,smd->snrm", ent_n, ent_cb_n)   # [S, ni, nr, ni(candidates)]
    inv_pred_idx = inv_sims.argmax(dim=-1)
    true_idx = torch.arange(n_instances).view(1, n_instances, 1).expand(S, n_instances, n_relations)
    inverse_correct_all = inv_pred_idx == true_idx

    unique_mask = torch.zeros(S, n_instances, n_relations, dtype=torch.bool)
    for si in range(S):
        for r in range(n_relations):
            col = val_idx[si, :, r]
            counts = torch.bincount(col, minlength=V)
            unique_mask[si, :, r] = counts[col] == 1
    inverse_coverage = unique_mask.float().mean().item()
    inverse_acc = inverse_correct_all[unique_mask].float().mean().item() if unique_mask.any() else float("nan")

    # --- overwrite sub-probe: 25% of (n, r) slots get a NEW value, written sequentially on top of `memory` ---
    total_slots = n_instances * n_relations
    M = max(1, min(total_slots, round(overwrite_frac * total_slots)))
    ow_rng = np.random.default_rng(1_000_003 * min(seeds) + 7)  # independent of the per-seed draws above, same across seeds
    flat_sel = np.sort(ow_rng.choice(total_slots, size=M, replace=False))
    n_sel = torch.from_numpy(flat_sel // n_relations).long()
    r_sel = torch.from_numpy(flat_sel % n_relations).long()

    old_val_idx_sel = val_idx[:, n_sel, r_sel]           # [S, M] (paired fancy indexing)
    new_val_idx_sel = (old_val_idx_sel + 1) % V          # guaranteed different (V >= 8 in this grid)
    s_range_m = torch.arange(S).view(S, 1).expand(S, M)
    new_val_sel = codebook[s_range_m, new_val_idx_sel]   # [S, M, d]
    old_val_sel = values[:, n_sel, r_sel]                # [S, M, d] -- the clean value that WAS at that slot

    ow_memory = _overwrite_slots(memory, entities, relations, new_val_sel, n_sel, r_sel)
    ow_pred, _ = _batched_reads(ow_memory, entities, relations, values)
    post_pred = ow_pred[:, n_sel, r_sel]                 # [S, M, d] (paired fancy indexing)
    post_n = post_pred / (post_pred.norm(dim=-1, keepdim=True) + 1e-8)

    ow_sims = torch.einsum("smd,svd->smv", post_n, codebook)
    ow_pred_idx = ow_sims.argmax(dim=-1)
    overwrite_new_acc = (ow_pred_idx == new_val_idx_sel).float().mean().item()

    old_val_n = old_val_sel / (old_val_sel.norm(dim=-1, keepdim=True) + 1e-8)
    overwrite_stale_cosine = (post_n * old_val_n).sum(dim=-1).mean().item()

    return dict(
        forward_acc=forward_acc,
        forward_margin=forward_margin,
        inverse_acc=inverse_acc,
        inverse_coverage=inverse_coverage,
        overwrite_new_acc=overwrite_new_acc,
        overwrite_stale_cosine=overwrite_stale_cosine,
    )


# ---------------------------------------------------------------------------
# Grid driver with pruning + a soft wall-clock cap.
# ---------------------------------------------------------------------------
def run_grid(time_budget_s: float = TIME_BUDGET_S) -> List[CellResult]:
    results: List[CellResult] = []
    codecs: Dict[int, TPRCodec] = {d: TPRCodec(dim=d) for d in DIMS}
    combos = sorted({(ni, nr) for ni in N_INSTANCES_GRID for nr in N_RELATIONS_GRID}, key=lambda t: t[0] * t[1])
    t_start = time.perf_counter()
    out_of_time = False

    for dim in DIMS:
        for V in V_GRID:
            for source in SOURCES:
                max_failed_product: Optional[int] = None
                for ni, nr in combos:
                    product = ni * nr
                    skip = out_of_time or (max_failed_product is not None and product > PRUNE_MULT * max_failed_product)
                    if skip:
                        results.append(CellResult(
                            dim=dim, n_instances=ni, n_relations=nr, V=V, source=source, n_seeds=len(SEEDS),
                            product=product, forward_acc=float("nan"), forward_margin=float("nan"),
                            inverse_acc=float("nan"), inverse_coverage=float("nan"),
                            overwrite_new_acc=float("nan"), overwrite_stale_cosine=float("nan"),
                            skipped=True, runtime_s=0.0,
                        ))
                        continue
                    t0 = time.perf_counter()
                    m = measure_cell(dim, ni, nr, V, source, SEEDS, codec=codecs[dim])
                    dt = time.perf_counter() - t0
                    results.append(CellResult(
                        dim=dim, n_instances=ni, n_relations=nr, V=V, source=source, n_seeds=len(SEEDS),
                        product=product, skipped=False, runtime_s=dt, **m,
                    ))
                    if m["forward_acc"] < FAIL_THRESHOLD:
                        max_failed_product = product if max_failed_product is None else max(max_failed_product, product)
                    if time.perf_counter() - t_start > time_budget_s:
                        out_of_time = True
    return results


def write_csv(results: List[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CellResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)


def _max_product_at(results: List[CellResult], dim: int, V: int, source: str, key: str, thresh: float) -> int:
    best = 0
    for r in results:
        if r.dim == dim and r.V == V and r.source == source and not r.skipped:
            val = getattr(r, key)
            if val == val and val >= thresh:  # val==val filters NaN
                best = max(best, r.product)
    return best


def _at_capacity(results: List[CellResult], dim: int, V: int, source: str, product: int) -> Optional[CellResult]:
    cands = [r for r in results if r.dim == dim and r.V == V and r.source == source and r.product == product and not r.skipped]
    return cands[0] if cands else None


def write_markdown(results: List[CellResult], runtime_s: float, path: Path) -> None:
    lines = []
    lines.append("# Memory capacity curve (M57d)\n")
    n_skipped = sum(1 for r in results if r.skipped)
    lines.append(
        "Deterministic probe (no training) of `nsm_ct.entity_memory`'s order-3 "
        "`[d,d,d]` entity(x)relation(x)value tensor: how many (instance, attribute-"
        "relation) facts fit in ONE memory before recall breaks, using genuinely "
        "sequential `nsm_ct.entity_memory.write` calls (gate=1 overwrite) and the "
        "matched-filter `query`/`query_entity` reads the reactor uses. "
        f"Grid: dim in {DIMS}, n_instances in {N_INSTANCES_GRID}, n_relations in "
        f"{N_RELATIONS_GRID}, codebook size V in {V_GRID}, value source in {SOURCES}, "
        f"{len(SEEDS)} seeds each ({len(results)} cells, {n_skipped} pruned/skipped). "
        f"Full grid: `runs/capacity_curve.csv` (gitignored). Runtime: {runtime_s:.1f}s.\n"
    )
    lines.append(
        "| dim | V | source | max facts @fwd>=0.99 | @fwd>=0.95 | max facts @inv>=0.99 | @inv>=0.95 "
        "| overwrite new-value acc @fwd-0.95 cap | stale cosine @fwd-0.95 cap |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for dim in DIMS:
        for V in V_GRID:
            for source in SOURCES:
                f99 = _max_product_at(results, dim, V, source, "forward_acc", 0.99)
                f95 = _max_product_at(results, dim, V, source, "forward_acc", 0.95)
                i99 = _max_product_at(results, dim, V, source, "inverse_acc", 0.99)
                i95 = _max_product_at(results, dim, V, source, "inverse_acc", 0.95)
                cap = _at_capacity(results, dim, V, source, f95) if f95 else None
                ow_acc = f"{cap.overwrite_new_acc:.2f}" if cap else "n/a"
                ow_stale = f"{cap.overwrite_stale_cosine:.3f}" if cap else "n/a"
                lines.append(
                    f"| {dim} | {V} | {source} | {f99} | {f95} | {i99} | {i95} | {ow_acc} | {ow_stale} |"
                )
    lines.append("")

    lines.append("## So what\n")
    fact_ceilings = {dim: _max_product_at(results, dim, 32, "codec", "forward_acc", 0.95) for dim in DIMS}
    ceiling_str = ", ".join(f"dim={d}: {c} facts" for d, c in fact_ceilings.items())
    smallest_ok_48 = next((d for d in DIMS if fact_ceilings[d] >= 48), None)
    smallest_ok_passage = next((d for d in DIMS if fact_ceilings[d] >= 150), None)
    lines.append(
        f"Fact-count ceiling at forward recall >= 0.95, V=32, codec-realistic values, per dim: {ceiling_str}. "
        f"An episode with 8 entities x 6 facts (48 facts) needs dim >= "
        f"{smallest_ok_48 if smallest_ok_48 else '128 (not reached in this grid)'} by this bound. "
        f"A 50-entity passage at ~3-4 facts/entity (~150-200 facts) needs dim >= "
        f"{smallest_ok_passage if smallest_ok_passage else 'more than 128 -- outside this grid, extrapolate or shard the memory'}. "
        "Inverse recall ('who holds X?') is read off the SAME tensor via query_entity and is reported "
        "separately above because it can diverge from forward recall (see the table) -- if it ceilings "
        "lower, 'who is X?' breaks before 'what is X's value?' does at the same dim/fact-count, which "
        "matters more for resolver-style candidate generation than for plain attribute lookup."
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    t0 = time.perf_counter()
    results = run_grid()
    total_rt = time.perf_counter() - t0
    write_csv(results, RUNS_DIR / "capacity_curve.csv")
    write_markdown(results, total_rt, DEV_DIR / "CAPACITY_CURVE.md")
    n_skipped = sum(1 for r in results if r.skipped)
    print(f"{len(results)} cells ({n_skipped} skipped by pruning), runtime {total_rt:.1f}s")
    print(f"CSV: {RUNS_DIR / 'capacity_curve.csv'}")
    print(f"Markdown: {DEV_DIR / 'CAPACITY_CURVE.md'}")


if __name__ == "__main__":
    main()
