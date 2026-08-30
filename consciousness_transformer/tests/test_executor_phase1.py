"""EXECUTOR PHASE 1 tests: the bootstrap anchor
(dev/EXECUTOR_DESIGN.md Sec.1.4 -- "today's pipeline as one fixed program"),
the honesty machinery (<=1-WRITE, D1 hard/soft), trace round-tripping, the
``program_for_step`` exhaustiveness claim, and the Sec.3 compute
kill-criteria instrumentation.

Per generator (old L1-6, writeback, instance, rich, document -- single
passage), :meth:`nsm_ct.executor.Executor.run` must reproduce
``ClauseReactor.forward`` (eval mode, same seeded weights) within 1e-5 on
``answer_logits`` AND on every intermediate register the design's own test
seams expose: ``V_read`` (forward's ``_mem_read``), ``S_gate`` (forward's
``_write_trace["gate"]``), the resolved candidate index (forward's
``_write_trace["resolved_index"]``), and the final memory tensor
(forward's ``_memory``) -- the last one is what actually proves an
address-redirected write (``A_w``) landed on the same node forward()'s own
write did, since forward() has no direct ``A_w`` test seam of its own (see
``executor.py``'s module docstring finding 2 for why).
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from nsm_ct import ops, programs
from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch
from nsm_ct.curriculum2 import (
    DocumentGenerator,
    InstanceCurriculumGenerator,
    RichEpisodeGenerator,
    WriteBackCurriculumGenerator,
    generate_document_episodes,
)
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.executor import Executor
from nsm_ct.instances import InstanceRegistry
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.resolver import make_resolver
from nsm_ct.tpr import TPRCodec

DIM = 24
HIDDEN = 16


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


# ---------------------------------------------------------------------------
# Batch builders for the five generators this milestone's battery covers.
# ---------------------------------------------------------------------------
def _old_l1_6_batch():
    eps = CurriculumGenerator(max_level=6, seed=0).generate(8)
    batch = build_clause_batch(eps, None, _meaning(), _codec())
    model = ClauseReactor(dim=DIM, hidden=HIDDEN)
    return batch, model


def _writeback_batch():
    eps = WriteBackCurriculumGenerator(seed=0).generate(8)
    batch = build_clause_batch(eps, None, _meaning(), _codec())
    resolver = make_resolver("A", DIM, HIDDEN)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    return batch, model


def _instance_batch():
    eps = InstanceCurriculumGenerator(seed=1, inverse_frac=0.3).generate(10)
    batch = build_clause_batch(eps, None, _meaning(), _codec())
    resolver = make_resolver("A", DIM, HIDDEN, use_cand_feature=True, cand_feature_extra=1)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    return batch, model


def _rich_batch():
    eps = RichEpisodeGenerator(seed=2, inverse_frac=0.3).generate(10)
    batch = build_clause_batch(eps, None, _meaning(), _codec())
    resolver = make_resolver("A", DIM, HIDDEN)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    return batch, model


def _document_single_passage_batch():
    """DocumentRunner-STYLE single passage: one document's PASSAGE-0
    episodes (own ``InstanceRegistry``, per ``ltm.py``'s "Interface
    contract"), built into ONE batch, run with ``ltm=None`` (no prior
    passage exists yet for passage 0 -- see this file's module docstring
    and dev/EXECUTOR_DESIGN.md's own note that ``cand_from_ltm`` is not
    exercised by this milestone's generator battery)."""
    eps = generate_document_episodes(30, seed=3)
    docs: dict = {}
    for ep in eps:
        docs.setdefault(ep.meta["doc_id"], []).append(ep)
    passage0 = []
    for passages in docs.values():
        passages.sort(key=lambda e: e.meta["passage_index"])
        passage0.append(passages[0])
    registry = InstanceRegistry(dim=DIM, seed=0)
    batch = build_clause_batch(passage0, None, _meaning(), _codec(), document_registry=registry)
    resolver = make_resolver("A", DIM, HIDDEN, use_cand_feature=True, cand_feature_extra=2)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    return batch, model


GENERATORS = {
    "old_l1_6": _old_l1_6_batch,
    "writeback": _writeback_batch,
    "instance": _instance_batch,
    "rich": _rich_batch,
    "document_single_passage": _document_single_passage_batch,
}


# ---------------------------------------------------------------------------
# 1. The bootstrap anchor -- one test per generator.
# ---------------------------------------------------------------------------
def _assert_anchor(name, batch, model, *, tol=1e-5):
    model.eval()
    with torch.no_grad():
        fwd = model(batch, return_memory=True, return_mem_read=True, return_write_trace=True)
        got = Executor(model).run(batch, return_memory=True, return_mem_read=True,
                                   return_write_trace=True, return_registers=True)

    max_abs_logits = (got["answer_logits"] - fwd["answer_logits"]).abs().max().item()
    assert max_abs_logits < tol, f"[{name}] answer_logits diverged: max abs diff {max_abs_logits}"

    max_abs_vread = (got["_mem_read"] - fwd["_mem_read"]).abs().max().item()
    assert max_abs_vread < tol, f"[{name}] V_read (mem_read) diverged: {max_abs_vread}"

    max_abs_gate = (got["_write_trace"]["gate"] - fwd["_write_trace"]["gate"]).abs().max().item()
    assert max_abs_gate < tol, f"[{name}] S_gate diverged: {max_abs_gate}"

    assert torch.equal(got["_write_trace"]["resolved_index"], fwd["_write_trace"]["resolved_index"]), \
        f"[{name}] resolved candidate index (D_w argmax) diverged"

    max_abs_mem = (got["_memory"] - fwd["_memory"]).abs().max().item()
    assert max_abs_mem < tol, f"[{name}] final memory tensor diverged (A_w writes landed differently): {max_abs_mem}"

    print(f"[anchor:{name}] answer_logits={max_abs_logits:.2e} V_read={max_abs_vread:.2e} "
          f"S_gate={max_abs_gate:.2e} memory={max_abs_mem:.2e}")


def test_anchor_old_l1_6():
    _assert_anchor("old_l1_6", *_old_l1_6_batch())


def test_anchor_writeback():
    _assert_anchor("writeback", *_writeback_batch())


def test_anchor_instance():
    _assert_anchor("instance", *_instance_batch())


def test_anchor_rich():
    _assert_anchor("rich", *_rich_batch())


def test_anchor_document_single_passage():
    _assert_anchor("document_single_passage", *_document_single_passage_batch())


# ---------------------------------------------------------------------------
# 2. program_for_step exhaustiveness: every (row, step) of every generator
#    resolves to a real family name, never falls through.
# ---------------------------------------------------------------------------
def test_program_for_step_exhaustive_over_every_generator():
    from collections import Counter
    seen = Counter()
    for name, build in GENERATORS.items():
        batch, _model = build()
        b, T, _d = batch.entity.shape
        for t in range(T):
            fam = programs.program_for_step(batch, t)
            assert len(fam) == b, f"[{name}] t={t}: family list length {len(fam)} != batch {b}"
            for f in fam:
                assert f in programs.FAMILY_NAMES, f"[{name}] t={t}: unknown family {f!r} (fell through)"
                seen[f] += 1
    print("program_for_step family coverage across all five generators:", dict(seen))
    # plain_fact must appear (every generator has direct-addressed steps);
    # at least one candidate-bearing family must appear too (otherwise this
    # battery isn't exercising the collapse branch at all).
    assert seen["plain_fact"] > 0
    assert sum(seen[f] for f in programs.FAMILY_NAMES if f != "plain_fact") > 0


# ---------------------------------------------------------------------------
# 3. Honesty machinery: <=1-WRITE-per-clause fires on a malicious program.
# ---------------------------------------------------------------------------
def test_write_assertion_fires_on_malicious_two_write_program():
    d = 8
    model = ClauseReactor(dim=d, hidden=8)
    model.eval()
    ex = Executor(model)

    rf = ops.RegisterFile.create(1, d, {
        "A_e": "Addr", "A_r": "Addr", "V_v": "Vec", "V_read": "Vec",
        "S_gate": "Scalar", "S_owr": "Scalar", "S_neg": "Scalar",
    })
    rf.write("A_e", F.normalize(torch.randn(1, d), dim=-1), step=0)
    rf.write("A_r", F.normalize(torch.randn(1, d), dim=-1), step=0)
    rf.write("V_v", F.normalize(torch.randn(1, d), dim=-1), step=0)
    rf.write("V_read", torch.zeros(1, d), step=0)

    write_step = programs.Step("WRITE", ("M_mem", "A_e", "A_r", "V_v", "S_gate", "S_owr", "S_neg"), "M_mem", "Mem")
    malicious = [
        programs.Step("TICK", ("A_e", "A_r", "V_v", "V_read", "state"), "state", "GRU"),
        programs.Step("GATE", ("state",), "S_gate", "Scalar"),
        programs.Step("OVERWRITE", ("state",), "S_owr", "Scalar"),
        programs.Step("NEGATE", ("state", "V_v"), "S_neg", "Scalar"),
        write_step,
        write_step,   # the SECOND write -- must be rejected
    ]

    state = torch.zeros(1, model.gru.hidden_size)
    memory = torch.zeros(1, d, d, d)
    try:
        ex.execute_step_program(malicious, rf, state=state, memory=memory)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "a second WRITE in one clause's program must raise AssertionError"

    # A legitimate one-WRITE program must NOT raise.
    legit = malicious[:-1]
    rf2 = ops.RegisterFile.create(1, d, dict(rf.slot_type))
    rf2.write("A_e", rf.read("A_e"), step=0)
    rf2.write("A_r", rf.read("A_r"), step=0)
    rf2.write("V_v", rf.read("V_v"), step=0)
    rf2.write("V_read", rf.read("V_read"), step=0)
    ex.execute_step_program(legit, rf2, state=state, memory=memory)   # must not raise


# ---------------------------------------------------------------------------
# 4. D1: hard_keys=True -> Addr register == argmax; hard_keys=False raises
#    on an Addr-typed EMIT target (soft is only legal for Dist/Scalar).
# ---------------------------------------------------------------------------
def _select_emit_program():
    return [
        programs.Step("SELECT", ("P.score",), "D_w", "->Dist"),
        programs.Step("EMIT", ("D_w", "P.addr"), "A_w", "->Addr"),
    ]


def test_hard_keys_true_matches_argmax():
    d, C, b = 12, 4, 3
    model = ClauseReactor(dim=d, hidden=8)
    model.eval()
    ex = Executor(model, hard_keys=True)

    torch.manual_seed(0)
    logits = torch.randn(b, C)
    atoms = F.normalize(torch.randn(b, C, d), dim=-1)
    mask = torch.ones(b, C, dtype=torch.bool)

    rf = ops.RegisterFile.create(b, d, {"D_w": "Dist", "A_w": "Addr"})
    state = torch.zeros(b, model.gru.hidden_size)
    memory = torch.zeros(d, d, d)
    ex.execute_step_program(_select_emit_program(), rf, state=state, memory=memory,
                             candidates={"logits": logits, "mask": mask, "atoms": atoms})

    expected = atoms[torch.arange(b), logits.argmax(-1)]
    assert torch.allclose(rf.read("A_w"), expected, atol=1e-6)


def test_hard_keys_false_raises_on_addr_emit():
    d, C, b = 12, 4, 3
    model = ClauseReactor(dim=d, hidden=8)
    model.eval()
    ex = Executor(model, hard_keys=False)

    torch.manual_seed(1)
    logits = torch.randn(b, C)
    atoms = F.normalize(torch.randn(b, C, d), dim=-1)
    mask = torch.ones(b, C, dtype=torch.bool)

    rf = ops.RegisterFile.create(b, d, {"D_w": "Dist", "A_w": "Addr"})
    state = torch.zeros(b, model.gru.hidden_size)
    memory = torch.zeros(d, d, d)
    try:
        ex.execute_step_program(_select_emit_program(), rf, state=state, memory=memory,
                                 candidates={"logits": logits, "mask": mask, "atoms": atoms})
        raised = False
    except ValueError:
        raised = True
    assert raised, "hard_keys=False must raise when EMIT targets an Addr-typed register"


def test_hard_keys_false_is_fine_for_a_dist_or_scalar_target():
    """The D1 policy only forbids soft->Addr -- a soft distribution written
    to a Dist-typed register (D_w itself) must NOT raise."""
    d, C, b = 12, 4, 2
    model = ClauseReactor(dim=d, hidden=8)
    model.eval()
    ex = Executor(model, hard_keys=False)
    logits = torch.randn(b, C)
    mask = torch.ones(b, C, dtype=torch.bool)
    rf = ops.RegisterFile.create(b, d, {"D_w": "Dist"})
    state = torch.zeros(b, model.gru.hidden_size)
    memory = torch.zeros(d, d, d)
    ex.execute_step_program([programs.Step("SELECT", ("P.score",), "D_w", "->Dist")], rf,
                             state=state, memory=memory, candidates={"logits": logits, "mask": mask})
    w = rf.read("D_w")[:, :C]
    assert torch.allclose(w.sum(-1), torch.ones(b), atol=1e-5)   # a real softmax distribution, not one-hot


# ---------------------------------------------------------------------------
# 5. Trace format round-trips through JSON.
# ---------------------------------------------------------------------------
def test_trace_round_trips_through_json():
    batch, model = _instance_batch()
    model.eval()
    with torch.no_grad():
        out = Executor(model).run(batch)
    trace = out["_trace"]
    assert len(trace) > 0

    serializable = [[op, list(args), step] for (op, args, step) in trace]
    blob = json.dumps(serializable)
    back = json.loads(blob)
    restored = [(op, tuple(args), step) for op, args, step in back]
    assert restored == trace


# ---------------------------------------------------------------------------
# 6. Compute kill-criteria instrumentation (dev/EXECUTOR_DESIGN.md Sec.3):
#    per-clause wall-clock, Executor vs forward, and peak RSS.
# ---------------------------------------------------------------------------
def test_kill_criteria_wall_clock_and_rss():
    batch, model = _rich_batch()
    model.eval()
    T = batch.entity.shape[1]
    n_reps = 5

    with torch.no_grad():
        model(batch)   # warm-up (lazy caches, e.g. ops.permute's seeded perm cache)
        Executor(model).run(batch)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(batch)
        t_forward = (time.perf_counter() - t0) / n_reps

        ex = Executor(model)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            ex.run(batch)
        t_executor = (time.perf_counter() - t0) / n_reps

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    forward_per_clause = t_forward / T
    executor_per_clause = t_executor / T
    ratio = executor_per_clause / forward_per_clause if forward_per_clause > 0 else float("inf")

    print(f"[kill-criteria] forward/clause={forward_per_clause * 1e3:.4f}ms "
          f"executor/clause={executor_per_clause * 1e3:.4f}ms ratio={ratio:.2f}x "
          f"k_max={ex.k_max} peak_rss_kb before={rss_before} after={rss_after}")

    # Sec.3: per-clause wall-clock <= K_max x forward's own per-step cost.
    # A generous safety factor on top of k_max absorbs measurement noise on
    # a small CPU batch (this is instrumentation/reporting, not a tight
    # perf regression gate) -- see executor.py's finding 3 for why k_max
    # itself (ops.PATIENCE=6) is under-sized relative to the gold programs'
    # own op-loop lengths measured in this same file's Executor.
    safety_factor = 4
    assert ratio <= max(ex.k_max, 1) * safety_factor, (
        f"Executor is {ratio:.2f}x slower per clause than forward() -- exceeds "
        f"K_max({ex.k_max}) x safety_factor({safety_factor})")


# ---------------------------------------------------------------------------
# 7. programs.py sanity: program_length matches the counts cited in
#    executor.py's module docstring finding 3 (K_max/PATIENCE mismatch).
# ---------------------------------------------------------------------------
def test_program_lengths_reported():
    lengths = {name: (Executor.program_length(prog), Executor.program_length(prog, op_loop_only=False))
               for name, prog in programs.GOLD_PROGRAMS.items()}
    print("gold program lengths (op_loop_only, full):", lengths)
    for name, (loop_len, _full) in lengths.items():
        assert loop_len > 0
    # PATIENCE was raised 6->12 after this milestone's finding 3 (the dial
    # must fit the longest gold program); every gold program now fits.
    for name, (loop_len, _full) in lengths.items():
        assert loop_len <= ops.PATIENCE, (name, loop_len, ops.PATIENCE)
