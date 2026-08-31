
### LOFO instrument repair — balance works, pronoun inversion explained, and a control-signal information bottleneck found (BUILT + diagnosed)

Repairs: (1) inverse-frequency family weights in the trace loss —
pronoun/writeback/recall op-selection 0.000-0.440 → ~1.000 at smoke;
(2) use_cand_feature=True restored (gate #1 ran with the interaction
feature accidentally OFF), which also exposed and fixed a padding crash
in Executor.run(); (3) the pronoun "oracle < floor" inversion was NOT a
gold-program bug: M53a's placeholder value IS the gold answer, so the
floor scores high for free, and gate #1's crippled resolver sank the
oracle — with the feature restored, oracle ≥ floor. programs.py
untouched. 28 tests green.

FINDING (needed a director ruling): definite_desc_read and inverse_query
share byte-identical control-signal traces until the final step — the
D2 Harvard split (control-only selector inputs) cannot discriminate them
(op_acc pinned at 0.000 regardless of balance/scale; an information
bottleneck, not a training defect). RULING (director, D2-consistent):
the control signal may carry STRUCTURAL batch flags (is_q, inverse_mask,
cand_addr_mask, evidence-target/from_ltm presence) — perception-side
facts about clause SHAPE, the same facts the deterministic dispatcher
reads; the split forbids register CONTENTS, not structure. Extension
queued; gate #2 fires after it.
