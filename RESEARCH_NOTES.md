
### M58g RESOLVED — default parse cap hoisted; train_prose hang gone (2026-09-02)

Fix landed (8662172): quantum_parser/src/parser/data_structures.py:249
ParserConfig.max_parse_seconds default None -> 30.0. Every parse() call is
now bounded even when a caller forgets an explicit override; override still
available via config_override. eval_prose.py build_one already had a broad
`except Exception` so it drops the now-raised ParseResourceExceeded and
continues — no caller edit needed. VERIFY: tests/ suite 77 passed (no
regression from the cap); quantum_parser 63f/60p/11e are PRE-EXISTING
(identical with fix stashed out, all DSL grammar-load errors, unrelated).
PROOF: train_prose smoke (2 epochs) completed in 4.24 min instead of
hanging — 10s batch cap fired, per-episode probe dropped 5/42 culprits,
training proceeded and saved prose_smoke.pt. This closes the parser-hang
saga (3 uncapped call sites -> one default bound at the parser). Full
20-epoch prose-training run (the M58 payoff) re-fired on top of this.

### M61 — the prose-training payoff: does reading real text help it read real text (2026-09-02)

First full prose-training run on the cap-fixed pipeline (ba743ef).
train_prose.py, 20/20 epochs, held-out-DOCUMENT split (42 docs fully held
out), curriculum-frac 0.5, warm-started from frozen M60. Branch
prose-training-run1 (14cc65e, checkpoint prose_v1.pt). peak_rss 9.1GB;
build dropped 5/42 prose-eval + 22/256 training-mix over-long episodes
(the 30s default cap doing its job; no hang).

BEFORE (frozen M60, held-out docs, n=38):
  overall 0.579 | AGENT 0.500 OBJECT 0.750 PLACE 0.600 |
  synthetic 0.826 (19/23)  real 0.200 (3/15) |
  abstain 0.395  acc_when_confident 0.696 (n=23)
  curriculum retention 0.900 (instance .848 old .853 writeback 1.000)
AFTER (20 epochs, held-out docs, n=37):
  overall 0.622 | AGENT 0.550 OBJECT 0.750 PLACE 0.667 |
  synthetic 0.783 (18/23)  real 0.357 (5/14) |
  abstain 0.351  acc_when_confident 0.792 (n=24)
  curriculum retention 0.885 (instance .788 old .868 writeback 1.000)
DELTA: overall +0.043; real +0.157; synthetic -0.043;
  acc_when_confident +0.096; abstain -0.044; retention -0.015.

HONEST READ: direction is positive on real text (the point) and there is
NO catastrophic forgetting (retention held ~0.89). BUT magnitude is within
noise at this eval size: n~37-38, Wilson 95% on 0.622 is ~+/-0.15, so
+0.043 overall is NOT significant on its own; real +0.157 is just +2/~14
correct. synthetic dipped slightly (expected — capacity reallocated toward
real text). CAVEAT: BEFORE n=38 vs AFTER n=37 (wall-clock cap is
non-deterministic, one extra episode dropped on the AFTER pass) — eval
sets near-identical, not byte-identical. VERDICT: encouraging first signal,
not yet a claim. To confirm needs more held-out documents (bigger corpus)
and/or more epochs; a clean re-run with a fixed pre-parsed episode set
(so BEFORE/AFTER share an identical eval set) would remove the n-mismatch.
