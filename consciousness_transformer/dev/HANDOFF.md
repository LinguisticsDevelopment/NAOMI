# Session handoff (2026-08-30)

Branch: claude/m27-m28-cleanup (local worktree branch `m57-work` in
.claude/worktrees/repo-cleanup), all pushed. Venv:
/home/will/.claude/jobs/eb5a4538/tmp/venv (rebuild: uv venv + torch-cpu +
nltk wordnet/omw-1.4; USVS artifact via scripts/build_usvs.py).

Read in this order:
1. CLAUDE.md (repo root)      — working agreement: director/Sonnet-agent model, ops rules, M57 schema decision
2. dev/AURORA_SPRINT.md       — final section: 2026-08-30 reprioritization (clock DROPPED; depth first)
3. RESEARCH_NOTES.md tail     — M57a, M57b (PROVEN), "M57c battery #1" (read-path diagnosis)
4. dev/LTM_DESIGN_BRIEF.md    — the lead's 5 pending decisions on episodic LTM
5. dev/CAPACITY_CURVE.md      — dim48 holds ~256 facts @0.95 forward recall
6. dev/MIND_INTERFACE.md      — v2 locked design + entity-instance addendum

STATE: M57a (instances) and M57b (write-back) proven. M57c.2 (post-collapse
read at the resolved address + entity-axis inverse read) committed 7edb00d,
GATE PENDING on cloud battery #2 (routine trig_01Hwopgsf2TGfoJUEE1oWuhu;
logs land on branch m57c2-battery-logs). Also shipped since: minibatched
training (--batch-size/--threads, peak RSS), capacity curve, rich-episode
curriculum (3-8 entities, --rich-frac), provenance wired into live writes
(--audit). Next: battery with --rich-frac in the mix; LTM after the lead's
decisions; then M58 corpus converter + zero-shot prose number.

OPS (learned hard): user's machine is for unit tests/smokes ONLY — full-
scale training runs in the cloud routine (Agent isolation:"remote" silently
falls back to LOCAL — never trust it). Push the branch BEFORE any cloud
run. Cloud box = 4 cores/15GB: ≤2 arms concurrent, OMP_NUM_THREADS=2,
--batch-size 64; cloud agents must poll with foreground sleeps (wakeups
fire hours late). Earlier ops rules unchanged: smoke results never gate
validity — full-scale forced-wrong arms do; cheat baselines; held-out
atoms; commit+push per milestone with a ledger entry, win or lose.
