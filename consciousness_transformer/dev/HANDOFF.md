# Session handoff (2026-08-11)

Branch: claude/m27-m28-cleanup @ 5863ead, all pushed. Worktree:
.claude/worktrees/repo-cleanup. Venv: /home/will/.claude/jobs/eb5a4538/tmp/venv
(rebuild if gone: uv venv + torch-cpu + nltk wordnet/omw-1.4).

Read in this order:
1. dev/AURORA_SPRINT.md      — the clock (cluster until ~Sept 15) + current priorities incl. final reprioritization section
2. RESEARCH_NOTES.md         — tail from M52 onward (M55 gate, M56b/c, M-ES1 are load-bearing)
3. dev/MIND_INTERFACE.md     — the signed-off architecture + v2 addenda (entity instances, memory-as-graph)
4. dev/TRACK_C_DESIGN.md     — op-algebra spike (parked: more-research verdict)
5. dev/NEXT_ARC_PLAN.md      — the arc map (partially superseded by AURORA_SPRINT reprioritization)

NEXT TASK: M57 robust memory schema — spec is the AURORA_SPRINT
reprioritization section + MIND_INTERFACE v2 addenda + the write-back gap
(RESEARCH_NOTES M55-era chat decision: resolver redirects VALUE not
ADDRESS today; "she is tall" must write to MARY's node; gate = no-gold
inference test). Then M58 corpus converter + zero-shot.

Ops rules that matter (learned hard): one training run at a time, nohup+
nice to runs/*.log, watchers grep the run's UNIQUE script name; agents
foreground-only, no git, smokes ≤2.5min; smoke results NEVER gate
curriculum validity — full-scale wrong-binding arms do (3 inversions on
record); every capability curriculum needs its cheat-baseline at floor;
held-out-atom tests for perfect-looking results; commit+push per
milestone with ledger entry, win or lose.
