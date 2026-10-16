# NAOMI — working agreement

## Orchestration model (user directive, 2026-08-11)
- The main session (Fable/Opus) is the DIRECTOR: makes design decisions,
  writes specs and prompts, coordinates agents, reviews results, handles
  merges, commits, and pushes. It should MINIMIZE the code it writes
  itself — token economy.
- Implementation work goes to **Sonnet-model agents** (`model: "sonnet"`),
  in parallel when tasks are independent, single when not. Opus agents
  (low/medium effort) are permitted for genuinely difficult tasks, but
  with the director reviewing and giving feedback, Sonnet should almost
  always suffice — escalate only after a Sonnet attempt actually fails.
- Agents are foreground-only for training runs, never touch git, and
  smoke tests they run must be ≤2.5 min.

## Ops rules (learned hard — do not relitigate)
- ONE training run at a time; long runs via `nohup` + `nice`, logging to
  `runs/*.log`; watchers grep the run's UNIQUE script name.
- Smoke-scale results NEVER gate curriculum validity — only full-scale
  wrong-binding arms do (3 smoke→full inversions on record).
- Every capability curriculum ships with its cheat-baseline at floor.
- Perfect-looking results get held-out-atom tests before being believed.
- Curriculum design law: the gold-determinant must not itself be
  answer-predictive — it selects the READING; the answer must flow only
  through the bound choice.
- Commit + push per milestone with a ledger entry, win or lose.

## Progress tracking (user directive, 2026-10-16)
- Track progress against **`GOALS.md`** (repo root) — update the relevant
  goal's status/evidence there when a result lands. Do not introduce new
  M-numbers. Ledger entries (`consciousness_transformer/RESEARCH_NOTES.md`)
  are titled `"<Goal>: <result>"`, not by milestone number. Old M-numbers
  are still findable via `consciousness_transformer/dev/archive/MILESTONE_MAP.md`.

## Decisions log
Dated decisions (memory schema, LTM, executor D1/D2, the dropped sprint
clock, the 2026-10-16 encoder/decoder calls) are recorded in **`GOALS.md`**'s
"Decisions log" section, each filed under the goal it affects, not here.
One-line index of what's there:
- Memory schema (2026-08-11): attribute facts are ordinary tensor writes;
  the instance registry is an index only; provenance lives in an
  append-only log outside the tensor.
- LTM (2026-08-30): separate additive-read LTM tensor, STM-only writes,
  identity linking via the existing resolver, consolidation as a gated
  substate event. See `dev/LTM_DESIGN_BRIEF.md` §5.
- Executor (2026-08-30): hard selection for memory-key addresses, soft for
  distributions/scalars; the controller's self-loop carries control signals
  only, never register data. See `dev/EXECUTOR_DESIGN.md` §6.
- Sprint clock dropped (2026-08-30): Aurora deadline no longer governs
  sequencing.
- Encoder unfrozen / decoder stays on hold (2026-10-16): see `GOALS.md`
  goals 1 and 6.

## Pointers
- Current status, what's proven, what's next: `GOALS.md` (repo root).
- Setup, repo map, where to read next: `README.md` (repo root).
- Research ledger: `consciousness_transformer/RESEARCH_NOTES.md`.
- Venv recipe if missing: CPU-torch wheel index + `pip install -e .` +
  `nltk.download('wordnet'/'omw-1.4'/'omw-2.0')` — full recipe in
  `README.md`.
