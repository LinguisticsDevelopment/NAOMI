# NAOMI — working agreement

## Orchestration model (user directive, 2026-08-11)
- The main session (Fable/Opus) is the DIRECTOR: makes design decisions,
  writes specs and prompts, coordinates agents, reviews results, handles
  merges, commits, and pushes. It should MINIMIZE the code it writes
  itself — token economy.
- Implementation work goes to **Sonnet-model agents** (`model: "sonnet"`),
  in parallel when tasks are independent, single when not.
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
- Commit + push per milestone with a RESEARCH_NOTES ledger entry, win or
  lose.

## M57 memory-schema decision (2026-08-11)
- Attribute facts are ordinary writes INTO the entity⊗relation⊗value
  tensor memory: `e_instance ⊗ attr:<name> ⊗ value`. Inverse queries and
  resolver candidate evidence come from the same store via the existing
  query ops.
- The instance registry is an INDEX only (minted instance atoms, id →
  vector) so candidate generation can enumerate instances. No knowledge
  lives in it.
- Provenance (source, language, timestamp, trust) is a membrane-side
  append-only log, one record per gated write — it cannot live in the
  tensor (superposition destroys the audit trail).

## Pointers
- Session handoff: `consciousness_transformer/dev/HANDOFF.md` (read
  order for the design docs is in there).
- Sprint clock + priorities: `consciousness_transformer/dev/AURORA_SPRINT.md`.
- Venv recipe if missing: `uv venv` + torch-cpu + nltk wordnet/omw-1.4.
