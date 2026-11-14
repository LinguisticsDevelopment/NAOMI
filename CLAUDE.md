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

## Layout after the 2026-11-14 refactor (owner directive)
- Live code is small on purpose: `consciousness_transformer/src/naomi/`
  (the minimal read -> remember -> recall loop), a slim
  `consciousness_transformer/src/nsm_ct/` (reader, grounding, graph, tensor
  core), ~20 scripts, ~40 tests. Everything else is under
  `consciousness_transformer/archive/` (same relative paths). Do not build
  on archived code; bring a piece back only when the minimal loop needs it,
  rewritten small.
- `integration-2026-11` (2e18721) is the frozen pre-refactor snapshot —
  never move or delete it.
- While the work is this experimental, heavy validation is deferred (owner,
  2026-11-14): one cheat baseline per reported number is enough. The
  curriculum-era laws below (smoke/full inversions, held-out-atom tests,
  curriculum design law) apply again when learned components return.

## How to check work (owner, 2026-11-14)
- **Check the method before the machinery.** Before building, fixing or
  measuring anything, state in plain words which part of the architecture
  it is, what that part is supposed to learn versus be given, and what
  signal teaches it — and confirm that with the owner when it is not
  already settled. A passing gate on the wrong method is how the project
  ran in circles. Machinery-level validation (gates, controls, baselines)
  comes after the method is agreed, and stays light while experimental.

## Design rules (owner, 2026-11-14 — do not relitigate)
- **No hand-written language patterns in code.** No word lists, regexes or
  heuristics that encode English (or any language) in Python. Language
  rules live ONLY in the grammar files (`quantum_parser/grammars/*.json`);
  everything else is learned or language-neutral.
- **No rule-based answerer.** Answering is learned by the consciousness
  model operating on memory. The encoder only turns text into trees; it is
  never trained or asked to "comprehend".
- **Trees are graded, with reasons.** A tree gets a percentage score and a
  breakdown of what is wrong (clause dropped, token missing, role wrong);
  that breakdown is the training feedback.
- **Encoder and decoder are trained as a pair** (round trip text -> tree ->
  text), with the middle tree validated independently so the pair cannot
  drift into a private code. The decoder hold is lifted for this role only;
  speaking answers stays parked.

## Ops rules (learned hard — do not relitigate)

**Mainline.** `master` is mainline (owner, 2026-11-14) — build feature
branches off it; the director merges them back. (`integration-2026-11` was
the cleanup branch that brought `master` current; `claude/m27-m28-cleanup`
and `director-*` are historical. The director session may push: repo
`.claude/settings.json` allows normal pushes, denies force-push and branch
deletion. See `BRANCHES.md` for the
full inventory and why ~95 other branches are not where you think the
latest work is.) To propose a merge: `git merge --no-ff <branch> -m "..."`,
then resolve conflicts — `RESEARCH_NOTES.md` conflicts are almost always
"keep every entry, both sides, in date order" in the canonical
`consciousness_transformer/RESEARCH_NOTES.md`; root `RESEARCH_NOTES.md`
stays a one-line pointer, never a copy. Run the focused tests for whatever
the branch touched before committing the merge.

**Cloud-job recipe** (every cloud session reinstalls from scratch — budget
~2 min for this, it is not the 20-min reinstall it used to be if you use
the CPU wheel index):
```bash
cd consciousness_transformer
pip install --index-url https://download.pytorch.org/whl/cpu torch --extra-index-url https://pypi.org/simple numpy nltk pytest
pip install -e .
python -c "import nltk; [nltk.download(x) for x in ('wordnet','omw-1.4','omw-2.0')]"
python scripts/build_usvs.py
```

**Artifacts — force-add or lose them.** `runs/` and checkpoints are
gitignored by default. **Always `git add -f`** any checkpoint/gold/log a
run produces that a future session will need, and merge it back to
mainline before the branch is abandoned — a best checkpoint
(`v4b_all_hard_0.pt`, "THE CHECKPOINT" per its own ledger entry) was lost
exactly this way: never force-added, only its `.log`/`.curve.tsv` survived.
See `consciousness_transformer/runs/ARTIFACTS.md` for what's on mainline
and how to load it; keep that file current when you add more.

**Dispatched work — verify it landed.** Dispatched routines/agents have
gone missing (lost work, not just lost checkpoints) **twice** on record.
After every dispatched job, the director must confirm a branch actually
exists on `origin` (`git branch -r` / `git ls-remote`) before treating the
work as done — a "pushed" claim in an agent's final message is not
verification.

**Test tiers.** `pytest -m "not slow" -q` is the fast tier (~4 min on this
CPU box after the refactor) and is the default (see `pyproject.toml`'s `addopts`).
`pytest -m slow` is the full tier — quantum_parser-heavy tests
(full-corpus reparse, `test_fairytaleqa.py`, anything that builds gold over
real text) are the ones that blow the 10-min budget; run the full tier
before trusting a change that touches `clause.py`, `corpus.py`, or
`quantum_parser` broadly, not just the fast tier.

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
  substate event. See `dev/design/LTM_DESIGN_BRIEF.md` §5.
- Executor (2026-08-30): hard selection for memory-key addresses, soft for
  distributions/scalars; the controller's self-loop carries control signals
  only, never register data. See `dev/design/EXECUTOR_DESIGN.md` §6.
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
