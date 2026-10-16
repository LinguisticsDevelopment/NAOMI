# Resolver build plan (v1 membrane, M52-M55) — signed-off design: MIND_INTERFACE.md

Execution shape per phase: Sonnet agent(s) build on cheap CPU (foreground,
serial, no commits) → main session verifies independently → ONE solo niced
training run at a time (<30 min each; the box tolerates single runs, dies
under parallel load) → ledger entry → commit/push. Dials, LTM, workspace,
emit gate (v2 organs) are the NEXT plan — nothing here builds them, but
nothing here may preclude them.

## Phase 1 — M52: the model consumes what the parser produces (~half day)

Agent 1 (Sonnet): multi-arg plumbing + curriculum.
- Batch build (`clause_reactor.py`: `_context_steps`/`build_clause_batch`
  only — the model class itself should need NO change if multi-arg clauses
  unroll into consecutive (entity, role, value) steps sharing the entity).
- Curriculum: new multi-arg episode levels (give/take/put:
  AGENT/OBJECT/RECIPIENT/PLACE) in curriculum2.py, parser-verified
  templates; questions can query ANY role (queried-role plumbed per the v1
  IN table).
- Tiny smoke train (≤200 episodes, ≤2 min) to prove learnability plumbing;
  the REAL run is the main session's.
Main session: solo run — mixed old+new curriculum, dim 48/64, ~15-25 min.
Gates: L1-6 no regression; multi-arg levels clearly above floor; suites.

## Phase 2 — M53: membrane types + pronoun collapse, Tracks A & B (~1 day)

Agent 2 (Sonnet): the membrane + the data.
- Candidate-set types (candidates, priors, feature vectors) crossing
  encoder→reactor; pronoun feature vectors from USVS axes; discourse-level
  entity registry supplying candidates (STM-side entities only, per v1).
- Pronoun curriculum level with ANTI-RECENCY DESIGN: correct antecedent is
  not the most recent entity in ≥50% of episodes (feature cues + role
  constraints carry the signal); nearest-entity baseline scripted — it
  must sit near floor or the data design failed.
Agent 3 (Sonnet, after 2 reports): the two tracks.
- Track A: coref head (candidates vs STM readout). Track B: shared
  score(candidate, mem_read, state). Same interface, --track A|B flag,
  identical training script.
Main session: two solo runs (A, then B) on identical data.
Gates: pronoun accuracy >> nearest-entity floor; L1-6 + multi-arg no
regression; first A-vs-B table (accuracy, params).

## Phase 3 — M54: sense collapse joins (~half day)

Agent 4: homograph candidate sets through the same membrane (sense
handles + MFS priors); ambiguity episodes merged into the reactor
curriculum; Track A = M34 chooser head with context source swapped to
memory readout; Track B = the same shared scorer, no new code.
Main session: two solo runs. Gates: M32/M40 protocol — memory context vs
bag context (the M30 rematch, now with the context WSD always lacked);
pronoun + L1-6 no regression; A-vs-B table grows.

## Phase 4 — M55: parse-hypothesis collapse (~1 day)

Agent 5: parser top-K + margins exposed through input_encoder (structural
margin decides K>1); garden-path battery authored (low-margin sentences
whose correct reading needs a memory fact); hypothesis-stream scoring.
Track A rank head / Track B shared. Main session: two solo runs.
Gates: garden-path battery answer + chosen-parse accuracy; margins
correlate with human-hard sentences (sanity read); nothing regresses.

## Standing rules

- Agents: foreground-only, serial, no git, no training runs beyond smoke
  (≤2 min); strict file ownership per phase; main session verifies every
  gate independently before commit.
- One training run at a time, nohup+nice, durable logs in runs/.
- Every phase lands its own ledger entry, win or negative.
- A-vs-B is scored functionality → params → transfer; a B lag on one
  capability is a RESULT (evidence of a distinct mechanism), not a failure.
