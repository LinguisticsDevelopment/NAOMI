# archive/ — curriculum-era code, moved 2026-11-14

244 files moved here in the 2026-11-14 refactor to a minimal codebase:
`src/nsm_ct` modules (`clause_reactor.py`, `curriculum2.py`, `executor.py`,
`ops.py`, `membrane.py`, `wsd.py`, `resolver.py`, `ltm.py`, the old decoder,
the whole `mind/` package), 90+ scripts, 80+ tests. Nothing was deleted.

- **Paths mirror the old layout.** `archive/src/nsm_ct/X.py` was
  `src/nsm_ct/X.py`; likewise `archive/scripts/` and `archive/tests/`.
- **Not importable from the live tree and not run by the default test tier.**
  Do not build on it as is.
- **Pre-refactor tree:** branch `integration-2026-11`, commit 2e18721 (the
  whole repo exactly as it was, everything runnable). Do not move or delete
  that branch.
- **Module map before the refactor:** `README_pre_refactor.md` here.
- **Rule for bringing something back:** only when the minimal loop
  (`src/naomi`) needs it, and rewritten small — copy the idea or the few
  functions, not the module. Note it in the ledger
  (`../RESEARCH_NOTES.md`) and, if it is a result the goals depend on,
  re-measure it on the live code.
- Four small cuts kept the live tree importable: `nsm_ct/__init__.py` no
  longer imports episode/wsd; `src/nsm_ct/episode.py` is slimmed to the
  `Episode` dataclass + `_NAMES` (full original at
  `archive/src/nsm_ct/episode.py`); `corpus.py` inlines `_TRANSFER_ROLE_MAP`;
  candidate types moved from `membrane.py` to `src/nsm_ct/candidates.py`.
