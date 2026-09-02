# M62 — Teacher Gold-Volume Probe: STOPPED (network fetch blocked)

Date: 2026-09-02
Branch: `claude/m27-m28-cleanup` @ `3329859`, results pushed on `encoder-probe-logs`.

## Goal (as assigned)

Measure how much clean GROUNDED-TREE gold the deterministic parser can
manufacture from REAL graded (K-12) text — McGuffey's First and Second
Eclectic Readers (~60-100 sentences per grade) fetched from Project
Gutenberg — to gauge teacher-signal volume for distilling a learned
universal encoder (`dev/UNIVERSAL_ENCODER_DESIGN.md`).

## Outcome: no measurement was run

Per the task's explicit contingency ("IF network fetch is blocked/unavailable:
STOP the fetch, report that clearly, and DO NOT invent sentences"), this
probe stops at the corpus-acquisition step. **No sentences were fabricated
or substituted.** No parser measurement was performed, because there was no
real graded-text corpus to measure.

## Setup completed successfully

- `git fetch origin claude/m27-m28-cleanup && git checkout claude/m27-m28-cleanup` — HEAD `3329859` (descendant of `a8e69cf`).
- `pip install torch numpy nltk pytest requests` — ok (torch 2.14.0+cu130, numpy 2.4.6, nltk 3.10.3).
- `pip install -e .` — ok.
- `NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "nltk.download('wordnet'/'omw-1.4'/'omw-2.0')"` — ok (resolved via nltk's download infra / `raw.githubusercontent.com`, which IS reachable from this environment).
- `python scripts/build_usvs.py` — ok: `core_words=9946 axes=607 senses=117659 antonym_edges=11317 genus_edges=1546 prime_fallback_senses=1208`, fingerprint `e0daef638b640dd5`. Sanity checks passed (`sim(dog,puppy)=0.894`, `sim(hot,cold)=-0.013`, etc.).

So the pipeline itself (parser deps, USVS) is live and ready; only the
corpus step failed.

## Corpus fetch: BLOCKED by network egress policy

Attempted sources (all rejected):

| Host tried | Method | Result |
|---|---|---|
| `www.gutenberg.org` | curl CONNECT | `403` — "gateway answered 403 to CONNECT (policy denial or upstream failure)" |
| `gutenberg.org` | curl CONNECT | same `403` policy denial |
| `aleph.gutenberg.org` | curl CONNECT | same `403` policy denial |
| `www.gutenberg.org/ebooks/search/...` | WebFetch tool | `EGRESS_BLOCKED`: "Access to www.gutenberg.org is blocked by the network egress proxy." |
| `gutenberg.org/ebooks/14640` | WebFetch tool | `EGRESS_BLOCKED`: "Access to gutenberg.org is blocked by the network egress proxy." |
| `example.com`, `en.wikisource.org` | curl CONNECT | same `403` policy denial (confirms this is a general default-deny egress policy, not a Gutenberg-specific block) |

The agent-proxy status endpoint (`/__agentproxy/status`) confirms these as
`connect_rejected` / "policy denial" events, and lists the environment's
`noProxy` allowlist as: `api.anthropic.com*`, `registry.npmjs.org`, `jsr.io`/`npm.jsr.io`,
`pypi.org`/`files.pythonhosted.org`, `index.crates.io`, `proxy.golang.org`,
plus internal/cluster addresses. General web domains (Project Gutenberg
included) are not on it and are rejected by organization policy at the
proxy's CONNECT step — this is not a transient DNS/timeout failure.

One asymmetry worth noting: `raw.githubusercontent.com` (HTTP 301, i.e.
reachable) and nltk's own corpus-download path both worked, so the block is
not a blanket ban on all outbound traffic — only on the (non-allowlisted)
domains this probe needed, Project Gutenberg among them. No attempt was
made to route around this via unofficial GitHub mirrors of Gutenberg texts,
since the task's instruction was to fetch from Project Gutenberg
specifically and guessing at third-party mirror URLs was out of scope for
this probe.

## Data status

- `/tmp/grade1.txt`, `/tmp/grade2.txt` — **not created** (no sentences fetched, none fabricated).
- `dev/M62_gold_volume_probe.csv` (this branch) — **header only, zero data rows** (see file).

## Measurement: N/A

No clean-full-tree rate, cap-hit rate, parse-second percentiles,
length-distribution, failure-mode counts, or grade1-vs-grade2 delta were
computed, because step 2 (real graded-text acquisition) did not produce
input.
