# Prose failure taxonomy (M58a, updated M58c)

Generated from the corpus in `data/corpus/` (the M58c "parser round" script
that regenerated this file mirrors `scripts/convert_corpus.py --stats`, plus
the new `parsed-ambiguous` bucket that script's own printer doesn't know
about yet -- see the M58c note at the bottom). Every sentence in the corpus
gets EXACTLY one outcome: `ok`, the new `parsed-ambiguous` (a variant of
"ok" -- a usable clause was extracted, but the parser's top-K hypotheses
tied within margin; see item 2 below), or one of the SEVEN failure reasons
below (`fragment-skipped` is new in M58c). This is the REAL failure map the
M58c parser round was driven by (see dev/AURORA_SPRINT.md "Week 2").

## M58c: what changed and why

Five perception-side, additive fixes (dev/AURORA_SPRINT.md M58c), driven
directly by this file's M58a histogram:

1. **Lexicon coverage** (`nsm_ct/../quantum_parser/src/parser/pos_tagger.py`):
   hyphenated compounds resolve via their head token ("pine-tree" -> "tree");
   an on-demand WordNet consult (with light de-inflection: plural/verb/
   comparative-superlative suffix stripping) recovers real vocabulary the
   static generated lexicon doesn't have an entry for -- including a
   separate, PRE-EXISTING bug in `build_parser_lexicon.py`'s consonant-
   doubling heuristic that silently mis-generates multi-syllable -en/-er
   verbs ("happened"/"listened"/"clattered"), left as-is (not fixed at the
   source) because tests/test_spanish_freeze.py pins the static English
   lexicon's exact fingerprint+word count; closed-class gaps (reflexive/
   indefinite pronouns, the `'ll`/`'d`/`'m`/`'re`/`'ve` contraction
   particles, "until", "else") were hand-added the same way the M41
   recipe's determiners/prepositions were; and a final NAME/NUM fallback
   tier treats a bare, uninflected, otherwise-uncovered alphabetic token as
   a proper name (measured: character names are the dominant real-text OOV
   class -- 9 of 11 residual unknown-word tokens in the sample passage) and
   a digit sequence as NUM.
2. **Parse ties** (`nsm_ct/corpus.py`): a sentence whose top-K hypotheses
   tie within `_AMBIGUITY_MARGIN` is no longer an automatic failure -- the
   top-1 reading is extracted normally and tagged with a
   `HypothesisCandidateSet` (the M55a membrane shape) on
   `ParsedClause.hypotheses`; `taxonomy_counts` reports this as the new
   `"parsed-ambiguous"` outcome. A companion tie-breaker (prefer fewer
   unattached tokens, then a shorter total dependency span) is applied
   LOCALLY to the already-returned top-K in `nsm_ct.corpus._rerank_topk`,
   not inside quantum_parser's own scorer -- a version tried there first
   caused a measured regression (chart-parsing prunes hypotheses with the
   same key at MULTIPLE intermediate steps, not just once at the end, so
   even a strictly-refining extra tie-break component could change which
   hypothesis survived an earlier prune); see `scorer.completeness_key`'s
   own docstring for the full account.
3. **Dative role map** (`nsm_ct/clause.py`): "to" after a TRANSFER verb
   (give/hand/pass, ...) whose object `is_entity` now resolves to
   RECIPIENT, not PLACE ("gave the ball to john" == "gave john the ball" in
   extracted roles). Scoped narrowly (both conditions required) so a
   genuine locative "to" is never touched.
4. **Fragments/quotations** (`nsm_ct/corpus.py`): quotation-mark tokens are
   stripped before the parse call (several "no-parse" failures were
   otherwise-complete clauses derailed by a stray leftover quote mark); a
   verbless fragment (no content token has any plausible VERB/AUX reading)
   gets the new `"fragment-skipped"` code instead of `"no-parse"`; when a
   sentence still yields nothing, the quoted SPAN inside it (if any) is
   tried on its own, tagged `ParsedClause.source="quoted"` on success
   (speaker attribution is explicitly out of scope).
5. **Passage-level entity registry** (`nsm_ct/corpus.py`): `parse_passage`
   threads a `_PassageRegistry` (most-recently-mentioned entity name)
   across a passage's sentences, so a pronoun with a registered antecedent
   from an EARLIER sentence resolves to that name (a real fact) instead of
   being unconditionally flagged `"pronoun-unresolvable"`.

**Regression gate**: tests/test_parser_round.py's
`test_curriculum_byte_identical_regression` (plus two supporting proof
tests) confirms none of the above changes how any existing curriculum
sentence parses -- see that file's own docstring for the exact method (this
environment has no git to diff against a pre-change commit, so the test
instead proves, directly, that no new lexicon tier or the dative fix ever
activates on curriculum vocabulary/constructions, and reconstructs the one
tag that DOES change for a curriculum-pool name ("sandra") via a scoped
monkeypatch to show the reconstructed old tag parses identically).

## Before / after (M58a -> M58c)

| corpus | sentences | ok (before) | ok (after) | ok+parsed-ambiguous (after) | episodes (before) | episodes (after) |
|---|---:|---:|---:|---:|---:|---:|
| all | 206 | 34.5% | 44.7% | 63.6% | 33 | 48 |
| synthetic | 120 | 55.0% | 63.3% | 71.7% | 29 | 30 |
| real | 86 | **5.8%** | **18.6%** | **52.3%** | 4 | 18 |

Real-text `ok` alone more than TRIPLED (5.8% -> 18.6%); counting
`parsed-ambiguous` as usable for stream purposes (a real clause WAS
extracted, the parser just couldn't fully disambiguate among a tied top-K --
still a genuine fact, not a guess) real text reaches 52.3%, closing most of
the gap to synthetic graded-reader prose. `unknown-word` on real text fell
from 54.7% to 2.3% (47 -> 2 sentences) -- the single largest lever, exactly
as the M58a histogram predicted. `multiple-parses-unresolved` as a FAILURE
code is gone entirely (0.0% on both corpora): every sentence that used to
land there now either extracts as `parsed-ambiguous` or fails for a
different, more specific reason.

## Histogram (after M58c)

### all (206 sentences)

- ok: 92 (44.7%)
- parsed-ambiguous: 39 (18.9%)
- unknown-word: 2 (1.0%)
- no-parse: 1 (0.5%)
- multiple-parses-unresolved: 0 (0.0%)
- unsupported-construction: 10 (4.9%)
- pronoun-unresolvable: 36 (17.5%)
- no-relation-extracted: 24 (11.7%)
- fragment-skipped: 2 (1.0%)

### synthetic (120 sentences)

- ok: 76 (63.3%)
- parsed-ambiguous: 10 (8.3%)
- unknown-word: 0 (0.0%)
- no-parse: 0 (0.0%)
- multiple-parses-unresolved: 0 (0.0%)
- unsupported-construction: 0 (0.0%)
- pronoun-unresolvable: 15 (12.5%)
- no-relation-extracted: 19 (15.8%)
- fragment-skipped: 0 (0.0%)

### real (86 sentences)

- ok: 16 (18.6%)
- parsed-ambiguous: 29 (33.7%)
- unknown-word: 2 (2.3%)
- no-parse: 1 (1.2%)
- multiple-parses-unresolved: 0 (0.0%)
- unsupported-construction: 10 (11.6%)
- pronoun-unresolvable: 21 (24.4%)
- no-relation-extracted: 5 (5.8%)
- fragment-skipped: 2 (2.3%)

## Examples (up to 3 verbatim per class, corpus-wide)

### parsed-ambiguous (new)

- `real_gutenberg_busterbear#0`: "buster bear yawned as he lay on his comfortable bed of leaves and watched the first early morning sunbeams creeping through the green forest to chase out the black shadows ." -- 4 hyps
- `real_gutenberg_busterbear#0`: "once more he yawned , and slowly got to his feet and shook himself ." -- 4 hyps
- `real_gutenberg_busterbear#1`: "while he sat there , trying to make up his mind what would taste best , he was listening to the sounds that told of the waking of all the little people who live in the green forest ." -- 4 hyps

### unknown-word

- `real_gutenberg_busterbear#4`: "" i 'm going fishing , " said he in his deep grumbly-rumbly voice to no one in particular ." -- grumbly-rumbly
- `real_gutenberg_busterbear#6`: "said he in his deepest , most grumbly-rumbly voice ." -- grumbly-rumbly

(down from 47/86 (54.7%) to 2/86 (2.3%) on real text; the two residual
hits are BOTH the same reduplicated nonce compound "grumbly-rumbly" --
neither "grumbly" nor "rumbly" is a WordNet lemma even after hyphen-tail
lookup, an honest remaining gap, not a bug.)

### no-parse

- `real_gutenberg_busterbear#8`: "" here 's your trout , mr . otter , " said he , as little joe put his head out of water to see who had frightened him so ."

### unsupported-construction

- `real_gutenberg_busterbear#1`: "and grinned ." -- coordination
- `real_gutenberg_busterbear#6`: "" that 's a very fine looking trout ." -- quotation
- `real_gutenberg_busterbear#8`: "" come and get it . "" -- quotation

### pronoun-unresolvable

- `real_gutenberg_busterbear#0`: "then he walked over to a big pine-tree , stood up on his hind legs , reached as high up on the trunk of the tree as he could , and scratched the bark with his great claws ."
- `real_gutenberg_busterbear#0`: "after that he yawned until it seemed as if his jaws would crack , and then sat down to think what he wanted for breakfast ."
- `real_gutenberg_busterbear#3`: "and as buster listened it suddenly came to him just what he wanted for breakfast ."

(these are all subject-pronoun sentences whose intended antecedent is
several sentences back, past the registry's single-most-recent-entity
horizon, or is a possessive/embedded mention the registry doesn't track --
the registry (item 5) fixes the "earlier in the passage" case, not full
coreference.)

### no-relation-extracted

- `real_gutenberg_busterbear#5`: "why , little joe otter to be sure ."
- `real_gutenberg_busterbear#9`: "the fact is , he was afraid to ."
- `real_gutenberg_busterbear#9`: "buster did n't seem to mind ."

### fragment-skipped (new)

- `real_gutenberg_busterbear#1`: "thief ! """
- `real_gutenberg_busterbear#6`: "" woof , woof ! ""

## Remaining top-3 real-text failure classes (what the NEXT round should target)

1. **pronoun-unresolvable (21/86, 24.4%)** -- now the largest single
   failure class (it GREW in share, not in absolute terms it matters --
   see "why pronoun-unresolvable's share grew" below). The registry (item
   5) only fixes same-entity-recency; real narrative prose needs either a
   longer antecedent horizon (currently "most recent ANY entity", not
   gender/number-aware) or real coreference.
2. **no-relation-extracted (5/86, 5.8%)** -- transitive sentences with no
   PP and a non-locative verb ("buster did n't seem to mind ."); the
   clause extracts fine structurally but has no PLACE/RECIPIENT/AGENT-
   shaped fact for `_RELATION_QUESTION_TEMPLATE` to ask about. Needs either
   more question templates (e.g. a plain OBJECT/"what did X do" template)
   or is an honest ceiling of the current "queried role" table.
3. **unsupported-construction (10/86, 11.6%)** -- quotation-attribution
   sentences ("said he", "thought he") and coordination ("and grinned .")
   the grammar doesn't build a full clause for even after quote-stripping.
   Item 4's quoted-span fallback helps the SIMPLE cases (a complete quoted
   clause standing alone); these are the harder "narration wraps AROUND an
   attribution verb" shape, still open.

**Why pronoun-unresolvable's share grew even though the fix works**:
fixing `unknown-word` and `multiple-parses-unresolved` (which used to
short-circuit MANY sentences before they ever reached extraction) means far
more sentences now REACH the pronoun-resolution stage where they can be
counted. Measured directly (tests/test_parser_round.py-style A/B against a
registry-disabled run): the registry converts 1 sentence corpus-wide from
`pronoun-unresolvable` to `ok` -- a real but modest effect on THIS small
corpus, not the cause of the share increase.

## M58c methodology note (episode count, taxonomy code)

`scripts/convert_corpus.py`'s own `print_stats`/`write_taxonomy_doc`
functions iterate a fixed `FAILURE_REASONS` list and don't yet know about
the `"parsed-ambiguous"` outcome `nsm_ct.corpus.taxonomy_counts` now
emits (out of this round's file scope to touch -- it isn't one of the
files this round owns); this file was regenerated by a small script that
reproduces that logic with the new bucket included, matching its exact
histogram/example format otherwise. 48 episodes were produced this round
(up from 33) -- `make_episodes` still holds out exactly one clause per
passage, but more passages now have an eligible (PLACE/RECIPIENT/AGENT)
clause with enough distractors to ask about.
