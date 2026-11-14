# Decoder round-trip feasibility (tree -> text' -> tree')

2026-11-14. Prototype: `src/naomi/decoder.py`; tests: `tests/test_naomi_decoder.py` (7 pass, <1 s).
Reproduce: `cd src && python -m naomi.decoder --variant {decoder|decoder_oracle_prep|upper|orig} --n 300 --max-tok 12 --out X.jsonl`
(long set: `--n 100 --min-tok 12 --max-tok 20 --max-clauses 4`). Data: `runs/encoder_gold_v6.jsonl.gz`, tree = `lattice.trees[0]`,
seed 0 sample. No training, no git, nothing outside the three allowed files was touched.

## Verdict

**(c) The round-trip partner has to be a learned decoder.** The grammar can be run backwards only far enough to produce a
*validator* (is this token order grammar-legal and does the parser read it as the same tree?). It cannot be the *generator*:
the grammar file contains no words, the tree has lost the function words and the modifier attachments, and nothing in the
grammar prefers a natural order over any other legal one (the output is word salad that the parser happily accepts).
Details and the minimal tree/grammar additions that would make the validator role work are in sections 2, 4 and 5.

## 1. What the grammar file is and how the parser uses it

- `quantum_parser/grammars/english.json`: 52 rulesets in an `order` list, 110 rules. Pure category rewriting, **no words**.
  Rule = `anchor` category + `before[]` / `after[]` lists of adjacent categories (with `quantifier` one/many, optional
  `subtypes`), `result` category, `connections` (typed edges between `anchor`, `before[i]`, `after[i]`, `before[*]`), `consume`.
- Categories (NodeType): NOUN, VERBAL, NOMINAL, PREDICATE, CLAUSE, SPECIFIER (adverbs), DESCRIPTOR (adjectives, **and determiners
  and numerals**), MODIFIER (aux/modal/particle), COORD, SUBOORD, PREP (+ PREP_SPEC/DESC/MIX), PP_NOUN/PP_VERB/..., NIL (punct).
- Word -> category is NOT in the grammar file: it is `TAG_TO_NODE_TYPE` (enums.py) over a POS tagger whose closed-class words
  live in a hand-written Python dict (`pos_tagger.WORD_TAG_DICT`) and a generated WordNet lexicon.
  So every function word (`the`, `of`, `in`, `and`, auxiliaries) lives outside the grammar file.
- Parsing = bottom-up chart: apply rulesets in `order`, each rule combines the anchor with its *adjacent* neighbours, emits typed
  edges, consumes the neighbours; hypotheses scored, top-k kept. Edges emitted (counts of connection types over all rules):
  COORDINATION 42, MODIFICATION 33, SUBJECT 9, SPECIFICATION 8, SUBORDINATION 7, PREPOSITION 6, DESCRIPTION 2, APPOSITION 2,
  OBJECT 2, INDIRECT_OBJECT 1, COMPLEMENT 1.
- PLACE / SOURCE / AGENT / RECIPIENT / `OF` / `WITH` are not grammar relations. A PP is `PREP + NOUN` (PREPOSITION edge) attached by
  MODIFICATION; the role label is assigned afterwards in `nsm_ct/clause.py` from a hard-coded English table
  (`_PREP_RELATION`: in/on/at/inside/to/into/by/near -> PLACE, from -> SOURCE; other prepositions -> uppercased token).
- The tree the encoder is trained on is a FLATTENED view: `clause = predicate + [(relation, token)]`. Modifier roles
  (DESCRIPTION, SPECIFICATION) are listed under the clause, not under the noun they modify.

## 2. Can the grammar run backwards?

**Partly, as data.** `derive_order_table` reads the rules into 81 (head, relation, dependent, side, index) entries (69 need no extra
slot) plus a "lift" closure (NOUN serves as NOMINAL, VERBAL as PREDICATE). From it, with no language knowledge in code:

| fact derived from the rules | rule |
|---|---|
| SUBJECT dependent (NOMINAL) goes before the PREDICATE | clause1 |
| INDIRECT_OBJECT after verb at slot 0, OBJECT at slot 1 (or slot 0 if alone), COMPLEMENT (DESCRIPTOR) after | predicate1 |
| DESCRIPTION (DESCRIPTOR*) goes before its NOUN, any number | noun1 |
| SPECIFICATION: adverb before or after a VERBAL, before a DESCRIPTOR | verb1, adj1 |
| PP = a **PREP slot** then NOUN; PP attaches after VERBAL or after NOUN | normPP1, verb1, noun3 |
| subordination/coordination/relative/appositive need COORD / SUBOORD / relative pronoun / comma / dash / paren slots | rel1, appositive1... |

The table drives candidate generation: every role gets every (head, side, index) the grammar licenses; assignments that leave a
node unreachable from the predicate are dropped; same-side siblings are permuted where the grammar gives no order.

**What is missing from the tree to do this properly:**
1. **Prepositions.** The rules require a PREP token before every PP filler; the tree keeps the preposition only when the label is an
   open one (`OF`, `WITH`). PLACE/SOURCE/AGENT/RECIPIENT collapse many words into one label. The grammar names the *slot*, never the word.
2. **Modifier attachment.** DESCRIPTION/SPECIFICATION roles do not say which noun/verb they modify; PP roles do not say verb- vs
   noun-attachment. The decoder must enumerate (this is most of the candidate count).
3. **Function words in general** (determiners partly present as DESCRIPTION, auxiliaries/negation, conjunctions beyond the
   `discourse_links` coordinator, subordinators, relative pronouns, punctuation): not in the tree, not in the grammar file.
4. Nesting/subordination: SUBORDINATION edges are flattened into separate clauses; clause order and which clause embeds which is lost.
5. Tree noise: the gold trees contain parser slips (`to` as OBJECT, `will` as predicate, `thus` as SUBJECT); a faithful decoder
   inherits them.
6. Tense/number: carried by the surface form kept in the tree (`chased`), which only works while the encoder emits surface tokens.

**Coverage of surface tokens by tree nodes**, 2,000 random gold-v6 records (45,860 tokens; node = a role filler or predicate):

| POS | tokens | in tree | % |
|---|---|---|---|
| PRON | 4,536 | 3,293 | 72.6 |
| PROPN | 772 | 521 | 67.5 |
| VERB | 6,057 | 3,662 | 60.5 |
| DET | 4,687 | 2,787 | 59.5 |
| NOUN | 6,746 | 3,919 | 58.1 |
| ADJ | 3,307 | 1,539 | 46.5 |
| AUX | 2,790 | 1,261 | 45.2 |
| ADV | 2,962 | 765 | 25.8 |
| CCONJ | 2,399 | 217 | 9.0 |
| ADP | 4,636 | 168 | **3.6** |
| PUNCT | 6,229 | 0 | 0 |
| PART | 492 | 0 | 0 |
| SCONJ | 239 | 0 | 0 |
| all | 45,860 | 18,134 | 39.5 (45.8 excluding punctuation) |

Systematically absent: ADP, CCONJ, SCONJ, PART, PUNCT. Even content words are missing ~40% (second-clause material, PP-internal
chains, words the parser dropped). The "~24%" in `DECODER_MEMORY_DESIGN.md` is older/different gold; the shape is the same.
Determiners and auxiliaries are present about half the time only because DET->DESCRIPTOR and AUX-as-predicate leak into the tree.

## 3. Prototype (`src/naomi/decoder.py`)

- `derive_order_table(grammar)`: grammar JSON -> entries + lift (above). Works unchanged on `spanish.json`.
- `linearize(tree, table, pos)`: candidate token sequences from the tree's own tokens only; one `PREP` slot per PP, filled only by
  `default_slot_fill` (relation label lower-cased when it is an open PP role, else the slot stays empty). Token indices are ignored
  and role order is shuffled in the harness (`strip_positions`) so order cannot leak.
- `decode(tree, table, judge, pos)`: re-parse each candidate with the deterministic parser, keep the best edge-F1 against the input tree.
- This is **grammar-derived candidate generation + search-by-parser**, not grammar-backwards generation: the grammar bounds the
  candidates, the parser picks. Candidate caps: 60 per tree, 200 attachment assignments, permutations of same-side siblings capped at 6.
- Hand-written language patterns in the file: none (grep in the final report). The only items to flag: the two role labels
  `AGENT`/`RECIPIENT` (tree-contract names, plus the values of `nsm_ct.clause._PREP_RELATION`, imported) used to decide that a label has
  lost its preposition; POS tag names (`NOUN`, `ADP`...) in the evaluation. **"Would need to be in the grammar or learned":** which word fills a PREP/COORD/SUBOORD
  slot, determiners/aux/negation insertion, which of the legal orders is natural, `PLACE` -> preposition.

## 4. Round-trip measurement

Set S = 300 random gold-v6 records with <=2 clauses and <=12 tokens. Set L = 100 records with 13-20 tokens and <=4 clauses.
Re-parse uses the same deterministic parser (cap 1.5k hypotheses / 1 s per candidate). Edge-F1 is over (predicate, role, filler-string)
multisets pooled over clauses (positions ignored), against `trees[0]`. Content recall = share of the original's NOUN/PROPN/VERB/ADJ/ADV/NUM
tokens present in text'. BLEU = smoothed sentence BLEU-4 vs the original.

| set | arm | re-parses | edge-F1 | exact tree | content recall | BLEU-4 | s/tree (mean / median / max) |
|---|---|---|---|---|---|---|---|
| S | **decoder** (grammar candidates + parser pick) | 100% | 0.873 | **57.7%** | 0.755 | 0.245 | 0.42 / 0.02 / 10.2 |
| S | decoder + oracle prepositions (what-if, from the original) | 100% | 0.911 | 70.7% | 0.755 | 0.263 | 0.37 / 0.01 / 10.9 |
| S | control U: original, non-tree tokens deleted, original order | 99.3% | 0.811 | 52.7% | 0.755 | 0.332 | 0.10 |
| S | control 0: original sentence re-parsed (parser self-consistency) | 96.0% | 0.960 | 96.0% | 0.997 | 0.998 | 0.18 |
| L | **decoder** | 99% | 0.777 | **19.0%** | 0.609 | 0.124 | 4.1 / 0.97 / 58.8 |
| L | control U | 97% | 0.644 | 15.0% | 0.607 | 0.188 | 0.37 |
| L | control 0 | 95% | 0.950 | 95.0% | 1.000 | 1.000 | 0.62 |

(Control 0 below 100% = the 1 s re-parse cap, not a mismatch; every parse that finished matched.)

Reading the numbers:
- **Selection is circular by design**: the decoder searches for ANY legal order the parser reads as the input tree. So "exact 57.7%" means
  "a grammar-legal order exists in our candidate set that re-parses to this tree", not "this is a good sentence". It beats control U
  (0.873 vs 0.811 F1) because the original order of content words is not the only, or best, order for the parser.
- **Content-only sentences are parseable** (99%) but do not give the tree back: 52.7% exact. The missing function words block the
  round trip regardless of ordering: by relation, trees with a `PLACE` role (23% of S) re-parse exactly **0%** in both the decoder and
  control U (no preposition -> the PP becomes an OBJECT); with the oracle preposition they reach 57%; trees without PLACE: 75% (decoder).
- **Multi-clause and long trees collapse**: exact 64% (1 clause) / 37% (2) on S; 22/20/14% on L. Unknown attachments and clause order
  multiply candidates (mean 14 on S, 34.5 on L; 34/300 and 41/100 hit the 60 cap, so the true best may be outside the set).
- **Cost** is parser-bound and grows fast with length: median ~0.02 s on S but up to 59 s for a 20-token tree (the parser, not generation).
- BLEU/recall are low as expected (output is 4.9 tokens on average vs 8.4 original on S; 8.7 vs 16.3 on L).
- Parser-agreement is not fluency: the grammar is permissive, so first-found legal orders are often salad (examples 3, 4, 6).

### 8 examples (original -> tree -> text' -> tree')

1. Good (oracle PREP): `his conductor glanced at him scornfully .` tree `glanced: SUBJECT conductor, PLACE him, DESCRIPTION his, SPECIFICATION scornfully`
   -> `his conductor glanced at him scornfully` -> identical tree'. Without the preposition: `his conductor glanced scornfully him` -> `him` becomes OBJECT (F1 0.8).
2. Good (open label): `it was really splendid -- beyond description splendid .` tree `was: SUBJECT it, BEYOND description, COMPLEMENT splendid`
   -> `was it beyond description splendid` -> identical tree' (the `BEYOND` label supplied the word; `was it` order is not English).
3. Exact but salad: `said he , as he came out into the sunshine .` -> `the he came into sunshine out he said` (oracle PREP) -> tree' identical to the tree.
4. Exact but salad: `" i am the happy prince .` tree `am: SUBJECT i, DESCRIPTION the, DESCRIPTION happy, OBJECT prince` -> `am happy the i prince` -> tree' identical.
5. Good, simple: `then there would n't be any secret .` -> `be any then there secret` (exact; negation/modal dropped, order unnatural).
6. Bad, missing preposition: `at dinner everyone laughed but the foxy fellow .` tree `laughed: SUBJECT everyone, PLACE dinner` -> `everyone laughed dinner` -> tree' has `OBJECT dinner` (F1 0.67).
7. Bad, flattened coordination: `so anne went home and told the queen .` tree `went: anne, home` + `told: anne` -> `anne told AND anne home went` -> tree' has spurious `PLACE told`, `PLACE went` edges (F1 0.71); the coordinator comes from `discourse_links`.
8. Bad, noisy gold: `and how odd the directions will look !` tree predicate `will`, `OBJECT look` -> `the odd directions look will` -> parser reads predicate `look`, `OBJECT will` (F1 0.0). Long case (L): `she was thinking much of her lamb , and more of her father .` -> `was she thinking of lamb her more father`, F1 0.86, 12 s.

## 5. What a learned decoder needs instead

- **Input**: the clause tree with USVS-grounded fillers (sense candidates, surface token or lemma+features), role labels, clause links, plus
  grammar features it may use as data: per-node category, derived (side, index) constraints, list of required slots (PREP/COORD/SUBOORD).
- **Output**: a token sequence including function words and punctuation, generated by copy-from-tree pointers + a small output
  vocabulary; no free open vocabulary (the archived no-confab discipline stays).
- **Training data now**: 22,990 gold v6 (sentence, tree) pairs; any further text through the round trip (text -> encoder/parser -> tree -> decoder).
  Target is the *sentence*, but the loss should be partly on the *re-parsed tree* (compare trees, not wording) so the 60% of tokens
  with no tree node are not memorised.
- **Independent validation of the middle tree** (no checkpoint-shared code): (1) grammar legality: the derived order/slot table accepts the
  decoded sentence's tree; (2) grounding: every tree node maps to a token actually present in the source (and in text'); (3) coverage:
  content-token recall of the source vs tree; (4) agreement with the deterministic parser wherever it parses (it reaches 95-96% of
  sentences under a 1 s cap); (5) held-out words/atoms: perturb a content word and require the decoded text to carry it.
- **Main risk**: encoder and decoder co-adapt a private code (the tree carries order/hidden information in role order, extra nodes or
  index leakage, and the decoder reads it). Blocks: tree fields restricted to grammar-legal (role, token) edges with no order or index
  (the harness already shuffles roles and drops indices); middle tree must pass the parser/grammar judge in (4) so it stays in the
  parser's public code; grounding check (2) prevents invented nodes; held-out atoms (5) detect a memorised mapping.
- **Archived attempts (why they failed)**: Phase-1 `archive/src/nsm_ct/decoder.py` was a lookup/template realizer (sense->lemma, handle->name,
  membrane templates), no generation, only short answers. `decoder_trained.py` trained exact-text reconstruction with a closed function-word
  vocab mined from uncovered tokens: 2,047 content words got in, train token-acc 0.79 vs dev 0.16 (memorised), confabulated.
  COLAB RUN 1: reconstruction from gold tree exact 0.000, token-F1 0.360; round trip 0.323: the decoder was the bottleneck.
  Root cause per `DECODER_MEMORY_DESIGN.md`: trees cover ~24% of tokens, so exact reconstruction pushed memorisation. The v2 plan
  (retrieval over dialectic memory) went on hold 2026-10-16.

## 6. Spanish

`spanish.json` is byte-identical in rules to `english.json` (cloned; its metadata says so). The same `derive_order_table` runs on it unchanged
(same 81 entries), so the *format* transfers; the *content* does not: it still encodes pre-nominal adjectives (`DESCRIPTOR before NOUN`),
no post-nominal rule, no pro-drop, no clitics, no wh-inversion. A Spanish decoder would need those grammar additions first.

## Minimal additions that would make the grammar-as-validator useful (option b, for the validator role only)

1. Keep the preposition token in the tree (a PREP node or label = preposition word; stop collapsing to PLACE).
2. Keep each modifier's head (DESCRIPTION/SPECIFICATION/PP carry a head index) so attachment is not enumerated.
3. Add coordinator/subordinator tokens and clause nesting (not only `discourse_links` for coordination).
4. A grammar-side notion of preferred order (rule scores or a learned prior) so legality is not the only criterion.
5. Fix the parser's tree noise (`to` as OBJECT, modal as predicate) or the round trip validates garbage.
