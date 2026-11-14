# USVS function-word audit (read-only, 2026-11-14)

Scope: audit of the current USVS (`data/usvs/`, 607 axes, 9,946 core words, 117,659 senses) and the NSM resolver / encoder
feature pipeline against the owner's design direction (function words are a closed class *in* USVS; negation, time, space,
known-ness, question, possession, quantity, connectors, person are dimensions). No repo file was modified; scratch scripts
are in `/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_audit/` (`lib.py`, `a1..a7.py`).
Closed-class word lists were read at run time from `quantum_parser/src/parser/pos_tagger.py` (`WORD_TAG_DICT` 294 words,
`WORD_SUBTYPES`); nothing was copied into the repo.

Headline: **26 of the 65 prime axes are completely empty** (no sense and no core word has weight on them), including NOT, I, YOU,
THIS, THE_SAME, SOME, BEFORE, AFTER, ABOVE, BELOW, NEAR, IF, BECAUSE, MAYBE, CAN, MINE, DO, VERY, MORE. A prime-exponent word
(`not`, `i`, `above`, `before`, `if`...) therefore has **no** pure prime point in USVS; the only "pure prime" is the leaf label
returned by `NSMMeaningResolver`, outside the vector space. The vector a closed-class word gets in the encoder pipeline is, when it exists at all, a
WordNet noun/adverb that merely shares its spelling (`i` -> iodine, `he` -> helium, `am` -> americium, `at` -> astatine,
`a` -> angstrom, `to` -> "tho", `me` -> Maine, `it` -> information_technology, `who` -> World Health Organization, `us` -> United States).

## 1. Axis inventory

| kind | count | note |
|---|---|---|
| prime (`NSM`, first 65 axes) | 65 | order = `nsm_primes._PRIME_TABLE`; 26 never loaded (below) |
| `attr:*` (WordNet attribute/domain/topic nouns) | 498 | 21 have 0 senses, 42 have <=50; includes topic axes (baseball, golf, chess...) |
| `lex:*` (WordNet lexnames) | 44 | |
| total | 607 | mean 23.7 non-zero axes per sense (max 30); core words are dense (median 560 non-zero axes) |

`nsm_primes.py` DOES list allolexes/exponents (`"I~ME"`, `"SOMETHING~THING"`, `"OTHER~ELSE~ANOTHER"`, `"MUCH~MANY"`, `"WHEN~TIME"`,
`"LIKE~AS~WAY"`, `"(IS) MINE~HAVE"`...). `meaning._PRIME_EXPONENT_MAP` flattens them to 74 English single-form keys
(`i, me, you, this, not, because, if, when, where, now, before, after, above, below, near, inside, here, one, two, some, all, many, few, much, little, very, more, can, have, do, be, like, else, another, other...`). Notably absent from it: `that`, `these`, `those`, `he/she/it/we/they`, `the`, `a`, `no`, `n't`, `nobody`, `nothing`, `never`, `without`, `and/or/but/so`, `to/from/of/in/on/with`.

Prime axis load over the 117,659 senses (sense count; core words in parentheses where relevant):

| owner dimension | candidate axes (sense count) | status |
|---|---|---|
| negation | NOT (**0**; 0 core) | exists, EMPTY |
| time | NOW 477, WHEN 97,011, MOMENT 44, BEFORE **0**, AFTER **0**, A_LONG_TIME/A_SHORT_TIME/FOR_SOME_TIME **0**; `lex:noun.time` | partial; no tense |
| space | INSIDE 5,673, HERE 5,623, WHERE 79,773, SIDE 5,478, FAR 653, TOUCH 1,804; ABOVE/BELOW/NEAR/BE_SOMEWHERE **0** | partial; no source/goal (to/from), no out/outside |
| known-ness (the/a) | THIS **0**, THE_SAME **0**, OTHER 13,075 | missing (THIS/THE_SAME empty; nothing for known vs new) |
| unknown / question | none; closest = KNOW 3,640 (+NOT which is empty) | missing |
| possession | MINE **0**, PART 114,370 (97% of senses: non-discriminative), `lex:verb.possession`, `lex:noun.possession` | prime empty; WordNet lexname partial |
| quantity | ONE 114,588 (97%), TWO 42,479, SOME **0**, ALL 33,350, MUCH_MANY 50,759, LITTLE_FEW 9,669 | partial; no number scale |
| logical connectors | IF **0**, BECAUSE **0**, MAYBE **0**, CAN **0**; no AND/OR/BUT/SO/although axis | empty / missing |
| person | I **0**, YOU **0**, SOMEONE 57,569, PEOPLE 89,472, SOMETHING 115,764 (98%) | I/YOU empty |

## 2. Closed-class vocabulary: what the system gives each word

Method per word: (a) is it a prime exponent in `_PRIME_EXPONENT_MAP`; (b) USVS point = `word_coord` (core) and/or sense vectors from
`USVS.senses_of_surface(word, pos_hint)` (the call the gold builder/encoder candidate path uses; for function tags it looks up the raw surface only,
never morphy); (c) what `NSMMeaningResolver.resolve(word)` returns (prime leaf, or SOMETHING fallback) and what `encoder_model.build_features`
feeds (mean of `sense_dense` of the record's `token_sense_candidates`; zero vector when there are none).
Groups come from the parser tags (WORD_TAG_DICT) plus its `RELATIVE` (wh) and `NEGATIVE` subtypes; NUM is `QUANTIFIER` subtype numerals plus hundred/thousand/first/second/many/few
(no POS tag in the dict). WH, NEG, and ADV(closed) overlap other rows (where/when/why/how are ADV; who/what/which PRON; no/neither DET; not/n't PART).

| tag | n | prime exponent (resolver gives a pure prime leaf) | ...of those, USVS vector loads its own axis >=0.3 | non-exponent, WordNet-derived (noisy) USVS point | non-exponent, NO USVS point (zero vector) | resolver falls back to SOMETHING (non-exponent) |
|---|---|---|---|---|---|---|
| ADP | 39 | 6 (15%) | 1 (inside) | 19 (49%) | 14 (36%) | 33 (85%) |
| DET | 15 | 1 (7%) | 0 | 6 (40%) | 8 (53%) | 14 (93%) |
| PRON | 36 | 4 (11%) | 1 (someone) | 7 (19%) | 25 (69%) | 31 (86%) |
| AUX | 29 | 4 (14%) | 0 | 8 (28%) | 17 (59%) | 25 (86%) |
| PART | 3 | 1 (33%) | 0 | 0 | 2 (67%) | 2 (67%) |
| CCONJ | 5 | 0 | 0 | 4 (80%) | 1 (20%) | 5 (100%) |
| SCONJ | 9 | 2 (22%) | 0 | 2 (22%) | 5 (56%) | 7 (78%) |
| (7 tags above, total) | 136 | 18 (13%) | 2 | 46 (34%) | 72 (53%) | 118 |
| ADV (closed list) | 23 | 6 (26%) | 2 (here, now) | 16 (70%) | 1 (4%) | 17 (74%) |
| WH (who whom whose which what where when why how) | 9 | 2 (22%) | 0 | 2 (22%) | 5 (56%) | 7 (78%) |
| NEG (not n't never no nothing nobody none neither nor without cannot hardly rarely seldom) | 14 | 1 (7%) | 0 | 9 (64%) | 4 (29%) | 12 (86%) |
| NUM | 16 | 4 (25%) | 3 (one, two, many) | 12 (75%) | 0 | 12 (75%) |

Reading: **0 of 136** closed-class words (7 UPOS tags) has a pure or near-pure prime point in USVS. "Noisy WordNet-derived" means the point exists but is
dominated by `SOMETHING`/`lex:*`/`PART` noise from an unrelated homograph sense. The 13% "prime exponent" are pure only as resolver tree labels; for 16 of 18 of them the USVS vector
has ~0 on the very prime they name (e.g. `above` -> ABOVE 0.0, `not` -> NOT 0.0, `i` -> I 0.0, `before` -> BEFORE 0.0). Of 136, 72 are zero vectors in the encoder features
(`the, from, with, of, into, during, without, these, those, your, his, their, she, they, we, him, them, is, was, were, been, has, had, could, should, would, and, because, if, although, unless, since, until, whether, n't, 's`...).
Resolver detail: `nobody` -> SOMEONE (negation lost), `nothing` -> SOMETHING, `never/no/none/without/cannot` -> SOMETHING, `have` -> MINE, `be` -> BE_SOMEONE_SOMETHING, all pronouns other than i/me/you -> SOMETHING (the `PRON` context hint only helps a word with no senses, and WordNet senses exist for `he`, `it`, `who`).

## 3. Negation

NOT axis value: **0.0 for every sense and every core word in USVS** (0 of 117,659 senses, 0 of 9,946 core words).

| word | NOT | what it carries instead (top axes) | resolver |
|---|---|---|---|
| not | 0 | only `not.r.01`: SOMETHING 1.0, lex:adv.all 1.0, lex:noun.communication .94, ONE .62 | NOT leaf |
| n't | 0 | no point | SOMETHING |
| never | 0 | core, 2 axes: WHEN .3, lex:adv.all .3 | SOMETHING |
| no | 0 | `no.n.01` etc.: SOMETHING, lex:noun.communication, attr:medicine/mathematics | SOMETHING |
| nothing | 0 | core: lex:noun.quantity .89, WHERE .84, MUCH_MANY .81, HERE .78 | SOMETHING |
| nobody | 0 | `cipher.n.04`: ONE, lex:noun.person, lex:noun.Tops | SOMEONE |
| none | 0 | lex:noun.time, WHEN, SOMETHING | SOMETHING |
| neither | 0 | lex:adj.all .72, ONE .44, LITTLE_FEW .33 | SOMETHING |
| nor, without, cannot | 0 | no point at all | SOMETHING |

Morphological negatives (34 WordNet-style pairs checked: unhappy/happy, impossible/possible, dislike/like, disappear/appear, unknown/known, nonsense/sense, careless/careful, unable/able, unfair/fair,
inactive/active, disagree/agree, dishonest/honest, unkind/kind, impolite/polite, illegal/legal, irregular/regular, nonviolent/violent, useless/useful, hopeless/hopeful, harmless/harmful, unsafe/safe, incorrect/correct,
indirect/direct, discontinue/continue, disconnect/connect, unlock/lock, inaccurate/accurate, unequal/equal, immoral/moral, unlike/like, endless/end, unusual/usual, uncertain/certain, unclear/clear):
**NOT filled for the negative member in 0/34 (0%)**; positive partner empty 34/34 (trivially). 12 of the 34 negative forms are not even core words (unhappy, disagree, unkind, impolite, nonviolent, hopeless, unsafe, discontinue, disconnect, unlock, inaccurate, immoral).
Over all 116 antonym edges in the USVS meta that are exactly "negative affix + base" (un/in/im/il/ir/dis/non-, -less), NOT is nonzero for 0.

How antonyms are separated today (not by NOT): the meta holds 11,317 antonym edges (702 direct, 2,681 satellite_head, 7,934 satellite_satellite). Placement signs the **attr** axes by a gloss
polarity (`ground/polarity.py::gloss_polarity`, a fixed English cue-word list `_POS_CUES/_NEG_CUES` in code; 220 attr axes carry negative values, 6,670 core words have a negative entry). Result: antonym core pairs
are *closer* than random (mean cosine 0.47 vs 0.28 for random core pairs; the 116 morphological pairs average 0.51, median 0.51). In 41/116 (35%) the largest differing axis is a signed `attr:` axis
(e.g. impossible vs possible: attr:possibleness -0.69; unfair/fair: attr:fairness -0.71); otherwise it differs on unrelated lexname/prime noise (unhappy/happy cos 0.89 with zero sign-opposed axes; harmless/harmful cos 0.96).
`negation_base` (affix stripping that flips pole axes) exists in `polarity.py` but is called only inside that module's polarity-coordinate path, not in the built USVS.
Pole pairs (GOOD/BAD, BIG/SMALL, MUCH_MANY/LITTLE_FEW) are separate unsigned axes in the 607-axis space.

In the gold trees (2,000 records of `runs/encoder_gold_v6.jsonl.gz`): `not` 245 tokens and `n't` 62 tokens appear as tree nodes **0** times; negation survives only as a discourse link (`coordinator: NOT, prime: NOT`) 128 times
(AND 349, OR->MAYBE 41, BUT 14). `no` is a node 45/110, `nothing` 29/41, `never` 31/46, `nobody` 12/14, `without` 0/33.

## 4. Spot-checks (weights in USVS; "mean" = encoder-pipeline vector, mean over all raw senses)

- **Tense/time.** No tense axis anywhere; `ate` -> 2 senses (food/person nouns, WHEN 0.0), `eat` core NOW 0.0 / WHEN 0.06, `will` -> `will.n/v` (SOMETHING .91, WHEN .17); gold records carry no tense field (only the D6 `predicate_token_index` note
  that a carrier like `did` "keeps tense/polarity"). `yesterday` WHEN .91 / NOW .10, `tomorrow` WHEN .74 / NOW .04 (cosine yesterday/tomorrow 0.96: not distinguished), `before` WHEN .57 (BEFORE 0.0), `after` WHEN .49 (AFTER 0.0), `now` core NOW .43, `later/ago/soon/then` WHEN .58-.77. BEFORE/AFTER are empty for every word, so before/after cosine is 0.76 only through shared noise.
- **Prepositions.** `in` INSIDE .19 / WHERE .23 / PART .30 (mean over 7 senses incl. `inch`); `out` INSIDE .01; cosine in/out 0.88, in/on 0.86, above/below 0.82. `inside` core INSIDE .46 (the only clean one). `on, under, above, below, near, at, by` have WHERE/PART/MOVE ~0.1-0.4, no ABOVE/BELOW/NEAR weight.
  `from, with, of, into, onto, upon, among, during, toward(s), against, beside` have **no sense at all** (zero vector); `to` -> `tho.n.01` = PART 1.0. So to vs from and in vs out are not distinguishable by an axis.
- **Determiners.** `the` zero vector; `a` -> angstrom (ONE .36); `this` zero vector (THIS exponent, axis empty); `that/these/those` zero; `some` ALL .03/ONE .17 (SOME empty); `all` core ALL .49; `every` core ALL .64; `each` ALL .48/ONE .66; `no` ONE .29; `any` ALL .49.
- **Wh-words.** `who` -> World Health Org. (SOMETHING 1.0, PEOPLE .61); `what, how, which, whom, whose` zero; `where/when` zero point (resolver gives WHERE/WHEN leaf); `why` -> `why.n.01` (lex:noun.motive). No axis can mean "not known"; the nearest pieces (SOMEONE/SOMETHING/WHERE/WHEN + KNOW + NOT) exist except that NOT is empty and nothing marks the question.
- **Pronouns.** `i` iodine, `he` helium, `it` information_technology, `me` Maine, `us` United States; `you, she, they, we, him, them` zero. I and YOU axes empty; no pronoun gets I/YOU/SOMEONE except `someone` (core SOMEONE .36) and `somebody`.
  The resolver gives I only for `i/me` and YOU for `you`; he/she/it/they/we/him/them -> SOMETHING.
- **Connectors.** `because, if, although, unless, since, until, whether, and` zero vectors; `but` -> `merely.r.01`, `or` -> Oregon, `so` -> `sol`, `yet` -> WHEN 1.0 (yet.r.01), `while` WHEN 1.0, `then` WHEN .77. BECAUSE/IF/MAYBE axes empty. (In gold, `and/or/but/not` live as discourse-link coordinators.)
- **Possession.** `'s, of, his, your, their, has` zero; `my` -> `mu.n.01` (PART .62); `have` -> 20 senses, `lex:verb.possession` .31, MINE 0.0; `own`/`belong` carry `lex:verb.possession` .39-.55 + PART. MINE empty.
- **Numbers.** `one` core ONE .52, `two` ONE .55 / TWO .35, `three..ten` ONE .77-.92 + MUCH_MANY .48-.74 + lex:noun.quantity, essentially one profile: cosine three/four 0.994, three/ten 0.982, four/five 0.968 (L2 distance 0.21, max single-axis difference 0.11), so **three is not distinguishable from four**. `many` MUCH_MANY .59, `few` zero core (senses ONE .64), `first/second` ONE ~.1-.3 only, `hundred` ONE .25, `several` ONE .17, `both` ONE .53 / TWO .25.

## 5. Downstream treatment in the gold trees (first 2,000 records of `runs/encoder_gold_v6.jsonl.gz`)

Gold node = a clause role (`roles[].grounding.type`) or clause predicate (`predicate_grounding.type`); types `sense | entity | reference | elision | prime`. Prime is only I/YOU (`encoder_model.PRIMES = [YOU, I, <UNK_PRIME>]`).
`node_vector` (usvs_graded): `sense` -> mean of candidate sense vectors; `prime/entity/reference/elision` -> a reserved block (type marker + hashed surface identity) with **no USVS axes at all**.

Tokens that are tree nodes, by POS (distinct token indices):

| POS | tokens | tree nodes | % |
|---|---|---|---|
| DET | 5,044 | 2,917 | 58% |
| PRON | 4,331 | 3,122 | 72% |
| AUX | 2,873 | 1,356 | 47% |
| ADP | 4,741 | 127 | 3% |
| CCONJ | 2,421 | 145 | 6% |
| SCONJ | 224 | 0 | 0% |
| PART | 457 | 0 | 0% |
| ADV | 3,531 | 919 | 26% |
| NUM | 3 | 0 | 0 |

Grounding type of those nodes (R = role node, P = predicate node):

| POS | sense | entity | reference | prime | note |
|---|---|---|---|---|---|
| DET | 611 R | 2,306 R | 0 | 0 | `the` -> entity (no USVS axes); `a` -> sense (angstrom/vitamin_a); `either` -> sense |
| PRON | 152 R | 357 R | 2,291 R | 322 R | `she/her/it` -> reference; I/YOU -> prime; `nothing` -> sense (nothing.n.01) |
| AUX | 15 R + 267 P | 19 R + 1,304 P | 0 | 0 | mostly predicate `entity` (copula/do); `will/can/might` -> verb/noun senses |
| ADP | 116 R | 11 R | 0 | 0 | the few ADP nodes are `to` -> `tho.n.01`; prepositions otherwise become role labels (OF, WITH, BEFORE, AFTER, UNDER, WITHOUT...) not nodes |
| CCONJ | 16 R + 16 P | 79 R + 35 P | 0 | 0 | `and` -> entity |
| ADV | 865 R + 10 P | 44 R | 0 | 0 | |
| SCONJ, PART, NUM | 0 | 0 | 0 | 0 | not nodes |
| (token_index null or outside tokens) | 770 | 230 | 201 | 27 | synthesized/elided slots |

`token_sense_candidates` for function tokens: ADP 2,882/4,741 tokens have an entry, AUX 694/2,873, CCONJ 744/2,421, DET 1,206/5,044, PART 245/457, PRON 1,793/4,331, SCONJ 63/224 (0 empty lists; entries exist only when a sense was found).
Net: when a function word is a node it is either a hashed identity (`entity`, no meaning axes), an unresolved slot (`reference`), I/YOU (`prime`), or a **wrong-homograph WordNet sense**; most function-word tokens are not nodes at all.
Role relations on PP/clause links (OF 371, WITH 106, FOR 91, BEFORE 18, AFTER 13, UNDER 13, WITHOUT 7...) are role labels, outside USVS.

## 6. Gap list per owner dimension

| owner dimension | status | evidence / size |
|---|---|---|
| Negation (NOT) | (ii) axis exists, nothing placed on it | NOT empty; 0/34 morphological negatives, 0/116 affix antonym edges, 0 of not/n't/never/no/nothing/nobody/none/neither/nor/without/cannot. ~14 closed-class negatives + all un-/dis-/in-/non-/-less forms (116 antonym-edge pairs found, thousands of words overall) unplaced |
| Time BEFORE/NOW/AFTER/WHEN | (ii) partial | WHEN loaded (97k senses, noise-heavy) and NOW 477; BEFORE, AFTER, A_LONG_TIME, A_SHORT_TIME, FOR_SOME_TIME empty; yesterday/tomorrow and before/after not separable. Tense (past/future of verbs, will/ate) = (iii) no axis |
| Space (INSIDE/ABOVE/BELOW/NEAR...) | (ii) partial | INSIDE/HERE/WHERE/SIDE/FAR/TOUCH loaded; ABOVE, BELOW, NEAR, BE_SOMEWHERE empty; ~35% of ADP (14/39) zero vectors, 49% noisy homographs. Source/goal (to/from/into/out), contact (on/at) = (iii) no axis |
| Known-ness (the/a, this/that) | (iii) missing | THIS and THE_SAME exist but empty (0 words); nothing encodes known vs new; the/a/this/that/these/those: 5 of 6 zero vectors, `a` = angstrom |
| Unknown / question (wh) | (iii) missing | no axis means "not known"; who/what/which/whom/whose/how zero or WHO-organization; where/when only as resolver leaves; (also NOT is empty, so "someone + not-known" cannot be composed) |
| Possession | (ii)/(iii) | MINE empty, PART 97% of senses (non-discriminative); only `lex:verb.possession` (verbs) carries signal; 's/of/his/your/their zero vectors; `have` -> 20 senses |
| Quantity (ONE/TWO/SOME/ALL/MUCH_MANY) | (ii) partial | ONE/TWO/ALL/MUCH_MANY/LITTLE_FEW loaded; SOME empty (0 words); numbers 3..10 collapse to one profile (cos three/four 0.994); no composition (3 = 1 + 2) |
| Logical connectors (IF/BECAUSE/MAYBE/CAN) | (ii) axes exist, empty; AND/OR/BUT/SO/although = (iii) no axis | 0 senses on IF, BECAUSE, MAYBE, CAN; of the 14 CCONJ/SCONJ words, 6 are zero vectors (`and, because, if, although, unless, since, until, whether` minus those with senses) and 6 are homographs or WHEN-only (`or`->Oregon, `but`->merely, `so`->sol, `yet`, `while`, `though`); exact counts in section 2 (CCONJ 1 zero, SCONJ 5 zero); gold carries and/or/but/not only as discourse links |
| Person (I/YOU/SOMEONE) | (ii) | I, YOU empty; SOMEONE only via nouns (57k); every personal pronoun has either a zero vector or a homograph vector (iodine, helium, Maine, United States...) |
| Names | n/a (as designed) | gold `entity` nodes already use hashed identity, no USVS point |
| Idioms | n/a | no mechanism audited |

Axes that look redundant / unused for the minimal-set goal:
- 26 prime axes never loaded: I, YOU, THIS, THE_SAME, SOME, DONT_WANT, DO, BE_SOMEWHERE, THERE_IS, BE_SOMEONE_SOMETHING, MINE, BEFORE, AFTER, A_LONG_TIME, A_SHORT_TIME, FOR_SOME_TIME, ABOVE, BELOW, NEAR, NOT, MAYBE, CAN, BECAUSE, IF, VERY, MORE (consistent with gloss decomposition dropping function words as stopwords; polarity.py's docstring says so for negation).
- Near-constant primes that carry almost no discriminating signal: SOMETHING 98% of senses, PART 97%, ONE 97%, BODY 84%, WHEN 82%, KIND 81%, PEOPLE 76%, WHERE 68%.
- 21 `attr:` axes with zero senses (e.g. attr:hue, attr:ease, attr:passivity, attr:certainty, attr:sureness) and 42 with <=50 senses; many near-duplicate synonym attributes (possibleness/possibility/potentiality, fairness/equity, passiveness/passivity, ease/easiness/simplicity/simpleness, hue/chromaticity, assurance/self-assurance/self-confidence/sureness).
- Topic/domain `attr:` axes (baseball, golf, chess, tennis, card_game, greek_mythology, christianity...) are irrelevant to closed-class meaning; 187 attr axes cover <=200 senses.
- Separate unsigned pole axes GOOD/BAD, BIG/SMALL, MUCH_MANY/LITTLE_FEW (polarity.py designs them as signed `EVAL/SIZE/QTY` but the built space uses the unsigned pairs).

## 7. Where a per-language closed-class lexicon could live (what exists)

1. `data/usvs/dictionary.jsonl.gz` / `usvs_senses.json.gz`: sparse `{axis: weight}` per WordNet sense (the closest existing "word -> axis weights" format), but keyed by English sense ids, built by script, no closed-class entries.
2. `meaning._PRIME_EXPONENT_MAP` (from `nsm_primes` `exponent` strings with `~` allolexes): word -> prime name, 74 English keys, one prime per word (no weights, no multi-axis words, not per-language).
3. `explications.ExplicationStore` + `data/gold_explications.json` (16 verb entries, text explication -> prime tree) and `nsm_molecules` (37 molecules, all explications None): word -> prime tree, content words only.
4. `quantum_parser/src/parser/pos_tagger.py`: `WORD_TAG_DICT`/`WORD_SUBTYPES` and `SPANISH_WORD_TAG_DICT`/`SPANISH_WORD_SUBTYPES` are already per-language closed-class tables (word -> tag/subtype), in Python rather than data files; `quantum_parser/grammars/{english,spanish}.json` hold only `metadata/order/rulesets` (no lexicon section).
5. Nothing maps a closed-class word to USVS axis weights; sketch only: one `data/lexicon/<lang>.json` of `{"word": {"AXIS": weight, ...}}` entries (e.g. `not: {NOT: 1}`, `never: {NOT: 1, WHEN: 1}`), loaded next to `usvs.npz` and consulted before the WordNet sense lookup in `senses_of_surface`/`build_features`.

## Side observations (facts, no proposals)

- Hand-written English word lists currently live in code: `polarity.py` (`_POS_CUES`, `_NEG_CUES`, `_NEG_PREFIXES`, `_NEG_SUFFIX`) and `encoder_model.py` (`_INTERJECTION_LEXSET`, `compute_fired_rules` triggers); `quantum_parser` `pos_tagger.py` holds the closed-class table.
- `USVS.senses_of_surface` skips morphy for function tags on purpose ("function words are grammar, not content"); the consequence measured here is that function words get only raw-spelling homograph senses or nothing.
