# Function words in USVS: design proposal (for owner approval, 2026-11-15)

Status: PROPOSAL ONLY. Nothing is built. Everything marked **J** (judgment call) is a number or placement I chose
and the owner should review; weights of 1.0 for "the word simply is this concept" are low-risk, every fraction is J.
Evidence base: `USVS_FUNCTION_WORD_AUDIT.md`, `USVS_DESIGN_HISTORY.md` (cited "history s8" = its section 8),
`LIT_FUNCTION_WORDS_SEMANTICS.md`, `DECODER_ROUNDTRIP_FEASIBILITY.md`. Facts I re-checked against the data are marked
(checked); everything else is quoted from those reports.

Owner input folded in: option C (tree node for scope + graded dimension for degree) is confirmed. The owner also
corrected the literature report: "not unhappy" and "not happy" ARE distinguishable, because "unhappy" and "happy" already sit
at different points on the trusted existing dimensions, so the negation node applied to each gives a different result.
That is how this document treats them (B, E). One caveat found while checking it is in B-(i).

---

## A. The method in plain words

This is the **vocabulary layer of the encoder/decoder pair**: what a word-node's meaning point is, for the ~300 words per
language that are grammar (not, the, in, because, did, who, three...). Given by hand, once, as per-language data: (1) a
small set of named dimensions (mostly NSM primes that already exist but are empty, plus 7 new ones), and (2) a lexicon
file placing each closed-class word on them. Content words keep exactly the placement they have today. Learned: nothing in
this proposal; the encoder learns to emit the right nodes and the decoder to read them back, later. The tree already
exists in the parser; the change is that every word stays in it as an attached node, so scope comes from the tree and
degree/meaning comes from the node's point. **What would show it works:** (a) every closed-class word has a clean,
non-homograph point ("i" is not iodine); (b) existing synonym/antonym/random-pair scores do not move (within 0.01);
(c) a tree for each of the 8 sentences in E is produced with every non-punctuation token a node; (d) the tree-comparison
score drops, with a named reason, when a function word is dropped or swapped ("not" removed, "to" replaced by "from").
Not tested here: that a learned encoder/decoder can use it.

---

## B. The closed-class treatment

### Reconciliation with option C and the owner's position

A function word is **a tree node attached to what it modifies**; its **meaning is a point on named USVS dimensions**
(lexicon entry); nothing else special-cases it. The literature's option C has two halves:
- *Tree half* (operator node attached to its head; scope and nesting from the tree): adopted as is. It is the owner's
  "every word a node, trees nest".
- *Displacement half* (a generic rule that moves the head's point by the node's dimensions): this is the only part that
  could smell like a code path. I propose it stays **off in v1**; if ever used it is ONE rule, identical for every word
  and dimension (choice point 3). Behaviour then falls out of where the word sits, not out of code about "negation".

Where the evidence pulls against the owner (flagged, not overridden):
1. **A single NEG scalar cannot carry scope** (every student did not fail). The tree carries scope; the scalar carries degree.
2. **Not-A is mitigated, not full reversal** (Zhu 2014, Aina 2018). If "not" = 1.0 and composition were additive,
   "not happy" would equal "extremely unhappy". The proposal avoids this by never folding "not" into the head in v1.
3. **History s8.1 item 4** says do not encode antonymy as a sign flip on a shared axis. This proposal does not: the NOT
   dimension marks *explicit* negation (the word "not", affixes), never lexical antonymy (sad vs happy stays as is).

### Where each number comes from (answers "slightly unhappy ... didn't eat")

Rule of thumb: **each node's number is its own lexicon/sense entry; no number is created by composing.** A degree word is
its own node with its own point; the tree says what it attaches to.

| phrase | nodes (attachment) | numbers and their source |
|---|---|---|
| happy | happy | base point from existing USVS; NOT = 0 (unmarked) |
| unhappy | unhappy (one node) | base point from existing USVS (its own sense, existing placement). Optional extra NOT = 0.7 from the overlay (see (i)) |
| slightly unhappy | unhappy <- slightly | unhappy as above; slightly = {LITTLE_FEW 0.7} from the lexicon (J) |
| extremely unhappy | unhappy <- extremely | extremely = {VERY 1.0} (J) |
| not happy | happy <- not | happy base point; not = {NOT 1.0} from the lexicon |
| not unhappy | unhappy <- not | **unhappy's base point** + not = {NOT 1.0}. Differs from "not happy" because the head node differs, as the owner said |
| never | never (one node) | {NOT 1, WHEN 1, ALL 1}: "not at all times" (J: ALL) |
| nobody | nobody (one node) | {SOMEONE 1, SOME 1, NOT 1} |
| nothing | nothing | {SOMETHING 1, SOME 1, NOT 1} |
| did not eat | eat <- did, eat <- not | did = {BEFORE 1}; not = {NOT 1}; both attach to eat; order/scope not stored beyond that |

So the **0.2 / 1.0 gradation** in the owner's picture is: 0 = no NOT node attached and the word itself unmarked; 1.0 =
a negator word; the in-between values (0.2-0.7) appear in exactly two places: a *lexicon entry that is partially
negative* (affix overlay, 0.7) and a *degree node* (slightly 0.7 on LITTLE_FEW, very 0.4 on VERY). Nothing is stored
as "slightly negated = 0.2"; if the optional folded view (choice 3) is turned on, the rule "multiply the head's
extension dims by (1 + VERY - LITTLE_FEW), clip to [0,1]" would give slightly unhappy 0.7 x 0.3 = 0.21, unhappy 0.7,
very unhappy 0.98, extremely unhappy 1.0 (these 0.21 etc. are arithmetic on J weights, shown only as an illustration).

### Choice points (4)

1. **Where do the graded numbers live?** (a) each word's own entry + degree as separate attached nodes; (b) a modifier
   word writes a level into its head; (c) composed at read time. **Recommend (a)**: it is the only option where the
   number is data in the lexicon, nothing is inferred, and the decoder can read each word's contribution off its node.
2. **Does a content word get a NOT value for explicit negative morphology (un-, in-, dis-, non-, -less)?**
   (a) overlay from data, default 0.7; (b) no, rely on existing placement only; (c) split "un-"+"happy" into two nodes.
   **Recommend (b) for v1, (a) as a measured step-2 option** (see (i): existing placement does not separate unhappy from
   happy well). (c) rejected: it changes how content words are placed, which the owner trusts.
3. **Composition at tree-comparison time.** (a) none: compare node by node, shape differences get partial credit;
   (b) one deterministic rule on named dims (the "fold" above); (c) left to the learned models.
   **Recommend (a) in v1, (b) as a flagged experiment later**: (a) is what the owner described ("compare trees node by
   node by closeness in USVS"), needs no arithmetic and cannot hide a bug.
4. **How degree is represented.** (a) two non-negative axes, VERY (empty prime, "more intense") and LITTLE_FEW
   (populated prime, "less"); (b) one signed axis. **Recommend (a)**: history s8.1 item 8 says senses and consumers are
   non-negative, and both axes already exist.

### (i) Can NOT be added to existing senses without disturbing existing axes? From what data?

Yes mechanically. NOT is axis index 57 (checked: 0 of 117,659 senses and 0 core words have it). I would not rewrite
`usvs.npz`; I would add a small **overlay file** (sense id -> {NOT: 0.7}) merged at load, built deterministically from:
WordNet direct antonym pointers (data) AND an affix table in the lexicon's `morphology` section (data; today this
English list lives in code as `polarity._NEG_PREFIXES`). A pair is marked only when a pointer exists and the affixed form
is the other member plus an affix. Counted (checked): 948 of the ~1,834 unordered adjective antonym pairs are
affix pairs (also 211 adverb and 379 noun pairs); a random sample of 24 looked right (able/unable, just/unjust,
realistic/unrealistic...). Nothing in the 607 existing axes changes.

**Caveat from the audit (checked in the audit, not re-run by me):** the owner's premise (existing placement separates
the pair) holds for only some pairs. In 41 of 116 affix pairs the biggest difference is a signed attr axis; "unhappy" is
not even a core word, and its senses share their top axes with happy's (SOMETHING, lex:adj.all, ONE, LIVE...). I
checked that senses carry no negative values at all (history s2.1) and that the top axes of `unhappy.a.01` and `happy.a.01` overlap. So for pairs like happy/unhappy the NOT overlay
is the extra signal; where the pair is already separated it adds a consistent second signal. Hence "measure, then decide"
in choice 2.

### (ii) "not happy" (two nodes) vs "unhappy" (one node) when trees are compared

Different shapes, so with composition off they get partial credit: the two share a content-close head (happy vs
unhappy, close but not equal in USVS), and each has one unmatched piece (the "not" node vs the head's own difference).
They will NOT score as identical, and I think that is right (contrary vs contradictory, Horn/Krifka). If the owner
wants them closer, the lever is the optional fold (choice 3b) plus the NOT overlay (choice 2a), measured.

### (iii) What "composing" means at comparison time

In v1: nothing. Alignment of nodes, then per-node closeness in USVS (cosine with the existing IDF weighting) and edge
agreement. Recommended simplest fold, if the owner later wants it: for each content node, effective point = own point +
sum of the extension-dimension values of its attached function nodes, clipped to [0,1]; one rule for all words,
never stored in the tree. Learned models are free to learn better than this.

---

## C. Special dimensions

**Rule: fill existing empty prime axes first.** Existing axes are used through the lexicon (closed-class words get rows;
they are not WordNet senses, so the empty prime columns become populated by ~300 lexicon words, not by content words).

| dimension | 0 / 1 mean | graded? | used by | existing axis? |
|---|---|---|---|---|
| NOT | 0 not negated / 1 full negation operator | graded | not, n't, never, nobody, nothing, no, but (0.5) | **yes, prime NOT (empty)**; this is the owner's "dim1" |
| VERY | 0 unmarked / 1 maximal intensity | graded | very, extremely | yes, prime (empty) |
| LITTLE_FEW | 0 / 1 small amount | graded | slightly | yes, prime (populated) |
| MORE | 0 / 1 additional or greater | graded | more, again, also | yes, prime (empty) |
| THIS, THE_SAME | demonstrative / identity | graded | this, that, again | yes, primes (empty) |
| SOME, ALL, ONE, TWO, MUCH_MANY | quantity | graded | a, some, all, every, nobody, numerals | yes (SOME empty, others populated) |
| I, YOU | speaker / addressee | binary | i, you, my | yes, primes (empty) |
| MINE | belonging | graded | 's, my | yes, prime (empty); name is NSM's "(is) mine", used for any possessor |
| BEFORE, NOW, AFTER | past / present / future relative to speech time | graded | did, will, inflected verbs, before/after | yes, primes (BEFORE/AFTER empty, NOW 477 senses) |
| ABOVE, BELOW, NEAR, FAR, INSIDE, TOUCH | spatial relation | graded | on, under, with, in, that | yes (ABOVE/BELOW/NEAR empty) |
| IF, BECAUSE, MAYBE, CAN | logical | binary-ish | if, because, or, may, can | yes, primes (all empty) |
| BE_SOMEONE_SOMETHING | copula | binary | was, is | yes, prime (empty) |
| SOMEONE, SOMETHING, PEOPLE, WHERE, WHEN, LIKE, PART | person/thing/place/time/manner/part | graded | pronouns, wh-words, of | yes, populated |
| **KNOWN** (new) | 0 not marked as identifiable / 1 hearer can identify it | graded | the, this, that, he/she/it/they, my, 's | no: KNOW is a mental-verb axis (know, learn, understand) and would put "the" next to "know" |
| **UNKNOWN** (new) | 0 / 1 the speaker lacks this and asks for it | graded | who, what, where, when, why, how | no (KNOW + NOT would put "who" next to "ignorance", "unknown") |
| **GOAL** (new) | 1 end of a movement | graded | to, into | no |
| **SOURCE** (new) | 1 start of a movement | graded | from, out | no |
| **AND** (new) | 1 plain conjunction | graded | and, but | no (NSM has none) |
| **MALE**, **FEMALE** (new) | natural gender of a person | binary | he, she | no: existing `attr:gender`/`attr:sex` are WordNet attribute axes (624 / 635 senses) and are signed in core coords, not usable non-negatively |

**7 new axes** (KNOWN, UNKNOWN, GOAL, SOURCE, AND, MALE, FEMALE); 26 existing empty primes get filled; all weights >= 0.
Tense uses BEFORE/NOW/AFTER and does not need new axes. Possession does not need a new axis (MINE).

Deliberately left out: number (plural) marking on nouns, aspect (progressive/perfect), mood beyond CAN/MAYBE, focus
(only/even), politeness/evidentiality, case (the grammar's structural edge gives subject/object), grammatical gender on
articles, a separate NEW (new-referent) axis (absence of KNOWN is "not marked known"; "a" says ONE/SOME instead).
UD/UniMorph checklist (Polarity, Tense, Definite, PronType, Poss, Number, Gender): covered except Number and Aspect.

Cost note: new axes are appended after the 607; the base fingerprint `e0daef638b640dd5` is stored as the extension's
`extends` field, consumers trained at d=607 can slice `[:607]` (history s8.1 items 6, 9). I did not verify how the loader
would handle this and expect to find rough edges in step 1.

---

## D. The lexicon as data

**Location (recommend):** `quantum_parser/lexicons/<lang>.json`, next to `grammars/`. The parser needs the same closed
class table for POS/subtype, and nsm_ct already imports quantum_parser (the reverse import would be odd). Extension axis
list: `consciousness_transformer/data/usvs/ext_axes.json`; NOT overlay: `data/usvs/overlay_not.json`.

**Format sketch**
```json
{ "lang": "en", "schema": "lexicon-1", "extends_usvs": "e0daef638b640dd5",
  "entries": {
    "not":   {"pos": ["PART"], "dims": {"NOT": 1.0}},
    "never": {"pos": ["ADV"],  "dims": {"NOT": 1.0, "WHEN": 1.0, "ALL": 1.0}, "j": true},
    "when":  [{"pos": ["ADV"],   "dims": {"WHEN": 1.0, "UNKNOWN": 1.0}},
              {"pos": ["SCONJ"], "dims": {"WHEN": 1.0}}],
    "three": {"pos": ["NUM"], "kind": "name_like", "dims": {"MUCH_MANY": 0.3}, "j": true} },
  "morphology": {"prefix": {"un": {"dims": {"NOT": 0.7}}, "in": {...}, "dis": {...}, "non": {...}},
                 "suffix": {"less": {"dims": {"NOT": 0.7}}, "ed": {"pos": ["VERB"], "dims": {"BEFORE": 1.0}}},
                 "forms":  {"ate": {"lemma": "eat", "dims": {"BEFORE": 1.0}}} },
  "interjections": ["oh", "ah", "alas", "..."] }
```
Fields: `pos` (the existing tagger tags, so "can"/"will" as AUX are distinct from the nouns), `dims` (axis -> weight in
(0,1]), `kind` (word | name_like), `j` (flag for owner review). Closed-class reading chosen by (word, pos). `forms` is
for irregular inflection; WordNet's exception lists do not separate past from participle, so UniMorph-style tables
would be the data source (I did not check that one is available offline: unverified).

**Consulted before WordNet.** `USVS.senses_of_surface` already short-circuits function tags
(`_FUNCTION_POS_TAGS`) to "raw surface only". Change: for those tags, look up the lexicon first and return a lexicon
sense id `fn.<lang>.<word>.<k>`, minted exactly the way `interj.<word>.01` is today (post-fingerprint, additive, inert).
"i" tagged PRON then never reaches iodine. If the tagger says NOUN, WordNet is used as today.

**Stopwords.** `meaning._STOPWORDS` (149 English words) and `corpus.gloss_vocabulary` caused the empty axes. They keep
working in the content pipeline unchanged (no rebuild, so no change to content placement). Later they could be derived
from the lexicon's closed-class keys instead of a Python list; not needed now.

**Parser table.** `pos_tagger.WORD_TAG_DICT` (294 words), `WORD_SUBTYPES`, and the Spanish pair (293 words) become the
lexicon's `pos` (+ `subtype`) fields; the parser loads them from the JSON. Output must be tag-for-tag identical (step 3).

### English entries (the 54 words you listed; J = judgment call to review)

| word | dims | why |
|---|---|---|
| not | NOT 1 | the negator itself |
| n't | NOT 1 | clitic form of not |
| never | NOT 1, WHEN 1, ALL 1 **J** | "not at all times" |
| nobody | SOMEONE 1, SOME 1, NOT 1 | "not some someone"; negation is a filled dim, no special code |
| nothing | SOMETHING 1, SOME 1, NOT 1 | same |
| no | SOME 1, NOT 1 | determiner "not some"; answer "no" same entry |
| slightly | LITTLE_FEW 0.7 **J** | small degree |
| very | VERY 0.4 **J** | intensifier, weaker than extremely |
| extremely | VERY 1.0 **J** | maximal intensifier |
| more | MORE 1 | the prime itself |
| the | KNOWN 1 | identifiable referent |
| a | ONE 0.5, SOME 0.5 **J** | "one, some"; KNOWN absent = not marked known |
| this | THIS 1, KNOWN 0.8 **J** | demonstrative, identifiable |
| that | THIS 0.5, FAR 0.5, KNOWN 0.8 **J** | distal demonstrative via FAR; "that" as clause-introducer (SCONJ) is a separate weak entry THIS 0.3 **J** |
| some | SOME 1 | the prime |
| all | ALL 1 | the prime |
| every | ALL 1, ONE 0.5 **J** | distributive "all, one by one" |
| i | I 1 | the prime |
| you | YOU 1 | the prime |
| he | SOMEONE 1, KNOWN 1, MALE 1 | known person, male |
| she | SOMEONE 1, KNOWN 1, FEMALE 1 | known person, female |
| it | SOMETHING 1, KNOWN 1 | known thing |
| they | PEOPLE 1, KNOWN 1 **J** | plural known persons; singular "they" not covered |
| who | SOMEONE 1, UNKNOWN 1 | owner's "someone + not-known" |
| what | SOMETHING 1, UNKNOWN 1 | same |
| where | WHERE 1, UNKNOWN 1 | same |
| when (adv) | WHEN 1, UNKNOWN 1 | same |
| why | BECAUSE 1, UNKNOWN 1 | asks for the cause |
| how | LIKE 1, UNKNOWN 1 **J** | LIKE~AS~WAY: manner |
| in | INSIDE 1 | prime |
| into | INSIDE 1, GOAL 1 | movement ending inside |
| out (of) | INSIDE 1, SOURCE 1 | movement starting inside |
| on | TOUCH 0.7, ABOVE 0.5 **J** | contact, usually from above; imperfect |
| under | BELOW 1 | prime |
| to | GOAL 1 | end of movement; infinitive "to" is a different reading, not covered (G) |
| from | SOURCE 1 | start of movement |
| with | NEAR 0.6, TOUCH 0.3 **J** | weakest entry: accompaniment vs instrument not separated |
| of | PART 1 **J** | part/belonging; PART is a noisy axis (97% of senses), IDF handles it |
| 's | MINE 1 | possession as a preposition-like node |
| my | I 1, MINE 1 | "belonging to me" |
| and | AND 1 | plain conjunction |
| but | AND 1, NOT 0.5 **J** | "and, against expectation" |
| or | MAYBE 1 | NSM: disjunction is MAYBE (also the old `_COORD_PRIME`) |
| because | BECAUSE 1 | prime |
| if | IF 1 | prime |
| when (conn.) | WHEN 1 | prime, no UNKNOWN |
| did | BEFORE 1 | tense only; do-support has no content |
| will | AFTER 1 | future |
| can | CAN 1 | prime |
| may | MAYBE 1 | possibility |
| was | BE_SOMEONE_SOMETHING 1, BEFORE 1 | copula + past (not on your list; needed in E) |
| one | ONE 1 | prime |
| two | TWO 1 | prime |
| three | MUCH_MANY 0.3 **J**, kind name_like | name-like node (hashed identity, distinct from four); memory holds "three really means two and one" |
| twenty | MUCH_MANY 0.3 **J**, kind name_like | same; "twenty = two times ten" is a memory fact |

Not covered by an entry: "again" (used in E): {MORE 1, THE_SAME 0.5} **J**; "her" (object): same as she.

### Spanish (6 entries; same dimensions)

| word | dims | note |
|---|---|---|
| no | NOT 1 | |
| nunca | NOT 1, WHEN 1, ALL 1 **J** | as never |
| nadie | SOMEONE 1, SOME 1, NOT 1 | with "no": negative concord, see G |
| el / un | KNOWN 1 / ONE 0.5, SOME 0.5 **J** | grammatical gender deliberately not encoded |
| quién | SOMEONE 1, UNKNOWN 1 | |
| en | INSIDE 0.5, TOUCH 0.4 **J** | covers in/on/at; less sharp than English "in" |

---

## E. Trees

### What changes in the schema

Current contract (`naomi/types.py`): `ClauseTree{predicate, args[Filler(role, token, lemma, token_index, senses)],
negated}`, a flattened clause with role labels (PLACE, SOURCE, OF...). Tokens kept: ~40% (ADP 3.6%, CCONJ 9%, SCONJ and
PART 0%). Proposed contract:
- **Node**: `id`, `token_index` (None for a covert node), `surface`, `lemma`, `kind` in {word, name, covert},
  `lex` (lexicon key) or `senses` (candidates, memory chooses as today), `morph` (dims from inflection, e.g. BEFORE).
  `name` nodes carry no point. Punctuation is not a node.
- **Edge**: `(head_id, rel, dep_id)` where `rel` is one of the **parser's own structural edge types**, which exist
  already in the grammar: SUBJECT, OBJECT, INDIRECT_OBJECT, COMPLEMENT, DESCRIPTION (adjective/determiner/numeral ->
  noun), SPECIFICATION (adverb/degree/negator/auxiliary -> verb or adjective), MODIFICATION (PP/clause -> head),
  PREPOSITION (preposition -> its noun), SUBORDINATION (connector -> clause), COORDINATION, APPOSITION.
- **Gone**: semantic role labels (PLACE, SOURCE, RECIPIENT, AGENT, OF, WITH, QUANTITY...), `negated`, `is_hole`.
  A "hole" is simply a node whose UNKNOWN dimension is high.
- **Clauses nest**: a clause is a predicate node; a connector node has the embedded clause's predicate as its
  dependent. A clause view (`predicate + args`) can still be derived for memory.
- **Scoring** (`naomi/score.py`): edges become `(head node, rel, dep node)` over all nodes, plus a per-node closeness
  (USVS cosine, IDF-weighted, as `usvs_graded` does); diagnostics gain `function_node_dropped`, `function_node_swapped`
  (with the two words), alongside the existing kinds.

### Eight worked trees
(`X <- Y [REL]` = Y attaches to X by REL. Dims abbreviated from the lexicon in D.)

**1. she was slightly unhappy**
```
was      pred, {BE_SOMEONE_SOMETHING, BEFORE}
 <- she        [SUBJECT]       {SOMEONE, KNOWN, FEMALE}
 <- unhappy    [COMPLEMENT]    base point of unhappy (existing USVS)
     <- slightly [SPECIFICATION] {LITTLE_FEW 0.7}
```
**2. he was not happy**  (and, for contrast, **he was not unhappy**)
```
was <- he [SUBJECT]; was <- happy [COMPLEMENT]; happy <- not [SPECIFICATION] {NOT 1}
was <- he [SUBJECT]; was <- unhappy [COMPLEMENT]; unhappy <- not [SPECIFICATION] {NOT 1}
```
The two differ because the head differs: base(happy) vs base(unhappy), both existing placement, each with NOT 1
attached. This is the owner's correction and is shown here as the worked example. Residual question (G): how strongly
"not" should move a word that is already at the negative end; v1 does not move anything.

**3. nobody came to the house**
```
came  {come, BEFORE via morph}
 <- nobody [SUBJECT]       {SOMEONE, SOME, NOT}      one node: negation is its own dims
 <- to     [MODIFICATION]  {GOAL 1}
     <- house [PREPOSITION]
         <- the [DESCRIPTION] {KNOWN 1}
```
**4. the dog did not eat the bone**
```
eat
 <- dog [SUBJECT] <- the [DESCRIPTION] {KNOWN}
 <- did [SPECIFICATION] {BEFORE 1}
 <- not [SPECIFICATION] {NOT 1}       (from did + n't if contracted)
 <- bone [OBJECT] <- the [DESCRIPTION] {KNOWN}
```
Scope of "not" = the predicate eat (it attaches there). Whether it is "not(past(eat))" or "past(not(eat))" is not stored.

**5. jim's father gave her three apples**
```
gave  {give, BEFORE via morph "gave"}
 <- father [SUBJECT]
     <- 's [MODIFICATION]  {MINE 1}
         <- jim [PREPOSITION]  kind=name, no point
 <- her    [INDIRECT_OBJECT] {SOMEONE, KNOWN, FEMALE}
 <- apples [OBJECT]
     <- three [DESCRIPTION]  name_like {MUCH_MANY 0.3}; memory: "three = two and one"
```
Possession is the same shape as a preposition ("the father of jim": `of {PART}`), as the owner described.

**6. she said that he left because it rained**
```
said {say, BEFORE}
 <- she [SUBJECT]
 <- that [SUBORDINATION] {THIS 0.3}
     <- left [clause]  {leave, BEFORE}
         <- he [SUBJECT]
         <- because [SUBORDINATION] {BECAUSE 1}
             <- rained [clause] {rain, BEFORE} <- it [SUBJECT] {SOMETHING, KNOWN}
```
The "because" clause could also attach to "said"; attachment ambiguity is the parser's, not new.

**7. who ate the bone?**
```
ate {eat, BEFORE via morph}
 <- who [SUBJECT]  {SOMEONE 1, UNKNOWN 1}     this node IS the question
 <- bone [OBJECT] <- the [DESCRIPTION] {KNOWN}
```
No question label anywhere; "?" is punctuation, not a node.

**8. the king never saw the princess again**
```
saw {see, BEFORE via "saw" form}
 <- king [SUBJECT] <- the [DESCRIPTION] {KNOWN}
 <- never [SPECIFICATION] {NOT 1, WHEN 1, ALL 1}
 <- princess [OBJECT] <- the [DESCRIPTION] {KNOWN}
 <- again [SPECIFICATION] {MORE 1, THE_SAME 0.5}
```

---

## F. What this supersedes or touches

- **D3 (I/YOU as reserved non-USVS `prime` grounding, `encoder_model.PRIMES`):** superseded. i/you become lexicon
  points on the existing I/YOU axes; the `prime` grounding type and its reserved block go away.
- **D4 (QUANTITY/ADDITIVE/FOCUS structural labels, "not USVS coords"):** superseded for QUANTITY and ADDITIVE (nodes
  with MORE/AND/ALL/SOME dims); FOCUS (only/even) has no entry yet (G).
- **`_PREP_RELATION` in `nsm_ct/clause.py`** (in/on/at/to/into/by/near -> PLACE, from -> SOURCE, plus the passive "by"
  -> AGENT and dative "to" -> RECIPIENT guards): removed; the preposition stays a node and the role is its point. The
  passive-agent "by" has no entry; open.
- **`ground/polarity.py` cue lists (`_POS_CUES` 38 words, `_NEG_CUES` 42):** these are English word lists in code.
  Measured cost of REMOVING them (history s2.2, 3k corpus, held-out): syn>ant 0.693 -> 0.652 (-0.041), antonyms do not
  collapse to chance; and it needs a rebuild. **Recommend relocating them to data unchanged** (no rebuild, no cost).
  `_NEG_PREFIXES`/`negation_base` (not called by the build) move into the lexicon's affix table.
- **Interjection list `_INTERJECTION_LEXSET` (13 words) in `encoder_model.py`:** moves to the lexicon's
  `interjections`; the 20 pure interjections already have USVS ids.
- **Shared stopword list** (`meaning._STOPWORDS`): left as is for the content pipeline; no rebuild; later derivable
  from the lexicon.
- **"USVS is settled and not being revisited" (GOALS.md goal 1):** needs a wording change, not a reversal: content
  placement stays settled; function words are added as an extension (lexicon rows, 7 appended axes, optional NOT overlay).
  A Decisions-log entry is for the director.
- **Operator nodes in `meaning_graph.py` and `memory.py:165` (`apply_operator("NOT")` when `tree.negated`):** recommend
  **repurposed**. The reader stops emitting `negated` and operator nodes; every word is an ordinary node with a point.
  NOT/MAYBE/IF operators and truth tags remain in the module as something the learned consciousness may *do* (supersede,
  mark FALSE) when it reads high NOT/IF dims; history s5 notes this machinery was gate-tested and is kept as is.
- **Grammar subtype names** in `quantum_parser/grammars/english.json`: NEGATIVE (3 uses, one rule's note describes
  it as "Negation"), QUESTION (2 pushes), RELATIVE (4), MODAL, QUANTIFIER, POSSESSIVE, COMPARATIVE. By the owner's rule a
  grammar label for negation/questions is not allowed; see question 5.
- **Frozen encoder v6:** its prime/role vocabulary (`prime` in {YOU, I, UNK}, PLACE/SOURCE/OF/WITH... labels, discourse
  link coordinators) no longer matches. **Gold v6 (22,990 records) must be rebuilt and the encoder retrained**; seeds
  0.708 / 0.677 become historical; the POS cheat baseline must be re-scored on the new format.

---

## G. Risks and what is NOT solved

- **How strongly "not" moves an already-negative word** ("not unhappy"): the two trees differ (shown in E.2); what is
  open is whether NOT should pull toward the positive end and by how much. v1 moves nothing. Owner intent: graded.
- **Scope ambiguity** (every student did not fail): the tree records attachment, not scope order; open.
- **Double negation / negative concord** ("no vino nadie"): two NOT nodes are kept; whether they count as one is left
  to the learned models. Owner intent: Spanish "n't" equivalents just fill the dimension.
- **Yes/no questions** have no word carrying UNKNOWN (inversion only). A grammar-inserted covert node whose point is
  a lexicon entry would work; deferred, not designed.
- **Wh-words as relative pronouns** ("the man who left") share spelling with question "who" but are not UNKNOWN;
  needs a second lexicon reading selected by structure; unverified how the parser separates them.
- **Numbers**: one/two on ONE/TWO, larger ones name-like + memory facts; "twenty-three", "three hundred and four"
  compounding unsolved. Owner intent: name-like description or built from one and two.
- **Tense/aspect**: simple past and future only. Progressive/perfect auxiliaries (was eating, has eaten) unplaced;
  inflection needs a per-language forms table.
- **Idioms** ("not bad" = good): memory facts, "X really means Y"; not tested.
- **Content-word NOT** for unhappy-type pairs depends on the measurement in choice 2.
- **Homograph readings of closed-class words** (to, that, will, can, her): selected by the tagger's POS, which is itself
  hand-written data today.
- **"so"/"therefore", "by" (agent), "only/even"**: no entry yet.
- **Cross-language fit** shown for 6 Spanish words only; nothing tested on real Spanish text.

---

## H. Staged plan (each 1-5 hours for an implementation agent; light checks only)

1. **Extension axes + lexicon file + loader (4 h).** Produces `ext_axes.json`, `lexicons/en.json` (the 54 entries + the
   rest of the closed class), `es.json` (6 entries), and `fn.<lang>.<word>.<k>` senses minted post-fingerprint;
   `senses_of_surface` consults the lexicon first for function tags. **Check:** all 136 closed-class words of the 7 audit tags have a point;
   "i" is not iodine; base fingerprint unchanged; synonym AUC / syn>ant / random-pair unchanged within 0.01.
2. **NOT overlay from WordNet pointers + affix table (2-3 h).** Produces `overlay_not.json`, deterministic.
   **Check:** ~948 adjective pairs marked; 30 random pairs eyeballed; two rebuilds byte-identical; scores unchanged
   within 0.01 with and without the overlay; report unhappy/happy and a few pairs' cosine before/after (this is the
   measurement choice 2 waits on).
3. **Move the parser's closed-class tables to the lexicon (3 h).** `pos_tagger` reads JSON. **Check:** tag and subtype
   output identical on the 2,000-record sample, token by token; existing parser tests green.
4. **New tree contract + adapter (5 h).** Node/edge dataclasses; adapter from the parser's graph with all non-punct
   tokens as nodes. **Check:** the 8 sentences in E produce the expected trees; node coverage of non-punct tokens
   measured on the 2,000-record sample (today ~45.8% excluding punct).
5. **Node-by-node scorer with reasons (3-4 h).** `score.py` successor. **Check:** identity scores 100%; dropping "not"
   gives `function_node_dropped`; "to"->"from" swap is penalised by exactly the GOAL/SOURCE difference; print scores
   for not-happy vs unhappy vs not-unhappy vs happy for the owner to inspect.
6. **Optional (3 h): regenerate ~2k gold in the new format** and re-score the POS cheat baseline. No training.

Dispatched work must be verified on `origin` per the ops rule.

### Questions for the owner before step 1 (recommended answers)

1. Approve the 7 new axes (KNOWN, UNKNOWN, GOAL, SOURCE, AND, MALE, FEMALE)? **Yes**; MALE/FEMALE are the most
   droppable if you prefer fewer.
2. Lexicon at `quantum_parser/lexicons/<lang>.json`, serving both parser (pos/subtype) and USVS (dims)? **Yes.**
3. NOT values on content words (un-/in-/dis-/-less) via overlay, default 0.7? **Measure first (step 2), decide after.**
4. Comparison-time composition: none in v1? **Yes**, fold only as a later experiment.
5. Grammar subtype names (NEGATIVE, QUESTION, RELATIVE...): may they be renamed to purely structural classes defined
   in the lexicon (position, not meaning), or must rules read USVS dims? **Rename to structural classes.**
6. Numerals above two as name-like nodes plus memory facts ("three = two and one")? **Yes.**

---

## Summary for the owner

1. Every word stays in the tree as an attached node; its meaning is a point on named USVS dimensions from a per-language lexicon file.
2. Scope and nesting come from the tree (the parser's existing edge types); degree comes from the node's own numbers. Option C, per your confirmation.
3. Content words are untouched: same 607 axes, same placement, no rebuild; lexicon rows are added post-fingerprint like the interjections.
4. 26 empty prime axes (NOT, VERY, MORE, THIS, SOME, I, YOU, MINE, BEFORE, AFTER, IF, BECAUSE, MAYBE, CAN...) are filled by closed-class words; only 7 new axes (KNOWN, UNKNOWN, GOAL, SOURCE, AND, MALE, FEMALE).
5. "who/what/where" = someone/something/place + UNKNOWN; "the" = KNOWN; tense = BEFORE/NOW/AFTER; possession = MINE; "not" = NOT 1; no code path for any of them.
6. "not happy" vs "not unhappy" differ because their head words sit at different places; negation is a node attached to each. How hard NOT should pull an already-negative word is open.
7. Evidence pulls against "not = full reversal" (not-A stays near A); v1 therefore never folds "not" into its head, nodes are compared one by one.
8. Optional NOT overlay for un-/in-/dis- pairs from WordNet antonym pointers (948 adjective pairs), decided after a measurement: existing placement separates only some pairs.
9. Supersedes D3/D4, the preposition-to-role table, the role labels, `negated`; polarity cue lists move to data (removing them would cost 0.041 syn>ant).
10. Gold v6 and encoder v6 must be rebuilt; grammar subtype names like NEGATIVE need your ruling.
11. Not solved: scope order, negative concord, yes/no questions, relative who, numbers above two, aspect, "so", "by", idioms.
12. Six steps of 2-5 hours, light checks; six questions above need answers before step 1.
