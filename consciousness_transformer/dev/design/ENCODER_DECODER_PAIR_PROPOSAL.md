# Encoder + decoder as a pair — method statement for owner approval

Status: PROPOSAL (2026-11-16). Nothing is built or trained. Per the working agreement
this is the method check that comes before any machinery.

Facts it rests on: `dev/reports/GOLD_V7_STATS.md`, the last three ledger entries
(gold v7; grammar pass 2), `dev/reports/DECODER_ROUNDTRIP_FEASIBILITY.md`,
`src/naomi/types.py` (tree contract), `src/naomi/score.py` (meaning + shape scorer).

## 1. The situation in four lines

- The tree contract is settled: every word of a sentence is a node, attached to one head
  by a structural relation from the grammar; clauses nest; function words are grounded by
  the lexicon; nothing is dropped.
- The grammar parser is the only teacher. After two grammar passes it connects about
  35-38% of words to the sentence's main root on book prose; about 27% of sentences come
  out as one connected tree (61% of sentences of ten words or fewer, 0% of those over 35).
- Hand-written grammar rules have hit diminishing returns: the variants that connect more
  were 30-70% wrong when hand-checked.
- So the learned reader has to get past its teacher. That is what training the encoder
  and decoder as a pair is for.

## 2. What each part is

**Encoder (text -> tree). Learned.** For every word it chooses (a) which other word is
its head, or that it is the root, and (b) the relation label. That is the whole output.
Because every word is a node by contract, the encoder cannot drop or invent content: its
only freedom is where each word attaches.
- Given: the words, their tags and structural classes from the lexicon, and each word's
  candidate meaning points (USVS for content words, lexicon for function words).
- Not its job: choosing a word's sense (memory does that later), answering anything.
- This replaces the old encoder's output format (the v6 model predicts the old lossy
  clauses and must be retired, not adapted).

**Decoder (tree -> text). Learned.** Given a tree whose nodes are words with their
relations, it produces the sentence: the order of the words, and their surface forms.
- First version: ordering only (the tree's nodes carry the actual words). This is the
  minimum needed for the round trip.
- Second version: nodes carry lemma + meaning point instead of the surface word, and the
  decoder also chooses the word form. That is the step toward speaking from memory, and
  it stays parked until the first version works.

**The grammar. Given, used as a judge.** It does not generate, but it can say whether a
proposed attachment (head category, relation, dependent category, side) is one its rules
license. That check is independent of both learned parts.

**The scorer. Given.** Grades a tree against another (meaning + shape, with reasons) and
grades a single tree for connectedness.

## 3. How they are trained

**Stage A — start from the teacher (supervised).**
- Encoder learns the parser's attachments from gold v7. One rule matters: where the
  parser left a piece unconnected, that is treated as "unknown", not as a correct answer.
  The encoder is never taught to leave a piece dangling.
- Decoder learns to put the words of a tree back in order, from the sentences the parser
  connected fully (about 8,000) plus the internally connected pieces of the rest.

**Stage B — get past the teacher (the round trip, with the middle tree checked).**
For sentences the parser could not connect:
1. The encoder proposes its best few complete trees (every word attached).
2. Each proposal is checked independently of the decoder:
   - every attachment must be licensed by the grammar;
   - it must agree with the parser wherever the parser did commit;
   - it must be one connected tree.
3. Each surviving proposal is sent round the loop: tree -> decoder -> text -> encoder ->
   tree. The proposal that comes back most nearly unchanged (scorer: meaning + shape),
   with the text closest to the original, is kept as a new training example.
4. Retrain both on teacher data plus the kept examples; repeat.

This is self-training with an independent referee, not a hidden-gradient trick: every
kept tree has a record of why it was kept (which checks it passed, its scores), so the
training data stays traceable. The pair cannot drift into a private code because the
tree's content is fixed to the sentence's own words and every link must pass the grammar.

**Feedback with reasons.** Every rejected proposal has a reason (unlicensed link, piece
left out, came back different, order wrong). Those reasons are logged per round; they
are also the list of what the grammar or the models most need next.

## 4. How we will know it works (decided now)

Measured on held-out sentences never used in training:
- **On sentences the parser connects fully:** encoder agreement with the parser's tree
  (scorer). Baseline to beat: attach every word to the nearest verb.
- **On sentences the parser fragments** (the real test): share of sentences the encoder
  turns into one grammar-licensed connected tree, and a hand-check of 60 of them (right /
  debatable / wrong per attachment). Target: clearly better than the parser's 27%
  connected, with at most 15% wrong attachments in the hand-check.
- **Round trip:** tree -> text -> tree similarity, and how often the decoder reproduces
  the original word order.
- **Abandon or rethink if:** after stage B the hand-check shows more than 25% wrong
  attachments on the newly connected sentences (the referee is too permissive), or the
  connected share does not rise by at least 15 points over the parser.

## 5. Known risks

| Risk | Why it is real | What limits it |
|---|---|---|
| The grammar accepts wrong links | It accepted word salad when run backwards | agreement with parser where it committed; hand-check each round; a link type that fails hand-checks is excluded |
| Self-training reinforces early mistakes | kept examples feed the next round | keep only top-scoring proposals; cap new examples per round; compare each round on the fixed hand-check set |
| Decoder learns order from punctuation or length cues | ordering is easy to shortcut | score order within each head's children, not whole-string match |
| Long sentences stay out of reach | 0% connected over 35 words today | report by length band; splitting long sentences at the parser's own clause boundaries is allowed as data preparation, not a rule in the reader |
| The old encoder work is thrown away | v6 predicts the old format | its training code and features (tags, sense candidates) carry over; only the output side changes |

## 6. Size

Small models on CPU, as before: stage A is a few hours; each stage-B round is
propose (minutes) + referee (parser/grammar checks, about an hour) + retrain (a few
hours). Three rounds to a first verdict.

## 7. Questions for the owner

1. **Encoder output as "for each word, its head and relation".** Agreed? (It is the
   direct reading of "every word a node attached to what it modifies".)
2. **Decoder first version = word order only**, with word-form choice from lemma and
   meaning as the second version. Agreed?
3. **Stage B as self-training with the grammar, the parser and the round trip as
   referees**, rather than back-propagating through the pair. Agreed? (Your earlier
   options: this is your case 4, round trip, with case 1, tree correctness, as the
   check on the middle tree.)
4. **"Unknown, not wrong"** for the parser's unconnected pieces in stage A. Agreed?
