# The consciousness, text in / text out — method statement for owner approval

Status: APPROVED (owner, 2026-11-17): 1 yes, 2 yes, 3 yes, 4 yes. Owner note on 1: the decoder should
be choosing the word that best matches the node's meaning, not just ordering given words -- so
this version runs the decoder both ways and reports both (see section 1, step 4).

Originally drafted 2026-11-17. Replaces the scoring of v1 (`CONSCIOUSNESS_V1_PROPOSAL.md`), whose
"pick a memory node, map it to the closest multiple-choice option" test the owner judged broken:
it grades word-picking through a hand-tuned mapping (a 20% meaning threshold), and cannot give
each answer a percentage. Owner direction: force the full chain encoder -> thinker -> decoder and
score the answer text with a robust measure. v1b (tree-attention pointer) was stopped before it
produced anything.

## 1. The chain

1. Story text -> encoder -> trees -> memory (unchanged).
2. Question text -> encoder -> question tree (unchanged).
3. **Thinker (learned):** given memory and the question tree, writes an ANSWER TREE, or
   abstains. It builds the tree step by step: take a node from memory (optionally with part of
   its subtree), or a node of the question, and attach it to the answer tree under a relation.
4. Answer tree -> decoder -> answer text. Two settings, both reported: (a) nodes carry their words,
   decoder only orders them (decoder v1); (b) nodes carry only meaning (USVS point, relation, tag),
   and the decoder chooses each word from the words of the story and question by best match to
   that meaning, then orders them. (b) is the honest chain; (a) isolates the thinker's share.
5. The answer text is scored against the reference answer text.

Given: reading, memory, USVS, decoder (frozen). Learned here: which nodes go in the answer tree,
how they attach, when to stop, when to abstain.

Limit of this version: the answer's words come from the story or the question, because the
decoder only orders the words it is given (choosing word forms from meaning is decoder v2,
still parked). Most FairytaleQA answers are phrases from the text; the rest are reported
separately as "needs new words", not tuned on.

## 2. Scoring every answer with a percentage

Two measures per answer, both reported:
- **Text overlap, ROUGE-L F1**: the measure FairytaleQA's own authors use. Independent of
  the project's tools, so a reading bug cannot inflate it.
- **Tree score**: answer text and reference both read into trees and graded by the tree
  scorer (meaning + shape), giving a % and reasons (word missing, extra word, role wrong,
  opposite meaning). This is the feedback with reasons.

Plus a pair-consistency check: the decoded answer read back into a tree must match the tree the
thinker wrote (the round trip), so the thinker and the decoder cannot drift into a private code.

Baselines (cheat, fixed now): return the story sentence with the most word overlap with the
question, whole; and echo the question. Random sentence as the floor. Controls kept from v1:
empty memory -> abstain; question from another story -> abstain >= 90%; questions shuffled within
a story -> at the floor. 30 answers hand-read.

## 3. Training signal

Still only the answer. Stage 1: the reference answer read into a tree is the target; the thinker
is trained to build it from memory (where the reference's words can be found in memory; items
where they cannot are skipped and counted). Stage 2: fine-tune on the final text score itself
(reward = ROUGE-L of the decoded answer), so the whole chain is judged on what it says.

## 4. Data and "basic first"

FairytaleQA free-form answers (all ~3k items in the repo, split by whole story); MCTest options
only as a secondary check. Early training is weighted toward questions whose answer is in a short
sentence (where reading is reliable), with longer ones added in stages. Every result is reported
by answer-sentence length, and the short-only vs all-lengths comparison stays as an explicit test
(owner condition: no lock to basic sentences).

## 5. Abandon or rethink if

- ROUGE-L does not beat the overlap-sentence cheat on held-out stories, or
- the round-trip check fails on more than 20% of answers, or
- empty memory still produces answers.

## 6. Questions for the owner

1. Answer words limited to words in the story or the question for now (decoder v2 parked)?
2. Both measures (ROUGE-L for honesty, tree score % with reasons for feedback)?
3. Training: first toward the reference answer tree, then on the final text score?
4. Basic first as a weighting that widens in stages, with every length always reported?
