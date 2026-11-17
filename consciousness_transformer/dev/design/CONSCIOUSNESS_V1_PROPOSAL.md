# The consciousness, first version — method statement for owner approval

Status: DRAFT (2026-11-17). Nothing is built. Per the working agreement this is the
method check before any machinery. It depends on one measurement still running
(`dev/reports/READING_READINESS.md`: is reading good enough on simple lesson text?).

## 1. Which part this is

The thinking part: a learned policy that, given a question and a memory full of trees,
decides which memory operations to apply and returns an answer — or says "not stated".

| | Given (not learned here) | Learned here |
|---|---|---|
| Reading | grammar parser + encoder (frozen) turn text into trees | — |
| Memory | layered graph of trees; every node also a tensor (`src/naomi/memory.py`) | — |
| Meaning | USVS points, lexicon, antonym relations | — |
| Operations | a small fixed set, below | **which operation, on what, in what order, and when to stop** |
| Answering | — | **which memory node(s) answer, and whether to abstain** |

Weights hold how to look things up and combine them, never a fact: the same trained
policy with an empty memory knows nothing (house rule, unchanged).

## 2. Operations (first set — deliberately small)

1. **FIND** — memory nodes most similar to a given tree or node (tensor view).
2. **FOLLOW** — from a node, step along one of its tree links (to its head, or to a child
   by relation label), or to the same entity's other mentions.
3. **COMPARE** — how similar two nodes are (the tree scorer's meaning measure, incl.
   opposite detection and folded operators such as "not").
4. **ANSWER** — return a node (and its subtree) as the answer.
5. **ABSTAIN** — return "not stated".

Each step is logged, so every answer has a trace of which facts it used. Operations whose
result is a memory address are hard choices (decision 2026-08-30, D1); the controller's
own state carries control information only, never copied facts (D2).

## 3. The training signal

Only the answer. A lesson or story is read into memory; a question is read into a tree;
the policy runs; its answer node is compared, by meaning, with the correct answer read
into a tree. No hand-written programs, no per-step targets.

- **First version: one step.** FIND with the question, then choose among the top
  candidates which node answers (or abstain). This is learnable by ordinary supervised
  training (pick the right candidate) and gives a baseline.
- **Second version: several steps.** FIND / FOLLOW / COMPARE chains up to a small budget,
  trained from the answer alone (reinforcement-style), starting from the one-step policy.

Abstaining must pay better than answering wrong and worse than answering right, with the
wrong-passage control (question asked against a different story) as its test.

## 4. What it is taught from

The owner's plan: K-12 lessons, then conversation. Until lessons are chosen, the stand-in
is what the repository already has, with cheat baselines fixed earlier: MCTest stories
(children's stories, 4-choice questions) and FairytaleQA (stories, questions with
answers). Held out by whole story. Multiple choice is used only to score; the policy
answers with a memory node, and the chosen option is the one closest to that node.

## 5. How we will know it works (decided now)

- Beats the word-overlap cheat on held-out stories (MCTest 45%, FairytaleQA 56%).
- **Memory-severed control:** same policy, empty memory → at chance.
- **Question-shuffled control:** another question from the same story → at chance.
- **Wrong-passage control:** abstains at least 90% of the time.
- Every answer comes with its trace; 30 traces hand-read for whether the cited facts
  actually support the answer.

Abandon or rethink if the policy cannot beat the cheat after the one-step version, or if
the memory-severed control is not at chance (it would mean answers come from the weights).

## 6. Known risks

| Risk | Limit |
|---|---|
| Reading errors poison memory | measured first (reading readiness); train only on stories whose trees pass the trust check, report the rest separately |
| Answers that are phrases, lists or reasons ("why") | first version answers with a node and its subtree; "why" questions reported separately, not tuned on |
| Pronouns: "he ate it" must reach the right entity | FOLLOW-to-other-mentions needs entity linking in memory; first version uses only what memory already links, and reports pronoun-answer questions separately |
| Weights memorise story content | held out by whole story; memory-severed control |
| The multiple-choice options become a crutch | the policy never sees the options; they are only used to score its node |

## 7. Questions for the owner

1. **Answer as a pointer to memory first** (the node and its subtree), with speaking the
   answer left parked? (Recommended: yes — it keeps the decoder out of the scoring.)
2. **Training signal = the answer only**, starting one-step and moving to multi-step?
3. **MCTest + FairytaleQA as the stand-in for K-12 lessons** until lessons are chosen?
   And is there a particular K-12 source you have in mind?
4. **The five operations** as the first set — anything you consider essential that is
   missing (for example a way to count, or to order events in time)?
