# The thinker as a loop: trees in, trees out, with state and memory — for owner approval

Status: DRAFT (2026-11-27). Owner correction: thinkers v1-v5 were "bastardized" — each was given
a fixed menu of actions (pick a sentence, pick an anchor, keep/drop words, replace a node), which
predefines what thinking is. The intended thinker is a single learned transition, looped.

Sources found in the repo: the first scaffold (commit 2395e26, June 2026: inputs = text + parse +
"consciousness vector" + memory; outputs = a response and the NEXT consciousness state, "Head 2:
consciousness state transition"); `MIND_INTERFACE.md` §0 (state = GRU over clause + memory read;
memory written by the controller each step); `EXECUTOR_DESIGN.md` (the later op-menu executor —
this proposal moves away from its fixed op list). No doc records the "loop until done" rule; it
is taken from the owner's message of 2026-11-27.

## 1. The method

One learned function, the thinker T:

    (tree_out, state', memory') = T(tree_in, state, memory)

- Run it in a loop: tree_out becomes the next tree_in, state' the next state, memory' the next
  memory. Stop when it has settled (section 3), with a step cap.
- A tree is a set of nodes, each a tensor (USVS point + relation + a parent link). Trees in and
  trees out are the same kind of object. No text inside the loop.
- Reading a story: each sentence tree from the encoder goes in, the thinker updates memory.
  Asking: the question tree goes in, the loop runs, the settled tree_out is the answer.
- The decoder turns the final tree into text: each node's word is the word whose meaning is
  nearest to the node (from the words it has seen — decoder setting (b)), then ordered.

Given: the encoder (text -> tree), USVS, the decoder (tree -> text), the memory's form (layered
graph of tensor nodes). Learned: everything T does — what it reads from memory, what it writes,
what tree it puts out, and when it is done. No action menu: no "pick a sentence", "keep/drop",
"replace", "abstain" heads. Abstaining is just settling on an empty or near-empty tree.

## 2. Training signal

Only the end result: the settled tree_out against the reference answer read into a tree (the tree
score, which is soft enough to learn from directly through the unrolled loop), with ROUGE-L of the
decoded text reported as the honest measure. Nothing supervises intermediate steps, memory writes
or the stop. Same FairytaleQA split, cheat (best-overlap sentence) and controls (empty memory,
wrong story, question-only memory -> near-empty tree; shuffled -> floor), short-only vs
all-lengths test.

Known risk (from the record): a recurrent state can learn to carry the answer itself and treat
memory as decoration (the M57 "state-carry cheat"). Control for it: the question-only and
empty-memory controls, plus a run with memory writes blocked during the question.

## 3. Prior work this resembles

Universal Transformer / adaptive computation time and PonderNet (one block applied repeatedly with
a learned stop), deep equilibrium models (run to a fixed point), Neural Turing Machine / DNC (a
controller that reads and writes an external memory each step). Here the input and output of each
step are trees, and the memory is the project's graph.

## 4. Questions for the owner

1. Should the story also be read through the thinker (it writes memory itself, as above), or
   should memory start as the encoder writes it today and the thinker only update it?
2. "Until the state = output": stop when the tree it puts out is the same as the tree it was
   given (nothing left to change), or when the state stops changing? (Proposed: the tree.)
3. Training only on the final tree vs the reference tree, nothing on the steps in between?
4. Output words chosen by the decoder from the node's meaning (setting (b)), so the thinker never
   picks words directly?
