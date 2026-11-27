# The thinker composes an answer tree — method and test redesign, for owner approval

Status: APPROVED (owner, 2026-11-27): 1 A now, B later; 2 yes; 3 yes, but the unknown must be
LEARNED, not defined; 4 yes.

**Owner amendment to 2 (binding, 2026-11-27):** score the actual decoded output text against the
reference answer text: ROUGE-L F1, plus the tree score of the decoded text read back vs the
reference tree. No separate answer/frame parts; section 3's item 1 is superseded by this.

**Owner amendment to 3 (binding).** Nothing in code, data prep or the training target tells the
thinker which question node is the unknown. The thinker sees every question node's USVS point
(the UNKNOWN axis is just one of its dimensions) and chooses for itself which node to replace and
with what. It is rewarded only for the result: the output must say what the reference answer
says, and keep the rest of the question. So it should discover that the node sitting on UNKNOWN
is the one to replace. Whether it did is reported as a diagnostic (how often the replaced node is
the question's highest-UNKNOWN node), never used as a target. Sections 2-4 below are read with
this amendment: the gold tree in section 3 is NOT built by placing the reference at a chosen
slot; scoring is placement-free (section 3, as amended).

Originally drafted 2026-11-27. Owner direction: "it never is supposed to be picking a sentence, it is
supposed to output a tree/sentence at the end of the day." This proposal replaces the
anchor-in-memory design of thinkers v2-v4 and the test criteria built around it.

## 1. What went wrong with v2-v4, in method terms

v2-v4 all output an answer tree that is decoded and scored as text, but the tree was always
*one subtree cut out of one memory sentence*. So the thinker's job collapsed into "find the
sentence, then find the node", and every diagnosis (sentence chosen 33% wrong, anchor depth
wrong) was about that cutting, not about building an answer. Three consequences:
- answers that join two sentences, or that need the question's own words, cannot be built;
- the "best subtree" oracle (0.757) is a hard ceiling, set by the cutting, not by the task;
- the diagnostics ("right sentence?", "right anchor?") reward a step the owner never wanted.

What does NOT go away under any design: the thinker must still find where in memory the
information is. Changing the output does not make that free. It changes it from a one-shot
sentence choice into node-by-node attention over the whole memory graph, with partial credit.

## 2. Options (with the research behind each)

**A. Complete the question tree (recommended).** The answer is the question tree with its
unknown filled in: "Why was the king angry?" -> "The king was angry because the princess lied."
- The unknown node is found by its position on the USVS `UNKNOWN` axis (already in
  `data/usvs/ext_axes.json`, filled from the lexicon) — no grammar label, no word list.
- The thinker writes the filler tree node by node: each node is either *copied from a memory
  node* (pointer/copy, as in pointer-generator and Graph2Tree decoders) or taken from the
  question. It may take nodes from several sentences. Every output node keeps a link back to
  the memory node it came from (provenance), which is the anti-confabulation check.
- Research: QA2D (Demszky, Guu & Liang 2018) shows question + answer -> one declarative
  sentence is a well-defined target (86k human rewrites exist); that is the text form of A.
  Matching a question tree against memory trees is the classic tree-alignment approach to
  answer finding; here the matching is learned, not rule-based.
- Why it fits NAOMI: the output is a whole sentence the decoder can say; the question supplies
  the frame, so the thinker learns *binding* (what fills the unknown), which is what K-12
  lessons will need; short-to-long becomes a composition question, not a depth guess.

**B. Free composition from the memory graph.** The thinker grows an answer tree from scratch,
choosing a head and children from any memory or question node, with no question frame. The
most general, and where A should lead later, but it has no structure to lean on and the
least training signal. Not first.

**C. Retrieval then generation (the standard neural route).** Pick passages, then generate
text (BART on FairytaleQA reaches roughly 0.5 ROUGE-L in the dataset paper). It needs a
language model that "comprehends", which the design rules forbid, and it hides the tree.
Reported as an external reference only, not built.

## 3. New test criteria

Per answer, all reported:
1. **Tree score (primary).** Output tree vs gold tree, node and edge matching, giving a % with
   reasons: missing node, extra node, wrong relation, wrong meaning, unknown left unfilled.
   Placement-free (owner amendment): (a) *answer part* — the nodes the thinker put in, against
   the reference answer read into a tree; (b) *frame part* — how much of the question's other
   nodes the output kept. Neither names a slot; replacing the wrong question node loses (a) or (b).
2. **ROUGE-L of the decoded filler** against the reference answer, for comparison with
   v2-v4 and the literature.
3. **Provenance.** Share of output nodes traced to a memory node; nodes from nowhere count as
   confabulation.
4. **Round trip.** The decoded sentence read back must match the output tree (unchanged).

Dropped as criteria (kept only as diagnostics): right sentence, right anchor, anchor depth.
Controls unchanged: empty memory and wrong story -> abstain (unknown left unfilled, >= 90%),
shuffled questions -> floor. Cheat: fill the unknown with the subtree of the best-overlap
sentence that best matches the question (this is v2-v4's method, so it is now the baseline to
beat). Reported by answer length, short-only vs all-lengths kept as the explicit test.

## 4. Training signal

Stage 1: which question node to replace is never supervised. The filler's content may be
imitated (copy the memory nodes aligned to the reference by lemma; unaligned items counted, not
tuned) but the replace-choice is learned from the score. Stage 2: fine-tune on the tree score. Data: FairytaleQA full, split by story.

## 5. Abandon or rethink if

- the filler tree score does not beat the v2-v4-style cheat on held-out stories, or
- provenance falls below 90% (it is making things up), or
- empty memory still produces fillers.

## 6. Questions for the owner

1. Option A first (fill the question's unknown, output a full sentence), B later?
2. Tree score as the primary criterion, ROUGE-L secondary, sentence/anchor only as diagnostics?
3. Finding the unknown via the existing USVS `UNKNOWN` axis — acceptable, given it is lexicon data?
4. The old method (best subtree of the best sentence) becomes the cheat baseline — agreed?
