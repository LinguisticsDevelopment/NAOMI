# Mind v1: the one-step learned answering policy (2026-11-17)

Status: DONE (one-step version). Verdict: **does not beat the word-overlap cheat** (the proposal's own abandon-or-rethink criterion); controls behave as required except MCTest wrong-passage abstention (89%, target 90%). Single run, held-out test stories, one rater for the hand-read.

## What was built
Method as approved in `dev/design/CONSCIOUSNESS_V1_PROPOSAL.md`, one step only.

1. **Reading (given, frozen)**: sentence -> tree by the parser; where it leaves pieces, the A2 encoder fills the joining links (parser links kept). A tree is "trusted" when every added link is chosen by stage-A, A2 and control with minimum head probability >= 0.8 (READING_READINESS S3); parser failures are untrusted. Trust is used for reporting only. All strings cached once in `runs/mind_v1/read_cache/reads.jsonl` (not committed).
2. **Memory**: per item a fresh `Memory`; every story sentence stored as a tree with provenance (`sentence_index`).
3. **FIND -> candidates** (`naomi.mind_v1.extract`): every memory node has a tensor (the memory's own TPR scheme over the node's subtree, relative paths). For each content node of the question the top `ceil(30/|Q|)` (min 4) memory content nodes by tensor cosine are taken (balanced, so a frequently repeated name cannot crowd out the rest; about 30 in all), then every node one tree link away (parent, children) is added. Function words (lexicon `fn.` senses) are never candidates.
4. **Policy** (`naomi.mind_v1.Policy`, ~60k parameters): scores every candidate and ABSTAIN. Inputs are numeric only: the candidate's and its subtree's USVS vectors and the question's wh-node vector (the node whose meaning point carries the lexicon's UNKNOWN dimension), through one shared linear projection and their products; FIND similarity and ranks; COMPARE (meaning similarity to question nodes, sentence coverage, opposite flag); the relation path (up/down steps, first/last relation label) from the candidate to the nearest question-matching node; the candidate's own and parent's relation labels; agreement with the wh node's relation and parent; structure (depth, subtree size, positions, sentence length); mention count; question structure. No token, lemma or any word identity.
5. **Training signal = the answer only**: cross-entropy over candidates + ABSTAIN. Target = the set of correct candidates (marginal likelihood) when one is in the candidate set, else ABSTAIN; 20% of examples are wrong-passage items (the question against another story of the same dataset) with target ABSTAIN. A candidate is correct when its subtree's meaning% (tree scorer, option as reference) is strictly the best among the four options for the correct option and >= 20 (fixed; the scorer folds determiners into the noun, so a correct "the door" scores about 35-50 against "door").
6. **Answering**: `Mind.ask(question)` returns the chosen node rendered as its subtree text, a trace (candidates considered with probabilities, FIND/COMPARE values, the sentence(s) and tree the node came from, abstain probability), or "not stated". Multiple choice: the chosen node maps to the option with the best (meaning, overall) by the tree scorer; the policy never sees the options.

## Data
- Reading: 9,692 distinct strings (story sentences, questions, options of both datasets) read once; 4,850 parser-complete trees with all added links trusted, 3,846 with at least one untrusted added link, 893 one-word strings (trivially trusted), 127 parser failures (tree from the encoder alone, untrusted). 787 s on 4 cores.
- Items: MCTest mc160 400 items / 100 stories (train 212 + validation 68 carved from 20 train stories by a stable hash of the story id, test = dev 120); FairytaleQA same-passage entity items with four options 584 (train 439, val 63, test 82; the file's other 2,416 items have no options and are not scorable here). 984 scored items in all: train 651, val 131, test 202.

## Development note (what changed after the first build)
First features (FIND/COMPARE by USVS-similarity coverage) gave validation hit rate ~0.17 (answer-in-candidates ceiling 0.68): the feature means of correct and incorrect candidates were almost identical for the sentence-level signals. Added (before any test number was looked at): exact-concept coverage per sentence (lemma identity, weighted by in-story rarity), relation-path agreement between the wh node's path to its nearest matched question node and the candidate's path to the same concept in the story, and direct compatibility of the candidate's meaning point with the wh word's own non-UNKNOWN dimensions. Items were rebuilt once. All numbers below are from the final build.

## Results (held-out test stories: MCTest dev 120 items, FairytaleQA test 82 items)
Policy = `runs/mind_v1/policy_all_best.pt` (trained on all sentence lengths; 16-dim projection, 64 hidden, ~66k parameters; epoch 32 of 60 chosen on validation; selected as the best of three configurations by validation score, so validation numbers are optimistic, test numbers are not). "Acc" counts abstain as wrong; "random on abstain" credits an abstain with 0.25.

| | MCTest (120) | FairytaleQA (82) | pooled (202) |
|---|---|---|---|
| answer-in-candidates ceiling (test; train .734, val .679) | 0.583 | 0.756 | 0.654 |
| **policy: acc (abstain wrong)** | **0.208** | **0.317** | **0.253** |
| policy: answer rate | 0.675 | 0.683 | 0.678 |
| policy: acc when answering | 0.309 | 0.464 | 0.372 |
| policy: acc, random guess on abstain | 0.290 | 0.396 | 0.333 |
| policy forced to answer (no abstain option) | 0.250 | 0.366 | 0.297 |
| control: word-overlap cheat (IDF-weighted, ties split) | 0.410 | 0.564 | 0.473 |
| control: random | 0.25 | 0.25 | 0.25 |
| control: a random candidate mapped to an option (the mapping's own floor) | 0.137 | 0.162 | 0.147 |
| control: memory-severed (memory = the question only): acc / answer rate | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| control: question-shuffled (another question of the same story; 120 / 37 / 157 items): acc / answer rate | 0.117 / 0.592 | 0.189 / 0.757 | 0.134 / 0.631 |
| control: wrong-passage abstain rate (target >= 0.90) | 0.892 | 0.951 | 0.916 |

Reading the table plainly: the policy is above chance only when it answers (0.37 vs 0.25) and well below the cheat everywhere; as a system it does not beat the cheat (0.25-0.33 vs 0.47). Memory-severed: it abstains every time (what the house rule needs: no fact in the weights). Question-shuffled falls below chance as required (0.13; it answers 63% of the time with the wrong question, so it is not "reading the question" well enough to abstain). Wrong-passage: 91.6% pooled, but MCTest alone is 89.2%, just under the 90% target.

### By sentence length (answer sentence = the story sentence that best holds the correct option's content; by question length; by reading trust). Policy acc / answer rate / cheat / ceiling
| answer-sentence words | n | acc | answer rate | acc when answering | cheat | ceiling |
|---|---|---|---|---|---|---|
| <= 10 | 52 | 0.135 | 0.615 | 0.219 | 0.362 | 0.654 |
| 11-20 | 72 | 0.292 | 0.736 | 0.396 | 0.506 | 0.639 |
| 21+ | 78 | 0.295 | 0.667 | 0.442 | 0.516 | 0.667 |

MCTest alone: <=10 0.156 (n 45), 11-20 0.255 (51), 21+ 0.208 (24). FairytaleQA alone: <=10 0.000 (7), 11-20 0.381 (21), 21+ 0.333 (54). The long-sentence rows are mostly FairytaleQA, which the policy does better on (and where the cheat does better too), so the pooled "longer is better" pattern is a dataset mix effect, not evidence that long sentences are easier. The ceiling does not fall with length (0.65 / 0.64 / 0.67): reachability is not what length changes here.

| question words | n | acc | answer rate | cheat | ceiling |
|---|---|---|---|---|---|
| <= 6 | 54 | 0.241 | 0.722 | 0.511 | 0.685 |
| 7-9 | 80 | 0.263 | 0.675 | 0.401 | 0.600 |
| 10+ | 68 | 0.250 | 0.647 | 0.527 | 0.691 |

| reading of the answer sentence | n | acc | acc when answering | cheat | ceiling |
|---|---|---|---|---|---|
| trusted | 68 | 0.176 | 0.267 | 0.415 | 0.632 |
| untrusted | 134 | 0.291 | 0.424 | 0.502 | 0.664 |
Trusted vs untrusted is confounded with dataset (60 of the 68 trusted sentences are MCTest, 74 of 82 FairytaleQA answer sentences are untrusted). Within MCTest alone: trusted 0.167 (n 60), untrusted 0.250 (n 60); the policy is not better on well-read sentences. At test time only 68 of 202 answer sentences (34%) have a trusted reading.

### Short-to-long generalisation (the owner's condition)
Second model trained only on training items whose answer sentence is <= 10 words (147 of 651 items, 100 with a reachable answer; checkpoint chosen on the 29 short validation items), same hyper-parameters; both models on the same held-out items (pooled; per-dataset in `eval_summary.json`):

| answer-sentence words | n | trained on ALL lengths: acc / answer rate | trained on <= 10 only: acc / answer rate |
|---|---|---|---|
| <= 10 | 52 | 0.135 / 0.615 | 0.154 / 0.615 |
| 11-20 | 72 | 0.292 / 0.736 | 0.097 / 0.458 |
| 21+ | 78 | 0.295 / 0.667 | 0.077 / 0.333 |

**Plain statement: training on short sentences did not transfer, and this experiment cannot show whether the mechanism would.** The short-trained model is at or below the mapping floor (0.147) on every band, including the short one (0.154, equal to the all-lengths model's 0.135 on those 52 items). With 147 training items (100 with a reachable answer) it learned almost nothing, so the long-sentence failure is not evidence about composition; it is evidence that this much data and these features do not learn the task at any length. What the all-lengths model shows is that nothing in the features breaks on long sentences (accuracy and ceiling do not drop with length), which is the weaker, positive-but-unproven reading of the owner's condition.

### Hand-read of 30 traces (single rater = the implementing agent; held-out test; 10 right / 10 wrong / 10 abstain by the option mapping, seeded draw)
Files: `runs/mind_v1/eval_traces_30.json` (full traces), `runs/mind_v1/eval_traces_30_judgments.json` (judgment and note per trace). Does the cited sentence support the answer the policy gave?

| outcome (by option mapping) | supports | partly | does not | n |
|---|---|---|---|---|
| "right" | 4 | 2 | 4 | 10 |
| wrong | 0 | 6 | 4 | 10 |
| abstain | n/a | | | 10 |
Of the 10 answers scored right, only 4 are right for the right reason; 4 are luck of the closest-option mapping (e.g. chosen node "river" from "elk river castle" mapped to "Swam in the swimming pool"), so measured accuracy overstates what the policy knows. In 6 of the 10 wrong answers the cited sentence does contain the right answer but the policy picked another node of it (the rest of the sentence, or a neighbouring noun): the failure is node selection inside the right sentence more than finding the sentence. Of the 10 abstentions, the story visibly states the answer in 5 (e.g. "an inn", "a bear and a wolf", "high tower", "in splendor and happiness", "working in the garden"); 5 unclear. Typical wrong-parse effect: the question "What was the hardest thing for Tom and his friends to fix?" is read as `was(SUBJECT:what INDIRECT_OBJECT:thing(...) OBJECT:to(MODIFICATION:fix))`, so the wh node's relation cannot line up with a story relation.

## Limits, with numbers (the three that cost most accuracy)
1. **Choosing the node inside the candidates** (the learned part is weak): reachable 65% of items, policy picks a correct-scoring node for roughly 18% (val hit rate; forced-to-answer option accuracy 0.30 pooled, MCTest 0.25 = chance, FairytaleQA 0.37). Hand heuristics on the same features reach about the same validation hit rate (0.12-0.17 vs the policy's 0.17-0.21), i.e. the features, not the optimiser, are the bottleneck: before the final features, the correct and incorrect candidates had almost identical means on FIND, coverage and relation features; after adding concept-identity coverage and relation-path agreement only coverage separates them (0.30 vs 0.22).
2. **Reading, mostly of questions and long sentences**: only 34% of test answer sentences have a trusted reading (MCTest 50%, FairytaleQA 10%); question trees are often shaped around the auxiliary (`did` / `was` as root, wh word as its subject), so the relation path from the wh node to the matched words has no counterpart in the story tree; the path-agreement feature fires for 2% of the correct candidates. FIND plus one tree link reaches only 58% (MCTest) / 76% (FairytaleQA) of correct answers.
3. **Abstain vs answer with 651 training items**: 35% of training items have the answer outside the candidates and are trained as ABSTAIN, so the learned abstain score is partly "I cannot tell", answer rate 0.68, accuracy when answering 0.37; with abstain counted wrong the headline is 0.25 although forced answers would give 0.30 and a random guess on abstain 0.33. Validation accuracy swung between 0.15 and 0.29 from epoch to epoch (131 items), so checkpoint choice itself is noisy. Option-mapping noise is a fourth, smaller item: a random candidate mapped to an option already scores 0.147.

## Honest status
The machinery (candidates, trace, severed / shuffled / wrong-passage controls, by-length reporting) works and the severed control shows nothing is stored in the weights. The one-step learned policy does not beat the cheat and does not show composition; per the proposal this is the point to rethink before the multi-step version. Candidate ideas, not done: replace one-link expansion with FOLLOW steps (second version), give the policy the question's own FIND matches per word instead of a single max, learn the wh-relation alignment from a larger set (more stories), and score by node (answer-sentence containment) separately from node choice so the two problems are not tangled.

## Training wall-clock and tests
Reading 787 s (4 processes), item build 747 s (4 processes; run twice, the second after the feature additions), policy training 77 s (all-lengths, selected configuration; ~20 min cumulative over 9 runs including the sweeps) and 18 s (short), evaluation ~1 min. Tests: `pytest tests/test_mind_v1.py tests/test_naomi_no_literals.py tests/test_naomi_memory.py` 14 passed (24 s). The no-literals test gained an allow-list line for non-linguistic identifiers only (dict keys, split and dataset names, direction tags).

## Reproduce
```
python scripts/build_mind_data.py read --workers 4      # reads cache (not committed)
python scripts/build_mind_data.py items --workers 4     # items_{train,val,test}.pkl (not committed)
python scripts/train_mind_v1.py --variant all   --epochs 60 --P 16 --H 64 --drop 0.3 --wd 0.05 --lr 1e-3 --seed 0 --tag _cfgA
python scripts/train_mind_v1.py --variant short --epochs 60 --P 16 --H 64 --drop 0.3 --wd 0.05 --lr 1e-3 --seed 0 --tag _cfgA
cp runs/mind_v1/policy_all_cfgA_best.pt runs/mind_v1/policy_all_best.pt; cp runs/mind_v1/policy_short_cfgA_best.pt runs/mind_v1/policy_short_best.pt
python scripts/eval_mind_v1.py
```
`Mind(...).ask(question)` loads `runs/mind_v1/policy_all_best.pt` by default.
