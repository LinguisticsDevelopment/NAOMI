# Orderer on gappy fragments

Question: the orderer (decoder v1) was trained on whole connected sentences and their connected pieces; in the chain it orders ANSWER
fragments (a subtree cut from a story sentence, some descendants dropped). Can adding such fragments to training make it better there,
method unchanged (it orders the given nodes, no new word identity, no position input)?

## What was done
- Code: `naomi.orderer.sample_fragment` (a random root, random frontier children added until a drawn size; a child not added drops its
  whole subtree; sizes 2-15 weighted toward 5-12), `fragments_from_items` in `scripts/train_naomi_orderer.py` (root relation reset to ROOT,
  true order = source order), options `--fragments --frag-per-piece --frag-max --init --out-dir`; `scripts/eval_orderer_fragments.py`;
  `--orderer` and `--out_tag` options on `scripts/eval_mind_v2.py`. Fast test: `test_sample_fragment_is_connected_with_gaps`.
- Data: gold v7 TRAIN-split sources only (same as before). Mix per epoch = every full piece (as before) + 4 fragments per connected piece
  of >= 3 nodes (288k trees; ~85% fragments). Fine-tuned from `large_best.pt` (same net, 0.58M params), lr 1e-3, selection on a fragment
  dev set (dev-split sources, 5.9k fragments) averaged with full-piece dev heads-right.
- Runs (one at a time, nice, logs in `runs/naomi_orderer_frag/`):
  - `frag` (sizes 2-15): 37.5 min, early stop; best at epoch 18 (`frag_best.pt`).
  - `frag30` (continued from `frag_best.pt`, sizes 2-30, long tail weight 2, lr 5e-4): 7 min, early stop after 5 epochs; best = epoch 1
    (`frag30_best.pt`). Added after the chain check showed 346 of 939 answered test answers have 13+ words (outside the 2-15 range).
  Total training ~45 min. Dev fragment heads-right moved only 0.917 -> 0.926: the training loss keeps falling (0.113 -> 0.063), the dev
  does not, so fragments are largely ambiguous from tree features alone (a gapped fragment does not say where the missing words were).

## (a) Full-sentence ordering, TEST split sources (exact order / per-head right / pairwise)
| checkpoint | connected sentences (393) exact | heads right | pairwise | pieces (3709) exact | pairwise |
|---|---|---|---|---|---|
| old `large_best` | 0.7176 | 0.9225 | 0.9378 | 0.8008 | 0.9452 |
| `frag_best` (2-15) | 0.7048 (-1.3) | 0.9191 | 0.9384 | 0.7978 (-0.3) | 0.9445 |
| `frag30_best` (2-30) | 0.7125 (-0.5) | 0.9214 | 0.9388 | 0.8010 (+0.0) | 0.9448 |

`frag` loses 1.3 points exact on the 393 connected sentences (5 sentences), failing the 1-point bar on that number; `frag30` holds
(-0.5; pieces and pairwise unchanged).

## (b) Fragment ordering, TEST split sources (12,316 fragments, sizes drawn from the training distribution, 4 per piece)
Exact order / pairwise accuracy over all node pairs, by number of words:
| checkpoint | all | 2-4 | 5-8 | 9-12 | 13-15 |
|---|---|---|---|---|---|
| old | 0.7855 / 0.9333 | 0.917 / 0.961 | 0.751 / 0.926 | 0.562 / 0.885 | 0.410 / 0.865 |
| `frag` | 0.7935 / 0.9354 | 0.917 / 0.963 | 0.765 / 0.928 | 0.573 / 0.886 | 0.421 / 0.868 |
| `frag30` | 0.7957 / 0.9362 | 0.920 / 0.964 | 0.767 / 0.929 | 0.575 / 0.887 | 0.426 / 0.879 |

Longer fragments (sizes 2-30 test set, `eval_fragments_long.json`): all exact old 0.776, `frag` 0.778, `frag30` 0.782; 16-30 words:
old 0.364 / 0.885, `frag` 0.322 / 0.857, `frag30` 0.327 / 0.860 (both fine-tuned models are WORSE than the old one on 16+ word fragments;
the old one is better there, probably because long fragments look like whole sentences, which it was trained on at full weight).
The gain is real but small: +0.8 to +1.0 points exact overall, concentrated at 5-12 words.

## (c) In the chain: v2b thinker (`runs/mind_v2b/thinker_s2b_all_best.pt`), FairytaleQA TEST, 1007 items, 939 answered
Decoded ROUGE-L (abstentions count 0), same declared round-trip threshold (A = 99, `runs/mind_v2/roundtrip_val.json`):
| orderer | (a) order-only ROUGE-L | (b) meaning-only ROUGE-L | true word order |
|---|---|---|---|
| old | 0.3484 | 0.3420 | 0.3722 |
| `frag` | 0.3491 | 0.3427 | 0.3722 |
| `frag30` | 0.3486 | 0.3423 | 0.3722 |

Round-trip pass rate (reread of the decoded fragment vs the same words in true order, threshold 99), setting (a) / (b):
| orderer | 1-3 words (246) | 4-6 words (139) | 7+ words (554) | all |
|---|---|---|---|---|
| old | 0.984 / 0.955 | 0.842 / 0.827 | 0.332 / 0.314 | 0.578 / 0.558 |
| `frag` | 0.980 / 0.951 | 0.842 / 0.827 | 0.354 / 0.336 | 0.590 / 0.570 |
| `frag30` | 0.976 / 0.947 | 0.856 / 0.842 | 0.365 / 0.348 | 0.597 / 0.578 |
| true order | 1.0 (by construction) | | | |

Reproduction check: the old row equals the v2b report (0.348 / 0.372; 98 / 84 / 33). Outputs: `runs/mind_v2b/eval_test_orderer_{old,frag,frag30}.json`.

## Conclusion
Fragment training helps the orderer a little where the chain uses it: +1 point exact on 5-12-word fragments, +2 to +3 points round-trip
pass rate on 7+-word answers (33.2 -> 36.5 for `frag30`), and the 4-6 band moves from 84.2 to 85.6. It does not move decoded ROUGE-L
(0.3484 -> 0.3486, inside noise), and the gap to true word order (0.372) stays: most of it is information a tree does not carry (where
missing words were), not a modelling shortfall. Cost: `frag` regresses full sentences by 1.3 points exact; `frag30` (the better of the
two) by 0.5, fine-tuned models are worse on 16+-word fragments. The ceiling here is the representation, not the data volume.

Default: NOT changed (`runs/naomi_orderer/large_best.pt` stays). Condition (a) holds for `frag30`, but (c) is an improvement only on the
round-trip rate, with ROUGE-L unchanged, and `frag30` was picked after seeing test numbers (two variants). To adopt it:
`--orderer runs/naomi_orderer_frag/frag30_best.pt` already works in `scripts/eval_mind_v2.py`; flipping the default is the one-line
default of that option (plus the hard-coded path in `scripts/roundtrip_controls.py`). Director's call.
