# USVS content-word diagnosis (read-only, planning phase, 2026-11-15)

Purpose: establish by measurement what is and is not wrong with content-word placement in USVS today, and fix the baseline scorecard any alternative will be compared with. **Nothing in the repo or in `data/usvs/` was changed** (file mtimes unchanged; no git). Scratch scripts/logs/JSON: `SCRATCH = /tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag/`. Convention: sections 1-8 are measurement; interpretation is confined to section 9 (and labelled INTERPRETATION there). Artifact: `data/usvs/` fingerprint `e0daef638b640dd5`, 607 base axes, 9,946 core words, 117,659 senses (+20 interjections); base loaded with `load_usvs(..., extended=False)`.

## 0. Method and definitions

- **Core level** = `core_coords` (9,946 x 607, dense, signed). **Sense level** = `sense_dense()` of the stored sparse rows (117,679 x 607, non-negative, top-24).
- **Weighting**: `raw` = plain cosine. `idf` = the scorer's weighted cosine (`src/naomi/score.py:185-194, 248-263, 303-305`): `w_j = log((N+1)/(df_j+1))`, df = senses with non-zero value; dot = `sum w a b`. My `w` computed on the 607 base axes equals `axis_weights()` of the extended artifact to <=0.023 on every axis with df>0; it differs by up to 3.4 only on 21 formerly empty prime axes that the function-word lexicon now fills (irrelevant for content-vs-content cosine).
- **Pair sets** (WordNet via nltk; `pairs.py`): *synonym* = lemma pairs sharing a synset, both in core (7,480; n 3,460 / v 2,347 / a 1,498 / r 175); the project's own morphy-expanded edge set (22,856, equal to the ledger's count) is reported as a control. *Antonym* = direct lemma antonym pointers, both lemmas in core (737; a 462, n 136, v 101, r 38); sense level uses the exact antonym synsets. *Hypernym* = direct (instance-)hypernym synset pairs (core: 15,654 lemma pairs; sense: 20,000 sampled of 97,666). *Sister* = synsets sharing a direct hypernym, up to 3 pairs per parent (core 2,553; sense 20,000 sampled). *Random* = 20,000 uniform pairs per level, plus 4,000 POS-matched pairs per POS. Sense-level "synonym" = first sense (MFS order) of each lemma, pairs with identical first sense dropped (6,871 left); same-synset synonyms share one vector, so this set is a weak synonym test, hence the `similar_to` control in 2a.
- **AUC(x>y)** = P(cos_x > cos_y) + 0.5 P(tie), rank based. `AUC ant<rand` = P(antonym pair less similar than a random pair): 0.5 = indistinguishable, <0.5 = antonyms look MORE similar than random. `P(antonym closer than a random synonym pair)` = 1 - AUC(syn>ant).
- **Caveat on in-sample numbers**: the production core was placed using all synonym/similar/also_see edges, so core-level synonym/hypernym numbers on the production artifact are in-sample (history report records 0.9535 synonym AUC for that reason). The M24-compliant numbers are section 1 and 2a'.

## 1. Reproduction of the project's own held-out harness

Harness: `src/nsm_ct/ground/ablation.py` (`metric_table`, M24-safe: placement propagates only over train pairs, scored on disjoint test pairs, extras filtered against test splits), driven by `archive/scripts/probe_m28_baseline.py`, `archive/scripts/ablate_signal.py`, `archive/scripts/probe_normalize.py`, `archive/scripts/probe_simlex.py`. All ran read-only (they only print). My driver for the 10k "winners" set is `h1_harness.py` (builds the vocab/graph exactly as `usvs.build_usvs` does: domain features merged, also_see close edges, 607 axes).

| Metric | Recorded (RESEARCH_NOTES) | Reproduced now | Command |
|---|---|---|---|
| 3k, placement anchored -> placed syn>ant (M28.0) | 0.404 -> 0.693 | **0.404 -> 0.693** (n 2,927 syn / 230 ant) | `python archive/scripts/probe_m28_baseline.py --n 3000` |
| 3k syn>ant / synonym AUC / similar AUC / random cos (M29 table, split 0.5) | 0.695 / 0.868 / 0.747 / 0.255 | 0.693 / 0.867 / 0.748 / 0.253 (jitters .45/.5/.55 syn>ant 0.683/0.693/0.709) | `python h1_harness.py 3000` |
| 3k hypernym AUC (anchored containment) | ~0.727 | 0.727 (hypernym_cos AUC 0.835, antonym_auc 0.261) | same |
| **10k syn>ant** | 0.676 | **0.675** mean of jitters (0.666 / 0.673 / 0.686) | `python h1_harness.py 10000 winners` |
| 10k synonym AUC | 0.885 | 0.882 (0.877 / 0.882 / 0.888) | same |
| 10k similar AUC | 0.803 | 0.814 (0.808 / 0.816 / 0.819) | same |
| 10k random-pair cos | 0.233 | 0.234 (0.226 / 0.234 / 0.241) | same |
| 10k hypernym AUC (anchored) / hypernym_cos AUC / antonym_auc | "hypernym 0.72" / - / "+0.005" | 0.710 / 0.831 / 0.222 | same |
| 3k raw vs tanh normalisation, held-out syn>ant (M20) | raw 0.673, tanh **0.756** (honest) | raw 0.693, center 0.644, standardize 0.670, minmax 0.717, **tanh 0.775**; random cos raw 0.265, tanh 0.156 | `python archive/scripts/probe_normalize.py --n 3000` |
| SimLex-999 rho, 738 core-core pairs, cosine + antonym store (M30) | 0.417 | **0.417** | `python archive/scripts/probe_simlex.py` |
| SimLex rho all 999, plain cosine / with antonym edges | 0.333 / 0.380 | 0.352 / 0.394 (the live artifact now has M46 sense layer + extended lexicon, so non-core pairs use MFS sense vectors) | same |

Reading: the harness runs and reproduces the ledger to within its own jitter band, except (i) similar AUC +0.011 and the tanh figure 0.775 vs 0.756 (the M20 number came from a different split/axis set; I did not hunt for the exact configuration), and (ii) the 'syn>ant 0.756 / minimality curve peaking K~80-120' of M20.1 (`probe_minimality.py`, leave-one-out) which I did **not** re-run (see section 10). The `dimensionality_spectrum` history numbers (~46 / ~17) are reproduced in section 4.

## 2. Content-word placement at both levels


### 2a. Cosine distributions (production artifact)

Pair definitions are in section 0. `raw` = plain cosine; `idf` = the scorer's weighted cosine (`sum w*a*b / sqrt(sum w*a^2 * sum w*b^2)`, `w = log((N+1)/(df+1))`). **Core-level numbers on the production artifact are in-sample** (the core was placed by propagating over all synonym/similar/also_see edges, i.e. over these very pairs); held-out core numbers follow in 2a'. Sense-level antonym pairs use the exact antonym synsets; sense-level `syn` is MFS(a) vs MFS(b) of a shared-synset lemma pair (pairs that share the synset at MFS have identical vectors and are excluded, n=6,871 of 7,480).

| level|weighting | synonym mean/median | antonym (direct) mean/median | hypernym-hyponym | co-hyponym (sisters) | random |
|---|---|---|---|---|---|
| core|raw | 0.715 / 0.716 | 0.501 / 0.507 | 0.458 / 0.464 | 0.432 / 0.433 | 0.280 / 0.287 |
| core|idf | 0.568 / 0.539 | 0.231 / 0.224 | 0.231 / 0.147 | 0.210 / 0.127 | 0.076 / 0.041 |
| sense|raw | 0.514 / 0.496 | 0.859 / 0.892 | 0.741 / 0.763 | 0.860 / 0.910 | 0.397 / 0.412 |
| sense|idf | 0.216 / 0.134 | 0.712 / 0.751 | 0.616 / 0.645 | 0.781 / 0.854 | 0.111 / 0.070 |

| level|weighting | AUC syn>random | AUC syn>ant (P cos_syn>cos_ant) | AUC ant<random (P ant less similar than random) | **(b)** P(antonym pair closer than a random synonym pair) | AUC hyp>random | AUC sister>random | AUC hyp>sister |
|---|---|---|---|---|---|---|---|
| core|raw | 0.955 | 0.764 | 0.237 | 0.236 | 0.727 | 0.700 | 0.537 |
| core|idf | 0.978 | 0.750 | 0.271 | 0.250 | 0.763 | 0.745 | 0.528 |
| sense|raw | 0.689 | 0.068 | 0.017 | 0.932 | 0.949 | 0.979 | 0.229 |
| sense|idf | 0.671 | 0.074 | 0.020 | 0.926 | 0.971 | 0.986 | 0.251 |

Extra: core-level with the project's own (morphy-expanded) synonym edge set (n=22,856, same as the ledger's edge count): raw mean 0.710, AUC>random 0.952, AUC>ant 0.755. Sense-level, all 3,992 direct antonym lemma pointers (not restricted to core words): mean cos raw 0.850 / idf 0.732; P(syn>ant) 0.078 / 0.067.

**Sense-level near-synonym control** (because same-synset synonyms share one vector, the MFS-pair 'synonym' set above is not a clean synonym test). WordNet `similar_to` satellite->head pairs (n=10,693), the closest sense-level synonymy WordNet records: mean cos raw 0.755 / idf 0.459; random adjective-synset pair mean 0.581 / 0.258; AUC>random-adjective 0.774 / 0.781; antonym mean 0.859 / 0.712; P(similar_to pair more similar than an antonym pair) = 0.255 (raw) / 0.207 (idf), i.e. an antonym pair is closer than a near-synonym pair in 0.745 / 0.793 of draws.

### 2a'. Core level on genuinely held-out pairs (M24 rule; placement re-run with 50% of synonym/similar pairs withheld, 10k winners set)

| weighting | mean syn | mean ant | mean is-a | mean random | AUC syn>rand | AUC syn>ant | AUC ant<rand | AUC is-a>rand |
|---|---|---|---|---|---|---|---|---|
| raw | 0.583 | 0.452 | 0.525 | 0.236 | 0.884 | 0.673 | 0.222 | 0.832 |
| idf | 0.392 | 0.201 | 0.349 | 0.059 | 0.920 | 0.690 | 0.237 | 0.874 |

n: test syn 11,387, test ant 814, test is-a 2,717, test similar 2021, random 4,000. Cross-check: the harness' own `discrimination()` on this placement returns 0.6731 (n_syn 11,387, n_ant 814).

### 2a''. By part of speech (production artifact; POS from the pair set; random pairs POS-matched)

| level|weighting | POS | n syn | n ant | syn mean | ant mean | rand mean (POS-matched) | AUC syn>rand | AUC syn>ant | AUC ant<rand | P(ant closer than syn) | AUC hyp>rand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| core|raw | n | 3460 | 136 | 0.761 | 0.491 | 0.262 | 0.971 | 0.809 | 0.244 | 0.191 | 0.735 |
| core|raw | v | 2347 | 101 | 0.652 | 0.494 | 0.442 | 0.831 | 0.782 | 0.415 | 0.218 | 0.534 |
| core|raw | a | 1498 | 462 | 0.698 | 0.495 | 0.425 | 0.851 | 0.738 | 0.399 | 0.262 | - |
| core|raw | r | 175 | 38 | 0.812 | 0.631 | 0.566 | 0.814 | 0.775 | 0.403 | 0.225 | - |
| core|idf | n | 3460 | 136 | 0.635 | 0.282 | 0.079 | 0.979 | 0.788 | 0.263 | 0.212 | 0.758 |
| core|idf | v | 2347 | 101 | 0.464 | 0.194 | 0.151 | 0.914 | 0.858 | 0.399 | 0.142 | 0.600 |
| core|idf | a | 1498 | 462 | 0.551 | 0.197 | 0.150 | 0.923 | 0.728 | 0.382 | 0.272 | - |
| core|idf | r | 175 | 38 | 0.767 | 0.558 | 0.677 | 0.622 | 0.706 | 0.617 | 0.294 | - |
| sense|raw | n | 3039 | 136 | 0.529 | 0.906 | 0.398 | 0.703 | 0.046 | 0.012 | 0.954 | 0.940 |
| sense|raw | v | 2325 | 101 | 0.478 | 0.804 | 0.442 | 0.586 | 0.065 | 0.042 | 0.935 | 0.921 |
| sense|raw | a | 1370 | 462 | 0.536 | 0.859 | 0.581 | 0.422 | 0.080 | 0.097 | 0.920 | - |
| sense|raw | r | 137 | 38 | 0.564 | 0.842 | 0.684 | 0.322 | 0.127 | 0.231 | 0.873 | - |
| sense|idf | n | 3039 | 136 | 0.244 | 0.839 | 0.127 | 0.682 | 0.043 | 0.012 | 0.958 | 0.960 |
| sense|idf | v | 2325 | 101 | 0.166 | 0.739 | 0.132 | 0.637 | 0.049 | 0.042 | 0.951 | 0.938 |
| sense|idf | a | 1370 | 462 | 0.228 | 0.660 | 0.258 | 0.435 | 0.089 | 0.097 | 0.911 | - |
| sense|idf | r | 137 | 38 | 0.334 | 0.815 | 0.673 | 0.198 | 0.098 | 0.275 | 0.902 | - |

Hypernym/sister POS split (mean cosine): core raw hyp n/v 0.455/0.463, sister n/v 0.434/0.427; sense raw hyp n/v 0.740/0.749, sister n/v 0.865/0.831.

### 2c. Forty concrete pairs (cosine; core = placed core coordinate, sense = first WordNet sense of the stated POS; `-` = word not a core word)

| pair | core raw | core idf | sense raw | sense idf | senses compared |
|---|---|---|---|---|---|
| happy/unhappy (a) | - | - | 0.89 | 0.90 | happy.a.01 / unhappy.a.01 |
| happy/sad (a) | 0.79 | 0.31 | 0.85 | 0.88 | happy.a.01 / sad.a.01 |
| hot/cold (a) | -0.01 | -0.70 | 0.92 | 0.75 | hot.a.01 / cold.a.01 |
| big/small (a) | 0.49 | 0.33 | 0.99 | 0.97 | large.a.01 / small.a.01 |
| good/bad (a) | 0.20 | -0.17 | 0.86 | 0.74 | good.a.01 / bad.a.01 |
| love/hate (v) | - | - | 0.82 | 0.87 | love.v.01 / hate.v.01 |
| up/down (r) | - | - | 0.99 | 1.00 | up.r.01 / down.r.01 |
| open/closed (a) | 0.57 | 0.27 | 0.88 | 0.58 | open.a.01 / closed.a.01 |
| alive/dead (a) | 0.73 | 0.93 | 0.88 | 0.55 | alive.a.01 / dead.a.01 |
| rich/poor (a) | 0.33 | -0.44 | 0.62 | 0.24 | rich.a.01 / hapless.s.01 |
| win/lose (v) | 0.59 | 0.27 | 0.23 | 0.01 | win.v.01 / lose.v.01 |
| buy/sell (v) | 0.54 | 0.59 | 0.84 | 0.80 | buy.v.01 / sell.v.01 |
| give/take (v) | 0.40 | 0.11 | 0.42 | 0.09 | give.v.01 / take.v.01 |
| dog/cat (n) | 0.52 | 0.75 | 0.84 | 0.83 | dog.n.01 / cat.n.01 |
| dog/bone (n) | 0.34 | 0.13 | 0.64 | 0.40 | dog.n.01 / bone.n.01 |
| bone/stone (n) | 0.20 | 0.04 | 0.47 | 0.03 | bone.n.01 / rock.n.01 |
| eat/chew (v) | 0.38 | 0.26 | 0.77 | 0.83 | eat.v.01 / chew.v.01 |
| eat/drink (v) | 0.24 | 0.15 | 0.87 | 0.82 | eat.v.01 / drink.v.01 |
| run/walk (v) | 0.59 | 0.37 | 0.89 | 0.89 | run.v.01 / walk.v.01 |
| king/queen (n) | 0.81 | 0.83 | 0.26 | 0.03 | king.n.01 / queen.n.01 |
| man/woman (n) | 0.64 | 0.64 | 0.99 | 0.99 | man.n.01 / woman.n.01 |
| boy/girl (n) | 0.73 | 0.35 | 0.44 | 0.29 | male_child.n.01 / girl.n.01 |
| day/night (n) | 0.61 | 0.40 | 0.55 | 0.40 | day.n.01 / night.n.01 |
| red/blue (a) | 0.75 | 0.51 | 0.81 | 0.46 | red.s.01 / blue.s.01 |
| red/color (n) | 0.67 | 0.35 | 0.86 | 0.66 | red.n.01 / color.n.01 |
| apple/fruit (n) | 0.37 | 0.08 | 0.59 | 0.18 | apple.n.01 / fruit.n.01 |
| apple/pear (n) | 0.35 | 0.93 | 0.87 | 0.86 | apple.n.01 / pear.n.01 |
| car/vehicle (n) | 0.86 | 0.54 | 0.89 | 0.68 | car.n.01 / vehicle.n.01 |
| happy/joyful (a) | 0.97 | 0.95 | 0.90 | 0.83 | happy.a.01 / joyful.a.01 |
| big/large (a) | 0.72 | 0.72 | 1.00 | 1.00 | large.a.01 / large.a.01 |
| quick/fast (a) | 0.65 | 0.48 | 0.94 | 0.67 | quick.s.01 / fast.a.01 |
| dog/puppy (n) | - | - | 0.89 | 0.89 | dog.n.01 / puppy.n.01 |
| hot/warm (a) | 0.78 | 0.84 | 0.91 | 0.61 | hot.a.01 / warm.a.01 |
| tall/short (a) | 0.65 | 0.62 | 0.88 | 0.52 | tall.a.01 / short.a.01 |
| old/young (a) | 0.69 | 0.53 | 0.90 | 0.57 | old.a.01 / young.a.01 |
| fast/slow (a) | 0.67 | 0.81 | 0.94 | 0.61 | fast.a.01 / slow.a.01 |
| true/false (a) | -0.04 | -0.83 | 0.87 | 0.86 | true.a.01 / false.a.01 |
| begin/end (v) | 0.50 | 0.08 | 0.27 | 0.02 | get_down.v.07 / end.v.01 |
| dog/justice (n) | 0.55 | 0.06 | 0.47 | 0.05 | dog.n.01 / justice.n.01 |
| table/run (n) | 0.19 | 0.05 | 0.52 | 0.19 | table.n.01 / run.n.01 |

(4 of the 40 pairs have a word outside the 9,946-word core, so the core columns are empty for them.) Note `big/large`: the first adjective sense of both is `large.a.01`, so the sense cosine is 1.0 by identity. `rich/poor`: the first adjective sense of `poor` is `hapless.s.01`, not the 'lacking money' sense, so that sense-level row compares different senses than the reader's intuition.

### 2d. Sign-opposition and where the sign is lost

| core-level pair set | n | frac with ANY axis where a*b<0 | frac with an axis where a*b < -0.01 | mean # opposed axes | frac cos<0 | mean cos |
|---|---|---|---|---|---|---|
| direct_ant_737 | 737 | 0.733 | 0.168 | 6.66 | 0.0353 | 0.501 |
| tiered_default_ant | 3383 | 0.865 | 0.174 | 8.61 | 0.0089 | 0.482 |
| tiered_all_ant | 11317 | 0.873 | 0.115 | 8.66 | 0.0037 | 0.476 |
| syn | 7480 | 0.239 | 0.009 | 0.44 | 0.0001 | 0.715 |
| rand | 10000 | 0.301 | 0.005 | 0.86 | 0.0002 | 0.281 |

Sense level: minimum value in the whole sense matrix = 0.0, so **0 of 737** direct antonym pairs (and 0 of any pairs) are sign-opposed on any axis. Core level: min -1.0, max 1.0.

Counterfactual (reconstructed, NOT the shipped artifact; `/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag/m4_signs_pairs.py`): rebuilding `sense_usvs_weights`' blend before the top-24/clip step on 3,944 random senses, 97.9% of blends contain a negative entry, but negative mass is only 1.12% of total absolute mass, and 100% of it sits on `attr:` axes. With signs retained and no sparsification the 732 antonym synset pairs would have mean cosine 0.816 vs 0.495 for random sense pairs; 40% would be opposed on some axis vs 54% for random pairs; AUC(antonym less similar than random) 0.093. I.e. the sign that is dropped is tiny relative to the shared structure; keeping it alone would not separate opposites.

Code locations of the sign path: core signs are created only in `placement.anchored_coordinate` (`src/nsm_ct/ground/placement.py:37-48`: `sign = gloss_polarity(w)` then `M[r, j] = +-1` on the word's `attr:<dimension>` axes only), where `gloss_polarity` (`src/nsm_ct/ground/polarity.py:77-100`, cue sets at 48-62) counts words from two hand-written English cue lists in the word's first gloss. Propagation (`placement.relax`, alpha 0.7, 20 iterations) then spreads them. Signs are dropped at the sense level in `sense_usvs_weights` (`src/nsm_ct/ground/usvs.py:306-315`): `cut = np.partition(blend, -top_k)[-top_k]; blend = np.where(blend >= max(cut, 1e-9), blend, 0.0)` zeroes every entry below the 24th-largest value and every negative (the cut is floored at +1e-9), then `blend / blend.max()` (line 315) and `row[j]=1.0` for the `lex:` axis (lines 674-676). `save_usvs` (745-766) stores `sense_axis_val` float32 sparse, so the artifact format has no negative storage in practice but would accept it.

## 3. Axis usage

### 3.1 Per-axis table

The full per-axis table (607 rows: kind, fraction of senses, fraction of core words, mean weight, IDF) is in `/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag/m3_axes_eigen.json` (`axis_rows`); summary below.

| kind | axes | axes with no sense | axes with no core word | mean frac of senses | mean frac of core words (|v|>1e-6) |
|---|---|---|---|---|---|
| prime | 65 | 26 | 26 | 0.1607 | 0.409 |
| attr | 498 | 21 | 0 | 0.0105 | 0.610 |
| lex | 44 | 0 | 0 | 0.1819 | 0.673 |

**Near-constant axes (loaded by >50% of the 117,659 senses)** with IDF weight:

| axis | frac senses | frac core (>1e-6) | mean weight when present | IDF |
|---|---|---|---|---|
| SOMETHING | 0.984 | 0.806 | 0.858 | 0.016 |
| ONE | 0.974 | 0.748 | 0.374 | 0.027 |
| PART | 0.972 | 0.727 | 0.329 | 0.029 |
| BODY | 0.840 | 0.722 | 0.240 | 0.174 |
| WHEN | 0.824 | 0.700 | 0.201 | 0.193 |
| KIND | 0.805 | 0.700 | 0.210 | 0.216 |
| lex:noun.act | 0.788 | 0.681 | 0.228 | 0.238 |
| PEOPLE | 0.760 | 0.698 | 0.198 | 0.274 |
| WHERE | 0.678 | 0.693 | 0.167 | 0.389 |
| lex:adj.all | 0.652 | 0.677 | 0.374 | 0.427 |
| lex:noun.artifact | 0.625 | 0.702 | 0.289 | 0.470 |
| lex:noun.communication | 0.574 | 0.685 | 0.226 | 0.556 |

Between 25% and 50% of senses: SOMEONE 0.49, lex:noun.person 0.48, lex:noun.cognition 0.44, MUCH_MANY 0.43, MOVE 0.41, lex:noun.attribute 0.37, TWO 0.36, lex:noun.location 0.29, lex:noun.group 0.29, ALL 0.28, attr:law 0.27, SMALL 0.26, attr:colloquialism 0.25.

**Empty axes**: 47 axes have no sense on them: the 26 prime axes I, YOU, THIS, THE_SAME, SOME, DONT_WANT, DO, BE_SOMEWHERE, THERE_IS, BE_SOMEONE_SOMETHING, MINE, BEFORE, AFTER, A_LONG_TIME, A_SHORT_TIME, FOR_SOME_TIME, ABOVE, BELOW, NEAR, NOT, MAYBE, CAN, BECAUSE, IF, VERY, MORE (also empty in core coordinates) and 21 `attr:` axes (attr:microbiology, attr:assurance, attr:authority, attr:certainty, attr:self-assurance, attr:self-confidence, attr:sureness, attr:bus, attr:train, attr:foregone_conclusion, attr:sure_thing, attr:evenness, attr:invariability, attr:ease, attr:easiness, attr:simpleness, attr:simplicity, attr:passiveness, attr:passivity, attr:chromaticity, attr:hue), which are populated only in core coordinates. A further 22 axes have 1-50 senses (e.g. MOMENT 44, attr:confidence 37, attr:superlative 50, attr:drug 20, attr:microorganism 41, attr:ceramics 40). (The 26 empty prime axes are exactly the ones the 2026-11-15 closed-class lexicon now fills for function-word senses; for content words they stay empty.)

**Topic-like axes**: of the 498 `attr:` axes, 249 come from WordNet's adjective->noun `attribute` relation (gradable dimensions such as temperature, quality, size; mean fraction of senses 0.39%) and 246 from topic/region/usage domains (law, music, chemistry, united_kingdom, slang, ...; mean fraction of senses 1.69%), 3 from both (classification: `/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag/m3_axes_eigen.py`, using `nsm_ct.wordnet.attributes` / `domains` over the core words). The 44 `lex:` axes are WordNet lexicographer categories (noun.animal, verb.motion, ...) and are the broadest topic-like axes: all senses have exactly one at weight 1.0 (`usvs.py:674-676`). Mean number of `lex:` axes per sense 8.0 (median 8): senses also receive `lex:` weight through the blended core coordinates of their genus and gloss words.

### 3.2 Axes per sense / per core word

Senses: mean 23.67; percentiles 0/5/25/50/75/95/100 = 0/24/24/24/24/25/30 (top-24 cap; 0.54% of senses have no prime axis). By kind (mean/median/p95): prime 10.4/11/13, attr 5.2/5/9, lex 8.0/8/11.

Core words (9,946; non-zero means |v|>1e-6): the distribution is **bimodal**: percentiles 0/5/25/50/75/95/100 = 1/2/4/537/549/554/558. 33.0% of core words (3,286) have at most 10 non-zero axes (nouns 41%, adverbs 71%, adjectives 15%, verbs 6% of each POS group; POS = first sense; examples: water, city, river, food, law, many); 66.1% have at least 400. For the dense words the ~540 non-zero axes are mostly propagation dust: median non-zero magnitude 9.2e-05, 99th percentile 0.15; axes with |v|>0.1 per dense word: mean 7.2 (p5 4, p95 12); |v|>0.3: mean 3.8. Random pair cosine between two dense words 0.40, two sparse words 0.12, one of each 0.20. Of the 737 core direct antonym pairs, 126 have at least one sparse member.

### 3.3 Share of the squared norm by axis kind

| vectors|weighting | prime axes | attr axes | lex axes |
|---|---|---|---|
| sense|raw | 50.5% (median 51.7%, pooled 51.6%) | 5.5% (median 2.7%, pooled 6.1%) | 44.0% (median 42.8%, pooled 42.2%) |
| sense|idf | 16.3% (median 11.5%, pooled 16.4%) | 18.5% (median 10.9%, pooled 21.9%) | 65.3% (median 68.4%, pooled 61.6%) |
| core|raw | 56.7% (median 60.5%, pooled 59.4%) | 10.4% (median 3.0%, pooled 11.6%) | 32.9% (median 29.2%, pooled 29.0%) |
| core|idf | 22.5% (median 14.4%, pooled 21.9%) | 29.0% (median 15.5%, pooled 41.7%) | 48.5% (median 45.6%, pooled 36.4%) |

Within `attr`: domain-type axes take 4.8% (raw) / 15.3% (IDF) of the average sense's squared norm, relation-type (gradable) axes 0.6% / 2.9%.

### 3.4 Near-duplicate axes (column correlation > 0.9)

Over core words: 221 axis pairs with Pearson r>0.9 (252 with r>0.8) among 581 axes with variance. Over senses: 193 pairs (r>0.8: 217) among 560. Top 60 core pairs (they are all `attr:` axes derived from WordNet synonymous noun lemmas of the same synset, e.g. speed/swiftness/fastness, plus a Confederate-states cluster):

| axis A | axis B | r |
|---|---|---|
| attr:bus | attr:train | 1.000 |
| attr:caliber | attr:calibre | 1.000 |
| attr:individualism | attr:individuality | 1.000 |
| attr:individuality | attr:individuation | 1.000 |
| attr:individualism | attr:individuation | 1.000 |
| attr:matureness | attr:maturity | 1.000 |
| attr:curliness | attr:waviness | 1.000 |
| attr:fastness | attr:speed | 1.000 |
| attr:fastness | attr:swiftness | 1.000 |
| attr:speed | attr:swiftness | 1.000 |
| attr:south | attr:confederate_states_of_america | 1.000 |
| attr:confederacy | attr:confederate_states | 1.000 |
| attr:confederate_states_of_america | attr:dixie | 1.000 |
| attr:dixie | attr:dixieland | 1.000 |
| attr:confederate_states_of_america | attr:dixieland | 1.000 |
| attr:confederate_states | attr:dixie | 1.000 |
| attr:south | attr:dixie | 1.000 |
| attr:south | attr:confederacy | 1.000 |
| attr:south | attr:confederate_states | 1.000 |
| attr:confederate_states | attr:confederate_states_of_america | 1.000 |
| attr:confederate_states | attr:dixieland | 1.000 |
| attr:south | attr:dixieland | 1.000 |
| attr:confederacy | attr:dixie | 1.000 |
| attr:confederacy | attr:confederate_states_of_america | 1.000 |
| attr:confederacy | attr:dixieland | 1.000 |
| attr:coloring | attr:colouring | 1.000 |
| attr:ideology | attr:political_orientation | 1.000 |
| attr:political_orientation | attr:political_theory | 1.000 |
| attr:ideology | attr:political_theory | 1.000 |
| attr:conventionalism | attr:conventionality | 1.000 |
| attr:convention | attr:conventionalism | 1.000 |
| attr:deepness | attr:depth | 1.000 |
| attr:convention | attr:conventionality | 1.000 |
| attr:independence | attr:independency | 1.000 |
| attr:emotionalism | attr:emotionality | 1.000 |
| attr:normalcy | attr:normality | 1.000 |
| attr:usefulness | attr:utility | 1.000 |
| attr:correctitude | attr:propriety | 1.000 |
| attr:properness | attr:propriety | 1.000 |
| attr:correctitude | attr:properness | 1.000 |
| attr:luminosity | attr:luminousness | 1.000 |
| attr:brightness | attr:brightness_level | 1.000 |
| attr:luminance | attr:luminosity | 1.000 |
| attr:luminance | attr:luminousness | 1.000 |
| attr:brightness | attr:luminousness | 1.000 |
| attr:brightness | attr:luminosity | 1.000 |
| attr:brightness_level | attr:luminance | 1.000 |
| attr:brightness_level | attr:luminousness | 1.000 |
| attr:brightness_level | attr:luminosity | 1.000 |
| attr:brightness | attr:luminance | 1.000 |
| attr:intensity | attr:volume | 1.000 |
| attr:loudness | attr:volume | 1.000 |
| attr:intensity | attr:loudness | 1.000 |
| attr:limpidity | attr:pellucidity | 1.000 |
| attr:clarity | attr:limpidity | 1.000 |
| attr:clarity | attr:clearness | 1.000 |
| attr:clearness | attr:lucidness | 1.000 |
| attr:limpidity | attr:lucidity | 1.000 |
| attr:limpidity | attr:lucidness | 1.000 |
| attr:clarity | attr:pellucidity | 1.000 |

## 4. Effective dimensionality

| matrix | participation ratio | components for 90% | 95% | 99% | top-1 share | top-5 share |
|---|---|---|---|---|---|---|
| core|raw|uncentered | 8.2 | 64 | 106 | 246 | 0.333 | 0.476 |
| core|raw|centered | 32.3 | 80 | 131 | 270 | 0.114 | 0.298 |
| core|idf|uncentered | 91.2 | 194 | 259 | 374 | 0.061 | 0.148 |
| core|idf|centered | 122.2 | 199 | 263 | 376 | 0.028 | 0.111 |
| sense|raw|uncentered | 5.4 | 43 | 66 | 162 | 0.419 | 0.548 |
| sense|raw|centered | 38.6 | 59 | 87 | 203 | 0.064 | 0.263 |
| sense|idf|uncentered | 40.4 | 115 | 177 | 310 | 0.107 | 0.262 |
| sense|idf|centered | 59.4 | 124 | 186 | 316 | 0.056 | 0.206 |

Unit-row (cosine geometry) variants, in the JSON file. Definitions: eigenvalues of X^T X (squared singular values, as in the project's `axes.dimensionality_spectrum`); participation ratio = (sum lam)^2 / sum lam^2; 'centered' subtracts the column mean first; `idf` multiplies columns by sqrt(w). Note the uncentered spectra are dominated by one constant direction (top-1 share 33% core / 42% senses) which is why the uncentered PR (8.2 / 5.4) is so small; centered PR is 32.3 / 38.6.

**Reproduction of the history's number.** `axes.dimensionality_spectrum` (anchored feature matrix before propagation, first 3,000 words), 3k vocab / 234 axes: intrinsic_dim_mass (90%) **46**, participation ratio **17.31** (history: ~46 axes, ~17 effective; matches). The same function on the shipped 10k vocabulary with the 607-axis set: 101 / 25.29 (3,000 words), 84 / 22.35 (all 9,946 words). (`m3b_hist_dim.py`.) So the anchored-feature effective dimensionality grew from ~17 to ~22-25 and the 90% count from 46 to 84-101 when the axis set grew from 234 to 607; on the *placed* core the PR is 8.2 uncentered / 32 centered and on *sense vectors* 5.4 / 38.6 (IDF-centered 59). The history's 'fidelity peak K~80-120' is a different quantity (held-out syn>ant by leave-one-out axis contribution on the 234-axis space) and was not re-run (see 'not measured').

### 4.2 Keeping only the top-K axes by IDF-weighted energy

Axes ranked by `idf_j * sum_i x_ij^2` over that level's matrix. Held-out core (re-placed, section 2a') and production sense vectors, IDF-weighted cosine unless stated:

| setting | idf AUC syn>rand | idf AUC syn>ant | idf AUC ant<rand | idf mean random cos | raw AUC syn>rand | raw AUC syn>ant |
|---|---|---|---|---|---|---|
| held-out core K=20 | 0.858 | 0.633 | 0.226 | 0.136 | 0.854 | 0.618 |
| held-out core K=50 | 0.893 | 0.651 | 0.190 | 0.086 | 0.886 | 0.632 |
| held-out core K=100 | 0.912 | 0.652 | 0.176 | 0.077 | 0.879 | 0.637 |
| held-out core K=200 | 0.916 | 0.672 | 0.188 | 0.068 | 0.883 | 0.650 |
| held-out core K=607 | 0.920 | 0.690 | 0.237 | 0.059 | 0.884 | 0.673 |

| setting | kept prime/attr/lex | idf AUC syn>rand | idf AUC syn>ant | idf AUC ant<rand | idf AUC hyp>rand | idf AUC sister>rand | idf mean random | raw AUC syn>rand | raw AUC syn>ant |
|---|---|---|---|---|---|---|---|---|---|
| production core K=20 | 8/0/12 | 0.937 | 0.677 | 0.191 | 0.713 | 0.689 | 0.180 | 0.932 | 0.648 |
| production core K=50 | 14/7/29 | 0.957 | 0.695 | 0.160 | 0.758 | 0.737 | 0.116 | 0.948 | 0.671 |
| production core K=100 | 28/33/39 | 0.970 | 0.718 | 0.172 | 0.767 | 0.751 | 0.099 | 0.948 | 0.706 |
| production core K=200 | 36/121/43 | 0.974 | 0.737 | 0.187 | 0.758 | 0.741 | 0.088 | 0.955 | 0.729 |
| production core K=400 | 37/319/44 | 0.977 | 0.755 | 0.233 | 0.760 | 0.741 | 0.079 | 0.955 | 0.750 |
| production core K=607 | 65/498/44 | 0.978 | 0.750 | 0.271 | 0.763 | 0.745 | 0.076 | 0.955 | 0.764 |
| production sense K=20 | 0/0/20 | 0.659 | 0.116 | 0.052 | 0.912 | 0.937 | 0.156 | 0.651 | 0.099 |
| production sense K=50 | 11/1/38 | 0.668 | 0.077 | 0.021 | 0.970 | 0.980 | 0.125 | 0.666 | 0.065 |
| production sense K=100 | 30/28/42 | 0.669 | 0.070 | 0.018 | 0.972 | 0.985 | 0.126 | 0.686 | 0.065 |
| production sense K=200 | 34/122/44 | 0.670 | 0.068 | 0.018 | 0.971 | 0.986 | 0.117 | 0.689 | 0.066 |
| production sense K=400 | 38/318/44 | 0.671 | 0.071 | 0.019 | 0.971 | 0.986 | 0.112 | 0.689 | 0.067 |
| production sense K=607 | 65/498/44 | 0.671 | 0.074 | 0.020 | 0.971 | 0.986 | 0.111 | 0.689 | 0.068 |

The top-30 axes by IDF energy: core: lex:noun.animal, SMALL, lex:noun.person, lex:noun.plant, lex:noun.body, lex:noun.substance, lex:noun.food, lex:adv.all, TWO, lex:noun.object, lex:noun.state, LIVE, lex:adj.all, MOVE, LIKE, lex:noun.artifact, lex:noun.group, ALL, MUCH_MANY, SOMEONE, lex:noun.communication, lex:noun.feeling, lex:noun.location, lex:noun.cognition, OTHER, lex:noun.quantity, TRUE, lex:verb.change, lex:noun.time, attr:cooking. Senses: lex:noun.plant, lex:noun.animal, lex:adv.all, lex:noun.person, lex:adj.pert, lex:adj.all, lex:noun.food, lex:noun.substance, lex:noun.state, lex:noun.location, lex:noun.artifact, lex:noun.body, lex:noun.attribute, lex:noun.group, lex:noun.object, lex:noun.cognition, lex:noun.Tops, lex:verb.contact, lex:noun.communication, lex:verb.change, SMALL, lex:noun.quantity, lex:verb.motion, LIVE, LIKE, lex:verb.communication, MOVE, lex:verb.cognition, lex:noun.time, attr:biology. (Production core rows are in-sample; sense `syn` is the MFS-pair set, so its AUCs are weak by construction; `syn>ant` at sense level never exceeds 0.23 at any K.)

## 5. Where the discriminating information lives (by axis kind)

| setting | AUC syn>rand | AUC syn>ant | AUC ant<rand | AUC is-a>rand | mean random cos |
|---|---|---|---|---|---|
| held-out core|raw|prime | 0.826 | 0.658 | 0.271 | 0.687 | 0.361 |
| held-out core|idf|prime | 0.854 | 0.697 | 0.280 | 0.700 | 0.165 |
| held-out core|raw|attr | 0.884 | 0.722 | 0.287 | 0.621 | 0.028 |
| held-out core|idf|attr | 0.889 | 0.729 | 0.290 | 0.623 | 0.021 |
| held-out core|raw|lex | 0.890 | 0.586 | 0.159 | 0.855 | 0.101 |
| held-out core|idf|lex | 0.897 | 0.600 | 0.156 | 0.860 | 0.086 |
| held-out core|raw|prime+attr | 0.825 | 0.685 | 0.312 | 0.683 | 0.300 |
| held-out core|idf|prime+attr | 0.887 | 0.724 | 0.322 | 0.687 | 0.071 |
| held-out core|raw|prime+lex | 0.884 | 0.636 | 0.183 | 0.839 | 0.267 |
| held-out core|idf|prime+lex | 0.904 | 0.632 | 0.164 | 0.881 | 0.091 |
| held-out core|raw|attr+lex | 0.905 | 0.650 | 0.207 | 0.855 | 0.078 |
| held-out core|idf|attr+lex | 0.918 | 0.680 | 0.226 | 0.857 | 0.050 |
| held-out core|raw|ALL | 0.884 | 0.673 | 0.222 | 0.832 | 0.236 |
| held-out core|idf|ALL | 0.920 | 0.690 | 0.237 | 0.874 | 0.059 |

| setting | syn mean | ant mean | rand mean | AUC syn>rand | AUC syn>ant | AUC ant<rand | AUC hyp>rand | AUC sister>rand |
|---|---|---|---|---|---|---|---|---|
| production sense|raw|prime | 0.751 | 0.884 | 0.651 | 0.649 | 0.213 | 0.132 | 0.673 | 0.863 |
| production sense|idf|prime | 0.363 | 0.646 | 0.281 | 0.609 | 0.228 | 0.164 | 0.742 | 0.868 |
| production sense|raw|attr | 0.128 | 0.483 | 0.060 | 0.608 | 0.224 | 0.164 | 0.687 | 0.808 |
| production sense|idf|attr | 0.102 | 0.454 | 0.044 | 0.608 | 0.224 | 0.163 | 0.691 | 0.810 |
| production sense|raw|lex | 0.308 | 0.879 | 0.163 | 0.664 | 0.069 | 0.016 | 0.970 | 0.979 |
| production sense|idf|lex | 0.256 | 0.811 | 0.121 | 0.667 | 0.079 | 0.021 | 0.971 | 0.978 |
| production sense|raw|prime+attr | 0.686 | 0.850 | 0.588 | 0.646 | 0.195 | 0.121 | 0.687 | 0.886 |
| production sense|idf|prime+attr | 0.197 | 0.547 | 0.137 | 0.613 | 0.176 | 0.118 | 0.775 | 0.909 |
| production sense|raw|prime+lex | 0.539 | 0.878 | 0.418 | 0.690 | 0.069 | 0.017 | 0.950 | 0.978 |
| production sense|idf|prime+lex | 0.257 | 0.773 | 0.135 | 0.666 | 0.075 | 0.019 | 0.972 | 0.982 |
| production sense|raw|attr+lex | 0.285 | 0.843 | 0.149 | 0.668 | 0.062 | 0.015 | 0.971 | 0.983 |
| production sense|idf|attr+lex | 0.209 | 0.729 | 0.096 | 0.675 | 0.075 | 0.021 | 0.970 | 0.984 |
| production core|raw|prime | 0.822 | 0.662 | 0.423 | 0.890 | 0.723 | 0.266 | 0.655 | 0.629 |
| production core|idf|prime | 0.695 | 0.403 | 0.208 | 0.935 | 0.794 | 0.280 | 0.662 | 0.628 |
| production core|raw|attr | 0.644 | 0.131 | 0.049 | 0.963 | 0.822 | 0.365 | 0.660 | 0.661 |
| production core|idf|attr | 0.636 | 0.112 | 0.038 | 0.969 | 0.824 | 0.369 | 0.662 | 0.663 |
| production core|raw|lex | 0.616 | 0.519 | 0.121 | 0.947 | 0.621 | 0.121 | 0.780 | 0.772 |
| production core|idf|lex | 0.604 | 0.475 | 0.103 | 0.953 | 0.650 | 0.124 | 0.782 | 0.772 |
| production core|raw|prime+attr | 0.751 | 0.490 | 0.353 | 0.902 | 0.786 | 0.345 | 0.634 | 0.609 |
| production core|idf|prime+attr | 0.600 | 0.169 | 0.091 | 0.968 | 0.800 | 0.356 | 0.645 | 0.624 |
| production core|raw|prime+lex | 0.755 | 0.618 | 0.315 | 0.954 | 0.702 | 0.161 | 0.745 | 0.719 |
| production core|idf|prime+lex | 0.595 | 0.418 | 0.113 | 0.965 | 0.706 | 0.136 | 0.778 | 0.760 |
| production core|raw|attr+lex | 0.593 | 0.357 | 0.097 | 0.959 | 0.696 | 0.234 | 0.774 | 0.767 |
| production core|idf|attr+lex | 0.575 | 0.233 | 0.064 | 0.973 | 0.735 | 0.265 | 0.771 | 0.761 |

Measured reading of these tables is in section 9 (interpretation is kept separate).

## 6. Coverage of relations (counts only; WordNet 3.0 via nltk; script `m6_relations.py`)

- Synsets in WordNet 117,659 (n 82,115, v 13,767, adjective head `a` 7,463, adjective satellite `s` 10,693, r 3,621); USVS has 117,679 senses (= WordNet + 20 interjections).
- **Direct antonym pointers** (lemma-level, `Lemma.antonyms()`): directed pointers by POS a 4,024, r 710, n 2,152, v 1,093. Unordered lemma pairs: a 2,012, r 355, n 1,076, v 549; total **3,992**. Distinct synset pairs: a 1,999, r 320, n 975, v 508. **Distinct synsets involved: 7,394** = 6.3% of all synsets (a 3,846 = 51.5% of head adjectives, n 1,922 = 2.3%, v 996 = 7.2%, r 630 = 17.4%). Of the 3,992 lemma pairs, 737 have both lemmas among the 9,946 core words (a 462, n 136, v 101, r 38). The USVS tiered store has 11,317 word pairs (direct 702, satellite_head 2,681, satellite_satellite 7,934 per the history report).
- **Adjective clusters**: 7,463 head synsets, 10,693 satellites (every satellite has exactly one `similar_to` head: 0 without, 0 with several). Heads with a direct antonym 3,846; antonymous head-head pairs (bipolar clusters) **1,999**; heads that own satellites 2,512 (of which 2,510 have a direct antonym), mean satellites per head-with-satellites **4.26** (median 3, max 146; 1.43 over all heads). 14,532 adjective synsets (80.0%) are in a bipolar cluster (an antonymous head pair plus their satellites); 3,617 head adjectives have no direct antonym (2 of them own satellites).
- **`attribute` relation**: 1,278 directed links (both directions listed) = 639 adjective-synset -> noun-synset links; 620 distinct adjective synsets (all head `a`), 320 distinct noun synsets.
- **Distance to a synset with a direct antonym** (undirected multi-source BFS over synsets, 0 = the synset itself has one; edge types: `similar_to`, `also_see`, derivationally-related (lemma-level, mapped to synset), `pertainym`, `hypernym`/instance-hypernym), share of the 117,659 WordNet senses within d steps:

| edge types | d=0 | <=1 | <=2 | <=3 | d<=3 by POS n/v/a/s/r (%) |
|---|---|---|---|---|---|
| sim+also+deriv+pert+hyper | 6.3% | 23.7% | 43.8% | 68.0% | 59/93/74/100/67 |
| sim+also+deriv+pert (no hypernym) | 6.3% | 18.4% | 24.2% | 27.0% | 12/31/59/100/66 |
| sim+also+deriv+pert+hyper+attr+verbgroup | 6.3% | 23.8% | 44.1% | 68.2% | 59/94/74/100/67 |
| sim only | 6.3% | 15.4% | 15.4% | 15.4% | 2/7/52/100/17 |
| deriv only | 6.3% | 9.6% | 12.3% | 14.6% | 9/24/55/17/17 |
| hyper only | 6.3% | 12.3% | 24.1% | 41.0% | 41/74/52/0/17 |

- **Encoder gold v6** (`runs/encoder_gold_v6.jsonl.gz`, 22,990 sentences; content tokens = UPOS NOUN/VERB/ADJ/ADV, n=211,042: NOUN 73,410, VERB 69,403, ADJ 37,586, ADV 30,643; each token's `chosen_sense`): share within 0/1/2/3 steps of an antonym-bearing synset, all five edge types: 27.8% / 54.0% / 76.4% / 89.3%; without hypernym edges: 27.8% / 42.7% / 56.5% / 62.5%. (The gold's chosen sense is the teacher's pick, which is MFS-like.)
- **Hypernym depth** (`min_depth`; root = 0): nouns n=82,115, mean 7.96, median 8, histogram {'0': 1, '1': 3, '2': 22, '3': 228, '4': 2020, '5': 6249, '6': 12267, '7': 18936, '8': 14155, '9': 11042, '10': 7207, '11': 4267, '12': 2505, '13': 1383, '14': 846, '15': 449, '16': 341, '17': 164, '18': 30}; `max_depth` mean 8.55. Verbs n=13,767, mean 2.53, median 2, histogram {'0': 559, '1': 3286, '2': 3927, '3': 3001, '4': 1591, '5': 729, '6': 358, '7': 149, '8': 60, '9': 63, '10': 32, '11': 11, '12': 1} (559 verb synsets are roots with no hypernym).

## 7. What depends on the current representation (blast radius; LIVE code only, archive/ excluded)

| Where | What it assumes | What would break if sense vectors/axes changed |
|---|---|---|
| `src/nsm_ct/ground/usvs.py:235` `SENSE_TOP_K=24`, `306-315` (top-k, `np.where(blend >= max(cut,1e-9), blend, 0)`, `/blend.max()`), `674-676` (`lex:` axis = 1.0) | sense rows non-negative, <=24 (+lex) axes, max-normalised | This IS the sense builder; a new builder replaces it. `build_usvs` is called by `scripts/build_usvs.py`. |
| `usvs.py:360-375, 390-397, 745-796` (`USVS` dataclass, `sense_dense`, `save/_load_base`) | dense axes order = `meta["axes"]`; sparse CSR (`sense_indptr/idx/val` float32); `core_coords` [W,D] float32; `sense_axis_val` any sign storable | Storage format tolerates signs/other counts; consumers below do not tolerate axis changes. |
| `usvs.py:525-581` `extend()` and `quantum_parser/lexicons/{en,es}.json` | lexicon `dims` use axis NAMES; `axis_index[a]` raises `KeyError` on an unknown name (line 559). The two lexicons reference **41 base axes** (ABOVE, AFTER, ALL, BECAUSE, BEFORE, BELOW, BE_SOMEONE_SOMETHING, CAN, FAR, GOOD, HERE, I, IF, INSIDE, LIKE, LITTLE_FEW, MAYBE, MINE, MORE, MOVE, MUCH_MANY, NEAR, NOT, NOW, ONE, OTHER, PART, PEOPLE, SIDE, SOME, SOMEONE, SOMETHING, THE_SAME, THINK, THIS, TOUCH, TWO, VERY, WHEN, WHERE, YOU) + 7 extension axes | Removing/renaming/merging any of those 41 prime axes makes `load_usvs()` fail for everyone (all consumers below) until the lexicon data is re-mapped. 48 axes are used in total. |
| `usvs.py:698-699` fingerprint (sha256 of axes + meta); `tests/test_lexicon_usvs.py:22,44-49,55,62-70` | fingerprint `e0daef638b640dd5`, 607/614 axes, `ext.axes[:607]==base.axes`, `core_coords` base columns equal, `sense_axis_val >= 0`, content senses byte-identical on base axes | These tests fail by design on any rebuild; extension `ext_axes.json` field `"extends": "e0daef638b640dd5"` also pins it. |
| `src/naomi/score.py:185-194` `axis_weights`; `248-263` `_Space`; `303-305` `_cos` (`min(1.0, ...)`, no lower clamp); `352-367` reasons via `sense_signature` | IDF from df of non-zero cells (a signed space still works); vectors scaled by sqrt(w); unit dims for names/unsensed lemmas use `max(w)` | Weighting formula and the `fn.` operator folding (`FN_PREFIX`, line ~37 in `reader.py`) assume operator senses live on the lexicon axes; "folding equals separate slots" holds only for disjoint dimensions (`tests/test_naomi_score.py:80-96`). Thresholds pinned in tests (e.g. `test_naomi_score.py:125` dog/bone < 70) would shift. |
| `src/naomi/score.py:210-` `sense_opposite`, `_antonym_pairs` | uses `usvs.antonyms` tiered store (not vectors) | Independent of vector change (placement-inert store), but they are the only thing making happy/unhappy score 66.7. |
| `src/nsm_ct/encoder_model.py:216-240` `build_features` | mean of `sense_dense` over candidate senses -> `sense_feat[T, d_axes]`; `:568` `nn.Linear(d_axes, d_sense)`; `:609` `usvs_head = Linear(hidden, d_axes + reserved_size())` | Checkpoints `runs/arms/v6_all_hard_seed{0,1}.pt` have `d_axes=607`, `sense_proj.weight (16,607)`, `usvs_head.weight (4708,128)` (4708 = 607 + 5 + 4096). Any change of axis count or meaning invalidates them (retrain). |
| `src/nsm_ct/usvs_graded.py:248-275` `node_vector`; `:494` `S[i,j] = rw * max(0.0, _cosine(...))`; `:158-205` reserved block after `len(usvs.axes)` | L2-normalised mean of sense vectors; negative cosine clamped to 0 in the soft-matching score | A signed space would have negative cosines silently clamped; vector dimension = `len(axes)+4101` everywhere. |
| `src/naomi/reader.py:79-86, 187-195, 265-268` | the seam uses the extended (614) artifact; the legacy encoder path loads `extended=False` "v6 encoder was trained on the 607 base axes" | Legacy fallback encoder breaks on a new axis set. |
| `scripts/train_encoder.py:280-293, 520`, `eval_encoder.py:52-64`, `rescore_encoder.py:67-87`, `probe_encoder_complement.py:294-299`, `encoder_vs_parser_probe.py:279-365`, `hand_gold.py:678,693`, `build_encoder_gold_v2.py:517,583`, `build_holdout_gold.py:43`, `gen_hard_gold.py:113`, `naomi_run.py:135` | `load_usvs()`; `d_axes=len(usvs.axes)`; `senses_of`/`senses_of_surface` for candidates | Gold builders use only **sense ids** (candidates), so gold files (`runs/encoder_gold_v6.jsonl.gz`, `hard_gold_train.jsonl`, holdouts) store no vectors and stay valid if sense ids and the lemma index are unchanged. Training targets (`node_vector`) are recomputed at train time. |
| tests touching vectors: `test_usvs_graded.py:216-219,297,410`, `test_usvs_aux_head.py:65-374`, `test_encoder_model.py` (many `d_axes=len(usvs.axes)`), `test_decisions_d1_d6.py:136`, `test_pos_lemmatization.py:50`, `test_lexicon_usvs.py`, `test_naomi_score.py`, `test_hard_gold_gen.py` | shapes follow `len(axes)`; a few pin 607/614 | Mostly shape-agnostic except `test_lexicon_usvs.py` and checkpoint-based tests. |
| `src/naomi/{memory,mind,decoder}.py` | grep finds **no** USVS vector use | Not affected. |
| `quantum_parser/` Python | grep finds no use of `sense_dense`/axes | Only the lexicon JSON data above is coupled. |

Artifacts built from USVS that a change would rebuild: `data/usvs/{usvs.npz, usvs_senses.json.gz, usvs_meta.json, dictionary.jsonl.gz (36 MB, sense signatures)}` (blobs are gitignored; `usvs_meta.json` is the committed pin); the two v6 encoder checkpoints (and any retrain); nothing else caches vectors (`DecompCache` is in-memory only; no embedding cache found). The IDF weights are computed on load, not stored. Not examined: `colab/`, `logs/`, `configs/`.

## 8. BASELINE SCORECARD ("how good USVS is today for content words")

Numbers are rounded; "H" = held-out (M24-safe) core coordinates, "P" = production artifact (in-sample at core level).

| # | Quantity | Value | Reproduce with (SCRATCH dir) |
|---|---|---|---|
| 1 | H core raw: syn>ant (10k, jitter mean; band) | 0.675 (0.666-0.686) | `python h1_harness.py 10000 winners` |
| 2 | H core raw: synonym AUC vs random | 0.882 | same |
| 3 | H core raw: similar AUC vs random | 0.814 | same |
| 4 | H core raw: hypernym AUC (anchored) / (cosine) | 0.710 / 0.831 | same |
| 5 | H core raw: random-pair mean cosine | 0.234 | same |
| 6 | H core raw: P(antonym less similar than random) | 0.222 (antonyms look MORE similar than random in 78% of draws) | same (`antonym_auc`) |
| 7 | H core idf: synonym AUC / syn>ant / P(ant<rand) | 0.920 / 0.690 / 0.237 | `python h2_heldout_core.py; python m5_heldout_core.py` |
| 8 | SimLex rho: core-core + antonym store / all 999 plain | 0.417 / 0.352 | `python archive/scripts/probe_simlex.py` (repo) |
| 9 | Sense raw|idf: mean cos of direct-antonym synset pairs | 0.859 / 0.712 | `python scorecard.py` |
| 10 | Sense raw|idf: mean cos of random sense pairs | 0.396 / 0.112 | same |
| 11 | Sense raw|idf: P(antonym pair closer than a random synonym (MFS) pair) | 0.932 / 0.926 | same |
| 12 | Sense raw|idf: P(antonym closer than a `similar_to` near-synonym pair) | 0.745 / 0.793 | `python m2b_extra.py` |
| 13 | Sense raw|idf: P(antonym less similar than random) | 0.017 / 0.020 | `scorecard.py` |
| 14 | Sense raw: mean cos sisters / parent-child / antonyms | 0.860 / 0.741 / 0.859 | `python m2_profile.py` |
| 15 | Core raw (P): P(antonym closer than random synonym) / antonym-sign-opposed (any axis a*b<-0.01), random pairs in brackets | 0.236 / 16.8% (0.5%) | `scorecard.py` |
| 16 | Sense: antonym pairs sign-opposed on any axis; min value in matrix | 0% ; 0.0 | same |
| 17 | Sense space: SOMETHING / ONE / PART / WHEN share of senses | 98.4 / 97.4 / 97.2 / 82.4 % | same |
| 18 | Sense squared norm on prime axes, raw vs idf; lex axes idf | 51.6% -> 16.4%; lex 61.6% | same |
| 19 | Empty axes (no sense) | 47 of 607 (26 prime + 21 attr) | same |
| 20 | Effective dimensionality: sense PR (centered raw / idf), n90 | 38.6 / 59.4 ; 59 / 124 | same (`m3_axes_eigen.py` for full table) |
| 21 | Core PR centered, n90; anchored-feature PR/n90 at 3k-234 vs 10k-607 | 32.3, 80 ; 17.3/46 vs 22.4/84 | `m3b_hist_dim.py` |
| 22 | Pairs, sense raw|idf: happy/unhappy ; happy/sad | 0.887|0.900 ; 0.847|0.883 | `scorecard.py` |
| 23 | Pairs, sense raw|idf: dog/cat ; eat/chew ; bone/stone ; dog/bone | 0.839|0.827 ; 0.773|0.826 ; 0.471|0.032 ; 0.644|0.395 | same |
| 24 | Tree scorer meaning% (overall%): dog->cat ; ate->chewed ; bone->stone | 94.24 (97.1) ; 94.21 (97.1) ; 67.8 (83.9) | `python scorecard.py` (`m8_tree_cases.py` for more) |
| 25 | Tree scorer meaning%: happy->sad ; happy->unhappy (opposite rule fires) | 96.1 ; 66.7 | same |

Note on #24: the ledger lists "dog->cat 97.1, ate->chewed 94.2, bone->stone 83.9" under "(meaning/shape)". With the live scorer those figures correspond to *overall* (dog->cat 97.1 and bone->stone 83.9) and *meaning* (94.2); dog->cat and ate->chewed are a tie (meaning 94.24 vs 94.21), not dog/cat above ate/chewed by 3 points. The qualitative weakness stands. Without IDF the same cases give meaning 94.6 / 92.4 / 82.4 / 94.9.

## 9. Findings

### 9.1 Demonstrably a problem (measured)
1. **Sense vectors put opposites almost on top of each other.** Mean cosine of direct antonym synset pairs 0.859 raw / 0.712 IDF vs random 0.396 / 0.112; an antonym pair is more similar than a random pair in 98% of draws (AUC 0.017-0.020), closer than a near-synonym (`similar_to`) pair in 75-79%. happy/unhappy 0.89, happy/sad 0.85, hot/cold 0.92, good/bad 0.86, big/small 0.99, up/down 0.99. IDF does not fix it. (Scorers only get opposites right through the WordNet antonym store; happy/sad, not a recorded pair, scores 96%.)
2. **Signs exist only on `attr:` axes of core words and are absent from senses**: min sense value 0.0 of 2,785,213 non-zeros; core has negatives on 220 axes. Only 16.8% of core antonym pairs are strongly (a*b<-0.01) opposed on some axis (random pairs 0.5%); the "73% opposed on any axis" is propagation dust (median non-zero core magnitude 9e-5). Keeping signs in the sense blend would not by itself help: negative mass is 1.1% of the blend's absolute mass and the unsparsified antonym cosine would still be 0.82 vs 0.49 random.
3. **Held-out core antonymy is weak**: syn>ant 0.675, and antonyms are more similar than random in 78% of draws (core AUC ant<rand 0.222); per-POS in-sample syn>ant 0.74-0.81.
4. **Sister and genus structure dominate sense similarity**: sister pairs 0.860 vs parent-child 0.741 (AUC sister>parent-child 0.77), sisters as close as antonyms. 44 `lex:` axes carry 61.6% of the IDF-weighted squared norm of an average sense (prime 16.4%, attr 21.9%); with only the 20 top-IDF-energy axes (all `lex:`) hypernym AUC is still 0.91. dog/cat (0.84 / 0.83) is at least as close as eat/chew (0.77 / 0.83); bone/stone IDF 0.03 but raw 0.47.
5. **Ubiquitous axes**: SOMETHING 98.4%, ONE 97.4%, PART 97.2%, BODY 84%, WHEN 82%, KIND 81%, `lex:noun.act` 79%, PEOPLE 76%; 12 axes above 50%. Raw prime share of the squared norm is 52%; IDF cuts it to 16% (IDF is therefore doing necessary work, and is opt-out in the scorer).
6. **Dead and dust axes**: 47 axes empty at sense level (26 primes empty everywhere, 21 attr only in core), 22 more with <=50 senses. 33% of core words (3,286; nouns 41%, adverbs 71%; includes water, city, river, food, law) have <=10 non-zero axes, i.e. are effectively unplaced; dense core words carry ~537 non-zero axes but only ~7 above 0.1.
7. **Redundant axes**: 221 axis pairs correlate >0.9 over core words (193 over senses), almost all duplicates among WordNet noun-attribute lemmas (speed/swiftness/fastness; six Confederate-states axes at r=1.0; brightness/luminosity/luminance...). 246 of 498 `attr:` axes are topic/region/usage domains (mean 1.7% of senses) and the other 249 are gradable-dimension axes (0.4% of senses).
8. **The space is far larger than what carries the signal**: sense PR 38.6 (59 IDF), n90 59 (IDF 124); held-out core IDF top-100 axes keep synonym AUC 0.912 of 0.920 (syn>ant 0.652 of 0.690), top-50 0.893; the effective count grew from ~17 (234 axes) to ~22-25 (607 axes) while 90% energy needs 84-101 anchored axes.
9. **Sense-level synonymy is not representable**: same-synset synonyms share one point; MFS-pair "synonyms" have mean cosine 0.514 vs 0.397 random (AUC 0.69), and for adjectives/adverbs below random-pair level (AUC 0.42 / 0.32 raw). Word-level synonymy is only present in core coordinates.
10. **Coverage of any antonym-based design is thin outside adjectives**: 3,992 direct lemma pairs touch 7,394 synsets = 6.3% of all synsets (adjective heads 51.5%, nouns 2.3%, verbs 7.2%, adverbs 17.4%); within 3 steps over similar/also-see/derivational/pertainym only 27% of senses (nouns 12%) reach an antonym-bearing synset; adding hypernym edges, 68% (noun 59%). For content tokens of the encoder gold: 27.8% sit on such a synset, 62.5% within 3 steps without hypernym edges and 89.3% with them.

### 9.2 Merely unintuitive / harmless
- `big/large` cosine 1.0 because both resolve to `large.a.01`; many "unintuitive" pair values come from which WordNet sense is first (`stone` -> `rock.n.01`, `poor` -> `hapless.s.01`, `boy` -> `male_child.n.01`).
- Uncentered spectra look 1-dimensional-ish (PR 5-8) because of one common direction; centered PR 32-39.
- Raw core cosine of random pairs is 0.28 (0.234 held-out); IDF brings it to 0.07-0.11.
- The 'cosine is a genericity match' effect is neutralised by IDF at the node level in the scorer.
- Satellite_satellite tier artifacts, lemma-level conflation: known, recorded, not re-measured.

### 9.3 Working well and to be preserved (measured)
- Synonymy and relatedness in the word core: held-out synonym AUC 0.882 (0.920 IDF), similar 0.814, SimLex 0.417 (word2vec-class ~0.41-0.44 per ledger), hypernym AUC vs random 0.83; hypernym vs random at sense level 0.95-0.97; sister vs random 0.98.
- Sense layer separates topics: unrelated senses are far apart under IDF (random 0.11), dog/justice 0.05, bone/stone 0.03.
- The reproduction of the ledger is stable: the harness regenerates M28.0/M29 numbers to within jitter, so it is a usable regression yardstick.
- Placement-inert antonym store and tiers (11,317 pairs) correctly flag opposites (the only reason scorer opposites fire); `fn.` lexicon extension is additive and does not change content-word vectors (base columns byte-identical).
- Attr-axis sign anchoring on core words gives the held-out syn>ant gain 0.404 -> 0.693 (and per kind, `attr` axes alone carry the best syn>ant: 0.729 IDF).
- Kind split of discrimination information (held-out core, IDF): attr-only syn>ant 0.729, syn AUC 0.889; lex-only syn AUC 0.897 but syn>ant 0.600; prime-only 0.854 / 0.697; all axes 0.920 / 0.690. At sense level no kind separates antonyms from near-synonyms (P(syn>ant) <= 0.23 for every kind combination, lowest 0.18-0.19 for prime+attr), while `lex:` alone gives hypernym AUC 0.97 (same as all axes).

No redesign recommendations are made here.

## 10. Could not measure / limits

- `probe_minimality.py` (M20.1 leave-one-out contribution ranking; "fidelity peak K~80-120", "last ~100 axes hurt") was not re-run on any axis set (long, leave-one-out); the top-K-by-IDF-energy ablation above is a different ranking and is not comparable to it. The M20 tanh 0.756 is reproduced only approximately (0.775 on the current 3k split).
- Sense-level synonym AUC cannot be measured cleanly (same-synset synonyms are identical vectors); I used MFS pairs and `similar_to` pairs as proxies. Sense-level hypernym similarity is partly circular (senses are built from their hypernym lemmas' core coordinates, `usvs.py:292-299`).
- Core-level production numbers are in-sample; held-out core numbers exist only for raw/IDF/kind/top-K in 2a'/5, not by POS and not for sisters.
- "Up/down as content adverbs": measured as `up.r.01`/`down.r.01` (0.99); not a content-vs-closed-class judgement.
- Tree-scorer cases were reproduced on hand-built trees only (no encoder/parser). Words `unhappy`, `puppy`, `love`, `hate`, `up`, `down` are not core words, so core columns are empty for 4 of 40 pairs (happy/unhappy, love/hate, up/down, dog/puppy).
- Hypernym-distance BFS treats hypernym edges as undirected; `derivationally_related` and `pertainym` are lemma-level links mapped to synsets (no sense filtering); results include satellites reaching their head's antonym at distance 1.
- Gold-token coverage uses the gold's `chosen_sense`, not a WSD system. I did not examine `colab/`, `logs/`, `configs/` for USVS consumers.
- Spanish/OMW and the closed-class lexicon were out of scope.

## Appendix A. Scripts (all under SCRATCH; none write outside it)

`common.py`, `scorecard_lib.py` (shared helpers); `pairs.py` (pair sets -> `pairs.pkl`); `h1_harness.py` (original harness, 3k/10k); `h2_heldout_core.py` (re-place held-out core, saves `heldout_core.pkl`); `m5_heldout_core.py` (held-out core profile, kinds, top-K); `m2_profile.py` (2a/2b/4.2/5 production); `m2b_extra.py` (similar_to control, POS hyper/sister, sparse core words); `m3_axes_eigen.py` (axis usage, duplicates, shares, spectra); `m3b_hist_dim.py` (project's `dimensionality_spectrum`); `m3c_core_density.py`; `m4_signs_pairs.py` (sign opposition, counterfactual blend, 40 pairs); `m6_relations.py`, `m6b_fix.py` (relation coverage); `m8_tree_cases.py` (tree scorer cases); `scorecard.py` (below); `gen_report.py` (assembles this file from the JSON outputs). Repo scripts run unmodified: `archive/scripts/probe_m28_baseline.py`, `probe_normalize.py`, `probe_simlex.py`. Logs: `h1_*.log`, `h2.log`, `m*.log`.

## Appendix B. `scorecard.py` (re-creates scorecard rows 9-11, 13, 15-20, 22-25 in ~2 min)

```python
"""USVS content-word BASELINE SCORECARD (read-only).  python scorecard.py  [~2 min]
Part A (production artifact, data/usvs, 607 base axes, no writes): core- and sense-level cosine metrics +/- IDF,
        sign-opposition counts, axis usage, eigenspectra, 5 concrete pairs.
Part B (live scorer): the RESEARCH_NOTES tree cases.
Held-out core numbers and SimLex come from separate commands listed in the report (h1_harness.py, probe_simlex.py)."""
import sys, json, random
import numpy as np
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
from nltk.corpus import wordnet as wn
from nsm_ct.ground.usvs import load_usvs
DATA = "/home/user/NAOMI/consciousness_transformer/data/usvs"
u = load_usvs(DATA, extended=False); D = len(u.axes)
core = {w: i for i, w in enumerate(u.core_words)}; sidx = {s: i for i, s in enumerate(u.sense_ids)}
C = u.core_coords.astype(np.float64)
M = np.zeros((len(u.sense_ids), D)); M[np.repeat(np.arange(len(M)), np.diff(u.sense_indptr)), u.sense_axis_idx] = u.sense_axis_val
w_idf = np.log((len(M) + 1.0) / ((M != 0).sum(0) + 1.0))              # == naomi.score.axis_weights on the base artifact
kd = np.array(["attr" if a.startswith("attr:") else "lex" if a.startswith("lex:") else "prime" for a in u.axes])

def wcos(A, B, w=None):
    w = np.ones(A.shape[1]) if w is None else w
    num = (A * B * w).sum(1); den = np.sqrt((A * A * w).sum(1) * (B * B * w).sum(1))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
def auc(pos, neg):                                                   # P(pos > neg) + .5 P(tie)
    x = np.concatenate([pos, neg]); uq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

# ---- pair sets (WordNet) ----
rng = random.Random(0)
syn = {}                                                             # lemma pairs sharing a synset, both in core
for ss in wn.all_synsets():
    ls = sorted({l.name().lower() for l in ss.lemmas() if l.name().lower() in core})
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)): syn.setdefault((ls[i], ls[j]), 1)
syn = list(syn)
mfs = {x: (u.senses_of(x) or [None])[0] for x in u.core_words}
ant = {}                                                             # direct lemma antonym pointers, both lemmas in core
for ss in wn.all_synsets():
    for l in ss.lemmas():
        for a in l.antonyms():
            x, y = l.name().lower(), a.name().lower()
            if x in core and y in core and x != y: ant[(frozenset((x, y)), frozenset((ss.name(), a.synset().name())))] = (x, y, ss.name(), a.synset().name())
ant = list(ant.values())
W, N = len(u.core_words), len(u.sense_ids)
rc = [(rng.randrange(W), rng.randrange(W)) for _ in range(20000)]; rs = [(rng.randrange(N), rng.randrange(N)) for _ in range(20000)]
synS = [(sidx[mfs[a]], sidx[mfs[b]]) for a, b in syn if mfs[a] and mfs[b] and mfs[a] != mfs[b]]
def cosl(X, prs, w): return wcos(X[[a for a, b in prs]], X[[b for a, b in prs]], w)
R = {}
for lvl, X, S_, A_, Rn in (("core", C, [(core[a], core[b]) for a, b in syn], [(core[a], core[b]) for a, b, *_ in ant], rc),
                           ("sense", M, synS, [(sidx[s1], sidx[s2]) for _, _, s1, s2 in ant], rs)):
    for wn_, w in (("raw", None), ("idf", w_idf)):
        cs, ca, cr = cosl(X, S_, w), cosl(X, A_, w), cosl(X, Rn, w)
        R[f"{lvl}|{wn_}"] = dict(syn_n=len(cs), ant_n=len(ca), syn_mean=cs.mean(), ant_mean=ca.mean(), rand_mean=cr.mean(),
            AUC_syn_vs_rand=auc(cs, cr), P_ant_less_similar_than_rand=auc(-ca, -cr), P_ant_closer_than_random_synonym=1 - auc(cs, ca))
# sign opposition (core): antonym pairs with a strongly sign-opposed axis (product < -0.01) vs random pairs
A_ = np.array([(core[a], core[b]) for a, b, *_ in ant]); pr = C[A_[:, 0]] * C[A_[:, 1]]
R["sign"] = dict(core_ant_strongly_opposed=float((pr < -0.01).any(1).mean()), core_ant_any_opposed=float((pr < -1e-6).any(1).mean()),
                 core_rand_strongly_opposed=float(((C[[a for a, b in rc]] * C[[b for a, b in rc]]) < -0.01).any(1).mean()),
                 sense_min_value=float(M.min()), sense_ant_opposed=float(((M[A_[:, 0] * 0 + [sidx[s1] for *_, s1, s2 in ant]] * M[[sidx[s2] for *_, s1, s2 in ant]]) < 0).any(1).mean()))
# axis usage
fs = (M != 0).mean(0); E = (M ** 2)
R["axes"] = dict(SOMETHING=float(fs[u.axes.index("SOMETHING")]), ONE=float(fs[u.axes.index("ONE")]), PART=float(fs[u.axes.index("PART")]), WHEN=float(fs[u.axes.index("WHEN")]),
    n_empty_sense_axes=int(((M != 0).sum(0) == 0).sum()), prime_share_sqnorm_raw=float(E[:, kd == "prime"].sum() / E.sum()),
    prime_share_sqnorm_idf=float((E * w_idf)[:, kd == "prime"].sum() / (E * w_idf).sum()), lex_share_sqnorm_idf=float((E * w_idf)[:, kd == "lex"].sum() / (E * w_idf).sum()),
    mean_axes_per_sense=float((M != 0).sum(1).mean()))
def spec(X, center):
    n = X.shape[0]; G = np.zeros((D, D)); mu = X.mean(0) if center else 0
    for i in range(0, n, 20000): B = X[i:i + 20000] - mu; G += B.T @ B
    lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None); cum = np.cumsum(lam) / lam.sum()
    return dict(PR=float(lam.sum() ** 2 / (lam ** 2).sum()), n90=int(np.searchsorted(cum, .9) + 1), n99=int(np.searchsorted(cum, .99) + 1))
R["spectra"] = {f"{l}|{c}": spec(X, c == "centered") for l, X in (("core", C), ("sense", M)) for c in ("uncentered", "centered")}
R["spectra_idf_sense_centered"] = spec(M * np.sqrt(w_idf), True)
# concrete pairs (first sense of the given POS)
def first(word, pos):
    ss = u.senses_of(word); return next((s for s in ss if s.rsplit(".", 2)[1].replace("s", "a") == pos), ss[0] if ss else None)
for a, b, p in (("happy", "unhappy", "a"), ("happy", "sad", "a"), ("dog", "cat", "n"), ("eat", "chew", "v"), ("bone", "stone", "n"), ("dog", "bone", "n"), ("hot", "cold", "a")):
    va, vb = M[[sidx[first(a, p)]]], M[[sidx[first(b, p)]]]
    R[f"pair {a}/{b} sense raw|idf"] = [float(wcos(va, vb)[0]), float(wcos(va, vb, w_idf)[0])]
# Part B: tree cases
from naomi.reader import make_node
from naomi.score import score_pair
from naomi.types import REL_ROOT, Tree
ue = load_usvs(DATA)
def Nn(rel, tok, idx, pos, *k): n = make_node(rel, tok, idx, pos); n.children = list(k); return n
ate = lambda v="ate", s="dog", o="bone": Tree(sentence="", root=Nn(REL_ROOT, v, 2, "VERB", Nn("SUBJECT", s, 1, "NOUN"), Nn("OBJECT", o, 4, "NOUN")))
she = lambda a: Tree(sentence="", root=Nn(REL_ROOT, "was", 1, "AUX", Nn("SUBJECT", "she", 0, "PRON"), Nn("COMPLEMENT", a, 3, "ADJ")))
for k, (x, y) in {"tree dog->cat": (ate(), ate(s="cat")), "tree ate->chewed": (ate(), ate(v="chewed")), "tree bone->stone": (ate(), ate(o="stone")),
                  "tree happy->sad": (she("happy"), she("sad")), "tree happy->unhappy": (she("happy"), she("unhappy"))}.items():
    ps = score_pair(x, y, ue); R[k] = dict(meaning=ps.meaning, overall=ps.overall, reasons=[r["kind"] for r in ps.reasons])
print(json.dumps(R, indent=1, default=float))
```
