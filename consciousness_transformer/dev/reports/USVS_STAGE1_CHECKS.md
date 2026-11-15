# USVS stage 1 (revised): traceable scales vs simpler baselines (read-only checks, 2026-11-15)

No git, no change to existing files or `data/usvs/*`. Scripts + every number: `dev/reports/usvs_review_scripts/stage1/` (`results.json`; run order `s1_folds, s1_pairs, s1_dev, s1_select, s1_b2dev, s1_test, s1_explain, s1_explain2, s1_extra`). Big caches (sense matrix, edges, eval sets) are reused from `probes/` (`mkbase.py` regenerates them). Single split, single seed; no confidence intervals. Opposite AUCs below are small numbers: 0.5 = opposites look like random pairs, near 0 = opposites look MORE similar than random.

## 0. Protocol (fixed before any test number was read)
- **Families.** The 3,802 recorded antonym pairs were grouped by union-find: a shared member or shared cluster head, or a derivational / pertainym / attribute edge between members of two pairs (happy/unhappy, happiness/unhappiness, happily/unhappily land together). `also_see` was NOT used for grouping: with it one family holds 1,263 of 3,802 pairs. 2,543 families, largest 18 pairs. Seed 20261115, greedy fill to 60/20/20.
- **Fold sizes (pairs):** build 2,280 (n 581, v 290, a 1,203, r 206); dev 762 (n 193, v 100, a 396, r 73); test 760 (n 201, v 118, a 400, r 41).
- **Dev selection rule** (written first): among configs with no damage (dev hypernym AUC, similar-vs-random AUC and SimLex each >= B1 - 0.01), maximise P(dev opposite less similar than random pair); ties -> fewer steps, larger decay. Grid, 768 configs: edge set {sim; +deriv; +pert; +also_see} x hyp step {no, yes} x degree cap {none, 30} x attribute merge {no, yes} x decay {.5,.7,.85} x steps {1..4} x importance {a,b}. B2 grid: r {0,1,2} x decay {.3,.5,.7,.9}. Test set looked at once with these frozen (a later exploratory "wide" config is labelled as such).
- **Views.** "all" = all 760 test pairs, unplaced pairs get the named block alone. "placed" = both ends sit on at least one scale (65 pairs, 8.6%); baselines are scored on those same 65. "b2flagged" = the 153 pairs (20.1%) that B2 flags. Damage sets (hypernym, similar_to, synonym, SimLex, random) are the same pair sets as the earlier probes and are NOT held out from spreading: similar_to pairs are edges the construction follows, so similar-vs-random AUC is inflated for the construction; hypernym and SimLex are the cleaner damage checks.
- **Frozen dev settings.** Construction (a) and (b): edges `similar_to` + one hypernym/hyponym step, no deriv/pert/also_see, step cap 1, decay 0.85, no degree cap, no attribute merge (dev ant<rand .0737 (a), .0696 (b) vs B1 .0689). Many configs tie, so these are partly tie-break outcomes; steps > 1 add nothing (`similar_to` is a star: head <-> satellites). B2: r = 2 steps per side, decay 0.9 per step (dev .1018; edge of the grid). Dev spread over all 768 configs: ant<rand .049 to .074 against .069 for B1 (no config meaningfully helps dev opposites).
- **Comparison formula (exact).** For senses a,b with named-block similarity N(a,b) (B1: IDF-weighted cosine over 595 axes), shared scales S = scales on which both have importance: `m_s = min(imp_a,s, imp_b,s)`; `agree = sum_s m_s * (1 - |pos_a,s - pos_b,s| / 2) / sum_s m_s`; `g = mean_s m_s`; `sim = (1 - g) * N + g * agree`; S empty -> g = 0, sim = N. Position in [-1,1]; recorded members +1/-1 (forced); a word reached from both ends gets `d^k_plus - d^k_minus` (shortest path each side), clipped. Importance (a) = d^(steps to nearest member); (b) = share of the sense's own non-antonym dictionary links (plus recorded build antonym links) that land on senses placed on that scale. A middle word keeps non-zero importance in both. The constants (decay, caps, degree factors) are assumptions or dev-chosen, not derived.
- **Null control.** Right-hand members of the 2,280 build pairs permuted within part of speech (same number, same degree distribution), frozen construction re-run, 5 seeds.

## 1. Main table (test fold; frozen settings)
Columns: mean similarity of test opposites; P(synonym pair closer than a test opposite); P(opposite less similar than random pair) overall and by POS (n/v/a/r); hypernym AUC; similar-vs-random AUC; SimLex rho. B1 is the reference.

| row (all 760 pairs) | mean opp sim | syn>opp | opp<rand | opp<rand n/v/a/r | hyp AUC | sim/rand | SimLex |
|---|---|---|---|---|---|---|---|
| B0 sense vectors + IDF | .725 | .071 | .020 | .016/.038/.098/.210 | .972 | .780 | .269 |
| B1 B0 minus 12 ubiquitous axes | .650 | .132 | .077 | .043/.038/.137/.216 | .935 | .726 | .274 |
| B2 relation at compare time (r=2, d=.9) | .566 | .183 | .109 | .059/.040/.196/.449 | .935 | .724 | .305 |
| construction, importance (a) | .618 | .143 | .079 | .055/.044/.137/.216 | .937 | .892* | .316 |
| construction, importance (b) | .643 | .133 | .076 | .040/.041/.137/.216 | .937 | .880* | .292 |
| null control (a), 5 seeds | .650-.651 | .132 | .077 | n/a | .937 | .892* | .292-.293 |
| null control (b), 5 seeds | .650 | .132 | .077 | n/a | .937 | .880* | .279 |

*inflated: the construction follows the very similar_to edges these pairs are drawn from. Null "all" rows equal B1 on opposites to 3 decimals because only 9-10 test pairs share a null scale.

| row (placed-only: 65 pairs, 8.6%, n 51 / v 14 / a 0 / r 0) | mean opp sim | syn>opp | opp<rand | opp<rand n / v |
|---|---|---|---|---|
| B0 | .750 | .067 | .022 | .024 / .040 |
| B1 | .737 | .081 | .044 | .054 / .039 |
| B2 | .566 | .139 | .067 | .090 / .039 |
| construction (a) | .360 | .206 | .075 | .102 / .082 |
| construction (b) | .647 | .096 | .037 | .043 / .063 |
| null (a), shared pairs only (9-10 pairs) | .92-.99 | .001-.022 | .000-.007 | n/a |
| null (b), shared pairs only (9-10 pairs) | .93-.95 | .005-.008 | .001 | n/a |

Of the 61 test pairs that share a scale (a): 52 sit on opposite poles, 9 on the same pole. Coverage-matched B1 on those 65: opp mean .737. On B2's own 153 flagged pairs: B1 .573, B2 .156, construction (a) .514, (b) .561 (B2 acts only where it flags, so this is a coverage statement, not an independent test).

**Unrelated-pair control** (n/v/a random same-POS pairs 4,000 each; 30 everyday unrelated pairs; "pushed apart" = sim falls by more than .15 vs B1; "pulled together" = rises by more than .15):

| row | random pairs changed at all | pushed apart / pulled together (random) | everyday-30 pushed / pulled |
|---|---|---|---|
| B2 | 0.03% (3 of 12,000) | 2 / 0 | 0 / 0 |
| construction (a) | 0.04% (5 of 12,000) | 2 / 1 | 0 / 2 of 30 (red/blue, eat/drink) |
| construction (b) | 0.04% (5 of 12,000) | 1 / 0 | 0 / 0 |
| B0 (IDF, no axis drop) | 99% (differs from B1 everywhere) | n/a | n/a |

Worst false positives (frozen a; chains from the build pair): *pushed apart* embark.v.02 / dissolve.v.10 .83 -> .25 (get_down.v.07 -[has-hyponym]-> embark.v.02, end.v.02 -[has-hyponym]-> dissolve.v.10; the two recorded opposites sit on opposite poles); triangulate.v.01 / reunify.v.01 .60 -> .22 (divide.v.01 vs unite.v.01 hyponyms). *Pulled together* red.s.01 / blue.s.01 .25 -> .89 (both similar_to chromatic.a.03, same pole of chromatic/achromatic); eat.v.01 / drink.v.01 .82 -> .97 (both hyponyms of consume.v.02, pole of consume/abstain); tram.v.01 / turn.v.06 .80 -> .97 (both under travel.v.01). All hypernym/hyponym-step artefacts; both directions of error exist, and a shared pole raises similarity as readily as opposite poles lower it.

## 2. What the construction places (coverage, scales, middles)
Scales: 2,280 (one per build pair, 0 merged in the frozen config; an attribute-merge run gave 2,261 = 19 merges, no dev gain). Placed nodes per scale (frozen): median 5, p90 14, max 228, 22% hold only the two members. 955 senses (6.0% of placed) sit on more than one scale (max 32). 406 non-member senses are reached from both ends and sit near the middle (pos 0.00 or +-0.15) with importance .85, e.g. gradient.n.02 on abruptness/gradualness, evaluate.v.02 on accept/reject, color.n.01 on achromatic/chromatic, miss.v.07 on overshoot/undershoot; the frozen edges are similar_to + hypernym/hyponym and the examples are hypernym/hyponym coincidences.

Share of the 117,659 WordNet senses placed, and share of the 211,042 content tokens in gold v6 (n/v/a/r = nouns/verbs/adjectives/adverbs). "Frozen edges" = similar_to + hyp; "wide" = all four edges + hyp (NOT dev-chosen, illustration only; steps counted per side). Hyp step counts as a step.

| recorded set | edges | steps | senses | n / v / a / r | gold tokens | gold NOUN/VERB/ADJ/ADV |
|---|---|---|---|---|---|---|
| build 60% | frozen | 0 (members only) | 3.8% | 1.4/4.2/12.8/11.2% | 18.7% | 6/21/35/25% |
| build 60% | frozen | 1 (= all steps) | 13.4% | 4.7/18.5/49.8/11.2% | 33.2% | 19/37/60/25% |
| build 60% | wide | 2 | 25.7% | 14.3/35.5/65.2/47.8% | 56.5% | 38/62/88/49% |
| build 60% | wide | 3 | 35.4% | 23.6/54.1/70.3/56.6% | 68.6% | 52/82/91/50% |
| build 60% | wide | 4 | 41.4% | 29.7/64.5/73.0/61.6% | 74.6% | 62/88/93/51% |
| all 3,802 recorded | frozen | 1 | 21.3% | 7.4/28.1/80.0/17.4% | 47.3% | 26/49/99/30% |
| all 3,802 recorded | wide | 3 | 44.5% | 30.9/68.6/83.2/66.1% | 78.0% | 63/92/100/56% |

Coverage of what matters here: only 8.6% of test opposite pairs have both ends on any scale under the frozen config (36% under wide, but the wide run places 25,437 senses on more than one scale, max 254 scales per sense, and its test numbers are no better, section 5).

## 3. Traceability check (15 test-fold-pair members placed by the construction)
Sampled with seed 3 from members of test-fold pairs that are placed but not recorded in build. The frozen config placed no test-pair adjective, so the 5 adjectives come from the wide config (marked w). Judge = me, single reader; "far-fetched" = chain valid but the sense inherits a pole too loosely.

| sense (gloss) | scale (recorded ends) | pos / imp a / imp b | chain | verdict |
|---|---|---|---|---|
| reactive.a.01 (w) | instability <-> stability | +.49 / .49 / 1.0 | instability.n.04 -deriv-> unstable.s.01 -similar_to-> reactive | far-fetched |
| reassuring.a.01 (w) | discouraging <-> encouraging | -.70 / .70 / .83 | encouraging -also_see-> reassuring | sensible |
| far.a.01 (w) | accessible <-> inaccessible | -.34 / .34 / .33 | inaccessible -similar_to-> outback.s.01 -deriv-> farness.n.01 -deriv-> far | far-fetched |
| white.a.01 (w) | dark.a.02 <-> light.a.02 | -.70 / .70 / 1.0 | light.a.02 -also_see-> white | sensible |
| paranormal.a.02 (w) | sensory <-> extrasensory | +.70 / .70 / 1.0 | extrasensory -also_see-> paranormal | sensible |
| mental_illness.n.01 | sanity <-> insanity | +.85 / .85 / .20 | insanity -has-hypernym-> mental_illness | sensible |
| decrease.n.04 (act of reducing) | appreciation <-> depreciation | -.85 / .85 / .03 | depreciation -has-hypernym-> decrease | far-fetched (generic) |
| impotence.n.02 | fertility <-> sterility | -.85 / .85 / .33 | sterility -has-hyponym-> impotence | sensible |
| increase.n.02 | waning <-> waxing | -.85 / .85 / .11 | waxing -has-hypernym-> increase | sensible |
| conformist.n.01 | conservative <-> liberal | +.85 / .85 / .33 | conservative -has-hyponym-> conformist | sensible |
| unscrew.v.02 | loosen <-> stiffen | +.85 / .85 / 1.0 | loosen -has-hyponym-> unscrew | sensible |
| assimilate.v.05 (become similar in sound) | change.v.02 <-> stay.v.01 | +.85 / .85 / .50 | change -has-hyponym-> assimilate | sensible |
| decrease.v.02 (make smaller) | maximize <-> minimize | -.85 / .85 / .06 | minimize -has-hypernym-> decrease | far-fetched (generic) |
| contract_in.v.01 (consent to pay union) | accept.v.03 <-> refuse.v.01 | +.85 / .85 / 1.0 | accept -has-hyponym-> contract_in | far-fetched |
| miss.v.07 (fail to reach) | overshoot <-> undershoot | 0.00 / .85 / .67 | overshoot / undershoot -has-hypernym-> miss | sensible (clean neutral middle) |

Verdicts: 10 sensible, 5 far-fetched, 0 wrong-sense (adjectives 3/2/0, nouns 4/1/0, verbs 3/2/0). All 15 can be explained by a 1-3 edge chain naming the recorded pair. Caveats: (1) a hypernym of a pole word gets almost the full pole position, so generic hypernyms (decrease) get a confident value that is probably wrong for most scales; importance (b) shows this (.03-.11) while (a) does not (.85). (2) Importance (b) also gives RECORDED members low weight (happy .35, rich .33), so the two importance definitions disagree on which words matter most. (3) How the 61 test pairs that share a scale were reached under the frozen config: all 61 through two hypernym or two hyponym steps from the two poles of one build pair (e.g. appropriateness / inappropriateness under propriety / impropriety; activator / inhibitor under catalyst / anticatalyst): near-duplicate opposites one level down or up from a recorded pair, not new scales.

## 4. Degree and negation (arithmetic on the scale; constants are my assumptions)
slightly X = pos x 0.4; very X = move 50% of remaining distance to the end; not X = pos x -0.6; importance unchanged; the named block of every operator form is X's own (no operator touches it). Full formula similarity, with the recorded or scale-end opposite at -1 (or +1) with importance 1. Words: first adjective sense; fold = fold of the word's recorded pair.

| word | fold | in build scale? | frozen (a) pos / X~very / X~slightly / X~not / not~opp / X~opp / X~syn |
|---|---|---|---|
| happy, hot, good, rich, clean, strong, easy, safe | build | recorded member | +-1.0 / 1.00 / .70 / .20 / .80 / .00 / .79-.93 |
| big (large.a.01) | test | not placed (frozen); placed by spreading (wide) | wide (a): +.70 / .95 / .85 / .61 / .61 / .22 / .84 |
| fast | dev | not placed in either config | - |

So 8 of the 10 words are build-fold recorded members, where the ordering very > slightly > not > opposite holds by construction (not X .80 vs its opposite, X .00 vs opposite). Under importance (b) the same words give X~opp .05-.72 (happy .72) because link-share importance is .33-.88, which leaves most weight on the named block (where happy/unhappy are .9 similar); the formula then cannot separate recorded opposites. Non-recorded words, aggregated over 2,000 placed non-member adjectives with |pos| >= .2 (seeded):

| config | very>slightly>not | not>opp (X~not > X~opp) | not~opp > X~opp | not~opp < 1 | not~opp > X~not | X~syn > X~not | mean X~very / slightly / not / not~opp / opp |
|---|---|---|---|---|---|---|---|
| frozen (a) | 100% | 100% | 100% | 100% | 100% | 100% | .97 / .78 / .42 / .68 / .10 |
| frozen (b) | 100% | 98% | 100% | 100% | 62% | 67% | .97 / .81 / .48 / .55 / .13 |
| wide (a), not dev-chosen | 92% | 100% | 100% | 100% | 15% | 59% | .95 / .88 / .68 / .51 / .18 |
| wide (b), not dev-chosen | 74% | 100% | 100% | 100% | 72% | 97% | .89 / .84 / .58 / .65 / .23 |

Reading: "not X near the opposite but not on it" holds in all rows (always < 1, always above X~opp). Whether "not X" is nearer the opposite than X is depends on config (15% to 100%), because it depends on |pos| (not X lies nearer the opposite pole than X only when |pos| > .45) and on the weight of the named block. The frozen (a) rows are 100% because every frozen non-member sits at pos .85 (one hypernym step), i.e. one value for all words: those orderings are the arithmetic, not evidence about words.

## 5. Everyday pairs and the wide config
`rec` = fold of the recorded antonym edge for that sense pair (buy/sell is in BUILD in this split, not held out, unlike the earlier probes). B1 / B2 / construction (a, frozen) / construction (a, wide).

| pair | rec | B1 | B2 | frozen a | wide a |
|---|---|---|---|---|---|
| happy/unhappy | build | .90 | .00 | .00 | .55 |
| hot/cold, good/bad, love/hate, buy/sell | build | .63/.68/.88/.81 | .00 | .00 | .41/.59/.41/.00 |
| happy/sad | none | .88 | .09 | .88 | .63 |
| big/small | test | .96 | .96 | .96 | .96 |
| alive/dead | dev | .27 | .27 | .27 | .27 |
| red/blue | none | .25 | .25 | .89 | .77 |
| eat/drink | none | .82 | .82 | .97 | .93 |
| dog/cat, dog/bone, bone/stone, eat/chew, king/queen, quick/fast, big/large | none | unchanged by every row | | | |

The wide (not dev-chosen) config lifts coverage of test pairs sharing a scale to 36% (274 pairs, 223 opposite poles, 51 same pole) but test opposite AUC is .0565 against B1 .0766 (all pairs), because 51 same-pole pairs and wide-spread words with 16 to 47 scales average the agreement toward the middle; hypernym AUC .948, SimLex .329, similar-vs-random .938. Dev already showed the same trade (best dev ant<rand with also_see + hyp: .0687, cov .15).

## What this supports / does not support
**Supports.** (1) Every value the construction gives can be explained by a 1-3 edge chain from a recorded opposite pair (15 of 15 explained; 10 sensible, 5 far-fetched, 0 wrong sense by one reader). (2) Where a test pair shares a scale, the construction separates it: 52 of 61 on opposite poles, mean similarity .74 -> .35 on the placed pairs, a clearly better result than the null control (null: 9-10 shared pairs, mean .92-.99, ant<rand .000-.007 against .065-.075). (3) It does not damage hypernym structure (.935 -> .937) or SimLex (.274 -> .316 / .292), and creates few false oppositions among unrelated pairs (5 of 12,000 random pairs altered; 2 of 30 everyday pairs wrongly pulled together). (4) The arithmetic for degree and negation is well-defined with a neutral middle (miss.v.07 at 0 with importance .85), and "not X" lands near but never on the opposite in every row.

**Does not support.** (1) Generalisation to opposites it was not told about is small: 8.6% of test pairs have both ends placed, all of them near-duplicate pairs reached through hypernym/hyponym steps of a recorded pair; overall opposite AUC is B1-level (.079 vs .077, with the null control also .077). (2) It is not better than the simplest baseline: B2 (opposition as a relation at comparison time) flags 20% of test pairs, not 8.6%, reaches opp<rand .109 and mean opp similarity .566 over all pairs and SimLex .305, with no stored coordinates. On the pairs the construction places, it separates more strongly (.36 vs .57) but at a fraction of the reach. (3) Importance definition (b) weakens the result (mean opp sim .643 vs .618) and gives recorded members low weight; (a) gives generic hypernyms full weight. (4) Same-pole sharing raises similarity (red/blue .25 -> .89, eat/drink .82 -> .97), so scales add false closeness as well as separation. (5) The degree/negation orderings for non-recorded words are mostly arithmetic on one or two position values; the frozen config places non-members only at +-.85 or 0.

**Unknown.** Whether wider edge types or a different family rule (e.g. also_see families kept apart, or whole-language held out) would change the generalisation figure, since the family split here is imperfect (hypernym-level near-duplicates cross folds); behaviour with all 3,802 pairs recorded rather than 60% (coverage would rise to 21% of senses frozen, 45% wide, test accuracy not measured); effects on the encoder, scorer or memory (nothing downstream was run); stability across seeds and splits (one split, 5 null seeds that coincide in three cases because coverage is tiny); a second reader for the traceability verdicts.
