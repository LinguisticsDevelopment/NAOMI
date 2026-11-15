# USVS feasibility probes: clean-up, importance+position, graph-derived basis (read-only, planning, 2026-11-15)

Nothing in the repo or `data/usvs/` was changed (no git). Scratch: `SCRATCH = /tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_probes/` (all numbers below are in `SCRATCH/results.json`; `show.py <substring>` prints rows). No scipy/sklearn in the env (numpy + torch 2.x only); eigensolver = `torch.lobpcg` on a sparse tensor, dense `torch.linalg.eigh` for the adjective subgraph. No recommendation is made.

## 0. Common protocol (identical for every probe)

- **Level**: SENSE level (117,659 WordNet synsets, the stored 607-axis sense vectors `M`, base artifact `e0daef638b640dd5`). No probe changes core coordinates, so the held-out core harness was not re-run (nothing to re-place); core numbers stay as in the diagnosis.
- **Hold-out (M24 rule)**: for every WordNet relation type separately (antonym, similar_to, also_see, derivational, pertainym, attribute, hypernym, verb_group, entailment, cause) a seeded (`random.Random(0)`) 20% of the edges is held out; **everything built (lookup, poles, values, graph, embeddings) uses only the 80%**. Scored: antonym test = 760 held-out distinct antonym synset pairs (of 3,802; n 205 / v 107 / a 386 / r 62), similar test = 2,139 held-out similar_to (satellite-head) pairs, hypernym test = 10,000 sampled from the 19.5k held-out hypernym edges, sister test = 10,000 pairs whose parent edges are not both in train, MFS-synonym pairs (6,871, lemma pairs sharing a synset, not graph edges), 20,000 random sense pairs (+4,000 per POS), SimLex-999 at sense level (first sense of the stated POS; 999 pairs; SimLex itself is never used for building, but 80% of its antonym pairs ARE recorded edges in the train 80%, so SimLex numbers for lookup/graph variants are partly "recorded-relation" effects), 17 everyday pairs (senses as in diagnosis 2c).
- **Columns**: `syn/rand` = AUC(MFS synonym > random); `sim/rand` = AUC(held-out similar_to > random adjective); `sim>ant` / `syn>ant` = P(a held-out similar_to / MFS-synonym pair is closer than a held-out antonym pair); `ant<rand` = P(held-out antonym pair LESS similar than a random pair) (0.5 = indistinguishable, ~0.02 = antonyms look more similar than random in 98% of draws); `hyp`, `sis` = AUC vs random (hypernym test / sister test); `rand` = mean random-pair similarity; `SL` = SimLex rho.
- **Graph probes (P3)**: the 20% held-out similar_to/hypernym edges also isolate many leaf senses from the train graph, so some test pairs have an unplaced end. Rows say `all` (unplaced pairs score 0) or `placed` (only pairs with both ends placed; `cov` = share of antonym test pairs kept).
- Scripts (re-run order): `mkbase.py` (edges, split, pair sets, gold tokens), `evalib.py` (metrics), `p1.py`, `p2lib.py` `p2runA.py` `p2runA2.py` `p2runB.py` `p2deg.py`, `glib.py` `p3a.py` `p3adj.py` `p3b.py` `p3b_repro.py` `p3hyb.py` `p3names.py`, `cov.py`.

## 1. One-page summary

Baseline = stored sense vectors, plain cosine (raw) or the scorer's IDF-weighted cosine. "By-POS syn>ant" = P(MFS-synonym closer than antonym) n/v/a/r (adjectives are the easy case).

| setting | syn/rand | sim/rand | sim>ant | syn>ant | ant<rand | hyp | sis | rand | SL | syn>ant n/v/a/r | dims | coverage: senses changed or placed (gold content tokens) | deterministic / reproducible | dims nameable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline raw | .693 | .779 | .293 | .082 | .023 | .952 | .978 | .395 | .258 | .057/.069/.144/.190 | 607 | n/a | yes | yes (named) |
| baseline IDF | .673 | .780 | .192 | .069 | .018 | .972 | .986 | .111 | .269 | .034/.033/.137/.172 | 607 | n/a | yes | yes |
| P1 IDF + drop 12 ubiquitous axes | .656 | .726 | .179 | .133 | .078 | .935 | .967 | .080 | .274 | .045/.032/.217/.179 | 595 | 100% (100%) | yes | yes |
| P1 IDF + top-100 axes | .671 | .774 | .199 | .067 | .016 | .973 | .985 | .125 | .264 | n/a | 100 | 100% (100%) | yes | yes |
| P1 + recorded-opposite lookup, held out | same as its base row (0.0% of 760 test pairs caught) | | | | | | | | +.03 | | | lookup knows 5.1% (11.1% with satellites) of senses; gold 21.6% (26.4%) | yes | yes |
| P1 + lookup, test pairs NOT held out (cheat ceiling) | | | 1.00 | .999 | .994 | | | | .305 | | | the same senses | yes | yes |
| P2-A values on existing axes (no lex axes, m=3, x10, cosine) | .661 | .747 | .111 | .070 | .024 | .962 | .981 | .099 | .260 | .033/.034/.080/.135 | 607 x 2 | 5.2% (12.5%) | yes | yes |
| P2-B2 merged scale dims, IDF, lambda=3, cosine | .645 | .709 | .099 | .101 | .041 | .961 | .981 | .094 | .291 | .079/.038/.133/.192 | 607 x 2 + 2,537 | 26.2% (53.8%) | yes | scales named by their poles (not by content) |
| P2-B2 raw weights, lambda=3 | .596 | .582 | .128 | .167 | .109 | .893 | .950 | .332 | .235 | .193/.103/.140/.252 | 607 x 2 + 2,537 | 26.2% (53.8%) | yes | as above |
| P3a spectral (hyp .05, k=100), all | .728 | .536 | .603 | .750 | .631 | .469 | .464 | .115 | .426 | .687/.730/.704/.758 | 100 | 78.8% placed (82.4%) | seeded LOBPCG; eigenvectors unique only up to near-degenerate eigenspaces | narrow topic clusters; yes for ~11/15 |
| P3a spectral, no hypernym (k=50), placed only | .742 | .848 | .871 | .849 | .787 | .748 | .749 | .011 | .622 | .883/.855/.805/.973 | 50 | 20.8% placed (51.2%); 38% of antonym test pairs | as above | as above |
| P3b fitted signed embedding (k=100), all | .717 | .573 | .627 | .705 | .582 | .579 | .579 | .017 | .523 | .642/.691/.678/.806 | 100 | 87.3% placed (88.2%) | seed 0; bit-identical with 1 thread, max diff 1.8e-3 with 4 threads | mostly not (1 of 15) |
| P3c hybrid [IDF sense ; fitted k=100], gamma=1 | .741 | .853 | .469 | .304 | .142 | .974 | .986 | .070 | .530 | .276/.255/.357/.488 | 607 + 100 | 87.3% (88.2%) | as P3b | 607 named + 100 unnamed |
| P3c hybrid, gamma=2 | .731 | .839 | .561 | .504 | .370 | .935 | .957 | .044 | .563 | .516/.524/.487/.620 | 607 + 100 | same | as P3b | same |

(P1 raw-weight "drop 12" and "drop 30" rows are in section 2; `syn>ant n` etc. for the P1 top-100 row were not printed.) Cells of a P3 row marked "all" include pairs that have no graph position (score 0); the `placed` rows exclude them.

## 2. Probe 1: clean-up only (existing vectors, different comparison)

**Done.** Weights per axis `w_j`: raw (1); IDF (`log((N+1)/(df+1))`, equals `score.axis_weights` on the base artifact); drop the k = 3/12/30 highest-df axes (SOMETHING, ONE, PART for k=3; +BODY, WHEN, KIND, lex:noun.act, PEOPLE, WHERE, lex:adj.all, lex:noun.artifact, lex:noun.communication for k=12); drop 47 dead axes + 112 near-duplicate axes (greedy, column correlation > 0.9 over senses; keep lowest index); keep top-K = 50/100/200 axes by IDF-weighted energy. Each combined with a recorded-opposite lookup that sets the pair similarity to 0: (i) direct (exact synset pair in the train antonym set), (ii) cluster (satellite -> head via train similar_to, then (head, head) pair in the train set), (iii) same lookups built from ALL antonym edges (test pairs not held out). Script `p1.py`.

| setting | syn/rand | sim/rand | sim>ant | syn>ant | ant<rand | hyp | sis | rand | SL | SL with lookup (train) |
|---|---|---|---|---|---|---|---|---|---|---|
| a raw | .693 | .779 | .293 | .082 | .023 | .952 | .978 | .395 | .258 | .294 (cluster) |
| b IDF | .673 | .780 | .192 | .069 | .018 | .972 | .986 | .111 | .269 | .298 |
| c IDF drop 3 / raw drop 3 | .669 / .656 | .781 / .794 | .192 / .276 | .068 / .061 | .018 / .012 | .972 | .986 | .105 / .184 | .269 / .266 | .299 / .302 |
| c IDF drop 12 / raw drop 12 | .656 / .650 | .726 / .726 | .179 / .181 | .133 / .136 | .078 / .079 | .935 | .967 | .080 / .106 | .274 / .284 | .298 / .309 |
| c IDF drop 30 / raw drop 30 | .615 / .615 | .679 / .676 | .215 / .214 | .153 / .154 | .104 / .105 | .849 | .915 | .051 / .057 | .209 / .209 | .231 / .231 |
| c IDF dead+dup / + drop 12 | .673 / .656 | .782 / .728 | .192 / .178 | .067 / .132 | .017 / .077 | .972 / .935 | .986 / .967 | .112 / .081 | .269 / .273 | .299 |
| d top-50 / 100 / 200 (IDF) | .669 / .671 / .671 | .757 / .774 / .784 | .193 / .199 / .197 | .071 / .067 / .066 | .018 / .016 / .017 | .970 / .973 / .972 | .980 / .985 / .986 | .124 / .125 / .117 | .273 / .264 / .262 | .304 / .297 / .296 |

**Lookup catch rate (share of the 760 held-out antonym pairs flagged as opposite)**: direct lookup built on train edges 0.0% (every test pair is a different synset pair); cluster/satellite inference on train edges 0.0% (n/v/a/r all 0.0%). Reason, measured: all 3,802 WordNet antonym pointers sit on head/non-satellite synsets (0 pairs have a satellite end), so satellite-via-head inference can only flag pairs whose HEAD pair is already recorded; of 760 test pairs only 43 have either end in a train pair and 28 have both. Lookup built from all edges flags 100% (that is the lookup answering its own table: a cheat ceiling, sim>ant 1.00). False flags: 0 of 2,139 similar_to, 0 of 20,000 random, 1 of 6,871 MFS-synonym pairs. Lookup effect on SimLex (+.03: 0.269 -> 0.298) comes from SimLex antonym pairs that are in the train edges.

**What would have to be true for it to work in the real system.** The opposite pair must be recorded in the store (the project's tiered store, 11,317 pairs, is exactly this); for unrecorded opposites (happy/sad, buy/sell when held out) the comparison alone cannot do anything. Dropping ubiquitous axes is the only clean-up with a held-out effect: dropping 12 raises syn>ant .069 -> .133 and ant<rand .018 -> .078, at the price of sim/rand .780 -> .726 and hyp .972 -> .935; dropping 30 doubles the effect and costs hyp .85, SimLex .21.

**Fragile.** The gain from axis dropping is small relative to the gap (antonyms stay more similar than random in 92% of draws at k=12); IDF barely helps antonymy (it does remove the generic floor: random .395 -> .111); the satellite lookup needs the head pair recorded; all numbers use first-sense/MFS choices for the "synonym" set (same-synset synonyms share one vector, so a clean sense-level synonym test does not exist; `sim` uses similar_to pairs instead).

## 3. Probe 2: importance (weight) + position (value)

**Done (`p2lib.py`, `p2runA.py`, `p2runA2.py`, `p2runB.py`).** Weight channel = existing non-negative sense values, `raw`, or `idf` (`M * sqrt(IDF)`, which makes the plain cosine equal the scorer's IDF cosine). Value channel only from the 3,042 train antonym pairs. Similarity: (i) cosine on `[w ; w*v]`; (ii) Gower as specified (`sum min(w_a,w_b)*agree / sum min(w_a,w_b)`, `agree = 1 - |v_a - v_b|/2`); (iii) Gower with the denominator `sum max(w_a,w_b)` ("Gower-max"). **Finding: Gower as specified is degenerate here**: with value +1 on almost every dimension both words agree on nearly everything, so every pair scores ~0.98 and AUCs sit at 0.48-0.50 (random pair mean 0.982, `P2A|...|gower` rows); only the max-normalised version keeps weight information (rand .083, behaves like a weighted Jaccard).

*Variant A (values on existing axes).* For each train antonym pair, members = head + its train satellites per pole; shared_j = min(mean weight of pole 1, mean weight of pole 2); score_j = shared_j * IDF_j; the top m = 1 or 3 axes (ties by axis index) get value +1 on pole 1 and -1 on pole 2 members (default +1 elsewhere; first assignment wins on conflicts). Run twice: all axes allowed (the rule then picks `lex:adj.all`, `lex:adv.all`, `lex:noun.attribute`: 773 of 3,042 pairs choose `lex:adj.all`), and with `lex:` axes excluded (`nolex`; then `attr:music`, `attr:fashion`, LIVE, TRUE, ... are chosen). Scaling x1/x3/x10 multiplies the existing weight on the chosen axes of cluster members. 6,067 senses (5.2%) receive any non-default value.

| A variant (IDF weights, cosine) | ant_tr (train antonyms, in-sample) | ant test mean (held out; base .718) | sim>ant | syn>ant | ant<rand | hyp | sim/rand | rand |
|---|---|---|---|---|---|---|---|---|
| all axes, m=1, x1 / x3 / x10 | .399 / .194 / .069 | .718 / .715 / .712 | .176 / .160 / .137 | .072 / .073 / .078 | .022 / .024 / .031 | .963 / .956 / .946 | .766 / .756 / .731 | .109 / .104 / .099 |
| no-lex, m=1, x1 / x3 / x10 | .649 / .462 / .211 | .724 / .721 / .714 | .185 / .165 / .137 | .068 / .067 / .070 | .018 / .018 / .022 | .972 / .971 / .964 | .776 / .780 / .757 | .110 / .107 / .102 |
| no-lex, m=3, x1 / x3 / x10 | .595 / .352 / .121 | .723 / .719 / .713 | .180 / .149 / .111 | .068 / .067 / .070 | .018 / .018 / .024 | .972 / .971 / .962 | .773 / .775 / .747 | .110 / .106 / .099 |

Raw-weight versions are in `results.json` (`P2A|raw|...`): same pattern (held-out antonym mean .84 -> .83 at x10; hyp .95 -> .92).

**Distinguishing-dimension share** (best shared axis per antonym pair by the rule above; share = w_d^2 / |w|^2 averaged over both members): with `lex:` axes allowed raw 23% / IDF 36% (but the chosen axis is a lexicographer category, not a differentia); with `lex:` excluded raw **4.4%** (median 1.6%, p90 12.8%), IDF 9.5% train / 8.8% test (median 5%, p90 24%). The analytic 1 - 2 w_d^2/|w|^2 therefore predicts opposites still at cosine ~0.9 under the weights as they are, consistent with the measured .72-.84.

*Variant B (new scale dimensions).* B1: one dimension per train antonym head pair (3,042 dims), +1/-1 on the two heads. B2: merge pairs whose members share an attribute noun (train `attribute` edges) or are linked by also_see (2,537 dims); poles by signed colouring (also_see = same pole; attribute-only links oriented by cosine to the poles already collected); 20 further adjectives assigned to a scale by attribute noun and nearest pole. Propagation (deterministic, a sense keeps the highest-weight entry per dimension; relative weight g): L0 heads g=1; L1 satellites (train similar_to) g=.5; L2 derivational + pertainym neighbours g=.25; L3 one hypernym/hyponym step g=.125. Weight on a scale dimension c = lambda * g * |w_sense|, so the scale's share of the squared weight is lambda^2/(1+lambda^2): lambda .3 / .9 / 3 = 8% / 45% / 90% (grid "x1/x3/x10" of a 0.3 base). Flat vector `[w ; w*v]` (existing dims v=+1) + `[c ; c*v]`.

Coverage after each propagation step (B1; B2 within 0.1 point): L0 5,950 senses (5.1%; gold content tokens 21.6%); L1 13,014 (11.1%; 26.4%); L2 18,899 (16.1%; 37.9%); L3 30,798 (26.2%; 53.7%). By POS at L3: n 16.7%, v 35.4%, a 58.7%, r 42.6%. Gold by UPOS at L3: NOUN 42%, VERB 64%, ADJ 80%, ADV 26%. Held-out antonym test pairs (760) after L3, B1 / B2: **separated by construction (same scale, opposite poles) 171 / 182**, same pole (wrong) 22 / 37, only one end on a scale 210 / 190, neither end 357 / 351. All 3,042 train pairs separated (3,035 in B2).

| B (cosine unless noted) | lambda (scale share) | ant_tr (in-sample) | ant test mean | sim>ant | syn>ant | ant<rand | hyp | sis | sim/rand | rand | SL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 IDF | .3 (8%) | .664 | .720 | .180 | .068 | .018 | .972 | .986 | .777 | .110 | |
| B1 IDF | .9 (45%) | .392 | .694 | .133 | .069 | .020 | .972 | .986 | .755 | .106 | |
| B1 IDF | 3 (90%) | .070 | .587 | .104 | .098 | .042 | .961 | .981 | .725 | .093 | .268 |
| B2 IDF | .3 / .9 / 3 | .665 / .395 / .073 | .720 / .696 / .596 | .180 / .131 / .099 | .068 / .070 / .101 | .018 / .019 / .041 | .972 / .972 / .961 | .986 / .986 / .981 | .776 / .747 / .709 | .110 / .106 / .094 | .291 (3) |
| B2 IDF Gower-max | .3 / .9 / 3 | .432 / .366 / .244 | .459 / .445 / .412 | .204 / .185 / .146 | .068 / .068 / .072 | .019 / .019 / .024 | .973 / .972 / .964 | | .800 / .791 / .759 | .082 / .080 / .076 | |
| B2 raw | .3 / .9 / 3 | .777 / .461 / .085 | .839 / .810 / .693 | .255 / .148 / .128 | .081 / .091 / .167 | .023 / .033 / .109 | .953 / .948 / .893 | | .767 / .682 / .582 | .392 / .378 / .332 | |

How much importance before held-out opposites separate: at 90% scale share, held-out antonym mean cosine only falls .72 -> .60 (IDF) and ant<rand .018 -> .041 (raw: .023 -> .109), because only ~22-24% of held-out pairs get a shared scale at all; the in-sample pairs fall to .07. The cost appears at the same setting: sim/rand .780 -> .709, hyp .972 -> .961 (IDF), and with raw weights hyp .952 -> .893. The sim>ant column FALLS (.19 -> .10) because held-out similar_to pairs are exactly the satellites whose cluster membership is unknown (13 of 2,139 share a scale): an unlabelled satellite looks unrelated to labelled cluster members. Gower-spec rows (degenerate) are in `results.json` (`P2B...|gower`).

**Degree / negation by arithmetic only** (`p2deg.py`; values in [-2,2], recorded poles +1/-1 from the recorded opposite sense; very X = 1.5, slightly X = 0.3, not X = -1, opp(X) = the recorded opposite sense's own weights with value -1; IDF weights; scale share 90%, lambda=3). Full 5x5 for happy.a.01 (rows/cols X, very, slightly, not, opp):

| | X | very X | slightly X | not X | opp(X)=unhappy |
|---|---|---|---|---|---|
| X | 1.00 | .98 | .89 | .10 | .09 |
| very X | .98 | 1.00 | .78 | -.10 | -.11 |
| slightly X | .89 | .78 | 1.00 | .54 | .53 |
| not X | .10 | -.10 | .54 | 1.00 | .99 |
| opp(X) | .09 | -.11 | .53 | .99 | 1.00 |

Same at lambda=3 for hot/cold, big(large)/small, good/bad: (X,very) .98, (X,slightly) .89, (X,not) .10, (X,opp) .075/.097/.074, (not X, opp) **.975 / .997 / .974**. At lambda=.9 (45% share): happy (X,very) .983, (X,slightly) .945, (X,not) .552, (X,opp) .497, (not X, opp) .945; hot/big/good (X,opp) .414/.537/.411, (not X, opp) .862/.985/.859. Reading: ordering very > slightly > not > opposite is produced by the arithmetic, but **"not X" and the recorded opposite collapse onto each other (.86-.997)** because both are value -1 on the same scale; they differ only through the remaining weight differences. Degree and negation are distinguishable from X, but negation vs the lexical opposite is not distinguished by this scheme without a further channel.

**What would have to be true.** (1) A pole relation must exist for the pair or for a related pair (coverage 26% of senses after one hypernym step; 5% direct; nouns 17%, adverbs 43%, adjectives 59%); (2) the differentia must carry real weight (4-9% share today; the held-out gain needs the scale at ~90% share); (3) a way to label members of a bipolar cluster that the recorded pair does not reach (held-out satellites); (4) the decision whether operators act on values (O2 of the alternatives report).

**Fragile / went wrong.** Held-out generalisation is almost all via derivational/hypernym relatives of recorded pairs (near-duplicates of train pairs, e.g. happiness/unhappiness), not via new scales; the poles are only as good as propagation (22-37 of the 193-219 held-out pairs that share a scale are on the SAME pole); attribute merging gave 3,042 -> 2,537 dims and 20 extra adjectives only (there are 639 attribute edges); red/blue goes UP (.46 -> .83) when both sit on one scale with the same pole (scales add shared mass); scaling a weight makes both channels follow it (hurts sim/rand and hyp); Gower as specified is degenerate (flat ~0.98).

## 4. Probe 3: graph-derived basis

**Graph (`glib.py`).** Nodes = the 117,659 synsets. Train edges (80%): positive weight 1: similar_to 8.6k, also_see 1.2k, derivational (lemma-level -> synset) 25.5k, pertainym 5.3k, attribute 0.5k, verb_group 0.7k, entailment 0.3k, cause 0.2k; hypernym/instance-hypernym 78k at weight 0.3 (variants 0.05 and 0, i.e. "without"); negative weight -1: antonym 3,042. Duplicate undirected pairs summed. 121,893 edges in the 0.3/0.05 variants, 43,792 without hypernyms.

**(a) Signed spectral embedding** (`p3a.py`): normalised signed Laplacian `I - D^-1/2 A D^-1/2` with `D = sum |w|`; the 200 smallest eigenvectors (largest of the signed normalised adjacency) by `torch.lobpcg` on the sparse giant component, seed 0, 120 iterations, k = 10/20/50/100/200 = first k columns (nested), cosine of rows (row scale irrelevant). Eigenvalue ladder (of the adjacency, 1 = Laplacian 0): .99983, .9992 (#2), .9981 (#10), .9973 (#20), .9950 (#50), .9923 (#100), .9883 (#200): no gap, nearly degenerate (residual max 1.0e-3, median 0). Giant component and fallback:

| variant | giant comp (senses placed) | outside giant | of which no train edge | components | gold tokens placed |
|---|---|---|---|---|---|
| hyp .3 / .05 | 92,739 (78.8%; n 78.3% v 91.9% a 76.2% r 53.9%) | 24,920 (n 17,814 v 1,117 a 4,321 r 1,668) | 14,903 | 17,341 | 82.4% (NOUN 93% VERB 97% ADJ 89% ADV 15%) |
| no hypernym | 24,487 (20.8%) | 93,172 | 72,011 | 78,872 | 51.2% |

Deterministic fallback used: senses outside the giant component get a zero graph block (the hybrid then equals the existing vector; pure-graph pairs score 0, "all" rows; "placed" rows drop them). A per-component embedding would need coordinates shared across components, which a spectral basis does not provide; the 20% held-out hypernym/similar edges are what push 14.9k leaves out (the diagnosis' lemma-hypernym relation otherwise connects them).

| spectral, all (unplaced = 0) | k | syn/rand | sim/rand | sim>ant | syn>ant | ant<rand | hyp | sis | rand | SL |
|---|---|---|---|---|---|---|---|---|---|---|
| hyp .3 | 10 / 20 / 50 / 100 / 200 | .707 / .719 / .743 / .748 / .767 | .504 / .505 / .506 / .512 / .548 | .498 / .495 / .508 / .509 / .561 | .709 / .713 / .721 / .713 / .730 | .569 / .564 / .559 / .543 / .560 | .436 / .438 / .450 / .471 / .513 | .434 / .437 / .451 / .471 / .513 | .323 / .308 / .251 / .219 / .132 | .325 / .321 / .323 / .362 / .403 |
| hyp .05 | 10 / 20 / 50 / 100 / 200 | .682 / .699 / .719 / .728 / .733 | .519 / .525 / .534 / .536 / .553 | .576 / .583 / .601 / .603 / .604 | .738 / .747 / .753 / .750 / .742 | .628 / .634 / .638 / .631 / .615 | .449 / .451 / .459 / .469 / .492 | .443 / .446 / .455 / .464 / .488 | .257 / .214 / .145 / .115 / .084 | .332 / .369 / .405 / .426 / .417 |
| no hyp, placed only (38% of ant pairs) | 10 / 50 / 200 | .737 / .742 / .713 | .818 / .848 / .846 | .866 / .871 / .880 | .846 / .849 / .859 | .776 / .787 / .812 | .744 / .748 / .724 | .735 / .749 / .716 | .018 / .011 / .003 | .642 / .622 / .602 |
| hyp .05, k=100, placed only (65% of ant pairs) | 100 | .713 | .773 | .719 | .713 | .624 | .771 | .747 | .184 | .473 |

By POS (`all`, hyp .05, k=100): syn>ant n/v/a/r .687/.730/.704/.758; ant<rand .588/.624/.589/.702; held-out antonym mean cosine n/v/a/r **-.032/+.038/-.079/-.290** (POS-matched random pairs .119/.295/.038/.023). Hypernym/sister AUCs below .5 in `all` rows are an artefact of the pair set: 69% of hypernym test pairs have an unplaced end (their child lost its only train edge) and score 0.

**Adjective-only subgraph (reproduces the alternatives report's probe on this split; `p3adj.py`)**: 18,156 adjective synsets, train similar_to/also_see (+1), antonym (-1); giant 4,764 nodes (1,260 further components of size >= 2); dense eigh, unnormalised and normalised Laplacian; both give the same: held-out antonym mean cosine **-0.86 (k=5), -0.85 (10), -0.84 (20), -0.79 (50), -0.67 (100), -0.36 (200)**, ant<rand .93/.93/.94/.94/.93/.76 (67 of 760 test pairs placed, = 17% of adjective test pairs; syn-pair coverage only 1.3%, so syn>ant is not reliable there: .94-.95 on ~86 pairs). So the strong signed-spectral result exists, but only for adjective head pairs inside the 4,764-node giant, which is 8.8% of all held-out antonym pairs.

**(b) Fitted signed embedding** (`p3b.py`, fixed seed, fully logged). Objective on unit-normalised rows z: `L = sum_{E+} w_ij (1 - z_i.z_j)/sum w + 3 * mean_{E-} relu(z_i.z_j + 0.5) + mean_{100k random pairs} (z_i.z_j)^2` (attract positives, margin -0.5 for antonym edges, decorrelate random pairs); same train edges/weights as (a) with hyp .3; init N(0,1)/sqrt(k), `torch.manual_seed(0)`, Adam lr .02, 400 full-batch iterations. Convergence k=20 (train antonym cosine -0.003 -> -0.714 at it 200 -> -0.700 at 400): loss 2.542 (it 0) -> 0.114 (200) -> 0.096 (400); pos term 1.000 -> .063 -> .045; random-pair term .050 -> .050 -> .051 (full curves for k=20/50/100 in `p3b_curves.json`). Reproducibility: two runs with seed 0 and 4 threads differ by up to **1.8e-3** per coordinate (not bit-identical; metrics identical to 4 decimals); with 1 thread two runs are **bit-identical** (sha1 equal, `p3b_repro.log`); seed 1 gives different coordinates with similar metrics (placed rows: ant<rand .575 vs .573, SimLex .458 vs .417 at k=20).

| fitted | syn/rand | sim/rand | sim>ant | syn>ant | ant<rand | hyp | sis | rand | SL | placed-only ant<rand / hyp / cov |
|---|---|---|---|---|---|---|---|---|---|---|
| k=20 | .660 | .564 | .606 | .670 | .563 | .558 | .554 | .027 | .417 | .573 / .692 / .758 |
| k=50 | .707 | .569 | .625 | .704 | .583 | .574 | .571 | .021 | .512 | .597 / .739 / .758 |
| k=100 | .717 | .573 | .627 | .705 | .582 | .579 | .579 | .017 | .523 | .597 / .753 / .758 |

By POS (k=100, all): syn>ant n/v/a/r .642/.691/.678/.806; ant<rand .546/.565/.590/.753.

**(c) Hybrid** (`p3hyb.py`): flat vector `[ w_idf/|w_idf| ; gamma * g/|g| ]`, g = graph block of the sense (zero if unplaced), cosine on the concatenation.

| hybrid | gamma | syn/rand | sim/rand | sim>ant | syn>ant | ant<rand | hyp | sis | rand | SL |
|---|---|---|---|---|---|---|---|---|---|---|
| IDF block only (baseline) | 0 | .673 | .780 | .192 | .069 | .018 | .972 | .986 | .111 | .269 |
| + fitted k=100 | .5 / 1 / 2 | .722 / .741 / .731 | .819 / .853 / .839 | .270 / .469 / .561 | .109 / .304 / .504 | .026 / .142 / .370 | .977 / .974 / .935 | .988 / .986 / .957 | .095 / .070 / .044 | .386 / .530 / .563 |
| + fitted k=50 | 1 | .736 | .841 | .463 | .305 | .141 | .970 | .984 | .072 | .524 |
| + spectral hyp .05 k=200 | .5 / 1 / 2 | .739 / .755 / .746 | .807 / .816 / .786 | .280 / .450 / .497 | .127 / .386 / .532 | .034 / .252 / .382 | .973 / .949 / .865 | .987 / .971 / .901 | .110 / .108 / .103 | .382 / .488 / .473 |
| + spectral hyp .05 k=50 | .5 / 1 / 2 | .738 / .745 / .734 | .802 / .785 / .733 | .285 / .457 / .502 | .138 / .414 / .566 | .039 / .276 / .413 | .970 / .922 / .795 | .985 / .956 / .843 | .122 / .138 / .152 | .376 / .459 / .441 |

By POS (hybrid fitted k=100, gamma 1): syn>ant n/v/a/r .276/.255/.357/.488, ant<rand .138/.122/.263/.577, held-out antonym mean cosine .44/.42/.38/.38 (random .08/.10/.15/.45). With gamma=2 syn>ant .516/.524/.487/.620.

**(d) Nameability (honest judgement, `p3names.py`, lists in `p3names.json`).** *Spectral, first 15 eigenvectors (hyp .05):* 12 high + 12 low loading synsets per dimension. Each is a narrow topical cluster, not a general axis, and several are near-duplicates (dims 3 and 7 both "parasitic plants"; 4 and 8 South American nations; 5 and 9 stars; 2 and 12 constructed languages; 11 and 14 ears/body). Examples: d0 hi bird genera (accipiter, platalea, ...) / lo waning, ebb, monotony; d1 hi ordinals (sixth ... zeroth) / lo cardinals (nineteen, billion, ...) = "ordinal vs cardinal numerals"; d6 hi fish families (-idae) / lo herring, salmon; d9 hi stars (sirius, rigel, ...) / lo meteorites; d12 hi Bantu languages (Sotho, Zulu) / lo constructed languages; d10 hi electric charge (positive, negative, anode) / lo neutral, uncharged, absorb. Verdict: **nameable 11 of 15 (as a topic; the two poles are usually two different topics)**, vague 3 (d5, d11, d13: mixed poles), 1 mixed; lexicographer-file purity of the 12 top words 9-12/12 for most dims; USVS-space intruder check (10 top words + 1 random sense, intruder = lowest mean cosine; 100 draws/dim; chance .09): mean hit .83. *Fitted k=50, 15 highest-variance dimensions:* coherence and purity are lower (mean pairwise USVS cosine .26 vs .53; intruder hit .42; top-12 lexname purity usually 3-7/12). Verdict: **nameable 1 of 15 (d46 "fairness / justice / equitable"), vague 8 (one pole nameable: d6 hi unattractive/introspection, lo neuroscientists; d12 lo soil/loam; d0 lo fish/fauna; d4 lo Romance languages; d3 lo reply/answer; d37 disproportionate vs cherished; d20 unsympathetic vs consciousness; d8 kits/systems vs marriage), not nameable 6 (d2, d25, d17, d34, d23, d38 mixed)**. A fitted basis is not rotation-canonical, so the same objective with another seed gives different (equally unnamed) coordinates.

**What would have to be true for it to work in the real system.** The graph-derived coordinates carry antonym/relatedness information that the stored vectors lack (the hybrid at gamma 1 raises syn>ant .069 -> .30 and SimLex .27 -> .53 while keeping hyp .974), so a graph block next to the existing block is the version that does not lose the existing structure; to be usable as the meaning space it would need (1) positions for the 21% of senses that have no train edge (a fallback that is not the zero vector), (2) a canonical orientation/rotation or the dimensions are not stable identifiers across rebuilds, (3) stable labels (nameability above is poor for the fitted basis and topic-local for the spectral one), (4) a decision whether closed-class words (outside WordNet) enter the graph.

**Fragile / went wrong.** Held-out antonymy is carried partly by neighbours: an antonym pair whose derivational/hypernym relatives are recorded opposites is pushed apart through them (sibling leakage, as in P2); overall held-out antonyms are still more similar than random in 37-86% of draws depending on gamma (ant<rand .14 at gamma 1); the global signed spectral basis is dominated by localised eigenvectors (first 15 = small taxon/topic clusters, eigenvalues 0.9998-0.998, no gap), which is why the adjective-only graph (-0.85 antonym cosine) works and the full graph (+0.12 to -0.03 antonym cosine, hyp AUC .47) does not; hypernym edges at weight .3 hurt antonymy vs .05 (ant<rand .543 vs .631 at k=100); the "no hypernym" setting is excellent where it places (ant<rand .79-.81) but places 21% of senses; SimLex gains are partly in-sample for recorded relations (80% of SimLex antonym pairs are train edges); the 4-thread fit is not bit-reproducible; hold-out of 20% of hypernym/similar_to edges removes the only train edge of many leaf senses (14.9k senses), so graph coverage in a real build (all edges) would be higher than reported here.

## 5. The 17 everyday pairs (sense level; `rec`: T = antonym edge in the 80% used to build, H = held out, - = not a recorded antonym pair)

| pair | rec | baseline raw | baseline IDF | P1 IDF drop-12 | P2-A no-lex m3 x10 | P2-B2 lambda=3 | P3a spectral hyp .05 k100 | P3b fitted k100 | P3c hybrid gamma 1 |
|---|---|---|---|---|---|---|---|---|---|
| happy/unhappy | T | .89 | .90 | .90 | .03 | .08 | -.72 | -.70 | .10 |
| happy/sad | - | .85 | .88 | .88 | .02 | .08 | -.70 | -.68 | .10 |
| hot/cold | T | .92 | .75 | .63 | .40 | .07 | -.89 | -.74 | .01 |
| good/bad | T | .86 | .74 | .68 | .14 | .07 | -.20 | -.58 | .08 |
| big/small | T | .99 | .97 | .96 | .07 | .10 | -.25 | -.61 | .18 |
| love/hate | T | .82 | .87 | .88 | .04 | .09 | -.97 | -.52 | .18 |
| buy/sell | H | .84 | .80 | .81 | .80 | .66 | .93 | .79 | .80 |
| alive/dead | T | .88 | .55 | .27 | .53 | .05 | -.58 | -.74 | -.10 |
| dog/cat | - | .84 | .83 | .84 | .83 | .83 | .97 | .48 | .66 |
| dog/bone | - | .64 | .40 | .38 | .40 | .32 | .00 | .04 | .22 |
| bone/stone | - | .47 | .03 | .00 | .03 | .02 | .05 | .14 | .09 |
| eat/chew | - | .77 | .83 | .83 | .83 | .69 | .96 | .92 | .87 |
| eat/drink | - | .87 | .82 | .82 | .82 | .69 | .44 | .72 | .77 |
| king/queen | - | .26 | .03 | .01 | .02 | .01 | .14 | .02 | .03 |
| red/blue | - | .81 | .46 | .25 | .46 | .83 | 1.00 | .98 | .72 |
| big/large | - | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| quick/fast | - | .94 | .67 | .43 | .67 | .67 | .96 | .99 | .83 |

Reading the rows honestly: every "T" row that falls is in-sample (that pair, or its cluster, was a build input); the only held-out antonym pair, buy/sell, is NOT moved by P1, P2-A (.80), P2-B2 (.66 vs .80 idf) and is MORE similar in the spectral (.93) and fitted (.79) graph spaces (no recorded neighbour pushes it apart); happy/sad drops in P2 only because sad is a satellite of unhappy (cluster inference, not held out in the build). Spectral/fitted graph values are raw graph-embedding cosines (unplaced -> 0); dog/cat .97 vs eat/drink .44 in the spectral space shows the noun-hierarchy bias.

## 6. What the probes do NOT show

- **No downstream effect**: nothing tested the tree scorer, memory read/recall, the encoder or any learned component; the numbers are pair-ranking statistics on WordNet-derived pairs, which are also what the vectors were built from (circularity of WordNet-built vectors scored on WordNet relations).
- **Held-out is relation-edge level, not concept level**: 20% of edges removed, but derivational/hypernym relatives of a held-out pair remain in train, so P2-B and P3 "generalisation" to held-out antonyms is mostly morphological/taxonomic siblings of recorded pairs. No held-out CLUSTER, held-out LEMMA FAMILY or held-out language was tested; no second inventory (Spanish/OMW) was tried.
- **Sense-level only; first-sense/MFS choices** drive the synonym sets and SimLex (the gold chosen_sense is MFS-like); same-synset synonyms are identical vectors, so a clean sense-level synonym AUC does not exist (`sim` uses similar_to pairs, `syn` uses MFS lemma pairs).
- **Core-coordinate harness not re-run** (no probe changes core coordinates).
- **Single seed / single split** for the 80/20 hold-out and for the fits (seed 1 checked for one fit at k=20 only); no confidence intervals; the adjective-subgraph spectral numbers rest on 67 antonym pairs and ~86 synonym pairs.
- **Gower as specified is degenerate** here; Gower-max is my substitution, not the owner's formula. Weight scaling in P2 (lambda) is a grid, not derived from definitions; no weight source other than the stored vectors (genus/differentia/depth variants, P23 of the alternatives report) was tested.
- **Closed-class/function words, trees, composition (degree, negation inside trees), NSM primes** are out of scope; the degree/negation test is arithmetic on flat vectors only.
- **Nameability** is my own judgement of word lists plus an automatic intruder proxy measured in the existing USVS space; no independent human or LLM labeller, no held-out membership test, no second labelling run.
- **Eigen-solver accuracy**: LOBPCG residuals <= 1e-3 after 120 iterations on near-degenerate eigenvalues; individual eigenvectors are not unique, only subspaces; k-nesting of the first k columns therefore depends on the solver.
- **Coverage on gold** uses the gold v6 `chosen_sense` of the 211,042 content tokens, not a WSD system; it measures reach of a probe, not accuracy there.
