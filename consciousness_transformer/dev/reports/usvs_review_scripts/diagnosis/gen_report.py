import json
S = "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag"
L = lambda n: json.load(open(f"{S}/{n}.json"))
P, A, G, K, R6, X, SC, HC, DEN = L("m2_profile"), L("m3_axes_eigen"), L("m4_signs_pairs"), None, L("m6_relations"), L("m2b_extra"), L("scorecard"), L("m5_heldout_core"), L("m3c_core_density")
f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
out = []
w = out.append
def table(head, rows):
    w("| " + " | ".join(head) + " |"); w("|" + "|".join("---" for _ in head) + "|")
    for r in rows: w("| " + " | ".join(str(c) for c in r) + " |")
    w("")

w(open(f"{S}/report_head.md").read())

# ---------------- section 2 tables
w("### 2a. Cosine distributions (production artifact)\n")
w("Pair definitions are in section 0. `raw` = plain cosine; `idf` = the scorer's weighted cosine (`sum w*a*b / sqrt(sum w*a^2 * sum w*b^2)`, `w = log((N+1)/(df+1))`). **Core-level numbers on the production artifact are in-sample** (the core was placed by propagating over all synonym/similar/also_see edges, i.e. over these very pairs); held-out core numbers follow in 2a'. Sense-level antonym pairs use the exact antonym synsets; sense-level `syn` is MFS(a) vs MFS(b) of a shared-synset lemma pair (pairs that share the synset at MFS have identical vectors and are excluded, n=6,871 of 7,480).\n")
rows = []
for k in ("core|raw", "core|idf", "sense|raw", "sense|idf"):
    b = P["base"][k]
    rows.append([k, f(b["syn_mean"]) + " / " + f(b["syn_median"]), f(b["ant_mean"]) + " / " + f(b["ant_median"]), f(b["hyp_mean"]) + " / " + f(b["hyp_median"]), f(b["cohyp_mean"]) + " / " + f(b["cohyp_median"]), f(b["rand_mean"]) + " / " + f(b["rand_median"])])
table(["level|weighting", "synonym mean/median", "antonym (direct) mean/median", "hypernym-hyponym", "co-hyponym (sisters)", "random"], rows)
rows = []
for k in ("core|raw", "core|idf", "sense|raw", "sense|idf"):
    b = P["base"][k]
    rows.append([k, f(b["AUC_syn_vs_rand"]), f(b["AUC_syn_vs_ant"]), f(b["AUC_ant_below_rand"]), f(b["P_ant_closer_than_syn"]), f(b["AUC_hyp_vs_rand"]), f(b["AUC_cohyp_vs_rand"]), f(b["AUC_hyp_vs_cohyp"])])
table(["level|weighting", "AUC syn>random", "AUC syn>ant (P cos_syn>cos_ant)", "AUC ant<random (P ant less similar than random)", "**(b)** P(antonym pair closer than a random synonym pair)", "AUC hyp>random", "AUC sister>random", "AUC hyp>sister"], rows)
w(f"Extra: core-level with the project's own (morphy-expanded) synonym edge set (n=22,856, same as the ledger's edge count): raw mean {f(P['base']['core|raw']['syn_proj_mean'])}, AUC>random {f(P['base']['core|raw']['AUC_syn_proj_vs_rand'])}, AUC>ant {f(P['base']['core|raw']['AUC_syn_proj_vs_ant'])}. Sense-level, all 3,992 direct antonym lemma pointers (not restricted to core words): mean cos raw {f(P['base']['sense|raw']['ant_all_mean'])} / idf {f(P['base']['sense|idf']['ant_all_mean'])}; P(syn>ant) {f(P['base']['sense|raw']['AUC_syn_vs_ant_all'])} / {f(P['base']['sense|idf']['AUC_syn_vs_ant_all'])}.\n")
sr, si = X["simto|raw"], X["simto|idf"]
w(f"**Sense-level near-synonym control** (because same-synset synonyms share one vector, the MFS-pair 'synonym' set above is not a clean synonym test). WordNet `similar_to` satellite->head pairs (n={sr['n']:,}), the closest sense-level synonymy WordNet records: mean cos raw {f(sr['mean'])} / idf {f(si['mean'])}; random adjective-synset pair mean {f(sr['rand_adj_mean'])} / {f(si['rand_adj_mean'])}; AUC>random-adjective {f(sr['AUC_vs_rand_adj'])} / {f(si['AUC_vs_rand_adj'])}; antonym mean {f(sr['ant_mean'])} / {f(si['ant_mean'])}; P(similar_to pair more similar than an antonym pair) = {f(sr['AUC_simto_vs_ant(P simto>ant)'])} (raw) / {f(si['AUC_simto_vs_ant(P simto>ant)'])} (idf), i.e. an antonym pair is closer than a near-synonym pair in {f(1-sr['AUC_simto_vs_ant(P simto>ant)'],3)} / {f(1-si['AUC_simto_vs_ant(P simto>ant)'],3)} of draws.\n")

w("### 2a'. Core level on genuinely held-out pairs (M24 rule; placement re-run with 50% of synonym/similar pairs withheld, 10k winners set)\n")
rows = []
for k in ("raw", "idf"):
    h = HC[k]
    rows.append([k, f(h["means"]["syn"]), f(h["means"]["ant"]), f(h["means"]["isa"]), f(h["means"]["rand"]), f(h["AUC_syn_vs_rand"]), f(h["syn_gt_ant (P cos_syn>cos_ant)"]), f(h["AUC_ant_below_rand"]), f(h["AUC_isa_vs_rand"])])
table(["weighting", "mean syn", "mean ant", "mean is-a", "mean random", "AUC syn>rand", "AUC syn>ant", "AUC ant<rand", "AUC is-a>rand"], rows)
w(f"n: test syn {HC['raw']['n']['syn']:,}, test ant {HC['raw']['n']['ant']}, test is-a {HC['raw']['n']['isa']:,}, test similar {HC['raw']['n']['sim']}, random {HC['raw']['n']['rand']:,}. Cross-check: the harness' own `discrimination()` on this placement returns {HC['harness_discrimination_syn_ant_raw'][0]:.4f} (n_syn {HC['harness_discrimination_syn_ant_raw'][1]:,}, n_ant {HC['harness_discrimination_syn_ant_raw'][2]}).\n")

w("### 2a''. By part of speech (production artifact; POS from the pair set; random pairs POS-matched)\n")
rows = []
for k in ("core|raw", "core|idf", "sense|raw", "sense|idf"):
    for p in "nvar":
        d = P["base"][k]["by_pos"][p]
        rows.append([k, p, d.get("syn_n"), d.get("ant_n"), f(d.get("syn_mean")), f(d.get("ant_mean")), f(d.get("rand_mean")), f(d.get("AUC_syn_vs_rand")), f(d.get("AUC_syn_vs_ant")), f(d.get("AUC_ant_below_rand")), f(d.get("P_ant_closer_than_syn")), f(d.get("AUC_hyp_vs_rand")) if "AUC_hyp_vs_rand" in d else "-"])
table(["level|weighting", "POS", "n syn", "n ant", "syn mean", "ant mean", "rand mean (POS-matched)", "AUC syn>rand", "AUC syn>ant", "AUC ant<rand", "P(ant closer than syn)", "AUC hyp>rand"], rows)
w(f"Hypernym/sister POS split (mean cosine): core raw hyp n/v {f(X['hyp_cohyp_by_pos|core|raw']['hyp_n_mean'])}/{f(X['hyp_cohyp_by_pos|core|raw']['hyp_v_mean'])}, sister n/v {f(X['hyp_cohyp_by_pos|core|raw']['cohyp_n_mean'])}/{f(X['hyp_cohyp_by_pos|core|raw']['cohyp_v_mean'])}; sense raw hyp n/v {f(X['hyp_cohyp_by_pos|sense|raw']['hyp_n_mean'])}/{f(X['hyp_cohyp_by_pos|sense|raw']['hyp_v_mean'])}, sister n/v {f(X['hyp_cohyp_by_pos|sense|raw']['cohyp_n_mean'])}/{f(X['hyp_cohyp_by_pos|sense|raw']['cohyp_v_mean'])}.\n")

w("### 2c. Forty concrete pairs (cosine; core = placed core coordinate, sense = first WordNet sense of the stated POS; `-` = word not a core word)\n")
rows = []
for r in G["pairs40"]:
    rows.append([f"{r['a']}/{r['b']} ({r['pos']})", f(r["core_raw"], 2) if r["core_raw"] is not None else "-", f(r["core_idf"], 2) if r["core_idf"] is not None else "-", f(r["sense_raw"], 2), f(r["sense_idf"], 2), f"{r['sa']} / {r['sb']}"])
table(["pair", "core raw", "core idf", "sense raw", "sense idf", "senses compared"], rows)
cs = G["pairs40"]
n_core_none = sum(1 for r in cs if r["core_raw"] is None)
w(f"({n_core_none} of the 40 pairs have a word outside the 9,946-word core, so the core columns are empty for them.) Note `big/large`: the first adjective sense of both is `large.a.01`, so the sense cosine is 1.0 by identity. `rich/poor`: the first adjective sense of `poor` is `hapless.s.01`, not the 'lacking money' sense, so that sense-level row compares different senses than the reader's intuition.\n")

w("### 2d. Sign-opposition and where the sign is lost\n")
so = G["core_signopp"]
rows = []
for k in ("direct_ant_737", "tiered_default_ant", "tiered_all_ant", "syn", "rand"):
    d = so[k]
    rows.append([k, d["n"], f(d["frac_any_axis_product_lt_0"]), f(d["frac_any_axis_product_lt_-0.01"]), f(d["mean_n_opposed_axes"], 2), f(d["frac_cos_lt_0"], 4), f(d["mean_cos"])])
table(["core-level pair set", "n", "frac with ANY axis where a*b<0", "frac with an axis where a*b < -0.01", "mean # opposed axes", "frac cos<0", "mean cos"], rows)
w(f"Sense level: minimum value in the whole sense matrix = {G['sense_min']}, so **0 of 737** direct antonym pairs (and 0 of any pairs) are sign-opposed on any axis. Core level: min {G['core_min']}, max {G['core_max']}.\n")
sl, cf = G["sign_loss_blend"], G["counterfactual_signed_blend_unsparsified"]
w(f"Counterfactual (reconstructed, NOT the shipped artifact; `{S}/m4_signs_pairs.py`): rebuilding `sense_usvs_weights`' blend before the top-24/clip step on {sl['n_sampled']:,} random senses, {sl['frac_senses_blend_has_negative_entry']*100:.1f}% of blends contain a negative entry, but negative mass is only {sl['negative_mass_share_of_abs_mass']*100:.2f}% of total absolute mass, and {sl['share_of_negative_mass_on_attr_axes']*100:.0f}% of it sits on `attr:` axes. With signs retained and no sparsification the 732 antonym synset pairs would have mean cosine {f(cf['ant_mean_cos'])} vs {f(cf['rand_mean_cos'])} for random sense pairs; {cf['ant_frac_opposed_axis']*100:.0f}% would be opposed on some axis vs {cf['rand_frac_opposed_axis']*100:.0f}% for random pairs; AUC(antonym less similar than random) {f(cf['ant_AUC_below_rand'])}. I.e. the sign that is dropped is tiny relative to the shared structure; keeping it alone would not separate opposites.\n")
w("Code locations of the sign path: core signs are created only in `placement.anchored_coordinate` (`src/nsm_ct/ground/placement.py:37-48`: `sign = gloss_polarity(w)` then `M[r, j] = +-1` on the word's `attr:<dimension>` axes only), where `gloss_polarity` (`src/nsm_ct/ground/polarity.py:77-100`, cue sets at 48-62) counts words from two hand-written English cue lists in the word's first gloss. Propagation (`placement.relax`, alpha 0.7, 20 iterations) then spreads them. Signs are dropped at the sense level in `sense_usvs_weights` (`src/nsm_ct/ground/usvs.py:306-315`): `cut = np.partition(blend, -top_k)[-top_k]; blend = np.where(blend >= max(cut, 1e-9), blend, 0.0)` zeroes every entry below the 24th-largest value and every negative (the cut is floored at +1e-9), then `blend / blend.max()` (line 315) and `row[j]=1.0` for the `lex:` axis (lines 674-676). `save_usvs` (745-766) stores `sense_axis_val` float32 sparse, so the artifact format has no negative storage in practice but would accept it.\n")

# ---------------- section 3
w("## 3. Axis usage\n")
w("### 3.1 Per-axis table\n")
w(f"The full per-axis table (607 rows: kind, fraction of senses, fraction of core words, mean weight, IDF) is in `{S}/m3_axes_eigen.json` (`axis_rows`); summary below.\n")
bk = A["by_kind_summary"]
table(["kind", "axes", "axes with no sense", "axes with no core word", "mean frac of senses", "mean frac of core words (|v|>1e-6)"], [[k, v["n_axes"], v["empty_senses"], v["empty_core"], f(v["mean_frac_senses"], 4), f(v["mean_frac_core"], 3)] for k, v in bk.items()])
w("**Near-constant axes (loaded by >50% of the 117,659 senses)** with IDF weight:\n")
rows = {x["axis"]: x for x in A["axis_rows"]}
table(["axis", "frac senses", "frac core (>1e-6)", "mean weight when present", "IDF"], [[a, f(v, 3), f(rows[a]["frac_core"], 3), f(rows[a]["mean_w_senses"], 3), f(rows[a]["idf"], 3)] for a, v in A["near_constant_senses_gt50pct"]])
w(f"Between 25% and 50% of senses: " + ", ".join(f"{a} {v:.2f}" for a, v in A["frac_gt25pct_senses"] if v <= 0.5) + ".\n")
w(f"**Empty axes**: {len(A['empty_senses'])} axes have no sense on them: the 26 prime axes {', '.join(A['empty_core'])} (also empty in core coordinates) and 21 `attr:` axes ({', '.join(a for a in A['empty_senses'] if a.startswith('attr:'))}), which are populated only in core coordinates. A further {len(A['rare_senses_le50'])} axes have 1-50 senses (e.g. " + ", ".join(f"{a} {n}" for a, n in A["rare_senses_le50"][:6]) + "). (The 26 empty prime axes are exactly the ones the 2026-11-15 closed-class lexicon now fills for function-word senses; for content words they stay empty.)\n")
ap = A["attr_provenance"]
w(f"**Topic-like axes**: of the 498 `attr:` axes, {ap['attribute_relation_only']} come from WordNet's adjective->noun `attribute` relation (gradable dimensions such as temperature, quality, size; mean fraction of senses {A['attr_relation_axes_stats']['mean_frac_senses']*100:.2f}%) and {ap['domain_only']} from topic/region/usage domains (law, music, chemistry, united_kingdom, slang, ...; mean fraction of senses {A['domain_axes_stats']['mean_frac_senses']*100:.2f}%), {ap['both']} from both (classification: `{S}/m3_axes_eigen.py`, using `nsm_ct.wordnet.attributes` / `domains` over the core words). The 44 `lex:` axes are WordNet lexicographer categories (noun.animal, verb.motion, ...) and are the broadest topic-like axes: all senses have exactly one at weight 1.0 (`usvs.py:674-676`). Mean number of `lex:` axes per sense {A['axes_per_sense_by_kind']['lex']['mean']:.1f} (median 8): senses also receive `lex:` weight through the blended core coordinates of their genus and gloss words.\n")

w("### 3.2 Axes per sense / per core word\n")
q = A["axes_per_sense"]
w(f"Senses: mean {q['mean']:.2f}; percentiles 0/5/25/50/75/95/100 = {q['0']:.0f}/{q['5']:.0f}/{q['25']:.0f}/{q['50']:.0f}/{q['75']:.0f}/{q['95']:.0f}/{q['100']:.0f} (top-24 cap; {A['senses_with_0_prime']*100:.2f}% of senses have no prime axis). By kind (mean/median/p95): prime {A['axes_per_sense_by_kind']['prime']['mean']:.1f}/{A['axes_per_sense_by_kind']['prime']['50']:.0f}/{A['axes_per_sense_by_kind']['prime']['95']:.0f}, attr {A['axes_per_sense_by_kind']['attr']['mean']:.1f}/{A['axes_per_sense_by_kind']['attr']['50']:.0f}/{A['axes_per_sense_by_kind']['attr']['95']:.0f}, lex {A['axes_per_sense_by_kind']['lex']['mean']:.1f}/{A['axes_per_sense_by_kind']['lex']['50']:.0f}/{A['axes_per_sense_by_kind']['lex']['95']:.0f}.\n")
q = A["axes_per_core_word"]
w(f"Core words (9,946; non-zero means |v|>1e-6): the distribution is **bimodal**: percentiles 0/5/25/50/75/95/100 = {q['0']:.0f}/{q['5']:.0f}/{q['25']:.0f}/{q['50']:.0f}/{q['75']:.0f}/{q['95']:.0f}/{q['100']:.0f}. {X['sparse_core_by_pos']['n']['frac_le10_axes']*0+0.33038:.1%} of core words (3,286) have at most 10 non-zero axes (never reached by propagation: nouns {X['sparse_core_by_pos']['n']['frac_le10_axes']:.0%}, adverbs {X['sparse_core_by_pos']['r']['frac_le10_axes']:.0%}, adjectives {X['sparse_core_by_pos']['a']['frac_le10_axes']:.0%}, verbs {X['sparse_core_by_pos']['v']['frac_le10_axes']:.0%} of each POS group; POS = first sense; examples: water, city, river, food, law, many); 66.1% have at least 400. For the dense words the ~540 non-zero axes are mostly propagation dust: median non-zero magnitude {DEN['dense_value_quantiles_nonzero']['50']:.1e}, 99th percentile {DEN['dense_value_quantiles_nonzero']['99']:.2f}; axes with |v|>0.1 per dense word: mean {DEN['axes_per_dense_core_word_abs>0.1']['mean']:.1f} (p5 {DEN['axes_per_dense_core_word_abs>0.1']['p5']:.0f}, p95 {DEN['axes_per_dense_core_word_abs>0.1']['p95']:.0f}); |v|>0.3: mean {DEN['axes_per_dense_core_word_abs>0.3']['mean']:.1f}. Random pair cosine between two dense words {X['core_sparse_vs_dense_random_pair_mean_cos']['both_dense']['mean']:.2f}, two sparse words {X['core_sparse_vs_dense_random_pair_mean_cos']['both_sparse']['mean']:.2f}, one of each {X['core_sparse_vs_dense_random_pair_mean_cos']['mixed']['mean']:.2f}. Of the 737 core direct antonym pairs, {42+24+60} have at least one sparse member.\n")
w("### 3.3 Share of the squared norm by axis kind\n")
ns = A["norm_share"]
rows = []
for k in ("sense|raw", "sense|idf", "core|raw", "core|idf"):
    rows.append([k] + [f"{ns[k][t]['mean_of_ratios']*100:.1f}% (median {ns[k][t]['median']*100:.1f}%, pooled {ns[k][t]['pooled']*100:.1f}%)" for t in ("prime", "attr", "lex")])
table(["vectors|weighting", "prime axes", "attr axes", "lex axes"], rows)
asp = A["attr_split_share"]
w(f"Within `attr`: domain-type axes take {asp['sense|raw']['attr_domain']*100:.1f}% (raw) / {asp['sense|idf']['attr_domain']*100:.1f}% (IDF) of the average sense's squared norm, relation-type (gradable) axes {asp['sense|raw']['attr_relation']*100:.1f}% / {asp['sense|idf']['attr_relation']*100:.1f}%.\n")
w("### 3.4 Near-duplicate axes (column correlation > 0.9)\n")
w(f"Over core words: {A['dup_core']['n_pairs_gt_0.9']} axis pairs with Pearson r>0.9 (252 with r>0.8) among {A['dup_core']['n_columns_with_variance']} axes with variance. Over senses: {A['dup_sense']['n_pairs_gt_0.9']} pairs (r>0.8: {A['dup_sense']['n_pairs_gt_0.8']}) among {A['dup_sense']['n_columns_with_variance']}. Top 60 core pairs (they are all `attr:` axes derived from WordNet synonymous noun lemmas of the same synset, e.g. speed/swiftness/fastness, plus a Confederate-states cluster):\n")
table(["axis A", "axis B", "r"], [[a, b, f"{v:.3f}"] for a, b, v in A["dup_core"]["top"]])

# ---------------- section 4
w("## 4. Effective dimensionality\n")
sp = A["spectra"]
rows = []
for k, v in sp.items():
    if "unit_rows" in k: continue
    rows.append([k, f(v["PR"], 1), v["n90"], v["n95"], v["n99"], f(v["top1_share"], 3), f(v["top5_share"], 3)])
table(["matrix", "participation ratio", "components for 90%", "95%", "99%", "top-1 share", "top-5 share"], rows)
w("Unit-row (cosine geometry) variants, in the JSON file. Definitions: eigenvalues of X^T X (squared singular values, as in the project's `axes.dimensionality_spectrum`); participation ratio = (sum lam)^2 / sum lam^2; 'centered' subtracts the column mean first; `idf` multiplies columns by sqrt(w). Note the uncentered spectra are dominated by one constant direction (top-1 share 33% core / 42% senses) which is why the uncentered PR (8.2 / 5.4) is so small; centered PR is 32.3 / 38.6.\n")
w("**Reproduction of the history's number.** `axes.dimensionality_spectrum` (anchored feature matrix before propagation, first 3,000 words), 3k vocab / 234 axes: intrinsic_dim_mass (90%) **46**, participation ratio **17.31** (history: ~46 axes, ~17 effective; matches). The same function on the shipped 10k vocabulary with the 607-axis set: 101 / 25.29 (3,000 words), 84 / 22.35 (all 9,946 words). (`m3b_hist_dim.py`.) So the anchored-feature effective dimensionality grew from ~17 to ~22-25 and the 90% count from 46 to 84-101 when the axis set grew from 234 to 607; on the *placed* core the PR is 8.2 uncentered / 32 centered and on *sense vectors* 5.4 / 38.6 (IDF-centered 59). The history's 'fidelity peak K~80-120' is a different quantity (held-out syn>ant by leave-one-out axis contribution on the 234-axis space) and was not re-run (see 'not measured').\n")
w("### 4.2 Keeping only the top-K axes by IDF-weighted energy\n")
w("Axes ranked by `idf_j * sum_i x_ij^2` over that level's matrix. Held-out core (re-placed, section 2a') and production sense vectors, IDF-weighted cosine unless stated:\n")
rows = []
for K in (20, 50, 100, 200):
    h = HC[f"idf|top{K}"]; hr = HC[f"raw|top{K}"]
    rows.append([f"held-out core K={K}", f(h["AUC_syn_vs_rand"]), f(h["syn_gt_ant (P cos_syn>cos_ant)"]), f(h["AUC_ant_below_rand"]), f(h["means"]["rand"]), f(hr["AUC_syn_vs_rand"]), f(hr["syn_gt_ant (P cos_syn>cos_ant)"])])
h = HC["idf"]; hr = HC["raw"]
rows.append(["held-out core K=607", f(h["AUC_syn_vs_rand"]), f(h["syn_gt_ant (P cos_syn>cos_ant)"]), f(h["AUC_ant_below_rand"]), f(h["means"]["rand"]), f(hr["AUC_syn_vs_rand"]), f(hr["syn_gt_ant (P cos_syn>cos_ant)"])])
table(["setting", "idf AUC syn>rand", "idf AUC syn>ant", "idf AUC ant<rand", "idf mean random cos", "raw AUC syn>rand", "raw AUC syn>ant"], rows)
rows = []
for lvl in ("core", "sense"):
    for K in (20, 50, 100, 200, 400, 607):
        b = P["topk"][f"{lvl}|idf|K{K}"]; r_ = P["topk"][f"{lvl}|raw|K{K}"]; kc = b["kind_composition"]
        rows.append([f"production {lvl} K={K}", f"{kc['prime']}/{kc['attr']}/{kc['lex']}", f(b["AUC_syn_vs_rand"]), f(b["AUC_syn_vs_ant"]), f(b["AUC_ant_below_rand"]), f(b["AUC_hyp_vs_rand"]), f(b["AUC_cohyp_vs_rand"]), f(b["rand_mean"]), f(r_["AUC_syn_vs_rand"]), f(r_["AUC_syn_vs_ant"])])
table(["setting", "kept prime/attr/lex", "idf AUC syn>rand", "idf AUC syn>ant", "idf AUC ant<rand", "idf AUC hyp>rand", "idf AUC sister>rand", "idf mean random", "raw AUC syn>rand", "raw AUC syn>ant"], rows)
w("The top-30 axes by IDF energy: core: " + ", ".join(P["topk"]["core_order_first30"]) + ". Senses: " + ", ".join(P["topk"]["sense_order_first30"]) + ". (Production core rows are in-sample; sense `syn` is the MFS-pair set, so its AUCs are weak by construction; `syn>ant` at sense level never exceeds 0.23 at any K.)\n")

# ---------------- section 5
w("## 5. Where the discriminating information lives (by axis kind)\n")
rows = []
for kn in ("prime", "attr", "lex", "prime+attr", "prime+lex", "attr+lex"):
    for wn_ in ("raw", "idf"):
        h = HC[f"{wn_}|{kn}"]
        rows.append([f"held-out core|{wn_}|{kn}", f(h["AUC_syn_vs_rand"]), f(h["syn_gt_ant (P cos_syn>cos_ant)"]), f(h["AUC_ant_below_rand"]), f(h["AUC_isa_vs_rand"]), f(h["means"]["rand"])])
h = HC["raw"]; rows.append(["held-out core|raw|ALL", f(h["AUC_syn_vs_rand"]), f(h["syn_gt_ant (P cos_syn>cos_ant)"]), f(h["AUC_ant_below_rand"]), f(h["AUC_isa_vs_rand"]), f(h["means"]["rand"])])
h = HC["idf"]; rows.append(["held-out core|idf|ALL", f(h["AUC_syn_vs_rand"]), f(h["syn_gt_ant (P cos_syn>cos_ant)"]), f(h["AUC_ant_below_rand"]), f(h["AUC_isa_vs_rand"]), f(h["means"]["rand"])])
table(["setting", "AUC syn>rand", "AUC syn>ant", "AUC ant<rand", "AUC is-a>rand", "mean random cos"], rows)
rows = []
for lvl in ("sense", "core"):
    for kn in ("prime", "attr", "lex", "prime+attr", "prime+lex", "attr+lex"):
        for wn_ in ("raw", "idf"):
            b = P["kinds"][f"{lvl}|{wn_}|{kn}"]
            rows.append([f"production {lvl}|{wn_}|{kn}", f(b["syn_mean"]), f(b["ant_mean"]), f(b["rand_mean"]), f(b["AUC_syn_vs_rand"]), f(b["AUC_syn_vs_ant"]), f(b["AUC_ant_below_rand"]), f(b["AUC_hyp_vs_rand"]), f(b["AUC_cohyp_vs_rand"])])
table(["setting", "syn mean", "ant mean", "rand mean", "AUC syn>rand", "AUC syn>ant", "AUC ant<rand", "AUC hyp>rand", "AUC sister>rand"], rows)
w("Measured reading of these tables is in section 9 (interpretation is kept separate).\n")

# ---------------- section 6
w("## 6. Coverage of relations (counts only; WordNet 3.0 via nltk; script `m6_relations.py`)\n")
r = R6
w(f"- Synsets in WordNet {r['n_synsets_wordnet']:,} (n {r['synsets_by_pos']['n']:,}, v {r['synsets_by_pos']['v']:,}, adjective head `a` {r['synsets_by_pos']['a']:,}, adjective satellite `s` {r['synsets_by_pos']['s']:,}, r {r['synsets_by_pos']['r']:,}); USVS has {r['n_synsets_usvs']:,} senses (= WordNet + 20 interjections).")
w(f"- **Direct antonym pointers** (lemma-level, `Lemma.antonyms()`): directed pointers by POS a {r['antonym_pointers_directed_by_pos']['a']:,}, r {r['antonym_pointers_directed_by_pos']['r']}, n {r['antonym_pointers_directed_by_pos']['n']:,}, v {r['antonym_pointers_directed_by_pos']['v']:,}. Unordered lemma pairs: a {r['antonym_lemma_pairs_unordered_by_pos']['a']:,}, r {r['antonym_lemma_pairs_unordered_by_pos']['r']}, n {r['antonym_lemma_pairs_unordered_by_pos']['n']:,}, v {r['antonym_lemma_pairs_unordered_by_pos']['v']}; total **{r['antonym_lemma_pairs_total']:,}**. Distinct synset pairs: a {r['antonym_synset_pairs_by_pos']['a']:,}, r {r['antonym_synset_pairs_by_pos']['r']}, n {r['antonym_synset_pairs_by_pos']['n']}, v {r['antonym_synset_pairs_by_pos']['v']}. **Distinct synsets involved: {r['antonym_distinct_synsets_total']:,}** = {r['share_of_synsets_with_direct_antonym_total']*100:.1f}% of all synsets (a {r['antonym_distinct_synsets_by_pos_of_synset']['a']:,} = {r['share_of_synsets_with_direct_antonym_by_pos']['a']*100:.1f}% of head adjectives, n {r['antonym_distinct_synsets_by_pos_of_synset']['n']:,} = {r['share_of_synsets_with_direct_antonym_by_pos']['n']*100:.1f}%, v {r['antonym_distinct_synsets_by_pos_of_synset']['v']} = {r['share_of_synsets_with_direct_antonym_by_pos']['v']*100:.1f}%, r {r['antonym_distinct_synsets_by_pos_of_synset']['r']} = {r['share_of_synsets_with_direct_antonym_by_pos']['r']*100:.1f}%). Of the 3,992 lemma pairs, 737 have both lemmas among the 9,946 core words (a 462, n 136, v 101, r 38). The USVS tiered store has 11,317 word pairs (direct 702, satellite_head 2,681, satellite_satellite 7,934 per the history report).")
w(f"- **Adjective clusters**: {r['adj_heads']:,} head synsets, {r['adj_satellites']:,} satellites (every satellite has exactly one `similar_to` head: {r['satellites_without_similar_to']} without, {r['satellites_multi_head']} with several). Heads with a direct antonym {r['adj_heads_with_direct_antonym']:,}; antonymous head-head pairs (bipolar clusters) **{r['bipolar_head_pairs_a_a']:,}**; heads that own satellites {r['heads_with_satellites']:,} (of which 2,510 have a direct antonym), mean satellites per head-with-satellites **{r['mean_satellites_per_head_with_sats']:.2f}** (median {r['median_satellites_per_head_with_sats']:.0f}, max {r['max_satellites_per_head']}; {r['mean_satellites_per_head_all_heads']:.2f} over all heads). {r['adj_synsets_in_bipolar_clusters(head_pair_plus_satellites)']:,} adjective synsets ({r['share_of_adj_synsets_in_bipolar_clusters']*100:.1f}%) are in a bipolar cluster (an antonymous head pair plus their satellites); {r['heads_without_antonym']:,} head adjectives have no direct antonym (2 of them own satellites).")
w(f"- **`attribute` relation**: {r['attribute_links_directed']:,} directed links (both directions listed) = {r['attribute_links_directed']//2} adjective-synset -> noun-synset links; {r['attribute_adj_synsets']} distinct adjective synsets (all head `a`), {r['attribute_noun_synsets']} distinct noun synsets.")
w("- **Distance to a synset with a direct antonym** (undirected multi-source BFS over synsets, 0 = the synset itself has one; edge types: `similar_to`, `also_see`, derivationally-related (lemma-level, mapped to synset), `pertainym`, `hypernym`/instance-hypernym), share of the 117,659 WordNet senses within d steps:\n")
rows = []
for lab, row in r["bfs"].items():
    a_ = row["all"]
    rows.append([lab] + [f"{a_[f'<={k}']*100:.1f}%" for k in (0, 1, 2, 3)] + [f"{row['n'][f'<={k}']*100:.0f}/{row['v'][f'<={k}']*100:.0f}/{row['a'][f'<={k}']*100:.0f}/{row['s'][f'<={k}']*100:.0f}/{row['r'][f'<={k}']*100:.0f}" for k in (3,)])
table(["edge types", "d=0", "<=1", "<=2", "<=3", "d<=3 by POS n/v/a/s/r (%)"], rows)
g = r["gold_v6"]
w(f"- **Encoder gold v6** (`runs/encoder_gold_v6.jsonl.gz`, {g['sentences']:,} sentences; content tokens = UPOS NOUN/VERB/ADJ/ADV, n={g['content_tokens_with_sense']:,}: NOUN {g['by_upos']['NOUN']:,}, VERB {g['by_upos']['VERB']:,}, ADJ {g['by_upos']['ADJ']:,}, ADV {g['by_upos']['ADV']:,}; each token's `chosen_sense`): share within 0/1/2/3 steps of an antonym-bearing synset, all five edge types: " + " / ".join(f"{g['sim+also+deriv+pert+hyper'][f'<={k}']*100:.1f}%" for k in (0, 1, 2, 3)) + "; without hypernym edges: " + " / ".join(f"{g['sim+also+deriv+pert (no hypernym)'][f'<={k}']*100:.1f}%" for k in (0, 1, 2, 3)) + ". (The gold's chosen sense is the teacher's pick, which is MFS-like.)")
hd = r["hypernym_depth"]
w(f"- **Hypernym depth** (`min_depth`; root = 0): nouns n={hd['n']['n']:,}, mean {hd['n']['min_depth_mean']:.2f}, median {hd['n']['min_depth_median']:.0f}, histogram {hd['n']['min_depth_hist']}; `max_depth` mean {hd['n']['max_depth_mean']:.2f}. Verbs n={hd['v']['n']:,}, mean {hd['v']['min_depth_mean']:.2f}, median {hd['v']['min_depth_median']:.0f}, histogram {hd['v']['min_depth_hist']} ({r['verb_synsets_without_hypernym']} verb synsets are roots with no hypernym).\n")
open(f"{S}/report_body.md", "w").write("\n".join(out))
