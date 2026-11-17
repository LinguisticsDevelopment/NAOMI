# Universal Dependencies as a second teacher: feasibility (2026-11-16, planning only)

Nothing trained; no existing file changed; git untouched. Scripts: `dev/reports/treebank_feasibility_scripts/` (frozen table
`conversion_table_v2.json`, converter `convert.py`, alignment `align.py`, measurements, `out/` = raw parser/encoder results). Data:
`data/ud/` (130 MB, not committed). All figures below come from those scripts. Single rater (me) for every hand judgement.

## 1. Data and licences (UD v2.18 files, raw.githubusercontent.com, all fetched)
| treebank | licence (README / LICENSE.txt) | sent / tokens (incl. punct) train+dev+test | genre | >20 words / >35 words |
|---|---|---|---|---|
| EWT | CC BY-SA 4.0 (annotations; texts "(c) various", LDC web treebank, some public domain) | 12,544+2,001+2,077 = 16,622 / 254,820 | weblog, newsgroup, email, review, Yahoo answers | 24% / 5% (train); median 12 |
| GUM | CC BY-NC-SA 4.0 | 11,314+1,575+1,464 = 14,353 / 256,739 | ~15 genres (fiction, news, bio, academic, speech, conversation, textbook, letter...) | 29% / 7% |
| LinES | CC BY-NC-SA 4.0 ("citation corpus") | 3,457+1,118+1,121 = 5,696 / 106,307 | translated fiction, manuals, parliament | 28% / 6% |
| ParTUT | CC BY-NC-SA 4.0 (inherited from ParTUT) | 1,781+156+153 = 2,090 / 49,478 | legal, news, wiki, talks | 45% / 13% (train) |
| es_GSD | CC BY-SA 4.0 (annotations); Google: text "ownership asserted by no one, some may be copyrighted" | 14,186+1,400+427 = 16,013 / 431,729 | blog, news, reviews, wiki | 52% / 17% |
| es_AnCora | CC BY 4.0 (changed from GNU licence in 2021) | 14,287+1,654+1,721 = 17,662 / 560,138 | news | 64% / 31% |
(Token counts = my own counts of syntactic words; GUM has no Reddit text in this version. LinES has no README in the repo.)
Reading of the terms: derived data, model training and redistribution of a converted copy are permitted by all six licences with
attribution; BY-SA (EWT, GSD) adds share-alike: a converted copy shipped in the repo must itself carry CC BY-SA 4.0; BY-NC-SA
(GUM, LinES, ParTUT) adds non-commercial: usable only while the repo/model stays non-commercial. AnCora (BY) is the freest. None of the
licences clears the copyright of the underlying texts (EWT and GSD READMEs say so explicitly); a converted copy with the sentence
text inside it inherits that gap. Trees without the text (heads+labels over token ids) would avoid shipping the text.

**Tokenisation vs the project** (`nsm_ct.corpus._tokenize_words`, which `normalize`/the tagger use; script `tokenisation.py`):
identical lower-cased token sequences on 75.3% of EWT-dev sentences, 80.0% of GUM-dev, 57.8% of es_GSD-dev. Same as UD:
`do n't`, `'s 're 've 'll 'd 'm` split off (possessive `'s` too, UD: `case`). Differs: project splits every `. : ,` (`alt.animals.cat`, URLs,
`e.g.`, `1,000`), UD keeps them whole; UD splits `its`->`it 's`, `ill`->`i ll`, `a+`->`a +`, `ask(heart)`, `..`; project does not split
`12's`/`area's` (UD keeps them whole, project splits), no hyphen splitting in either; Spanish `del/al` are one UD multi-word token (two
syntactic words `de el`) and one project token; clitics (`dame`) likewise. 20 contraction/punctuation-heavy EWT examples: 10 identical, 10 differ
(causes above). Handled by character-stream alignment (`align.py`): UD words merged into one project token where the project is coarser
(head = the member whose head lies outside the group); project tokens carved out of one UD word are masked. On the 1,500-sentence sample:
0 character-stream mismatches, 13% of sentences have at least one masked token, 1.3% of project nodes masked.

## 2. Scheme comparison (project side measured on gold v7, parser-committed links; probes parsed with `Reader.read_parser`)
Edge labels in gold v7 (nodes): DESCRIPTION 86k, MODIFICATION 75k, SUBJECT 57k, PREPOSITION 49k, OBJECT 42k, SPECIFICATION 32k,
COORDINATION 30k, INDIRECT_OBJECT 10.6k, COMPLEMENT 6.5k, SUBORDINATION 4.1k, APPOSITION 0.3k (+ ROOT, UNATTACHED). Commonest
(head tag <- dep tag, side of the dependent; % of that label's committed edges, `gold_conventions.json` has the top 8):
- DESCRIPTION: NOUN<-DET before 61%, NOUN<-ADJ before 20%, VERB<-DET 3%. SUBJECT: VERB<-PRON before 34%, VERB<-NOUN 18%, AUX<-PRON 17%, AUX<-NOUN 7%.
- OBJECT: VERB<-NOUN after 32%, VERB<-PRON 16%, AUX<-NOUN 11%, VERB<-VERB 5.5%. INDIRECT_OBJECT: VERB<-PRON 22%, VERB<-NOUN 22%, AUX<-NOUN 10% (it is the label for a second post-verbal nominal, often the real object).
- PREPOSITION: ADP<-NOUN after 62%, ADP<-PRON 16%, ADP<-VERB 5% (gerund). MODIFICATION: VERB<-ADP after 25%, VERB<-AUX before 12%, NOUN<-ADP after 10%, ADP<-VERB 7%.
- SPECIFICATION: VERB<-ADV after 20% / before 12%, DET<-ADV before 12%, ADJ<-ADV before 11%, AUX<-ADV 11%. COMPLEMENT: AUX<-ADJ after 38%, VERB<-ADJ 17%, VERB<-DET 16%.
- COORDINATION: CCONJ<-VERB after 23% and before 21%, CCONJ<-NOUN both sides 15%, VERB<-CCONJ before 8%. SUBORDINATION: VERB<-PRON after 24%, VERB<-SCONJ 11%, VERB<-ADV 15%. APPOSITION: NOUN<-NOUN after 39%.

| construction | project convention (gold v7 / probes) | UD convention | class |
|---|---|---|---|
| subject | SUBJECT of the verb, or of the AUX/copula when that is head (26% of SUBJECT edges have an AUX head) | `nsubj` of the content predicate | (i) label; head differs when AUX is head: (iii) |
| auxiliary | main verb heads, aux = MODIFICATION (12.1k edges), but AUX heads the verb in 5.7k pairs (~30% of aux-verb pairs); aligned with UD: verb head 57%, aux head 18%, other 25% (1,375 aux relations) | `aux` dependent of the verb | (i) for the first analysis; the project's own choice between the two is not predictable from tags/features: (iii) |
| copula + predicate | be-form heads; predicate COMPLEMENT (ADJ), OBJECT (noun), MODIFICATION (PP), SPECIFICATION (adv); `she is happy` -> is(SUBJECT:she COMPLEMENT:happy) | predicate heads, `cop` child, subject on predicate | (ii) swap + move subject/aux/clausal dependents; label by predicate tag (i) |
| preposition | ADP heads its noun (PREPOSITION); the ADP hangs from verb/noun by MODIFICATION: `sat(MODIFICATION:on(PREPOSITION:mat))` | noun heads, `case` child, noun is `obl`/`nmod` | (ii) insert the ADP between noun and its head (chain for `out of`); PP attachment itself is a content choice |
| coordination | conjunction on top and in the conjuncts' place: `went(SUBJECT:and(COORDINATION:tom COORDINATION:mary))`, clauses `and(COORDINATION:ran(..) COORDINATION:walked(..))` | first conjunct heads, `conj` chain, `cc` on the next conjunct | (ii) lift; comma lists without conjunction, several `and`s, leading conjunction: (iii) |
| subordinator / complementiser | `because`/`that` head their clause: `stayed(SUBORDINATION:because(MODIFICATION:rained(..)))`, `think(SUBORDINATION:that(MODIFICATION:left))` | `mark` is a child of the clause verb; clause is `advcl`/`ccomp` | (ii) swap; label SUBORDINATION vs OBJECT for `ccomp` is mixed in the parser: (iii) small |
| relative clause | noun -SUBORDINATION-> clause verb; relative pronoun MODIFICATION of that verb (`man(SUBORDINATION:lives(MODIFICATION:who ..))`); with `that`: SUBORDINATION:that(..) | `acl:relcl` noun->verb; pronoun is `nsubj`/`obj`/`obl` of the verb | (i) for the link; pronoun relabel needs FEATS PronType=Rel: not implemented (table gap) |
| possessive | `'s` heads the possessor (PREPOSITION) and hangs MODIFICATION from the possessed noun; `his dog`: DESCRIPTION | `nmod:poss`, `'s` = `case` of the possessor | (ii) same swap as ADP; pronoun (i) |
| negation | `not`/`n't` MODIFICATION of the verb | `advmod` + Polarity=Neg | (i) |
| det / adjective / numeral / adverb | DESCRIPTION of the noun; adverb SPECIFICATION of verb/adj/adv | `det`,`amod`,`nummod`; `advmod` | (i); but in noun-noun compounds the parser hangs det/adj on the nearest noun, UD on the head noun: (iii) |
| infinitive marker | `to` heads its verb (`wants(OBJECT:to(MODIFICATION:go))`); parser labels it OBJECT or MODIFICATION (54 vs 34 on dev) | `mark` PART, verb is `xcomp`/`advcl` | (ii) swap; label (iii) |
| reported speech | quotation clause is OBJECT of the speech verb, speaker SUBJECT | `ccomp` (own verb), or `parataxis` | (i) `ccomp`; `parataxis` (iii) |
| names / numerals / compounds | PROPN/NUM are `name` nodes; no compound relation (parser: IO, PREPOSITION or SUBJECT for the first noun) | `flat`, `compound` (first or last token heads) | (iii): the grammar has no equivalent, table gives DESCRIPTION |
| apposition | APPOSITION, almost unused (244 edges); parser uses MODIFICATION | `appos` | (i) |

## 3. Draft conversion table and converter
`conversion_table_v2.json` (frozen; v0 draft, v1 and v2 = the two allowed iterations on the dev half): ~35 first-match rules
`{ud relation(+subtype base), dep_upos, head_upos, dep FEATS} -> project label`, plus four declared operations in a fixed order:
`drop_punct`, `copula_swap` (cop becomes head, takes predicate's place and relation, receives subject/aux/clausal dependents; predicate gets label from its UPOS),
`phrase_head_swap` x2 (`case` -> ADP heads its noun and takes its relation; `mark` -> subordinator heads its clause; chains allowed; `cc` and
conjuncts that carry their own marker move up), `coord_lift` (first `cc` replaces the first conjunct, all conjuncts become COORDINATION children;
no `cc`: later conjuncts hang under the first). No word lists: only UD relation names, UPOS and FEATS (e.g. `Polarity=Neg`, `VerbForm=Ger`).
`convert.py` emits the project JSON tree (`naomi.types.Node/Tree`, `source: "ud_converted"`, `to_project_tree`).
Lemma/POS: taken from UD columns (the table is keyed on UD UPOS/FEATS; the project tagger errs on web text and would corrupt the keys);
for encoder training the project tagger would be run on the same tokens afterwards (features need its subtypes), the alignment above supports that.
Result on all treebanks: 100% of 38,676 English and 33,674 Spanish sentences become one-root, fully attached trees, 0 UD relations without a rule.
Known gaps, found after the second iteration and NOT fixed (iteration cap): a leading `cc` is taken as the coordinator before the real one
(3.7% of 6,393 EWT coordination groups; seen in the hand judgement); a leftover `cc` is labelled MODIFICATION (parser: COORDINATION);
`cc` of a copular root should travel with the copula; relative-pronoun label.

## 4. Agreement with the project's parser (1,500 EWT sentences, 375 per length band <=10 / 11-20 / 21-35 / 36+, seed 0; dev/test halves 750/750)
Parser run (10 s cap, 4 workers, 22 min): 48 failed (3.2%), 1,452 usable: **293 fully connected (20%)**, 1,159 left in pieces (36+ words: 0 connected).
Agreement = same head token (root = head none); head+relation = same head and same label. Compared over the parser's committed links.
| table | half | connected: head / head+rel (sent / links) | pieces, committed links: head / head+rel (sent / links) |
|---|---|---|---|
| v0 draft | dev | .682 / .611 | .606 / .533 |
| v1 | dev | .682 / .621 | .606 / .546 |
| v2 | dev | .682 / .622 (156 / 1,474) | .606 / .546 (577 / 10,811) |
| v0 | test | .665 / .592 | .612 / .540 |
| **v2 frozen** | **test** | **.665 / .601 (137 / 1,298)** | **.612 / .552 (582 / 10,461)** |
Iterations changed labels only (head agreement is fixed by the operations plus UD). Frozen test, connected, by length: <=10 .699/.624, 11-20 .619/.556,
21-35 .683/.643. By parser relation (head / head+rel): DESCRIPTION .805/.795, PREPOSITION .790/.766, SUBJECT .743/.650, ROOT .730/.730, OBJECT .568/.410,
MODIFICATION .585/.537, SPECIFICATION .556/.489, COORDINATION .634/.585, INDIRECT_OBJECT .388/.061, SUBORDINATION .400/.400. By UD construction: det .81, nmod .78,
obl .78, nsubj .72, obj .74, case (the ADP's own attachment) .51, advmod .55, aux .57, cc .40, compound .22. Pooled dev+test, all 24,044 compared links: 9,224 head
mismatches (38%), 1,449 label-only mismatches (6%). **For sentences the parser leaves in pieces, the UD tree agrees with the parser's committed links
61% (head) / 55% (head+rel)**, only slightly below the connected sentences (66.5 / 60.1), so UD contradicts the teacher about as much there as elsewhere; but 39% head
disagreement on links the parser did commit is the noise floor any UD-based referee has to live with.

Top 15 disagreements, pooled (count of the 9,224 head + 1,449 label mismatches; C = connected-sentence count; two examples each,
parser head vs UD head; "UD" = converted tree):
| # | key (UD relation, dep tag; parser label) | n (C) | examples | class |
|---|---|---|---|---|
| 1 | head, `case` ADP, MODIFICATION | 1,109 (113) | `want to party hardy for my birthday`: for->want vs ->party; `got it in one`: in->it vs ->got | analysis difference (PP attachment) / parser error |
| 2 | head, `det` | 550 (42) | `the my tho waterfront`: the->tho vs ->waterfront; `a protein skimmer`: a->protein vs ->skimmer | analysis difference: grammar has no compound |
| 3 | head, `advmod` | 499 (58) | `will probably die`: probably->will vs ->die; `just off their beach`: just->are vs ->beach | analysis diff. (scope; aux as head) |
| 4 | head, `nsubj` PRON | 231 (36) | `tourists that i 've come across`: i->'ve vs ->come; `because it was missing`: it->was vs ->missing | analysis diff. (AUX head) |
| 5 | head, `compound` NOUN, PREPOSITION | 211 (20) | `for my birth certificate`: birth->for vs ->certificate; `for summer employment` | parser error / grammar gap |
| 6 | head, `obl` NOUN, OBJECT | 189 (30) | `a lot more concerned`: lot->be vs ->more; `send for my birth certificate`: certificate->send vs ->for | parser error + analysis diff. |
| 7 | head, `mark` PART (`to`) | 182 (20) | `is believed to be`: to->yet vs ->believed; `scheduled to have lunch`: to->see vs ->scheduled | parser error |
| 8 | head, `amod` | 163 (14) | `unitary executive theory`: unitary->theory vs ->executive; `authentic british meat pies` | analysis diff. (compound) |
| 9 | head, `cc` | 161 (8) | `and it is a war`: and->is vs ->war; `and he would always laugh`: and->would vs ->laugh | conversion-table gap (copular/leading cc) + analysis diff. |
| 10 | head, `obj` NOUN | 139 (22) | `it was missing one lousy nail`: nail->was vs ->missing; `we are still reviewing the attachment`: attachment->are vs ->reviewing | analysis diff. (AUX head) |
| 11 | head, `aux` | 135 (17) | `because it was missing`: was->because vs ->missing; `that lists`/`are reviewing`: are->that vs ->reviewing | analysis diff. (aux duality) |
| 12 | head, `compound` PROPN, PREPOSITION | 132 (7) | `hurricane katrina`: hurricane->of vs ->katrina; `sunday afternoon`: sunday->on vs ->afternoon | parser error / grammar gap |
| 13 | head, `compound` NOUN, INDIRECT_OBJECT | 127 (17) | `need a protein skimmer`: protein->need vs ->skimmer; `do the epm test` | parser error / grammar gap |
| 14 | head, `nsubj` NOUN | 123 (12) | `the mouse will probably die`: mouse->will vs ->die; `opinions expressed herein are`: opinions->expressed vs ->are | analysis diff. (AUX head) |
| 15 | label, `obj` NOUN, parser INDIRECT_OBJECT vs conv OBJECT | 114 (21) | `have a natural instinct`; `take time to discover` | parser error (INDIRECT_OBJECT over-used) |
Tokenisation mismatch accounts for none of the top 15 (masked tokens are excluded). Table gaps are the minority (item 9, relative-pronoun and
`cc` labels); items 1-8, 10-15 are what the project's grammar does differently (or wrongly) with those constructions, not what the table can repair.
A genuine caveat: because several of these are parser errors, "agreement with the parser" understates how good the converted UD trees are.

## 5. Does it supply the missing evidence?
Encoder `runs/naomi_encoder/main_best.pt` on the same UD sentences the parser leaves in pieces (frozen v2 table). Links the encoder ADDS = its head for tokens the parser
left UNATTACHED. Out-of-domain for the encoder (web text, trained on fiction): on committed links it still equals the parser 89-91% (test .906).
| set | added links (n) | head agrees with converted UD | head+relation agrees | "attach to previous token" baseline |
|---|---|---|---|---|
| test half | 3,031 | **26.4%** | **16.6%** | 19.0% |
| dev half | 2,717 | 26.7% | 15.2% | 19.2% |
By length (test): <=10 words .41 / .23 (n=139), 11-20 .30 / .18, 21-35 .26 / .17, 36+ .24 / .16 (n=1,624). By the encoder's label (head / h+r, test):
PREPOSITION .53/.43 (n=176), DESCRIPTION .46/.40 (296), SPECIFICATION .34/.15, COORDINATION .20/.16 (366), MODIFICATION .21/.12 (968), OBJECT .19/.10 (530),
SUBJECT .24/.10, SUBORDINATION .27/.20, COMPLEMENT .23/.07, INDIRECT_OBJECT .32/0. For comparison, on committed links the same UD trees agree with the parser 61% and with
the encoder 61%. Against the hand-check (32% right / 34% debatable / 34% wrong; Gutenberg-style text, other sentences): strict UD agreement on added links (26% head) is of the same size as
"right", consistent but not a test of the same links. Reading: UD gives a verdict on 100% of the added links (every word has a head), says about three quarters differ, and
cannot say how many of those are UD/conversion noise (39% of committed parser links differ too) versus encoder errors. Whether "UD agrees => right" has high precision
is not established here; it would need hand-judging a sample of agreeing and disagreeing added links.

**Reliability of converted between-clause links** (60 links, 10 per type, from EWT dev/test + GUM dev sentences of 21-45 words, converted with v2; `handjudge_sample.py`; judged in the project's format: right = head and
label as the project's conventions above would give; debatable = defensible alternative / lossy; wrong). Lenient on labels inside the clause (e.g. relative pronoun as SUBJECT).
| type | right | debatable | wrong |
|---|---|---|---|
| clause coordination | 6 | 3 | 1 |
| subordinate (`advcl` + SCONJ) | 7 | 3 | 0 |
| relative (`acl:relcl`) | 9 | 1 | 0 |
| complement (`ccomp` + `that/if`) | 9 | 1 | 0 |
| reported speech (`ccomp` no mark / `parataxis`, sentence has quote marks) | 5 | 4 | 1 |
| infinitival / purpose (`xcomp`/`advcl` + `to`) | 8 | 2 | 0 |
| **all** | **44 (73%)** | **14 (23%)** | **2 (3%)** |
Examples. Right: `what I can do in this field and what I have to more focus on` -> `know(OBJECT:and(COORDINATION:do(..) COORDINATION:have(..)))`. Debatable: `hesitant about turning over`
-> `hesitant(SUBORDINATION:about(MODIFICATION:turning(..)))` (project would make the gerund a PREPOSITION child; the `VerbForm=Ger` rule did not fire), and a six-clause `and ... and ...` chain (extra `and`s dangle as MODIFICATION).
Wrong: `And I wish to start ..., and it relates to ...` -> the leading `And` was chosen as coordinator (the known bug); `express ... [23]` footnote number linked as `parataxis` COORDINATION (not a clause).
Caveats: I wrote the table and judged its output (self-judgement, lenient); this measures fidelity of translation, not whether the project's grammar would ever produce these links; the 10 infinitival items
are all control/raising `to`, whose label (OBJECT vs MODIFICATION) the parser itself splits 54/34.

## 6. How much would it add
Converted with v2 (`amounts.py`; "between-clause link" = a non-root VERB attached to something, same definition applied to gold v7 fully connected sentences):
| source | sentences | tokens (words) | >20 words | >35 words | between-clause links |
|---|---|---|---|---|---|
| gold v7, fully connected | 8,214 | 81,532 | 586 (7%) | 16 | 9,312 |
| EWT | 16,537 | 225,053 | 3,610 (22%) | 727 | 20,326 |
| GUM | 14,353 | 222,235 | 4,167 (29%) | 956 | 19,555 |
| LinES | 5,696 | 93,511 | 1,570 (28%) | 352 | 9,103 |
| ParTUT | 2,090 | 43,750 | 901 (43%) | 240 | 3,350 |
| English total | 38,676 | 584,549 | 10,248 (26%) | 2,275 | 52,334 |
| es_GSD / AnCora | 16,013 / 17,661 | 384,254 / 494,513 | 8,396 / 11,322 | 2,757 / 5,448 | 26,205 / 36,457 |
English clause links by UD relation (EWT+GUM+LinES+ParTUT, verbs): advcl 10.2k, acl (incl. relative) 11.1k, conj 9.2k, xcomp 6.7k, ccomp 4.1k, parataxis 2.8k, csubj 1.0k. (The count includes participial `amod` verbs and 'root' verbs that
end up under a coordinator; order of magnitude only.) Long-sentence material: ~17x gold's >20-word sentences, ~140x its >35 (BY-SA/BY only: EWT 3,610 + 727).
Domain: the project corpus is children's stories and Gutenberg prose (gold v7 median 17 words, 25% of sentences contain quote marks). EWT is web text (median 11 words, 5% quote marks,
16% of its word types unseen in gold v7); GUM fiction is closer (median 10, 10% unseen, 1,070 sentences, 242 over 20 words), LinES 11% unseen. No stories-with-dialogue in EWT/ParTUT; informal spelling and fragments in EWT.
The domain gap hits vocabulary and reported speech most; clause-linking constructions (relative, subordinate, coordination) are shared.

## 7. Spanish (frozen table, unchanged, 300 random es_GSD sentences; `spanish.py`)
300/300 trees with exactly one root, every token attached, no cycle (depth median 8, p90 12); 7,099 nodes; 113 sentences contain multi-word tokens (`del`, `al`, clitics) handled at UD-word level.
Labels: DESCRIPTION 26%, MODIFICATION 21%, PREPOSITION 17%, COORDINATION 8%, OBJECT 6%, SUBJECT 6%, SPECIFICATION 5%, ROOT 4%, SUBORDINATION 4%, APPOSITION 2%, COMPLEMENT 0.7%.
27 UD relation types occur; all are covered by a rule. The same on 300 AnCora sentences (28 types, 300/300 well-formed). Nothing in the table was changed (it contains only UD names/UPOS/FEATS).
Caveat: well-formedness is largely guaranteed by the operations (they only rewire a tree), so this shows coverage, not correctness; there is no accuracy claim, and no Spanish parser output to compare with.
Language-specific parts left outside the table: the project tokeniser (no `del`/clitic splitting), the tagger and lexicon for Spanish.

## 8. Risks and conflicts with the project rules
- **Two teachers with different habits.** Where they differ (auxiliary head vs verb head, noun compounds, PP attachment, `to`), 38% of committed links differ in head. An encoder trained on both would see contradictory targets for the same
  construction unless the source is a feature or the loss is masked per construction; it would also pick up UD's web-text register.
- **Is the table "language rules outside the grammar file"?** Reading A (yes): the table encodes, per language-independent UD relation, how English grammar is to be shaped; its rules were tuned on English parser agreement, the operations
  encode English-specific decisions (ADP heads noun, conjunction on top), and a Spanish grammar would need its own check. It is hand-written and sits outside `quantum_parser/grammars/*.json`. Reading B (no): it contains no word, regex or tag-sequence of
  any language, only UD's own language-neutral relation inventory mapped to the project's relation inventory, it is data like the lexicon (per-corpus, logged, repeatable; the owner's "fitting is acceptable if transparent"), and the same
  table served Spanish unchanged. The project's rule says language rules live only in grammar files; whether a corpus-conversion table counts is an owner decision. Parser-agreement tuning (v1/v2 used parser labels) also leaks the parser's habits into the
  "independent" teacher, so the table should be fixed from principle, not from parser agreement, if independence matters (head agreement is unaffected by tuning).
- **UD is someone else's analysis.** The project defines meaning in USVS; UD sets heads by content-word dependency, `obl`/`nmod` choices and `compound`/`flat` conventions the project's grammar does not have. A converted tree is a UD analysis wearing the project's labels.
- **Licences:** share-alike for a committed converted copy (EWT, GSD); non-commercial (GUM, LinES, ParTUT); text copyright not cleared. Keep converted data out of the repo or release it as BY-SA.
- **Tokenisation:** 1.3% of nodes lost and 13% of sentences affected in English; Spanish needs merge rules for `del/al` and clitics; the project splits on every `.` which breaks abbreviations/URLs/numbers.
- **What the encoder learns where teachers disagree:** with both teachers in one loss, the average; with UD only on between-piece links, it learns UD's analysis exactly where the parser is silent, which is the intended use, but those links then carry UD's habits
  (aux and compounds) into trees whose committed parts follow the parser's habits: the same sentence can mix two conventions.
- **"Trees are graded, with reasons":** every converted link carries its rule index/flags (`conv["flags"]`: which rule or operation produced the label), so a UD-supervised link is traceable; grading would need a reason set for UD-vs-parser disagreement
  (head differs / label differs / UD construction) and a decision which side is "unknown, not wrong" (UD could fill exactly the parser's UNATTACHED tops).
- **Alternatives for the same evidence (not tested):** more grammar rules (the 2nd pass raised connection 27->35-38% and the variants beyond were 30-70% wrong); a different referee signal; meaning-level supervision (question answering on
  FairytaleQA-style text) which asks "is the tree usable?", not "is the link UD-right?", but gives no per-link evidence.

## 9. What would have to be true (measured value in brackets)
1. A converted copy can be shipped/used: licence terms acceptable (BY-SA for EWT/GSD ok if the converted data is BY-SA; NC ones only if non-commercial; text copyright uncleared) [partly met; owner/legal call].
2. The conversion is complete and deterministic: every sentence becomes one connected tree, no relation without a rule [met: 38,676 EN, 300+300 ES, 0 unmapped].
3. The conversion is faithful to the project's conventions [partly: 73% right / 23% debatable / 3% wrong on 60 between-clause links, single self-rater; head agreement with parser 66.5%, head+rel 60.1% on connected; known table gaps listed].
4. Disagreement with the parser is mostly the parser's or analysis difference, not the table [likely: top-15 above has 2 table-gap items; unproven for the long tail].
5. UD supplies material where the parser is silent: [met: 10.2k EN sentences >20 words vs 586; 2.3k over 35 vs 16; 52k clause links vs 9.3k].
6. UD's verdicts separate right from wrong added links: [not shown: encoder added links agree 26% head / 17% head+rel; parser's own committed links agree 61%; precision of "UD agrees => right" and recall of "UD differs => wrong" unmeasured, needs a hand-judged sample of both groups].
7. Domain gap tolerable: [open: web text, 16% unseen types, 5% quotes vs 25%; GUM fiction closer]. 
8. Tokenisation reconciled: [met for English with masking: 1.3% nodes; Spanish needs `del/al`+clitic merging, not built].
9. The owner accepts a conversion table as data outside the grammar files, and a policy for teacher conflict (mask, feature, or UD only for UNATTACHED tops) [decision, not measurable].
10. Training on the second teacher improves the long-sentence hand-check (<=15% wrong added links; currently 34%) [untested; no training allowed here].
