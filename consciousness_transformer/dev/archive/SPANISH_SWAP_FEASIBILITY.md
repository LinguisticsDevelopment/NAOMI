# Spanish grammar-swap-test feasibility probe

Status: MEASUREMENT ONLY. No encoder/model code changed. Branch:
`claude/m27-m28-cleanup`. Question asked: train the encoder on English, then
at test time feed Spanish grammar+senses+text to the SAME weights — do the
two data prerequisites for that swap exist?

Setup executed: `pip install torch numpy nltk pytest`, `pip install -e .`,
NLTK `wordnet`/`omw-1.4`/`omw-2.0` downloaded (their zips were not
auto-extracted by `nltk.download`; extracted manually under
`/root/nltk_data/corpora/`), `python scripts/build_usvs.py` (no prebuilt
artifact existed — only a stale `usvs_meta.json`; a fresh build took 180s:
9,946 core words / 607 axes / 117,659 senses).

---

## Check 1 — does USVS resolve SPANISH words to senses?

**Result: NO, out of the box.** `USVS.senses_of(word)` (the encoder's sense
input) is built entirely from `nsm_ct/wordnet.py`'s `all_senses()`, which
calls `s.lemmas()` with **no `lang` argument** — English lemmas only. Tested
directly against the built artifact:

```
senses_of('perro') -> []
senses_of('gato')  -> []
senses_of('casa')  -> []
senses_of('correr')-> []
senses_of('comer') -> ['comer.n.01', 'arrival.n.03']   # accidental: English surname sense
senses_of('nino')  -> []                                # missing tilde (typo in the task's own word list)
senses_of('niño')  -> []                                # correct spelling, still empty
senses_of('agua')  -> ['agua.n.01']                     # accidental: English lemma "agua toad"
senses_of('grande')-> []
senses_of('rojo')  -> []
```

**8 of 9 genuinely empty; the 2 non-empty hits are coincidences** — `comer`
and `agua` happen to also be English WordNet lemma strings (a surname sense,
a toad species), not real Spanish resolution. **0/9 correct.**

**Exact gap, and what it takes to close it:** NLTK's WordNet reader ships
OMW (Open Multilingual Wordnet) with a Spanish layer already present in the
downloaded corpus, under `omw-1.4/mcr/wn-data-spa.tab` and
`omw-2.0/mcr/wn-data-spa.tab` (`mcr` = Multilingual Central Repository,
which bundles Spanish/Catalan/Basque/Galician). It just isn't in
`wn.langs()` until `wn.add_omw()` runs (lazy — happens automatically the
first time any `lang=` call is made). Once loaded:

```python
wn.synset('dog.n.01').lemmas(lang='spa')
# -> [Lemma('dog.n.01.can'), ..., Lemma('dog.n.01.perro'), ...]
```

This is a **build-flag change, not an architectural one**: `wordnet.py`'s
`all_senses()` (and/or `usvs.py`'s sense-lemma collection in `build_usvs`)
would need `lemmas(lang="spa")` unioned in alongside the English lemmas per
sense, keyed to the SAME `sense_id` (WordNet synset IDs are the
interlingual index OMW hangs off of — this is the entire premise the
project's own `scripts/probe_spanish_freeze.py` already documents:
"gato" -> `cat.n.01` -> the same vector as "cat"). **Crucially, no
re-grounding is needed** — sense signatures are already grounded per
synset ID, language-independent; adding Spanish only extends the
*lemma → sense_id lookup table*, not the grounding itself.

**Enabling tweak tried (flagged, not applied to `src/`):** to verify this
precisely rather than assert it, `scripts/probe_spanish_lattice_sample.py`
(new, this probe) builds a **standalone, ~15-line supplementary Spanish
lemma index** over the 117,659 sense_ids already in the built USVS artifact
(`wn.synset(sid).lemmas(lang="spa")` per sense_id, no changes to
`nsm_ct/ground/usvs.py` or `nsm_ct/wordnet.py`). Build cost: **~4.3s**,
producing **88,942 Spanish lemmas**. Re-tested the same 9 words against it:

```
perro  -> ['dog.n.01', 'rotter.n.01']
gato   -> ['cat.n.01', 'caterpillar.n.02', 'dodger.n.01', 'kitty.n.04', 'tom.n.02']
casa   -> ['building.n.01', 'diggings.n.02', 'dwelling.n.01', 'family.n.01', 'firm.n.01']
correr -> ['draw.v.20', 'fiddle.v.01', 'flow.v.01', 'fly.v.02', 'pour.v.04']
comer  -> ['consume.v.02', 'consume.v.05', 'eat.v.01', 'eat.v.02', 'feed.v.06']
nino   -> []                              # still empty: missing tilde, not a system gap
niño   -> ['baby.n.01', 'chap.n.01', 'child.n.01', 'child.n.02', 'child.n.03']
agua   -> ['agua.n.01', 'body_of_water.n.01', 'rain.n.02', 'water.n.01', 'water.n.03']
grande -> ['ace.s.01', 'authoritative.s.01', 'bad.s.01', 'big.s.01', 'big.s.03']
rojo   -> ['bloodshot.s.01', 'bolshevik.n.02', 'communist.n.02', 'crimson.s.03', 'red.n.01']
```

**8/9 now resolve** (the one miss, `nino`, is a missing-accent typo — `niño`
resolves fine). Confirmed the resolved sense_ids are already grounded in the
USVS (e.g. `dog.n.01`, `cat.n.01`, `child.n.01` each carry a populated
24-axis signature, unchanged by this indexing addition). **This closes Check
1's gap in principle** but was deliberately kept out of `src/` per the
task's scope — it lives only in the new probe script, clearly labeled as a
prototype, not a production change.

One narrower limitation surfaced while building the §3 sample below: OMW-es
lemmas are **uninflected dictionary forms**. Conjugated Spanish verb tokens
("está", "fue", "tomó") do **not** hit the lemma index — only base nouns did
in the sample. `scripts/build_parser_lexicon.py`'s own module docstring
already documents this as a known, deliberate scope-narrowing ("No
verb-conjugation generation... Spanish conjugation is a different, much
larger system this script does not attempt"). This is a real, separate gap
from Check 1's core indexing question — it affects predicate sense-grounding
specifically, not noun/entity resolution.

---

## Check 2 — does quantum_parser TAG + PARSE Spanish end-to-end?

**Result: YES**, given correctly-accented input — and there is substantially
more Spanish infrastructure already built on this branch than the task
description assumed:

- `quantum_parser/grammars/spanish.json` exists (a deliberate clone of
  `english.json` — rules match abstract Tag/NodeType/SubType categories, not
  literal words, per its own metadata and `tests/test_spanish_freeze.py`).
- `quantum_parser/src/parser/pos_tagger.py` has `tag_spanish_sentence()` and
  a `SPANISH_WORD_TAG_DICT` for closed-class words.
- `quantum_parser/data/es_lexicon.json.gz` is already built (337KB, from
  `scripts/build_parser_lexicon.py --lang spa`, which sources
  `wn.all_lemma_names(pos=p, lang="spa")` — the same OMW-es layer as Check
  1) and sits alongside `en_lexicon.json.gz`.
- `nsm_ct/input_encoder.py`'s `ParserInputEncoder` already accepts
  `lang="es"` (selects the Spanish tagger + grammar + a Spanish
  preposition→role map, `_install_spanish_prep_relation`).
- `nsm_ct/meaning_es.py` (`SpanishMeaningResolver`), `nsm_ct/curriculum2.py`
  (`TEMPLATES_ES`, `TRANSFER_TEMPLATES_ES`, `_PLACES_ES`,
  `_TRANSFER_OBJECTS_ES`, `verify_templates_es`), and
  `scripts/probe_spanish_freeze.py` (a prior, more thorough "Spanish Freeze
  Test" stream-equivalence probe) all already exist on this branch.

Parsed the task's 3 requested sentences through the exact gold-builder path
(`ParserInputEncoder(lang="es")._tag` → `._parse_graph` →
`clause.extract_discourse`):

```
"el perro corre ."        -> OK: SUBJECT=perro -> CLAUSE=corre. Tagged DET/NOUN/VERB/PUNCT correctly.
"la nina come pan ."      -> OK: SUBJECT=nina, OBJECT=pan -> CLAUSE=come. Tagged correctly.
"maria esta en la casa ." -> FAILS as literally typed (no clause at all: roots=[0,1,2,5], no SUBJECT edge).
```

**Where it fails, precisely:** the third sentence's ASCII-only spelling
collides two distinct Spanish words. `esta` (no accent) is the demonstrative
"this" (feminine) and tags as `DET`; `está` (accented, "is") is the verb and
tags as `VERB` via `SPANISH_WORD_TAG_DICT`. Without the accent the tagger
picks the (also valid) determiner reading, no verb node exists, and no
clause forms. This is a **diacritics issue in the task's own test-sentence
spelling**, not a tagger/grammar/grounding gap — re-run with correct
Spanish orthography:

```
"maría está en la casa ." -> OK: SUBJECT=maría, PLACE=casa -> CLAUSE=está
  MODIFICATION+PREPOSITION edges correct; tagged NOUN/VERB/ADP/DET/NOUN/PUNCT.
```

All three sentences parse to grounded, correctly-role-labeled trees once
properly accented. This matches the "nino"/"niño" accent sensitivity
already seen in Check 1 — Spanish diacritics matter throughout this
pipeline and un-accented test input silently degrades results without
raising an error.

---

## VERDICT: READY, contingent on the two gaps above being closed

Both checks pass in principle. Neither is a structural/architectural
blocker:

1. **Check 1's gap** (USVS Spanish lemma indexing) is closed by a small,
   verified, additive change — extend `all_senses()`/`build_usvs`'s
   sense-lemma collection with `lemmas(lang="spa")` (or build a
   supplementary Spanish index at load time, as this probe does
   standalone). No re-grounding required. **Narrower remaining gap:**
   conjugated verb forms won't resolve via base OMW-es lemmas without
   Spanish lemmatization/conjugation-form generation (documented, separate
   scope from the base indexing fix; `build_parser_lexicon.py` already
   flags this for the parser-tagger lexicon).
2. **Check 2** already works end-to-end on this branch (tagger, grammar,
   lexicon, prep-role map, and full clause extraction all present and
   tested) — no gap to close beyond correct input accenting.
3. **A separate, orthogonal gap this probe surfaced:** there is **no
   committed V2 encoder-gold BUILDER script** in this repo for *either*
   language — `runs/encoder_gold_v2.jsonl` is referenced by
   `scripts/train_encoder.py`/`scripts/eval_encoder.py` and read by
   `nsm_ct/encoder_model._gold_sites`, but nothing in `scripts/` or
   `src/nsm_ct/` turns raw text into that file. Building an actual
   Spanish (or English) gold set at scale needs that builder written first;
   this probe's sample generator (`scripts/probe_spanish_lattice_sample.py`)
   is a minimal, ad hoc stand-in for that missing piece, not a substitute
   for it.

**Sample produced:** `runs/spanish_gold_probe.jsonl`, 20 records, built from
already-verified `curriculum2.TEMPLATES_ES`/`TRANSFER_TEMPLATES_ES`
sentences (PLACE/MOVE/TRANSFER shapes over the 6 names × 6 places × 6
transfer objects), each parsed through the real
`ParserInputEncoder(lang="es")` → `clause.extract_discourse` path and
grounded via the standalone Spanish lemma index from Check 1. All 20 match
the V2 record shape (`text`/`tokens`/`pos`/`lattice.trees`/
`lattice.discourse_links_per_tree`/`token_sense_candidates`) and pass a
shape check run through the real reader, `nsm_ct.encoder_model._gold_sites`,
with no exceptions. Grounding-type breakdown across the 20 records: 26
`sense`-type slots (place/object nouns — cocina, jardín, llave, oficina, ...)
and 46 `entity`-type slots (person-name subjects/sources, and — the
predicate-verb limitation above — the conjugated predicate verbs
está/fue/tomó, which correctly fall back to `entity` per the contract's
documented behavior for a content token the lemma index doesn't cover).

**Bottom line:** a real Spanish candidate-lattice test set for the
grammar-swap test is buildable now, at small additional engineering cost —
(a) add the Spanish lemma-index extension to `build_usvs`/`wordnet.py`
(small, flagged above, not done here), (b) decide how to handle conjugated
Spanish verbs for predicate sense-grounding (lemmatize before lookup, or
extend the tagger lexicon's stored base-lemma per inflected form), and (c)
write the actual V2 gold-builder script this repo is currently missing for
either language (this probe's sample generator is a stand-in, not that
script).
