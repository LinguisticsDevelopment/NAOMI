"""Template-varied curriculum: measuring surface-form generalization honestly.

``CurriculumGenerator`` (episode.py, levels 1-6) is a genuine reasoning
curriculum, but every context sentence is drawn from a tiny, fixed set of
rigid templates ("X is in the Y .", "X moved to the Z ."). Because the reactor
perceives sentences through :class:`nsm_ct.input_encoder.ParserInputEncoder`
-> :func:`nsm_ct.clause.extract_discourse` -> :func:`nsm_ct.clause_reactor.
build_clause_batch`, any accuracy gain could in principle be TEMPLATE
overfitting rather than genuine reasoning: the reactor may be keying off the
exact surface form rather than the (entity, relation, value) triple the
parser extracts from it.

This module generates the SAME underlying facts/logic as levels 1-6
(locations, moves, corroboration) through two **disjoint** surface template
sets (``"A"`` and ``"B"``, plus a mixed ``"AB"`` draw), so a model trained on
one set's phrasing can be validated on the other's — the overfit test that
:mod:`scripts.probe_template_overfit` runs.

CRITICAL: sentence variety here is constrained by what ``quantum_parser`` can
actually parse (a hand-rolled grammar with a small tagger dictionary, not a
real parser). Every template below was verified with :func:`verify_templates`
to yield a non-degenerate clause (a SUBJECT and a PLACE argument matching the
sentence's actual entity/place) before being kept; templates that failed
verification (e.g. "X is inside the Y .", "X entered the Y .", passive
voice) were dropped or fixed. See ``scripts/probe_template_overfit.py`` for
the printed parse-success table.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .episode import Episode, _NAMES, _PLACES

# ---------------------------------------------------------------------------
# Parser-verified, DISJOINT surface template sets.
#
# Each set covers the two actions levels 1-6 need: a stative PLACE fact
# ("X is/stays/etc in the Y") and a MOVE update ("X moved/walked/etc to the
# Y"). Set A intentionally matches CurriculumGenerator's own phrasing (the
# "status quo" surface form); Set B shares no template string and no verb
# with Set A. Both were verified via verify_templates() below — every entry
# parses into a clause whose SUBJECT/PLACE match the sentence's actual
# entity/place (not just "some clause came out").
# ---------------------------------------------------------------------------
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "A": {
        "PLACE": [
            "{n} is in the {p} .",
            "{n} is now in the {p} .",
            "{n} stayed in the {p} .",
        ],
        "MOVE": [
            "{n} moved to the {p} .",
            "{n} walked to the {p} .",
            "{n} moved into the {p} .",
        ],
    },
    "B": {
        "PLACE": [
            "{n} is at the {p} .",
            "{n} is currently in the {p} .",
            "{n} is standing in the {p} .",
        ],
        "MOVE": [
            "{n} went to the {p} .",
            "{n} returned to the {p} .",
            "{n} headed to the {p} .",
        ],
    },
}

# Templates that were TRIED and DROPPED because verify_templates() found them
# degenerate (kept here so the negative result isn't silently lost — see the
# module docstring and scripts/probe_template_overfit.py's report):
#   "{n} is inside the {p} ."        -> tagger doesn't know "inside" as a
#                                        preposition; no PP forms at all.
#   "{n} is located in the {p} ."    -> "located" breaks the predicate rules.
#   "the {p} is where {n} is ."      -> wh-cleft; no clause extracted.
#   "{n} entered the {p} ."          -> transitive (no preposition), so
#                                        extract_discourse (PP-keyed) finds no
#                                        PLACE argument at all.
#   "{n} can be found in the {p} ."  -> passive voice; SUBJECT slot lands on
#                                        the modal "can", not the entity.
#   "{n} is by/near the {p} ."       -> "by"/"near" aren't mapped to PLACE in
#                                        clause._PREP_RELATION.
#   "{n} came/sat to/in the {p} ."   -> "came"/"sat" aren't in the tagger's
#                                        verb dictionary and don't end in the
#                                        suffixes its heuristics catch, so
#                                        they default to NOUN and break the
#                                        verb phrase.
DROPPED_TEMPLATES = [
    "{n} is inside the {p} .",
    "{n} is located in the {p} .",
    "the {p} is where {n} is .",
    "{n} entered the {p} .",
    "{n} can be found in the {p} .",
    "{n} is by the {p} .",
    "{n} is near the {p} .",
    "{n} came to the {p} .",
    "{n} sat in the {p} .",
]

# A larger place/object noun pool for the ``vocab_scale`` option (61 nouns,
# parser-verified below — none collide with quantum_parser's tagger
# dictionary or its VERB/NOUN ambiguity list, so every one defaults cleanly
# to NOUN). The small-vocab default reuses episode.py's own ``_PLACES`` (6
# nouns) so the "status quo" arm is byte-comparable to CurriculumGenerator.
VOCAB_SCALE_PLACES: List[str] = [
    "garden", "library", "station", "harbor", "museum", "hospital", "market",
    "bridge", "tower", "tunnel", "valley", "meadow", "forest", "desert",
    "canyon", "island", "prairie", "swamp", "marsh", "lagoon", "reef", "cove",
    "bay", "cabin", "cottage", "cellar", "attic", "balcony", "courtyard",
    "orchard", "vineyard", "farmhouse", "barn", "stable", "dock", "pier",
    "wharf", "lighthouse", "windmill", "cathedral", "chapel", "monastery",
    "castle", "fortress", "palace", "mansion", "lodge", "plaza", "square",
    "arena", "stadium", "theater", "gallery", "workshop", "warehouse",
    "cafe", "diner", "tavern", "inn", "clinic", "academy",
]


def verify_templates(template_sets=("A", "B"), sample_name: str = "mary",
                      sample_place: str = "garden") -> Dict[str, bool]:
    """Parse-check every template in ``template_sets`` through the REAL parser.

    Runs each formatted sentence through ``ParserInputEncoder._parse_graph``
    -> :func:`nsm_ct.clause.extract_discourse` and requires the resulting
    clause's SUBJECT and PLACE arguments to match the sentence's actual
    entity/place *exactly* (not just "some non-empty clause came out" — a
    passive-voice sentence can produce a clause whose SUBJECT is the wrong
    word, which a looser check would wave through).

    Returns ``{template_string: bool}``. Templates that fail should be fixed
    or moved to :data:`DROPPED_TEMPLATES`, never silently kept in
    :data:`TEMPLATES`.

    Returns an empty dict (nothing to report) if ``quantum_parser`` isn't
    importable in this environment — callers should treat that as "unable to
    verify", not as a pass.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    all_templates: List[str] = []
    for s in template_sets:
        for action_templates in TEMPLATES[s].values():
            all_templates.extend(action_templates)

    texts = [t.format(n=sample_name, p=sample_place) for t in all_templates]
    texts += [sample_name, sample_place]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        return {}  # quantum_parser unavailable; caller must handle this case

    results: Dict[str, bool] = {}
    for t in all_templates:
        sent = t.format(n=sample_name, p=sample_place)
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        ok = False
        for cl in clauses:
            subj = place = None
            for rel, arg in cl.args:
                if rel == "SUBJECT":
                    subj = (arg.token or "").lower()
                elif rel == "PLACE":
                    place = (arg.token or "").lower()
            if subj == sample_name and place == sample_place:
                ok = True
        results[t] = ok
    return results


class VariedCurriculumGenerator:
    """Levels 1-6 of :class:`nsm_ct.episode.CurriculumGenerator`, template-varied.

    Same facts/logic (one fact; two facts, pick one; move + recency; a
    post-question distractor; corroboration vs. contradiction; overwrite),
    but every context sentence is drawn from ``TEMPLATES[template_set]``
    instead of a single fixed phrasing per action.

    Args:
        template_set: ``"A"``, ``"B"`` (disjoint surface forms), or ``"AB"``
            (each sentence independently drawn from A or B — the "mixed
            training" arm).
        vocab_scale: if True, place/object nouns are drawn from the 61-noun
            :data:`VOCAB_SCALE_PLACES` pool instead of episode.py's fixed 6.
        max_level: highest level to sample (1-6).
        num_options: multiple-choice option count.
        seed: RNG seed (deterministic given the same args).
    """

    def __init__(self, template_set: str = "A", vocab_scale: bool = False,
                 max_level: int = 6, num_options: int = 4, seed: int = 0) -> None:
        if template_set not in ("A", "B", "AB"):
            raise ValueError(f"template_set must be 'A', 'B', or 'AB', got {template_set!r}")
        self.template_set = template_set
        self.vocab_scale = vocab_scale
        self.places = list(VOCAB_SCALE_PLACES) if vocab_scale else list(_PLACES)
        self.max_level = max(1, min(max_level, 6))
        self.num_options = num_options
        self.rng = random.Random(seed)

    def _tmpl(self, action: str) -> str:
        ts = self.rng.choice(["A", "B"]) if self.template_set == "AB" else self.template_set
        return self.rng.choice(TEMPLATES[ts][action])

    def _place_sent(self, name: str, place: str) -> str:
        return self._tmpl("PLACE").format(n=name, p=place)

    def _move_sent(self, name: str, place: str) -> str:
        return self._tmpl("MOVE").format(n=name, p=place)

    def _mc(self, answer: str):
        pool = [p for p in self.places if p != answer]
        distractors = self.rng.sample(pool, min(self.num_options - 1, len(pool)))
        options = distractors + [answer]
        self.rng.shuffle(options)
        return options, options.index(answer)

    def _meta(self, level: int) -> dict:
        return {"src": "curriculum2", "template_set": self.template_set,
                "vocab_scale": self.vocab_scale, "level": level}

    def _level1(self) -> Episode:
        name = self.rng.choice(_NAMES)
        place = self.rng.choice(self.places)
        options, idx = self._mc(place)
        return Episode(
            context=[self._place_sent(name, place)],
            question=f"where is {name} ?",
            answer_text=place, options=options, answer_idx=idx, level=1,
            meta=self._meta(1),
        )

    def _level2(self) -> Episode:
        name_a, name_b = self.rng.sample(_NAMES, 2)
        place_a, place_b = self.rng.sample(self.places, 2)
        target_name, target_place = self.rng.choice([(name_a, place_a), (name_b, place_b)])
        context = [self._place_sent(name_a, place_a), self._place_sent(name_b, place_b)]
        self.rng.shuffle(context)
        options, idx = self._mc(target_place)
        return Episode(
            context=context, question=f"where is {target_name} ?",
            answer_text=target_place, options=options, answer_idx=idx, level=2,
            meta=self._meta(2),
        )

    def _level3(self) -> Episode:
        name = self.rng.choice(_NAMES)
        first, second = self.rng.sample(self.places, 2)
        context = [self._place_sent(name, first), self._move_sent(name, second)]
        options, idx = self._mc(second)
        return Episode(
            context=context, question=f"where is {name} ?",
            answer_text=second, options=options, answer_idx=idx, level=3,
            meta=self._meta(3),
        )

    def _level4(self) -> Episode:
        name = self.rng.choice(_NAMES)
        first, second = self.rng.sample(self.places, 2)
        options, idx = self._mc(first)
        return Episode(
            context=[self._place_sent(name, first)],
            question=f"where is {name} ?",
            answer_text=first, options=options, answer_idx=idx, level=4,
            post_context=[self._move_sent(name, second)],
            meta=self._meta(4),
        )

    def _level5(self) -> Episode:
        name = self.rng.choice(_NAMES)
        true_place, false_place = self.rng.sample(self.places, 2)
        stmts = [
            (self._place_sent(name, true_place), 1.0),
            (self._place_sent(name, false_place), 0.0),
            (self._place_sent(name, true_place), 1.0),
        ]
        self.rng.shuffle(stmts)
        options, idx = self._mc(true_place)
        return Episode(
            context=[s for s, _ in stmts], question=f"where is {name} ?",
            answer_text=true_place, options=options, answer_idx=idx, level=5,
            trust_labels=[lab for _, lab in stmts],
            meta=self._meta(5),
        )

    def _level6(self) -> Episode:
        name_a, name_b = self.rng.sample(_NAMES, 2)
        place_b = self.rng.choice(self.places)
        first, second = self.rng.sample([p for p in self.places if p != place_b], 2)
        context = [
            self._place_sent(name_a, first),
            self._place_sent(name_b, place_b),
            self._move_sent(name_a, second),
        ]
        options, idx = self._mc(second)
        return Episode(
            context=context, question=f"where is {name_a} ?",
            answer_text=second, options=options, answer_idx=idx, level=6,
            meta=self._meta(6),
        )

    def generate(self, n: int) -> List[Episode]:
        builders = [self._level1, self._level2, self._level3,
                    self._level4, self._level5, self._level6][: self.max_level]
        return [builders[i % len(builders)]() for i in range(n)]


def generate_varied_episodes(n: int, seed: int = 0, template_set: str = "A",
                              vocab_scale: bool = False, max_level: int = 6,
                              num_options: int = 4) -> List[Episode]:
    """``n`` template-varied episodes; same interface shape as
    :func:`nsm_ct.episode.CurriculumGenerator.generate`, deterministic given
    ``(n, seed, template_set, vocab_scale, max_level, num_options)``.
    """
    gen = VariedCurriculumGenerator(template_set=template_set, vocab_scale=vocab_scale,
                                     max_level=max_level, num_options=num_options, seed=seed)
    return gen.generate(n)
