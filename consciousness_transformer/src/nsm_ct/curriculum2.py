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

# Set "C": templates originally dropped, then FIXED by the M36 parser round
# (tagger lexicon: inside/near as ADP, come/came + sit/sat as VERB; clause.py:
# by/near -> PLACE frame entries; the _LOCATIVE_TRANSITIVE_VERBS allow-list for
# bare-object "entered") and the M38 parser round-2 fixes (scorer tie-break:
# ParseChart prefers the hypothesis with a SUBJECT edge on ties; passive voice:
# new aux1 rule + found->VERB + inf1/neg1 unscoped-pattern fixes). Kept as a
# separate set so A/B stay byte-identical to the M35 overfit-audit runs.
TEMPLATES["C"] = {
    "PLACE": [
        "{n} is inside the {p} .",
        "{n} is by the {p} .",
        "{n} is near the {p} .",
        "{n} sat in the {p} .",
        "{n} is located in the {p} .",
        "{n} can be found in the {p} .",
    ],
    "MOVE": [
        "{n} came to the {p} .",
        "{n} entered the {p} .",
    ],
}

# Templates that were TRIED and DROPPED because verify_templates() found them
# degenerate (kept here so the negative result isn't silently lost).
#   "the {p} is where {n} is ."      -> wh-cleft; no equative ruleset exists,
#                                        and there's a real prerequisite bug:
#                                        verb1's adverbial-SPECIFIER rule has
#                                        no subtype filter, so it grabs a
#                                        relative "where" before rel2 ever
#                                        sees it (same unscoped-pattern family
#                                        as the round-2 inf1/neg1 fixes -- a
#                                        round-3 candidate). Single template,
#                                        no curriculum need; deferred.
DROPPED_TEMPLATES = [
    "the {p} is where {n} is .",
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
        scaled: if True, use the denser "scaled" builders instead of the
            plain levels 1-6 — same facts/logic per level, but each episode
            packs ``min_facts``-``max_facts`` context sentences over
            ``min_entities``-``max_entities`` distinct entities (distractor
            facts about entities other than the one being asked about),
            instead of the 1-3 sentences / 1-2 entities of the plain levels.
            This is the M-scaling experiment's data axis (see
            scripts/probe_scaled_training.py); every sentence still comes
            from :data:`TEMPLATES` (no new surface shapes), so parseability
            is inherited from the already-verified A/B sets.
        min_facts, max_facts: per-episode context-sentence count range,
            ``scaled`` mode only (default 4-8).
        min_entities, max_entities: per-episode distinct-entity count range,
            ``scaled`` mode only (default 2-4; clamped to ``len(_NAMES)``,
            the fixed 6-name pool shared with episode.py — the noun/place
            vocabulary scales via ``vocab_scale``, but entity NAMES do not).
    """

    def __init__(self, template_set: str = "A", vocab_scale: bool = False,
                 max_level: int = 6, num_options: int = 4, seed: int = 0,
                 scaled: bool = False, min_facts: int = 4, max_facts: int = 8,
                 min_entities: int = 2, max_entities: int = 4) -> None:
        if template_set not in ("A", "B", "AB"):
            raise ValueError(f"template_set must be 'A', 'B', or 'AB', got {template_set!r}")
        if min_facts > max_facts:
            raise ValueError(f"min_facts ({min_facts}) must be <= max_facts ({max_facts})")
        if min_entities > max_entities:
            raise ValueError(f"min_entities ({min_entities}) must be <= max_entities ({max_entities})")
        self.template_set = template_set
        self.vocab_scale = vocab_scale
        self.places = list(VOCAB_SCALE_PLACES) if vocab_scale else list(_PLACES)
        self.max_level = max(1, min(max_level, 6))
        self.num_options = num_options
        self.rng = random.Random(seed)
        self.scaled = scaled
        self.min_facts, self.max_facts = min_facts, max_facts
        self.max_entities = min(max_entities, len(_NAMES))
        self.min_entities = min(min_entities, self.max_entities)

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

    def _meta(self, level: int, **extra) -> dict:
        meta = {"src": "curriculum2", "template_set": self.template_set,
                "vocab_scale": self.vocab_scale, "level": level, "scaled": self.scaled}
        meta.update(extra)
        return meta

    # -- "scaled" mode density helpers --------------------------------------
    def _n_facts(self) -> int:
        return self.rng.randint(self.min_facts, self.max_facts)

    def _n_entities(self) -> int:
        return self.rng.randint(self.min_entities, self.max_entities)

    def _interleave(self, own: List[str], extra: List[str]) -> List[str]:
        """Merge ``extra`` into ``own`` at random positions, preserving
        ``own``'s relative order (each extra item is independently inserted,
        so two ``own`` items are never swapped past each other). This matters
        for recency-sensitive levels where ``own`` is a chronological
        fact/move chain for the entity being asked about."""
        merged = list(own)
        for item in extra:
            merged.insert(self.rng.randrange(len(merged) + 1), item)
        return merged

    def _distractor_stmts(self, entity_pool: List[str], count: int) -> List[str]:
        """``count`` extra PLACE facts about entities drawn from
        ``entity_pool`` (never the entity/entities being asked about, so
        padding can never contradict what the episode is actually testing).
        """
        pool = entity_pool or [nm for nm in _NAMES]
        return [self._place_sent(self.rng.choice(pool), self.rng.choice(self.places))
                for _ in range(max(0, count))]

    def _dense_context(self, own: List[str], exclude: List[str], n_entities: int,
                        n_facts: int) -> List[str]:
        """Pad ``own`` (the target entity's/entities' own chronological facts)
        up to ``n_facts`` total context items using up to ``n_entities -
        len(exclude)`` OTHER distinct entities (never ``exclude``), padding
        any remaining budget with repeats of those same other entities (never
        introducing entities beyond that count, so the realized entity count
        never exceeds ``n_entities``). ``own``'s relative order is preserved.
        """
        n_facts = max(n_facts, len(own))
        budget = n_facts - len(own)
        pool = [nm for nm in _NAMES if nm not in exclude]
        k_others = min(max(n_entities - len(exclude), 0), len(pool), budget)
        others = self.rng.sample(pool, k_others) if pool and k_others else []
        extra = [self._place_sent(o, self.rng.choice(self.places)) for o in others]
        # Any leftover padding budget repeats the SAME (safe, non-excluded)
        # entities already chosen -- never a fresh independent draw from the
        # whole pool, which would let realized entity count silently exceed
        # ``n_entities`` (a single fallback entity when ``others`` is empty).
        fallback = others or ([self.rng.choice(pool)] if pool else list(_NAMES))
        extra += self._distractor_stmts(fallback, budget - len(extra))
        return self._interleave(own, extra)

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

    # -----------------------------------------------------------------
    # "scaled" mode: the SAME facts/logic as levels 1-6, denser -- more
    # context facts per episode and more distinct entities in view, via
    # distractor facts about entities other than the one being asked about.
    # Every sentence still comes from TEMPLATES (no new surface shapes), so
    # parseability is inherited from the module's already-verified A/B sets;
    # density comes purely from generating MORE of those verified sentences
    # per episode. This is the M-scaling experiment's data axis (see
    # scripts/probe_scaled_training.py).
    # -----------------------------------------------------------------
    def _scaled_level1(self) -> Episode:
        # One fact about the target, padded with distractor facts about
        # other entities -- level 1's single-fact recall under more data /
        # more distinct entities in view.
        name = self.rng.choice(_NAMES)
        place = self.rng.choice(self.places)
        own = [self._place_sent(name, place)]
        n_entities, n_facts = self._n_entities(), self._n_facts()
        context = self._dense_context(own, [name], n_entities, n_facts)
        self.rng.shuffle(context)  # no recency to preserve: a single stative fact
        options, idx = self._mc(place)
        return Episode(
            context=context, question=f"where is {name} ?",
            answer_text=place, options=options, answer_idx=idx, level=1,
            meta=self._meta(1, n_facts=len(context), n_entities=n_entities),
        )

    def _scaled_level2(self) -> Episode:
        # Many entities each with one fact; ask about one -- level 2 with
        # 2-4 entities instead of a fixed 2, padded with distractor facts
        # (about entities already in play) to reach the target fact count.
        n_entities = self._n_entities()
        entities = self.rng.sample(_NAMES, n_entities)
        places_k = self.rng.sample(self.places, min(n_entities, len(self.places)))
        own = [self._place_sent(nm, pl) for nm, pl in zip(entities, places_k)]
        ti = self.rng.randrange(len(own))
        target_name, target_place = entities[ti], places_k[ti]
        n_facts = max(self._n_facts(), len(own))
        others = [nm for nm in entities if nm != target_name] or entities
        extra = self._distractor_stmts(others, n_facts - len(own))
        context = own + extra
        self.rng.shuffle(context)
        options, idx = self._mc(target_place)
        return Episode(
            context=context, question=f"where is {target_name} ?",
            answer_text=target_place, options=options, answer_idx=idx, level=2,
            meta=self._meta(2, n_facts=len(context), n_entities=n_entities),
        )

    def _scaled_level3(self) -> Episode:
        # Move + recency, denser: a 2-4-hop move chain for the target (own
        # chronological order preserved), interleaved with distractor facts
        # about other entities.
        name = self.rng.choice(_NAMES)
        n_moves = self.rng.randint(1, 3)
        chain = self.rng.sample(self.places, min(n_moves + 1, len(self.places)))
        own = [self._place_sent(name, chain[0])] + [self._move_sent(name, p) for p in chain[1:]]
        final = chain[-1]
        n_entities, n_facts = self._n_entities(), self._n_facts()
        context = self._dense_context(own, [name], n_entities, n_facts)
        options, idx = self._mc(final)
        return Episode(
            context=context, question=f"where is {name} ?",
            answer_text=final, options=options, answer_idx=idx, level=3,
            meta=self._meta(3, n_facts=len(context), n_entities=n_entities),
        )

    def _scaled_level4(self) -> Episode:
        # Post-question distractors, denser: several entities' facts before
        # the question, then the corrupting move PLUS extra unrelated facts
        # after it -- the response-timing probe under more data.
        name = self.rng.choice(_NAMES)
        first, second = self.rng.sample(self.places, 2)
        n_entities, n_facts = self._n_entities(), self._n_facts()
        n_facts = max(n_facts, 2)  # >= 1 pre-question fact + 1 post-question move
        context = self._dense_context([self._place_sent(name, first)], [name],
                                       n_entities, n_facts - 1)
        post = [self._move_sent(name, second)]
        post += self._distractor_stmts([nm for nm in _NAMES if nm != name],
                                        n_facts - len(context) - 1)
        self.rng.shuffle(post)
        options, idx = self._mc(first)
        return Episode(
            context=context, question=f"where is {name} ?",
            answer_text=first, options=options, answer_idx=idx, level=4,
            post_context=post,
            meta=self._meta(4, n_facts=len(context) + len(post), n_entities=n_entities),
        )

    def _scaled_level5(self) -> Episode:
        # Corroboration vs contradiction, denser: 2-3 corroborating
        # statements, 1-2 contradicting ones, plus distractor facts about
        # other entities (labeled trustworthy -- they're true, just
        # irrelevant to the question, so they can never mask the answer).
        name = self.rng.choice(_NAMES)
        true_place, false_place = self.rng.sample(self.places, 2)
        n_corrob, n_contra = self.rng.randint(2, 3), self.rng.randint(1, 2)
        own = [(self._place_sent(name, true_place), 1.0) for _ in range(n_corrob)]
        own += [(self._place_sent(name, false_place), 0.0) for _ in range(n_contra)]
        self.rng.shuffle(own)
        n_entities = self._n_entities()
        n_facts = max(self._n_facts(), len(own))
        budget = n_facts - len(own)
        pool = [nm for nm in _NAMES if nm != name]
        k_others = min(max(n_entities - 1, 0), len(pool), budget)
        others = self.rng.sample(pool, k_others) if pool and k_others else []
        extra = [(self._place_sent(o, self.rng.choice(self.places)), 1.0) for o in others]
        fallback = others or [self.rng.choice(pool)]  # bound realized entities to n_entities
        extra += [(s, 1.0) for s in self._distractor_stmts(fallback, budget - len(extra))]
        combined = self._interleave(own, extra)
        context = [s for s, _ in combined]
        options, idx = self._mc(true_place)
        return Episode(
            context=context, question=f"where is {name} ?",
            answer_text=true_place, options=options, answer_idx=idx, level=5,
            trust_labels=[lab for _, lab in combined],
            meta=self._meta(5, n_facts=len(context), n_entities=n_entities),
        )

    def _scaled_level6(self) -> Episode:
        # Overwrite, denser: the target's place -> move (own chronological
        # order preserved) amid several unrelated entities' facts, not just
        # the one distractor of the plain level.
        name_a, name_b = self.rng.sample(_NAMES, 2)
        place_b = self.rng.choice(self.places)
        first, second = self.rng.sample([p for p in self.places if p != place_b], 2)
        own = [self._place_sent(name_a, first), self._move_sent(name_a, second)]
        required = [self._place_sent(name_b, place_b)]
        n_entities = self._n_entities()
        n_facts = max(self._n_facts(), len(own) + len(required))
        budget = n_facts - len(own) - len(required)
        pool = [nm for nm in _NAMES if nm != name_a]  # name_b may recur; only name_a is off-limits
        k_others = min(max(n_entities - 2, 0), len(pool), budget)
        others = self.rng.sample(pool, k_others) if pool and k_others else []
        extra = required + [self._place_sent(o, self.rng.choice(self.places)) for o in others]
        fallback = others or [name_b]  # name_b is already realized via `required`; bound entities
        extra += self._distractor_stmts(fallback, budget - len(others))
        context = self._interleave(own, extra)
        options, idx = self._mc(second)
        return Episode(
            context=context, question=f"where is {name_a} ?",
            answer_text=second, options=options, answer_idx=idx, level=6,
            meta=self._meta(6, n_facts=len(context), n_entities=n_entities),
        )

    def generate(self, n: int) -> List[Episode]:
        if self.scaled:
            builders = [self._scaled_level1, self._scaled_level2, self._scaled_level3,
                        self._scaled_level4, self._scaled_level5, self._scaled_level6]
        else:
            builders = [self._level1, self._level2, self._level3,
                        self._level4, self._level5, self._level6]
        builders = builders[: self.max_level]
        return [builders[i % len(builders)]() for i in range(n)]


def generate_varied_episodes(n: int, seed: int = 0, template_set: str = "A",
                              vocab_scale: bool = False, max_level: int = 6,
                              num_options: int = 4, scaled: bool = False,
                              min_facts: int = 4, max_facts: int = 8,
                              min_entities: int = 2, max_entities: int = 4) -> List[Episode]:
    """``n`` template-varied episodes; same interface shape as
    :func:`nsm_ct.episode.CurriculumGenerator.generate`, deterministic given
    ``(n, seed, template_set, vocab_scale, max_level, num_options, scaled,
    min_facts, max_facts, min_entities, max_entities)``. See
    :class:`VariedCurriculumGenerator` for what ``scaled`` and the
    facts/entities range do.
    """
    gen = VariedCurriculumGenerator(template_set=template_set, vocab_scale=vocab_scale,
                                     max_level=max_level, num_options=num_options, seed=seed,
                                     scaled=scaled, min_facts=min_facts, max_facts=max_facts,
                                     min_entities=min_entities, max_entities=max_entities)
    return gen.generate(n)


def generate_scaled_episodes(n: int, seed: int = 0, template_set: str = "AB",
                              vocab_scale: bool = True, max_level: int = 6,
                              num_options: int = 4, min_facts: int = 4, max_facts: int = 8,
                              min_entities: int = 2, max_entities: int = 4) -> List[Episode]:
    """The M-scaling experiment's episode source: mixed A+B templates, the
    61-noun ``vocab_scale`` pool, and denser episodes (4-8 facts, 2-4
    entities, distractors) by default -- levels 1-6's own facts/logic, just
    more of it per episode. A thin, self-documenting wrapper around
    :func:`generate_varied_episodes` with ``scaled=True``. See
    scripts/probe_scaled_training.py.
    """
    return generate_varied_episodes(n, seed=seed, template_set=template_set,
                                     vocab_scale=vocab_scale, max_level=max_level,
                                     num_options=num_options, scaled=True,
                                     min_facts=min_facts, max_facts=max_facts,
                                     min_entities=min_entities, max_entities=max_entities)


# ---------------------------------------------------------------------------
# M52 -- multi-arg TRANSFER curriculum (give/hand/pass/take:
# AGENT/OBJECT/RECIPIENT/SOURCE/PLACE). The curriculum half of the M52
# plumbing prerequisite for the resolver work (dev/MIND_INTERFACE.md,
# dev/RESOLVER_BUILD_PLAN.md Phase 1); the reactor half is
# :mod:`nsm_ct.clause_reactor`'s per-role step unrolling
# (``_context_steps``/``_queried_role``).
#
# LANDMINE AVOIDED (deviation from the literal spec template, recorded here
# so it isn't silently "fixed" back): the natural dative phrasing
# "{giver} gave the {obj} to {receiver} in the {place} ." does NOT parse
# cleanly through the real quantum_parser -- clause.py's ``_PREP_RELATION``
# maps "to" to PLACE unconditionally (the directional-PLACE convention
# "moved to the office" needs it), so "to {receiver}" comes out mislabeled
# PLACE, indistinguishable from the real "in the {place}" PP (two PLACE
# args, no RECIPIENT role at all -- verified empirically, not just read off
# the code). clause.py is out of scope for M52 (FILES OWNED), so the fix is
# a template choice: the double-object construction
# ("{giver} gave {receiver} the {obj} ...") instead, which quantum_parser
# resolves to a clean INDIRECT_OBJECT edge for {receiver}. Verified via
# :func:`verify_transfer_templates` below, exactly like ``verify_templates``
# gates :data:`TEMPLATES`.
# ---------------------------------------------------------------------------
_TRANSFER_OBJECTS: List[str] = ["ball", "box", "key", "book", "letter", "coin"]

# GIVE-family (double-object: SUBJECT/INDIRECT_OBJECT/OBJECT/PLACE all
# resolve correctly) + the TAKE/SOURCE variant (SUBJECT/SOURCE/OBJECT/PLACE).
TRANSFER_TEMPLATES: Dict[str, str] = {
    "GIVE": "{giver} gave {receiver} the {obj} in the {place} .",
    "HAND": "{giver} handed {receiver} the {obj} in the {place} .",
    "PASS": "{giver} passed {receiver} the {obj} in the {place} .",
    "TAKE": "{taker} took the {obj} from {source} in the {place} .",
}


def verify_transfer_templates(sample_giver: str = "mary", sample_receiver: str = "john",
                               sample_obj: str = "ball", sample_place: str = "garden"
                               ) -> Dict[str, Dict[str, object]]:
    """Parse-check every :data:`TRANSFER_TEMPLATES` entry through the REAL
    parser, requiring EVERY role (not just SUBJECT+PLACE, unlike
    :func:`verify_templates`) to resolve to the exact expected token.

    Returns ``{template_key: {"sentence": str, "ok": bool, "roles": {rel:
    token}}}`` -- the per-template table the M52 gate reports. A template
    that fails must be fixed or dropped, never silently kept (mirrors
    :func:`verify_templates`'s contract). Returns an empty dict if
    ``quantum_parser`` isn't importable in this environment.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    expected = {
        "GIVE": {"SUBJECT": sample_giver, "INDIRECT_OBJECT": sample_receiver,
                  "OBJECT": sample_obj, "PLACE": sample_place},
        "HAND": {"SUBJECT": sample_giver, "INDIRECT_OBJECT": sample_receiver,
                  "OBJECT": sample_obj, "PLACE": sample_place},
        "PASS": {"SUBJECT": sample_giver, "INDIRECT_OBJECT": sample_receiver,
                  "OBJECT": sample_obj, "PLACE": sample_place},
        "TAKE": {"SUBJECT": sample_giver, "SOURCE": sample_receiver,
                  "OBJECT": sample_obj, "PLACE": sample_place},
    }
    texts = {k: t.format(giver=sample_giver, receiver=sample_receiver, obj=sample_obj,
                          place=sample_place, taker=sample_giver, source=sample_receiver)
             for k, t in TRANSFER_TEMPLATES.items()}
    tok = SimpleTokenizer.build(
        list(texts.values()) + [sample_giver, sample_receiver, sample_obj, sample_place],
        extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        return {}  # quantum_parser unavailable; caller must handle this case

    results: Dict[str, Dict[str, object]] = {}
    for key, sent in texts.items():
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        want = expected[key]
        got: Dict[str, str] = {}
        ok = False
        for cl in clauses:
            roles = {rel: (arg.token or "").lower() for rel, arg in cl.args}
            if all(roles.get(rel) == val for rel, val in want.items()):
                ok = True
                got = roles
                break
            if len(roles) > len(got):   # keep the closest miss for the report
                got = roles
        results[key] = {"sentence": sent, "ok": ok, "roles": got}
    return results


class TransferCurriculumGenerator:
    """Multi-arg transfer episodes (M52): give/hand/pass/take events over
    AGENT(SUBJECT)/OBJECT/RECIPIENT(INDIRECT_OBJECT)/SOURCE/PLACE roles --
    the curriculum half of the M52 plumbing (the reactor half is
    :func:`nsm_ct.clause_reactor.build_clause_batch`'s per-role step
    unrolling, keyed off the transferred OBJECT as the shared entity). Two
    levels:

        7 ("transfer_place"): one transfer event; the question asks WHERE
          the transferred object ended up. Answerable regardless of which
          of the 4 verbs was used, since the PLACE role is written the same
          way no matter which participant role the verb assigns.
        8 ("transfer_who"): a GIVE-family transfer event (give/hand/pass
          only -- "took" has no RECIPIENT, so a "who has X" question after
          a bare TAKE would have no ground truth). The question asks WHO
          now HAS the object (the RECIPIENT) -- this requires the model to
          route the question's queried role (RECIPIENT, not the PLACE every
          earlier level used) to the memory slot the statement actually
          wrote (:func:`nsm_ct.clause_reactor._queried_role`).

    Every context sentence is one of :data:`TRANSFER_TEMPLATES` (parser-
    verified by :func:`verify_transfer_templates`); object nouns are drawn
    from :data:`_TRANSFER_OBJECTS`. Deterministic given ``seed``, like every
    other generator in this module.
    """

    def __init__(self, num_options: int = 4, seed: int = 0) -> None:
        self.num_options = num_options
        self.rng = random.Random(seed)

    def _mc_places(self, answer: str):
        pool = [p for p in _PLACES if p != answer]
        distractors = self.rng.sample(pool, min(self.num_options - 1, len(pool)))
        options = distractors + [answer]
        self.rng.shuffle(options)
        return options, options.index(answer)

    def _mc_names(self, answer: str):
        # Deliberately does NOT exclude the other participant (e.g. the
        # giver, for a "who has X" question): including it as a candidate
        # distractor is the real test -- a model that just answers "a name
        # mentioned in the sentence" gets this wrong roughly half the time.
        pool = [n for n in _NAMES if n != answer]
        distractors = self.rng.sample(pool, min(self.num_options - 1, len(pool)))
        options = distractors + [answer]
        self.rng.shuffle(options)
        return options, options.index(answer)

    def _sample_participants(self):
        giver, receiver = self.rng.sample(_NAMES, 2)
        obj = self.rng.choice(_TRANSFER_OBJECTS)
        place = self.rng.choice(_PLACES)
        return giver, receiver, obj, place

    def _level7(self) -> Episode:
        """Where did the transferred object end up? Any of the 4 verbs."""
        giver, receiver, obj, place = self._sample_participants()
        verb = self.rng.choice(list(TRANSFER_TEMPLATES))
        sent = TRANSFER_TEMPLATES[verb].format(
            giver=giver, receiver=receiver, obj=obj, place=place,
            taker=giver, source=receiver)
        options, idx = self._mc_places(place)
        return Episode(
            context=[sent], question=f"where is the {obj} ?",
            answer_text=place, options=options, answer_idx=idx, level=7,
            meta={"src": "curriculum2", "kind": "transfer_place", "verb": verb},
        )

    def _level8(self) -> Episode:
        """Who now has the transferred object? GIVE-family only (needs a
        RECIPIENT; a bare TAKE has none)."""
        giver, receiver, obj, place = self._sample_participants()
        verb = self.rng.choice(["GIVE", "HAND", "PASS"])
        sent = TRANSFER_TEMPLATES[verb].format(giver=giver, receiver=receiver, obj=obj, place=place)
        options, idx = self._mc_names(receiver)
        return Episode(
            context=[sent], question=f"who has the {obj} ?",
            answer_text=receiver, options=options, answer_idx=idx, level=8,
            meta={"src": "curriculum2", "kind": "transfer_who", "verb": verb},
        )

    def generate(self, n: int) -> List[Episode]:
        builders = [self._level7, self._level8]
        return [builders[i % len(builders)]() for i in range(n)]


def generate_transfer_episodes(n: int, seed: int = 0, num_options: int = 4) -> List[Episode]:
    """``n`` M52 multi-arg transfer episodes, deterministic given ``(n, seed,
    num_options)``. See :class:`TransferCurriculumGenerator`."""
    return TransferCurriculumGenerator(num_options=num_options, seed=seed).generate(n)
