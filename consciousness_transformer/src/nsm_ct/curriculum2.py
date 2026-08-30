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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .episode import Episode, _NAMES, _PLACES, _AMBIGUITY_FAMILIES  # noqa: F401
from .membrane import NAME_GENDER  # noqa: F401  (M53a pronoun curriculum genders)

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


# ---------------------------------------------------------------------------
# M53a -- pronoun-binding curriculum, ANTI-RECENCY BY DESIGN.
#
# dev/MIND_INTERFACE.md's v1 "entity" IN row + dev/RESOLVER_BUILD_PLAN.md
# Phase 2: a pronoun's true antecedent must NOT be recoverable by "pick the
# most recently introduced entity" alone, or the curriculum would let a
# resolver (or a non-resolver) win by a shortcut that has nothing to do with
# coreference. Every episode: two people go to two DIFFERENT places -- one
# with a female name, one with a male name (episode.py's fixed 6-name pool;
# genders are nsm_ct.membrane.NAME_GENDER, the same hand-specified,
# transparent, closed-class table the mention feature vectors use) -- so
# GENDER alone disambiguates which one a "she"/"he" refers to, independent
# of which was introduced first. A :class:`PronounCurriculumGenerator`
# instance alternates, call by call, whether the antecedent is the
# FIRST-introduced (older -> anti-recency) or SECOND-introduced (recent)
# person, which guarantees >=50% anti-recency for ANY ``n`` and ANY
# ``seed`` (the alternation is a counter, not an RNG draw).
#
# The pronoun sentence ("she found the ball .") is a NEW template shape (not
# covered by TEMPLATES/verify_templates above) parser-verified below by
# :func:`verify_pronoun_templates`, mirroring :func:`verify_transfer_templates`'s
# contract: every role must resolve to the exact expected token, never just
# "some clause came out".
# ---------------------------------------------------------------------------
_FEMALE_NAMES: List[str] = [n for n, g in NAME_GENDER.items() if g == "F"]
# LANDMINE FOUND (M53a build, NOT fixed here -- quantum_parser/ is out of
# scope for this task): quantum_parser.pos_tagger.simple_tag()'s
# unconditional "ends in -ed -> VERB" suffix heuristic misfires on "fred"
# (absent from both WORD_TAG_DICT and the WordNet-derived lexicon, so it
# falls through to that heuristic) -- confirmed empirically: "fred is in
# the garden ." parses to an EMPTY clause (extract_discourse returns no
# SUBJECT/PLACE at all; build_clause_batch silently drops the whole context
# step). This is a PRE-EXISTING defect: episode.py's fixed _NAMES pool
# already includes "fred", so ANY level 1-13 episode that happens to sample
# "fred" as a context subject is ALREADY silently degraded this way, with
# or without M53a. Not fixed here; excluded from this generator's own name
# pool only, because the anti-recency design depends on BOTH people's PLACE
# facts parsing correctly (a dropped step would desync the entity registry
# from ep.meta's ``registry_order``/``gold_antecedent`` bookkeeping). See
# RESEARCH_NOTES.md M53a entry.
_MALE_NAMES: List[str] = [n for n, g in NAME_GENDER.items() if g == "M" and n != "fred"]

# The pronoun-mention template. Context sentences reuse TEMPLATES["B"]["MOVE"][0]
# verbatim ("{n} went to the {p} .", already verify_templates()-covered above);
# this is the one genuinely NEW sentence shape M53a introduces.
PRONOUN_FIND_TEMPLATE = "{pronoun} found the {obj} ."
_PRONOUN_CONTEXT_TEMPLATE = TEMPLATES["B"]["MOVE"][0]
assert _PRONOUN_CONTEXT_TEMPLATE == "{n} went to the {p} ."


def verify_pronoun_templates(sample_name: str = "mary", sample_place: str = "garden",
                              sample_obj: str = "ball",
                              sample_pronouns: Tuple[str, ...] = ("she", "he", "it", "they"),
                              ) -> Dict[str, Dict[str, object]]:
    """Parse-check the pronoun curriculum's sentence shapes through the REAL
    parser (the ``verify_*`` pattern :func:`verify_templates` /
    :func:`verify_transfer_templates` already establish): the context
    sentence must yield SUBJECT/PLACE exactly matching the sample, and the
    pronoun sentence must yield SUBJECT=pronoun, OBJECT=sample_obj exactly.
    ``sample_pronouns`` defaults to all four candidates the design doc
    mentions (she/he/it/they) so the table also documents which ones parse
    cleanly even though the generator below only USES she/he (see its
    docstring for why it/they are reported-not-used). Returns an empty dict
    if ``quantum_parser`` isn't importable (caller must treat that as
    "unable to verify", not a pass) -- same contract as the other verify_*
    functions in this module.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    ctx_sent = _PRONOUN_CONTEXT_TEMPLATE.format(n=sample_name, p=sample_place)
    pronoun_sents = {pr: PRONOUN_FIND_TEMPLATE.format(pronoun=pr, obj=sample_obj)
                      for pr in sample_pronouns}
    texts = [ctx_sent, *pronoun_sents.values(), sample_name, sample_place, sample_obj]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        return {}  # quantum_parser unavailable; caller must handle this case

    results: Dict[str, Dict[str, object]] = {}

    graph = parser._parse_graph(ctx_sent)
    clauses, _links = extract_discourse(graph)
    ok = any((arg.token or "").lower() == sample_name for rel, arg in
             (ra for cl in clauses for ra in cl.args) if rel == "SUBJECT") and \
        any((arg.token or "").lower() == sample_place for rel, arg in
            (ra for cl in clauses for ra in cl.args) if rel == "PLACE")
    results["CONTEXT_MOVE"] = {"sentence": ctx_sent, "ok": ok}

    for pr, sent in pronoun_sents.items():
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        ok, roles = False, {}
        for cl in clauses:
            r = {rel: (arg.token or "").lower() for rel, arg in cl.args}
            if r.get("SUBJECT") == pr and r.get("OBJECT") == sample_obj:
                ok, roles = True, r
                break
            if len(r) > len(roles):
                roles = r
        results[f"PRONOUN_FIND[{pr}]"] = {"sentence": sent, "ok": ok, "roles": roles}
    return results


class PronounCurriculumGenerator:
    """M53a pronoun-binding episodes: anti-recency by design.

    Episode shape:
        "{name_a} went to the {place_a} ." "{name_b} went to the {place_b} ."
        "{pronoun} found the {obj} ." Q: "where is the {obj} ?"

    ``pronoun`` is "she" or "he"; its gender picks out exactly ONE of
    name_a/name_b (one female name, one male name -- genders from
    :data:`nsm_ct.membrane.NAME_GENDER`), so the gold answer is that
    person's place -- genuinely requires resolving the pronoun, not just
    reading off the most recent fact. Whether the antecedent is name_a
    (older, introduced first -- the ANTI-RECENCY case) or name_b (more
    recent) alternates by an internal counter every call, guaranteeing
    exactly-or-more-than 50% anti-recency for any ``n``.

    it/they were evaluated (:func:`verify_pronoun_templates` includes them;
    both parse cleanly -- "it"/"they found the ball ." both yield a clean
    SUBJECT/OBJECT split) but are NOT generated here: "it" needs an OBJECT
    antecedent registry (a different candidate-set population than the
    person registry this level exercises) and "they" needs a genuine
    multi-person-antecedent design (a coordinated-subject registry) -- both
    real, both natural M53b-or-later extensions, both out of scope for the
    single-focused anti-recency gender battery M53a's gate asks for.

    Gold antecedent + the data the nearest-entity baseline needs (see
    :func:`nearest_entity_baseline`) are recorded in ``ep.meta`` --
    ``gold_antecedent``, ``gold_place``, ``other_entity``, ``other_place``,
    ``pronoun``, ``pronoun_sentence_index`` (which context sentence carries
    the pronoun -- :mod:`nsm_ct.clause_reactor`'s batch path keys off this),
    and ``antecedent_recency`` ("old" | "recent", the anti-recency
    accounting label).
    """

    def __init__(self, num_options: int = 4, seed: int = 0, *,
                 female_names: Optional[List[str]] = None,
                 male_names: Optional[List[str]] = None) -> None:
        """``female_names``/``male_names`` (M56b, RESEARCH_NOTES M56/M56b)
        override the module-level :data:`_FEMALE_NAMES`/:data:`_MALE_NAMES`
        pools this generator samples antecedents from -- default ``None``
        reproduces every pre-M56b call exactly (same globals, same rng
        draws). This is what makes the held-out-name ablation possible: a
        caller passes a TRAIN pool (all names minus one held-out
        female/male) to one generator and the held-out singleton(s) to a
        second, so `scripts/train_resolver.py`'s ablation runner can build
        train-name and held-out-name episode sets from the SAME template
        machinery with no leak (see tests/test_curriculum2_name_split.py)."""
        self.num_options = num_options
        self.rng = random.Random(seed)
        self._count = 0
        self._female_names = list(female_names) if female_names is not None else _FEMALE_NAMES
        self._male_names = list(male_names) if male_names is not None else _MALE_NAMES

    def _mc_places(self, answer: str, required: List[str]):
        opts = list(dict.fromkeys([answer, *required]))
        pool = [p for p in _PLACES if p not in opts]
        while len(opts) < self.num_options and pool:
            opts.append(pool.pop(self.rng.randrange(len(pool))))
        opts = opts[: self.num_options]
        self.rng.shuffle(opts)
        return opts, opts.index(answer)

    def _episode(self) -> Episode:
        antecedent_first = (self._count % 2 == 0)   # alternate -> exact >=50% anti-recency
        self._count += 1

        gender = self.rng.choice(["F", "M"])
        pronoun = "she" if gender == "F" else "he"
        female_name = self.rng.choice(self._female_names)
        male_name = self.rng.choice(self._male_names)
        antecedent_name = female_name if gender == "F" else male_name
        other_name = male_name if gender == "F" else female_name
        name_a, name_b = ((antecedent_name, other_name) if antecedent_first
                          else (other_name, antecedent_name))

        place_a, place_b = self.rng.sample(_PLACES, 2)
        gold_place = place_a if antecedent_first else place_b
        other_place = place_b if antecedent_first else place_a
        obj = self.rng.choice(_TRANSFER_OBJECTS)

        context = [
            _PRONOUN_CONTEXT_TEMPLATE.format(n=name_a, p=place_a),
            _PRONOUN_CONTEXT_TEMPLATE.format(n=name_b, p=place_b),
            PRONOUN_FIND_TEMPLATE.format(pronoun=pronoun, obj=obj),
        ]
        options, idx = self._mc_places(gold_place, [other_place])
        return Episode(
            context=context, question=f"where is the {obj} ?",
            answer_text=gold_place, options=options, answer_idx=idx, level=14,
            meta={
                "src": "curriculum2", "kind": "pronoun_binding",
                "gold_antecedent": antecedent_name, "gold_place": gold_place,
                "other_entity": other_name, "other_place": other_place,
                "pronoun": pronoun, "pronoun_sentence_index": 2,
                "antecedent_recency": "old" if antecedent_first else "recent",
                "registry_order": [name_a, name_b],
            },
        )

    def generate(self, n: int) -> List[Episode]:
        return [self._episode() for _ in range(n)]


def generate_pronoun_episodes(n: int, seed: int = 0, num_options: int = 4, *,
                               female_names: Optional[List[str]] = None,
                               male_names: Optional[List[str]] = None) -> List[Episode]:
    """``n`` M53a pronoun-binding episodes, deterministic given ``(n, seed,
    num_options, female_names, male_names)``. See
    :class:`PronounCurriculumGenerator`; ``female_names``/``male_names`` are
    the M56b held-out-name-ablation override (default ``None`` = pre-M56b
    behavior, the fixed module-level name pools)."""
    return PronounCurriculumGenerator(num_options=num_options, seed=seed,
                                       female_names=female_names, male_names=male_names).generate(n)


def nearest_entity_baseline(episodes: List[Episode]) -> Dict[str, float]:
    """The scripted NEAREST-ENTITY coreference baseline
    (dev/RESOLVER_BUILD_PLAN.md Phase 2's data-design check): resolve every
    pronoun to the MOST RECENTLY introduced entity (name_b -- the second
    "went to" sentence), predict that entity's place, score against the
    episode's actual (gender-resolved) gold answer. Reads ``ep.meta``
    directly (no parser needed -- the curriculum already knows the
    registry order and the recency label).

    By design (see :class:`PronounCurriculumGenerator`): on the
    "antecedent_recency": "recent" half this baseline is trivially correct
    (it names the entity that happens to BE the antecedent); on the "old"
    half it is trivially WRONG (``other_place != gold_place`` always,
    places are sampled distinct) -- i.e. the anti-recency half's accuracy
    must be exactly 0.0, and the overall accuracy must equal the "recent"
    fraction (<=50% is guaranteed by the >=50% anti-recency alternation).
    Non-pronoun episodes are ignored (not part of this baseline's question).
    """
    total = anti = correct = anti_correct = 0
    for ep in episodes:
        if ep.meta.get("kind") != "pronoun_binding":
            continue
        recency = ep.meta["antecedent_recency"]
        # nearest (most-recently-introduced) entity's place:
        nearest_place = ep.meta["gold_place"] if recency == "recent" else ep.meta["other_place"]
        hit = nearest_place == ep.answer_text
        total += 1
        correct += int(hit)
        if recency == "old":
            anti += 1
            anti_correct += int(hit)
    return {
        "n": total,
        "accuracy": correct / total if total else float("nan"),
        "n_anti_recency": anti,
        "anti_recency_accuracy": anti_correct / anti if anti else float("nan"),
    }


# ---------------------------------------------------------------------------
# M57b -- resolver-driven write-BACK curriculum (CLAUDE.md's M57
# memory-schema decision). :class:`PronounCurriculumGenerator` above tests
# whether the collapsed resolver choice can redirect a WRITE's VALUE ("she
# found the ball ." -> WHERE is the ball, resolved from the antecedent's own
# place). This generator tests the opposite direction: whether the
# collapsed choice can redirect the write's ADDRESS -- "she is tall ." must
# write to the referent's OWN node (mary's), not to a fixed pronoun-mention
# placeholder, or a later question naming mary directly ("what is mary
# like ?") reads back nothing.
#
# v2 REDESIGN (fixes a full-scale leak the v1 shape had -- see
# RESEARCH_NOTES M57b): v1 always asked about the pronoun's TRUE referent
# ("target_name == antecedent_name", always), so the pronoun sentence's own
# stated attribute was ALWAYS the answer -- a GRU could carry that value from
# the pronoun step straight through to the question step and answer
# correctly with NO binding at all (measured, full scale: a CHEAT arm with
# candidate sets stripped entirely, and a WRONG-BINDING arm with only the
# resolver's aux-loss target corrupted, BOTH scored 1.000 on write-back
# questions). v2 closes this two ways at once:
#
#   1. Every entity now gets its OWN named attribute statement FIRST
#      ("mary is kind .", "john is strong ."), and the pronoun statement
#      comes LAST ("she is tall .") -- its write OVERWRITES the referent's
#      earlier named attribute at the SAME (entity, rel:ATTR) address (the
#      reactor's own learned overwrite gate does this; nothing here
#      special-cases it -- see :func:`nsm_ct.clause_reactor._writeback_steps`
#      below). The referent's earlier (now-stale) value is retained in
#      ``ep.meta["stale_attr"]`` and is ALWAYS one of the multiple-choice
#      options -- it is the signature wrong answer a failed redirect (or a
#      GRU merely carrying the pronoun step's value forward without binding)
#      would produce.
#   2. The question no longer always asks about the referent: ``target_name``
#      is sampled UNIFORMLY over BOTH entities. When it names the referent,
#      the answer is the PRONOUN's overwrite value (answerable only via a
#      correct address redirect); when it names the other entity, the
#      answer is that entity's OWN (never-overwritten) named attribute -- a
#      control condition requiring no redirect at all, so the SAME question
#      template can't be answered by a single fixed "always echo the
#      pronoun's value" shortcut.
#
# Same anti-recency-style discipline as PronounCurriculumGenerator: gender
# picks which of name_a/name_b the pronoun refers to (an internal counter
# alternates which slot -- first or second introduced -- is the antecedent,
# so recency alone can never fully predict the answer either).
#
# Design law (CLAUDE.md: "the gold-determinant must not itself be
# answer-predictive"): TWO features select the READING here, and NEITHER may
# predict the ANSWER VALUE --
#   - GENDER (which name the pronoun picks out as the referent);
#   - target selection (``question_targets_referent`` -- whether the
#     question asks about the referent or the other entity).
# The three attribute values in play (``stale_attr``/``other_attr``/
# ``pronoun_attr``) are drawn by ``self.rng.sample(_ATTR_VALUES, 3)`` AFTER
# gender/antecedent and target are both chosen, from the SAME fixed word
# pool, WITHOUT ever consulting gender, name identity, antecedent-recency, or
# target selection. This makes the joint draw a literal product distribution
# by construction: P(answer | gender) == P(answer | question_targets_referent)
# == P(answer) for every value in the pool, since the RNG call that picks the
# attribute triple never branches on either determinant at all -- zero
# mutual information, not an approximation (verified empirically in
# tests/test_writeback.py's determinant/answer independence check, since
# "the code never reads the value" is easy to state but the test is what
# actually catches a future edit that breaks this).
#
# Validity machinery (replacing v1's ``wrong_binding`` aux-gold-corruption
# arm -- see :class:`nsm_ct.clause_reactor.ClauseBatch`'s
# ``cand_forced_index`` field, :meth:`nsm_ct.clause_reactor.ClauseReactor.
# _collapse`'s teacher-forcing paragraph, and
# ``scripts/train_writeback.py``'s ``--force-binding {gold,wrong}``): this
# generator carries NO wrong-binding flag of its own anymore --
# ``ep.meta["gold_antecedent"]`` is ALWAYS the true referent. Forcing the
# collapse to the wrong candidate now happens at COLLAPSE time (a
# teacher-forced one-hot weight, independent of the resolver's own logits
# AND of the curriculum's gold_index), not by lying to the resolver's own
# supervision target -- task pressure simply overrode that corrupted target
# (measured: v1's wrong_binding arm ALSO scored 1.000 -- corrupting only the
# aux loss's target was insufficient; only forcing the COLLAPSE ITSELF
# closes the gap honestly).
#
# No parser dependency (:func:`nsm_ct.clause_reactor._writeback_steps`
# builds every triple straight from ``ep.meta``, like the L9-11 reasoning
# stream) -- ``context``/``question`` strings here exist for
# readability/provenance only, mirroring how :func:`_reasoning_steps` treats
# its own episodes' text. See that function's docstring for why: an
# attribute VALUE is a genuine adjective, and quantum_parser's tagger is
# noun-lexicon-biased (curriculum2.py's own "fred" landmine note above), so
# betting a new sentence shape's parse reliability against the one thing
# THIS milestone is meant to test would be a needless risk.
# ---------------------------------------------------------------------------
_ATTR_VALUES: List[str] = ["tall", "quiet", "clever", "brave", "curious", "gentle"]


class WriteBackCurriculumGenerator:
    """M57b write-back episodes (v2): resolver-driven ADDRESS redirection.

    Episode shape:
        "{name_a} went to the {place_a} ."
        "{name_b} went to the {place_b} ."
        "{antecedent_name} is {stale_attr} ."   -- referent's OWN named attribute
        "{other_name} is {other_attr} ."        -- other entity's OWN (persisting) named attribute
        "{pronoun} is {pronoun_attr} ."         -- the write-back collapse step: OVERWRITES
                                                     the referent's rel:ATTR slot
        Q: "what is {target_name} like ?"        (target_name sampled uniformly
                                                     over {antecedent_name, other_name})

    Answer key: if ``target_name`` is the referent, the answer is
    ``pronoun_attr`` (answerable only via a correct address redirect +
    overwrite); otherwise it's ``other_attr`` (the other entity's own,
    never-overwritten attribute -- a redirect-free control condition). See
    the module docstring above for the full v2 leak-fix rationale and its
    design-law argument.

    ``female_names``/``male_names`` (default ``None`` = the module pools)
    mirror :class:`PronounCurriculumGenerator`'s own held-out-name-ablation
    override for the same reason (a future M57-family ablation may want
    train/held-out name splits without touching this generator's internals).
    """

    def __init__(self, num_options: int = 4, seed: int = 0, *,
                 female_names: Optional[List[str]] = None,
                 male_names: Optional[List[str]] = None) -> None:
        self.num_options = num_options
        self.rng = random.Random(seed)
        self._count = 0
        self._female_names = list(female_names) if female_names is not None else _FEMALE_NAMES
        self._male_names = list(male_names) if male_names is not None else _MALE_NAMES

    def _mc_attrs(self, answer: str, required: List[str]):
        opts = list(dict.fromkeys([answer, *required]))
        pool = [a for a in _ATTR_VALUES if a not in opts]
        while len(opts) < self.num_options and pool:
            opts.append(pool.pop(self.rng.randrange(len(pool))))
        opts = opts[: self.num_options]
        self.rng.shuffle(opts)
        return opts, opts.index(answer)

    def _episode(self) -> Episode:
        antecedent_first = (self._count % 2 == 0)   # alternate -> exact >=50% anti-recency
        self._count += 1

        gender = self.rng.choice(["F", "M"])
        pronoun = "she" if gender == "F" else "he"
        female_name = self.rng.choice(self._female_names)
        male_name = self.rng.choice(self._male_names)
        antecedent_name = female_name if gender == "F" else male_name
        other_name = male_name if gender == "F" else female_name
        name_a, name_b = ((antecedent_name, other_name) if antecedent_first
                          else (other_name, antecedent_name))

        place_a, place_b = self.rng.sample(_PLACES, 2)

        # Three distinct attribute draws, independent of gender/antecedent-
        # recency/target-selection -- see the module note above for the
        # zero-MI-by-construction argument. stale_attr: the referent's own
        # named (soon overwritten) attribute; other_attr: the other entity's
        # own (persisting) named attribute; pronoun_attr: the write-back
        # overwrite value.
        stale_attr, other_attr, pronoun_attr = self.rng.sample(_ATTR_VALUES, 3)

        # Which entity the question asks about -- sampled UNIFORMLY,
        # independent of gender/attribute draws (the second half of the
        # design law: not only WHO the pronoun refers to, but WHICH entity
        # gets asked about, must carry zero information about the answer).
        target_name = self.rng.choice([antecedent_name, other_name])
        targets_referent = target_name == antecedent_name
        answer = pronoun_attr if targets_referent else other_attr

        context = [
            f"{name_a} went to the {place_a} .",
            f"{name_b} went to the {place_b} .",
            f"{antecedent_name} is {stale_attr} .",
            f"{other_name} is {other_attr} .",
            f"{pronoun} is {pronoun_attr} .",
        ]
        options, idx = self._mc_attrs(answer, [stale_attr, other_attr, pronoun_attr])
        return Episode(
            context=context, question=f"what is {target_name} like ?",
            answer_text=answer, options=options, answer_idx=idx, level=15,
            meta={
                "src": "curriculum2", "kind": "writeback",
                "name_a": name_a, "name_b": name_b,
                "place_a": place_a, "place_b": place_b,
                "pronoun": pronoun,
                "gold_antecedent": antecedent_name,   # ALWAYS the true referent (v2: no wrong_binding flag)
                "true_antecedent": antecedent_name,
                "other_entity": other_name,
                "stale_attr": stale_attr,
                "other_attr": other_attr,
                "pronoun_attr": pronoun_attr,
                "target_name": target_name,
                "question_targets_referent": targets_referent,
                "pronoun_sentence_index": 4,
                "antecedent_recency": "old" if antecedent_first else "recent",
                "registry_order": [name_a, name_b],
            },
        )

    def generate(self, n: int) -> List[Episode]:
        return [self._episode() for _ in range(n)]


def generate_writeback_episodes(n: int, seed: int = 0, num_options: int = 4, *,
                                 female_names: Optional[List[str]] = None,
                                 male_names: Optional[List[str]] = None) -> List[Episode]:
    """``n`` M57b write-back episodes, deterministic given ``(n, seed,
    num_options, female_names, male_names)``. See
    :class:`WriteBackCurriculumGenerator`."""
    return WriteBackCurriculumGenerator(num_options=num_options, seed=seed,
                                         female_names=female_names, male_names=male_names).generate(n)


# ---------------------------------------------------------------------------
# M57c -- instance atoms + definite-description referring expressions
# (dev/MIND_INTERFACE.md's v2 addendum, CLAUDE.md's M57 memory-schema
# decision): the two-Marys curriculum.
#
# World, every episode: THREE person instances, roles fixed "a"/"b"/"c"
# (matching nsm_ct.clause_reactor._instance_steps's own bookkeeping exactly,
# so this module -- torch/codec-free by its own constraint -- never needs
# to touch an instance id or atom; it only ever names ROLES). "a" and "b"
# share ONE surface name (``shared_name`` -- the two-Marys premise) but
# differ in KIND (occupation), which is always unique per instance by
# construction -- a definite description ("the doctor") is therefore
# ALWAYS unambiguous, the property this whole curriculum leans on. "c"
# carries a DISTINCT name and kind. Gender is sampled independently, with
# a/b sharing gender in EXACTLY half of episodes (alternated by episode
# count parity, the same deterministic-50%-not-statistical-50% discipline
# WriteBackCurriculumGenerator's ``antecedent_first`` already uses) --
# "kind always disambiguates, gender sometimes does" (the milestone spec's
# own phrase): a pronoun's evidence (attr:gender) is therefore sometimes a
# clean single-candidate match and sometimes genuinely tied, while a
# definite description's evidence (attr:kind) never ties.
#
# Context, in order: every instance is introduced with its own kind +
# gender + named-place fact (three ordinary attribute writes, addressed
# directly by :func:`nsm_ct.clause_reactor._instance_steps` via that
# instance's own minted atom -- no ambiguity in the WRITE itself, even
# though the SURFACE TEXT below may repeat a name); then every instance
# gets ONE baseline TRAIT statement (the "stale" value); then ONE more
# statement -- the OVERWRITE -- referring to exactly one instance (the
# "referent") via a referring expression instead of a unique name, and
# OVERWRITING that instance's trait slot (the M57b overwrite shape, scaled
# from two entities to three, same reactor-owned gate, no special-casing).
#
# Referring devices (sampled uniformly per episode):
#   - "definite_description" ("the {kind} is {attr} .") -- ALWAYS
#     eligible, referent may be any of a/b/c.
#   - "pronoun" ("she/he is {attr} .") -- referent may be any of a/b/c;
#     evidence is attr:gender, genuinely tied when a/b share gender AND
#     the referent's gender also matches c's (the harder, honest case,
#     left in on purpose rather than filtered out).
#   - "ambiguous_name" ("{shared_name} is {attr} .") -- referent restricted
#     to {a, b} (c's name is never ambiguous). The gold referent is
#     resolved by DISCOURSE RECENCY: the generator places whichever of a/b
#     is the referent LAST among the three baseline statements (temporally
#     closest to the overwrite) -- the SAME recency convention this
#     codebase already uses for pronoun antecedents elsewhere
#     (WriteBackCurriculumGenerator's antecedent_first/antecedent_recency).
#     This is a legitimate discourse phenomenon (real ambiguous repeated
#     names really do get disambiguated this way), not a surface-form leak
#     of the FINAL ANSWER: answering correctly still requires the write to
#     land on the right node and the question to read the right node
#     afterward (CLAUDE.md's design law is about the answer-bearing
#     statement's surface identifiability, not about what legitimately
#     determines a referent -- see nsm_ct.clause_reactor._instance_steps's
#     own docstring for the fuller argument).
#
# Question ("target" mode, ``inverse_frac`` fraction excluded -- see
# below): "what is {referring expression} like ?", TARGET sampled
# uniformly over a/b/c, INDEPENDENT of which instance is the referent
# (mirrors WriteBackCurriculumGenerator's target_name independence: not
# only WHO the referring expression names, but WHICH instance the
# QUESTION asks about, must carry zero information about the answer on
# its own). The referring expression is always UNAMBIGUOUS -- c's own
# unique name, or a definite description for a/b -- but a/b's definite-
# description question ALSO carries the same candidate/evidence-relation
# machinery (nsm_ct.clause_reactor._instance_steps's question branch),
# exercising the identical resolution mechanism at READ time that the
# overwrite step exercises at WRITE time -- the unification
# dev/MIND_INTERFACE.md's addendum promises. Answer: the OVERWRITE value
# if target == referent (answerable only via a correct redirect +
# overwrite, then a correct redirect + read), else the target's OWN
# baseline value (the redirect-free control condition). ``stale_attr``
# (the referent's own pre-overwrite baseline) is always among the MC
# options, mirroring WriteBackCurriculumGenerator exactly.
#
# INVERSE-QUERY episodes ("who is {trait} ?", ``inverse_frac`` of the
# mix): a DIFFERENT, deliberately simpler mechanism -- no candidate/
# evidence-relation machinery at all, a pure generation-plus-contrastive
# task (see _instance_steps's own docstring for why: the resolver
# contract has no clean slot to carry an arbitrary trait word into the
# collapse decision at the question step itself). ``query_trait`` is
# sampled from the three CURRENTLY-HELD trait values only (the referent's
# NEW overwrite value, or one of the other two instances' still-standing
# baselines) -- the referent's own STALE value is excluded, since nobody
# currently holds it after the overwrite (every inverse-query episode is
# therefore always answerable). Options are IDENTITY vectors, one per
# instance (a fixed 3, independent of ``num_options`` -- a deliberate
# simplification, see :func:`nsm_ct.clause_reactor._instance_option_vec`):
# c's own unique name, or "{shared_name} the {kind}" for a/b (the kind
# suffix is what keeps the two Marys' OPTIONS distinct meaning-vectors,
# mirroring how their own instance ATOMS are kept distinct on the memory
# side -- the same identity-fusion fix, now on the answer side too).
# ---------------------------------------------------------------------------
_KIND_VALUES: List[str] = ["doctor", "teacher", "nurse", "pilot", "chef"]


class InstanceCurriculumGenerator:
    """M57c instance-atom episodes (the two-Marys curriculum). See the
    module comment immediately above for the full design.

    ``inverse_frac`` (default 0.3): the fraction of episodes generated as
    inverse-query ("who is {trait} ?") rather than target-question ("what
    is {referring expression} like ?") -- a single generator with a mixing
    knob, per the milestone spec's "generate as a mixable fraction".
    ``names``/``kinds``/``traits`` (default ``None`` = the module pools)
    mirror :class:`WriteBackCurriculumGenerator`'s own override pattern.
    """

    def __init__(self, num_options: int = 4, seed: int = 0, *,
                 inverse_frac: float = 0.3,
                 names: Optional[List[str]] = None,
                 kinds: Optional[List[str]] = None,
                 traits: Optional[List[str]] = None) -> None:
        self.num_options = num_options
        self.inverse_frac = inverse_frac
        self.rng = random.Random(seed)
        self._count = 0
        self._names = list(names) if names is not None else list(_NAMES)
        self._kinds = list(kinds) if kinds is not None else list(_KIND_VALUES)
        self._traits = list(traits) if traits is not None else list(_ATTR_VALUES)

    def _mc_attrs(self, answer: str, required: List[str]):
        opts = list(dict.fromkeys([answer, *required]))
        pool = [a for a in self._traits if a not in opts]
        while len(opts) < self.num_options and pool:
            opts.append(pool.pop(self.rng.randrange(len(pool))))
        opts = opts[: self.num_options]
        self.rng.shuffle(opts)
        return opts, opts.index(answer)

    def _episode(self) -> Episode:
        instance_seed = self._count
        self._count += 1

        shared_name, distinct_name = self.rng.sample(self._names, 2)
        kind_a, kind_b, kind_c = self.rng.sample(self._kinds, 3)

        # Gender: a/b share gender in EXACTLY half of episodes (parity
        # alternation, not an RNG draw -- see module comment above);
        # c's gender is independent.
        same_gender_pair = (instance_seed % 2 == 0)
        gender_a = self.rng.choice(["F", "M"])
        gender_b = gender_a if same_gender_pair else ("M" if gender_a == "F" else "F")
        gender_c = self.rng.choice(["F", "M"])

        place_a, place_b, place_c = self.rng.sample(_PLACES, 3)
        baseline_a, baseline_b, baseline_c, overwrite_attr = self.rng.sample(self._traits, 4)

        device = self.rng.choice(["definite_description", "pronoun", "ambiguous_name"])
        referent = self.rng.choice(["a", "b"] if device == "ambiguous_name" else ["a", "b", "c"])

        # role -> (name, kind, gender, place, baseline_trait)
        roles = {
            "a": (shared_name, kind_a, gender_a, place_a, baseline_a),
            "b": (shared_name, kind_b, gender_b, place_b, baseline_b),
            "c": (distinct_name, kind_c, gender_c, place_c, baseline_c),
        }
        stale_attr = roles[referent][4]
        registry_order = [f"inst:{shared_name}#1", f"inst:{shared_name}#2", f"inst:{distinct_name}#1"]
        gold_instance_id = {"a": registry_order[0], "b": registry_order[1], "c": registry_order[2]}[referent]

        meta: Dict[str, object] = {
            "src": "curriculum2", "kind": "instance",
            "instance_seed": instance_seed,
            "shared_name": shared_name, "distinct_name": distinct_name,
            "kind_a": kind_a, "kind_b": kind_b, "kind_c": kind_c,
            "gender_a": gender_a, "gender_b": gender_b, "gender_c": gender_c,
            "place_a": place_a, "place_b": place_b, "place_c": place_c,
            "baseline_a": baseline_a, "baseline_b": baseline_b, "baseline_c": baseline_c,
            "overwrite_attr": overwrite_attr,
            "referring_device": device,
            "referent_role": referent,
            "stale_attr": stale_attr,
            "registry_order": registry_order,
            "gold_instance_id": gold_instance_id,
            "same_gender_pair": same_gender_pair,
        }

        # Human-readable context text ONLY (curriculum2/_instance_steps is
        # parser-free -- see that function's docstring -- these strings are
        # never parsed, provenance/debugging only).
        context: List[str] = []
        for r in ("a", "b", "c"):
            name, kind, gender, place, _b = roles[r]
            context.append(f"{name} is a {kind} .")
            context.append(f"{name} is {'female' if gender == 'F' else 'male'} .")
            context.append(f"{name} went to the {place} .")
        baseline_order = ["a", "b", "c"]
        if device == "ambiguous_name":
            baseline_order = [r for r in baseline_order if r != referent] + [referent]
        for r in baseline_order:
            name, kind, _g, _p, baseline = roles[r]
            ref_text = f"the {kind}" if r in ("a", "b") else name
            context.append(f"{ref_text} is {baseline} .")
        if device == "pronoun":
            mention = "she" if roles[referent][2] == "F" else "he"
        elif device == "definite_description":
            mention = f"the {roles[referent][1]}"
        else:
            mention = shared_name
        context.append(f"{mention} is {overwrite_attr} .")

        if self.rng.random() < self.inverse_frac:
            # Currently-held trait values only (the referent's own STALE
            # baseline was just overwritten-away -- nobody holds it, so
            # it is never asked about; see module comment above).
            currently_held = [("overwrite", referent, overwrite_attr)]
            currently_held += [("baseline", r, roles[r][4]) for r in ("a", "b", "c") if r != referent]
            _src, answer_role, query_trait = self.rng.choice(currently_held)

            def _render(r: str) -> str:
                name, kind = roles[r][0], roles[r][1]
                return f"{name} the {kind}" if r in ("a", "b") else name

            options = [_render(r) for r in ("a", "b", "c")]
            answer = _render(answer_role)
            answer_idx = options.index(answer)
            question = f"who is {query_trait} ?"
            meta["question_mode"] = "inverse"
            meta["query_trait"] = query_trait
            meta["answer_role"] = answer_role
        else:
            target = self.rng.choice(["a", "b", "c"])
            targets_referent = target == referent
            answer = overwrite_attr if targets_referent else roles[target][4]
            options, answer_idx = self._mc_attrs(answer, [baseline_a, baseline_b, baseline_c, overwrite_attr])
            target_ref_text = f"the {roles[target][1]}" if target in ("a", "b") else roles[target][0]
            question = f"what is {target_ref_text} like ?"
            meta["question_mode"] = "target"
            meta["target_role"] = target
            meta["question_targets_referent"] = targets_referent

        return Episode(
            context=context, question=question, answer_text=answer,
            options=options, answer_idx=answer_idx, level=16, meta=meta,
        )

    def generate(self, n: int) -> List[Episode]:
        return [self._episode() for _ in range(n)]


def generate_instance_episodes(n: int, seed: int = 0, num_options: int = 4, *,
                                inverse_frac: float = 0.3,
                                names: Optional[List[str]] = None,
                                kinds: Optional[List[str]] = None,
                                traits: Optional[List[str]] = None) -> List[Episode]:
    """``n`` M57c instance-atom episodes, deterministic given ``(n, seed,
    num_options, inverse_frac, names, kinds, traits)``. See
    :class:`InstanceCurriculumGenerator`."""
    return InstanceCurriculumGenerator(num_options=num_options, seed=seed, inverse_frac=inverse_frac,
                                        names=names, kinds=kinds, traits=traits).generate(n)


# ---------------------------------------------------------------------------
# RICH-EPISODE curriculum (the "stop requiring minimal episodes" priority,
# CLAUDE.md's 2026-08-30 reprioritization / dev/AURORA_SPRINT.md): a direct
# generalization of M57c's two-Marys world (InstanceCurriculumGenerator,
# fixed 3 entities / 1 overwrite statement) to N entities (3-8, sampled per
# episode) and K referring/overwrite statements (1-4, sampled per episode),
# with a richer attribute vocabulary (three DISTINCT attribute RELATIONS --
# trait/mood/size -- not just one). The reactor-side generalization is
# :func:`nsm_ct.clause_reactor._rich_steps`, a NEW function sibling to
# :func:`nsm_ct.clause_reactor._instance_steps` rather than a refactor of it
# -- ``_instance_steps`` is UNCHANGED (still exclusively serves
# ``ep.meta["kind"] == "instance"``), matching this codebase's own
# established convention (M54/M55a/M57b/M57c each added their own dedicated
# ``_xxx_steps`` function rather than editing an earlier, already-proven
# one). Every :class:`~nsm_ct.membrane.EntityCandidateSet` mechanism this
# needs -- addr_redirect, evidence_relation, forced_index, the entity-axis
# inverse read -- is REUSED unchanged; no new ``ClauseBatch`` field is
# needed at all (``cand_entity``'s ``Cmax`` padding dimension already
# handles a variable per-(row, step) candidate-set size generically).
#
# World, every episode: N person instances, roles are plain integer indices
# 0..N-1 (curriculum2 stays torch/codec-free -- see instances.py's own
# module docstring -- so, exactly like InstanceCurriculumGenerator, this
# module only ever names ROLES; :func:`nsm_ct.clause_reactor._rich_steps`
# mints the real per-episode :class:`~nsm_ct.instances.InstanceRegistry`
# atoms). ``n_shared_name_pairs`` entities pair up sharing ONE surface name
# (the two-Marys premise, generalized to 0-2 pairs); every OTHER entity has
# its own name. KIND is always sampled WITHOUT REPLACEMENT across ALL N
# entities (never just within a name-sharing pair), so a definite
# description ("the {kind}") is ALWAYS globally unambiguous, exactly like
# InstanceCurriculumGenerator's kind_a/kind_b/kind_c. GENDER is independent
# per entity (F/M); a pronoun's evidence is therefore sometimes tied
# (several entities sharing a gender) and sometimes not, same "kind always
# disambiguates, gender sometimes does" texture M57c established -- but
# UNLIKE M57c's pronoun device (which leaves a genuine gender tie in the
# curriculum on purpose, "the harder case"), RICH's stricter honesty
# contract (see below) means a pronoun referring statement is only ever
# GENERATED when the referent's gender happens to be globally unique at
# that point -- a tie simply makes that referent/device combination
# ineligible for THIS statement, not an accepted ambiguous case.
#
# Each entity gets, besides kind/gender/a named place: 1-3 attribute facts
# over DISTINCT attribute RELATIONS drawn from a small pool
# (:data:`_RICH_ATTR_RELATIONS` -- trait/mood/size, each with its OWN value
# pool, :data:`_RICH_ATTR_VALUES`) -- this is the literal fix for the
# "just one value pool" limitation of WriteBack/InstanceCurriculumGenerator.
# Values are DISTINCT within a relation across the whole episode (no two
# entities ever hold the same trait value at the same time) -- enforced at
# assignment time and re-enforced on every overwrite, so an inverse query
# ("who is tall ?") is always unambiguous and a definite-description
# question's multiple-choice distractors are never accidentally correct
# for a different entity.
#
# K referring statements (1-4) each pick a DISTINCT (entity, relation) pair
# still held by that entity, an ELIGIBLE referring device
# (definite_description/pronoun/ambiguous_name), and a NEW value (distinct
# from every value currently held under that relation) that OVERWRITES the
# old one -- the M57b/M57c overwrite shape, scaled to K independent
# overwrites instead of one. HONESTY, non-negotiable (locked design):
#   1. The selecting evidence (kind for definite_description, gender for
#      pronoun, discourse recency for ambiguous_name) carries ZERO
#      information about the ANSWER -- it only ever determines WHICH
#      entity a referring expression names, a question wholly independent
#      of what attribute value that entity ends up holding (kind/gender
#      are sampled before any attribute value; recency is a fact about
#      discourse ORDER, not content).
#   2. Every referring statement's referent is UNIQUELY DETERMINED by the
#      evidence available at that point in the discourse --
#      construct-and-verify, not filter-and-hope: :func:`_verify_unique_referent`
#      (definite_description/pronoun) and :func:`_verify_recency_referent`
#      (ambiguous_name) are called at generation time and RAISE on a
#      genuine tie (see their own docstrings -- directly unit-testable by
#      constructing a collision by hand). Eligibility is pre-filtered
#      (only a referent with globally-unique gender is offered the
#      "pronoun" device at all) so these should never actually fire in
#      normal generation; they are the honesty contract's enforcement
#      mechanism, not decoration. Once a name-sharing pair is referred to
#      by ANY statement, BOTH members are excluded from every LATER
#      referring statement in the episode -- otherwise an intervening
#      mention could silently change which member is "more recent",
#      corrupting the very recency evidence an earlier ambiguous_name
#      statement already committed to.
#   3. The answer is never recoverable from surface form alone: multiple
#      referring statements over multiple relations, plus each entity's
#      own NAMED baseline statements over the SAME relations, means no
#      single sentence's surface shape predicts the question's answer (the
#      M55b/M57b/M57c "the answer-bearing statement must not be surface-
#      identifiable" law).
#
# The question (ONE per episode, the reactor's own per-episode contract --
# see :func:`nsm_ct.clause_reactor._rich_steps`'s docstring for why this
# means ``ClauseBatch.inverse_mask``'s "at most one inverse step per
# episode" seam is automatically preserved): target entity sampled
# UNIFORMLY over all N, target relation sampled uniformly over that
# entity's OWN held relations, phrased with an UNAMBIGUOUS referring
# expression (the entity's own unique name if it isn't name-sharing, else
# "the {kind}" -- never a pronoun, mirroring InstanceCurriculumGenerator's
# question phrasing exactly). Answer = the CURRENT (post-overwrite) value;
# the STALE (pre-overwrite) value is always among the 4 MC options when the
# targeted slot was in fact overwritten -- the signature wrong answer, same
# convention as every earlier M57-family generator.
#
# An INVERSE-QUERY variant ("who is {value} ?") is a mixable fraction
# (``inverse_frac``, default 0.0 -- opt-in, unlike ``question_mode`` for
# InstanceCurriculumGenerator which defaults 0.3): a value CURRENTLY held by
# some entity (post-overwrite state only -- an overwritten-away stale value
# is never asked about, so every inverse episode is always answerable) is
# sampled uniformly over every (entity, relation, value) triple in the
# final state; the 4 MC options (``num_options``, NOT "one option per
# entity" -- N may exceed 4) are the answering entity plus 3 distractors
# sampled uniformly from the rest, rendered as unambiguous identity strings
# ("{name} the {kind}" for a name-sharing entity, else the bare name) and
# grounded downstream as the REAL per-episode instance atoms themselves
# (:func:`nsm_ct.clause_reactor.build_clause_batch`'s M57c.2 inverse-option
# grounding, extended one branch for ``kind == "rich"``).
#
# DEVIATION FROM A LITERAL "n_shared_name_pairs in [0, 2], n_entities in
# [3, 8]" (recorded here, not silently reconciled): this module's fixed
# name pool (``episode._NAMES``, 6 names) cannot supply 8 fully-distinct
# entity names, so at ``n_entities == 8`` exactly 2 shared pairs are
# REQUIRED (not merely allowed) to fit within the pool; at 7 entities at
# least 1 pair is required. The generator computes this feasible range
# itself (raising ``ValueError`` only for a caller-supplied name pool too
# small for the requested ``max_entities``); the default configuration
# (entities 3-8, pairs 0-2, the 6-name pool) is feasible for every sampled
# ``n_entities`` without ever hitting the error path.
# ---------------------------------------------------------------------------
_RICH_KIND_VALUES: List[str] = [
    "doctor", "teacher", "nurse", "pilot", "chef", "farmer", "artist",
    "judge", "clerk", "diver",
]
_RICH_ATTR_RELATIONS: List[str] = ["trait", "mood", "size"]
_RICH_ATTR_VALUES: Dict[str, List[str]] = {
    "trait": ["tall", "quiet", "clever", "brave", "curious", "gentle",
              "witty", "calm", "patient", "honest", "humble", "stubborn"],
    "mood": ["happy", "anxious", "cheerful", "grumpy", "eager", "serene",
             "nervous", "proud", "restless", "hopeful", "weary", "content"],
    "size": ["large", "small", "tiny", "huge", "narrow", "wide",
              "compact", "broad", "slender", "bulky", "thin", "massive"],
}


def _verify_unique_referent(device: str, referent: int, evidence: Dict[int, object]) -> None:
    """Assert ``referent`` is the UNIQUE entity (among every key in
    ``evidence``, e.g. every entity's kind for ``"definite_description"``,
    every entity's gender for ``"pronoun"``) sharing ``evidence[referent]``'s
    value -- the RICH curriculum's honesty contract, non-negotiable (see
    the module comment above): every referring statement's referent must be
    uniquely resolvable from the evidence available in the discourse, never
    left as an accepted tie the way InstanceCurriculumGenerator's pronoun
    device deliberately allows. Raises ``AssertionError`` on a genuine
    collision -- directly testable by handing this a rigged ``evidence``
    dict with two entities sharing a value.
    """
    val = evidence[referent]
    matches = [i for i, v in evidence.items() if v == val]
    assert matches == [referent], (
        f"{device} evidence is not unique for referent {referent}: "
        f"ties with {[m for m in matches if m != referent]}")


def _verify_recency_referent(referent: int, partner: int, mention_order: Dict[int, int]) -> None:
    """Assert ``referent``'s discourse position (``mention_order`` -- higher
    = more recent) is STRICTLY more recent than ``partner``'s -- the
    recency evidence an ``"ambiguous_name"`` referring statement relies on
    to disambiguate a repeated name. Raises ``AssertionError`` otherwise;
    directly testable with a rigged ``mention_order`` dict."""
    assert mention_order[referent] > mention_order[partner], (
        f"ambiguous_name referent {referent} (order {mention_order[referent]}) is not "
        f"more recent than partner {partner} (order {mention_order[partner]})")


def _predict_registry_ids(entity_order: List[int], names: List[str]) -> Dict[int, str]:
    """The ``inst:<name>#<n>`` id :class:`nsm_ct.instances.InstanceRegistry`
    will mint for each entity when :func:`nsm_ct.clause_reactor._rich_steps`
    mints them in ``entity_order`` sequence -- computed here, in this
    torch/codec-free module, by literally reproducing
    :meth:`~nsm_ct.instances.InstanceRegistry.mint`'s id-format convention
    (lowercase name, 1-based count of PRIOR mints of that same name),
    exactly like :class:`InstanceCurriculumGenerator` predicts its own
    fixed 3-role ``registry_order`` by hand. Returns ``{entity_index: id}``.
    """
    counts: Dict[str, int] = {}
    ids: Dict[int, str] = {}
    for i in entity_order:
        key = names[i].lower()
        counts[key] = counts.get(key, 0) + 1
        ids[i] = f"inst:{key}#{counts[key]}"
    return ids


class RichEpisodeGenerator:
    """RICH-EPISODE curriculum: N entities (3-8), K referring/overwrite
    statements (1-4), a small pool of DISTINCT attribute relations
    (trait/mood/size). See the module comment immediately above for the
    full design and its honesty machinery.

    Args:
        min_entities, max_entities: per-episode entity-count range.
        max_shared_name_pairs: upper bound on name-sharing pairs (0-2
            default); the actual per-episode count is clamped up when the
            name pool can't otherwise fit ``n_entities`` distinct names
            (see the module comment's "DEVIATION" note).
        min_referring, max_referring: per-episode referring/overwrite
            statement count range (K).
        min_attrs, max_attrs: per-entity distinct-attribute-relation count
            range (clamped to ``len(attr_relations)``).
        inverse_frac: fraction of episodes generated as an inverse-query
            ("who is {value} ?") rather than a target question -- default
            ``0.0`` (opt-in; InstanceCurriculumGenerator's own default is
            0.3, kept different here since RICH is the newer, less-proven
            curriculum -- callers/scripts choose their own mix).
        names/kinds/attr_relations/attr_values: pool overrides, mirroring
            every earlier M57-family generator's override pattern.
    """

    def __init__(self, num_options: int = 4, seed: int = 0, *,
                 min_entities: int = 3, max_entities: int = 8,
                 max_shared_name_pairs: int = 2,
                 min_referring: int = 1, max_referring: int = 4,
                 min_attrs: int = 1, max_attrs: int = 3,
                 inverse_frac: float = 0.0,
                 plural_frac: float = 0.0,
                 names: Optional[List[str]] = None,
                 kinds: Optional[List[str]] = None,
                 attr_relations: Optional[List[str]] = None,
                 attr_values: Optional[Dict[str, List[str]]] = None) -> None:
        if max_entities < min_entities:
            raise ValueError(f"max_entities ({max_entities}) must be >= min_entities ({min_entities})")
        if max_referring < min_referring:
            raise ValueError(f"max_referring ({max_referring}) must be >= min_referring ({min_referring})")
        self.num_options = num_options
        self.rng = random.Random(seed)
        self._count = 0
        self.min_entities, self.max_entities = min_entities, max_entities
        self.max_shared_name_pairs = max_shared_name_pairs
        self.min_referring, self.max_referring = min_referring, max_referring
        self.min_attrs, self.max_attrs = min_attrs, max_attrs
        self.inverse_frac = inverse_frac
        # M57e (morphology signals, dev/AURORA_SPRINT.md's 2026-08-11
        # reprioritization): fraction of episodes that additionally mint a
        # PLURAL group referent (the DRT move, dev/MIND_INTERFACE.md's v2
        # addendum) from an explicit coordination sentence -- see the
        # "optional PLURAL group" block in :meth:`_episode` for the full
        # mechanism. Default ``0.0`` -- opt-in, like ``inverse_frac`` --
        # AND, critically, zero EXTRA rng draws at exactly 0.0 (the `and`
        # short-circuit there), so every episode this generator already
        # produced is byte-identical (tests/test_morphology.py's
        # plural_frac=0 regression checks this against a fixed seed).
        self.plural_frac = plural_frac
        self._names = list(names) if names is not None else list(_NAMES)
        self._kinds = list(kinds) if kinds is not None else list(_RICH_KIND_VALUES)
        self._attr_relations = list(attr_relations) if attr_relations is not None else list(_RICH_ATTR_RELATIONS)
        self._attr_values = ({k: list(v) for k, v in attr_values.items()} if attr_values is not None
                              else {k: list(v) for k, v in _RICH_ATTR_VALUES.items()})
        if self.max_entities > len(self._kinds):
            raise ValueError("max_entities must be <= len(kinds pool) (kind is always globally unique)")
        for rel in self._attr_relations:
            if len(self._attr_values.get(rel, [])) < self.max_entities:
                raise ValueError(f"attr_values[{rel!r}] pool must have >= max_entities distinct values")

    def _mc_attr_options(self, relation: str, answer: str, required: List[str]):
        opts = list(dict.fromkeys([answer, *required]))
        pool = [v for v in self._attr_values[relation] if v not in opts]
        while len(opts) < self.num_options and pool:
            opts.append(pool.pop(self.rng.randrange(len(pool))))
        opts = opts[: self.num_options]
        self.rng.shuffle(opts)
        return opts, opts.index(answer)

    def _episode(self) -> Episode:
        instance_seed = self._count
        self._count += 1
        rng = self.rng

        n = rng.randint(self.min_entities, self.max_entities)

        # -- names: n_shared_name_pairs clamped to what the fixed pool can
        # fit (see class/module "DEVIATION" note).
        pool_n = len(self._names)
        lo_pairs = max(0, n - pool_n)
        hi_pairs = min(self.max_shared_name_pairs, n // 2)
        if lo_pairs > hi_pairs:
            raise ValueError(f"cannot fit {n} entities in a {pool_n}-name pool with at most "
                              f"{self.max_shared_name_pairs} shared pairs")
        n_pairs = rng.randint(lo_pairs, hi_pairs)
        n_solo = n - 2 * n_pairs
        chosen = rng.sample(self._names, n_pairs + n_solo)
        shared_names, solo_names = chosen[:n_pairs], chosen[n_pairs:]
        names: List[str] = []
        for nm in shared_names:
            names += [nm, nm]
        names += solo_names
        name_groups: Dict[str, List[int]] = {}
        for i, nm in enumerate(names):
            if names.count(nm) == 2:
                name_groups.setdefault(nm, []).append(i)

        kinds = rng.sample(self._kinds, n)
        genders = [rng.choice(["F", "M"]) for _ in range(n)]
        gender_counts: Dict[str, int] = {}
        for g in genders:
            gender_counts[g] = gender_counts.get(g, 0) + 1
        places = [rng.choice(_PLACES) for _ in range(n)]

        held_relations: List[List[str]] = []
        for _ in range(n):
            k = rng.randint(min(self.min_attrs, len(self._attr_relations)),
                             min(self.max_attrs, len(self._attr_relations)))
            held_relations.append(rng.sample(self._attr_relations, k))

        current_value: List[Dict[str, str]] = [{} for _ in range(n)]
        relation_in_use: Dict[str, Set[str]] = {rel: set() for rel in self._attr_relations}
        for rel in self._attr_relations:
            holders = [i for i in range(n) if rel in held_relations[i]]
            if not holders:
                continue
            vals = rng.sample(self._attr_values[rel], len(holders))
            for i, v in zip(holders, vals):
                current_value[i][rel] = v
                relation_in_use[rel].add(v)
        initial_values = [dict(dv) for dv in current_value]

        # -- discourse order for registration/attribute facts (also the
        # InstanceRegistry mint order -- see _rich_steps). Adjusted below,
        # pair by pair, so an ambiguous_name referent is provably the more
        # RECENT of its name-sharing pair.
        entity_order = list(range(n))
        rng.shuffle(entity_order)

        held_pairs = [(i, rel) for i in range(n) for rel in held_relations[i]]
        rng.shuffle(held_pairs)
        n_referring = min(rng.randint(self.min_referring, self.max_referring), len(held_pairs))

        used_name_groups: Set[str] = set()
        used_pairs: Set[Tuple[int, str]] = set()
        statements: List[dict] = []
        for referent, rel in held_pairs:
            if len(statements) >= n_referring:
                break
            group_name = names[referent]
            group = name_groups.get(group_name)
            if group is not None and group_name in used_name_groups:
                continue   # group already touched -- protects a committed recency reading
            eligible_devices = ["definite_description"]
            if gender_counts[genders[referent]] == 1:
                eligible_devices.append("pronoun")
            if group is not None:
                eligible_devices.append("ambiguous_name")
            device = rng.choice(eligible_devices)

            if device == "definite_description":
                _verify_unique_referent(device, referent, dict(enumerate(kinds)))
                mention_word = kinds[referent]
            elif device == "pronoun":
                _verify_unique_referent(device, referent, dict(enumerate(genders)))
                mention_word = "she" if genders[referent] == "F" else "he"
            else:
                partner = next(j for j in group if j != referent)
                if entity_order.index(referent) < entity_order.index(partner):
                    pi, pj = entity_order.index(referent), entity_order.index(partner)
                    entity_order[pi], entity_order[pj] = entity_order[pj], entity_order[pi]
                _verify_recency_referent(referent, partner,
                                          {e: pos for pos, e in enumerate(entity_order)})
                mention_word = group_name

            candidates_new = [v for v in self._attr_values[rel] if v not in relation_in_use[rel]]
            if not candidates_new:
                continue   # relation's value pool exhausted by holders -- try the next pair
            new_value = rng.choice(candidates_new)
            stale_value = current_value[referent][rel]
            relation_in_use[rel].discard(stale_value)
            relation_in_use[rel].add(new_value)
            current_value[referent][rel] = new_value

            statements.append({
                "referent": referent, "relation": rel, "device": device,
                "stale_value": stale_value, "new_value": new_value,
                "mention_word": mention_word,
            })
            used_pairs.add((referent, rel))
            if group is not None:
                used_name_groups.add(group_name)

        assert statements, "RichEpisodeGenerator produced zero referring statements"
        final_values = current_value

        registry_ids = _predict_registry_ids(entity_order, names)
        registry_order = [registry_ids[i] for i in entity_order]
        for stmt in statements:
            stmt["gold_instance_id"] = registry_ids[stmt["referent"]]

        # -- optional PLURAL group (M57e, dev/AURORA_SPRINT.md's 2026-08-11
        # reprioritization "morphological signals (number/gender subtypes)
        # flowing parser->membrane->memory attributes"): an explicit
        # coordination sentence ("A and B went to the park .") mints a NEW
        # group discourse referent (the DRT move, dev/MIND_INTERFACE.md's v2
        # addendum) holding attr:number=pl plus one attr:member fact per
        # member (nsm_ct.clause_reactor._rich_steps mints it and writes
        # those facts -- see that function), followed by a plural-pronoun
        # overwrite statement ("they are {value} .") whose gold referent is
        # the group -- resolved via NUMBER evidence (attr:number) instead of
        # kind/gender, mirroring every K referring statement's own
        # evidence-relation mechanism exactly. ``self.plural_frac > 0 and
        # rng.random() < ...`` short-circuits: at the default ``0.0`` no rng
        # draw happens AT ALL, so every existing draw sequence is
        # byte-identical (tests/test_morphology.py's plural_frac=0
        # regression checks this).
        has_group = self.plural_frac > 0 and n >= 2 and rng.random() < self.plural_frac
        group_members: Optional[Tuple[int, int]] = None
        group_relation: Optional[str] = None
        group_value: Optional[str] = None
        group_instance_id: Optional[str] = None
        if has_group:
            # Members must have DISTINCT surface names -- "john and john
            # went to the park ." is nonsense prose (and would silently
            # reuse a name-sharing pair's own identity-disambiguation
            # premise for an unrelated purpose). Skip the group entirely if
            # every pair in this episode happens to share a name (only
            # possible at n==2 with the two sharing a name, since
            # name-sharing pairs are LIMITED TO 2 members each).
            distinct_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if names[i] != names[j]]
            if not distinct_pairs:
                has_group = False
        if has_group:
            m0, m1 = rng.choice(distinct_pairs)
            group_relation = rng.choice(self._attr_relations)
            candidates_new = [v for v in self._attr_values[group_relation]
                               if v not in relation_in_use[group_relation]]
            if not candidates_new:
                has_group = False   # relation pool exhausted -- skip, mirrors the K-statement "continue"
            else:
                group_members = (m0, m1)
                group_value = rng.choice(candidates_new)
                relation_in_use[group_relation].add(group_value)
                # _rich_steps mints the group AFTER every individual (see its
                # own comment) -- always its first (only) mint, so this is a
                # closed-form prediction, exactly like _predict_registry_ids
                # above predicts the individuals' ids.
                group_instance_id = "inst:group#1"
                _verify_unique_referent(
                    "plural_pronoun", "group",
                    {**{i: "sg" for i in range(n)}, "group": "pl"})

        # Human-readable context text ONLY -- like InstanceCurriculumGenerator,
        # this module is parser-free by design; _rich_steps grounds every
        # fact directly via minted atoms, never by parsing these strings.
        context: List[str] = []
        for i in entity_order:
            context.append(f"{names[i]} is a {kinds[i]} .")
            context.append(f"{names[i]} is {'female' if genders[i] == 'F' else 'male'} .")
            context.append(f"{names[i]} went to the {places[i]} .")
            for rel in held_relations[i]:
                context.append(f"{names[i]} is {initial_values[i][rel]} .")
        for stmt in statements:
            referent = stmt["referent"]
            if stmt["device"] == "pronoun":
                mention = stmt["mention_word"]
            elif stmt["device"] == "definite_description":
                mention = f"the {stmt['mention_word']}"
            else:
                mention = stmt["mention_word"]
            context.append(f"{mention} is {stmt['new_value']} .")
        if has_group:
            m0, m1 = group_members
            context.append(f"{names[m0]} and {names[m1]} went to the park .")
            context.append(f"they are {group_value} .")

        meta: Dict[str, object] = {
            "src": "curriculum2", "kind": "rich",
            "instance_seed": instance_seed,
            "n_entities": n,
            "n_shared_name_pairs": n_pairs,
            "entity_order": list(entity_order),
            "names": list(names),
            "kinds": list(kinds),
            "genders": list(genders),
            "places": list(places),
            "held_relations": [list(r) for r in held_relations],
            "initial_values": initial_values,
            "final_values": [dict(dv) for dv in final_values],
            "name_groups": {k: list(v) for k, v in name_groups.items()},
            "registry_order": registry_order,
            "referring_statements": statements,
            "n_referring_statements": len(statements),
            "has_group": has_group,
            "group_members": list(group_members) if group_members is not None else None,
            "group_relation": group_relation,
            "group_value": group_value,
            "group_instance_id": group_instance_id,
        }

        def _render(i: int) -> str:
            return f"{names[i]} the {kinds[i]}" if names[i] in name_groups else names[i]

        if rng.random() < self.inverse_frac:
            triples = [(i, rel, v) for i in range(n) for rel, v in final_values[i].items()]
            answer_entity, query_relation, query_value = rng.choice(triples)
            n_opts = min(self.num_options, n)
            others = [j for j in range(n) if j != answer_entity]
            distractors = rng.sample(others, n_opts - 1)
            option_entities = distractors + [answer_entity]
            rng.shuffle(option_entities)
            options = [_render(i) for i in option_entities]
            answer = _render(answer_entity)
            answer_idx = option_entities.index(answer_entity)
            question = f"who is {query_value} ?"
            meta["question_mode"] = "inverse"
            meta["query_value"] = query_value
            meta["query_relation"] = query_relation
            meta["answer_entity"] = answer_entity
            meta["answer_instance_id"] = registry_ids[answer_entity]
            meta["option_entities"] = option_entities
            meta["inverse_option_ids"] = [registry_ids[i] for i in option_entities]
        else:
            # M57e: a "target"-mode question MAY ask about the plural group
            # instead of a single entity ("what are A and B like ?") --
            # gated the same way (has_group and rng.random() < 0.5) so a
            # non-group episode, or plural_frac == 0.0, never draws this rng
            # call at all. The group has no prior value (its ONE statement
            # is the first and only write to it), so there is never a stale
            # option here -- unlike an individual target, which may or may
            # not have been overwritten.
            target_is_group = has_group and rng.random() < 0.5
            if target_is_group:
                target = None
                target_relation = group_relation
                answer = group_value
                overwritten = False
                stale = None
                options, answer_idx = self._mc_attr_options(target_relation, answer, [answer])
                m0, m1 = group_members
                question = f"what are {names[m0]} and {names[m1]} like ?"
                question_device = "plural_pronoun"
                target_instance_id = group_instance_id
            else:
                target = rng.choice(range(n))
                target_relation = rng.choice(held_relations[target])
                answer = final_values[target][target_relation]
                overwritten = (target, target_relation) in used_pairs
                stale = initial_values[target].get(target_relation) if overwritten else None
                required = [answer] + ([stale] if stale is not None else [])
                options, answer_idx = self._mc_attr_options(target_relation, answer, required)
                is_solo = names[target] not in name_groups
                target_ref_text = names[target] if is_solo else f"the {kinds[target]}"
                question = f"what is {target_ref_text} like ?"
                overwriting_stmt = next((s for s in statements
                                          if s["referent"] == target and s["relation"] == target_relation), None)
                question_device = overwriting_stmt["device"] if overwriting_stmt else None
                target_instance_id = registry_ids[target]
            meta["question_mode"] = "target"
            meta["target_entity"] = target
            meta["target_relation"] = target_relation
            meta["target_instance_id"] = target_instance_id
            meta["question_targets_overwritten"] = overwritten
            meta["stale_value_for_question"] = stale
            meta["question_device"] = question_device
            meta["target_is_group"] = target_is_group

        return Episode(
            context=context, question=question, answer_text=answer,
            options=options, answer_idx=answer_idx, level=17, meta=meta,
        )

    def generate(self, n: int) -> List[Episode]:
        return [self._episode() for _ in range(n)]


def generate_rich_episodes(n: int, seed: int = 0, num_options: int = 4, *,
                            min_entities: int = 3, max_entities: int = 8,
                            max_shared_name_pairs: int = 2,
                            min_referring: int = 1, max_referring: int = 4,
                            min_attrs: int = 1, max_attrs: int = 3,
                            inverse_frac: float = 0.0,
                            plural_frac: float = 0.0,
                            names: Optional[List[str]] = None,
                            kinds: Optional[List[str]] = None,
                            attr_relations: Optional[List[str]] = None,
                            attr_values: Optional[Dict[str, List[str]]] = None) -> List[Episode]:
    """``n`` RICH-EPISODE curriculum episodes, deterministic given all
    keyword arguments. See :class:`RichEpisodeGenerator`."""
    return RichEpisodeGenerator(
        num_options=num_options, seed=seed, min_entities=min_entities, max_entities=max_entities,
        max_shared_name_pairs=max_shared_name_pairs, min_referring=min_referring, max_referring=max_referring,
        min_attrs=min_attrs, max_attrs=max_attrs, inverse_frac=inverse_frac, plural_frac=plural_frac,
        names=names, kinds=kinds,
        attr_relations=attr_relations, attr_values=attr_values).generate(n)


# ---------------------------------------------------------------------------
# M59b -- CROSS-PASSAGE document curriculum for episodic LTM (M59a). See
# src/nsm_ct/ltm.py's module docstring ("Interface contract for the
# curriculum agent") for the binding contract this generator's output must
# satisfy -- nothing here builds LTM, this is the SHAPE the curriculum has to
# hand back -- and dev/LTM_DESIGN_BRIEF.md Sec.5 for the locked LTM design
# this milestone tests.
#
# A DOCUMENT is an ORDERED list of PASSAGES: Episodes sharing one
# ``meta["doc_id"]``, each carrying its position via ``meta["passage_index"]``
# (0-based, contiguous). Every entity mentioned anywhere in a document is
# minted from the SAME :class:`~nsm_ct.instances.InstanceRegistry`,
# constructed once by the CALLER (curriculum2 stays torch/codec-free -- see
# every earlier generator's own module comment -- so this module only ever
# PREDICTS the ids :func:`nsm_ct.clause_reactor._document_steps` will mint,
# the same way :func:`_predict_registry_ids` already does for
# :class:`RichEpisodeGenerator`).
#
# Two-passage default (``n_passages``, configurable to 3):
#   passage 0 (REGISTRATION): N entities (``min_entities``-``max_entities``,
#     default 3-5), ALL DISTINCT NAMES (no internal name-sharing -- the
#     interesting collision this curriculum tests is ACROSS passages, not
#     within passage 0), each with kind/gender/place + 2-3 attribute facts
#     (the trait/mood/size relation pool :class:`RichEpisodeGenerator`
#     already uses) -- no referring statements, every entity addressed
#     directly by its own freshly-minted atom.
#   [passage 1, ONLY when ``n_passages == 3``]: a FILLER passage -- 1-2
#     unrelated distractor entities, registered the same way (kind/gender/
#     place only, no attribute facts, no candidate sets) -- purely to
#     exercise an EXTRA consolidation hop between the referent's
#     introduction and its cross-passage mention.
#   final passage (MENTION + QUESTION, index ``n_passages - 1``): re-mentions
#     ONE passage-0 entity (the "referent") by NAME, under one of two
#     conditions sampled 50/50:
#       - "same": the SAME referent -- no contradicting kind; an OPTIONAL
#         confirming description ("mary the doctor", ~50% of "same"
#         episodes).
#       - "new": a DIFFERENT person introduced under the SAME name, with a
#         kind that NEVER matches any kind already used in the document
#         ("a doctor named mary").
#     Exactly ONE :class:`~nsm_ct.membrane.EntityCandidateSet`
#     (``addr_redirect=True``) at the mention step: candidates = [the
#     referent's own LTM instance id (``from_ltm=1``), a freshly-minted NEW
#     instance id (``from_ltm=0``)] -- ALWAYS both, regardless of which
#     condition was realized (honesty invariant #1 below: the candidate
#     SET's shape never reveals the condition). The mention statement then
#     writes ONE attribute fact (relation R, a fresh value) onto whichever
#     candidate the gold link names; R is always a relation the REFERENT
#     already held in passage 0 (so a type-(iii) question is always
#     constructible).
#   Then ONE question, sampled among (availability-gated):
#     (i) a passage-0 fact about the referent under a DIFFERENT relation
#       than R (never touched this passage) -- answerable ONLY via the
#       additive LTM read (this passage's own STM starts at zero and never
#       states it).
#     (ii) relation R's CURRENT value on whoever the mention actually linked
#       to (the referent, "same"; the NEW instance, "new") -- tests that the
#       WRITE landed on the correct address, not just that recall works.
#       Options include the OTHER candidate's own R value as a "wrong
#       identity" distractor.
#     (iii) ("new" condition only) relation R's value on the ORIGINAL
#       referent, whom this passage's mention statement did NOT touch (it
#       wrote to the NEW instance's own address instead) -- options include
#       the NEW instance's just-written value as the competing distractor,
#       checking the referent's LTM slot was not clobbered.
#   Every question addresses its target entity DIRECTLY (via "the {kind}",
#   globally unique across passage 0 + the NEW instance by construction) --
#   never a second candidate set: the cross-passage LINK decision is tested
#   exactly once, at the mention step.
#
# HONESTY (non-negotiable, mirrors RichEpisodeGenerator's own contract):
#   1. Zero mutual information between CONDITION and the eventual ANSWER
#      value: attribute values are sampled the same way regardless of which
#      condition was drawn, so knowing the condition alone never predicts
#      the specific value -- only WHOSE slot it lands on.
#   2. The link decision is uniquely determined by the evidence actually
#      available to the mention: a "kind" description names a kind that
#      matches EXACTLY ONE of the two candidates' live ``attr:kind`` slot,
#      asserted at generation (see the assertion in :meth:`_episode_group`,
#      the same contract :func:`_verify_unique_referent` enforces for
#      :class:`RichEpisodeGenerator`). A BARE name mention ("same", no
#      description) carries NO kind evidence at all -- its
#      ``evidence_target`` grounds via the "name:" convention
#      (:func:`nsm_ct.clause_reactor._ground_evidence_target`), which by
#      construction does NOT correlate with either candidate's ``attr:kind``
#      readout, so the resolver must lean on ``cand_from_ltm``/recency
#      instead of a leaked kind signal -- documented here, not silently
#      smoothed over.
#   3. The answer is never recoverable from surface form alone: the mention
#      word is the SAME string ("mary") whether the condition is "same" or
#      "new"; only the (possibly absent) description and the live memory
#      state disambiguate.
# ---------------------------------------------------------------------------
def _predict_document_new_id(passage0_names: List[str], shared_name: str) -> str:
    """The instance id :meth:`nsm_ct.instances.InstanceRegistry.mint` will
    assign to the freshly-minted "NEW" candidate a document's mention step
    always mints (see the module comment above): passage 0 mints
    ``passage0_names`` in order, each name DISTINCT (this generator's own
    construction), so the mention's own mint is the SECOND mint of
    ``shared_name`` -- ``"inst:<shared_name>#2"`` -- deterministically,
    mirroring :func:`_predict_registry_ids`'s own id-format reproduction."""
    key = shared_name.lower()
    n = sum(1 for nm in passage0_names if nm.lower() == key) + 1
    return f"inst:{key}#{n}"


class DocumentGenerator:
    """M59b cross-passage document curriculum: episodic LTM's curriculum
    generator. See the module comment immediately above for the full design
    and its honesty machinery. ``generate(n)`` returns ``n`` DOCUMENTS'
    worth of episodes flattened into one list (``n * n_passages`` episodes
    total) -- the caller groups them back into documents by
    ``meta["doc_id"]``, sorted by ``meta["passage_index"]`` (see
    src/nsm_ct/ltm.py's module docstring; ``scripts/train_ltm.py`` is the
    reference caller).

    Args:
        n_passages: 2 (default) or 3 -- see the module comment's passage
            layout.
        min_entities, max_entities: passage-0 entity-count range (default
            3-5).
        min_attrs, max_attrs: per-entity distinct-attribute-relation count
            range (default 2-3 -- >= 2 so a type-(i) question, which needs a
            relation DIFFERENT from the mention's own R, is always
            constructible for the referent).
        names/kinds/attr_relations/attr_values: pool overrides, mirroring
            every earlier M57-family generator's override pattern.
    """

    def __init__(self, num_options: int = 4, seed: int = 0, *,
                 n_passages: int = 2,
                 min_entities: int = 3, max_entities: int = 5,
                 min_attrs: int = 2, max_attrs: int = 3,
                 names: Optional[List[str]] = None,
                 kinds: Optional[List[str]] = None,
                 attr_relations: Optional[List[str]] = None,
                 attr_values: Optional[Dict[str, List[str]]] = None) -> None:
        if n_passages not in (2, 3):
            raise ValueError(f"n_passages must be 2 or 3, got {n_passages}")
        if max_entities < min_entities:
            raise ValueError(f"max_entities ({max_entities}) must be >= min_entities ({min_entities})")
        if max_attrs < min_attrs:
            raise ValueError(f"max_attrs ({max_attrs}) must be >= min_attrs ({min_attrs})")
        self.num_options = num_options
        self.rng = random.Random(seed)
        self._count = 0
        self.n_passages = n_passages
        self.min_entities, self.max_entities = min_entities, max_entities
        self.min_attrs, self.max_attrs = min_attrs, max_attrs
        self._names = list(names) if names is not None else list(_NAMES)
        self._kinds = list(kinds) if kinds is not None else list(_RICH_KIND_VALUES)
        self._attr_relations = list(attr_relations) if attr_relations is not None else list(_RICH_ATTR_RELATIONS)
        self._attr_values = ({k: list(v) for k, v in attr_values.items()} if attr_values is not None
                              else {k: list(v) for k, v in _RICH_ATTR_VALUES.items()})
        # +3: passage-0 entities, the NEW candidate's own kind, and up to 2
        # filler entities (n_passages == 3) must ALL carry globally distinct
        # kinds (see module comment: "the {kind}" must stay unambiguous).
        if self.max_entities + 3 > len(self._kinds):
            raise ValueError("kinds pool too small for max_entities + the NEW candidate + filler entities "
                              f"(need > {self.max_entities + 3}, have {len(self._kinds)})")
        for rel in self._attr_relations:
            if len(self._attr_values.get(rel, [])) < self.max_entities:
                raise ValueError(f"attr_values[{rel!r}] pool must have >= max_entities distinct values")

    def _mc_attr_options(self, relation: str, answer: str, required: List[str]):
        opts = list(dict.fromkeys([answer, *required]))
        pool = [v for v in self._attr_values[relation] if v not in opts]
        while len(opts) < self.num_options and pool:
            opts.append(pool.pop(self.rng.randrange(len(pool))))
        opts = opts[: self.num_options]
        self.rng.shuffle(opts)
        return opts, opts.index(answer)

    def _episode_group(self) -> List[Episode]:
        rng = self.rng
        doc_seed = self._count
        self._count += 1
        doc_id = f"doc:{doc_seed}"

        n = rng.randint(self.min_entities, self.max_entities)
        names = rng.sample(self._names, n)                       # distinct within passage 0
        kinds = rng.sample(self._kinds, n)                       # globally unique so far
        genders = [rng.choice(["F", "M"]) for _ in range(n)]
        places = [rng.choice(_PLACES) for _ in range(n)]
        held_relations: List[List[str]] = [
            rng.sample(self._attr_relations,
                       rng.randint(min(self.min_attrs, len(self._attr_relations)),
                                   min(self.max_attrs, len(self._attr_relations))))
            for _ in range(n)
        ]
        current_value: List[Dict[str, str]] = [{} for _ in range(n)]
        relation_in_use: Dict[str, Set[str]] = {rel: set() for rel in self._attr_relations}
        for rel in self._attr_relations:
            holders = [i for i in range(n) if rel in held_relations[i]]
            if not holders:
                continue
            vals = rng.sample(self._attr_values[rel], len(holders))
            for i, v in zip(holders, vals):
                current_value[i][rel] = v
                relation_in_use[rel].add(v)
        initial_values = [dict(dv) for dv in current_value]

        entity_order = list(range(n))
        rng.shuffle(entity_order)
        registry_ids = _predict_registry_ids(entity_order, names)
        registry_order = [registry_ids[i] for i in entity_order]

        used_kinds: Set[str] = set(kinds)

        # -- optional filler passage (n_passages == 3 only): see module
        # comment. Predicts each filler entity's own registry id the same
        # way _predict_registry_ids does (each is the FIRST -- only -- mint
        # of its own name, since filler names never collide with passage-0's
        # or each other's).
        filler_entities: List[Dict[str, object]] = []
        if self.n_passages == 3:
            n_filler = rng.randint(1, 2)
            filler_name_pool = [nm for nm in self._names if nm not in names]
            filler_names = (rng.sample(filler_name_pool, min(n_filler, len(filler_name_pool)))
                             if filler_name_pool else [])
            filler_kind_pool = [k for k in self._kinds if k not in used_kinds]
            filler_kinds = rng.sample(filler_kind_pool, len(filler_names))
            used_kinds |= set(filler_kinds)
            for nm, kd in zip(filler_names, filler_kinds):
                filler_entities.append({
                    "id": f"inst:{nm.lower()}#1",
                    "name": nm, "kind": kd,
                    "gender": rng.choice(["F", "M"]), "place": rng.choice(_PLACES),
                })

        # -- the referent + mention (final passage).
        referent = rng.randrange(n)
        shared_name = names[referent]
        original_kind = kinds[referent]
        referent_instance_id = registry_ids[referent]
        mention_relation = rng.choice(held_relations[referent])
        other_relations = [r for r in held_relations[referent] if r != mention_relation]
        has_untouched = bool(other_relations)
        untouched_relation = rng.choice(other_relations) if has_untouched else None

        condition = "same" if rng.random() < 0.5 else "new"
        has_description = condition == "same" and rng.random() < 0.5

        new_kind: Optional[str] = None
        if condition == "new":
            new_kind_pool = [k for k in self._kinds if k not in used_kinds]
            new_kind = rng.choice(new_kind_pool)
            used_kinds.add(new_kind)

        value_before = initial_values[referent][mention_relation]
        pool_mention_value = [v for v in self._attr_values[mention_relation]
                               if v not in relation_in_use[mention_relation]]
        mention_new_value = rng.choice(pool_mention_value)

        new_id = _predict_document_new_id(names, shared_name)
        link_candidates = [referent_instance_id, new_id]
        gold_link = referent_instance_id if condition == "same" else new_id

        # Honesty invariant #2 (see module comment): when kind evidence IS
        # offered (a description, or the "new" condition), it must uniquely
        # pick out the gold link -- a genuine construct-and-verify assertion,
        # not decoration (mirrors _verify_unique_referent's own contract).
        if condition == "new" or has_description:
            evidence_kind = new_kind if condition == "new" else original_kind
            kind_of = {referent_instance_id: original_kind,
                       new_id: (new_kind if condition == "new" else None)}
            matches = [cid for cid, k in kind_of.items() if k == evidence_kind]
            assert matches == [gold_link], (
                f"document link evidence not unique for doc {doc_id}: {matches} vs gold {gold_link}")

        available_types = (["i"] if has_untouched else []) + ["ii"]
        if condition == "new" and has_untouched:
            available_types.append("iii")
        question_type = rng.choice(available_types)

        if question_type == "i":
            q_target_kind = original_kind
            answer = initial_values[referent][untouched_relation]
            required = [answer]
            question_relation = untouched_relation
        elif question_type == "ii":
            q_target_kind = original_kind if condition == "same" else new_kind
            answer = mention_new_value
            required = [answer, value_before]
            question_relation = mention_relation
        else:  # "iii"
            q_target_kind = original_kind
            answer = value_before
            required = [answer, mention_new_value]
            question_relation = mention_relation
        options, answer_idx = self._mc_attr_options(question_relation, answer, required)
        question = f"what is the {q_target_kind} like ?"

        base_meta: dict = {
            "src": "curriculum2", "kind": "document",
            "doc_id": doc_id, "n_passages": self.n_passages,
            "instance_seed": doc_seed,
            "n_entities": n, "entity_order": list(entity_order),
            "names": list(names), "kinds": list(kinds), "genders": list(genders), "places": list(places),
            "held_relations": [list(r) for r in held_relations],
            "initial_values": initial_values,
            "registry_order": registry_order,
            "filler_entities": filler_entities,
            "referent": referent, "shared_name": shared_name, "original_kind": original_kind,
            "referent_instance_id": referent_instance_id,
            "condition": condition, "has_description": has_description,
            "new_kind": new_kind,
            "mention_relation": mention_relation, "mention_new_value": mention_new_value,
            "value_before": value_before,
            "untouched_relation": untouched_relation,
            "link_candidates": link_candidates, "gold_link": gold_link,
            "question_type": question_type,
        }

        episodes: List[Episode] = []
        p0_context: List[str] = []
        for i in entity_order:
            p0_context.append(f"{names[i]} is a {kinds[i]} .")
            p0_context.append(f"{names[i]} is {'female' if genders[i] == 'F' else 'male'} .")
            p0_context.append(f"{names[i]} went to the {places[i]} .")
            for rel in held_relations[i]:
                p0_context.append(f"{names[i]} is {initial_values[i][rel]} .")
        episodes.append(Episode(
            context=p0_context, question="(registration passage; no question)",
            answer_text="idk", options=["idk"], answer_idx=0, level=18,
            meta=dict(base_meta, passage_index=0),
        ))

        if self.n_passages == 3:
            fp_context: List[str] = []
            for fe in filler_entities:
                fp_context.append(f"{fe['name']} is a {fe['kind']} .")
                fp_context.append(f"{fe['name']} is {'female' if fe['gender'] == 'F' else 'male'} .")
                fp_context.append(f"{fe['name']} went to the {fe['place']} .")
            episodes.append(Episode(
                context=fp_context, question="(filler passage; no question)",
                answer_text="idk", options=["idk"], answer_idx=0, level=18,
                meta=dict(base_meta, passage_index=1),
            ))

        pf_context: List[str] = []
        if condition == "new":
            pf_context.append(f"a {new_kind} named {shared_name} .")
        mention_text = f"{shared_name} the {original_kind}" if has_description else shared_name
        pf_context.append(f"{mention_text} is {mention_new_value} .")
        episodes.append(Episode(
            context=pf_context, question=question, answer_text=answer,
            options=options, answer_idx=answer_idx, level=18,
            meta=dict(base_meta, passage_index=self.n_passages - 1),
        ))

        return episodes

    def generate(self, n: int) -> List[Episode]:
        episodes: List[Episode] = []
        for _ in range(n):
            episodes += self._episode_group()
        return episodes


def generate_document_episodes(n: int, seed: int = 0, num_options: int = 4, *,
                                n_passages: int = 2,
                                min_entities: int = 3, max_entities: int = 5,
                                min_attrs: int = 2, max_attrs: int = 3,
                                names: Optional[List[str]] = None,
                                kinds: Optional[List[str]] = None,
                                attr_relations: Optional[List[str]] = None,
                                attr_values: Optional[Dict[str, List[str]]] = None) -> List[Episode]:
    """``n`` M59b DOCUMENTS' worth of episodes (``n * n_passages`` episodes
    total, flattened -- group by ``meta["doc_id"]`` to recover documents),
    deterministic given all keyword arguments. See :class:`DocumentGenerator`.
    """
    return DocumentGenerator(
        num_options=num_options, seed=seed, n_passages=n_passages,
        min_entities=min_entities, max_entities=max_entities,
        min_attrs=min_attrs, max_attrs=max_attrs,
        names=names, kinds=kinds, attr_relations=attr_relations, attr_values=attr_values).generate(n)


# ---------------------------------------------------------------------------
# M54b -- entity-keyed, binding-critical sense-ambiguity curriculum.
#
# RESEARCH_NOTES.md M54's finding: the M32/episode.py ambiguity curriculum
# (``episode.ambiguity_episode`` / ``generate_ambiguity_episodes``) LEAKS --
# MFS-floor task accuracy (0.910) sits almost at the gold ceiling (0.917),
# meaning the model never needs the SENSE at all to answer. This generator
# fixes that BY CONSTRUCTION, mirroring the M53a anti-recency discipline
# exactly: the disambiguating cue for the homograph lives in an EARLIER
# sentence keyed to a SPECIFIC entity ("mary went to the river ."), a
# DIFFERENT entity gets a decoy sentence supporting the WRONG sense's answer
# ("john went to the money ."), and only THEN does the homograph event happen
# ("mary sat on the bank ."). Both senses' associated content is present in
# EVERY episode, attached to different entities -- so a bag-of-words
# association over the whole episode is uninformative BY DESIGN (see
# :func:`association_only_baseline`, which must land at chance on generated
# data) and only correctly binding the homograph to the RIGHT entity's own
# prior fact -- via genuine entity-keyed memory, not a side-channel -- can
# discriminate. See ``nsm_ct.clause_reactor._ambiguity_steps``'s "M54b"
# addendum for the one clause_reactor.py change this needs (gated on
# ``ep.meta["kind"] == "sense_binding"``, so the M32 curriculum's own
# behavior is completely unaffected).
#
# Vocabulary: reuses episode.py's _AMBIGUITY_FAMILIES (senses/answers/
# anchors) as the closed-class word list -- NOT episode.py's own episode
# SHAPE, which is untouched by anything below.
#
# Two exhaustive, PARSER-MEASURED exclusions (found by literally parsing
# every anchor/answer combination through quantum_parser -- see
# :func:`verify_sense_binding_templates`, which reproduces this audit and
# reports it, exactly like :func:`verify_pronoun_templates` reports the
# fred landmine instead of silently working around it):
#
#   1. ``_SENSE_BINDING_EXCLUDE_FAMILY`` -- "pool"'s sense-A answer, "swim",
#      does not parse as a PLACE noun in "{name} went to the {p} ." (the
#      generic MOVE template this generator uses for BOTH the disambiguating
#      cue and the decoy -- the same parser-verified shape as
#      ``TEMPLATES["B"]["MOVE"][0]``). Every OTHER answer word across all 31
#      families' both senses (62 words total) parses cleanly. "pool" is
#      dropped entirely (not just its A sense) for simplicity.
#
#   2. ``_SENSE_BINDING_EXCLUDE_ANCHOR`` -- a (family, sense_key) that cannot
#      be the ANCHOR (the sense whose event actually happens, entity-keyed)
#      because its own anchor sentence fails one of two checks: (a) the
#      clause's SUBJECT isn't the {name} at all (three anchors are built
#      around a non-person subject -- "the doctor examined the pupil .",
#      "the bird had a long bill .", "the game ended in a tie ." -- pupil/B,
#      bill/B, tie/B), or (b) the SAME clause contains a second content word
#      besides SUBJECT/homograph -- e.g. "cave" in "{name} saw a bat in the
#      cave ." -- which is EXACTLY the M54 same-clause leak this curriculum
#      exists to avoid, so any anchor with that shape is disqualified as a
#      binding-critical ANCHOR outright (bat/A, plant/B [also a genuine
#      no-hit-clause parse miss], seal/A, seal/B, pupil/A, hood/A [no-hit-
#      clause], jam/A, jam/B, mole/A) plus the pre-existing "hammered"
#      landmine (nail/B, RESEARCH_NOTES M53a/M54 -- fred's sibling, not
#      fixed here, quantum_parser/ out of scope). A family with BOTH senses
#      excluded (seal, jam, pupil) is simply never used as the anchor side;
#      it remains fully available as the DECOY side (only the answer WORD is
#      used there, never the anchor sentence).
#
# Landmine avoided (mirrors M53a's fred exclusion exactly): "bill" is BOTH a
# curriculum NAME (episode.py's _NAMES) and an ambiguity-family HOMOGRAPH --
# picking "bill" as an entity name in a "bill"-family episode would make the
# SUBJECT token and the homograph token the SAME SURFACE WORD, an
# unnecessary and confusing self-collision. "bill" is excluded from this
# generator's own entity-name pool entirely (like fred is excluded from
# PronounCurriculumGenerator's), leaving 4 usable names.
# ---------------------------------------------------------------------------
_SENSE_BINDING_NAMES: List[str] = [n for n in _NAMES if n not in ("fred", "bill")]

_SENSE_BINDING_EXCLUDE_FAMILY = {"pool"}
_SENSE_BINDING_EXCLUDE_ANCHOR = {
    ("bat", "A"), ("plant", "B"), ("seal", "A"), ("seal", "B"),
    ("pupil", "A"), ("pupil", "B"), ("nail", "B"), ("hood", "A"),
    ("bill", "B"), ("tie", "B"), ("jam", "A"), ("jam", "B"), ("mole", "A"),
}

# The one new sentence shape this curriculum introduces: reuses
# TEMPLATES["B"]["MOVE"][0] verbatim (already verify_templates()-covered
# above) with a sense's ANSWER WORD standing in for the place noun -- the
# disambiguating cue AND the decoy are the exact same shape, just keyed to
# different entities and different sense answers.
_SENSE_BINDING_CUE_TEMPLATE = TEMPLATES["B"]["MOVE"][0]
assert _SENSE_BINDING_CUE_TEMPLATE == "{n} went to the {p} ."


def _sense_binding_eligible_pairs() -> List[Tuple[str, str, bool]]:
    """``[(family, gold_key, flipped)]`` for every (family, sense) that may
    be the ANCHOR (see module-level exclusions above). ``flipped`` is
    ``gold_sense != mfs_sense`` computed off the REAL synset ids (not an
    A/B-key shortcut -- a handful of families' ``mfs`` synset matches
    NEITHER curriculum sense, e.g. "seal"'s mfs is ``sealing_wax.n.01``,
    a real WordNet quirk episode.py's own module docstring already flags --
    those are always-flipped regardless of which key is chosen, which this
    computation gets right for free by comparing synset strings directly)."""
    pairs = []
    for word, fam in _AMBIGUITY_FAMILIES.items():
        if word in _SENSE_BINDING_EXCLUDE_FAMILY:
            continue
        for key in ("A", "B"):
            if (word, key) in _SENSE_BINDING_EXCLUDE_ANCHOR:
                continue
            flipped = fam["senses"][key]["synset"] != fam["mfs"]
            pairs.append((word, key, flipped))
    return pairs


_SENSE_BINDING_ELIGIBLE = _sense_binding_eligible_pairs()


def verify_sense_binding_templates(sample_gold_name: str = "mary",
                                    sample_other_name: str = "john"
                                    ) -> Dict[str, Dict[str, object]]:
    """Parse-check this curriculum's sentence shapes through the REAL parser
    (the ``verify_*`` pattern every other generator in this module follows):
    every eligible anchor's clause must have SUBJECT == the sample name and
    carry NO other content word besides the homograph (the same-clause-leak
    check M54 measured missing), and every family's BOTH answer words must
    parse as a clean SUBJECT/PLACE pair through
    :data:`_SENSE_BINDING_CUE_TEMPLATE`. Returns
    ``{check_name: {"sentence": str, "ok": bool, ...}}``. Returns an empty
    dict if ``quantum_parser`` isn't importable (caller must treat that as
    "unable to verify", not a pass) -- same contract as
    :func:`verify_templates` / :func:`verify_transfer_templates` /
    :func:`verify_pronoun_templates`.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    # Only answer words this generator can actually produce -- i.e. every
    # family EXCEPT _SENSE_BINDING_EXCLUDE_FAMILY ("pool"'s sense-A answer,
    # "swim", is the one word in the whole 62-word vocabulary that fails this
    # check -- feed the pre-exclusion set to see it fail explicitly).
    answers = sorted({fam["senses"][k]["answer"] for w, fam in _AMBIGUITY_FAMILIES.items()
                       for k in ("A", "B") if w not in _SENSE_BINDING_EXCLUDE_FAMILY})
    cue_texts = [_SENSE_BINDING_CUE_TEMPLATE.format(n=sample_gold_name, p=a) for a in answers]
    anchor_texts = {(w, k): _AMBIGUITY_FAMILIES[w]["senses"][k]["anchor"].format(name=sample_gold_name)
                    for (w, k, _f) in _SENSE_BINDING_ELIGIBLE}
    texts = cue_texts + list(anchor_texts.values()) + [sample_gold_name, sample_other_name] + answers
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        return {}  # quantum_parser unavailable; caller must handle this case

    results: Dict[str, Dict[str, object]] = {}
    for a in answers:
        sent = _SENSE_BINDING_CUE_TEMPLATE.format(n=sample_gold_name, p=a)
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        ok = any((arg.token or "").lower() == sample_gold_name for rel, arg in
                 (ra for cl in clauses for ra in cl.args) if rel == "SUBJECT") and \
            any((arg.token or "").lower() == a for rel, arg in
                (ra for cl in clauses for ra in cl.args) if rel == "PLACE")
        results[f"CUE[{a}]"] = {"sentence": sent, "ok": ok}

    for (w, k), sent in anchor_texts.items():
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        hit = None
        for cl in clauses:
            if any((arg.token or "").lower() == w for _rel, arg in cl.args):
                hit = cl
                break
        ok, roles, extra = False, {}, []
        if hit is not None:
            roles = {rel: (arg.token or "").lower() for rel, arg in hit.args}
            extra = [t for _rel, arg in hit.args for t in [(arg.token or "").lower()]
                     if t and t != w and t != sample_gold_name]
            ok = roles.get("SUBJECT") == sample_gold_name and not extra
        results[f"ANCHOR[{w}.{k}]"] = {"sentence": sent, "ok": ok, "roles": roles, "context_leak": extra}
    return results


class SenseBindingCurriculumGenerator:
    """M54b binding-critical ambiguity episodes: entity-keyed, decoy-balanced.

    Episode shape (order: cues first in random relative order, ANCHOR always
    last so its sense-binding step's memory READ sees both cues already
    written):

        "{gold_name} went to the {gold_answer} ."
        "{other_name} went to the {other_answer} ."
        "{gold_name} <anchor verb phrase with the homograph> ."
        Q: "what kind of {homograph} is it ?"
        options: [gold_answer, other_answer] (shuffled)

    ``gold_answer``/``other_answer`` are the homograph family's two senses'
    answer words (:data:`nsm_ct.episode._AMBIGUITY_FAMILIES`) -- e.g. for
    "bank": river (the A/river-bank sense) vs money (the B/financial-
    institution sense). Both are present in EVERY episode, each attached to
    a DIFFERENT entity, so a bag-of-words association over the whole episode
    is uninformative by construction (:func:`association_only_baseline`
    must land at chance on generated data) -- only correctly binding the
    homograph event to ``gold_name``'s own earlier fact discriminates.

    Flip balance mirrors :class:`PronounCurriculumGenerator`'s anti-recency
    discipline exactly: an internal COUNTER (not RNG) alternates wanting a
    "flipped" episode (``gold_sense != mfs_sense`` -- the half where always-
    guessing MFS is wrong) vs "unflipped", guaranteeing (as close to) exact
    50/50 as :data:`_SENSE_BINDING_ELIGIBLE`'s pool allows (47 eligible
    (family, gold_key) pairs: 22 unflipped, 25 flipped -- both sides always
    have candidates, so the alternation never has to fall back).

    ``ep.meta`` carries ``kind="sense_binding"`` (the marker
    ``clause_reactor._ambiguity_steps`` gates its M54b addendum on),
    ``family``, ``homograph``, ``gold_sense``, ``mfs_sense``, ``sense_key``,
    ``flipped``, ``entity`` (the gold/anchor-bearing name --
    ``clause_reactor`` reads this directly, mirroring
    ``_pronoun_context_step``'s own ``ep.meta["gold_antecedent"]`` placeholder
    pattern), ``other_entity``, ``gold_answer``, ``other_answer``.
    """

    def __init__(self, num_options: int = 2, seed: int = 0) -> None:
        self.num_options = num_options
        self.rng = random.Random(seed)
        self._count = 0

    def _episode(self) -> Episode:
        want_flipped = (self._count % 2 == 1)
        self._count += 1
        candidates = [(w, k) for (w, k, f) in _SENSE_BINDING_ELIGIBLE if f == want_flipped]
        if not candidates:   # defensive; both buckets are non-empty today
            candidates = [(w, k) for (w, k, _f) in _SENSE_BINDING_ELIGIBLE]
        family, gold_key = self.rng.choice(candidates)
        fam = _AMBIGUITY_FAMILIES[family]
        other_key = "B" if gold_key == "A" else "A"
        gold_sense_obj, other_sense_obj = fam["senses"][gold_key], fam["senses"][other_key]
        gold_answer, other_answer = gold_sense_obj["answer"], other_sense_obj["answer"]

        gold_name, other_name = self.rng.sample(_SENSE_BINDING_NAMES, 2)
        gold_cue = _SENSE_BINDING_CUE_TEMPLATE.format(n=gold_name, p=gold_answer)
        other_cue = _SENSE_BINDING_CUE_TEMPLATE.format(n=other_name, p=other_answer)
        cues = [gold_cue, other_cue]
        self.rng.shuffle(cues)
        anchor = gold_sense_obj["anchor"].format(name=gold_name)
        context = cues + [anchor]

        options = [gold_answer, other_answer]
        self.rng.shuffle(options)
        return Episode(
            context=context, question=f"what kind of {fam['word']} is it ?",
            answer_text=gold_answer, options=options, answer_idx=options.index(gold_answer),
            level=15,
            meta={
                "src": "curriculum2", "kind": "sense_binding",
                "family": family, "homograph": fam["word"],
                "gold_sense": gold_sense_obj["synset"], "mfs_sense": fam["mfs"],
                "sense_key": gold_key, "flipped": gold_sense_obj["synset"] != fam["mfs"],
                "entity": gold_name, "other_entity": other_name,
                "gold_answer": gold_answer, "other_answer": other_answer,
            },
        )

    def generate(self, n: int) -> List[Episode]:
        return [self._episode() for _ in range(n)]


def generate_sense_binding_episodes(n: int, seed: int = 0, num_options: int = 2) -> List[Episode]:
    """``n`` M54b binding-critical ambiguity episodes, deterministic given
    ``(n, seed, num_options)``. See :class:`SenseBindingCurriculumGenerator`."""
    return SenseBindingCurriculumGenerator(num_options=num_options, seed=seed).generate(n)


def association_only_baseline(episodes: List[Episode], dim: int = 64) -> Dict[str, float]:
    """The scripted ASSOCIATION-ONLY baseline (M54b's honesty gate, computed
    WITHOUT training/parsing): answer = whichever option's USVS handle is
    most cosine-similar to the mean of every content word's USVS handle,
    bagged over the episode's context sentences (names and ungroundable
    tokens dropped, exactly like the reactor drops ungroundable words --
    no parser call needed, this reads raw ``ep.context`` text directly).

    By :class:`SenseBindingCurriculumGenerator`'s construction, BOTH
    options' associated content word (``gold_answer`` and ``other_answer``)
    appear in every episode with equal footing (one per entity) -- so this
    baseline must land at chance (~0.5) on generated data; a value far from
    that means the curriculum's balance is broken, not that this baseline is
    unusually clever. Only ``kind == "sense_binding"`` episodes are scored;
    others are ignored (mirrors :func:`nearest_entity_baseline`'s own
    kind-filtering contract).
    """
    import re

    from .usvs_bridge import usvs_handle

    names = {n.lower() for n in _NAMES}
    total = correct = 0
    for ep in episodes:
        if ep.meta.get("kind") != "sense_binding":
            continue
        words = [w for sent in ep.context for w in re.findall(r"[a-z]+", sent.lower())
                 if w not in names]
        vecs = [usvs_handle(w, dim) for w in words]
        vecs = [v for v in vecs if v is not None]
        if not vecs or not ep.options:
            continue
        bag = np.mean(np.stack(vecs), axis=0)
        bag_n = bag / (np.linalg.norm(bag) + 1e-8)
        opt_vecs = [usvs_handle(o, dim) for o in ep.options]
        sims = [
            float(np.dot(bag_n, v / (np.linalg.norm(v) + 1e-8))) if v is not None else -1e9
            for v in opt_vecs
        ]
        pred = int(np.argmax(sims))
        total += 1
        correct += int(pred == ep.answer_idx)
    return {"n": total, "accuracy": correct / total if total else float("nan")}


# ---------------------------------------------------------------------------
# M55a: the garden-path curriculum (dev/TRACK_C_DESIGN.md Sec 1.10,
# RESEARCH_NOTES M55a). Built ONLY from the ambiguity shape
# scripts/probe_m55_hyp_survey.py's survey proved real: "{n} can {h} ."
# produces an EXACT structural-score tie (margin 0.0) between two
# non-equivalent readings for a specific set of verb/noun homographs (the
# M41 WordNet-lexicon-backed tag lattice) -- "can" as a transitive VERB with
# the homograph as its OBJECT vs the homograph as the main VERB with "can"
# as a bare modal. See :mod:`nsm_ct.clause_reactor`'s ``_garden_path_steps``
# for how the membrane candidate set is built from this.
# ---------------------------------------------------------------------------

# The 14 homographs the survey confirmed give an EXACT (margin=0.0) tie
# between the OBJECT reading (predicate=can, SUBJECT=n, OBJECT=h) and the
# VERB reading (predicate=h, SUBJECT=n, "can" a bare modal) for "{n} can
# {h} ." AND parse cleanly (SUBJECT=h, PLACE=p, single dominant reading) as
# "the {h} is in the {p} ." -- both checked empirically, not assumed (see
# the survey script and RESEARCH_NOTES M55a for the numbers). "flies"
# (plural of "fly") dropped as redundant with "fly".
GARDEN_PATH_HOMOGRAPHS: Tuple[str, ...] = (
    "bear", "book", "date", "duck", "fish", "fly", "park", "rock",
    "run", "saw", "time", "train", "watch",
)

# fred/bill excluded (parser landmines: "fred" is absent from the WordNet
# lexicon and falls through the -ed suffix heuristic to VERB; "bill" is
# itself a common-noun/verb homograph, risking collision when used as a
# proper name) -- same exclusion M53a/M54b already established.
_GARDEN_PATH_NAMES: Tuple[str, ...] = tuple(n for n in _NAMES if n not in ("fred", "bill"))

_GARDEN_PATH_MOVE_TEMPLATE = TEMPLATES["B"]["MOVE"][0]
assert _GARDEN_PATH_MOVE_TEMPLATE == "{n} went to the {p} ."
_GARDEN_PATH_HOMOGRAPH_PLACE_TEMPLATE = "the {h} is in the {p} ."
_GARDEN_PATH_AMBIGUOUS_TEMPLATE = "{n} can {h} ."

# ---------------------------------------------------------------------------
# M55b (RESEARCH_NOTES M55a's own flagged caveat: "the gold reading is
# currently an internal counter, NOT inferable from the context facts ...
# a resolver trained on this could not exceed 0.5 legitimately"). The fix,
# mirroring M54b's binding-critical move exactly (SenseBindingCurriculumGenerator
# above): an entity-keyed TRAIT marker fact, attached to name_a, that the
# gold reading is a deterministic function of -- plus a two-sided DECOY (the
# opposite marker, attached to a DIFFERENT entity never mentioned in the
# ambiguous sentence or question), so bag-of-words association stays
# uninformative by construction.
#
# The convention is arbitrary but fixed, like every other scripted word-to-
# meaning mapping in this module (see e.g. SenseBindingCurriculumGenerator's
# family/answer words): "market" (a place thematically tied to ACQUIRING/
# HAVING things) marks the OBJECT reading ("can" = transitively get/hold the
# homograph); "stadium" (a place thematically tied to physical ABILITY)
# marks the VERB reading ("can" = bare modal, an ability statement). Both
# words are checked disjoint from _PLACES, GARDEN_PATH_HOMOGRAPHS, and
# episode._AMBIGUITY_FAMILIES's vocabulary below (test_curriculum2.py) so
# they never collide with place_a/place_b/the homograph or the sense-binding
# curriculum's own word bank.
#
# The marker is written to memory under a DEDICATED relation (``rel:TRAIT``,
# nsm_ct.clause_reactor._garden_path_steps) -- never rel:PLACE -- so it can
# never collide with (overwrite) either reading's own PLACE address, and it
# reaches the collapse step's resolver ONLY through the controller's running
# ``state`` (see nsm_ct.resolver.RankHead's docstring for why: membrane.py's
# per-candidate Addr register is a single (entity, relation) slot per
# candidate, already spoken for by each reading's own PLACE query).
_GARDEN_PATH_TRAIT_WORDS: Dict[str, str] = {"object": "market", "verb": "stadium"}
_GARDEN_PATH_TRAIT_TEMPLATE = TEMPLATES["A"]["PLACE"][0]
assert _GARDEN_PATH_TRAIT_TEMPLATE == "{n} is in the {p} ."


def verify_garden_path_templates(sample_name: str = "mary", sample_place_a: str = "garden",
                                  sample_place_b: str = "kitchen",
                                  homographs: Tuple[str, ...] = GARDEN_PATH_HOMOGRAPHS,
                                  ) -> Dict[str, Dict[str, object]]:
    """Parse-check the garden-path curriculum's sentence shapes through the
    REAL parser (the ``verify_*`` pattern every other generator in this
    module follows). Checks, per homograph:

    - ``CONTEXT_HOM[{h}]``: "the {h} is in the {p} ." yields SUBJECT={h},
      PLACE={p} as its BEST (top-scoring) hypothesis.
    - ``AMBIGUOUS[{h}]``: "{n} can {h} ." yields >=2 top hypotheses with an
      EXACT score tie (margin 0.0) whose extracted role signatures differ
      AND both carry a SUBJECT (neither is a broken/degenerate parse) --
      the structural garden-path ambiguity :func:`GardenPathCurriculumGenerator`
      is built on. Returns an empty dict if ``quantum_parser`` isn't
      importable (caller must treat that as "unable to verify", not a
      pass) -- same contract as :func:`verify_templates` /
      :func:`verify_pronoun_templates` / :func:`verify_sense_binding_templates`.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    ctx_texts = [_GARDEN_PATH_MOVE_TEMPLATE.format(n=sample_name, p=sample_place_a)]
    hom_place_texts = {h: _GARDEN_PATH_HOMOGRAPH_PLACE_TEMPLATE.format(h=h, p=sample_place_b)
                        for h in homographs}
    ambiguous_texts = {h: _GARDEN_PATH_AMBIGUOUS_TEMPLATE.format(n=sample_name, h=h)
                        for h in homographs}
    texts = (ctx_texts + list(hom_place_texts.values()) + list(ambiguous_texts.values())
             + [sample_name, sample_place_a, sample_place_b] + list(homographs))
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        return {}  # quantum_parser unavailable; caller must handle this case

    def _signature(graph) -> frozenset:
        clauses, _links = extract_discourse(graph)
        return frozenset(((cl.predicate or "").lower(), rel, (arg.token or "").lower())
                          for cl in clauses for rel, arg in cl.args)

    results: Dict[str, Dict[str, object]] = {}
    for h, sent in hom_place_texts.items():
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        roles = {rel: (arg.token or "").lower() for cl in clauses for rel, arg in cl.args}
        ok = roles.get("SUBJECT") == h and roles.get("PLACE") == sample_place_b
        results[f"CONTEXT_HOM[{h}]"] = {"sentence": sent, "ok": ok, "roles": roles}

    for h, sent in ambiguous_texts.items():
        graphs, scores, margin = parser._parse_topk_one(sent, k=2)
        ok = False
        sig0 = sig1 = None
        if len(scores) >= 2 and margin == 0.0:
            sig0, sig1 = _signature(graphs[0]), _signature(graphs[1])
            has_subj0 = any(rel == "SUBJECT" for _p, rel, _a in sig0)
            has_subj1 = any(rel == "SUBJECT" for _p, rel, _a in sig1)
            ok = sig0 != sig1 and bool(sig0) and bool(sig1) and has_subj0 and has_subj1
        results[f"AMBIGUOUS[{h}]"] = {"sentence": sent, "ok": ok, "n_hyp": len(scores),
                                       "margin": margin, "sig0": sig0, "sig1": sig1}
    return results


def verify_garden_path_trait_templates(sample_name_a: str = "mary", sample_name_b: str = "john",
                                        ) -> Dict[str, Dict[str, object]]:
    """Parse-check the M55b TRAIT marker sentences (both directions,
    both entities) through the REAL parser -- the ``verify_*`` pattern every
    other generator in this module follows. Each must yield SUBJECT=name,
    PLACE=trait_word with no other content word (the same-clause-leak check
    :func:`verify_sense_binding_templates` already runs for its own cue
    sentences). Returns an empty dict if ``quantum_parser`` isn't importable
    (caller must treat that as "unable to verify", not a pass)."""
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    texts = [_GARDEN_PATH_TRAIT_TEMPLATE.format(n=n, p=p)
             for n in (sample_name_a, sample_name_b) for p in _GARDEN_PATH_TRAIT_WORDS.values()]
    tok = SimpleTokenizer.build(texts + [sample_name_a, sample_name_b] + list(_GARDEN_PATH_TRAIT_WORDS.values()),
                                 extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        return {}  # quantum_parser unavailable; caller must handle this case

    results: Dict[str, Dict[str, object]] = {}
    for n in (sample_name_a, sample_name_b):
        for reading, trait in _GARDEN_PATH_TRAIT_WORDS.items():
            sent = _GARDEN_PATH_TRAIT_TEMPLATE.format(n=n, p=trait)
            graph = parser._parse_graph(sent)
            clauses, _links = extract_discourse(graph)
            ok = any(
                {rel: (arg.token or "").lower() for rel, arg in cl.args}.get("SUBJECT") == n
                and {rel: (arg.token or "").lower() for rel, arg in cl.args}.get("PLACE") == trait
                and len(cl.args) == 2
                for cl in clauses)
            results[f"TRAIT[{n}.{reading}]"] = {"sentence": sent, "ok": ok}
    return results


class GardenPathCurriculumGenerator:
    """M55b binding-critical garden-path episodes: entity-keyed TRAIT marker
    facts (mirroring :class:`SenseBindingCurriculumGenerator`'s M54b decoy
    structure) determine which reading is gold, replacing M55a's internal
    counter (RESEARCH_NOTES M55a's own flagged caveat: "the gold reading is
    currently an internal counter, NOT inferable from the context facts ...
    a resolver trained on this could not exceed 0.5 legitimately").

    Episode shape (order: TRAIT cues first in random relative order --
    mirrors :class:`SenseBindingCurriculumGenerator`'s own "cues first"
    discipline -- then the two independently-true PLACE facts, the ANCHOR
    (the ambiguous sentence itself) always last):

        "{name_a_or_b} is in the {trait_word} ."      -- TRAIT cue (shuffled order)
        "{name_b_or_a} is in the {other_trait_word} ." -- TRAIT cue (decoy)
        "{name_a} went to the {place_a} ."              -- VERB-reading answer source
        "the {h} is in the {place_b} ."                 -- OBJECT-reading answer source
        "{name_a} can {h} ."                             -- the garden-path sentence
        Q: "where is {name_a} ?"

    "{name_a} can {h} ." is still the EXACT parser-score tie (margin 0.0,
    :data:`GARDEN_PATH_HOMOGRAPHS`, see :func:`verify_garden_path_templates`)
    between the OBJECT reading ("can" as a transitive VERB, homograph as its
    OBJECT -- "name_a now has/is with the homograph", so the correct place
    is the homograph's OWN place, ``place_b``) and the VERB reading
    (homograph as the main VERB, "can" a bare modal -- an ability statement
    that doesn't touch place, so the answer stays ``place_a``) -- UNCHANGED
    from M55a. What M55b changes is WHICH of those two readings is gold:

    ``gold_reading`` ("object" | "verb") is still chosen by an internal
    COUNTER (not RNG, mirroring :class:`PronounCurriculumGenerator`'s
    anti-recency discipline) for an exact 50/50 split -- but the counter's
    choice is then RECORDED as ``name_a``'s own TRAIT marker fact
    (``trait_word = _GARDEN_PATH_TRAIT_WORDS[gold_reading]``), so the
    reading genuinely FOLLOWS from a context fact keyed to the ambiguous
    sentence's own subject, exactly M54b's "the gold answer must follow
    from a memory-reachable fact, not a hidden field" move, applied to
    structural (not lexical) ambiguity. ``name_b`` -- a DIFFERENT entity,
    sampled distinct from ``name_a``, never mentioned in the ambiguous
    sentence or the question -- carries the OPPOSITE marker: the two-sided
    decoy. Both trait words are present in EVERY episode, one per entity,
    so a bag-of-words association can't use them
    (:func:`garden_path_association_baseline` must stay at chance) and only
    binding the marker to ``name_a`` SPECIFICALLY discriminates. Both PLACE
    facts (``place_a`` for name_a, ``place_b`` for the homograph) are STILL
    always present regardless of which reading is gold (M55a's own
    association-defeating property, preserved unchanged), and the parser's
    own top-1 hypothesis is still the deterministic OBJECT tie-break
    (:func:`garden_path_parser_top1_baseline` still lands at chance -- the
    marker redesign never touches the ambiguous sentence itself).

    ``ep.meta`` carries everything M55a's did (``garden_path=True``,
    ``name_a``, ``gp_homograph``, ``place_a``, ``place_b``,
    ``gold_reading``) PLUS ``other_entity`` (the decoy name), ``trait_word``
    (name_a's own marker), ``other_trait_word`` (the decoy's marker), and
    ``cue_order`` (``("a", "b")`` or ``("b", "a")`` -- which marker sentence
    comes first in ``ep.context``).
    """

    def __init__(self, num_options: int = 2, seed: int = 0, *,
                 names: Tuple[str, ...] = _GARDEN_PATH_NAMES,
                 homographs: Tuple[str, ...] = GARDEN_PATH_HOMOGRAPHS) -> None:
        self.num_options = num_options
        self.rng = random.Random(seed)
        self._count = 0
        self._names = names
        self._homographs = homographs

    def _episode(self) -> Episode:
        want_object = (self._count % 2 == 0)   # alternate -> exact 50/50
        self._count += 1
        gold_reading = "object" if want_object else "verb"
        other_reading = "verb" if want_object else "object"

        name_a, name_b = self.rng.sample(self._names, 2)
        homograph = self.rng.choice(self._homographs)
        place_a, place_b = self.rng.sample(_PLACES, 2)
        gold_place = place_b if want_object else place_a
        other_place = place_a if want_object else place_b

        trait_word = _GARDEN_PATH_TRAIT_WORDS[gold_reading]
        other_trait_word = _GARDEN_PATH_TRAIT_WORDS[other_reading]
        cue_a = _GARDEN_PATH_TRAIT_TEMPLATE.format(n=name_a, p=trait_word)
        cue_b = _GARDEN_PATH_TRAIT_TEMPLATE.format(n=name_b, p=other_trait_word)
        cues = [("a", cue_a), ("b", cue_b)]
        self.rng.shuffle(cues)
        cue_order = tuple(tag for tag, _sent in cues)

        context = [sent for _tag, sent in cues] + [
            _GARDEN_PATH_MOVE_TEMPLATE.format(n=name_a, p=place_a),
            _GARDEN_PATH_HOMOGRAPH_PLACE_TEMPLATE.format(h=homograph, p=place_b),
            _GARDEN_PATH_AMBIGUOUS_TEMPLATE.format(n=name_a, h=homograph),
        ]
        options = [gold_place, other_place]
        self.rng.shuffle(options)
        return Episode(
            context=context, question=f"where is {name_a} ?",
            answer_text=gold_place, options=options, answer_idx=options.index(gold_place),
            level=16,
            meta={
                "src": "curriculum2", "kind": "garden_path", "garden_path": True,
                # NOTE: keyed "gp_homograph", not "homograph" -- the latter is
                # the M54 ambiguity-episode marker (episode.py's
                # generate_ambiguity_episodes / clause_reactor.py's
                # `elif ep.meta.get("homograph")` branch); reusing it here
                # would silently route garden-path episodes through the WRONG
                # step-builder (_ambiguity_steps) since that check comes first.
                "name_a": name_a, "gp_homograph": homograph,
                "place_a": place_a, "place_b": place_b,
                "gold_reading": gold_reading,
                "other_entity": name_b, "trait_word": trait_word,
                "other_trait_word": other_trait_word, "cue_order": cue_order,
            },
        )

    def generate(self, n: int) -> List[Episode]:
        return [self._episode() for _ in range(n)]


def generate_garden_path_episodes(n: int, seed: int = 0, num_options: int = 2) -> List[Episode]:
    """``n`` M55a garden-path episodes, deterministic given ``(n, seed,
    num_options)``. See :class:`GardenPathCurriculumGenerator`."""
    return GardenPathCurriculumGenerator(num_options=num_options, seed=seed).generate(n)


def garden_path_parser_top1_baseline(episodes: List[Episode], parser) -> Dict[str, float]:
    """The scripted PARSER-TOP-1 baseline (M55a's honesty gate (a)): parse
    the ambiguous sentence with the REAL parser, take ``best_hypothesis()``
    (the parser's own top choice, exactly what :meth:`nsm_ct.input_encoder.
    ParserInputEncoder._parse_graph` already uses everywhere else), read off
    whether that ONE hypothesis is the OBJECT or VERB reading (has an
    OBJECT-relation edge on the "can" clause, or not), predict accordingly
    -- NO memory, no context facts consulted, exactly what a model would get
    "for free" if the parser's own score already picked the right reading.
    Must sit near floor (~0.5, chance) on generated data by
    :class:`GardenPathCurriculumGenerator`'s 50/50 balance -- if it doesn't,
    the episode doesn't belong (RESEARCH_NOTES M53a/M54b's "capability
    curricula must make the capability NECESSARY" discipline). ``parser`` is
    a :class:`~nsm_ct.input_encoder.ParserInputEncoder`; episodes without
    ``kind == "garden_path"`` are ignored (mirrors
    :func:`nearest_entity_baseline`'s own kind-filtering contract).
    """
    from .clause import extract_discourse

    total = correct = 0
    for ep in episodes:
        if ep.meta.get("kind") != "garden_path":
            continue
        name_a, homograph = ep.meta["name_a"], ep.meta["gp_homograph"]
        sentence = _GARDEN_PATH_AMBIGUOUS_TEMPLATE.format(n=name_a, h=homograph)
        graph = parser._parse_graph(sentence)
        clauses, _links = extract_discourse(graph)
        is_object_reading = any(
            rel == "OBJECT" and (arg.token or "").lower() == homograph
            for cl in clauses for rel, arg in cl.args)
        pred_place = ep.meta["place_b"] if is_object_reading else ep.meta["place_a"]
        total += 1
        correct += int(pred_place == ep.answer_text)
    return {"n": total, "accuracy": correct / total if total else float("nan")}


def garden_path_association_baseline(episodes: List[Episode], dim: int = 64) -> Dict[str, float]:
    """The scripted ASSOCIATION-ONLY baseline (M55a's honesty gate (b),
    mirroring :func:`association_only_baseline`'s exact method): answer =
    whichever option's USVS handle is most cosine-similar to the mean of
    every content word's USVS handle, bagged over the episode's context
    sentences (names dropped). By :class:`GardenPathCurriculumGenerator`'s
    construction BOTH place words (``place_a``, ``place_b``) appear in
    every episode's context with equal footing regardless of which is
    gold, so this must land at chance (~0.5). Only ``kind ==
    "garden_path"`` episodes are scored.
    """
    import re

    from .usvs_bridge import usvs_handle

    names = {n.lower() for n in _NAMES}
    total = correct = 0
    for ep in episodes:
        if ep.meta.get("kind") != "garden_path":
            continue
        words = [w for sent in ep.context for w in re.findall(r"[a-z]+", sent.lower())
                 if w not in names]
        vecs = [usvs_handle(w, dim) for w in words]
        vecs = [v for v in vecs if v is not None]
        if not vecs or not ep.options:
            continue
        bag = np.mean(np.stack(vecs), axis=0)
        bag_n = bag / (np.linalg.norm(bag) + 1e-8)
        opt_vecs = [usvs_handle(o, dim) for o in ep.options]
        sims = [
            float(np.dot(bag_n, v / (np.linalg.norm(v) + 1e-8))) if v is not None else -1e9
            for v in opt_vecs
        ]
        pred = int(np.argmax(sims))
        total += 1
        correct += int(pred == ep.answer_idx)
    return {"n": total, "accuracy": correct / total if total else float("nan")}


# ===========================================================================
# Spanish Freeze Test (dev/ROADMAP_LONG_TERM.md "The Spanish Freeze Test")
# ===========================================================================
# Append-only from here down: nothing above this line is touched. Perception-
# side, curriculum-scale surface templates for the L1-6-equivalent PLACE/MOVE
# facts + the TRANSFER TAKE/SOURCE shape, translated (not re-derived) from
# TEMPLATES["A"]/TRANSFER_TEMPLATES above, each parse-verified through the
# REAL Spanish grammar (grammars/spanish.json, cloned from english.json --
# see its metadata) + tagger (quantum_parser.pos_tagger.tag_spanish_sentence)
# exactly like verify_templates()/verify_transfer_templates() gate the
# English sets.
#
# Deliberate, documented exclusions (task explicitly allows "handle it or
# exclude it explicitly" for both):
#
# - **Pro-drop** ("encontró la pelota .", no subject at all): quantum_parser
#   is a constituency-rule engine keyed on tokens actually present -- there
#   is no rule that synthesizes an implicit SUBJECT node from verb
#   person/number morphology, and clause.extract_discourse's
#   _primary_discourse requires a real SUBJECT edge on every branch (no
#   subject -> zero clauses extracted, a silent content-loss failure mode,
#   not a crash). Building that synthesis is a grammar-engine feature, out
#   of scope here; every Spanish template below has an explicit subject.
# - **GIVE-family dative transfer** ("... dio la pelota a juan ."): mirrors
#   the ALREADY-DOCUMENTED English GIVE landmine above (this file, near
#   TRANSFER_TEMPLATES) almost exactly -- Spanish "a" is used for BOTH the
#   MOVE template's motion destination ("fue a la oficina") and the dative
#   recipient, and clause.py's ``_PREP_RELATION`` (extended, not edited --
#   see input_encoder._install_spanish_prep_relation) is a flat
#   token->role dict with no verb-sensitivity, so it cannot resolve "a" to
#   PLACE for one verb and RECIPIENT for another. English's escape hatch
#   (the double-object construction, "gave {receiver} the {obj}") does not
#   exist in Spanish without a clitic ("le dio la pelota a juan"), and the
#   clitic breaks SUBJECT extraction outright: "juan le dio ..." puts TWO
#   bare NOMINALs before the verb ("juan", "le"), and clause1's
#   before-quantifier "one" rule binds the NEAREST one ("le") as SUBJECT --
#   verified empirically, not just reasoned about (see
#   scripts/probe_spanish_freeze.py's template table). The TAKE/SOURCE
#   variant is used instead (below): "de" -> SOURCE has no motion-PLACE
#   ambiguity in these templates, so it needs no clitic and no landmine.
# - **Wh-question subject-verb inversion** ("¿ dónde está la pelota ?"):
#   english.json's question1 ruleset requires the copula tagged MODIFIER
#   (Tag.AUX, PROGRESSIVE) -- English gets this via ``AMBIGUOUS_WORDS``
#   exposing BOTH the AUX and VERB reading of "is" to the parse lattice
#   (declaratives need the VERB reading for predicate1's PP rule; questions
#   need the AUX reading for question1). Spanish "está" is tagged VERB only
#   (single tag, needed for "mary está en el jardín ." to parse at all);
#   making it ambiguous the same way would require gating
#   ``pos_tagger.get_possible_tags`` by language, since that function
#   currently has no language parameter and consults English-only tables --
#   a real architectural change, not attempted here to avoid ANY risk to
#   the English byte-identity gate. The question TEMPLATE below is still
#   authored and parse-checked (tokenization of "¿" decided: its own
#   whitespace-separated PUNCT token, mirroring how trailing "?" is
#   already tokenized) so the actual outcome is measured and reported
#   honestly, not assumed.
# ---------------------------------------------------------------------------

# Spanish place table, matching episode._PLACES 1:1. ``det``/``a_det`` are the
# FULL determined NP and its "a"+article motion-destination form (al/a la) --
# Spanish grammatical gender means the article travels with the noun, so
# (unlike English's fixed "the", baked into the template string) it is
# precomputed here and substituted as a whole. Hand-verified against a
# dictionary (not the trailing-vowel gender heuristic
# scripts/probe_spanish_freeze.py uses for the ~60-word VOCAB_SCALE_PLACES
# OMW coverage report, which is reporting-only and not fed into these
# templates).
_PLACES_ES: Dict[str, Dict[str, str]] = {
    "kitchen":  {"noun": "cocina",     "det": "la cocina",     "a_det": "a la cocina"},
    "garden":   {"noun": "jardín",     "det": "el jardín",     "a_det": "al jardín"},
    "office":   {"noun": "oficina",    "det": "la oficina",    "a_det": "a la oficina"},
    "bedroom":  {"noun": "dormitorio", "det": "el dormitorio", "a_det": "al dormitorio"},
    "hallway":  {"noun": "pasillo",    "det": "el pasillo",    "a_det": "al pasillo"},
    "bathroom": {"noun": "baño",       "det": "el baño",       "a_det": "al baño"},
}
assert set(_PLACES_ES) == set(_PLACES), "every episode._PLACES noun needs a Spanish entry"

# Spanish transfer-object table, matching this module's own _TRANSFER_OBJECTS
# 1:1 (grammatical-gender note from the task: "la pelota"/"el libro" -- both
# genders genuinely occur, not cherry-picked).
_TRANSFER_OBJECTS_ES: Dict[str, Dict[str, str]] = {
    "ball":   {"noun": "pelota", "det": "la pelota"},
    "box":    {"noun": "caja",   "det": "la caja"},
    "key":    {"noun": "llave",  "det": "la llave"},
    "book":   {"noun": "libro",  "det": "el libro"},
    "letter": {"noun": "carta",  "det": "la carta"},
    "coin":   {"noun": "moneda", "det": "la moneda"},
}
assert set(_TRANSFER_OBJECTS_ES) == set(_TRANSFER_OBJECTS), \
    "every _TRANSFER_OBJECTS noun needs a Spanish entry"

# Entity names are the UNTRANSLATED English _NAMES strings (mary/john/sandra/
# daniel/bill/fred), used verbatim as Spanish proper nouns. Deliberate:
# clause.py's is_entity()/EntityTracker/_ENTITY_NAMES are hardcoded to
# episode.py's English _NAMES and are out of scope for this task (FILES
# OWNED) -- reusing the identical strings means entity-atom resolution needs
# ZERO clause.py changes and produces EXACTLY (not just "up to naming")
# identical entity vectors across the English/Spanish clause streams (the
# entity column of scripts/probe_spanish_freeze.py's stream-equivalence
# table). Proper names not translating is also just linguistically ordinary.

# Spanish surface templates, mirroring TEMPLATES["A"]'s two actions.
# {n} = name (untranslated), {p} = the FULL determined place NP (_PLACES_ES
# "det" for PLACE, "a_det" for MOVE -- the a+el contraction lives in the
# table, not the template string).
TEMPLATES_ES: Dict[str, Dict[str, List[str]]] = {
    "A": {
        "PLACE": [
            "{n} está en {p} .",
            "{n} está ahora en {p} .",
        ],
        "MOVE": [
            "{n} fue {p} .",
        ],
    },
}

# Spanish question template (see the module-note above on why subject-verb
# inversion is not expected to work end-to-end; parse-checked honestly by
# verify_templates_es below rather than assumed). "¿" tokenized as its own
# leading whitespace-separated PUNCT token (decision recorded, mirrors how
# trailing "?" is already a separate token in the English templates).
QUESTION_TEMPLATE_ES = "¿ dónde está {p} ?"


def _check_subject_place(parser, extract_discourse, sent: str, name: str, place_noun: str) -> bool:
    """Shared body: parse ``sent``, require a clause whose SUBJECT/PLACE args
    equal ``name``/``place_noun`` exactly (same contract as verify_templates)."""
    graph = parser._parse_graph(sent)
    clauses, _links = extract_discourse(graph)
    for cl in clauses:
        subj = place = None
        for rel, arg in cl.args:
            if rel == "SUBJECT":
                subj = (arg.token or "").lower()
            elif rel == "PLACE":
                place = (arg.token or "").lower()
        if subj == name and place == place_noun:
            return True
    return False


def verify_templates_es(sample_name: str = "mary", sample_place: str = "garden"
                         ) -> Dict[str, bool]:
    """Spanish counterpart of :func:`verify_templates`: parse-check every
    :data:`TEMPLATES_ES` entry through the REAL Spanish grammar/tagger,
    requiring the resulting clause's SUBJECT and PLACE arguments to match
    the sentence's actual entity/place exactly (same non-degenerate-clause
    contract as the English gate). Returns ``{template_string: bool}``
    (formatted keys, like ``verify_templates``'s pre-format keys are English-
    template strings -- Spanish's per-noun article means the template
    string itself isn't reusable across places, unlike English's fixed
    "the"). Empty dict if quantum_parser isn't importable.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    place = _PLACES_ES[sample_place]
    texts = [t.format(n=sample_name, p=place["det"]) for t in TEMPLATES_ES["A"]["PLACE"]]
    texts += [t.format(n=sample_name, p=place["a_det"]) for t in TEMPLATES_ES["A"]["MOVE"]]
    texts += [sample_name, place["noun"]]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="es")
    if getattr(parser, "_parser", None) is None:
        return {}

    results: Dict[str, bool] = {}
    for t in TEMPLATES_ES["A"]["PLACE"]:
        sent = t.format(n=sample_name, p=place["det"])
        results[sent] = _check_subject_place(parser, extract_discourse, sent,
                                              sample_name, place["noun"])
    for t in TEMPLATES_ES["A"]["MOVE"]:
        sent = t.format(n=sample_name, p=place["a_det"])
        results[sent] = _check_subject_place(parser, extract_discourse, sent,
                                              sample_name, place["noun"])
    return results


# TAKE/SOURCE transfer template only -- see the module note above for why
# the GIVE/dative family is excluded. Carries its own "en {place}" PP (like
# English TRANSFER_TEMPLATES["TAKE"]'s trailing "in the {place}") so the two
# languages' clause streams expose the SAME relation set (SUBJECT/SOURCE/
# OBJECT/PLACE) for scripts/probe_spanish_freeze.py's relation-match gate --
# an earlier version omitted it, which under-counted relation agreement for
# a template-asymmetry reason having nothing to do with translation fidelity
# (fixed before this file's first measured run).
TRANSFER_TEMPLATES_ES: Dict[str, str] = {
    "TAKE": "{taker} tomó {obj} de {source} en {place} .",
}


def verify_transfer_templates_es(sample_taker: str = "mary", sample_source: str = "john",
                                  sample_obj: str = "ball", sample_place: str = "garden"
                                  ) -> Dict[str, Dict[str, object]]:
    """Spanish counterpart of :func:`verify_transfer_templates`, TAKE/SOURCE
    only (see module note). Returns ``{template_key: {"sentence": str, "ok":
    bool, "roles": {rel: token}}}``, same shape/contract as the English gate.
    Empty dict if quantum_parser isn't importable.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    obj = _TRANSFER_OBJECTS_ES[sample_obj]
    place = _PLACES_ES[sample_place]
    expected = {"SUBJECT": sample_taker, "SOURCE": sample_source, "OBJECT": obj["noun"],
                "PLACE": place["noun"]}
    texts = {k: t.format(taker=sample_taker, source=sample_source, obj=obj["det"], place=place["det"])
             for k, t in TRANSFER_TEMPLATES_ES.items()}
    tok = SimpleTokenizer.build(
        list(texts.values()) + [sample_taker, sample_source, obj["noun"], place["noun"]],
        extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="es")
    if getattr(parser, "_parser", None) is None:
        return {}

    out: Dict[str, Dict[str, object]] = {}
    for k, sent in texts.items():
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        roles: Dict[str, str] = {}
        ok = False
        for cl in clauses:
            cand = {rel: (arg.token or "").lower() for rel, arg in cl.args}
            if all(cand.get(rel) == val for rel, val in expected.items()):
                roles = cand
                ok = True
                break
            if len(cand) > len(roles):
                roles = cand
        out[k] = {"sentence": sent, "ok": ok, "roles": roles}
    return out


# Pronoun-find template (see module note: parse-verified only -- clause.py's
# English-only _PRONOUNS blocks entity-level extraction; this is reported,
# not hidden).
PRONOUN_FIND_TEMPLATE_ES = "{pronoun} encontró {obj} ."
_PRONOUN_CONTEXT_TEMPLATE_ES = TEMPLATES_ES["A"]["MOVE"][0]


def verify_pronoun_templates_es(sample_pronouns=("ella", "él"), sample_obj: str = "ball"
                                 ) -> Dict[str, Dict[str, object]]:
    """Parse-check :data:`PRONOUN_FIND_TEMPLATE_ES` for each of
    ``sample_pronouns``: reports whether the SUBJECT edge lands on the
    pronoun token at all (structural, grammar-layer check -- this is NOT
    the same as clause.py recognizing it as an entity; see the module note
    and membrane.py's ``_PRONOUN_EXTRA_ES`` docstring for that separate,
    documented gap). Returns ``{pronoun: {"sentence": str, "subject_ok":
    bool}}``. Empty dict if quantum_parser isn't importable.
    """
    from .clause import extract_discourse
    from .input_encoder import ParserInputEncoder
    from .nsm_primes import PRIME_NAMES
    from .structure import PARSE_LABELS
    from .tokenizer import SimpleTokenizer

    obj = _TRANSFER_OBJECTS_ES[sample_obj]
    texts = [PRONOUN_FIND_TEMPLATE_ES.format(pronoun=pr, obj=obj["det"]) for pr in sample_pronouns]
    tok = SimpleTokenizer.build(texts + list(sample_pronouns) + [obj["noun"]],
                                 extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="es")
    if getattr(parser, "_parser", None) is None:
        return {}

    out: Dict[str, Dict[str, object]] = {}
    for pr in sample_pronouns:
        sent = PRONOUN_FIND_TEMPLATE_ES.format(pronoun=pr, obj=obj["det"])
        graph = parser._parse_graph(sent)
        clauses, _links = extract_discourse(graph)
        subj_ok = any((arg.token or "").lower() == pr
                       for cl in clauses for rel, arg in cl.args if rel == "SUBJECT")
        out[pr] = {"sentence": sent, "subject_ok": subj_ok}
    return out


def generate_freeze_pairs(n: int, seed: int = 0) -> List[Dict[str, object]]:
    """``n`` parallel English/Spanish (context, shape) pairs -- the Freeze
    Test's raw material: same underlying facts (same rng draw of name(s) +
    place/object), same seed, only the surface language differs. Each item:
    ``{"shape": "PLACE"|"MOVE"|"TRANSFER", "en": [sentences], "es":
    [sentences], **draw}`` where ``draw`` carries the entities/places drawn
    (so a caller can independently verify a downstream comparison uses the
    SAME facts, not just eyeball the strings). Deterministic given ``seed``,
    like every other generator in this module. Shapes are drawn uniformly;
    PLACE/MOVE use the canonical (index-0) template from each language's set
    (:data:`TEMPLATES`/:data:`TEMPLATES_ES`) -- entities vary per draw, not
    phrasing, so a "translated episode" means translated STRUCTURE with the
    same random facts, exactly what the roadmap's freeze test asks to
    compare.

    Name pool excludes "fred" (the SAME pre-existing English-tagger landmine
    ``_MALE_NAMES``/``_SENSE_BINDING_NAMES`` above already exclude it for:
    the unconditional "ends in -ed -> VERB" suffix heuristic mistags lower-
    case "fred" as a VERB, which breaks ENGLISH clause extraction entirely
    -- confirmed empirically here too (any pair drawing "fred" produced
    zero English clauses, an English-side failure with nothing to do with
    Spanish). Excluding it is not a Spanish-side workaround; it is reusing
    the codebase's own established exclusion for a bug this task did not
    introduce and is not chartered to fix (quantum_parser/pos_tagger.py's
    ``simple_tag`` suffix heuristics).
    """
    rng = random.Random(seed)
    shapes = ["PLACE", "MOVE", "TRANSFER"]
    freeze_names = [n for n in _NAMES if n != "fred"]
    out: List[Dict[str, object]] = []
    for _ in range(n):
        shape = rng.choice(shapes)
        if shape in ("PLACE", "MOVE"):
            name = rng.choice(freeze_names)
            place = rng.choice(list(_PLACES_ES))
            en_t = TEMPLATES["A"][shape][0]
            es_t = TEMPLATES_ES["A"][shape][0]
            es_p = _PLACES_ES[place]["det"] if shape == "PLACE" else _PLACES_ES[place]["a_det"]
            out.append({
                "shape": shape, "name": name, "place": place,
                "en": [en_t.format(n=name, p=place)],
                "es": [es_t.format(n=name, p=es_p)],
            })
        else:
            taker, source = rng.sample(freeze_names, 2)
            obj = rng.choice(list(_TRANSFER_OBJECTS_ES))
            place = rng.choice(list(_PLACES_ES))
            en_t = TRANSFER_TEMPLATES["TAKE"]
            es_t = TRANSFER_TEMPLATES_ES["TAKE"]
            out.append({
                "shape": shape, "taker": taker, "source": source, "obj": obj, "place": place,
                "en": [en_t.format(taker=taker, source=source, obj=obj, place=place)],
                "es": [es_t.format(taker=taker, source=source,
                                    obj=_TRANSFER_OBJECTS_ES[obj]["det"],
                                    place=_PLACES_ES[place]["det"])],
            })
    return out
