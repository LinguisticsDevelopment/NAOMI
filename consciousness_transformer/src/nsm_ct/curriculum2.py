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
from typing import Dict, List, Optional, Tuple

import numpy as np

from .episode import Episode, _NAMES, _PLACES, _AMBIGUITY_FAMILIES
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


class GardenPathCurriculumGenerator:
    """M55a garden-path episodes: memory-coherence-flavored collapse over a
    genuine parser-score tie.

    Episode shape:
        "{name_a} went to the {place_a} ."          -- VERB-reading answer source
        "the {h} is in the {place_b} ."              -- OBJECT-reading answer source
        "{name_a} can {h} ."                          -- the garden-path sentence
        Q: "where is {name_a} ?"

    "{name_a} can {h} ." is an EXACT parser-score tie (margin 0.0,
    :data:`GARDEN_PATH_HOMOGRAPHS`, see :func:`verify_garden_path_templates`
    and ``scripts/probe_m55_hyp_survey.py``) between: the OBJECT reading
    ("can" as a transitive VERB, homograph as its OBJECT -- semantically
    placeholder-bound here as "name_a now has/is with the homograph", so
    the correct place is the homograph's OWN place, ``place_b``) and the
    VERB reading (homograph as the main VERB, "can" a bare modal, no object
    at all -- an ability statement that doesn't touch place, so the correct
    answer stays ``place_a``, unaffected).

    ``gold_reading`` ("object" | "verb") alternates by an internal COUNTER
    (not RNG, mirroring :class:`PronounCurriculumGenerator`'s anti-recency
    discipline and :class:`SenseBindingCurriculumGenerator`'s flip balance)
    -- exactly 50/50 for any ``n``. Both context facts (``place_a`` for
    name_a, ``place_b`` for the homograph) are ALWAYS present regardless of
    which reading is gold, so a bag-of-words association can't shortcut
    (:func:`garden_path_association_baseline` must land at chance), and the
    parser's own top-1 hypothesis for the ambiguous sentence is a
    DETERMINISTIC tie-break (always the OBJECT reading, per
    ``completeness_key``'s "more core-role edges wins" rule -- verified
    empirically for every homograph in :data:`GARDEN_PATH_HOMOGRAPHS`), so
    always-trust-the-parser's-top-1 also lands at chance
    (:func:`garden_path_parser_top1_baseline`).

    ``ep.meta`` carries ``garden_path=True`` (the marker
    ``clause_reactor.build_clause_batch`` gates its M55a branch on),
    ``name_a``, ``homograph``, ``place_a``, ``place_b``, ``gold_reading``.
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

        name_a = self.rng.choice(self._names)
        homograph = self.rng.choice(self._homographs)
        place_a, place_b = self.rng.sample(_PLACES, 2)
        gold_place = place_b if want_object else place_a
        other_place = place_a if want_object else place_b

        context = [
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
