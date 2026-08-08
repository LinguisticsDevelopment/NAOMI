"""Reasoning episodes and the sources that produce them.

An **episode** is the unit the model learns from, matching the "kindergartener"
regime: a *context stream* of statements followed by a *question*. The model is
meant to absorb the statements into memory one by one (gated by its state) and
then recognize the question and answer it.

Three sources sit behind one interface:

* :class:`CurriculumGenerator` — offline, always-runnable, reasoning-shaped
  episodes at escalating difficulty (one fact -> pick among facts -> recency).
  These genuinely require storing and retrieving facts; they are not surface
  pattern-completion.
* :class:`BabiSource` — loads Facebook's bAbI tasks (the canonical
  facts-then-question reasoning corpus). Falls back to the generator if the data
  cannot be obtained in this environment.
* :class:`TextbookSource` — a **stub** for the north star: ingest a textbook
  chapter as the context stream and answer its (often multiple-choice) homework
  questions. Not implemented yet.

Answers are available in two forms so both training modes work: ``answer_text``
(open-ended) and ``options`` + ``answer_idx`` (multiple choice — the densest
training signal, kept as a first-class citizen).
"""

from __future__ import annotations

import abc
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

from . import reasoning_oracle


@dataclass
class Episode:
    """A single reasoning episode.

    Attributes:
        context: Ordered statements (the stream to absorb into memory).
        question: The question that should trigger a response.
        answer_text: The correct answer as a string (open-ended supervision).
        options: For multiple choice, the candidate answers (correct one included).
        answer_idx: Index of the correct option in ``options`` (MC supervision).
        level: Curriculum difficulty (0 = easiest). Provenance/debug only.
        meta: Free-form provenance.
    """

    context: List[str]
    question: str
    answer_text: str
    options: Optional[List[str]] = None
    answer_idx: Optional[int] = None
    level: int = 0
    post_context: List[str] = field(default_factory=list)  # distractors AFTER the question
    trust_labels: Optional[List[float]] = None  # per-context-item: 1 trustworthy, 0 contradicted (metrics only)
    disjuncts: Optional[List[str]] = None  # the OR alternatives (e.g. ["kitchen", "office"])
    truth_labels_per_disjunct: Optional[List[float]] = None  # per-disjunct truth (metrics only)
    # Multi-question episodes: item indices of ALL questions in the stream and the
    # correct option index per question (the shared `options` list scores them all).
    question_positions: Optional[List[int]] = None
    question_targets: Optional[List[int]] = None
    # Reasoning levels (oracle-provided): the gold derivation chain and whether the
    # query is derivable at all (False -> the gold answer is "idk"/abstain).
    gold_chain: Optional[List] = None
    answerable: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.options is not None:
            if self.answer_idx is None or not 0 <= self.answer_idx < len(self.options):
                raise ValueError("answer_idx must index into options")
            if self.options[self.answer_idx] != self.answer_text:
                # Keep them consistent; the text is the source of truth.
                self.answer_text = self.options[self.answer_idx]

    @property
    def is_multiple_choice(self) -> bool:
        return self.options is not None


class AbstractEpisodeSource(abc.ABC):
    """Interface for anything that yields :class:`Episode` objects."""

    @abc.abstractmethod
    def generate(self, n: int) -> List[Episode]:
        """Return ``n`` episodes."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Curriculum generator (offline default)
# ---------------------------------------------------------------------------
_NAMES = ["mary", "john", "sandra", "daniel", "bill", "fred"]
_PLACES = ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom"]
# Reasoning-level vocab (self-contained; no WordNet dependency).
_OBJECTS = ["stove", "window", "painting", "mirror", "clock"]   # "can see the X"
_ISA = [("robin", "bird", "fly"), ("trout", "fish", "swim"),
        ("beagle", "dog", "bark"), ("oak", "tree", "grow")]       # (sub, super, ability)
_ABILITIES = [a for _s, _p, a in _ISA]
_IDK = "idk"  # the abstain atom (a first-class "I don't know")
# Deep variable-length chains (L12/L13). Distinct tokens per chain; depth k in 2..4.
_CHAIN_NOUNS = ["robin", "sparrow", "beagle", "trout", "oak", "maple",
                "sedan", "hawk", "perch", "cedar", "finch", "spaniel"]
_LADDER_RELS = ["PLACE", "CAN_SEE", "CAN_HOLD", "CAN_OPEN", "CAN_REACH"]
_LADDER_VALS = ["kitchen", "stove", "key", "door", "chest",
                "lamp", "gate", "box", "drawer", "hatch"]
_REL_PHRASE = {"PLACE": "is in", "CAN_SEE": "can see", "CAN_HOLD": "can hold",
               "CAN_OPEN": "can open", "CAN_REACH": "can reach"}


class CurriculumGenerator(AbstractEpisodeSource):
    """Generates reasoning-shaped episodes at escalating difficulty.

    Levels (a mini bAbI-style curriculum):
        1. One fact, ask it back. Requires storing and recalling a single fact.
        2. Two facts about different people; ask about one. Requires storing
           several facts and retrieving the *relevant* one.
        3. A person moves; ask where they are now. Requires updating/recency.
        4. A fact, the question, THEN a distractor fact (post-question). The
           answer is fixed before the question, so the model must learn to
           respond *at the question* — not just at the last item. This is the
           probe for emergent response timing.
        5. Corroboration vs contradiction: two sources agree, one disagrees;
           the answer is the corroborated place. Probes emergent trust.
        6. Overwrite update: a person's location is updated amid an unrelated
           fact; the answer is the new place. Probes overwrite-not-forget memory.
        7. Disjunction (logical OR): "X is in the A or the B." Half the time it is
           left UNRESOLVED — the answer is **maybe** (a first-class uncertain
           answer, the NSM MAYBE); half the time a negation ("X is not in the A")
           RESOLVES it — the answer is the other place. Recency/majority cannot do
           this; only storing the OR and deciding truth on the negation can.
        8. Negation removes a value: assert A, assert B, then "X is not in the B".
           The answer is A — the negation must *remove* B (recency alone would
           wrongly answer B). Probes that NOT subtracts a stored value.

    Every episode is emitted with both multiple-choice options and an
    open-ended answer, so either training mode can consume it.

    Args:
        max_level: Highest difficulty level to sample (1-6).
        num_options: Number of multiple-choice options per question.
        seed: RNG seed.
    """

    def __init__(self, max_level: int = 3, num_options: int = 4, seed: int = 0) -> None:
        self.max_level = max(1, min(max_level, 13))
        self.num_options = num_options
        self.rng = random.Random(seed)

    def _mc(self, answer: str) -> tuple[List[str], int]:
        distractors = self.rng.sample([p for p in _PLACES if p != answer], self.num_options - 1)
        options = distractors + [answer]
        self.rng.shuffle(options)
        return options, options.index(answer)

    def _mc_with(self, answer: str, required: List[str]) -> tuple[List[str], int]:
        """MC options that must contain ``answer`` and every string in ``required``.

        Used by the logical levels so the option set always pits the answer against
        the disjuncts and the "maybe" alternative (the model cannot win by guessing).
        """
        opts = list(dict.fromkeys([answer] + required))         # de-dup, keep order
        pool = [p for p in _PLACES if p not in opts]
        while len(opts) < self.num_options and pool:
            opts.append(pool.pop(self.rng.randrange(len(pool))))
        opts = opts[: self.num_options]
        self.rng.shuffle(opts)
        return opts, opts.index(answer)

    def _mc_pool(self, answer: str, pool: List[str], required: List[str] = ()) -> tuple[List[str], int]:
        """MC options over an arbitrary ``pool`` that must contain ``answer`` + ``required``."""
        opts = list(dict.fromkeys([answer, *required]))
        extra = [p for p in pool if p not in opts]
        while len(opts) < self.num_options and extra:
            opts.append(extra.pop(self.rng.randrange(len(extra))))
        opts = opts[: self.num_options]
        self.rng.shuffle(opts)
        return opts, opts.index(answer)

    @staticmethod
    def _chain_tuples(chain) -> List[tuple]:
        return [(s.derived, s.rule, s.support) for s in chain]

    def _level1(self) -> Episode:
        name = self.rng.choice(_NAMES)
        place = self.rng.choice(_PLACES)
        options, idx = self._mc(place)
        return Episode(
            context=[f"{name} is in the {place} ."],
            question=f"where is {name} ?",
            answer_text=place,
            options=options,
            answer_idx=idx,
            level=1,
        )

    def _level2(self) -> Episode:
        name_a, name_b = self.rng.sample(_NAMES, 2)
        place_a, place_b = self.rng.sample(_PLACES, 2)
        target_name, target_place = self.rng.choice([(name_a, place_a), (name_b, place_b)])
        context = [f"{name_a} is in the {place_a} .", f"{name_b} is in the {place_b} ."]
        self.rng.shuffle(context)
        options, idx = self._mc(target_place)
        return Episode(
            context=context,
            question=f"where is {target_name} ?",
            answer_text=target_place,
            options=options,
            answer_idx=idx,
            level=2,
        )

    def _level3(self) -> Episode:
        name = self.rng.choice(_NAMES)
        first, second = self.rng.sample(_PLACES, 2)
        context = [
            f"{name} is in the {first} .",
            f"{name} moved to the {second} .",
        ]
        options, idx = self._mc(second)  # recency: answer is the most recent place
        return Episode(
            context=context,
            question=f"where is {name} ?",
            answer_text=second,
            options=options,
            answer_idx=idx,
            level=3,
        )

    def _level4(self) -> Episode:
        # A *corrupting* distractor AFTER the question: the same person "moves"
        # afterwards. The answer is their location when asked (the first place),
        # so absorbing the trailing item and then answering gives the WRONG
        # (updated) place. The model must respond at/before the question rather
        # than naively reading the final state — emergent response timing.
        name = self.rng.choice(_NAMES)
        first, second = self.rng.sample(_PLACES, 2)
        options, idx = self._mc(first)
        return Episode(
            context=[f"{name} is in the {first} ."],
            question=f"where is {name} ?",
            answer_text=first,
            options=options,
            answer_idx=idx,
            level=4,
            post_context=[f"{name} moved to the {second} ."],
        )

    def _level5(self) -> Episode:
        # Corroboration vs contradiction: two sources agree on the true place,
        # one contradicts. The answer is the corroborated (majority) place, and
        # the contradicting statement may come last (so recency alone fails). The
        # model must learn to TRUST the corroborated info — emergently, since
        # answering right requires discounting the contradiction.
        name = self.rng.choice(_NAMES)
        true_place, false_place = self.rng.sample(_PLACES, 2)
        stmts = [
            (f"{name} is in the {true_place} .", 1.0),
            (f"{name} is in the {false_place} .", 0.0),
            (f"{name} is in the {true_place} .", 1.0),
        ]
        self.rng.shuffle(stmts)
        options, idx = self._mc(true_place)
        return Episode(
            context=[s for s, _ in stmts],
            question=f"where is {name} ?",
            answer_text=true_place,
            options=options,
            answer_idx=idx,
            level=5,
            trust_labels=[lab for _, lab in stmts],
        )

    def _level6(self) -> Episode:
        # Overwrite stress: a person's location is updated amid an unrelated
        # fact. The answer is the UPDATED place, so memory must overwrite that
        # person's slot in place (not the distractor's, and without forgetting
        # it). Exercises content-addressed overwrite (distinct from level 3's
        # bare update and level 4's post-question move).
        name_a, name_b = self.rng.sample(_NAMES, 2)
        place_b = self.rng.choice(_PLACES)
        first, second = self.rng.sample([p for p in _PLACES if p != place_b], 2)
        context = [
            f"{name_a} is in the {first} .",
            f"{name_b} is in the {place_b} .",
            f"{name_a} moved to the {second} .",
        ]
        options, idx = self._mc(second)
        return Episode(
            context=context,
            question=f"where is {name_a} ?",
            answer_text=second,
            options=options,
            answer_idx=idx,
            level=6,
        )

    def _level7(self) -> Episode:
        # Disjunction. "X is in the A or the B." Either left UNRESOLVED (answer the
        # NSM "maybe") or RESOLVED by a negation ("X is not in the A" -> answer B).
        # The option set always includes "maybe", A and B, so the model must (a)
        # store the OR and (b) decide truth on the negation — recency/majority fail.
        name = self.rng.choice(_NAMES)
        a, b = self.rng.sample(_PLACES, 2)
        resolved = self.rng.random() < 0.5
        if resolved:
            context = [f"{name} is in the {a} or the {b} .",
                       f"{name} is not in the {a} ."]
            answer, truth = b, [0.0, 1.0]
        else:
            context = [f"{name} is in the {a} or the {b} ."]
            answer, truth = "maybe", [0.5, 0.5]
        options, idx = self._mc_with(answer, ["maybe", a, b])
        return Episode(
            context=context,
            question=f"where is {name} ?",
            answer_text=answer,
            options=options,
            answer_idx=idx,
            level=7,
            disjuncts=[a, b],
            truth_labels_per_disjunct=truth,
            meta={"resolved": resolved},
        )

    def _level8(self) -> Episode:
        # Negation removes a value. Assert A, assert B, then "not in B": the answer
        # is A. Recency would wrongly answer B (the last positive place); only NOT
        # subtracting B from memory leaves A.
        name = self.rng.choice(_NAMES)
        a, b = self.rng.sample(_PLACES, 2)
        context = [f"{name} is in the {a} .",
                   f"{name} is in the {b} .",
                   f"{name} is not in the {b} ."]
        options, idx = self._mc_with(a, ["maybe", b])
        return Episode(
            context=context,
            question=f"where is {name} ?",
            answer_text=a,
            options=options,
            answer_idx=idx,
            level=8,
            disjuncts=[a, b],
            truth_labels_per_disjunct=[1.0, 0.0],
        )

    def _level9(self) -> Episode:
        # Conditional (modus ponens): a rule + its antecedent fact. The answer (what
        # they can see) is NEVER stated — only firing the rule derives it. Retrieval
        # and recency both fail; the loop must compose rule + fact.
        name = self.rng.choice(_NAMES)
        a = self.rng.choice(_PLACES)
        x = self.rng.choice(_OBJECTS)
        facts: List[reasoning_oracle.Triple] = [(name, "PLACE", a)]
        rules = [reasoning_oracle.conditional_rule(a, x)]
        ans, chain = reasoning_oracle.derive(facts, rules, (name, "CAN_SEE"))
        context = [f"if {name} is in the {a} , {name} can see the {x} .",
                   f"{name} is in the {a} ."]
        options, idx = self._mc_pool(ans, _OBJECTS, [_IDK])
        return Episode(
            context=context, question=f"what can {name} see ?",
            answer_text=ans, options=options, answer_idx=idx, level=9,
            answerable=True, gold_chain=self._chain_tuples(chain),
            meta={"facts": facts, "query": (name, "CAN_SEE"),
                  "rule": ((name, "PLACE", a), (name, "CAN_SEE", x))},
        )

    def _level10(self) -> Episode:
        # Transitivity / inheritance: "a robin is a bird. a bird can fly." -> a robin
        # can fly. A 2-hop chain over an is-a edge; the ability is never asserted of
        # the subtype directly.
        sub, sup, act = self.rng.choice(_ISA)
        facts: List[reasoning_oracle.Triple] = [(sub, "IS_A", sup), (sup, "CAN", act)]
        rules = [reasoning_oracle.INHERITANCE]
        ans, chain = reasoning_oracle.derive(facts, rules, (sub, "CAN"))
        context = [f"a {sub} is a {sup} .", f"a {sup} can {act} ."]
        options, idx = self._mc_pool(ans, _ABILITIES, [_IDK])
        return Episode(
            context=context, question=f"what can a {sub} do ?",
            answer_text=ans, options=options, answer_idx=idx, level=10,
            answerable=True, gold_chain=self._chain_tuples(chain),
            meta={"facts": facts, "query": (sub, "CAN")},
        )

    def _level11(self) -> Episode:
        # Unanswerable -> abstain. The rule is about place A but the person is in a
        # DIFFERENT place B, so the antecedent never fires; nothing derives what they
        # can see. The gold answer is "idk" (the tempting object X is an option).
        name = self.rng.choice(_NAMES)
        a, b = self.rng.sample(_PLACES, 2)
        x = self.rng.choice(_OBJECTS)
        facts: List[reasoning_oracle.Triple] = [(name, "PLACE", b)]
        rules = [reasoning_oracle.conditional_rule(a, x)]
        ans, _chain = reasoning_oracle.derive(facts, rules, (name, "CAN_SEE"))
        assert ans is None, "level 11 must be unanswerable"
        context = [f"if {name} is in the {a} , {name} can see the {x} .",
                   f"{name} is in the {b} ."]
        options, idx = self._mc_pool(_IDK, _OBJECTS, [x])
        return Episode(
            context=context, question=f"what can {name} see ?",
            answer_text=_IDK, options=options, answer_idx=idx, level=11,
            answerable=False, gold_chain=[],
            meta={"facts": facts, "query": (name, "CAN_SEE"),
                  "rule": ((name, "PLACE", a), (name, "CAN_SEE", x))},
        )

    def _level12(self) -> Episode:
        # Deep transitivity, variable depth k (2..4). TWO disjoint is-a chains ending in
        # DIFFERENT abilities; the question targets one chain's root. The distractor chain is
        # essential: with a single ability the model could just retrieve it without chaining
        # (the is-a links would be a red herring). Facts shuffled (no pre-chaining at ingest).
        # Half the time the queried chain is broken -> unanswerable -> abstain.
        k = self.rng.randint(2, 4)
        names = self.rng.sample(_CHAIN_NOUNS, 2 * (k + 1))
        n, m = names[: k + 1], names[k + 1:]               # queried chain n, distractor chain m
        act, act2 = self.rng.sample(_ABILITIES, 2)
        broken = self.rng.random() < 0.4
        edges = [(n[i], "IS_A", n[i + 1]) for i in range(k)]
        if broken:
            edges.pop(self.rng.randrange(len(edges)))       # snap a link in the queried chain
        facts = (edges + [(n[k], "CAN", act)]
                 + [(m[i], "IS_A", m[i + 1]) for i in range(k)] + [(m[k], "CAN", act2)])
        self.rng.shuffle(facts)
        rules = [reasoning_oracle.IS_A_TRANS, reasoning_oracle.INHERITANCE]
        ans, chain = reasoning_oracle.derive(facts, rules, (n[0], "CAN"))
        ctx = [f"a {e} is a {v} ." for (e, _r, v) in facts if _r == "IS_A"]
        ctx += [f"a {e} can {v} ." for (e, _r, v) in facts if _r == "CAN"]
        self.rng.shuffle(ctx)
        q = f"what can a {n[0]} do ?"
        meta = {"facts": facts, "query": (n[0], "CAN"), "chain_len": k}
        if ans is None:                                     # broken chain -> abstain
            options, idx = self._mc_pool(_IDK, _ABILITIES, [act, act2])
            return Episode(context=ctx, question=q, answer_text=_IDK, options=options,
                           answer_idx=idx, level=12, answerable=False, gold_chain=[], meta=meta)
        options, idx = self._mc_pool(ans, _ABILITIES, [act2, _IDK])
        return Episode(context=ctx, question=q, answer_text=ans, options=options,
                       answer_idx=idx, level=12, answerable=True,
                       gold_chain=self._chain_tuples(chain), meta=meta)

    def _level13(self) -> Episode:
        # Chained modus ponens, variable depth k (2..4): cascading conditionals
        # (if in A -> sees X; if sees X -> holds Y; ...) + the seed fact. The answer needs
        # all k rules to fire in sequence. Half the time the seed fact is missing/wrong.
        k = self.rng.randint(2, 4)
        name = self.rng.choice(_NAMES)
        rels = _LADDER_RELS[: k + 1]                        # rels[0..k]
        vals = self.rng.sample(_LADDER_VALS, k + 1)         # vals[0..k]
        rules_struct = [((name, rels[i], vals[i]), (name, rels[i + 1], vals[i + 1]))
                        for i in range(k)]
        rules = [reasoning_oracle.Rule((a,), c, name="mp") for a, c in rules_struct]
        self.rng.shuffle(rules_struct)   # shuffle the cascade: answer isn't positionally last
        broken = self.rng.random() < 0.4
        facts = [] if broken else [(name, rels[0], vals[0])]
        ans, chain = reasoning_oracle.derive(facts, rules, (name, rels[k]))
        ctx = [f"if {name} {_REL_PHRASE[rels[i]]} the {vals[i]} , "
               f"{name} {_REL_PHRASE[rels[i + 1]]} the {vals[i + 1]} ." for i in range(k)]
        if not broken:
            ctx.append(f"{name} {_REL_PHRASE[rels[0]]} the {vals[0]} .")
        qword = _REL_PHRASE[rels[k]].split()[-1]            # "see"/"hold"/"open"/"reach"
        question = f"what can {name} {qword} ?"
        if ans is None:                                     # missing seed -> abstain
            options, idx = self._mc_pool(_IDK, _LADDER_VALS, [vals[k]])
            return Episode(context=ctx, question=question, answer_text=_IDK,
                           options=options, answer_idx=idx, level=13, answerable=False,
                           gold_chain=[],
                           meta={"facts": facts, "rules": rules_struct,
                                 "query": (name, rels[k]), "chain_len": k})
        options, idx = self._mc_pool(ans, _LADDER_VALS, [_IDK])
        return Episode(context=ctx, question=question, answer_text=ans,
                       options=options, answer_idx=idx, level=13, answerable=True,
                       gold_chain=self._chain_tuples(chain),
                       meta={"facts": facts, "rules": rules_struct,
                             "query": (name, rels[k]), "chain_len": k})

    def base_facts(self) -> List[str]:
        """Base 'facts we know about the world' to seed long-term memory."""
        facts = [f"the {p} is a place ." for p in _PLACES]
        facts += [f"{n} is a person ." for n in _NAMES]
        return facts

    def generate(self, n: int) -> List[Episode]:
        builders = [self._level1, self._level2, self._level3, self._level4,
                    self._level5, self._level6, self._level7, self._level8,
                    self._level9, self._level10, self._level11,
                    self._level12, self._level13][: self.max_level]
        return [builders[i % len(builders)]() for i in range(n)]


def chained_question_episode(seed: int = 0):
    """Build a multi-question episode for the consistency probe.

    Stream: two facts, then ``Q1`` (about A), ``Q2`` (about B), ``Q1'`` (A again).
    All questions are answered in one **unreset** run; the repeat ``Q1'`` (after
    the intervening ``Q2``) should match ``Q1``. A fixed option set (all places)
    is shared across questions so a single batch can score every readout.

    Returns:
        ``(episode, positions, gold, repeat_pairs)`` where ``positions`` are the
        item indices of the questions in the stream, ``gold`` the correct option
        index per question, and ``repeat_pairs`` the (Q1, Q1') columns to compare.
    """
    rng = random.Random(seed)
    a, b = rng.sample(_NAMES, 2)
    pa, pb = rng.sample(_PLACES, 2)
    facts = [f"{a} is in the {pa} .", f"{b} is in the {pb} ."]
    rng.shuffle(facts)
    # One option set scores every question: both answers + distractors (4 options,
    # matching the curriculum so chained episodes batch with single-question ones).
    options = [pa, pb] + rng.sample([p for p in _PLACES if p not in (pa, pb)], 2)
    rng.shuffle(options)
    base = len(facts)
    positions = [base, base + 1, base + 2]                  # Q1, Q2, Q1'
    gold = [options.index(pa), options.index(pb), options.index(pa)]
    ep = Episode(
        context=facts,
        question=f"where is {a} ?",
        answer_text=pa,
        options=options,
        answer_idx=options.index(pa),
        post_context=[f"where is {b} ?", f"where is {a} ?"],
        question_positions=positions,
        question_targets=gold,
    )
    return ep, positions, gold, [(0, 2)]


def generate_chained_episodes(n: int, seed: int = 0) -> List[Episode]:
    """``n`` chained multi-question episodes (for training the capability)."""
    return [chained_question_episode(seed=seed * 100_000 + i)[0] for i in range(n)]


# ---------------------------------------------------------------------------
# M32 — ambiguity-bearing comprehension curriculum
# ---------------------------------------------------------------------------
# Homograph families where WordNet carries clearly distinct senses and BOTH
# senses fit the curriculum's simple declarative world. Each family names two
# senses (A, B): a real WordNet synset, the word that stands in for "the right
# answer under that sense" (itself USVS-groundable), an ANCHOR sentence that
# always contains the bare homograph, and DISTRACTOR sentences that reinforce
# the same reading (sometimes by repeating the word, matching the style of the
# original 4 families). ``mfs`` is the family's most-frequent-sense synset
# (``wn.synsets(word)[0]``), recorded as a constant so generation never needs
# a live WordNet call (offline, like the rest of this module) — verified
# against a live lookup in tests/test_m32_ambiguity.py.
#
# M32.2 (this batch): grown from 4 to 31 families to test whether the
# 24.7k-param sense chooser's leave-one-family-out failure on "bank" was a
# too-few-shots artifact (bank was one of only 3 training families) rather
# than something about "bank" itself — see scripts/train_sense_chooser.py's
# leave-one-family-out rotation over ALL families now, not just 4.
#
# Most families are constructed so ``mfs`` equals sense A's synset (so within
# a family sense-flip is close to a coin flip); a handful of words (seal,
# crane, hood, cell, mole) have a WordNet MFS that lands on a rare/awkward
# sense unrelated to BOTH curriculum-natural readings (e.g. crane's "most
# frequent sense" by ``wn.synsets`` order is the writer Stephen Crane, not
# the bird or the machine) — those families are ~100% sense-flipped and are
# kept anyway (real WordNet quirk, not a generator bug); see
# tests/test_m32_ambiguity.py for the accounting.
#
# Dropped during construction (pre-flight: both senses must have a nonzero
# ``usvs_sense_handle(d=128)`` and cosine(A, B) < 0.8, else the chooser has no
# vector-space signal to discriminate on):
#   - "tank" (army tank vs. storage tank): cosine(A, B) = 0.875 at d=128 —
#     both senses' USVS coordinates are too close (both cluster near a
#     generic "large enclosed container" region) to be discriminable.
# Trimmed (passed pre-flight but cut to keep the family count near the ~30
# target, favoring the words explicitly called out in the design brief):
#   bow, key, drill, trunk, block, spade, club — all had valid, discriminable
#   sense pairs but the smallest A/B cosine margins among the non-hinted
#   candidates, so cutting here preserved the rest of the pool's average
#   discriminability while trimming headcount.
_AMBIGUITY_FAMILIES = {
    "bank": {
        "word": "bank",
        "mfs": "bank.n.01",
        "senses": {
            "A": {
                "synset": "bank.n.01",
                "answer": "river",
                "anchor": "{name} sat on the bank .",
                "distractors": [
                    "the river flowed past the bank .",
                    "the water was cold .",
                    "{name} watched the fish swim .",
                ],
            },
            "B": {
                "synset": "depository_financial_institution.n.01",
                "answer": "money",
                "anchor": "{name} walked into the bank .",
                "distractors": [
                    "the teller counted the money .",
                    "the vault door was heavy .",
                    "{name} opened a new account .",
                ],
            },
        },
    },
    "bat": {
        "word": "bat",
        "mfs": "bat.n.01",
        "senses": {
            "A": {
                "synset": "bat.n.01",
                "answer": "cave",
                "anchor": "{name} saw a bat in the cave .",
                "distractors": [
                    "the bat flew in the dark .",
                    "the cave was silent .",
                    "{name} heard a squeak .",
                ],
            },
            "B": {
                "synset": "bat.n.05",
                "answer": "game",
                "anchor": "{name} picked up the bat .",
                "distractors": [
                    "the bat hit the ball .",
                    "the crowd cheered .",
                    "{name} played in the game .",
                ],
            },
        },
    },
    "plant": {
        "word": "plant",
        "mfs": "plant.n.01",
        "senses": {
            "A": {
                "synset": "plant.n.01",
                "answer": "factory",
                "anchor": "{name} worked at the plant .",
                "distractors": [
                    "the plant made cars .",
                    "the machines were loud .",
                    "{name} wore a hard hat .",
                ],
            },
            "B": {
                "synset": "plant.n.02",
                "answer": "garden",
                "anchor": "{name} watered the plant .",
                "distractors": [
                    "the plant grew leaves .",
                    "the soil was dry .",
                    "{name} placed it near the window .",
                ],
            },
        },
    },
    "organ": {
        "word": "organ",
        "mfs": "organ.n.01",
        "senses": {
            "A": {
                "synset": "organ.n.01",
                "answer": "body",
                "anchor": "{name} studied the organ .",
                "distractors": [
                    "the organ pumped blood .",
                    "the doctor examined the patient .",
                    "{name} read the chart .",
                ],
            },
            "B": {
                "synset": "organ.n.05",
                "answer": "music",
                "anchor": "{name} played the organ .",
                "distractors": [
                    "the organ filled the church .",
                    "the music was loud .",
                    "{name} pressed the keys .",
                ],
            },
        },
    },
    "star": {
        "word": "star",
        "mfs": "star.n.01",
        "senses": {
            "A": {
                "synset": "star.n.01",
                "answer": "sky",
                "anchor": "{name} looked at the star .",
                "distractors": [
                    "the sky was dark and clear .",
                    "the light took years to arrive .",
                    "{name} pointed at the sky .",
                ],
            },
            "B": {
                "synset": "star.n.04",
                "answer": "actor",
                "anchor": "{name} watched the star .",
                "distractors": [
                    "the actor took a bow .",
                    "the audience clapped loudly .",
                    "{name} read the movie poster .",
                ],
            },
        },
    },
    "spring": {
        "word": "spring",
        "mfs": "spring.n.01",
        "senses": {
            "A": {
                "synset": "spring.n.01",
                "answer": "season",
                "anchor": "{name} waited for spring .",
                "distractors": [
                    "the flowers began to bloom .",
                    "the snow finally melted .",
                    "the days grew warmer .",
                ],
            },
            "B": {
                "synset": "spring.n.02",
                "answer": "coil",
                "anchor": "{name} pressed on the spring .",
                "distractors": [
                    "the coil bounced right back .",
                    "the mattress felt bouncy .",
                    "{name} adjusted the mechanism .",
                ],
            },
        },
    },
    "seal": {
        "word": "seal",
        "mfs": "sealing_wax.n.01",
        "senses": {
            "A": {
                "synset": "seal.n.09",
                "answer": "ocean",
                "anchor": "{name} saw a seal on the rocks .",
                "distractors": [
                    "the ocean waves crashed nearby .",
                    "{name} watched it dive underwater .",
                    "the fur looked wet and sleek .",
                ],
            },
            "B": {
                "synset": "seal.n.02",
                "answer": "stamp",
                "anchor": "{name} pressed the seal into the wax .",
                "distractors": [
                    "the stamp marked the letter .",
                    "the wax was still warm .",
                    "{name} sealed the envelope .",
                ],
            },
        },
    },
    "bark": {
        "word": "bark",
        "mfs": "bark.n.01",
        "senses": {
            "A": {
                "synset": "bark.n.01",
                "answer": "tree",
                "anchor": "{name} touched the bark .",
                "distractors": [
                    "the tree had thick roots .",
                    "the wood felt rough .",
                    "{name} peeled a piece off .",
                ],
            },
            "B": {
                "synset": "bark.n.02",
                "answer": "dog",
                "anchor": "{name} heard the bark .",
                "distractors": [
                    "the dog ran to the door .",
                    "the sound was loud and sudden .",
                    "{name} looked outside .",
                ],
            },
        },
    },
    "crane": {
        "word": "crane",
        "mfs": "crane.n.01",
        "senses": {
            "A": {
                "synset": "crane.n.05",
                "answer": "bird",
                "anchor": "{name} watched the crane .",
                "distractors": [
                    "the bird waded through the marsh .",
                    "its long neck curved gracefully .",
                    "{name} took a photograph .",
                ],
            },
            "B": {
                "synset": "crane.n.04",
                "answer": "machine",
                "anchor": "{name} operated the crane .",
                "distractors": [
                    "the machine lifted the heavy beam .",
                    "the construction site was noisy .",
                    "{name} watched it swing slowly .",
                ],
            },
        },
    },
    "pitcher": {
        "word": "pitcher",
        "mfs": "pitcher.n.01",
        "senses": {
            "A": {
                "synset": "pitcher.n.01",
                "answer": "baseball",
                "anchor": "{name} was the pitcher .",
                "distractors": [
                    "the baseball game began .",
                    "the crowd cheered from the stands .",
                    "{name} threw a fastball .",
                ],
            },
            "B": {
                "synset": "pitcher.n.02",
                "answer": "water",
                "anchor": "{name} filled the pitcher .",
                "distractors": [
                    "the water poured out slowly .",
                    "the glass was nearly full .",
                    "{name} set it on the table .",
                ],
            },
        },
    },
    "mouse": {
        "word": "mouse",
        "mfs": "mouse.n.01",
        "senses": {
            "A": {
                "synset": "mouse.n.01",
                "answer": "rodent",
                "anchor": "{name} saw a mouse .",
                "distractors": [
                    "the rodent scurried under the couch .",
                    "{name} set a small trap .",
                    "the tiny footprints were everywhere .",
                ],
            },
            "B": {
                "synset": "mouse.n.04",
                "answer": "computer",
                "anchor": "{name} clicked the mouse .",
                "distractors": [
                    "the computer screen lit up .",
                    "{name} opened a new window .",
                    "the cursor moved across the screen .",
                ],
            },
        },
    },
    "bass": {
        "word": "bass",
        "mfs": "bass.n.01",
        "senses": {
            "A": {
                "synset": "bass.n.01",
                "answer": "music",
                "anchor": "{name} played the bass .",
                "distractors": [
                    "the music sounded deep and low .",
                    "{name} tuned the instrument .",
                    "the band practiced all afternoon .",
                ],
            },
            "B": {
                "synset": "bass.n.08",
                "answer": "fish",
                "anchor": "{name} caught a bass .",
                "distractors": [
                    "the fish jumped out of the water .",
                    "the lake was calm and still .",
                    "{name} used a small net .",
                ],
            },
        },
    },
    "date": {
        "word": "date",
        "mfs": "date.n.01",
        "senses": {
            "A": {
                "synset": "date.n.01",
                "answer": "calendar",
                "anchor": "{name} wrote down the date .",
                "distractors": [
                    "the calendar hung on the wall .",
                    "the meeting was set for noon .",
                    "{name} circled the day .",
                ],
            },
            "B": {
                "synset": "date.n.08",
                "answer": "fruit",
                "anchor": "{name} ate a date .",
                "distractors": [
                    "the fruit was sweet and sticky .",
                    "the palm tree grew in the desert .",
                    "{name} spit out the pit .",
                ],
            },
        },
    },
    "palm": {
        "word": "palm",
        "mfs": "palm.n.01",
        "senses": {
            "A": {
                "synset": "palm.n.01",
                "answer": "hand",
                "anchor": "{name} looked at the palm .",
                "distractors": [
                    "the hand felt smooth and warm .",
                    "{name} made a fist .",
                    "the fingers curled slowly .",
                ],
            },
            "B": {
                "synset": "palm.n.03",
                "answer": "tree",
                "anchor": "{name} climbed the palm .",
                "distractors": [
                    "the tree swayed in the wind .",
                    "the leaves were long and green .",
                    "{name} found a coconut .",
                ],
            },
        },
    },
    "racket": {
        "word": "racket",
        "mfs": "racket.n.01",
        "senses": {
            "A": {
                "synset": "racket.n.01",
                "answer": "noise",
                "anchor": "{name} heard a racket .",
                "distractors": [
                    "the noise was loud and sudden .",
                    "{name} covered both ears .",
                    "the neighbors complained loudly .",
                ],
            },
            "B": {
                "synset": "racket.n.04",
                "answer": "tennis",
                "anchor": "{name} picked up the racket .",
                "distractors": [
                    "the tennis ball bounced once .",
                    "{name} served the ball .",
                    "the match had just begun .",
                ],
            },
        },
    },
    "pupil": {
        "word": "pupil",
        "mfs": "student.n.01",
        "senses": {
            "A": {
                "synset": "student.n.01",
                "answer": "school",
                "anchor": "{name} was a pupil at the school .",
                "distractors": [
                    "the teacher gave a lesson .",
                    "the classroom was quiet .",
                    "{name} raised a hand .",
                ],
            },
            "B": {
                "synset": "pupil.n.02",
                "answer": "eye",
                "anchor": "the doctor examined the pupil .",
                "distractors": [
                    "the eye was slightly dilated .",
                    "the light was too bright .",
                    "{name} used a small light .",
                ],
            },
        },
    },
    "fan": {
        "word": "fan",
        "mfs": "fan.n.01",
        "senses": {
            "A": {
                "synset": "fan.n.01",
                "answer": "wind",
                "anchor": "{name} turned on the fan .",
                "distractors": [
                    "the wind blew across the room .",
                    "the air felt cooler now .",
                    "{name} felt relieved .",
                ],
            },
            "B": {
                "synset": "sports_fan.n.01",
                "answer": "sports",
                "anchor": "{name} was a big fan .",
                "distractors": [
                    "the sports team scored a goal .",
                    "the crowd cheered loudly .",
                    "{name} wore a team jersey .",
                ],
            },
        },
    },
    "yard": {
        "word": "yard",
        "mfs": "yard.n.01",
        "senses": {
            "A": {
                "synset": "yard.n.01",
                "answer": "length",
                "anchor": "{name} measured one yard .",
                "distractors": [
                    "the length was exact .",
                    "the tape measure was long .",
                    "{name} wrote down the number .",
                ],
            },
            "B": {
                "synset": "yard.n.02",
                "answer": "grass",
                "anchor": "{name} played in the yard .",
                "distractors": [
                    "the grass was green and soft .",
                    "the fence surrounded the house .",
                    "{name} ran across the lawn .",
                ],
            },
        },
    },
    "staff": {
        "word": "staff",
        "mfs": "staff.n.01",
        "senses": {
            "A": {
                "synset": "staff.n.01",
                "answer": "workers",
                "anchor": "{name} hired more staff .",
                "distractors": [
                    "the workers arrived early .",
                    "the office was busy .",
                    "{name} assigned new tasks .",
                ],
            },
            "B": {
                "synset": "staff.n.02",
                "answer": "stick",
                "anchor": "{name} carried a staff .",
                "distractors": [
                    "the stick was tall and sturdy .",
                    "{name} leaned on it while walking .",
                    "the wood was polished smooth .",
                ],
            },
        },
    },
    "nail": {
        "word": "nail",
        "mfs": "nail.n.01",
        "senses": {
            "A": {
                "synset": "nail.n.01",
                "answer": "finger",
                "anchor": "{name} painted the nail .",
                "distractors": [
                    "the finger looked shiny now .",
                    "{name} chose a bright color .",
                    "the polish dried quickly .",
                ],
            },
            "B": {
                "synset": "nail.n.02",
                "answer": "hammer",
                "anchor": "{name} hammered the nail .",
                "distractors": [
                    "the hammer struck hard .",
                    "the wood held firm .",
                    "{name} built a small shelf .",
                ],
            },
        },
    },
    "ball": {
        "word": "ball",
        "mfs": "ball.n.01",
        "senses": {
            "A": {
                "synset": "ball.n.01",
                "answer": "game",
                "anchor": "{name} kicked the ball .",
                "distractors": [
                    "the game was very exciting .",
                    "the team scored a point .",
                    "{name} ran across the field .",
                ],
            },
            "B": {
                "synset": "ball.n.09",
                "answer": "dance",
                "anchor": "{name} attended the ball .",
                "distractors": [
                    "the dance floor was crowded .",
                    "the orchestra played all night .",
                    "{name} wore a fine gown .",
                ],
            },
        },
    },
    "hood": {
        "word": "hood",
        "mfs": "hood.n.01",
        "senses": {
            "A": {
                "synset": "hood.n.09",
                "answer": "car",
                "anchor": "{name} opened the hood .",
                "distractors": [
                    "the engine was still hot .",
                    "the car needed more oil .",
                    "{name} checked the battery .",
                ],
            },
            "B": {
                "synset": "hood.n.08",
                "answer": "head",
                "anchor": "{name} wore a hood .",
                "distractors": [
                    "the fabric covered the head .",
                    "the wind was cold outside .",
                    "{name} pulled the drawstring tight .",
                ],
            },
        },
    },
    "iron": {
        "word": "iron",
        "mfs": "iron.n.01",
        "senses": {
            "A": {
                "synset": "iron.n.01",
                "answer": "metal",
                "anchor": "{name} studied the iron .",
                "distractors": [
                    "the metal was heavy and gray .",
                    "the ore came from the mine .",
                    "{name} tested its strength .",
                ],
            },
            "B": {
                "synset": "iron.n.04",
                "answer": "clothes",
                "anchor": "{name} used the iron .",
                "distractors": [
                    "the clothes were badly wrinkled .",
                    "the shirt looked smooth now .",
                    "{name} set it on the board .",
                ],
            },
        },
    },
    "cell": {
        "word": "cell",
        "mfs": "cell.n.01",
        "senses": {
            "A": {
                "synset": "cell.n.02",
                "answer": "biology",
                "anchor": "{name} studied the cell .",
                "distractors": [
                    "the biology lesson continued .",
                    "the membrane was clearly visible .",
                    "{name} drew a small diagram .",
                ],
            },
            "B": {
                "synset": "cell.n.07",
                "answer": "prison",
                "anchor": "{name} was locked in a cell .",
                "distractors": [
                    "the prison guard walked by .",
                    "the bars felt cold and metal .",
                    "{name} sat on the bench .",
                ],
            },
        },
    },
    "pool": {
        "word": "pool",
        "mfs": "pool.n.01",
        "senses": {
            "A": {
                "synset": "pool.n.01",
                "answer": "swim",
                "anchor": "{name} jumped into the pool .",
                "distractors": [
                    "the water was cool and clear .",
                    "{name} swam several laps .",
                    "the sun felt warm outside .",
                ],
            },
            "B": {
                "synset": "pool.n.09",
                "answer": "billiards",
                "anchor": "{name} played pool .",
                "distractors": [
                    "the billiard balls scattered .",
                    "{name} lined up a shot .",
                    "the table felt smooth .",
                ],
            },
        },
    },
    "court": {
        "word": "court",
        "mfs": "court.n.01",
        "senses": {
            "A": {
                "synset": "court.n.01",
                "answer": "judge",
                "anchor": "{name} appeared in court .",
                "distractors": [
                    "the judge listened carefully .",
                    "the lawyer presented the case .",
                    "{name} answered every question .",
                ],
            },
            "B": {
                "synset": "court.n.04",
                "answer": "game",
                "anchor": "{name} practiced on the court .",
                "distractors": [
                    "the game began at noon .",
                    "{name} scored a basket .",
                    "the crowd applauded loudly .",
                ],
            },
        },
    },
    "bill": {
        "word": "bill",
        "mfs": "bill.n.01",
        "senses": {
            "A": {
                "synset": "bill.n.01",
                "answer": "law",
                "anchor": "{name} proposed a new bill .",
                "distractors": [
                    "the law was debated for hours .",
                    "the senate voted the next day .",
                    "{name} explained the details .",
                ],
            },
            "B": {
                "synset": "beak.n.02",
                "answer": "bird",
                "anchor": "the bird had a long bill .",
                "distractors": [
                    "{name} watched it peck at seeds .",
                    "the feathers were bright blue .",
                    "{name} took a photograph .",
                ],
            },
        },
    },
    "tie": {
        "word": "tie",
        "mfs": "necktie.n.01",
        "senses": {
            "A": {
                "synset": "necktie.n.01",
                "answer": "shirt",
                "anchor": "{name} wore a tie .",
                "distractors": [
                    "the shirt was pressed neatly .",
                    "{name} looked very formal .",
                    "the suit fit well .",
                ],
            },
            "B": {
                "synset": "tie.n.03",
                "answer": "score",
                "anchor": "the game ended in a tie .",
                "distractors": [
                    "the score was perfectly even .",
                    "{name} could not believe it .",
                    "both teams cheered loudly .",
                ],
            },
        },
    },
    "jam": {
        "word": "jam",
        "mfs": "jam.n.01",
        "senses": {
            "A": {
                "synset": "jam.n.01",
                "answer": "bread",
                "anchor": "{name} spread jam on the toast .",
                "distractors": [
                    "the bread tasted very sweet .",
                    "the fruit flavor was strong .",
                    "{name} licked the spoon .",
                ],
            },
            "B": {
                "synset": "fix.n.01",
                "answer": "trouble",
                "anchor": "{name} was in a real jam .",
                "distractors": [
                    "the trouble seemed hard to escape .",
                    "{name} needed help quickly .",
                    "the situation felt tense .",
                ],
            },
        },
    },
    "ring": {
        "word": "ring",
        "mfs": "ring.n.01",
        "senses": {
            "A": {
                "synset": "ring.n.01",
                "answer": "sound",
                "anchor": "{name} heard a ring .",
                "distractors": [
                    "the sound echoed loudly .",
                    "the phone kept buzzing .",
                    "{name} answered right away .",
                ],
            },
            "B": {
                "synset": "ring.n.08",
                "answer": "jewelry",
                "anchor": "{name} wore a ring .",
                "distractors": [
                    "the jewelry sparkled brightly .",
                    "the diamond caught the light .",
                    "{name} admired it closely .",
                ],
            },
        },
    },
    "mole": {
        "word": "mole",
        "mfs": "gram_molecule.n.01",
        "senses": {
            "A": {
                "synset": "mole.n.06",
                "answer": "animal",
                "anchor": "{name} saw a mole in the garden .",
                "distractors": [
                    "the animal dug a small tunnel .",
                    "the dirt piled up near the hole .",
                    "{name} watched it disappear .",
                ],
            },
            "B": {
                "synset": "counterspy.n.01",
                "answer": "spy",
                "anchor": "{name} suspected a mole .",
                "distractors": [
                    "the spy passed secret information .",
                    "the agency launched an investigation .",
                    "{name} reviewed the evidence .",
                ],
            },
        },
    },
}


def ambiguity_episode(rng: random.Random) -> Episode:
    """One ambiguity-bearing episode: 2-4 facts fix ONE sense of a homograph,
    then a question whose correct option depends on that sense.

    Metadata carries the homograph, its gold sense (the synset the CONTEXT
    actually means) and the MFS sense (``wn.synsets(word)[0]``, a fixed
    per-family constant) so evaluation can compare grounding methods without
    a live WordNet call at generation time.
    """
    family_name = rng.choice(list(_AMBIGUITY_FAMILIES))
    fam = _AMBIGUITY_FAMILIES[family_name]
    sense_key = rng.choice(["A", "B"])
    other_key = "B" if sense_key == "A" else "A"
    sense, other = fam["senses"][sense_key], fam["senses"][other_key]
    name = rng.choice(_NAMES)
    k = rng.randint(2, 1 + len(sense["distractors"]))  # anchor + (k-1) distractors, k in [2,4]
    picked_distractors = rng.sample(sense["distractors"], k - 1)
    context = [sense["anchor"]] + picked_distractors
    context = [s.format(name=name) for s in context]
    rng.shuffle(context)
    options = [sense["answer"], other["answer"]]
    rng.shuffle(options)
    answer = sense["answer"]
    return Episode(
        context=context,
        question=f"what kind of {fam['word']} is it ?",
        answer_text=answer,
        options=options,
        answer_idx=options.index(answer),
        level=0,
        meta={
            "family": family_name,
            "homograph": fam["word"],
            "gold_sense": sense["synset"],
            "mfs_sense": fam["mfs"],
            "sense_key": sense_key,
        },
    )


def generate_ambiguity_episodes(n: int, seed: int = 0) -> List[Episode]:
    """``n`` ambiguity-bearing episodes (M32), deterministic given ``seed``.

    Cycles across all homograph families with a single shared RNG stream, so
    two calls with the same ``(n, seed)`` are byte-identical.
    """
    rng = random.Random(seed)
    return [ambiguity_episode(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# bAbI source (real corpus, with offline fallback)
# ---------------------------------------------------------------------------
_BABI_URL = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"


class BabiSource(AbstractEpisodeSource):
    """Loads Facebook bAbI episodes, with a graceful fallback.

    bAbI is exactly the facts-then-question format. Answers are single words;
    we synthesize multiple-choice options from the task's answer vocabulary so
    MC training works too.

    If the data cannot be downloaded or found (common in sandboxed
    environments), :meth:`generate` logs a warning and transparently falls back
    to :class:`CurriculumGenerator` so training is never blocked.

    Args:
        task: bAbI task number (1 = single supporting fact).
        path: Optional path to an already-extracted bAbI ``tasks_1-20_v1-2`` dir
            or a single task file. If ``None``, attempts a download to a cache.
        num_options: Number of MC options to synthesize.
        seed: RNG seed (used for option synthesis and the fallback).
    """

    def __init__(
        self,
        task: int = 1,
        path: Optional[str] = None,
        num_options: int = 4,
        seed: int = 0,
    ) -> None:
        self.task = task
        self.path = path
        self.num_options = num_options
        self.seed = seed
        self.rng = random.Random(seed)

    # -- parsing ------------------------------------------------------------
    @staticmethod
    def parse_babi(lines: List[str]) -> List[Episode]:
        """Parse bAbI-format lines into :class:`Episode` objects (open-ended)."""
        episodes: List[Episode] = []
        context: List[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            num, _, rest = line.partition(" ")
            if num == "1":
                context = []  # new story
            if "\t" in rest:
                question, answer, *_ = rest.split("\t")
                episodes.append(
                    Episode(
                        context=list(context),
                        question=question.strip().lower(),
                        answer_text=answer.strip().lower(),
                        level=0,
                        meta={"source": "babi", "task": "loaded"},
                    )
                )
            else:
                context.append(rest.strip().lower())
        return episodes

    def _add_options(self, episodes: List[Episode]) -> List[Episode]:
        """Synthesize MC options from the corpus answer vocabulary."""
        vocab = sorted({ep.answer_text for ep in episodes})
        if len(vocab) < self.num_options:
            return episodes  # not enough distinct answers; leave open-ended
        for ep in episodes:
            distractors = [a for a in vocab if a != ep.answer_text]
            chosen = self.rng.sample(distractors, self.num_options - 1)
            options = chosen + [ep.answer_text]
            self.rng.shuffle(options)
            ep.options = options
            ep.answer_idx = options.index(ep.answer_text)
        return episodes

    # -- loading ------------------------------------------------------------
    def _candidate_files(self, root: str) -> List[str]:
        suffix = f"qa{self.task}_"
        hits = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.startswith(suffix) and fn.endswith("_train.txt"):
                    hits.append(os.path.join(dirpath, fn))
        return hits

    def _load_lines(self) -> Optional[List[str]]:
        # 1) explicit path
        if self.path and os.path.exists(self.path):
            if os.path.isfile(self.path):
                with open(self.path, encoding="utf-8") as fh:
                    return fh.readlines()
            files = self._candidate_files(self.path)
            if files:
                with open(files[0], encoding="utf-8") as fh:
                    return fh.readlines()
        # 2) attempt download to cache
        try:
            cache = os.path.join(os.path.expanduser("~"), ".cache", "nsm_ct_babi.tar.gz")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            if not os.path.exists(cache):
                urllib.request.urlretrieve(_BABI_URL, cache)  # may be blocked
            with tarfile.open(cache, "r:gz") as tf:
                member = next(
                    m for m in tf.getmembers()
                    if f"qa{self.task}_" in os.path.basename(m.name) and m.name.endswith("_train.txt")
                )
                fh = tf.extractfile(member)
                if fh is not None:
                    return fh.read().decode("utf-8").splitlines()
        except Exception:
            return None
        return None

    def generate(self, n: int) -> List[Episode]:
        lines = self._load_lines()
        if not lines:
            print("[BabiSource] bAbI data unavailable; falling back to CurriculumGenerator.")
            return CurriculumGenerator(seed=self.seed).generate(n)
        episodes = self.parse_babi(lines)
        if not episodes:
            print("[BabiSource] bAbI parse produced nothing; falling back to CurriculumGenerator.")
            return CurriculumGenerator(seed=self.seed).generate(n)
        episodes = self._add_options(episodes)
        # Repeat/truncate to exactly n.
        if len(episodes) >= n:
            return episodes[:n]
        return [episodes[i % len(episodes)] for i in range(n)]


# ---------------------------------------------------------------------------
# Textbook source (north-star stub)
# ---------------------------------------------------------------------------
class TextbookSource(AbstractEpisodeSource):
    """STUB: ingest a textbook chapter as context, answer its homework questions.

    This is the project's north star — "read a textbook and use the homework
    questions". It is intentionally unimplemented: turning real chapter prose
    into a clean context stream and aligning end-of-chapter (often
    multiple-choice) questions is a substantial effort.

    TODO(textbook): implement chapter segmentation -> context stream, and
    homework-question extraction -> Episode (prefer multiple choice).
    """

    def __init__(self, chapter_path: Optional[str] = None) -> None:
        self.chapter_path = chapter_path

    def generate(self, n: int) -> List[Episode]:
        raise NotImplementedError(
            "TextbookSource is a stub for the read-a-textbook / homework-questions "
            "north star. Implement chapter -> context stream and homework -> Episode."
        )


def make_source(name: str, *, seed: int = 0, max_level: int = 3, babi_task: int = 1,
                babi_path: Optional[str] = None) -> AbstractEpisodeSource:
    """Factory: build an episode source by config name."""
    name = name.lower()
    if name == "curriculum":
        return CurriculumGenerator(max_level=max_level, seed=seed)
    if name == "babi":
        return BabiSource(task=babi_task, path=babi_path, seed=seed)
    if name == "textbook":
        return TextbookSource()
    raise ValueError(f"Unknown episode source: {name!r}")


def split_episodes(
    episodes: List[Episode], val_fraction: float, seed: int = 0
):
    """Deterministically split into (train, val). (Moved from the deleted
    token-stack dataset module — splitting is episode plumbing.)"""
    rng = random.Random(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]
