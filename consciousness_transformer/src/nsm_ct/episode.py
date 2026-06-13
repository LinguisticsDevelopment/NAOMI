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
        self.max_level = max(1, min(max_level, 8))
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

    def base_facts(self) -> List[str]:
        """Base 'facts we know about the world' to seed long-term memory."""
        facts = [f"the {p} is a place ." for p in _PLACES]
        facts += [f"{n} is a person ." for n in _NAMES]
        return facts

    def generate(self, n: int) -> List[Episode]:
        builders = [self._level1, self._level2, self._level3, self._level4,
                    self._level5, self._level6, self._level7, self._level8][: self.max_level]
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
