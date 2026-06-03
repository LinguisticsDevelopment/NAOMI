"""Toy training data and dataset plumbing.

Phase 1 trains on a tiny **synthetic** reading-comprehension set generated
programmatically. The examples are deliberately, obviously fake (colored
animals, counting fruit) — they are *not* meant to resemble real elementary
comprehension data. Their only job is to make the end-to-end training and
evaluation loops run.

Real data goes through the same :class:`ComprehensionExample` interface; the
``load_*`` hooks at the bottom mark exactly where CommonLit / RACE-easy / etc.
loaders should be plugged in.
"""

from __future__ import annotations

import json
import random
from typing import List

import torch
from torch.utils.data import Dataset

from .config import Config
from .data_structures import ComprehensionExample
from .features import Batch, EncodedExample, FeatureBuilder, collate
from .nsm_primes import PRIME_NAMES
from .tokenizer import SimpleTokenizer

# Parse-tree labels the mock parser can emit (kept in sync with parser_interface).
_PARSE_LABELS = ["S", "CONTENT", "FUNC", "PUNCT", "NUM"]


# ---------------------------------------------------------------------------
# Synthetic toy-data generation
# ---------------------------------------------------------------------------
def _color_example(rng: random.Random) -> ComprehensionExample:
    animals = ["cat", "dog", "bird", "fish", "frog"]
    colors = ["red", "blue", "green", "yellow", "purple"]
    animal = rng.choice(animals)
    color = rng.choice(colors)
    distractors = rng.sample([c for c in colors if c != color], 3)
    options = distractors + [color]
    rng.shuffle(options)
    passage = f"The {animal} is {color}. The {animal} likes to play."
    question = f"What color is the {animal} ?"
    return ComprehensionExample(
        passage=passage,
        question=question,
        options=options,
        answer_idx=options.index(color),
        meta={"template": "color"},
    )


def _count_example(rng: random.Random) -> ComprehensionExample:
    fruits = ["apples", "pears", "plums", "grapes"]
    number_words = ["one", "two", "three", "four", "five", "six"]
    fruit = rng.choice(fruits)
    n = rng.randint(0, len(number_words) - 1)
    correct = number_words[n]
    distractors = rng.sample([w for w in number_words if w != correct], 3)
    options = distractors + [correct]
    rng.shuffle(options)
    passage = f"There are {correct} {fruit} on the table. Nobody ate any {fruit}."
    question = f"How many {fruit} are there ?"
    return ComprehensionExample(
        passage=passage,
        question=question,
        options=options,
        answer_idx=options.index(correct),
        meta={"template": "count"},
    )


def _location_example(rng: random.Random) -> ComprehensionExample:
    objects = ["ball", "book", "cup", "hat"]
    places = ["box", "chair", "shelf", "bed", "floor"]
    obj = rng.choice(objects)
    place = rng.choice(places)
    distractors = rng.sample([p for p in places if p != place], 3)
    options = distractors + [place]
    rng.shuffle(options)
    passage = f"The {obj} is on the {place}. It did not move."
    question = f"Where is the {obj} ?"
    return ComprehensionExample(
        passage=passage,
        question=question,
        options=options,
        answer_idx=options.index(place),
        meta={"template": "location"},
    )


_TEMPLATES = [_color_example, _count_example, _location_example]


def generate_toy_dataset(num_examples: int = 100, seed: int = 0) -> List[ComprehensionExample]:
    """Generate ``num_examples`` obviously-synthetic comprehension examples.

    Args:
        num_examples: How many examples to generate.
        seed: RNG seed for reproducibility.

    Returns:
        A list of :class:`ComprehensionExample`.
    """
    rng = random.Random(seed)
    examples: List[ComprehensionExample] = []
    for i in range(num_examples):
        template = _TEMPLATES[i % len(_TEMPLATES)]
        examples.append(template(rng))
    return examples


def split_examples(
    examples: List[ComprehensionExample], val_fraction: float, seed: int = 0
) -> tuple[List[ComprehensionExample], List[ComprehensionExample]]:
    """Deterministically split into (train, val)."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Tokenizer construction
# ---------------------------------------------------------------------------
def build_tokenizer(examples: List[ComprehensionExample]) -> SimpleTokenizer:
    """Build a tokenizer covering the examples, NSM primes, and parse labels."""
    texts: List[str] = []
    for ex in examples:
        texts.append(ex.passage)
        texts.append(ex.question)
        texts.extend(ex.options)
    extra = list(PRIME_NAMES) + _PARSE_LABELS
    return SimpleTokenizer.build(texts, extra_tokens=extra)


# ---------------------------------------------------------------------------
# Torch Dataset / DataLoader
# ---------------------------------------------------------------------------
class ComprehensionDataset(Dataset):
    """Wraps examples + a :class:`FeatureBuilder`, yielding encoded examples."""

    def __init__(self, examples: List[ComprehensionExample], feature_builder: FeatureBuilder) -> None:
        self.examples = examples
        self.feature_builder = feature_builder

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.feature_builder.build(self.examples[idx])


def make_dataloader(
    dataset: ComprehensionDataset, pad_id: int, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    """Build a DataLoader whose ``collate_fn`` produces a :class:`Batch`."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda items: collate(items, pad_id=pad_id),
    )


# ---------------------------------------------------------------------------
# Real-data hooks (intentionally unimplemented)
# ---------------------------------------------------------------------------
def load_jsonl_examples(path: str) -> List[ComprehensionExample]:
    """Load comprehension examples from a JSONL file.

    Each line must be a JSON object with keys ``passage``, ``question``,
    ``options`` (length-4 list), and ``answer_idx`` (int 0-3). This is the
    canonical on-disk format to convert any real corpus into.
    """
    examples: List[ComprehensionExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                ComprehensionExample(
                    passage=obj["passage"],
                    question=obj["question"],
                    options=obj["options"],
                    answer_idx=int(obj["answer_idx"]),
                    meta=obj.get("meta", {}),
                )
            )
    return examples


def load_commonlit(path: str) -> List[ComprehensionExample]:
    """HOOK: load the CommonLit corpus. Not implemented.

    TODO(real-data): convert CommonLit passages + questions into
    :class:`ComprehensionExample` (4 options, single correct index) and return
    them. See :func:`load_jsonl_examples` for the target format.
    """
    raise NotImplementedError(
        "CommonLit loader not implemented. Convert the corpus to JSONL "
        "(passage/question/options/answer_idx) and use load_jsonl_examples, "
        "or implement the mapping here."
    )


def load_race_easy(path: str) -> List[ComprehensionExample]:
    """HOOK: load the RACE-easy corpus. Not implemented.

    TODO(real-data): RACE already ships 4-option multiple choice, so the mapping
    is mostly mechanical. Convert to JSONL or build examples directly here.
    """
    raise NotImplementedError(
        "RACE-easy loader not implemented. RACE is already 4-way multiple "
        "choice; map its fields onto ComprehensionExample here."
    )
