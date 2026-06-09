"""NSM Consciousness Transformer — a stateful reasoning-loop scaffold.

A transformer used as a **state-transition function**: it threads an abstract
consciousness state across a stream of input sentences, gates writes into working
memory ("absorb facts as the state calls for it"), and — when a question arrives
— recognizes it and answers from memory. Trained in a "kindergartener" regime on
reasoning episodes (context stream → question → answer).

The genuinely hard pieces (the real NSM/geometric semantic mapper, a consistent
parser, what the consciousness state *means*) are mocked or stubbed behind clean
interfaces and clearly marked. See README.md and RESEARCH_NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agent import Mind, Psyche
from .config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from .data_structures import (
    CausalRelation,
    CausalTable,
    ConsciousnessState,
    ParseNode,
    ParseTree,
)
from .dataset import (
    EpisodeBatch,
    EpisodeDataset,
    build_answer_vocab,
    build_tokenizer,
    collate,
    make_dataloader,
    split_episodes,
)
from .episode import (
    AbstractEpisodeSource,
    BabiSource,
    CurriculumGenerator,
    Episode,
    TextbookSource,
    make_source,
)
from .input_encoder import (
    AbstractInputEncoder,
    ParserInputEncoder,
    TokenInputEncoder,
    make_input_encoder,
)
from .long_term_memory import LongTermMemory
from .losses import LossBreakdown, compute_losses
from .memory import MemoryState, WorkingMemory
from .model import (
    ACTION_ABSORB,
    ACTION_APPEND,
    ACTION_NAMES,
    ACTION_RESPOND,
    ACTION_SKIP,
    ConsciousnessTransformer,
    StepOutput,
)
from .nsm_primes import NUM_PRIMES, PRIMES, PRIME_NAMES, NSMPrime, PrimeCategory
from .parser_interface import AbstractParser, MockNaomiParser
from .tokenizer import SimpleTokenizer
from .wsd import (
    IterativeSenseResolver,
    MockSenseInventory,
    Sense,
    SenseInventory,
    SenseResolver,
    WordNetSenseInventory,
    WSDModule,
)

__all__ = [
    "Config", "ModelConfig", "TrainConfig", "DataConfig", "load_config",
    "Episode", "AbstractEpisodeSource", "CurriculumGenerator", "BabiSource",
    "TextbookSource", "make_source",
    "AbstractInputEncoder", "TokenInputEncoder", "ParserInputEncoder", "make_input_encoder",
    "SimpleTokenizer", "EpisodeBatch", "EpisodeDataset", "collate", "make_dataloader",
    "build_tokenizer", "build_answer_vocab", "split_episodes",
    "WorkingMemory", "MemoryState", "LongTermMemory",
    "ConsciousnessTransformer", "StepOutput",
    "ACTION_ABSORB", "ACTION_APPEND", "ACTION_RESPOND", "ACTION_SKIP", "ACTION_NAMES",
    "Psyche", "Mind", "LossBreakdown", "compute_losses",
    "ParseTree", "ParseNode", "CausalTable", "CausalRelation", "ConsciousnessState",
    "AbstractParser", "MockNaomiParser",
    "PRIMES", "PRIME_NAMES", "NUM_PRIMES", "NSMPrime", "PrimeCategory",
    "Sense", "SenseInventory", "MockSenseInventory", "WordNetSenseInventory",
    "WSDModule", "SenseResolver", "IterativeSenseResolver",
    "Stack", "build_default_stack",
]


@dataclass
class Stack:
    """A fully wired training/inference stack."""

    tokenizer: SimpleTokenizer
    answer_vocab: dict
    encoder: AbstractInputEncoder
    model: ConsciousnessTransformer
    memory: WorkingMemory
    psyche: Psyche
    long_term: object = None  # Optional[LongTermMemory]

    @property
    def mind(self) -> Psyche:  # backward-compat alias (the loop is now "Psyche")
        return self.psyche


def build_default_stack(config: Config, episodes) -> Stack:
    """Assemble tokenizer + input encoder + model + memory + :class:`Mind`.

    Swap ``config.input_encoder`` to ``"parser"`` to feed the experimental
    parser's structure; everything else is unchanged. The hard semantic mapper
    remains out of scope (mocked).

    Args:
        config: Loaded :class:`Config`.
        episodes: Episodes used to build the tokenizer and answer vocabulary.
    """
    tokenizer = build_tokenizer(episodes)
    answer_vocab = build_answer_vocab(episodes)
    encoder = make_input_encoder(config.input_encoder, tokenizer)
    model = ConsciousnessTransformer(tokenizer.vocab_size, len(answer_vocab), config.model)
    memory = WorkingMemory(config.model.memory_dim, config.model.consciousness_dim)
    long_term = None
    if config.model.use_long_term:
        long_term = LongTermMemory(
            config.model.memory_dim, config.model.consciousness_dim,
            max_size=config.model.ltm_max_size,
        )
    psyche = Psyche(
        model, memory, config.data.answer_mode,
        reasoning_hops=config.model.reasoning_hops, long_term=long_term,
        pad_id=tokenizer.pad_id,
    )
    return Stack(
        tokenizer=tokenizer,
        answer_vocab=answer_vocab,
        encoder=encoder,
        model=model,
        memory=memory,
        psyche=psyche,
        long_term=long_term,
    )
