"""NSM Consciousness Transformer — a research scaffold.

A minimal, end-to-end-runnable skeleton for a "consciousness transformer" that
consumes tokenized text, serialized parse trees, a consciousness vector, and
retrieved memory, and emits both a language-model response and a consciousness
state transition. The genuinely hard pieces (semantic mapping onto NSM primes,
the real NAOMI parser, real memory, the real consciousness objective) are
**mocked behind clean interfaces** and clearly marked. See README.md and
RESEARCH_NOTES.md.
"""

from __future__ import annotations

from .config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from .data_structures import (
    CausalRelation,
    CausalTable,
    ComprehensionExample,
    ConsciousnessState,
    ParseNode,
    ParseTree,
)
from .features import Batch, EncodedExample, FeatureBuilder, collate
from .memory import AbstractMemory, MockMemoryStore
from .model import ConsciousnessTransformer, ModelOutput
from .nsm_primes import NUM_PRIMES, PRIMES, PRIME_NAMES, NSMPrime, PrimeCategory
from .parser_interface import AbstractParser, MockNaomiParser
from .semantic_mapper import (
    AbstractSemanticMapper,
    MockSemanticMapper,
    SemanticRepresentation,
)
from .tokenizer import SimpleTokenizer

__all__ = [
    "Config",
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "load_config",
    "ParseTree",
    "ParseNode",
    "CausalTable",
    "CausalRelation",
    "ConsciousnessState",
    "ComprehensionExample",
    "SimpleTokenizer",
    "AbstractParser",
    "MockNaomiParser",
    "AbstractSemanticMapper",
    "MockSemanticMapper",
    "SemanticRepresentation",
    "AbstractMemory",
    "MockMemoryStore",
    "FeatureBuilder",
    "EncodedExample",
    "Batch",
    "collate",
    "ConsciousnessTransformer",
    "ModelOutput",
    "PRIMES",
    "PRIME_NAMES",
    "NUM_PRIMES",
    "NSMPrime",
    "PrimeCategory",
    "build_default_stack",
]


def build_default_stack(config: "Config", examples):
    """Convenience: build a tokenizer + the mock NLP stack + a FeatureBuilder.

    This wires together the default *mock* parser, semantic mapper, and memory
    store. Swap any argument for a real implementation of the matching abstract
    interface to graduate from scaffold to system.

    Args:
        config: Loaded :class:`Config`.
        examples: Examples used to build the tokenizer vocabulary.

    Returns:
        ``(tokenizer, feature_builder)``.
    """
    # Imported here to avoid a heavy import at package load time.
    from .dataset import build_tokenizer

    tokenizer = build_tokenizer(examples)
    parser = MockNaomiParser()
    mapper = MockSemanticMapper()
    memory = MockMemoryStore(dim=config.model.memory_dim)
    feature_builder = FeatureBuilder(
        tokenizer=tokenizer,
        parser=parser,
        semantic_mapper=mapper,
        memory=memory,
        config=config,
    )
    return tokenizer, feature_builder
