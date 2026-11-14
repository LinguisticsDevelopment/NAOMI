"""NSM Consciousness Transformer — a meaning-space substrate with a learned mind.

The living architecture (see MIND_ARCHITECTURE.md): everything the system holds
and manipulates is a **meaning object** over ~65 NSM primes — never tokens. The
substrate (meaning graph + TPR encoding + grounded lexical space in `ground/`)
is deterministic and inspectable; the only learned parts are small cognitive
policies (`clause_psyche`, `mind/`) that decide how to perceive, recall, infer,
and respond. Knowledge lives in the graph, never in weights.

The original token-stack scaffold (a transformer as state-transition function
over token ids) was removed after the clause/mind line superseded it; it lives
in git history up to tag `token-stack-final` if ever needed.
"""

from __future__ import annotations

from .data_structures import (
    CausalRelation,
    CausalTable,
    ConsciousnessState,
    ParseNode,
    ParseTree,
)
from .input_encoder import (
    AbstractInputEncoder,
    ParserInputEncoder,
    TokenInputEncoder,
)
from .nsm_primes import NUM_PRIMES, PRIMES, PRIME_NAMES, NSMPrime, PrimeCategory
from .tokenizer import SimpleTokenizer

__all__ = [
    'AbstractInputEncoder','TokenInputEncoder','ParserInputEncoder','SimpleTokenizer',
    'ParseTree','ParseNode','CausalTable','CausalRelation','ConsciousnessState',
    'PRIMES','PRIME_NAMES','NUM_PRIMES','NSMPrime','PrimeCategory',
]
