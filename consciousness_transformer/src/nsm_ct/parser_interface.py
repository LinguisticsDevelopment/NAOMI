"""Abstract syntactic-parser interface plus a mock implementation.

The real NAOMI parser (the Go/Python parsers elsewhere in this repo) would
implement :class:`AbstractParser` and return a :class:`ParseTree`. We do **not**
integrate it here — per the project brief, the parser is mocked behind this
interface so the transformer scaffold can be developed and tested independently.

PLUG-IN POINT: To use the real parser, implement :class:`AbstractParser` with an
adapter that calls NAOMI and converts its output into our :class:`ParseTree`
shape, then pass that adapter wherever :class:`MockNaomiParser` is constructed.
"""

from __future__ import annotations

import abc
from typing import List

from .data_structures import ParseNode, ParseTree
from .tokenizer import basic_tokenize

# A tiny, obviously-fake POS guesser. Real syntax is NOT happening here.
_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "but", "if", "because", "with", "at", "by", "it", "he",
    "she", "they", "this", "that", "?", ".", ",",
}


def _mock_pos(token: str) -> str:
    """Return a fake coarse POS label for a token. Not linguistically real."""
    if token in {".", ",", "?", "!"}:
        return "PUNCT"
    if token in _FUNCTION_WORDS:
        return "FUNC"
    if token.isdigit():
        return "NUM"
    return "CONTENT"


class AbstractParser(abc.ABC):
    """Interface every parser (mock or real) must satisfy."""

    @abc.abstractmethod
    def parse(self, text: str) -> ParseTree:
        """Parse ``text`` and return a :class:`ParseTree`."""
        raise NotImplementedError


class MockNaomiParser(AbstractParser):
    """A stand-in for the real NAOMI parser.

    It produces a flat, one-level tree: a root ``S`` node whose children are one
    leaf per surface token, each tagged with a fake POS label. This is enough to
    exercise serialization and the model's parse-input pathway without pretending
    to do real syntax.

    TODO(parser): replace with a real NAOMI adapter producing genuine hierarchy.
    """

    def parse(self, text: str) -> ParseTree:
        tokens = basic_tokenize(text)
        children = [ParseNode(label=_mock_pos(tok), token=tok) for tok in tokens]
        root = ParseNode(label="S", token=None, children=children)
        return ParseTree(root=root, text=text)
