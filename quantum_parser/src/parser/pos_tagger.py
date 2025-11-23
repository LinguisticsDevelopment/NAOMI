"""
Simple POS tagger for automatic word tagging.

Uses a basic rule-based approach with a word dictionary.
For production, integrate spaCy or similar.
"""

from typing import List
from .data_structures import Word
from .enums import Tag


# Simple word → POS tag dictionary (expandable)
WORD_TAG_DICT = {
    # Determiners
    "the": Tag.DET, "a": Tag.DET, "an": Tag.DET,
    "this": Tag.DET, "that": Tag.DET, "these": Tag.DET, "those": Tag.DET,
    "my": Tag.DET, "your": Tag.DET, "his": Tag.DET, "her": Tag.DET,
    "its": Tag.DET, "our": Tag.DET, "their": Tag.DET,

    # Coordinating conjunctions
    "and": Tag.CCONJ, "or": Tag.CCONJ, "but": Tag.CCONJ,
    "so": Tag.CCONJ, "yet": Tag.CCONJ, "for": Tag.CCONJ,

    # Prepositions
    "in": Tag.ADP, "on": Tag.ADP, "at": Tag.ADP, "to": Tag.ADP,
    "from": Tag.ADP, "with": Tag.ADP, "by": Tag.ADP, "about": Tag.ADP,
    "under": Tag.ADP, "over": Tag.ADP, "through": Tag.ADP,
    "into": Tag.ADP, "of": Tag.ADP, "for": Tag.ADP,
    "before": Tag.ADP, "after": Tag.ADP, "between": Tag.ADP,
    "among": Tag.ADP, "during": Tag.ADP, "without": Tag.ADP,
    "within": Tag.ADP, "toward": Tag.ADP, "towards": Tag.ADP,

    # Common adverbs
    "very": Tag.ADV, "quickly": Tag.ADV, "slowly": Tag.ADV,
    "extremely": Tag.ADV, "really": Tag.ADV, "quite": Tag.ADV,
    "too": Tag.ADV, "also": Tag.ADV, "always": Tag.ADV,
    "never": Tag.ADV, "often": Tag.ADV, "sometimes": Tag.ADV,
    "here": Tag.ADV, "there": Tag.ADV, "now": Tag.ADV,

    # Common adjectives
    "big": Tag.ADJ, "small": Tag.ADJ, "red": Tag.ADJ,
    "blue": Tag.ADJ, "green": Tag.ADJ, "yellow": Tag.ADJ,
    "happy": Tag.ADJ, "sad": Tag.ADJ, "good": Tag.ADJ,
    "bad": Tag.ADJ, "new": Tag.ADJ, "old": Tag.ADJ,
    "young": Tag.ADJ, "hot": Tag.ADJ, "cold": Tag.ADJ,
    "tall": Tag.ADJ, "short": Tag.ADJ, "long": Tag.ADJ,
    "shiny": Tag.ADJ, "beautiful": Tag.ADJ, "ugly": Tag.ADJ,

    # Auxiliary verbs (be, have, do as auxiliaries)
    "be": Tag.AUX, "am": Tag.AUX, "is": Tag.AUX, "are": Tag.AUX,
    "was": Tag.AUX, "were": Tag.AUX, "been": Tag.AUX, "being": Tag.AUX,
    "have": Tag.AUX, "has": Tag.AUX, "had": Tag.AUX, "having": Tag.AUX,
    "do": Tag.AUX, "does": Tag.AUX, "did": Tag.AUX,

    # Modal verbs
    "can": Tag.AUX, "could": Tag.AUX,
    "may": Tag.AUX, "might": Tag.AUX,
    "must": Tag.AUX,
    "shall": Tag.AUX, "should": Tag.AUX,
    "will": Tag.AUX, "would": Tag.AUX,

    # Common verbs (infinitive/base form)
    "run": Tag.VERB, "runs": Tag.VERB, "running": Tag.VERB, "ran": Tag.VERB,
    "walk": Tag.VERB, "walks": Tag.VERB, "walked": Tag.VERB,
    "see": Tag.VERB, "sees": Tag.VERB, "saw": Tag.VERB, "seen": Tag.VERB,
    "make": Tag.VERB, "makes": Tag.VERB, "made": Tag.VERB,
    "go": Tag.VERB, "goes": Tag.VERB, "went": Tag.VERB, "gone": Tag.VERB,
    "get": Tag.VERB, "gets": Tag.VERB, "got": Tag.VERB, "gotten": Tag.VERB,
    "give": Tag.VERB, "gives": Tag.VERB, "gave": Tag.VERB, "given": Tag.VERB,
    "take": Tag.VERB, "takes": Tag.VERB, "took": Tag.VERB, "taken": Tag.VERB,
    "chase": Tag.VERB, "chases": Tag.VERB, "chased": Tag.VERB,
    "catch": Tag.VERB, "catches": Tag.VERB, "caught": Tag.VERB,
    "jump": Tag.VERB, "jumps": Tag.VERB, "jumped": Tag.VERB,

    # Common nouns
    "dog": Tag.NOUN, "dogs": Tag.NOUN, "cat": Tag.NOUN, "cats": Tag.NOUN,
    "bird": Tag.NOUN, "birds": Tag.NOUN, "mouse": Tag.NOUN, "mice": Tag.NOUN,
    "man": Tag.NOUN, "men": Tag.NOUN, "woman": Tag.NOUN, "women": Tag.NOUN,
    "child": Tag.NOUN, "children": Tag.NOUN, "boy": Tag.NOUN, "boys": Tag.NOUN,
    "girl": Tag.NOUN, "girls": Tag.NOUN, "person": Tag.NOUN, "people": Tag.NOUN,
    "book": Tag.NOUN, "books": Tag.NOUN, "table": Tag.NOUN, "tables": Tag.NOUN,
    "chair": Tag.NOUN, "chairs": Tag.NOUN, "house": Tag.NOUN, "houses": Tag.NOUN,
    "park": Tag.NOUN, "parks": Tag.NOUN, "sky": Tag.NOUN, "ball": Tag.NOUN,
    "teacher": Tag.NOUN, "telescope": Tag.NOUN, "mat": Tag.NOUN,
}


# Ambiguous words - words that can have multiple POS tags
AMBIGUOUS_WORDS = {
    # Verb/Noun ambiguity
    "book": [Tag.VERB, Tag.NOUN],
    "run": [Tag.VERB, Tag.NOUN],
    "time": [Tag.VERB, Tag.NOUN],
    "duck": [Tag.VERB, Tag.NOUN],
    "light": [Tag.VERB, Tag.NOUN, Tag.ADJ],
    "bear": [Tag.VERB, Tag.NOUN],
    "date": [Tag.VERB, Tag.NOUN],
    "rock": [Tag.VERB, Tag.NOUN],
    "park": [Tag.VERB, Tag.NOUN],
    "saw": [Tag.VERB, Tag.NOUN],
    "watch": [Tag.VERB, Tag.NOUN],
    "train": [Tag.VERB, Tag.NOUN],
    "fly": [Tag.VERB, Tag.NOUN],
    "flies": [Tag.VERB, Tag.NOUN],
    "fish": [Tag.VERB, Tag.NOUN],
    "can": [Tag.AUX, Tag.NOUN],  # Modal or noun (can of soup)
    "will": [Tag.AUX, Tag.NOUN],  # Modal or noun (last will)
    "may": [Tag.AUX, Tag.NOUN],  # Modal or noun (month of May)

    # Be/have/do can be main verbs or auxiliaries
    "be": [Tag.AUX, Tag.VERB],
    "am": [Tag.AUX, Tag.VERB],
    "is": [Tag.AUX, Tag.VERB],
    "are": [Tag.AUX, Tag.VERB],
    "was": [Tag.AUX, Tag.VERB],
    "were": [Tag.AUX, Tag.VERB],
    "been": [Tag.AUX, Tag.VERB],
    "being": [Tag.AUX, Tag.VERB],
    "have": [Tag.AUX, Tag.VERB],
    "has": [Tag.AUX, Tag.VERB],
    "had": [Tag.AUX, Tag.VERB],
    "do": [Tag.AUX, Tag.VERB],
    "does": [Tag.AUX, Tag.VERB],
    "did": [Tag.AUX, Tag.VERB],

    # Verb/Adjective ambiguity
    "close": [Tag.VERB, Tag.ADJ],
    "clean": [Tag.VERB, Tag.ADJ],
    "dry": [Tag.VERB, Tag.ADJ],
    "open": [Tag.VERB, Tag.ADJ],
    "separate": [Tag.VERB, Tag.ADJ],

    # Noun/Adjective ambiguity
    "fast": [Tag.NOUN, Tag.ADJ, Tag.ADV],
    "well": [Tag.NOUN, Tag.ADJ, Tag.ADV],
    "right": [Tag.NOUN, Tag.ADJ, Tag.ADV],
    "left": [Tag.NOUN, Tag.ADJ, Tag.VERB],

    # Preposition/Adverb ambiguity
    "up": [Tag.ADP, Tag.ADV],
    "down": [Tag.ADP, Tag.ADV],
    "out": [Tag.ADP, Tag.ADV],

    # Common pronoun "her" - DET or PRON
    "her": [Tag.DET, Tag.PRON],
}


def get_possible_tags(word: Word) -> List[Tag]:
    """
    Get all possible POS tags for a word.

    Args:
        word: Word object to check

    Returns:
        List of possible POS tags (includes current tag if not in ambiguous dict)
    """
    text_lower = word.text.lower()

    # Check if word is in ambiguous dictionary
    if text_lower in AMBIGUOUS_WORDS:
        return AMBIGUOUS_WORDS[text_lower].copy()

    # Not ambiguous - return current tag as only option
    return [word.pos]


def simple_tag(text: str) -> Tag:
    """
    Tag a single word using simple rules.

    Args:
        text: Word to tag

    Returns:
        POS tag
    """
    text_lower = text.lower()

    # Check dictionary
    if text_lower in WORD_TAG_DICT:
        return WORD_TAG_DICT[text_lower]

    # Simple heuristics
    # Capitalized words (not at start) → Proper noun
    if text[0].isupper() and text not in ["I", "A"]:
        return Tag.PROPN

    # Ends in -ly → probably adverb
    if text_lower.endswith("ly"):
        return Tag.ADV

    # Ends in -ing → verb or adjective
    if text_lower.endswith("ing"):
        return Tag.VERB

    # Ends in -ed → verb
    if text_lower.endswith("ed"):
        return Tag.VERB

    # Ends in -s (but not -ss) → could be verb or plural noun
    if text_lower.endswith("s") and not text_lower.endswith("ss"):
        # Default to noun (plural)
        return Tag.NOUN

    # Default: noun
    return Tag.NOUN


def tag_sentence(sentence: str) -> List[Word]:
    """
    Tag an entire sentence.

    Args:
        sentence: Input sentence as string

    Returns:
        List of Word objects with POS tags
    """
    # Simple tokenization (split on whitespace)
    tokens = sentence.split()

    words = []
    for token in tokens:
        if token.strip():  # Skip empty tokens
            tag = simple_tag(token)
            words.append(Word(token, tag))

    return words


def tag_words(text_list: List[str]) -> List[Word]:
    """
    Tag a list of word strings.

    Args:
        text_list: List of word strings

    Returns:
        List of Word objects with POS tags
    """
    return [Word(text, simple_tag(text)) for text in text_list]


# Optional: Try to import spaCy for better tagging
try:
    import spacy
    _nlp = None

    def tag_sentence_spacy(sentence: str) -> List[Word]:
        """
        Tag sentence using spaCy (if available).

        Args:
            sentence: Input sentence

        Returns:
            List of Word objects
        """
        global _nlp
        if _nlp is None:
            _nlp = spacy.load("en_core_web_sm")

        doc = _nlp(sentence)

        # Map spaCy tags to our Tag enum
        SPACY_TAG_MAP = {
            "DET": Tag.DET,
            "NOUN": Tag.NOUN,
            "PROPN": Tag.PROPN,
            "VERB": Tag.VERB,
            "AUX": Tag.AUX,
            "ADJ": Tag.ADJ,
            "ADV": Tag.ADV,
            "ADP": Tag.ADP,
            "CCONJ": Tag.CCONJ,
            "SCONJ": Tag.SCONJ,
            "PRON": Tag.PRON,
            "NUM": Tag.NUM,
            "PART": Tag.PART,
            "INTJ": Tag.INTJ,
            "PUNCT": Tag.PUNCT,
            "SYM": Tag.SYM,
            "X": Tag.X,
        }

        words = []
        for token in doc:
            tag = SPACY_TAG_MAP.get(token.pos_, Tag.NOUN)
            words.append(Word(token.text, tag))

        return words

    # Use spaCy by default if available
    tag_sentence = tag_sentence_spacy

except ImportError:
    # spaCy not available, use simple tagger
    pass
