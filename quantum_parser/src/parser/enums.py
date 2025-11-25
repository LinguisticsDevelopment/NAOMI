"""
Enumerations for parser types.

Based on Universal Dependencies POS tags and custom node/connection types.
"""

from enum import Enum, auto


class Tag(Enum):
    """Part-of-Speech tags (Universal Dependencies)."""

    ADJ = auto()    # Adjective
    ADP = auto()    # Adposition (preposition/postposition)
    ADV = auto()    # Adverb
    AUX = auto()    # Auxiliary verb
    CCONJ = auto()  # Coordinating conjunction
    DET = auto()    # Determiner
    INTJ = auto()   # Interjection
    NOUN = auto()   # Noun
    NUM = auto()    # Numeral
    PART = auto()   # Particle
    PRON = auto()   # Pronoun
    PROPN = auto()  # Proper noun
    PUNCT = auto()  # Punctuation
    SCONJ = auto()  # Subordinating conjunction
    SYM = auto()    # Symbol
    VERB = auto()   # Verb
    X = auto()      # Other
    LOG = auto()    # Logical operator (custom)


class NodeType(Enum):
    """Syntactic node types."""

    NIL = auto()

    # Base types
    NOUN = auto()
    VERBAL = auto()

    # Phrases
    PREDICATE = auto()   # Complete verb phrase
    NOMINAL = auto()     # Noun phrase
    CLAUSE = auto()      # Full clause

    # Modifiers
    SPECIFIER = auto()   # Adverbs, determiners
    DESCRIPTOR = auto()  # Adjectives
    MODIFIER = auto()    # Modals, auxiliaries

    # Coordination/Subordination
    COORD = auto()       # Coordinator (and, or, but)
    SUBOORD = auto()     # Subordinator (because, if, that)

    # Prepositional
    PREP = auto()        # Preposition
    PREP_SPEC = auto()
    PREP_DESC = auto()
    PREP_MIX = auto()
    PP_NOUN = auto()     # Prepositional phrase with noun object
    PP_MIX = auto()
    PP_DESC = auto()
    PP_SPEC = auto()
    PP_VERB = auto()

    # Other
    INTJ = auto()        # Interjection


# Default POS → NodeType mapping
TAG_TO_NODE_TYPE = {
    Tag.ADJ: NodeType.DESCRIPTOR,
    Tag.ADP: NodeType.PREP,
    Tag.ADV: NodeType.SPECIFIER,
    Tag.AUX: NodeType.MODIFIER,
    Tag.CCONJ: NodeType.COORD,
    Tag.DET: NodeType.DESCRIPTOR,
    Tag.INTJ: NodeType.INTJ,
    Tag.NOUN: NodeType.NOUN,
    Tag.NUM: NodeType.DESCRIPTOR,
    Tag.PART: NodeType.MODIFIER,
    Tag.PRON: NodeType.NOUN,
    Tag.PROPN: NodeType.NOUN,
    Tag.PUNCT: NodeType.NIL,
    Tag.SCONJ: NodeType.SUBOORD,
    Tag.SYM: NodeType.NIL,
    Tag.VERB: NodeType.VERBAL,
    Tag.X: NodeType.NIL,
    Tag.LOG: NodeType.COORD,
}


class ConnectionType(Enum):
    """Edge types representing grammatical relationships."""

    # Argument structure
    SUBJECT = auto()              # Nominal → Predicate (doer of action)
    OBJECT = auto()               # Nominal → Verbal (receiver of action)
    INDIRECT_OBJECT = auto()      # Nominal → Verbal (recipient)
    SUBJECT_COMPLEMENT = auto()   # Descriptor/Nominal → Predicate (copula)

    # Modification
    DESCRIPTION = auto()          # Descriptor → Noun
    SPECIFICATION = auto()        # Specifier → Descriptor/Verbal
    MODIFICATION = auto()         # Modifier → various
    COMPLEMENT = auto()           # Predicate complement

    # Coordination
    COORDINATION = auto()         # Coord → coordinated elements

    # Prepositional
    PREPOSITION = auto()          # Simplified preposition connection
    PREPOSITION_FROM = auto()     # Source of PP relationship
    PREPOSITION_TO = auto()       # Target of PP relationship

    # Subordination
    SUBORDINATION = auto()        # Simplified subordination connection
    SUBORDINATION_FROM = auto()   # Main clause
    SUBORDINATION_TO = auto()     # Embedded clause

    # Apposition
    APPOSITION = auto()           # Noun-noun apposition


class SubCat(Enum):
    """Subcategory types for agreement features."""

    CLAUSE = auto()       # Clause type (subordinate, independent)
    VERB = auto()         # Verb form (modal, nominal)
    DESCRIPTOR = auto()   # Descriptor form (comparative, superlative)
    PREPOSITION = auto()  # Preposition/PP form
    GENDER = auto()       # Gender agreement
    NUMBER = auto()       # Number agreement
    QUESTIONS = auto()    # Question markers
    LOGIC = auto()        # Logical operators


class SubType(Enum):
    """Specific morphological/syntactic feature values."""

    # Clause types
    SUBORDINATE = auto()
    INDEPENDENT = auto()

    # Verb forms
    NOMINAL = auto()
    MODAL = auto()
    PERFECT = auto()       # Perfect aspect (have + past participle)
    PROGRESSIVE = auto()   # Progressive aspect (be + present participle)
    PASSIVE = auto()       # Passive voice (be + past participle)

    # Descriptor/Specifier forms
    COMPARATIVE = auto()
    SUPERLATIVE = auto()
    POSSESSIVE = auto()    # Possessive marking
    RELATIVE = auto()      # Relative pronouns/adverbs
    NEGATIVE = auto()      # Negation marking
    INFINITIVE = auto()    # Infinitive marker
    PARTICIPLE = auto()    # Present/past participle

    # Adjective position (for Spanish)
    POST_NOMINAL = auto()   # Adjective comes after noun
    PRE_NOMINAL = auto()    # Adjective comes before noun

    # Preposition/PP forms
    P_MIX = auto()
    P_DESC = auto()
    P_SPEC = auto()
    P_VERB = auto()
    P_NORM = auto()

    # Gender
    MASCULINE = auto()
    FEMININE = auto()
    NEUTER = auto()

    # Number
    SINGULAR = auto()
    PLURAL = auto()

    # Verb person (for conjugation and implied subjects)
    FIRST_PERSON = auto()   # I/we (yo/nosotros)
    SECOND_PERSON = auto()  # you (tú/vosotros/usted)
    THIRD_PERSON = auto()   # he/she/it/they (él/ella/ellos)

    # Questions
    QUESTION = auto()

    # Logical operators
    L_AND = auto()
    L_OR = auto()
    L_XOR = auto()
    L_NAND = auto()
    L_IF = auto()
    L_XIF = auto()
    L_NOT = auto()
    L_NOR = auto()
    L_XNOR = auto()


# SubType → SubCat mapping (what category does each subtype belong to?)
SUBTYPE_TO_SUBCAT = {
    # Clause
    SubType.SUBORDINATE: SubCat.CLAUSE,
    SubType.INDEPENDENT: SubCat.CLAUSE,

    # Verb
    SubType.NOMINAL: SubCat.VERB,
    SubType.MODAL: SubCat.VERB,
    SubType.PERFECT: SubCat.VERB,
    SubType.PROGRESSIVE: SubCat.VERB,
    SubType.PASSIVE: SubCat.VERB,

    # Descriptor
    SubType.COMPARATIVE: SubCat.DESCRIPTOR,
    SubType.SUPERLATIVE: SubCat.DESCRIPTOR,
    SubType.POSSESSIVE: SubCat.DESCRIPTOR,
    SubType.NEGATIVE: SubCat.DESCRIPTOR,
    SubType.RELATIVE: SubCat.DESCRIPTOR,
    SubType.INFINITIVE: SubCat.DESCRIPTOR,
    SubType.PARTICIPLE: SubCat.DESCRIPTOR,

    # Preposition
    SubType.P_MIX: SubCat.PREPOSITION,
    SubType.P_DESC: SubCat.PREPOSITION,
    SubType.P_SPEC: SubCat.PREPOSITION,
    SubType.P_VERB: SubCat.PREPOSITION,
    SubType.P_NORM: SubCat.PREPOSITION,

    # Gender
    SubType.MASCULINE: SubCat.GENDER,
    SubType.FEMININE: SubCat.GENDER,
    SubType.NEUTER: SubCat.GENDER,

    # Number
    SubType.SINGULAR: SubCat.NUMBER,
    SubType.PLURAL: SubCat.NUMBER,

    # Questions
    SubType.QUESTION: SubCat.QUESTIONS,

    # Logic
    SubType.L_AND: SubCat.LOGIC,
    SubType.L_OR: SubCat.LOGIC,
    SubType.L_XOR: SubCat.LOGIC,
    SubType.L_NAND: SubCat.LOGIC,
    SubType.L_IF: SubCat.LOGIC,
    SubType.L_XIF: SubCat.LOGIC,
    SubType.L_NOT: SubCat.LOGIC,
    SubType.L_NOR: SubCat.LOGIC,
    SubType.L_XNOR: SubCat.LOGIC,
}


def get_subcat(subtype: SubType) -> SubCat:
    """Get the subcategory for a given subtype."""
    return SUBTYPE_TO_SUBCAT.get(subtype)
