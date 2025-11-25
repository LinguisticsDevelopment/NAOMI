"""
Simple POS tagger for automatic word tagging.

Uses a basic rule-based approach with a word dictionary.
For production, integrate spaCy or similar.
"""

from typing import List
from .data_structures import Word
from .enums import Tag, SubType


# Simple word → POS tag dictionary (expandable)
WORD_TAG_DICT = {
    # Determiners
    "the": Tag.DET, "a": Tag.DET, "an": Tag.DET,
    "this": Tag.DET, "that": Tag.DET, "these": Tag.DET, "those": Tag.DET,
    "my": Tag.DET, "your": Tag.DET, "his": Tag.DET, "her": Tag.DET,
    "its": Tag.DET, "our": Tag.DET, "their": Tag.DET,
    "no": Tag.DET, "neither": Tag.DET, "either": Tag.DET,

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
    "hardly": Tag.ADV, "rarely": Tag.ADV, "seldom": Tag.ADV,
    "here": Tag.ADV, "there": Tag.ADV, "now": Tag.ADV,
    "where": Tag.ADV, "when": Tag.ADV, "why": Tag.ADV, "how": Tag.ADV,

    #  Pronouns (including relative pronouns)
    "who": Tag.PRON, "whom": Tag.PRON, "whose": Tag.PRON,
    "which": Tag.PRON, "that": Tag.PRON, "what": Tag.PRON,

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
    "leave": Tag.VERB, "leaves": Tag.VERB, "left": Tag.VERB,
    "eat": Tag.VERB, "eats": Tag.VERB, "ate": Tag.VERB, "eaten": Tag.VERB,
    "fly": Tag.VERB, "flies": Tag.VERB, "flew": Tag.VERB, "flown": Tag.VERB,
    "swim": Tag.VERB, "swims": Tag.VERB, "swam": Tag.VERB,
    "like": Tag.VERB, "likes": Tag.VERB, "liked": Tag.VERB,
    "want": Tag.VERB, "wants": Tag.VERB, "wanted": Tag.VERB,
    "live": Tag.VERB, "lives": Tag.VERB, "lived": Tag.VERB,

    # Particles (negation, infinitive marker, possession, etc.)
    "not": Tag.PART, "n't": Tag.PART, "to": Tag.PART, "'s": Tag.PART,

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
    "water": Tag.NOUN, "place": Tag.NOUN, "tail": Tag.NOUN, "tails": Tag.NOUN,
    "thing": Tag.NOUN, "things": Tag.NOUN, "fun": Tag.NOUN,
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


# Word → SubType dictionary for morphological features
WORD_SUBTYPES = {
    # Modal verbs
    "can": [SubType.MODAL],
    "could": [SubType.MODAL],
    "may": [SubType.MODAL],
    "might": [SubType.MODAL],
    "must": [SubType.MODAL],
    "shall": [SubType.MODAL],
    "should": [SubType.MODAL],
    "will": [SubType.MODAL],
    "would": [SubType.MODAL],

    # Perfect auxiliaries (have)
    "have": [SubType.PERFECT],
    "has": [SubType.PERFECT],
    "had": [SubType.PERFECT],

    # Progressive/passive auxiliaries (be)
    "be": [SubType.PROGRESSIVE],
    "am": [SubType.PROGRESSIVE],
    "is": [SubType.PROGRESSIVE],
    "are": [SubType.PROGRESSIVE],
    "was": [SubType.PROGRESSIVE],
    "were": [SubType.PROGRESSIVE],
    "been": [SubType.PROGRESSIVE],
    "being": [SubType.PROGRESSIVE],

    # Possessive marker
    "'s": [SubType.POSSESSIVE],

    # Infinitive marker (when used as particle)
    "to": [SubType.INFINITIVE],

    # Negation markers
    "not": [SubType.NEGATIVE],
    "n't": [SubType.NEGATIVE],
    "no": [SubType.NEGATIVE],

    # Relative pronouns
    "who": [SubType.RELATIVE],
    "whom": [SubType.RELATIVE],
    "whose": [SubType.RELATIVE],
    "which": [SubType.RELATIVE],
    "that": [SubType.RELATIVE],
    "what": [SubType.RELATIVE],

    # Relative adverbs
    "where": [SubType.RELATIVE],
    "when": [SubType.RELATIVE],
    "why": [SubType.RELATIVE],
    "how": [SubType.RELATIVE],

    # Negative adverbs
    "never": [SubType.NEGATIVE],
    "hardly": [SubType.NEGATIVE],
    "rarely": [SubType.NEGATIVE],
    "seldom": [SubType.NEGATIVE],
}


# ========== SPANISH LANGUAGE SUPPORT ==========

# Spanish word → POS tag dictionary
SPANISH_WORD_TAG_DICT = {
    # Determiners (articles)
    "el": Tag.DET, "la": Tag.DET, "los": Tag.DET, "las": Tag.DET,
    "un": Tag.DET, "una": Tag.DET, "unos": Tag.DET, "unas": Tag.DET,
    "este": Tag.DET, "esta": Tag.DET, "estos": Tag.DET, "estas": Tag.DET,
    "ese": Tag.DET, "esa": Tag.DET, "esos": Tag.DET, "esas": Tag.DET,
    "aquel": Tag.DET, "aquella": Tag.DET, "aquellos": Tag.DET, "aquellas": Tag.DET,
    "mi": Tag.DET, "mis": Tag.DET, "tu": Tag.DET, "tus": Tag.DET,
    "su": Tag.DET, "sus": Tag.DET, "nuestro": Tag.DET, "nuestra": Tag.DET,
    "nuestros": Tag.DET, "nuestras": Tag.DET,

    # Common nouns
    "perro": Tag.NOUN, "perros": Tag.NOUN, "gato": Tag.NOUN, "gatos": Tag.NOUN,
    "casa": Tag.NOUN, "casas": Tag.NOUN, "libro": Tag.NOUN, "libros": Tag.NOUN,
    "mesa": Tag.NOUN, "mesas": Tag.NOUN, "silla": Tag.NOUN, "sillas": Tag.NOUN,
    "hombre": Tag.NOUN, "hombres": Tag.NOUN, "mujer": Tag.NOUN, "mujeres": Tag.NOUN,
    "niño": Tag.NOUN, "niños": Tag.NOUN, "niña": Tag.NOUN, "niñas": Tag.NOUN,
    "día": Tag.NOUN, "días": Tag.NOUN, "noche": Tag.NOUN, "noches": Tag.NOUN,
    "coche": Tag.NOUN, "coches": Tag.NOUN, "agua": Tag.NOUN, "comida": Tag.NOUN,
    "parque": Tag.NOUN, "parques": Tag.NOUN, "ciudad": Tag.NOUN, "ciudades": Tag.NOUN,
    "ratón": Tag.NOUN, "ratones": Tag.NOUN, "pájaro": Tag.NOUN, "pájaros": Tag.NOUN,

    # Adjectives (common)
    "grande": Tag.ADJ, "grandes": Tag.ADJ, "pequeño": Tag.ADJ, "pequeña": Tag.ADJ,
    "pequeños": Tag.ADJ, "pequeñas": Tag.ADJ, "blanco": Tag.ADJ, "blanca": Tag.ADJ,
    "blancos": Tag.ADJ, "blancas": Tag.ADJ, "negro": Tag.ADJ, "negra": Tag.ADJ,
    "negros": Tag.ADJ, "negras": Tag.ADJ, "rojo": Tag.ADJ, "roja": Tag.ADJ,
    "rojos": Tag.ADJ, "rojas": Tag.ADJ, "azul": Tag.ADJ, "azules": Tag.ADJ,
    "verde": Tag.ADJ, "verdes": Tag.ADJ, "bueno": Tag.ADJ, "buena": Tag.ADJ,
    "buenos": Tag.ADJ, "buenas": Tag.ADJ, "malo": Tag.ADJ, "mala": Tag.ADJ,
    "malos": Tag.ADJ, "malas": Tag.ADJ, "viejo": Tag.ADJ, "vieja": Tag.ADJ,
    "viejos": Tag.ADJ, "viejas": Tag.ADJ, "nuevo": Tag.ADJ, "nueva": Tag.ADJ,
    "nuevos": Tag.ADJ, "nuevas": Tag.ADJ, "bonito": Tag.ADJ, "bonita": Tag.ADJ,
    "bonitos": Tag.ADJ, "bonitas": Tag.ADJ, "feo": Tag.ADJ, "fea": Tag.ADJ,
    "feos": Tag.ADJ, "feas": Tag.ADJ,

    # Verbs - conjugated forms (ser, estar, ir, tener, hacer, etc.)
    # SER (to be - permanent)
    "soy": Tag.VERB, "eres": Tag.VERB, "es": Tag.VERB, "somos": Tag.VERB, "sois": Tag.VERB, "son": Tag.VERB,
    # ESTAR (to be - temporary)
    "estoy": Tag.VERB, "estás": Tag.VERB, "está": Tag.VERB, "estamos": Tag.VERB, "estáis": Tag.VERB, "están": Tag.VERB,
    # CORRER (to run)
    "corro": Tag.VERB, "corres": Tag.VERB, "corre": Tag.VERB, "corremos": Tag.VERB, "corréis": Tag.VERB, "corren": Tag.VERB,
    # COMER (to eat)
    "como": Tag.VERB, "comes": Tag.VERB, "come": Tag.VERB, "comemos": Tag.VERB, "coméis": Tag.VERB, "comen": Tag.VERB,
    # VER (to see)
    "veo": Tag.VERB, "ves": Tag.VERB, "ve": Tag.VERB, "vemos": Tag.VERB, "veis": Tag.VERB, "ven": Tag.VERB,
    # HACER (to do/make)
    "hago": Tag.VERB, "haces": Tag.VERB, "hace": Tag.VERB, "hacemos": Tag.VERB, "hacéis": Tag.VERB, "hacen": Tag.VERB,
    # TENER (to have)
    "tengo": Tag.VERB, "tienes": Tag.VERB, "tiene": Tag.VERB, "tenemos": Tag.VERB, "tenéis": Tag.VERB, "tienen": Tag.VERB,
    # VIVIR (to live)
    "vivo": Tag.VERB, "vives": Tag.VERB, "vive": Tag.VERB, "vivimos": Tag.VERB, "vivís": Tag.VERB, "viven": Tag.VERB,

    # LAVAR (to wash - reflexive: lavarse)
    "lavo": Tag.VERB, "lavas": Tag.VERB, "lava": Tag.VERB, "lavamos": Tag.VERB, "laváis": Tag.VERB, "lavan": Tag.VERB,
    # LLAMAR (to call - reflexive: llamarse)
    "llamo": Tag.VERB, "llamas": Tag.VERB, "llama": Tag.VERB, "llamamos": Tag.VERB, "llamáis": Tag.VERB, "llaman": Tag.VERB,
    # LEVANTAR (to get up - reflexive: levantarse)
    "levanto": Tag.VERB, "levantas": Tag.VERB, "levanta": Tag.VERB, "levantamos": Tag.VERB, "levantáis": Tag.VERB, "levantan": Tag.VERB,

    # Past participles (can be used as adjectives)
    "leído": Tag.VERB, "leída": Tag.VERB, "leídos": Tag.VERB, "leídas": Tag.VERB,  # read
    "construido": Tag.VERB, "construida": Tag.VERB, "construidos": Tag.VERB, "construidas": Tag.VERB,  # built
    "escrito": Tag.VERB, "escrita": Tag.VERB, "escritos": Tag.VERB, "escritas": Tag.VERB,  # written
    "hecho": Tag.VERB, "hecha": Tag.VERB, "hechos": Tag.VERB, "hechas": Tag.VERB,  # made/done
    "roto": Tag.VERB, "rota": Tag.VERB, "rotos": Tag.VERB, "rotas": Tag.VERB,  # broken
    "abierto": Tag.VERB, "abierta": Tag.VERB, "abiertos": Tag.VERB, "abiertas": Tag.VERB,  # open
    "cerrado": Tag.VERB, "cerrada": Tag.VERB, "cerrados": Tag.VERB, "cerradas": Tag.VERB,  # closed

    # Adverbs
    "muy": Tag.ADV, "bien": Tag.ADV, "mal": Tag.ADV, "aquí": Tag.ADV, "ahí": Tag.ADV,
    "allí": Tag.ADV, "siempre": Tag.ADV, "nunca": Tag.ADV, "rápido": Tag.ADV,
    "rápidamente": Tag.ADV, "lentamente": Tag.ADV, "ya": Tag.ADV, "también": Tag.ADV,
    "tampoco": Tag.ADV, "mucho": Tag.ADV, "poco": Tag.ADV,

    # Prepositions
    "a": Tag.ADP, "de": Tag.ADP, "en": Tag.ADP, "con": Tag.ADP, "por": Tag.ADP,
    "para": Tag.ADP, "sin": Tag.ADP, "sobre": Tag.ADP, "bajo": Tag.ADP,
    "entre": Tag.ADP, "desde": Tag.ADP, "hasta": Tag.ADP, "hacia": Tag.ADP,

    # Coordinating conjunctions
    "y": Tag.CCONJ, "o": Tag.CCONJ, "pero": Tag.CCONJ, "ni": Tag.CCONJ,

    # Pronouns (subject)
    "yo": Tag.PRON, "tú": Tag.PRON, "él": Tag.PRON, "ella": Tag.PRON,
    "nosotros": Tag.PRON, "nosotras": Tag.PRON, "vosotros": Tag.PRON, "vosotras": Tag.PRON,
    "ellos": Tag.PRON, "ellas": Tag.PRON, "usted": Tag.PRON, "ustedes": Tag.PRON,

    # Reflexive pronouns
    "me": Tag.PRON, "te": Tag.PRON, "se": Tag.PRON, "nos": Tag.PRON, "os": Tag.PRON,
}


# Spanish word → SubType dictionary (gender, number, position, person)
SPANISH_WORD_SUBTYPES = {
    # Determiners - masculine singular
    "el": [SubType.MASCULINE, SubType.SINGULAR],
    "un": [SubType.MASCULINE, SubType.SINGULAR],
    "este": [SubType.MASCULINE, SubType.SINGULAR],
    "ese": [SubType.MASCULINE, SubType.SINGULAR],
    "aquel": [SubType.MASCULINE, SubType.SINGULAR],
    "mi": [SubType.SINGULAR],
    "tu": [SubType.SINGULAR],
    "su": [SubType.SINGULAR],
    "nuestro": [SubType.MASCULINE, SubType.SINGULAR],

    # Determiners - feminine singular
    "la": [SubType.FEMININE, SubType.SINGULAR],
    "una": [SubType.FEMININE, SubType.SINGULAR],
    "esta": [SubType.FEMININE, SubType.SINGULAR],
    "esa": [SubType.FEMININE, SubType.SINGULAR],
    "aquella": [SubType.FEMININE, SubType.SINGULAR],
    "nuestra": [SubType.FEMININE, SubType.SINGULAR],

    # Determiners - masculine plural
    "los": [SubType.MASCULINE, SubType.PLURAL],
    "unos": [SubType.MASCULINE, SubType.PLURAL],
    "estos": [SubType.MASCULINE, SubType.PLURAL],
    "esos": [SubType.MASCULINE, SubType.PLURAL],
    "aquellos": [SubType.MASCULINE, SubType.PLURAL],
    "mis": [SubType.PLURAL],
    "tus": [SubType.PLURAL],
    "sus": [SubType.PLURAL],
    "nuestros": [SubType.MASCULINE, SubType.PLURAL],

    # Determiners - feminine plural
    "las": [SubType.FEMININE, SubType.PLURAL],
    "unas": [SubType.FEMININE, SubType.PLURAL],
    "estas": [SubType.FEMININE, SubType.PLURAL],
    "esas": [SubType.FEMININE, SubType.PLURAL],
    "aquellas": [SubType.FEMININE, SubType.PLURAL],
    "nuestras": [SubType.FEMININE, SubType.PLURAL],

    # Nouns - masculine singular
    "perro": [SubType.MASCULINE, SubType.SINGULAR],
    "gato": [SubType.MASCULINE, SubType.SINGULAR],
    "libro": [SubType.MASCULINE, SubType.SINGULAR],
    "hombre": [SubType.MASCULINE, SubType.SINGULAR],
    "niño": [SubType.MASCULINE, SubType.SINGULAR],
    "día": [SubType.MASCULINE, SubType.SINGULAR],
    "coche": [SubType.MASCULINE, SubType.SINGULAR],
    "parque": [SubType.MASCULINE, SubType.SINGULAR],
    "ratón": [SubType.MASCULINE, SubType.SINGULAR],
    "pájaro": [SubType.MASCULINE, SubType.SINGULAR],

    # Nouns - feminine singular
    "casa": [SubType.FEMININE, SubType.SINGULAR],
    "mesa": [SubType.FEMININE, SubType.SINGULAR],
    "silla": [SubType.FEMININE, SubType.SINGULAR],
    "mujer": [SubType.FEMININE, SubType.SINGULAR],
    "niña": [SubType.FEMININE, SubType.SINGULAR],
    "noche": [SubType.FEMININE, SubType.SINGULAR],
    "agua": [SubType.FEMININE, SubType.SINGULAR],
    "comida": [SubType.FEMININE, SubType.SINGULAR],
    "ciudad": [SubType.FEMININE, SubType.SINGULAR],

    # Nouns - masculine plural
    "perros": [SubType.MASCULINE, SubType.PLURAL],
    "gatos": [SubType.MASCULINE, SubType.PLURAL],
    "libros": [SubType.MASCULINE, SubType.PLURAL],
    "hombres": [SubType.MASCULINE, SubType.PLURAL],
    "niños": [SubType.MASCULINE, SubType.PLURAL],
    "días": [SubType.MASCULINE, SubType.PLURAL],
    "coches": [SubType.MASCULINE, SubType.PLURAL],
    "parques": [SubType.MASCULINE, SubType.PLURAL],
    "ratones": [SubType.MASCULINE, SubType.PLURAL],
    "pájaros": [SubType.MASCULINE, SubType.PLURAL],

    # Nouns - feminine plural
    "casas": [SubType.FEMININE, SubType.PLURAL],
    "mesas": [SubType.FEMININE, SubType.PLURAL],
    "sillas": [SubType.FEMININE, SubType.PLURAL],
    "mujeres": [SubType.FEMININE, SubType.PLURAL],
    "niñas": [SubType.FEMININE, SubType.PLURAL],
    "noches": [SubType.FEMININE, SubType.PLURAL],
    "ciudades": [SubType.FEMININE, SubType.PLURAL],

    # Adjectives - masculine singular (post-nominal by default)
    "grande": [SubType.POST_NOMINAL],  # Can be any gender
    "grandes": [SubType.POST_NOMINAL, SubType.PLURAL],
    "pequeño": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "blanco": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "negro": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "rojo": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "azul": [SubType.SINGULAR, SubType.POST_NOMINAL],  # Invariant gender
    "verde": [SubType.SINGULAR, SubType.POST_NOMINAL],  # Invariant gender
    "bueno": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "malo": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "viejo": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "nuevo": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "bonito": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "feo": [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL],

    # Adjectives - feminine singular
    "pequeña": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "blanca": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "negra": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "roja": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "buena": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "mala": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "vieja": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "nueva": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "bonita": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],
    "fea": [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL],

    # Adjectives - masculine plural
    "pequeños": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "blancos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "negros": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "rojos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "azules": [SubType.PLURAL, SubType.POST_NOMINAL],
    "verdes": [SubType.PLURAL, SubType.POST_NOMINAL],
    "buenos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "malos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "viejos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "nuevos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "bonitos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],
    "feos": [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL],

    # Adjectives - feminine plural
    "pequeñas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "blancas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "negras": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "rojas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "buenas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "malas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "viejas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "nuevas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "bonitas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],
    "feas": [SubType.FEMININE, SubType.PLURAL, SubType.POST_NOMINAL],

    # Verbs - SER (to be - permanent) with person/number
    "soy": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "eres": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "es": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "somos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "sois": [SubType.SECOND_PERSON, SubType.PLURAL],
    "son": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - ESTAR (to be - temporary)
    "estoy": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "estás": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "está": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "estamos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "estáis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "están": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - CORRER (to run)
    "corro": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "corres": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "corre": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "corremos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "corréis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "corren": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - COMER (to eat)
    "como": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "comes": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "come": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "comemos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "coméis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "comen": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - VER (to see)
    "veo": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "ves": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "ve": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "vemos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "veis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "ven": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - HACER (to do/make)
    "hago": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "haces": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "hace": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "hacemos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "hacéis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "hacen": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - TENER (to have)
    "tengo": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "tienes": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "tiene": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "tenemos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "tenéis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "tienen": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - VIVIR (to live)
    "vivo": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "vives": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "vive": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "vivimos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "vivís": [SubType.SECOND_PERSON, SubType.PLURAL],
    "viven": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - LAVAR (to wash)
    "lavo": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "lavas": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "lava": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "lavamos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "laváis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "lavan": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - LLAMAR (to call)
    "llamo": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "llamas": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "llama": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "llamamos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "llamáis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "llaman": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Verbs - LEVANTAR (to get up)
    "levanto": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "levantas": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "levanta": [SubType.THIRD_PERSON, SubType.SINGULAR],
    "levantamos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "levantáis": [SubType.SECOND_PERSON, SubType.PLURAL],
    "levantan": [SubType.THIRD_PERSON, SubType.PLURAL],

    # Past participles with gender/number (used as adjectives)
    "leído": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.SINGULAR],
    "leída": [SubType.PARTICIPLE, SubType.FEMININE, SubType.SINGULAR],
    "leídos": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.PLURAL],
    "leídas": [SubType.PARTICIPLE, SubType.FEMININE, SubType.PLURAL],
    "construido": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.SINGULAR],
    "construida": [SubType.PARTICIPLE, SubType.FEMININE, SubType.SINGULAR],
    "construidos": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.PLURAL],
    "construidas": [SubType.PARTICIPLE, SubType.FEMININE, SubType.PLURAL],
    "escrito": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.SINGULAR],
    "escrita": [SubType.PARTICIPLE, SubType.FEMININE, SubType.SINGULAR],
    "escritos": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.PLURAL],
    "escritas": [SubType.PARTICIPLE, SubType.FEMININE, SubType.PLURAL],
    "hecho": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.SINGULAR],
    "hecha": [SubType.PARTICIPLE, SubType.FEMININE, SubType.SINGULAR],
    "hechos": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.PLURAL],
    "hechas": [SubType.PARTICIPLE, SubType.FEMININE, SubType.PLURAL],
    "roto": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.SINGULAR],
    "rota": [SubType.PARTICIPLE, SubType.FEMININE, SubType.SINGULAR],
    "rotos": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.PLURAL],
    "rotas": [SubType.PARTICIPLE, SubType.FEMININE, SubType.PLURAL],
    "abierto": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.SINGULAR],
    "abierta": [SubType.PARTICIPLE, SubType.FEMININE, SubType.SINGULAR],
    "abiertos": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.PLURAL],
    "abiertas": [SubType.PARTICIPLE, SubType.FEMININE, SubType.PLURAL],
    "cerrado": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.SINGULAR],
    "cerrada": [SubType.PARTICIPLE, SubType.FEMININE, SubType.SINGULAR],
    "cerrados": [SubType.PARTICIPLE, SubType.MASCULINE, SubType.PLURAL],
    "cerradas": [SubType.PARTICIPLE, SubType.FEMININE, SubType.PLURAL],

    # Pronouns - subject with person/number
    "yo": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "tú": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "él": [SubType.THIRD_PERSON, SubType.SINGULAR, SubType.MASCULINE],
    "ella": [SubType.THIRD_PERSON, SubType.SINGULAR, SubType.FEMININE],
    "nosotros": [SubType.FIRST_PERSON, SubType.PLURAL, SubType.MASCULINE],
    "nosotras": [SubType.FIRST_PERSON, SubType.PLURAL, SubType.FEMININE],
    "vosotros": [SubType.SECOND_PERSON, SubType.PLURAL, SubType.MASCULINE],
    "vosotras": [SubType.SECOND_PERSON, SubType.PLURAL, SubType.FEMININE],
    "ellos": [SubType.THIRD_PERSON, SubType.PLURAL, SubType.MASCULINE],
    "ellas": [SubType.THIRD_PERSON, SubType.PLURAL, SubType.FEMININE],
    "usted": [SubType.SECOND_PERSON, SubType.SINGULAR],  # Formal "you"
    "ustedes": [SubType.SECOND_PERSON, SubType.PLURAL],  # Formal "you" plural

    # Reflexive pronouns with person/number
    "me": [SubType.FIRST_PERSON, SubType.SINGULAR],
    "te": [SubType.SECOND_PERSON, SubType.SINGULAR],
    "se": [SubType.THIRD_PERSON],  # Can be singular or plural
    "nos": [SubType.FIRST_PERSON, SubType.PLURAL],
    "os": [SubType.SECOND_PERSON, SubType.PLURAL],

    # Negation
    "nunca": [SubType.NEGATIVE],
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
        List of Word objects with POS tags and subtypes
    """
    # Simple tokenization (split on whitespace)
    tokens = sentence.split()

    words = []
    for token in tokens:
        if token.strip():  # Skip empty tokens
            tag = simple_tag(token)

            # Look up subtypes from dictionary
            subtypes = WORD_SUBTYPES.get(token.lower(), []).copy()

            # Add suffix-based subtypes
            token_lower = token.lower()

            # -ing suffix → PARTICIPLE (for present participles)
            if token_lower.endswith("ing") and tag == Tag.VERB:
                if SubType.PARTICIPLE not in subtypes:
                    subtypes.append(SubType.PARTICIPLE)

            words.append(Word(token, tag, subtypes))

    return words


def tag_words(text_list: List[str]) -> List[Word]:
    """
    Tag a list of word strings.

    Args:
        text_list: List of word strings

    Returns:
        List of Word objects with POS tags and subtypes
    """
    words = []
    for text in text_list:
        tag = simple_tag(text)
        subtypes = WORD_SUBTYPES.get(text.lower(), []).copy()

        # Add suffix-based subtypes
        text_lower = text.lower()
        if text_lower.endswith("ing") and tag == Tag.VERB:
            if SubType.PARTICIPLE not in subtypes:
                subtypes.append(SubType.PARTICIPLE)

        words.append(Word(text, tag, subtypes))

    return words


# ========== SPANISH TAGGING FUNCTIONS ==========

def simple_tag_spanish(text: str) -> Tag:
    """
    Tag a single Spanish word using simple rules.

    Args:
        text: Spanish word to tag

    Returns:
        POS tag
    """
    text_lower = text.lower()

    # Check Spanish dictionary
    if text_lower in SPANISH_WORD_TAG_DICT:
        return SPANISH_WORD_TAG_DICT[text_lower]

    # Spanish heuristics
    # Capitalized words → Proper noun
    if text[0].isupper():
        return Tag.PROPN

    # Ends in -mente → adverb (Spanish)
    if text_lower.endswith("mente"):
        return Tag.ADV

    # Ends in -ar, -er, -ir → verb infinitive
    if text_lower.endswith(("ar", "er", "ir")):
        return Tag.VERB

    # Ends in -ando, -iendo → gerund (participle)
    if text_lower.endswith(("ando", "iendo")):
        return Tag.VERB

    # Ends in -ado, -ido → past participle
    if text_lower.endswith(("ado", "ido")):
        return Tag.VERB

    # Default: noun
    return Tag.NOUN


def tag_spanish_sentence(sentence: str) -> List[Word]:
    """
    Tag a Spanish sentence.

    Args:
        sentence: Input Spanish sentence as string

    Returns:
        List of Word objects with POS tags and subtypes
    """
    # Simple tokenization (split on whitespace)
    tokens = sentence.split()

    words = []
    for token in tokens:
        if token.strip():  # Skip empty tokens
            tag = simple_tag_spanish(token)

            # Look up subtypes from Spanish dictionary
            subtypes = SPANISH_WORD_SUBTYPES.get(token.lower(), []).copy()

            # Add suffix-based subtypes for Spanish
            token_lower = token.lower()

            # -ando/-iendo suffix → PARTICIPLE (gerund)
            if token_lower.endswith(("ando", "iendo")) and tag == Tag.VERB:
                if SubType.PARTICIPLE not in subtypes:
                    subtypes.append(SubType.PARTICIPLE)

            # -ado/-ido suffix → PARTICIPLE (past participle)
            if token_lower.endswith(("ado", "ido")) and tag == Tag.VERB:
                if SubType.PARTICIPLE not in subtypes:
                    subtypes.append(SubType.PARTICIPLE)

            words.append(Word(token, tag, subtypes))

    return words


def tag_spanish_words(text_list: List[str]) -> List[Word]:
    """
    Tag a list of Spanish word strings.

    Args:
        text_list: List of Spanish word strings

    Returns:
        List of Word objects with POS tags and subtypes
    """
    words = []
    for text in text_list:
        tag = simple_tag_spanish(text)
        subtypes = SPANISH_WORD_SUBTYPES.get(text.lower(), []).copy()

        # Add suffix-based subtypes
        text_lower = text.lower()
        if text_lower.endswith(("ando", "iendo", "ado", "ido")) and tag == Tag.VERB:
            if SubType.PARTICIPLE not in subtypes:
                subtypes.append(SubType.PARTICIPLE)

        words.append(Word(text, tag, subtypes))

    return words


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
