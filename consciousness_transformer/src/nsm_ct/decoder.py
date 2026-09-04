"""Phase-1 decoder — rule-grounded short-answer realizer (M65).

``dev/DECODER_DESIGN.md`` §2-4: the decoder consumes the RESOLVED grounded
answer structure comprehension hands off (every sense chosen, every reference
bound) and realizes surface text through exactly two lookups — USVS
``sense_id -> lemma`` and entity ``handle -> name`` — plus the closed-class
words `mind/membrane.py`'s existing ``(subject, relation, value) -> text``
templates already contribute (PLACE / IS_A / CAN*). There is no generative
model and no vocabulary distribution anywhere in this module: every content
word in the output is a value comprehension supplied on the input structure,
so when a grounding is missing there is nothing to look up and the decoder
realizes abstention (design §4.1) — never invented content.
"""

from __future__ import annotations

from typing import Dict, Optional

from .mind import membrane

ABSTAIN_TEXT = "I don't know."

# ATTRIBUTE has no curriculum relation in membrane.RELATION_TEMPLATES (the
# curriculum never asked "what colour"); this is the one new relation the
# decoder adds, in exactly membrane's own "{s} <verb> {v} ." shape.
_ATTRIBUTE_TEMPLATE = "{s} is {v} ."

# Relations membrane.render_fact already realizes forward — reused as-is.
_MEMBRANE_RELATIONS = frozenset(membrane.RELATION_TEMPLATES)


def _entity_name(handle: Optional[str], entity_book: Optional[Dict[str, str]]) -> Optional[str]:
    """entity handle -> its surface name (the entity book, design §3.1).

    ``entity_book`` (handle -> name) lets a caller supply the real book; absent
    one, fall back to clause.py's ``var:<name>`` / ``h_<name>`` handle
    convention (a name is always its own realization — a lookup, not a guess).
    """
    if not handle:
        return None
    if entity_book is not None:
        return entity_book.get(handle)
    name = handle.split(":")[-1]
    if name.startswith("h_"):
        name = name[2:]
    return name.capitalize() if name else None


def _sense_lemma(usvs, sense_id: Optional[str]) -> Optional[str]:
    """sense_id -> its canonical lemma, via USVS's sense_id -> lemmas index.

    The canonical lemma is the sense's first lemma (WordNet MFS-lemma order,
    `ground/usvs.py`'s ``USVS.__post_init__``): ``sense_lemmas[sense_ids.index
    (sense_id)][0]``, read through the already-built ``_sense_index``.
    """
    if usvs is None or not sense_id:
        return None
    idx = usvs._sense_index.get(sense_id)
    if idx is None:
        return None
    lemmas = usvs.sense_lemmas[idx]
    return lemmas[0].replace("_", " ") if lemmas else None


def realize_word(grounding: Optional[Dict], usvs=None,
                  entity_book: Optional[Dict[str, str]] = None) -> Optional[str]:
    """A single RESOLVED grounding -> a surface word, or ``None`` if unrealizable.

    This is the only place a content word is produced (design §3.1/§4.1): the
    key (``sense_id`` / ``handle``) is a value comprehension already put on the
    structure, never one the decoder mints. An unresolved grounding (still
    carrying a ``candidates`` set) or a content grounding with no ``handle``
    (§2.1: required for content) is not realizable and yields ``None`` — the
    caller's job is then to abstain, never to fabricate a substitute.
    """
    if not grounding or grounding.get("candidates") is not None:
        return None
    kind = grounding.get("type")
    handle = grounding.get("handle")
    if kind == "sense":
        return _sense_lemma(usvs, grounding.get("sense_id")) if handle else None
    if kind == "entity":
        return _entity_name(handle, entity_book) if handle else None
    if kind == "prime":
        prime = grounding.get("prime")
        return prime.lower() if prime else None
    return None


def _finish(text: str) -> str:
    """Tidy a rendered fragment for standalone use: trailing ' .' -> '.', cap first letter."""
    text = text.strip()
    if text.endswith(" ."):
        text = text[:-2] + "."
    elif not text.endswith("."):
        text += "."
    return text[:1].upper() + text[1:] if text else text


def _role_by_index(roles, idx):
    if idx is None or not (0 <= idx < len(roles)):
        return None
    return roles[idx]


def _role_by_relation(roles, relation):
    for role in roles:
        if role.get("relation") == relation:
            return role
    return None


def _short_form(role: Dict, usvs, entity_book) -> Optional[str]:
    """The focused constituent's NP alone (design §3.2's short-answer projection)."""
    grounding = role.get("grounding")
    word = realize_word(grounding, usvs, entity_book)
    if word is None:
        return None
    if grounding.get("type") == "entity":
        return _finish(word)                     # proper name: no article
    if role.get("relation") == "PLACE":
        return _finish(f"the {word}")             # common-noun place: "The garden."
    return _finish(word)                          # attribute / other content: "Red."


def _full_form(clause: Dict, usvs, entity_book) -> Optional[str]:
    """The whole clause, realized via the shared (subject, relation, value) templates."""
    roles = clause.get("roles", [])
    subj_role = _role_by_relation(roles, "SUBJECT")
    if subj_role is None:
        return None
    subj_grounding = subj_role.get("grounding")
    subj_word = realize_word(subj_grounding, usvs, entity_book)
    if subj_word is None:
        return None
    if subj_grounding.get("type") == "sense":       # common-noun subject needs its article
        subj_word = f"the {subj_word}"

    value_role = next((r for r in roles if r.get("relation") != "SUBJECT"), None)
    if value_role is None:
        return None
    value_word = realize_word(value_role.get("grounding"), usvs, entity_book)
    if value_word is None:
        return None

    relation = value_role.get("relation")
    if relation == "ATTRIBUTE":
        text = _ATTRIBUTE_TEMPLATE.format(s=subj_word, v=value_word)
    elif relation in _MEMBRANE_RELATIONS:
        text = membrane.render_fact(subj_word, relation, value_word)
    else:
        return None
    return _finish(text)


def realize(answer: Dict, *, usvs=None, entity_book: Optional[Dict[str, str]] = None,
            form: str = "short") -> str:
    """Realize a RESOLVED grounded answer structure (design §2) as surface text.

    ``form`` selects the short-answer projection (default, the K-12 ladder's
    register) or the full clause; both are faithful projections of the same
    resolved structure (design §3.2), never a content decision. Any missing or
    unresolved grounding — including every grounding under the no-confabulation
    ablation (design §4.2), where the memory->decoder link is severed — collapses
    the result to abstention, by construction: there is nothing left to look up.
    """
    kind = answer.get("answer_kind")
    if kind == "abstain":
        return ABSTAIN_TEXT
    if kind == "verdict":
        verdict = answer.get("verdict")
        if verdict == "yes":
            return "Yes."
        if verdict == "no":
            return "No."
        return ABSTAIN_TEXT

    clause = answer.get("answer_clause")
    if not clause:
        return ABSTAIN_TEXT

    if form == "full":
        text = _full_form(clause, usvs, entity_book)
        return text if text is not None else ABSTAIN_TEXT

    roles = clause.get("roles", [])
    focus = answer.get("focus") or {}
    role = None
    if focus.get("slot") == "role":
        role = _role_by_index(roles, focus.get("role_index"))
    if role is None:
        return ABSTAIN_TEXT
    text = _short_form(role, usvs, entity_book)
    return text if text is not None else ABSTAIN_TEXT


__all__ = ["realize", "realize_word", "ABSTAIN_TEXT"]
