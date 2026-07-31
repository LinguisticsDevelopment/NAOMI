"""Deterministic coreference for the conversation (M13 stage 3).

The honest, rule-based stand-in for the hard problem: resolve a pronoun
(he/she/it/they/…) to the right antecedent across sentences using **recency +
agreement (gender/number) + grammatical salience (prefer the previous subject)** —
Centering-theory-lite. No learning. Names co-refer trivially (one entity per name);
this fills the pronoun→entity gap that pure recency (``clause.EntityTracker``) gets
wrong when several entities are in play.

Limits (documented, deferred): split antecedents ("Tom and Sue … they"), bridging
("the house … the door"), and world-knowledge coreference are out of scope.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# pronoun → (gender, number): m/f/n gender, s/p number
_PRONOUNS = {
    "he": ("m", "s"), "him": ("m", "s"), "his": ("m", "s"),
    "she": ("f", "s"), "her": ("f", "s"),
    "it": ("n", "s"), "its": ("n", "s"),
    "they": (None, "p"), "them": (None, "p"), "their": (None, "p"),
}
# small name→gender lexicon (covers the curriculum names; extend as needed)
_GENDER = {
    "mary": "f", "sandra": "f", "alice": "f", "sue": "f", "anna": "f", "emma": "f",
    "john": "m", "bill": "m", "fred": "m", "daniel": "m", "tom": "m", "bob": "m",
}


def is_pronoun(tok: str) -> bool:
    return tok in _PRONOUNS


def gender_of(entity: str) -> Optional[str]:
    """Best-effort gender for an entity name (``None`` = unknown, agrees with anything)."""
    return _GENDER.get(entity)


def _agrees(pron_gender: Optional[str], pron_num: str, ent_gender: Optional[str]) -> bool:
    if pron_num == "p":                      # 'they' — number agreement not tracked per-entity
        return True
    if pron_gender is None or ent_gender is None:
        return True                          # unknown on either side → compatible
    return pron_gender == ent_gender


class Coref:
    """Tracks mentioned entities and resolves pronouns over them."""

    def __init__(self) -> None:
        self.mentions: List[Tuple[str, Optional[str], bool]] = []  # (entity, gender, was_subject)

    def register(self, entity: str, *, subject: bool = False) -> None:
        """Record a concrete (non-pronoun, non-variable) entity mention."""
        if is_pronoun(entity) or entity.startswith("?"):
            return
        self.mentions.append((entity, gender_of(entity), subject))

    def resolve(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to the best antecedent, or ``None`` if none is compatible."""
        info = _PRONOUNS.get(pronoun)
        if info is None:
            return None
        pg, pn = info
        cands = [(e, g, subj) for (e, g, subj) in self.mentions if _agrees(pg, pn, g)]
        if not cands:
            return None
        if pg in ("m", "f"):                             # prefer a KNOWN-gender antecedent
            exact = [c for c in cands if c[1] == pg]
            cands = exact or cands
        subjects = [c for c in cands if c[2]]            # salience: prefer the prior subject
        return (subjects[-1] if subjects else cands[-1])[0]


__all__ = ["Coref", "is_pronoun", "gender_of"]
