"""Extracted from membrane.py for corpus.py (Candidate/CandidateSet/Hypothesis sets + name/pronoun tables)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class Candidate:
    """One membrane candidate: an opaque key (entity-atom name today; a
    sense handle / parse-hypothesis id for M54/M55) plus its STRUCTURAL
    prior. Priors are uniform in v1 (nothing here is learned -- perception
    stays untrained and honest about uncertainty).

    M55a (dev/design/TRACK_C_DESIGN.md Sec 1.10, the "M55 gold program" gap):
    ``query_entity``/``query_relation`` -- a candidate's OWN memory-query
    address. Sec 1.10 found that for a garden-path parse hypothesis, WHICH
    (entity, relation) is even worth querying is itself part of what that
    ONE hypothesis asserts (two readings of "the horse raced past the barn
    fell" disagree about which token plays which role) -- a register kind
    the pre-M55a membrane didn't have (every prior ``mem_query`` use reads
    its ``Addr`` from a clause-level register SHARED by every candidate:
    the mention feature, the sense step's fixed ``rel:SENSE`` address).
    String tokens (grounded downstream by :mod:`nsm_ct.clause_reactor`,
    the same convention as ``key`` itself -- membrane stays torch/codec-
    free), ``None`` for every entity/sense candidate (those keep their
    existing shared-address behavior UNCHANGED; only
    :class:`HypothesisCandidateSet` populates these)."""

    key: str
    prior: float = 1.0
    query_entity: Optional[str] = None
    query_relation: Optional[str] = None



@dataclass
class CandidateSet:
    """A slot perception couldn't resolve deterministically: candidates +
    priors + provenance (which clause/step, which surface token). Kept
    deliberately minimal so M54 (sense candidates) and M55 (parse
    hypotheses) can reuse it unchanged -- only :class:`EntityCandidateSet`
    below is populated in M53a."""

    candidates: List[Candidate] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def keys(self) -> List[str]:
        return [c.key for c in self.candidates]

    @property
    def priors(self) -> np.ndarray:
        return np.array([c.prior for c in self.candidates], dtype=np.float32)


NAME_GENDER: Dict[str, str] = {"mary": "F", "sandra": "F",
                                "john": "M", "daniel": "M", "bill": "M", "fred": "M"}

@dataclass
class HypothesisCandidateSet(CandidateSet):
    """The v1 IN-table "parse hypothesis" row: a garden-path sentence's
    top-K parser hypotheses (:meth:`nsm_ct.input_encoder.ParserInputEncoder.
    _parse_topk_one`), each carrying its OWN (entity, relation) query
    address (``Candidate.query_entity``/``query_relation`` -- see that
    class's M55a docstring) -- WHICH fact is worth checking to decide if
    this reading coheres with memory differs per candidate, unlike
    :class:`EntityCandidateSet`/:class:`SenseCandidateSet` where every
    candidate in a set shares one clause-level address.

    ``gold_index`` is the curriculum's gold reading's position in the
    candidate list (M55a PLACEHOLDER mode only, no resolver yet -- mirrors
    :class:`EntityCandidateSet`/:class:`SenseCandidateSet`'s own
    ``gold_index`` contract exactly)."""

    gold_index: Optional[int] = None


def hypothesis_candidate_set(readings: List[Tuple[str, float, str, str]], *,
                              gold_index: Optional[int] = None,
                              provenance: Optional[Dict[str, object]] = None
                              ) -> HypothesisCandidateSet:
    """M55a's builder, mirroring :func:`pronoun_entity_candidate_set` /
    :func:`sense_candidate_set`'s own factory pattern.

    ``readings`` is ``[(key, prior, query_entity_token, query_relation_token),
    ...]`` -- one 4-tuple per top-K parse hypothesis. Building the ACTUAL
    (entity, relation) address a reading asserts requires interpreting its
    own clause structure, which is task/curriculum-specific (today, only
    :func:`nsm_ct.clause_reactor._garden_path_steps`), so the caller
    supplies it rather than this function re-deriving it -- the same
    division of labor :func:`pronoun_entity_candidate_set` already has with
    its caller over ``gold_antecedent``."""
    candidates = [Candidate(key=key, prior=prior, query_entity=qe, query_relation=qr)
                  for key, prior, qe, qr in readings]
    return HypothesisCandidateSet(candidates=candidates, provenance=dict(provenance or {}),
                                   gold_index=gold_index)

PronounMorphology = Tuple[str, str, str]   # (gender, number, person)

PRONOUN_MORPHOLOGY: Dict[str, PronounMorphology] = {
    # English
    "he":    ("M", "sg", "3"),
    "she":   ("F", "sg", "3"),
    "it":    ("unknown", "sg", "3"),
    "they":  ("unknown", "pl", "3"),
    "him":   ("M", "sg", "3"),
    "her":   ("F", "sg", "3"),
    "them":  ("unknown", "pl", "3"),
    "his":   ("M", "sg", "3"),
    "hers":  ("F", "sg", "3"),
    "their": ("unknown", "pl", "3"),
    # Spanish
    "él":    ("M", "sg", "3"),
    "ella":  ("F", "sg", "3"),
    "ellos": ("M", "pl", "3"),
    "ellas": ("F", "pl", "3"),
    "ello":  ("unknown", "sg", "3"),
    "le":    ("unknown", "sg", "3"),
    "la":    ("F", "sg", "3"),
    "lo":    ("unknown", "sg", "3"),
    "los":   ("M", "pl", "3"),
    "las":   ("F", "pl", "3"),
}
