"""Hand-authored gold: a human-writable spec -> a schema-valid v2 lattice record.

`dev/ENCODER_IO_CONTRACT_V2.md` is the schema this emits. The point of this
module is the **schema-writability gate** (lead, 2026-09-07): the deterministic
teacher can only parse declaratives, so the structures the LEARNED encoder
exists for -- imperatives (synthesized SUBJECT), interjections, elision /
fragments, dropped arguments -- appear in ZERO teacher gold and must be
hand-fabricated. That is only viable if a *human* can author a record.

The division of labour this module implements:

  HUMAN writes                      | MACHINE fills
  ----------------------------------|---------------------------------------
  the surface string                | `tokens`, `pos` (the real parser tagger)
  clause split + `utterance_kind`   | `token_index` (consume-on-match walk)
  predicate + role RELATIONS        | `grounding.type` (sense/entity routing)
  the filler WORD, or               | `candidates` = `usvs.senses_of(lemma)`
    PRIME(...) / CTX(of="...") /    | `token_sense_candidates` (whole table)
    MEM(...)                        | `ref` (content-name -> a real pointer)
  the antecedent BY CONTENT         | the `ctx:ci/ti/slot[/j]` handle strings
  the prior-context sentences       | the context[] lattices (recursively)

Every record this module emits is checked by `check_record` against the ONE
sacred invariant: `encoder_model.linearize_tree` must run on it, and every
oracle action it produces must be admitted by `encoder_model.legal_action_types`
(a masked-out gold action is a -inf logit -> +inf training loss). It also
round-trips: the action sequence is de-linearized back to a clause skeleton and
compared with the authored tree.

Usage as a library:

    from hand_gold import hand_gold_record, C, W, PRIME, CTX, MEM, check_record

    rec = hand_gold_record(
        "come here !",
        [C(kind="imperative", predicate=W("come"),
           roles=[("SUBJECT", PRIME("YOU")), ("PLACE", W("here"))])],
    )
    check_record(rec, usvs)     # raises on any violation
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
_QP_ROOT = _HERE.parent.parent / "quantum_parser"
if str(_QP_ROOT) not in sys.path:
    sys.path.insert(0, str(_QP_ROOT))

from nsm_ct.clause import is_entity  # noqa: E402
from nsm_ct import encoder_model as em  # noqa: E402


# ---------------------------------------------------------------------------
# The human-facing spec vocabulary
# ---------------------------------------------------------------------------

@dataclass
class W:
    """A filler with a SURFACE token. Grounding routed automatically:
    pronoun/name -> `reference`/`entity`; a lemma USVS covers -> `sense` with
    the full candidate list; anything else -> `entity` (contract S4.2's
    explicit "ungrounded content word" case).

    `lemma` overrides which lemma the sense candidates are retrieved for
    (an inflected surface form -- "flew" -- is not a WordNet lemma, so
    `senses_of` returns [] for it). OFF by default: the teacher bulk gold
    retrieves on the raw surface token, and hand gold is UNIONed with it, so
    diverging here would give the encoder two different gtype targets for the
    same shape of token. See dev/HAND_GOLD_DRAFT.md, open decision D5.
    """
    word: str
    lemma: Optional[str] = None
    force_entity: bool = False


@dataclass
class PRIME:
    """A synthesized filler licensed by the grammar to exactly ONE referent --
    the imperative addressee (contract S5). Resolved, no candidate set, no
    surface token. NOTE `encoder_model.PRIMES` currently admits only "YOU";
    anything else trains as `<UNK_PRIME>`."""
    prime: str = "YOU"
    word: Optional[str] = None       # readability only; defaults to prime.lower()


@dataclass
class CTX:
    """An unresolved slot pointing into the record's serialized `context[]`.

    The human names the antecedent BY CONTENT (`of="milk"`, or
    `of=PREDICATE` for the context clause's predicate); this module resolves
    that to a real `(context_index, tree_index, clause, slot, role_index)`
    pointer and builds the readable `ctx:...` handle strings.

    `candidates` is the retrieval-time candidate SET the encoder must emit
    (contract S7 scores set RECALL, never the pick); by default it is every
    context node of the compatible kind. `ref` addresses the gold antecedent.
    """
    gtype: str                        # "elision" | "reference"
    of: Optional[str] = None          # antecedent word, or PREDICATE sentinel
    method: Optional[str] = None      # default derived from gtype/slot
    word: Optional[str] = None        # surface token, if the slot has one
    lemma: Optional[str] = None
    scope: str = "roles"              # candidate set: "roles" | "predicates" | "all" | "clauses"
    context_index: Optional[int] = None   # restrict the search to one context entry


PREDICATE = "\x00PREDICATE"           # `of=PREDICATE` sentinel
CLAUSE = "\x00CLAUSE"                 # `of=CLAUSE` -- propositional anaphora
                                      # (the antecedent is a whole context
                                      # clause, `ref.slot: null`)


@dataclass
class MEM:
    """An unresolved slot whose candidates come from RUN-TIME memory
    (`retrieval.source: "memory"`, `candidates: null`) -- the form the teacher
    already emits for bare pronouns, and the form a hand record uses when the
    antecedent is the speaker / the situation rather than a serialized
    sentence."""
    gtype: str = "reference"
    word: Optional[str] = None
    method: str = "coref"


@dataclass
class C:
    """One clause of the authored tree."""
    predicate: object                 # W | CTX | MEM | None
    roles: List[Tuple[str, object]] = field(default_factory=list)
    kind: str = "proposition"         # utterance_kind
    is_question: bool = False


# ---------------------------------------------------------------------------
# Machine side: tagging, sense retrieval, token_index recovery
# ---------------------------------------------------------------------------

_TAGGERS: Dict[str, object] = {}


def tag(text: str, lang: str = "en"):
    """Tokens + POS from the REAL parser tagger, so a hand record's surface
    fields are byte-identical in provenance to teacher-bulk gold."""
    if lang not in _TAGGERS:
        if lang == "es":
            from src.parser.pos_tagger import tag_spanish_sentence as fn  # type: ignore
        else:
            from src.parser.pos_tagger import tag_sentence as fn  # type: ignore
        _TAGGERS[lang] = fn
    words = _TAGGERS[lang](text)
    return [w.text for w in words], [w.pos.name for w in words]


class _Matcher:
    """Left-to-right consume-on-match token_index recovery -- the same walk
    `scripts/build_encoder_gold_v2.py` and `encoder_model.clause_node_order`
    use, so repeated words land on the same indices."""

    def __init__(self, tokens: Sequence[str]) -> None:
        self._pos: Dict[str, List[int]] = {}
        for i, t in enumerate(tokens):
            self._pos.setdefault(t.lower(), []).append(i)

    def match(self, word: Optional[str]) -> Optional[int]:
        if not word:
            return None
        dq = self._pos.get(word.lower())
        if not dq:
            return None
        return dq.pop(0)


def _sense_grounding(usvs, word: str, lemma: Optional[str]) -> Dict[str, object]:
    key = (lemma or word).lower()
    cands = list(usvs.senses_of(key))
    if cands:
        return {"type": "sense", "candidates": cands,
                "retrieval": {"source": "lexicon", "method": "lemma_senses", "ref": None}}
    return {"type": "entity", "candidates": None}


_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they",
             "him", "her", "them", "us", "me"}


def ground_W(usvs, spec: W) -> Dict[str, object]:
    w = spec.word.lower()
    if spec.force_entity:
        return {"type": "entity", "candidates": None}
    if w in _PRONOUNS:
        # identical to the teacher's `ground_word` for a bare pronoun
        return {"type": "reference", "candidates": None,
                "retrieval": {"source": "memory", "method": "coref", "ref": None}}
    if is_entity(spec.word):
        return {"type": "entity", "candidates": None}
    return _sense_grounding(usvs, spec.word, spec.lemma)


# ---------------------------------------------------------------------------
# context[] -- antecedent addressing by CONTENT
# ---------------------------------------------------------------------------

def _handle(ci: int, ti: int, slot: str, ridx: Optional[int]) -> str:
    return f"ctx:{ci}/{ti}/{slot}" + (f"/{ridx}" if ridx is not None else "")


def _context_nodes(context: Sequence[dict]) -> List[dict]:
    """Flatten every addressable node of a serialized context[] into
    {handle, word, slot, context_index, tree_index, clause, role_index}."""
    out: List[dict] = []
    for ci, entry in enumerate(context):
        for ti, tree in enumerate(entry["lattice"]["trees"]):
            for cj, clause in enumerate(tree["clauses"]):
                out.append({"handle": _handle(ci, ti, "predicate", None),
                            "word": clause.get("predicate"), "slot": "predicate",
                            "context_index": ci, "tree_index": ti, "clause": cj,
                            "role_index": None})
                for rj, role in enumerate(clause["roles"]):
                    out.append({"handle": _handle(ci, ti, "role", rj),
                                "word": role.get("word"), "slot": "role",
                                "context_index": ci, "tree_index": ti, "clause": cj,
                                "role_index": rj})
                out.append({"handle": _handle(ci, ti, "clause", cj), "word": None,
                            "slot": None, "context_index": ci, "tree_index": ti,
                            "clause": cj, "role_index": None})
    return out


_SCOPE_SLOT = {"roles": {"role"}, "predicates": {"predicate"},
               "clauses": {None}, "all": {"role", "predicate"}}


def ground_CTX(usvs, spec: CTX, context: Sequence[dict]) -> Dict[str, object]:
    if not context:
        raise ValueError(f"CTX slot {spec!r} needs a context[]")
    nodes = _context_nodes(context)
    if spec.context_index is not None:
        nodes = [n for n in nodes if n["context_index"] == spec.context_index]
    want = _SCOPE_SLOT[spec.scope]
    pool = [n for n in nodes if n["slot"] in want]
    if not pool:
        raise ValueError(f"CTX slot {spec!r}: empty candidate pool (scope={spec.scope})")

    gold: Optional[dict] = None
    if spec.of == PREDICATE:
        gold = next((n for n in nodes if n["slot"] == "predicate"), None)
        if gold is not None and gold["handle"] not in {n["handle"] for n in pool}:
            pool = pool + [gold]
    elif spec.of == CLAUSE:
        gold = next((n for n in nodes if n["slot"] is None), None)
        if gold is not None and gold["handle"] not in {n["handle"] for n in pool}:
            pool = pool + [gold]
    elif spec.of is not None:
        matches = [n for n in nodes if (n["word"] or "").lower() == spec.of.lower()]
        if not matches:
            raise ValueError(f"CTX slot: antecedent {spec.of!r} not found in context[]")
        gold = matches[0]
        if gold["handle"] not in {n["handle"] for n in pool}:
            pool = pool + [gold]

    method = spec.method or ("elision_inherit_predicate"
                             if (gold and gold["slot"] == "predicate" and spec.gtype == "elision")
                             else "elision_inherit_arg" if spec.gtype == "elision" else "coref")
    ref = {"source": "context",
           "clause": gold["clause"] if gold else None,
           "context_index": gold["context_index"] if gold else None,
           "tree_index": gold["tree_index"] if gold else None,
           "slot": gold["slot"] if gold else None,
           "role_index": gold["role_index"] if gold else None,
           "handle": None}
    return {"type": spec.gtype,
            "candidates": [n["handle"] for n in pool],
            "retrieval": {"source": "context", "method": method, "ref": ref}}


def ground_MEM(spec: MEM) -> Dict[str, object]:
    return {"type": spec.gtype, "candidates": None,
            "retrieval": {"source": "memory", "method": spec.method, "ref": None}}


# ---------------------------------------------------------------------------
# The record builder
# ---------------------------------------------------------------------------

def _filler_word(spec) -> Optional[str]:
    if isinstance(spec, W):
        return spec.word
    if isinstance(spec, PRIME):
        return spec.word if spec.word is not None else spec.prime.lower()
    if isinstance(spec, (CTX, MEM)):
        return spec.word
    return None


def _needs_token_index(spec) -> bool:
    """A PRIME filler never has a surface token (contract S5). CTX/MEM slots
    have one only when the fragment spells the slot out ("me too")."""
    if isinstance(spec, PRIME):
        return False
    return _filler_word(spec) is not None


def _ground(usvs, spec, context: Sequence[dict]) -> Dict[str, object]:
    if isinstance(spec, W):
        return ground_W(usvs, spec)
    if isinstance(spec, PRIME):
        return {"type": "prime", "prime": spec.prime, "candidates": None}
    if isinstance(spec, CTX):
        return ground_CTX(usvs, spec, context)
    if isinstance(spec, MEM):
        return ground_MEM(spec)
    raise TypeError(f"unknown filler spec {spec!r}")


def build_tree(usvs, clauses: Sequence[C], tokens: Sequence[str],
               context: Sequence[dict]) -> Dict[str, object]:
    matcher = _Matcher(tokens)
    out_clauses = []
    for cl in clauses:
        pg = _ground(usvs, cl.predicate, context) if cl.predicate is not None else \
            {"type": "elision", "candidates": None,
             "retrieval": {"source": "memory", "method": "elision_inherit_predicate", "ref": None}}
        pred_word = _filler_word(cl.predicate)
        # the predicate's token_index is not stored on the node (the oracle
        # re-derives it, encoder_model._predicate_token_index) but it MUST be
        # consumed from the matcher here so the roles' indices line up.
        if pg["type"] in ("sense", "entity") and pred_word:
            matcher.match(pred_word)
        roles = []
        for relation, spec in cl.roles:
            word = _filler_word(spec)
            tidx = matcher.match(word) if _needs_token_index(spec) else None
            roles.append({"relation": relation, "word": word, "token_index": tidx,
                          "is_entity": bool(word) and is_entity(word),
                          "grounding": _ground(usvs, spec, context)})
        out_clauses.append({
            "predicate": pred_word if pg["type"] in ("sense", "entity") else None,
            "predicate_grounding": pg,
            "is_question": cl.is_question,
            "utterance_kind": cl.kind,
            "roles": roles,
        })
    return {"clauses": out_clauses}


def build_token_sense_candidates(usvs, tokens: Sequence[str]) -> List[dict]:
    """Identical to the teacher's: `senses_of` on the raw surface token, one
    sparse entry per covered token."""
    out = []
    for i, tok in enumerate(tokens):
        cands = list(usvs.senses_of(tok))
        if cands:
            out.append({"index": i, "token": tok,
                        "sense_candidates": cands, "chosen_sense": cands[0]})
    return out


def context_entry(usvs, text: str, clauses: Sequence[C], lang: str = "en") -> dict:
    """One `context[]` entry: only `text` + `lattice` are required to
    dereference a pointer (contract S4.4); `tokens`/`pos`/
    `token_sense_candidates` are carried too since they are free here."""
    tokens, pos = tag(text, lang)
    tree = build_tree(usvs, clauses, tokens, context=[])
    return {"text": text, "tokens": tokens, "pos": pos,
            "lattice": {"trees": [tree], "discourse_links_per_tree": [[]]},
            "token_sense_candidates": build_token_sense_candidates(usvs, tokens)}


def hand_gold_record(text: str, clauses_spec: Sequence[C], *,
                     usvs, context: Optional[Sequence[dict]] = None,
                     trees_spec: Optional[Sequence[Sequence[C]]] = None,
                     lang: str = "en", meta: Optional[dict] = None) -> dict:
    """text + a human-friendly clause spec -> a v2-contract-valid record.

    `trees_spec` authors a genuinely ambiguous sentence as a multi-tree
    forest; the default is the one-tree (unambiguous) lattice, which is what
    the margin-pruning decision (lead, 2026-09-07) says most sentences should
    carry.
    """
    context = list(context or [])
    tokens, pos = tag(text, lang)
    specs = list(trees_spec) if trees_spec else [list(clauses_spec)]
    trees = [build_tree(usvs, s, tokens, context) for s in specs]
    record = {
        "text": text,
        "tokens": tokens,
        "pos": pos,
        "lattice": {"trees": trees, "discourse_links_per_tree": [[] for _ in trees]},
        "token_sense_candidates": build_token_sense_candidates(usvs, tokens),
    }
    if context:
        record["context"] = context
    if meta:
        record["meta"] = meta
    return record


# ---------------------------------------------------------------------------
# The gate: schema validity + oracle round-trip + action legality
# ---------------------------------------------------------------------------

_GTYPES = set(em.GROUNDING_TYPES)
_SOURCES = set(em.SOURCES)
_KINDS = set(em.CLAUSE_KINDS)
_CORE_ROLES = {"SUBJECT", "OBJECT", "INDIRECT_OBJECT", "PLACE", "SOURCE",
               "AGENT", "RECIPIENT"}


def check_schema(record: dict) -> List[str]:
    """Contract S1-S4 field/type checks. Returns a list of problems."""
    errs: List[str] = []
    for f in ("text", "tokens", "pos", "lattice", "token_sense_candidates"):
        if f not in record:
            errs.append(f"missing top-level field {f!r}")
    if errs:
        return errs
    T = len(record["tokens"])
    if len(record["pos"]) != T:
        errs.append("len(pos) != len(tokens)")
    lat = record["lattice"]
    if not lat.get("trees"):
        errs.append("lattice.trees is empty (contract requires >=1)")
    if len(lat.get("discourse_links_per_tree", [])) != len(lat.get("trees", [])):
        errs.append("discourse_links_per_tree not parallel to trees")

    tsc = {e["index"]: e for e in record["token_sense_candidates"]}
    for e in record["token_sense_candidates"]:
        if not (0 <= e["index"] < T):
            errs.append(f"token_sense_candidates index {e['index']} out of range")
        if e.get("chosen_sense") != (e["sense_candidates"] or [None])[0]:
            errs.append(f"chosen_sense != sense_candidates[0] at {e['index']}")

    ctx = record.get("context", [])
    ctx_handles = {n["handle"] for n in _context_nodes(ctx)} if ctx else set()

    def check_grounding(g: dict, where: str, tidx: Optional[int]) -> None:
        gt = g.get("type")
        if gt not in _GTYPES:
            errs.append(f"{where}: bad grounding.type {gt!r}")
            return
        if gt in ("sense", "reference", "elision"):
            r = g.get("retrieval")
            if not isinstance(r, dict):
                errs.append(f"{where}: type {gt} needs a retrieval object")
                return
            if r.get("source") not in _SOURCES:
                errs.append(f"{where}: bad retrieval.source {r.get('source')!r}")
            if not r.get("method"):
                errs.append(f"{where}: retrieval.method missing")
            if r.get("source") == "context":
                ref = r.get("ref") or {}
                if ref.get("source") != "context":
                    errs.append(f"{where}: ref.source must be 'context'")
                for cand in (g.get("candidates") or []):
                    if cand not in ctx_handles:
                        errs.append(f"{where}: candidate {cand!r} does not dereference")
                h = _handle(ref.get("context_index"), ref.get("tree_index"),
                            ref.get("slot") or "clause", ref.get("role_index"))
                if ref.get("slot") is not None and h not in ctx_handles:
                    errs.append(f"{where}: ref {h!r} does not dereference")
        elif gt == "prime":
            if not g.get("prime"):
                errs.append(f"{where}: type prime needs a prime")
            if g.get("candidates") is not None:
                errs.append(f"{where}: resolved grounding must have candidates=null")
        elif gt == "entity":
            if g.get("candidates") is not None:
                errs.append(f"{where}: entity must have candidates=null")
        if gt == "sense" and tidx is not None:
            authored = g.get("candidates")
            table = tsc.get(tidx, {}).get("sense_candidates")
            if authored is not None and table is not None and authored != table:
                errs.append(f"{where}: node candidates disagree with "
                            f"token_sense_candidates[{tidx}] (contract S4.2)")
            if table is None:
                errs.append(f"{where}: sense slot at token {tidx} has no "
                            f"token_sense_candidates entry")

    for ti, tree in enumerate(lat["trees"]):
        for ci, clause in enumerate(tree["clauses"]):
            if clause.get("utterance_kind", "proposition") not in _KINDS:
                errs.append(f"tree{ti}/clause{ci}: bad utterance_kind")
            pg = clause["predicate_grounding"]
            if pg["type"] in ("sense", "entity") and not clause.get("predicate"):
                errs.append(f"tree{ti}/clause{ci}: grounded predicate with null surface")
            if pg["type"] not in ("sense", "entity") and clause.get("predicate"):
                errs.append(f"tree{ti}/clause{ci}: elided predicate must have "
                            f"predicate=null")
            pidx = em._predicate_token_index(record, clause)
            check_grounding(pg, f"tree{ti}/clause{ci}/predicate", pidx)
            for rj, role in enumerate(clause["roles"]):
                w = f"tree{ti}/clause{ci}/role{rj}({role['relation']})"
                ti_ = role["token_index"]
                if ti_ is not None and not (0 <= ti_ < T):
                    errs.append(f"{w}: token_index out of range")
                if ti_ is not None and role["word"] and \
                        record["tokens"][ti_].lower() != role["word"].lower():
                    errs.append(f"{w}: token_index does not point at word")
                if role["word"] is None and ti_ is not None:
                    errs.append(f"{w}: token_index without a surface word")
                check_grounding(role["grounding"], w, ti_)
    return errs


def replay(record: dict, steps: Sequence[em.Step]) -> List[str]:
    """Replay the oracle derivation through the SAME state machine
    `teacher_force_loss` uses, asserting every action is admitted by the
    legality mask. A masked-out gold action is a -inf logit -> +inf loss:
    the one invariant that must never break."""
    errs: List[str] = []
    T = len(record["tokens"])
    i, open_clause, has_clause = 0, False, False
    for k, step in enumerate(steps):
        legal = em.legal_action_types(open_clause, i, T, has_clause)
        if step.action not in legal:
            errs.append(f"step {k}: {step.action} ILLEGAL at "
                        f"(open={open_clause}, i={i}, T={T}, has_clause={has_clause}); "
                        f"legal={legal}")
            return errs
        if step.action == "OPEN_CLAUSE":
            open_clause = True
        elif step.action == "CLOSE_CLAUSE":
            open_clause, has_clause = False, True
        if step.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and \
                step.token_index is not None:
            i = step.token_index + 1
    if steps and steps[-1].action != "STOP":
        errs.append("derivation does not end in STOP")
    return errs


def _steps_to_skeleton(steps: Sequence[em.Step]) -> List[Tuple]:
    """De-linearize: the action sequence -> the clause skeleton it builds."""
    out, cur, kind = [], None, None
    for step in steps:
        if step.action == "OPEN_CLAUSE":
            cur, kind = [], step.kind
        elif step.action in ("GROUND", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"):
            cur.append((step.role, step.token_index, step.gtype, step.source, step.prime))
        elif step.action == "CLOSE_CLAUSE":
            out.append((kind, tuple(cur)))
            cur, kind = None, None
    return out


def _tree_to_skeleton(record: dict, tree: dict) -> List[Tuple]:
    """The same skeleton read directly off the authored tree."""
    out = []
    for clause in tree["clauses"]:
        nodes = []
        for role, g, tidx in em.clause_node_order(record, clause):
            gt = g["type"]
            src = (g.get("retrieval") or {}).get("source")
            nodes.append((role, tidx, gt, src if gt != "prime" else None,
                          g.get("prime") if gt == "prime" else None))
        out.append((clause.get("utterance_kind", "proposition"), tuple(nodes)))
    return out


def check_record(record: dict, usvs=None, *, strict: bool = True) -> Dict[str, object]:
    """The full gate. Returns a report; raises AssertionError when `strict`
    and anything failed."""
    report: Dict[str, object] = {"text": record["text"], "schema": [], "trees": []}
    report["schema"] = check_schema(record)

    for ti, tree in enumerate(record["lattice"]["trees"]):
        tr: Dict[str, object] = {"tree": ti}
        try:
            steps = em.linearize_tree(record, tree)
        except Exception as exc:  # linearize must not blow up
            tr["linearize_error"] = repr(exc)
            report["trees"].append(tr)
            continue
        tr["n_steps"] = len(steps)
        tr["actions"] = [s.action for s in steps]
        tr["legality"] = replay(record, steps)
        got, want = _steps_to_skeleton(steps), _tree_to_skeleton(record, tree)
        tr["roundtrip"] = [] if got == want else [f"skeleton mismatch:\n  from steps: {got}\n  from tree : {want}"]
        report["trees"].append(tr)

    # context[] entries must themselves be legal records
    for ci, entry in enumerate(record.get("context", [])):
        sub = {"text": entry["text"], "tokens": entry["tokens"], "pos": entry["pos"],
               "lattice": entry["lattice"],
               "token_sense_candidates": entry.get("token_sense_candidates", [])}
        for tree in entry["lattice"]["trees"]:
            errs = replay(sub, em.linearize_tree(sub, tree))
            if errs:
                report.setdefault("context_errors", []).append(f"context[{ci}]: {errs}")

    problems = list(report["schema"]) + list(report.get("context_errors", []))
    for tr in report["trees"]:
        problems += tr.get("legality", []) + tr.get("roundtrip", [])
        if "linearize_error" in tr:
            problems.append(tr["linearize_error"])
    report["ok"] = not problems
    report["problems"] = problems
    if strict and problems:
        raise AssertionError(f"{record['text']!r}: " + "\n  ".join(problems))
    return report


def check_trains(records: Sequence[dict], usvs) -> Dict[str, float]:
    """End-to-end: build features, run the real teacher-forced loss on a
    freshly-initialized policy. A masked-out gold action would make this
    +inf (or NaN); a finite loss is the invariant holding through the ACTUAL
    training path, not just the replay above."""
    import torch

    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes))
    out = {}
    total = 0.0
    for rec in records:
        feats = em.build_features(rec, usvs, pos_vocab, model.hash_buckets)
        steps = em.linearize_tree(rec, rec["lattice"]["trees"][0])
        loss = em.teacher_force_loss(model, feats, steps)
        assert torch.isfinite(loss), f"non-finite teacher-forced loss on {rec['text']!r}"
        v = float(loss.detach())
        out[rec["text"]] = v
        total += v
    out["__mean__"] = total / max(len(records), 1)
    return out


def load_usvs_default():
    from nsm_ct.ground.usvs import load_usvs
    return load_usvs(_HERE.parent / "data" / "usvs")
