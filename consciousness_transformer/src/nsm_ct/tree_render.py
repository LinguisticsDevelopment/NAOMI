"""Compact, LLM-friendly flattening of a parse tree, for the parse judge
(dev/PARSE_JUDGE.md) and for `scripts/probe_encoder_complement.py`'s
human-readable renderer (`render_block`'s per-clause lines factor through
here now).

Two tree shapes exist in this codebase (dev/ENCODER_IO_CONTRACT_V2.md):

- **normalized / decode-time** shape, what `encoder_model.beam_decode`
  returns and what the judge always renders: a clause is
  `{"predicate": {"token_index": int|None, "grounding": {...}}, "roles":
  [{"relation": str, "token_index": int|None, "grounding": {...}}, ...],
  "utterance_kind": str}`.
- **gold lattice** shape, what `runs/encoder_gold_v2.jsonl` records store:
  a clause is `{"predicate": str|None, "predicate_grounding": {...},
  "roles": [{"relation", "word", "token_index", "is_entity", "grounding"}],
  "utterance_kind": str}`.

`normalize_gold_tree` converts the latter to the former (lifted from
`probe_encoder_complement.gold_top1_tree`, generalized to any tree in a
record's forest); `render_tree` only ever consumes the normalized shape.
"""

from __future__ import annotations

from typing import Optional

from .clause import MODIFIER_RELATIONS

MAX_CHARS = 600
MAX_GLOSS_WORDS = 8

_POS_TO_WORDNET = {
    "NOUN": "n", "PROPN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}


def _predicate_token_index(record: dict, clause: dict) -> Optional[int]:
    pred_tok = clause.get("predicate")
    if pred_tok is None:
        return None
    claimed = {r["token_index"] for r in clause["roles"] if r["token_index"] is not None}
    for t, tok in enumerate(record["tokens"]):
        if tok == pred_tok and t not in claimed:
            return t
    return None


def normalize_gold_tree(record: dict, tree: dict) -> dict:
    """Gold-lattice-shaped tree -> the same clause/node shape `beam_decode`
    emits, so `render_tree` (and any other decode-time consumer) applies
    unmodified. `tree` is one entry of `record["lattice"]["trees"]`."""
    clauses = []
    for cl in tree.get("clauses", []):
        pg = cl["predicate_grounding"]
        pred_idx = (_predicate_token_index(record, cl)
                    if pg["type"] in ("sense", "entity") else cl.get("predicate_token_index"))
        pred = {"relation": "PREDICATE", "token_index": pred_idx, "grounding": pg}
        roles = [{"relation": r["relation"], "token_index": r["token_index"], "grounding": r["grounding"]}
                 for r in cl["roles"]]
        clauses.append({"predicate": pred, "roles": roles,
                         "utterance_kind": cl.get("utterance_kind", "proposition")})
    return {"clauses": clauses}


def _gloss_of(sense_id: str) -> Optional[str]:
    try:
        from nltk.corpus import wordnet as wn
        definition = wn.synset(sense_id).definition()
    except Exception:
        return None
    words = definition.split()
    if len(words) > MAX_GLOSS_WORDS:
        definition = " ".join(words[:MAX_GLOSS_WORDS]) + "..."
    return definition


def _first_sense(node: dict, record: dict) -> Optional[str]:
    g = node.get("grounding") or {}
    cands = g.get("candidates")
    if cands:
        return cands[0]
    idx = node.get("token_index")
    if idx is None:
        return None
    for entry in record.get("token_sense_candidates", []):
        if entry["index"] == idx:
            sc = entry.get("sense_candidates")
            return sc[0] if sc else None
    return None


def _render_node(node: dict, record: dict) -> str:
    tokens = record["tokens"]
    g = node.get("grounding") or {}
    gtype = g.get("type")
    idx = node.get("token_index")

    if gtype == "prime":
        return g.get("prime") or "?"
    if gtype == "elision":
        return "<elided>"
    if gtype == "reference":
        tok = tokens[idx] if idx is not None and 0 <= idx < len(tokens) else None
        return f"<ref:{tok}>" if tok else "<ref:pronoun>"

    if idx is None or not (0 <= idx < len(tokens)):
        return "<elided>"
    tok = tokens[idx]

    if gtype == "sense":
        sense = _first_sense(node, record)
        gloss = _gloss_of(sense) if sense else None
        return f"{tok} ({gloss})" if gloss else tok
    # "entity" and any other resolved type: bare surface token
    return tok


def _split_core_modifier(roles: list) -> tuple:
    core = [r for r in roles if r.get("relation") not in MODIFIER_RELATIONS]
    modifiers = [r for r in roles if r.get("relation") in MODIFIER_RELATIONS]
    return core, modifiers


def render_tree(record: dict, tree: dict) -> str:
    """Flatten one normalized-shape tree (see module docstring) to a compact
    string: the sentence, then one `[kind] PREDICATE=... | ROLE=... | ...`
    line per clause, core roles before modifier roles. Capped at ~600 chars
    (`MAX_CHARS`) total."""
    lines = [record["text"]]
    clauses = tree.get("clauses", []) if tree else []
    if not clauses:
        lines.append("<no clauses emitted>")
    for clause in clauses:
        kind = clause.get("utterance_kind", "proposition")
        parts = [f"PREDICATE={_render_node(clause.get('predicate', {}), record)}"]
        core, modifiers = _split_core_modifier(clause.get("roles", []))
        for r in core + modifiers:
            parts.append(f"{r.get('relation', '?')}={_render_node(r, record)}")
        lines.append(f"[{kind}] " + " | ".join(parts))
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS - 3] + "..."
    return text


def node_detail_compact(node: dict, record: dict) -> str:
    """`type:detail` string for the diagnostic (verbose, non-LLM) renderer
    `scripts/probe_encoder_complement.py` uses -- e.g. `sense:thief.n.01`,
    `prime:YOU`, `reference:memory`. Shares `_first_sense` with `render_tree`
    so both renderers agree on "what is the top candidate sense" for a node;
    unlike `render_tree` this reports the raw sense id, not its gloss."""
    g = node.get("grounding") or {}
    gtype = g.get("type")
    if gtype == "sense":
        sense = _first_sense(node, record)
        return f"sense:{sense}" if sense else "sense:?"
    if gtype == "prime":
        return f"prime:{g.get('prime')}"
    if gtype in ("reference", "elision"):
        retrieval = g.get("retrieval") or {}
        source = retrieval.get("source", g.get("source"))
        return f"{gtype}:{source}"
    return str(gtype)


def pos_agrees_with_first_sense(record: dict, token_index: int) -> Optional[bool]:
    """MockJudge heuristic: does the token's coarse POS tag agree with the
    WordNet POS of its first (MFS) sense candidate? Returns None when there
    is no sense candidate or no POS-mapping for this coarse tag (function
    words, PUNCT, etc. -- not evidence either way)."""
    pos_list = record.get("pos", [])
    if token_index is None or not (0 <= token_index < len(pos_list)):
        return None
    coarse = pos_list[token_index]
    wn_pos = _POS_TO_WORDNET.get(coarse)
    if wn_pos is None:
        return None
    entry = next((e for e in record.get("token_sense_candidates", [])
                  if e["index"] == token_index), None)
    if not entry or not entry.get("sense_candidates"):
        return None
    first = entry["sense_candidates"][0]
    try:
        sense_pos = first.split(".")[-2]
    except IndexError:
        return None
    return sense_pos == wn_pos
