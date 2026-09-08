"""Render runs/hand_gold_draft.jsonl into the examples section of
dev/HAND_GOLD_DRAFT.md, spliced in place between the BEGIN/END GENERATED
EXAMPLES markers -- so the readable trees can never drift from the records.
The prose around the markers is hand-written and is left untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
IN = _HERE.parent / "runs" / "hand_gold_draft.jsonl"
DOC = _HERE.parent / "dev" / "HAND_GOLD_DRAFT.md"
BEGIN = "<!-- BEGIN GENERATED EXAMPLES -->"
END = "<!-- END GENERATED EXAMPLES -->"

CAT_TITLE = {
    "imperative": "A. IMPERATIVES — a SUBJECT the string does not contain",
    "interjection": "B. INTERJECTIONS — literal/gloss sense + utterance_kind, NO appraisal node",
    "elision": "C. ELISION / FRAGMENTS — a context_ref slot, antecedent named by CONTENT",
    "synth_argument": "D. SYNTHESIZED / DROPPED ARGUMENTS — an absent argument that is not the addressee",
}
CAT_ORDER = ["imperative", "interjection", "elision", "synth_argument"]


def g_str(g: dict) -> str:
    t = g["type"]
    if t == "prime":
        return f"prime:{g['prime']}  [RESOLVED]"
    if t == "entity":
        return "entity  [RESOLVED, no sense]"
    r = g.get("retrieval") or {}
    cands = g.get("candidates")
    if t == "sense":
        n = len(cands or [])
        head = ", ".join((cands or [])[:3]) + (", …" if n > 3 else "")
        return f"sense <- lexicon/{r.get('method')}  ({n} candidates: {head})"
    ref = r.get("ref") or {}
    tgt = "—"
    if ref.get("slot") == "role":
        tgt = f"context[{ref['context_index']}] clause {ref['clause']} role {ref['role_index']}"
    elif ref.get("slot") == "predicate":
        tgt = f"context[{ref['context_index']}] clause {ref['clause']} PREDICATE"
    elif ref.get("source") == "context":
        tgt = f"context[{ref['context_index']}] clause {ref['clause']} (WHOLE CLAUSE)"
    if r.get("source") == "memory":
        return f"{t} <- memory/{r.get('method')}  (candidates retrieved at run time)"
    return (f"{t} <- context/{r.get('method')}\n"
            f"          candidates: {cands}\n"
            f"          gold ref  : {tgt}")


def clause_str(clause: dict, indent: str = "    ") -> str:
    lines = [f"{indent}clause  utterance_kind={clause['utterance_kind']}  "
             f"is_question={clause['is_question']}"]
    p = clause.get("predicate")
    pshow = repr(p) if p else "(none — elided)"
    lines.append(f"{indent}  PREDICATE  {pshow}")
    lines.append(f"{indent}      {g_str(clause['predicate_grounding'])}")
    for role in clause["roles"]:
        w = role["word"]
        ti = role["token_index"]
        surf = f"{w!r}@{ti}" if ti is not None else (f"{w!r} (NO surface token)" if w else "(NO surface token)")
        lines.append(f"{indent}  {role['relation']:<16} {surf}")
        lines.append(f"{indent}      {g_str(role['grounding'])}")
    return "\n".join(lines)


def render(rec: dict) -> str:
    m = rec["meta"]
    out = [f"#### `{rec['text']}`", ""]
    out.append("```")
    out.append(f"text   : {rec['text']}")
    out.append(f"tokens : {rec['tokens']}")
    out.append(f"pos    : {rec['pos']}")
    for ci, entry in enumerate(rec.get("context", [])):
        out.append(f"context[{ci}]: {entry['text']}")
        for clause in entry["lattice"]["trees"][0]["clauses"]:
            out.append(clause_str(clause, indent="      "))
    out.append("tree:")
    for clause in rec["lattice"]["trees"][0]["clauses"]:
        out.append(clause_str(clause))
    out.append("```")
    out.append("")
    out.append(f"**Why it is a hard case.** {m['why_hard']}")
    out.append("")
    out.append(f"**Design choice.** {m['design_choice']}")
    if m.get("unsure"):
        out.append("")
        out.append("**UNSURE — needs the lead:**")
        for u in m["unsure"]:
            out.append(f"- {u}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    recs = [json.loads(l) for l in IN.open() if l.strip()]
    parts = []
    for cat in CAT_ORDER:
        parts.append(f"### {CAT_TITLE[cat]}\n")
        for rec in recs:
            if rec["meta"]["category"] == cat:
                parts.append(render(rec))
    body = "\n".join(parts)
    doc = DOC.read_text()
    head, _, rest = doc.partition(BEGIN)
    _, _, tail = rest.partition(END)
    if not head or not tail:
        print(f"{DOC}: BEGIN/END GENERATED EXAMPLES markers not found", file=sys.stderr)
        return 1
    DOC.write_text(f"{head}{BEGIN}\n\n{body}\n{END}{tail}")
    print(f"spliced {len(recs)} examples into {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
