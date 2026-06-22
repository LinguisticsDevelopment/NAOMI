"""Respond with thinking — verbalize the *actual* derivation (M6).

Unlike LLM chain-of-thought, this is **faithful by construction**: the explanation
is rendered from the `DerivStep` provenance the M2 executor / M4 conscious-loop
already produce, so "Because a robin is a bird and a bird can fly, a robin can fly"
is the *real* derivation, not a plausible story. Knowledge is rendered through the
membrane; nothing is invented here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import membrane, ops


def verbalize_answer(query: Tuple[str, str], answer: Optional[str]) -> str:
    """Render the answer to ``(subject, relation)`` as a sentence (or abstention)."""
    s, r = query
    if answer is None or answer == ops.ABSTAIN:
        return "I don't know."
    if answer == "maybe":
        return "Maybe."
    return _cap(membrane.render_fact(s, r, answer))


def verbalize_verdict(answer: Optional[str]) -> str:
    """Render a yes/no verdict (``true``/``false``/``Unknown``/abstain) as a reply."""
    if answer == "true":
        return "Yes."
    if answer == "false":
        return "No."
    return "I don't know."


def _relevant_steps(support: List, target) -> List:
    """The minimal sub-chain that actually derives ``target`` (backward trace).

    Keeps only the ``DerivStep``s on the derivation path to the answer — still 100%
    faithful (a subset of real steps), just without the irrelevant closure branches.
    Ordered premises-first.
    """
    by_derived = {}
    for st in support:
        by_derived.setdefault(st.derived, st)
    keep, seen, stack = [], set(), [target]
    while stack:
        t = stack.pop()
        st = by_derived.get(t)
        if st is None or t in seen:
            continue
        seen.add(t)
        keep.append(st)
        stack.extend(st.support)
    keep.reverse()                                   # answer's step last (premises first)
    return keep


def verbalize_trace(query: Tuple[str, str], answer: Optional[str], support: List) -> str:
    """Verbalize the faithful reasoning chain + the answer.

    Args:
        query: ``(subject, relation)``.
        answer: the derived value (or ``idk``/``None`` for abstain).
        support: the list of ``reasoning_oracle.DerivStep`` that derived the answer
            (the executor's provenance). Pruned to the answer-relevant path; each
            step renders as "Because <support facts>, <derived>".
    """
    steps = support
    if answer not in (None, ops.ABSTAIN, "maybe"):
        steps = _relevant_steps(support, (query[0], query[1], answer)) or support
    lines: List[str] = []
    for step in steps:
        facts = " and ".join(membrane.render_clause(*f) for f in step.support)
        concl = membrane.render_clause(*step.derived)
        lines.append(f"Because {facts}, {concl}")
    reason = ". ".join(lines)
    ans = verbalize_answer(query, answer)
    return f"{_cap(reason)}. {ans}" if lines else ans


def _cap(text: str) -> str:
    """Capitalize the first letter and tidy the trailing ' .' to '.'."""
    text = text.strip()
    if text.endswith(" ."):
        text = text[:-2] + "."
    return text[:1].upper() + text[1:] if text else text


__all__ = ["verbalize_answer", "verbalize_verdict", "verbalize_trace"]
