"""Talk to the mind — natural-language question in, answer + faithful reasoning out.

End-to-end M6 demo: the membrane parses the curriculum's natural language into
meaning objects, the symbolic validator (the M2 executor over the episode's
facts+rules) derives the answer with provenance, and the verbalizer renders the
*actual* DerivStep chain back into text. The "because …" is the real derivation,
not a story.

Run:  python scripts/talk_to_mind.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.mind import membrane, ops, teacher  # noqa: E402
from nsm_ct.mind.verbalize import verbalize_trace  # noqa: E402


def _pick(level: int, seed: int = 7):
    for ep in CurriculumGenerator(max_level=13, seed=seed).generate(400):
        if ep.level == level and ep.answerable:
            return ep
    return None


def main() -> None:
    for level, kind in [(9, "modus ponens"), (10, "inheritance"), (12, "deep is-a chain")]:
        ep = _pick(level)
        if ep is None:
            continue
        print(f"\n=== L{level} — {kind} ===")
        print("Heard:")
        for sent in ep.context:
            obj = membrane.parse(sent)
            print(f"  \"{sent}\"   ->  {obj}")
        print(f"Asked:  \"{ep.question}\"   ->  {membrane.parse(ep.question)}")

        res = teacher.replay(ep)
        support = next((s.support for s in res["trace"] if s.op == ops.INFER), [])
        query = tuple(ep.meta["query"])
        print("Thinks & answers:")
        print(f"  {verbalize_trace(query, res['answer'], support)}")


if __name__ == "__main__":
    main()
