"""Referee: three independent checks on a proposed tree, each returning per-link findings with reasons.

(i)   LICENSED: a link (head, relation, dependent, side) is licensed (a) by the grammar file read as data
      (``naomi.decoder.derive_order_table``) and (b) by ATTESTATION: the same configuration -- head tag + structural
      classes, relation, dependent tag + structural classes, side -- was committed by the parser in TRAIN gold at least
      ``min_count`` times. Both are data lookups; no rule is written here.
(ii)  AGREES WITH THE PARSER: of the links the parser committed (not UNATTACHED), the share the proposal keeps.
(iii) ROUND TRIP: proposal -> decoder v1 -> word sequence -> encoder -> tree', graded by ``score_pair`` plus the distance of
      the decoded order from the original. It uses the encoder twice, so it is not independent of the encoder.

Everything returned is plain python (JSON-serialisable).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .tree import walk
from .types import REL_ROOT, REL_UNATTACHED, Tree

CT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ATTESTED = CT_DIR / "runs" / "naomi_orderer" / "attested_configs.json"
DEFAULT_ORDERER = CT_DIR / "runs" / "naomi_orderer" / "large_best.pt"
DEFAULT_ENCODER = CT_DIR / "runs" / "naomi_encoder" / "main_best.pt"
BEFORE, AFTER = "before", "after"


# ----------------------------------------------------------------------------- links
def subs_key(subs: Iterable[str], classes: Optional[set]) -> str:
    return "|".join(sorted(s for s in subs if classes is None or s in classes))


def link_row(head_pos, head_subs, rel, dep_pos, dep_subs, dep_before_head: bool, classes: Optional[set]) -> tuple:
    return (head_pos, subs_key(head_subs, classes), rel, dep_pos, subs_key(dep_subs, classes),
            BEFORE if dep_before_head else AFTER)


def tree_links(tree: Tree, classes: Optional[set] = None) -> List[dict]:
    """One record per non-root, non-UNATTACHED real node that has a real head."""
    out = []
    for n, _d, p in walk(tree.root):
        if p is None or p.token_index is None or n.token_index is None or n.rel in (REL_ROOT, REL_UNATTACHED):
            continue
        out.append({"dep": n.token_index, "dep_token": n.token, "head": p.token_index, "head_token": p.token,
                    "rel": n.rel, "side": BEFORE if n.token_index < p.token_index else AFTER,
                    "row": link_row(p.pos, p.subtypes, n.rel, n.pos, n.subtypes, n.token_index < p.token_index, classes)})
    return sorted(out, key=lambda r: r["dep"])


# ----------------------------------------------------------------------------- (b) attested table
class Attested:
    def __init__(self, counts: Dict[tuple, int], classes: Sequence[str], min_count: int = 1):
        self.counts, self.classes, self.min_count = dict(counts), set(classes), min_count
        self.pos_counts: Counter = Counter()
        for (hp, _hs, rel, dp, _ds, side), c in self.counts.items():
            self.pos_counts[(hp, rel, dp, side)] += c

    def count(self, row: tuple) -> int:
        return self.counts.get(row, 0)

    def pos_count(self, row: tuple) -> int:
        return self.pos_counts.get((row[0], row[2], row[3], row[5]), 0)

    def licensed(self, row: tuple) -> bool:
        return self.count(row) >= self.min_count

    def to_json(self) -> dict:
        return {"classes": sorted(self.classes), "min_count": self.min_count,
                "n_links": int(sum(self.counts.values())),
                "columns": ["head_pos", "head_classes", "relation", "dep_pos", "dep_classes", "side", "count"],
                "rows": [list(k) + [c] for k, c in sorted(self.counts.items(), key=lambda kc: -kc[1])]}

    @classmethod
    def from_json(cls, d: dict) -> "Attested":
        return cls({tuple(r[:6]): r[6] for r in d["rows"]}, d["classes"], d["min_count"])

    @classmethod
    def load(cls, path=DEFAULT_ATTESTED) -> "Attested":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def attested_from_items(items, classes: set) -> Counter:
    """Count the parser's committed links in prepared items (``infos`` + ``sv`` as in the encoder's prep cache)."""
    cnt: Counter = Counter()
    for it in items:
        sv, infos = it["sv"], it["infos"]
        for i, ok in enumerate(sv.sup):
            if ok and sv.head[i] > 0 and infos[sv.head[i] - 1].is_node and infos[i].is_node:
                j = sv.head[i] - 1
                cnt[link_row(infos[j].pos, infos[j].subtypes, sv.rel[i], infos[i].pos, infos[i].subtypes, i < j, classes)] += 1
    return cnt


# ----------------------------------------------------------------------------- (a) grammar as data
class GrammarLicense:
    def __init__(self, table=None):
        if table is None:
            from .decoder import derive_order_table, load_grammar
            table = derive_order_table(load_grammar())
        self.table = table

    def check(self, head_pos: str, rel: str, dep_pos: str, side: str) -> Tuple[bool, str]:
        hc, dc = self.table.tag_to_cat.get(head_pos), self.table.tag_to_cat.get(dep_pos)
        if hc is None or dc is None:
            return False, "category unknown to the grammar"
        es = self.table.find(rel, hc, dc, allow_extra=True)
        if not es:
            return False, f"no grammar rule builds {rel} from head category {hc} and dependent category {dc}"
        if not any(e.side == side for e in es):
            return False, f"grammar licenses {rel} {hc}/{dc} only on the {'after' if side == BEFORE else 'before'} side"
        return True, "licensed by rule " + sorted({e.rule for e in es if e.side == side})[0]


# ----------------------------------------------------------------------------- the referee
class Referee:
    def __init__(self, attested: Attested, grammar: Optional[GrammarLicense] = None, orderer=None, encoder=None,
                 usvs=None, min_agreement: float = 0.9, min_roundtrip: float = 80.0):
        self.attested, self.grammar = attested, grammar or GrammarLicense()
        self.orderer, self.encoder, self.usvs = orderer, encoder, usvs
        self.min_agreement, self.min_roundtrip = min_agreement, min_roundtrip

    @classmethod
    def default(cls, roundtrip: bool = True) -> "Referee":
        att = Attested.load(DEFAULT_ATTESTED)
        if not roundtrip:
            return cls(att)
        from .encoder import NaomiEncoder
        from .orderer import Orderer
        from .reader import _sense_space
        usvs = _sense_space()
        return cls(att, orderer=Orderer.load(str(DEFAULT_ORDERER), usvs), encoder=NaomiEncoder.load(str(DEFAULT_ENCODER), usvs),
                   usvs=usvs)

    # (i)
    def check_links(self, tree: Tree) -> List[dict]:
        out = []
        for L in tree_links(tree, self.attested.classes):
            hp, _hs, rel, dp, _ds, side = L["row"]
            ok_a, why_a = self.grammar.check(hp, rel, dp, side)
            cnt = self.attested.count(L["row"])
            ok_b = cnt >= self.attested.min_count
            out.append({"dep": L["dep"], "dep_token": L["dep_token"], "head": L["head"], "head_token": L["head_token"],
                        "rel": rel, "side": side, "head_pos": hp, "dep_pos": dp,
                        "licensed_grammar": ok_a, "grammar_reason": why_a,
                        "attested_count": cnt, "attested_count_pos_only": self.attested.pos_count(L["row"]),
                        "licensed_attested": ok_b,
                        "attested_reason": (f"attested {cnt}x in train gold" if ok_b else
                                            f"configuration attested {cnt}x (< {self.attested.min_count}) in train gold")})
        return out

    # (ii)
    @staticmethod
    def agree_parser(tree: Tree, parser_tree: Tree) -> dict:
        prop = {L["dep"]: L for L in tree_links(tree)}
        root_tok = next((n.token_index for n, _d, _p in walk(tree.root) if n.rel == REL_ROOT and n.token_index is not None), None)
        committed = tree_links(parser_tree)
        kept_head = kept_both = 0
        changed = []
        for L in committed:
            P = prop.get(L["dep"])
            same_h = P is not None and P["head"] == L["head"]
            kept_head += int(same_h)
            if same_h and P["rel"] == L["rel"]:
                kept_both += 1
                continue
            changed.append({"dep": L["dep"], "dep_token": L["dep_token"], "parser_head": L["head"], "parser_rel": L["rel"],
                            "proposal_head": None if P is None else P["head"], "proposal_rel": None if P is None else P["rel"],
                            "reason": "relation changed" if same_h else "head changed"})
        n = len(committed)
        return {"parser_links": n, "kept_same_head_and_relation": kept_both, "kept_same_head": kept_head,
                "share_kept": round(kept_both / n, 4) if n else None,
                "share_kept_head_only": round(kept_head / n, 4) if n else None, "changed": changed,
                "proposal_root": root_tok}

    # (iii)
    def _tail(self, sentence: str) -> List[str]:
        """The sentence's own trailing non-node tokens (final punctuation, by tag), so the decoded text ends the way the
        original did; the order-only decoder places no punctuation itself."""
        if not sentence:
            return []
        infos = self.encoder._infos(sentence)
        k = len(infos)
        while k > 0 and not infos[k - 1].is_node:
            k -= 1
        return [t.token for t in infos[k:]]

    def roundtrip(self, tree: Tree, sentence: str = "") -> dict:
        from .orderer import kendall_distance
        from .score import score_pair
        nodes = self.orderer.order_tree(tree)
        words = [n.token for n in nodes]
        pos = [n.token_index for n in nodes]
        out = {"decoded_equals_original_order": pos == sorted(pos), "order_kendall_distance": round(kendall_distance(pos), 4),
               "decoded_text": " ".join(words + self._tail(sentence))}
        for name, text in (("with_final_punctuation", out["decoded_text"]), ("no_punctuation", " ".join(words))):
            sp = score_pair(tree, self.encoder.encode(text), self.usvs)
            out[name] = {"meaning": round(float(sp.meaning), 2), "shape": round(float(sp.shape), 2), "overall": round(float(sp.overall), 2)}
        out["overall"] = out["with_final_punctuation" if sentence else "no_punctuation"]["overall"]
        return out

    # all
    def referee(self, tree: Tree, sentence: str = "", parser_tree: Optional[Tree] = None, roundtrip: bool = True) -> dict:
        links = self.check_links(tree)
        res = {"sentence": sentence, "n_links": len(links), "links": links,
               "n_unlicensed_grammar": sum(not L["licensed_grammar"] for L in links),
               "n_unattested": sum(not L["licensed_attested"] for L in links)}
        reasons = [f"link {L['dep_token']}#{L['dep']} -> {L['head_token']}#{L['head']} ({L['rel']}): {L['attested_reason']}"
                   for L in links if not L["licensed_attested"]]
        if parser_tree is not None:
            res["parser_agreement"] = self.agree_parser(tree, parser_tree)
            s = res["parser_agreement"]["share_kept"]
            if s is not None and s < self.min_agreement:
                reasons.append(f"keeps only {s:.2f} of the parser's committed links (< {self.min_agreement})")
        if roundtrip and self.orderer is not None and self.encoder is not None:
            res["roundtrip"] = self.roundtrip(tree, sentence)
            if res["roundtrip"]["overall"] < self.min_roundtrip:
                reasons.append(f"round trip overall {res['roundtrip']['overall']} < {self.min_roundtrip}")
        res["accept"] = not reasons
        res["reasons"] = reasons
        return res


_DEFAULT: Optional[Referee] = None


def referee(tree: Tree, sentence: str = "", parser_tree: Optional[Tree] = None, ctx: Optional[Referee] = None) -> dict:
    """All findings plus one accept / reject with reasons. ``ctx`` defaults to the checked-in artifacts."""
    global _DEFAULT
    if ctx is None:
        if _DEFAULT is None:
            _DEFAULT = Referee.default()
        ctx = _DEFAULT
    return ctx.referee(tree, sentence, parser_tree)
