"""One-off: merge the operator's hand-judged labels (ENCODER_VS_PARSER.md
report work, branch encoder-vs-parser) into runs/evp_judgeset.jsonl and
write the final runs/encoder_vs_parser_judged.jsonl. Not part of the
reusable pipeline; run once, left for provenance."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABELS = {
    # SECTION 1: complement, encoder rank-1 (C0-C79)
    "C0": "PARTIAL", "C1": "EMPTY", "C2": "WRONG", "C3": "PARTIAL", "C4": "PARTIAL",
    "C5": "WRONG", "C6": "PARTIAL", "C7": "WRONG", "C8": "PARTIAL", "C9": "EMPTY",
    "C10": "WRONG", "C11": "WRONG", "C12": "USABLE", "C13": "PARTIAL", "C14": "USABLE",
    "C15": "PARTIAL", "C16": "USABLE", "C17": "PARTIAL", "C18": "PARTIAL", "C19": "WRONG",
    "C20": "EMPTY", "C21": "WRONG", "C22": "USABLE", "C23": "PARTIAL", "C24": "USABLE",
    "C25": "PARTIAL", "C26": "WRONG", "C27": "PARTIAL", "C28": "PARTIAL", "C29": "EMPTY",
    "C30": "EMPTY", "C31": "PARTIAL", "C32": "WRONG", "C33": "EMPTY", "C34": "WRONG",
    "C35": "EMPTY", "C36": "USABLE", "C37": "WRONG", "C38": "USABLE", "C39": "USABLE",
    "C40": "PARTIAL", "C41": "PARTIAL", "C42": "PARTIAL", "C43": "PARTIAL", "C44": "WRONG",
    "C45": "PARTIAL", "C46": "EMPTY", "C47": "PARTIAL", "C48": "WRONG", "C49": "USABLE",
    "C50": "EMPTY", "C51": "PARTIAL", "C52": "WRONG", "C53": "PARTIAL", "C54": "PARTIAL",
    "C55": "EMPTY", "C56": "PARTIAL", "C57": "PARTIAL", "C58": "USABLE", "C59": "USABLE",
    "C60": "PARTIAL", "C61": "PARTIAL", "C62": "PARTIAL", "C63": "USABLE", "C64": "PARTIAL",
    "C65": "PARTIAL", "C66": "USABLE", "C67": "PARTIAL", "C68": "PARTIAL", "C69": "USABLE",
    "C70": "USABLE", "C71": "PARTIAL", "C72": "EMPTY", "C73": "WRONG", "C74": "EMPTY",
    "C75": "WRONG", "C76": "PARTIAL", "C77": "PARTIAL", "C78": "PARTIAL", "C79": "WRONG",
    # SECTION 2: complement, POS-cheat heuristic (H0-H79, same 80 sentences)
    "H0": "WRONG", "H1": "EMPTY", "H2": "WRONG", "H3": "USABLE", "H4": "WRONG",
    "H5": "WRONG", "H6": "WRONG", "H7": "WRONG", "H8": "PARTIAL", "H9": "WRONG",
    "H10": "WRONG", "H11": "WRONG", "H12": "PARTIAL", "H13": "PARTIAL", "H14": "WRONG",
    "H15": "WRONG", "H16": "PARTIAL", "H17": "WRONG", "H18": "PARTIAL", "H19": "PARTIAL",
    "H20": "EMPTY", "H21": "PARTIAL", "H22": "PARTIAL", "H23": "WRONG", "H24": "PARTIAL",
    "H25": "PARTIAL", "H26": "WRONG", "H27": "WRONG", "H28": "PARTIAL", "H29": "EMPTY",
    "H30": "EMPTY", "H31": "WRONG", "H32": "PARTIAL", "H33": "EMPTY", "H34": "WRONG",
    "H35": "EMPTY", "H36": "USABLE", "H37": "WRONG", "H38": "PARTIAL", "H39": "PARTIAL",
    "H40": "PARTIAL", "H41": "WRONG", "H42": "WRONG", "H43": "WRONG", "H44": "WRONG",
    "H45": "WRONG", "H46": "EMPTY", "H47": "PARTIAL", "H48": "WRONG", "H49": "USABLE",
    "H50": "WRONG", "H51": "WRONG", "H52": "WRONG", "H53": "WRONG", "H54": "EMPTY",
    "H55": "WRONG", "H56": "WRONG", "H57": "PARTIAL", "H58": "USABLE", "H59": "PARTIAL",
    "H60": "PARTIAL", "H61": "WRONG", "H62": "WRONG", "H63": "PARTIAL", "H64": "WRONG",
    "H65": "PARTIAL", "H66": "PARTIAL", "H67": "PARTIAL", "H68": "WRONG", "H69": "WRONG",
    "H70": "WRONG", "H71": "PARTIAL", "H72": "WRONG", "H73": "WRONG", "H74": "EMPTY",
    "H75": "WRONG", "H76": "PARTIAL", "H77": "PARTIAL", "H78": "EMPTY", "H79": "WRONG",
    # SECTION 3: calibration, teacher-OK, encoder rank-1 (K0-K39)
    "K0": "USABLE", "K1": "PARTIAL", "K2": "WRONG", "K3": "USABLE", "K4": "PARTIAL",
    "K5": "WRONG", "K6": "USABLE", "K7": "PARTIAL", "K8": "USABLE", "K9": "PARTIAL",
    "K10": "PARTIAL", "K11": "PARTIAL", "K12": "PARTIAL", "K13": "PARTIAL", "K14": "USABLE",
    "K15": "USABLE", "K16": "USABLE", "K17": "PARTIAL", "K18": "EMPTY", "K19": "PARTIAL",
    "K20": "USABLE", "K21": "PARTIAL", "K22": "PARTIAL", "K23": "PARTIAL", "K24": "PARTIAL",
    "K25": "USABLE", "K26": "WRONG", "K27": "PARTIAL", "K28": "USABLE", "K29": "USABLE",
    "K30": "PARTIAL", "K31": "USABLE", "K32": "WRONG", "K33": "PARTIAL", "K34": "USABLE",
    "K35": "PARTIAL", "K36": "PARTIAL", "K37": "USABLE", "K38": "PARTIAL", "K39": "PARTIAL",
}

items = []
with open(ROOT / "runs" / "evp_judgeset.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        iid = rec["item_id"]
        assert iid in LABELS, f"missing label for {iid}"
        rec["label"] = LABELS[iid]
        items.append(rec)

assert len(items) == len(LABELS) == 200

out_path = ROOT / "runs" / "encoder_vs_parser_judged.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for rec in items:
        f.write(json.dumps(rec) + "\n")
print(f"wrote {len(items)} judged items -> {out_path}")

from collections import Counter

for section in ("complement_encoder", "complement_cheat", "calibration_encoder"):
    sub = [r for r in items if r["section"] == section]
    c = Counter(r["label"] for r in sub)
    n = len(sub)
    any_correct = c["USABLE"] + c["PARTIAL"]
    print(f"{section}: n={n} {dict(c)} any_correct={any_correct}/{n}={any_correct/n:.1%}")
