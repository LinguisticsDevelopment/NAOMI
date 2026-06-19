"""Fetch the ProofWriter OWA subset into data/proofwriter/ (reproducible, M8).

Downloads the official Allen AI ProofWriter release and extracts the open-world
(OWA) meta JSONL for depths 0,1,2,3,5 — the structured facts/rules/questions the
mind substrate ingests. The data is git-ignored (data/*.jsonl) and regenerable
by re-running this. If the network policy blocks the download, see the M8 plan's
fallback (a vendored subset or the synthetic broad rule-reasoning generator).

Run:  python scripts/fetch_proofwriter.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile

URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
DEPTHS = ("0", "1", "2", "3", "5")
WANT = ("meta-train.jsonl", "meta-dev.jsonl", "meta-test.jsonl")


def main() -> None:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "proofwriter")
    os.makedirs(out, exist_ok=True)
    cache = "/tmp/proofwriter.zip"
    if not os.path.exists(cache):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, cache)
    z = zipfile.ZipFile(cache)
    kept = 0
    for n in z.namelist():
        leaf = n.split("/")[-1]
        if "/OWA/depth-" in n and leaf in WANT:
            depth = n.split("/OWA/depth-")[1].split("/")[0]
            if depth in DEPTHS:
                dst = os.path.join(out, f"owa-depth{depth}-{leaf.split('-')[-1]}")
                with z.open(n) as src, open(dst, "wb") as f:
                    f.write(src.read())
                kept += 1
    print(f"extracted {kept} OWA files -> {out}")


if __name__ == "__main__":
    main()
