"""Refresh the privately held integrity manifest from the configured input catalogue.

For every registered input: record tier + path; if the file is present (and not a huge raw-tier asset unless
--hash-raw) compute its sha256/byte_count. Unstaged inputs are listed with empty sha256 so 00_verify_inputs
reports them as 'to stage'. Run after staging to refresh hashes.
"""
from __future__ import annotations

import csv
import os
import sys

from wlpr.io import manifest, paths


def main() -> None:
    paths.assert_not_source_tree()
    hash_raw = "--hash-raw" in sys.argv
    idx = paths._inputs_index()
    rows = []
    for iid in sorted(idx):
        tier, rel = idx[iid]
        p = os.path.join(paths.TIERS[tier], rel)
        sha, nbytes = "", ""
        if tier != "raw" or hash_raw:
            if os.path.isfile(p):
                sha = manifest.sha256_file(p); nbytes = os.path.getsize(p)
            elif os.path.isdir(p):  # directory asset (e.g. pv_checkpoints) -> structure+size fingerprint
                sha, nbytes = manifest.sha256_dir(p)
        rows.append(dict(input_id=iid, tier=tier, relative_path=rel, sha256=sha, byte_count=nbytes,
                         row_count="", materialize=("stream" if tier == "raw" else "copy"),
                         schema_path="", evidence_level="", allowed_use="", forbidden_use="", provenance=""))
    os.makedirs(os.path.dirname(manifest.MANIFEST), exist_ok=True)
    with open(manifest.MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest.COLUMNS)
        w.writeheader(); w.writerows(rows)
    staged = sum(1 for r in rows if r["sha256"])
    print(f"[manifest] {len(rows)} inputs; {staged} staged+hashed; {len(rows)-staged} pending staging")
    print(f"[saved] {manifest.MANIFEST}")


if __name__ == "__main__":
    main()
