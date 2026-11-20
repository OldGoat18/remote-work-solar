"""Step 00 — verify inputs.

Reconciles configs/inputs.yaml against data/frozen/manifests/input_manifest.csv and sha256-verifies
every present frozen/derived asset (raw tier hashed only with --hash-raw). Exit 1 on any
missing/mismatched asset. This is the integrity gate the whole pipeline rests on.
"""
from __future__ import annotations

import sys

from wlpr.io import manifest, paths


def main() -> None:
    paths.assert_not_source_tree()
    hash_raw = "--hash-raw" in sys.argv
    r = manifest.verify(hash_raw=hash_raw)
    print(f"[00] manifest: checked {r['checked']} hashes | "
          f"missing {len(r['missing'])} | mismatched {len(r['mismatched'])}")
    for iid in r["missing"]:
        print(f"  [MISSING] {iid}")
    for iid, want, got in r["mismatched"]:
        print(f"  [SHA MISMATCH] {iid}: want {want}.. got {got}..")
    if not r["ok"]:
        raise SystemExit("00_verify_inputs: FAIL")
    print("00_verify_inputs: OK")


if __name__ == "__main__":
    main()
