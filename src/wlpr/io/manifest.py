"""Input manifest: sha256-verify every frozen/raw/derived-snapshot asset.

The privately held manifest is the provenance and integrity registry.
Columns: input_id, tier, relative_path, sha256, byte_count, row_count, materialize, schema_path,
evidence_level, allowed_use, forbidden_use, provenance.

`verify()` checks registered manifest entries and hashes each present asset (raw tier is
hashed lazily / optionally, since the 298 GB mobility tar is verified by a recorded one-time hash).
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import csv
import hashlib
import os

from wlpr.io import paths

MANIFEST = os.path.join(paths.TIERS["frozen"], _asset_location('asset_0119'), "input_manifest.csv")
COLUMNS = ["input_id", "tier", "relative_path", "sha256", "byte_count", "row_count",
           "materialize", "schema_path", "evidence_level", "allowed_use", "forbidden_use", "provenance"]


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: str) -> tuple[str, int]:
    """Directory fingerprint: sha256 over the sorted 'relpath\\tsize' listing (structure + per-file sizes, no
    content read — fast even for the 24k-file PV-checkpoints tree). Returns (hexdigest, total_bytes). Catches
    any added / removed / resized file; a same-size content edit is out of scope (use sha256_file for that)."""
    h = hashlib.sha256()
    total = 0
    entries = []
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, path).replace("\\", "/")
            sz = os.path.getsize(fp)
            entries.append((rel, sz)); total += sz
    for rel, sz in sorted(entries):
        h.update(f"{rel}\t{sz}\n".encode("utf-8"))
    return h.hexdigest(), total


def load() -> list[dict]:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify(hash_raw: bool = False) -> dict:
    """Return {ok, checked, missing[], mismatched[]}. raw-tier assets hashed only if hash_raw."""
    rows = load()
    missing, mismatched, checked = [], [], 0
    for r in rows:
        tier = r["tier"]
        base = paths.TIERS[tier]
        p = os.path.normpath(os.path.join(base, r["relative_path"]))
        if not os.path.exists(p):
            missing.append(r["input_id"])
            continue
        if tier == "raw" and not hash_raw:
            continue  # huge restricted raw (e.g. 298 GB tar) -- skip per-run hashing
        want = (r.get("sha256") or "").strip()
        if want:
            if os.path.isfile(p):
                got = sha256_file(p)
            elif os.path.isdir(p):
                got = sha256_dir(p)[0]
            else:
                continue
            checked += 1
            if got != want:
                mismatched.append((r["input_id"], want[:12], got[:12]))
    return {"ok": not missing and not mismatched, "checked": checked,
            "missing": missing, "mismatched": mismatched}
