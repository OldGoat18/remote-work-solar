"""Download and normalize the public e-Stat time-use curve used by the WFH state layer.

Source table 0003457699 is the 2021 Japanese Social Life Basic Survey table
"time-of-day activity rate".  The query is fixed to weekday, both sexes,
employed persons, the activity "work", and Japan total.  The table already
provides 96 quarter-hour intervals, so no interpolation is applied.

The generated Parquet is a public reconstruction of the missing frozen input;
it is deliberately not labelled as byte-identical to the legacy file.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

import base64
import csv
import gzip
import hashlib
import json
import os
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SID = "0003457699"
MODEL_URL = "https://www.e-stat.go.jp/dbview/api_get_model"
RESULT_URL = "https://www.e-stat.go.jp/dbview/api_get_result"
RAW_DIR = ROOT / _asset_location('asset_0002') / _asset_location('asset_0120') / _asset_location('asset_0198') / _asset_location('asset_0197')
OUT = ROOT / _asset_location('asset_0002') / _asset_location('asset_0121') / _asset_location('asset_0124') / "timeuse_activity_rate_15min.parquet"
MANIFEST = ROOT / _asset_location('asset_0002') / _asset_location('asset_0121') / _asset_location('asset_0119') / "input_manifest.csv"


class _ValueParser(HTMLParser):
    """Extract table value cells without adding a runtime HTML dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []
        self._capture = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "td":
            return
        attrs_d = dict(attrs)
        if "stat-dbview-value" in (attrs_d.get("class") or "").split():
            self._capture = True
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._capture:
            self.values.append(" ".join("".join(self._buf).split()))
            self._capture = False
            self._buf = []


_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "WFH-LV-PV-reproduction/1.0",
    "Referer": f"https://www.e-stat.go.jp/dbview?sid={SID}",
    "X-Requested-With": "XMLHttpRequest",
})


def _post_json(url: str, params: dict[str, str], payload: dict | None = None) -> dict:
    """Use one HTTP session because the result endpoint keeps viewer state in cookies."""
    last = None
    for attempt in range(4):
        response = _SESSION.post(url, params=params, data=payload or {}, timeout=180)
        if response.status_code != 408:
            response.raise_for_status()
            return response.json()
        last = response
        time.sleep(2.0 * (attempt + 1))
    assert last is not None
    last.raise_for_status()
    return last.json()


def _encode_selection(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def _selection_request(model: dict) -> dict:
    """Build the same compressed form payload as the public e-Stat viewer."""
    base = {
        "rows": [], "cols": [], "tops": [], "apiTops": [],
        "annotationFlg": model["annotationFlg"],
        "rowNoDataDispFlg": model["rowNoDataDispFlg"],
        "colNoDataDispFlg": model["colNoDataDispFlg"],
        "commaType": model["commaType"],
        "replaceSpChars": model["replaceSpChars"],
        "graphAxis": model["graphAxis"], "graphBasis": model["graphBasis"],
        "graphSort": model["graphSort"], "graphTitle": model["graphTitle"],
        "graphType": model["graphType"], "inputNumberOfCols": 99999,
        "inputNumberOfRows": 99999, "movementId": 0,
        "leftMoveFlg": model["leftMoveFlg"], "rightMoveFlg": model["rightMoveFlg"],
        "underMoveFlg": model["underMoveFlg"], "upMoveFlg": model["upMoveFlg"],
        "currentCols": None, "currentRows": None, "mode": "table",
    }
    # Fixed, auditable selection: weekday, total sex, employed, work, age total,
    # Japan total.  Matter 8 contains the 96 quarter-hour columns.
    selected = {3: ["1"], 4: ["0"], 5: ["1"], 6: ["05"], 7: ["00"], 1: ["00000"]}
    for _, matter in sorted(model["matters"].items(), key=lambda item: int(item[1]["matterId"])):
        position = matter["position"]
        if position in ("row", "col"):
            if matter["matterId"] == 8 or position == "col":
                codes = [item["code"] for item in matter["listData"].values()]
            else:
                codes = selected[matter["matterId"]]
            items = []
            for code in codes:
                item = matter["listData"]["code" + code]
                items.append({
                    "name": item["name"], "code": code,
                    "unit": item.get("unitName", ""),
                    "explanation": item.get("explanation", ""),
                })
            entry = {
                "matterId": matter["matterId"], "tableName": matter["tableName"],
                "dispTableName": matter["dispTableName"],
                "positionNum": matter["positionNum"], "listData": items,
                "allSelected": int(len(items) == len(matter["listData"])),
            }
            base["rows" if position == "row" else "cols"].append(entry)
        else:
            item = next(iter(matter["listData"].values()))
            entry = {
                "matterId": matter["matterId"], "tableName": matter["tableName"],
                "dispTableName": matter["dispTableName"],
                "positionNum": matter["positionNum"],
                "listData": [{"name": item["name"], "code": item["code"], "explanation": ""}],
                "allSelected": 1,
            }
            base["tops"].append(entry)
            base["apiTops"].append(entry)
    post = dict(base)
    for key in ("rows", "cols", "tops", "apiTops"):
        post[key] = _encode_selection(base[key])
    return post


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _update_manifest(path: Path, rows: list[dict]) -> None:
    with path.open(encoding="utf-8", newline="") as f:
        existing = list(csv.DictReader(f))
    row = next((x for x in existing if x["input_id"] == "timeuse_15min"), None)
    if row is None:
        raise RuntimeError("timeuse_15min is missing from the input manifest")
    row.update({
        "sha256": _sha256(OUT), "byte_count": str(OUT.stat().st_size), "row_count": "96",
        "materialize": "public_rebuild", "evidence_level": "public_reconstructed",
        "allowed_use": "WFH state-layer time shape",
        "forbidden_use": "not a measured load series",
        "provenance": "e-Stat 0003457699;Reiwa3 Social Life Basic Survey;weekday;work;Japan;15min",
    })
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing[0].keys())
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    model = _post_json(MODEL_URL, {"sid": SID}, {})
    (RAW_DIR / f"estat_{SID}_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    response = _post_json(RESULT_URL, {"sid": SID}, _selection_request(model))
    (RAW_DIR / f"estat_{SID}_result.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    if response.get("error"):
        raise RuntimeError(response)
    parser = _ValueParser()
    parser.feed(response.get("table", ""))
    if len(parser.values) != 96:
        raise RuntimeError(f"e-Stat table returned {len(parser.values)} values, expected 96")
    values = pd.to_numeric(pd.Series(parser.values), errors="coerce")
    if values.isna().any() or (values < 0).any() or (values > 100).any():
        raise RuntimeError(f"invalid e-Stat activity rates: {parser.values[:10]}")
    out = pd.DataFrame({
        "day_type": "weekday", "activity": "work", "slot": range(96), "rate_pct": values.astype(float),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    _update_manifest(MANIFEST, [])
    print(f"[timeuse] source=e-Stat:{SID} rows={len(out)}")
    print(f"[timeuse] rate_pct min={out.rate_pct.min():.3f} max={out.rate_pct.max():.3f}")
    print(f"[timeuse] wrote {OUT} sha256={_sha256(OUT)}")
    print(f"[timeuse] raw files in {RAW_DIR}")


if __name__ == "__main__":
    main()
