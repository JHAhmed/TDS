"""
TDS 2026 May - GA8 MLOps & Fine-Tuning
Single FastAPI app exposing all seven graded endpoints:

  POST /build-corpus   (Q1)
  POST /bqml           (Q2)
  POST /promote        (Q3)
  POST /adapt          (Q4)
  POST /quantize       (Q5)
  POST /pipeline       (Q6)
  POST /verify-bundle  (Q7)

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
State is in-process. Deploy as ONE instance (no autoscaling), otherwise
the stateful endpoints (/bqml, /quantize, /pipeline) will fail replays.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="TDS GA8")

MAX_SAFE = 2**53 - 1
BAD = JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def cj(x: Any) -> str:
    """Compact JSON, non-ASCII emitted directly."""
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return sha256_hex(s.encode("utf-8"))


def is_str(v: Any) -> bool:
    return isinstance(v, str)


def nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v != ""


def is_safe_int(v: Any, *, minimum: int | None = None) -> bool:
    if isinstance(v, bool) or not isinstance(v, int):
        return False
    if abs(v) > MAX_SAFE:
        return False
    if minimum is not None and v < minimum:
        return False
    return True


def is_finite(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def in_unit(v: Any) -> bool:
    return is_finite(v) and 0.0 <= float(v) <= 1.0


def r12(x: float) -> float:
    return round(float(x), 12)


def usort(xs):
    return sorted(xs, key=lambda s: s.encode("utf-8"))


def codes(xs) -> list[str]:
    return usort(set(xs))


def is_hex(s: Any, n: int) -> bool:
    return isinstance(s, str) and len(s) == n and re.fullmatch(r"[0-9a-f]+", s) is not None


TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$"
)
_DIM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def parse_ts(s: Any) -> float | None:
    """Return epoch milliseconds, or None when the instant is invalid."""
    if not isinstance(s, str):
        return None
    m = TS_RE.match(s)
    if not m:
        return None
    y, mo, d, hh, mm, ss = (int(m.group(i)) for i in range(1, 7))
    frac = m.group(7) or "0"
    off = m.group(8)
    if not 1 <= mo <= 12:
        return None
    dim = _DIM[mo - 1]
    if mo == 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
        dim = 29
    if not 1 <= d <= dim or hh > 23 or mm > 59 or ss > 59:
        return None
    if off == "Z":
        off_min = 0
    else:
        sign = 1 if off[0] == "+" else -1
        oh, om = int(off[1:3]), int(off[4:6])
        if om > 59 or oh > 14 or (oh == 14 and om != 0):
            return None
        off_min = sign * (oh * 60 + om)
    ms = int((frac + "000")[:3])
    days = _days_from_civil(y, mo, d)
    return (days * 86400 + hh * 3600 + mm * 60 + ss) * 1000 + ms - off_min * 60000


def _days_from_civil(y: int, m: int, d: int) -> int:
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def iso_utc(ms: float) -> str:
    ms = int(ms)
    days, rem = divmod(ms, 86400000)
    y, mo, d = _civil_from_days(days)
    msec = rem % 1000
    rem //= 1000
    hh, rem = divmod(rem, 3600)
    mm, ss = divmod(rem, 60)
    return f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}:{ss:02d}.{msec:03d}Z"


def _civil_from_days(z: int):
    z += 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + (3 if mp < 10 else -9)
    return y + (m <= 2), m, d


_WS = re.compile(r"\s+", re.UNICODE)


def canon_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = _WS.sub(" ", s)
    return s.strip()


_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def word_set(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


async def body(request: Request):
    try:
        return await request.json()
    except Exception:
        return None


# ==========================================================================
# Q1  POST /build-corpus
# ==========================================================================
URI_RE = re.compile(r"^gs://[^/\s]+/.+$")
DEC_RE = re.compile(r"^\d+$")
ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}


def _row_ok(r: Any) -> bool:
    if not isinstance(r, dict) or set(r.keys()) != ROW_KEYS:
        return False
    for k in ("id", "entity", "eventTime", "text"):
        if not isinstance(r[k], str):
            return False
    if not is_safe_int(r["revision"], minimum=0):
        return False
    return parse_ts(r["eventTime"]) is not None


@app.post("/build-corpus")
async def build_corpus(request: Request):
    data = await body(request)
    if not isinstance(data, dict):
        return BAD
    policy = data.get("policy")
    objects = data.get("objects")
    if not isinstance(policy, dict) or not isinstance(objects, list):
        return BAD

    policy_ok = (
        parse_ts(policy.get("minTime")) is not None
        and parse_ts(policy.get("maxTime")) is not None
        and in_unit(policy.get("contaminationThreshold"))
    )
    min_ms = parse_ts(policy.get("minTime"))
    max_ms = parse_ts(policy.get("maxTime"))
    thr = float(policy["contaminationThreshold"]) if in_unit(policy.get("contaminationThreshold")) else None

    rejected_objects: list[dict] = []
    lineage: list[dict] = []
    rows: list[dict] = []

    for obj in objects:
        cs: list[str] = []
        o = obj if isinstance(obj, dict) else {}
        uri = o.get("uri")
        if not isinstance(uri, str) or not URI_RE.match(uri):
            cs.append("URI_INVALID")

        g, fg = o.get("generation"), o.get("fetchedGeneration")
        g_ok = isinstance(g, str) and DEC_RE.match(g) is not None
        fg_ok = isinstance(fg, str) and DEC_RE.match(fg) is not None
        if not (g_ok and fg_ok):
            cs.append("GENERATION_INVALID")
        if g != fg:
            cs.append("GENERATION_MISMATCH")

        crc = o.get("crc32c")
        crc_ok = isinstance(crc, str) and re.fullmatch(r"[0-9a-f]{8}", crc) is not None
        if not crc_ok:
            cs.append("CRC32C_INVALID")

        content = o.get("content")
        content_is_str = isinstance(content, str)
        if not content_is_str:
            cs.append("SCHEMA_INVALID")
        elif crc_ok:
            actual = _crc32c_hex(content.encode("utf-8"))
            if actual != crc:
                cs.append("CRC32C_MISMATCH")

        if o.get("schemaId") != "training-v1":
            cs.append("SCHEMA_INVALID")

        parsed_rows: list[dict] = []
        if content_is_str:
            lines = [ln for ln in content.split("\n") if ln.strip() != ""]
            if not lines:
                cs.append("SCHEMA_INVALID")
            for ln in lines:
                try:
                    row = json.loads(ln)
                except Exception:
                    cs.append("JSONL_INVALID")
                    continue
                if not _row_ok(row):
                    cs.append("SCHEMA_INVALID")
                    continue
                parsed_rows.append(row)

        if cs:
            rejected_objects.append(
                {"uri": uri if isinstance(uri, str) else None, "reasonCodes": codes(cs)}
            )
            continue

        lineage.append({"uri": uri, "generation": g, "crc32c": crc, "schemaId": o["schemaId"]})
        for row in parsed_rows:
            rows.append(
                {
                    "id": row["id"],
                    "entity": canon_text(row["entity"]),
                    "eventTime": iso_utc(parse_ts(row["eventTime"])),
                    "revision": row["revision"],
                    "text": canon_text(row["text"]),
                }
            )

    rejected_rows: list[dict] = []

    # dedupe
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(cj([r["entity"], r["eventTime"], r["text"]]), []).append(r)
    kept: list[dict] = []
    for _, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: (-r["revision"], r["id"].encode("utf-8")))
        kept.append(grp_sorted[0])
        for loser in grp_sorted[1:]:
            rejected_rows.append({"id": loser["id"], "reasonCodes": ["DUPLICATE"]})

    survivors: list[dict] = []
    if not policy_ok:
        for r in kept:
            rejected_rows.append({"id": r["id"], "reasonCodes": ["POLICY_INVALID"]})
    else:
        for r in kept:
            t = parse_ts(r["eventTime"])
            if t < min_ms or t > max_ms:
                rejected_rows.append({"id": r["id"], "reasonCodes": ["OUT_OF_WINDOW"]})
            else:
                survivors.append(r)

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for r in survivors:
        b = hashlib.sha256(r["entity"].encode("utf-8")).digest()[0] % 10
        splits["train" if b <= 5 else ("validation" if b <= 7 else "test")].append(r)

    train_words = [word_set(r["text"]) for r in splits["train"]]
    if thr is not None:
        for name in ("validation", "test"):
            keep = []
            for r in splits[name]:
                ws = word_set(r["text"])
                if any(jaccard(ws, tw) >= thr for tw in train_words):
                    rejected_rows.append({"id": r["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
                else:
                    keep.append(r)
            splits[name] = keep

    def ser(r: dict) -> str:
        return cj(
            {
                "id": r["id"],
                "entity": r["entity"],
                "eventTime": r["eventTime"],
                "revision": r["revision"],
                "text": r["text"],
            }
        )

    digests = {}
    out_splits = {}
    for name in ("train", "validation", "test"):
        ordered = sorted(splits[name], key=lambda r: (r["id"].encode("utf-8"), ser(r).encode("utf-8")))
        out_splits[name] = [json.loads(ser(r)) for r in ordered]
        blob = "".join(ser(r) + "\n" for r in ordered).encode("utf-8")
        digests[name] = sha256_hex(blob)

    rejected_objects.sort(key=lambda e: ((e["uri"] or "").encode("utf-8"), cj(e).encode("utf-8")))
    rejected_rows.sort(key=lambda e: (e["id"].encode("utf-8"), cj(e).encode("utf-8")))
    lineage.sort(key=lambda e: (e["uri"].encode("utf-8"), cj(e).encode("utf-8")))

    return {
        "splits": out_splits,
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows,
        "digests": digests,
        "lineage": lineage,
    }


_CRC32C_TABLE: list[int] = []


def _crc32c_hex(data: bytes) -> str:
    global _CRC32C_TABLE
    if not _CRC32C_TABLE:
        poly = 0x82F63B78
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ (poly if c & 1 else 0)
            _CRC32C_TABLE.append(c)
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return format(crc ^ 0xFFFFFFFF, "08x")


# ==========================================================================
# Q2  POST /bqml
# ==========================================================================
BQML_RUNS: dict[str, dict] = {}


def _sel_valid(d: dict) -> bool:
    run_id = d.get("runId")
    if not nonempty_str(run_id) or len(run_id) > 128:
        return False
    if not isinstance(d.get("forbiddenFeatures"), list) or not all(
        is_str(x) for x in d["forbiddenFeatures"]
    ):
        return False
    if not is_safe_int(d.get("numTrialsLimit"), minimum=1):
        return False
    rows = d.get("rows")
    if not isinstance(rows, list) or not rows:
        return False
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            return False
        if not nonempty_str(r.get("id")) or r["id"] in seen:
            return False
        seen.add(r["id"])
        if not nonempty_str(r.get("entity")):
            return False
        if parse_ts(r.get("eventTime")) is None or parse_ts(r.get("predictionTime")) is None:
            return False
        if not is_safe_int(r.get("version"), minimum=0):
            return False
        if r.get("split") not in ("TRAIN", "EVAL"):
            return False
        feats = r.get("features")
        if not isinstance(feats, dict):
            return False
        for k, v in feats.items():
            if not nonempty_str(k) or not isinstance(v, dict):
                return False
            if parse_ts(v.get("availableAt")) is None:
                return False
    trials = d.get("trials")
    if not isinstance(trials, list):
        return False
    tseen = set()
    for t in trials:
        if not isinstance(t, dict):
            return False
        if not is_safe_int(t.get("trialId"), minimum=0) or t["trialId"] in tseen:
            return False
        tseen.add(t["trialId"])
        if t.get("status") not in ("SUCCEEDED", "FAILED"):
            return False
    return True


def _bqml_select(d: dict) -> dict:
    run_id = d.get("runId") if nonempty_str(d.get("runId")) else None
    if not _sel_valid(d):
        return {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    rows = d["rows"]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(cj([r["entity"], iso_utc(parse_ts(r["eventTime"]))]), []).append(r)
    kept = [
        sorted(g, key=lambda r: (-r["version"], r["id"].encode("utf-8")))[0]
        for g in groups.values()
    ]

    forbidden = set(d["forbiddenFeatures"])
    names = set(kept[0]["features"].keys()) if kept else set()
    for r in kept:
        names &= set(r["features"].keys())
    eligible_feats = []
    for n in names:
        if n in forbidden:
            continue
        if all(
            parse_ts(r["features"][n]["availableAt"]) <= parse_ts(r["predictionTime"]) for r in kept
        ):
            eligible_feats.append(n)
    feature_names = usort(eligible_feats)
    train_ids = usort([r["id"] for r in kept if r["split"] == "TRAIN"])
    eval_ids = usort([r["id"] for r in kept if r["split"] == "EVAL"])
    digest = sha256_str(
        cj({"trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": feature_names})
    )

    cs: list[str] = []
    trials = d["trials"]
    if len(trials) > d["numTrialsLimit"]:
        cs.append("TRIAL_LIMIT_EXCEEDED")
    ok = [t for t in trials if t["status"] == "SUCCEEDED" and is_finite(t.get("evalMetric"))]
    if not ok:
        cs.append("NO_SUCCESSFUL_TRIAL")
    selected = None
    if ok and not cs:
        selected = sorted(ok, key=lambda t: (-float(t["evalMetric"]), t["trialId"]))[0]["trialId"]

    return {
        "runId": run_id,
        "selectedTrialId": selected,
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": codes(cs),
    }


def _eval_row_ok(r: Any) -> bool:
    return (
        isinstance(r, dict)
        and r.get("label") in (0, 1)
        and not isinstance(r.get("label"), bool)
        and r.get("prediction") in (0, 1)
        and not isinstance(r.get("prediction"), bool)
        and nonempty_str(r.get("slice"))
    )


@app.post("/bqml")
async def bqml(request: Request):
    d = await body(request)
    if not isinstance(d, dict):
        return BAD
    phase = d.get("phase")

    if phase == "select":
        run_id = d.get("runId")
        fingerprint = cj({k: v for k, v in sorted(d.items()) if k != "phase"})
        if nonempty_str(run_id) and run_id in BQML_RUNS:
            prev = BQML_RUNS[run_id]
            if prev["fingerprint"] != fingerprint:
                return JSONResponse(status_code=409, content={"error": "RUN_ID_CONFLICT"})
            return prev["response"]
        resp = _bqml_select(d)
        if nonempty_str(run_id):
            BQML_RUNS[run_id] = {"fingerprint": fingerprint, "response": resp}
        return resp

    if phase == "evaluate":
        cs: list[str] = []
        run_id = d.get("runId")
        sel = d.get("selectedTrialId")
        dig = d.get("datasetDigest")

        input_ok = (
            nonempty_str(run_id)
            and len(run_id) <= 128
            and in_unit(d.get("metricFloor"))
            and isinstance(d.get("requiredSlices"), dict)
            and all(nonempty_str(k) and in_unit(v) for k, v in d["requiredSlices"].items())
            and is_safe_int(d.get("bytesProcessed"), minimum=0)
            and is_safe_int(d.get("maxBytes"), minimum=0)
            and isinstance(d.get("rows"), list)
        )
        if not input_ok:
            cs.append("INVALID_INPUT")

        lineage_ok = (
            is_safe_int(sel, minimum=0)
            and is_hex(dig, 64)
            and nonempty_str(run_id)
            and run_id in BQML_RUNS
            and BQML_RUNS[run_id]["response"].get("selectedTrialId") == sel
            and BQML_RUNS[run_id]["response"].get("datasetDigest") == dig
            and not BQML_RUNS[run_id]["response"].get("reasonCodes")
        )
        if not lineage_ok:
            cs.append("INVALID_LINEAGE")

        rows = d.get("rows") if isinstance(d.get("rows"), list) else []
        rows_ok = bool(rows) and all(_eval_row_ok(r) for r in rows)
        if rows and not rows_ok:
            cs.append("INVALID_TEST_ROW")

        test_metric = None
        slice_pass = True
        if rows_ok and input_ok:
            correct = sum(1 for r in rows if r["label"] == r["prediction"])
            test_metric = r12(correct / len(rows))
            if test_metric < float(d["metricFloor"]):
                cs.append("AGGREGATE_FLOOR")
            for name, floor in d["requiredSlices"].items():
                sub = [r for r in rows if r["slice"] == name]
                if not sub:
                    cs.append(f"MISSING_SLICE:{name}")
                    slice_pass = False
                    continue
                acc = r12(sum(1 for r in sub if r["label"] == r["prediction"]) / len(sub))
                if acc < float(floor):
                    cs.append(f"SLICE_FLOOR:{name}")
                    slice_pass = False
        else:
            slice_pass = False

        if not input_ok or not lineage_ok:
            slice_pass = False

        if is_safe_int(d.get("bytesProcessed"), minimum=0) and is_safe_int(
            d.get("maxBytes"), minimum=0
        ):
            if d["bytesProcessed"] > d["maxBytes"]:
                cs.append("BYTE_LIMIT")

        decision = "admit" if (not cs and test_metric is not None) else "reject"
        return {
            "runId": run_id if is_str(run_id) else None,
            "selectedTrialId": sel if is_safe_int(sel, minimum=0) else None,
            "datasetDigest": dig if is_hex(dig, 64) else None,
            "testMetric": test_metric,
            "criticalSlicePass": bool(slice_pass),
            "decision": decision,
            "bytesProcessed": d.get("bytesProcessed") if is_safe_int(d.get("bytesProcessed"), minimum=0) else None,
            "reasonCodes": codes(cs),
        }

    return BAD


# ==========================================================================
# Q3  POST /promote
# ==========================================================================
VER_RE = re.compile(r"^[1-9]\d*$")
PROMOTE_ALIAS: dict[str, str] = {}


@app.post("/promote")
async def promote(request: Request):
    d = await body(request)
    if not isinstance(d, dict):
        return BAD
    policy = d.get("policy")
    versions = d.get("versions")
    champ = d.get("championVersion")
    if not isinstance(policy, dict) or not isinstance(versions, list) or not is_str(champ):
        return BAD

    as_of = parse_ts(d.get("asOf"))
    req_slices = policy.get("requiredSlices")
    policy_ok = (
        as_of is not None
        and nonempty_str(policy.get("datasetDigest"))
        and nonempty_str(policy.get("schemaDigest"))
        and is_safe_int(policy.get("maxAgeSeconds"), minimum=0)
        and in_unit(policy.get("accuracyFloor"))
        and isinstance(req_slices, dict)
        and all(nonempty_str(k) and in_unit(v) for k, v in req_slices.items())
        and is_finite(policy.get("maxLatencyMs"))
        and float(policy.get("maxLatencyMs", -1)) >= 0
        and is_safe_int(policy.get("maxSizeBytes"), minimum=0)
        and in_unit(policy.get("minImprovement"))
    )

    failed: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for v in versions:
        vid = v.get("version") if isinstance(v, dict) else None
        key = vid if is_str(vid) else cj(vid)
        counts[key] = counts.get(key, 0) + 1

    eligible: dict[str, dict] = {}

    for v in versions:
        vid = v.get("version") if isinstance(v, dict) else None
        key = vid if is_str(vid) else cj(vid)
        cs = failed.setdefault(key, [])
        canonical = is_str(vid) and VER_RE.match(vid) is not None and int(vid) <= MAX_SAFE
        if not canonical:
            cs.append("INVALID_VERSION")
        if counts[key] > 1:
            cs.append("DUPLICATE_VERSION")
        if not policy_ok:
            cs.append("INVALID_POLICY")

        ev = v.get("evaluation") if isinstance(v, dict) else None
        if not isinstance(ev, dict):
            cs.append("MISSING_EVALUATION")
            continue

        created = parse_ts(ev.get("createdAt"))
        if created is None:
            cs.append("INVALID_TIMESTAMP")
        elif as_of is not None:
            if created > as_of:
                cs.append("FUTURE_EVALUATION")
            elif policy_ok and created < as_of - policy["maxAgeSeconds"] * 1000:
                cs.append("STALE_EVALUATION")

        acc, lat, size = ev.get("accuracy"), ev.get("latencyMs"), ev.get("sizeBytes")
        if not (is_finite(acc) and is_finite(lat) and is_finite(size)):
            cs.append("NON_FINITE")
        else:
            if not in_unit(acc) or float(lat) < 0 or not is_safe_int(size, minimum=0):
                cs.append("METRIC_RANGE")

        if ev.get("artifactDigest") != v.get("artifactDigest") or not nonempty_str(
            v.get("artifactDigest")
        ):
            cs.append("ARTIFACT_MISMATCH")
        if policy_ok:
            if ev.get("datasetDigest") != policy["datasetDigest"]:
                cs.append("DATASET_MISMATCH")
            if ev.get("schemaDigest") != policy["schemaDigest"]:
                cs.append("SCHEMA_MISMATCH")

        slices = ev.get("slices") if isinstance(ev.get("slices"), dict) else {}
        if policy_ok:
            for name, floor in req_slices.items():
                if name not in slices:
                    cs.append(f"MISSING_SLICE:{name}")
                    continue
                val = slices[name]
                if not in_unit(val):
                    cs.append(f"SLICE_RANGE:{name}")
                elif float(val) < float(floor):
                    cs.append(f"SLICE_FLOOR:{name}")

        if policy_ok and is_finite(acc) and in_unit(acc) and float(acc) < float(policy["accuracyFloor"]):
            cs.append("ACCURACY_FLOOR")
        if policy_ok and is_finite(lat) and float(lat) >= 0 and float(lat) > float(policy["maxLatencyMs"]):
            cs.append("LATENCY_LIMIT")
        if policy_ok and is_safe_int(size, minimum=0) and size > policy["maxSizeBytes"]:
            cs.append("SIZE_LIMIT")

        if not cs and canonical:
            eligible[vid] = {"version": vid, "ev": ev}

    failed_out = {k: codes(v) for k, v in failed.items()}

    ranked = sorted(
        eligible.values(),
        key=lambda e: (
            -float(e["ev"]["accuracy"]),
            float(e["ev"]["latencyMs"]),
            int(e["ev"]["sizeBytes"]),
            int(e["version"]),
        ),
    )
    eligible_versions = sorted(eligible.keys(), key=lambda s: int(s))

    listed = {v.get("version") for v in versions if isinstance(v, dict)}
    if champ not in listed or champ not in eligible or not policy_ok or not ranked:
        return {
            "action": "block",
            "championVersion": champ,
            "selectedVersion": None,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_out,
            "aliasMutation": None,
            "evidence": None,
        }

    challenger = ranked[0]
    champ_acc = float(eligible[champ]["ev"]["accuracy"])
    improvement = r12(float(challenger["ev"]["accuracy"]) - champ_acc)

    if improvement >= float(policy["minImprovement"]) and challenger["version"] != champ:
        PROMOTE_ALIAS["champion"] = challenger["version"]
        return {
            "action": "promote",
            "championVersion": champ,
            "selectedVersion": challenger["version"],
            "eligibleVersions": eligible_versions,
            "failedGates": failed_out,
            "aliasMutation": {"alias": "champion", "version": challenger["version"]},
            "evidence": challenger["ev"],
        }

    return {
        "action": "retain",
        "championVersion": champ,
        "selectedVersion": champ,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_out,
        "aliasMutation": None,
        "evidence": eligible[champ]["ev"],
    }


# ==========================================================================
# Q4  POST /adapt
# ==========================================================================
PRIORITY = ["prompt_only", "retrieval", "lora", "qlora"]


def _choose(d: dict) -> dict:
    policy = d.get("policy")
    cands = d.get("candidates")

    policy_ok = isinstance(policy, dict) and (
        in_unit(policy.get("minQuality"))
        and isinstance(policy.get("freshnessRequired"), bool)
        and is_finite(policy.get("maxLatencyMs"))
        and float(policy.get("maxLatencyMs", -1)) >= 0
        and is_finite(policy.get("maxMemoryMb"))
        and float(policy.get("maxMemoryMb", -1)) >= 0
        and is_safe_int(policy.get("maxLabeledExamples"), minimum=0)
        and is_finite(policy.get("maxTotalCost"))
        and float(policy.get("maxTotalCost", -1)) >= 0
        and is_safe_int(policy.get("horizonRequests"), minimum=0)
    )

    by_name: dict[str, dict] = {}
    shape_ok = isinstance(cands, list)
    if shape_ok:
        for c in cands:
            if not isinstance(c, dict) or c.get("name") not in PRIORITY or c["name"] in by_name:
                shape_ok = False
                break
            by_name[c["name"]] = c
        if set(by_name.keys()) != set(PRIORITY):
            shape_ok = False

    def cand_ok(c):
        return (
            isinstance(c.get("available"), bool)
            and in_unit(c.get("quality"))
            and isinstance(c.get("freshness"), bool)
            and is_finite(c.get("latencyMs"))
            and float(c["latencyMs"]) >= 0
            and is_finite(c.get("memoryMb"))
            and float(c["memoryMb"]) >= 0
            and is_safe_int(c.get("labeledExamples"), minimum=0)
            and is_finite(c.get("oneTimeCost"))
            and float(c["oneTimeCost"]) >= 0
            and is_finite(c.get("recurringCost"))
            and float(c["recurringCost"]) >= 0
        )

    if not policy_ok or not shape_ok or not all(cand_ok(c) for c in by_name.values()):
        return {
            "selected": None,
            "eligible": [],
            "totalCosts": {n: None for n in PRIORITY},
            "reasonCodes": {n: ["INVALID_INPUT"] for n in PRIORITY},
        }

    total_costs, reason_codes, eligible = {}, {}, []
    for name in PRIORITY:
        c = by_name[name]
        total = r12(float(c["oneTimeCost"]) + policy["horizonRequests"] * float(c["recurringCost"]))
        total_costs[name] = total
        cs = []
        if not c["available"]:
            cs.append("UNAVAILABLE")
        if float(c["quality"]) < float(policy["minQuality"]):
            cs.append("QUALITY_FLOOR")
        if policy["freshnessRequired"] and not c["freshness"]:
            cs.append("FRESHNESS_REQUIRED")
        if float(c["latencyMs"]) > float(policy["maxLatencyMs"]):
            cs.append("LATENCY_LIMIT")
        if float(c["memoryMb"]) > float(policy["maxMemoryMb"]):
            cs.append("MEMORY_LIMIT")
        if c["labeledExamples"] > policy["maxLabeledExamples"]:
            cs.append("DATA_LIMIT")
        if total > float(policy["maxTotalCost"]):
            cs.append("COST_LIMIT")
        reason_codes[name] = codes(cs)
        if not cs:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes,
    }


LORA_SUFFIX = (".lora_A.weight", ".lora_B.weight")
UNSAFE_EXT = (".bin", ".pt", ".pth", ".pkl", ".pickle")
ADAPTER_SET = ["adapter_config.json", "adapter_model.safetensors"]
CKPT_KEYS = {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}


def _repair(d: dict) -> dict:
    cs: list[str] = []

    # --- tokens / labels
    tokens = d.get("tokens")
    tokens_ok = isinstance(tokens, list) and len(tokens) > 0
    if tokens_ok:
        for t in tokens:
            if (
                not isinstance(t, dict)
                or not is_safe_int(t.get("id"), minimum=0)
                or t.get("role") not in ("system", "user", "assistant")
                or not isinstance(t.get("padding"), bool)
                or not is_str(t.get("text"))
            ):
                tokens_ok = False
                break
    if tokens_ok:
        labels = [
            t["id"] if (t["role"] == "assistant" and not t["padding"]) else -100 for t in tokens
        ]
    else:
        cs.append("INVALID_TOKEN")
        labels = [-100] * (len(tokens) if isinstance(tokens, list) else 0)

    # --- chat template
    template_pass = d.get("templateApplications") == 1 and not isinstance(
        d.get("templateApplications"), bool
    )
    if not template_pass:
        cs.append("CHAT_TEMPLATE_COUNT")

    # --- PEFT parameters
    params = d.get("parameters")
    allowed = d.get("allowedTargets")
    allowed_ok = (
        isinstance(allowed, list)
        and len(allowed) > 0
        and all(nonempty_str(x) for x in allowed)
        and len(set(allowed)) == len(allowed)
    )
    params_ok = isinstance(params, list)
    seen_names = set()
    if params_ok:
        for p in params:
            if (
                not isinstance(p, dict)
                or not nonempty_str(p.get("name"))
                or p["name"] in seen_names
                or not is_str(p.get("target"))
                or not is_safe_int(p.get("numel"), minimum=1)
            ):
                params_ok = False
                break
            seen_names.add(p["name"])

    trainable, count = [], 0
    if params_ok and allowed_ok:
        aset = set(allowed)
        for p in params:
            if p["target"] in aset and p["name"].endswith(LORA_SUFFIX):
                trainable.append(p["name"])
                count += p["numel"]
        trainable = usort(trainable)
    peft_pass = params_ok and allowed_ok and len(trainable) > 0
    if not peft_pass:
        cs.append("INVALID_PARAMETER")

    # --- inference mode
    if d.get("inferenceMode") is not False:
        cs.append("INFERENCE_MODE")

    # --- adapter artifacts
    files = d.get("artifactFiles")
    files_ok = isinstance(files, list) and all(is_str(f) for f in files)
    adapter_files = usort(set(files)) if files_ok else []
    if not files_ok or sorted(files) != sorted(ADAPTER_SET):
        cs.append("ADAPTER_FILE_SET")
    if files_ok and any(str(f).endswith(UNSAFE_EXT) or f in ("model.safetensors",) for f in files):
        cs.append("FULL_MODEL_ARTIFACT")

    # --- checkpoint
    ckpt = d.get("checkpoint")
    ckpt_ok = isinstance(ckpt, dict) and CKPT_KEYS.issubset(set(ckpt.keys()))
    if not ckpt_ok:
        cs.append("INCOMPLETE_CHECKPOINT")

    # --- lineage
    base_rev = d.get("baseRevision")
    if not is_hex(base_rev, 40):
        cs.append("MUTABLE_BASE_REVISION")
    exp = d.get("expectedDigests") if isinstance(d.get("expectedDigests"), dict) else {}
    lineage_pass = True
    for field in ("datasetDigest", "codeDigest", "configDigest"):
        val = d.get(field)
        if not is_hex(val, 64) or exp.get(field) != val:
            lineage_pass = False
    if not lineage_pass:
        cs.append("LINEAGE_MISMATCH")

    # --- effective batch
    mb, ga, rep, exp_b = (
        d.get("microBatch"),
        d.get("gradientAccumulation"),
        d.get("replicas"),
        d.get("expectedEffectiveBatch"),
    )
    batch_ok = all(is_safe_int(x, minimum=1) for x in (mb, ga, rep, exp_b)) and mb * ga * rep == exp_b
    if not batch_ok:
        cs.append("EFFECTIVE_BATCH_MISMATCH")

    # --- eval isolation
    tr, ev = d.get("trainRowIds"), d.get("evalRowIds")

    def idset_ok(x):
        return (
            isinstance(x, list)
            and len(x) > 0
            and all(nonempty_str(i) for i in x)
            and len(set(x)) == len(x)
        )

    eval_isolated = idset_ok(tr) and idset_ok(ev) and not (set(tr) & set(ev))
    if not eval_isolated:
        cs.append("EVAL_LEAKAGE")

    eval_det = d.get("dropoutActiveDuringEval") is False
    if not eval_det:
        cs.append("EVAL_DROPOUT_ACTIVE")

    # --- resume
    uw, rw, tol = d.get("uninterruptedWeights"), d.get("resumedWeights"), d.get("resumeTolerance")
    resume_pass = (
        isinstance(uw, list)
        and isinstance(rw, list)
        and len(uw) > 0
        and len(uw) == len(rw)
        and all(is_finite(x) for x in uw)
        and all(is_finite(x) for x in rw)
        and is_finite(tol)
        and float(tol) >= 0
        and all(abs(float(a) - float(b)) <= float(tol) for a, b in zip(uw, rw))
    )
    if not resume_pass:
        cs.append("RESUME_DIVERGENCE")

    return {
        "labels": labels,
        "templatePass": bool(template_pass),
        "trainableParams": trainable,
        "trainableCount": count,
        "peftConfigPass": bool(peft_pass),
        "adapterFiles": adapter_files,
        "checkpointComplete": bool(ckpt_ok),
        "lineagePass": bool(lineage_pass and is_hex(base_rev, 40)),
        "evalIsolated": bool(eval_isolated),
        "evaluationDeterministic": bool(eval_det),
        "resumePass": bool(resume_pass),
        "reasonCodes": codes(cs),
    }


@app.post("/adapt")
async def adapt(request: Request):
    d = await body(request)
    if not isinstance(d, dict):
        return BAD
    op = d.get("operation")
    if op == "choose":
        return _choose(d)
    if op == "repair":
        return _repair(d)
    return BAD


# ==========================================================================
# Q5  POST /quantize
# ==========================================================================
FREEZES: dict[str, dict] = {}


def _inventory(files: dict) -> tuple[list[dict], int, str]:
    inv = []
    for name in sorted(files.keys(), key=lambda s: s.encode("utf-8")):
        raw = files[name].encode("utf-8")
        inv.append({"name": name, "bytes": len(raw), "sha256": sha256_hex(raw)})
    total = sum(e["bytes"] for e in inv)
    return inv, total, sha256_str(cj(inv))


def _do_freeze(d: dict):
    freeze_id = d.get("freezeId")
    if not nonempty_str(freeze_id) or len(freeze_id) > 128:
        return BAD
    cands = d.get("candidates")
    if not isinstance(cands, list) or not cands:
        return BAD

    allowed = d.get("allowedUnsupportedReasons")
    top_ok = (
        nonempty_str(d.get("calibrationDigest"))
        and nonempty_str(d.get("tokenizerDigest"))
        and isinstance(allowed, list)
        and all(nonempty_str(x) for x in allowed)
        and len(set(allowed)) == len(allowed)
    )
    allowed_set = set(allowed) if isinstance(allowed, list) else set()

    names_seen = set()
    out = []
    for c in cands:
        cs: list[str] = []
        c = c if isinstance(c, dict) else {}
        name = c.get("name")
        name_ok = nonempty_str(name) and name not in names_seen
        if name_ok:
            names_seen.add(name)
        files = c.get("files")
        files_ok = (
            isinstance(files, dict)
            and len(files) > 0
            and all(nonempty_str(k) and is_str(v) for k, v in files.items())
        )
        if not top_ok or not name_ok or not files_ok:
            cs.append("INVALID_INPUT")

        reason = c.get("unsupportedReason")
        if reason is not None:
            if not (nonempty_str(reason) and reason in allowed_set):
                cs.append("UNALLOWED_UNSUPPORTED_REASON")
        else:
            if c.get("loadable") is not True:
                cs.append("NOT_LOADABLE")
            if c.get("calibrationDigest") != d.get("calibrationDigest"):
                cs.append("CALIBRATION_MISMATCH")
            if c.get("tokenizerDigest") != d.get("tokenizerDigest"):
                cs.append("TOKENIZER_MISMATCH")

        if files_ok:
            inv, total, pkg = _inventory(files)
        else:
            inv, total, pkg = [], None, None

        if cs:
            status = "invalid"
        elif reason is not None:
            status = "unsupported"
        else:
            status = "frozen"

        out.append(
            {
                "name": name if is_str(name) else cj(name),
                "status": status,
                "inventory": inv,
                "totalBytes": total,
                "packageDigest": pkg,
                "reasonCodes": codes(cs),
            }
        )

    out.sort(key=lambda e: e["name"].encode("utf-8"))
    return {"freezeId": freeze_id, "candidates": out}


def _quantize_select(d: dict):
    freeze_id = d.get("freezeId")
    cands = d.get("candidates")
    policy = d.get("policy")
    rows = d.get("rows")
    if (
        not isinstance(cands, list)
        or not isinstance(rows, list)
        or not isinstance(policy, dict)
    ):
        return BAD

    cs_global: list[str] = []
    stored = FREEZES.get(freeze_id) if nonempty_str(freeze_id) else None
    frozen_ok = stored is not None and cands == stored["response"]["candidates"]
    if stored is None:
        cs_global.append("NOT_FROZEN")
    elif not frozen_ok:
        cs_global.append("INVALID_LINEAGE")

    order = policy.get("candidateOrder")
    policy_ok = (
        is_safe_int(policy.get("maxBytes"), minimum=0)
        and in_unit(policy.get("aggregateFloor"))
        and isinstance(policy.get("requiredSlices"), dict)
        and all(nonempty_str(k) and in_unit(v) for k, v in policy["requiredSlices"].items())
        and is_finite(policy.get("maxLatencyMs"))
        and float(policy.get("maxLatencyMs", -1)) >= 0
        and isinstance(order, list)
        and all(nonempty_str(x) for x in order)
        and len(set(order)) == len(order)
    )
    names = [c.get("name") for c in cands if isinstance(c, dict)]
    if policy_ok and set(order) != set(names):
        policy_ok = False
    if not policy_ok:
        cs_global.append("INVALID_POLICY")

    lat = d.get("latencies") if isinstance(d.get("latencies"), dict) else {}
    rows_ok = all(
        isinstance(r, dict)
        and r.get("label") in (0, 1)
        and not isinstance(r.get("label"), bool)
        and nonempty_str(r.get("slice"))
        and isinstance(r.get("predictions"), dict)
        for r in rows
    )

    results = []
    admitted = []
    for c in cands:
        c = c if isinstance(c, dict) else {}
        name = c.get("name")
        cs = list(cs_global)

        inv = c.get("inventory")
        manifest_ok = isinstance(inv, list) and all(
            isinstance(e, dict) and nonempty_str(e.get("name")) and is_safe_int(e.get("bytes"), minimum=0)
            and is_hex(e.get("sha256"), 64)
            for e in inv
        )
        total = None
        if manifest_ok:
            total = sum(e["bytes"] for e in inv)
            if c.get("packageDigest") != sha256_str(cj(inv)):
                manifest_ok = False
        if not manifest_ok:
            cs.append("INVALID_MANIFEST")
            total = None

        preds_ok = bool(rows) and rows_ok and all(
            r["predictions"].get(name) in (0, 1)
            and not isinstance(r["predictions"].get(name), bool)
            for r in rows
        )
        if not preds_ok:
            cs.append("INVALID_PREDICTIONS")

        agg = None
        slices_out: dict[str, Any] = {}
        if preds_ok:
            agg = r12(sum(1 for r in rows if r["label"] == r["predictions"][name]) / len(rows))
            if policy_ok and agg < float(policy["aggregateFloor"]):
                cs.append("AGGREGATE_FLOOR")
            if policy_ok:
                for sname, floor in policy["requiredSlices"].items():
                    sub = [r for r in rows if r["slice"] == sname]
                    if not sub:
                        cs.append(f"MISSING_SLICE:{sname}")
                        slices_out[sname] = None
                        continue
                    val = r12(
                        sum(1 for r in sub if r["label"] == r["predictions"][name]) / len(sub)
                    )
                    slices_out[sname] = val
                    if val < float(floor):
                        cs.append(f"SLICE_FLOOR:{sname}")
        elif policy_ok:
            for sname in policy["requiredSlices"]:
                slices_out[sname] = None

        if policy_ok and total is not None and total > policy["maxBytes"]:
            cs.append("SIZE_LIMIT")

        lv = lat.get(name)
        latency = float(lv) if is_finite(lv) and float(lv) >= 0 else None
        if policy_ok and latency is not None and latency > float(policy["maxLatencyMs"]):
            cs.append("LATENCY_LIMIT")
        if latency is None:
            cs.append("INVALID_POLICY")

        ok = (
            not cs
            and c.get("status") == "frozen"
            and total is not None
            and latency is not None
        )
        results.append(
            {
                "name": name,
                "aggregate": agg,
                "slices": slices_out,
                "totalBytes": total,
                "latencyMs": latency,
                "admitted": bool(ok),
                "reasonCodes": codes(cs),
            }
        )
        if ok:
            admitted.append((total, latency, order.index(name) if policy_ok else 0, c, name))

    order_index = {n: i for i, n in enumerate(order)} if policy_ok else {}
    results.sort(
        key=lambda r: (order_index.get(r["name"], len(order_index)), str(r["name"]).encode("utf-8"))
    )

    winner = None
    if admitted:
        admitted.sort(key=lambda t: (t[0], t[1], t[2]))
        winner = admitted[0]

    return {
        "freezeId": freeze_id,
        "selected": winner[4] if winner else None,
        "results": results,
        "packageManifest": winner[3] if winner else None,
    }


@app.post("/quantize")
async def quantize(request: Request):
    d = await body(request)
    if not isinstance(d, dict):
        return BAD
    phase = d.get("phase")

    if phase == "freeze":
        resp = _do_freeze(d)
        if isinstance(resp, JSONResponse):
            return resp
        freeze_id = d["freezeId"]
        fingerprint = cj({k: v for k, v in sorted(d.items()) if k != "phase"})
        if freeze_id in FREEZES:
            if FREEZES[freeze_id]["fingerprint"] != fingerprint:
                return JSONResponse(status_code=409, content={"error": "FREEZE_ID_CONFLICT"})
            return FREEZES[freeze_id]["response"]
        FREEZES[freeze_id] = {"fingerprint": fingerprint, "response": resp}
        return resp

    if phase == "select":
        return _quantize_select(d)

    return BAD


# ==========================================================================
# Q6  POST /pipeline
# ==========================================================================
DAG = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
PARENT = {DAG[i]: (DAG[i - 1] if i else None) for i in range(len(DAG))}
INPUT_KEYS = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]
DEPS = {
    "verify_data": ["generation", "checksum"],
    "prepare": ["canonicalData", "prepareCode", "prepareConfig"],
    "train": ["prepareArtifact", "trainCode", "trainConfig", "runtime"],
    "evaluate": ["trainArtifact", "canonicalData", "evaluateCode", "evaluateConfig"],
    "register": ["evaluateArtifact", "schemaDigest"],
    "publish": ["registerArtifact", "publishConfig"],
}
EVENT_KEYS = {
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
}
STATUSES = ("started", "succeeded", "retryable_failed", "terminal_failed")

SESSIONS: dict[str, dict] = {}


def conflict(code: str):
    return JSONResponse(status_code=409, content={"error": code})


@app.post("/pipeline")
async def pipeline(request: Request):
    d = await body(request)
    if not isinstance(d, dict):
        return conflict("INVALID_REQUEST")

    session = d.get("session")
    revision = d.get("revision")
    inputs = d.get("inputs")
    events = d.get("events")
    if (
        not nonempty_str(session)
        or not is_safe_int(revision, minimum=1)
        or not isinstance(inputs, dict)
        or not all(nonempty_str(inputs.get(k)) for k in INPUT_KEYS)
        or not isinstance(events, list)
    ):
        return conflict("INVALID_REQUEST")

    st = SESSIONS.setdefault(
        session, {"revision": None, "inputs_fp": None, "cache": {}, "nodes": {}, "events": {}}
    )
    inputs_fp = cj({k: inputs[k] for k in sorted(inputs.keys())})

    if st["revision"] == revision:
        if st["inputs_fp"] != inputs_fp:
            return conflict("REVISION_CONFLICT")
    else:
        st["revision"] = revision
        st["inputs_fp"] = inputs_fp
        st["nodes"] = {}

    cache = st["cache"]

    def keys_now():
        ks: dict[str, str | None] = {}
        arts: dict[str, str] = {}
        for node in DAG:
            parent = PARENT[node]
            if parent is not None:
                pk = ks.get(parent)
                if pk is None or pk not in cache:
                    ks[node] = None
                    continue
            parts = []
            for dep in DEPS[node]:
                if dep.endswith("Artifact"):
                    pname = dep[: -len("Artifact")]
                    pkey = ks.get(pname)
                    parts.append(cache[pkey]["artifact"])
                else:
                    parts.append(inputs[dep])
            ks[node] = sha256_str(cj(parts))
            if ks[node] in cache:
                arts[node] = cache[ks[node]]["artifact"]
        return ks

    # ---- snapshot for rollback
    snap = (
        json.loads(cj(st["cache"])),
        json.loads(cj(st["nodes"])),
        json.loads(cj(st["events"])),
    )

    accepted: list[str] = []
    ignored: list[str] = []

    for raw in events:
        if not isinstance(raw, dict) or set(raw.keys()) != EVENT_KEYS:
            st["cache"], st["nodes"], st["events"] = snap
            return conflict("INVALID_EVENT")
        eid = raw["eventId"]
        if (
            not nonempty_str(eid)
            or not is_safe_int(raw["revision"], minimum=1)
            or not is_str(raw["node"])
            or not is_safe_int(raw["attempt"], minimum=1)
            or not is_str(raw["status"])
        ):
            st["cache"], st["nodes"], st["events"] = snap
            return conflict("INVALID_EVENT")

        canon = cj({k: raw[k] for k in sorted(EVENT_KEYS)})
        if eid in st["events"]:
            if st["events"][eid] == canon:
                ignored.append(eid)
                continue
            st["cache"], st["nodes"], st["events"] = snap
            return conflict("EVENT_ID_CONFLICT")

        node = raw["node"]
        status = raw["status"]
        ks = keys_now()

        # ---- ignore conditions
        if raw["revision"] != revision or node not in DAG or status not in STATUSES:
            ignored.append(eid)
            continue
        key = ks.get(node)
        if key is None or raw["key"] != key:
            ignored.append(eid)
            continue
        if status == "succeeded":
            if not nonempty_str(raw["artifactDigest"]):
                ignored.append(eid)
                continue
            if node in ("register", "publish"):
                if raw["receiptId"] != f"receipt:{node}:{key}":
                    ignored.append(eid)
                    continue
            elif raw["receiptId"] is not None:
                ignored.append(eid)
                continue
        else:
            if raw["artifactDigest"] is not None or raw["receiptId"] is not None:
                ignored.append(eid)
                continue

        cached = key in cache
        prev = st["nodes"].get(node)
        if prev is not None and prev.get("key") != key:
            prev = None

        if cached:
            if status == "succeeded" and raw["artifactDigest"] != cache[key]["artifact"]:
                st["cache"], st["nodes"], st["events"] = snap
                return conflict("EVIDENCE_CONFLICT")
            st["cache"], st["nodes"], st["events"] = snap
            return conflict("STATUS_CONFLICT")

        if prev is None:
            if status == "started" and raw["attempt"] == 1:
                st["nodes"][node] = {"status": "started", "attempt": 1, "key": key, "eventId": eid}
            else:
                ignored.append(eid)
                continue
        elif prev["status"] == "started":
            if raw["attempt"] < prev["attempt"]:
                ignored.append(eid)
                continue
            if status in ("succeeded", "retryable_failed", "terminal_failed") and raw["attempt"] == prev["attempt"]:
                if status == "succeeded":
                    cache[key] = {"artifact": raw["artifactDigest"], "eventId": eid}
                    st["nodes"][node] = {
                        "status": "succeeded",
                        "attempt": raw["attempt"],
                        "key": key,
                        "eventId": eid,
                    }
                else:
                    st["nodes"][node] = {
                        "status": status,
                        "attempt": raw["attempt"],
                        "key": key,
                        "eventId": eid,
                    }
            else:
                st["cache"], st["nodes"], st["events"] = snap
                return conflict("STATUS_CONFLICT")
        elif prev["status"] == "retryable_failed":
            if raw["attempt"] < prev["attempt"]:
                ignored.append(eid)
                continue
            if status == "started" and raw["attempt"] == prev["attempt"] + 1:
                st["nodes"][node] = {
                    "status": "started",
                    "attempt": raw["attempt"],
                    "key": key,
                    "eventId": eid,
                }
            else:
                st["cache"], st["nodes"], st["events"] = snap
                return conflict("STATUS_CONFLICT")
        else:  # terminal_failed or succeeded (non-cached shouldn't happen)
            st["cache"], st["nodes"], st["events"] = snap
            return conflict("STATUS_CONFLICT")

        st["events"][eid] = canon
        accepted.append(eid)

    # ---- build response
    ks = keys_now()
    nodes_out = []
    upstream_terminal = False
    upstream_pending = False
    for node in DAG:
        key = ks.get(node)
        dd: dict[str, Any] = {}
        for dep in DEPS[node]:
            if dep.endswith("Artifact"):
                pname = dep[: -len("Artifact")]
                pkey = ks.get(pname)
                dd[dep] = cache[pkey]["artifact"] if pkey and pkey in cache else None
            else:
                dd[dep] = inputs[dep]
        dd["cacheKey"] = key

        if upstream_terminal:
            entry = {"action": "block", "reasonCodes": ["UPSTREAM_TERMINAL"], "triggeringEventIds": []}
        elif key is None or upstream_pending:
            entry = {"action": "block", "reasonCodes": ["UPSTREAM_PENDING"], "triggeringEventIds": []}
        elif key in cache:
            entry = {
                "action": "reuse",
                "reasonCodes": ["CACHE_HIT"],
                "triggeringEventIds": [cache[key]["eventId"]],
            }
        else:
            state = st["nodes"].get(node)
            if state is not None and state.get("key") != key:
                state = None
            if state is None:
                entry = {"action": "rerun", "reasonCodes": ["CACHE_MISS"], "triggeringEventIds": []}
                upstream_pending = True
            elif state["status"] == "started":
                entry = {
                    "action": "block",
                    "reasonCodes": ["RUNNING"],
                    "triggeringEventIds": [state["eventId"]],
                }
                upstream_pending = True
            elif state["status"] == "retryable_failed":
                entry = {
                    "action": "rerun",
                    "reasonCodes": ["RETRYABLE_FAILURE"],
                    "triggeringEventIds": [state["eventId"]],
                }
                upstream_pending = True
            else:  # terminal_failed
                entry = {
                    "action": "block",
                    "reasonCodes": ["TERMINAL_FAILURE"],
                    "triggeringEventIds": [state["eventId"]],
                }
                upstream_terminal = True

        entry.update({"node": node, "dependencyDigests": dd})
        nodes_out.append(
            {
                "node": node,
                "action": entry["action"],
                "reasonCodes": entry["reasonCodes"],
                "dependencyDigests": dd,
                "triggeringEventIds": entry["triggeringEventIds"],
            }
        )

    return {
        "revision": revision,
        "acceptedEventIds": accepted,
        "ignoredEventIds": ignored,
        "nodes": nodes_out,
    }


# ==========================================================================
# Q7  POST /verify-bundle
# ==========================================================================
REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]
MARKER_PREFIX = "<!-- tds-model-card"
MANIFEST_FIELDS = [
    "task",
    "datasetDigest",
    "codeDigest",
    "trainingConfigDigest",
    "modelArtifactDigest",
    "evaluationArtifactDigest",
]


def _find_markers(readme: str) -> list[str]:
    """Return each marker payload, treating braces inside JSON strings as text."""
    out = []
    i = 0
    while True:
        j = readme.find(MARKER_PREFIX, i)
        if j < 0:
            break
        k = j + len(MARKER_PREFIX)
        in_str = False
        esc = False
        end = -1
        while k < len(readme):
            ch = readme[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif readme.startswith("-->", k):
                    end = k
                    break
            k += 1
        if end < 0:
            out.append(readme[j + len(MARKER_PREFIX):])
            i = len(readme)
        else:
            out.append(readme[j + len(MARKER_PREFIX):end])
            i = end + 3
    return out


@app.post("/verify-bundle")
async def verify_bundle(request: Request):
    d = await body(request)
    if not isinstance(d, dict):
        return BAD
    policy = d.get("policy")
    files = d.get("files")
    if not isinstance(policy, dict) or not isinstance(files, dict):
        return BAD

    v: list[str] = []

    rs = policy.get("requiredSlices")
    policy_ok = (
        isinstance(rs, list)
        and len(rs) > 0
        and all(nonempty_str(x) for x in rs)
        and len(set(rs)) == len(rs)
        and nonempty_str(policy.get("license"))
        and nonempty_str(policy.get("intendedUse"))
        and nonempty_str(policy.get("limitations"))
    )
    if not policy_ok:
        v.append("INVALID_POLICY")

    for name, content in files.items():
        if not is_str(content):
            v.append(f"INVALID_FILE:{name}")
    for name in REQUIRED_FILES:
        if name not in files or not is_str(files.get(name)):
            v.append(f"MISSING_FILE:{name}")

    for name in files:
        if name not in REQUIRED_FILES:
            v.append("UNTRACKED_FILE")
        if str(name).endswith(UNSAFE_EXT):
            v.append("UNSAFE_WEIGHTS")

    def raw(name):
        c = files.get(name)
        return c.encode("utf-8") if is_str(c) else None

    # ---- inventory
    recomputed = []
    for name in sorted([n for n in files if n != "inventory.json"], key=lambda s: s.encode("utf-8")):
        b = raw(name)
        if b is None:
            continue
        recomputed.append({"name": name, "bytes": len(b), "sha256": sha256_hex(b)})
    inventory_digest = sha256_str(cj(recomputed))

    if is_str(files.get("inventory.json")):
        try:
            parsed_inv = json.loads(files["inventory.json"])
            if files["inventory.json"] != cj(recomputed):
                v.append("INVENTORY_MISMATCH")
            if not isinstance(parsed_inv, list):
                v.append("INVALID_FILE:inventory.json")
        except Exception:
            v.append("INVALID_JSON:inventory.json")
            v.append("INVENTORY_MISMATCH")

    # ---- adapter_config.json
    cfg = None
    if is_str(files.get("adapter_config.json")):
        try:
            cfg = json.loads(files["adapter_config.json"])
        except Exception:
            v.append("INVALID_JSON:adapter_config.json")
        if cfg is not None:
            tm = cfg.get("target_modules") if isinstance(cfg, dict) else None
            if (
                not isinstance(cfg, dict)
                or not is_safe_int(cfg.get("r"), minimum=1)
                or not isinstance(tm, list)
                or not tm
                or not all(nonempty_str(x) for x in tm)
                or len(set(tm)) != len(tm)
            ):
                v.append("INVALID_ADAPTER_CONFIG")

    # ---- training_manifest.json
    man = None
    if is_str(files.get("training_manifest.json")):
        try:
            man = json.loads(files["training_manifest.json"])
        except Exception:
            v.append("INVALID_JSON:training_manifest.json")
        if man is not None and not isinstance(man, dict):
            v.append("INVALID_TRAINING_MANIFEST")
            man = None
    if isinstance(man, dict):
        if not is_hex(man.get("baseRevision"), 40):
            v.append("MUTABLE_BASE_REVISION")
        for f in MANIFEST_FIELDS:
            if not nonempty_str(man.get(f)):
                v.append(f"MISSING_MANIFEST_FIELD:{f}")
        mb = raw("adapter_model.safetensors")
        if mb is not None and man.get("modelArtifactDigest") != sha256_hex(mb):
            v.append("MODEL_ARTIFACT_MISMATCH")
        eb = raw("evaluation.json")
        if eb is not None and man.get("evaluationArtifactDigest") != sha256_hex(eb):
            v.append("EVALUATION_ARTIFACT_MISMATCH")

    # ---- evaluation.json
    ev = None
    if is_str(files.get("evaluation.json")):
        try:
            ev = json.loads(files["evaluation.json"])
        except Exception:
            v.append("INVALID_JSON:evaluation.json")
        if ev is not None and not isinstance(ev, dict):
            v.append("INVALID_EVALUATION")
            ev = None
    if isinstance(ev, dict):
        if isinstance(man, dict) and ev.get("modelArtifactDigest") != man.get("modelArtifactDigest"):
            v.append("EVALUATION_DIGEST_MISMATCH")
        if not in_unit(ev.get("aggregate")):
            v.append("INVALID_AGGREGATE")
        slices = ev.get("slices") if isinstance(ev.get("slices"), dict) else {}
        if policy_ok:
            for s in rs:
                if s not in slices:
                    v.append(f"MISSING_SLICE:{s}")
                elif not in_unit(slices[s]):
                    v.append(f"SLICE_RANGE:{s}")

    # ---- model card
    readme = files.get("README.md")
    if is_str(readme):
        markers = _find_markers(readme)
        if len(markers) == 0:
            v.append("MODEL_CARD_COUNT")
            v.append("MISSING_MODEL_CARD")
        elif len(markers) > 1:
            v.append("MODEL_CARD_COUNT")
        else:
            try:
                card = json.loads(markers[0].strip())
            except Exception:
                card = None
            if not isinstance(card, dict):
                v.append("INVALID_MODEL_CARD")
            else:
                mismatch = False
                if isinstance(man, dict):
                    for f in ("task", "baseRevision", "datasetDigest", "modelArtifactDigest"):
                        if card.get(f) != man.get(f):
                            mismatch = True
                else:
                    mismatch = True
                if policy_ok:
                    for f in ("license", "intendedUse", "limitations"):
                        if card.get(f) != policy.get(f):
                            mismatch = True
                else:
                    mismatch = True
                if mismatch:
                    v.append("MODEL_CARD_MISMATCH")

    violations = codes(v)
    return {
        "decision": "admit" if not violations else "reject",
        "violations": violations,
        "inventoryDigest": inventory_digest,
    }


@app.get("/")
async def root():
    return {"ok": True, "endpoints": ["/build-corpus", "/bqml", "/promote", "/adapt", "/quantize", "/pipeline", "/verify-bundle"]}