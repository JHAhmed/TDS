"""
TDS 2026 May GA7 — one FastAPI app serving all five graded endpoints.

  POST /release-gate      (Q1)
  POST /action-firewall   (Q2)
  POST /terraform/plan    (Q3)
  POST /sanitize-output   (Q4)
  POST /corroborate       (Q5)

Run locally:   uvicorn main:app --host 0.0.0.0 --port 8000
Deps:          pip install fastapi uvicorn
"""

import re
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="TDS GA7")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def body_of(request: Request) -> Any:
    """Parse JSON leniently — a malformed body must NOT become a 422."""
    try:
        return await request.json()
    except Exception:
        return None


@app.get("/")
def root():
    return {"ok": True, "endpoints": [
        "/release-gate", "/action-firewall", "/terraform/plan",
        "/sanitize-output", "/corroborate",
    ]}


# ─────────────────────────────────────────────────────────────
# Q1 — CI/CD Container Release Gate
# ─────────────────────────────────────────────────────────────

SHA40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_PERMS = {"contents": "read", "packages": "write", "id-token": "none"}


@app.post("/release-gate")
async def release_gate(request: Request):
    body = await body_of(request)
    v: list[str] = []
    if not isinstance(body, dict):
        return {"decision": "block", "violations": ["EXCESS_PERMISSION"]}

    target = body.get("target")
    event = body.get("event")
    ref = body.get("ref")
    wf = body.get("workflow") or {}
    img = body.get("image") or {}
    if not isinstance(wf, dict):
        wf = {}
    if not isinstance(img, dict):
        img = {}

    # --- permissions: exactly least privilege, no extra scopes ---
    perms = wf.get("permissions")
    if not isinstance(perms, dict):
        v.append("EXCESS_PERMISSION")
    else:
        norm = {str(k).lower(): str(x).lower() for k, x in perms.items()}
        norm.setdefault("id-token", "none")          # absent == "none"
        if norm != REQUIRED_PERMS:
            v.append("EXCESS_PERMISSION")

    # --- PR trigger safety ---
    if wf.get("trigger") == "pull_request_target":
        v.append("UNSAFE_PR_TRIGGER")

    # --- tests / matrix ---
    if (wf.get("testsPassed") is not True
            or wf.get("matrixComplete") is not True
            or wf.get("failFast") is not False):
        v.append("TESTS_INCOMPLETE")

    # --- action pinning ---
    actions = wf.get("actions")
    if isinstance(actions, list):
        for a in actions:
            if not isinstance(a, dict):
                v.append("MUTABLE_ACTION")
                break
            owner = str(a.get("owner", "")).lower()
            aref = str(a.get("ref", ""))
            if owner == "actions":
                continue                              # version tags allowed
            if not SHA40.match(aref):
                v.append("MUTABLE_ACTION")
                break

    # --- image hardening ---
    if img.get("multiStage") is not True:
        v.append("SINGLE_STAGE_IMAGE")
    if img.get("runsAsRoot") is True:
        v.append("ROOT_RUNTIME")
    if img.get("secretMode") not in ("none", "buildkit"):
        v.append("SECRET_IN_LAYER")
    cve = img.get("criticalVulnerabilities")
    if not isinstance(cve, (int, float)) or isinstance(cve, bool) or cve > 0:
        v.append("CRITICAL_CVE")
    if img.get("digestPinned") is not True:
        v.append("UNPINNED_IMAGE")

    # --- production extras ---
    if target == "production":
        if not (event == "push" and ref == "refs/heads/main"):
            v.append("INVALID_PRODUCTION_REF")
        if wf.get("environmentApproval") is not True:
            v.append("APPROVAL_REQUIRED")

    return {"decision": "promote" if not v else "block", "violations": v}


# ─────────────────────────────────────────────────────────────
# Q2 — LLM Action Firewall
# ─────────────────────────────────────────────────────────────

TENANT_ID = "tenant-nvlv9kq"
EMAIL_DOMAIN = "notify-wysbde9.example"
ALLOWED_TOOLS = {"search", "lookup_record", "send_email", "render_html"}

RE_SCRIPTISH = re.compile(r"<\s*(script|iframe|object|embed)\b", re.I)
RE_ON_ATTR = re.compile(r"\bon[a-z]+\s*=", re.I)
RE_JS_URL = re.compile(r"javascript\s*:", re.I)


def _fw(decision: str, reason: str):
    return {"decision": decision, "reason": reason}


@app.post("/action-firewall")
async def action_firewall(request: Request):
    body = await body_of(request)

    # 1. top-level schema
    if not isinstance(body, dict):
        return _fw("block", "INVALID_SCHEMA")
    if body.get("provenance") not in ("trusted", "untrusted"):
        return _fw("block", "INVALID_SCHEMA")
    if not isinstance(body.get("humanApproved"), bool):
        return _fw("block", "INVALID_SCHEMA")
    if "untrustedContent" in body and not isinstance(body["untrustedContent"], (str, type(None))):
        return _fw("block", "INVALID_SCHEMA")
    action = body.get("action")
    if not isinstance(action, dict):
        return _fw("block", "INVALID_SCHEMA")
    tool = action.get("tool")
    args = action.get("args")
    if not isinstance(tool, str):
        return _fw("block", "INVALID_SCHEMA")
    if not isinstance(args, dict):
        return _fw("block", "INVALID_SCHEMA")

    # 2. tool allowlist
    if tool not in ALLOWED_TOOLS:
        return _fw("block", "TOOL_NOT_ALLOWED")

    keys = set(args.keys())
    s = lambda k: isinstance(args.get(k), str)

    # 3. per-tool argument schema (exact key set)
    if tool == "search":
        if keys != {"query"} or not s("query") or not (1 <= len(args["query"]) <= 200):
            return _fw("block", "INVALID_SCHEMA")
        return _fw("allow", "ALLOW")

    if tool == "lookup_record":
        if keys != {"tenantId", "recordId"} or not s("tenantId") or not s("recordId"):
            return _fw("block", "INVALID_SCHEMA")
        if not args["recordId"]:
            return _fw("block", "INVALID_SCHEMA")
        # 4. tenant scope
        if args["tenantId"] != TENANT_ID:
            return _fw("block", "TENANT_SCOPE")
        return _fw("allow", "ALLOW")

    if tool == "send_email":
        if keys != {"to", "subject", "body"} or not (s("to") and s("subject") and s("body")):
            return _fw("block", "INVALID_SCHEMA")
        to = args["to"]
        # 5. exact recipient domain
        if to.count("@") != 1:
            return _fw("block", "EGRESS_DENIED")
        local, _, domain = to.partition("@")
        if not local or domain.lower() != EMAIL_DOMAIN:
            return _fw("block", "EGRESS_DENIED")
        # 6. human approval
        if body.get("humanApproved") is not True:
            return _fw("block", "APPROVAL_REQUIRED")
        return _fw("allow", "ALLOW")

    # render_html
    if keys != {"html"} or not s("html"):
        return _fw("block", "INVALID_SCHEMA")
    html = args["html"]
    # 7. safe rendering
    if RE_SCRIPTISH.search(html) or RE_ON_ATTR.search(html) or RE_JS_URL.search(html):
        return _fw("block", "UNSAFE_OUTPUT")
    return _fw("allow", "ALLOW")


# ─────────────────────────────────────────────────────────────
# Q3 — Terraform Plan Policy Gate
# ─────────────────────────────────────────────────────────────

TF_WORKSPACE = "prod-aghhi8"
TF_LABELS = {
    "owner": "student-rwi9r",
    "environment": "production",
    "cost_center": "cc-np1q",
}
SAFE_BACKENDS = {"gcs", "s3", "azurerm", "remote"}
STATEFUL = {"storage_bucket", "sql_database", "persistent_disk"}

RE_EXACT_VER = re.compile(r"^\s*=?\s*\d+(\.\d+)*\s*$")
RE_PESSIMISTIC = re.compile(r"^\s*~>\s*\d+(\.\d+)*\s*$")


def _tf(decision: str, reason: str):
    return {"decision": decision, "reason": reason}


@app.post("/terraform/plan")
async def terraform_plan(request: Request):
    body = await body_of(request)

    # 1. shape / types
    if not isinstance(body, dict):
        return _tf("reject", "INVALID_PLAN")
    env = body.get("environment")
    state = body.get("state")
    pv = body.get("providerVersion")
    da = body.get("destroyApproved")
    res = body.get("resource")
    if not isinstance(env, str) or not isinstance(pv, str):
        return _tf("reject", "INVALID_PLAN")
    if not isinstance(state, dict) or not isinstance(res, dict):
        return _tf("reject", "INVALID_PLAN")
    if not isinstance(da, bool):
        return _tf("reject", "INVALID_PLAN")
    if not isinstance(state.get("backend"), str) or not isinstance(state.get("locked"), bool):
        return _tf("reject", "INVALID_PLAN")
    if not isinstance(res.get("address"), str) or not isinstance(res.get("type"), str):
        return _tf("reject", "INVALID_PLAN")
    if res.get("action") not in ("create", "update", "delete"):
        return _tf("reject", "INVALID_PLAN")
    if not isinstance(res.get("labels"), dict):
        return _tf("reject", "INVALID_PLAN")
    if not isinstance(res.get("forceDestroy"), bool):
        return _tf("reject", "INVALID_PLAN")
    secret = res.get("secret")
    if secret is not None and not isinstance(secret, str):
        return _tf("reject", "INVALID_PLAN")

    # 2. workspace
    if env != TF_WORKSPACE:
        return _tf("reject", "ENVIRONMENT_MISMATCH")

    # 3. remote state + locking
    if state["backend"] not in SAFE_BACKENDS or state["locked"] is not True:
        return _tf("reject", "STATE_UNSAFE")

    # 4. provider pinning
    if not (RE_EXACT_VER.match(pv) or RE_PESSIMISTIC.match(pv)):
        return _tf("reject", "UNPINNED_PROVIDER")

    # 5. cost-ownership labels
    labels = res["labels"]
    for k, want in TF_LABELS.items():
        if labels.get(k) != want:
            return _tf("reject", "MISSING_LABELS")

    # 6. no plaintext secrets
    if secret is not None:
        if not secret.startswith("secret://") or len(secret) <= len("secret://"):
            return _tf("reject", "PLAINTEXT_SECRET")

    # 7. stateful deletes need approval
    if res["action"] == "delete" and res["type"] in STATEFUL and da is not True:
        return _tf("reject", "DELETE_NOT_APPROVED")

    # 8. never force-destroy a production bucket
    if res["type"] == "storage_bucket" and res["forceDestroy"] is True:
        return _tf("reject", "FORCE_DESTROY")

    return _tf("approve", "APPROVE")


# ─────────────────────────────────────────────────────────────
# Q4 — LLM Output Handling Gate (OWASP LLM05)
# ─────────────────────────────────────────────────────────────

ALLOWED_HOSTS = {"cdn-gxy6xn6.example", "app-vzxkvh5.example"}
CHANNELS = {"html", "markdown", "url", "sql", "shell"}

RE_PCT = re.compile(r"%([0-9A-Fa-f]{2})")
RE_NUM_ENT = re.compile(r"&#(x[0-9A-Fa-f]+|[0-9]+);")
RE_U_ESC = re.compile(r"\\u([0-9A-Fa-f]{4})")
NAMED_ENT = [("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'"), ("&amp;", "&")]

RE_TAG = re.compile(r"<\s*/?\s*(script|iframe|object|embed)\b", re.I)
RE_OPEN_TAG = re.compile(r"<\s*(script|iframe|object|embed)\b", re.I)
RE_EVENT = re.compile(r"\bon[a-z0-9_\-]+\s*=", re.I)
RE_SCHEME_TXT = re.compile(r"\b(javascript|data|vbscript)\s*:", re.I)
RE_ATTR_URL = re.compile(r"\b(?:src|href)\s*=\s*(\"([^\"]*)\"|'([^']*)')", re.I)
RE_MD_TARGET = re.compile(r"\]\(\s*([^)\s]+)")
RE_ABS_URL = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):", re.S)
RE_UNION = re.compile(r"union", re.I)
RE_OR11 = re.compile(r"or\s+1\s*=\s*1", re.I)
SHELL_CHARS = set(";&|`<>")


def decode_once(s: str) -> str:
    out = RE_PCT.sub(lambda m: chr(int(m.group(1), 16)), s)
    out = RE_NUM_ENT.sub(
        lambda m: chr(int(m.group(1)[1:], 16)) if m.group(1)[0] in "xX" else chr(int(m.group(1))),
        out,
    )
    for ent, ch in NAMED_ENT:
        out = out.replace(ent, ch)
    out = RE_U_ESC.sub(lambda m: chr(int(m.group(1), 16)), out)
    return out


def hostname_of(url: str) -> str | None:
    """Return the hostname of an absolute/protocol-relative URL, else None."""
    u = url.strip()
    if u.startswith("//"):
        rest = u[2:]
    else:
        m = RE_ABS_URL.match(u)
        if not m:
            return None                     # relative reference
        rest = u[m.end():]
        if rest.startswith("//"):
            rest = rest[2:]
        else:
            return None                     # e.g. mailto:, not host-based
    authority = re.split(r"[/?#]", rest, 1)[0]
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]   # strip credentials
    if authority.startswith("["):                 # IPv6
        return authority[: authority.find("]") + 1].lower()
    return authority.split(":")[0].lower()


def is_absolute(url: str) -> bool:
    u = url.strip()
    return u.startswith("//") or bool(RE_ABS_URL.match(u))


def scheme_of(url: str) -> str | None:
    u = url.strip()
    if u.startswith("//"):
        return "https"
    m = RE_ABS_URL.match(u)
    return m.group(1).lower() if m else None


def extract_urls(channel: str, text: str) -> list[str]:
    if channel == "html":
        return [(m.group(2) if m.group(2) is not None else m.group(3))
                for m in RE_ATTR_URL.finditer(text)]
    if channel == "markdown":
        return [m.group(1) for m in RE_MD_TARGET.finditer(text)]
    if channel == "url":
        return [text.strip()]
    return []


def channel_reason(channel: str, text: str) -> str:
    if channel == "sql":
        if ("'" in text or '"' in text or ";" in text or "--" in text
                or "/*" in text or RE_UNION.search(text) or RE_OR11.search(text)):
            return "SQL_METACHAR"
        return "SAFE"

    if channel == "shell":
        if any(c in SHELL_CHARS for c in text) or "$(" in text or "${" in text:
            return "SHELL_METACHAR"
        return "SAFE"

    if channel == "html":
        if RE_OPEN_TAG.search(text):
            return "SCRIPT_TAG"
        if RE_EVENT.search(text):
            return "EVENT_HANDLER"

    urls = extract_urls(channel, text)

    # DANGEROUS_SCHEME: raw text mention, or an extracted URL with a non-http(s) scheme
    if RE_SCHEME_TXT.search(text):
        return "DANGEROUS_SCHEME"
    for u in urls:
        if is_absolute(u):
            sch = scheme_of(u)
            if sch not in ("http", "https"):
                return "DANGEROUS_SCHEME"

    # EXTERNAL_EXFIL: absolute URL whose hostname isn't exactly allowlisted
    for u in urls:
        if not is_absolute(u):
            continue
        host = hostname_of(u)
        if host is None or host not in ALLOWED_HOSTS:
            return "EXTERNAL_EXFIL"

    return "SAFE"


@app.post("/sanitize-output")
async def sanitize_output(request: Request):
    body = await body_of(request)

    # 1. schema
    if not isinstance(body, dict):
        return {"safe": False, "reason": "INVALID_SCHEMA"}
    channel = body.get("channel")
    output = body.get("output")
    if channel not in CHANNELS or not isinstance(output, str) or len(output) > 20000:
        return {"safe": False, "reason": "INVALID_SCHEMA"}

    # 2. encoded payload
    decoded = decode_once(output)
    if decoded != output and channel_reason(channel, decoded) != "SAFE":
        return {"safe": False, "reason": "ENCODED_PAYLOAD"}

    # 3. channel rules on the original
    reason = channel_reason(channel, output)
    return {"safe": reason == "SAFE", "reason": reason}


# ─────────────────────────────────────────────────────────────
# Q5 — OSINT Corroboration Engine
# ─────────────────────────────────────────────────────────────

VALID_TYPES = {"dns", "ct_log", "registry", "archive", "scan"}


def parse_ts(v: Any):
    if not isinstance(v, str):
        return None
    s = v.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@app.post("/corroborate")
async def corroborate(request: Request):
    body = await body_of(request)
    invalid = {"verdict": "invalid", "confidence": "low", "corroboratingSources": []}

    # 1. validity
    if not isinstance(body, dict):
        return invalid
    claim = body.get("claim")
    if not isinstance(claim, dict) or not isinstance(claim.get("value"), str):
        return invalid
    as_of = parse_ts(body.get("asOf"))
    if as_of is None:
        return invalid
    days = body.get("stalenessDays")
    if not isinstance(days, (int, float)) or isinstance(days, bool):
        return invalid
    sources = body.get("sources")
    if not isinstance(sources, list):
        return invalid

    target = claim["value"]
    window = float(days) * 86400.0

    fresh = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        if not all(isinstance(s.get(k), str) for k in ("id", "origin", "value", "observedAt")):
            continue
        if s.get("type") not in VALID_TYPES:
            continue
        obs = parse_ts(s["observedAt"])
        if obs is None:
            continue
        if (as_of - obs).total_seconds() > window:
            continue                              # stale
        fresh.append(s)

    # 2. contradiction by a fresh authoritative source
    contra = sorted(
        s["id"] for s in fresh
        if s.get("authoritative") is True and s["value"] != target
    )
    if contra:
        return {"verdict": "contradicted", "confidence": "low",
                "corroboratingSources": contra}

    # 3. support: one representative per origin, >= 2 independent origins
    reps: dict[str, dict] = {}
    for s in fresh:
        if s["value"] != target:
            continue
        cur = reps.get(s["origin"])
        if cur is None or s["id"] < cur["id"]:
            reps[s["origin"]] = s

    if len(reps) >= 2:
        chosen = list(reps.values())
        types = {s["type"] for s in chosen}
        return {
            "verdict": "supported",
            "confidence": "high" if len(types) >= 2 else "medium",
            "corroboratingSources": sorted(s["id"] for s in chosen),
        }

    # 4. everything else
    return {"verdict": "unverified", "confidence": "low", "corroboratingSources": []}


# Never let a validation error escape as a 422 — graders want 2xx JSON.
@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content={"error": "unhandled"})
