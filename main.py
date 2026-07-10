# Q7, Q8

"""
Dual-purpose API for:
1) Invoice extraction from messy free-text → strongly-typed JSON
2) Semantic ranking of candidate passages via text-embedding-3-small

Requirements:
  pip install fastapi uvicorn openai numpy pydantic

Env:
  export OPENAI_API_KEY=sk-...

Run:
  uvicorn this_file:app --host 0.0.0.0 --port 8000

Endpoints (use public URLs for submission):
  POST /extract  — invoice → JSON
  POST /rank     — semantic search ranking
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
# Use a capable model with good structured-output support
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

app = FastAPI(title="Invoice Extractor + Semantic Ranker")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema: dict[str, Any] = Field(alias="schema")  # JSON Schema from grader

    class Config:
        populate_by_name = True


class RankRequest(BaseModel):
    query_id: str
    query: str
    candidates: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You extract invoice fields from messy free-text documents for an ERP.

Rules (strict):
- vendor: biller's proper name, exactly as written in the text.
- currency: ISO 4217 code only: USD, EUR, GBP, INR, or JPY.
  Map words/symbols: euros/€→EUR, pounds sterling/£→GBP, ₹/rupees→INR, yen/¥→JPY, dollars/$→USD.
- total_amount: integer in the main currency unit (no symbols, no separators).
  Parse: "12,480", Indian grouping "1,24,800", "12K"→12000, spelled-out amounts like "twelve thousand four hundred eighty"→12480.
- invoice_date: normalize to YYYY-MM-DD.
- due_in_days: integer days until due.
  "Net 30"→30, "payable within 45 days"→45, "due in two weeks"→14, "due in one month"→30.
- is_paid: true if wording indicates paid/settled ("paid in full", "payment received"); false if unpaid/pending ("awaiting payment", "balance due").
- priority: exactly one of: low | normal | high | urgent.
- contact_email: lowercased email only.
- line_items: array of {sku, quantity, unit_price} in the order they appear; quantity and unit_price are integers.
- item_count: number of line_items (must equal len(line_items)).

Return ONLY fields required by the provided JSON schema. No extras. No nulls for required fields. Numbers must be integers where the schema says so."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts with text-embedding-3-small."""
    # OpenAI allows many inputs per call; batch to stay safe
    out: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        # API returns data sorted by index
        ordered = sorted(resp.data, key=lambda d: d.index)
        out.extend([d.embedding for d in ordered])
    return out


# ---------------------------------------------------------------------------
# Endpoint 1: Invoice extraction
# ---------------------------------------------------------------------------

@app.post("/extract")
def extract_invoice(body: ExtractRequest) -> dict[str, Any]:
    """
    Parse messy invoice free-text into the exact schema the grader expects.
    Uses the request's JSON Schema for structured OpenAI output.
    """
    json_schema = body.schema
    if not isinstance(json_schema, dict):
        raise HTTPException(status_code=400, detail="schema must be a JSON object")

    # OpenAI strict structured outputs expect name + schema wrapper
    # Ensure schema is a proper object schema
    schema_for_api = json_schema
    if schema_for_api.get("type") != "object":
        # still pass through; grader sends object schemas
        pass

    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Document ID: {body.document_id}\n\n"
                        f"Invoice text:\n{body.text}\n\n"
                        "Extract all fields. Follow the extraction rules exactly."
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice_extraction",
                    "strict": True,
                    "schema": _ensure_strict_schema(schema_for_api),
                },
            },
        )
    except Exception as e:
        # Fallback without strict mode if schema isn't fully compatible
        try:
            completion = client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Document ID: {body.document_id}\n\n"
                            f"Invoice text:\n{body.text}\n\n"
                            "Return a single JSON object with EXACTLY these keys:\n"
                            "vendor, currency, total_amount, invoice_date, due_in_days, "
                            "is_paid, priority, contact_email, line_items, item_count.\n"
                            f"JSON Schema for reference:\n{json_schema}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e2:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {e2}") from e2

    raw = completion.choices[0].message.content
    if not raw:
        raise HTTPException(status_code=502, detail="Empty model response")

    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {e}") from e

    # Light post-normalization so grader strict-match is more reliable
    data = _normalize_extracted(data)
    return data


def _ensure_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    OpenAI strict mode requires additionalProperties: false and required listing all properties.
    Best-effort fix if the grader schema is almost ready.
    """
    import copy

    s = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" and "properties" in node:
            props = node["properties"]
            node.setdefault("additionalProperties", False)
            node["required"] = list(props.keys())
            for v in props.values():
                walk(v)
        if node.get("type") == "array" and "items" in node:
            walk(node["items"])
        if "$defs" in node:
            for v in node["$defs"].values():
                walk(v)
        if "definitions" in node:
            for v in node["definitions"].values():
                walk(v)

    walk(s)
    return s


def _normalize_extracted(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce types / casing for exact-match grading."""
    out = dict(data)

    if "contact_email" in out and isinstance(out["contact_email"], str):
        out["contact_email"] = out["contact_email"].strip().lower()

    if "currency" in out and isinstance(out["currency"], str):
        out["currency"] = out["currency"].strip().upper()

    if "priority" in out and isinstance(out["priority"], str):
        out["priority"] = out["priority"].strip().lower()

    for key in ("total_amount", "due_in_days", "item_count"):
        if key in out and out[key] is not None:
            try:
                out[key] = int(out[key])
            except (TypeError, ValueError):
                pass

    if "is_paid" in out:
        v = out["is_paid"]
        if isinstance(v, str):
            out["is_paid"] = v.strip().lower() in {"true", "yes", "1", "paid"}
        else:
            out["is_paid"] = bool(v)

    if "line_items" in out and isinstance(out["line_items"], list):
        cleaned = []
        for item in out["line_items"]:
            if not isinstance(item, dict):
                continue
            cleaned.append(
                {
                    "sku": str(item.get("sku", "")),
                    "quantity": int(item.get("quantity", 0)),
                    "unit_price": int(item.get("unit_price", 0)),
                }
            )
        out["line_items"] = cleaned
        out["item_count"] = len(cleaned)

    # Keep only expected keys, in stable order
    expected = [
        "vendor",
        "currency",
        "total_amount",
        "invoice_date",
        "due_in_days",
        "is_paid",
        "priority",
        "contact_email",
        "line_items",
        "item_count",
    ]
    return {k: out[k] for k in expected if k in out}


# ---------------------------------------------------------------------------
# Endpoint 2: Semantic ranking
# ---------------------------------------------------------------------------

@app.post("/rank")
def rank_passages(body: RankRequest) -> dict[str, list[int]]:
    """
    Embed query + candidates with text-embedding-3-small,
    return indices of the top-3 by cosine similarity (order among top-3 free).
    """
    if not body.candidates:
        raise HTTPException(status_code=400, detail="candidates must be non-empty")

    texts = [body.query, *body.candidates]
    try:
        vectors = embed_texts(texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}") from e

    q = np.array(vectors[0], dtype=np.float64)
    scores: list[tuple[int, float]] = []
    for i, emb in enumerate(vectors[1:]):
        c = np.array(emb, dtype=np.float64)
        scores.append((i, cosine_similarity(q, c)))

    scores.sort(key=lambda t: t[1], reverse=True)
    top3 = [idx for idx, _ in scores[:3]]

    return {"ranking": top3}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))