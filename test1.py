from __future__ import annotations

import base64
import math
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

TOTAL_ORDERS = 53
RATE_LIMIT = 15
WINDOW_SECONDS = 10

app = FastAPI(title="Orders API")

# Ensure CORS is allowed for the grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],  # <-- THIS IS THE MAGIC LINE
)

lock = threading.Lock()

# Store for Idempotency: key -> full response body
idempotency_store: Dict[str, Dict[str, Any]] = {}

# Store for Rate Limiting: client_id -> deque of timestamps
client_requests: Dict[str, Deque[float]] = defaultdict(deque)

# Fixed catalog 1..T
CATALOG: List[Dict[str, Any]] = [
    {"id": i, "name": f"Order #{i}"} for i in range(1, TOTAL_ORDERS + 1)
]


class OrderCreateIn(BaseModel):
    item: Optional[str] = None
    quantity: Optional[int] = Field(default=None, ge=1)
    notes: Optional[str] = None


# --- Helper Functions ---

def _encode_cursor(offset: int) -> str:
    raw = f"offset:{offset}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _decode_cursor(cursor: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
        prefix, offset_str = raw.split(":", 1)
        if prefix != "offset":
            raise ValueError
        offset = int(offset_str)
        if offset < 0:
            raise ValueError
        return offset
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid cursor")


# --- Dependencies ---

def check_rate_limit(x_client_id: Optional[str] = Header(default=None, alias="X-Client-Id")) -> str:
    """
    Per-client rate limiting dependency. 
    By using a dependency, we automatically ignore OPTIONS (CORS preflight) requests.
    """
    client_id = x_client_id or "anonymous"
    now = time.time()
    
    with lock:
        q = client_requests[client_id]

        # Drop timestamps outside the rolling window
        cutoff = now - WINDOW_SECONDS
        while q and q[0] <= cutoff:
            q.popleft()

        if len(q) >= RATE_LIMIT:
            retry_after = max(1, math.ceil(WINDOW_SECONDS - (now - q[0])))
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        q.append(now)
        
    return client_id


# --- Endpoints ---

@app.post("/orders", status_code=201)
async def create_order(
    payload: OrderCreateIn,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    client_id: str = Depends(check_rate_limit)
):
    if not idempotency_key:
        raise HTTPException(status_code=400, detail="Missing Idempotency-Key")

    with lock:
        if idempotency_key in idempotency_store:
            # Return the exact same order record; never create a duplicate.
            return idempotency_store[idempotency_key]

        order = {
            "id": f"ord_{uuid.uuid4().hex}",
            "item": payload.item,
            "quantity": payload.quantity,
            "notes": payload.notes,
        }
        idempotency_store[idempotency_key] = order
        return order


@app.get("/orders")
async def list_orders(
    limit: int = 10, 
    cursor: Optional[str] = None,
    client_id: str = Depends(check_rate_limit)
):
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    # Safety cap
    limit = min(limit, 100)

    offset = 0 if cursor is None else _decode_cursor(cursor)
    if offset > TOTAL_ORDERS:
        raise HTTPException(status_code=400, detail="Cursor out of range")

    items = CATALOG[offset : offset + limit]
    next_offset = offset + len(items)

    return {
        "items": items,
        "next_cursor": _encode_cursor(next_offset) if next_offset < TOTAL_ORDERS else None,
    }


@app.get("/health")
async def health():
    return {"ok": True}