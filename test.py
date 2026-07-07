# main.py
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Assigned values
ALLOWED_CORS_ORIGIN = "https://app-61pz70.example.com"
RATE_LIMIT_BUCKET = 13
RATE_LIMIT_WINDOW_SECONDS = 10

# Set this to the exam page origin used by the verifier.
# Example:
# EXAM_PAGE_ORIGIN="https://your-exam-page.example.com"
EXAM_PAGE_ORIGIN = os.getenv("https://exam.sanand.workers.dev/", "")

# Set your logged-in address here or via env var.
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "22f3003202@ds.study.iitm.ac.in")

app = FastAPI(title="Ping Service")

allowed_origins = [ALLOWED_CORS_ORIGIN]
if EXAM_PAGE_ORIGIN and EXAM_PAGE_ORIGIN != ALLOWED_CORS_ORIGIN:
    allowed_origins.append(EXAM_PAGE_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["X-Client-Id", "X-Request-ID", "Content-Type"],
    expose_headers=["X-Request-ID"],
)

# Sliding-window rate-limit state
client_hits: Dict[str, Deque[float]] = defaultdict(deque)
rate_limit_lock = asyncio.Lock()


def _retry_after_seconds(now: float, oldest_hit: float) -> int:
    remaining = RATE_LIMIT_WINDOW_SECONDS - (now - oldest_hit)
    return max(1, int(remaining + 0.999999))


async def _check_rate_limit(client_id: str) -> Optional[int]:
    now = time.time()
    async with rate_limit_lock:
        q = client_hits[client_id]

        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while q and q[0] <= cutoff:
            q.popleft()

        if len(q) >= RATE_LIMIT_BUCKET:
            return _retry_after_seconds(now, q[0])

        q.append(now)
        return None


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def per_client_rate_limit_middleware(request: Request, call_next):
    # Do not block CORS preflight.
    if request.method == "OPTIONS":
        return await call_next(request)

    client_id = request.headers.get("X-Client-Id", "anonymous")
    retry_after = await _check_rate_limit(client_id)

    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


@app.get("/ping")
async def ping(request: Request):
    return {
        "email": EMAIL_ADDRESS,
        "request_id": request.state.request_id,
    }


@app.options("/ping")
async def ping_options():
    # CORSMiddleware handles the actual CORS preflight response.
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)