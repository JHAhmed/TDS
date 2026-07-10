from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from redis.asyncio import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

app = FastAPI(title="Redis Counter API")
redis_client: Redis | None = None


@app.on_event("startup")
async def startup() -> None:
    global redis_client
    redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    # Fail fast if Redis is unreachable
    await redis_client.ping()


@app.on_event("shutdown")
async def shutdown() -> None:
    if redis_client is not None:
        await redis_client.aclose()


@app.post("/hit/{key}")
async def hit(key: str):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    count = await redis_client.incr(f"counter:{key}")
    return {"key": key, "count": count}


@app.get("/count/{key}")
async def count(key: str):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    value = await redis_client.get(f"counter:{key}")
    return {"key": key, "count": int(value) if value is not None else 0}


@app.get("/healthz")
async def healthz():
    if redis_client is None:
        return {"status": "fail", "redis": "down"}

    try:
        await redis_client.ping()
        return {"status": "ok", "redis": "up"}
    except Exception:
        return {"status": "fail", "redis": "down"}