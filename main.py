# main.py
import os
import time
import uuid
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import jwt

APP_EMAIL = os.getenv("APP_EMAIL", "22f3003202@ds.study.iitm.ac.in")
ALLOWED_ORIGIN = "https://dash-wyqi1o.example.com"
# Constants provided by the grader
ISSUER = "https://idp.exam.local"
AUDIENCE = "tds-ivgy8zrw.apps.exam.local"

# The RS256 Public Key
PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2okOHspNjgA+2rTLbeuY
cxiP/hG8C6Sb9iwg3yiLAA4HCnpITcbWCSelbvbYGuc3EbNy4xFyf5Cbj5DHJMID
EkryOgyd2giIIIBOUBj8S63uGcnRpOBh9NFatfNwheKuzsPuVNldu6A9cNteNpXc
WyJjG2axVfmq7i6SuKr1JoWYG7xTTAvKPujSl4OtsQfO3h5NepzdfXpr28oNnzfW
ed+zclR6BcmNNo/WVfJ4xyCLSf0BCOgdTgW6PdaChd1l9VDetJZVEgC5tkyvXsfI
SI6iyrYbKR0NEBSqq4XkadEjsCs4F1RncsS4LlgniT7GlkL9Mce3b0wGLs9/7ZIX
dQIDAQAB
-----END PUBLIC KEY-----"""

app = FastAPI()

class TokenRequest(BaseModel):
    token: str


class RequestTimingAndIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        request_id = str(uuid.uuid4())

        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)

        duration = max(time.perf_counter() - start, 0.0)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.6f}"
        return response


class StrictOriginCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        origin = request.headers.get("origin")
        is_allowed = origin == ALLOWED_ORIGIN

        # Handle preflight explicitly.
        if request.method == "OPTIONS" and request.url.path == "/stats":
            if not is_allowed:
                return Response(status_code=403)

            response = Response(status_code=204)
            response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = request.headers.get(
                "access-control-request-headers", "*"
            )
            response.headers["Access-Control-Max-Age"] = "86400"
            response.headers["Vary"] = "Origin"
            return response

        response = await call_next(request)

        # Add ACAO only for the exact allowed origin.
        if is_allowed:
            response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
            response.headers["Vary"] = "Origin"

        return response


app.add_middleware(RequestTimingAndIDMiddleware)
app.add_middleware(StrictOriginCORSMiddleware)


def parse_values(values: str) -> list[int]:
    if values is None or values.strip() == "":
        raise HTTPException(status_code=422, detail="Query parameter 'values' is required")

    parts = [p.strip() for p in values.split(",")]
    if any(p == "" for p in parts):
        raise HTTPException(status_code=422, detail="Invalid comma-separated integers")

    try:
        return [int(p) for p in parts]
    except ValueError:
        raise HTTPException(status_code=422, detail="All values must be integers")


@app.get("/")
def read_root():
    return {"message": "This is main.py!"}


@app.get("/stats")
async def stats(values: str = Query(..., description="Comma-separated integers")):
    nums = parse_values(values)
    count = len(nums)
    total = sum(nums)
    minimum = min(nums)
    maximum = max(nums)
    mean = total / count

    return {
        "email": APP_EMAIL,
        "count": count,
        "sum": total,
        "min": minimum,
        "max": maximum,
        "mean": mean,
    }


@app.options("/stats")
async def stats_preflight():
    # Preflight is handled by middleware; this exists only to make the route explicit.
    return Response(status_code=204)


@app.post("/verify")
def verify_token(request: TokenRequest):
    try:
        payload = jwt.decode(
            request.token,
            key=PUBLIC_KEY,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
            leeway=60  # <-- ADD THIS: Allows for 60 seconds of clock skew between servers
        )
        
        return {
            "valid": True,
            "email": payload.get("email"),
            "sub": payload.get("sub"),
            "aud": payload.get("aud")
        }
        
    except jwt.InvalidTokenError as e:
        # Optional: Print the actual PyJWT error to your terminal for debugging
        print(f"Token validation failed: {e}") 
        
        return JSONResponse(
            status_code=401,
            content={"valid": False}
        )
    try:
        # PyJWT's decode function automatically verifies the signature, 
        # expiration (exp), audience (aud), and issuer (iss) when provided.
        payload = jwt.decode(
            request.token,
            key=PUBLIC_KEY,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER
        )
        
        # If decode succeeds, the token is perfectly valid
        return {
            "valid": True,
            "email": payload.get("email"),
            "sub": payload.get("sub"),
            "aud": payload.get("aud")
        }
        
    except jwt.InvalidTokenError:
        # This catches PyJWT's ExpiredSignatureError, InvalidAudienceError, 
        # InvalidIssuerError, and basic DecodeError (tampering).
        # We use JSONResponse directly to avoid FastAPI's default {"detail": ...} wrapper
        return JSONResponse(
            status_code=401,
            content={"valid": False}
        )