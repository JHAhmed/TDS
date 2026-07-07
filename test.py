import time
import uuid
import threading
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# --- ASSIGNED VALUES & CONFIGURATION ---
RATE_LIMIT_REQUESTS = 13
RATE_LIMIT_WINDOW = 10

# TODO: 1. Put your actual login email here
YOUR_EMAIL = "22f3003202@ds.study.iitm.ac.in" 

# TODO: 2. Add the origin of the exam page you are currently viewing this from 
# (e.g., "https://tds-2-qona.onrender.com" or "http://localhost:3000")
EXAM_PAGE_ORIGIN = "https://exam.sanand.workers.dev/tds-2026-05-ga2"

ALLOWED_ORIGINS = [
    "https://app-61pz70.example.com",
    EXAM_PAGE_ORIGIN
]

app = FastAPI()

# --- MIDDLEWARE 1: Request Context ---
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Read existing ID or generate a new one
        req_id = request.headers.get("X-Request-ID")
        if not req_id:
            req_id = str(uuid.uuid4())
            
        # Store in request state for the endpoint to use
        request.state.request_id = req_id
        
        # Process the request
        response = await call_next(request)
        
        # Append the ID to the response headers
        response.headers["X-Request-ID"] = req_id
        return response


# --- MIDDLEWARE 3: Rate Limiting ---
# (Numbered as 3 to match your prompt, but placed here for execution ordering)
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.client_requests = defaultdict(deque)
        self.lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        # Bypass rate limiting for CORS preflight OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)
            
        client_id = request.headers.get("X-Client-Id", "anonymous")
        now = time.time()
        
        with self.lock:
            q = self.client_requests[client_id]
            
            # Remove timestamps older than the 10-second window
            while q and q[0] <= now - RATE_LIMIT_WINDOW:
                q.popleft()
                
            # Check bucket limit
            if len(q) >= RATE_LIMIT_REQUESTS:
                return JSONResponse(
                    status_code=429, 
                    content={"detail": "Too Many Requests"}
                )
                
            # Log the request
            q.append(now)
            
        return await call_next(request)


# --- MIDDLEWARE COMPOSITION (ORDER MATTERS) ---
# Middlewares wrap the application from the bottom up. 
# The last one added is the outermost layer.

# 1. Innermost layer (runs right before the route)
app.add_middleware(RequestContextMiddleware)

# 2. Middle layer (checks rate limits)
app.add_middleware(RateLimitMiddleware)

# 3. Outermost layer (Middleware 2 - CORS)
# Placing this outermost guarantees OPTIONS requests are answered successfully 
# without triggering the rate limiter. No wildcards are used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ENDPOINT ---
@app.get("/ping")
async def ping(request: Request):
    return {
        "email": YOUR_EMAIL,
        "request_id": request.state.request_id
    }