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

YOUR_EMAIL = "22f3003202@ds.study.iitm.ac.in" 
EXAM_PAGE_ORIGIN = "https://exam.sanand.workers.dev"

ALLOWED_ORIGINS = [
    "https://app-61pz70.example.com",
    EXAM_PAGE_ORIGIN
]

app = FastAPI()

# --- MIDDLEWARES ---

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Ignore OPTIONS preflight requests here
        if request.method == "OPTIONS":
            return await call_next(request)
            
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
# In FastAPI, the LAST middleware added is the OUTERMOST layer.

# 1. Innermost layer (Runs right before the route)
app.add_middleware(RateLimitMiddleware)

# 2. Middle layer (Sets request ID, catches 429s from rate limiter to add headers)
app.add_middleware(RequestContextMiddleware)

# 3. Outermost layer (Middleware 2 - CORS)
# Placing this outermost guarantees OPTIONS requests are answered successfully 
# without triggering the inner middlewares.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]  # <-- CRITICAL for the grader to read the header
)


# --- ENDPOINT ---
@app.get("/ping")
async def ping(request: Request):
    return {
        "email": YOUR_EMAIL,
        "request_id": request.state.request_id
    }