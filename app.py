from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import os
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import openai

import time
import uuid
from datetime import datetime, timezone
from collections import deque
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Instrumented API")

# --- Globals & State ---
# Track startup time for /healthz uptime calculation
START_TIME = time.time()

# Prometheus counter for all HTTP requests
REQUEST_COUNTER = Counter(
    "http_requests_total", 
    "Total number of HTTP requests to any endpoint"
)

# In-memory buffer for logs, kept to a max size to prevent memory leaks
LOG_BUFFER = deque(maxlen=1000)

app = FastAPI()

# Enable CORS to allow the browser to verify the endpoint directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Assigned constants
API_KEY = "ak_wxkxdqp9va50jgypae8qir3q"
EMAIL = "22f3003202@ds.study.iitm.ac.in"

# Pydantic schemas for request validation
class Event(BaseModel):
    user: str
    amount: float
    ts: int

class AnalyticsRequest(BaseModel):
    events: List[Event]

client = openai.OpenAI()

# --- Request & Response Schemas ---

class ExtractRequest(BaseModel):
    text: str

class InvoiceResponse(BaseModel):
    vendor: str = Field(description="The vendor name extracted from the text.")
    amount: float = Field(description="The total due amount as a numeric float or integer.")
    currency: str = Field(description="The 3-letter currency code, strictly uppercase (e.g., USD, EUR, GBP).")
    date: str = Field(description="The payment due date structured exactly as YYYY-MM-DD.")

    @validator('currency')
    def validate_currency(cls, v):
        v_clean = v.strip().upper()
        if len(v_clean) != 3 or not v_clean.isalpha():
            raise ValueError("Currency must be a 3-letter uppercase alphabetic code.")
        return v_clean

    @validator('date')
    def validate_date(cls, v):
        # Basic check to ensure it adheres to YYYY-MM-DD pattern length and structure
        v_clean = v.strip()
        parts = v_clean.split('-')
        if len(parts) != 3 or len(parts[0]) != 4 or len(parts[1]) != 2 or len(parts[2]) != 2:
            raise ValueError("Date must be in the exact format YYYY-MM-DD.")
        return v_clean


# --- Middleware ---
@app.middleware("http")
async def instrumentation_middleware(request: Request, call_next):
    # 1. Increment Prometheus counter for every single request
    REQUEST_COUNTER.inc()
    
    # 2. Generate required log fields
    req_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    path = request.url.path
    
    # Structure the JSON log entry
    log_entry = {
        "level": "INFO",
        "ts": ts,
        "path": path,
        "request_id": req_id,
        "method": request.method
    }
    
    # Append to our in-memory deque
    LOG_BUFFER.append(log_entry)
    
    # Process the actual request
    response = await call_next(request)
    return response

@app.get("/")
def read_root():
    return {"message": "This is app.py!"}

@app.post("/analytics")
async def process_analytics(payload: AnalyticsRequest, request: Request):
    # 1. Authorization Check
    # We use request.headers.get() to gracefully catch missing headers 
    # and return a 401 instead of FastAPI's default 422.
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing or wrong API key")

    events = payload.events
    
    # 2. Aggregation Logic
    total_events = len(events)
    unique_users = set()
    revenue = 0.0
    user_revenue = {}

    for event in events:
        # Track unique users
        unique_users.add(event.user)
        
        # Track positive revenue and user-specific revenue
        if event.amount > 0:
            revenue += event.amount
            user_revenue[event.user] = user_revenue.get(event.user, 0.0) + event.amount
    
    # 3. Determine the top user
    top_user = ""
    if user_revenue:
        # Returns the dictionary key (user) with the highest mapped value (revenue)
        top_user = max(user_revenue, key=user_revenue.get)

    # 4. Return structured result
    return {
        "email": EMAIL,
        "total_events": total_events,
        "unique_users": len(unique_users),
        "revenue": revenue,
        "top_user": top_user
    }

@app.get("/work")
def do_work(n: int = 0):
    # Returns the specified format. Using a dummy email as requested.
    return {"email": "student@example.com", "done": n}

@app.get("/metrics")
def get_metrics():
    # Expose Prometheus metrics in the standard text format
    metrics_data = generate_latest()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
def health_check():
    # Calculate non-negative float uptime in seconds
    uptime_s = max(0.0, time.time() - START_TIME)
    return {"status": "ok", "uptime_s": uptime_s}

@app.get("/logs/tail")
def tail_logs(limit: int = 10):
    # Slice the deque to get the last N entries
    # deque doesn't support direct slicing, so we convert to a list
    logs_list = list(LOG_BUFFER)
    
    # If limit is greater than available logs, this safely returns what we have
    return logs_list[-limit:]

@app.post(
    "/extract", 
    response_model=InvoiceResponse, 
    status_code=status.HTTP_200_OK,
    responses={422: {"description": "Malformed input or extraction failure"}}
)
async def extract_invoice(payload: ExtractRequest):
    # Handle empty or purely whitespace garbage inputs immediately
    if not payload.text or not payload.text.strip():
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Malformed or empty input text provided."}
        )

    try:
        # Utilizing OpenAI's native Structured Outputs parser
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Highly cost-efficient and performant for extraction tasks
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an accurate data extraction system. Extract the requested fields "
                        "from the invoice text. Ensure the amount is numeric, currency is a 3-letter uppercase string, "
                        "and date is formatted exactly as YYYY-MM-DD. If fields are completely ambiguous or missing, "
                        "do not hallucinate wild data; let the parsing layer handle validation exceptions safely."
                    )
                },
                {"role": "user", "content": payload.text}
            ],
            response_format=InvoiceResponse,
            temperature=0.0, # Enforce deterministic extraction
        )

        extracted_data = completion.choices[0].message.parsed
        
        # Check if the model explicitly refused or failed to output structured data
        if completion.choices[0].message.refusal or not extracted_data:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": "Model refused or failed validation for this text layout."}
            )

        return extracted_data

    except Exception as e:
        # Intercept any parsing or API connectivity exceptions to satisfy the requirement
        # that malformed text/garbage input must never throw an unhandled HTTP 500 error.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": f"Unprocessable invoice data or parsing error: {str(e)}"}
        )