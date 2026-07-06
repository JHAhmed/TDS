from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

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