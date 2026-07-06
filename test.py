import base64
import time
from typing import Optional, Dict, List

from fastapi import FastAPI, Header, HTTPException, Query, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS as requested by the grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Assigned Values ---
TOTAL_ORDERS = 53
RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW = 10  # seconds

# --- In-Memory Data Stores ---
# 1. Fixed catalog for GET /orders (IDs 1 to 53)
orders_catalog = [{"id": i, "description": f"Order {i}"} for i in range(1, TOTAL_ORDERS + 1)]

# 2. Store for Idempotency: Maps idempotency keys to order objects
idempotency_store: Dict[str, dict] = {}

# 3. Store for Rate Limiting: Maps client IDs to a list of request timestamps
rate_limit_store: Dict[str, List[float]] = {}


# --- Models ---
class OrderCreate(BaseModel):
    item: str
    quantity: int = 1


# --- Dependencies ---
def rate_limiter(x_client_id: str = Header(...)):
    """
    Per-client rate limiting dependency.
    Buckets requests by X-Client-Id. Allows R requests per 10 seconds.
    """
    now = time.time()
    
    # Initialize client bucket if it doesn't exist
    if x_client_id not in rate_limit_store:
        rate_limit_store[x_client_id] = []
        
    # Filter out timestamps older than the 10-second window
    client_timestamps = [ts for ts in rate_limit_store[x_client_id] if now - ts < RATE_LIMIT_WINDOW]
    rate_limit_store[x_client_id] = client_timestamps
    
    # Check if the client has exceeded the limit
    if len(client_timestamps) >= RATE_LIMIT_REQUESTS:
        oldest_request = client_timestamps[0]
        retry_after = max(1, int(RATE_LIMIT_WINDOW - (now - oldest_request)))
        
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
            headers={"Retry-After": str(retry_after)}
        )
        
    # Log the new request timestamp
    rate_limit_store[x_client_id].append(now)
    return x_client_id


# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "This is test.py!"}

@app.post("/orders", status_code=201)
def create_order(
    order: OrderCreate,
    response: Response,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    client_id: str = Depends(rate_limiter)
):
    """
    1. Idempotent Order Creation
    """
    # Check if we have already processed this key
    if idempotency_key in idempotency_store:
        # Return the exact same response for repeated requests
        response.status_code = 201 
        return idempotency_store[idempotency_key]
    
    # Generate a new mock order id (using 100+ to separate from the fixed catalog 1..53)
    new_order_id = 100 + len(idempotency_store)
    
    new_order = {
        "id": new_order_id,
        "item": order.item,
        "quantity": order.quantity,
        "status": "created"
    }
    
    # Save to idempotency store
    idempotency_store[idempotency_key] = new_order
    
    return new_order


@app.get("/orders")
def get_orders(
    limit: int = Query(10, gt=0, le=100),
    cursor: Optional[str] = None,
    client_id: str = Depends(rate_limiter)
):
    """
    2. Cursor Pagination
    """
    start_id = 1
    
    # Decode the opaque cursor if provided
    if cursor:
        try:
            # The cursor is a base64 encoded string of the next starting ID
            decoded_cursor = base64.b64decode(cursor.encode()).decode()
            start_id = int(decoded_cursor)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor format.")

    # Find the starting index in our fixed catalog
    remaining_orders = [order for order in orders_catalog if order["id"] >= start_id]
    
    # Slice exactly up to the limit
    page_items = remaining_orders[:limit]
    
    # Generate the next cursor if there are more items left
    next_cursor = None
    if len(remaining_orders) > limit:
        next_id = remaining_orders[limit]["id"]
        # Encode the ID to keep the cursor opaque
        next_cursor = base64.b64encode(str(next_id).encode()).decode()
        
    return {
        "items": page_items,
        "next_cursor": next_cursor
    }