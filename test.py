# Q3, Q4

import os
from typing import Optional, List, Any, Dict, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(title="Dynamic Invoice & Data Extractor API")

# Enable CORS (Required by grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client (automatically reads OPENAI_API_KEY from environment)
client = OpenAI()

# --- Request Schemas ---

class DynamicExtractRequest(BaseModel):
    text: str
    # Use an alias so we don't conflict with Pydantic's internal 'schema' namespace
    schema_definition: Dict[str, str] = Field(..., alias="schema") 

    class Config:
        populate_by_name = True


# --- Dynamic Model Builder Helper ---

def build_dynamic_model(schema_dict: Dict[str, str], model_name: str = "DynamicExtraction") -> Any:
    """
    Reads the simplified schema dictionary provided in the request
    and dynamically builds a Pydantic model for OpenAI to enforce.
    """
    fields = {}
    
    for key, field_type in schema_dict.items():
        # Map requested types to Python/Pydantic types
        # Everything is wrapped in Optional to allow 'null' if missing from text
        if field_type == "string":
            fields[key] = (Optional[str], Field(default=None))
        elif field_type == "integer":
            fields[key] = (Optional[int], Field(default=None))
        elif field_type == "float":
            fields[key] = (Optional[float], Field(default=None))
        elif field_type == "boolean":
            fields[key] = (Optional[bool], Field(default=None))
        elif field_type == "date":
            # Add explicit instructions for the date format directly into the dynamic field description
            fields[key] = (Optional[str], Field(default=None, description="STRICTLY ISO format YYYY-MM-DD"))
        elif field_type == "array[string]":
            fields[key] = (Optional[List[str]], Field(default=None))
        elif field_type == "array[integer]":
            fields[key] = (Optional[List[int]], Field(default=None))
        else:
            # Fallback to string if an unknown type is passed
            fields[key] = (Optional[str], Field(default=None))
            
    # Dynamically create and return the Pydantic class
    return create_model(model_name, **fields)


class LineItem(BaseModel):
    sku: str = Field(..., description="Stock Keeping Unit identifier code exactly as written.")
    quantity: int = Field(..., description="The quantity of the item as an integer.")
    unit_price: int = Field(..., description="The unit price as an integer (main unit).")

class InvoiceSchema(BaseModel):
    vendor: str = Field(..., description="The biller's proper name, exactly as written.")
    currency: str = Field(..., description="The ISO 4217 code (e.g., USD, EUR, GBP, INR, JPY) matching the text currency indicator.")
    total_amount: int = Field(..., description="Integer representing the total amount in the main unit (no separators, symbols, or decimals). Parse text numbers or abbreviations like 12K into integers.")
    invoice_date: str = Field(..., description="Normalized invoice date in YYYY-MM-DD format.")
    due_in_days: int = Field(..., description="The payment terms converted to an integer number of days (e.g., 'Net 30' -> 30, 'due in two weeks' -> 14).")
    is_paid: bool = Field(..., description="Boolean inference based on text wording (e.g., 'paid in full' -> true, 'awaiting payment' -> false).")
    priority: Literal["low", "normal", "high", "urgent"] = Field(..., description="The implied priority category.")
    contact_email: str = Field(..., description="The lowercased contact email address found in the document.")
    line_items: List[LineItem] = Field(..., description="Array of line items in the order they appear.")
    item_count: int = Field(..., description="The total count of line items present in the array.")


class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema_definition: Dict[str, Any] = Field(..., alias="schema") 
    
    class Config:
        populate_by_name = True


# --- API Endpoint ---

@app.post("/extract")
async def extract_invoice(payload: ExtractRequest):
    try:
        # Request a strongly typed parse from OpenAI
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert data extraction microservice for an ERP system. "
                        "Read the provided free-text document carefully and extract all required properties. "
                        "Follow these rules strictly:\n"
                        "1. currency must be normalized to a standard 3-letter ISO 4217 code.\n"
                        "2. total_amount, quantity, and unit_price must be pure integers.\n"
                        "3. invoice_date must follow YYYY-MM-DD format.\n"
                        "4. due_in_days must be calculated as an absolute integer number of days.\n"
                        "5. contact_email must be converted entirely to lowercase.\n"
                        "6. line_items must match the original order of appearance, and item_count must equal len(line_items)."
                    )
                },
                {
                    "role": "user",
                    "content": payload.text
                }
            ],
            response_format=InvoiceSchema,
        )
        
        extracted_data = completion.choices[0].message.parsed
        
        if not extracted_data:
            raise HTTPException(status_code=500, detail="LLM failed to produce valid structured schema outputs.")
        
        # Return exact JSON fields to the grader without markdown wrappers or extra keys
        return JSONResponse(
            status_code=200,
            content=extracted_data.model_dump()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failure on document {payload.document_id}: {str(e)}"
        )

# --- API Endpoint ---

@app.post("/dynamic-extract")
async def dynamic_extract_data(payload: DynamicExtractRequest):
    try:
        # 1. Build the exact Pydantic model required for this specific request
        DynamicModel = build_dynamic_model(payload.schema_definition)
        
        # 2. Call OpenAI enforcing the dynamic schema
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise data extraction API. Extract the requested fields from the text based on the schema. "
                        "Rules:\n"
                        "- If a field cannot be found in the text, you MUST return null.\n"
                        "- Dates MUST be formatted as YYYY-MM-DD.\n"
                        "- Numbers must be purely JSON numbers, not strings.\n"
                        "- Do not guess or hallucinate information not present in the text."
                    )
                },
                {
                    "role": "user",
                    "content": payload.text
                }
            ],
            response_format=DynamicModel,
        )
        
        result = completion.choices[0].message.parsed
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to parse model response.")

        # 3. Dump the model to a dictionary. 
        # Using standard model_dump() ensures missing fields remain explicitly as 'null' instead of being deleted.
        return JSONResponse(
            status_code=200,
            content=result.model_dump()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)