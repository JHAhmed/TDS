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




class InvoiceExtraction(BaseModel):
    invoice_no: Optional[str] = Field(default=None, description="The invoice number. null if not found.")
    date: Optional[str] = Field(default=None, description="Date of the invoice in strictly YYYY-MM-DD format. null if not found.")
    vendor: Optional[str] = Field(default=None, description="The vendor, seller, or company name. null if not found.")
    amount: Optional[float] = Field(default=None, description="The subtotal amount BEFORE tax. null if not found.")
    tax: Optional[float] = Field(default=None, description="The tax or VAT amount. null if not found.")
    currency: Optional[str] = Field(default=None, description="The currency of the invoice (e.g., INR, USD). null if not found.")

# 2. Input Schema (API Request)
class InvoiceRequest(BaseModel):
    invoice_text: str

@app.post("/extract", response_model=InvoiceExtraction)
async def extract_invoice(request: InvoiceRequest):
    """
    Extracts structured fields from raw invoice text using OpenAI Structured Outputs.
    """
    system_prompt = (
        "You are an expert financial data extraction assistant. "
        "Extract the exact fields requested from the provided invoice text. "
        "Strict rules:\n"
        "1. 'date' must be strictly in ISO format (YYYY-MM-DD).\n"
        "2. 'amount' is the subtotal BEFORE tax.\n"
        "3. 'tax' is the tax amount only.\n"
        "4. If a field cannot be found, explicitly return null."
    )
    
    # Use OpenAI's Beta Parse endpoint for guaranteed Structured Outputs
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # Fast, cost-effective, and highly capable for text extraction
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.invoice_text}
        ],
        response_format=InvoiceExtraction,
        temperature=0.0  # Lowest temperature for deterministic, factual extraction
    )
    
    # The .parsed property directly yields the Pydantic object, 
    # which FastAPI automatically serializes to the exact required JSON schema.
    return completion.choices[0].message.parsed

# --- API Endpoint ---

@app.post("/dynamic-extract")
async def dynamic_extract_data(payload: DynamicExtractRequest):
    try:
        # 1. Build the exact Pydantic model required for this specific request
        DynamicModel = build_dynamic_model(payload.schema_definition)
        
        # 2. Call OpenAI enforcing the dynamic schema
        completion = client.beta.chat.completions.parse(
            model="gpt-5.4-mini",
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