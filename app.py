# Q2

import os
import re
import json
import base64
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel



app = FastAPI(title="Multimodal QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class AnswerImageRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class AnswerImageResponse(BaseModel):
    answer: str


def _infer_mime(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _to_data_url(image_base64: str) -> str:
    s = image_base64.strip()
    if s.startswith("data:"):
        return s

    s = re.sub(r"\s+", "", s)
    try:
        raw = base64.b64decode(s, validate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    mime = _infer_mime(raw)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"


def _extract_answer(raw_text: str) -> str:
    text = raw_text.strip()

    # Remove common wrappers
    text = text.strip("`").strip()

    # Try to parse JSON first
    def try_parse_json(candidate: str) -> Dict[str, Any] | None:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "answer" in obj:
                return obj
        except Exception:
            return None
        return None

    obj = try_parse_json(text)
    if obj is None:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            obj = try_parse_json(m.group(0))

    if obj is not None:
        ans = str(obj.get("answer", "")).strip()
        return ans

    # Fallback: use first non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        first = re.sub(r"^(answer\s*[:\-]\s*)", "", first, flags=re.I).strip()
        return first.strip('"\'')

    return text.strip('"\'')




class ExtractionRequest(BaseModel):
    invoice_text: str

class InvoiceExtraction(BaseModel):
    """
    Strict schema enforced by OpenAI's Structured Outputs.
    All fields are Optional to allow 'null' if the data cannot be found.
    """
    invoice_no: Optional[str] = Field(None, description="The invoice or bill number.")
    date: Optional[str] = Field(None, description="The invoice date, STRICTLY formatted as ISO 8601 YYYY-MM-DD.")
    vendor: Optional[str] = Field(None, description="The name of the seller or vendor.")
    amount: Optional[float] = Field(None, description="The subtotal amount before taxes. Do not include currency symbols.")
    tax: Optional[float] = Field(None, description="The tax or VAT amount only. Do not include currency symbols.")
    currency: Optional[str] = Field(None, description="The 3-letter currency code (e.g., INR, USD).")



@app.post("/extract", response_model=InvoiceExtraction)
async def extract_invoice_data(payload: ExtractionRequest):
    try:
        # Call OpenAI with strict schema enforcement
        completion = client.beta.chat.completions.parse(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise data extraction API. Extract the requested fields from the invoice text. "
                        "If a field cannot be found, return null. "
                        "The 'date' MUST be in YYYY-MM-DD format. "
                        "'amount' is the subtotal before tax. 'tax' is the tax amount only."
                    )
                },
                {
                    "role": "user",
                    "content": payload.invoice_text
                }
            ],
            response_format=InvoiceExtraction,
        )
        
        result = completion.choices[0].message.parsed
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to parse model response.")

        # FastAPI will automatically serialize this Pydantic model to the exact required JSON
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))









@app.get("/")
def get_root():
    return {"message": "this is app.py"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/answer-image", response_model=AnswerImageResponse)
async def answer_image(payload: AnswerImageRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    image_url = _to_data_url(payload.image_base64)

    prompt = (
        "Answer the user's question using only the information visible in the image.\n"
        "Return ONLY a JSON object with exactly one key: answer.\n"
        "The value must be a string.\n"
        "For numeric answers, return only the number, with no currency symbols, commas, or units.\n"
        "Do not include any extra text."
    )

    try:
        resp = await client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt + "\n\nQuestion: " + payload.question},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            temperature=0,
            max_output_tokens=100,
        )

        raw = getattr(resp, "output_text", "") or ""
        answer = _extract_answer(raw)

        if not answer:
            raise HTTPException(status_code=500, detail="Model returned an empty answer")

        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")