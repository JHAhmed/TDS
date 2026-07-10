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

class ExtractRequest(BaseModel):
    invoice_text: str


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def parse_date(text: str) -> Optional[str]:
    patterns = [
        r"\b(\d{4})-(\d{2})-(\d{2})\b",
        r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b",
        r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",
    ]

    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    m = re.search(patterns[0], text)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(patterns[1], text)
    if m:
        d, mo, y = m.groups()
        try:
            return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(patterns[2], text, flags=re.I)
    if m:
        d, month_name, y = m.groups()
        month = month_map.get(month_name.lower())
        if month:
            try:
                return datetime(int(y), month, int(d)).strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def parse_money(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def find_first(patterns, text, flags=re.I):
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return clean_text(m.group(1))
    return None


@app.post("/extract")
def extract_invoice(payload: ExtractRequest):
    text = payload.invoice_text or ""

    invoice_no = find_first(
        [
            r"Invoice\s*No[:\s]*([A-Za-z0-9\-\/]+)",
            r"Invoice\s*Number[:\s]*([A-Za-z0-9\-\/]+)",
            r"Inv(?:oice)?\s*#[:\s]*([A-Za-z0-9\-\/]+)",
        ],
        text,
    )

    vendor = find_first(
        [
            r"Vendor[:\s]*(.+)",
            r"Supplier[:\s]*(.+)",
            r"Billed\s*From[:\s]*(.+)",
            r"From[:\s]*(.+)",
        ],
        text,
    )
    if vendor:
        vendor = vendor.split("\n")[0].strip()

    date = parse_date(text)

    # Subtotal / amount before tax
    amount_raw = find_first(
        [
            r"Subtotal[:\s]*([A-Za-z₹Rs\.\s0-9,]+)",
            r"Sub\s*Total[:\s]*([A-Za-z₹Rs\.\s0-9,]+)",
            r"Amount\s*Before\s*Tax[:\s]*([A-Za-z₹Rs\.\s0-9,]+)",
            r"Net\s*Amount[:\s]*([A-Za-z₹Rs\.\s0-9,]+)",
        ],
        text,
    )
    amount = parse_money(amount_raw)

    # Tax / GST
    tax_raw = find_first(
        [
            r"GST(?:\s*\([^)]*\))?[:\s]*([A-Za-z₹Rs\.\s0-9,]+)",
            r"Tax[:\s]*([A-Za-z₹Rs\.\s0-9,]+)",
            r"VAT[:\s]*([A-Za-z₹Rs\.\s0-9,]+)",
        ],
        text,
    )
    tax = parse_money(tax_raw)

    return {
        "invoice_no": invoice_no,
        "date": date,
        "vendor": vendor,
        "amount": amount,
        "tax": tax,
        "currency": "INR",
    }


@app.get("/")
def get_root():
    return {"message": "this is main.py"}

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