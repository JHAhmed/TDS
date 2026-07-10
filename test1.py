import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(title="Arithmetic Word Problem Solver Service")

# Initialize OpenAI Client (reads OPENAI_API_KEY from environment)
client = OpenAI()

# --- Request/Response Schemas ---

class ProblemRequest(BaseModel):
    problem_id: str
    problem: str

class SolverResponse(BaseModel):
    """
    Schema enforced directly on the OpenAI API response to guarantee format.
    """
    reasoning: str = Field(
        ..., 
        description="Detailed step-by-step math reasoning showing how you derived the answer. Must be at least 80 characters long."
    )
    answer: int = Field(
        ..., 
        description="The final single integer answer. Do not include currency symbols or decimals."
    )

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning_length(cls, v: str) -> str:
        if len(v) < 80:
            raise ValueError("Reasoning must be at least 80 characters long.")
        return v


# --- API Endpoint ---

@app.post("/solve")
async def solve_problem(payload: ProblemRequest):
    try:
        # Call OpenAI with strict schema enforcement
        completion = client.beta.chat.completions.parse(
            model="gpt-5.4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise mathematical solver service. Your task is to solve the word problem provided. "
                        "Identify and ignore any distractor numbers that are irrelevant to the core question. "
                        "Provide a detailed step-by-step reasoning that is at least 80 characters long, "
                        "and output the final answer as a single integer."
                    )
                },
                {
                    "role": "user",
                    "content": payload.problem
                }
            ],
            response_format=SolverResponse,
        )
        
        # The parsed object is automatically validated against our Pydantic schema
        result = completion.choices[0].message.parsed
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to parse model response.")

        # Return the exact JSON structure required by the grader (no markdown, no extra keys)
        return JSONResponse(
            status_code=200,
            content={
                "reasoning": result.reasoning,
                "answer": result.answer
            }
        )

    except Exception as e:
        # Fallback handling for validation failures or API errors
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing problem {payload.problem_id}: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)