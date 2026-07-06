from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import jwt

app = FastAPI()

# Constants provided by the grader
ISSUER = "https://idp.exam.local"
AUDIENCE = "tds-ivgy8zrw.apps.exam.local"

# The RS256 Public Key
PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2okOHspNjgA+2rTLbeuY
cxiP/hG8C6Sb9iwg3yiLAA4HCnpITcbWCSelbvbYGuc3EbNy4xFyf5Cbj5DHJMID
EkryOgyd2giIIIBOUBj8S63uGcnRpOBh9NFatfNwheKuzsPuVNldu6A9cNteNpXc
WyJjG2axVfmq7i6SuKr1JoWYG7xTTAvKPujSl4OtsQfO3h5NepzdfXpr28oNnzfW
ed+zclR6BcmNNo/WVfJ4xyCLSf0BCOgdTgW6PdaChd1l9VDetJZVEgC5tkyvXsfI
SI6iyrYbKR0NEBSqq4XkadEjsCs4F1RncsS4LlgniT7GlkL9Mce3b0wGLs9/7ZIX
dQIDAQAB
-----END PUBLIC KEY-----"""

class TokenRequest(BaseModel):
    token: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Token Verification API!"}


@app.post("/verify")
def verify_token(request: TokenRequest):
    try:
        # PyJWT's decode function automatically verifies the signature, 
        # expiration (exp), audience (aud), and issuer (iss) when provided.
        payload = jwt.decode(
            request.token,
            key=PUBLIC_KEY,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER
        )
        
        # If decode succeeds, the token is perfectly valid
        return {
            "valid": True,
            "email": payload.get("email"),
            "sub": payload.get("sub"),
            "aud": payload.get("aud")
        }
        
    except jwt.InvalidTokenError:
        # This catches PyJWT's ExpiredSignatureError, InvalidAudienceError, 
        # InvalidIssuerError, and basic DecodeError (tampering).
        # We use JSONResponse directly to avoid FastAPI's default {"detail": ...} wrapper
        return JSONResponse(
            status_code=401,
            content={"valid": False}
        )