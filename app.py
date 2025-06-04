import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import jwt
from jwt import PyJWKClient

load_dotenv()

app = FastAPI(
    title="Pattern Paradise Embedder API",
    description="Text embeddings secured via Keycloak (OAuth2 + Password Grant).",
    version="1.0.0"
)

KEYCLOAK_ISSUER = os.getenv("KEYCLOAK_ISSUER_URL")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")
KEYCLOAK_URL = os.getenv("KEYCLOAK_CALLABLE_URL")
JWKS_URL = f"{KEYCLOAK_URL}/protocol/openid-connect/certs"
TOKEN_URL = f"{KEYCLOAK_URL}/protocol/openid-connect/token"


oauth2_scheme = OAuth2PasswordBearer(tokenUrl=TOKEN_URL)
jwk_client = PyJWKClient(JWKS_URL)
model = SentenceTransformer("all-MiniLM-L6-v2")


class EmbedRequest(BaseModel):
    texts: List[str]


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=KEYCLOAK_CLIENT_ID,
            issuer=KEYCLOAK_ISSUER,
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=403, detail=f"Invalid token: {str(e)}")


@app.get("/embed", tags=["Embedding"])
def embed_get(text: str, user=Depends(get_current_user)):
    vec = model.encode(text).tolist()
    return {"embedding": vec}


@app.post("/embed", tags=["Embedding"])
def embed_post(data: EmbedRequest, user=Depends(get_current_user)):
    vectors = model.encode(data.texts).tolist()
    return {"embeddings": vectors}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2Password": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": TOKEN_URL,
                    "scopes": { "openid": "OpenID Connect scope for user identity" }
                }
            }
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"OAuth2Password": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
