import os
from fastapi import HTTPException, status, Security
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

# Setup SlowAPI Limiter instance for endpoint rate limiting
limiter = Limiter(key_func=get_remote_address)

# Scheme X-API-Key Header
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key_header: str = Security(api_key_header_scheme)) -> str:
    """
    Validasi API Key dari X-API-Key HTTP Request Header.
    Nilai rahasia dicocokkan dengan environment variable API_KEY dari .env.
    Mengembalikan HTTP 401 Unauthorized jika API Key tidak sesuai atau tidak disertakan.
    """
    expected_api_key = os.getenv("API_KEY")

    if not api_key_header or api_key_header != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key_header
