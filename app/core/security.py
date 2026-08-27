import os
import secrets
from fastapi import HTTPException, status, Security, Request
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter


def get_real_client_ip(request: Request) -> str:
    """
    Membaca Client IP asli saat berada di balik Reverse Proxy (Nginx/Cloudflare/Traefik).
    Mengecek header X-Forwarded-For atau X-Real-IP paling kiri, dengan fallback ke client.host.
    """
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # X-Forwarded-For dapat berisi list IP terpisah koma: client, proxy1, proxy2
        client_ip = x_forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip

    x_real_ip = request.headers.get("X-Real-IP")
    if x_real_ip and x_real_ip.strip():
        return x_real_ip.strip()

    return request.client.host if request.client and request.client.host else "127.0.0.1"


# Setup SlowAPI Limiter instance berbasis Real Client IP
limiter = Limiter(key_func=get_real_client_ip)

# Scheme X-API-Key Header
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key_header: str = Security(api_key_header_scheme)) -> str:
    """
    Validasi API Key dari X-API-Key HTTP Request Header.
    Menggunakan secrets.compare_digest (Constant-Time Comparison) untuk mencegah Timing Attack.
    Mengembalikan HTTP 401 Unauthorized jika API Key tidak sesuai atau tidak disertakan.
    """
    expected_api_key = os.getenv("API_KEY")

    if not expected_api_key:
        logger_warn = "Environment variable API_KEY belum dikonfigurasi di server."
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Server API Key is not configured."
        )

    if not api_key_header or not secrets.compare_digest(api_key_header, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )

    return api_key_header
