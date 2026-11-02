
"""
FastAPI Application Entrypoint
Public API with basic request protection and configurable CORS.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from core.config import BASE_DIR
from api.routes import router as api_router


STATIC_DIR = os.path.join(BASE_DIR, "static")

# Anonymous public API limits.
RATE_WINDOW_SECONDS = 60
CHAT_REQUESTS_PER_WINDOW = 10
OTHER_API_REQUESTS_PER_WINDOW = 60
MAX_REQUEST_BODY_BYTES = 32 * 1024

# These operations must not be publicly accessible until authentication
# and per-user isolation are implemented.

PUBLIC_DEPLOYMENT = (
    os.getenv("PUBLIC_DEPLOYMENT", "false").lower() == "true"
)

BLOCKED_PUBLIC_PATHS = {
    "/api/memory/add",
    "/api/memory/clear",
    "/api/models/preload",
    "/api/models/evict",
    "/api/mcp/invoke",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Initializing Adaptive Multi-SLM AI Server...")
    os.makedirs(STATIC_DIR, exist_ok=True)
    yield
    print("[Server] Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Intelligent Multi-Model AI: Adaptive SLM Routing",
        description="Adaptive model routing with cloud inference and optional local inference.",
        version="1.0.0",
        lifespan=lifespan
    )

    # Same-origin frontend works without CORS.
    # Add external frontend origins through Render's CORS_ORIGINS variable.
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # In-memory fixed-window rate limiter.
    # This is a basic single-instance protection, not distributed rate limiting.
    request_counts = {}

    @app.middleware("http")
    async def public_api_protection(request: Request, call_next):

        path = request.url.path

        if path.startswith("/api/"):

            # Disable sensitive operations until proper authentication exists.
            if PUBLIC_DEPLOYMENT and path in BLOCKED_PUBLIC_PATHS:
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "This operation is disabled on the public deployment."
                    }
                )

            # Reject oversized requests.
            content_length = request.headers.get("content-length")

            if content_length:
                try:
                    if int(content_length) > MAX_REQUEST_BODY_BYTES:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Request body is too large."}
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header."}
                    )

            # Use the client address resolved by the ASGI server.
            client_ip = (
                request.client.host
                if request.client is not None
                else "unknown"
            )

            now = time.monotonic()
            window = int(now // RATE_WINDOW_SECONDS)

            is_chat = path == "/api/chat"

            limit = (
                CHAT_REQUESTS_PER_WINDOW
                if is_chat
                else OTHER_API_REQUESTS_PER_WINDOW
            )

            route_class = "chat" if is_chat else "api"
            key = (client_ip, route_class, window)

            count = request_counts.get(key, 0)

            if count >= limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded. Please try again later."
                    },
                    headers={"Retry-After": "60"}
                )

            request_counts[key] = count + 1

            # Periodically discard expired counters.
            if len(request_counts) > 5000:
                request_counts.clear()

        return await call_next(request)

    app.include_router(api_router, prefix="/api")

    if os.path.exists(STATIC_DIR):
        app.mount(
            "/static",
            StaticFiles(directory=STATIC_DIR),
            name="static"
        )

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(STATIC_DIR, "index.html")

        if os.path.exists(index_file):
            return FileResponse(index_file)

        return {
            "status": "online",
            "message": "Static index.html not yet initialized"
        }

    return app


app = create_app()
