
import os
import time

from core.providers.gemini_provider import (
    generate_response as generate_gemini
)
from core.providers.groq_provider import (
    generate_response as generate_groq
)
from core.providers.openrouter_provider import (
    generate_response as generate_openrouter
)


PROVIDER_CONFIG = {
    "gemini": {
        "key": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.8-flash",
        "generate": generate_gemini,
    },
    "groq": {
        "key": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-20b",
        "generate": generate_groq,
    },
    "openrouter": {
        "key": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "openrouter/free",
        "generate": generate_openrouter,
    },
}


def get_provider_order():
    """Return configured cloud providers in preferred order."""
    configured_order = os.getenv(
        "CLOUD_PROVIDER_ORDER",
        "gemini,groq,openrouter"
    )

    order = [
        name.strip().lower()
        for name in configured_order.split(",")
        if name.strip()
    ]

    return [
        name for name in order
        if name in PROVIDER_CONFIG
        and os.getenv(PROVIDER_CONFIG[name]["key"])
    ]


def generate_cloud_response(messages):
    """
    Try configured cloud providers in order.

    Returns:
        (result, failures)

    result contains provider, model, response and inference latency.
    If every provider fails or no key is configured, result is None.
    """
    failures = []

    for provider_name in get_provider_order():
        config = PROVIDER_CONFIG[provider_name]
        model = os.getenv(
            config["model_env"],
            config["default_model"]
        )

        started = time.perf_counter()

        try:
            response = config["generate"](messages, model)

            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                2
            )

            return {
                "provider": provider_name,
                "model": model,
                "response": response,
                "inference_latency_ms": latency_ms,
            }, failures

        except Exception as exc:
            print(
                f"[Provider Registry] {provider_name} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            failures.append({
                "provider": provider_name,
                "error": str(exc)[:250],
            })

    return None, failures
