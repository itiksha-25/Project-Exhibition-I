
import os

from core.providers.openai_compatible import (
    generate_response as generate_compatible_response
)


def generate_response(messages, model=None):
    model = model or os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free"
    )

    return generate_compatible_response(
        messages=messages,
        model=model,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        provider_name="OpenRouter",
        site_url=os.getenv(
            "OPENROUTER_SITE_URL",
            "https://intelligent-slm-routing-engine.onrender.com/"
        ),
        app_name="Adaptive SLM Routing Engine"
    )
