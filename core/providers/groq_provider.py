
import os

from core.providers.openai_compatible import (
    generate_response as generate_compatible_response
)


def generate_response(messages, model=None):
    model = model or os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    return generate_compatible_response(
        messages=messages,
        model=model,
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        provider_name="Groq"
    )
