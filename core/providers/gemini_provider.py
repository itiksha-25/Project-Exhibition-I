
import json
import os
import urllib.error
import urllib.request


def _extract_text(message):
    """Extract text from common chat message formats."""
    if isinstance(message, str):
        return message

    if isinstance(message, dict):
        for key in ("content", "text"):
            value = message.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = []
                for part in value:
                    if isinstance(part, dict):
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                return "\n".join(parts)

    return str(message)


def _convert_messages(messages):
    """Convert existing application messages into Gemini contents."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    contents = []

    for message in messages:
        text = _extract_text(message).strip()
        if not text:
            continue

        role = "user"
        if isinstance(message, dict):
            if message.get("role") in ("assistant", "model"):
                role = "model"

        contents.append({
            "role": role,
            "parts": [{"text": text}]
        })

    if not contents:
        raise ValueError("No usable message content was provided.")

    return contents


def generate_response(messages, model=None):
    """Generate a response using Google's Gemini REST API."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )

    payload = {
        "contents": _convert_messages(messages),
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        # Do not expose request credentials or raw upstream details.
        raise RuntimeError(
            f"Gemini API request failed with HTTP {exc.code}."
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError("Could not connect to Gemini API.") from exc

    candidates = result.get("candidates", [])

    if not candidates:
        raise RuntimeError("Gemini returned no response candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "\n".join(
        part["text"] for part in parts if isinstance(part.get("text"), str)
    ).strip()

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    return answer
