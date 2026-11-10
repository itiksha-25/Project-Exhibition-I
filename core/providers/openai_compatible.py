
import json
import os
import time
import urllib.error
import urllib.request


def _normalise_messages(messages):
    """Convert application messages into OpenAI-compatible chat messages."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if not isinstance(messages, list):
        raise ValueError("Messages must be a string or a list.")

    normalised = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"

        content = message.get("content", message.get("text", ""))

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            content = "\n".join(parts)

        if not isinstance(content, str) or not content.strip():
            continue

        normalised.append({
            "role": role,
            "content": content.strip()
        })

    if not normalised:
        raise ValueError("No usable message content was provided.")

    return normalised


def generate_response(
    messages,
    model,
    api_key_env,
    base_url,
    provider_name,
    site_url=None,
    app_name=None
):
    """Call an OpenAI-compatible hosted inference API."""

    api_key = os.getenv(api_key_env)

    if not api_key:
        raise RuntimeError(f"{api_key_env} is not configured.")

    if not model:
        raise RuntimeError(f"No model configured for {provider_name}.")

    payload = {
        "model": model,
        "messages": _normalise_messages(messages),
        "max_tokens": 2048
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if site_url:
        headers["HTTP-Referer"] = site_url

    if app_name:
        headers["X-Title"] = app_name

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    retryable_statuses = (408, 429, 500, 502, 503, 504)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))

            choices = result.get("choices", [])

            if not choices:
                raise RuntimeError(
                    f"{provider_name} returned no response choices."
                )

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                )

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(
                    f"{provider_name} returned an empty response."
                )

            return content.strip()

        except urllib.error.HTTPError as exc:
            if exc.code not in retryable_statuses:
                raise RuntimeError(
                    f"{provider_name} request failed with HTTP {exc.code}."
                ) from exc

            if attempt == 2:
                raise RuntimeError(
                    f"{provider_name} is temporarily unavailable "
                    f"(HTTP {exc.code})."
                ) from exc

            time.sleep(2 ** attempt)

        except urllib.error.URLError as exc:
            if attempt == 2:
                raise RuntimeError(
                    f"Could not connect to {provider_name}."
                ) from exc

            time.sleep(2 ** attempt)

    raise RuntimeError(f"{provider_name} did not return a response.")
