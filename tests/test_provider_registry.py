import pytest

from core.providers import registry


def test_provider_order_uses_only_configured_keys(monkeypatch):
    monkeypatch.setenv(
        "CLOUD_PROVIDER_ORDER",
        "groq,gemini,openrouter"
    )

    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    assert registry.get_provider_order() == [
        "groq",
        "openrouter"
    ]


def test_cloud_provider_fallback(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")

    def failing_provider(messages, model):
        raise RuntimeError("Simulated provider failure")

    def working_provider(messages, model):
        return "Test response"

    test_config = {
        "groq": {
            "key": "GROQ_API_KEY",
            "model_env": "GROQ_MODEL",
            "default_model": "test-groq-model",
            "generate": failing_provider,
        },
        "gemini": {
            "key": "GEMINI_API_KEY",
            "model_env": "GEMINI_MODEL",
            "default_model": "test-gemini-model",
            "generate": working_provider,
        },
    }

    monkeypatch.setattr(
        registry,
        "PROVIDER_CONFIG",
        test_config
    )
    monkeypatch.setenv("CLOUD_PROVIDER_ORDER", "groq,gemini")

    result, failures = registry.generate_cloud_response(
        [{"role": "user", "content": "Hello"}]
    )

    assert result is not None
    assert result["provider"] == "gemini"
    assert result["model"] == "test-gemini-model"
    assert result["response"] == "Test response"
    assert failures[0]["provider"] == "groq"


def test_no_configured_cloud_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result, failures = registry.generate_cloud_response(
        [{"role": "user", "content": "Hello"}]
    )

    assert result is None
    assert failures == []