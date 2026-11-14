"""
Orchestrator Model Switcher & Inference Engine
Handles Ollama model serving, token streaming, TTFT/TPS calculation, and mock fallback.
"""

import time
import json
import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Any, Tuple
import httpx
import ollama

from core.config import OLLAMA_HOST, DOMAINS
from .memory_manager import AdaptiveMemoryManager, LatencyDecomposition


# Domain specialist fallback responses for graceful offline resilience
KNOWLEDGE_RESPONSES = {
    "mathematics": (
        "Let's solve this step by step.\n\n"
        "1. Identify the given expression and objective.\n"
        "2. Applying the analytical rule:\n"
        "   For a composite function f(g(x)), by chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x).\n"
        "3. Simplifying the resulting algebraic terms yields an exact closed-form solution.\n"
        "Result: Verification confirmed with high-precision arithmetic."
    ),
    "medical": (
        "Clinical Summary:\n\n"
        "- Pathophysiology: Characterized by chronic metabolic dysfunction and cellular signaling alteration.\n"
        "- Differential Considerations: Distinguishing Type-1 (autoimmune islet destruction) from Type-2 (peripheral insulin resistance).\n"
        "- First-Line Pharmacotherapy: Metformin (AMPK activator) titrated to renal clearance (eGFR > 30 mL/min).\n"
        "- Monitoring: Glycated hemoglobin (HbA1c target < 7.0%), lipid panel, and renal microalbuminuria screening."
    ),
    "code": (
        "```python\n"
        "# High-performance asynchronous implementation\n"
        "import asyncio\n"
        "import httpx\n\n"
        "async def fetch_endpoint(url: str, timeout: float = 5.0) -> dict:\n"
        "    async with httpx.AsyncClient(timeout=timeout) as client:\n"
        "        response = await client.get(url)\n"
        "        response.raise_for_status()\n"
        "        return response.json()\n\n"
        "async def main():\n"
        "    results = await asyncio.gather(\n"
        "        fetch_endpoint('https://api.example.com/data/1'),\n"
        "        fetch_endpoint('https://api.example.com/data/2')\n"
        "    )\n"
        "    print(f'Processed {len(results)} responses.')\n"
        "```"
    ),
    "science": (
        "Scientific Principles:\n\n"
        "1. Thermodynamics & Entropy: The Second Law states that in any isolated system, total entropy change ΔS ≥ 0.\n"
        "2. Molecular Mechanism: Energy dispersion occurs towards microstates with maximum statistical probability.\n"
        "3. Real-world Implication: Irreversibility dictates the arrow of thermodynamic time."
    ),
    "writing": (
        "In the quiet stillness before the dawn, shadows stretch across the forgotten corridors. "
        "Every whisper carries the cadence of deliberate thought—crafted not merely to inform, but to resonate. "
        "Through balanced rhetoric and vivid prose, the core message emerges with crystalline clarity."
    ),
    "general": (
        "Here is a comprehensive overview:\n\n"
        "Taking into account all critical factors, the best approach involves systematic evaluation: "
        "first isolating the core requirements, next reviewing trade-offs between efficiency and simplicity, "
        "and finally implementing an iterative strategy that guarantees consistent results."
    )
}


class ModelSwitcher:
    """
    Manages active model execution, context injection, and real-time streaming tokens.
    """

    def __init__(self, memory_manager: Optional[AdaptiveMemoryManager] = None):
        self.memory_manager = memory_manager or AdaptiveMemoryManager()
        self.ollama_host = OLLAMA_HOST
        self._available_ollama_models: List[str] = []
        self.refresh_available_models()

    def refresh_available_models(self) -> List[str]:
        """Queries local Ollama instance for pulled models."""
        try:
            client = ollama.Client(host=self.ollama_host)
            models_info = client.list()
            # ollama.list() returns either dict or ModelList
            if hasattr(models_info, "models"):
                self._available_ollama_models = [m.model for m in models_info.models]
            elif isinstance(models_info, dict):
                self._available_ollama_models = [m.get("name") for m in models_info.get("models", [])]
        except Exception:
            self._available_ollama_models = []
        return self._available_ollama_models

    def is_model_available(self, model_name: str) -> bool:
    """Checks whether the exact model or its equivalent tag is pulled locally."""
    if not self._available_ollama_models:
        self.refresh_available_models()

    requested = model_name.strip().lower()

    return any(
        model.strip().lower() == requested
        or model.strip().lower() == f"{requested}:latest"
        or requested == model.strip().lower().removesuffix(":latest")
        for model in self._available_ollama_models
    )

    def resolve_best_model(self, requested_model: str, domain: str) -> str:
    """
    Selects the requested local model if available,
    otherwise uses the domain fallback or another available model.
    """
    if self.is_model_available(requested_model):
        return requested_model

    domain_meta = DOMAINS.get(domain, DOMAINS["general"])
    fallback = domain_meta.get("fallback_model", "llama3.2:3b")

    if self.is_model_available(fallback):
        return fallback

    if self._available_ollama_models:
        return self._available_ollama_models[0]

    raise RuntimeError(
        "No local Ollama models are available. Pull a model or configure cloud inference."
    )

    def generate_chat_response(
    self,
    messages: List[Dict[str, str]],
    target_model: str,
    domain: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Executes non-streaming local Ollama inference and records telemetry.
    TTFT is measured as response latency because this uses non-streaming mode.
    """
    resolved_model = self.resolve_best_model(target_model, domain)

    success, swap_decomp, evicted = self.memory_manager.stage_model(
        resolved_model
    )

    t_start = time.perf_counter()

    try:
        client = ollama.Client(host=self.ollama_host)

        response = client.chat(
            model=resolved_model,
            messages=messages,
            options={"temperature": 0.2},
            keep_alive="10m"
        )

        response_text = response["message"]["content"]

        t_end = time.perf_counter()

        # Non-streaming API: actual first-token timing is unavailable.
        response_latency_ms = round((t_end - t_start) * 1000.0, 2)

        token_count = max(
            1,
            int(len(response_text.split()) * 1.3)
        )

    except Exception as e:
        raise RuntimeError(
            f"Ollama inference failed: {type(e).__name__}: {e}"
        ) from e

    total_inference_time = max(0.001, t_end - t_start)

    tps = round(token_count / total_inference_time, 1)

    telemetry = {
        "model_used": resolved_model,
        "domain": domain,
        "ttft_ms": None,
        "response_latency_ms": response_latency_ms,
        "tps": tps,
        "tokens_generated_estimated": token_count,
        "total_inference_seconds": round(total_inference_time, 3),
        "swap_decomposition": (
            swap_decomp.model_dump()
            if hasattr(swap_decomp, "model_dump")
            else {}
        ),
        "evicted_model": evicted
    }

    return response_text, telemetry

    async def stream_chat_response(
    self,
    messages: List[Dict[str, str]],
    target_model: str,
    domain: str,
    ttft_ms = None 
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streams local Ollama inference with TTFT and throughput telemetry.
    """

    resolved_model = self.resolve_best_model(target_model, domain)

    success, swap_decomp, evicted = self.memory_manager.stage_model(
        resolved_model
    )

    swap_data = (
        swap_decomp.model_dump()
        if hasattr(swap_decomp, "model_dump")
        else {}
    )

    yield {
        "type": "staging",
        "model": resolved_model,
        "swap": swap_data,
        "evicted": evicted
    }

    t_start = time.perf_counter()
    first_token_emitted = False
    chunk_count = 0
    response_parts = []

    try:
        client = ollama.Client(host=self.ollama_host)

        stream = client.chat(
            model=resolved_model,
            messages=messages,
            stream=True,
            options={"temperature": 0.2},
            keep_alive="10m"
        )

        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")

            if token:
                if not first_token_emitted:
                    ttft_ms = round(
                        (time.perf_counter() - t_start) * 1000.0,
                        2
                    )

                    yield {
                        "type": "ttft",
                        "ttft_ms": ttft_ms
                    }

                    first_token_emitted = True

                chunk_count += 1
                response_parts.append(token)

                yield {
                    "type": "token",
                    "content": token
                }

    except Exception as e:
        yield {
            "type": "error",
            "error": f"Ollama inference failed: {type(e).__name__}"
        }
        return

    t_end = time.perf_counter()
    total_time = max(0.001, t_end - t_start)

    if not first_token_emitted:
        yield {
            "type": "error",
            "error": "The model returned an empty response."
        }
        return

    response_text = "".join(response_parts)

    estimated_tokens = max(
        1,
        int(len(response_text.split()) * 1.3)
    )

    tps = round(estimated_tokens / total_time, 1)

    yield {
        "type": "done",
        "model": resolved_model,
        "chunks": chunk_count,
        "tokens_generated_estimated": estimated_tokens,
        "tps": tps,
        "total_seconds": round(total_time, 3)
    }