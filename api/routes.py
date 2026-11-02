"""
REST & Streaming API Routes
Intelligent Multi-Model AI: Adaptive SLM Routing & Context-Aware Inference
"""

import json
import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import DOMAINS, SYSTEM_SPECIFICATIONS
from core.router.intent_classifier import IntentRouter
from core.memory.vector_store import VectorMemoryStore
from core.memory.context_normalizer import UniversalContextEnvelope
from core.memory.decay_manager import MemoryDecayManager
from core.orchestrator.memory_manager import AdaptiveMemoryManager
from core.orchestrator.model_switcher import ModelSwitcher
from core.orchestrator.telemetry import TelemetryTracker
from core.benchmarks.quantization_matrix import get_quantization_comparison
from core.benchmarks.benchmark_suite import BenchmarkSuite
from core.mcp.server import (
    route_and_infer as mcp_route_and_infer,
    classify_intent as mcp_classify_intent,
    query_semantic_memory as mcp_query_memory,
    add_memory_fact as mcp_add_fact,
    get_system_telemetry as mcp_get_telemetry,
    run_quantization_benchmark as mcp_run_benchmark,
    get_system_resource,
    get_domains_resource,
    get_benchmarks_resource
)

router = APIRouter()

# Singletons across API lifecycle
intent_router = IntentRouter()
memory_store = VectorMemoryStore()
decay_manager = MemoryDecayManager()
memory_manager = AdaptiveMemoryManager()
model_switcher = ModelSwitcher(memory_manager=memory_manager)
telemetry_tracker = TelemetryTracker()
benchmark_suite = BenchmarkSuite(model_switcher=model_switcher)


# Request & Response Schemas
class ChatRequest(BaseModel):
    query: str = Field(..., description="User prompt text")
    domain_override: Optional[str] = Field(None, description="Optional manual domain override")
    conversation_id: Optional[str] = Field("default", description="Conversation session ID")
    stream: bool = Field(True, description="Whether to stream response tokens")


class RouteRequest(BaseModel):
    query: str = Field(..., description="Prompt to classify")


class MemoryFactRequest(BaseModel):
    fact: str = Field(..., description="Factual text statement")
    domain: str = Field("general", description="Associated domain")
    importance: float = Field(1.0, description="Salience weight [0.0 - 1.0]")


class ModelActionRequest(BaseModel):
    model_name: str = Field(..., description="Target model string")


class McpInvokeRequest(BaseModel):
    tool_name: str = Field(..., description="MCP tool identifier")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool input arguments")


# ==========================================
# CHAT & INFERENCE ENDPOINTS
# ==========================================

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Main conversational pipeline:
    1. Route query (<25ms)
    2. Retrieve memory (<15ms)
    3. Compile Universal Context Envelope (UCE)
    4. Stage model & execute inference
    5. Stream tokens and emit rich telemetry
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Routing
    t_route_0 = time.perf_counter()
    route_res = intent_router.route(req.query, forced_domain=req.domain_override)
    t_route_ms = route_res.latency_ms

    # 2. Memory Retrieval
    retrieved, mem_latency_ms = memory_store.search(
        query=req.query,
        top_k=2,
        min_similarity=0.12
    )
    injected_facts = [r["text"] for r in retrieved]

    # 3. Create CDU & Compile Context Envelope
    cdu = UniversalContextEnvelope.create_cdu(
        role="user",
        content=req.query,
        source_model="client",
        domain=route_res.domain
    )
    compiled_messages = UniversalContextEnvelope.compile_for_model(
        cdus=[cdu],
        target_model=route_res.target_model,
        target_domain=route_res.domain,
        injected_facts=injected_facts
    )

    if req.stream:
        async def event_generator():
            # Send initial metadata header event
            initial_event = {
                "type": "meta",
                "routing": route_res.model_dump(),
                "memory": {
                    "injected_count": len(retrieved),
                    "retrieved_facts": retrieved,
                    "lookup_latency_ms": mem_latency_ms
                }
            }
            yield f"data: {json.dumps(initial_event)}\n\n"

            accumulated_tokens = []
            ttft_value = 0.0

            async for chunk in model_switcher.stream_chat_response(
                messages=compiled_messages,
                target_model=route_res.target_model,
                domain=route_res.domain
            ):
                if chunk["type"] == "error":
                    yield f"data: {json.dumps(chunk)}\n\n"
                    return

                if chunk["type"] == "token":
                    accumulated_tokens.append(chunk["content"])
                elif chunk["type"] == "ttft":
                    ttft_value = chunk["ttft_ms"]

                yield f"data: {json.dumps(chunk)}\n\n"

            # Auto-save interaction to vector memory
            full_response = "".join(accumulated_tokens)
            memory_store.insert(
                text=f"User: {req.query[:80]} | Response: {full_response[:100]}",
                domain=route_res.domain,
                source_model=route_res.target_model,
                importance=0.75
            )

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    else:
        # Non-streaming fallback
        try:
            response_text, telemetry = model_switcher.generate_chat_response(
                messages=compiled_messages,
                target_model=route_res.target_model,
                domain=route_res.domain
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail="Inference unavailable. No model provider could complete the request."
            ) from exc

        # Record telemetry
        telemetry_tracker.record_inference(
            domain=route_res.domain,
            model=telemetry["model_used"],
            ttft_ms=telemetry["ttft_ms"],
            tps=telemetry["tps"],
            route_latency_ms=t_route_ms,
            mem_latency_ms=mem_latency_ms,
            is_hot=telemetry["swap_decomposition"]["is_hot_switch"],
            swap_total_ms=telemetry["swap_decomposition"]["t_total_ms"]
        )

        # Store to vector memory
        memory_store.insert(
            text=f"User: {req.query[:80]} | Response: {response_text[:100]}",
            domain=route_res.domain,
            source_model=telemetry["model_used"],
            importance=0.75
        )

        return {
            "output": response_text,
            "routing": route_res.model_dump(),
            "telemetry": telemetry,
            "memory": {
                "injected_count": len(retrieved),
                "retrieved_facts": retrieved,
                "lookup_latency_ms": mem_latency_ms
            }
        }


# ==========================================
# ROUTER LAB ENDPOINTS (Om's Research)
# ==========================================

@router.post("/route")
def route_standalone_endpoint(req: RouteRequest):
    """Stand-alone intent classification endpoint (<25ms budget)."""
    return intent_router.route(req.query).model_dump()


@router.get("/router/benchmark")
def router_benchmark_endpoint():
    """Returns classification accuracy, latency distribution, and confusion matrix."""
    return intent_router.get_benchmark_report()


# ==========================================
# SYSTEM & MEMORY ENDPOINTS
# ==========================================

@router.get("/system/stats")
def get_system_stats_endpoint():
    """Returns OS RAM metrics, loaded models, LRU order, and swap history."""
    mem_stats = memory_manager.get_hardware_memory_stats()
    telemetry_summary = telemetry_tracker.get_summary()

    return {
        "memory": mem_stats.model_dump(),
        "active_model": memory_manager.active_model_name,
        "lru_queue": memory_manager.lru_queue,
        "models_catalog": {k: v.model_dump() for k, v in memory_manager.models.items()},
        "telemetry": telemetry_summary
    }


@router.post("/models/preload")
def preload_model_endpoint(req: ModelActionRequest):
    """Pre-stages a model in warm memory cache."""
    success, decomp, evicted = memory_manager.stage_model(req.model_name)
    return {
        "success": success,
        "model": req.model_name,
        "evicted": evicted,
        "swap_decomposition": decomp.model_dump()
    }


@router.post("/models/evict")
def evict_models_endpoint():
    """Forces zero-leakage memory eviction across all models."""
    memory_manager.evict_all()
    return {"status": "evicted_all", "memory": memory_manager.get_hardware_memory_stats().model_dump()}


# ==========================================
# VECTOR & CONTEXT MEMORY
# ==========================================

@router.get("/memory/search")
def search_memory_endpoint(q: str, limit: int = 4, domain: Optional[str] = None):
    """Searches persistent vector store with cosine similarity."""
    results, latency_ms = memory_store.search(q, top_k=limit, domain_filter=domain)
    return {
        "query": q,
        "latency_ms": latency_ms,
        "count": len(results),
        "results": results
    }


@router.post("/memory/add")
def add_memory_endpoint(req: MemoryFactRequest):
    """Stores a fact or rule in long-term vector store."""
    item_id = memory_store.insert(
        text=req.fact,
        domain=req.domain,
        source_model="api_user",
        importance=req.importance
    )
    return {"status": "success", "item_id": item_id}


@router.post("/memory/clear")
def clear_memory_endpoint():
    """Clears all entries in vector storage."""
    memory_store.clear()
    return {"status": "memory_cleared"}


@router.get("/memory/decay-curve")
def memory_decay_curve_endpoint(half_life: float = 3600.0):
    """Returns theoretical memory decay curve points for UI visualization."""
    points = []
    # Project over 0 to 4 half-lives
    steps = [0, 300, 900, 1800, 3600, 7200, 10800, 14400]
    decay_mgr = MemoryDecayManager(half_life_seconds=half_life)
    for s in steps:
        points.append({
            "seconds": s,
            "minutes": round(s / 60, 1),
            "salience": decay_mgr.compute_salience(1.0, float(s))
        })
    return {"half_life_seconds": half_life, "curve": points}


# ==========================================
# BENCHMARK SUITE
# ==========================================

@router.get("/benchmarks/matrix")
def get_benchmarks_matrix_endpoint():
    """Returns the empirical asymmetric quantization trade-off matrix."""
    return get_quantization_comparison()


@router.get("/benchmarks/run")
def run_benchmark_endpoint(domain: str = "mathematics", limit: int = 1):
    """Executes live empirical benchmark evaluation."""
    return benchmark_suite.run_domain_benchmark(domain=domain, sample_limit=limit)


# ==========================================
# MODEL CONTEXT PROTOCOL (MCP) INTERFACE
# ==========================================

@router.post("/mcp/invoke")
def mcp_invoke_endpoint(req: McpInvokeRequest):
    """Allows web frontend or external clients to execute MCP tools directly."""
    tool = req.tool_name.strip()
    args = req.arguments

    t0 = time.perf_counter()
    try:
        if tool == "route_and_infer":
            raw = mcp_route_and_infer(
                query=args.get("query", ""),
                domain_override=args.get("domain_override")
            )
        elif tool == "classify_intent":
            raw = mcp_classify_intent(query=args.get("query", ""))
        elif tool == "query_semantic_memory":
            raw = mcp_query_memory(
                query=args.get("query", ""),
                top_k=args.get("top_k", 3),
                domain_filter=args.get("domain_filter")
            )
        elif tool == "add_memory_fact":
            raw = mcp_add_fact(
                fact=args.get("fact", ""),
                domain=args.get("domain", "general"),
                importance=args.get("importance", 1.0)
            )
        elif tool == "get_system_telemetry":
            raw = mcp_get_telemetry()
        elif tool == "run_quantization_benchmark":
            raw = mcp_run_benchmark(domain=args.get("domain", "mathematics"))
        else:
            raise HTTPException(status_code=404, detail=f"MCP tool '{tool}' not found")

        t_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        parsed = json.loads(raw) if isinstance(raw, str) else raw

        return {
            "status": "success",
            "tool": tool,
            "duration_ms": t_ms,
            "result": parsed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mcp/resources")
def mcp_resources_endpoint(uri: str = "system://status"):
    """Fetches MCP resource contents."""
    if uri == "system://status":
        raw = get_system_resource()
    elif uri == "system://domains":
        raw = get_domains_resource()
    elif uri == "system://benchmarks":
        raw = get_benchmarks_resource()
    else:
        raise HTTPException(status_code=404, detail=f"Resource '{uri}' not found")

    return json.loads(raw) if isinstance(raw, str) else raw


# ==========================================
# METADATA & SYSTEM SPECIFICATIONS
# ==========================================

@router.get("/specs")
def get_system_specs_endpoint():
    """Returns core architectural specifications and subsystem descriptions."""
    return {
        "project_title": "Adaptive Multi-SLM AI Orchestrator",
        "subtitle": "Intelligent Multi-Model Routing, Quantization & Context-Aware Inference",
        "domains": DOMAINS,
        "specifications": SYSTEM_SPECIFICATIONS
    }


@router.get("/models")
def get_models_catalog_endpoint():
    """Returns available local Ollama models and domain configuration."""
    available = model_switcher.refresh_available_models()
    return {
        "available_local_models": available,
        "domain_assignments": DOMAINS
    }
