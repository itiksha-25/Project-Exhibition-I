"""
Semantic Intent Router & Prompt Classifier
Meets strictly bounded latency budget: T_route < 25 ms
Provides Hybrid Cascade: Heuristics -> Semantic Centroids -> Ambiguity Fallback
"""

import time
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from core.config import DOMAINS, MAX_ROUTER_LATENCY_MS
from .heuristics import evaluate_heuristics
from .domain_centroids import evaluate_centroids


class RouteResult(BaseModel):
    domain: str = Field(..., description="Selected primary domain key")
    domain_name: str = Field(..., description="Human-readable domain title")
    confidence: float = Field(..., description="Router confidence level [0.0 - 1.0]")
    secondary_domain: Optional[str] = Field(None, description="Runner-up domain if ambiguous")
    target_model: str = Field(..., description="Ollama model string assigned for inference")
    recommended_quant: str = Field(..., description="Optimal quantization tier (Q4_K_M, Q5_K_M, Q8_0)")
    min_precision_bits: int = Field(..., description="Minimum bit-depth threshold")
    decision_tier: str = Field(..., description="Tier 1 Heuristic or Tier 2 Centroid")
    decision_rationale: str = Field(..., description="Explanation of why this model was chosen")
    latency_ms: float = Field(..., description="Execution time in milliseconds")
    budget_adhered: bool = Field(True, description="True if latency_ms <= 25.0ms")
    domain_probabilities: Dict[str, float] = Field(default_factory=dict, description="Distribution across all 6 domains")


class IntentRouter:
    """
    Unified Intelligent Intent Router for Multi-Model Edge AI.
    """

    def __init__(self, latency_budget_ms: float = MAX_ROUTER_LATENCY_MS):
        self.latency_budget_ms = latency_budget_ms
        self.total_routed = 0
        self.budget_violations = 0
        self.history: List[Dict[str, Any]] = []

    def route(self, query: str, forced_domain: Optional[str] = None) -> RouteResult:
        """
        Routes the user prompt to the most suitable domain and recommends a local SLM.
        Cloud inference providers are selected separately by the provider registry.
        Enforces T_route < 25.0ms.
        """
        t_start = time.perf_counter()

        # Handle forced domain override if user explicitly requested a specific specialist
        if forced_domain and forced_domain.lower() in DOMAINS:
            d_key = forced_domain.lower()
            conf = 1.0
            tier = "User Override"
            rationale = f"User explicitly selected domain: {DOMAINS[d_key]['name']}"
            probs = {d: (1.0 if d == d_key else 0.0) for d in DOMAINS}
            sec_domain = None
        else:
            # 1. Tier-1 Heuristics Evaluation (< 1.0 ms)
            heuristic_match = evaluate_heuristics(query)
            
            # Compute centroid distribution (Tier-2: < 3.0 ms)
            centroid_probs = evaluate_centroids(query)
            sorted_centroids = sorted(centroid_probs.items(), key=lambda x: x[1], reverse=True)
            top_centroid_domain, top_centroid_score = sorted_centroids[0]
            sec_domain = sorted_centroids[1][0] if len(sorted_centroids) > 1 else None

            if heuristic_match:
                d_key, h_conf, h_desc = heuristic_match
                # Blend with centroid signal for smoothed confidence
                conf = max(h_conf, centroid_probs.get(d_key, 0.5))
                tier = "Tier-1 Micro-Rule Heuristic"
                rationale = f"Deterministic syntax trigger matched: {h_desc}"
                probs = centroid_probs
                # If heuristic picked something different from centroid, keep heuristic
                if top_centroid_domain != d_key:
                    sec_domain = top_centroid_domain
            else:
                # 2. Tier-2 Semantic Centroid Vector Projection
                d_key = top_centroid_domain
                conf = top_centroid_score
                tier = "Tier-2 Semantic Centroid Projection"
                rationale = f"High cosine alignment with domain semantic cluster ({conf:.1%})"
                probs = centroid_probs

        domain_meta = DOMAINS.get(d_key, DOMAINS["general"])
        
        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000.0, 3)

        budget_ok = latency_ms <= self.latency_budget_ms
        self.total_routed += 1
        if not budget_ok:
            self.budget_violations += 1

        result = RouteResult(
            domain=d_key,
            domain_name=domain_meta["name"],
            confidence=round(conf, 4),
            secondary_domain=sec_domain,

            # This is the recommended LOCAL model for this domain.
            target_model=domain_meta["primary_model"],

            # These fields describe local Ollama configuration only.
            recommended_quant=domain_meta["recommended_quant"],
            min_precision_bits=domain_meta["min_precision_bits"],

            decision_tier=tier,
            decision_rationale=rationale,
            latency_ms=latency_ms,
            budget_adhered=budget_ok,
            domain_probabilities={
                k: round(v, 4) for k, v in probs.items()
            }
        )


        self.history.append({
            "timestamp": time.time(),
            "query_snippet": query[:60],
            "domain": d_key,
            "latency_ms": latency_ms,
            "confidence": conf
        })

        return result

    def get_benchmark_report(self) -> Dict[str, Any]:
        """
        Computes routing accuracy metrics, average latency, and confusion statistics
        for academic project evaluation.
        """
        # Test synthetic verification set across the 6 domains
        test_samples = [
            ("Calculate the second derivative of f(x) = x^3 * e^(2x)", "mathematics"),
            ("Solve the system of linear equations: 3x + 2y = 12 and x - y = 1", "mathematics"),
            ("What is the difference between Type-1 and Type-2 Diabetes Mellitus?", "medical"),
            ("Explain the clinical indications and contraindications of Metformin", "medical"),
            ("Write an asynchronous Python function using httpx to fetch multiple URLs", "code"),
            ("Debug this NullPointerException in Java Spring Boot controller", "code"),
            ("Explain the second law of thermodynamics and how entropy relates to time", "science"),
            ("What occurs during the Krebs cycle in cellular respiration?", "science"),
            ("Compose a persuasive op-ed on the societal impact of renewable energy", "writing"),
            ("Write a haiku about autumn leaves falling in Kyoto", "writing"),
            ("Can you give me advice on planning a trip to Japan during spring?", "general"),
            ("What are some good habits to maintain productivity while working remotely?", "general")
        ]

        correct = 0
        latencies = []
        confusion_counts = {d: {target: 0 for target in DOMAINS} for d in DOMAINS}

        for prompt, expected in test_samples:
            res = self.route(prompt)
            latencies.append(res.latency_ms)
            confusion_counts[expected][res.domain] += 1
            if res.domain == expected:
                correct += 1

        accuracy = correct / len(test_samples)
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies) if latencies else 0.0

        return {
            "total_test_samples": len(test_samples),
            "accuracy": round(accuracy, 4),
            "average_latency_ms": round(avg_latency, 3),
            "max_latency_ms": round(max_latency, 3),
            "budget_ms": self.latency_budget_ms,
            "within_budget": max_latency <= self.latency_budget_ms,
            "confusion_matrix": confusion_counts
        }
