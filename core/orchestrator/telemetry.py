"""
Telemetry & Hardware Metric Tracker
Tracks real-time system metrics, swap frequencies, cache-hit ratios, and latency distributions.
"""

import time
from typing import Dict, List, Any, Optional
from collections import deque


class TelemetryTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.ttft_history: deque = deque(maxlen=window_size)
        self.tps_history: deque = deque(maxlen=window_size)
        self.routing_latencies: deque = deque(maxlen=window_size)
        self.memory_retrieval_latencies: deque = deque(maxlen=window_size)
        self.swap_events: deque = deque(maxlen=window_size)
        
        self.total_queries = 0
        self.hot_switches = 0
        self.cold_switches = 0

    def record_inference(
    self,
    domain: str,
    model: str,
    ttft_ms: Optional[float],
    tps: Optional[float],
    route_latency_ms: float,
    mem_latency_ms: float,
    is_hot: bool = False,
    swap_total_ms: float = 0.0,
    provider: str = "ollama"
):
    self.total_queries += 1

    if provider == "ollama":
        if is_hot:
            self.hot_switches += 1
        else:
            self.cold_switches += 1

        self.swap_events.append({
            "timestamp": time.time(),
            "domain": domain,
            "model": model,
            "provider": provider,
            "is_hot": is_hot,
            "swap_total_ms": swap_total_ms
        })

    if ttft_ms is not None:
        self.ttft_history.append(ttft_ms)

    if tps is not None:
        self.tps_history.append(tps)

    self.routing_latencies.append(route_latency_ms)
    self.memory_retrieval_latencies.append(mem_latency_ms)
    def get_summary(self) -> Dict[str, Any]:
        avg_ttft = sum(self.ttft_history) / len(self.ttft_history) if self.ttft_history else 0.0
        avg_tps = sum(self.tps_history) / len(self.tps_history) if self.tps_history else 0.0
        avg_route = sum(self.routing_latencies) / len(self.routing_latencies) if self.routing_latencies else 0.0
        avg_mem = sum(self.memory_retrieval_latencies) / len(self.memory_retrieval_latencies) if self.memory_retrieval_latencies else 0.0

        cache_hit_rate = (self.hot_switches / self.total_queries * 100.0) if self.total_queries > 0 else 100.0

        return {
            "total_queries": self.total_queries,
            "hot_switches": self.hot_switches,
            "cold_switches": self.cold_switches,
            "cache_hit_ratio_percent": round(cache_hit_rate, 1),
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_tps": round(avg_tps, 1),
            "avg_routing_latency_ms": round(avg_route, 3),
            "avg_memory_retrieval_ms": round(avg_mem, 3),
            "recent_swaps": list(self.swap_events)[-5:]
        }
