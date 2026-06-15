"""Operational metrics: latency and a rough cost estimate.

Latency is read straight from recorded predictions (wall-clock captured during
a live run). Cost is deliberately approximate — the OpenAI client does not
surface token usage, so we estimate tokens as ~chars/4 when not recorded and
multiply by a configurable price. Treat the dollar figure as an order-of-
magnitude guide, not a bill.
"""

from __future__ import annotations

from statistics import mean

from eval._common import Prediction

CHARS_PER_TOKEN = 4.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _estimated_tokens(pred: Prediction) -> int:
    if pred.estimated_tokens is not None:
        return pred.estimated_tokens
    chars = len(pred.question) + len(pred.answer) + sum(len(e) for e in pred.evidence)
    return int(chars / CHARS_PER_TOKEN)


def aggregate(predictions: list[Prediction], *, usd_per_1k_tokens: float) -> dict:
    latencies = [p.latency_ms for p in predictions if p.latency_ms is not None]
    tokens = [_estimated_tokens(p) for p in predictions]
    total_tokens = sum(tokens)
    return {
        "n": len(predictions),
        "latency_ms": {
            "mean": mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 50) if latencies else None,
            "p95": _percentile(latencies, 95) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "estimated_tokens_total": total_tokens,
        "estimated_cost_usd": round((total_tokens / 1000.0) * usd_per_1k_tokens, 4),
        "cost_is_approximate": True,
    }
