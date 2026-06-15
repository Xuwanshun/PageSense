"""Shared models, loaders, and guardrails for the eval framework.

Everything here is import-safe and offline: no network, no Paddle. The only
third-party dependency is PyYAML (dev-only, see requirements-dev.txt).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = EVAL_ROOT / "reports"


# ── Gold dataset models ──────────────────────────────────────────────────────
class GoldSource(BaseModel):
    """One graded gold citation. ``pages``/``sections`` are the acceptable
    evidence locations; ``relevance`` (0-3) feeds graded nDCG."""

    source: str
    pages: list[int] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    relevance: int = Field(default=1, ge=0, le=3)


class GoldExpected(BaseModel):
    sources: list[GoldSource] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    # When True the system is expected to refuse / say it cannot answer.
    no_answer: bool = False
    # Suggestion-type questions: the recommendation must cite manual evidence.
    requires_citation: bool = False
    # Suggestion-type questions: whether the recommendation is expected to be
    # actionable (used only as a label for the human/LLM judge, never asserted
    # offline).
    actionable: bool | None = None


QuestionType = Literal["factual", "procedure", "suggestion", "no_answer"]


class GoldQuestion(BaseModel):
    id: str
    type: QuestionType = "factual"
    question: str
    expected: GoldExpected = Field(default_factory=GoldExpected)


class GoldSet(BaseModel):
    version: int = 1
    questions: list[GoldQuestion]


# ── Prediction models (the offline scoring input) ────────────────────────────
class PredSource(BaseModel):
    """A retrieved/cited source. Field names mirror ``MultiAgentQAResponse``
    source payloads so live runs can serialize directly."""

    chunk_id: str | None = None
    source_filename: str | None = None
    page_number: int | None = None
    document_id: str | None = None
    sections: list[str] = Field(default_factory=list)
    score: float = 0.0


class Prediction(BaseModel):
    id: str
    question: str = ""
    answer: str = ""
    sources: list[PredSource] = Field(default_factory=list)
    # Optional retrieved-context text, populated only by live runs, so an
    # LLM judge can assess faithfulness/groundedness from a recorded file.
    evidence: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    estimated_tokens: int | None = None


class PredictionFile(BaseModel):
    run: dict[str, Any] = Field(default_factory=dict)
    predictions: list[Prediction]


# ── Loaders ──────────────────────────────────────────────────────────────────
class EvalConfig(BaseModel):
    top_k: int = 5
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5])
    page_window: int = 1
    citation_top_n: int = 2
    # Stronger-than-generator judge to reduce self-preference bias (live only).
    judge_model: str = "gpt-4o"
    usd_per_1k_tokens: float = 0.005


def load_config(path: Path) -> EvalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvalConfig.model_validate(raw)


def load_gold(path: Path) -> GoldSet:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GoldSet.model_validate(raw)


def load_predictions(path: Path) -> PredictionFile:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PredictionFile.model_validate(raw)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def new_report_dir(name: str) -> Path:
    out = REPORTS_DIR / f"{utc_stamp()}-{name}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Source matching (filename + page-window tolerant) ────────────────────────
def _norm_name(value: str | None) -> str:
    if not value:
        return ""
    return Path(value).stem.strip().lower()


def source_matches(pred: PredSource, gold: GoldSource, *, page_window: int) -> bool:
    """A predicted source matches a gold source when the file name agrees and
    the predicted page falls within ``page_window`` of any gold page. If the
    gold source lists no pages, file-name agreement alone is enough."""
    if _norm_name(pred.source_filename) != _norm_name(gold.source):
        return False
    if not gold.pages:
        return True
    if pred.page_number is None:
        return False
    return any(abs(pred.page_number - gp) <= page_window for gp in gold.pages)


def best_relevance(pred: PredSource, golds: list[GoldSource], *, page_window: int) -> int:
    """Highest graded relevance among gold sources this prediction matches
    (0 if it matches none)."""
    rels = [g.relevance for g in golds if source_matches(pred, g, page_window=page_window)]
    return max(rels) if rels else 0


# ── Cloud-cost guardrail ─────────────────────────────────────────────────────
class CloudCostError(SystemExit):
    """Raised (as a clean exit) when a model-calling run lacks confirmation."""


def require_cloud_confirmation(*, live: bool, confirm_cloud_cost: bool, n_calls: int, what: str) -> None:
    """Gate any code path that calls a paid model.

    Offline runs never reach here. Live runs must pass ``--confirm-cloud-cost``;
    otherwise we print a rough estimate and exit without spending anything.
    """
    if not live:
        raise CloudCostError(f"{what} requires --live (it calls OpenAI). Refusing to run in offline mode.")
    if not confirm_cloud_cost:
        raise CloudCostError(
            f"{what} would make ~{n_calls} OpenAI calls (embeddings + generation"
            f"{', + judge' if 'judge' in what.lower() else ''}). "
            "Re-run with --confirm-cloud-cost to proceed."
        )


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_ms)``."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000.0
