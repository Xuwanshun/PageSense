"""
Faithfulness verification and answer correction.

After the synthesis agent produces an answer, the FaithfulnessChecker
breaks it into individual claims and checks each one against the retrieved
source passages.  If unsupported or inferred claims are found, the
FaithfulnessCorrector rewrites the answer to remove or hedge them.

Controlled by ``use_faithfulness_check`` in Settings.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from config import Settings
from document_process.clients import build_openai_client
from rag.chunk import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

FAITHFULNESS_CHECK_PROMPT = """You are a faithfulness verifier for a document QA system.
Your job is to check whether each claim in a generated answer is directly supported \
by the provided source passages.

This is a strict factual grounding check — not a quality or style review.

Query that was asked: {query}

Source passages used for generation:
{source_passages}

Generated answer to verify:
{generated_answer}

Instructions:
1. Break the generated answer into individual claims (facts, numbers, conclusions)
2. For each claim, find the source chunk(s) that support it
3. Mark each claim as:
   - SUPPORTED: claim is directly stated or numerically/logically entailed by a source chunk
   - INFERRED: claim is a reasonable inference but not directly stated — flag for review
   - UNSUPPORTED: claim has no basis in the provided source chunks — this is a hallucination
4. For table/figure claims: verify that cited numbers/trends appear in the chunk description,
   not just that a table on the topic exists
5. Do not penalize correct claims just because they seem obvious — only flag what \
cannot be traced to a source

Output format (strict JSON):
{{
  "claims": [
    {{
      "claim_text": "...",
      "status": "SUPPORTED|INFERRED|UNSUPPORTED",
      "source_chunk_ids": ["chunk_id_1"],
      "note": "optional short explanation if INFERRED or UNSUPPORTED"
    }}
  ],
  "overall_verdict": "FAITHFUL|PARTIALLY_FAITHFUL|UNFAITHFUL",
  "confidence_score": 0.0,
  "recommended_action": "return_as_is|flag_for_review|regenerate_without_unsupported_claims"
}}"""


FAITHFULNESS_CORRECTION_PROMPT = """A faithfulness check found the following issues in a generated answer.
Rewrite the answer removing or correcting unsupported claims.

Original query: {query}

Original answer: {original_answer}

Faithfulness issues found:
{unsupported_claims}

Verified source passages:
{source_passages}

Rewrite rules:
1. Remove or soften any UNSUPPORTED claim — do not replace with a guess
2. For INFERRED claims: add hedging language ("the document suggests...", "based on the table...")
3. Keep all SUPPORTED claims exactly as they were — do not rephrase correct information
4. If removing unsupported claims leaves the answer unable to address the query, \
state explicitly: "The retrieved documents do not contain sufficient information to answer this."
5. Maintain the same structure and citation format as the original answer

Rewritten answer:"""


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ClaimVerdict:
    claim_text: str
    status: str  # SUPPORTED | INFERRED | UNSUPPORTED
    source_chunk_ids: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class FaithfulnessResult:
    claims: list[ClaimVerdict]
    overall_verdict: str  # FAITHFUL | PARTIALLY_FAITHFUL | UNFAITHFUL
    confidence_score: float
    recommended_action: str  # return_as_is | flag_for_review | regenerate_without_unsupported_claims


# ── Internal helpers ──────────────────────────────────────────────────────────


def _format_source_passages(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for chunk in chunks:
        parent_title = chunk.metadata.get("parent_title") or ""
        region_types = chunk.metadata.get("region_types") or []
        region_label = "TABLE" if "table" in region_types else "FIGURE" if "figure" in region_types else "PROSE"
        lines.append(f"[CHUNK {chunk.chunk_id} | {region_label} | {parent_title}]\n{chunk.text}")
    return "\n\n".join(lines)


def _format_problem_claims(claims: list[ClaimVerdict]) -> str:
    lines: list[str] = []
    for claim in claims:
        if claim.status in ("UNSUPPORTED", "INFERRED"):
            lines.append(f'Claim: "{claim.claim_text}"\nStatus: {claim.status}\nNote: {claim.note or "N/A"}')
    return "\n\n".join(lines)


def _parse_check_response(raw: str) -> FaithfulnessResult:
    """Parse the JSON faithfulness check response with graceful fallback."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        payload = {}
        if match:
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    raw_claims = payload.get("claims") or []
    claims: list[ClaimVerdict] = []
    for c in raw_claims:
        if not isinstance(c, dict):
            continue
        claims.append(
            ClaimVerdict(
                claim_text=str(c.get("claim_text") or ""),
                status=str(c.get("status") or "UNSUPPORTED"),
                source_chunk_ids=[str(x) for x in (c.get("source_chunk_ids") or [])],
                note=str(c.get("note") or ""),
            )
        )

    return FaithfulnessResult(
        claims=claims,
        overall_verdict=str(payload.get("overall_verdict") or "FAITHFUL"),
        confidence_score=float(payload.get("confidence_score") or 1.0),
        recommended_action=str(payload.get("recommended_action") or "return_as_is"),
    )


# ── Public API ────────────────────────────────────────────────────────────────


class FaithfulnessChecker:
    """
    Verifies that a generated answer is grounded in the retrieved source passages,
    and rewrites it if unsupported claims are found.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check(
        self,
        query: str,
        source_chunks: list[RetrievedChunk],
        generated_answer: str,
    ) -> FaithfulnessResult:
        """
        Break the answer into claims and verify each against source chunks.

        Falls back to a FAITHFUL verdict on LLM failure so the pipeline is
        never blocked by a verification error.
        """
        source_passages = _format_source_passages(source_chunks)
        user_prompt = FAITHFULNESS_CHECK_PROMPT.format(
            query=query,
            source_passages=source_passages,
            generated_answer=generated_answer,
        )

        try:
            client = build_openai_client(self.settings)
            raw = client.generate_text(
                system_prompt="You are a faithfulness verifier. Respond with valid JSON only.",
                user_prompt=user_prompt,
            )
            return _parse_check_response(raw)
        except Exception as exc:
            logger.warning("Faithfulness check failed, treating answer as faithful: %s", exc)
            return FaithfulnessResult(
                claims=[],
                overall_verdict="FAITHFUL",
                confidence_score=1.0,
                recommended_action="return_as_is",
            )

    def correct(
        self,
        query: str,
        original_answer: str,
        result: FaithfulnessResult,
        source_chunks: list[RetrievedChunk],
    ) -> str:
        """
        Rewrite the answer to remove or hedge any unsupported/inferred claims
        identified by a prior ``check()`` call.

        Returns the original answer on LLM failure.
        """
        problem_claims = [c for c in result.claims if c.status in ("UNSUPPORTED", "INFERRED")]
        if not problem_claims:
            return original_answer

        source_passages = _format_source_passages(source_chunks)
        unsupported_block = _format_problem_claims(problem_claims)

        user_prompt = FAITHFULNESS_CORRECTION_PROMPT.format(
            query=query,
            original_answer=original_answer,
            unsupported_claims=unsupported_block,
            source_passages=source_passages,
        )

        try:
            client = build_openai_client(self.settings)
            return client.generate_text(
                system_prompt=(
                    "You are a faithfulness editor. "
                    "Rewrite the answer to remove unsupported claims while preserving all supported content."
                ),
                user_prompt=user_prompt,
            ).strip()
        except Exception as exc:
            logger.warning("Faithfulness correction failed, returning original answer: %s", exc)
            return original_answer
