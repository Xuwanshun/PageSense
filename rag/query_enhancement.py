from __future__ import annotations

import re

from config import Settings
from document_process.clients import build_openai_client


def classify_query(question: str, settings: Settings) -> str:
    """Return 'simple' or 'complex' based on the query's structure."""
    client = build_openai_client(settings)
    result = (
        client.generate_text(
            system_prompt=(
                "Classify the user query as either 'simple' or 'complex'.\n"
                "A query is complex if it:\n"
                "- Contains multiple questions or sub-topics\n"
                "- Requires reasoning across multiple facts\n"
                "- Uses conjunctions that imply multiple intents ('and', 'also', 'as well as')\n"
                "Otherwise it is simple.\n"
                "Respond with exactly one word: 'simple' or 'complex'."
            ),
            user_prompt=question,
        )
        .strip()
        .lower()
    )
    return "complex" if "complex" in result else "simple"


def hyde_enhance(question: str, settings: Settings) -> str:
    """Generate a hypothetical answer paragraph for use as the retrieval query."""
    client = build_openai_client(settings)
    return client.generate_text(
        system_prompt=(
            "Write a short hypothetical answer paragraph (2-4 sentences) that a document "
            "in a technical corpus might contain in response to the following question. "
            "Write it as if it were extracted verbatim from such a document, not as a "
            "direct answer to the user."
        ),
        user_prompt=question,
    ).strip()


def decompose_query(question: str, settings: Settings) -> list[str]:
    """Break a complex query into 2-4 independent sub-queries."""
    client = build_openai_client(settings)
    result = client.generate_text(
        system_prompt=(
            "Break the user query into 2-4 independent sub-queries, each targeting a "
            "single fact or concept. Return each sub-query on its own line, numbered "
            "(e.g. '1. ...'). Do not include any explanation — only the numbered sub-queries."
        ),
        user_prompt=question,
    ).strip()
    sub_queries: list[str] = []
    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned:
            sub_queries.append(cleaned)
    return sub_queries or [question]
