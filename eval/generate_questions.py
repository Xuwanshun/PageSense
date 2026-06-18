"""
Generate a labeled evaluation dataset from preprocessed document chunks.

For each sampled chunk, the LLM produces:
  - question:            something a reader would ask that requires that chunk to answer
  - ground_truth:        a concise correct answer grounded in the chunk text
  - relevant_chunk_ids:  [chunk_id]  (the source chunk; retriever must surface it)

Output is a JSONL file, one JSON object per line.

Usage:
    python -m eval.generate_questions \
        --document-id <id> \
        --output eval/test_questions.jsonl \
        --n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from config import Settings
from document_process.clients import build_openai_client

logger = logging.getLogger(__name__)


def _load_chunks(processed_dir: Path, document_id: str) -> list[dict]:
    path = processed_dir / document_id / "chunks.json"
    if not path.exists():
        raise FileNotFoundError(f"chunks.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _generate_qa_for_chunk(client, chunk: dict) -> dict | None:
    text = chunk.get("text", "").strip()
    if len(text) < 100:
        return None
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    try:
        raw = client.client.chat.completions.create(
            model=client.model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an exam question writer. Given a passage from a research paper, "
                        "write ONE specific factual question that can only be answered using the passage, "
                        "and a concise ground-truth answer (1-3 sentences). "
                        "Avoid vague questions like 'What is discussed here?' — be specific about a concept, "
                        "number, comparison, or mechanism described in the passage. "
                        'Return JSON: {"question": "...", "ground_truth": "..."}'
                    ),
                },
                {"role": "user", "content": f"Passage:\n{text[:1200]}"},
            ],
        )
        payload = json.loads(raw.choices[0].message.content or "{}")
        q = payload.get("question", "").strip()
        gt = payload.get("ground_truth", "").strip()
        if not q or not gt:
            return None
        return {
            "question": q,
            "ground_truth": gt,
            "relevant_chunk_ids": [chunk_id],
            "source_page": chunk.get("metadata", {}).get("page_number"),
        }
    except Exception as exc:
        logger.warning("Failed to generate QA for chunk %s: %s", chunk_id, exc)
        return None


def generate(
    document_id: str,
    output_path: Path,
    *,
    n: int = 20,
    seed: int = 42,
    settings: Settings | None = None,
) -> list[dict]:
    resolved = settings or Settings()
    client = build_openai_client(resolved)
    chunks = _load_chunks(resolved.processed_documents_dir, document_id)

    random.seed(seed)
    # Prefer chunks with more text (more likely to have answerable content)
    chunks_sorted = sorted(chunks, key=lambda c: len(c.get("text", "")), reverse=True)
    # Sample from top 60% to avoid very short tail chunks
    pool = chunks_sorted[: max(n * 3, len(chunks_sorted) // 2 + 1)]
    sample = random.sample(pool, min(n, len(pool)))

    results: list[dict] = []
    for i, chunk in enumerate(sample, start=1):
        logger.info("Generating QA %d/%d for chunk %s", i, len(sample), chunk.get("chunk_id"))
        qa = _generate_qa_for_chunk(client, chunk)
        if qa:
            results.append(qa)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Wrote %d QA pairs to %s", len(results), output_path)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate eval QA pairs from document chunks")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", default="eval/test_questions.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(
        args.document_id,
        Path(args.output),
        n=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
