from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from config import Settings
from document_process.clients import build_openai_client
from document_process.models import ProcessedChunk, ProcessedDocument
from rag.chunk import ChunkRecord, RetrievedChunk, chunk_records_from_processed_chunks
from rag.embed import EmbeddingBackend, build_embedding_backend
from rag.hybrid import BM25Index, apply_region_boost, expand_to_parent_context, rrf_fuse


@dataclass(frozen=True)
class QAResponse:
    question: str
    answer: str
    sources: list[dict[str, Any]]


class VectorStore(Protocol):
    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None: ...

    def query(
        self, embedding: list[float], top_k: int, *, doc_filter: list[str] | None = None
    ) -> list[RetrievedChunk]: ...

    def bm25_query(
        self, query: str, top_k: int, *, doc_filter: list[str] | None = None
    ) -> list[RetrievedChunk]: ...

    def get_all_chunks(self, *, doc_filter: list[str] | None = None) -> list[RetrievedChunk]: ...

    def get_all_descriptors(self, processed_documents_dir: Path) -> list[dict[str, Any]]: ...


class JsonVectorStore:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.store_path.exists():
            return []
        payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        return payload.get("rows", [])

    def _save_rows(self, rows: list[dict[str, Any]]) -> None:
        self.store_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        existing = {row["chunk_id"]: row for row in self._load_rows()}
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            existing[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }
        self._save_rows(list(existing.values()))

    def query(self, embedding: list[float], top_k: int, *, doc_filter: list[str] | None = None) -> list[RetrievedChunk]:
        scored: list[RetrievedChunk] = []
        filter_set = set(doc_filter) if doc_filter is not None else None
        for row in self._load_rows():
            if filter_set is not None:
                doc_id = row.get("metadata", {}).get("document_id")
                if doc_id not in filter_set:
                    continue
            score = _cosine_similarity(embedding, row.get("embedding", []))
            scored.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row.get("text", ""),
                    metadata=row.get("metadata", {}),
                    score=score,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def bm25_query(self, query: str, top_k: int, *, doc_filter: list[str] | None = None) -> list[RetrievedChunk]:
        rows = self._load_rows()
        filter_set = set(doc_filter) if doc_filter is not None else None
        chunk_ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for row in rows:
            if filter_set is not None:
                doc_id = row.get("metadata", {}).get("document_id")
                if doc_id not in filter_set:
                    continue
            chunk_ids.append(row["chunk_id"])
            texts.append(row.get("text", ""))
            metadatas.append(row.get("metadata", {}))
        index = BM25Index()
        index.build(chunk_ids, texts, metadatas)
        return index.query(query, top_k)

    def get_all_chunks(self, *, doc_filter: list[str] | None = None) -> list[RetrievedChunk]:
        filter_set = set(doc_filter) if doc_filter is not None else None
        result: list[RetrievedChunk] = []
        for row in self._load_rows():
            if filter_set is not None:
                doc_id = row.get("metadata", {}).get("document_id")
                if doc_id not in filter_set:
                    continue
            result.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row.get("text", ""),
                    metadata=row.get("metadata", {}),
                    score=0.0,
                )
            )
        return result

    def get_all_descriptors(self, processed_documents_dir: Path) -> list[dict[str, Any]]:
        seen_ids: set[str] = set()
        for row in self._load_rows():
            doc_id = row.get("metadata", {}).get("document_id")
            if doc_id:
                seen_ids.add(str(doc_id))
        result: list[dict[str, Any]] = []
        for doc_id in seen_ids:
            doc_path = processed_documents_dir / doc_id / "document.json"
            if not doc_path.exists():
                continue
            try:
                payload = json.loads(doc_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            embedding = payload.get("summary_embedding")
            if embedding:
                result.append({"document_id": doc_id, "summary_embedding": embedding})
        return result


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "rag_agent_pdf") -> None:
        try:
            import chromadb  # type: ignore
        except Exception as exc:
            raise RuntimeError("chromadb is not installed.") from exc

        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection(name=collection_name)

    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )

    def query(self, embedding: list[float], top_k: int, *, doc_filter: list[str] | None = None) -> list[RetrievedChunk]:
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if doc_filter:
            query_kwargs["where"] = {"document_id": {"$in": doc_filter}}
        response = self.collection.query(**query_kwargs)
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text or "",
                metadata=metadata or {},
                score=1.0 - float(distance),
            )
            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False)
        ]

    def bm25_query(self, query: str, top_k: int, *, doc_filter: list[str] | None = None) -> list[RetrievedChunk]:
        # ChromaDB does not expose raw text for BM25 — return empty so the
        # caller falls back to dense-only retrieval gracefully.
        return []

    def get_all_chunks(self, *, doc_filter: list[str] | None = None) -> list[RetrievedChunk]:
        return []

    def get_all_descriptors(self, processed_documents_dir: Path) -> list[dict[str, Any]]:
        return []


class DocumentRetriever:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_backend = embedding_backend or build_embedding_backend(settings)
        self.vector_store = vector_store or build_vector_store(settings)

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        embeddings = self.embedding_backend.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.upsert(chunks, embeddings)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        doc_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_backend.embed_texts([question])[0]
        return self.vector_store.query(
            query_embedding,
            top_k or self.settings.default_top_k,
            doc_filter=doc_filter or None,
        )

    def hybrid_retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        doc_filter: list[str] | None = None,
        parent_threshold: int = 2,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval: dense vector + BM25 sparse, fused via RRF.

        Post-processing steps applied in order:
        1. RRF fusion of dense and sparse ranked lists
        2. Region-type boost (1.3× for table/figure when query has data intent)
        3. Parent-context expansion (replace clusters of siblings with full section)
        """
        k = top_k or self.settings.default_top_k
        fetch_k = k * 3  # over-fetch so fusion has enough candidates

        query_embedding = self.embedding_backend.embed_texts([question])[0]
        dense_results = self.vector_store.query(
            query_embedding, fetch_k, doc_filter=doc_filter or None
        )
        sparse_results = self.vector_store.bm25_query(
            question, fetch_k, doc_filter=doc_filter or None
        )

        fused = rrf_fuse(dense_results, sparse_results)
        fused = apply_region_boost(fused, question)
        fused = fused[:k]

        all_chunks = self.vector_store.get_all_chunks(doc_filter=doc_filter or None)
        fused = expand_to_parent_context(fused, all_chunks, sibling_threshold=parent_threshold)

        return fused

    def filter_by_relevance(self, query_embedding: list[float], threshold: float) -> list[str]:
        descriptors = self.vector_store.get_all_descriptors(self.settings.processed_documents_dir)
        matching: list[str] = []
        for desc in descriptors:
            emb = desc.get("summary_embedding")
            if not emb:
                continue
            if _cosine_similarity(query_embedding, emb) >= threshold:
                matching.append(str(desc["document_id"]))
        return matching

    def index_processed_chunks(
        self,
        chunks: list[ProcessedChunk],
        *,
        document_id: str | None = None,
        source_filename: str | None = None,
    ) -> int:
        records = chunk_records_from_processed_chunks(
            chunks,
            document_id=document_id,
            source_filename=source_filename,
        )
        if not records:
            return 0
        self.upsert_chunks(records)
        return len(records)

    def answer_question(self, question: str, *, top_k: int | None = None) -> QAResponse:
        retrieved = self.retrieve(question, top_k=top_k)
        return answer_question(question, retrieved, settings=self.settings)


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.prefer_chroma:
        try:
            return ChromaVectorStore(settings.vectorstore_dir)
        except RuntimeError:
            pass
    return JsonVectorStore(settings.vectorstore_dir / "store.json")


def load_processed_document_bundle(document_dir: Path) -> tuple[ProcessedDocument | None, list[ProcessedChunk]]:
    document_payload = _load_json(_artifact_path(document_dir, "document.json"))
    chunks_payload = _load_json(_artifact_path(document_dir, "chunks.json")) or []
    document = ProcessedDocument.model_validate(document_payload) if isinstance(document_payload, dict) else None
    chunks = [ProcessedChunk.model_validate(item) for item in chunks_payload if isinstance(item, dict)]
    return document, chunks


def index_processed_document(
    document_id_or_path: str | Path,
    *,
    settings: Settings | None = None,
    retriever: DocumentRetriever | None = None,
) -> int:
    resolved_settings = settings or Settings()
    active_retriever = retriever or DocumentRetriever(resolved_settings)
    document_dir = _resolve_processed_document_dir(document_id_or_path, resolved_settings)
    document, chunks = load_processed_document_bundle(document_dir)
    return active_retriever.index_processed_chunks(
        chunks,
        document_id=document.document_id if document else document_dir.name,
        source_filename=document.source_filename if document else None,
    )


def index_all_processed_documents(
    *,
    settings: Settings | None = None,
    retriever: DocumentRetriever | None = None,
) -> dict[str, int]:
    resolved_settings = settings or Settings()
    active_retriever = retriever or DocumentRetriever(resolved_settings)
    indexed: dict[str, int] = {}
    for document_dir in sorted(path for path in resolved_settings.processed_documents_dir.iterdir() if path.is_dir()):
        document, chunks = load_processed_document_bundle(document_dir)
        if not chunks:
            continue
        document_id = document.document_id if document else document_dir.name
        indexed[document_id] = active_retriever.index_processed_chunks(
            chunks,
            document_id=document_id,
            source_filename=document.source_filename if document else None,
        )
    return indexed


def answer_corpus_question(
    question: str,
    *,
    settings: Settings | None = None,
    retriever: DocumentRetriever | None = None,
    top_k: int | None = None,
) -> QAResponse:
    resolved_settings = settings or Settings()
    active_retriever = retriever or DocumentRetriever(resolved_settings)
    return active_retriever.answer_question(question, top_k=top_k)


def answer_question(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    settings: Settings,
) -> QAResponse:
    if not retrieved_chunks:
        return QAResponse(
            question=question,
            answer="I cannot answer from the indexed documents because no relevant context was retrieved.",
            sources=[],
        )
    prompt = _build_qa_prompt(question, retrieved_chunks)
    answer = _generate_qa_answer(prompt=prompt, settings=settings)
    return QAResponse(
        question=question,
        answer=answer,
        sources=[_source_payload(chunk) for chunk in retrieved_chunks],
    )


def _build_qa_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_sections = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        page_number = chunk.metadata.get("page_number")
        label = f"Source {index} | chunk={chunk.chunk_id}"
        if page_number is not None:
            label += f" | page={page_number}"
        context_sections.append(f"[{label}]\n{chunk.text.strip()}")
    context = "\n\n".join(section for section in context_sections if section.strip())
    return (
        "Answer the question using only the provided context.\n"
        "Do not invent facts.\n"
        "If the answer is not in the context, say: I cannot answer from the provided context.\n"
        "Keep the answer concise and cite chunk/page identifiers when available.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}"
    )


def _generate_qa_answer(*, prompt: str, settings: Settings) -> str:
    client = build_openai_client(settings)
    return client.generate_text(
        system_prompt="You are a grounded QA assistant.",
        user_prompt=prompt,
    ).strip()


def _source_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "page_number": chunk.metadata.get("page_number"),
        "document_id": chunk.metadata.get("document_id"),
        "source_filename": chunk.metadata.get("source_filename") or chunk.metadata.get("source_file"),
        "score": round(chunk.score, 4),
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return dot / (left_norm * right_norm)


def _artifact_path(document_dir: Path, filename: str) -> Path:
    direct_path = document_dir / filename
    if direct_path.exists():
        return direct_path
    return document_dir / "structured" / filename


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_processed_document_dir(document_id_or_path: str | Path, settings: Settings) -> Path:
    candidate = Path(document_id_or_path)
    if candidate.exists():
        return candidate
    return settings.processed_documents_dir / candidate
