"""
API Routes — FastAPI endpoints for the Enterprise RAG system.

Provides /upload, /query, /query/stream, and /health endpoints with
Pydantic request/response models, async processing, latency tracking,
and comprehensive error handling.
"""

import logging
import shutil
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.parser import PDFParser
from app.ingestion.chunker import SemanticChunker

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of results")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Min similarity")


class CitationResponse(BaseModel):
    """A single citation reference."""

    document: str
    page: Any
    chunk_id: str
    score: float
    source: str = ""


class MetricsResponse(BaseModel):
    """Latency and usage metrics."""

    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    chunks_retrieved: int
    avg_similarity: float


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""

    answer: str
    citations: list[CitationResponse]
    metrics: MetricsResponse


class UploadResponse(BaseModel):
    """Response body for /upload endpoint."""

    document_id: str
    title: str
    chunks_created: int
    total_pages: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    index_size: int
    embedding_model: str
    generation_model: str


# ------------------------------------------------------------------
# Dependency references (set by main.py at startup)
# ------------------------------------------------------------------


class _AppState:
    """Mutable container for app-level dependencies (set at startup)."""

    embedding_service: Any = None
    vector_store: Any = None
    retrieval_pipeline: Any = None
    answer_generator: Any = None


app_state = _AppState()


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
) -> UploadResponse:
    """
    Upload a PDF document for ingestion.

    Parses the PDF, chunks the text, generates embeddings, and indexes
    the chunks in the FAISS vector store.
    """
    start = time.perf_counter()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    settings = get_settings()

    # Save uploaded file to disk
    upload_path = settings.upload_dir / file.filename
    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc

    try:
        # Parse PDF
        parser = PDFParser()
        document = parser.parse(upload_path, title=title)

        # Chunk document
        chunker = SemanticChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = chunker.chunk_document(document)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text content could be extracted from the PDF.",
            )

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        vectors = app_state.embedding_service.embed_texts(texts)

        # Build metadata for vector store
        metadata_list = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        # Add to vector store
        app_state.vector_store.add_documents(vectors, metadata_list)
        app_state.vector_store.save()

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Document '%s' ingested: %d chunks, %.1fms",
            document.title,
            len(chunks),
            elapsed_ms,
        )

        return UploadResponse(
            document_id=document.document_id,
            title=document.title,
            chunks_created=len(chunks),
            total_pages=document.total_pages,
            processing_time_ms=round(elapsed_ms, 1),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", file.filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the knowledge base and get a grounded answer.

    Performs semantic retrieval, then generates an answer using the
    retrieved context with citations.
    """
    total_start = time.perf_counter()

    if app_state.vector_store.size == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents indexed. Please upload documents first.",
        )

    try:
        # Retrieval
        retrieval_result = app_state.retrieval_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )

        # Generation
        generation_result = app_state.answer_generator.generate(
            query=request.query,
            context=retrieval_result.context_text,
            citations=retrieval_result.citations,
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        # Calculate average similarity
        scores = [c.score for c in retrieval_result.chunks]
        avg_sim = sum(scores) / len(scores) if scores else 0.0

        return QueryResponse(
            answer=generation_result.answer,
            citations=[CitationResponse(**c) for c in retrieval_result.citations],
            metrics=MetricsResponse(
                retrieval_latency_ms=round(retrieval_result.latency_ms, 1),
                generation_latency_ms=round(generation_result.latency_ms, 1),
                total_latency_ms=round(total_ms, 1),
                chunks_retrieved=len(retrieval_result.chunks),
                avg_similarity=round(avg_sim, 4),
            ),
        )

    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest) -> StreamingResponse:
    """
    Query with streaming response.

    Returns a Server-Sent Events stream of answer tokens.
    """
    if app_state.vector_store.size == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents indexed. Please upload documents first.",
        )

    retrieval_result = app_state.retrieval_pipeline.retrieve(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    def event_stream():
        import json
        # Send citations first
        citations_data = json.dumps({"type": "citations", "data": retrieval_result.citations})
        yield f"data: {citations_data}\n\n"

        # Stream answer tokens
        for token in app_state.answer_generator.generate_stream(
            query=request.query,
            context=retrieval_result.context_text,
            citations=retrieval_result.citations,
        ):
            chunk_data = json.dumps({"type": "token", "data": token})
            yield f"data: {chunk_data}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health check endpoint."""
    settings = get_settings()

    return HealthResponse(
        status="healthy",
        index_size=app_state.vector_store.size if app_state.vector_store else 0,
        embedding_model=settings.embedding_model,
        generation_model=settings.generation_model,
    )
