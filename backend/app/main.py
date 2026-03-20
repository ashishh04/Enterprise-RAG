"""
FastAPI Application Entry Point — Enterprise RAG Knowledge Assistant.

Configures the FastAPI app with CORS, structured logging, lifespan
management (loads FAISS index and embedding model on startup), and
registers all API routes.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.embeddings.embedder import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.retriever import RetrievalPipeline
from app.generation.generator import AnswerGenerator
from app.api.routes import router, app_state


def _setup_logging(level: str) -> None:
    """Configure structured logging."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: initialize services on startup, cleanup on shutdown.
    """
    settings = get_settings()
    _setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Enterprise RAG Knowledge Assistant — Starting Up")
    logger.info("=" * 60)

    # --- Initialize Embedding Service ---
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        cache_dir=str(settings.cache_dir),
    )
    app_state.embedding_service = embedding_service

    # --- Initialize Vector Store ---
    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension,
        index_dir=str(settings.index_dir),
    )
    vector_store.load()  # Load existing index if available
    app_state.vector_store = vector_store
    logger.info("FAISS index loaded: %d vectors", vector_store.size)

    # --- Initialize Retrieval Pipeline ---
    retrieval_pipeline = RetrievalPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
        max_context_tokens=settings.max_context_tokens,
    )
    app_state.retrieval_pipeline = retrieval_pipeline

    # --- Initialize Answer Generator ---
    answer_generator = AnswerGenerator(
        api_token=settings.hf_api_token,
        model_name=settings.generation_model,
    )
    app_state.answer_generator = answer_generator

    logger.info("All services initialized successfully.")
    logger.info("=" * 60)

    yield  # Application runs

    # --- Shutdown ---
    logger.info("Shutting down — saving index...")
    vector_store.save()
    logger.info("Shutdown complete.")


# ------------------------------------------------------------------
# App creation
# ------------------------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description=(
        "A production-grade Retrieval-Augmented Generation system for "
        "enterprise document search and AI-powered Q&A."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS ---
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request timing middleware ---
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Add server-timing header to all responses."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["Server-Timing"] = f"total;dur={elapsed:.1f}"
    return response


# --- Register routes ---
app.include_router(router, prefix="/api", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Enterprise RAG Knowledge Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
