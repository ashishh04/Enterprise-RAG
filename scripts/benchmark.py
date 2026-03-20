"""
Benchmark Script — Measures latency for ingestion, embedding, retrieval, and generation.

Usage:
    python scripts/benchmark.py --pdf <path_to_pdf> --queries <num_queries>

Requires the backend's .env to be configured with a valid HF_API_TOKEN.
"""

import argparse
import os
import statistics
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))


def run_benchmark(pdf_path: str, num_queries: int = 10) -> None:
    """Run ingestion + query benchmark."""
    from app.config import get_settings
    from app.ingestion.parser import PDFParser
    from app.ingestion.chunker import SemanticChunker
    from app.embeddings.embedder import EmbeddingService
    from app.vectorstore.faiss_store import FAISSVectorStore
    from app.retrieval.retriever import RetrievalPipeline
    from app.generation.generator import AnswerGenerator

    settings = get_settings()

    print("=" * 60)
    print("Enterprise RAG — Latency Benchmark")
    print("=" * 60)

    # --- Ingestion ---
    print("\n[1/5] Parsing PDF...")
    t0 = time.perf_counter()
    parser = PDFParser()
    document = parser.parse(pdf_path)
    parse_time = (time.perf_counter() - t0) * 1000
    print(f"  Parsed: {document.total_pages} pages in {parse_time:.1f}ms")

    print("\n[2/5] Chunking text...")
    t0 = time.perf_counter()
    chunker = SemanticChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk_document(document)
    chunk_time = (time.perf_counter() - t0) * 1000
    print(f"  Created: {len(chunks)} chunks in {chunk_time:.1f}ms")

    # --- Embedding ---
    print("\n[3/5] Generating embeddings...")
    t0 = time.perf_counter()
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
    )
    texts = [c.text for c in chunks]
    vectors = embedding_service.embed_texts(texts)
    embed_time = (time.perf_counter() - t0) * 1000
    print(f"  Embedded: {len(vectors)} chunks in {embed_time:.1f}ms")
    print(f"  Per-chunk: {embed_time / max(len(vectors), 1):.1f}ms")

    # --- Indexing ---
    print("\n[4/5] Indexing into FAISS...")
    t0 = time.perf_counter()
    store = FAISSVectorStore(dimension=embedding_service.dimension)
    metadata_list = [
        {"chunk_id": c.chunk_id, "document_id": c.document_id, "text": c.text, **c.metadata}
        for c in chunks
    ]
    store.add_documents(vectors, metadata_list)
    index_time = (time.perf_counter() - t0) * 1000
    print(f"  Indexed: {store.size} vectors in {index_time:.1f}ms")

    # --- Query Benchmark ---
    print(f"\n[5/5] Running {num_queries} queries...")
    pipeline = RetrievalPipeline(
        embedding_service=embedding_service,
        vector_store=store,
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
        max_context_tokens=settings.max_context_tokens,
    )

    sample_queries = [
        "What are the main findings?",
        "Summarize the key points.",
        "What methodology was used?",
        "What conclusions were drawn?",
        "What data was analyzed?",
        "What are the recommendations?",
        "What challenges were identified?",
        "What is the scope of this document?",
        "What metrics are discussed?",
        "What are the limitations?",
    ]

    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    generator = AnswerGenerator(
        api_token=settings.hf_api_token,
        model_name=settings.generation_model,
    )

    for i in range(min(num_queries, len(sample_queries))):
        query = sample_queries[i % len(sample_queries)]
        t_total = time.perf_counter()

        result = pipeline.retrieve(query)
        retrieval_latencies.append(result.latency_ms)

        t_gen = time.perf_counter()
        gen_result = generator.generate(query, result.context_text, result.citations)
        gen_ms = (time.perf_counter() - t_gen) * 1000
        generation_latencies.append(gen_ms)

        total_ms = (time.perf_counter() - t_total) * 1000
        total_latencies.append(total_ms)

        print(f"  Query {i + 1}: retrieval={result.latency_ms:.1f}ms, generation={gen_ms:.1f}ms, total={total_ms:.1f}ms")

    # --- Report ---
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n📄 Ingestion:")
    print(f"   Parse time:     {parse_time:.1f}ms")
    print(f"   Chunk time:     {chunk_time:.1f}ms")
    print(f"   Embed time:     {embed_time:.1f}ms")
    print(f"   Index time:     {index_time:.1f}ms")
    print(f"   Total ingest:   {parse_time + chunk_time + embed_time + index_time:.1f}ms")

    if retrieval_latencies:
        print(f"\n🔍 Retrieval (n={len(retrieval_latencies)}):")
        print(f"   p50: {statistics.median(retrieval_latencies):.1f}ms")
        print(f"   p95: {sorted(retrieval_latencies)[int(len(retrieval_latencies) * 0.95)]:.1f}ms")
        print(f"   avg: {statistics.mean(retrieval_latencies):.1f}ms")

    if generation_latencies:
        print(f"\n⚡ Generation (n={len(generation_latencies)}):")
        print(f"   p50: {statistics.median(generation_latencies):.1f}ms")
        print(f"   p95: {sorted(generation_latencies)[int(len(generation_latencies) * 0.95)]:.1f}ms")
        print(f"   avg: {statistics.mean(generation_latencies):.1f}ms")

    if total_latencies:
        print(f"\n🏁 Total Latency (n={len(total_latencies)}):")
        print(f"   p50: {statistics.median(total_latencies):.1f}ms")
        print(f"   p95: {sorted(total_latencies)[int(len(total_latencies) * 0.95)]:.1f}ms")
        print(f"   avg: {statistics.mean(total_latencies):.1f}ms")

    print(f"\n📊 Index stats: {store.size} vectors, {embedding_service.dimension}d")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise RAG Benchmark")
    parser.add_argument("--pdf", required=True, help="Path to a PDF file to benchmark.")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries to run.")
    args = parser.parse_args()

    run_benchmark(args.pdf, args.queries)
