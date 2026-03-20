"""
Answer Generator — Grounded answer generation via Hugging Face Inference API.

Uses huggingface_hub InferenceClient to generate answers from a
language model, strictly grounded in the provided context.
Includes a strict system prompt to minimize hallucination and
enforce citation formatting.
"""

import logging
import time
from dataclasses import dataclass
from typing import Generator, Optional

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Your role is to answer questions accurately 
using ONLY the provided context documents.

STRICT RULES:
1. ONLY use information from the provided context to answer questions.
2. If the answer cannot be found in the provided context, respond with: 
   "Information not found in provided documents."
3. Always cite your sources using the format: [Document Name – Page X]
4. Be precise, professional, and concise.
5. If multiple documents provide relevant information, synthesize the answer and cite all sources.
6. Never make up information or use external knowledge.
7. If the context is ambiguous, acknowledge the ambiguity.

Format your response clearly with citations inline where relevant."""


@dataclass
class GenerationResult:
    """Result from the answer generation step."""

    answer: str
    model_name: str
    latency_ms: float
    error: Optional[str] = None


class AnswerGenerator:
    """
    Generates grounded answers using Hugging Face Inference API.

    Features:
    - Strict system prompt to prevent hallucination
    - Citation enforcement
    - Retry with exponential backoff
    - Streaming support
    """

    def __init__(
        self,
        api_token: str,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        self._client = InferenceClient(
            model=model_name,
            token=api_token,
        )

        logger.info("AnswerGenerator initialized with model: %s", model_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(
        self,
        query: str,
        context: str,
        citations: list[dict],
    ) -> GenerationResult:
        """
        Generate a grounded answer for the query using provided context.

        Args:
            query: The user's question.
            context: Concatenated context chunks with citation markers.
            citations: List of citation metadata dicts.

        Returns:
            GenerationResult with the answer and metrics.
        """
        start = time.perf_counter()

        if not context.strip():
            return GenerationResult(
                answer="No relevant documents found. Please upload documents first or refine your query.",
                model_name=self._model_name,
                latency_ms=0.0,
            )

        user_message = self._format_user_message(query, context)

        try:
            response = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self._max_new_tokens,
                temperature=self._temperature,
            )

            answer = response.choices[0].message.content.strip()
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Generation completed: model=%s, latency=%.1fms, answer_len=%d",
                self._model_name,
                elapsed_ms,
                len(answer),
            )

            return GenerationResult(
                answer=answer,
                model_name=self._model_name,
                latency_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Generation failed: %s", exc)
            raise

    def generate_stream(
        self,
        query: str,
        context: str,
        citations: list[dict],
    ) -> Generator[str, None, None]:
        """
        Stream a grounded answer token-by-token.

        Args:
            query: The user's question.
            context: Concatenated context chunks.
            citations: Citation metadata.

        Yields:
            Text chunks as they are generated.
        """
        if not context.strip():
            yield "No relevant documents found. Please upload documents first or refine your query."
            return

        user_message = self._format_user_message(query, context)

        try:
            stream = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self._max_new_tokens,
                temperature=self._temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield f"\n\n[Error: Generation failed — {exc}]"

    @staticmethod
    def _format_user_message(query: str, context: str) -> str:
        """Format the user message with context and question."""
        return (
            f"CONTEXT DOCUMENTS:\n"
            f"==================\n"
            f"{context}\n"
            f"==================\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Please answer the question using ONLY the context above. "
            f"Include citations in the format [Document Name – Page X]."
        )
