"""Gemini grounded answer generation."""

from __future__ import annotations

from typing import Any

from src.config import RAGConfig


NOT_FOUND_ANSWER = "I could not find enough information in the uploaded documents."


class GeminiAnswerGenerator:
    """Generate answers grounded only in retrieved chunks."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()

    def generate(self, question: str, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return NOT_FOUND_ANSWER
        if not self.config.gemini_configured:
            raise ValueError("GEMINI_API_KEY is required for RAG answer generation.")

        import google.generativeai as genai

        genai.configure(api_key=self.config.gemini_api_key)
        model = genai.GenerativeModel(self.config.gemini_generation_model)
        context = "\n\n".join(
            f"[chunk_id={chunk.get('chunk_id')} file={chunk.get('file_name')}]\n{chunk.get('chunk_text')}"
            for chunk in chunks
        )
        prompt = f"""
You answer questions about clinical trial / TMF documents.
Use ONLY the retrieved context below.
If the answer is not explicitly supported by the context, answer exactly:
{NOT_FOUND_ANSWER}

Question:
{question}

Retrieved context:
{context}
"""
        response = model.generate_content(prompt)
        answer = getattr(response, "text", "") or ""
        return answer.strip() or NOT_FOUND_ANSWER
