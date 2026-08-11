import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ValidationError
from groq import Groq
from dotenv import load_dotenv

from app.pdf_parser import parse_pdf_into_sections

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "qa_reference_docs"
TOP_K = 5

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)


class Issue(BaseModel):
    issue_id: str
    type: str
    severity: str
    flagged_text: str
    reason: str
    source_file: str
    source_section: str


class ReviewResult(BaseModel):
    status: str
    issues: list[Issue]
    summary: str


def retrieve_context(draft_text: str, top_k: int = TOP_K) -> list[dict]:
    query_embedding = embed_model.encode([draft_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    context = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context.append({"text": doc, "source_file": meta["source_file"], "heading": meta["main_heading"]})
    return context


def build_prompt(draft_text: str, context: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{c['source_file']} - {c['heading']}]\n{c['text']}" for c in context
    )

    return f"""You are a QA reviewer checking a content draft against approved reference material.

REFERENCE MATERIAL:
{context_block}

DRAFT TO REVIEW:
{draft_text}

Only flag a claim if it is clearly contradicted or unsupported by the reference material above.
Do not flag everything - correct statements should not be flagged.

Respond with ONLY valid JSON in this exact shape, no other text:
{{
  "status": "pass" or "needs_revision",
  "issues": [
    {{
      "issue_id": "issue_1",
      "type": "factual_error" or "unsupported_claim" or "writing_violation",
      "severity": "high" or "medium" or "low",
      "flagged_text": "the exact sentence from the draft",
      "reason": "why this is a problem",
      "source_file": "which reference file supports this",
      "source_section": "which section in that file"
    }}
  ],
  "summary": "short overall summary"
}}"""


def call_llm_with_retry(prompt: str, max_retries: int = 2) -> dict:
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == max_retries:
                raise
            continue


def review_draft(draft_text: str) -> ReviewResult:
    context = retrieve_context(draft_text)
    prompt = build_prompt(draft_text, context)
    raw_result = call_llm_with_retry(prompt)

    try:
        return ReviewResult(**raw_result)
    except ValidationError as e:
        raise ValueError(f"LLM returned malformed review JSON: {e}")


if __name__ == "__main__":
    draft_path = "data/drafts_to_review/homepage_and_product_overview.pdf"
    sections = parse_pdf_into_sections(draft_path)
    full_draft_text = "\n\n".join(f"{s.main_heading} - {s.sub_heading}: {s.text}" for s in sections)

    result = review_draft(full_draft_text)
    print(result.model_dump_json(indent=2))