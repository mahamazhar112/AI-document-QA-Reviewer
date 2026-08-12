import json
import os
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ValidationError
from groq import Groq
from dotenv import load_dotenv

from app.pdf_parser import parse_pdf_into_sections

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "qa_reference_docs"
TOP_K = 4

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


def group_by_main_heading(sections) -> dict:
    grouped = defaultdict(list)
    for s in sections:
        if s.main_heading.strip().lower() == "contents":
            continue
        grouped[s.main_heading].append(s)
    return grouped


def retrieve_context(query_text: str, top_k: int = TOP_K) -> list[dict]:
    query_embedding = embed_model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    context = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context.append({"text": doc, "source_file": meta["source_file"], "heading": meta["main_heading"]})
    return context


def build_prompt(main_heading: str, section_text: str, context: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{c['source_file']} - {c['heading']}]\n{c['text']}" for c in context
    )

    return f"""You are a QA reviewer checking one section of a content draft against approved reference material.

REFERENCE MATERIAL:
{context_block}

DRAFT SECTION: "{main_heading}"
{section_text}

Rules:
- Only flag a claim if it is clearly contradicted or unsupported by the reference material above.
- Do not flag correct statements.
- Ignore sentences that are meta-commentary about the draft itself (e.g. "other sections of this draft...", "the copy is intentionally written as...", "some statements are narrow and factual while others..."). These are not product claims and must never be flagged.
- Only evaluate actual claims about the product, its features, plans, security, or behavior.
- If the same underlying problem appears more than once in this section, report it once.
- If nothing is wrong, return an empty issues list.

Respond with ONLY valid JSON, no other text:
{{
  "issues": [
    {{
      "type": "factual_error" or "unsupported_claim" or "writing_violation",
      "severity": "high" or "medium" or "low",
      "flagged_text": "the exact sentence from the draft",
      "reason": "why this is a problem",
      "source_file": "which reference file supports this",
      "source_section": "which section in that file"
    }}
  ]
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


def review_main_section(main_heading: str, sub_sections: list) -> list[dict]:
    combined_text = "\n\n".join(
        f"{s.sub_heading}: {s.text}" if s.sub_heading else s.text
        for s in sub_sections
    )

    context = retrieve_context(f"{main_heading}: {combined_text}")
    prompt = build_prompt(main_heading, combined_text, context)
    raw_result = call_llm_with_retry(prompt)

    return raw_result.get("issues", [])


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def deduplicate_issues(raw_issues: list[dict]) -> list[dict]:
    """
    Multiple retrieved reference chunks can independently confirm the same
    problem, which causes the LLM to report the same flagged sentence more
    than once (each time citing a different source). We keep only the first
    occurrence of each distinct flagged_text, so one real problem = one
    reported issue.
    """
    seen = set()
    deduped = []
    for issue in raw_issues:
        key = normalize_text(issue.get("flagged_text", ""))
        if key and key not in seen:
            seen.add(key)
            deduped.append(issue)
    return deduped


def review_draft(draft_path: str) -> ReviewResult:
    sections = parse_pdf_into_sections(draft_path)
    grouped = group_by_main_heading(sections)

    all_issues = []
    for main_heading, sub_sections in grouped.items():
        all_issues.extend(review_main_section(main_heading, sub_sections))

    deduped_issues = deduplicate_issues(all_issues)

    for i, issue in enumerate(deduped_issues, start=1):
        issue["issue_id"] = f"issue_{i}"

    try:
        validated_issues = [Issue(**issue) for issue in deduped_issues]
    except ValidationError as e:
        raise ValueError(f"LLM returned malformed issue JSON: {e}")

    status = "needs_revision" if validated_issues else "pass"
    summary = (
        f"Found {len(validated_issues)} issue(s) across {len(grouped)} sections."
        if validated_issues
        else "No issues found. Draft is consistent with reference material."
    )

    return ReviewResult(status=status, issues=validated_issues, summary=summary)


if __name__ == "__main__":
    result = review_draft("data/drafts_to_review/homepage_and_product_overview.pdf")
    print(result.model_dump_json(indent=2))
    print(f"\nTotal issues found: {len(result.issues)}")