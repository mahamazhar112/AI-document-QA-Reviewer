import json
import os

from groq import Groq
from dotenv import load_dotenv

from app import db
from app.review_engine import retrieve_context, call_llm_with_retry

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TOOL_CLASSIFY_PROMPT = """Classify this follow-up message into one tool call and extract its parameters.

Tools:
- "explain_issue": user asks why an issue was flagged or wants its source (needs issue_id, e.g. "issue_3")
- "find_rule": user asks what the rule/policy says about a topic (needs a short topic string)
- "check_claim": user gives a specific claim and wants it checked against reference material (needs the claim text)
- "load_review": user wants to see a full past review (needs review_id if mentioned, else null)

Message: "{message}"

Respond with ONLY valid JSON, no other text:
{{"tool": "explain_issue" or "find_rule" or "check_claim" or "load_review", "issue_id": "issue_N or null", "query": "topic or claim text or null", "review_id": number or null}}"""


def classify_tool_call(message: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": TOOL_CLASSIFY_PROMPT.format(message=message)}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"tool": None}


def explain_issue(review_id: int, issue_id: str) -> dict:
    issue = db.get_issue_by_id(review_id, issue_id)
    if not issue:
        return {"found": False, "message": f"No issue '{issue_id}' found for review {review_id}."}

    return {
        "found": True,
        "issue_id": issue["issue_id"],
        "flagged_text": issue["flagged_text"],
        "reason": issue["reason"],
        "source_file": issue["source_file"],
        "source_section": issue["source_section"],
        "severity": issue["severity"],
    }


def find_rule(topic: str) -> dict:
    context = retrieve_context(topic, top_k=3)
    return {"topic": topic, "matches": context}


def check_claim(claim_text: str) -> dict:
    context = retrieve_context(claim_text, top_k=4)
    context_block = "\n\n".join(f"[{c['source_file']} - {c['heading']}]\n{c['text']}" for c in context)

    prompt = f"""Check this claim against the reference material. Say whether it is supported, contradicted, or unsupported.

REFERENCE MATERIAL:
{context_block}

CLAIM:
{claim_text}

Respond with ONLY valid JSON:
{{"verdict": "supported" or "contradicted" or "unsupported", "reason": "short explanation", "source_file": "file name or null"}}"""

    result = call_llm_with_retry(prompt)
    result["claim"] = claim_text
    return result


def load_review(review_id: int) -> dict:
    review = db.load_review(review_id)
    if not review:
        return {"found": False, "message": f"No review found with id {review_id}."}
    return {"found": True, **review}


def handle_tool_call(session_id: str, message: str, last_review_id: int | None) -> dict:
    parsed = classify_tool_call(message)
    tool = parsed.get("tool")

    if tool == "explain_issue":
        review_id = parsed.get("review_id") or last_review_id
        issue_id = parsed.get("issue_id")
        if not review_id or not issue_id:
            return {"tool": "explain_issue", "error": "Missing review or issue reference."}
        return {"tool": "explain_issue", "result": explain_issue(review_id, issue_id)}

    if tool == "find_rule":
        query = parsed.get("query") or message
        return {"tool": "find_rule", "result": find_rule(query)}

    if tool == "check_claim":
        query = parsed.get("query") or message
        return {"tool": "check_claim", "result": check_claim(query)}

    if tool == "load_review":
        review_id = parsed.get("review_id") or last_review_id
        if not review_id:
            return {"tool": "load_review", "error": "No review reference available."}
        return {"tool": "load_review", "result": load_review(review_id)}

    return {"tool": None, "error": "Could not understand the request."}