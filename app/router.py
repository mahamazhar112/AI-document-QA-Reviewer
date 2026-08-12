import json
import os

from groq import Groq
from dotenv import load_dotenv

from app import db
from app.memory import memory
from app.review_engine import review_draft
from app.tools import handle_tool_call

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CLASSIFY_PROMPT = """Classify the user's message into exactly one category:

- "review": user wants a draft reviewed/checked against reference material, or is naming/uploading a document to review
- "tool": user is asking about a previous review, an issue, a source, or a rule (e.g. "why was issue 2 flagged", "show me the source", "find the rule about security wording")
- "chat": greetings, small talk, or anything not covered above

Message: "{message}"

Respond with ONLY valid JSON, no other text:
{{"path": "review" or "tool" or "chat"}}"""


def classify_intent(message: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(message=message)}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw).get("path", "chat")
    except json.JSONDecodeError:
        return "chat"


def normal_reply(session_id: str, message: str) -> str:
    history = memory.get_history(session_id)
    chat_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]
    chat_messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful QA reviewer assistant."}] + chat_messages,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def route_message(session_id: str, message: str, document_path: str | None = None) -> dict:
    memory.add_message(session_id, "user", message)
    path = classify_intent(message)

    if path == "review" and document_path:
        result = review_draft(document_path)
        review_id = db.save_review(session_id, document_path, result)
        memory.set_last_review_id(session_id, review_id)

        reply = {
            "path": "review",
            "review_id": review_id,
            "status": result.status,
            "summary": result.summary,
            "issues": [issue.model_dump() for issue in result.issues],
        }

    elif path == "tool":
        last_review_id = memory.get_last_review_id(session_id)
        reply = handle_tool_call(session_id, message, last_review_id)
        reply["path"] = "tool"

    else:
        answer = normal_reply(session_id, message)
        reply = {"path": "chat", "answer": answer}

    memory.add_message(session_id, "assistant", json.dumps(reply))
    return reply