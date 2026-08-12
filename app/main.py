from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.db import init_db, load_review
from app.router import route_message

app = FastAPI(title="AI Document QA Reviewer")

init_db()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_path: str | None = None


class ChatResponse(BaseModel):
    path: str
    data: dict


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = route_message(request.session_id, request.message, request.document_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    path = result.get("path", "chat")
    return ChatResponse(path=path, data=result)


@app.get("/reviews/{review_id}")
def get_review(review_id: int):
    review = load_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found.")
    return review


@app.get("/reviews/{review_id}/issues")
def get_review_issues(review_id: int):
    review = load_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found.")
    return {"review_id": review_id, "issues": review["issues"]}


@app.get("/")
def root():
    return {"status": "ok", "service": "AI Document QA Reviewer"}