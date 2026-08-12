AI Document QA Reviewer

An AI-powered QA reviewer that checks content drafts against approved product documentation, compliance policies, and writing guidelines using a multi-source RAG pipeline. Built as a capstone project for the Visionerds AI Engineering Internship.

What it does:

Given a content draft (e.g. a homepage, help article, or pricing page), the system:

Parses the draft into sections
Retrieves the relevant product facts and writing/compliance rules for each section
Uses an LLM to compare the draft against that reference material
Returns a structured, validated list of issues (factual errors, unsupported claims, writing violations) with severity, reasoning, and source citations
Persists the review in SQLite so follow-up questions ("why was issue 3 flagged?") can be answered from the saved data
Supports natural conversation — review requests, follow-up questions, and general chat are routed to the right handler automatically

Architecture:
Draft submitted
      ↓
Extract claims (section by section)
      ↓
Retrieve matching source ← Product docs (facts) + Policies (rules)
      ↓
Compare claim vs source
      ↓
  Match → no flag        Mismatch → flag issue (with severity)
      ↓                              ↓
      └──────→ Save review (SQLite, issue_id) ←──────┘
                        ↓
                Follow-up Q&A (by issue_id)

Reference documents are chunked using font-based section detection (not fixed-size character splitting), embedded with sentence-transformers, and stored in a persistent Chroma vector database. Each chunk is tagged with a source_type (product fact, writing rule, compliance rule, QA rubric, approved example) to keep facts and rules distinguishable during retrieval.

Tech stack:
LLM: Groq (llama-3.3-70b-versatile)
Retrieval: ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
Backend: FastAPI + Pydantic
Persistence: SQLite
PDF parsing: PyMuPDF (font-based section chunking)
Testing: pytest (unit + mocked LLM tests, one live integration test)
CI: GitHub Actions
Project structure
app/
├── pdf_parser.py     # font-based section chunking for PDFs
├── metadata.py        # tags reference files with source_type
├── ingest.py           # builds the Chroma vector index
├── review_engine.py    # per-section retrieval + LLM review + dedup + validation
├── db.py                # SQLite schema and queries
├── memory.py            # per-session conversation memory
├── router.py             # classifies messages into review / tool / chat paths
├── tools.py               # explain_issue, find_rule, check_claim, load_review
└── main.py                 # FastAPI app (/chat, /reviews/{id}, /reviews/{id}/issues)

data/
├── reference_docs/    # product manual, plans, admin guide, writing guide, compliance rules, QA rubric
└── drafts_to_review/  # sample drafts with planted factual errors and unsupported claims

tests/
└── test_review_engine.py

.github/workflows/ci.yml
Setup
bash
git clone https://github.com/mahamazhar112/AI-Document-QA-Reviewer.git
cd AI-Document-QA-Reviewer

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
Running the project

1. Build the reference index (run once, or whenever reference docs change):

bash
python -m app.ingest

2. Start the API:

bash
uvicorn app.main:app --reload

3. Open the interactive docs:

http://127.0.0.1:8000/docs
API endpoints
Endpoint	Description
POST /chat	Main entrypoint — send a session_id, message, and optional document_path to review a draft, ask a follow-up question, or chat normally
GET /reviews/{review_id}	Fetch a full review with all its issues
GET /reviews/{review_id}/issues	Fetch just the issues for a review

Example — review a draft:

json
POST /chat
{
  "session_id": "demo-1",
  "message": "review this draft",
  "document_path": "data/drafts_to_review/homepage_and_product_overview.pdf"
}

Example — follow-up question:

json
POST /chat
{
  "session_id": "demo-1",
  "message": "why was issue_3 flagged?"
}

The system automatically links this to the last review in that session — no need to repeat the review ID.

Example — ask about a rule directly:

json
POST /chat
{
  "session_id": "demo-1",
  "message": "what does the rule say about security wording?"
}
Running tests
bash
pytest tests/ -v

The suite includes fast unit tests (deduplication, section grouping, Pydantic validation), mocked LLM tests (JSON parsing, retry logic), and one live integration test that is automatically skipped if no GROQ_API_KEY is available or the Groq rate limit has been reached.

CI

Every push runs the test suite automatically via GitHub Actions (.github/workflows/ci.yml).

Known limitations:
Conversation memory is in-process and resets if the server restarts (reviews themselves are safely persisted in SQLite)
No deployment/containerization — runs locally via uvicorn
Follow-up questions are resolved against the most recent review in a session, not a specific review chosen by the user unless stated
Near-duplicate issues are deduplicated by exact flagged-text match; semantically similar but differently-worded duplicates may still appear separately
