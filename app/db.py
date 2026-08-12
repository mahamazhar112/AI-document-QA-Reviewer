import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = "qa_reviewer.db"


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER NOT NULL,
                issue_id TEXT NOT NULL,
                type TEXT NOT NULL,
                severity TEXT NOT NULL,
                flagged_text TEXT NOT NULL,
                reason TEXT NOT NULL,
                source_file TEXT,
                source_section TEXT,
                FOREIGN KEY (review_id) REFERENCES reviews (id)
            )
        """)
        conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def save_review(session_id: str, document_name: str, review_result) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO reviews (session_id, document_name, status, summary) VALUES (?, ?, ?, ?)",
            (session_id, document_name, review_result.status, review_result.summary),
        )
        review_id = cursor.lastrowid

        for issue in review_result.issues:
            conn.execute(
                """INSERT INTO issues
                   (review_id, issue_id, type, severity, flagged_text, reason, source_file, source_section)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id,
                    issue.issue_id,
                    issue.type,
                    issue.severity,
                    issue.flagged_text,
                    issue.reason,
                    issue.source_file,
                    issue.source_section,
                ),
            )
        conn.commit()
        return review_id


def load_review(review_id: int) -> dict | None:
    with get_connection() as conn:
        review_row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if not review_row:
            return None

        issue_rows = conn.execute("SELECT * FROM issues WHERE review_id = ?", (review_id,)).fetchall()

        return {
            "id": review_row["id"],
            "session_id": review_row["session_id"],
            "document_name": review_row["document_name"],
            "status": review_row["status"],
            "summary": review_row["summary"],
            "created_at": review_row["created_at"],
            "issues": [dict(row) for row in issue_rows],
        }


def get_issue_by_id(review_id: int, issue_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM issues WHERE review_id = ? AND issue_id = ?",
            (review_id, issue_id),
        ).fetchone()
        return dict(row) if row else None


def list_reviews_by_session(session_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]