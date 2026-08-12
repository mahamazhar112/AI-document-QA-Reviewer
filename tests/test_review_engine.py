import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from app.review_engine import (
    normalize_text,
    deduplicate_issues,
    group_by_main_heading,
    call_llm_with_retry,
    Issue,
    ReviewResult,
)
from app.pdf_parser import Section


# ---------- normalize_text ----------

def test_normalize_text_lowercases_and_trims_whitespace():
    assert normalize_text("  Hello   World  ") == "hello world"


def test_normalize_text_treats_case_variants_as_equal():
    assert normalize_text("Growth Plan") == normalize_text("growth plan")


# ---------- deduplicate_issues ----------

def test_deduplicate_issues_removes_exact_duplicates():
    issues = [
        {"flagged_text": "All exports are instant.", "reason": "A"},
        {"flagged_text": "All exports are instant.", "reason": "B"},
        {"flagged_text": "All exports are instant.", "reason": "C"},
    ]
    result = deduplicate_issues(issues)
    assert len(result) == 1
    assert result[0]["reason"] == "A"  # keeps first occurrence


def test_deduplicate_issues_ignores_case_and_whitespace_differences():
    issues = [
        {"flagged_text": "Unlimited reporting history.", "reason": "A"},
        {"flagged_text": "  unlimited   reporting history.  ", "reason": "B"},
    ]
    result = deduplicate_issues(issues)
    assert len(result) == 1


def test_deduplicate_issues_keeps_genuinely_distinct_issues():
    issues = [
        {"flagged_text": "Growth includes SAML SSO.", "reason": "A"},
        {"flagged_text": "Enterprise has unlimited retention.", "reason": "B"},
    ]
    result = deduplicate_issues(issues)
    assert len(result) == 2


def test_deduplicate_issues_handles_empty_list():
    assert deduplicate_issues([]) == []


# ---------- group_by_main_heading ----------

def make_section(main_heading, sub_heading="", text="body"):
    return Section(main_heading=main_heading, sub_heading=sub_heading, text=text,
                    doc_title="Test Doc", source_file="test.pdf")


def test_group_by_main_heading_skips_contents_section():
    sections = [make_section("Contents"), make_section("1. Overview")]
    grouped = group_by_main_heading(sections)
    assert "Contents" not in grouped
    assert "1. Overview" in grouped


def test_group_by_main_heading_groups_subsections_together():
    sections = [
        make_section("6. Reporting", "Core jobs"),
        make_section("6. Reporting", "Review notes"),
    ]
    grouped = group_by_main_heading(sections)
    assert len(grouped["6. Reporting"]) == 2


# ---------- Pydantic validation ----------

def test_issue_model_raises_on_missing_required_field():
    with pytest.raises(ValidationError):
        Issue(issue_id="issue_1", type="factual_error", severity="high")
        # missing flagged_text, reason, source_file, source_section


def test_review_result_accepts_empty_issue_list():
    result = ReviewResult(status="pass", issues=[], summary="No issues found.")
    assert result.status == "pass"
    assert result.issues == []


# ---------- call_llm_with_retry (mocked, no real API calls) ----------

def _mock_groq_response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    return mock_response


@patch("app.review_engine.client.chat.completions.create")
def test_call_llm_with_retry_parses_valid_json(mock_create):
    mock_create.return_value = _mock_groq_response('{"issues": []}')
    result = call_llm_with_retry("any prompt")
    assert result == {"issues": []}
    assert mock_create.call_count == 1


@patch("app.review_engine.client.chat.completions.create")
def test_call_llm_with_retry_strips_markdown_code_fences(mock_create):
    mock_create.return_value = _mock_groq_response('```json\n{"issues": []}\n```')
    result = call_llm_with_retry("any prompt")
    assert result == {"issues": []}


@patch("app.review_engine.client.chat.completions.create")
def test_call_llm_with_retry_retries_then_succeeds_on_malformed_json(mock_create):
    mock_create.side_effect = [
        _mock_groq_response("this is not valid json"),
        _mock_groq_response('{"issues": []}'),
    ]
    result = call_llm_with_retry("any prompt", max_retries=2)
    assert result == {"issues": []}
    assert mock_create.call_count == 2


@patch("app.review_engine.client.chat.completions.create")
def test_call_llm_with_retry_raises_after_exhausting_retries(mock_create):
    mock_create.return_value = _mock_groq_response("still not json")
    with pytest.raises(Exception):
        call_llm_with_retry("any prompt", max_retries=1)


# ---------- Real integration test (skipped if no API key, or if rate-limited) ----------

@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="No GROQ_API_KEY set")
def test_review_draft_end_to_end_on_real_draft():
    from app.review_engine import review_draft
    from groq import RateLimitError

    try:
        result = review_draft("data/drafts_to_review/homepage_and_product_overview.pdf")
    except RateLimitError:
        pytest.skip("Groq daily rate limit reached — skipping live integration test")

    assert result.status in ("pass", "needs_revision")
    assert isinstance(result.issues, list)