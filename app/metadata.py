from pathlib import Path

FACT_FILES = {
    "product_manual",
    "plans_limits_and_entitlements",
    "admin_and_configuration_guide",
    "reporting_data_and_integrations",
    "product_release_notes",
}

SOURCE_TYPE_MAP = {
    "brand_and_writing_guide": "writing_rule",
    "claims_compliance_and_security_rules": "compliance_rule",
    "content_qa_rubric": "qa_rubric",
    "approved_content_examples": "approved_example",
}


def infer_source_type(filename: str) -> str:
    stem = Path(filename).stem.lower()

    if stem in FACT_FILES:
        return "product_fact"

    if stem in SOURCE_TYPE_MAP:
        return SOURCE_TYPE_MAP[stem]

    return "unknown"