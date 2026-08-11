"""
pdf_parser.py

Parses our reference and draft PDFs into clean chunks using font info,
instead of naive character splitting.

Every PDF in this dataset uses a consistent style:
- ~8.0pt regular   -> repeated page header/footer line, e.g. "NovaFlow CX
                       Product Manual | 3"  -> we skip this
- ~14.5pt bold      -> main section heading, e.g. "6. Reporting"
- ~11.2pt bold      -> sub-heading within a section, e.g. "Core jobs",
                       "Example in practice", "Review notes"
- ~9.2 / 9.1pt reg. -> body text and bullet points

A chunk = one (main_heading, sub_heading) pair + the body text under it.
This keeps each chunk focused on one real idea, which makes retrieval much
more precise than fixed-size character chunks.
"""

import fitz  # pymupdf
from pathlib import Path
from dataclasses import dataclass

MAIN_HEADING_MIN_SIZE = 14.0   # >= this and bold -> main section heading
SUB_HEADING_MIN_SIZE = 10.5    # >= this and bold (but below main) -> sub-heading
PAGE_LABEL_MAX_SIZE = 8.5      # small repeated header/footer line -> skip


@dataclass
class Section:
    main_heading: str
    sub_heading: str
    text: str
    doc_title: str
    source_file: str


def parse_pdf_into_sections(pdf_path: str) -> list[Section]:
    doc = fitz.open(pdf_path)
    doc_title = None
    sections: list[Section] = []

    current_main = None
    current_sub = None
    current_lines: list[str] = []

    def flush():
        if current_main and current_lines:
            body = " ".join(current_lines).strip()
            if body:
                sections.append(
                    Section(
                        main_heading=current_main,
                        sub_heading=current_sub or "",
                        text=body,
                        doc_title=doc_title or "",
                        source_file=Path(pdf_path).name,
                    )
                )

    for page in doc:
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue

                spans = line["spans"]
                size = spans[0]["size"] if spans else 0
                is_bold = spans and "Bold" in spans[0]["font"]

                # skip the repeated page-label line entirely
                if size <= PAGE_LABEL_MAX_SIZE:
                    continue

                if is_bold and size >= MAIN_HEADING_MIN_SIZE:
                    # first big bold line in the whole doc is the title, not
                    # a section -- capture it once, then treat the rest as
                    # real main-section headings
                    if current_main is None and doc_title is None:
                        doc_title = line_text
                        continue
                    flush()
                    current_main = line_text
                    current_sub = None
                    current_lines = []
                    continue

                if is_bold and SUB_HEADING_MIN_SIZE <= size < MAIN_HEADING_MIN_SIZE:
                    flush()
                    current_sub = line_text
                    current_lines = []
                    continue

                # regular body text -> belongs to current (main, sub) section
                current_lines.append(line_text)

    flush()
    doc.close()
    return sections


def parse_folder(folder_path: str) -> list[Section]:
    """Parses every PDF in a folder (non-recursive) and returns all sections."""
    all_sections: list[Section] = []
    for pdf_file in sorted(Path(folder_path).glob("*.pdf")):
        all_sections.extend(parse_pdf_into_sections(str(pdf_file)))
    return all_sections