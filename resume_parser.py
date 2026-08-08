"""
TalentMatch AI
Resume Parser

Supported:
    PDF
    DOCX
    TXT

PDF engine:
    pypdf

DOCX engine:
    python-docx

IMPORTANT:
    No PyMuPDF / fitz is required.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader
from docx import Document


# ============================================================
# Custom Exception
# ============================================================

class ResumeParsingError(Exception):
    """Raised when a resume cannot be parsed."""

    pass


# ============================================================
# Supported Extensions
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted resume text.
    """

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize non-breaking spaces
    text = text.replace("\xa0", " ")

    # Normalize spaces and tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text,
    )

    # Clean individual lines
    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


# ============================================================
# PDF Parser
# ============================================================

def parse_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF using pypdf.

    No PyMuPDF / fitz is used.
    """

    if not file_bytes:

        raise ResumeParsingError(
            "The PDF file is empty."
        )

    pdf_stream = None

    try:

        # ----------------------------------------------------
        # Confirm PDF signature
        # ----------------------------------------------------

        if not file_bytes.startswith(b"%PDF"):

            raise ResumeParsingError(
                "The uploaded file does not appear to be "
                "a valid PDF."
            )

        # ----------------------------------------------------
        # Load PDF into memory
        # ----------------------------------------------------

        pdf_stream = io.BytesIO(
            file_bytes
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # strict=False allows pypdf to handle many PDFs
        # that contain minor structural issues.
        # ----------------------------------------------------

        reader = PdfReader(
            pdf_stream,
            strict=False,
        )

        # ----------------------------------------------------
        # Check encryption
        # ----------------------------------------------------

        if reader.is_encrypted:

            raise ResumeParsingError(
                "This PDF is password protected. "
                "Please upload an unlocked PDF resume."
            )

        # ----------------------------------------------------
        # Check pages
        # ----------------------------------------------------

        page_count = len(
            reader.pages
        )

        if page_count == 0:

            raise ResumeParsingError(
                "The PDF does not contain any pages."
            )

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        extracted_pages = []

        successful_pages = 0

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:

                page_text = page.extract_text()

                if page_text:

                    page_text = clean_text(
                        page_text
                    )

                    if page_text:

                        extracted_pages.append(
                            page_text
                        )

                        successful_pages += 1

            except Exception as exc:

                print(
                    f"[PDF] Page {page_number} "
                    f"extraction warning: {repr(exc)}"
                )

                continue

        # ----------------------------------------------------
        # Combine extracted pages
        # ----------------------------------------------------

        text = "\n\n".join(
            extracted_pages
        )

        text = clean_text(
            text
        )

        # ----------------------------------------------------
        # No text
        # ----------------------------------------------------

        if not text:

            raise ResumeParsingError(
                "No readable text could be extracted "
                "from this PDF. This usually means the "
                "PDF is scanned or image-based. "
                "Please upload a text-based PDF, DOCX "
                "or TXT resume."
            )

        print(
            f"[PDF] Successfully parsed "
            f"{successful_pages}/{page_count} pages."
        )

        print(
            f"[PDF] Extracted {len(text)} characters."
        )

        return text

    except ResumeParsingError:

        raise

    except Exception as exc:

        print(
            "[PDF] Parser error:"
        )

        print(
            repr(exc)
        )

        raise ResumeParsingError(
            "Unable to read this PDF resume. "
            "The PDF may be corrupted, malformed, "
            "or protected."
        ) from exc

    finally:

        if pdf_stream is not None:

            try:
                pdf_stream.close()

            except Exception:
                pass


# ============================================================
# DOCX Parser
# ============================================================

def parse_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX resume.

    Includes paragraphs and tables.
    """

    if not file_bytes:

        raise ResumeParsingError(
            "The DOCX file is empty."
        )

    document_stream = None

    try:

        document_stream = io.BytesIO(
            file_bytes
        )

        document = Document(
            document_stream
        )

        parts = []

        # ----------------------------------------------------
        # Paragraphs
        # ----------------------------------------------------

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                parts.append(
                    text
                )

        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------

        for table in document.tables:

            for row in table.rows:

                cells = []

                for cell in row.cells:

                    cell_text = (
                        cell.text.strip()
                    )

                    if cell_text:

                        cells.append(
                            cell_text
                        )

                if cells:

                    parts.append(
                        " | ".join(cells)
                    )

        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        text = "\n".join(
            parts
        )

        text = clean_text(
            text
        )

        if not text:

            raise ResumeParsingError(
                "No readable text could be extracted "
                "from this DOCX resume."
            )

        print(
            f"[DOCX] Extracted {len(text)} characters."
        )

        return text

    except ResumeParsingError:

        raise

    except Exception as exc:

        print(
            "[DOCX] Parser error:",
            repr(exc),
        )

        raise ResumeParsingError(
            "Unable to read the DOCX resume. "
            "Please make sure the document is valid."
        ) from exc

    finally:

        if document_stream is not None:

            try:
                document_stream.close()

            except Exception:
                pass


# ============================================================
# TXT Parser
# ============================================================

def parse_txt(file_bytes: bytes) -> str:
    """
    Extract text from a TXT resume.
    """

    if not file_bytes:

        raise ResumeParsingError(
            "The TXT file is empty."
        )

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ]

    text = None

    for encoding in encodings:

        try:

            text = file_bytes.decode(
                encoding
            )

            break

        except UnicodeDecodeError:

            continue

    if text is None:

        raise ResumeParsingError(
            "Unable to decode the TXT resume."
        )

    text = clean_text(
        text
    )

    if not text:

        raise ResumeParsingError(
            "The TXT resume contains no readable text."
        )

    print(
        f"[TXT] Extracted {len(text)} characters."
    )

    return text


# ============================================================
# Main Resume Parser
# ============================================================

def parse_resume(
    file_bytes: bytes,
    filename: str,
) -> str:
    """
    Automatically determine the resume format
    and extract its text.
    """

    if not file_bytes:

        raise ResumeParsingError(
            "The uploaded resume is empty."
        )

    if not filename:

        raise ResumeParsingError(
            "The uploaded file has no filename."
        )

    # --------------------------------------------------------
    # Get extension
    # --------------------------------------------------------

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    print(
        f"[PARSER] Filename: {filename}"
    )

    print(
        f"[PARSER] Detected extension: {extension}"
    )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if extension not in SUPPORTED_EXTENSIONS:

        raise ResumeParsingError(
            "Unsupported resume format. "
            "Please upload a PDF, DOCX or TXT file."
        )

    # --------------------------------------------------------
    # Route parser
    # --------------------------------------------------------

    if extension == ".pdf":

        print(
            "[PARSER] Using pypdf PDF parser."
        )

        return parse_pdf(
            file_bytes
        )

    if extension == ".docx":

        print(
            "[PARSER] Using python-docx parser."
        )

        return parse_docx(
            file_bytes
        )

    if extension == ".txt":

        print(
            "[PARSER] Using built-in TXT parser."
        )

        return parse_txt(
            file_bytes
        )

    raise ResumeParsingError(
        "Unable to determine the resume file format."
    )


# ============================================================
# Parser Status
# ============================================================

def parser_status():
    """
    Return parser availability information.
    """

    return {
        "status": "ready",
        "pdf_engine": "pypdf",
        "docx_engine": "python-docx",
        "txt_engine": "built-in Python",
        "pymupdf_required": False,
        "supported_formats": [
            "PDF",
            "DOCX",
            "TXT",
        ],
    }
