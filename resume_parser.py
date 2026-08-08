"""
===============================================================
TalentMatch AI
Resume Parser

Responsibilities:
• Parse PDF resumes using pypdf
• Parse DOCX resumes using python-docx
• Parse TXT resumes
• Validate file types
• Clean extracted text
• Provide useful errors
• Designed for Render Free / 512 MB RAM

Supported:
    PDF
    DOCX
    TXT

PDF engine:
    pypdf

DOCX engine:
    python-docx

No PyMuPDF dependency required.
===============================================================
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader
from docx import Document


# =============================================================
# Custom Exception
# =============================================================

class ResumeParsingError(Exception):
    """
    Raised when a resume cannot be parsed.
    """
    pass


# =============================================================
# Supported Extensions
# =============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


# =============================================================
# Text Cleaning
# =============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted resume text.

    Removes:
    • Excess whitespace
    • Repeated blank lines
    • Null characters
    • Unnecessary spacing

    Returns:
        Clean readable text.
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

    # Remove excessive spaces/tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    # Strip whitespace from each line
    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if line:
            lines.append(line)

    text = "\n".join(lines)

    return text.strip()


# =============================================================
# PDF Parser
# =============================================================

def parse_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF using pypdf.

    Parameters:
        file_bytes:
            Raw PDF bytes.

    Returns:
        Extracted text.

    Raises:
        ResumeParsingError
    """

    if not file_bytes:
        raise ResumeParsingError(
            "The PDF file is empty."
        )

    try:

        # -----------------------------------------------------
        # Load PDF from memory
        # -----------------------------------------------------

        pdf_stream = io.BytesIO(
            file_bytes
        )

        reader = PdfReader(
            pdf_stream
        )

        # -----------------------------------------------------
        # Validate PDF
        # -----------------------------------------------------

        if not reader.pages:

            raise ResumeParsingError(
                "The PDF does not contain any pages."
            )

        extracted_pages = []

        # -----------------------------------------------------
        # Extract each page
        # -----------------------------------------------------

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                page_text = page.extract_text()

                if page_text:

                    extracted_pages.append(
                        page_text
                    )

            except Exception as exc:

                # Continue processing other pages
                print(
                    f"Warning: Could not extract "
                    f"text from PDF page "
                    f"{page_number}: {exc}"
                )

        # -----------------------------------------------------
        # Combine pages
        # -----------------------------------------------------

        text = "\n\n".join(
            extracted_pages
        )

        text = clean_text(
            text
        )

        # -----------------------------------------------------
        # Check whether text was extracted
        # -----------------------------------------------------

        if not text:

            raise ResumeParsingError(
                "No readable text could be extracted "
                "from this PDF. The PDF may be scanned "
                "or image-based. Please upload a text-based "
                "PDF, DOCX or TXT resume."
            )

        return text

    except ResumeParsingError:

        raise

    except Exception as exc:

        print(
            "PDF parsing error:",
            repr(exc)
        )

        raise ResumeParsingError(
            "Unable to read the PDF resume. "
            "Please make sure the PDF is valid and "
            "not password protected."
        ) from exc

    finally:

        try:
            pdf_stream.close()
        except Exception:
            pass


# =============================================================
# DOCX Parser
# =============================================================

def parse_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX resume.

    Includes:
    • Paragraphs
    • Tables

    Parameters:
        file_bytes:
            Raw DOCX bytes.

    Returns:
        Extracted text.
    """

    if not file_bytes:

        raise ResumeParsingError(
            "The DOCX file is empty."
        )

    try:

        document_stream = io.BytesIO(
            file_bytes
        )

        document = Document(
            document_stream
        )

        parts = []

        # -----------------------------------------------------
        # Paragraphs
        # -----------------------------------------------------

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                parts.append(
                    text
                )

        # -----------------------------------------------------
        # Tables
        # -----------------------------------------------------

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

        return text

    except ResumeParsingError:

        raise

    except Exception as exc:

        print(
            "DOCX parsing error:",
            repr(exc)
        )

        raise ResumeParsingError(
            "Unable to read the DOCX resume. "
            "Please make sure the document is valid."
        ) from exc


# =============================================================
# TXT Parser
# =============================================================

def parse_txt(file_bytes: bytes) -> str:
    """
    Extract text from a TXT resume.

    Attempts UTF-8 first, then common fallbacks.
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

    return text


# =============================================================
# Main Resume Parser
# =============================================================

def parse_resume(
    file_bytes: bytes,
    filename: str,
) -> str:
    """
    Automatically determine the resume format
    and extract its text.

    Parameters:
        file_bytes:
            Uploaded file bytes.

        filename:
            Original uploaded filename.

    Returns:
        Clean resume text.

    Raises:
        ResumeParsingError
    """

    if not file_bytes:

        raise ResumeParsingError(
            "The uploaded resume is empty."
        )

    if not filename:

        raise ResumeParsingError(
            "The uploaded file has no filename."
        )

    # ---------------------------------------------------------
    # Determine extension
    # ---------------------------------------------------------

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    # ---------------------------------------------------------
    # Validate extension
    # ---------------------------------------------------------

    if extension not in SUPPORTED_EXTENSIONS:

        raise ResumeParsingError(
            "Unsupported resume format. "
            "Please upload a PDF, DOCX or TXT file."
        )

    # ---------------------------------------------------------
    # Route to parser
    # ---------------------------------------------------------

    if extension == ".pdf":

        return parse_pdf(
            file_bytes
        )

    if extension == ".docx":

        return parse_docx(
            file_bytes
        )

    if extension == ".txt":

        return parse_txt(
            file_bytes
        )

    # ---------------------------------------------------------
    # Safety fallback
    # ---------------------------------------------------------

    raise ResumeParsingError(
        "Unable to determine the resume file format."
    )


# =============================================================
# Parser Status
# =============================================================

def parser_status():
    """
    Return parser availability information.

    Useful for diagnostics.
    """

    return {
        "status": "ready",
        "pdf_engine": "pypdf",
        "docx_engine": "python-docx",
        "txt_engine": "built-in Python",
        "supported_formats": [
            "PDF",
            "DOCX",
            "TXT",
        ],
    }
