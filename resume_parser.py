"""
=========================================================
TalentMatch AI
Resume Parser
=========================================================

Production resume text extraction for:

    PDF
    DOCX
    TXT

Designed for:
    Render Free
    512 MB RAM

The parser processes uploaded resumes in memory and
does not permanently store candidate documents.

=========================================================
"""

from io import BytesIO
from pathlib import Path
from typing import Union

from config import (
    SUPPORTED_EXTENSIONS,
    MAX_UPLOAD_SIZE,
)


# =========================================================
# Exceptions
# =========================================================

class ResumeParsingError(Exception):
    """Raised when a resume cannot be parsed."""
    pass


# =========================================================
# File Validation
# =========================================================

def validate_resume(
    filename: str,
    file_size: int
) -> str:
    """
    Validate the uploaded resume.

    Returns:
        Normalized file extension.
    """

    if not filename:
        raise ResumeParsingError(
            "No resume filename was provided."
        )

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ResumeParsingError(
            f"Unsupported resume format. "
            f"Supported formats: {supported}"
        )

    if file_size <= 0:
        raise ResumeParsingError(
            "The uploaded resume is empty."
        )

    if file_size > MAX_UPLOAD_SIZE:
        raise ResumeParsingError(
            "Resume exceeds the maximum allowed "
            "file size of 10 MB."
        )

    return extension


# =========================================================
# Text Cleaning
# =========================================================

def clean_text(text: str) -> str:
    """
    Normalize extracted resume text.

    This reduces unnecessary whitespace while
    preserving the actual resume content.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    # Remove null characters
    text = text.replace(
        "\x00",
        ""
    )

    # Normalize spaces on each line
    lines = []

    for line in text.split("\n"):

        line = " ".join(
            line.split()
        )

        if line:
            lines.append(line)

    # Rebuild document
    text = "\n".join(lines)

    # Prevent excessive blank lines
    while "\n\n\n" in text:

        text = text.replace(
            "\n\n\n",
            "\n\n"
        )

    return text.strip()


# =========================================================
# PDF Parser
# =========================================================

def extract_pdf(
    file_bytes: bytes
) -> str:
    """
    Extract text from a PDF stored in memory.

    PyMuPDF is used because it is lightweight and
    generally faster than heavier PDF processing
    alternatives.
    """

    try:

        import fitz

    except ImportError as exc:

        raise ResumeParsingError(
            "PDF support requires PyMuPDF."
        ) from exc

    try:

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        pages = []

        try:

            for page in document:

                text = page.get_text()

                if text:
                    pages.append(text)

        finally:

            document.close()

        return clean_text(
            "\n".join(pages)
        )

    except Exception as exc:

        raise ResumeParsingError(
            f"Unable to read PDF resume: {exc}"
        ) from exc


# =========================================================
# DOCX Parser
# =========================================================

def extract_docx(
    file_bytes: bytes
) -> str:
    """
    Extract text from a DOCX resume.

    Includes both paragraphs and table content,
    because many professional resumes use tables
    for skills, education, or experience.
    """

    try:

        from docx import Document

    except ImportError as exc:

        raise ResumeParsingError(
            "DOCX support requires python-docx."
        ) from exc

    try:

        document = Document(
            BytesIO(file_bytes)
        )

        parts = []

        # -------------------------------------------------
        # Paragraphs
        # -------------------------------------------------

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                parts.append(text)

        # -------------------------------------------------
        # Tables
        # -------------------------------------------------

        for table in document.tables:

            for row in table.rows:

                cells = []

                for cell in row.cells:

                    text = cell.text.strip()

                    if text:
                        cells.append(text)

                if cells:

                    parts.append(
                        " | ".join(cells)
                    )

        return clean_text(
            "\n".join(parts)
        )

    except Exception as exc:

        raise ResumeParsingError(
            f"Unable to read DOCX resume: {exc}"
        ) from exc


# =========================================================
# TXT Parser
# =========================================================

def extract_txt(
    file_bytes: bytes
) -> str:
    """
    Extract text from a plain-text resume.

    Attempts UTF-8 first and then falls back to
    common encodings.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
    ]

    for encoding in encodings:

        try:

            return clean_text(
                file_bytes.decode(
                    encoding
                )
            )

        except UnicodeDecodeError:

            continue

    raise ResumeParsingError(
        "Unable to decode the text resume."
    )


# =========================================================
# Main Parser
# =========================================================

def parse_resume(
    file_bytes: bytes,
    filename: str
) -> str:
    """
    Main resume parsing interface.

    Parameters
    ----------
    file_bytes:
        Uploaded resume contents.

    filename:
        Original uploaded filename.

    Returns
    -------
    str
        Cleaned resume text.
    """

    if not isinstance(
        file_bytes,
        bytes
    ):

        raise ResumeParsingError(
            "Resume data must be provided as bytes."
        )

    extension = validate_resume(
        filename=filename,
        file_size=len(file_bytes)
    )

    # -----------------------------------------------------
    # Select parser
    # -----------------------------------------------------

    if extension == ".pdf":

        text = extract_pdf(
            file_bytes
        )

    elif extension == ".docx":

        text = extract_docx(
            file_bytes
        )

    elif extension == ".txt":

        text = extract_txt(
            file_bytes
        )

    else:

        raise ResumeParsingError(
            f"Unsupported extension: {extension}"
        )

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    text = clean_text(text)

    if not text:

        raise ResumeParsingError(
            "No readable text could be extracted "
            "from the resume."
        )

    # Prevent accidental processing of extremely
    # large extracted documents.
    #
    # This protects the small Render instance from
    # pathological input files.
    MAX_TEXT_LENGTH = 200_000

    if len(text) > MAX_TEXT_LENGTH:

        text = text[:MAX_TEXT_LENGTH]

    return text


# =========================================================
# Resume Information
# =========================================================

def get_resume_info(
    filename: str,
    text: str
) -> dict:
    """
    Generate lightweight metadata about a parsed resume.

    No sensitive personal information is extracted here.
    """

    return {
        "filename": Path(
            filename
        ).name,

        "format": Path(
            filename
        ).suffix.lower(),

        "characters": len(text),

        "words": len(
            text.split()
        ),
    }
