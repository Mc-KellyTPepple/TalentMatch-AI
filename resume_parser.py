"""
===============================================================
TalentMatch AI
Resume Parser

Supports:
• PDF
• DOCX
• TXT

Designed for:
• FastAPI
• Render Free
• 512 MB RAM
• In-memory processing
• No permanent resume storage
===============================================================
"""

from io import BytesIO
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
# Clean Text
# =============================================================

def clean_text(text):
    """
    Normalize extracted resume text.
    """

    if text is None:
        return ""

    text = str(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Remove excessive spaces
    lines = []

    for line in text.split("\n"):

        line = " ".join(
            line.strip().split()
        )

        if line:
            lines.append(line)

    return "\n".join(lines)


# =============================================================
# PDF Parser
# =============================================================

def parse_pdf(file_bytes):
    """
    Extract text from a PDF resume.

    Uses pypdf and processes the document
    entirely in memory.
    """

    try:

        if not file_bytes:
            raise ResumeParsingError(
                "The PDF file is empty."
            )

        pdf_stream = BytesIO(
            file_bytes
        )

        reader = PdfReader(
            pdf_stream
        )

        if reader.is_encrypted:

            try:
                reader.decrypt("")
            except Exception:

                raise ResumeParsingError(
                    "The PDF is password protected. "
                    "Please upload an unlocked PDF."
                )

        pages_text = []

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:
                    pages_text.append(
                        text
                    )

            except Exception as exc:

                print(
                    "PDF page extraction warning:",
                    repr(exc)
                )

        text = "\n".join(
            pages_text
        )

        text = clean_text(
            text
        )

        if not text:

            raise ResumeParsingError(
                "No readable text was found in the PDF. "
                "Please upload a text-based PDF rather than "
                "a scanned image-only PDF."
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
            "Please make sure the PDF is valid and not corrupted."
        ) from exc


# =============================================================
# DOCX Parser
# =============================================================

def parse_docx(file_bytes):
    """
    Extract text from a DOCX resume.
    """

    try:

        if not file_bytes:
            raise ResumeParsingError(
                "The DOCX file is empty."
            )

        document = Document(
            BytesIO(file_bytes)
        )

        paragraphs = []

        # -----------------------------------------------------
        # Paragraphs
        # -----------------------------------------------------

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(
                    text
                )

        # -----------------------------------------------------
        # Tables
        #
        # Many CVs store important information
        # inside tables.
        # -----------------------------------------------------

        for table in document.tables:

            for row in table.rows:

                cells = []

                for cell in row.cells:

                    text = cell.text.strip()

                    if text:
                        cells.append(
                            text
                        )

                if cells:

                    paragraphs.append(
                        " | ".join(cells)
                    )

        text = "\n".join(
            paragraphs
        )

        text = clean_text(
            text
        )

        if not text:

            raise ResumeParsingError(
                "No readable text was found in the DOCX resume."
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

def parse_txt(file_bytes):
    """
    Extract text from a TXT resume.
    """

    try:

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
                "No readable text was found in the TXT resume."
            )

        return text

    except ResumeParsingError:
        raise

    except Exception as exc:

        print(
            "TXT parsing error:",
            repr(exc)
        )

        raise ResumeParsingError(
            "Unable to read the TXT resume."
        ) from exc


# =============================================================
# Main Resume Parser
# =============================================================

def parse_resume(
    file_bytes,
    filename
):
    """
    Automatically select the correct parser.

    Supported:

        .pdf
        .docx
        .txt
    """

    if not filename:

        raise ResumeParsingError(
            "Resume filename is missing."
        )

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    # =========================================================
    # PDF
    # =========================================================

    if extension == ".pdf":

        return parse_pdf(
            file_bytes
        )

    # =========================================================
    # DOCX
    # =========================================================

    if extension == ".docx":

        return parse_docx(
            file_bytes
        )

    # =========================================================
    # TXT
    # =========================================================

    if extension == ".txt":

        return parse_txt(
            file_bytes
        )

    # =========================================================
    # Unsupported
    # =========================================================

    raise ResumeParsingError(
        "Unsupported resume format. "
        "Please upload a PDF, DOCX or TXT file."
    )
