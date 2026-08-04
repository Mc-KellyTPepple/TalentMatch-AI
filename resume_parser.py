"""
TalentMatch AI
Resume Parser

Supports:
- PDF (.pdf)
- Microsoft Word (.docx)

Extracts clean text for downstream NLP processing.
"""

from pathlib import Path
from typing import Union

import fitz  # PyMuPDF
from docx import Document

from config import ALLOWED_EXTENSIONS


class ResumeParser:
    """
    Parse PDF and DOCX resumes into plain text.
    """

    def __init__(self):
        pass

    # --------------------------------------------------

    @staticmethod
    def validate_file(file_path: Union[str, Path]) -> Path:
        """
        Validate file existence and extension.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}"
            )

        return file_path

    # --------------------------------------------------

    @staticmethod
    def parse_pdf(file_path: Path) -> str:
        """
        Extract text from PDF.
        """

        text = []

        document = fitz.open(file_path)

        for page in document:

            page_text = page.get_text("text")

            if page_text:

                text.append(page_text)

        document.close()

        return "\n".join(text)

    # --------------------------------------------------

    @staticmethod
    def parse_docx(file_path: Path) -> str:
        """
        Extract text from DOCX.
        """

        document = Document(file_path)

        paragraphs = [

            paragraph.text

            for paragraph in document.paragraphs

            if paragraph.text.strip()

        ]

        return "\n".join(paragraphs)

    # --------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize whitespace.
        """

        lines = [

            line.strip()

            for line in text.splitlines()

            if line.strip()

        ]

        return "\n".join(lines)

    # --------------------------------------------------

    def parse(self, file_path: Union[str, Path]) -> str:
        """
        Main parser entry point.
        """

        file_path = self.validate_file(file_path)

        if file_path.suffix.lower() == ".pdf":

            text = self.parse_pdf(file_path)

        else:

            text = self.parse_docx(file_path)

        return self.clean_text(text)


if __name__ == "__main__":

    parser = ResumeParser()

    resume_text = parser.parse("sample_resume.pdf")

    print(resume_text[:1000])
