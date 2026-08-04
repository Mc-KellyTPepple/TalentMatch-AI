"""
TalentMatch AI
Text Preprocessing Utilities

Shared preprocessing for:
    • Resumes
    • Job Descriptions
    • Interview Questions

Designed for lightweight NLP on Render Free.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

# ----------------------------------------------------
# Regular Expressions
# ----------------------------------------------------

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(\+?\d[\d\s().-]{7,}\d)"
)

MULTISPACE_PATTERN = re.compile(r"\s+")

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+)"
)

# ----------------------------------------------------

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ----------------------------------------------------

def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


# ----------------------------------------------------

def remove_emails(text: str) -> str:
    return EMAIL_PATTERN.sub(" ", text)


# ----------------------------------------------------

def remove_phone_numbers(text: str) -> str:
    return PHONE_PATTERN.sub(" ", text)


# ----------------------------------------------------

def normalize_whitespace(text: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", text).strip()


# ----------------------------------------------------

def lowercase(text: str) -> str:
    return text.lower()


# ----------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Main preprocessing pipeline.
    """

    text = normalize_unicode(text)

    text = remove_urls(text)

    text = remove_emails(text)

    text = remove_phone_numbers(text)

    text = lowercase(text)

    text = normalize_whitespace(text)

    return text


# ----------------------------------------------------

def sentence_split(text: str) -> List[str]:

    sentences = re.split(r"[.!?]", text)

    return [

        sentence.strip()

        for sentence in sentences

        if sentence.strip()

    ]


# ----------------------------------------------------

def tokenize(text: str) -> List[str]:

    return preprocess_text(text).split()


# ----------------------------------------------------

if __name__ == "__main__":

    sample = """

    John Doe

    john@email.com

    https://example.com

    Python Developer with FastAPI experience.

    """

    print(preprocess_text(sample))
