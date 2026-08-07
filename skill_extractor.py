"""
===============================================================
TalentMatch AI
Skill Extraction Engine

Purpose:
    Extract relevant skills from uploaded resumes using the
    skill artifacts generated during training.

Designed for:
    - Free Render deployment
    - 512 MB RAM
    - Low CPU usage
    - Fast inference
    - No model retraining
    - No large NLP libraries

Uses:
    models/
        skills.json.gz
        skill_frequency.json.gz
        synonyms.json.gz

===============================================================
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Set


# ==============================================================
# Configuration
# ==============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

SKILLS_FILE = MODELS_DIR / "skills.json.gz"
FREQUENCY_FILE = MODELS_DIR / "skill_frequency.json.gz"
SYNONYMS_FILE = MODELS_DIR / "synonyms.json.gz"


# ==============================================================
# Text normalization
# ==============================================================

def normalize_text(text: str) -> str:
    """
    Normalize resume text for efficient skill matching.

    This intentionally avoids heavyweight NLP libraries.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Normalize common separators.
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Preserve characters useful for technical skills:
    # C++, C#, .NET, Node.js, etc.
    text = re.sub(
        r"[^a-z0-9+#./&\-\s]",
        " ",
        text
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==============================================================
# Load compressed artifacts
# ==============================================================

def _load_gzip_json(path: Path, default):
    """
    Safely load a gzip-compressed JSON artifact.
    """

    if not path.exists():
        return default

    try:

        with gzip.open(
            path,
            "rt",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:
        return default


# ==============================================================
# Skill database
# ==============================================================

@lru_cache(maxsize=1)
def load_skills() -> List[str]:
    """
    Load the trained skill vocabulary.

    Cached after first load so the file is not repeatedly read
    for every resume request.
    """

    skills = _load_gzip_json(
        SKILLS_FILE,
        []
    )

    if not isinstance(skills, list):
        return []

    # Normalize and remove duplicates.
    skills = {
        str(skill).strip().lower()
        for skill in skills
        if str(skill).strip()
    }

    return sorted(
        skills,
        key=lambda x: (-len(x), x)
    )


@lru_cache(maxsize=1)
def load_skill_frequency() -> Dict[str, int]:
    """
    Load skill frequency information generated during training.
    """

    data = _load_gzip_json(
        FREQUENCY_FILE,
        {}
    )

    if not isinstance(data, dict):
        return {}

    return {
        str(k).lower(): int(v)
        for k, v in data.items()
        if str(k).strip()
    }


@lru_cache(maxsize=1)
def load_synonyms() -> Dict[str, List[str]]:
    """
    Load the synonym relationships generated during training.
    """

    data = _load_gzip_json(
        SYNONYMS_FILE,
        {}
    )

    if not isinstance(data, dict):
        return {}

    normalized = {}

    for key, values in data.items():

        key = str(key).strip().lower()

        if not key:
            continue

        if isinstance(values, list):

            normalized[key] = [
                str(value).strip().lower()
                for value in values
                if str(value).strip()
            ]

    return normalized


# ==============================================================
# Build lookup structures
# ==============================================================

@lru_cache(maxsize=1)
def build_skill_lookup() -> Dict[str, str]:
    """
    Create a normalized lookup table.

    Example:

        "machine learning" -> "machine learning"
        "ml"               -> "machine learning"

    The canonical skill name is returned.
    """

    skills = load_skills()

    lookup = {}

    for skill in skills:
        lookup[skill] = skill

    synonyms = load_synonyms()

    for canonical, alternatives in synonyms.items():

        # Canonical skill itself.
        lookup[canonical] = canonical

        for alternative in alternatives:

            alternative = alternative.strip().lower()

            if alternative:
                lookup[alternative] = canonical

    return lookup


# ==============================================================
# Phrase matching
# ==============================================================

def _contains_skill(
    text: str,
    skill: str
) -> bool:
    """
    Check whether a skill occurs as a complete phrase.

    Word boundaries prevent false matches such as:

        "R" matching "research"

    while still allowing technical skills such as:

        C++
        C#
        .NET
        Node.js
    """

    if not skill:
        return False

    # Escape the skill so symbols such as + and . are literal.
    escaped = re.escape(skill)

    # For normal alphabetic/number skills use boundaries.
    if re.search(r"[a-z0-9]", skill):

        pattern = (
            r"(?<![a-z0-9])"
            + escaped
            + r"(?![a-z0-9])"
        )

    else:

        pattern = escaped

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE
    ) is not None


# ==============================================================
# Main extraction
# ==============================================================

def extract_skills(
    text: str,
    max_skills: int = 100
) -> List[str]:
    """
    Extract skills from resume text.

    Parameters
    ----------
    text:
        Resume text.

    max_skills:
        Maximum number of returned skills.

    Returns
    -------
    List[str]
        Canonical skill names.
    """

    if not text:
        return []

    normalized = normalize_text(text)

    if not normalized:
        return []

    lookup = build_skill_lookup()

    found: Set[str] = set()

    # Sort by length so multi-word skills are matched before
    # shorter related terms.
    candidates = sorted(
        lookup.keys(),
        key=lambda x: (-len(x), x)
    )

    for candidate in candidates:

        if _contains_skill(
            normalized,
            candidate
        ):

            canonical = lookup[candidate]

            found.add(canonical)

            if len(found) >= max_skills:
                break

    # Rank using training frequency where available.
    frequency = load_skill_frequency()

    results = sorted(
        found,
        key=lambda skill: (
            -frequency.get(skill, 0),
            skill
        )
    )

    return results[:max_skills]


# ==============================================================
# Detailed extraction
# ==============================================================

def extract_skill_details(
    text: str,
    max_skills: int = 100
) -> List[Dict]:
    """
    Return skills together with their training frequency.

    Useful for the employer-facing API.
    """

    skills = extract_skills(
        text,
        max_skills=max_skills
    )

    frequency = load_skill_frequency()

    results = []

    for skill in skills:

        results.append({
            "skill": skill,
            "frequency": frequency.get(
                skill,
                0
            )
        })

    return results


# ==============================================================
# Required skills comparison
# ==============================================================

def compare_skills(
    resume_text: str,
    required_skills: List[str]
) -> Dict:
    """
    Compare resume skills against employer requirements.

    Returns:
        matched
        missing
        match_percentage
    """

    resume_skills = set(
        extract_skills(resume_text)
    )

    normalized_required = []

    lookup = build_skill_lookup()

    for skill in required_skills:

        normalized = normalize_text(skill)

        if not normalized:
            continue

        canonical = lookup.get(
            normalized,
            normalized
        )

        normalized_required.append(
            canonical
        )

    required_set = set(
        normalized_required
    )

    matched = sorted(
        resume_skills.intersection(
            required_set
        )
    )

    missing = sorted(
        required_set.difference(
            resume_skills
        )
    )

    if required_set:

        percentage = (
            len(matched)
            /
            len(required_set)
        ) * 100

    else:

        percentage = 0.0

    return {
        "matched": matched,
        "missing": missing,
        "required": sorted(required_set),
        "match_percentage": round(
            percentage,
            2
        )
    }


# ==============================================================
# Health check
# ==============================================================

def skill_engine_status() -> Dict:
    """
    Return information about the skill engine.

    Useful for debugging the deployed application.
    """

    skills = load_skills()
    frequency = load_skill_frequency()
    synonyms = load_synonyms()

    return {
        "status": "ready",
        "skill_count": len(skills),
        "frequency_entries": len(frequency),
        "synonym_entries": len(synonyms),
        "skills_file_exists": SKILLS_FILE.exists(),
        "frequency_file_exists": FREQUENCY_FILE.exists(),
        "synonyms_file_exists": SYNONYMS_FILE.exists()
    }


# ==============================================================
# Simple local test
# ==============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TalentMatch AI - Skill Extraction Engine")
    print("=" * 60)

    print(
        json.dumps(
            skill_engine_status(),
            indent=2
        )
    )

    sample_resume = """
    AI Engineer with experience in Python, Machine Learning,
    Deep Learning, TensorFlow, PyTorch, SQL, Docker,
    FastAPI, Computer Vision and Natural Language Processing.
    """

    skills = extract_skills(
        sample_resume
    )

    print("\nExtracted skills:")

    for skill in skills:
        print(
            f"  - {skill}"
        )
