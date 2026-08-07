"""
TalentMatch AI
Skill Extraction Engine

Purpose
-------
Extract relevant skills from uploaded resumes using the
trained skill artifacts generated during model training.

Artifacts used
--------------
models/
    skills.json.gz
    skill_frequency.json.gz
    synonyms.json.gz

Designed for
------------
- Render Free
- 512 MB RAM
- CPU inference
- Low CPU usage
- Fast repeated requests
- No model retraining
- No heavyweight NLP libraries
- No permanent resume storage

Architecture
------------
Resume text
    ↓
Text normalization
    ↓
Skill / synonym lookup
    ↓
Phrase matching
    ↓
Canonical skill mapping
    ↓
Frequency-based ranking
    ↓
Extracted skills

The module contains no machine-learning model.
It therefore adds very little memory overhead.
"""

from __future__ import annotations

import gzip
import json
import re

from functools import lru_cache
from typing import Dict, List, Set

from config import MODELS_DIR


# ============================================================
# Artifact Paths
# ============================================================

SKILLS_FILE = MODELS_DIR / "skills.json.gz"

FREQUENCY_FILE = (
    MODELS_DIR / "skill_frequency.json.gz"
)

SYNONYMS_FILE = (
    MODELS_DIR / "synonyms.json.gz"
)


# ============================================================
# Limits
# ============================================================

DEFAULT_MAX_SKILLS = 100

MAX_ALLOWED_SKILLS = 250


# ============================================================
# Text Normalization
# ============================================================

def normalize_text(
    text: str
) -> str:
    """
    Normalize resume text for skill matching.

    The implementation intentionally avoids NLP libraries.

    Technical characters such as:

        C++
        C#
        .NET
        Node.js
        C/C++
        REST/API

    are preserved where possible.
    """

    if not text:
        return ""

    text = str(text).lower()

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        " "
    )

    text = text.replace(
        "\r",
        " "
    )

    text = text.replace(
        "\n",
        " "
    )

    text = text.replace(
        "\t",
        " "
    )

    # --------------------------------------------------------
    # Preserve characters commonly used by technical skills.
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9+#./&\-\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Collapse whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Safe GZIP JSON Loader
# ============================================================

def _load_gzip_json(
    path,
    default
):
    """
    Safely load a gzip-compressed JSON artifact.

    If the artifact does not exist or cannot be read,
    the supplied default value is returned.

    This prevents the entire application from crashing
    because an optional skill artifact is unavailable.
    """

    if not path.exists():
        return default

    try:

        with gzip.open(
            path,
            "rt",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as exc:

        print(
            f"Warning: unable to load "
            f"{path.name}: {exc}"
        )

        return default


# ============================================================
# Skill Vocabulary
# ============================================================

@lru_cache(maxsize=1)
def load_skills() -> List[str]:
    """
    Load the trained skill vocabulary.

    The result is cached after the first request.

    Returns
    -------
    List[str]
        Normalized unique skill names.
    """

    skills = _load_gzip_json(
        SKILLS_FILE,
        []
    )

    if not isinstance(
        skills,
        list
    ):
        return []

    normalized: Set[str] = set()

    for skill in skills:

        if skill is None:
            continue

        skill = str(
            skill
        ).strip().lower()

        if skill:
            normalized.add(
                skill
            )

    # --------------------------------------------------------
    # Longer phrases first.
    #
    # Example:
    #
    # "machine learning"
    # before
    # "learning"
    # --------------------------------------------------------

    return sorted(
        normalized,
        key=lambda value: (
            -len(value),
            value
        )
    )


# ============================================================
# Skill Frequency
# ============================================================

@lru_cache(maxsize=1)
def load_skill_frequency() -> Dict[str, int]:
    """
    Load skill frequency information generated during training.

    Frequency is used only for ranking extracted skills.
    It does not determine whether a skill exists.
    """

    data = _load_gzip_json(
        FREQUENCY_FILE,
        {}
    )

    if not isinstance(
        data,
        dict
    ):
        return {}

    frequency = {}

    for key, value in data.items():

        try:

            normalized_key = str(
                key
            ).strip().lower()

            if not normalized_key:
                continue

            frequency[
                normalized_key
            ] = max(
                0,
                int(value)
            )

        except (
            TypeError,
            ValueError
        ):

            continue

    return frequency


# ============================================================
# Synonym Database
# ============================================================

@lru_cache(maxsize=1)
def load_synonyms() -> Dict[str, List[str]]:
    """
    Load trained skill synonym relationships.

    Example:

        machine learning:
            ml
            machine-learning

    The canonical skill remains the dictionary key.
    """

    data = _load_gzip_json(
        SYNONYMS_FILE,
        {}
    )

    if not isinstance(
        data,
        dict
    ):
        return {}

    normalized = {}

    for key, values in data.items():

        canonical = str(
            key
        ).strip().lower()

        if not canonical:
            continue

        if not isinstance(
            values,
            list
        ):
            continue

        alternatives = []

        for value in values:

            if value is None:
                continue

            value = str(
                value
            ).strip().lower()

            if value:
                alternatives.append(
                    value
                )

        normalized[
            canonical
        ] = alternatives

    return normalized


# ============================================================
# Skill Lookup
# ============================================================

@lru_cache(maxsize=1)
def build_skill_lookup() -> Dict[str, str]:
    """
    Build a normalized lookup table.

    Example
    -------

        "python" -> "python"

        "ml" ->
            "machine learning"

        "machine-learning" ->
            "machine learning"

    Returns
    -------
    Dict[str, str]
        Alternative skill → canonical skill.
    """

    lookup = {}

    # --------------------------------------------------------
    # Canonical skills
    # --------------------------------------------------------

    for skill in load_skills():

        lookup[
            skill
        ] = skill

    # --------------------------------------------------------
    # Synonyms
    # --------------------------------------------------------

    synonyms = load_synonyms()

    for canonical, alternatives in synonyms.items():

        # Make sure canonical skill exists.
        lookup[
            canonical
        ] = canonical

        for alternative in alternatives:

            alternative = (
                alternative
                .strip()
                .lower()
            )

            if alternative:

                lookup[
                    alternative
                ] = canonical

    return lookup


# ============================================================
# Skill Pattern Cache
# ============================================================

@lru_cache(maxsize=4096)
def _build_skill_pattern(
    skill: str
):
    """
    Build and cache a regular-expression pattern for a skill.

    Caching avoids recompiling the same expression during
    repeated resume-analysis requests.
    """

    if not skill:
        return None

    escaped = re.escape(
        skill
    )

    # --------------------------------------------------------
    # Skills containing normal alphanumeric characters
    # receive boundary protection.
    #
    # This prevents:
    #
    # "r"
    #
    # from matching:
    #
    # "research"
    # --------------------------------------------------------

    if re.search(
        r"[a-z0-9]",
        skill
    ):

        pattern = (
            r"(?<![a-z0-9])"
            + escaped
            + r"(?![a-z0-9])"
        )

    else:

        pattern = escaped

    try:

        return re.compile(
            pattern,
            flags=re.IGNORECASE
        )

    except re.error:

        return None


# ============================================================
# Phrase Matching
# ============================================================

def _contains_skill(
    text: str,
    skill: str
) -> bool:
    """
    Check whether a skill appears as a complete phrase.

    Regular-expression patterns are cached to reduce CPU
    overhead on repeated requests.
    """

    if not text or not skill:
        return False

    pattern = _build_skill_pattern(
        skill
    )

    if pattern is None:
        return False

    return (
        pattern.search(text)
        is not None
    )


# ============================================================
# Main Skill Extraction
# ============================================================

def extract_skills(
    text: str,
    max_skills: int = DEFAULT_MAX_SKILLS
) -> List[str]:
    """
    Extract canonical skills from resume text.

    Parameters
    ----------
    text:
        Resume text.

    max_skills:
        Maximum number of skills returned.

    Returns
    -------
    List[str]
        Canonical skill names ranked by training frequency.
    """

    if not text:
        return []

    # --------------------------------------------------------
    # Protect the server from unreasonable values.
    # --------------------------------------------------------

    try:

        max_skills = int(
            max_skills
        )

    except (
        TypeError,
        ValueError
    ):

        max_skills = DEFAULT_MAX_SKILLS

    max_skills = max(
        1,
        min(
            max_skills,
            MAX_ALLOWED_SKILLS
        )
    )

    # --------------------------------------------------------
    # Normalize resume
    # --------------------------------------------------------

    normalized = normalize_text(
        text
    )

    if not normalized:
        return []

    # --------------------------------------------------------
    # Build lookup
    # --------------------------------------------------------

    lookup = build_skill_lookup()

    if not lookup:
        return []

    found: Set[str] = set()

    # --------------------------------------------------------
    # Check candidate skills.
    #
    # The lookup is already ordered by skill length through
    # load_skills(), but synonym additions may not be.
    #
    # Sorting here ensures multi-word skills are checked first.
    # --------------------------------------------------------

    candidates = sorted(
        lookup.keys(),
        key=lambda value: (
            -len(value),
            value
        )
    )

    for candidate in candidates:

        if _contains_skill(
            normalized,
            candidate
        ):

            canonical = lookup[
                candidate
            ]

            found.add(
                canonical
            )

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if not found:
        return []

    # --------------------------------------------------------
    # Rank using training frequency.
    #
    # Higher-frequency skills appear first.
    # Alphabetical ordering provides deterministic results
    # when frequencies are equal.
    # --------------------------------------------------------

    frequency = load_skill_frequency()

    results = sorted(
        found,
        key=lambda skill: (
            -frequency.get(
                skill,
                0
            ),
            skill
        )
    )

    return results[
        :max_skills
    ]


# ============================================================
# Detailed Skill Extraction
# ============================================================

def extract_skill_details(
    text: str,
    max_skills: int = DEFAULT_MAX_SKILLS
) -> List[Dict]:
    """
    Return extracted skills together with training frequency.

    Example output:

        [
            {
                "skill": "python",
                "frequency": 1250
            }
        ]
    """

    skills = extract_skills(
        text,
        max_skills=max_skills
    )

    frequency = load_skill_frequency()

    return [
        {
            "skill": skill,
            "frequency": frequency.get(
                skill,
                0
            )
        }
        for skill in skills
    ]


# ============================================================
# Required Skill Comparison
# ============================================================

def compare_skills(
    resume_text: str,
    required_skills: List[str]
) -> Dict:
    """
    Compare resume skills against employer-required skills.

    Returns
    -------

        {
            "matched": [...],
            "missing": [...],
            "required": [...],
            "match_percentage": 75.0
        }

    Synonyms are normalized to their canonical skill names.
    """

    if not isinstance(
        required_skills,
        list
    ):

        required_skills = []

    # --------------------------------------------------------
    # Extract resume skills
    # --------------------------------------------------------

    resume_skills = set(
        extract_skills(
            resume_text
        )
    )

    # --------------------------------------------------------
    # Lookup canonical names
    # --------------------------------------------------------

    lookup = build_skill_lookup()

    normalized_required = set()

    for skill in required_skills:

        if skill is None:
            continue

        normalized = normalize_text(
            str(skill)
        )

        if not normalized:
            continue

        canonical = lookup.get(
            normalized,
            normalized
        )

        normalized_required.add(
            canonical
        )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    matched = sorted(
        resume_skills.intersection(
            normalized_required
        )
    )

    missing = sorted(
        normalized_required.difference(
            resume_skills
        )
    )

    required = sorted(
        normalized_required
    )

    # --------------------------------------------------------
    # Match percentage
    # --------------------------------------------------------

    if normalized_required:

        percentage = (
            len(matched)
            /
            len(normalized_required)
        ) * 100.0

    else:

        percentage = 0.0

    return {

        "matched": matched,

        "missing": missing,

        "required": required,

        "match_percentage": round(
            percentage,
            2
        )
    }


# ============================================================
# Skill Engine Health Check
# ============================================================

def skill_engine_status() -> Dict:
    """
    Return lightweight information about the skill engine.

    This endpoint can be used for deployment diagnostics.
    """

    skills = load_skills()

    frequency = load_skill_frequency()

    synonyms = load_synonyms()

    lookup = build_skill_lookup()

    return {

        "status": "ready",

        "skill_count": len(
            skills
        ),

        "frequency_entries": len(
            frequency
        ),

        "synonym_entries": len(
            synonyms
        ),

        "lookup_entries": len(
            lookup
        ),

        "skills_file_exists":
            SKILLS_FILE.exists(),

        "frequency_file_exists":
            FREQUENCY_FILE.exists(),

        "synonyms_file_exists":
            SYNONYMS_FILE.exists()
    }


# ============================================================
# Cache Management
# ============================================================

def clear_skill_caches() -> None:
    """
    Clear cached skill artifacts and compiled patterns.

    Normally this does not need to be called in production.

    It is useful during development/testing if the underlying
    skill artifacts are replaced without restarting Python.
    """

    load_skills.cache_clear()

    load_skill_frequency.cache_clear()

    load_synonyms.cache_clear()

    build_skill_lookup.cache_clear()

    _build_skill_pattern.cache_clear()


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "TalentMatch AI - Skill Extraction Engine"
    )

    print("=" * 60)

    print()

    print(
        json.dumps(
            skill_engine_status(),
            indent=2
        )
    )

    print()

    sample_resume = """
    AI Engineer with experience in Python,
    Machine Learning, Deep Learning,
    TensorFlow, PyTorch, SQL, Docker,
    FastAPI, Computer Vision and
    Natural Language Processing.
    """

    skills = extract_skills(
        sample_resume
    )

    print(
        "\nExtracted skills:"
    )

    for skill in skills:

        print(
            f"  - {skill}"
        )

    print()

    print(
        "Detailed skills:"
    )

    print(
        json.dumps(
            extract_skill_details(
                sample_resume
            ),
            indent=2
        )
    )
