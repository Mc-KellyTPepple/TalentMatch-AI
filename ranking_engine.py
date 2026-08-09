```python
"""
================================================================
TalentMatch AI
Ranking Engine
================================================================

Production ranking layer for TalentMatch AI.

Responsibilities
----------------
• Combine semantic and lexical matching
• Remove duplicate job listings
• Produce normalized match scores
• Generate evidence-based explanations
• Identify matched and missing TF-IDF terms
• Produce candidate/employer-friendly summaries
• Return lightweight JSON-compatible results

IMPORTANT
---------
The Match Score is an AI compatibility score.

It is NOT:
• probability of employment
• probability of interview
• probability of hiring
• a replacement for human recruitment decisions

Designed for:
Render Free
512 MB RAM
"""

from typing import Any, Dict, List
import re


# ================================================================
# CONFIGURATION
# ================================================================

MIN_SCORE = 0.0
MAX_SCORE = 1.0

# Candidate retrieval multiplier.
#
# The prediction engine may return more candidates than the UI
# finally displays. This gives deduplication room without
# changing the underlying similarity calculation.
CANDIDATE_MULTIPLIER = 3

MAX_CANDIDATES = 50

MAX_MATCHED_TERMS = 8
MAX_MISSING_TERMS = 8


# ================================================================
# SCORE UTILITIES
# ================================================================

def clamp_score(score: float) -> float:
    return max(
        MIN_SCORE,
        min(MAX_SCORE, float(score))
    )


def score_to_percentage(score: float) -> float:
    """
    Preserve two decimal places.

    Example:
        0.4452 -> 44.52
    """
    score = clamp_score(score)
    return round(score * 100.0, 2)


def get_match_level(score: float) -> str:
    """
    Compatibility labels only.
    """

    percentage = score_to_percentage(score)

    if percentage >= 85:
        return "Excellent Match"

    if percentage >= 70:
        return "Strong Match"

    if percentage >= 55:
        return "Moderate Match"

    if percentage >= 40:
        return "Developing Match"

    return "Low Match"


# ================================================================
# TEXT NORMALIZATION
# ================================================================

def _normalize_text(value: Any) -> str:
    """
    Normalize text for duplicate detection only.

    This does NOT modify the text shown to the user.
    """

    if value is None:
        return ""

    text = str(value).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    return text.strip()


# ================================================================
# JOB TITLE
# ================================================================

def _get_job_title(job: Dict[str, Any]) -> str:

    for key in (
        "title",
        "job_title",
        "position",
        "role",
        "name",
    ):

        value = job.get(key)

        if value:
            return str(value)

    return ""


# ================================================================
# COMPANY
# ================================================================

def _get_company(job: Dict[str, Any]) -> str:

    for key in (
        "company",
        "company_name",
        "employer",
        "organization",
        "organisation",
    ):

        value = job.get(key)

        if value:
            return str(value)

    return ""


# ================================================================
# LOCATION
# ================================================================

def _get_location(job: Dict[str, Any]) -> str:

    for key in (
        "location",
        "city",
        "country",
        "job_location",
    ):

        value = job.get(key)

        if value:
            return str(value)

    return ""


# ================================================================
# DUPLICATE KEY
# ================================================================

def build_duplicate_key(job: Dict[str, Any]) -> str:
    """
    Build a conservative duplicate key.

    Priority:
        1. Stable job ID when available.
        2. Otherwise company + title + location + content.

    We intentionally include description/requirements so that two
    different employers can advertise the same job title without
    being incorrectly collapsed.
    """

    for key in (
        "job_id",
        "id",
        "listing_id",
        "jobid",
    ):

        value = job.get(key)

        if value not in (
            None,
            "",
        ):

            return (
                "id:"
                + _normalize_text(value)
            )

    parts = [

        _get_company(job),

        _get_job_title(job),

        _get_location(job),

        job.get(
            "description",
            "",
        ),

        job.get(
            "requirements",
            "",
        ),

    ]

    normalized = [
        _normalize_text(value)
        for value in parts
        if value
    ]

    return "|".join(normalized)


# ================================================================
# SCORE EXPLANATION
# ================================================================

def explain_score(
    semantic_score: float,
    tfidf_score: float,
    final_score: float,
    matched_terms: List[str] | None = None,
    missing_terms: List[str] | None = None,
) -> Dict[str, Any]:

    semantic_percentage = score_to_percentage(
        semantic_score
    )

    tfidf_percentage = score_to_percentage(
        tfidf_score
    )

    final_percentage = score_to_percentage(
        final_score
    )

    matched_terms = list(
        matched_terms or []
    )

    missing_terms = list(
        missing_terms or []
    )

    strengths = []

    if semantic_percentage >= 80:

        strengths.append(
            "Strong semantic alignment with the job."
        )

    elif semantic_percentage >= 60:

        strengths.append(
            "Good semantic alignment with the job."
        )

    elif semantic_percentage >= 40:

        strengths.append(
            "Moderate semantic alignment with the job."
        )

    if tfidf_percentage >= 60:

        strengths.append(
            "Strong overlap with job terminology."
        )

    elif tfidf_percentage >= 35:

        strengths.append(
            "Some relevant job terminology is present."
        )

    if matched_terms:

        strengths.append(
            "The resume shares several important terms "
            "with this job."
        )

    if not strengths:

        strengths.append(
            "Some relevant overlap was detected."
        )

    return {

        "match_score":
            final_percentage,

        "semantic_score":
            semantic_percentage,

        "keyword_score":
            tfidf_percentage,

        "match_level":
            get_match_level(
                final_score
            ),

        "strengths":
            strengths,

        "matched_terms":
            matched_terms,

        "potential_gaps":
            missing_terms,
    }


# ================================================================
# SINGLE JOB
# ================================================================

def rank_job(
    job: Dict[str, Any]
) -> Dict[str, Any]:

    semantic_score = float(
        job.get(
            "semantic_score",
            job.get(
                "semantic",
                0.0,
            ),
        )
    )

    tfidf_score = float(
        job.get(
            "tfidf_score",
            job.get(
                "keyword_score",
                0.0,
            ),
        )
    )

    final_score = float(
        job.get(
            "score",
            job.get(
                "final_score",
                0.0,
            ),
        )
    )

    explanation = explain_score(

        semantic_score=semantic_score,

        tfidf_score=tfidf_score,

        final_score=final_score,

        matched_terms=job.get(
            "matched_terms",
            [],
        ),

        missing_terms=job.get(
            "potential_gaps",
            [],
        ),
    )

    category = (
        job.get("category")
        or job.get("Category")
        or ""
    )

    title = _get_job_title(
        job
    )

    company = _get_company(
        job
    )

    location = _get_location(
        job
    )

    description = (
        job.get("description")
        or job.get("Description")
        or ""
    )

    requirements = (
        job.get("requirements")
        or job.get("Requirements")
        or ""
    )

    benefits = (
        job.get("benefits")
        or job.get("Benefits")
        or ""
    )

    result = {

        "match_score":
            explanation["match_score"],

        "semantic_score":
            explanation["semantic_score"],

        "keyword_score":
            explanation["keyword_score"],

        "match_level":
            explanation["match_level"],

        "title":
            str(title),

        "category":
            str(category),

        "company":
            str(company),

        "location":
            str(location),

        "description":
            str(description),

        "requirements":
            str(requirements),

        "benefits":
            str(benefits),

        "strengths":
            explanation["strengths"],

        "matched_terms":
            explanation["matched_terms"],

        "potential_gaps":
            explanation["potential_gaps"],
    }

    # Preserve stable identifiers when available.
    for key in (
        "job_id",
        "id",
        "listing_id",
    ):

        if job.get(key) not in (
            None,
            "",
        ):

            result[key] = job[key]

            break

    return result


# ================================================================
# DEDUPLICATION
# ================================================================

def deduplicate_jobs(
    jobs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate listings while preserving the highest-ranked
    version of each job.

    The first occurrence wins because jobs arrive already sorted
    by AI score.
    """

    unique = []

    seen = set()

    duplicate_count = 0

    for job in jobs:

        key = build_duplicate_key(
            job
        )

        if not key:

            unique.append(job)

            continue

        if key in seen:

            duplicate_count += 1

            continue

        seen.add(key)

        unique.append(job)

    return unique


# ================================================================
# RANK MULTIPLE JOBS
# ================================================================

def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:

    if not jobs:
        return []

    ranked = [
        rank_job(job)
        for job in jobs
    ]

    ranked.sort(
        key=lambda item: (
            item["match_score"],
            item["semantic_score"],
            item["keyword_score"],
        ),
        reverse=True,
    )

    ranked = deduplicate_jobs(
        ranked
    )

    ranked = ranked[
        :max(
            1,
            int(top_k),
        )
    ]

    for position, job in enumerate(
        ranked,
        start=1,
    ):

        job["rank"] = position

    return ranked


# ================================================================
# CANDIDATE SUMMARY
# ================================================================

def build_candidate_summary(
    ranked_jobs: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not ranked_jobs:

        return {

            "jobs_analyzed": 0,

            "best_match_score": 0,

            "best_match_level":
                "No Match",

            "average_match_score": 0,

            "strongest_match":
                None,

            "unique_jobs_returned":
                0,
        }

    scores = [
        float(
            job.get(
                "match_score",
                0,
            )
        )
        for job in ranked_jobs
    ]

    best = ranked_jobs[0]

    average_score = round(
        sum(scores)
        / len(scores),
        2,
    )

    return {

        "jobs_analyzed":
            len(ranked_jobs),

        "unique_jobs_returned":
            len(ranked_jobs),

        "best_match_score":
            best.get(
                "match_score",
                0,
            ),

        "best_match_level":
            best.get(
                "match_level",
                "Unknown",
            ),

        "average_match_score":
            average_score,

        "strongest_match":
            best.get(
                "title"
            )
            or best.get(
                "category"
            )
            or "Top recommended job",

        "score_interpretation":
            (
                "Compatibility score based on semantic "
                "and keyword similarity. It is not a "
                "probability of employment or hiring."
            ),
    }


# ================================================================
# MAIN RANKING INTERFACE
# ================================================================

def analyze_jobs(
    prediction_engine,
    resume_text: str,
    top_k: int = 10,
) -> Dict[str, Any]:

    if not resume_text:
        raise ValueError(
            "Resume text cannot be empty."
        )

    requested_k = max(
        1,
        int(top_k),
    )

    # ------------------------------------------------------------
    # Retrieve extra candidates so duplicate listings can be
    # removed without unnecessarily reducing the final result set.
    #
    # The prediction engine must allow this candidate count.
    # ------------------------------------------------------------

    candidate_k = min(
        max(
            requested_k
            * CANDIDATE_MULTIPLIER,
            requested_k,
        ),
        MAX_CANDIDATES,
    )

    predictions = (
        prediction_engine.hybrid_job_search(
            resume_text,
            top_k=candidate_k,
        )
    )

    ranked_jobs = rank_jobs(
        predictions,
        top_k=requested_k,
    )

    summary = build_candidate_summary(
        ranked_jobs
    )

    return {

        "summary":
            summary,

        "jobs":
            ranked_jobs,
    }


# ================================================================
# INTERVIEW QUESTIONS
# ================================================================

def format_interview_questions(
    questions: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:

    if not questions:
        return []

    formatted = []

    for position, question in enumerate(
        questions[:top_k],
        start=1,
    ):

        score = float(
            question.get(
                "score",
                0.0,
            )
        )

        formatted.append({

            "rank":
                position,

            "relevance_score":
                score_to_percentage(
                    score
                ),

            "question":
                question.get(
                    "question",
                    "",
                ),

            "ideal_answer":
                question.get(
                    "answer",
                    "",
                ),

            "role":
                question.get(
                    "role",
                    "",
                ),

            "category":
                question.get(
                    "category",
                    "",
                ),

            "difficulty":
                question.get(
                    "difficulty",
                    "",
                ),

            "experience":
                question.get(
                    "experience",
                    "",
                ),
        })

    return formatted
```
