"""
TalentMatch AI
Production Ranking / Response Formatting Layer

Responsibilities
----------------
- Combine semantic and lexical matching results
- Remove duplicate job listings
- Preserve the strongest duplicate
- Produce normalized match scores
- Generate evidence-based explanations
- Identify matched and potentially missing TF-IDF terms
- Produce candidate-friendly summaries
- Return lightweight JSON-compatible results

IMPORTANT
---------
The Match Score is an AI compatibility score.

It is NOT:
- probability of employment
- probability of interview
- probability of hiring
- a replacement for human recruitment decisions

Designed for:
- Render Free
- 512 MB RAM
- CPU inference
"""

from typing import Any, Dict, List
import re


# ===============================================================
# CONFIGURATION
# ===============================================================

MIN_SCORE = 0.0
MAX_SCORE = 1.0

CANDIDATE_MULTIPLIER = 3
MAX_CANDIDATES = 50

MAX_MATCHED_TERMS = 8
MAX_MISSING_TERMS = 8


# ===============================================================
# SCORE UTILITIES
# ===============================================================

def clamp_score(score: float) -> float:

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    if score < MIN_SCORE:
        return MIN_SCORE

    if score > MAX_SCORE:
        return MAX_SCORE

    return score


def score_to_percentage(score: float) -> float:

    return round(
        clamp_score(score) * 100.0,
        2,
    )


def get_match_level(score: float) -> str:

    percentage = score_to_percentage(
        score
    )

    if percentage >= 85:
        return "Excellent Match"

    if percentage >= 70:
        return "Strong Match"

    if percentage >= 55:
        return "Moderate Match"

    if percentage >= 40:
        return "Developing Match"

    return "Low Match"


# ===============================================================
# TEXT NORMALIZATION
# ===============================================================

def _normalize_text(value: Any) -> str:

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


# ===============================================================
# JOB TITLE
# ===============================================================

def _get_job_title(
    job: Dict[str, Any]
) -> str:

    for key in (
        "title",
        "job_title",
        "position",
        "role",
        "name",
    ):

        value = job.get(key)

        if value not in (
            None,
            "",
        ):

            return str(value)

    return ""


# ===============================================================
# COMPANY
# ===============================================================

def _get_company(
    job: Dict[str, Any]
) -> str:

    for key in (
        "company",
        "company_name",
        "employer",
        "organization",
        "organisation",
    ):

        value = job.get(key)

        if value not in (
            None,
            "",
        ):

            return str(value)

    return ""


# ===============================================================
# LOCATION
# ===============================================================

def _get_location(
    job: Dict[str, Any]
) -> str:

    for key in (
        "location",
        "city",
        "country",
        "job_location",
    ):

        value = job.get(key)

        if value not in (
            None,
            "",
        ):

            return str(value)

    return ""


# ===============================================================
# DUPLICATE KEY
# ===============================================================

def build_duplicate_key(
    job: Dict[str, Any]
) -> str:
    """
    Build a conservative duplicate key.

    Priority:
        1. Stable job ID.
        2. Company + title + location + description + requirements.

    This prevents two different employers with the same title
    from being incorrectly merged.
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

            normalized_id = _normalize_text(
                value
            )

            if normalized_id:

                return (
                    "id:"
                    + normalized_id
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

    normalized = [
        value
        for value in normalized
        if value
    ]

    if not normalized:
        return ""

    return "|".join(
        normalized
    )


# ===============================================================
# SCORE EXPLANATION
# ===============================================================

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

    # -----------------------------------------------------------
    # Semantic evidence
    # -----------------------------------------------------------

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

    else:

        strengths.append(
            "Limited semantic alignment with the job."
        )

    # -----------------------------------------------------------
    # Keyword evidence
    # -----------------------------------------------------------

    if tfidf_percentage >= 60:

        strengths.append(
            "Strong overlap with the job's terminology."
        )

    elif tfidf_percentage >= 35:

        strengths.append(
            "Relevant job terminology appears in the resume."
        )

    elif tfidf_percentage > 0:

        strengths.append(
            "Some job-related terminology overlaps."
        )

    # -----------------------------------------------------------
    # Actual matched terms
    # -----------------------------------------------------------

    if matched_terms:

        strengths.append(
            "Matched terms include: "
            + ", ".join(
                matched_terms[:MAX_MATCHED_TERMS]
            )
            + "."
        )

    # -----------------------------------------------------------
    # Fallback
    # -----------------------------------------------------------

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
            matched_terms[
                :MAX_MATCHED_TERMS
            ],

        "potential_gaps":
            missing_terms[
                :MAX_MISSING_TERMS
            ],
    }


# ===============================================================
# SINGLE JOB
# ===============================================================

def rank_job(
    job: Dict[str, Any],
    prediction_engine=None,
    resume_text: str = "",
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

    # -----------------------------------------------------------
    # TF-IDF evidence
    # -----------------------------------------------------------

    matched_terms = job.get(
        "matched_terms",
        [],
    )

    missing_terms = job.get(
        "potential_gaps",
        [],
    )

    job_index = job.get(
        "_job_index"
    )

    if (
        prediction_engine is not None
        and resume_text
        and job_index is not None
        and hasattr(
            prediction_engine,
            "get_tfidf_evidence",
        )
    ):

        try:

            evidence = (
                prediction_engine.get_tfidf_evidence(
                    resume_text=resume_text,
                    job_index=job_index,
                    max_terms=MAX_MATCHED_TERMS,
                )
            )

            if evidence:

                matched_terms = (
                    evidence[0]
                    or matched_terms
                )

                missing_terms = (
                    evidence[1]
                    or missing_terms
                )

        except Exception:
            # Evidence is an enhancement.
            # Never allow it to break job ranking.
            pass

    # -----------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------

    explanation = explain_score(

        semantic_score=semantic_score,

        tfidf_score=tfidf_score,

        final_score=final_score,

        matched_terms=matched_terms,

        missing_terms=missing_terms,
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

    # -----------------------------------------------------------
    # Stable identifiers
    # -----------------------------------------------------------

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


# ===============================================================
# DEDUPLICATION
# ===============================================================

def deduplicate_jobs(
    jobs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate listings while preserving the
    highest-ranked occurrence.
    """

    if not jobs:
        return []

    unique = []
    seen = set()

    for job in jobs:

        key = build_duplicate_key(
            job
        )

        # If a listing has absolutely no identifying
        # information, do not accidentally remove it.
        if not key:

            unique.append(job)
            continue

        if key in seen:
            continue

        seen.add(key)
        unique.append(job)

    return unique


# ===============================================================
# RANK MULTIPLE JOBS
# ===============================================================

def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = 10,
    prediction_engine=None,
    resume_text: str = "",
) -> List[Dict[str, Any]]:

    if not jobs:
        return []

    ranked = []

    for job in jobs:

        try:

            ranked.append(
                rank_job(
                    job,
                    prediction_engine=prediction_engine,
                    resume_text=resume_text,
                )
            )

        except Exception:

            # One malformed result should not
            # destroy the entire candidate analysis.
            continue

    if not ranked:
        return []

    # -----------------------------------------------------------
    # Highest AI score first
    # -----------------------------------------------------------

    ranked.sort(
        key=lambda item: (
            float(
                item.get(
                    "match_score",
                    0.0,
                )
            ),
            float(
                item.get(
                    "semantic_score",
                    0.0,
                )
            ),
            float(
                item.get(
                    "keyword_score",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    # -----------------------------------------------------------
    # Remove duplicate listings
    # -----------------------------------------------------------

    ranked = deduplicate_jobs(
        ranked
    )

    # -----------------------------------------------------------
    # Final result size
    # -----------------------------------------------------------

    try:
        requested_k = max(
            1,
            int(top_k),
        )
    except (TypeError, ValueError):
        requested_k = 10

    ranked = ranked[
        :requested_k
    ]

    # -----------------------------------------------------------
    # Assign rank
    # -----------------------------------------------------------

    for position, job in enumerate(
        ranked,
        start=1,
    ):

        job["rank"] = position

    return ranked


# ===============================================================
# CANDIDATE SUMMARY
# ===============================================================

def build_candidate_summary(
    ranked_jobs: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not ranked_jobs:

        return {

            "jobs_analyzed": 0,

            "unique_jobs_returned": 0,

            "best_match_score": 0,

            "best_match_level":
                "No Match",

            "average_match_score": 0,

            "strongest_match":
                None,

            "score_interpretation":
                (
                    "No sufficiently relevant jobs "
                    "were found."
                ),
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
            (
                best.get("title")
                or best.get("category")
                or "Top recommended job"
            ),

        "score_interpretation":
            (
                "Compatibility score based on semantic "
                "and keyword similarity. It is not a "
                "probability of employment or hiring."
            ),
    }


# ===============================================================
# MAIN RANKING INTERFACE
# ===============================================================

def analyze_jobs(
    prediction_engine,
    resume_text: str,
    top_k: int = 10,
) -> Dict[str, Any]:

    if prediction_engine is None:

        raise ValueError(
            "Prediction engine is required."
        )

    if not resume_text:

        raise ValueError(
            "Resume text cannot be empty."
        )

    resume_text = str(
        resume_text
    ).strip()

    if not resume_text:

        raise ValueError(
            "Resume text cannot be empty."
        )

    try:

        requested_k = max(
            1,
            int(top_k),
        )

    except (TypeError, ValueError):

        requested_k = 10

    # -----------------------------------------------------------
    # Retrieve extra candidates.
    # -----------------------------------------------------------

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
            resume_text=resume_text,
            top_k=candidate_k,
        )
    )

    ranked_jobs = rank_jobs(
        predictions,
        top_k=requested_k,
        prediction_engine=prediction_engine,
        resume_text=resume_text,
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


# ===============================================================
# INTERVIEW QUESTIONS
# ===============================================================

def format_interview_questions(
    questions: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:

    if not questions:
        return []

    try:

        limit = min(
            max(
                1,
                int(top_k),
            ),
            len(questions),
        )

    except (TypeError, ValueError):

        limit = min(
            10,
            len(questions),
        )

    formatted = []

    for position, question in enumerate(
        questions[:limit],
        start=1,
    ):

        try:

            score = float(
                question.get(
                    "score",
                    0.0,
                )
            )

        except (TypeError, ValueError):

            score = 0.0

        formatted.append({

            "rank":
                position,

            "relevance_score":
                score_to_percentage(
                    score
                ),

            "question":
                str(
                    question.get(
                        "question",
                        "",
                    )
                    or ""
                ),

            "ideal_answer":
                str(
                    question.get(
                        "answer",
                        "",
                    )
                    or ""
                ),

            "role":
                str(
                    question.get(
                        "role",
                        "",
                    )
                    or ""
                ),

            "category":
                str(
                    question.get(
                        "category",
                        "",
                    )
                    or ""
                ),

            "difficulty":
                str(
                    question.get(
                        "difficulty",
                        "",
                    )
                    or ""
                ),

            "experience":
                str(
                    question.get(
                        "experience",
                        "",
                    )
                    or ""
                ),
        })

    return formatted
