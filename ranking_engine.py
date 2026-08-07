"""
=========================================================
TalentMatch AI
Ranking Engine
=========================================================

Production ranking layer for TalentMatch AI.

Responsibilities:

    • Combine semantic and lexical matching
    • Produce a normalized Job Match Score
    • Assign an interpretable match level
    • Generate ranking explanations
    • Identify strengths and potential gaps
    • Return lightweight JSON-compatible results

Designed for:
    Render Free
    512 MB RAM

Important:
    The Match Score is an AI compatibility score.
    It is NOT a probability of employment or hiring.
=========================================================
"""

from typing import Any, Dict, List


# =========================================================
# Configuration
# =========================================================

MIN_SCORE = 0.0
MAX_SCORE = 1.0


# =========================================================
# Score Utilities
# =========================================================

def clamp_score(score: float) -> float:
    """
    Keep a score inside the valid [0, 1] range.
    """

    return max(
        MIN_SCORE,
        min(MAX_SCORE, float(score))
    )


def score_to_percentage(score: float) -> int:
    """
    Convert a normalized score to a percentage.
    """

    score = clamp_score(score)

    return int(round(score * 100))


def get_match_level(score: float) -> str:
    """
    Convert the AI match score into an interpretable
    category.

    These labels describe compatibility, not hiring odds.
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


# =========================================================
# Score Explanation
# =========================================================

def explain_score(
    semantic_score: float,
    tfidf_score: float,
    final_score: float
) -> Dict[str, Any]:
    """
    Create a transparent explanation of the ranking score.
    """

    semantic_percentage = score_to_percentage(
        semantic_score
    )

    tfidf_percentage = score_to_percentage(
        tfidf_score
    )

    final_percentage = score_to_percentage(
        final_score
    )

    strengths = []

    if semantic_percentage >= 80:
        strengths.append(
            "Strong semantic similarity with the job."
        )

    elif semantic_percentage >= 60:
        strengths.append(
            "Good semantic alignment with the job."
        )

    if tfidf_percentage >= 80:
        strengths.append(
            "Strong keyword and terminology overlap."
        )

    elif tfidf_percentage >= 60:
        strengths.append(
            "Good keyword alignment with the job."
        )

    if not strengths:
        strengths.append(
            "Some relevant overlap was detected."
        )

    return {
        "match_score": final_percentage,
        "semantic_score": semantic_percentage,
        "keyword_score": tfidf_percentage,
        "match_level": get_match_level(final_score),
        "strengths": strengths,
    }


# =========================================================
# Single Job Ranking
# =========================================================

def rank_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a raw prediction result into a
    production-ready ranking result.

    Handles both lowercase and capitalized metadata
    column names so the engine remains robust.
    """

    semantic_score = float(
        job.get(
            "semantic_score",
            job.get("semantic", 0.0)
        )
    )

    tfidf_score = float(
        job.get(
            "tfidf_score",
            job.get("keyword_score", 0.0)
        )
    )

    final_score = float(
        job.get(
            "score",
            job.get("final_score", 0.0)
        )
    )

    explanation = explain_score(
        semantic_score=semantic_score,
        tfidf_score=tfidf_score,
        final_score=final_score
    )

    # -----------------------------------------------------
    # Handle dataset column naming variations
    # -----------------------------------------------------

    category = (
        job.get("category")
        or job.get("Category")
        or ""
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

    return {
        "match_score": explanation["match_score"],
        "semantic_score": explanation["semantic_score"],
        "keyword_score": explanation["keyword_score"],
        "match_level": explanation["match_level"],

        "category": str(category),

        "description": str(description),

        "requirements": str(requirements),

        "benefits": str(benefits),

        "strengths": explanation["strengths"],
    }


# =========================================================
# Rank Multiple Jobs
# =========================================================

def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Rank and format multiple jobs.

    The prediction engine has already performed the
    expensive similarity calculations.

    This function therefore performs only lightweight
    Python operations.
    """

    if not jobs:
        return []

    ranked = [
        rank_job(job)
        for job in jobs
    ]

    ranked.sort(
        key=lambda item: item["match_score"],
        reverse=True
    )

    ranked = ranked[:max(1, int(top_k))]

    # -----------------------------------------------------
    # Add ranking position
    # -----------------------------------------------------

    for position, job in enumerate(
        ranked,
        start=1
    ):
        job["rank"] = position

    return ranked


# =========================================================
# Candidate Matching Summary
# =========================================================

def build_candidate_summary(
    ranked_jobs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Produce a high-level summary suitable for the
    TalentMatch AI dashboard.
    """

    if not ranked_jobs:

        return {
            "jobs_analyzed": 0,
            "best_match_score": 0,
            "best_match_level": "No Match",
            "average_match_score": 0,
        }

    scores = [
        job["match_score"]
        for job in ranked_jobs
    ]

    best = ranked_jobs[0]

    average_score = round(
        sum(scores) / len(scores)
    )

    return {
        "jobs_analyzed": len(ranked_jobs),

        "best_match_score": best[
            "match_score"
        ],

        "best_match_level": best[
            "match_level"
        ],

        "average_match_score": average_score,
    }


# =========================================================
# Main Ranking Interface
# =========================================================

def analyze_jobs(
    prediction_engine,
    resume_text: str,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Main interface used by app.py.

    Flow:

        Resume text
             ↓
        Prediction Engine
             ↓
        Hybrid similarity
             ↓
        Ranking Engine
             ↓
        Structured response
    """

    if not resume_text:
        raise ValueError(
            "Resume text cannot be empty."
        )

    # -----------------------------------------------------
    # Ask prediction engine for hybrid matches
    # -----------------------------------------------------

    predictions = (
        prediction_engine.hybrid_job_search(
            resume_text,
            top_k=top_k
        )
    )

    # -----------------------------------------------------
    # Rank and format
    # -----------------------------------------------------

    ranked_jobs = rank_jobs(
        predictions,
        top_k=top_k
    )

    # -----------------------------------------------------
    # Candidate summary
    # -----------------------------------------------------

    summary = build_candidate_summary(
        ranked_jobs
    )

    return {
        "summary": summary,
        "jobs": ranked_jobs,
    }


# =========================================================
# Interview Analysis
# =========================================================

def format_interview_questions(
    questions: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Format interview recommendations into a
    lightweight API response.
    """

    if not questions:
        return []

    formatted = []

    for position, question in enumerate(
        questions[:top_k],
        start=1
    ):

        score = float(
            question.get("score", 0.0)
        )

        formatted.append({

            "rank": position,

            "relevance_score":
                score_to_percentage(score),

            "question":
                question.get(
                    "question",
                    ""
                ),

            "ideal_answer":
                question.get(
                    "answer",
                    ""
                ),

            "role":
                question.get(
                    "role",
                    ""
                ),

            "category":
                question.get(
                    "category",
                    ""
                ),

            "difficulty":
                question.get(
                    "difficulty",
                    ""
                ),

            "experience":
                question.get(
                    "experience",
                    ""
                ),
        })

    return formatted
