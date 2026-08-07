"""
============================================================
TalentMatch AI
Production Ranking Engine
============================================================

Converts raw AI similarity results into:

• Match percentage
• Match level
• Strengths
• Ranking
• Candidate summary

Designed for Render Free / 512 MB RAM.

No ML models are loaded here.
============================================================
"""

from typing import Any, Dict, List


# ============================================================
# Score Utilities
# ============================================================

def clamp_score(
    score: float
) -> float:

    return max(
        0.0,
        min(
            1.0,
            float(score)
        )
    )


def score_to_percentage(
    score: float
) -> int:

    return int(
        round(
            clamp_score(score) * 100
        )
    )


def get_match_level(
    score: float
) -> str:

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


# ============================================================
# Score Explanation
# ============================================================

def explain_score(
    semantic_score: float,
    tfidf_score: float,
    final_score: float
) -> Dict[str, Any]:

    semantic_percentage = (
        score_to_percentage(
            semantic_score
        )
    )

    tfidf_percentage = (
        score_to_percentage(
            tfidf_score
        )
    )

    final_percentage = (
        score_to_percentage(
            final_score
        )
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
            strengths
    }


# ============================================================
# Rank Single Job
# ============================================================

def rank_job(
    job: Dict[str, Any]
) -> Dict[str, Any]:

    semantic_score = float(
        job.get(
            "semantic_score",
            0.0
        )
    )

    tfidf_score = float(
        job.get(
            "tfidf_score",
            0.0
        )
    )

    final_score = float(
        job.get(
            "score",
            0.0
        )
    )

    explanation = explain_score(
        semantic_score,
        tfidf_score,
        final_score
    )

    return {

        "match_score":
            explanation["match_score"],

        "semantic_score":
            explanation["semantic_score"],

        "keyword_score":
            explanation["keyword_score"],

        "match_level":
            explanation["match_level"],

        "category":
            str(
                job.get(
                    "category",
                    ""
                )
            ),

        "description":
            str(
                job.get(
                    "description",
                    ""
                )
            ),

        "requirements":
            str(
                job.get(
                    "requirements",
                    ""
                )
            ),

        "benefits":
            str(
                job.get(
                    "benefits",
                    ""
                )
            ),

        "strengths":
            explanation["strengths"]
    }


# ============================================================
# Rank Multiple Jobs
# ============================================================

def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = 10
) -> List[Dict[str, Any]]:

    if not jobs:

        return []

    ranked = [
        rank_job(job)
        for job in jobs
    ]

    ranked.sort(
        key=lambda item:
            item["match_score"],
        reverse=True
    )

    ranked = ranked[
        :max(
            1,
            int(top_k)
        )
    ]

    for position, job in enumerate(
        ranked,
        start=1
    ):

        job["rank"] = position

    return ranked


# ============================================================
# Candidate Summary
# ============================================================

def build_candidate_summary(
    ranked_jobs: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not ranked_jobs:

        return {

            "jobs_analyzed": 0,

            "best_match_score": 0,

            "best_match_level":
                "No Match",

            "average_match_score": 0
        }

    scores = [
        job["match_score"]
        for job in ranked_jobs
    ]

    best = ranked_jobs[0]

    return {

        "jobs_analyzed":
            len(ranked_jobs),

        "best_match_score":
            best["match_score"],

        "best_match_level":
            best["match_level"],

        "average_match_score":
            round(
                sum(scores)
                / len(scores)
            )
    }


# ============================================================
# Main Job Analysis
# ============================================================

def analyze_jobs(
    prediction_engine,
    resume_text: str,
    top_k: int = 10
) -> Dict[str, Any]:

    if not resume_text:

        raise ValueError(
            "Resume text cannot be empty."
        )

    predictions = (
        prediction_engine.hybrid_job_search(
            resume_text,
            top_k=top_k
        )
    )

    ranked_jobs = rank_jobs(
        predictions,
        top_k=top_k
    )

    summary = (
        build_candidate_summary(
            ranked_jobs
        )
    )

    return {

        "summary":
            summary,

        "jobs":
            ranked_jobs
    }


# ============================================================
# Interview Formatting
# ============================================================

def format_interview_questions(
    questions: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:

    if not questions:

        return []

    formatted = []

    for position, question in enumerate(
        questions[:top_k],
        start=1
    ):

        score = float(
            question.get(
                "score",
                0.0
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
                )
        })

    return formatted
