"""
TalentMatch AI
Ranking / Response Formatting Layer
"""

from predict import engine


# ===============================================================
# JOB ANALYSIS
# ===============================================================

def analyze_jobs(
    prediction_engine,
    resume_text,
    top_k=10,
):
    """
    Run hybrid semantic + TF-IDF job matching.
    """

    if prediction_engine is None:

        raise ValueError(
            "Prediction engine is required."
        )

    if not resume_text:

        raise ValueError(
            "Resume text cannot be empty."
        )

    jobs = prediction_engine.hybrid_job_search(
        resume_text=resume_text,
        top_k=top_k,
    )

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------

    if jobs:

        semantic_scores = [
            float(
                job.get(
                    "semantic_score",
                    0.0,
                )
            )
            for job in jobs
        ]

        tfidf_scores = [
            float(
                job.get(
                    "tfidf_score",
                    0.0,
                )
            )
            for job in jobs
        ]

        final_scores = [
            float(
                job.get(
                    "score",
                    0.0,
                )
            )
            for job in jobs
        ]

        summary = {

            "jobs_found":
                len(jobs),

            "best_match_score":
                max(final_scores),

            "average_match_score":
                sum(final_scores)
                / len(final_scores),

            "best_semantic_score":
                max(semantic_scores),

            "best_tfidf_score":
                max(tfidf_scores),
        }

    else:

        summary = {

            "jobs_found":
                0,

            "best_match_score":
                0.0,

            "average_match_score":
                0.0,

            "best_semantic_score":
                0.0,

            "best_tfidf_score":
                0.0,
        }

    return {

        "jobs":
            jobs,

        "summary":
            summary,
    }


# ===============================================================
# INTERVIEW QUESTION FORMATTING
# ===============================================================

def format_interview_questions(
    questions,
    top_k=5,
):

    if not questions:

        return []

    limit = min(
        max(
            1,
            int(top_k),
        ),
        len(questions),
    )

    formatted = []

    for question in questions[:limit]:

        formatted.append({

            "question":
                question.get(
                    "question",
                    "",
                ),

            "answer":
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

            "score":
                round(
                    float(
                        question.get(
                            "score",
                            0.0,
                        )
                    ),
                    4,
                ),
        })

    return formatted
