"""
TalentMatch AI
Job Ranking and Result Formatting Engine

Responsibilities
----------------
• Convert raw prediction scores to percentages
• Determine match levels
• Explain matching signals
• Rank jobs
• Build candidate summary
• Format interview questions

IMPORTANT
---------
This module DOES NOT load ML models.

All expensive AI operations happen inside:

    PredictionEngine.hybrid_job_search()
    PredictionEngine.interview_questions()

Designed for:
    Render Free
    512 MB RAM
    CPU inference

Diagnostic mode
---------------
Every major operation is timed and printed to the
Render log so slow prediction operations can be
identified precisely.
"""

from typing import Any, Dict, List
import time
import traceback


# ============================================================
# DIAGNOSTIC LOGGER
# ============================================================

_MODULE_START = time.perf_counter()


def _log(message: str, *args: Any) -> None:
    """
    Print a timestamped diagnostic message.

    Uses f-string formatting correctly so Render logs
    contain actual values rather than literal formatting
    placeholders.
    """

    elapsed = time.perf_counter() - _MODULE_START

    if args:
        try:
            message = message.format(*args)
        except Exception:
            message = (
                f"{message} "
                + " ".join(str(arg) for arg in args)
            )

    print(
        f"[TalentMatch {elapsed:8.3f}s] {message}",
        flush=True,
    )


_log("ranking_engine.py imported successfully.")


# ============================================================
# SCORE UTILITIES
# ============================================================

def clamp_score(score: float) -> float:
    """
    Clamp a score into the [0, 1] interval.
    """

    try:
        value = float(score)

    except (
        TypeError,
        ValueError,
    ):
        value = 0.0

    if value != value:  # NaN
        value = 0.0

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def score_to_percentage(score: float) -> int:
    """
    Convert a normalized score in [0, 1]
    into an integer percentage.
    """

    return int(
        round(
            clamp_score(score) * 100
        )
    )


def get_match_level(score: float) -> str:
    """
    Convert a normalized score into a human-readable
    match category.
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


# ============================================================
# SCORE EXPLANATION
# ============================================================

def explain_score(
    semantic_score: float,
    tfidf_score: float,
    final_score: float,
) -> Dict[str, Any]:
    """
    Explain the semantic, keyword and final matching signals.
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
            strengths,
    }


# ============================================================
# SAFE TEXT
# ============================================================

def _safe_text(
    value: Any,
) -> str:
    """
    Convert arbitrary values into safe strings.

    Handles:
        None
        NaN
        numbers
        strings
    """

    if value is None:
        return ""

    try:

        if value != value:
            return ""

    except Exception:
        pass

    return str(value)


# ============================================================
# SAFE LIST
# ============================================================

def _safe_list(
    value: Any,
) -> List[Any]:
    """
    Ensure list-like backend fields are returned safely.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


# ============================================================
# RANK SINGLE JOB
# ============================================================

def rank_job(
    job: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert one raw PredictionEngine result into the
    frontend-ready job structure.

    Expected raw PredictionEngine fields:

        score
        semantic_score
        tfidf_score
        category
        description
        requirements
        benefits

    Additional fields are preserved where available.
    """

    if not isinstance(
        job,
        dict,
    ):

        job = {
            "category": str(job)
        }

    # --------------------------------------------------------
    # Extract scores
    # --------------------------------------------------------

    semantic_score = job.get(
        "semantic_score",
        0.0,
    )

    tfidf_score = job.get(
        "tfidf_score",
        0.0,
    )

    final_score = job.get(
        "score",
        0.0,
    )

    try:

        semantic_score = float(
            semantic_score
        )

    except (
        TypeError,
        ValueError,
    ):

        semantic_score = 0.0

    try:

        tfidf_score = float(
            tfidf_score
        )

    except (
        TypeError,
        ValueError,
    ):

        tfidf_score = 0.0

    try:

        final_score = float(
            final_score
        )

    except (
        TypeError,
        ValueError,
    ):

        final_score = 0.0

    # --------------------------------------------------------
    # Explain score
    # --------------------------------------------------------

    explanation = explain_score(
        semantic_score,
        tfidf_score,
        final_score,
    )

    # --------------------------------------------------------
    # Format result
    # --------------------------------------------------------

    return {

        "match_score":
            explanation[
                "match_score"
            ],

        "semantic_score":
            explanation[
                "semantic_score"
            ],

        "keyword_score":
            explanation[
                "keyword_score"
            ],

        "match_level":
            explanation[
                "match_level"
            ],

        "category":
            _safe_text(
                job.get(
                    "category",
                    job.get(
                        "title",
                        "",
                    ),
                )
            ),

        "title":
            _safe_text(
                job.get(
                    "title",
                    "",
                )
            ),

        "description":
            _safe_text(
                job.get(
                    "description",
                    "",
                )
            ),

        "requirements":
            _safe_text(
                job.get(
                    "requirements",
                    "",
                )
            ),

        "benefits":
            _safe_text(
                job.get(
                    "benefits",
                    "",
                )
            ),

        "strengths":
            explanation[
                "strengths"
            ],

        # ----------------------------------------------------
        # Optional skill information.
        # ----------------------------------------------------

        "matched_skills":
            _safe_list(
                job.get(
                    "matched_skills",
                    job.get(
                        "matching_skills",
                        [],
                    ),
                )
            ),

        "missing_skills":
            _safe_list(
                job.get(
                    "missing_skills",
                    job.get(
                        "skills_to_develop",
                        [],
                    ),
                )
            ),

        "skill_details":
            _safe_list(
                job.get(
                    "skill_details",
                    [],
                )
            ),
    }


# ============================================================
# RANK MULTIPLE JOBS
# ============================================================

def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rank and format multiple jobs.

    This function performs NO ML inference.
    """

    start = time.perf_counter()

    _log(
        "rank_jobs() started. Raw jobs received: {}",
        len(jobs)
        if isinstance(
            jobs,
            list,
        )
        else "invalid",
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not jobs:

        _log(
            "rank_jobs(): no jobs received."
        )

        return []

    if not isinstance(
        jobs,
        list,
    ):

        _log(
            "rank_jobs(): invalid jobs object: {}",
            type(jobs).__name__,
        )

        return []

    # --------------------------------------------------------
    # Rank each job
    # --------------------------------------------------------

    ranked = []

    for job in jobs:

        try:

            ranked.append(
                rank_job(job)
            )

        except Exception as exc:

            _log(
                "WARNING: Could not rank one job: {}",
                repr(exc),
            )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked.sort(
        key=lambda item:
            item.get(
                "match_score",
                0,
            ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    try:

        requested_top_k = max(
            1,
            int(top_k),
        )

    except (
        TypeError,
        ValueError,
    ):

        requested_top_k = 10

    ranked = ranked[
        :requested_top_k
    ]

    # --------------------------------------------------------
    # Add rank
    # --------------------------------------------------------

    for position, job in enumerate(
        ranked,
        start=1,
    ):

        job["rank"] = position

    elapsed = (
        time.perf_counter()
        - start
    )

    _log(
        "rank_jobs() completed in {:.3f}s. "
        "Returned {} jobs.",
        elapsed,
        len(ranked),
    )

    return ranked


# ============================================================
# CANDIDATE SUMMARY
# ============================================================

def build_candidate_summary(
    ranked_jobs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a lightweight summary from ranked jobs.
    """

    if not ranked_jobs:

        return {

            "jobs_analyzed":
                0,

            "best_match_score":
                0,

            "best_match_level":
                "No Match",

            "average_match_score":
                0,
        }

    scores = []

    for job in ranked_jobs:

        try:

            scores.append(
                int(
                    job.get(
                        "match_score",
                        0,
                    )
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            scores.append(0)

    best = ranked_jobs[0]

    average_score = round(
        sum(scores)
        / len(scores)
    )

    return {

        "jobs_analyzed":
            len(ranked_jobs),

        "best_match_score":
            best.get(
                "match_score",
                0,
            ),

        "best_match_level":
            best.get(
                "match_level",
                "No Match",
            ),

        "average_match_score":
            average_score,
    }


# ============================================================
# MAIN JOB ANALYSIS
# ============================================================

def analyze_jobs(
    prediction_engine,
    resume_text: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Run hybrid job search through PredictionEngine,
    then format and rank the results.

    ML inference happens ONLY inside:

        prediction_engine.hybrid_job_search()
    """

    start = time.perf_counter()

    _log(
        "=================================================="
    )

    _log(
        "analyze_jobs() STARTED"
    )

    _log(
        "Resume characters: {}",
        len(resume_text)
        if resume_text
        else 0,
    )

    _log(
        "Requested top_k: {}",
        top_k,
    )

    _log(
        "Prediction engine type: {}",
        type(
            prediction_engine
        ).__name__,
    )

    _log(
        "=================================================="
    )

    # ========================================================
    # VALIDATE RESUME
    # ========================================================

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

    # ========================================================
    # VALIDATE ENGINE
    # ========================================================

    if prediction_engine is None:

        raise RuntimeError(
            "Prediction engine is None."
        )

    if not hasattr(
        prediction_engine,
        "hybrid_job_search",
    ):

        raise RuntimeError(
            "Prediction engine does not provide "
            "hybrid_job_search()."
        )

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    _log(
        "Calling prediction_engine.hybrid_job_search()..."
    )

    hybrid_start = time.perf_counter()

    try:

        predictions = (
            prediction_engine.hybrid_job_search(
                resume_text,
                top_k=top_k,
            )
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - hybrid_start
        )

        _log(
            "hybrid_job_search() FAILED after {:.3f}s.",
            elapsed,
        )

        _log(
            "Exception type: {}",
            type(exc).__name__,
        )

        _log(
            "Exception: {}",
            repr(exc),
        )

        _log(
            "Full traceback:"
        )

        print(
            traceback.format_exc(),
            flush=True,
        )

        raise

    hybrid_elapsed = (
        time.perf_counter()
        - hybrid_start
    )

    _log(
        "hybrid_job_search() RETURNED after {:.3f}s.",
        hybrid_elapsed,
    )

    # ========================================================
    # NORMALIZE PREDICTION RESULT
    # ========================================================

    if predictions is None:

        _log(
            "WARNING: hybrid_job_search() returned None."
        )

        predictions = []

    elif not isinstance(
        predictions,
        list,
    ):

        _log(
            "WARNING: hybrid_job_search() returned {} "
            "instead of list.",
            type(predictions).__name__,
        )

        try:

            predictions = list(
                predictions
            )

        except Exception:

            predictions = []

    _log(
        "Raw prediction count: {}",
        len(predictions),
    )

    # ========================================================
    # INSPECT FIRST RESULT
    # ========================================================

    if predictions:

        first = predictions[0]

        if isinstance(
            first,
            dict,
        ):

            _log(
                "First prediction keys: {}",
                list(
                    first.keys()
                ),
            )

        else:

            _log(
                "First prediction type: {}",
                type(first).__name__,
            )

    # ========================================================
    # RANK RESULTS
    # ========================================================

    _log(
        "Starting rank_jobs()..."
    )

    ranking_start = time.perf_counter()

    ranked_jobs = rank_jobs(
        predictions,
        top_k=top_k,
    )

    ranking_elapsed = (
        time.perf_counter()
        - ranking_start
    )

    _log(
        "rank_jobs() finished in {:.3f}s.",
        ranking_elapsed,
    )

    # ========================================================
    # CANDIDATE SUMMARY
    # ========================================================

    _log(
        "Building candidate summary..."
    )

    summary_start = time.perf_counter()

    summary = (
        build_candidate_summary(
            ranked_jobs
        )
    )

    summary_elapsed = (
        time.perf_counter()
        - summary_start
    )

    _log(
        "Candidate summary completed in {:.3f}s.",
        summary_elapsed,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - start
    )

    _log(
        "=================================================="
    )

    _log(
        "analyze_jobs() COMPLETED in {:.3f}s.",
        total_elapsed,
    )

    _log(
        "Final ranked jobs: {}",
        len(ranked_jobs),
    )

    _log(
        "Best match: {}%",
        summary.get(
            "best_match_score",
            0,
        ),
    )

    _log(
        "=================================================="
    )

    return {

        "summary":
            summary,

        "jobs":
            ranked_jobs,
    }


# ============================================================
# INTERVIEW FORMATTING
# ============================================================

def format_interview_questions(
    questions: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Format raw PredictionEngine interview results.

    No ML inference occurs here.
    """

    start = time.perf_counter()

    _log(
        "format_interview_questions() started."
    )

    if not questions:

        _log(
            "No interview questions received."
        )

        return []

    try:

        limit = max(
            1,
            int(top_k),
        )

    except (
        TypeError,
        ValueError,
    ):

        limit = 5

    formatted = []

    for position, question in enumerate(
        questions[:limit],
        start=1,
    ):

        if not isinstance(
            question,
            dict,
        ):

            continue

        try:

            score = float(
                question.get(
                    "score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        formatted.append({

            "rank":
                position,

            "relevance_score":
                score_to_percentage(
                    score
                ),

            "question":
                _safe_text(
                    question.get(
                        "question",
                        "",
                    )
                ),

            "ideal_answer":
                _safe_text(
                    question.get(
                        "answer",
                        "",
                    )
                ),

            "role":
                _safe_text(
                    question.get(
                        "role",
                        "",
                    )
                ),

            "category":
                _safe_text(
                    question.get(
                        "category",
                        "",
                    )
                ),

            "difficulty":
                _safe_text(
                    question.get(
                        "difficulty",
                        "",
                    )
                ),

            "experience":
                _safe_text(
                    question.get(
                        "experience",
                        "",
                    )
                ),
        })

    elapsed = (
        time.perf_counter()
        - start
    )

    _log(
        "format_interview_questions() completed "
        "in {:.3f}s. Returned {} questions.",
        elapsed,
        len(formatted),
    )

    return formatted


# ============================================================
# MODULE READY
# ============================================================

_log(
    "ranking_engine.py ready. "
    "No ML models loaded."
)
