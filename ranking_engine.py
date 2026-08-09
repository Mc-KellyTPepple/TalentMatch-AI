"""
TalentMatch AI
Production Ranking and Explanation Layer

Responsibilities
----------------
- Rank AI job predictions
- Combine semantic and lexical evidence
- Remove duplicate job listings
- Normalize compatibility scores
- Generate evidence-based match explanations
- Identify matched resume/job terms
- Identify potential skill/keyword gaps
- Produce candidate-friendly summaries
- Produce employer-friendly screening information
- Return lightweight JSON-compatible dictionaries

Important
---------
The TalentMatch score is an AI compatibility score.

It is NOT:
- probability of employment
- probability of interview
- probability of hiring
- a replacement for human recruitment decisions

Designed for:
- Render Free
- approximately 512 MB RAM
- CPU inference
- lightweight FastAPI deployment
"""

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import re


# ================================================================
# CONFIGURATION
# ================================================================

MIN_SCORE = 0.0
MAX_SCORE = 1.0

# Retrieve extra candidates so duplicate listings can be removed
# without unnecessarily reducing the final result count.

CANDIDATE_MULTIPLIER = 3

MAX_CANDIDATES = 50

MAX_MATCHED_TERMS = 8
MAX_MISSING_TERMS = 8

# Maximum amount of text used when constructing duplicate
# fingerprints. This prevents unexpectedly large memory usage.

MAX_FINGERPRINT_TEXT = 3000

# Minimum useful token length for explanation analysis.

MIN_TERM_LENGTH = 3

# Common words that should not be presented as meaningful
# matching evidence.

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "you",
    "your",
    "our",
    "their",
    "they",
    "will",
    "have",
    "has",
    "had",
    "been",
    "being",
    "was",
    "were",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "must",
    "not",
    "but",
    "all",
    "any",
    "some",
    "more",
    "other",
    "into",
    "over",
    "under",
    "than",
    "then",
    "also",
    "such",
    "using",
    "used",
    "use",
    "work",
    "working",
    "job",
    "role",
    "position",
    "company",
    "team",
    "including",
    "required",
    "requirements",
    "preferred",
    "responsibilities",
    "experience",
    "years",
}


# ================================================================
# SCORE UTILITIES
# ================================================================

def clamp_score(score: float) -> float:
    """
    Clamp a score into the supported 0-1 range.
    """

    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0

    if value != value:
        value = 0.0

    return max(
        MIN_SCORE,
        min(MAX_SCORE, value),
    )


def score_to_percentage(score: float) -> float:
    """
    Convert a 0-1 compatibility score to a percentage.

    Example:
        0.4452 -> 44.52
    """

    return round(
        clamp_score(score) * 100.0,
        2,
    )


def get_match_level(score: float) -> str:
    """
    Return a human-readable compatibility category.

    These labels describe similarity only.
    They do not represent hiring probability.
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
    Normalize text for comparison and duplicate detection.

    This does NOT modify the original text returned to users.
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


def _tokenize_text(value: Any) -> List[str]:
    """
    Convert text into lightweight normalized tokens.

    No additional NLP model is required.
    """

    text = _normalize_text(value)

    if not text:
        return []

    tokens = re.findall(
        r"[a-z0-9][a-z0-9+#.\-]*",
        text,
    )

    cleaned = []

    for token in tokens:

        token = token.strip(
            ".-"
        )

        if len(token) < MIN_TERM_LENGTH:
            continue

        if token in STOPWORDS:
            continue

        cleaned.append(token)

    return cleaned


def _unique_preserve_order(
    values: List[str],
) -> List[str]:
    """
    Remove duplicate values while preserving their order.
    """

    seen = set()
    result = []

    for value in values:

        normalized = _normalize_text(
            value
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(value)

    return result


# ================================================================
# GENERIC JOB FIELD ACCESS
# ================================================================

def _get_first_value(
    job: Dict[str, Any],
    keys: Tuple[str, ...],
) -> str:
    """
    Safely retrieve the first non-empty field.
    """

    for key in keys:

        value = job.get(key)

        if value is None:
            continue

        if isinstance(value, str):

            if value.strip():
                return value.strip()

        else:

            text = str(value).strip()

            if text:
                return text

    return ""


# ================================================================
# JOB TITLE
# ================================================================

def _get_job_title(
    job: Dict[str, Any]
) -> str:

    return _get_first_value(
        job,
        (
            "title",
            "job_title",
            "position",
            "role",
            "name",
        ),
    )


# ================================================================
# COMPANY
# ================================================================

def _get_company(
    job: Dict[str, Any]
) -> str:

    return _get_first_value(
        job,
        (
            "company",
            "company_name",
            "employer",
            "organization",
            "organisation",
        ),
    )


# ================================================================
# LOCATION
# ================================================================

def _get_location(
    job: Dict[str, Any]
) -> str:

    return _get_first_value(
        job,
        (
            "location",
            "city",
            "country",
            "job_location",
        ),
    )


# ================================================================
# JOB CONTENT
# ================================================================

def _get_job_description(
    job: Dict[str, Any]
) -> str:

    return _get_first_value(
        job,
        (
            "description",
            "Description",
            "job_description",
        ),
    )


def _get_job_requirements(
    job: Dict[str, Any]
) -> str:

    return _get_first_value(
        job,
        (
            "requirements",
            "Requirements",
            "qualifications",
            "qualification",
        ),
    )


def _get_job_benefits(
    job: Dict[str, Any]
) -> str:

    return _get_first_value(
        job,
        (
            "benefits",
            "Benefits",
            "perks",
        ),
    )


def _get_job_category(
    job: Dict[str, Any]
) -> str:

    return _get_first_value(
        job,
        (
            "category",
            "Category",
        ),
    )


# ================================================================
# DUPLICATE KEY
# ================================================================

def build_duplicate_key(
    job: Dict[str, Any]
) -> str:
    """
    Build a conservative duplicate fingerprint.

    Priority:

    1. Stable job ID.
    2. Exact normalized job content.
    3. Conservative title/company/location/content fingerprint.

    We deliberately do NOT use only:

        company + title

    because an employer may legitimately advertise multiple
    openings with the same title.
    """

    # ------------------------------------------------------------
    # Stable identifier
    # ------------------------------------------------------------

    for key in (
        "job_id",
        "id",
        "listing_id",
        "jobid",
    ):

        value = job.get(key)

        if value is None:
            continue

        normalized = _normalize_text(
            value
        )

        if normalized:

            return (
                "id:"
                + normalized
            )

    # ------------------------------------------------------------
    # Extract fields
    # ------------------------------------------------------------

    title = _normalize_text(
        _get_job_title(job)
    )

    company = _normalize_text(
        _get_company(job)
    )

    location = _normalize_text(
        _get_location(job)
    )

    description = _normalize_text(
        _get_job_description(job)
    )

    requirements = _normalize_text(
        _get_job_requirements(job)
    )

    # ------------------------------------------------------------
    # Exact content fingerprint
    # ------------------------------------------------------------

    content_parts = [
        title,
        company,
        location,
        description[
            :MAX_FINGERPRINT_TEXT
        ],
        requirements[
            :MAX_FINGERPRINT_TEXT
        ],
    ]

    content = "|".join(
        content_parts
    )

    if content.strip("|"):

        digest = hashlib.sha1(
            content.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()

        return (
            "content:"
            + digest
        )

    return ""


# ================================================================
# EXPLANATION TERM EXTRACTION
# ================================================================

def _extract_job_terms(
    job: Dict[str, Any]
) -> List[str]:
    """
    Extract meaningful terms from the job.

    Requirements are given slightly more importance by appearing
    first in the term collection.
    """

    requirements = _get_job_requirements(
        job
    )

    description = _get_job_description(
        job
    )

    category = _get_job_category(
        job
    )

    title = _get_job_title(
        job
    )

    combined = " ".join(
        [
            requirements,
            requirements,
            title,
            category,
            description,
        ]
    )

    return _unique_preserve_order(
        _tokenize_text(combined)
    )


def _extract_resume_terms(
    resume_text: str
) -> List[str]:
    """
    Extract meaningful terms from the candidate resume.
    """

    return _unique_preserve_order(
        _tokenize_text(
            resume_text
        )
    )


def _derive_match_terms(
    resume_text: str,
    job: Dict[str, Any],
) -> List[str]:
    """
    Find meaningful terms present in both the resume and job.

    This is intentionally lightweight and deterministic.
    """

    resume_terms = set(
        _extract_resume_terms(
            resume_text
        )
    )

    job_terms = _extract_job_terms(
        job
    )

    matched = []

    for term in job_terms:

        if term in resume_terms:

            matched.append(term)

    return _unique_preserve_order(
        matched
    )[
        :MAX_MATCHED_TERMS
    ]


def _derive_missing_terms(
    resume_text: str,
    job: Dict[str, Any],
) -> List[str]:
    """
    Identify potentially important job terms absent from the
    resume.

    These are presented as potential gaps, not claims that the
    candidate lacks the underlying skill.
    """

    resume_terms = set(
        _extract_resume_terms(
            resume_text
        )
    )

    requirements = _get_job_requirements(
        job
    )

    title = _get_job_title(
        job
    )

    requirement_terms = _unique_preserve_order(
        _tokenize_text(
            requirements
        )
    )

    title_terms = _unique_preserve_order(
        _tokenize_text(
            title
        )
    )

    candidates = (
        requirement_terms
        + title_terms
    )

    missing = []

    for term in candidates:

        if term in resume_terms:
            continue

        if term in missing:
            continue

        missing.append(term)

        if len(missing) >= MAX_MISSING_TERMS:
            break

    return missing


# ================================================================
# SCORE EXPLANATION
# ================================================================

def explain_score(
    semantic_score: float,
    tfidf_score: float,
    final_score: float,
    matched_terms: Optional[List[str]] = None,
    missing_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build an evidence-based explanation.

    The explanation describes similarity signals only.
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

    matched_terms = _unique_preserve_order(
        list(
            matched_terms or []
        )
    )[
        :MAX_MATCHED_TERMS
    ]

    missing_terms = _unique_preserve_order(
        list(
            missing_terms or []
        )
    )[
        :MAX_MISSING_TERMS
    ]

    strengths = []

    # ------------------------------------------------------------
    # Semantic evidence
    # ------------------------------------------------------------

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
            "Limited semantic alignment was detected."
        )

    # ------------------------------------------------------------
    # Keyword evidence
    # ------------------------------------------------------------

    if tfidf_percentage >= 60:

        strengths.append(
            "Strong overlap with job terminology."
        )

    elif tfidf_percentage >= 35:

        strengths.append(
            "Some relevant job terminology is present."
        )

    else:

        strengths.append(
            "Limited direct keyword overlap was detected."
        )

    # ------------------------------------------------------------
    # Explicit matching terms
    # ------------------------------------------------------------

    if matched_terms:

        strengths.append(
            "Shared terms include: "
            + ", ".join(
                matched_terms
            )
            + "."
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

        "score_interpretation":
            (
                "Compatibility score based on semantic "
                "and keyword similarity. It is not a "
                "probability of employment, interview, "
                "or hiring."
            ),
    }


# ================================================================
# SINGLE JOB
# ================================================================

def rank_job(
    job: Dict[str, Any],
    resume_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert one prediction-engine result into a clean,
    employer/candidate-friendly result.
    """

    if not isinstance(
        job,
        dict,
    ):

        raise TypeError(
            "Each job prediction must be a dictionary."
        )

    # ------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------

    try:

        semantic_score = float(
            job.get(
                "semantic_score",
                job.get(
                    "semantic",
                    0.0,
                ),
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        semantic_score = 0.0

    try:

        tfidf_score = float(
            job.get(
                "tfidf_score",
                job.get(
                    "keyword_score",
                    0.0,
                ),
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        tfidf_score = 0.0

    try:

        final_score = float(
            job.get(
                "score",
                job.get(
                    "final_score",
                    0.0,
                ),
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        final_score = 0.0

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    matched_terms = job.get(
        "matched_terms",
        [],
    )

    potential_gaps = job.get(
        "potential_gaps",
        [],
    )

    if resume_text:

        derived_matched = _derive_match_terms(
            resume_text,
            job,
        )

        derived_missing = _derive_missing_terms(
            resume_text,
            job,
        )

        if derived_matched:
            matched_terms = derived_matched

        if derived_missing:
            potential_gaps = derived_missing

    # ------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------

    explanation = explain_score(

        semantic_score=semantic_score,

        tfidf_score=tfidf_score,

        final_score=final_score,

        matched_terms=matched_terms,

        missing_terms=potential_gaps,
    )

    # ------------------------------------------------------------
    # Job fields
    # ------------------------------------------------------------

    category = _get_job_category(
        job
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

    description = _get_job_description(
        job
    )

    requirements = _get_job_requirements(
        job
    )

    benefits = _get_job_benefits(
        job
    )

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

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

        "score_interpretation":
            explanation[
                "score_interpretation"
            ],
    }

    # ------------------------------------------------------------
    # Preserve stable identifiers
    # ------------------------------------------------------------

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

            result[key] = value

            break

    return result


# ================================================================
# DUPLICATION
# ================================================================

def deduplicate_jobs(
    jobs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate job listings.

    Jobs are expected to already be ranked, so the first occurrence
    is retained.

    The function does not modify the original job dictionaries.
    """

    if not jobs:
        return []

    unique = []

    seen = set()

    for job in jobs:

        key = build_duplicate_key(
            job
        )

        # If there is no usable identity/content key, retain the
        # listing rather than accidentally deleting it.

        if not key:

            unique.append(
                job
            )

            continue

        if key in seen:

            continue

        seen.add(key)

        unique.append(
            job
        )

    return unique


# ================================================================
# RANK MULTIPLE JOBS
# ================================================================

def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = 10,
    resume_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rank, explain and deduplicate job predictions.
    """

    if not jobs:
        return []

    try:

        requested_k = max(
            1,
            int(top_k),
        )

    except (
        TypeError,
        ValueError,
    ):

        requested_k = 10

    # ------------------------------------------------------------
    # Rank raw predictions
    # ------------------------------------------------------------

    ranked = [
        rank_job(
            job,
            resume_text=resume_text,
        )
        for job in jobs
    ]

    # ------------------------------------------------------------
    # Sort by compatibility
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Remove duplicates AFTER sorting
    #
    # This guarantees that the strongest version of a duplicate
    # listing survives.
    # ------------------------------------------------------------

    ranked = deduplicate_jobs(
        ranked
    )

    # ------------------------------------------------------------
    # Final top-K
    # ------------------------------------------------------------

    ranked = ranked[
        :requested_k
    ]

    # ------------------------------------------------------------
    # Assign final ranks
    # ------------------------------------------------------------

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
    """
    Build a lightweight candidate-facing summary.
    """

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

            "score_interpretation":
                (
                    "No compatible job matches were found "
                    "in the analyzed listings."
                ),
        }

    scores = []

    for job in ranked_jobs:

        try:

            score = float(
                job.get(
                    "match_score",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        scores.append(
            clamp_score(
                score / 100.0
                if score > 1
                else score
            )
            * 100.0
        )

    best = ranked_jobs[0]

    average_score = round(
        sum(scores)
        / len(scores),
        2,
    )

    strongest_match = (
        best.get("title")
        or best.get("category")
        or "Top recommended job"
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
            strongest_match,

        "score_interpretation":
            (
                "Compatibility score based on semantic "
                "and keyword similarity. It is not a "
                "probability of employment or hiring."
            ),
    }


# ================================================================
# EMPLOYER SUMMARY
# ================================================================

def build_employer_summary(
    ranked_jobs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Produce lightweight employer-facing information.

    This prepares the ranking layer for a future employer
    dashboard without requiring another ML model.
    """

    if not ranked_jobs:

        return {

            "candidates_found":
                0,

            "top_candidate_score":
                0,

            "top_candidate_level":
                "No Match",

            "screening_note":
                "No matching candidate information available.",
        }

    top_job = ranked_jobs[0]

    return {

        "candidates_found":
            len(ranked_jobs),

        "top_candidate_score":
            top_job.get(
                "match_score",
                0,
            ),

        "top_candidate_level":
            top_job.get(
                "match_level",
                "Unknown",
            ),

        "top_match_strengths":
            top_job.get(
                "strengths",
                [],
            ),

        "top_match_gaps":
            top_job.get(
                "potential_gaps",
                [],
            ),

        "screening_note":
            (
                "Use this ranking as decision-support evidence. "
                "Human review should consider qualifications, "
                "experience, interview performance and other "
                "relevant recruitment criteria."
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
    """
    Main interface used by the FastAPI application.
    """

    if resume_text is None:

        raise ValueError(
            "Resume text cannot be None."
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

    except (
        TypeError,
        ValueError,
    ):

        requested_k = 10

    # ------------------------------------------------------------
    # Retrieve additional candidates.
    #
    # This provides enough room for duplicate removal while
    # preserving the requested final result count.
    # ------------------------------------------------------------

    candidate_k = min(
        max(
            requested_k
            * CANDIDATE_MULTIPLIER,
            requested_k,
        ),
        MAX_CANDIDATES,
    )

    # ------------------------------------------------------------
    # Prediction engine
    # ------------------------------------------------------------

    predictions = (
        prediction_engine.hybrid_job_search(
            resume_text,
            top_k=candidate_k,
        )
    )

    if predictions is None:
        predictions = []

    # ------------------------------------------------------------
    # Ranking + explanations + deduplication
    # ------------------------------------------------------------

    ranked_jobs = rank_jobs(
        predictions,
        top_k=requested_k,
        resume_text=resume_text,
    )

    # ------------------------------------------------------------
    # Candidate summary
    # ------------------------------------------------------------

    summary = build_candidate_summary(
        ranked_jobs
    )

    # ------------------------------------------------------------
    # Employer summary
    # ------------------------------------------------------------

    employer_summary = (
        build_employer_summary(
            ranked_jobs
        )
    )

    # ------------------------------------------------------------
    # Final API payload
    # ------------------------------------------------------------

    return {

        "summary":
            summary,

        "employer_summary":
            employer_summary,

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
    """
    Convert prediction-engine interview results into
    API-friendly JSON.
    """

    if not questions:
        return []

    try:

        requested_k = max(
            1,
            int(top_k),
        )

    except (
        TypeError,
        ValueError,
    ):

        requested_k = 10

    formatted = []

    for position, question in enumerate(
        questions[
            :requested_k
        ],
        start=1,
    ):

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


# ================================================================
# PUBLIC API
# ================================================================

__all__ = [
    "clamp_score",
    "score_to_percentage",
    "get_match_level",
    "build_duplicate_key",
    "deduplicate_jobs",
    "explain_score",
    "rank_job",
    "rank_jobs",
    "build_candidate_summary",
    "build_employer_summary",
    "analyze_jobs",
    "format_interview_questions",
]
