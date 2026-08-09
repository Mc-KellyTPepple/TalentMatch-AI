"""
TalentMatch AI
Production Ranking / Response Formatting Layer

Responsibilities
-----------------
1. Receive predictions from PredictionEngine.
2. Remove duplicate job listings.
3. Normalize compatibility scores.
4. Generate lightweight explanations.
5. Identify matched and missing terms.
6. Produce candidate-friendly summaries.
7. Produce employer-friendly job information.
8. Preserve the response contract expected by app.py.

IMPORTANT
---------
This module performs NO model inference.

It does NOT:
- load MiniLM
- load TF-IDF
- calculate embeddings
- calculate semantic similarity
- calculate TF-IDF similarity

Those operations remain inside predict.py.

Designed for:
- Render Free
- 512 MB RAM
- CPU inference
"""

from typing import Any, Dict, List, Optional
import re


# ================================================================
# CONFIGURATION
# ================================================================

MIN_SCORE = 0.0
MAX_SCORE = 1.0

MAX_MATCHED_TERMS = 8
MAX_MISSING_TERMS = 8

DEFAULT_TOP_K = 10


# ================================================================
# SCORE UTILITIES
# ================================================================

def clamp_score(score: Any) -> float:
    """
    Safely constrain a similarity score to [0, 1].
    """

    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0

    if value != value:  # NaN
        value = 0.0

    return max(
        MIN_SCORE,
        min(MAX_SCORE, value)
    )


def score_to_percentage(score: Any) -> float:
    """
    Convert a normalized similarity score to a percentage.

    Example:
        0.4452 -> 44.52
    """

    return round(
        clamp_score(score) * 100.0,
        2
    )


def get_match_level(score: Any) -> str:
    """
    Human-readable compatibility label.

    These labels describe similarity only.
    They do NOT represent hiring probability.
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
# SAFE TEXT
# ================================================================

def _safe_text(value: Any) -> str:
    """
    Convert arbitrary values safely to text.
    """

    if value is None:
        return ""

    try:
        if value != value:
            return ""
    except Exception:
        pass

    return str(value).strip()


# ================================================================
# TEXT NORMALIZATION
# ================================================================

def _normalize_text(value: Any) -> str:
    """
    Normalize text for comparison/deduplication only.

    User-visible text is never modified by this function.
    """

    text = _safe_text(value).lower()

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ================================================================
# TOKENIZATION
# ================================================================

def _tokenize(value: Any) -> List[str]:
    """
    Lightweight tokenization used only for explanations.

    This is deliberately independent of the trained TF-IDF
    vectorizer so that ranking_engine.py does not introduce
    another model/artifact dependency.
    """

    text = _normalize_text(value)

    if not text:
        return []

    tokens = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,30}\b",
        text
    )

    return tokens


# ================================================================
# STOP WORDS
# ================================================================

_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "are",
    "was",
    "were",
    "will",
    "your",
    "you",
    "our",
    "their",
    "they",
    "job",
    "role",
    "work",
    "working",
    "years",
    "year",
    "required",
    "requirements",
    "experience",
    "candidate",
    "candidates",
    "ability",
    "including",
    "using",
    "used",
    "into",
    "within",
    "about",
    "through",
    "over",
    "under",
    "such",
    "also",
    "may",
    "must",
    "can",
    "should",
    "would",
    "more",
    "than",
    "other",
    "all",
    "any",
    "not",
    "but",
    "who",
    "what",
    "how",
    "where",
    "when",
    "why",
    "its",
    "our",
    "their",
    "an",
    "a",
    "of",
    "to",
    "in",
    "on",
    "at",
    "as",
    "by",
    "or",
}


# ================================================================
# JOB FIELDS
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
            return _safe_text(value)

    return ""


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
            return _safe_text(value)

    return ""


def _get_location(job: Dict[str, Any]) -> str:

    for key in (
        "location",
        "city",
        "country",
        "job_location",
    ):

        value = job.get(key)

        if value:
            return _safe_text(value)

    return ""


def _get_field(
    job: Dict[str, Any],
    name: str
) -> str:

    value = job.get(name)

    if value:
        return _safe_text(value)

    # Defensive support for datasets with capitalized fields.
    value = job.get(
        name.capitalize()
    )

    if value:
        return _safe_text(value)

    return ""


# ================================================================
# DUPLICATE KEY
# ================================================================

def build_duplicate_key(
    job: Dict[str, Any]
) -> str:
    """
    Build a conservative duplicate key.

    Priority:
        1. Stable job identifier.
        2. Company + title + location + description.
        3. Title + location + requirements.

    We deliberately do NOT deduplicate by title alone.
    """

    # ------------------------------------------------------------
    # Stable ID
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

            normalized_id = _normalize_text(
                value
            )

            if normalized_id:
                return "id:" + normalized_id

    # ------------------------------------------------------------
    # Main identity fields
    # ------------------------------------------------------------

    company = _normalize_text(
        _get_company(job)
    )

    title = _normalize_text(
        _get_job_title(job)
    )

    location = _normalize_text(
        _get_location(job)
    )

    description = _normalize_text(
        _get_field(
            job,
            "description"
        )
    )

    requirements = _normalize_text(
        _get_field(
            job,
            "requirements"
        )
    )

    # ------------------------------------------------------------
    # Strong duplicate signature
    # ------------------------------------------------------------

    if company or title or location:

        return "|".join(
            (
                company,
                title,
                location,
                description[:500],
                requirements[:500],
            )
        )

    return "|".join(
        (
            description[:500],
            requirements[:500],
        )
    )


# ================================================================
# TERM EXTRACTION
# ================================================================

def _extract_candidate_terms(
    resume_text: str
) -> List[str]:

    tokens = _tokenize(
        resume_text
    )

    result = []

    seen = set()

    for token in tokens:

        token_lower = token.lower()

        if token_lower in _STOP_WORDS:
            continue

        if len(token_lower) < 3:
            continue

        if token_lower in seen:
            continue

        seen.add(token_lower)

        result.append(
            token_lower
        )

    return result


def _extract_job_terms(
    job: Dict[str, Any]
) -> List[str]:

    parts = [

        _get_job_title(job),

        _get_field(
            job,
            "category"
        ),

        _get_field(
            job,
            "description"
        ),

        _get_field(
            job,
            "requirements"
        ),
    ]

    tokens = []

    seen = set()

    for part in parts:

        for token in _tokenize(part):

            token_lower = token.lower()

            if token_lower in _STOP_WORDS:
                continue

            if len(token_lower) < 3:
                continue

            if token_lower in seen:
                continue

            seen.add(token_lower)

            tokens.append(
                token_lower
            )

    return tokens


# ================================================================
# MATCHED / MISSING TERMS
# ================================================================

def extract_term_evidence(
    resume_text: str,
    job: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Produce lightweight evidence from the resume/job text.

    This does NOT alter the actual AI score.

    It is explanatory evidence only.
    """

    resume_terms = set(
        _extract_candidate_terms(
            resume_text
        )
    )

    job_terms = _extract_job_terms(
        job
    )

    matched = []

    seen = set()

    for term in job_terms:

        if term in resume_terms:

            if term not in seen:

                matched.append(term)
                seen.add(term)

        if len(matched) >= MAX_MATCHED_TERMS:
            break

    missing = []

    seen_missing = set()

    for term in job_terms:

        if term not in resume_terms:

            if term in seen_missing:
                continue

            missing.append(term)
            seen_missing.add(term)

        if len(missing) >= MAX_MISSING_TERMS:
            break

    return {
        "matched_terms": matched,
        "missing_terms": missing,
    }


# ================================================================
# SCORE EXPLANATION
# ================================================================

def explain_score(
    semantic_score: Any,
    tfidf_score: Any,
    final_score: Any,
    matched_terms: Optional[List[str]] = None,
    missing_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:

    semantic = clamp_score(
        semantic_score
    )

    tfidf = clamp_score(
        tfidf_score
    )

    final = clamp_score(
        final_score
    )

    semantic_percentage = score_to_percentage(
        semantic
    )

    tfidf_percentage = score_to_percentage(
        tfidf
    )

    final_percentage = score_to_percentage(
        final
    )

    matched_terms = list(
        matched_terms or []
    )[:MAX_MATCHED_TERMS]

    missing_terms = list(
        missing_terms or []
    )[:MAX_MISSING_TERMS]

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
            "Limited exact keyword overlap was detected."
        )

    # ------------------------------------------------------------
    # Term evidence
    # ------------------------------------------------------------

    if matched_terms:

        strengths.append(
            "The resume contains terms relevant to this role."
        )

    return {

        "match_score":
            final_percentage,

        "semantic_score":
            semantic_percentage,

        "keyword_score":
            tfidf_percentage,

        "match_level":
            get_match_level(final),

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
    job: Dict[str, Any],
    resume_text: str = "",
) -> Dict[str, Any]:

    if not isinstance(
        job,
        dict
    ):

        job = {}

    semantic_score = float(
        job.get(
            "semantic_score",
            job.get(
                "semantic",
                0.0
            )
        )
        or 0.0
    )

    tfidf_score = float(
        job.get(
            "tfidf_score",
            job.get(
                "keyword_score",
                0.0
            )
        )
        or 0.0
    )

    final_score = float(
        job.get(
            "score",
            job.get(
                "final_score",
                0.0
            )
        )
        or 0.0
    )

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    evidence = extract_term_evidence(
        resume_text,
        job
    )

    matched_terms = job.get(
        "matched_terms"
    )

    if not isinstance(
        matched_terms,
        list
    ):

        matched_terms = evidence[
            "matched_terms"
        ]

    missing_terms = job.get(
        "potential_gaps"
    )

    if not isinstance(
        missing_terms,
        list
    ):

        missing_terms = evidence[
            "missing_terms"
        ]

    explanation = explain_score(

        semantic_score=semantic_score,

        tfidf_score=tfidf_score,

        final_score=final_score,

        matched_terms=matched_terms,

        missing_terms=missing_terms,
    )

    # ------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------

    title = _get_job_title(
        job
    )

    company = _get_company(
        job
    )

    location = _get_location(
        job
    )

    category = _get_field(
        job,
        "category"
    )

    description = _get_field(
        job,
        "description"
    )

    requirements = _get_field(
        job,
        "requirements"
    )

    benefits = _get_field(
        job,
        "benefits"
    )

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    result = {

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

        "title":
            title,

        "category":
            category,

        "company":
            company,

        "location":
            location,

        "description":
            description,

        "requirements":
            requirements,

        "benefits":
            benefits,

        "strengths":
            explanation[
                "strengths"
            ],

        "matched_terms":
            explanation[
                "matched_terms"
            ],

        "potential_gaps":
            explanation[
                "potential_gaps"
            ],
    }

    # ------------------------------------------------------------
    # Preserve stable identifier
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
# DEDUPLICATION
# ================================================================

def deduplicate_jobs(
    jobs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove duplicate listings.

    Jobs should already be ordered by AI compatibility.
    Therefore the first occurrence is retained.
    """

    if not jobs:
        return []

    unique = []

    seen = set()

    for job in jobs:

        key = build_duplicate_key(
            job
        )

        # If there is genuinely no identity information,
        # preserve the result rather than accidentally deleting it.
        if not key:

            unique.append(job)
            continue

        if key in seen:
            continue

        seen.add(key)

        unique.append(job)

    return unique


# ================================================================
# RANK MULTIPLE JOBS
# ================================================================

def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    resume_text: str = "",
) -> List[Dict[str, Any]]:

    if not jobs:
        return []

    requested_k = max(
        1,
        int(top_k)
    )

    # ------------------------------------------------------------
    # Format jobs
    # ------------------------------------------------------------

    ranked = [

        rank_job(
            job,
            resume_text=resume_text
        )

        for job in jobs

        if isinstance(
            job,
            dict
        )
    ]

    if not ranked:
        return []

    # ------------------------------------------------------------
    # Sort by final compatibility
    # ------------------------------------------------------------

    ranked.sort(

        key=lambda item: (

            float(
                item.get(
                    "match_score",
                    0.0
                )
            ),

            float(
                item.get(
                    "semantic_score",
                    0.0
                )
            ),

            float(
                item.get(
                    "keyword_score",
                    0.0
                )
            ),
        ),

        reverse=True
    )

    # ------------------------------------------------------------
    # Deduplicate AFTER sorting.
    #
    # This guarantees that the highest scoring duplicate
    # survives.
    # ------------------------------------------------------------

    ranked = deduplicate_jobs(
        ranked
    )

    # ------------------------------------------------------------
    # Final top K
    # ------------------------------------------------------------

    ranked = ranked[
        :requested_k
    ]

    # ------------------------------------------------------------
    # Ranking positions
    # ------------------------------------------------------------

    for position, job in enumerate(
        ranked,
        start=1
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

            "jobs_analyzed":
                0,

            "unique_jobs_returned":
                0,

            "best_match_score":
                0,

            "best_match_level":
                "No Match",

            "average_match_score":
                0,

            "strongest_match":
                None,

            "score_interpretation":
                (
                    "No compatible job matches were "
                    "returned."
                ),
        }

    scores = [

        float(
            job.get(
                "match_score",
                0.0
            )
        )

        for job in ranked_jobs
    ]

    best = ranked_jobs[0]

    average_score = round(
        sum(scores) / len(scores),
        2
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
                0
            ),

        "best_match_level":
            best.get(
                "match_level",
                "Unknown"
            ),

        "average_match_score":
            average_score,

        "strongest_match":
            strongest_match,

        "score_interpretation":
            (
                "Compatibility score based on semantic "
                "and keyword similarity. It is not a "
                "probability of employment, interview, "
                "or hiring."
            ),
    }


# ================================================================
# MAIN RANKING INTERFACE
# ================================================================

def analyze_jobs(
    prediction_engine,
    resume_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """
    Main interface used by app.py.

    IMPORTANT:
    This function preserves the original application contract:

        {
            "summary": {...},
            "jobs": [...]
        }

    No additional model loading is performed here.
    """

    if prediction_engine is None:

        raise ValueError(
            "Prediction engine is required."
        )

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

    requested_k = max(
        1,
        int(top_k)
    )

    # ------------------------------------------------------------
    # Ask PredictionEngine for matches.
    #
    # IMPORTANT:
    # Do NOT calculate another embedding here.
    # ------------------------------------------------------------

    predictions = (
        prediction_engine.hybrid_job_search(
            resume_text=resume_text,
            top_k=requested_k,
        )
    )

    if predictions is None:
        predictions = []

    if not isinstance(
        predictions,
        list
    ):

        raise RuntimeError(
            "Prediction engine returned an invalid "
            "job prediction response."
        )

    # ------------------------------------------------------------
    # Lightweight ranking / explanation
    # ------------------------------------------------------------

    ranked_jobs = rank_jobs(

        jobs=predictions,

        top_k=requested_k,

        resume_text=resume_text,
    )

    # ------------------------------------------------------------
    # Candidate summary
    # ------------------------------------------------------------

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

    limit = min(
        max(
            1,
            int(top_k)
        ),
        len(questions)
    )

    formatted = []

    for position, question in enumerate(
        questions[:limit],
        start=1
    ):

        if not isinstance(
            question,
            dict
        ):
            continue

        score = clamp_score(
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
                _safe_text(
                    question.get(
                        "question",
                        ""
                    )
                ),

            "ideal_answer":
                _safe_text(
                    question.get(
                        "answer",
                        ""
                    )
                ),

            "role":
                _safe_text(
                    question.get(
                        "role",
                        ""
                    )
                ),

            "category":
                _safe_text(
                    question.get(
                        "category",
                        ""
                    )
                ),

            "difficulty":
                _safe_text(
                    question.get(
                        "difficulty",
                        ""
                    )
                ),

            "experience":
                _safe_text(
                    question.get(
                        "experience",
                        ""
                    )
                ),
        })

    return formatted
