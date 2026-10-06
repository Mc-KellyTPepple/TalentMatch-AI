"""
TalentMatch AI — Improved Ranking Engine
----------------------------------------
Drop-in replacement for ranking_engine.py.

Fixes:
1. Retrieves a larger candidate pool before final top-k selection.
2. Removes exact and near-duplicate job postings.
3. Rejects weak matches instead of presenting them as recommendations.
4. Preserves title/company metadata when available.
5. Produces more honest explanations and summary scores.
6. Keeps the existing analyze_jobs() / format_interview_questions()
   interfaces compatible with the current app.py.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


MIN_SCORE = 0.0
MAX_SCORE = 1.0

# A result below this is not presented as a recommendation.
MIN_RELEVANT_SCORE = 0.40

# A semantic-only match needs stronger semantic evidence.
MIN_SEMANTIC_SCORE = 0.45

# A strong lexical match can rescue a slightly lower semantic score.
MIN_KEYWORD_SCORE = 0.12

# Pull more candidates before filtering/deduplication.
CANDIDATE_POOL_MULTIPLIER = 8
MIN_CANDIDATE_POOL = 50

# Near-duplicate threshold for normalized job text.
DUPLICATE_SIMILARITY = 0.92


def clamp_score(score: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, float(score)))


def score_to_percentage(score: float) -> float:
    return round(clamp_score(score) * 100, 2)


def get_match_level(score: float) -> str:
    p = score_to_percentage(score)

    if p >= 80:
        return "Excellent Match"
    if p >= 65:
        return "Strong Match"
    if p >= 50:
        return "Good Match"
    if p >= 40:
        return "Developing Match"
    return "Low Match"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _first(job: Dict[str, Any], names: Tuple[str, ...]) -> str:
    for name in names:
        value = _text(job.get(name))
        if value:
            return value
    return ""


def _normalize_text(value: Any) -> str:
    text = _text(value).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _job_title(job: Dict[str, Any]) -> str:
    return _first(
        job,
        (
            "title",
            "job_title",
            "jobtitle",
            "position",
            "role",
            "job_role",
            "category",
            "Category",
        ),
    )


def _company(job: Dict[str, Any]) -> str:
    return _first(
        job,
        (
            "company",
            "company_name",
            "employer",
            "organization",
            "organisation",
        ),
    )


def _job_text(job: Dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            _job_title(job),
            _company(job),
            _first(job, ("category", "Category")),
            _first(job, ("description", "Description")),
            _first(job, ("requirements", "Requirements", "requirement")),
            _first(job, ("benefits", "Benefits")),
        )
        if part
    )


def _job_fingerprint(job: Dict[str, Any]) -> str:
    normalized = _normalize_text(_job_text(job))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _duplicate_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    Compare meaningful job text rather than only the category.
    This prevents legitimate different HR/Data Scientist jobs
    from being removed merely because they share a category.
    """
    ta = _normalize_text(_job_text(a))
    tb = _normalize_text(_job_text(b))

    if not ta or not tb:
        return 0.0

    return SequenceMatcher(None, ta, tb).ratio()


def _is_duplicate(candidate: Dict[str, Any], kept: List[Dict[str, Any]]) -> bool:
    title = _normalize_text(_job_title(candidate))
    company = _normalize_text(_company(candidate))
    fingerprint = _job_fingerprint(candidate)

    for existing in kept:
        if fingerprint == existing.get("_fingerprint"):
            return True

        existing_title = _normalize_text(_job_title(existing))
        existing_company = _normalize_text(_company(existing))

        # Same title + same company is a duplicate/alternate copy
        # unless the posting text is materially different.
        if title and company and title == existing_title and company == existing_company:
            if _duplicate_similarity(candidate, existing) >= 0.80:
                return True

        if _duplicate_similarity(candidate, existing) >= DUPLICATE_SIMILARITY:
            return True

    return False


def _is_relevant(job: Dict[str, Any]) -> bool:
    final_score = float(job.get("score", job.get("final_score", 0.0)))
    semantic_score = float(
        job.get("semantic_score", job.get("semantic", 0.0))
    )
    keyword_score = float(
        job.get(
            "tfidf_score",
            job.get("keyword_score", 0.0),
        )
    )

    if final_score < MIN_RELEVANT_SCORE:
        return False

    # Require either genuinely useful semantic evidence or
    # unusually strong lexical evidence.
    if semantic_score < MIN_SEMANTIC_SCORE and keyword_score < MIN_KEYWORD_SCORE:
        return False

    return True


def explain_score(
    semantic_score: float,
    tfidf_score: float,
    final_score: float,
) -> Dict[str, Any]:
    semantic_pct = score_to_percentage(semantic_score)
    keyword_pct = score_to_percentage(tfidf_score)
    final_pct = score_to_percentage(final_score)

    strengths: List[str] = []

    if semantic_pct >= 65:
        strengths.append("Strong semantic alignment with the role.")
    elif semantic_pct >= 50:
        strengths.append("Moderate semantic alignment with the role.")

    if keyword_pct >= 25:
        strengths.append("Strong overlap with job-specific terminology.")
    elif keyword_pct >= 12:
        strengths.append("Useful overlap with job-specific terminology.")

    if not strengths:
        strengths.append("The available evidence is limited.")

    return {
        "match_score": final_pct,
        "semantic_score": semantic_pct,
        "keyword_score": keyword_pct,
        "match_level": get_match_level(final_score),
        "strengths": strengths,
    }


def rank_job(job: Dict[str, Any]) -> Dict[str, Any]:
    semantic_score = float(
        job.get("semantic_score", job.get("semantic", 0.0))
    )
    tfidf_score = float(
        job.get("tfidf_score", job.get("keyword_score", 0.0))
    )
    final_score = float(
        job.get("score", job.get("final_score", 0.0))
    )

    explanation = explain_score(
        semantic_score,
        tfidf_score,
        final_score,
    )

    title = _job_title(job)
    company = _company(job)
    category = _first(job, ("category", "Category"))

    result = {
        "match_score": explanation["match_score"],
        "semantic_score": explanation["semantic_score"],
        "keyword_score": explanation["keyword_score"],
        "match_level": explanation["match_level"],
        "title": title or category or "Job Opportunity",
        "company": company,
        "category": category,
        "description": _first(job, ("description", "Description")),
        "requirements": _first(job, ("requirements", "Requirements", "requirement")),
        "benefits": _first(job, ("benefits", "Benefits")),
        "strengths": explanation["strengths"],
    }

    # Internal fields are used only during ranking/deduplication.
    result["_fingerprint"] = _job_fingerprint(job)
    return result


def _deduplicate_and_filter(
    jobs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    accepted: List[Dict[str, Any]] = []
    rejected = 0
    duplicates = 0

    for raw in jobs:
        if not _is_relevant(raw):
            rejected += 1
            continue

        ranked = rank_job(raw)

        if _is_duplicate(ranked, accepted):
            duplicates += 1
            continue

        accepted.append(ranked)

    return accepted, rejected, duplicates


def rank_jobs(
    jobs: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if not jobs:
        return []

    ranked, _, _ = _deduplicate_and_filter(jobs)

    ranked.sort(
        key=lambda item: item["match_score"],
        reverse=True,
    )

    ranked = ranked[: max(1, int(top_k))]

    for position, job in enumerate(ranked, start=1):
        job["rank"] = position
        job.pop("_fingerprint", None)

    return ranked


def build_candidate_summary(
    ranked_jobs: List[Dict[str, Any]],
    *,
    candidate_pool_size: int = 0,
    rejected_count: int = 0,
    duplicate_count: int = 0,
) -> Dict[str, Any]:
    if not ranked_jobs:
        return {
            "jobs_analyzed": 0,
            "candidate_pool_size": candidate_pool_size,
            "rejected_weak_matches": rejected_count,
            "duplicates_removed": duplicate_count,
            "best_match_score": 0,
            "best_match_level": "No Suitable Match",
            "average_match_score": 0,
        }

    scores = [float(job["match_score"]) for job in ranked_jobs]
    best = ranked_jobs[0]

    return {
        "jobs_analyzed": len(ranked_jobs),
        "candidate_pool_size": candidate_pool_size,
        "rejected_weak_matches": rejected_count,
        "duplicates_removed": duplicate_count,
        "best_match_score": best["match_score"],
        "best_match_level": best["match_level"],
        "average_match_score": round(sum(scores) / len(scores), 2),
    }


def analyze_jobs(
    prediction_engine,
    resume_text: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    requested = max(1, int(top_k))

    # Do NOT ask the predictor for only the final number of jobs.
    # Duplicates and weak matches must be removed first.
    candidate_pool = max(
        MIN_CANDIDATE_POOL,
        requested * CANDIDATE_POOL_MULTIPLIER,
    )

    predictions = prediction_engine.hybrid_job_search(
        resume_text,
        top_k=candidate_pool,
    )

    ranked, rejected, duplicates = _deduplicate_and_filter(predictions)

    ranked.sort(
        key=lambda item: item["match_score"],
        reverse=True,
    )

    ranked = ranked[:requested]

    for position, job in enumerate(ranked, start=1):
        job["rank"] = position
        job.pop("_fingerprint", None)

    summary = build_candidate_summary(
        ranked,
        candidate_pool_size=len(predictions),
        rejected_count=rejected,
        duplicate_count=duplicates,
    )

    return {
        "summary": summary,
        "jobs": ranked,
    }


def format_interview_questions(
    questions: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if not questions:
        return []

    formatted = []

    for position, question in enumerate(questions[:top_k], start=1):
        score = float(question.get("score", 0.0))

        formatted.append(
            {
                "rank": position,
                "relevance_score": score_to_percentage(score),
                "question": question.get("question", ""),
                "answer": question.get("answer", ""),
                "role": question.get("role", ""),
                "category": question.get("category", ""),
                "difficulty": question.get("difficulty", ""),
                "experience": question.get("experience", ""),
            }
        )

    return formatted
