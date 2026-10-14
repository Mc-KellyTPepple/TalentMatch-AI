"""
TalentMatch-AI — Groq Contextual Reranker

Optimized for an 8,000 TPM Groq limit (free tier):
  ~1,300 resume tokens + ~2,500 job tokens + 2,500 max output stays under it.

The reranker:
- sends at most 10 jobs
- compresses job descriptions
- limits resume size
- preserves exact candidate IDs
- validates Groq JSON
- rejects invented/duplicate IDs
- never prints the API key
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from groq import Groq


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_JOBS = 10
ABSOLUTE_MAX_JOBS = 10
DEFAULT_TIMEOUT_SECONDS = 45

MAX_RESUME_CHARS = 5000
MAX_TITLE_CHARS = 120
MAX_COMPANY_CHARS = 120
MAX_CATEGORY_CHARS = 120
MAX_DESCRIPTION_CHARS = 450
MAX_REQUIREMENTS_CHARS = 300
MAX_BENEFITS_CHARS = 0


VALID_DECISIONS = {
    "strong_match",
    "match",
    "potential_match",
    "weak_match",
    "not_a_match",
}


def txt(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def clip(value: Any, max_chars: int) -> str:
    text = txt(value)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def env_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(name)

    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return max(
        minimum,
        min(maximum, value),
    )


# =====================================================================
# JOB TITLE / COMPANY EXTRACTION
# =====================================================================


def extract_job_title(description: str, category: str = "") -> str:
    """Extract a plausible vacancy title without returning a whole sentence."""
    raw = str(description or "").replace("\r", "\n")

    # First, prefer explicitly labelled titles.
    explicit_patterns = [
        r"(?im)^\s*Job\s*Title\s*:\s*([^\n]{2,120})",
        r"(?im)^\s*Position\s*:\s*([^\n]{2,120})",
        r"(?im)^\s*Role\s*:\s*([^\n]{2,120})",
        r"(?im)^\s*Title\s*:\s*([^\n]{2,120})",
    ]

    for pattern in explicit_patterns:
        match = re.search(pattern, raw)
        if match:
            title = txt(match.group(1))
            title = re.split(
                r"\b(?:Company|Location|Salary|Description|Requirements|Responsibilities)\s*:",
                title,
                flags=re.IGNORECASE,
            )[0].strip(" -|,;.")
            if 2 <= len(title) <= MAX_TITLE_CHARS:
                return title

    # Then look for common job-advert wording.
    intro_patterns = [
        r"\b(?:is seeking|are seeking|seeks|seek|looking for|recruiting for|hiring)\s+(?:a|an|the)\s+(.{3,100}?)(?=\s+(?:to join|to work|who will|for our|for its|for the|in this role|with an|with a|at our|at the)\b|[.!?\n]|$)",
    ]

    for pattern in intro_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            title = txt(match.group(1)).strip(" -|,;.")

            # Remove common promotional adjectives preceding the role.
            title = re.sub(
                r"^(?:(?:talented|experienced|dynamic|qualified|motivated|driven|skilled|passionate|dedicated|strong)\s+and\s+)*(?:talented|experienced|dynamic|qualified|motivated|driven|skilled|passionate|dedicated|strong)\s+",
                "",
                title,
                flags=re.IGNORECASE,
            ).strip()

            if (
                3 <= len(title) <= MAX_TITLE_CHARS
                and len(title.split()) <= 12
            ):
                return title

    # Use the category as a safe fallback.
    category = txt(category)
    if category:
        return clip(category, MAX_TITLE_CHARS)

    return "Job Opportunity"


def extract_company(
    description: str,
) -> str:
    """Extract company name when explicitly present."""

    text = txt(description)

    if not text:
        return ""

    patterns = [
        r"\bCompany\s*:\s*([^\n]+)",
        r"\bEmployer\s*:\s*([^\n]+)",
        r"\bOrganization\s*:\s*([^\n]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            company = txt(
                match.group(1)
            )

            company = re.split(
                r"\b(?:Location|Salary|Job\s*Title|Position|Description|Requirements)\s*:",
                company,
                flags=re.IGNORECASE,
            )[0].strip()

            if company:
                return clip(
                    company,
                    MAX_COMPANY_CHARS,
                )

    return ""


# =====================================================================
# RESUME COMPRESSION
# =====================================================================

def compact_resume(
    resume_text: str,
) -> str:
    """Keep the resume compact enough for the Groq request."""

    text = txt(resume_text)

    if not text:
        return "No resume text supplied."

    if len(text) <= MAX_RESUME_CHARS:
        return text

    head_size = 3800
    tail_size = (
        MAX_RESUME_CHARS
        - head_size
    )

    return (
        text[:head_size].rstrip()
        + "\n...[resume shortened]...\n"
        + text[-tail_size:].lstrip()
    )


# =====================================================================
# JOB COMPRESSION
# =====================================================================

def compact_job(
    job: Dict[str, Any],
    candidate_id: int,
) -> Dict[str, Any]:
    """Create a compact Groq representation of one job."""

    description = txt(
        job.get("description")
    )

    requirements = txt(
        job.get("requirements")
    )

    benefits = txt(
        job.get("benefits")
    )

    category = txt(
        job.get("category")
    )

    title = txt(
        job.get("title")
    )

    if not title:
        title = extract_job_title(
            description,
            category,
        )

    company = txt(
        job.get("company")
    )

    if not company:
        company = extract_company(
            description
        )

    # Remove repeated Job Title text from description.
    cleaned_description = description

    if title and cleaned_description:

        cleaned_description = re.sub(
            r"\bJob\s*Title\s*:\s*"
            + re.escape(title),
            "",
            cleaned_description,
            flags=re.IGNORECASE,
        )

    return {
        "candidate_id": int(candidate_id),
        "title": clip(
            title,
            MAX_TITLE_CHARS,
        ),
        "company": clip(
            company,
            MAX_COMPANY_CHARS,
        ),
        "category": clip(
            category,
            MAX_CATEGORY_CHARS,
        ),
        "description": clip(
            cleaned_description,
            MAX_DESCRIPTION_CHARS,
        ),
        "requirements": clip(
            requirements,
            MAX_REQUIREMENTS_CHARS,
        ),
    }


# =====================================================================
# GROQ CLIENT
# =====================================================================

def get_groq_client() -> Optional[Groq]:
    """Return Groq client when API key exists."""

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


# =====================================================================
# PROMPTS
# =====================================================================

def build_system_prompt() -> str:
    return """
You are the contextual job-matching evaluator for TalentMatch-AI.

Evaluate each job against the candidate's actual professional background.

Prioritize:
1. Actual role/career relevance.
2. Technical and professional skills.
3. Relevant experience.
4. Seniority/level.
5. Industry/domain relevance.

Do NOT match merely because generic words overlap.

An AI/ML candidate should not receive a high score for HR simply because
the resume contains generic words such as management, team, communication,
research, or leadership.

Likewise, do not assume every Data Scientist role is appropriate merely
because the candidate has AI or Python experience. Judge the actual role
requirements.

Return ONLY valid JSON:

{
  "assessments": [
    {
      "candidate_id": 0,
      "overall_score": 0,
      "role_relevance": 0,
      "skills_match": 0,
      "experience_match": 0,
      "seniority_match": 0,
      "industry_match": 0,
      "decision": "strong_match",
      "matched_skills": [],
      "missing_skills": [],
      "reason": ""
    }
  ]
}

Scores must be integers from 0 to 100.
Keep each "reason" under 30 words and each skills list under 6 short items.

Decision must be exactly one of:
strong_match
match
potential_match
weak_match
not_a_match

Use the exact candidate_id supplied for each job.
Never invent, renumber, duplicate, or omit candidate IDs.
""".strip()


def build_user_prompt(
    resume_text: str,
    jobs: List[Dict[str, Any]],
) -> str:
    payload = {
        "candidate_resume": compact_resume(
            resume_text
        ),
        "jobs": jobs,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# =====================================================================
# RESPONSE VALIDATION
# =====================================================================

def clamp_score(
    value: Any,
) -> int:
    try:
        score = int(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0

    return max(
        0,
        min(100, score),
    )


def clean_string_list(
    value: Any,
    max_items: int = 8,
) -> List[str]:

    if not isinstance(
        value,
        list,
    ):
        return []

    result = []

    for item in value[:max_items]:

        item_text = txt(item)

        if item_text:
            result.append(
                clip(
                    item_text,
                    80,
                )
            )

    return result


def validate_assessment(
    assessment: Any,
    valid_ids: set[int],
) -> Optional[Dict[str, Any]]:

    if not isinstance(
        assessment,
        dict,
    ):
        return None

    try:
        candidate_id = int(
            assessment.get(
                "candidate_id"
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if candidate_id not in valid_ids:
        return None

    decision = txt(
        assessment.get(
            "decision"
        )
    ).lower()

    if decision not in VALID_DECISIONS:
        decision = "potential_match"

    reason = txt(
        assessment.get(
            "reason"
        )
    )

    if not reason:
        reason = (
            "Contextual AI evaluation completed."
        )

    return {
        "candidate_id": candidate_id,
        "overall_score": clamp_score(
            assessment.get(
                "overall_score"
            )
        ),
        "role_relevance": clamp_score(
            assessment.get(
                "role_relevance"
            )
        ),
        "skills_match": clamp_score(
            assessment.get(
                "skills_match"
            )
        ),
        "experience_match": clamp_score(
            assessment.get(
                "experience_match"
            )
        ),
        "seniority_match": clamp_score(
            assessment.get(
                "seniority_match"
            )
        ),
        "industry_match": clamp_score(
            assessment.get(
                "industry_match"
            )
        ),
        "decision": decision,
        "matched_skills": clean_string_list(
            assessment.get(
                "matched_skills"
            )
        ),
        "missing_skills": clean_string_list(
            assessment.get(
                "missing_skills"
            )
        ),
        "reason": clip(
            reason,
            600,
        ),
    }


def parse_json_response(
    content: str,
) -> Dict[str, Any]:

    text = txt(content)

    if not text:
        raise ValueError(
            "Groq returned an empty response."
        )

    # Remove markdown fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip()

    try:
        parsed = json.loads(text)

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    # Fallback: locate JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        parsed = json.loads(
            text[start:end + 1]
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    raise ValueError(
        "Groq returned invalid JSON."
    )


# =====================================================================
# MAIN RERANKER
# =====================================================================

def rerank_jobs_with_groq(
    resume_text: str,
    jobs: List[Dict[str, Any]],
    max_jobs: Optional[int] = None,
) -> Dict[str, Any]:

    model = os.getenv(
        "GROQ_MODEL",
        DEFAULT_MODEL,
    )

    configured_max = env_int(
        "GROQ_MAX_JOBS",
        DEFAULT_MAX_JOBS,
        1,
        ABSOLUTE_MAX_JOBS,
    )

    if max_jobs is not None:

        try:
            requested = int(
                max_jobs
            )
        except (
            TypeError,
            ValueError,
        ):
            requested = configured_max

        configured_max = max(
            1,
            min(
                requested,
                ABSOLUTE_MAX_JOBS,
            ),
        )

    timeout_seconds = env_int(
        "GROQ_TIMEOUT_SECONDS",
        DEFAULT_TIMEOUT_SECONDS,
        10,
        120,
    )

    client = get_groq_client()

    if client is None:

        return {
            "available": False,
            "model": model,
            "error": (
                "GROQ_API_KEY is not set."
            ),
            "assessments": [],
        }

    if not txt(resume_text):

        return {
            "available": False,
            "model": model,
            "error": "Resume text is empty.",
            "assessments": [],
        }

    if not jobs:

        return {
            "available": False,
            "model": model,
            "error": (
                "No jobs were supplied "
                "for contextual reranking."
            ),
            "assessments": [],
        }

    selected_jobs = jobs[
        :configured_max
    ]

    compact_jobs = [
        compact_job(
            job,
            index,
        )
        for index, job in enumerate(
            selected_jobs
        )
    ]

    valid_ids = {
        int(job["candidate_id"])
        for job in compact_jobs
    }

    system_prompt = (
        build_system_prompt()
    )

    user_prompt = build_user_prompt(
        resume_text,
        compact_jobs,
    )

    try:

        request = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2500,
            response_format={"type": "json_object"},
            timeout=timeout_seconds,
        )

        try:
            # gpt-oss models spend output tokens on hidden reasoning; keep it
            # low so the JSON answer is not truncated.
            response = client.chat.completions.create(
                reasoning_effort="low", **request
            )
        except TypeError:
            response = client.chat.completions.create(**request)
        except Exception as first_error:
            if "reasoning" in str(first_error).lower():
                response = client.chat.completions.create(**request)
            else:
                raise

        content = (
            response.choices[0]
            .message.content
        )

        parsed = parse_json_response(
            content
        )

        raw_assessments = (
            parsed.get(
                "assessments",
                [],
            )
        )

        if not isinstance(
            raw_assessments,
            list,
        ):
            raise ValueError(
                "Groq JSON did not contain "
                "an assessments list."
            )

        assessments = []
        seen_ids = set()

        for raw in raw_assessments:

            assessment = (
                validate_assessment(
                    raw,
                    valid_ids,
                )
            )

            if assessment is None:
                continue

            candidate_id = (
                assessment[
                    "candidate_id"
                ]
            )

            if candidate_id in seen_ids:
                continue

            seen_ids.add(
                candidate_id
            )

            assessments.append(
                assessment
            )

        if not assessments:

            raise ValueError(
                "Groq returned no valid "
                "job assessments."
            )

        # Preserve original candidate order.
        order = {
            index: index
            for index in range(
                len(compact_jobs)
            )
        }

        assessments.sort(
            key=lambda item: order.get(
                item[
                    "candidate_id"
                ],
                999999,
            )
        )

        return {
            "available": True,
            "model": model,
            "error": None,
            "assessments": assessments,
        }

    except Exception as exc:

        return {
            "available": False,
            "model": model,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "assessments": [],
        }


# =====================================================================
# SAFE DIAGNOSTICS
# =====================================================================

def groq_status() -> Dict[str, Any]:

    return {
        "available": bool(
            os.getenv(
                "GROQ_API_KEY"
            )
        ),
        "model": os.getenv(
            "GROQ_MODEL",
            DEFAULT_MODEL,
        ),
        "max_jobs": env_int(
            "GROQ_MAX_JOBS",
            DEFAULT_MAX_JOBS,
            1,
            ABSOLUTE_MAX_JOBS,
        ),
        "timeout_seconds": env_int(
            "GROQ_TIMEOUT_SECONDS",
            DEFAULT_TIMEOUT_SECONDS,
            10,
            120,
        ),
    }


if __name__ == "__main__":

    status = groq_status()

    print(
        "TalentMatch Groq Reranker"
    )
    print(
        "-------------------------"
    )
    print(
        "GROQ KEY:",
        "SET"
        if status["available"]
        else "MISSING",
    )
    print(
        "MODEL:",
        status["model"],
    )
    print(
        "MAX JOBS:",
        status["max_jobs"],
    )
    print(
        "TIMEOUT:",
        status["timeout_seconds"],
        "seconds",
    )