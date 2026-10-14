"""
TalentMatch AI - orchestration.

    CV text
      -> matcher.analyse_cv()          skills, level, ranked career roles, gaps
      -> live_jobs.store               current vacancies (Remotive, Arbeitnow)
      -> local scoring                 role alignment + skill overlap + TF-IDF
                                       + seniority + location eligibility
      -> Groq contextual rerank        optional; skipped safely if unavailable
      -> response                      roles, jobs, learn-next skills, interview prep

The local score is always computed, so the service still returns useful,
honest results when Groq or the vacancy feeds are down.
"""

from __future__ import annotations

import re
import threading
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import matcher
from groq_reranker import groq_status, rerank_jobs_with_groq
from live_jobs import store as live_store

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

FINAL_TOP_K = 10
GROQ_POOL = 10
MIN_LOCAL_JOB_SCORE = 28.0          # below this a live job is not shown
MIN_GROQ_SCORE = 40                 # AI says weaker than this -> dropped
GROQ_WEIGHT = 0.60
LOCAL_WEIGHT = 0.40
RESTRICTED_LOCATION_FACTOR = 0.55

WORLDWIDE_WORDS = ("worldwide", "anywhere", "global", "any location", "all countries", "international", "remote")
AFRICA = {
    "nigeria", "ghana", "kenya", "south africa", "egypt", "ethiopia", "tanzania", "uganda", "rwanda",
    "senegal", "cameroon", "morocco", "tunisia", "algeria", "zambia", "zimbabwe", "botswana", "namibia",
}

_SENIOR_RE = re.compile(r"\b(senior|sr\.?|staff|principal|lead|head of|director|vp|vice president|chief|manager)\b")
_JUNIOR_RE = re.compile(r"\b(intern|internship|trainee|graduate|entry[- ]level|apprentice)\b")
_JR_RE = re.compile(r"\b(junior|jr\.?|associate)\b")
_LEAD_RE = re.compile(r"\b(principal|head of|director|vp|vice president|chief)\b")


# --------------------------------------------------------------------------
# Live job index (rebuilt only when the cache changes)
# --------------------------------------------------------------------------

class _JobIndex:
    def __init__(self) -> None:
        self.version = -1
        self.jobs: List[Dict[str, Any]] = []
        self.skills: List[Dict[str, int]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.lock = threading.Lock()

    def ensure(self, jobs: List[Dict[str, Any]], version: int) -> None:
        if version == self.version and self.jobs is not None and len(self.jobs) == len(jobs):
            return
        with self.lock:
            if version == self.version and len(self.jobs) == len(jobs):
                return
            docs = [f"{j['title']} {j['title']} {j.get('category', '')} {j.get('text', '')}" for j in jobs]
            skills = [matcher.extract_skills(f"{j['title']} {j.get('text', '')} {' '.join(j.get('tags', []))}") for j in jobs]
            vectorizer = None
            matrix = None
            if docs:
                vectorizer = TfidfVectorizer(
                    lowercase=True, stop_words="english", ngram_range=(1, 2),
                    sublinear_tf=True, min_df=2, max_features=30000, dtype=np.float32,
                )
                try:
                    matrix = vectorizer.fit_transform(docs)
                except ValueError:  # empty vocabulary
                    vectorizer, matrix = None, None
            self.jobs, self.skills, self.vectorizer, self.matrix, self.version = jobs, skills, vectorizer, matrix, version


_INDEX = _JobIndex()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _job_level(title: str) -> Optional[int]:
    low = title.lower()
    if _JUNIOR_RE.search(low):
        return 0
    if _JR_RE.search(low):
        return 1
    if _LEAD_RE.search(low):
        return 4
    if _SENIOR_RE.search(low):
        return 3
    return None


def _recency(posted: str) -> float:
    try:
        days = (date.today() - datetime.fromisoformat(posted).date()).days
    except Exception:
        return 0.4
    if days <= 14:
        return 1.0
    if days <= 30:
        return 0.6
    return 0.3


def _eligibility(job: Dict[str, Any], country: str) -> Tuple[str, float]:
    """Return (label, score multiplier) for a candidate based in `country`."""
    location = (job.get("location") or "").lower()
    if not country:
        label = job.get("location") or ""
        return (f"Location: {label}" if label else ""), 1.0
    c = country.lower().strip()

    if c and c in location:
        return "Open to your location", 1.0
    if any(w in location for w in WORLDWIDE_WORDS):
        return "Open worldwide", 1.0
    if c in AFRICA and any(w in location for w in ("africa", "emea")):
        return "Open to your region", 1.0
    if job.get("visa"):
        return "Visa sponsorship mentioned - relocation required", 0.9
    if job.get("source") == "Remotive" and location:
        return f"Restricted to: {job.get('location')}", RESTRICTED_LOCATION_FACTOR
    if job.get("remote"):
        return "Remote, but check eligibility for your country", 0.85
    return f"On-site in {job.get('location') or 'another location'} - relocation required", RESTRICTED_LOCATION_FACTOR


def match_level(score: float) -> str:
    if score >= 80:
        return "Excellent Match"
    if score >= 65:
        return "Strong Match"
    if score >= 50:
        return "Good Match"
    if score >= 35:
        return "Potential Match"
    return "Low Match"


# --------------------------------------------------------------------------
# Live job scoring
# --------------------------------------------------------------------------

def score_live_jobs(resume_text: str, analysis: Dict[str, Any], country: str = "", remote_only: bool = False,
                    only_eligible: bool = False) -> Tuple[List[Dict[str, Any]], int]:
    jobs = live_store.get_jobs()
    if not jobs:
        return [], 0
    _INDEX.ensure(jobs, live_store.version)
    if _INDEX.matrix is None:
        return [], len(jobs)

    cv_skills = analysis["skills"]
    cv_level = analysis["experience"]["level_index"]
    top_roles = analysis["roles"][:4]

    query = _INDEX.vectorizer.transform([resume_text.lower()])
    cos = (_INDEX.matrix @ query.T).toarray().ravel()
    p95 = float(np.percentile(cos, 95)) if cos.size else 0.0
    p95 = p95 if p95 > 1e-6 else 1.0

    role_patterns = [(matcher._phrase_regex(
        next(r["aliases"] for r in matcher.ROLES if r["title"] == role["title"])), role["fit"] / 100.0, role["title"])
        for role in top_roles]

    scored: List[Dict[str, Any]] = []
    for idx, job in enumerate(_INDEX.jobs):
        if remote_only and not job.get("remote"):
            continue

        title_low = matcher.normalize(job["title"])
        align, aligned_role = 0.0, ""
        for pattern, fit, title in role_patterns:
            if pattern.search(title_low) and fit > align:
                align, aligned_role = fit, title

        job_skills = _INDEX.skills[idx]
        shared = [s for s in job_skills if s in cv_skills]
        total_w = sum(matcher.skill_weight(s) for s in job_skills)
        got_w = sum(matcher.skill_weight(s) for s in shared)
        skill_cov = min(1.0, (got_w / max(total_w, 8.0)) / 0.55)
        cos_rel = min(1.0, float(cos[idx]) / p95)

        # Needs some real evidence: a role-title hit or several shared skills.
        if align == 0.0 and len(shared) < 3:
            continue

        local = 0.38 * align + 0.34 * skill_cov + 0.20 * cos_rel + 0.08 * _recency(job.get("posted", ""))

        level_note = ""
        job_level = _job_level(job["title"])
        if job_level is not None:
            diff = job_level - cv_level
            if diff > 1:
                local *= max(0.5, 1 - 0.18 * (diff - 1))
                level_note = "More senior than your current level - a stretch role."
            elif diff < -1:
                local *= max(0.7, 1 - 0.10 * (-diff - 1))
                level_note = "Below your current level."

        label, factor = _eligibility(job, country)
        if only_eligible and country and factor < 0.8:
            continue
        local *= factor

        local_pct = round(min(98.0, local * 100.0), 1)
        if local_pct < MIN_LOCAL_JOB_SCORE:
            continue

        scored.append({
            "title": job["title"], "company": job.get("company", ""), "location": job.get("location", ""),
            "remote": bool(job.get("remote")), "url": job["url"], "source": job["source"],
            "posted": job.get("posted", ""), "salary": job.get("salary", ""), "job_type": job.get("job_type", ""),
            "category": job.get("category", ""),
            "description": job.get("text", "")[:700],
            "requirements": ", ".join(matcher.pretty_skills(sorted(job_skills, key=lambda s: -matcher.skill_weight(s))[:12])),
            "matched_skills": matcher.pretty_skills(sorted(shared, key=lambda s: -matcher.skill_weight(s))[:10]),
            "missing_skills": matcher.pretty_skills(sorted((s for s in job_skills if s not in cv_skills),
                                                           key=lambda s: -matcher.skill_weight(s))[:6]),
            "aligned_role": aligned_role, "eligibility": label, "level_note": level_note,
            "local_score": local_pct, "match_score": local_pct, "ai_reranked": False,
            "reason": _local_reason(aligned_role, shared, level_note),
        })

    scored.sort(key=lambda j: j["local_score"], reverse=True)
    return scored, len(jobs)


def _local_reason(aligned_role: str, shared: List[str], level_note: str) -> str:
    parts = []
    if aligned_role:
        parts.append(f"Title lines up with your best-fit role ({aligned_role}).")
    if shared:
        parts.append(f"Shares {len(shared)} skill{'s' if len(shared) != 1 else ''} with your CV: "
                     f"{', '.join(matcher.pretty_skills(shared[:5]))}.")
    if level_note:
        parts.append(level_note)
    return " ".join(parts) or "Matched on overall wording similarity with your CV."


# --------------------------------------------------------------------------
# Groq blending
# --------------------------------------------------------------------------

def _apply_groq(resume_text: str, jobs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pool = jobs[:GROQ_POOL]
    rest = jobs[GROQ_POOL:]
    candidates = [{
        "title": j["title"], "company": j["company"], "category": j.get("category", ""),
        "description": j["description"], "requirements": j["requirements"], "benefits": "",
    } for j in pool]

    result = rerank_jobs_with_groq(resume_text, candidates) or {}
    info = {"enabled": bool(result.get("available")), "model": result.get("model", ""), "error": result.get("error") or ""}
    if not result.get("available"):
        return jobs, info

    by_id = {int(a["candidate_id"]): a for a in result.get("assessments", [])}
    blended: List[Dict[str, Any]] = []
    for idx, job in enumerate(pool):
        a = by_id.get(idx)
        if not a:
            blended.append(job)  # keep local ranking for jobs the AI skipped
            continue
        if a["decision"] == "not_a_match" or a["overall_score"] < MIN_GROQ_SCORE:
            continue
        final = round(min(98.0, GROQ_WEIGHT * a["overall_score"] + LOCAL_WEIGHT * job["local_score"]), 1)
        job = dict(job)
        job.update({
            "match_score": final, "ai_score": a["overall_score"], "ai_reranked": True,
            "reason": a["reason"] or job["reason"],
            "matched_skills": a["matched_skills"] or job["matched_skills"],
            "missing_skills": a["missing_skills"] or job["missing_skills"],
            "decision": a["decision"],
        })
        blended.append(job)
    blended.sort(key=lambda j: j["match_score"], reverse=True)
    return blended + rest, info


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def analyze_cv(resume_text: str, top_k: int = FINAL_TOP_K, country: str = "", remote_only: bool = False,
               only_eligible: bool = False, use_ai: bool = True) -> Dict[str, Any]:
    top_k = max(1, min(int(top_k), FINAL_TOP_K))
    country = (country or "").strip()[:60]

    analysis = matcher.analyse_cv(resume_text, top_roles=5, location=country)
    notices: List[str] = []

    jobs: List[Dict[str, Any]] = []
    scanned = 0
    live_available = live_store.enabled()
    if analysis["has_strong_match"]:
        try:
            jobs, scanned = score_live_jobs(resume_text, analysis, country, remote_only, only_eligible)
        except Exception as exc:  # noqa: BLE001 - never lose the role analysis because of the job feed
            notices.append("Live vacancy matching hit an unexpected problem; role matches are still shown.")
            print(f"[TalentMatch] live job scoring failed: {exc!r}", flush=True)
    else:
        notices.append("Your CV did not clearly match any role in our library. Add specific skills, tools and job titles, "
                       "then try again.")

    groq_info = {"enabled": False, "model": "", "error": ""}
    if jobs and use_ai and groq_status()["available"]:
        jobs, groq_info = _apply_groq(resume_text, jobs)
        if not groq_info["enabled"]:
            notices.append("AI re-ranking was unavailable for this request; results use the skills and keyword engine.")

    jobs = jobs[:top_k]
    for job in jobs:
        job["match_level"] = match_level(job["match_score"])
        job["score"] = round(job["match_score"] / 100.0, 4)
        job["strengths"] = job["matched_skills"]

    if analysis["has_strong_match"] and not jobs:
        if scanned == 0:
            notices.append("Live vacancy feeds are not reachable right now. Use the search links on each role to find openings.")
        else:
            notices.append("No current live listing was a good enough match. Try removing the location filters, "
                           "or use the role search links below.")

    roles = []
    for r in analysis["roles"]:
        roles.append({
            "title": r["title"], "family": r["family"], "summary": r["summary"], "fit": r["fit"],
            "match_level": match_level(r["fit"]),
            "matched_skills": matcher.pretty_skills(r["matched_skills"]),
            "missing_skills": matcher.pretty_skills(r["missing_skills"]),
            "next_roles": r["next_roles"], "level_note": r["level_note"], "search_links": r["search_links"],
        })

    # Skills to learn next: missing core skills weighted by how well the role fits.
    gap: Dict[str, float] = {}
    gap_count: Dict[str, int] = {}
    for r in analysis["strong_roles"][:3]:
        for rank, skill in enumerate(r["missing_skills"][:5]):
            gap[skill] = gap.get(skill, 0.0) + (r["fit"] / 100.0) * (1.0 - rank * 0.1)
            gap_count[skill] = gap_count.get(skill, 0) + 1
    learn_next = [{"skill": matcher.pretty_skill(s), "roles": gap_count[s]}
                  for s in sorted(gap, key=gap.get, reverse=True)[:6]]

    interviews: List[Dict[str, Any]] = []
    for r in analysis["strong_roles"][:2]:
        interviews += matcher.interview_prep(r["title"], r["matched_skills"], r["missing_skills"], limit=5)

    exp = analysis["experience"]
    best = analysis["roles"][0] if analysis["roles"] else None
    scores = [j["match_score"] for j in jobs]

    return {
        "profile": {
            "experience_years": exp["years"], "level": exp["level"], "education": exp["education"],
            "skills": matcher.pretty_skills(analysis["skill_list"]),
            "cv_tips": analysis["tips"], "word_count": analysis["word_count"],
        },
        "roles": roles,
        "jobs": jobs,
        "learn_next": learn_next,
        "adjacent_roles": matcher.adjacent_roles(analysis["roles"]),
        "interviews": interviews,
        "notices": notices,
        "summary": {
            "jobs_analyzed": scanned,
            "jobs_matched": len(jobs),
            "best_role": best["title"] if best else "",
            "best_role_fit": best["fit"] if best else 0,
            "best_match_score": max(scores) if scores else 0,
            "best_match_level": match_level(max(scores)) if scores else "No live match",
            "average_match_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "groq_enabled": groq_info["enabled"],
            "groq_model": groq_info["model"],
            "groq_error": groq_info["error"],
            "live_jobs": live_store.status(),
            "live_jobs_requested": live_available,
        },
    }
