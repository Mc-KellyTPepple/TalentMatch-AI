"""
TalentMatch AI - CV analysis and role matching core.

Pure-Python + scikit-learn. No PyTorch / transformer model is loaded, so the
whole service fits comfortably inside Render's 512 MB free instance and
starts in about a second.

Pipeline
--------
CV text
  -> skill extraction      (clean vocabulary from roles_kb.py, with aliases)
  -> experience / level    (years from stated durations and date ranges)
  -> role scoring          (weighted skill coverage + title evidence + TF-IDF)
  -> gap analysis          (missing core skills, next career steps)
  -> CV quality tips
"""

from __future__ import annotations

import math
import re
from datetime import date
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote_plus

from sklearn.feature_extraction.text import TfidfVectorizer

from roles_kb import ROLES, SKILL_ALIASES

# Tokens too ambiguous to match as bare words ("go", "r" ...). They are
# matched only through the explicit surface forms below.
AMBIGUOUS = {"go", "r", "ap", "ar", "ts", "node"}
EXTRA_SURFACES = {
    "golang": "go",
    "r programming": "r",
    "r language": "r",
    "rstudio": "r",
    "ggplot": "r",
    "ggplot2": "r",
}

BOUNDARY_BEFORE = r"(?<![a-z0-9+#.])"
BOUNDARY_AFTER = r"(?![a-z0-9+#])"

# A role fit at or above this value is treated as a genuine match.
STRONG_FIT = 35.0

LEVELS = [
    (0.0, "Entry level"),
    (1.5, "Junior"),
    (3.5, "Mid-level"),
    (6.5, "Senior"),
    (10.0, "Expert / Lead"),
]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text


def _phrase_regex(phrases: Iterable[str]) -> "re.Pattern[str]":
    ordered = sorted({p for p in phrases if p}, key=len, reverse=True)
    body = "|".join(re.escape(p) for p in ordered)
    return re.compile(BOUNDARY_BEFORE + "(" + body + ")" + BOUNDARY_AFTER)


# ---------------------------------------------------------------------------
# Knowledge-base indexes (built once at import)
# ---------------------------------------------------------------------------

_SURFACE_TO_SKILL: Dict[str, str] = {}
_SKILL_DF: Dict[str, int] = {}

for _role in ROLES:
    for _skill in set(_role["core"]) | set(_role["secondary"]):
        _SKILL_DF[_skill] = _SKILL_DF.get(_skill, 0) + 1

for _skill in _SKILL_DF:
    if _skill not in AMBIGUOUS:
        _SURFACE_TO_SKILL[_skill] = _skill
for _skill, _alts in SKILL_ALIASES.items():
    for _alt in _alts:
        if _alt not in AMBIGUOUS and _alt.lower() != _skill:
            _SURFACE_TO_SKILL[_alt.lower()] = _skill
_SURFACE_TO_SKILL.update(EXTRA_SURFACES)

_SKILL_PATTERN = _phrase_regex(_SURFACE_TO_SKILL)

_N_ROLES = len(ROLES)


def skill_weight(skill: str) -> float:
    """Rarer skills carry more evidence than ones every role mentions."""
    df = max(1, _SKILL_DF.get(skill, 1))
    return 1.0 + math.log(_N_ROLES / df) / 2.0


_ROLE_ALIAS_PATTERNS = [_phrase_regex(role["aliases"]) for role in ROLES]


def _role_document(role: Dict) -> str:
    return " ".join(
        [role["title"]] * 3
        + role["aliases"]
        + role["core"] * 2
        + role["secondary"]
        + [role["summary"]]
    )


_VECTORIZER = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)
_ROLE_MATRIX = _VECTORIZER.fit_transform([_role_document(r) for r in ROLES])


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> Dict[str, int]:
    """Return {canonical_skill: occurrences} found in the text."""
    counts: Dict[str, int] = {}
    for match in _SKILL_PATTERN.finditer(normalize(text)):
        skill = _SURFACE_TO_SKILL.get(match.group(1))
        if skill:
            counts[skill] = counts.get(skill, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Experience, level and education
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:-|to|until|till|through)\s*"
    r"((?:19|20)\d{2}|present|current|now|date|ongoing|till date)\b"
)
_MONTH_RANGE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
    r"((?:19|20)\d{2})\s*(?:-|to|until|till)\s*"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?"
    r"((?:19|20)\d{2}|present|current|now|date|ongoing)\b"
)
_STATED_YEARS_RE = re.compile(
    r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)(?:'|\s)*(?:of\s+)?"
    r"(?:(?:professional|relevant|industry|work|working|hands-on|commercial)\s+)?"
    r"(?:experience|exp)\b"
)
_STATED_YEARS_ALT_RE = re.compile(
    r"\bexperience\s*(?:of|:)?\s*(?:over|more than|around|about)?\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b"
)

_EDUCATION = [
    (r"\b(ph\.?d|doctorate|doctor of philosophy)\b", "PhD / Doctorate", 5),
    (r"\b(m\.?sc|m\.?eng|mba|m\.?a\b|master'?s?|postgraduate|pgd|m\.?tech)\b", "Master's degree", 4),
    (r"\b(b\.?sc|b\.?eng|b\.?a\b|b\.?tech|bachelor'?s?|undergraduate degree|b\.?ed|llb|hnd)\b", "Bachelor's degree", 3),
    (r"\b(diploma|national diploma|\bond\b|associate degree|certificate)\b", "Diploma / Certificate", 2),
]

_SENIOR_TITLE_RE = re.compile(
    r"\b(senior|sr\.?|lead|principal|head of|director|manager|chief|staff engineer|supervisor)\b"
)
_JUNIOR_TITLE_RE = re.compile(
    r"\b(intern(?:ship)?|trainee|graduate|entry[- ]level|junior|jr\.?|apprentice|volunteer|nysc|youth corps)\b"
)


def _years_from_ranges(text: str, today_year: int) -> float:
    spans: List[Tuple[int, int]] = []
    for pattern in (_RANGE_RE, _MONTH_RANGE_RE):
        for m in pattern.finditer(text):
            start = int(m.group(1))
            end_raw = m.group(2)
            end = today_year if not end_raw.isdigit() else int(end_raw)
            if start > today_year or end < start or start < 1970:
                continue
            spans.append((start, min(end, today_year)))
    if not spans:
        return 0.0
    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    # A same-year span still represents some months of work.
    return float(sum(max(1, e - s) for s, e in merged))


def estimate_experience(text: str, today: Optional[date] = None) -> Dict:
    today = today or date.today()
    low = normalize(text)

    stated = [int(m.group(1)) for m in _STATED_YEARS_RE.finditer(low)]
    stated += [int(m.group(1)) for m in _STATED_YEARS_ALT_RE.finditer(low)]
    stated = [y for y in stated if 0 < y <= 40]

    ranged = _years_from_ranges(low, today.year)

    if stated:
        # What the candidate states about themselves is the best evidence.
        years = float(max(stated))
    else:
        years = ranged
    years = min(years, 40.0)

    head = low[:600]
    if years == 0 and _JUNIOR_TITLE_RE.search(low):
        years = 0.5
    if _SENIOR_TITLE_RE.search(head) and years < 5:
        years = max(years, 5.0)

    level = LEVELS[0][1]
    for threshold, name in LEVELS:
        if years >= threshold:
            level = name

    education, edu_rank = "", 0
    for pattern, label, rank in _EDUCATION:
        if re.search(pattern, low) and rank > edu_rank:
            education, edu_rank = label, rank

    return {
        "years": round(years, 1),
        "level": level,
        "education": education,
        "level_index": [name for _, name in LEVELS].index(level),
    }


# ---------------------------------------------------------------------------
# Role scoring
# ---------------------------------------------------------------------------

def _title_signal(text_low: str, head_low: str, role_index: int) -> float:
    pattern = _ROLE_ALIAS_PATTERNS[role_index]
    if pattern.search(head_low):
        return 1.0
    hits = len(pattern.findall(text_low))
    if hits >= 3:
        return 0.8
    if hits == 2:
        return 0.65
    if hits == 1:
        return 0.45
    return 0.0


def _coverage(role: Dict, cv_skills: Dict[str, int]) -> Tuple[float, List[str], List[str]]:
    total = 0.0
    got = 0.0
    matched: List[Tuple[float, str]] = []
    missing: List[Tuple[float, str]] = []
    for skill, tier in [(s, 2.0) for s in role["core"]] + [(s, 1.0) for s in role["secondary"]]:
        w = tier * skill_weight(skill)
        total += w
        if skill in cv_skills:
            got += w
            matched.append((w, skill))
        elif tier == 2.0:
            missing.append((w, skill))
    matched.sort(reverse=True)
    missing.sort(reverse=True)
    return (got / total if total else 0.0), [s for _, s in matched], [s for _, s in missing]


def _calibrate(raw: float) -> float:
    """Map the blended raw score (0-~0.9 in practice) onto a 0-98 scale.

    Capped below 100 on purpose: a CV is never a perfect match for a role."""
    return round(max(0.0, min(98.0, 100.0 * (raw / 0.9) ** 1.1)), 1)


def search_links(title: str, location: str = "") -> Dict[str, str]:
    q = quote_plus(title)
    loc = quote_plus(location) if location else ""
    return {
        "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={q}" + (f"&location={loc}" if loc else ""),
        "indeed": f"https://www.indeed.com/jobs?q={q}" + (f"&l={loc}" if loc else ""),
        "google": "https://www.google.com/search?q=" + quote_plus(f"{title} jobs {location}".strip()) + "&ibp=htl;jobs",
    }


def rank_roles(text: str, cv_skills: Dict[str, int], experience: Dict, top_k: int = 5,
               location: str = "") -> List[Dict]:
    text_low = normalize(text)
    head_low = text_low[:400]

    query = _VECTORIZER.transform([text_low])
    cosines = (_ROLE_MATRIX @ query.T).toarray().ravel()
    max_cos = float(cosines.max()) or 1.0

    scored: List[Dict] = []
    for idx, role in enumerate(ROLES):
        coverage, matched, missing = _coverage(role, cv_skills)
        title_sig = _title_signal(text_low, head_low, idx)
        cos_rel = float(cosines[idx]) / max_cos
        # Without any skill evidence a title-only hit is weak evidence.
        raw = 0.60 * coverage + 0.25 * title_sig * (0.4 + 0.6 * min(1.0, coverage / 0.25)) + 0.15 * cos_rel * min(1.0, coverage / 0.15 + 0.3)
        scored.append({
            "title": role["title"],
            "family": role["family"],
            "summary": role["summary"],
            "raw": raw,
            "fit": _calibrate(raw),
            "coverage": round(coverage * 100, 1),
            "title_evidence": title_sig > 0,
            "matched_skills": matched[:10],
            "missing_skills": missing[:6],
            "next_roles": role["next"][:4],
            "search_links": search_links(role["title"], location),
            "_index": idx,
        })

    scored.sort(key=lambda r: r["raw"], reverse=True)
    top = scored[:top_k]
    level = experience.get("level", "")
    for r in top:
        r["level_note"] = (
            f"Your CV reads as {level.lower()}; look for {level.lower()} versions of this role."
            if level else ""
        )
    return top


def adjacent_roles(ranked_all_top: List[Dict], limit: int = 3) -> List[str]:
    """Roles in the same family as the best match (useful for pivots)."""
    if not ranked_all_top:
        return []
    family = ranked_all_top[0]["family"]
    return [r["title"] for r in ranked_all_top[1:] if r["family"] != family][:limit]


# ---------------------------------------------------------------------------
# CV quality tips
# ---------------------------------------------------------------------------

def cv_tips(text: str, cv_skills: Dict[str, int]) -> List[str]:
    low = normalize(text)
    words = len(re.findall(r"\w+", low))
    tips: List[str] = []

    if words < 150:
        tips.append("Your CV is very short. Add detail on responsibilities, tools used and results for each role.")
    elif words > 1400:
        tips.append("Your CV is long. Aim for 1-2 pages and keep the most recent, most relevant experience.")

    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", low):
        tips.append("No email address detected. Make sure recruiters and automated systems can find your contact details.")

    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|k\b|m\b|x\b)|[$£€₦]\s?\d", low)
    if len(numbers) < 3:
        tips.append("Add measurable results (for example 'reduced processing time by 30%') - quantified achievements stand out.")

    if not re.search(r"\b(skills|technical skills|core competencies|key skills)\b", low):
        tips.append("Add a clearly labelled Skills section so keyword-based screening picks up your abilities.")
    if not re.search(r"\b(summary|profile|objective|about me)\b", low):
        tips.append("A 2-3 line professional summary at the top helps recruiters see your direction immediately.")
    if not re.search(r"\b(education|qualification|degree|university|college|certification)\b", low):
        tips.append("No education or certification section was detected.")
    if len(cv_skills) < 6:
        tips.append("Few recognisable skills were found. List specific tools, technologies and methods by name.")
    return tips[:6]


# ---------------------------------------------------------------------------
# Interview preparation
# ---------------------------------------------------------------------------

_BEHAVIOURAL = [
    ("Behavioural", "Tell me about a time you faced a difficult deadline. How did you prioritise and what was the outcome?",
     "Use STAR: situation, task, action, result. Quantify the result and show how you communicated with stakeholders."),
    ("Behavioural", "Describe a disagreement with a colleague or manager and how you resolved it.",
     "Show you listened first, focused on the shared goal and reached a concrete outcome."),
    ("Motivation", "Why do you want this role, and where do you see your career in three years?",
     "Connect your strongest skills to the role and show a realistic growth path."),
]


def interview_prep(role_title: str, matched: Sequence[str], missing: Sequence[str], limit: int = 6) -> List[Dict]:
    questions: List[Dict] = []
    for skill in list(matched)[:3]:
        questions.append({
            "role": role_title, "category": "Technical", "difficulty": "Medium",
            "question": f"Describe a real project where you used {skill}. What problem did it solve and what trade-offs did you make?",
            "answer": f"Pick one concrete example from your CV. Explain your specific contribution with {skill}, the decision you made, and the measurable result.",
        })
    for skill in list(missing)[:1]:
        questions.append({
            "role": role_title, "category": "Gap", "difficulty": "Hard",
            "question": f"This role often needs {skill}, which does not appear on your CV. How would you get up to speed?",
            "answer": f"Be honest. Name related experience, describe a learning plan for {skill} (course, small project, mentor) and give an example of learning something quickly before.",
        })
    for category, question, answer in _BEHAVIOURAL:
        questions.append({
            "role": role_title, "category": category, "difficulty": "Medium",
            "question": question, "answer": answer,
        })
    return questions[:limit]


# ---------------------------------------------------------------------------
# One-call profile analysis
# ---------------------------------------------------------------------------

def analyse_cv(text: str, top_roles: int = 5, location: str = "") -> Dict:
    skills = extract_skills(text)
    experience = estimate_experience(text)
    roles = rank_roles(text, skills, experience, top_k=max(top_roles, 8), location=location)
    top = roles[:top_roles]

    best = top[0] if top else None
    strong = [r for r in top if r["fit"] >= STRONG_FIT]
    ordered_skills = sorted(skills.items(), key=lambda kv: (-kv[1] * skill_weight(kv[0]), kv[0]))

    return {
        "skills": skills,
        "skill_list": [s for s, _ in ordered_skills][:30],
        "experience": experience,
        "roles": top,
        "all_ranked": roles,
        "best_role": best["title"] if best else "",
        "has_strong_match": bool(best and best["fit"] >= STRONG_FIT),
        "strong_roles": strong,
        "tips": cv_tips(text, skills),
        "word_count": len(re.findall(r"\w+", text)),
    }


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

_UPPER = {
    "aws", "gcp", "sql", "seo", "sem", "ppc", "crm", "etl", "nlp", "llm", "api", "hse", "ifrs", "gaap",
    "kyc", "aml", "hris", "plc", "scada", "hvac", "cad", "ui", "ux", "pmp", "dcf", "iam", "vpn", "lan",
    "wan", "bgp", "ospf", "vlan", "dns", "dhcp", "tcp/ip", "ci/cd", "html", "css", "jira", "sap", "erp",
    "pos", "kpis", "okrs", "gdpr", "nps", "ats", "dba", "siem", "owasp", "nist", "ccna", "ccnp", "nebosh",
    "iosh", "osha", "sox", "cpa", "acca", "icaew", "cia", "ga4", "spss", "m&e", "gis", "lms", "pms",
    "ncr", "spc", "fmea", "gmp", "haccp", "sdr", "bdr", "ar", "ap", "pid", "hazop",
}
_SPECIAL = {
    "node.js": "Node.js", "next.js": "Next.js", "c#": "C#", ".net": ".NET", "scikit-learn": "scikit-learn",
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "mongodb": "MongoDB", "postgresql": "PostgreSQL",
    "mysql": "MySQL", "power bi": "Power BI", "javascript": "JavaScript", "typescript": "TypeScript",
    "mlops": "MLOps", "fastapi": "FastAPI", "rest api": "REST API", "autocad": "AutoCAD",
    "quickbooks": "QuickBooks", "hubspot": "HubSpot", "solidworks": "SolidWorks", "github actions": "GitHub Actions",
    "mailchimp": "Mailchimp", "e-commerce": "E-commerce", "iso 27001": "ISO 27001", "iso 9001": "ISO 9001",
    "iso 45001": "ISO 45001", "iso 14001": "ISO 14001", "linkedin": "LinkedIn", "salesforce": "Salesforce",
    "wordpress": "WordPress", "woocommerce": "WooCommerce", "bigquery": "BigQuery", "mlflow": "MLflow",
    "hugging face": "Hugging Face", "after effects": "After Effects", "adobe xd": "Adobe XD",
    "jenkins": "Jenkins", "dbt": "dbt", "go": "Go", "r": "R", "solar pv": "Solar PV", "oil and gas": "Oil and gas",
    "hysys": "HYSYS", "pipesim": "PIPESIM", "matlab": "MATLAB", "revit": "Revit", "staad pro": "STAAD Pro",
    "etabs": "ETABS", "ansys": "ANSYS", "lean six sigma": "Lean Six Sigma", "tiktok": "TikTok",
    "youtube": "YouTube", "instagram": "Instagram", "facebook ads": "Facebook Ads", "google ads": "Google Ads",
    "photoshop": "Photoshop", "adobe photoshop": "Adobe Photoshop", "adobe illustrator": "Adobe Illustrator",
    "adobe premiere": "Adobe Premiere", "indesign": "InDesign", "coreldraw": "CorelDRAW",
    "microsoft office": "Microsoft Office", "ms project": "MS Project", "prince2": "PRINCE2",
    "kobo toolbox": "KoboToolbox", "nvivo": "NVivo", "stata": "Stata", "xero": "Xero", "sage": "Sage",
}


def pretty_skill(skill: str) -> str:
    key = str(skill).lower().strip()
    if key in _SPECIAL:
        return _SPECIAL[key]
    if key in _UPPER:
        return key.upper()
    words = []
    for word in key.split(" "):
        words.append(word.upper() if word in _UPPER else word[:1].upper() + word[1:])
    return " ".join(words)


def pretty_skills(skills: Iterable[str]) -> List[str]:
    return [pretty_skill(s) for s in skills]
