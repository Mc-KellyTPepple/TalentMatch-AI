"""End-to-end tests. Run from the project root:  python -m pytest -q"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["LIVE_JOBS"] = "0"        # tests never touch the network
os.environ.pop("GROQ_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

import app as application
import live_jobs
import ranking_engine
from sample_cvs import CVS

client = TestClient(application.app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    application._hits.clear()
    yield
    application._hits.clear()


def post(text, name="cv.txt", **fields):
    files = {"file": (name, text if isinstance(text, bytes) else text.encode(), "application/octet-stream")}
    return client.post("/analyze", files=files, data=fields)


# ----------------------------------------------------------------- basics
def test_health_is_instant_and_static_assets_served():
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.parametrize("key,expected", [
    ("ml", "Machine Learning Engineer"),
    ("accountant", "Accountant"),
    ("marketing", "Digital Marketing Specialist"),
    ("nurse", "Nurse / Healthcare Assistant"),
    ("petro", "Petroleum / Process Engineer"),
    ("grad", "Full Stack Developer"),
])
def test_best_role_is_correct(key, expected):
    r = post(CVS[key])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["roles"][0]["title"] == expected
    assert body["profile"]["skills"]
    assert body["interviews"]


def test_vague_cv_is_not_forced_into_a_match():
    body = post(CVS["vague"] + " " * 5 + "x" * 0).json()
    assert body["jobs"] == []
    assert any("did not clearly match" in n for n in body["notices"])
    assert body["interviews"] == []


# ----------------------------------------------------------- input errors
def test_rejects_bad_input():
    assert post(b"", "cv.txt").status_code == 400
    assert post("hello", "cv.txt").status_code == 400                  # too short
    assert post("some text " * 20, "cv.exe").status_code == 400         # wrong type
    assert post(b"x" * (10 * 1024 * 1024 + 5), "cv.txt").status_code == 413
    assert post(b"%PDF-1.4 not really a pdf", "cv.pdf").status_code == 400


def test_pdf_and_docx_upload():
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in CVS["accountant"].splitlines():
        c.drawString(40, y, line[:110]); y -= 14
    c.save()
    assert post(buf.getvalue(), "cv.pdf").json()["roles"][0]["title"] == "Accountant"

    from docx import Document
    d = Document()
    t = d.add_table(rows=1, cols=1)
    t.rows[0].cells[0].text = CVS["marketing"]          # content only inside a table
    out = io.BytesIO(); d.save(out)
    assert post(out.getvalue(), "cv.docx").json()["roles"][0]["title"] == "Digital Marketing Specialist"


# ------------------------------------------------------------- live jobs
def _job(i, title, text, location="Worldwide", source="Remotive", remote=True, posted="2026-10-08", visa=False):
    return {"id": f"t{i}", "source": source, "title": title, "company": f"Co{i}", "location": location,
            "remote": remote, "visa": visa, "url": f"https://example.com/{i}", "posted": posted,
            "job_type": "full time", "salary": "", "category": "", "tags": [], "text": text}


@pytest.fixture()
def feed():
    jobs = [
        _job(1, "Machine Learning Engineer", "PyTorch, Python, MLOps, Docker, AWS, model deployment, FastAPI, NLP, SQL"),
        _job(2, "Senior Machine Learning Engineer", "Python PyTorch TensorFlow Kubernetes MLflow deep learning AWS", location="USA Only"),
        _job(3, "Data Scientist", "Python scikit-learn statistics SQL pandas machine learning A/B testing"),
        _job(4, "Accountant", "IFRS reconciliation QuickBooks accounts payable general ledger tax"),
        _job(5, "Sales Representative", "cold calling CRM prospecting negotiation Salesforce"),
        _job(6, "AI Engineer", "LLM transformers Python PyTorch deep learning NLP Hugging Face", location="Berlin",
             source="Arbeitnow", remote=False, visa=True),
        _job(7, "Machine Learning Intern", "Python machine learning scikit-learn", location="Africa"),
        _job(8, "Backend Developer", "Python FastAPI PostgreSQL Docker REST API Redis", location="Europe Only"),
    ]
    live_jobs.store._jobs = jobs
    live_jobs.store._fetched_at = __import__("time").time()
    live_jobs.store._version += 1
    live_jobs.store._status = {"remotive": "ok", "arbeitnow": "ok"}
    os.environ["LIVE_JOBS"] = "1"
    yield jobs
    os.environ["LIVE_JOBS"] = "0"
    live_jobs.store._jobs = []


def test_live_jobs_are_ranked_sensibly(feed):
    body = post(CVS["ml"]).json()
    titles = [j["title"] for j in body["jobs"]]
    assert titles[0] in {"Machine Learning Engineer", "Senior Machine Learning Engineer", "AI Engineer"}
    assert "Accountant" not in titles and "Sales Representative" not in titles
    assert body["summary"]["jobs_analyzed"] == len(feed)
    for j in body["jobs"]:
        assert j["url"].startswith("https://") and j["source"] in {"Remotive", "Arbeitnow"}
        assert 0 < j["match_score"] <= 98


def test_location_eligibility_for_candidate_in_nigeria(feed):
    body = post(CVS["ml"], country="Nigeria").json()
    by_title = {j["title"]: j for j in body["jobs"]}
    assert by_title["Machine Learning Engineer"]["eligibility"] == "Open worldwide"
    assert by_title["Senior Machine Learning Engineer"]["eligibility"].startswith("Restricted")
    assert by_title["AI Engineer"]["eligibility"].startswith("Visa sponsorship")
    # Restricted jobs are down-ranked below the open-worldwide equivalent.
    assert by_title["Machine Learning Engineer"]["match_score"] > by_title["Senior Machine Learning Engineer"]["match_score"]

    only = post(CVS["ml"], country="Nigeria", only_eligible="true").json()
    titles = [j["title"] for j in only["jobs"]]
    assert "Senior Machine Learning Engineer" not in titles
    assert all(not j["eligibility"].startswith(("Restricted", "On-site")) for j in only["jobs"])


def test_remote_only_filter(feed):
    body = post(CVS["ml"], remote_only="true").json()
    assert body["jobs"] and all(j["remote"] for j in body["jobs"])


def test_seniority_penalty_for_stretch_roles(feed):
    live_jobs.store._jobs.append(_job(20, "Senior Full Stack Developer", "JavaScript React Node.js SQL Git HTML CSS TypeScript"))
    live_jobs.store._jobs.append(_job(21, "Full Stack Developer", "JavaScript React Node.js SQL Git HTML CSS TypeScript"))
    live_jobs.store._version += 1
    body = post(CVS["grad"]).json()           # an entry-level CV
    by_title = {j["title"]: j for j in body["jobs"]}
    assert by_title["Full Stack Developer"]["match_score"] > by_title["Senior Full Stack Developer"]["match_score"]
    assert "stretch" in by_title["Senior Full Stack Developer"]["level_note"]


def test_groq_blend_and_failure_fallback(feed, monkeypatch):
    monkeypatch.setattr(ranking_engine, "groq_status", lambda: {"available": True})

    def fake_ok(resume, jobs, max_jobs=None):
        assessments = [{"candidate_id": i, "overall_score": 90 if i == 0 else 20, "role_relevance": 90,
                        "decision": "strong_match" if i == 0 else "not_a_match", "matched_skills": ["Python"],
                        "missing_skills": [], "reason": "AI says fit."} for i in range(len(jobs))]
        return {"available": True, "model": "fake", "error": None, "assessments": assessments}

    monkeypatch.setattr(ranking_engine, "rerank_jobs_with_groq", fake_ok)
    body = post(CVS["ml"]).json()
    assert body["summary"]["groq_enabled"] is True
    assert len(body["jobs"]) == 1 and body["jobs"][0]["ai_reranked"] and body["jobs"][0]["reason"] == "AI says fit."

    monkeypatch.setattr(ranking_engine, "rerank_jobs_with_groq",
                        lambda *a, **k: {"available": False, "model": "x", "error": "RateLimitError", "assessments": []})
    body = post(CVS["ml"]).json()
    assert body["jobs"] and not body["jobs"][0]["ai_reranked"]
    assert any("AI re-ranking was unavailable" in n for n in body["notices"])


def test_feed_down_still_returns_roles():
    live_jobs.store._jobs = []
    body = post(CVS["ml"]).json()
    assert body["roles"] and body["jobs"] == []
    assert body["roles"][0]["search_links"]["linkedin"].startswith("https://www.linkedin.com/jobs/search")


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(application, "_RATE_LIMIT", 2)
    application._hits.clear()
    codes = [post(CVS["nurse"]).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    application._hits.clear()
