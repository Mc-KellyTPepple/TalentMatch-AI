
Those are **not valid Python** if they actually exist in the file.

Also, your `/api/parser/status` endpoint should exist in the deployed `app.py`. Since you got **Not Found**, I recommend replacing the entire `app.py` rather than trying to patch individual lines.

## 1. Replace your entire `app.py`

Delete everything currently in `app.py` and put this in its place:

```python
"""
TalentMatch AI
Production FastAPI Application

Designed for:
    Render Free
    512 MB RAM
    CPU inference

Features:
    PDF / DOCX / TXT resume parsing
    AI semantic job matching
    TF-IDF keyword matching
    Hybrid job ranking
    Resume skill extraction
    Interview question retrieval
"""

import gc
import traceback

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Request,
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# ============================================================
# INTERNAL MODULES
# ============================================================

from config import (
    MAX_UPLOAD_SIZE,
    MAX_RETURNED_JOBS,
    MAX_RETURNED_INTERVIEWS,
)

from resume_parser import (
    parse_resume,
    ResumeParsingError,
    parser_status,
)

from predict import engine

from ranking_engine import (
    analyze_jobs,
    format_interview_questions,
)

from skill_extractor import (
    extract_skill_details,
    skill_engine_status,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="TalentMatch AI",
    description=(
        "AI-powered resume analysis, job matching, "
        "skill extraction and interview preparation platform."
    ),
    version="1.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# ERROR HELPER
# ============================================================

def readable_error(error):
    """
    Convert an exception or object into a readable string.
    """

    if error is None:
        return "An unknown error occurred."

    if isinstance(error, str):
        return error

    if isinstance(error, dict):

        for key in (
            "error",
            "message",
            "detail",
            "msg",
        ):

            value = error.get(key)

            if value:

                if isinstance(value, str):
                    return value

                return str(value)

        return str(error)

    if isinstance(error, (list, tuple)):

        return "; ".join(
            str(item)
            for item in error
        )

    return str(error)


# ============================================================
# JSON ERROR RESPONSE
# ============================================================

def error_response(
    message,
    status_code=500,
):
    """
    Return a consistent JSON error response.
    """

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": readable_error(message),
        },
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_class=JSONResponse,
)
def health_check():

    return {
        "status": "healthy",
        "service": "TalentMatch AI",
        "version": "1.0.0",
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


# ============================================================
# RESUME ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded resume.

    Pipeline:

        Upload
          ↓
        Validation
          ↓
        Resume Parser
          ↓
        Skill Extraction
          ↓
        Job Matching
          ↓
        Interview Questions
          ↓
        JSON Response
    """

    file_bytes = None
    resume_text = None
    skill_details = None
    detected_skills = None
    analysis = None
    questions = None
    interview_results = None

    try:

        # ====================================================
        # VALIDATE FILENAME
        # ====================================================

        if not file.filename:

            return error_response(
                "Please upload a resume.",
                status_code=400,
            )

        filename = file.filename.strip()

        if not filename:

            return error_response(
                "Please upload a resume.",
                status_code=400,
            )


        # ====================================================
        # VALIDATE EXTENSION
        # ====================================================

        extension = ""

        if "." in filename:
            extension = (
                "." +
                filename.rsplit(".", 1)[1].lower()
            )

        supported_extensions = {
            ".pdf",
            ".docx",
            ".txt",
        }

        if extension not in supported_extensions:

            return error_response(
                (
                    "Unsupported resume format. "
                    "Please upload a PDF, DOCX or TXT file."
                ),
                status_code=400,
            )


        # ====================================================
        # READ FILE
        # ====================================================

        file_bytes = await file.read()

        file_size = len(file_bytes)


        # ====================================================
        # VALIDATE FILE SIZE
        # ====================================================

        if file_size <= 0:

            return error_response(
                "The uploaded resume is empty.",
                status_code=400,
            )


        if file_size > MAX_UPLOAD_SIZE:

            return error_response(
                (
                    "Resume exceeds the maximum "
                    "allowed file size."
                ),
                status_code=400,
            )


        # ====================================================
        # PARSE RESUME
        # ====================================================

        resume_text = parse_resume(
            file_bytes=file_bytes,
            filename=filename,
        )


        if not resume_text:

            return error_response(
                (
                    "No readable text could be extracted "
                    "from the uploaded resume."
                ),
                status_code=400,
            )


        # ====================================================
        # RELEASE RAW FILE
        # ====================================================

        file_bytes = None

        gc.collect()


        # ====================================================
        # EXTRACT SKILLS
        # ====================================================

        skill_details = extract_skill_details(
            resume_text,
            max_skills=100,
        )


        if not isinstance(skill_details, list):
            skill_details = []


        # ====================================================
        # BUILD SKILL LIST
        # ====================================================

        detected_skills = []

        for item in skill_details:

            if isinstance(item, dict):

                skill = item.get("skill")

                if skill:
                    detected_skills.append(
                        str(skill)
                    )

            elif isinstance(item, str):

                detected_skills.append(item)


        # ====================================================
        # REMOVE DUPLICATES
        # ====================================================

        detected_skills = list(
            dict.fromkeys(
                detected_skills
            )
        )


        # ====================================================
        # ANALYZE JOBS
        # ====================================================

        analysis = analyze_jobs(
            prediction_engine=engine,
            resume_text=resume_text,
            top_k=MAX_RETURNED_JOBS,
        )


        if not isinstance(analysis, dict):

            raise RuntimeError(
                "The job ranking engine returned "
                "an invalid response."
            )


        jobs = analysis.get(
            "jobs",
            [],
        )

        summary = analysis.get(
            "summary",
            {},
        )


        if not isinstance(jobs, list):
            jobs = []


        if not isinstance(summary, dict):
            summary = {}


        # ====================================================
        # INTERVIEW QUESTIONS
        # ====================================================

        questions = engine.interview_questions(
            resume_text=resume_text,
            top_k=MAX_RETURNED_INTERVIEWS,
        )


        if not isinstance(questions, list):
            questions = []


        # ====================================================
        # FORMAT INTERVIEW QUESTIONS
        # ====================================================

        interview_results = format_interview_questions(
            questions,
            top_k=MAX_RETURNED_INTERVIEWS,
        )


        if not isinstance(interview_results, list):
            interview_results = []


        # ====================================================
        # SKILL SUMMARY
        # ====================================================

        skill_summary = {
            "total_detected": len(detected_skills),
            "skills": detected_skills,
            "details": skill_details,
        }


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response = {
            "success": True,
            "summary": summary,
            "skills": skill_summary,
            "jobs": jobs,
            "interview_questions": interview_results,
        }


        return JSONResponse(
            status_code=200,
            content=response,
        )


    # ========================================================
    # RESUME PARSING ERROR
    # ========================================================

    except ResumeParsingError as exc:

        print(
            "Resume parsing error:",
            repr(exc),
        )

        return error_response(
            str(exc),
            status_code=400,
        )


    # ========================================================
    # HTTP ERROR
    # ========================================================

    except HTTPException as exc:

        return error_response(
            exc.detail,
            status_code=exc.status_code,
        )


    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as exc:

        print("=" * 60)
        print("TALENTMATCH AI ANALYSIS ERROR")
        print(repr(exc))
        print(traceback.format_exc())
        print("=" * 60)

        return error_response(
            (
                "Unable to analyze the resume. "
                "The server encountered an internal error."
            ),
            status_code=500,
        )


    finally:

        file_bytes = None
        resume_text = None
        skill_details = None
        detected_skills = None
        analysis = None
        questions = None
        interview_results = None

        gc.collect()


# ============================================================
# PARSER STATUS
# ============================================================

@app.get(
    "/api/parser/status",
    response_class=JSONResponse,
)
def parser_status_endpoint():

    try:

        status = parser_status()

        return {
            "success": True,
            "parser": status,
        }

    except Exception as exc:

        print(
            "Parser status error:",
            repr(exc),
        )

        return error_response(
            str(exc),
            status_code=500,
        )


# ============================================================
# SKILL ENGINE STATUS
# ============================================================

@app.get(
    "/api/skills/status",
    response_class=JSONResponse,
)
def skills_status():

    try:

        status = skill_engine_status()

        if status is None:

            return {
                "status": "unknown",
                "message": (
                    "Skill engine returned no status."
                ),
            }

        return status

    except Exception as exc:

        print(
            "Skill engine status error:",
            repr(exc),
        )

        return error_response(
            str(exc),
            status_code=500,
        )


# ============================================================
# API INFORMATION
# ============================================================

@app.get(
    "/api",
    response_class=JSONResponse,
)
def api_info():

    return {

        "name": "TalentMatch AI",

        "version": "1.0.0",

        "status": "online",

        "features": [
            "Resume parsing",
            "PDF parsing with pypdf",
            "DOCX parsing with python-docx",
            "TXT parsing",
            "Semantic job matching",
            "TF-IDF keyword matching",
            "Hybrid job ranking",
            "Resume skill extraction",
            "Skill frequency analysis",
            "Interview question retrieval",
        ],

        "supported_resume_formats": [
            "PDF",
            "DOCX",
            "TXT",
        ],

        "pdf_engine": "pypdf",

        "pymupdf_required": False,

        "deployment": {
            "platform": "Render",
            "mode": "CPU inference",
            "memory_target": "512 MB",
        },
    }


# ============================================================
# READINESS CHECK
# ============================================================

@app.get(
    "/ready",
    response_class=JSONResponse,
)
def readiness_check():

    try:

        current_parser_status = parser_status()

        parser_ready = (
            isinstance(
                current_parser_status,
                dict,
            )
            and
            current_parser_status.get("status")
            == "ready"
        )


        skill_status = skill_engine_status()

        if not isinstance(
            skill_status,
            dict,
        ):
            skill_status = {}


        skills_ready = bool(
            skill_status.get(
                "skills_file_exists",
                False,
            )
        )


        application_ready = (
            parser_ready
            and
            skills_ready
        )


        return {

            "status":
                "ready"
                if application_ready
                else "degraded",

            "prediction_engine":
                "loaded",

            "resume_parser":
                current_parser_status,

            "skill_engine":
                skill_status.get(
                    "status",
                    "unknown",
                ),

            "skills_file_exists":
                skills_ready,
        }


    except Exception as exc:

        print(
            "Readiness check error:",
            repr(exc),
        )

        return JSONResponse(

            status_code=503,

            content={

                "status":
                    "not_ready",

                "prediction_engine":
                    "unknown",

                "resume_parser":
                    "error",

                "skill_engine":
                    "error",

                "error":
                    readable_error(exc),
            },
        )


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown_event():

    gc.collect()
