"""
===============================================================
TalentMatch AI
Production FastAPI Application

Responsibilities:

    • Accept resume uploads
    • Parse PDF / DOCX / TXT resumes
    • Run AI job matching
    • Rank jobs
    • Generate candidate summary
    • Retrieve relevant interview questions
    • Return lightweight JSON responses
    • Serve a simple web interface

Designed for:
    Render Free
    512 MB RAM

The AI model is loaded ONCE through predict.py.
===============================================================
"""

# ============================================================
# Standard Library
# ============================================================

import time
from pathlib import Path

# ============================================================
# FastAPI
# ============================================================

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

# ============================================================
# Project Modules
# ============================================================

from resume_parser import (
    parse_resume,
    get_resume_info,
    ResumeParsingError,
)

from predict import engine

from ranking_engine import (
    analyze_jobs,
    format_interview_questions,
)

# ============================================================
# Application Configuration
# ============================================================

APP_NAME = "TalentMatch AI"

APP_VERSION = "1.0.0"

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

# Keep API responses small.
DEFAULT_TOP_K_JOBS = 10

DEFAULT_TOP_K_INTERVIEWS = 10

# ============================================================
# Application
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "AI-powered resume and job matching platform "
        "using semantic and keyword-based matching."
    ),
)

# ============================================================
# Templates
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)

# ============================================================
# Startup State
# ============================================================

START_TIME = time.time()


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    response_class=JSONResponse
)
def health_check():
    """
    Lightweight health endpoint.

    Render can use this endpoint to determine
    whether the service is alive.
    """

    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
    }


# ============================================================
# Root Web Page
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(
    request: Request
):
    """
    Serve the TalentMatch AI web interface.
    """

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "version": APP_VERSION,
        }
    )


# ============================================================
# Resume Upload Helper
# ============================================================

async def read_upload(
    file: UploadFile
) -> bytes:
    """
    Read uploaded resume safely into memory.

    The file is NOT permanently stored.

    This is important for Render's limited
    filesystem and memory environment.
    """

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Please provide a resume file."
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported resume format. "
                "Please upload PDF, DOCX or TXT."
            )
        )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded resume is empty."
        )

    if len(contents) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Resume exceeds the 10 MB limit."
        )

    return contents


# ============================================================
# Main Resume Analysis Endpoint
# ============================================================

@app.post(
    "/api/analyze"
)
async def analyze_resume(
    file: UploadFile = File(...)
):
    """
    Complete TalentMatch AI analysis.

    Pipeline:

        Resume Upload
             ↓
        Resume Parser
             ↓
        Resume Text
             ↓
        Prediction Engine
             ↓
        Hybrid Matching
             ↓
        Ranking Engine
             ↓
        Interview Retrieval
             ↓
        JSON Response
    """

    start_time = time.time()

    # --------------------------------------------------------
    # Read resume
    # --------------------------------------------------------

    contents = await read_upload(
        file
    )

    # --------------------------------------------------------
    # Parse resume
    # --------------------------------------------------------

    try:

        resume_text = parse_resume(
            file_bytes=contents,
            filename=file.filename
        )

    except ResumeParsingError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while processing the resume."
            )
        )

    # --------------------------------------------------------
    # Resume information
    # --------------------------------------------------------

    resume_info = get_resume_info(
        filename=file.filename,
        text=resume_text
    )

    # --------------------------------------------------------
    # Job analysis
    # --------------------------------------------------------

    try:

        analysis = analyze_jobs(
            prediction_engine=engine,
            resume_text=resume_text,
            top_k=DEFAULT_TOP_K_JOBS
        )

    except Exception as exc:

        print(
            f"Job analysis error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The AI job matching engine "
                "could not complete the analysis."
            )
        )

    # --------------------------------------------------------
    # Interview recommendations
    # --------------------------------------------------------

    try:

        questions = engine.interview_questions(
            resume_text,
            top_k=DEFAULT_TOP_K_INTERVIEWS
        )

        interview_questions = (
            format_interview_questions(
                questions,
                top_k=DEFAULT_TOP_K_INTERVIEWS
            )
        )

    except Exception as exc:

        print(
            f"Interview retrieval error: {exc}"
        )

        interview_questions = []

    # --------------------------------------------------------
    # Processing time
    # --------------------------------------------------------

    processing_time = round(
        time.time() - start_time,
        3
    )

    # --------------------------------------------------------
    # Final API response
    # --------------------------------------------------------

    return {
        "success": True,

        "service": APP_NAME,

        "resume": resume_info,

        "summary": analysis[
            "summary"
        ],

        "jobs": analysis[
            "jobs"
        ],

        "interview_questions":
            interview_questions,

        "processing_time_seconds":
            processing_time,
    }


# ============================================================
# Lightweight Job Matching Endpoint
# ============================================================

@app.post(
    "/api/jobs"
)
async def match_jobs(
    file: UploadFile = File(...)
):
    """
    Return only job recommendations.

    This endpoint is useful when the frontend
    only needs job matching.
    """

    contents = await read_upload(
        file
    )

    try:

        resume_text = parse_resume(
            file_bytes=contents,
            filename=file.filename
        )

        analysis = analyze_jobs(
            prediction_engine=engine,
            resume_text=resume_text,
            top_k=DEFAULT_TOP_K_JOBS
        )

        return {
            "success": True,
            "summary": analysis["summary"],
            "jobs": analysis["jobs"],
        }

    except ResumeParsingError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        print(
            f"Job matching error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Job matching failed."
        )


# ============================================================
# Interview Preparation Endpoint
# ============================================================

@app.post(
    "/api/interview"
)
async def interview_analysis(
    file: UploadFile = File(...)
):
    """
    Retrieve interview questions relevant
    to the uploaded resume.
    """

    contents = await read_upload(
        file
    )

    try:

        resume_text = parse_resume(
            file_bytes=contents,
            filename=file.filename
        )

        questions = engine.interview_questions(
            resume_text,
            top_k=DEFAULT_TOP_K_INTERVIEWS
        )

        formatted = format_interview_questions(
            questions,
            top_k=DEFAULT_TOP_K_INTERVIEWS
        )

        return {
            "success": True,
            "questions": formatted,
        }

    except ResumeParsingError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        print(
            f"Interview analysis error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Interview question retrieval failed."
            )
        )


# ============================================================
# API Information
# ============================================================

@app.get(
    "/api"
)
def api_info():
    """
    Basic API information.
    """

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",

        "features": [
            "Resume parsing",
            "Semantic job matching",
            "TF-IDF keyword matching",
            "Hybrid job ranking",
            "Match explanations",
            "Interview question retrieval",
        ],

        "endpoints": {
            "health": "/health",
            "web": "/",
            "full_analysis": "/api/analyze",
            "job_matching": "/api/jobs",
            "interview": "/api/interview",
        },
    }


# ============================================================
# Error Handler
# ============================================================

@app.exception_handler(
    ResumeParsingError
)
async def resume_error_handler(
    request: Request,
    exc: ResumeParsingError
):

    return JSONResponse(
        status_code=400,

        content={
            "success": False,
            "error": str(exc),
        }
    )


# ============================================================
# Local Development
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
