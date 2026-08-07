"""
============================================================
TalentMatch AI
Production FastAPI Application
============================================================

Features:

• Resume upload
• PDF/DOCX/TXT extraction
• AI job matching
• Match score
• Match explanation
• Interview preparation
• Employer-friendly dashboard

Designed for Render Free / 512 MB RAM.

Uploaded resumes are processed in memory and are not
permanently stored.
============================================================
"""

import gc

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Request
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse
)

from fastapi.templating import Jinja2Templates

from resume_parser import (
    parse_resume,
    ResumeParsingError
)

from skill_extractor import (
    extract_skills
)

from predict import engine

from ranking_engine import (
    analyze_jobs,
    format_interview_questions
)

from config import (
    MAX_UPLOAD_SIZE,
    TOP_K_JOBS,
    TOP_K_INTERVIEWS
)


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="TalentMatch AI",
    description=(
        "AI-powered resume and job matching platform."
    ),
    version="1.0.0"
)


# ============================================================
# Templates
# ============================================================

templates = Jinja2Templates(
    directory="templates"
)


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health"
)
def health():

    return {

        "status":
            "healthy",

        "service":
            "TalentMatch AI"
    }


# ============================================================
# Home Page
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(
    request: Request
):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# ============================================================
# Resume Analysis
# ============================================================

@app.post(
    "/api/analyze"
)
async def analyze_resume(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Please upload a resume."
        )

    # --------------------------------------------------------
    # Read uploaded file
    #
    # We deliberately read only MAX_UPLOAD_SIZE + 1 bytes.
    # This prevents oversized uploads from consuming RAM.
    # --------------------------------------------------------

    data = await file.read(
        MAX_UPLOAD_SIZE + 1
    )

    if len(data) > MAX_UPLOAD_SIZE:

        del data

        raise HTTPException(
            status_code=413,
            detail=(
                "Resume exceeds the maximum "
                "allowed size of 10 MB."
            )
        )

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Uploaded resume is empty."
        )

    # --------------------------------------------------------
    # Parse resume
    # --------------------------------------------------------

    try:

        resume_text = parse_resume(
            file_bytes=data,
            filename=file.filename
        )

    except ResumeParsingError as exc:

        del data
        gc.collect()

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    # --------------------------------------------------------
    # Release uploaded binary immediately.
    # --------------------------------------------------------

    del data

    # --------------------------------------------------------
    # Job matching
    # --------------------------------------------------------

    try:

        job_analysis = analyze_jobs(
            prediction_engine=engine,
            resume_text=resume_text,
            top_k=TOP_K_JOBS
        )

        # ----------------------------------------------------
        # Skill extraction
        # ----------------------------------------------------

        skills = extract_skills(
            resume_text,
            engine.skills
        )

        # ----------------------------------------------------
        # Interview preparation
        # ----------------------------------------------------

        interview_predictions = (
            engine.interview_questions(
                resume_text,
                top_k=TOP_K_INTERVIEWS
            )
        )

        interview_questions = (
            format_interview_questions(
                interview_predictions,
                top_k=TOP_K_INTERVIEWS
            )
        )

    except Exception as exc:

        gc.collect()

        print(
            "Analysis error:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to complete AI analysis."
            )
        )

    # --------------------------------------------------------
    # Return lightweight JSON
    # --------------------------------------------------------

    response = {

        "success":
            True,

        "resume": {

            "filename":
                file.filename,

            "characters":
                len(resume_text),

            "words":
                len(resume_text.split())
        },

        "skills":
            skills,

        "matching":
            job_analysis,

        "interview":
            interview_questions
    }

    # --------------------------------------------------------
    # Release temporary text after response has been built.
    # --------------------------------------------------------

    del resume_text

    gc.collect()

    return JSONResponse(
        content=response
    )


# ============================================================
# Error Handler
# ============================================================

@app.exception_handler(
    Exception
)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    print(
        "Unhandled application error:",
        repr(exc)
    )

    return JSONResponse(

        status_code=500,

        content={
            "success":
                False,

            "error":
                "An internal server error occurred."
        }
    )
