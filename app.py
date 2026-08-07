"""
TalentMatch AI
Production FastAPI Application

Designed for:
Render Free
512 MB RAM
CPU inference

Architecture:

Browser
   ↓
FastAPI
   ↓
Resume Parser
   ↓
Prediction Engine
   ↓
Ranking Engine
   ↓
Interview Retrieval
   ↓
JSON / HTML Response
"""

import os
import gc

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

from resume_parser import (
    parse_resume,
    ResumeParsingError,
)

from predict import engine

from ranking_engine import (
    analyze_jobs,
    format_interview_questions,
)


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="TalentMatch AI",
    description=(
        "AI-powered resume and job matching platform."
    ),
    version="1.0.0",
)


# ============================================================
# Static Files
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
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
    "/health",
    response_class=JSONResponse
)
def health_check():

    return {
        "status": "healthy",
        "service": "TalentMatch AI",
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
    "/analyze"
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
    # Upload is processed in memory.
    # It is NOT permanently stored.
    # --------------------------------------------------------

    file_bytes = await file.read()

    try:

        # ----------------------------------------------------
        # Parse resume
        # ----------------------------------------------------

        resume_text = parse_resume(
            file_bytes=file_bytes,
            filename=file.filename,
        )

        # ----------------------------------------------------
        # Release uploaded bytes as early as possible.
        # ----------------------------------------------------

        del file_bytes

        # ----------------------------------------------------
        # Analyze jobs
        # ----------------------------------------------------

        analysis = analyze_jobs(
            prediction_engine=engine,
            resume_text=resume_text,
        )

        # ----------------------------------------------------
        # Retrieve relevant interview questions
        # ----------------------------------------------------

        questions = engine.interview_questions(
            resume_text=resume_text
        )

        interview_results = (
            format_interview_questions(
                questions,
                top_k=5
            )
        )

        # ----------------------------------------------------
        # Candidate response
        # ----------------------------------------------------

        response = {

            "success": True,

            "summary": analysis[
                "summary"
            ],

            "jobs": analysis[
                "jobs"
            ],

            "interview_questions":
                interview_results,

        }

        # ----------------------------------------------------
        # Release large temporary objects
        # ----------------------------------------------------

        del resume_text
        del analysis
        del questions
        del interview_results

        gc.collect()

        return response

    except ResumeParsingError as exc:

        return JSONResponse(

            status_code=400,

            content={
                "success": False,
                "error": str(exc),
            }
        )

    except Exception as exc:

        print(
            "Analysis error:",
            repr(exc)
        )

        gc.collect()

        return JSONResponse(

            status_code=500,

            content={
                "success": False,
                "error": (
                    "Unable to analyze the resume. "
                    "Please try again."
                ),
            }
        )


# ============================================================
# API Information
# ============================================================

@app.get(
    "/api"
)
def api_info():

    return {

        "name": "TalentMatch AI",

        "version": "1.0.0",

        "features": [

            "Resume parsing",

            "Semantic job matching",

            "TF-IDF keyword matching",

            "Hybrid job ranking",

            "Interview question retrieval",

        ],

        "status": "online",
    }


# ============================================================
# Application Shutdown
# ============================================================

@app.on_event("shutdown")
def shutdown_event():

    gc.collect()
