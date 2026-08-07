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
    Skill Extractor
        ↓
    Prediction Engine
        ↓
    Ranking Engine
        ↓
    Interview Retrieval
        ↓
    JSON / HTML Response

Features:
    • PDF / DOCX / TXT resume parsing
    • AI semantic job matching
    • TF-IDF keyword matching
    • Hybrid job ranking
    • Resume skill extraction
    • Skill frequency information
    • Interview question retrieval
    • Employer-friendly candidate summary

Memory strategy:
    • Single prediction engine
    • CPU inference
    • No permanent resume storage
    • Upload size protection
    • Temporary objects released after processing
"""

# ============================================================
# Imports
# ============================================================

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


# ============================================================
# Internal Modules
# ============================================================

from config import (
    MAX_UPLOAD_SIZE,
    MAX_RETURNED_JOBS,
    MAX_RETURNED_INTERVIEWS,
)

from resume_parser import (
    parse_resume,
    ResumeParsingError,
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
# Application
# ============================================================

app = FastAPI(

    title="TalentMatch AI",

    description=(
        "AI-powered resume analysis, "
        "job matching, skill extraction "
        "and interview preparation platform."
    ),

    version="1.0.0",
)


# ============================================================
# Static Files
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory="static"
    ),
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
    """
    Lightweight Render health check.

    This endpoint intentionally does not perform AI inference.
    """

    return {

        "status": "healthy",

        "service": "TalentMatch AI",

        "version": "1.0.0",
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
    """
    Serve the TalentMatch AI web interface.
    """

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
    """
    Analyze an uploaded resume.

    Processing pipeline:

        Uploaded Resume
              ↓
        File Validation
              ↓
        Resume Parsing
              ↓
        Skill Extraction
              ↓
        Hybrid Job Matching
              ↓
        Interview Retrieval
              ↓
        JSON Response

    The uploaded document is processed in memory and
    is not permanently stored.
    """

    # ========================================================
    # Validate filename
    # ========================================================

    if not file.filename:

        raise HTTPException(

            status_code=400,

            detail=(
                "Please upload a resume."
            )
        )


    # ========================================================
    # Read upload
    # ========================================================

    file_bytes = None

    try:

        # ----------------------------------------------------
        # Read file into memory.
        #
        # MAX_UPLOAD_SIZE is enforced immediately afterward.
        # ----------------------------------------------------

        file_bytes = await file.read()

        file_size = len(
            file_bytes
        )

        # ----------------------------------------------------
        # Protect Render memory.
        # ----------------------------------------------------

        if file_size <= 0:

            return JSONResponse(

                status_code=400,

                content={

                    "success": False,

                    "error":
                        "The uploaded resume is empty.",
                }
            )


        if file_size > MAX_UPLOAD_SIZE:

            return JSONResponse(

                status_code=400,

                content={

                    "success": False,

                    "error": (
                        "Resume exceeds the maximum "
                        "allowed file size of 10 MB."
                    ),
                }
            )


        # ====================================================
        # Parse Resume
        # ====================================================

        resume_text = parse_resume(

            file_bytes=file_bytes,

            filename=file.filename,
        )


        # ----------------------------------------------------
        # Release raw document bytes immediately.
        # ----------------------------------------------------

        del file_bytes

        file_bytes = None


        # ====================================================
        # Extract Skills
        # ====================================================

        skill_details = extract_skill_details(

            resume_text,

            max_skills=100
        )


        # ====================================================
        # Simplified Skill List
        # ====================================================

        detected_skills = [

            item["skill"]

            for item in skill_details
        ]


        # ====================================================
        # Analyze Jobs
        # ====================================================

        analysis = analyze_jobs(

            prediction_engine=engine,

            resume_text=resume_text,

            top_k=MAX_RETURNED_JOBS,
        )


        # ====================================================
        # Retrieve Interview Questions
        # ====================================================

        questions = engine.interview_questions(

            resume_text=resume_text,

            top_k=MAX_RETURNED_INTERVIEWS,
        )


        # ====================================================
        # Format Interview Results
        # ====================================================

        interview_results = (
            format_interview_questions(

                questions,

                top_k=MAX_RETURNED_INTERVIEWS,
            )
        )


        # ====================================================
        # Candidate Skill Summary
        # ====================================================

        skill_summary = {

            "total_detected": len(
                detected_skills
            ),

            "skills": detected_skills,

            "details": skill_details,
        }


        # ====================================================
        # Candidate Response
        # ====================================================

        response = {

            "success": True,

            "summary": analysis[
                "summary"
            ],

            "skills": skill_summary,

            "jobs": analysis[
                "jobs"
            ],

            "interview_questions":
                interview_results,

        }


        # ====================================================
        # Release Temporary Objects
        # ====================================================

        del resume_text

        del analysis

        del questions

        del interview_results

        del skill_details

        del detected_skills

        gc.collect()


        # ====================================================
        # Return
        # ====================================================

        return response


    # ========================================================
    # Resume Parsing Error
    # ========================================================

    except ResumeParsingError as exc:

        return JSONResponse(

            status_code=400,

            content={

                "success": False,

                "error": str(exc),
            }
        )


    # ========================================================
    # Unexpected Error
    # ========================================================

    except Exception as exc:

        print(
            "TalentMatch AI analysis error:",
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


    # ========================================================
    # Always Release Uploaded Bytes
    # ========================================================

    finally:

        if file_bytes is not None:

            del file_bytes

        gc.collect()


# ============================================================
# Skill Engine Status
# ============================================================

@app.get(
    "/api/skills/status",
    response_class=JSONResponse
)
def skills_status():
    """
    Return the status of the skill extraction engine.

    Useful for deployment diagnostics.
    """

    try:

        return skill_engine_status()

    except Exception as exc:

        return JSONResponse(

            status_code=500,

            content={

                "status": "error",

                "error": str(exc),
            }
        )


# ============================================================
# API Information
# ============================================================

@app.get(
    "/api",
    response_class=JSONResponse
)
def api_info():
    """
    Basic API information.
    """

    return {

        "name":
            "TalentMatch AI",

        "version":
            "1.0.0",

        "status":
            "online",

        "features": [

            "Resume parsing",

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

        "deployment": {

            "platform":
                "Render",

            "mode":
                "CPU inference",

            "memory_target":
                "512 MB",
        }
    }


# ============================================================
# Lightweight Readiness Check
# ============================================================

@app.get(
    "/ready",
    response_class=JSONResponse
)
def readiness_check():
    """
    Determine whether the application has its critical
    runtime components available.

    Unlike /health, this checks the skill artifact layer.
    """

    try:

        skill_status = skill_engine_status()

        skills_ready = (
            skill_status.get(
                "skills_file_exists",
                False
            )
        )

        return {

            "status":
                "ready"
                if skills_ready
                else "degraded",

            "prediction_engine":
                "loaded",

            "skill_engine":
                skill_status.get(
                    "status",
                    "unknown"
                ),
        }

    except Exception as exc:

        return JSONResponse(

            status_code=503,

            content={

                "status": "not_ready",

                "error": str(exc),
            }
        )


# ============================================================
# Application Shutdown
# ============================================================

@app.on_event(
    "shutdown"
)
def shutdown_event():
    """
    Perform lightweight cleanup when the Render instance
    shuts down.
    """

    gc.collect()
