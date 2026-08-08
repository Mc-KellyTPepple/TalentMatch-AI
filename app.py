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
# Application
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
# Helper: Readable Error
# ============================================================

def readable_error(error):
    """
    Convert an exception/object into a clean string.

    Prevents the frontend from displaying:

        [object Object]
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

            value = error.get(
                key
            )

            if value:

                if isinstance(
                    value,
                    str,
                ):

                    return value

                return str(value)

        return str(error)

    if isinstance(
        error,
        (list, tuple),
    ):

        return "; ".join(
            str(item)
            for item in error
        )

    return str(error)


# ============================================================
# Helper: JSON Error Response
# ============================================================

def error_response(
    message,
    status_code=500,
):
    """
    Return a consistent JSON error response.

    IMPORTANT:
        error is ALWAYS a string.
    """

    return JSONResponse(

        status_code=status_code,

        content={

            "success": False,

            "error": readable_error(
                message
            ),
        },
    )


# ============================================================
# Health Check
# ============================================================

@app.get(
    "/health",
    response_class=JSONResponse,
)
def health_check():
    """
    Lightweight Render health check.

    Does not perform AI inference.
    """

    return {

        "status":
            "healthy",

        "service":
            "TalentMatch AI",

        "version":
            "1.0.0",
    }


# ============================================================
# Home Page
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
):
    """
    Serve the TalentMatch AI web interface.
    """

    return templates.TemplateResponse(

        "index.html",

        {
            "request":
                request,
        },
    )


# ============================================================
# Resume Analysis
# ============================================================

@app.post(
    "/analyze",
)
async def analyze_resume(
    file: UploadFile = File(...),
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

    The uploaded document is processed in memory
    and is not permanently stored.
    """

    # --------------------------------------------------------
    # Temporary references
    # --------------------------------------------------------

    file_bytes = None
    resume_text = None
    skill_details = None
    detected_skills = None
    analysis = None
    questions = None
    interview_results = None

    try:

        # ====================================================
        # Validate filename
        # ====================================================

        if not file.filename:

            return error_response(
                "Please upload a resume.",
                status_code=400,
            )


        # ====================================================
        # Validate extension
        # ====================================================

        filename = file.filename.lower()

        supported_extensions = (
            ".pdf",
            ".docx",
            ".txt",
        )

        if not filename.endswith(
            supported_extensions
        ):

            return error_response(

                (
                    "Unsupported resume format. "
                    "Please upload a PDF, DOCX or TXT file."
                ),

                status_code=400,
            )


        # ====================================================
        # Read uploaded file
        # ====================================================

        file_bytes = await file.read()

        file_size = len(
            file_bytes
        )


        # ====================================================
        # Validate size
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
                    "allowed file size of 10 MB."
                ),

                status_code=400,
            )


        # ====================================================
        # Parse Resume
        # ====================================================

        resume_text = parse_resume(

            file_bytes=file_bytes,

            filename=file.filename,
        )


        # ====================================================
        # Verify extracted text
        # ====================================================

        if not resume_text:

            return error_response(

                (
                    "No readable text could be extracted "
                    "from the uploaded resume."
                ),

                status_code=400,
            )


        # ====================================================
        # Release raw file
        # ====================================================

        file_bytes = None

        gc.collect()


        # ====================================================
        # Extract Resume Skills
        # ====================================================

        skill_details = extract_skill_details(

            resume_text,

            max_skills=100,
        )


        # ====================================================
        # Normalize Skill Details
        # ====================================================

        if skill_details is None:

            skill_details = []


        if not isinstance(
            skill_details,
            list,
        ):

            skill_details = []


        # ====================================================
        # Build Simplified Skill List
        # ====================================================

        detected_skills = []

        for item in skill_details:

            if isinstance(
                item,
                dict,
            ):

                skill = item.get(
                    "skill"
                )

                if skill:

                    detected_skills.append(
                        str(skill)
                    )

            elif isinstance(
                item,
                str,
            ):

                detected_skills.append(
                    item
                )


        # ====================================================
        # Remove Duplicate Skills
        # ====================================================

        detected_skills = list(
            dict.fromkeys(
                detected_skills
            )
        )


        # ====================================================
        # Analyze Jobs
        # ====================================================

        analysis = analyze_jobs(

            prediction_engine=engine,

            resume_text=resume_text,

            top_k=MAX_RETURNED_JOBS,
        )


        # ====================================================
        # Validate Ranking Response
        # ====================================================

        if not isinstance(
            analysis,
            dict,
        ):

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


        if jobs is None:

            jobs = []


        if not isinstance(
            jobs,
            list,
        ):

            jobs = []


        if summary is None:

            summary = {}


        if not isinstance(
            summary,
            dict,
        ):

            summary = {}


        # ====================================================
        # Retrieve Interview Questions
        # ====================================================

        questions = engine.interview_questions(

            resume_text=resume_text,

            top_k=MAX_RETURNED_INTERVIEWS,
        )


        if questions is None:

            questions = []


        # ====================================================
        # Format Interview Questions
        # ====================================================

        interview_results = (
            format_interview_questions(

                questions,

                top_k=MAX_RETURNED_INTERVIEWS,
            )
        )


        if interview_results is None:

            interview_results = []


        if not isinstance(
            interview_results,
            list,
        ):

            interview_results = []


        # ====================================================
        # Candidate Skill Summary
        # ====================================================

        skill_summary = {

            "total_detected":
                len(
                    detected_skills
                ),

            "skills":
                detected_skills,

            "details":
                skill_details,
        }


        # ====================================================
        # Candidate Response
        # ====================================================

        response = {

            "success":
                True,

            "summary":
                summary,

            "skills":
                skill_summary,

            "jobs":
                jobs,

            "interview_questions":
                interview_results,
        }


        # ====================================================
        # Return JSON
        # ====================================================

        return JSONResponse(

            status_code=200,

            content=response,
        )


    # ========================================================
    # Resume Parsing Error
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
    # HTTP Error
    # ========================================================

    except HTTPException as exc:

        return error_response(

            exc.detail,

            status_code=exc.status_code,
        )


    # ========================================================
    # Unexpected Error
    # ========================================================

    except Exception as exc:

        print(
            "=================================================="
        )

        print(
            "TalentMatch AI ANALYSIS ERROR"
        )

        print(
            repr(exc)
        )

        print(
            traceback.format_exc()
        )

        print(
            "=================================================="
        )

        return error_response(

            (
                "Unable to analyze the resume. "
                "The server encountered an internal error. "
                "Please try again."
            ),

            status_code=500,
        )


    # ========================================================
    # Always Release Temporary Objects
    # ========================================================

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
# Parser Status
# ============================================================

@app.get(
    "/api/parser/status",
    response_class=JSONResponse,
)
def parser_status_endpoint():
    """
    Return the active resume parser configuration.

    This is especially useful for confirming that Render
    is actually running the new pypdf implementation.
    """

    try:

        return parser_status()

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
# Skill Engine Status
# ============================================================

@app.get(
    "/api/skills/status",
    response_class=JSONResponse,
)
def skills_status():
    """
    Return the status of the skill extraction engine.
    """

    try:

        status = skill_engine_status()

        if status is None:

            return {

                "status":
                    "unknown",

                "message":
                    "Skill engine returned no status.",
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
# API Information
# ============================================================

@app.get(
    "/api",
    response_class=JSONResponse,
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

        "pdf_engine":
            "pypdf",

        "pymupdf_required":
            False,

        "deployment": {

            "platform":
                "Render",

            "mode":
                "CPU inference",

            "memory_target":
                "512 MB",
        },
    }


# ============================================================
# Lightweight Readiness Check
# ============================================================

@app.get(
    "/ready",
    response_class=JSONResponse,
)
def readiness_check():
    """
    Determine whether the application has its critical
    runtime components available.
    """

    try:

        # ----------------------------------------------------
        # Parser status
        # ----------------------------------------------------

        current_parser_status = parser_status()

        parser_ready = (
            isinstance(
                current_parser_status,
                dict,
            )
            and
            current_parser_status.get(
                "status"
            ) == "ready"
        )


        # ----------------------------------------------------
        # Skill engine status
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Overall readiness
        # ----------------------------------------------------

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
