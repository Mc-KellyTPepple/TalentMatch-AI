"""
TalentMatch AI
Production FastAPI Application

Optimized for:
- Render Free
- 512 MB RAM
- CPU inference
- Lazy AI model loading
- PDF/DOCX/TXT resume parsing

Important memory strategy:
- Do NOT load the prediction engine during application startup.
- Load the prediction engine only when /analyze is requested.
- Do not permanently store uploaded resumes.
- Release large temporary objects after processing.
"""

# ============================================================
# STANDARD LIBRARY
# ============================================================

import gc
import traceback

# ============================================================
# FASTAPI
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
from fastapi.staticfiles import StaticFiles

# ============================================================
# CONFIG
# ============================================================

from config import (
    MAX_UPLOAD_SIZE,
    MAX_RETURNED_JOBS,
    MAX_RETURNED_INTERVIEWS,
)

# ============================================================
# RESUME PARSER
#
# IMPORTANT:
# This module should NOT load any large ML model.
# ============================================================

from resume_parser import (
    parse_resume,
    ResumeParsingError,
    parser_status,
)

# ============================================================
# SKILL EXTRACTOR
#
# This should only load lightweight skill data.
# ============================================================

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
# LAZY AI ENGINE
# ============================================================

_prediction_engine = None


def get_prediction_engine():
    """
    Load the AI prediction engine only when it is actually needed.

    This is critical for Render Free because importing predict.py
    during application startup may load the ONNX model and large
    embedding files into memory.
    """

    global _prediction_engine

    if _prediction_engine is None:

        print("Loading TalentMatch AI prediction engine...")

        from predict import engine

        _prediction_engine = engine

        print("TalentMatch AI prediction engine loaded.")

        # Give Python a chance to release temporary allocations.
        gc.collect()

    return _prediction_engine


# ============================================================
# HELPER: READABLE ERROR
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
# HELPER: JSON ERROR RESPONSE
# ============================================================

def error_response(
    message,
    status_code=500,
):

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

        "memory_strategy": "lazy_model_loading",
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

        "prediction_engine": "lazy_loaded",

        "deployment": {

            "platform": "Render",

            "mode": "CPU inference",

            "memory_target": "512 MB",
        },
    }


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

        if status is None:

            return {
                "status": "unknown",
                "message": "Parser returned no status.",
            }

        return status

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
# READINESS CHECK
# ============================================================

@app.get(
    "/ready",
    response_class=JSONResponse,
)
def readiness_check():

    try:

        # ----------------------------------------------------
        # Parser
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
        # Skill engine
        # ----------------------------------------------------

        current_skill_status = skill_engine_status()

        if not isinstance(
            current_skill_status,
            dict,
        ):

            current_skill_status = {}

        skills_ready = bool(
            current_skill_status.get(
                "skills_file_exists",
                False,
            )
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # We deliberately do NOT load the prediction engine
        # here.
        #
        # /ready must remain lightweight.
        # ----------------------------------------------------

        application_ready = (
            parser_ready
            and skills_ready
        )

        return {

            "status":
                "ready"
                if application_ready
                else "degraded",

            "prediction_engine":
                "lazy_loaded",

            "prediction_engine_loaded":
                _prediction_engine is not None,

            "resume_parser":
                current_parser_status,

            "skill_engine":
                current_skill_status,

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

                "status": "not_ready",

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
# RESUME ANALYSIS
# ============================================================

@app.post(
    "/analyze",
)
async def analyze_resume(
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Temporary objects
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
        # VALIDATE FILENAME
        # ====================================================

        if not file.filename:

            return error_response(
                "Please upload a resume.",
                status_code=400,
            )

        # ====================================================
        # VALIDATE EXTENSION
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
        # READ FILE
        # ====================================================

        file_bytes = await file.read()

        file_size = len(file_bytes)

        # ====================================================
        # VALIDATE SIZE
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
        # PARSE RESUME
        # ====================================================

        resume_text = parse_resume(

            file_bytes=file_bytes,

            filename=file.filename,
        )

        # ====================================================
        # RELEASE RAW FILE
        # ====================================================

        file_bytes = None

        # ====================================================
        # VERIFY TEXT
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
        # LIMIT EXCESSIVELY LARGE TEXT
        #
        # Prevents an unusually large document from consuming
        # excessive memory during NLP processing.
        # ====================================================

        MAX_RESUME_TEXT_CHARS = 100000

        if len(resume_text) > MAX_RESUME_TEXT_CHARS:

            resume_text = resume_text[
                :MAX_RESUME_TEXT_CHARS
            ]

        # ====================================================
        # EXTRACT SKILLS
        # ====================================================

        skill_details = extract_skill_details(

            resume_text,

            max_skills=100,
        )

        if skill_details is None:
            skill_details = []

        if not isinstance(
            skill_details,
            list,
        ):
            skill_details = []

        # ====================================================
        # BUILD SKILL LIST
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
        # REMOVE DUPLICATES
        # ====================================================

        detected_skills = list(
            dict.fromkeys(
                detected_skills
            )
        )

        # ====================================================
        # LOAD AI ENGINE LAZILY
        #
        # THIS is the important memory optimization.
        # ====================================================

        prediction_engine = (
            get_prediction_engine()
        )

        # ====================================================
        # IMPORT RANKING ENGINE LAZILY
        #
        # If ranking_engine imports large objects, this also
        # prevents those objects from being loaded at startup.
        # ====================================================

        from ranking_engine import (
            analyze_jobs,
            format_interview_questions,
        )

        # ====================================================
        # JOB ANALYSIS
        # ====================================================

        analysis = analyze_jobs(

            prediction_engine=prediction_engine,

            resume_text=resume_text,

            top_k=MAX_RETURNED_JOBS,
        )

        # ====================================================
        # VALIDATE RANKING RESPONSE
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

        if not isinstance(
            jobs,
            list,
        ):

            jobs = []

        if not isinstance(
            summary,
            dict,
        ):

            summary = {}

        # ====================================================
        # INTERVIEW QUESTIONS
        # ====================================================

        questions = prediction_engine.interview_questions(

            resume_text=resume_text,

            top_k=MAX_RETURNED_INTERVIEWS,
        )

        if questions is None:
            questions = []

        # ====================================================
        # FORMAT QUESTIONS
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
        # SKILL SUMMARY
        # ====================================================

        skill_summary = {

            "total_detected":
                len(detected_skills),

            "skills":
                detected_skills,

            "details":
                skill_details,
        }

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "success": True,

            "summary":
                summary,

            "skills":
                skill_summary,

            "jobs":
                jobs,

            "interview_questions":
                interview_results,
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
    # MEMORY ERROR
    # ========================================================

    except MemoryError:

        print(
            "=================================================="
        )

        print(
            "TalentMatch AI MEMORY ERROR"
        )

        print(
            "The 512 MB Render instance ran out of memory."
        )

        print(
            "=================================================="
        )

        return error_response(

            (
                "The server ran out of memory while "
                "processing this resume. Please try "
                "a smaller resume."
            ),

            status_code=503,
        )

    # ========================================================
    # UNEXPECTED ERROR
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
    # CLEANUP
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
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown_event():

    global _prediction_engine

    _prediction_engine = None

    gc.collect()
