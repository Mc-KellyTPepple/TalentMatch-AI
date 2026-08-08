"""
===============================================================
TalentMatch AI
Production FastAPI Application
===============================================================

Render Free / 512 MB diagnostic version.

IMPORTANT DESIGN:

- FastAPI startup must remain lightweight.
- predict.py is NOT imported during startup.
- ranking_engine is NOT imported during startup.
- Resume parsing is performed only during /analyze.
- Skill extraction is performed only during /analyze.
- AI prediction engine is loaded only during /analyze.
- Every important operation prints timing information.
- Full exceptions and tracebacks are printed to Render logs.
- HEAD / is supported so Render health checks do not produce 405.
===============================================================
"""

# ============================================================
# STANDARD LIBRARY
# ============================================================

import gc
import threading
import traceback
import time
import os


# ============================================================
# STARTUP TIMER
# ============================================================

APP_START_TIME = time.perf_counter()


def elapsed():
    """Return elapsed application time in seconds."""
    return round(
        time.perf_counter() - APP_START_TIME,
        3,
    )


def log(message, *args):
    """
    Timestamped diagnostic logger.

    Render logs will show exactly where the application is.
    """

    timestamp = elapsed()

    if args:
        print(
            f"[TalentMatch {timestamp:>8.3f}s] "
            + message,
            *args,
            flush=True,
        )
    else:
        print(
            f"[TalentMatch {timestamp:>8.3f}s] "
            + message,
            flush=True,
        )


log("==================================================")
log("TalentMatch AI application module loading")
log("Python PID: %s", os.getpid())
log("==================================================")


# ============================================================
# FASTAPI
# ============================================================

log("Importing FastAPI...")

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

log("FastAPI imported successfully.")


# ============================================================
# STATIC / TEMPLATES
# ============================================================

log("Importing Jinja2Templates and StaticFiles...")

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

log("Template/static imports complete.")


# ============================================================
# CONFIG
# ============================================================

log("Importing config...")

try:

    from config import (
        MAX_UPLOAD_SIZE,
        MAX_RETURNED_JOBS,
        MAX_RETURNED_INTERVIEWS,
    )

    log(
        "Config imported successfully."
    )

    log(
        "MAX_UPLOAD_SIZE=%s",
        MAX_UPLOAD_SIZE,
    )

    log(
        "MAX_RETURNED_JOBS=%s",
        MAX_RETURNED_JOBS,
    )

    log(
        "MAX_RETURNED_INTERVIEWS=%s",
        MAX_RETURNED_INTERVIEWS,
    )

except Exception as exc:

    log(
        "CONFIG IMPORT FAILED"
    )

    log(
        "Exception type: %s",
        type(exc).__name__,
    )

    log(
        "Exception: %r",
        exc,
    )

    log(
        "Traceback:\n%s",
        traceback.format_exc(),
    )

    raise


# ============================================================
# APPLICATION
# ============================================================

log("Creating FastAPI application...")

app = FastAPI(
    title="TalentMatch AI",
    description=(
        "AI-powered resume analysis, job matching, "
        "skill extraction and interview preparation platform."
    ),
    version="1.0.0",
)

log("FastAPI application created.")


# ============================================================
# STATIC FILES
# ============================================================

log("Mounting static directory...")

try:

    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static",
    )

    log(
        "Static directory mounted successfully."
    )

except Exception as exc:

    log(
        "STATIC DIRECTORY ERROR"
    )

    log(
        "Exception type: %s",
        type(exc).__name__,
    )

    log(
        "Exception: %r",
        exc,
    )

    log(
        "Traceback:\n%s",
        traceback.format_exc(),
    )

    raise


# ============================================================
# TEMPLATES
# ============================================================

log("Creating Jinja2 template engine...")

try:

    templates = Jinja2Templates(
        directory="templates"
    )

    log(
        "Template engine initialized successfully."
    )

except Exception as exc:

    log(
        "TEMPLATE ENGINE ERROR"
    )

    log(
        "Exception type: %s",
        type(exc).__name__,
    )

    log(
        "Exception: %r",
        exc,
    )

    log(
        "Traceback:\n%s",
        traceback.format_exc(),
    )

    raise


# ============================================================
# LAZY AI ENGINE
# ============================================================

_prediction_engine = None

_prediction_engine_lock = threading.Lock()


# ============================================================
# LAZY MODULE REFERENCES
# ============================================================

_resume_parser_module = None
_skill_extractor_module = None
_ranking_engine_module = None


# ============================================================
# LAZY RESUME PARSER
# ============================================================

def get_resume_parser():

    global _resume_parser_module

    if _resume_parser_module is not None:

        return _resume_parser_module

    log(
        "Lazy-loading resume_parser..."
    )

    start = time.perf_counter()

    try:

        import resume_parser

        _resume_parser_module = resume_parser

        log(
            "resume_parser loaded in %.3f seconds.",
            time.perf_counter() - start,
        )

        return _resume_parser_module

    except Exception as exc:

        log(
            "RESUME_PARSER IMPORT FAILED"
        )

        log(
            "Exception type: %s",
            type(exc).__name__,
        )

        log(
            "Exception: %r",
            exc,
        )

        log(
            "Traceback:\n%s",
            traceback.format_exc(),
        )

        raise


# ============================================================
# LAZY SKILL EXTRACTOR
# ============================================================

def get_skill_extractor():

    global _skill_extractor_module

    if _skill_extractor_module is not None:

        return _skill_extractor_module

    log(
        "Lazy-loading skill_extractor..."
    )

    start = time.perf_counter()

    try:

        import skill_extractor

        _skill_extractor_module = skill_extractor

        log(
            "skill_extractor loaded in %.3f seconds.",
            time.perf_counter() - start,
        )

        return _skill_extractor_module

    except Exception as exc:

        log(
            "SKILL_EXTRACTOR IMPORT FAILED"
        )

        log(
            "Exception type: %s",
            type(exc).__name__,
        )

        log(
            "Exception: %r",
            exc,
        )

        log(
            "Traceback:\n%s",
            traceback.format_exc(),
        )

        raise


# ============================================================
# LAZY RANKING ENGINE
# ============================================================

def get_ranking_engine():

    global _ranking_engine_module

    if _ranking_engine_module is not None:

        return _ranking_engine_module

    log(
        "Lazy-loading ranking_engine..."
    )

    start = time.perf_counter()

    try:

        import ranking_engine

        _ranking_engine_module = ranking_engine

        log(
            "ranking_engine loaded in %.3f seconds.",
            time.perf_counter() - start,
        )

        return _ranking_engine_module

    except Exception as exc:

        log(
            "RANKING_ENGINE IMPORT FAILED"
        )

        log(
            "Exception type: %s",
            type(exc).__name__,
        )

        log(
            "Exception: %r",
            exc,
        )

        log(
            "Traceback:\n%s",
            traceback.format_exc(),
        )

        raise


# ============================================================
# LAZY PREDICTION ENGINE
# ============================================================

def get_prediction_engine():

    global _prediction_engine

    if _prediction_engine is not None:

        log(
            "Prediction engine already initialized."
        )

        return _prediction_engine

    log(
        "=================================================="
    )

    log(
        "LAZY AI ENGINE INITIALIZATION STARTED"
    )

    log(
        "=================================================="
    )

    start = time.perf_counter()

    with _prediction_engine_lock:

        if _prediction_engine is not None:

            return _prediction_engine

        try:

            log(
                "Importing predict.py..."
            )

            import_start = time.perf_counter()

            from predict import engine

            log(
                "predict.py imported in %.3f seconds.",
                time.perf_counter() - import_start,
            )

            if engine is None:

                raise RuntimeError(
                    "predict.py returned a null engine."
                )

            _prediction_engine = engine

            log(
                "Prediction engine object created."
            )

            log(
                "Prediction engine type: %s",
                type(engine).__name__,
            )

            log(
                "Prediction engine initialization phase "
                "completed in %.3f seconds.",
                time.perf_counter() - start,
            )

            log(
                "=================================================="
            )

            return _prediction_engine

        except MemoryError:

            _prediction_engine = None

            log(
                "=================================================="
            )

            log(
                "PREDICTION ENGINE MEMORY ERROR"
            )

            log(
                "Render memory limit was exceeded."
            )

            log(
                "=================================================="
            )

            gc.collect()

            raise

        except Exception as exc:

            _prediction_engine = None

            log(
                "=================================================="
            )

            log(
                "PREDICTION ENGINE INITIALIZATION FAILED"
            )

            log(
                "Exception type: %s",
                type(exc).__name__,
            )

            log(
                "Exception: %r",
                exc,
            )

            log(
                "Full traceback:"
            )

            log(
                "%s",
                traceback.format_exc(),
            )

            log(
                "=================================================="
            )

            gc.collect()

            raise


# ============================================================
# PREDICTION ENGINE STATUS
# ============================================================

def prediction_engine_status():

    engine = _prediction_engine

    if engine is None:

        return "not_loaded"

    try:

        if hasattr(
            engine,
            "is_loaded",
        ):

            return (
                "loaded"
                if bool(
                    engine.is_loaded()
                )
                else "not_loaded"
            )

        loaded_state = getattr(
            engine,
            "_loaded",
            False,
        )

        if not loaded_state:

            loaded_state = getattr(
                engine,
                "_model_loaded",
                False,
            )

        return (
            "loaded"
            if bool(loaded_state)
            else "not_loaded"
        )

    except Exception as exc:

        log(
            "Prediction engine status check failed: %r",
            exc,
        )

        return "unavailable"


# ============================================================
# ERROR HELPERS
# ============================================================

def readable_error(error):

    if error is None:

        return (
            "An unknown error occurred."
        )

    if isinstance(
        error,
        str,
    ):

        return error

    if isinstance(
        error,
        dict,
    ):

        for key in (
            "error",
            "message",
            "detail",
            "msg",
        ):

            value = error.get(key)

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


def error_response(
    message,
    status_code=500,
):

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
# ROOT / HEAD
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
):

    log(
        "GET / received."
    )

    start = time.perf_counter()

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )

    log(
        "GET / completed in %.3f seconds.",
        time.perf_counter() - start,
    )

    return response


@app.head("/")
async def home_head():

    log(
        "HEAD / received."
    )

    return JSONResponse(
        content=None,
        status_code=200,
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_class=JSONResponse,
)
def health_check():

    log(
        "GET /health received."
    )

    return {

        "status":
            "healthy",

        "service":
            "TalentMatch AI",

        "version":
            "1.0.0",

        "uptime_seconds":
            elapsed(),

        "memory_strategy":
            "lazy_model_loading",

        "prediction_engine":
            prediction_engine_status(),
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get(
    "/api",
    response_class=JSONResponse,
)
def api_info():

    log(
        "GET /api received."
    )

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

        "docx_engine":
            "python-docx",

        "pymupdf_required":
            False,

        "prediction_engine":
            prediction_engine_status(),

        "deployment": {

            "platform":
                "Render",

            "mode":
                "CPU inference",

            "memory_target":
                "512 MB",

            "model_loading":
                "lazy",
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

        log(
            "Checking parser status..."
        )

        parser_module = (
            get_resume_parser()
        )

        status = (
            parser_module.parser_status()
        )

        log(
            "Parser status: %s",
            status,
        )

        return status

    except Exception as exc:

        log(
            "PARSER STATUS ERROR"
        )

        log(
            "%s",
            traceback.format_exc(),
        )

        return error_response(
            str(exc),
            500,
        )


# ============================================================
# SKILL STATUS
# ============================================================

@app.get(
    "/api/skills/status",
    response_class=JSONResponse,
)
def skills_status():

    try:

        log(
            "Checking skill engine status..."
        )

        skill_module = (
            get_skill_extractor()
        )

        status = (
            skill_module.skill_engine_status()
        )

        log(
            "Skill engine status: %s",
            status,
        )

        return status

    except Exception as exc:

        log(
            "SKILL ENGINE STATUS ERROR"
        )

        log(
            "%s",
            traceback.format_exc(),
        )

        return error_response(
            str(exc),
            500,
        )


# ============================================================
# READINESS
# ============================================================

@app.get(
    "/ready",
    response_class=JSONResponse,
)
def readiness_check():

    log(
        "GET /ready received."
    )

    try:

        # ----------------------------------------------------
        # Do NOT load prediction engine.
        # ----------------------------------------------------

        parser_module = (
            get_resume_parser()
        )

        skill_module = (
            get_skill_extractor()
        )

        parser_status_value = (
            parser_module.parser_status()
        )

        skill_status_value = (
            skill_module.skill_engine_status()
        )

        log(
            "Parser readiness: %s",
            parser_status_value,
        )

        log(
            "Skill readiness: %s",
            skill_status_value,
        )

        parser_ready = (

            isinstance(
                parser_status_value,
                dict,
            )

            and

            parser_status_value.get(
                "status"
            )
            == "ready"
        )

        if not isinstance(
            skill_status_value,
            dict,
        ):

            skill_status_value = {}

        skills_file_exists = bool(
            skill_status_value.get(
                "skills_file_exists",
                False,
            )
        )

        skill_ready = (

            skills_file_exists

            or

            skill_status_value.get(
                "status"
            )
            == "ready"
        )

        application_ready = (
            parser_ready
            and
            skill_ready
        )

        return {

            "status":
                "ready"
                if application_ready
                else "degraded",

            "lazy_loading":
                True,

            "prediction_engine":
                prediction_engine_status(),

            "resume_parser":
                parser_status_value,

            "skill_engine":
                skill_status_value,

            "skills_file_exists":
                skills_file_exists,
        }

    except Exception as exc:

        log(
            "READINESS CHECK FAILED"
        )

        log(
            "Exception type: %s",
            type(exc).__name__,
        )

        log(
            "Exception: %r",
            exc,
        )

        log(
            "Traceback:\n%s",
            traceback.format_exc(),
        )

        return JSONResponse(

            status_code=503,

            content={

                "status":
                    "not_ready",

                "prediction_engine":
                    prediction_engine_status(),

                "error":
                    readable_error(exc),
            },
        )


# ============================================================
# ANALYZE RESUME
# ============================================================

@app.post(
    "/analyze",
)
async def analyze_resume(
    file: UploadFile = File(...),
):

    request_start = time.perf_counter()

    log(
        "=================================================="
    )

    log(
        "POST /analyze RECEIVED"
    )

    log(
        "Filename: %s",
        file.filename,
    )

    log(
        "Content type: %s",
        file.content_type,
    )

    log(
        "=================================================="
    )

    file_bytes = None
    resume_text = None
    skill_details = None
    detected_skills = None
    analysis = None
    questions = None
    interview_results = None

    try:

        # ====================================================
        # FILENAME
        # ====================================================

        if not file.filename:

            return error_response(
                "Please upload a resume.",
                400,
            )

        # ====================================================
        # EXTENSION
        # ====================================================

        filename = (
            file.filename.lower()
        )

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
                400,
            )

        log(
            "File extension validated."
        )

        # ====================================================
        # READ FILE
        # ====================================================

        log(
            "Beginning upload read..."
        )

        read_start = time.perf_counter()

        chunks = []

        total_size = 0

        chunk_size = 256 * 1024

        while True:

            chunk = await file.read(
                chunk_size
            )

            if not chunk:
                break

            total_size += len(chunk)

            if (
                total_size
                >
                MAX_UPLOAD_SIZE
            ):

                log(
                    "Upload exceeded maximum size."
                )

                return error_response(
                    (
                        "Resume exceeds the maximum "
                        "allowed file size."
                    ),
                    400,
                )

            chunks.append(chunk)

        log(
            "Upload read completed."
        )

        log(
            "Bytes received: %s",
            total_size,
        )

        log(
            "Upload read time: %.3f seconds.",
            time.perf_counter() - read_start,
        )

        if total_size <= 0:

            return error_response(
                "The uploaded resume is empty.",
                400,
            )

        # ====================================================
        # COMBINE
        # ====================================================

        combine_start = time.perf_counter()

        file_bytes = b"".join(
            chunks
        )

        del chunks

        log(
            "File bytes combined in %.3f seconds.",
            time.perf_counter() - combine_start,
        )

        # ====================================================
        # PARSER
        # ====================================================

        log(
            "Loading resume parser..."
        )

        parser_module = (
            get_resume_parser()
        )

        log(
            "Starting resume parsing..."
        )

        parse_start = time.perf_counter()

        resume_text = (
            parser_module.parse_resume(
                file_bytes=file_bytes,
                filename=file.filename,
            )
        )

        log(
            "Resume parsing completed in %.3f seconds.",
            time.perf_counter() - parse_start,
        )

        file_bytes = None

        if not resume_text:

            return error_response(
                (
                    "No readable text could be extracted "
                    "from the uploaded resume."
                ),
                400,
            )

        log(
            "Extracted resume characters: %s",
            len(resume_text),
        )

        # ====================================================
        # LIMIT TEXT
        # ====================================================

        MAX_RESUME_TEXT_CHARS = 100000

        if (
            len(resume_text)
            >
            MAX_RESUME_TEXT_CHARS
        ):

            log(
                "Resume text exceeds %s characters. "
                "Truncating.",
                MAX_RESUME_TEXT_CHARS,
            )

            resume_text = resume_text[
                :MAX_RESUME_TEXT_CHARS
            ]

        # ====================================================
        # SKILL EXTRACTION
        # ====================================================

        log(
            "Loading skill extractor..."
        )

        skill_module = (
            get_skill_extractor()
        )

        log(
            "Starting skill extraction..."
        )

        skill_start = time.perf_counter()

        skill_details = (
            skill_module.extract_skill_details(
                resume_text,
                max_skills=100,
            )
        )

        log(
            "Skill extraction completed in %.3f seconds.",
            time.perf_counter() - skill_start,
        )

        if not isinstance(
            skill_details,
            list,
        ):

            skill_details = []

        # ====================================================
        # DETECTED SKILLS
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

        detected_skills = list(
            dict.fromkeys(
                detected_skills
            )
        )

        log(
            "Detected skills: %s",
            len(detected_skills),
        )

        # ====================================================
        # PREDICTION ENGINE
        # ====================================================

        log(
            "Requesting prediction engine..."
        )

        prediction_start = (
            time.perf_counter()
        )

        prediction_engine = (
            get_prediction_engine()
        )

        log(
            "Prediction engine obtained in %.3f seconds.",
            time.perf_counter()
            -
            prediction_start,
        )

        # ====================================================
        # RANKING ENGINE
        # ====================================================

        ranking_module = (
            get_ranking_engine()
        )

        log(
            "Starting job analysis..."
        )

        jobs_start = time.perf_counter()

        analysis = (
            ranking_module.analyze_jobs(
                prediction_engine=
                    prediction_engine,

                resume_text=
                    resume_text,

                top_k=
                    MAX_RETURNED_JOBS,
            )
        )

        log(
            "Job analysis completed in %.3f seconds.",
            time.perf_counter()
            -
            jobs_start,
        )

        if not isinstance(
            analysis,
            dict,
        ):

            raise RuntimeError(
                "Job ranking engine returned "
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

        log(
            "Jobs returned: %s",
            len(jobs),
        )

        # ====================================================
        # INTERVIEW QUESTIONS
        # ====================================================

        log(
            "Generating interview questions..."
        )

        interview_start = (
            time.perf_counter()
        )

        questions = (
            prediction_engine.interview_questions(
                resume_text=
                    resume_text,

                top_k=
                    MAX_RETURNED_INTERVIEWS,
            )
        )

        log(
            "Interview retrieval completed in %.3f seconds.",
            time.perf_counter()
            -
            interview_start,
        )

        if not isinstance(
            questions,
            list,
        ):

            questions = []

        # ====================================================
        # FORMAT QUESTIONS
        # ====================================================

        interview_results = (
            ranking_module.format_interview_questions(
                questions,
                top_k=
                    MAX_RETURNED_INTERVIEWS,
            )
        )

        if not isinstance(
            interview_results,
            list,
        ):

            interview_results = []

        log(
            "Interview questions returned: %s",
            len(interview_results),
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "success":
                True,

            "summary":
                summary,

            "skills": {

                "total_detected":
                    len(detected_skills),

                "skills":
                    detected_skills,

                "details":
                    skill_details,
            },

            "jobs":
                jobs,

            "interview_questions":
                interview_results,
        }

        total_time = (
            time.perf_counter()
            -
            request_start
        )

        log(
            "=================================================="
        )

        log(
            "ANALYSIS COMPLETED SUCCESSFULLY"
        )

        log(
            "Total request time: %.3f seconds.",
            total_time,
        )

        log(
            "=================================================="
        )

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # ========================================================
    # PARSER ERROR
    # ========================================================

    except Exception as exc:

        log(
            "=================================================="
        )

        log(
            "TALENTMATCH AI ANALYSIS ERROR"
        )

        log(
            "Exception type: %s",
            type(exc).__name__,
        )

        log(
            "Exception: %r",
            exc,
        )

        log(
            "Full traceback:"
        )

        log(
            "%s",
            traceback.format_exc(),
        )

        log(
            "Total request time before failure: %.3f seconds.",
            time.perf_counter()
            -
            request_start,
        )

        log(
            "=================================================="
        )

        if isinstance(
            exc,
            MemoryError,
        ):

            return error_response(
                (
                    "The server ran out of memory "
                    "while analyzing the resume."
                ),
                503,
            )

        if isinstance(
            exc,
            ResumeParsingError
            if "ResumeParsingError"
            in globals()
            else type(None),
        ):

            return error_response(
                str(exc),
                400,
            )

        if isinstance(
            exc,
            HTTPException,
        ):

            return error_response(
                exc.detail,
                exc.status_code,
            )

        return error_response(
            (
                "Unable to analyze the resume. "
                "See server diagnostics for the "
                "exact failure."
            ),
            500,
        )

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        file_bytes = None
        resume_text = None
        skill_details = None
        detected_skills = None
        analysis = None
        questions = None
        interview_results = None

        log(
            "Request temporary objects released."
        )

        gc.collect()


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event(
    "startup"
)
async def startup_event():

    log(
        "=================================================="
    )

    log(
        "FASTAPI STARTUP EVENT"
    )

    log(
        "Prediction engine will NOT be loaded."
    )

    log(
        "Ranking engine will NOT be loaded."
    )

    log(
        "Resume parser will NOT be loaded."
    )

    log(
        "Skill extractor will NOT be loaded."
    )

    log(
        "Application startup is intentionally lightweight."
    )

    log(
        "=================================================="
    )


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event(
    "shutdown"
)
def shutdown_event():

    global _prediction_engine
    global _resume_parser_module
    global _skill_extractor_module
    global _ranking_engine_module

    log(
        "=================================================="
    )

    log(
        "TalentMatch AI shutdown started."
    )

    _prediction_engine = None
    _resume_parser_module = None
    _skill_extractor_module = None
    _ranking_engine_module = None

    gc.collect()

    log(
        "TalentMatch AI shutdown complete."
    )

    log(
        "=================================================="
    )


# ============================================================
# MODULE READY
# ============================================================

log(
    "=================================================="
)

log(
    "TalentMatch AI app.py module loaded successfully."
)

log(
    "Total module load time: %.3f seconds.",
    elapsed(),
)

log(
    "Prediction engine status: %s",
    prediction_engine_status(),
)

log(
    "=================================================="
)
