"""
================================================================
TalentMatch AI
Production FastAPI Application
================================================================

Optimized for:

- Render Free
- 512 MB RAM
- CPU inference
- Lazy AI model loading
- PDF / DOCX / TXT resume parsing

DIAGNOSTIC VERSION

This version adds detailed diagnostic logging around:

1. Application startup
2. /health
3. /ready
4. /analyze
5. File upload
6. Resume parsing
7. Skill extraction
8. Prediction engine loading
9. Ranking engine loading
10. Job analysis
11. Interview generation
12. Response construction
13. Exceptions
14. Cleanup
15. Memory usage where available

IMPORTANT:

The prediction engine is NOT loaded during startup.

It is loaded only when /analyze is called.

================================================================
"""

# ================================================================
# STANDARD LIBRARY
# ================================================================

import gc
import os
import sys
import time
import threading
import traceback

# ================================================================
# FASTAPI
# ================================================================

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

# ================================================================
# CONFIG
# ================================================================

from config import (
    MAX_UPLOAD_SIZE,
    MAX_RETURNED_JOBS,
    MAX_RETURNED_INTERVIEWS,
)

# ================================================================
# RESUME PARSER
# ================================================================

from resume_parser import (
    parse_resume,
    ResumeParsingError,
    parser_status,
)

# ================================================================
# SKILL EXTRACTOR
# ================================================================

from skill_extractor import (
    extract_skill_details,
    skill_engine_status,
)

# ================================================================
# APPLICATION
# ================================================================

app = FastAPI(
    title="TalentMatch AI",
    description=(
        "AI-powered resume analysis, job matching, "
        "skill extraction and interview preparation platform."
    ),
    version="1.0.0",
)

# ================================================================
# DIAGNOSTIC LOGGER
# ================================================================

START_TIME = time.time()


def log(message):
    """
    Central diagnostic logger.

    flush=True ensures messages appear immediately
    in Render logs.
    """

    print(
        f"[TalentMatch DEBUG] {message}",
        flush=True,
    )


def log_separator(title=""):
    """
    Print a highly visible diagnostic separator.
    """

    print(
        "",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    if title:
        print(
            f"[TalentMatch DEBUG] {title}",
            flush=True,
        )

    print(
        "=" * 70,
        flush=True,
    )


def log_exception(title, exc):
    """
    Print complete exception diagnostics.
    """

    log_separator(title)

    print(
        f"Exception type: {type(exc).__name__}",
        flush=True,
    )

    print(
        f"Exception repr: {repr(exc)}",
        flush=True,
    )

    print(
        f"Exception str: {str(exc)}",
        flush=True,
    )

    print(
        "Full traceback:",
        flush=True,
    )

    print(
        traceback.format_exc(),
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )


# ================================================================
# OPTIONAL MEMORY DIAGNOSTICS
# ================================================================

def memory_status():
    """
    Return lightweight process memory information.

    This does not require psutil.

    On Linux / Render, /proc/self/status is normally available.
    """

    try:

        status_path = "/proc/self/status"

        if not os.path.exists(status_path):
            return {
                "available": False,
            }

        memory = {}

        with open(
            status_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            for line in f:

                if (
                    line.startswith("VmRSS:")
                    or line.startswith("VmSize:")
                    or line.startswith("VmPeak:")
                ):

                    parts = line.split()

                    if len(parts) >= 2:

                        memory[
                            parts[0].rstrip(":")
                        ] = f"{parts[1]} {parts[2] if len(parts) > 2 else 'kB'}"

        memory["available"] = True

        return memory

    except Exception as exc:

        log(
            f"Memory diagnostic failed: {repr(exc)}"
        )

        return {
            "available": False,
            "error": str(exc),
        }


def log_memory(label):
    """
    Print current memory information.
    """

    try:

        log(
            f"MEMORY [{label}]: {memory_status()}"
        )

    except Exception as exc:

        log(
            f"Unable to log memory: {repr(exc)}"
        )


# ================================================================
# STARTUP DIAGNOSTICS
# ================================================================

log_separator(
    "TalentMatch AI application module loading"
)

log(
    f"Python version: {sys.version}"
)

log(
    f"Platform: {sys.platform}"
)

log(
    f"Working directory: {os.getcwd()}"
)

log(
    f"MAX_UPLOAD_SIZE: {MAX_UPLOAD_SIZE}"
)

log(
    f"MAX_RETURNED_JOBS: {MAX_RETURNED_JOBS}"
)

log(
    f"MAX_RETURNED_INTERVIEWS: {MAX_RETURNED_INTERVIEWS}"
)

log_memory(
    "application import"
)

# ================================================================
# STATIC FILES
# ================================================================

try:

    app.mount(
        "/static",
        StaticFiles(
            directory="static"
        ),
        name="static",
    )

    log(
        "Static files mounted successfully."
    )

except Exception as exc:

    log_exception(
        "STATIC FILE MOUNT ERROR",
        exc,
    )

    raise

# ================================================================
# TEMPLATES
# ================================================================

try:

    templates = Jinja2Templates(
        directory="templates"
    )

    log(
        "Jinja2 templates initialized successfully."
    )

except Exception as exc:

    log_exception(
        "TEMPLATE INITIALIZATION ERROR",
        exc,
    )

    raise

# ================================================================
# LAZY AI ENGINE
# ================================================================

_prediction_engine = None

_prediction_engine_lock = threading.Lock()


# ================================================================
# GET PREDICTION ENGINE
# ================================================================

def get_prediction_engine():

    global _prediction_engine

    # ------------------------------------------------------------
    # Already initialized
    # ------------------------------------------------------------

    if _prediction_engine is not None:

        log(
            "Prediction engine already exists. "
            "Reusing existing engine."
        )

        return _prediction_engine

    # ------------------------------------------------------------
    # Lock initialization
    # ------------------------------------------------------------

    log(
        "Prediction engine is not initialized."
    )

    log(
        "Waiting for prediction-engine initialization lock..."
    )

    with _prediction_engine_lock:

        log(
            "Prediction-engine initialization lock acquired."
        )

        # --------------------------------------------------------
        # Check again
        # --------------------------------------------------------

        if _prediction_engine is not None:

            log(
                "Another request initialized the prediction "
                "engine while this request was waiting."
            )

            return _prediction_engine

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------

        log_separator(
            "STARTING LAZY PREDICTION ENGINE INITIALIZATION"
        )

        log_memory(
            "before predict.py import"
        )

        start_time = time.time()

        try:

            log(
                "About to execute: from predict import engine"
            )

            log(
                "IMPORTANT: Any import-time error in predict.py "
                "will appear immediately below."
            )

            # ====================================================
            # THIS IS THE IMPORTANT IMPORT
            # ====================================================

            from predict import engine

            elapsed = (
                time.time() - start_time
            )

            log(
                f"predict.py imported successfully "
                f"in {elapsed:.2f} seconds."
            )

            log(
                f"Imported engine object type: "
                f"{type(engine).__name__}"
            )

            # ----------------------------------------------------
            # Validate
            # ----------------------------------------------------

            if engine is None:

                raise RuntimeError(
                    "predict.py returned a null "
                    "prediction engine."
                )

            _prediction_engine = engine

            log(
                "Prediction engine object stored successfully."
            )

            # ----------------------------------------------------
            # Inspect engine
            # ----------------------------------------------------

            try:

                log(
                    "Prediction engine attributes:"
                )

                log(
                    str(
                        [
                            attr
                            for attr in dir(
                                _prediction_engine
                            )
                            if not attr.startswith("__")
                        ]
                    )
                )

            except Exception as inspect_error:

                log(
                    "Unable to inspect prediction engine: "
                    f"{repr(inspect_error)}"
                )

            # ----------------------------------------------------
            # Engine status
            # ----------------------------------------------------

            try:

                if hasattr(
                    _prediction_engine,
                    "is_loaded",
                ):

                    loaded = (
                        _prediction_engine.is_loaded()
                    )

                    log(
                        f"Prediction engine is_loaded(): "
                        f"{loaded}"
                    )

                else:

                    log(
                        "Prediction engine has no is_loaded() method."
                    )

            except Exception as status_error:

                log(
                    "Could not determine prediction engine "
                    f"loaded state: {repr(status_error)}"
                )

            log_memory(
                "after predict.py import"
            )

            gc.collect()

            log(
                "Garbage collection completed after "
                "prediction-engine initialization."
            )

            log_separator(
                "PREDICTION ENGINE INITIALIZATION SUCCESS"
            )

            return _prediction_engine

        # ========================================================
        # MEMORY ERROR
        # ========================================================

        except MemoryError as exc:

            _prediction_engine = None

            log_memory(
                "MEMORY ERROR state"
            )

            log_exception(
                "PREDICTION ENGINE MEMORY ERROR",
                exc,
            )

            gc.collect()

            raise

        # ========================================================
        # ALL OTHER ERRORS
        # ========================================================

        except Exception as exc:

            _prediction_engine = None

            log_memory(
                "prediction engine initialization failure"
            )

            log_exception(
                "PREDICTION ENGINE INITIALIZATION ERROR",
                exc,
            )

            gc.collect()

            raise


# ================================================================
# PREDICTION ENGINE STATUS
# ================================================================

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
            "Prediction engine status error: "
            f"{repr(exc)}"
        )

        return "unavailable"


# ================================================================
# READABLE ERROR
# ================================================================

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


# ================================================================
# ERROR RESPONSE
# ================================================================

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


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get(
    "/health",
    response_class=JSONResponse,
)
def health_check():

    log(
        "GET /health received."
    )

    result = {

        "status": "healthy",

        "service":
            "TalentMatch AI",

        "version":
            "1.0.0",

        "memory_strategy":
            "lazy_model_loading",

        "prediction_engine":
            prediction_engine_status(),
    }

    log(
        f"/health response: {result}"
    )

    return result


# ================================================================
# HOME PAGE
# ================================================================

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

    try:

        response = templates.TemplateResponse(
            "index.html",
            {
                "request": request,
            },
        )

        log(
            "index.html rendered successfully."
        )

        return response

    except Exception as exc:

        log_exception(
            "HOME PAGE ERROR",
            exc,
        )

        raise


# ================================================================
# API INFORMATION
# ================================================================

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


# ================================================================
# PARSER STATUS
# ================================================================

@app.get(
    "/api/parser/status",
    response_class=JSONResponse,
)
def parser_status_endpoint():

    log(
        "GET /api/parser/status received."
    )

    try:

        status = parser_status()

        log(
            f"Parser status: {status}"
        )

        if status is None:

            return {

                "status":
                    "unknown",

                "message":
                    "Parser returned no status.",
            }

        return status

    except Exception as exc:

        log_exception(
            "PARSER STATUS ERROR",
            exc,
        )

        return error_response(
            str(exc),
            status_code=500,
        )


# ================================================================
# SKILL ENGINE STATUS
# ================================================================

@app.get(
    "/api/skills/status",
    response_class=JSONResponse,
)
def skills_status():

    log(
        "GET /api/skills/status received."
    )

    try:

        status = skill_engine_status()

        log(
            f"Skill engine status: {status}"
        )

        if status is None:

            return {

                "status":
                    "unknown",

                "message":
                    "Skill engine returned no status.",
            }

        return status

    except Exception as exc:

        log_exception(
            "SKILL ENGINE STATUS ERROR",
            exc,
        )

        return error_response(
            str(exc),
            status_code=500,
        )


# ================================================================
# READINESS CHECK
# ================================================================

@app.get(
    "/ready",
    response_class=JSONResponse,
)
def readiness_check():

    log(
        "GET /ready received."
    )

    try:

        # ========================================================
        # PARSER
        # ========================================================

        current_parser_status = (
            parser_status()
        )

        log(
            f"Parser readiness: "
            f"{current_parser_status}"
        )

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

        # ========================================================
        # SKILLS
        # ========================================================

        current_skill_status = (
            skill_engine_status()
        )

        log(
            f"Skill-engine readiness: "
            f"{current_skill_status}"
        )

        if not isinstance(
            current_skill_status,
            dict,
        ):

            current_skill_status = {}

        skills_file_exists = bool(

            current_skill_status.get(
                "skills_file_exists",
                False,
            )
        )

        skills_status_value = (
            current_skill_status.get(
                "status"
            )
        )

        skills_ready = (

            skills_file_exists

            or

            skills_status_value == "ready"
        )

        # ========================================================
        # PREDICTION ENGINE
        #
        # DO NOT LOAD IT HERE
        # ========================================================

        prediction_status = (
            prediction_engine_status()
        )

        application_ready = (

            parser_ready

            and

            skills_ready
        )

        result = {

            "status":
                (
                    "ready"
                    if application_ready
                    else "degraded"
                ),

            "prediction_engine":
                prediction_status,

            "prediction_engine_status":
                prediction_status,

            "lazy_loading":
                True,

            "resume_parser":
                current_parser_status,

            "skill_engine":
                current_skill_status,

            "skills_file_exists":
                skills_file_exists,
        }

        log(
            f"/ready response: {result}"
        )

        return result

    except Exception as exc:

        log_exception(
            "READINESS CHECK ERROR",
            exc,
        )

        return JSONResponse(

            status_code=503,

            content={

                "status":
                    "not_ready",

                "prediction_engine":
                    prediction_engine_status(),

                "resume_parser":
                    "error",

                "skill_engine":
                    "error",

                "error":
                    readable_error(
                        exc
                    ),
            },
        )


# ================================================================
# RESUME ANALYSIS
# ================================================================

@app.post(
    "/analyze",
)
async def analyze_resume(
    file: UploadFile = File(...),
):

    request_start = time.time()

    log_separator(
        "NEW /analyze REQUEST"
    )

    log_memory(
        "start of /analyze"
    )

    file_bytes = None
    resume_text = None
    skill_details = None
    detected_skills = None
    analysis = None
    questions = None
    interview_results = None

    try:

        # ========================================================
        # REQUEST INFORMATION
        # ========================================================

        log(
            f"Uploaded filename: {file.filename}"
        )

        log(
            f"Uploaded content type: "
            f"{file.content_type}"
        )

        log(
            f"Configured maximum upload size: "
            f"{MAX_UPLOAD_SIZE} bytes"
        )

        # ========================================================
        # VALIDATE FILENAME
        # ========================================================

        if not file.filename:

            log(
                "ERROR: No filename supplied."
            )

            return error_response(
                "Please upload a resume.",
                status_code=400,
            )

        # ========================================================
        # VALIDATE EXTENSION
        # ========================================================

        filename = (
            file.filename.lower()
        )

        supported_extensions = (
            ".pdf",
            ".docx",
            ".txt",
        )

        log(
            f"Detected filename: {filename}"
        )

        if not filename.endswith(
            supported_extensions
        ):

            log(
                "ERROR: Unsupported file extension."
            )

            return error_response(

                (
                    "Unsupported resume format. "
                    "Please upload a PDF, DOCX or TXT file."
                ),

                status_code=400,
            )

        log(
            "File extension validation passed."
        )

        # ========================================================
        # READ FILE
        # ========================================================

        log_separator(
            "READING UPLOADED FILE"
        )

        chunks = []

        total_size = 0

        chunk_size = (
            1024 * 1024
        )

        read_start = time.time()

        while True:

            chunk = await file.read(
                chunk_size
            )

            if not chunk:

                break

            total_size += len(
                chunk
            )

            log(
                f"Received chunk: "
                f"{len(chunk)} bytes; "
                f"total: {total_size} bytes"
            )

            if (
                total_size
                >
                MAX_UPLOAD_SIZE
            ):

                log(
                    "ERROR: Upload exceeded "
                    "MAX_UPLOAD_SIZE."
                )

                return error_response(

                    (
                        "Resume exceeds the maximum "
                        "allowed file size."
                    ),

                    status_code=400,
                )

            chunks.append(
                chunk
            )

        read_elapsed = (
            time.time()
            - read_start
        )

        log(
            f"File read completed in "
            f"{read_elapsed:.3f} seconds."
        )

        log(
            f"Total uploaded bytes: "
            f"{total_size}"
        )

        # ========================================================
        # EMPTY FILE
        # ========================================================

        if total_size <= 0:

            log(
                "ERROR: Uploaded file is empty."
            )

            return error_response(
                "The uploaded resume is empty.",
                status_code=400,
            )

        # ========================================================
        # COMBINE CHUNKS
        # ========================================================

        log(
            "Combining uploaded chunks..."
        )

        file_bytes = b"".join(
            chunks
        )

        del chunks

        log(
            f"Combined file size: "
            f"{len(file_bytes)} bytes"
        )

        log_memory(
            "after file upload"
        )

        # ========================================================
        # PARSE RESUME
        # ========================================================

        log_separator(
            "STARTING RESUME PARSING"
        )

        parse_start = time.time()

        log(
            f"Calling parse_resume() for "
            f"{file.filename}"
        )

        resume_text = parse_resume(

            file_bytes=file_bytes,

            filename=file.filename,
        )

        parse_elapsed = (
            time.time()
            - parse_start
        )

        log(
            f"parse_resume() completed in "
            f"{parse_elapsed:.3f} seconds."
        )

        # ========================================================
        # RELEASE RAW FILE
        # ========================================================

        file_bytes = None

        gc.collect()

        log(
            "Raw uploaded file bytes released."
        )

        log_memory(
            "after resume parsing"
        )

        # ========================================================
        # VERIFY EXTRACTED TEXT
        # ========================================================

        if not resume_text:

            log(
                "ERROR: Parser returned empty resume text."
            )

            return error_response(

                (
                    "No readable text could be extracted "
                    "from the uploaded resume."
                ),

                status_code=400,
            )

        log(
            f"Extracted resume text length: "
            f"{len(resume_text)} characters."
        )

        log(
            f"First 500 characters of extracted text: "
            f"{resume_text[:500]!r}"
        )

        # ========================================================
        # LIMIT RESUME TEXT
        # ========================================================

        MAX_RESUME_TEXT_CHARS = 100000

        if (
            len(resume_text)
            >
            MAX_RESUME_TEXT_CHARS
        ):

            log(
                f"Resume text exceeds "
                f"{MAX_RESUME_TEXT_CHARS} characters."
            )

            resume_text = resume_text[
                :MAX_RESUME_TEXT_CHARS
            ]

            log(
                "Resume text truncated."
            )

        # ========================================================
        # SKILL EXTRACTION
        # ========================================================

        log_separator(
            "STARTING SKILL EXTRACTION"
        )

        skill_start = time.time()

        log(
            "Calling extract_skill_details()..."
        )

        skill_details = (
            extract_skill_details(

                resume_text,

                max_skills=100,
            )
        )

        skill_elapsed = (
            time.time()
            - skill_start
        )

        log(
            f"Skill extraction completed in "
            f"{skill_elapsed:.3f} seconds."
        )

        log(
            f"Raw skill_details type: "
            f"{type(skill_details).__name__}"
        )

        log(
            f"Raw skill_details count: "
            f"{len(skill_details) if isinstance(skill_details, list) else 'N/A'}"
        )

        if skill_details is None:

            skill_details = []

        if not isinstance(
            skill_details,
            list,
        ):

            log(
                "WARNING: skill_details was not a list. "
                "Replacing with []."
            )

            skill_details = []

        # ========================================================
        # BUILD DETECTED SKILLS
        # ========================================================

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
            f"Detected unique skills: "
            f"{len(detected_skills)}"
        )

        log(
            f"Detected skills: "
            f"{detected_skills}"
        )

        log_memory(
            "after skill extraction"
        )

        # ========================================================
        # LOAD AI ENGINE
        # ========================================================

        log_separator(
            "REQUESTING LAZY AI ENGINE"
        )

        log(
            "Resume parsing and skill extraction succeeded."
        )

        log(
            "The next operation may load the ML model."
        )

        log_memory(
            "before prediction engine"
        )

        engine_start = time.time()

        prediction_engine = (
            get_prediction_engine()
        )

        engine_elapsed = (
            time.time()
            - engine_start
        )

        log(
            f"get_prediction_engine() completed in "
            f"{engine_elapsed:.3f} seconds."
        )

        log(
            f"Prediction engine type: "
            f"{type(prediction_engine).__name__}"
        )

        log(
            f"Prediction engine status: "
            f"{prediction_engine_status()}"
        )

        log_memory(
            "after prediction engine"
        )

        # ========================================================
        # RANKING ENGINE IMPORT
        # ========================================================

        log_separator(
            "LOADING RANKING ENGINE"
        )

        ranking_import_start = time.time()

        log(
            "About to execute:"
        )

        log(
            "from ranking_engine import "
            "analyze_jobs, format_interview_questions"
        )

        from ranking_engine import (
            analyze_jobs,
            format_interview_questions,
        )

        ranking_import_elapsed = (
            time.time()
            - ranking_import_start
        )

        log(
            f"ranking_engine imported successfully "
            f"in {ranking_import_elapsed:.3f} seconds."
        )

        # ========================================================
        # JOB ANALYSIS
        # ========================================================

        log_separator(
            "STARTING JOB ANALYSIS"
        )

        log(
            f"MAX_RETURNED_JOBS = "
            f"{MAX_RETURNED_JOBS}"
        )

        job_start = time.time()

        log(
            "Calling analyze_jobs()..."
        )

        analysis = analyze_jobs(

            prediction_engine=
                prediction_engine,

            resume_text=
                resume_text,

            top_k=
                MAX_RETURNED_JOBS,
        )

        job_elapsed = (
            time.time()
            - job_start
        )

        log(
            f"analyze_jobs() completed in "
            f"{job_elapsed:.3f} seconds."
        )

        log(
            f"Analysis type: "
            f"{type(analysis).__name__}"
        )

        # ========================================================
        # VALIDATE ANALYSIS
        # ========================================================

        if not isinstance(
            analysis,
            dict,
        ):

            log(
                "ERROR: analyze_jobs() did not return a dict."
            )

            raise RuntimeError(

                "The job ranking engine returned "
                "an invalid response."
            )

        log(
            f"Analysis keys: "
            f"{list(analysis.keys())}"
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

            log(
                "WARNING: jobs is not a list. "
                "Replacing with []."
            )

            jobs = []

        if summary is None:
            summary = {}

        if not isinstance(
            summary,
            dict,
        ):

            log(
                "WARNING: summary is not a dict. "
                "Replacing with {}."
            )

            summary = {}

        log(
            f"Number of jobs returned: "
            f"{len(jobs)}"
        )

        log(
            f"Summary returned: "
            f"{summary}"
        )

        # ========================================================
        # INTERVIEW QUESTIONS
        # ========================================================

        log_separator(
            "STARTING INTERVIEW QUESTION GENERATION"
        )

        log(
            f"MAX_RETURNED_INTERVIEWS = "
            f"{MAX_RETURNED_INTERVIEWS}"
        )

        interview_start = time.time()

        log(
            "Calling prediction_engine.interview_questions()..."
        )

        questions = (
            prediction_engine.interview_questions(

                resume_text=
                    resume_text,

                top_k=
                    MAX_RETURNED_INTERVIEWS,
            )
        )

        interview_elapsed = (
            time.time()
            - interview_start
        )

        log(
            f"interview_questions() completed in "
            f"{interview_elapsed:.3f} seconds."
        )

        log(
            f"Questions type: "
            f"{type(questions).__name__}"
        )

        if questions is None:

            questions = []

        log(
            f"Raw questions count: "
            f"{len(questions) if isinstance(questions, list) else 'N/A'}"
        )

        # ========================================================
        # FORMAT QUESTIONS
        # ========================================================

        log(
            "Formatting interview questions..."
        )

        format_start = time.time()

        interview_results = (
            format_interview_questions(

                questions,

                top_k=
                    MAX_RETURNED_INTERVIEWS,
            )
        )

        format_elapsed = (
            time.time()
            - format_start
        )

        log(
            f"format_interview_questions() completed in "
            f"{format_elapsed:.3f} seconds."
        )

        if interview_results is None:

            interview_results = []

        if not isinstance(
            interview_results,
            list,
        ):

            log(
                "WARNING: interview_results was not a list. "
                "Replacing with []."
            )

            interview_results = []

        log(
            f"Final interview results count: "
            f"{len(interview_results)}"
        )

        # ========================================================
        # SKILL SUMMARY
        # ========================================================

        skill_summary = {

            "total_detected":
                len(detected_skills),

            "skills":
                detected_skills,

            "details":
                skill_details,
        }

        # ========================================================
        # FINAL RESPONSE
        # ========================================================

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

        # ========================================================
        # FINAL DIAGNOSTICS
        # ========================================================

        total_elapsed = (
            time.time()
            - request_start
        )

        log_separator(
            "ANALYSIS SUCCESS"
        )

        log(
            f"Jobs returned: {len(jobs)}"
        )

        log(
            f"Interviews returned: "
            f"{len(interview_results)}"
        )

        log(
            f"Skills detected: "
            f"{len(detected_skills)}"
        )

        log(
            f"Total request time: "
            f"{total_elapsed:.3f} seconds"
        )

        log_memory(
            "before sending response"
        )

        log(
            "Returning HTTP 200 response."
        )

        return JSONResponse(

            status_code=200,

            content=response,
        )

    # ============================================================
    # RESUME PARSING ERROR
    # ============================================================

    except ResumeParsingError as exc:

        log_exception(
            "RESUME PARSING ERROR",
            exc,
        )

        return error_response(
            str(exc),
            status_code=400,
        )

    # ============================================================
    # HTTP ERROR
    # ============================================================

    except HTTPException as exc:

        log_exception(
            "HTTP ERROR",
            exc,
        )

        return error_response(
            exc.detail,
            status_code=exc.status_code,
        )

    # ============================================================
    # MEMORY ERROR
    # ============================================================

    except MemoryError as exc:

        log_exception(
            "TALENTMATCH AI MEMORY ERROR",
            exc,
        )

        log_memory(
            "after MemoryError"
        )

        return error_response(

            (
                "The server ran out of memory while "
                "processing this resume. Please try "
                "a smaller resume."
            ),

            status_code=503,
        )

    # ============================================================
    # UNEXPECTED ERROR
    # ============================================================

    except Exception as exc:

        total_elapsed = (
            time.time()
            - request_start
        )

        log_exception(
            "TALENTMATCH AI ANALYSIS ERROR",
            exc,
        )

        log(
            f"Request failed after "
            f"{total_elapsed:.3f} seconds."
        )

        log(
            f"Current prediction engine status: "
            f"{prediction_engine_status()}"
        )

        log_memory(
            "analysis exception"
        )

        # ========================================================
        # IMPORTANT DEBUG RESPONSE
        # ========================================================
        #
        # This intentionally exposes the actual exception to the
        # frontend while debugging.
        #
        # Once the deployment is working, change:
        #
        #     DEBUG_MODE = True
        #
        # to:
        #
        #     DEBUG_MODE = False
        #
        # ========================================================

        DEBUG_MODE = True

        if DEBUG_MODE:

            return JSONResponse(

                status_code=500,

                content={

                    "success":
                        False,

                    "error":
                        "TalentMatch AI analysis failed.",

                    "exception_type":
                        type(exc).__name__,

                    "exception":
                        str(exc),

                    "diagnostic":
                        (
                            "Detailed traceback has been "
                            "written to the Render service log."
                        ),
                },
            )

        return error_response(

            (
                "Unable to analyze the resume. "
                "The server encountered an internal error. "
                "Please try again."
            ),

            status_code=500,
        )

    # ============================================================
    # CLEANUP
    # ============================================================

    finally:

        log(
            "Starting /analyze request cleanup..."
        )

        file_bytes = None

        resume_text = None

        skill_details = None

        detected_skills = None

        analysis = None

        questions = None

        interview_results = None

        try:

            await file.close()

            log(
                "Uploaded file handle closed."
            )

        except Exception as close_error:

            log(
                "Unable to close upload file: "
                f"{repr(close_error)}"
            )

        gc.collect()

        total_elapsed = (
            time.time()
            - request_start
        )

        log(
            f"Cleanup complete. "
            f"Total request lifetime: "
            f"{total_elapsed:.3f} seconds."
        )

        log_memory(
            "after /analyze cleanup"
        )

        log_separator(
            "END /analyze REQUEST"
        )


# ================================================================
# SHUTDOWN
# ================================================================

@app.on_event(
    "shutdown"
)
def shutdown_event():

    global _prediction_engine

    log_separator(
        "TALENTMATCH AI SHUTDOWN"
    )

    log(
        "Clearing prediction engine reference..."
    )

    _prediction_engine = None

    gc.collect()

    log_memory(
        "after shutdown cleanup"
    )

    log(
        "TalentMatch AI application shutdown complete."
    )
