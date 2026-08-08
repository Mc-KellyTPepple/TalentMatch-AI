"""
TalentMatch AI
Production FastAPI Application

Optimized for:

- Render Free
- 512 MB RAM
- CPU inference
- Lazy AI model loading
- PDF / DOCX / TXT resume parsing

Memory strategy:

- DO NOT load predict.py during application startup
- Load AI prediction engine only when /analyze is requested
- Keep /health, /ready and /api lightweight
- Read uploaded files with a hard size limit
- Do not permanently store uploaded resumes
- Limit extracted resume text
- Lazily import ranking_engine
- Release temporary objects after processing
- Avoid unnecessary model initialization
"""

# ============================================================
# STANDARD LIBRARY
# ============================================================

import gc
import traceback
import threading


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
# resume_parser.py must NOT load a large ML model.
# ============================================================

from resume_parser import (
    parse_resume,
    ResumeParsingError,
    parser_status,
)


# ============================================================
# SKILL EXTRACTOR
#
# This should only load lightweight skill information.
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

# Prevent two simultaneous requests from loading the
# prediction engine twice.
_prediction_engine_lock = threading.Lock()


def get_prediction_engine():
    """
    Load the AI prediction engine only when required.

    IMPORTANT:

    predict.py is deliberately NOT imported when FastAPI
    starts.

    This keeps /health, /ready and /api lightweight and
    prevents the SentenceTransformer model and embedding
    matrices from being loaded unnecessarily.

    Returns:
        PredictionEngine instance
    """

    global _prediction_engine

    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if _prediction_engine is not None:

        return _prediction_engine

    # --------------------------------------------------------
    # Protect initialization from concurrent requests
    # --------------------------------------------------------

    with _prediction_engine_lock:

        # Another request may have loaded it while this
        # request was waiting for the lock.

        if _prediction_engine is not None:

            return _prediction_engine

        print(
            "=================================================="
        )

        print(
            "Loading TalentMatch AI prediction engine..."
        )

        print(
            "=================================================="
        )

        try:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # This is the FIRST import of predict.py.
            #
            # Therefore the heavy prediction engine is not
            # initialized during normal FastAPI startup.
            # ------------------------------------------------

            from predict import engine

            _prediction_engine = engine

            # ------------------------------------------------
            # Verify engine state when supported
            # ------------------------------------------------

            if hasattr(
                _prediction_engine,
                "is_loaded",
            ):

                if not _prediction_engine.is_loaded():

                    raise RuntimeError(
                        "Prediction engine failed to initialize."
                    )

            print(
                "TalentMatch AI prediction engine loaded."
            )

            gc.collect()

            return _prediction_engine

        except MemoryError:

            _prediction_engine = None

            gc.collect()

            print(
                "Prediction engine could not be loaded "
                "because the server ran out of memory."
            )

            raise

        except Exception:

            _prediction_engine = None

            gc.collect()

            print(
                "Prediction engine initialization failed."
            )

            raise


# ============================================================
# PREDICTION ENGINE STATUS
# ============================================================

def prediction_engine_status():
    """
    Return the current prediction engine status.

    This function NEVER loads the prediction engine.

    Possible results:

        not_loaded
        loaded
        unavailable
    """

    engine = _prediction_engine

    if engine is None:

        return "not_loaded"

    try:

        # ----------------------------------------------------
        # Preferred method
        # ----------------------------------------------------

        if hasattr(
            engine,
            "is_loaded",
        ):

            return (
                "loaded"
                if engine.is_loaded()
                else "not_loaded"
            )

        # ----------------------------------------------------
        # Compatibility fallback
        # ----------------------------------------------------

        loaded_state = getattr(
            engine,
            "_loaded",
            False,
        )

        return (
            "loaded"
            if bool(loaded_state)
            else "not_loaded"
        )

    except Exception:

        return "unavailable"


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

            "success":
                False,

            "error":
                readable_error(
                    message
                ),
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
    """
    Lightweight Render health check.

    IMPORTANT:

    This endpoint does NOT load the prediction engine.
    """

    return {

        "status":
            "healthy",

        "service":
            "TalentMatch AI",

        "version":
            "1.0.0",

        "memory_strategy":
            "lazy_model_loading",

        "prediction_engine":
            prediction_engine_status(),
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
):

    return templates.TemplateResponse(

        "index.html",

        {
            "request":
                request,
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

        status = parser_status()

        if status is None:

            return {

                "status":
                    "unknown",

                "message":
                    "Parser returned no status.",
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
# READINESS CHECK
# ============================================================

@app.get(
    "/ready",
    response_class=JSONResponse,
)
def readiness_check():

    try:

        # ====================================================
        # PARSER STATUS
        # ====================================================

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


        # ====================================================
        # SKILL ENGINE STATUS
        # ====================================================

        current_skill_status = (
            skill_engine_status()
        )

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


        # ====================================================
        # PREDICTION ENGINE STATUS
        #
        # IMPORTANT:
        #
        # DO NOT call get_prediction_engine() here.
        #
        # /ready must remain lightweight.
        # ====================================================

        engine = _prediction_engine

        prediction_status = (
            "not_loaded"
        )

        if engine is not None:

            try:

                prediction_status = (

                    "loaded"

                    if engine.is_loaded()

                    else "not_loaded"
                )

            except AttributeError:

                # Compatibility with an older predict.py
                # that does not yet contain is_loaded().

                prediction_status = (

                    "loaded"

                    if getattr(
                        engine,
                        "_loaded",
                        False,
                    )

                    else "not_loaded"
                )

            except Exception:

                prediction_status = (
                    "unavailable"
                )


        # ====================================================
        # APPLICATION READINESS
        #
        # The AI engine does NOT need to be loaded for the
        # application itself to be ready.
        #
        # It will load when /analyze is called.
        # ====================================================

        application_ready = (

            parser_ready

            and

            skills_ready
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "status":

                "ready"

                if application_ready

                else "degraded",

            # ------------------------------------------------
            # REQUIRED FIX
            # ------------------------------------------------

            "prediction_engine":

                "loaded"

                if engine is not None
                and engine.is_loaded()

                else "not_loaded",

            # ------------------------------------------------
            # Additional explicit status
            # ------------------------------------------------

            "prediction_engine_status":
                prediction_status,

            "lazy_loading":
                True,

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


# ============================================================
# RESUME ANALYSIS
# ============================================================

@app.post(
    "/analyze",
)
async def analyze_resume(
    file: UploadFile = File(...),
):

    # ========================================================
    # TEMPORARY OBJECTS
    # ========================================================

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
        # READ FILE WITH MEMORY LIMIT
        #
        # Instead of blindly calling:
        #
        #     await file.read()
        #
        # read in chunks and stop once the maximum size
        # is exceeded.
        # ====================================================

        chunks = []

        total_size = 0

        chunk_size = 1024 * 1024

        while True:

            chunk = await file.read(
                chunk_size
            )

            if not chunk:

                break

            total_size += len(chunk)

            if total_size > MAX_UPLOAD_SIZE:

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


        # ====================================================
        # EMPTY FILE
        # ====================================================

        if total_size <= 0:

            return error_response(

                "The uploaded resume is empty.",

                status_code=400,
            )


        # ====================================================
        # COMBINE CHUNKS
        # ====================================================

        file_bytes = b"".join(
            chunks
        )

        del chunks


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

        gc.collect()


        # ====================================================
        # VERIFY EXTRACTED TEXT
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
        # LIMIT RESUME TEXT
        #
        # Prevents huge documents from consuming excessive
        # memory during embedding and TF-IDF processing.
        # ====================================================

        MAX_RESUME_TEXT_CHARS = 100000

        if len(
            resume_text
        ) > MAX_RESUME_TEXT_CHARS:

            resume_text = resume_text[
                :MAX_RESUME_TEXT_CHARS
            ]


        # ====================================================
        # EXTRACT SKILLS
        # ====================================================

        skill_details = (
            extract_skill_details(

                resume_text,

                max_skills=100,
            )
        )


        if skill_details is None:

            skill_details = []


        if not isinstance(
            skill_details,
            list,
        ):

            skill_details = []


        # ====================================================
        # BUILD DETECTED SKILLS
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
        # THIS is the only place where the heavy AI engine
        # should be loaded.
        # ====================================================

        prediction_engine = (
            get_prediction_engine()
        )


        # ====================================================
        # LAZY IMPORT RANKING ENGINE
        # ====================================================

        from ranking_engine import (

            analyze_jobs,

            format_interview_questions,
        )


        # ====================================================
        # JOB ANALYSIS
        # ====================================================

        analysis = analyze_jobs(

            prediction_engine=
                prediction_engine,

            resume_text=
                resume_text,

            top_k=
                MAX_RETURNED_JOBS,
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
        # INTERVIEW QUESTIONS
        # ====================================================

        questions = (
            prediction_engine.interview_questions(

                resume_text=
                    resume_text,

                top_k=
                    MAX_RETURNED_INTERVIEWS,
            )
        )


        if questions is None:

            questions = []


        # ====================================================
        # FORMAT QUESTIONS
        # ====================================================

        interview_results = (

            format_interview_questions(

                questions,

                top_k=
                    MAX_RETURNED_INTERVIEWS,
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
                len(
                    detected_skills
                ),

            "skills":
                detected_skills,

            "details":
                skill_details,
        }


        # ====================================================
        # FINAL RESPONSE
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
        # RETURN
        # ====================================================

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

@app.on_event(
    "shutdown"
)
def shutdown_event():

    global _prediction_engine

    _prediction_engine = None

    gc.collect()
