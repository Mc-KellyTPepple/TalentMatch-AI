"""
TalentMatch AI
FastAPI Application

Responsibilities
----------------
- Receive CV uploads
- Extract PDF/DOCX/TXT text
- Run hybrid job matching
- Return ranked jobs
- Retrieve interview questions
- Provide health/status endpoints

Deployment
----------
Render Free
CPU inference
512 MB target
"""


# ===============================================================
# ENVIRONMENT
# ===============================================================

import os

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "HF_HOME",
    "/tmp/huggingface",
)


# ===============================================================
# STANDARD LIBRARY
# ===============================================================

import io
import time
import traceback
from pathlib import Path


# ===============================================================
# FASTAPI
# ===============================================================

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    HTMLResponse,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from fastapi.templating import (
    Jinja2Templates,
)

from starlette.requests import Request


# ===============================================================
# PROJECT
# ===============================================================

from predict import engine

from ranking_engine import (
    analyze_jobs,
    format_interview_questions,
)


# ===============================================================
# APPLICATION
# ===============================================================

app = FastAPI(
    title="TalentMatch AI",
    description=(
        "AI-powered resume and job matching "
        "using semantic and TF-IDF ranking."
    ),
    version="1.0.0",
)


# ===============================================================
# CORS
# ===============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================================================
# DIRECTORIES
# ===============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

STATIC_DIR = (
    BASE_DIR / "static"
)

TEMPLATE_DIR = (
    BASE_DIR / "templates"
)


# ===============================================================
# STATIC FILES
# ===============================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(
                STATIC_DIR
            )
        ),
        name="static",
    )


# ===============================================================
# TEMPLATES
# ===============================================================

templates = None

if TEMPLATE_DIR.exists():

    templates = Jinja2Templates(
        directory=str(
            TEMPLATE_DIR
        )
    )


# ===============================================================
# LOGGER
# ===============================================================

_MODULE_START = time.perf_counter()


def log(message, *args):

    elapsed = (
        time.perf_counter()
        - _MODULE_START
    )

    if args:

        try:

            message = message.format(
                *args
            )

        except Exception:

            message = (
                str(message)
                + " "
                + " ".join(
                    str(x)
                    for x in args
                )
            )

    print(
        f"[TalentMatch {elapsed:8.3f}s] {message}",
        flush=True,
    )


log(
    "app.py imported successfully."
)


# ===============================================================
# TEXT EXTRACTION
# ===============================================================

def extract_text_from_upload(
    filename,
    data,
):

    filename = str(
        filename or ""
    ).lower()


    # -----------------------------------------------------------
    # TXT
    # -----------------------------------------------------------

    if filename.endswith(".txt"):

        return data.decode(
            "utf-8",
            errors="ignore",
        )


    # -----------------------------------------------------------
    # PDF
    # -----------------------------------------------------------

    if filename.endswith(".pdf"):

        try:

            from pypdf import PdfReader

            reader = PdfReader(
                io.BytesIO(data)
            )

            pages = []

            for page in reader.pages:

                try:

                    text = (
                        page.extract_text()
                        or ""
                    )

                    if text.strip():

                        pages.append(
                            text
                        )

                except Exception:

                    continue

            return "\n".join(pages)

        except Exception as exc:

            raise ValueError(
                "Could not read PDF file: "
                f"{exc}"
            )


    # -----------------------------------------------------------
    # DOCX
    # -----------------------------------------------------------

    if filename.endswith(".docx"):

        try:

            from docx import Document

            document = Document(
                io.BytesIO(data)
            )

            paragraphs = []

            for paragraph in document.paragraphs:

                text = (
                    paragraph.text
                    or ""
                )

                if text.strip():

                    paragraphs.append(
                        text
                    )

            return "\n".join(
                paragraphs
            )

        except Exception as exc:

            raise ValueError(
                "Could not read DOCX file: "
                f"{exc}"
            )


    raise ValueError(
        "Unsupported CV format. "
        "Please upload PDF, DOCX or TXT."
    )


# ===============================================================
# CLEAN RESUME
# ===============================================================

def clean_resume_text(text):

    if not text:
        return ""

    lines = [
        line.strip()
        for line in str(text).splitlines()
        if line.strip()
    ]

    text = "\n".join(lines)

    # Protect the service from unnecessarily large requests.
    text = text[:30000]

    return text.strip()


# ===============================================================
# ROOT
# ===============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
):

    if templates is not None:

        try:

            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request
                },
            )

        except Exception as exc:

            log(
                "Template rendering failed: {}",
                repr(exc),
            )

    return HTMLResponse(
        """
        <html>
        <head>
            <title>TalentMatch AI</title>
        </head>
        <body>
            <h1>TalentMatch AI</h1>
            <p>API is running.</p>
        </body>
        </html>
        """
    )


# ===============================================================
# HEALTH
# ===============================================================

@app.get("/health")
async def health():

    return {

        "status":
            "healthy",

        "service":
            "TalentMatch AI",

        "prediction_engine":
            (
                "loaded"
                if engine.is_loaded()
                else "not_loaded"
            ),

        "loading":
            bool(
                engine.load_status().get(
                    "loading",
                    False,
                )
            ),
    }


# ===============================================================
# STATUS
# ===============================================================

@app.get("/status")
async def status():

    return {

        "service":
            "TalentMatch AI",

        "prediction_engine":
            engine.load_status(),
    }


# ===============================================================
# CV ANALYSIS
# ===============================================================

@app.post("/analyze")
async def analyze_cv(
    file: UploadFile = File(...),
    top_k: int = Form(10),
):

    start = time.perf_counter()

    try:

        top_k = max(
            1,
            min(
                int(top_k),
                20,
            ),
        )

    except (
        TypeError,
        ValueError,
    ):

        top_k = 10


    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No CV filename supplied.",
        )


    try:

        data = await file.read()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded CV.",
        )


    if not data:

        raise HTTPException(
            status_code=400,
            detail="Uploaded CV is empty.",
        )


    try:

        resume_text = (
            extract_text_from_upload(
                file.filename,
                data,
            )
        )

        resume_text = clean_resume_text(
            resume_text
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


    if not resume_text:

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found "
                "in the uploaded CV."
            ),
        )


    try:

        result = analyze_jobs(
            prediction_engine=engine,
            resume_text=resume_text,
            top_k=top_k,
        )

    except Exception as exc:

        log(
            "CV analysis failed: {}",
            repr(exc),
        )

        print(
            traceback.format_exc(),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "TalentMatch AI could not "
                "complete CV analysis."
            ),
        )


    elapsed = (
        time.perf_counter()
        - start
    )


    return {

        "success":
            True,

        "filename":
            file.filename,

        "resume_characters":
            len(resume_text),

        "processing_time_seconds":
            round(
                elapsed,
                3,
            ),

        "summary":
            result.get(
                "summary",
                {},
            ),

        "jobs":
            result.get(
                "jobs",
                [],
            ),
    }


# ===============================================================
# INTERVIEW
# ===============================================================

@app.post("/interview")
async def interview(
    file: UploadFile = File(...),
    top_k: int = Form(5),
):

    start = time.perf_counter()

    try:

        top_k = max(
            1,
            min(
                int(top_k),
                10,
            ),
        )

    except (
        TypeError,
        ValueError,
    ):

        top_k = 5


    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No CV supplied.",
        )


    data = await file.read()


    if not data:

        raise HTTPException(
            status_code=400,
            detail="Uploaded CV is empty.",
        )


    try:

        resume_text = (
            extract_text_from_upload(
                file.filename,
                data,
            )
        )

        resume_text = clean_resume_text(
            resume_text
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


    if not resume_text:

        raise HTTPException(
            status_code=400,
            detail="No readable text found in CV.",
        )


    try:

        questions = (
            engine.interview_questions(
                resume_text,
                top_k=top_k,
            )
        )

        formatted = (
            format_interview_questions(
                questions,
                top_k=top_k,
            )
        )

    except Exception as exc:

        log(
            "Interview retrieval failed: {}",
            repr(exc),
        )

        print(
            traceback.format_exc(),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve interview questions."
            ),
        )


    elapsed = (
        time.perf_counter()
        - start
    )


    return {

        "success":
            True,

        "processing_time_seconds":
            round(
                elapsed,
                3,
            ),

        "questions":
            formatted,
    }


# ===============================================================
# API ALIAS
# ===============================================================

@app.post("/api/analyze")
async def api_analyze_cv(
    file: UploadFile = File(...),
    top_k: int = Form(10),
):

    return await analyze_cv(
        file=file,
        top_k=top_k,
    )


# ===============================================================
# STARTUP
# ===============================================================

@app.on_event("startup")
async def startup_event():

    log(
        "TalentMatch AI application started."
    )

    log(
        "MiniLM will load lazily."
    )

    log(
        "Job artifacts will load on first analysis."
    )

    log(
        "Interview artifacts will load only when "
        "/interview is requested."
    )

    log(
        "Prediction engine status: {}",
        engine.load_status(),
    )


# ===============================================================
# SHUTDOWN
# ===============================================================

@app.on_event("shutdown")
async def shutdown_event():

    log(
        "Application shutting down..."
    )

    try:

        engine.shutdown()

    except Exception as exc:

        log(
            "Shutdown warning: {}",
            repr(exc),
        )


# ===============================================================
# LOCAL DEVELOPMENT
# ===============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                "8000",
            )
        ),
        workers=1,
    )
