"""
===============================================================
TalentMatch AI
Production Configuration
===============================================================

Optimized for:

    Render Free
    512 MB RAM
    CPU inference

This file contains configuration only.

No models are loaded here.
No large files are opened here.

===============================================================
"""

from pathlib import Path
import os


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"


# ============================================================
# Embedding Model
# ============================================================

# all-MiniLM-L6-v2 is intentionally used because it provides
# a good balance between:
#
#     • embedding quality
#     • inference speed
#     • model size
#     • RAM usage
#
# This is much more suitable for a 512 MB deployment than
# larger transformer models.

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Compute Device
# ============================================================

# Render Free does not provide a GPU.

DEVICE = "cpu"


# ============================================================
# CPU Thread Protection
# ============================================================

# Prevent numerical libraries from creating too many CPU
# threads on a small Render instance.
#
# These environment variables are normally set before Python
# libraries perform heavy numerical initialization.

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1"
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1"
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1"
)

os.environ.setdefault(
    "VECLIB_MAXIMUM_THREADS",
    "1"
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1"
)


# ============================================================
# Trained Artifacts
# ============================================================

JOB_EMBEDDINGS = (
    MODELS_DIR /
    "job_embeddings.npz"
)


JOB_METADATA = (
    MODELS_DIR /
    "job_metadata.parquet"
)


INTERVIEW_EMBEDDINGS = (
    MODELS_DIR /
    "interview_vectors.npz"
)


INTERVIEW_METADATA = (
    MODELS_DIR /
    "interview_metadata.parquet"
)


TFIDF_MODEL = (
    MODELS_DIR /
    "tfidf_vectorizer.pkl"
)


SKILLS = (
    MODELS_DIR /
    "skills.json.gz"
)


# ============================================================
# Ranking Configuration
# ============================================================

SEMANTIC_WEIGHT = float(
    os.getenv(
        "SEMANTIC_WEIGHT",
        "0.70"
    )
)


TFIDF_WEIGHT = float(
    os.getenv(
        "TFIDF_WEIGHT",
        "0.30"
    )
)


# ============================================================
# Validate Ranking Weights
# ============================================================

_weight_total = (
    SEMANTIC_WEIGHT +
    TFIDF_WEIGHT
)

if abs(_weight_total - 1.0) > 0.001:

    raise ValueError(
        "SEMANTIC_WEIGHT + TFIDF_WEIGHT "
        "must equal 1.0"
    )


# ============================================================
# Result Limits
# ============================================================

TOP_K_JOBS = int(
    os.getenv(
        "TOP_K_JOBS",
        "10"
    )
)


TOP_K_INTERVIEWS = int(
    os.getenv(
        "TOP_K_INTERVIEWS",
        "5"
    )
)


# ============================================================
# Hard Safety Limits
# ============================================================

# Prevent environment variables or API requests from
# accidentally requesting excessive results.

TOP_K_JOBS = min(
    max(1, TOP_K_JOBS),
    10
)


TOP_K_INTERVIEWS = min(
    max(1, TOP_K_INTERVIEWS),
    5
)


# ============================================================
# Upload Protection
# ============================================================

# Maximum uploaded resume size:
#
# 10 MB
#
# This protects the Render instance from unnecessarily large
# uploads.

MAX_UPLOAD_SIZE = (
    10 * 1024 * 1024
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}


# ============================================================
# Resume Processing Limits
# ============================================================

# Maximum extracted text processed from a resume.

MAX_RESUME_TEXT_LENGTH = 200_000


# ============================================================
# API Response Limits
# ============================================================

MAX_RETURNED_JOBS = 10

MAX_RETURNED_INTERVIEWS = 5


# ============================================================
# Server Configuration
# ============================================================

HOST = "0.0.0.0"


PORT = int(
    os.getenv(
        "PORT",
        "10000"
    )
)


# ============================================================
# Production Settings
# ============================================================

# Keep API documentation enabled because it is useful when
# demonstrating TalentMatch AI to employers.
#
# These values can be used by app.py.

APP_NAME = "TalentMatch AI"

APP_VERSION = "1.0.0"

APP_DESCRIPTION = (
    "AI-powered resume analysis, "
    "job matching and interview preparation."
)


# ============================================================
# Startup Validation
# ============================================================

# Fail early with a clear message if a required artifact
# is missing instead of allowing a confusing error later.

REQUIRED_ARTIFACTS = [
    JOB_EMBEDDINGS,
    JOB_METADATA,
    INTERVIEW_EMBEDDINGS,
    INTERVIEW_METADATA,
    TFIDF_MODEL,
    SKILLS,
]


def validate_artifacts():
    """
    Verify that required model artifacts exist.

    This is called by app.py during startup.
    """

    missing = [
        str(path)
        for path in REQUIRED_ARTIFACTS
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing TalentMatch AI artifacts:\n"
            + "\n".join(missing)
        )
