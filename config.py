"""
TalentMatch AI
Production Configuration

Optimized for Render Free / 512 MB RAM.
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

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEVICE = "cpu"


# ============================================================
# Trained Artifacts
# ============================================================

JOB_EMBEDDINGS = (
    MODELS_DIR / "job_embeddings.npz"
)

JOB_METADATA = (
    MODELS_DIR / "job_metadata.parquet"
)

INTERVIEW_EMBEDDINGS = (
    MODELS_DIR / "interview_vectors.npz"
)

INTERVIEW_METADATA = (
    MODELS_DIR / "interview_metadata.parquet"
)

TFIDF_MODEL = (
    MODELS_DIR / "tfidf_vectorizer.pkl"
)

SKILLS = (
    MODELS_DIR / "skills.json.gz"
)


# ============================================================
# Ranking
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
# Result Limits
# ============================================================

TOP_K_JOBS = 10

TOP_K_INTERVIEWS = 5


# ============================================================
# Upload Protection
# ============================================================

MAX_UPLOAD_SIZE = 10 * 1024 * 1024


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}


# ============================================================
# Resume Processing Limits
# ============================================================

MAX_RESUME_TEXT_LENGTH = 200_000


# ============================================================
# Memory Protection
# ============================================================

# Maximum number of jobs examined by the API.
#
# The trained embedding file can contain many jobs.
# Keeping the full array is still necessary for fast
# similarity search, but these limits prevent excessively
# large responses.

MAX_RETURNED_JOBS = 10

MAX_RETURNED_INTERVIEWS = 5


# ============================================================
# Server
# ============================================================

HOST = "0.0.0.0"

PORT = int(
    os.getenv(
        "PORT",
        "10000"
    )
)
