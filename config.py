"""
TalentMatch AI
Production configuration

Optimized for Render Free / low-memory deployment.
"""

from pathlib import Path
import os

# =========================================================
# Project directories
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"

# =========================================================
# Model
# =========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cpu"

# =========================================================
# Artifact paths
# =========================================================

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

# =========================================================
# Matching
# =========================================================

SEMANTIC_WEIGHT = 0.70

TFIDF_WEIGHT = 0.30

TOP_K_JOBS = 5

TOP_K_INTERVIEWS = 5

# =========================================================
# Upload protection
# =========================================================

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}

# =========================================================
# Resource limits
# =========================================================

MAX_RESUME_TEXT_LENGTH = 200_000

# Prevent excessive tokenizer/thread memory.
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false"
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1"
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1"
)
