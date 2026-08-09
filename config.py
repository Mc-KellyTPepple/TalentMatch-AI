"""
TalentMatch AI
Deployment Configuration

All inference artifacts are located in:

    models/

This configuration is designed for:
    - FastAPI
    - Render Free
    - CPU inference
    - 512 MB RAM target
"""

from pathlib import Path


# ===============================================================
# BASE DIRECTORIES
# ===============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"


# ===============================================================
# HUGGING FACE MODEL
# ===============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cpu"


# ===============================================================
# JOB ARTIFACTS
# ===============================================================

JOB_EMBEDDINGS = (
    MODELS_DIR / "job_embeddings.npz"
)

JOB_METADATA = (
    MODELS_DIR / "job_metadata.parquet"
)


# ===============================================================
# TF-IDF ARTIFACTS
# ===============================================================

TFIDF_MODEL = (
    MODELS_DIR / "tfidf_vectorizer.pkl"
)

JOB_TFIDF_MATRIX = (
    MODELS_DIR / "job_tfidf_matrix.npz"
)


# ===============================================================
# INTERVIEW ARTIFACTS
# ===============================================================

INTERVIEW_EMBEDDINGS = (
    MODELS_DIR / "interview_vectors.npz"
)

INTERVIEW_METADATA = (
    MODELS_DIR / "interview_metadata.parquet"
)


# ===============================================================
# RANKING
# ===============================================================

# These values must match the values used when you designed
# your hybrid ranking system.

SEMANTIC_WEIGHT = 0.70

TFIDF_WEIGHT = 0.30


# ===============================================================
# LIMITS
# ===============================================================

TOP_K_JOBS = 20

TOP_K_INTERVIEWS = 10


# ===============================================================
# TF-IDF TRAINING CONTRACT
# ===============================================================

# These describe the exact process used to create
# job_tfidf_matrix.npz.

TFIDF_MAX_FEATURES = 4000

TFIDF_STOP_WORDS = "english"

TFIDF_NGRAM_RANGE = (1, 2)

TFIDF_NORM = "l2"

TFIDF_USE_IDF = True

TFIDF_SMOOTH_IDF = True


# ===============================================================
# REQUIRED FILES
# ===============================================================

REQUIRED_JOB_ARTIFACTS = (
    JOB_EMBEDDINGS,
    JOB_METADATA,
    JOB_TFIDF_MATRIX,
    TFIDF_MODEL,
)

REQUIRED_INTERVIEW_ARTIFACTS = (
    INTERVIEW_EMBEDDINGS,
    INTERVIEW_METADATA,
)
