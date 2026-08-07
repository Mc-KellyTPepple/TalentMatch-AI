"""
=========================================================
TalentMatch AI
Configuration
=========================================================

Central configuration used throughout the application.

Every module imports from here instead of hardcoding paths.
"""

from pathlib import Path
import torch

# ==========================================================
# Project Directories
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

MODELS_DIR = ROOT_DIR / "models"
UPLOAD_DIR = ROOT_DIR / "uploads"
STATIC_DIR = ROOT_DIR / "static"
TEMPLATE_DIR = ROOT_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)

# ==========================================================
# Model Files
# ==========================================================

JOB_EMBEDDINGS = MODELS_DIR / "job_embeddings.npz"
JOB_METADATA = MODELS_DIR / "job_metadata.parquet"

INTERVIEW_EMBEDDINGS = MODELS_DIR / "interview_vectors.npz"
INTERVIEW_METADATA = MODELS_DIR / "interview_metadata.parquet"

TFIDF_MODEL = MODELS_DIR / "tfidf_vectorizer.pkl"

SKILLS = MODELS_DIR / "skills.json.gz"
SKILL_FREQUENCY = MODELS_DIR / "skill_frequency.json.gz"
SYNONYMS = MODELS_DIR / "synonyms.json.gz"
SKILL_GRAPH = MODELS_DIR / "skill_graph.json"

MODEL_CONFIG = MODELS_DIR / "model_config.json"
METADATA = MODELS_DIR / "metadata.json"
EVALUATION = MODELS_DIR / "evaluation.json"
HASHES = MODELS_DIR / "artifact_hashes.json"

# ==========================================================
# HuggingFace Sentence Transformer
# ==========================================================

# The model will automatically download on first startup.
# It is cached locally afterwards.

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIM = 384

NORMALIZE_EMBEDDINGS = True

# ==========================================================
# Resume Processing
# ==========================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024      # 10 MB

# ==========================================================
# Ranking Configuration
# ==========================================================

TOP_K_JOBS = 10

TOP_K_INTERVIEWS = 10

SEMANTIC_WEIGHT = 0.80

TFIDF_WEIGHT = 0.20

# ==========================================================
# FastAPI
# ==========================================================

HOST = "0.0.0.0"

PORT = 10000

# ==========================================================
# Torch
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# Application
# ==========================================================

APP_NAME = "TalentMatch AI"

APP_VERSION = "1.0.0"

# ==========================================================
# Utility
# ==========================================================

def verify_required_files():
    """
    Verify every required model artifact exists.
    """

    required = [
        JOB_EMBEDDINGS,
        JOB_METADATA,
        INTERVIEW_EMBEDDINGS,
        INTERVIEW_METADATA,
        TFIDF_MODEL,
        SKILLS,
        SKILL_FREQUENCY,
        SYNONYMS,
        SKILL_GRAPH,
        MODEL_CONFIG,
        METADATA,
        EVALUATION,
        HASHES,
    ]

    missing = [f.name for f in required if not f.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n"
            + "\n".join(missing)
        )

    return True
