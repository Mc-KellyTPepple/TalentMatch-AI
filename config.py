"""
TalentMatch AI
Configuration
"""

from pathlib import Path
import os

# -------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"

UPLOAD_DIR = BASE_DIR / "uploads"

TEMPLATE_DIR = BASE_DIR / "templates"

STATIC_DIR = BASE_DIR / "static"

# -------------------------------------------------------
# DATASETS
# -------------------------------------------------------

RESUME_DATASET = DATASETS_DIR / "resumes.csv"

JOB_DATASET = DATASETS_DIR / "job_descriptions.csv"

INTERVIEW_DATASET = DATASETS_DIR / "interview_questions.csv"

# -------------------------------------------------------
# MODEL FILES
# -------------------------------------------------------

ONNX_MODEL = MODELS_DIR / "sentence_model.onnx"

LABEL_ENCODER = MODELS_DIR / "label_encoder.pkl"

MODEL_METADATA = MODELS_DIR / "metadata.json"

# -------------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SEQUENCE_LENGTH = 256

EMBEDDING_DIM = 384

# -------------------------------------------------------
# ATS SCORING
# -------------------------------------------------------

SEMANTIC_WEIGHT = 0.55

SKILL_WEIGHT = 0.25

EXPERIENCE_WEIGHT = 0.10

EDUCATION_WEIGHT = 0.05

QUALITY_WEIGHT = 0.05

# -------------------------------------------------------
# FILES
# -------------------------------------------------------

ALLOWED_EXTENSIONS = {

    ".pdf",

    ".docx"

}

MAX_UPLOAD_MB = 10

# -------------------------------------------------------
# INFERENCE
# -------------------------------------------------------

TOP_INTERVIEW_QUESTIONS = 5

TOP_MATCHES = 10

MINIMUM_MATCH_SCORE = 40

# -------------------------------------------------------
# RENDER
# -------------------------------------------------------

HOST = "0.0.0.0"

PORT = int(os.getenv("PORT", 8000))

# -------------------------------------------------------
# RANDOMNESS
# -------------------------------------------------------

SEED = 42

# -------------------------------------------------------
# LOGGING
# -------------------------------------------------------

LOG_LEVEL = "INFO"

# -------------------------------------------------------
# CREATE DIRECTORIES
# -------------------------------------------------------

for directory in [

    DATASETS_DIR,

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    MODELS_DIR,

    UPLOAD_DIR

]:

    directory.mkdir(

        parents=True,

        exist_ok=True

    )
