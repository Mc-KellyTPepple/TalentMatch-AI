"""
=======================================================================
TalentMatch AI
Production Prediction Engine
=======================================================================

Responsibilities
----------------
- Lazy-load MiniLM
- Load precomputed job embeddings
- Load precomputed TF-IDF matrix
- Load TF-IDF vectorizer
- Perform semantic matching
- Perform TF-IDF matching
- Perform hybrid ranking
- Remove duplicate jobs from final recommendations
- Generate deterministic match explanations
- Generate employer-facing screening information
- Retrieve interview questions
- Maintain low-memory CPU inference

IMPORTANT
---------
The job TF-IDF matrix was generated from:

    category
    description
    requirements
    benefits

using:

    TfidfVectorizer(
        max_features=4000,
        dtype=np.float32,
        stop_words="english",
        ngram_range=(1, 2)
    )

The saved matrix is a CUSTOM NumPy CSR archive containing:

    data
    indices
    indptr
    shape

It MUST NOT be loaded using scipy.sparse.load_npz().

The following artifact order MUST remain identical:

    job_embeddings
            ↓
    job_metadata
            ↓
    job_tfidf_matrix

This file deliberately does NOT rebuild or modify any training artifact.
=======================================================================
"""

# =====================================================================
# ENVIRONMENT
# =====================================================================

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

os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    "/tmp/huggingface/transformers",
)

os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    "/tmp/huggingface/sentence_transformers",
)


# =====================================================================
# STANDARD LIBRARY
# =====================================================================

import gc
import re
import threading
import time
from pathlib import Path


# =====================================================================
# NUMPY / PANDAS / JOBLIB
# =====================================================================

import joblib
import numpy as np
import pandas as pd


# =====================================================================
# CONFIG
# =====================================================================

from config import (
    EMBEDDING_MODEL,
    JOB_EMBEDDINGS,
    JOB_METADATA,
    JOB_TFIDF_MATRIX,
    INTERVIEW_EMBEDDINGS,
    INTERVIEW_METADATA,
    TFIDF_MODEL,
    DEVICE,
    SEMANTIC_WEIGHT,
    TFIDF_WEIGHT,
    TOP_K_JOBS,
    TOP_K_INTERVIEWS,
)


# =====================================================================
# LOGGER
# =====================================================================

_MODULE_START = time.perf_counter()


def _log(message, *args):
    """
    Lightweight deployment-safe logger.
    """

    elapsed = (
        time.perf_counter()
        - _MODULE_START
    )

    if args:

        try:
            message = message.format(*args)

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


_log("predict.py imported.")


# =====================================================================
# PREDICTION ENGINE
# =====================================================================

class PredictionEngine:

    # =================================================================
    # SINGLETON
    # =================================================================

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):

        if cls._instance is None:

            with cls._instance_lock:

                if cls._instance is None:

                    cls._instance = (
                        super().__new__(cls)
                    )

                    cls._instance._initialized = False

        return cls._instance


    # =================================================================
    # INITIALIZATION
    # =================================================================

    def __init__(self):

        if self._initialized:
            return

        # -------------------------------------------------------------
        # Shared embedding model
        # -------------------------------------------------------------

        self.model = None

        # -------------------------------------------------------------
        # Job artifacts
        # -------------------------------------------------------------

        self.job_embeddings = None
        self.job_metadata = None

        self.vectorizer = None
        self.job_tfidf_matrix = None

        # -------------------------------------------------------------
        # Interview artifacts
        # -------------------------------------------------------------

        self.interview_embeddings = None
        self.interview_metadata = None

        # -------------------------------------------------------------
        # State
        # -------------------------------------------------------------

        self._model_loaded = False
        self._jobs_loaded = False
        self._interviews_loaded = False
        self._loading = False

        self._prediction_lock = threading.Lock()

        self._initialized = True

        _log(
            "PredictionEngine initialized."
        )

        _log(
            "Heavy artifacts will load lazily."
        )


    # =================================================================
    # STATUS
    # =================================================================

    def is_loaded(self):

        return bool(
            self._model_loaded
            and self._jobs_loaded
        )


    def load_status(self):

        return {

            "loaded":
                self.is_loaded(),

            "loading":
                bool(self._loading),

            "model_loaded":
                self.model is not None,

            "job_embeddings_loaded":
                self.job_embeddings is not None,

            "job_metadata_loaded":
                self.job_metadata is not None,

            "tfidf_vectorizer_loaded":
                self.vectorizer is not None,

            "tfidf_matrix_loaded":
                self.job_tfidf_matrix is not None,

            "interview_embeddings_loaded":
                self.interview_embeddings is not None,

            "interview_metadata_loaded":
                self.interview_metadata is not None,
        }


    # =================================================================
    # FILE VALIDATION
    # =================================================================

    @staticmethod
    def _require_file(path):

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

        if not path.is_file():

            raise RuntimeError(
                f"Artifact is not a file: {path}"
            )

        return path


    # =================================================================
    # TORCH CONFIGURATION
    # =================================================================

    @staticmethod
    def _configure_torch():

        try:

            import torch

            torch.set_num_threads(1)

            try:

                torch.set_num_interop_threads(1)

            except RuntimeError:

                pass

        except Exception as exc:

            _log(
                "PyTorch configuration warning: {}",
                repr(exc),
            )


    # =================================================================
    # LOAD EMBEDDINGS
    # =================================================================

    @staticmethod
    def _load_embeddings(path):

        path = Path(path)

        PredictionEngine._require_file(path)

        # -------------------------------------------------------------
        # NPY
        # -------------------------------------------------------------

        if path.suffix.lower() == ".npy":

            embeddings = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if embeddings.ndim != 2:

                raise RuntimeError(
                    "Expected 2D embeddings, got "
                    f"{embeddings.shape}"
                )

            if embeddings.dtype != np.float16:

                embeddings = embeddings.astype(
                    np.float16
                )

            return embeddings

        # -------------------------------------------------------------
        # NPZ
        # -------------------------------------------------------------

        loaded = np.load(
            path,
            allow_pickle=False,
        )

        try:

            if "embeddings" in loaded.files:

                embeddings = loaded["embeddings"]

            elif len(loaded.files) == 1:

                embeddings = loaded[
                    loaded.files[0]
                ]

            else:

                raise RuntimeError(
                    "Could not determine embedding array "
                    f"in {path}. Keys={loaded.files}"
                )

            if embeddings.ndim != 2:

                raise RuntimeError(
                    "Expected 2D embeddings, got "
                    f"{embeddings.shape}"
                )

            if embeddings.dtype != np.float16:

                embeddings = embeddings.astype(
                    np.float16
                )

            return embeddings

        finally:

            loaded.close()


    # =================================================================
    # LOAD CUSTOM TF-IDF CSR MATRIX
    # =================================================================

    @staticmethod
    def _load_custom_tfidf_matrix(path):

        """
        Reconstruct the exact CSR matrix created by:

            np.savez_compressed(
                matrix_path,
                data=...,
                indices=...,
                indptr=...,
                shape=...
            )
        """

        path = Path(path)

        PredictionEngine._require_file(path)

        _log(
            "Loading custom TF-IDF CSR archive: {}",
            path,
        )

        archive = np.load(
            path,
            allow_pickle=False,
        )

        try:

            required = {
                "data",
                "indices",
                "indptr",
                "shape",
            }

            missing = (
                required
                - set(archive.files)
            )

            if missing:

                raise RuntimeError(
                    "TF-IDF archive is missing required "
                    f"arrays: {sorted(missing)}"
                )

            data = np.asarray(
                archive["data"],
                dtype=np.float32,
            )

            indices = np.asarray(
                archive["indices"],
                dtype=np.int32,
            )

            indptr = np.asarray(
                archive["indptr"],
                dtype=np.int32,
            )

            shape_array = np.asarray(
                archive["shape"],
                dtype=np.int64,
            )

            if shape_array.size != 2:

                raise RuntimeError(
                    "Invalid TF-IDF matrix shape."
                )

            shape = (
                int(shape_array[0]),
                int(shape_array[1]),
            )

        finally:

            archive.close()

        # -------------------------------------------------------------
        # Structural validation
        # -------------------------------------------------------------

        if indptr.size != shape[0] + 1:

            raise RuntimeError(
                "Invalid TF-IDF indptr length. "
                f"Expected {shape[0] + 1}, "
                f"got {indptr.size}."
            )

        if indices.size != data.size:

            raise RuntimeError(
                "TF-IDF indices/data length mismatch."
            )

        if indptr[-1] != data.size:

            raise RuntimeError(
                "TF-IDF indptr/data mismatch."
            )

        if np.any(indptr < 0):

            raise RuntimeError(
                "TF-IDF indptr contains negative values."
            )

        if np.any(np.diff(indptr) < 0):

            raise RuntimeError(
                "TF-IDF indptr is not monotonic."
            )

        if indices.size:

            if np.min(indices) < 0:

                raise RuntimeError(
                    "TF-IDF indices contain negative values."
                )

            if np.max(indices) >= shape[1]:

                raise RuntimeError(
                    "TF-IDF indices exceed matrix columns."
                )

        from scipy.sparse import csr_matrix

        matrix = csr_matrix(
            (
                data,
                indices,
                indptr,
            ),
            shape=shape,
            dtype=np.float32,
        )

        return matrix


    # =================================================================
    # LOAD EMBEDDING MODEL
    # =================================================================

    def _load_embedding_model(self):

        if self.model is not None:

            self._model_loaded = True

            return

        self._configure_torch()

        _log(
            "Loading Hugging Face model: {}",
            EMBEDDING_MODEL,
        )

        from sentence_transformers import (
            SentenceTransformer,
        )

        start = time.perf_counter()

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu",
            cache_folder=os.environ.get(
                "SENTENCE_TRANSFORMERS_HOME"
            ),
        )

        self.model.eval()

        self._model_loaded = True

        _log(
            "MiniLM loaded in {:.3f}s.",
            time.perf_counter() - start,
        )


    # =================================================================
    # LOAD JOB ARTIFACTS
    # =================================================================

    def _load_jobs(self):

        if self._jobs_loaded:
            return

        with self._prediction_lock:

            if self._jobs_loaded:
                return

            self._loading = True

            start = time.perf_counter()

            try:

                _log(
                    "=============================================="
                )

                _log(
                    "Loading job matching artifacts..."
                )

                # -----------------------------------------------------
                # Model
                # -----------------------------------------------------

                self._load_embedding_model()

                # -----------------------------------------------------
                # Job embeddings
                # -----------------------------------------------------

                self.job_embeddings = (
                    self._load_embeddings(
                        JOB_EMBEDDINGS
                    )
                )

                _log(
                    "Job embeddings: shape={} dtype={}",
                    self.job_embeddings.shape,
                    self.job_embeddings.dtype,
                )

                # -----------------------------------------------------
                # Job metadata
                # -----------------------------------------------------

                self._require_file(
                    JOB_METADATA
                )

                self.job_metadata = (
                    pd.read_parquet(
                        JOB_METADATA
                    )
                )

                self.job_metadata.columns = [
                    str(c).strip().lower()
                    for c in self.job_metadata.columns
                ]

                # -----------------------------------------------------
                # Embedding / metadata alignment
                # -----------------------------------------------------

                if (
                    len(self.job_embeddings)
                    != len(self.job_metadata)
                ):

                    raise RuntimeError(
                        "JOB ARTIFACT ALIGNMENT ERROR: "
                        f"job_embeddings rows="
                        f"{len(self.job_embeddings)}, "
                        f"job_metadata rows="
                        f"{len(self.job_metadata)}"
                    )

                _log(
                    "Job metadata rows: {:,}",
                    len(self.job_metadata),
                )

                # -----------------------------------------------------
                # TF-IDF vectorizer
                # -----------------------------------------------------

                self._require_file(
                    TFIDF_MODEL
                )

                self.vectorizer = joblib.load(
                    TFIDF_MODEL
                )

                if not hasattr(
                    self.vectorizer,
                    "vocabulary_",
                ):

                    raise RuntimeError(
                        "Loaded TF-IDF vectorizer does not "
                        "contain vocabulary_."
                    )

                vectorizer_features = len(
                    self.vectorizer.vocabulary_
                )

                _log(
                    "TF-IDF vocabulary size: {:,}",
                    vectorizer_features,
                )

                # -----------------------------------------------------
                # Custom TF-IDF matrix
                # -----------------------------------------------------

                self.job_tfidf_matrix = (
                    self._load_custom_tfidf_matrix(
                        JOB_TFIDF_MATRIX
                    )
                )

                _log(
                    "TF-IDF matrix shape: {}",
                    self.job_tfidf_matrix.shape,
                )

                # -----------------------------------------------------
                # CRITICAL ALIGNMENT CHECKS
                # -----------------------------------------------------

                if (
                    self.job_tfidf_matrix.shape[0]
                    != len(self.job_metadata)
                ):

                    raise RuntimeError(
                        "TF-IDF/job metadata alignment error: "
                        f"TF-IDF rows="
                        f"{self.job_tfidf_matrix.shape[0]}, "
                        f"metadata rows="
                        f"{len(self.job_metadata)}"
                    )

                if (
                    self.job_tfidf_matrix.shape[0]
                    != len(self.job_embeddings)
                ):

                    raise RuntimeError(
                        "TF-IDF/job embedding alignment error: "
                        f"TF-IDF rows="
                        f"{self.job_tfidf_matrix.shape[0]}, "
                        f"embedding rows="
                        f"{len(self.job_embeddings)}"
                    )

                if (
                    self.job_tfidf_matrix.shape[1]
                    != vectorizer_features
                ):

                    raise RuntimeError(
                        "TF-IDF feature mismatch: "
                        f"matrix columns="
                        f"{self.job_tfidf_matrix.shape[1]}, "
                        f"vectorizer vocabulary="
                        f"{vectorizer_features}"
                    )

                # -----------------------------------------------------
                # READY
                # -----------------------------------------------------

                self._jobs_loaded = True

                _log(
                    "Job matching artifacts aligned successfully."
                )

                _log(
                    "Job artifacts loaded in {:.3f}s.",
                    time.perf_counter() - start,
                )

                _log(
                    "=============================================="
                )

            except Exception:

                self._cleanup_jobs()

                raise

            finally:

                self._loading = False

                gc.collect()


    # =================================================================
    # LOAD INTERVIEW ARTIFACTS
    # =================================================================

    def _load_interviews(self):

        if self._interviews_loaded:
            return

        with self._prediction_lock:

            if self._interviews_loaded:
                return

            self._loading = True

            start = time.perf_counter()

            try:

                _log(
                    "Loading interview artifacts..."
                )

                self._load_embedding_model()

                self.interview_embeddings = (
                    self._load_embeddings(
                        INTERVIEW_EMBEDDINGS
                    )
                )

                self._require_file(
                    INTERVIEW_METADATA
                )

                self.interview_metadata = (
                    pd.read_parquet(
                        INTERVIEW_METADATA
                    )
                )

                self.interview_metadata.columns = [
                    str(c).strip().lower()
                    for c in self.interview_metadata.columns
                ]

                if (
                    len(self.interview_embeddings)
                    != len(self.interview_metadata)
                ):

                    raise RuntimeError(
                        "Interview embedding/metadata "
                        "alignment error: "
                        f"embeddings="
                        f"{len(self.interview_embeddings)}, "
                        f"metadata="
                        f"{len(self.interview_metadata)}"
                    )

                self._interviews_loaded = True

                _log(
                    "Interview artifacts loaded in {:.3f}s.",
                    time.perf_counter() - start,
                )

            except Exception:

                self._cleanup_interviews()

                raise

            finally:

                self._loading = False

                gc.collect()


    # =================================================================
    # EMBEDDING
    # =================================================================

    def embed(self, text):

        if text is None:

            raise ValueError(
                "Text cannot be None."
            )

        text = str(text).strip()

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        self._load_embedding_model()

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=1,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )


    # =================================================================
    # TEXT NORMALIZATION
    # =================================================================

    @staticmethod
    def _normalize_text(text):
        """
        Normalize text for deterministic duplicate detection
        and keyword comparison.

        This does NOT modify the original job text returned to
        the frontend.
        """

        if text is None:

            return ""

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9+#.\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()


    @staticmethod
    def _tokenize_text(text):
        """
        Lightweight tokenization.

        Designed for low-memory deployment and deliberately
        avoids loading another NLP model.
        """

        normalized = (
            PredictionEngine
            ._normalize_text(text)
        )

        if not normalized:

            return set()

        tokens = re.findall(
            r"[a-z][a-z0-9+#.]{1,}",
            normalized,
        )

        return set(tokens)


    # =================================================================
    # JOB IDENTITY
    # =================================================================

    @classmethod
    def _job_identity(cls, row):
        """
        Create a stable content-based identity for duplicate jobs.

        The identity deliberately ignores the original dataframe
        index so that duplicated records representing the same job
        are grouped together.
        """

        title = cls._normalize_text(
            row.get("title", "")
        )

        category = cls._normalize_text(
            row.get("category", "")
        )

        description = cls._normalize_text(
            row.get("description", "")
        )

        requirements = cls._normalize_text(
            row.get("requirements", "")
        )

        benefits = cls._normalize_text(
            row.get("benefits", "")
        )

        # Prefer the substantial job content.
        identity_text = "|".join(
            [
                title,
                category,
                description,
                requirements,
                benefits,
            ]
        )

        import hashlib

        return hashlib.sha1(
            identity_text.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()


    # =================================================================
    # DUPLICATE REMOVAL
    # =================================================================

    @classmethod
    def _deduplicate_results(
        cls,
        results,
        top_k,
    ):
        """
        Remove duplicate jobs while preserving ranking order.

        Because results are already sorted by final score,
        the first occurrence of a duplicate is the strongest
        occurrence and is retained.
        """

        if not results:

            return []

        seen = set()
        unique = []

        duplicate_count = 0

        for result in results:

            identity = result.get(
                "_job_identity"
            )

            if not identity:

                identity = cls._job_identity(
                    result
                )

            if identity in seen:

                duplicate_count += 1

                continue

            seen.add(identity)

            result["duplicate_count"] = 0

            unique.append(result)

            if len(unique) >= int(top_k):

                break

        if duplicate_count:

            _log(
                "Removed {} duplicate job result(s).",
                duplicate_count,
            )

        # Internal field should not leak to API.
        for result in unique:

            result.pop(
                "_job_identity",
                None,
            )

        return unique


    # =================================================================
    # TOP INDICES
    # =================================================================

    @staticmethod
    def _top_indices(scores, top_k):

        if scores is None:

            return np.empty(
                0,
                dtype=np.int64,
            )

        total = int(
            scores.shape[0]
        )

        if total == 0:

            return np.empty(
                0,
                dtype=np.int64,
            )

        top_k = min(
            max(1, int(top_k)),
            total,
        )

        if top_k >= total:

            return np.argsort(
                scores
            )[::-1]

        indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        order = np.argsort(
            scores[indices]
        )[::-1]

        return indices[order]


    # =================================================================
    # KEYWORD EXTRACTION
    # =================================================================

    def _extract_keyword_analysis(
        self,
        resume_text,
        row,
    ):
        """
        Deterministic keyword overlap analysis.

        This is intentionally lightweight.

        It is not presented as a semantic NLP explanation.
        It simply identifies meaningful token overlap between
        the resume and job requirements/description.
        """

        resume_tokens = self._tokenize_text(
            resume_text
        )

        requirement_text = " ".join(
            [
                self._safe_value(
                    row.get(
                        "requirements",
                        "",
                    )
                ),
                self._safe_value(
                    row.get(
                        "description",
                        "",
                    )
                ),
                self._safe_value(
                    row.get(
                        "category",
                        "",
                    )
                ),
                self._safe_value(
                    row.get(
                        "title",
                        "",
                    )
                ),
            ]
        )

        job_tokens = self._tokenize_text(
            requirement_text
        )

        if not resume_tokens or not job_tokens:

            return {
                "matched_keywords": [],
                "missing_keywords": [],
                "keyword_overlap": 0.0,
            }

        matched = sorted(
            resume_tokens.intersection(
                job_tokens
            )
        )

        # Ignore extremely common generic words.
        generic_words = {
            "work",
            "working",
            "team",
            "teams",
            "using",
            "use",
            "strong",
            "experience",
            "skills",
            "skill",
            "knowledge",
            "ability",
            "role",
            "job",
            "candidate",
            "required",
            "requirements",
        }

        matched = [
            word
            for word in matched
            if word not in generic_words
        ]

        # Limit output so a job result does not become huge.
        matched = matched[:15]

        # For "missing" terms, prioritize terms from
        # requirements rather than the complete description.
        requirement_only = self._tokenize_text(
            self._safe_value(
                row.get(
                    "requirements",
                    "",
                )
            )
        )

        missing = sorted(
            requirement_only
            - resume_tokens
        )

        missing = [
            word
            for word in missing
            if word not in generic_words
            and len(word) >= 3
        ][:15]

        overlap = (
            len(matched)
            /
            max(
                1,
                len(job_tokens),
            )
        )

        return {
            "matched_keywords": matched,
            "missing_keywords": missing,
            "keyword_overlap": float(
                min(
                    overlap,
                    1.0,
                )
            ),
        }


    # =================================================================
    # MATCH LEVEL
    # =================================================================

    @staticmethod
    def _match_level(score):
        """
        User-facing interpretation of the hybrid score.

        The thresholds are intentionally conservative and are
        based on the current score range rather than claiming
        that the system predicts hiring probability.
        """

        score = float(score)

        if score >= 0.70:

            return "Strong Match"

        if score >= 0.55:

            return "Good Match"

        if score >= 0.40:

            return "Developing Match"

        if score >= 0.25:

            return "Low Match"

        return "Limited Match"


    # =================================================================
    # MATCH EXPLANATION
    # =================================================================

    def _build_match_explanation(
        self,
        resume_text,
        row,
        score,
        semantic_score,
        tfidf_score,
    ):
        """
        Build an explainable, deterministic summary.

        Important:
        This does NOT claim that the model understands why an
        employer would hire the candidate. It explains the
        numerical signals used by TalentMatch AI.
        """

        semantic_weight = float(
            SEMANTIC_WEIGHT
        )

        tfidf_weight = float(
            TFIDF_WEIGHT
        )

        semantic_contribution = (
            float(semantic_score)
            * semantic_weight
        )

        keyword_contribution = (
            float(tfidf_score)
            * tfidf_weight
        )

        keyword_analysis = (
            self._extract_keyword_analysis(
                resume_text,
                row,
            )
        )

        matched = keyword_analysis[
            "matched_keywords"
        ]

        missing = keyword_analysis[
            "missing_keywords"
        ]

        # -------------------------------------------------------------
        # Primary signal
        # -------------------------------------------------------------

        if semantic_contribution > (
            keyword_contribution * 1.25
        ):

            primary_signal = (
                "Strong semantic alignment"
            )

        elif keyword_contribution > (
            semantic_contribution * 1.25
        ):

            primary_signal = (
                "Strong keyword alignment"
            )

        else:

            primary_signal = (
                "Balanced semantic and keyword alignment"
            )

        # -------------------------------------------------------------
        # Explanation
        # -------------------------------------------------------------

        if matched:

            keyword_text = ", ".join(
                matched[:6]
            )

            explanation = (
                f"{primary_signal}. "
                f"The resume shows overlap with "
                f"job-related terms including "
                f"{keyword_text}."
            )

        else:

            explanation = (
                f"{primary_signal}. "
                "The match is driven mainly by "
                "semantic similarity rather than "
                "direct keyword overlap."
            )

        # -------------------------------------------------------------
        # Strengths
        # -------------------------------------------------------------

        strengths = []

        if semantic_score >= 0.60:

            strengths.append(
                "Strong semantic alignment with the role."
            )

        elif semantic_score >= 0.45:

            strengths.append(
                "Moderate semantic alignment with the role."
            )

        if tfidf_score >= 0.20:

            strengths.append(
                "Relevant terminology overlaps with the job."
            )

        if matched:

            strengths.append(
                "Relevant terms: "
                + ", ".join(
                    matched[:8]
                )
                + "."
            )

        if not strengths:

            strengths.append(
                "The profile has some measurable similarity "
                "to the job requirements."
            )

        # -------------------------------------------------------------
        # Gaps
        # -------------------------------------------------------------

        gaps = []

        if tfidf_score < 0.15:

            gaps.append(
                "Limited direct keyword overlap with the job text."
            )

        if missing:

            gaps.append(
                "Potentially missing or unconfirmed terms: "
                + ", ".join(
                    missing[:8]
                )
                + "."
            )

        if not gaps:

            gaps.append(
                "No major keyword gap was detected by the "
                "current screening signals."
            )

        return {

            "match_level":
                self._match_level(score),

            "match_explanation":
                explanation,

            "primary_signal":
                primary_signal,

            "semantic_contribution":
                float(
                    semantic_contribution
                ),

            "keyword_contribution":
                float(
                    keyword_contribution
                ),

            "matched_keywords":
                matched,

            "missing_keywords":
                missing,

            "strengths":
                strengths,

            "gaps":
                gaps,
        }


    # =================================================================
    # EMPLOYER-FACING SUMMARY
    # =================================================================

    @staticmethod
    def _employer_recommendation(
        score,
        semantic_score,
        tfidf_score,
    ):
        """
        Screening recommendation based only on the current
        numerical matching signals.

        This is NOT a hiring decision and is intentionally
        framed as a screening signal.
        """

        score = float(score)
        semantic_score = float(
            semantic_score
        )
        tfidf_score = float(
            tfidf_score
        )

        if (
            score >= 0.70
            and semantic_score >= 0.60
        ):

            return {
                "recommendation":
                    "Prioritize for review",

                "screening_signal":
                    "High alignment",

                "reason":
                    "Strong overall and semantic alignment "
                    "with the job profile.",
            }

        if score >= 0.55:

            return {
                "recommendation":
                    "Consider for review",

                "screening_signal":
                    "Good alignment",

                "reason":
                    "The candidate shows meaningful alignment "
                    "with the role.",
            }

        if score >= 0.40:

            return {
                "recommendation":
                    "Review selectively",

                "screening_signal":
                    "Developing alignment",

                "reason":
                    "The candidate shows some relevant "
                    "alignment, but additional screening "
                    "may be useful.",
            }

        return {
            "recommendation":
                "Lower priority",

            "screening_signal":
                "Limited alignment",

            "reason":
                "The current semantic and keyword signals "
                "show limited alignment with the role.",
        }


    # =================================================================
    # TF-IDF SCORING
    # =================================================================

    def _tfidf_scores_from_text(
        self,
        resume_text,
    ):

        if self.vectorizer is None:

            self._load_jobs()

        query = self.vectorizer.transform(
            [str(resume_text)]
        )

        scores = (
            self.job_tfidf_matrix
            @ query.T
        ).toarray().ravel()

        del query

        return np.asarray(
            scores,
            dtype=np.float32,
        )


    # =================================================================
    # SEMANTIC JOB SEARCH
    # =================================================================

    def semantic_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS,
    ):

        self._load_jobs()

        query = self.embed(
            resume_text
        )

        scores = np.dot(
            self.job_embeddings,
            query,
        )

        indices = self._top_indices(
            scores,
            top_k,
        )

        results = []

        for idx in indices:

            idx = int(idx)

            row = self.job_metadata.iloc[idx]

            result = self._job_result(
                row=row,
                score=float(
                    scores[idx]
                ),
                semantic_score=float(
                    scores[idx]
                ),
                tfidf_score=0.0,
                resume_text=resume_text,
            )

            results.append(result)

        return self._deduplicate_results(
            results,
            top_k,
        )


    # =================================================================
    # TF-IDF ONLY
    # =================================================================

    def tfidf_scores(
        self,
        resume_text,
    ):

        self._load_jobs()

        return self._tfidf_scores_from_text(
            resume_text
        )


    # =================================================================
    # HYBRID JOB SEARCH
    # =================================================================

    def hybrid_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS,
    ):

        start = time.perf_counter()

        self._load_jobs()

        # -------------------------------------------------------------
        # ONE RESUME EMBEDDING
        # -------------------------------------------------------------

        embed_start = time.perf_counter()

        query = self.embed(
            resume_text
        )

        _log(
            "Resume embedding generated in {:.3f}s.",
            time.perf_counter()
            - embed_start,
        )

        # -------------------------------------------------------------
        # SEMANTIC
        # -------------------------------------------------------------

        semantic_start = time.perf_counter()

        semantic_scores = np.dot(
            self.job_embeddings,
            query,
        )

        _log(
            "Semantic scoring completed in {:.3f}s.",
            time.perf_counter()
            - semantic_start,
        )

        # -------------------------------------------------------------
        # TF-IDF
        # -------------------------------------------------------------

        tfidf_start = time.perf_counter()

        lexical_scores = (
            self._tfidf_scores_from_text(
                resume_text
            )
        )

        _log(
            "TF-IDF scoring completed in {:.3f}s.",
            time.perf_counter()
            - tfidf_start,
        )

        # -------------------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------------------

        if (
            semantic_scores.shape
            != lexical_scores.shape
        ):

            raise RuntimeError(
                "Semantic and TF-IDF score arrays "
                "are not aligned: "
                f"semantic={semantic_scores.shape}, "
                f"tfidf={lexical_scores.shape}"
            )

        # -------------------------------------------------------------
        # HYBRID SCORE
        #
        # IMPORTANT:
        # This formula is unchanged from the working version.
        # -------------------------------------------------------------

        final_scores = (
            semantic_scores
            * float(SEMANTIC_WEIGHT)
            +
            lexical_scores
            * float(TFIDF_WEIGHT)
        )

        # -------------------------------------------------------------
        # IMPORTANT DUPLICATE HANDLING
        #
        # Request more candidates than the final display count.
        #
        # Example:
        #
        # TOP_K_JOBS = 10
        #
        # We inspect more than 10 candidates because several
        # of the first candidates may be duplicates.
        # -------------------------------------------------------------

        candidate_count = min(
            len(final_scores),
            max(
                int(top_k) * 3,
                int(top_k) + 10,
            ),
        )

        indices = self._top_indices(
            final_scores,
            candidate_count,
        )

        jobs = []

        for idx in indices:

            idx = int(idx)

            row = self.job_metadata.iloc[idx]

            jobs.append(
                self._job_result(
                    row=row,
                    score=float(
                        final_scores[idx]
                    ),
                    semantic_score=float(
                        semantic_scores[idx]
                    ),
                    tfidf_score=float(
                        lexical_scores[idx]
                    ),
                    resume_text=resume_text,
                )
            )

        # -------------------------------------------------------------
        # Remove duplicates AFTER ranking.
        #
        # Therefore the strongest duplicate survives.
        # -------------------------------------------------------------

        jobs = self._deduplicate_results(
            jobs,
            top_k,
        )

        _log(
            "Hybrid search completed in {:.3f}s.",
            time.perf_counter() - start,
        )

        _log(
            "Unique jobs returned: {}",
            len(jobs),
        )

        return jobs


    # =================================================================
    # JOB RESULT
    # =================================================================

    def _job_result(
        self,
        row,
        score,
        semantic_score,
        tfidf_score,
        resume_text=None,
    ):
        """
        Build the complete API-safe job result.

        Existing fields are preserved.
        New explanation/employer fields are additive.
        """

        title = self._safe_value(
            row.get(
                "title",
                "",
            )
        )

        category = self._safe_value(
            row.get(
                "category",
                "",
            )
        )

        description = self._safe_value(
            row.get(
                "description",
                "",
            )
        )

        requirements = self._safe_value(
            row.get(
                "requirements",
                "",
            )
        )

        benefits = self._safe_value(
            row.get(
                "benefits",
                "",
            )
        )

        result = {

            # ---------------------------------------------------------
            # EXISTING CORE FIELDS
            # ---------------------------------------------------------

            "score":
                float(score),

            "semantic_score":
                float(semantic_score),

            "tfidf_score":
                float(tfidf_score),

            "category":
                category,

            "title":
                title,

            "description":
                description,

            "requirements":
                requirements,

            "benefits":
                benefits,
        }

        # -------------------------------------------------------------
        # Explanation
        # -------------------------------------------------------------

        if resume_text is not None:

            explanation = (
                self._build_match_explanation(
                    resume_text=resume_text,
                    row=row,
                    score=score,
                    semantic_score=semantic_score,
                    tfidf_score=tfidf_score,
                )
            )

            result.update(
                explanation
            )

            # ---------------------------------------------------------
            # Employer-facing screening information
            # ---------------------------------------------------------

            employer = (
                self._employer_recommendation(
                    score=score,
                    semantic_score=semantic_score,
                    tfidf_score=tfidf_score,
                )
            )

            result["employer_summary"] = {

                "candidate_fit":
                    explanation[
                        "match_level"
                    ],

                "recommendation":
                    employer[
                        "recommendation"
                    ],

                "screening_signal":
                    employer[
                        "screening_signal"
                    ],

                "reason":
                    employer[
                        "reason"
                    ],

                "strengths":
                    explanation[
                        "strengths"
                    ],

                "gaps":
                    explanation[
                        "gaps"
                    ],
            }

        # -------------------------------------------------------------
        # Internal duplicate identity.
        #
        # Removed before result reaches the frontend.
        # -------------------------------------------------------------

        result["_job_identity"] = (
            self._job_identity(row)
        )

        return result


    # =================================================================
    # INTERVIEW QUESTIONS
    # =================================================================

    def interview_questions(
        self,
        resume_text,
        top_k=TOP_K_INTERVIEWS,
    ):

        self._load_interviews()

        query = self.embed(
            resume_text
        )

        scores = np.dot(
            self.interview_embeddings,
            query,
        )

        indices = self._top_indices(
            scores,
            top_k,
        )

        questions = []

        for idx in indices:

            idx = int(idx)

            row = self.interview_metadata.iloc[
                idx
            ]

            questions.append({

                "score":
                    float(
                        scores[idx]
                    ),

                "question":
                    self._safe_value(
                        row.get(
                            "question",
                            "",
                        )
                    ),

                "answer":
                    self._safe_value(
                        row.get(
                            "answer",
                            "",
                        )
                    ),

                "role":
                    self._safe_value(
                        row.get(
                            "role",
                            "",
                        )
                    ),

                "category":
                    self._safe_value(
                        row.get(
                            "category",
                            "",
                        )
                    ),

                "difficulty":
                    self._safe_value(
                        row.get(
                            "difficulty",
                            "",
                        )
                    ),

                "experience":
                    self._safe_value(
                        row.get(
                            "experience",
                            "",
                        )
                    ),
            })

        return questions


    # =================================================================
    # SAFE VALUE
    # =================================================================

    @staticmethod
    def _safe_value(value):

        if value is None:

            return ""

        try:

            if pd.isna(value):

                return ""

        except Exception:

            pass

        return str(value)


    # =================================================================
    # CLEANUP JOBS
    # =================================================================

    def _cleanup_jobs(self):

        self.job_embeddings = None
        self.job_metadata = None
        self.vectorizer = None
        self.job_tfidf_matrix = None

        self._jobs_loaded = False

        gc.collect()


    # =================================================================
    # CLEANUP INTERVIEWS
    # =================================================================

    def _cleanup_interviews(self):

        self.interview_embeddings = None
        self.interview_metadata = None

        self._interviews_loaded = False

        gc.collect()


    # =================================================================
    # SHUTDOWN
    # =================================================================

    def shutdown(self):

        with self._prediction_lock:

            _log(
                "Releasing prediction engine..."
            )

            self._cleanup_jobs()

            self._cleanup_interviews()

            self.model = None

            self._model_loaded = False

            gc.collect()

            _log(
                "Prediction engine released."
            )


# =====================================================================
# GLOBAL SINGLETON
# =====================================================================

engine = PredictionEngine()

_log(
    "Global prediction engine created."
)

_log(
    "No Hugging Face model loaded at import time."
)
