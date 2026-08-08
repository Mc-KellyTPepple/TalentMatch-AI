"""
================================================================
TalentMatch AI
Production Prediction Engine
================================================================

Responsibilities
-----------------
• Lazy-load the MiniLM embedding model
• Semantic job matching
• TF-IDF job matching
• Hybrid job ranking
• Interview question retrieval
• Safe artifact loading
• Singleton model management

Deployment
----------
Render Free
512 MB RAM
CPU inference

Memory Strategy
---------------
• No model loading during module import
• Lazy-load all heavy AI artifacts
• Singleton prediction engine
• CPU-only inference
• One PyTorch CPU thread
• Disable tokenizer parallelism
• Keep stored embeddings in float16
• Convert query embeddings to float32
• Avoid unnecessary array copies
• Use NumPy dot products for normalized embeddings
• Use argpartition for efficient top-k selection
• Keep TF-IDF sparse
• Explicit temporary-object cleanup
• No resume storage
================================================================
"""


# ================================================================
# IMPORTANT
# Set CPU/thread environment variables BEFORE importing
# libraries that may import PyTorch/BLAS.
# ================================================================

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


# ================================================================
# STANDARD LIBRARY
# ================================================================

import gc
import threading


# ================================================================
# NUMPY / JOBLIB / PANDAS
# ================================================================

import joblib
import numpy as np
import pandas as pd


# ================================================================
# CONFIG
# ================================================================

from config import (
    EMBEDDING_MODEL,
    JOB_EMBEDDINGS,
    JOB_METADATA,
    INTERVIEW_EMBEDDINGS,
    INTERVIEW_METADATA,
    TFIDF_MODEL,
    DEVICE,
    SEMANTIC_WEIGHT,
    TFIDF_WEIGHT,
    TOP_K_JOBS,
    TOP_K_INTERVIEWS,
)


# ================================================================
# PREDICTION ENGINE
# ================================================================

class PredictionEngine:
    """
    Lightweight singleton prediction engine.

    IMPORTANT
    ---------
    Importing this module does NOT load:

        • SentenceTransformer
        • PyTorch model
        • Job embeddings
        • Interview embeddings
        • Job metadata
        • Interview metadata
        • TF-IDF matrix

    Heavy artifacts are loaded only when the first prediction
    request is made.

    This is important for Render Free's 512 MB RAM limit.
    """

    _instance = None
    _instance_lock = threading.Lock()

    # ============================================================
    # Singleton
    # ============================================================

    def __new__(cls):

        if cls._instance is None:

            with cls._instance_lock:

                if cls._instance is None:

                    cls._instance = super().__new__(
                        cls
                    )

                    cls._instance._initialized = False
                    cls._instance._model_loaded = False
                    cls._instance._loading = False

        return cls._instance

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self):

        if self._initialized:
            return

        # --------------------------------------------------------
        # Heavy objects remain None.
        # --------------------------------------------------------

        self.model = None

        self.job_embeddings = None
        self.job_metadata = None
        self.job_tfidf_matrix = None

        self.interview_embeddings = None
        self.interview_metadata = None

        self.vectorizer = None

        # --------------------------------------------------------
        # Loading state
        # --------------------------------------------------------

        self._model_loaded = False
        self._loading = False

        self._initialized = True

        print(
            "=============================================="
        )

        print(
            "TalentMatch AI prediction engine initialized."
        )

        print(
            "Heavy AI artifacts will be loaded lazily."
        )

        print(
            "=============================================="
        )

    # ============================================================
    # PUBLIC LOADED STATUS
    # ============================================================

    def is_loaded(self):
        """
        Return True only when the complete prediction engine
        has successfully loaded.

        This method is intentionally lightweight.

        It is used by app.py for:

            "prediction_engine": (
                "loaded"
                if engine.is_loaded()
                else "not_loaded"
            )
        """

        return bool(
            self._model_loaded
            and self.model is not None
            and self.job_embeddings is not None
            and self.job_metadata is not None
            and self.vectorizer is not None
            and self.job_tfidf_matrix is not None
            and self.interview_embeddings is not None
            and self.interview_metadata is not None
        )

    # ============================================================
    # PUBLIC LOADING STATUS
    # ============================================================

    def load_status(self):
        """
        Return a small diagnostic dictionary.

        Does not trigger model loading.
        """

        return {
            "loaded": self.is_loaded(),
            "loading": bool(self._loading),
            "model_loaded": self.model is not None,
            "job_embeddings_loaded": (
                self.job_embeddings is not None
            ),
            "job_metadata_loaded": (
                self.job_metadata is not None
            ),
            "tfidf_loaded": (
                self.vectorizer is not None
            ),
            "interview_embeddings_loaded": (
                self.interview_embeddings is not None
            ),
            "interview_metadata_loaded": (
                self.interview_metadata is not None
            ),
        }

    # ============================================================
    # CONFIGURE PYTORCH
    # ============================================================

    @staticmethod
    def _configure_torch():
        """
        Restrict PyTorch CPU threading.

        Render Free has limited memory and CPU resources.
        Excessive worker/thread creation can increase memory
        usage significantly.
        """

        try:

            import torch

            torch.set_num_threads(1)

            try:

                torch.set_num_interop_threads(1)

            except RuntimeError:

                # PyTorch may reject this if parallel work has
                # already started. This is harmless.
                pass

        except Exception as exc:

            print(
                "Warning: could not configure PyTorch threads:",
                repr(exc),
            )

    # ============================================================
    # LAZY LOAD ALL ARTIFACTS
    # ============================================================

    def _load_model(self):
        """
        Load the complete prediction engine exactly once.

        This function is called only when an actual AI
        prediction is requested.
        """

        # --------------------------------------------------------
        # Already loaded
        # --------------------------------------------------------

        if self.is_loaded():
            return

        # --------------------------------------------------------
        # Prevent multiple threads from loading the model
        # simultaneously.
        # --------------------------------------------------------

        with self._instance_lock:

            if self.is_loaded():
                return

            self._loading = True

            try:

                print(
                    "=============================================="
                )

                print(
                    "Loading TalentMatch AI prediction artifacts..."
                )

                print(
                    "=============================================="
                )

                # =================================================
                # CPU CONFIGURATION
                # =================================================

                self._configure_torch()

                device = "cpu"

                print(
                    "Configured inference device:",
                    device,
                )

                # =================================================
                # SENTENCE TRANSFORMER
                # =================================================

                print(
                    "Loading embedding model:",
                    EMBEDDING_MODEL,
                )

                # Import only now.
                from sentence_transformers import (
                    SentenceTransformer,
                )

                self.model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device=device,
                )

                self.model.eval()

                print(
                    "Embedding model loaded."
                )

                # =================================================
                # JOB EMBEDDINGS
                # =================================================

                print(
                    "Loading job embeddings..."
                )

                self.job_embeddings = (
                    self._load_embeddings(
                        JOB_EMBEDDINGS
                    )
                )

                print(
                    "Job embeddings shape:",
                    self.job_embeddings.shape,
                )

                print(
                    "Job embeddings dtype:",
                    self.job_embeddings.dtype,
                )

                # =================================================
                # JOB METADATA
                # =================================================

                print(
                    "Loading job metadata..."
                )

                self.job_metadata = pd.read_parquet(
                    JOB_METADATA
                )

                self.job_metadata.columns = [
                    str(column)
                    .strip()
                    .lower()
                    for column
                    in self.job_metadata.columns
                ]

                # -------------------------------------------------
                # Validate job counts
                # -------------------------------------------------

                if len(
                    self.job_embeddings
                ) != len(
                    self.job_metadata
                ):

                    raise RuntimeError(
                        "Job embedding count does not match "
                        "job metadata count."
                    )

                print(
                    "Job metadata rows:",
                    len(self.job_metadata),
                )

                # =================================================
                # TF-IDF VECTORIZER
                # =================================================

                print(
                    "Loading TF-IDF vectorizer..."
                )

                self.vectorizer = joblib.load(
                    TFIDF_MODEL
                )

                # =================================================
                # TF-IDF JOB MATRIX
                # =================================================

                print(
                    "Building TF-IDF job matrix..."
                )

                self.job_tfidf_matrix = (
                    self._build_job_tfidf_matrix()
                )

                print(
                    "TF-IDF matrix shape:",
                    self.job_tfidf_matrix.shape,
                )

                # =================================================
                # INTERVIEW EMBEDDINGS
                # =================================================

                print(
                    "Loading interview embeddings..."
                )

                self.interview_embeddings = (
                    self._load_embeddings(
                        INTERVIEW_EMBEDDINGS
                    )
                )

                print(
                    "Interview embeddings shape:",
                    self.interview_embeddings.shape,
                )

                print(
                    "Interview embeddings dtype:",
                    self.interview_embeddings.dtype,
                )

                # =================================================
                # INTERVIEW METADATA
                # =================================================

                print(
                    "Loading interview metadata..."
                )

                self.interview_metadata = (
                    pd.read_parquet(
                        INTERVIEW_METADATA
                    )
                )

                self.interview_metadata.columns = [
                    str(column)
                    .strip()
                    .lower()
                    for column
                    in self.interview_metadata.columns
                ]

                # -------------------------------------------------
                # Validate interview counts
                # -------------------------------------------------

                if len(
                    self.interview_embeddings
                ) != len(
                    self.interview_metadata
                ):

                    raise RuntimeError(
                        "Interview embedding count does not "
                        "match interview metadata count."
                    )

                print(
                    "Interview metadata rows:",
                    len(self.interview_metadata),
                )

                # =================================================
                # FINAL CLEANUP
                # =================================================

                gc.collect()

                # -------------------------------------------------
                # Mark loaded only after EVERYTHING succeeds.
                # -------------------------------------------------

                self._model_loaded = True

                print(
                    "=============================================="
                )

                print(
                    "TalentMatch AI prediction engine READY."
                )

                print(
                    "=============================================="
                )

            except Exception:

                # -------------------------------------------------
                # If loading fails, do not leave the engine in a
                # falsely-loaded state.
                # -------------------------------------------------

                self._model_loaded = False

                self._cleanup_loaded_artifacts()

                raise

            finally:

                self._loading = False

                gc.collect()

    # ============================================================
    # LOAD EMBEDDINGS
    # ============================================================

    @staticmethod
    def _load_embeddings(path):
        """
        Load embedding arrays while keeping them compact.

        Supports:

            .npy
            .npz

        float16 is preferred because it significantly reduces
        RAM consumption.

        Returns
        -------
        numpy.ndarray
        """

        path = str(path)

        # ========================================================
        # NPY
        # ========================================================

        if path.lower().endswith(
            ".npy"
        ):

            embeddings = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )

            # ----------------------------------------------------
            # Keep float16 if already compact.
            # ----------------------------------------------------

            if embeddings.dtype == np.float16:

                return embeddings

            # ----------------------------------------------------
            # Convert other dtypes to float16.
            # ----------------------------------------------------

            return embeddings.astype(
                np.float16,
                copy=False,
            )

        # ========================================================
        # NPZ
        # ========================================================

        loaded = np.load(
            path,
            allow_pickle=False,
        )

        try:

            if "embeddings" in loaded.files:

                embeddings = loaded[
                    "embeddings"
                ]

            elif len(
                loaded.files
            ) == 1:

                embeddings = loaded[
                    loaded.files[0]
                ]

            else:

                raise RuntimeError(
                    "No 'embeddings' array found in "
                    f"{path}."
                )

            # ----------------------------------------------------
            # Force compact float16 storage.
            # ----------------------------------------------------

            if embeddings.dtype != np.float16:

                embeddings = embeddings.astype(
                    np.float16,
                    copy=False,
                )

            return embeddings

        finally:

            loaded.close()

    # ============================================================
    # BUILD TF-IDF JOB MATRIX
    # ============================================================

    def _build_job_tfidf_matrix(self):
        """
        Build one sparse TF-IDF matrix for all jobs.

        The matrix remains sparse, avoiding unnecessary dense
        memory allocation.
        """

        preferred_columns = (
            "category",
            "description",
            "requirements",
            "benefits",
        )

        available_columns = [
            column
            for column
            in preferred_columns
            if column
            in self.job_metadata.columns
        ]

        # --------------------------------------------------------
        # Fallback to all metadata columns if the preferred
        # columns are unavailable.
        # --------------------------------------------------------

        if not available_columns:

            available_columns = list(
                self.job_metadata.columns
            )

        documents = []

        for row in self.job_metadata[
            available_columns
        ].itertuples(
            index=False,
            name=None,
        ):

            parts = []

            for value in row:

                if value is None:
                    continue

                try:

                    if pd.isna(value):
                        continue

                except Exception:

                    pass

                parts.append(
                    str(value)
                )

            documents.append(
                " ".join(parts)
            )

        matrix = self.vectorizer.transform(
            documents
        )

        del documents

        gc.collect()

        return matrix

    # ============================================================
    # EMBEDDING
    # ============================================================

    def embed(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Convert text into a normalized MiniLM embedding.

        Returns
        -------
        numpy.ndarray
            float32 normalized embedding.
        """

        if text is None:

            raise ValueError(
                "Text cannot be None."
            )

        text = str(
            text
        ).strip()

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        # --------------------------------------------------------
        # Lazy-load model and artifacts.
        # --------------------------------------------------------

        self._load_model()

        # --------------------------------------------------------
        # Generate normalized embedding.
        # --------------------------------------------------------

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=1,
        )

        # --------------------------------------------------------
        # Ensure float32 query.
        # --------------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        return embedding

    # ============================================================
    # TOP-K SELECTION
    # ============================================================

    @staticmethod
    def _top_indices(
        scores: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        """
        Return indices of the highest-scoring items.

        Uses np.argpartition so the complete score array does
        not need to be sorted.
        """

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
            max(
                1,
                int(top_k),
            ),
            total,
        )

        # --------------------------------------------------------
        # If requesting everything, normal sorting is fine.
        # --------------------------------------------------------

        if top_k >= total:

            return np.argsort(
                scores
            )[::-1]

        # --------------------------------------------------------
        # Select only the top-k candidates.
        # --------------------------------------------------------

        indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        # --------------------------------------------------------
        # Sort only the selected candidates.
        # --------------------------------------------------------

        order = np.argsort(
            scores[indices]
        )[::-1]

        return indices[
            order
        ]

    # ============================================================
    # SEMANTIC JOB SEARCH
    # ============================================================

    def semantic_job_search(
        self,
        resume_text: str,
        top_k: int = TOP_K_JOBS,
    ):
        """
        Perform semantic-only job matching.

        Normalized embeddings allow cosine similarity to be
        calculated with a simple dot product.
        """

        self._load_model()

        top_k = min(
            max(
                1,
                int(top_k),
            ),
            int(TOP_K_JOBS),
        )

        query = self.embed(
            resume_text
        )

        # --------------------------------------------------------
        # cosine similarity =
        #
        # normalized_job_embedding · normalized_query
        # --------------------------------------------------------

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

            idx = int(
                idx
            )

            row = self.job_metadata.iloc[
                idx
            ]

            results.append({

                "score": round(
                    float(
                        scores[idx]
                    ),
                    4,
                ),

                "category": self._safe_value(
                    row.get(
                        "category",
                        "",
                    )
                ),

                "description": self._safe_value(
                    row.get(
                        "description",
                        "",
                    )
                ),

                "requirements": self._safe_value(
                    row.get(
                        "requirements",
                        "",
                    )
                ),

                "benefits": self._safe_value(
                    row.get(
                        "benefits",
                        "",
                    )
                ),
            })

        # --------------------------------------------------------
        # Release temporary arrays.
        # --------------------------------------------------------

        del query
        del scores
        del indices

        gc.collect()

        return results

    # ============================================================
    # TF-IDF SCORES
    # ============================================================

    def tfidf_scores(
        self,
        resume_text: str,
    ):
        """
        Calculate lexical similarity between the resume
        and all available jobs.

        The TF-IDF matrix remains sparse.
        """

        self._load_model()

        if resume_text is None:

            raise ValueError(
                "Resume text cannot be None."
            )

        query = self.vectorizer.transform(
            [
                str(
                    resume_text
                )
            ]
        )

        # --------------------------------------------------------
        # Sparse matrix multiplication.
        # --------------------------------------------------------

        scores = (
            self.job_tfidf_matrix
            @ query.T
        ).toarray().ravel()

        del query

        return scores

    # ============================================================
    # HYBRID JOB SEARCH
    # ============================================================

    def hybrid_job_search(
        self,
        resume_text: str,
        top_k: int = TOP_K_JOBS,
    ):
        """
        Combine semantic and TF-IDF similarity.

        final_score =
            semantic_score * SEMANTIC_WEIGHT
            +
            tfidf_score * TFIDF_WEIGHT
        """

        self._load_model()

        top_k = min(
            max(
                1,
                int(top_k),
            ),
            int(TOP_K_JOBS),
        )

        # ========================================================
        # SEMANTIC
        # ========================================================

        query = self.embed(
            resume_text
        )

        semantic_scores = np.dot(
            self.job_embeddings,
            query,
        )

        # ========================================================
        # TF-IDF
        # ========================================================

        lexical_scores = self.tfidf_scores(
            resume_text
        )

        # ========================================================
        # HYBRID SCORE
        # ========================================================

        final_scores = (
            semantic_scores
            * float(
                SEMANTIC_WEIGHT
            )
            +
            lexical_scores
            * float(
                TFIDF_WEIGHT
            )
        )

        # ========================================================
        # TOP-K
        # ========================================================

        indices = self._top_indices(
            final_scores,
            top_k,
        )

        jobs = []

        for idx in indices:

            idx = int(
                idx
            )

            row = self.job_metadata.iloc[
                idx
            ]

            jobs.append({

                "score": round(
                    float(
                        final_scores[idx]
                    ),
                    4,
                ),

                "semantic_score": round(
                    float(
                        semantic_scores[idx]
                    ),
                    4,
                ),

                "tfidf_score": round(
                    float(
                        lexical_scores[idx]
                    ),
                    4,
                ),

                "category": self._safe_value(
                    row.get(
                        "category",
                        "",
                    )
                ),

                "description": self._safe_value(
                    row.get(
                        "description",
                        "",
                    )
                ),

                "requirements": self._safe_value(
                    row.get(
                        "requirements",
                        "",
                    )
                ),

                "benefits": self._safe_value(
                    row.get(
                        "benefits",
                        "",
                    )
                ),
            })

        # ========================================================
        # CLEANUP
        # ========================================================

        del query
        del semantic_scores
        del lexical_scores
        del final_scores
        del indices

        gc.collect()

        return jobs

    # ============================================================
    # INTERVIEW QUESTION RETRIEVAL
    # ============================================================

    def interview_questions(
        self,
        resume_text: str,
        top_k: int = TOP_K_INTERVIEWS,
    ):
        """
        Retrieve interview questions most relevant to the
        candidate's resume.
        """

        self._load_model()

        top_k = min(
            max(
                1,
                int(top_k),
            ),
            int(TOP_K_INTERVIEWS),
        )

        query = self.embed(
            resume_text
        )

        # --------------------------------------------------------
        # Stored interview embeddings remain float16.
        #
        # Query is float32.
        #
        # NumPy computes the similarity without creating a
        # permanent float32 copy of the complete matrix.
        # --------------------------------------------------------

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

            idx = int(
                idx
            )

            row = self.interview_metadata.iloc[
                idx
            ]

            questions.append({

                "score": round(
                    float(
                        scores[idx]
                    ),
                    4,
                ),

                "question": self._safe_value(
                    row.get(
                        "question",
                        "",
                    )
                ),

                "answer": self._safe_value(
                    row.get(
                        "answer",
                        "",
                    )
                ),

                "role": self._safe_value(
                    row.get(
                        "role",
                        "",
                    )
                ),

                "category": self._safe_value(
                    row.get(
                        "category",
                        "",
                    )
                ),

                "difficulty": self._safe_value(
                    row.get(
                        "difficulty",
                        "",
                    )
                ),

                "experience": self._safe_value(
                    row.get(
                        "experience",
                        "",
                    )
                ),
            })

        # ========================================================
        # CLEANUP
        # ========================================================

        del query
        del scores
        del indices

        gc.collect()

        return questions

    # ============================================================
    # SAFE METADATA VALUE
    # ============================================================

    @staticmethod
    def _safe_value(
        value,
    ):
        """
        Convert None/NaN metadata values into clean strings.

        Prevents backend objects such as NaN from being returned
        directly to the frontend.
        """

        if value is None:

            return ""

        try:

            if pd.isna(value):

                return ""

        except Exception:

            pass

        return str(
            value
        )

    # ============================================================
    # CLEANUP
    # ============================================================

    def _cleanup_loaded_artifacts(self):
        """
        Release partially loaded artifacts if initialization
        fails.

        This prevents a failed initialization from leaving
        large objects in RAM.
        """

        self.model = None

        self.job_embeddings = None
        self.job_metadata = None
        self.job_tfidf_matrix = None

        self.interview_embeddings = None
        self.interview_metadata = None

        self.vectorizer = None

        self._model_loaded = False

        gc.collect()

    # ============================================================
    # EXPLICIT SHUTDOWN
    # ============================================================

    def shutdown(self):
        """
        Release loaded AI resources.

        Used by FastAPI shutdown handling.
        """

        with self._instance_lock:

            print(
                "Releasing TalentMatch AI prediction engine..."
            )

            self._cleanup_loaded_artifacts()

            print(
                "TalentMatch AI prediction engine released."
            )


# ================================================================
# GLOBAL SINGLETON
# ================================================================

"""
IMPORTANT
---------

This creates ONLY the lightweight PredictionEngine object.

It does NOT load:

    • SentenceTransformer
    • PyTorch
    • Job embeddings
    • Interview embeddings
    • TF-IDF matrix
    • Parquet metadata

The heavy model/artifacts are loaded by:

    engine.embed(...)
    engine.semantic_job_search(...)
    engine.tfidf_scores(...)
    engine.hybrid_job_search(...)
    engine.interview_questions(...)

This is what allows app.py to safely import:

    from predict import engine

without immediately consuming the majority of the
512 MB Render memory limit.
"""

engine = PredictionEngine()
