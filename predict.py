"""
================================================================
TalentMatch AI
Production Prediction Engine
================================================================

Responsibilities
----------------
• Lazy-load the MiniLM embedding model
• Semantic job matching
• TF-IDF job matching
• Hybrid job ranking
• Interview question retrieval

Deployment
----------
Render Free
512 MB RAM
CPU inference

Memory strategy
---------------
• Lazy model loading
• One prediction engine
• One CPU worker
• CPU-only inference
• PyTorch thread limits
• float16 stored embeddings
• float32 query embeddings
• No permanent resume storage
• No skills database loaded here
• No unnecessary pandas conversions
• Efficient top-k selection
• Explicit temporary-object cleanup
• Small metadata datasets
================================================================
"""

# ================================================================
# IMPORTANT: Set CPU/thread environment BEFORE importing torch
# ================================================================

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ================================================================
# Imports
# ================================================================

import gc
import json
import threading

import joblib
import numpy as np
import pandas as pd

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
# Prediction Engine
# ================================================================


class PredictionEngine:
    """
    Lightweight singleton prediction engine.

    IMPORTANT:
    The SentenceTransformer model is NOT loaded during import.

    It is loaded only when an actual prediction is requested.

    This substantially reduces Render startup memory usage.
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

                    cls._instance = super().__new__(cls)

                    cls._instance._initialized = False
                    cls._instance._model_loaded = False

        return cls._instance

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(self):

        if self._initialized:
            return

        # --------------------------------------------------------
        # Lightweight attributes only.
        #
        # DO NOT load SentenceTransformer here.
        # --------------------------------------------------------

        self.model = None

        self.job_embeddings = None
        self.job_metadata = None
        self.job_tfidf_matrix = None

        self.interview_embeddings = None
        self.interview_metadata = None

        self.vectorizer = None

        self._model_loaded = False

        self._initialized = True

        print("==============================================")
        print("TalentMatch AI prediction engine initialized.")
        print("Model loading deferred until first prediction.")
        print("==============================================")

    # ============================================================
    # Configure PyTorch
    # ============================================================

    @staticmethod
    def _configure_torch():

        """
        Restrict PyTorch CPU threading.

        This is important on Render Free because excessive
        thread creation can increase memory consumption.
        """

        try:

            import torch

            torch.set_num_threads(1)

            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch may reject this if parallel work
                # has already started. It is safe to continue.
                pass

        except Exception as exc:

            print(
                "Warning: could not configure PyTorch threads:",
                repr(exc)
            )

    # ============================================================
    # Lazy Model Loading
    # ============================================================

    def _load_model(self):

        """
        Load all prediction artifacts exactly once.

        This method is called only when an actual prediction
        is required.
        """

        if self._model_loaded:
            return

        with self._instance_lock:

            if self._model_loaded:
                return

            print("==============================================")
            print("Loading TalentMatch AI prediction artifacts...")
            print("==============================================")

            # ----------------------------------------------------
            # Configure CPU
            # ----------------------------------------------------

            self._configure_torch()

            # ----------------------------------------------------
            # Force CPU
            # ----------------------------------------------------

            device = "cpu"

            print(
                "Configured inference device:",
                device
            )

            # ====================================================
            # Sentence Transformer
            # ====================================================

            print(
                "Loading embedding model:",
                EMBEDDING_MODEL
            )

            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device
            )

            self.model.eval()

            print("Embedding model loaded.")

            # ====================================================
            # Job Embeddings
            # ====================================================

            print("Loading job embeddings...")

            self.job_embeddings = self._load_embeddings(
                JOB_EMBEDDINGS
            )

            print(
                "Job embeddings shape:",
                self.job_embeddings.shape
            )

            # ====================================================
            # Job Metadata
            # ====================================================

            print("Loading job metadata...")

            self.job_metadata = pd.read_parquet(
                JOB_METADATA
            )

            self.job_metadata.columns = [
                str(column).strip().lower()
                for column in self.job_metadata.columns
            ]

            # ----------------------------------------------------
            # Make sure embedding count and metadata count match.
            # ----------------------------------------------------

            if len(self.job_embeddings) != len(
                self.job_metadata
            ):

                raise RuntimeError(
                    "Job embedding count does not match "
                    "job metadata count."
                )

            print(
                "Job metadata rows:",
                len(self.job_metadata)
            )

            # ====================================================
            # TF-IDF Vectorizer
            # ====================================================

            print("Loading TF-IDF vectorizer...")

            self.vectorizer = joblib.load(
                TFIDF_MODEL
            )

            # ====================================================
            # Build TF-IDF Matrix
            # ====================================================

            print("Building TF-IDF job matrix...")

            self.job_tfidf_matrix = (
                self._build_job_tfidf_matrix()
            )

            # ====================================================
            # Interview Embeddings
            # ====================================================

            print("Loading interview embeddings...")

            self.interview_embeddings = (
                self._load_embeddings(
                    INTERVIEW_EMBEDDINGS
                )
            )

            print(
                "Interview embeddings shape:",
                self.interview_embeddings.shape
            )

            # ====================================================
            # Interview Metadata
            # ====================================================

            print("Loading interview metadata...")

            self.interview_metadata = pd.read_parquet(
                INTERVIEW_METADATA
            )

            self.interview_metadata.columns = [
                str(column).strip().lower()
                for column in self.interview_metadata.columns
            ]

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
                len(self.interview_metadata)
            )

            # ====================================================
            # Cleanup
            # ====================================================

            gc.collect()

            self._model_loaded = True

            print("==============================================")
            print("TalentMatch AI prediction engine ready.")
            print("==============================================")

    # ============================================================
    # Load Embeddings Efficiently
    # ============================================================

    @staticmethod
    def _load_embeddings(path):
        """
        Load embeddings while preserving float16 storage.

        Supports both:

            .npy

        and:

            .npz

        formats.

        Stored embeddings remain float16 to reduce RAM.
        """

        path = str(path)

        # --------------------------------------------------------
        # NPY
        # --------------------------------------------------------

        if path.lower().endswith(".npy"):

            embeddings = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False
            )

            return embeddings

        # --------------------------------------------------------
        # NPZ
        # --------------------------------------------------------

        loaded = np.load(
            path,
            allow_pickle=False
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
                    f"No 'embeddings' array found in {path}."
                )

            # ----------------------------------------------------
            # Ensure compact dtype.
            # ----------------------------------------------------

            if embeddings.dtype != np.float16:

                embeddings = embeddings.astype(
                    np.float16,
                    copy=False
                )

            # ----------------------------------------------------
            # Make a compact contiguous array only when needed.
            # ----------------------------------------------------

            return embeddings

        finally:

            loaded.close()

    # ============================================================
    # Build TF-IDF Matrix
    # ============================================================

    def _build_job_tfidf_matrix(self):

        preferred_columns = (
            "category",
            "description",
            "requirements",
            "benefits",
        )

        available = [
            column
            for column in preferred_columns
            if column in self.job_metadata.columns
        ]

        if not available:

            available = list(
                self.job_metadata.columns
            )

        # --------------------------------------------------------
        # There are only 325 jobs.
        #
        # Building the documents once at startup is inexpensive.
        # --------------------------------------------------------

        documents = []

        for row in self.job_metadata[
            available
        ].itertuples(
            index=False,
            name=None
        ):

            parts = []

            for value in row:

                if value is not None:

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
    # Public Readiness
    # ============================================================

    def is_loaded(self):

        return bool(
            self._model_loaded
            and self.model is not None
        )

    # ============================================================
    # Embedding
    # ============================================================

    def embed(
        self,
        text: str
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

        text = str(text).strip()

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        # --------------------------------------------------------
        # Ensure model is loaded.
        # --------------------------------------------------------

        self._load_model()

        # --------------------------------------------------------
        # CPU inference.
        # SentenceTransformer handles inference_mode/no_grad
        # internally for encode().
        # --------------------------------------------------------

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=1
        )

        # --------------------------------------------------------
        # Ensure exactly float32.
        # --------------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        return embedding

    # ============================================================
    # Efficient Top-K
    # ============================================================

    @staticmethod
    def _top_indices(
        scores: np.ndarray,
        top_k: int
    ) -> np.ndarray:

        """
        Return indices of highest-scoring items.

        Uses argpartition instead of sorting the complete array.
        """

        if scores is None:

            return np.empty(
                0,
                dtype=np.int64
            )

        total = int(
            scores.shape[0]
        )

        if total == 0:

            return np.empty(
                0,
                dtype=np.int64
            )

        top_k = min(
            max(
                1,
                int(top_k)
            ),
            total
        )

        # --------------------------------------------------------
        # Small dataset optimization.
        # --------------------------------------------------------

        if top_k >= total:

            return np.argsort(
                scores
            )[::-1]

        # --------------------------------------------------------
        # Partial selection.
        # --------------------------------------------------------

        indices = np.argpartition(
            scores,
            -top_k
        )[-top_k:]

        # --------------------------------------------------------
        # Sort only selected elements.
        # --------------------------------------------------------

        order = np.argsort(
            scores[indices]
        )[::-1]

        return indices[order]

    # ============================================================
    # Semantic Job Search
    # ============================================================

    def semantic_job_search(
        self,
        resume_text: str,
        top_k: int = TOP_K_JOBS
    ):

        """
        Perform semantic-only job matching.
        """

        self._load_model()

        top_k = min(
            max(
                1,
                int(top_k)
            ),
            TOP_K_JOBS
        )

        query = self.embed(
            resume_text
        )

        # --------------------------------------------------------
        # Stored embeddings are normalized.
        #
        # cosine similarity = dot product
        # --------------------------------------------------------

        scores = np.dot(
            self.job_embeddings,
            query
        )

        indices = self._top_indices(
            scores,
            top_k
        )

        results = []

        for idx in indices:

            idx = int(idx)

            row = self.job_metadata.iloc[
                idx
            ]

            results.append({

                "score": round(
                    float(
                        scores[idx]
                    ),
                    4
                ),

                "category": self._safe_value(
                    row.get(
                        "category",
                        ""
                    )
                ),

                "description": self._safe_value(
                    row.get(
                        "description",
                        ""
                    )
                ),

                "requirements": self._safe_value(
                    row.get(
                        "requirements",
                        ""
                    )
                ),

                "benefits": self._safe_value(
                    row.get(
                        "benefits",
                        ""
                    )
                ),
            })

        del query
        del scores

        gc.collect()

        return results

    # ============================================================
    # TF-IDF Scores
    # ============================================================

    def tfidf_scores(
        self,
        resume_text: str
    ):

        """
        Calculate lexical similarity against all jobs.
        """

        self._load_model()

        query = self.vectorizer.transform(
            [str(resume_text)]
        )

        scores = (
            self.job_tfidf_matrix
            @ query.T
        ).toarray().ravel()

        del query

        return scores

    # ============================================================
    # Hybrid Job Search
    # ============================================================

    def hybrid_job_search(
        self,
        resume_text: str,
        top_k: int = TOP_K_JOBS
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
                int(top_k)
            ),
            TOP_K_JOBS
        )

        # ========================================================
        # Semantic similarity
        # ========================================================

        query = self.embed(
            resume_text
        )

        semantic_scores = np.dot(
            self.job_embeddings,
            query
        )

        # ========================================================
        # TF-IDF similarity
        # ========================================================

        lexical_scores = self.tfidf_scores(
            resume_text
        )

        # ========================================================
        # Hybrid score
        # ========================================================

        final_scores = (
            semantic_scores
            * float(SEMANTIC_WEIGHT)
            +
            lexical_scores
            * float(TFIDF_WEIGHT)
        )

        # ========================================================
        # Top-K
        # ========================================================

        indices = self._top_indices(
            final_scores,
            top_k
        )

        jobs = []

        for idx in indices:

            idx = int(idx)

            row = self.job_metadata.iloc[
                idx
            ]

            jobs.append({

                "score": round(
                    float(
                        final_scores[idx]
                    ),
                    4
                ),

                "semantic_score": round(
                    float(
                        semantic_scores[idx]
                    ),
                    4
                ),

                "tfidf_score": round(
                    float(
                        lexical_scores[idx]
                    ),
                    4
                ),

                "category": self._safe_value(
                    row.get(
                        "category",
                        ""
                    )
                ),

                "description": self._safe_value(
                    row.get(
                        "description",
                        ""
                    )
                ),

                "requirements": self._safe_value(
                    row.get(
                        "requirements",
                        ""
                    )
                ),

                "benefits": self._safe_value(
                    row.get(
                        "benefits",
                        ""
                    )
                ),
            })

        # ========================================================
        # Cleanup
        # ========================================================

        del query
        del semantic_scores
        del lexical_scores
        del final_scores

        gc.collect()

        return jobs

    # ============================================================
    # Interview Question Retrieval
    # ============================================================

    def interview_questions(
        self,
        resume_text: str,
        top_k: int = TOP_K_INTERVIEWS
    ):

        """
        Retrieve interview questions most relevant
        to the candidate's resume.
        """

        self._load_model()

        top_k = min(
            max(
                1,
                int(top_k)
            ),
            TOP_K_INTERVIEWS
        )

        query = self.embed(
            resume_text
        )

        # --------------------------------------------------------
        # Interview embeddings:
        #
        # 14,817 × 384 float16
        #
        # Only the temporary score vector is float32.
        # --------------------------------------------------------

        scores = np.dot(
            self.interview_embeddings,
            query
        )

        indices = self._top_indices(
            scores,
            top_k
        )

        questions = []

        for idx in indices:

            idx = int(idx)

            row = self.interview_metadata.iloc[
                idx
            ]

            questions.append({

                "score": round(
                    float(
                        scores[idx]
                    ),
                    4
                ),

                "question": self._safe_value(
                    row.get(
                        "question",
                        ""
                    )
                ),

                "answer": self._safe_value(
                    row.get(
                        "answer",
                        ""
                    )
                ),

                "role": self._safe_value(
                    row.get(
                        "role",
                        ""
                    )
                ),

                "category": self._safe_value(
                    row.get(
                        "category",
                        ""
                    )
                ),

                "difficulty": self._safe_value(
                    row.get(
                        "difficulty",
                        ""
                    )
                ),

                "experience": self._safe_value(
                    row.get(
                        "experience",
                        ""
                    )
                ),
            })

        del query
        del scores

        gc.collect()

        return questions

    # ============================================================
    # Safe Metadata Conversion
    # ============================================================

    @staticmethod
    def _safe_value(value):

        """
        Convert pandas NaN/None into clean strings.
        """

        if value is None:

            return ""

        try:

            if pd.isna(value):

                return ""

        except Exception:

            pass

        return str(value)


# ================================================================
# Global Singleton
# ================================================================

# IMPORTANT:
#
# This creates only the lightweight engine object.
#
# It DOES NOT load SentenceTransformer here.
#
# The model is loaded on the first actual prediction request.
# ================================================================

engine = PredictionEngine()
