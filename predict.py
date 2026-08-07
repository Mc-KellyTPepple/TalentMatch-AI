"""
===============================================================
TalentMatch AI
Production Prediction Engine
===============================================================

Responsibilities:

    • Load trained artifacts once
    • Semantic job matching
    • TF-IDF job matching
    • Hybrid job ranking
    • Interview question retrieval

Deployment target:

    Render Free
    512 MB RAM
    CPU inference

Memory strategy:

    • Singleton model loading
    • CPU-only inference
    • float32 runtime embeddings
    • Keep stored embeddings compact
    • Avoid unnecessary array copies
    • Avoid pandas conversions during inference
    • Use NumPy dot products for normalized embeddings
    • Efficient top-k selection with argpartition
    • Explicit cleanup of temporary arrays
    • No candidate resume storage
===============================================================
"""

# ============================================================
# Imports
# ============================================================

import gc
import gzip
import json

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    JOB_EMBEDDINGS,
    JOB_METADATA,
    INTERVIEW_EMBEDDINGS,
    INTERVIEW_METADATA,
    TFIDF_MODEL,
    SKILLS,
    DEVICE,
    SEMANTIC_WEIGHT,
    TFIDF_WEIGHT,
    TOP_K_JOBS,
    TOP_K_INTERVIEWS,
)


# ============================================================
# Prediction Engine
# ============================================================

class PredictionEngine:
    """
    Production AI inference engine.

    The singleton pattern ensures that the embedding model
    and trained artifacts are loaded only once per process.

    This is important for Render Free because repeatedly
    loading SentenceTransformer would waste significant RAM.
    """

    _instance = None

    # ========================================================
    # Singleton
    # ========================================================

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            cls._instance._loaded = False

        return cls._instance

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self):

        if self._loaded:
            return

        print("==============================================")
        print("Starting TalentMatch AI prediction engine...")
        print("==============================================")

        # ----------------------------------------------------
        # Force CPU inference
        # ----------------------------------------------------
        #
        # Render Free does not provide a GPU.
        #
        # DEVICE should normally be "cpu" in config.py.
        #

        print("Inference device:", DEVICE)

        # ====================================================
        # Sentence Transformer
        # ====================================================

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )

        # Disable training behavior.

        self.model.eval()

        # ====================================================
        # Job Embeddings
        # ====================================================

        print("Loading job embeddings...")

        loaded_jobs = np.load(
            JOB_EMBEDDINGS,
            allow_pickle=False
        )

        job_embeddings = loaded_jobs["embeddings"]

        # ----------------------------------------------------
        # Runtime representation
        # ----------------------------------------------------
        #
        # Your stored embeddings are float16.
        #
        # That is excellent for disk/RAM storage.
        #
        # For CPU similarity calculations we convert only
        # the small job matrix to float32.
        #

        if job_embeddings.dtype != np.float32:

            job_embeddings = job_embeddings.astype(
                np.float32
            )

        self.job_embeddings = job_embeddings

        del loaded_jobs
        del job_embeddings

        # ====================================================
        # Job Metadata
        # ====================================================

        print("Loading job metadata...")

        self.job_metadata = pd.read_parquet(
            JOB_METADATA
        )

        # Normalize metadata column names.

        self.job_metadata.columns = [
            str(column)
            .strip()
            .lower()
            for column in self.job_metadata.columns
        ]

        # ====================================================
        # Interview Embeddings
        # ====================================================

        print("Loading interview embeddings...")

        loaded_interviews = np.load(
            INTERVIEW_EMBEDDINGS,
            allow_pickle=False
        )

        interview_embeddings = (
            loaded_interviews["embeddings"]
        )

        # ----------------------------------------------------
        # IMPORTANT MEMORY DECISION
        # ----------------------------------------------------
        #
        # Your interview matrix is:
        #
        #   14,817 × 384
        #
        # Stored as float16:
        #
        #   ~10.85 MB
        #
        # Keeping it as float16 saves RAM.
        #
        # NumPy dot products can operate using the compact
        # representation without creating a permanent
        # float32 copy.
        #

        self.interview_embeddings = interview_embeddings

        del loaded_interviews
        del interview_embeddings

        # ====================================================
        # Interview Metadata
        # ====================================================

        print("Loading interview metadata...")

        self.interview_metadata = pd.read_parquet(
            INTERVIEW_METADATA
        )

        # Normalize column names.

        self.interview_metadata.columns = [
            str(column)
            .strip()
            .lower()
            for column in self.interview_metadata.columns
        ]

        # ====================================================
        # TF-IDF Vectorizer
        # ====================================================

        print("Loading TF-IDF vectorizer...")

        self.vectorizer = joblib.load(
            TFIDF_MODEL
        )

        # ====================================================
        # Build TF-IDF Job Matrix
        # ====================================================

        print("Building TF-IDF job index...")

        preferred_columns = [
            "category",
            "description",
            "requirements",
            "benefits",
        ]

        text_columns = [
            column
            for column in preferred_columns
            if column in self.job_metadata.columns
        ]

        if text_columns:

            job_documents = (
                self.job_metadata[
                    text_columns
                ]
                .fillna("")
                .astype(str)
                .agg(
                    " ".join,
                    axis=1
                )
                .tolist()
            )

        else:

            job_documents = (
                self.job_metadata
                .fillna("")
                .astype(str)
                .agg(
                    " ".join,
                    axis=1
                )
                .tolist()
            )

        # Build sparse TF-IDF matrix.

        self.job_tfidf_matrix = (
            self.vectorizer.transform(
                job_documents
            )
        )

        # Temporary list no longer required.

        del job_documents

        # ====================================================
        # Skills
        # ====================================================

        print("Loading skill database...")

        try:

            with gzip.open(
                SKILLS,
                "rt",
                encoding="utf-8"
            ) as f:

                self.skills = set(
                    json.load(f)
                )

        except Exception as exc:

            # Skills are supplementary to matching.
            #
            # Do not prevent the entire API from starting
            # if the optional skill database has an issue.

            print(
                "Warning: skill database could not be loaded:",
                exc
            )

            self.skills = set()

        # ====================================================
        # Cleanup
        # ====================================================

        gc.collect()

        # ====================================================
        # Ready
        # ====================================================

        self._loaded = True

        print("==============================================")
        print("TalentMatch AI prediction engine ready.")
        print("==============================================")


    # ========================================================
    # Embedding
    # ========================================================

    def embed(
        self,
        text: str
    ) -> np.ndarray:
        """
        Convert text into a normalized MiniLM embedding.

        Returns:
            float32 NumPy array
        """

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return np.asarray(
            embedding,
            dtype=np.float32
        )


    # ========================================================
    # Efficient Top-K
    # ========================================================

    @staticmethod
    def _top_indices(
        scores: np.ndarray,
        top_k: int
    ) -> np.ndarray:
        """
        Return indices of the top-k scores.

        Uses argpartition rather than sorting the entire
        array. This reduces unnecessary CPU work.
        """

        total = len(scores)

        if total == 0:

            return np.empty(
                0,
                dtype=np.int64
            )

        top_k = min(
            max(1, int(top_k)),
            total
        )

        # If requesting nearly everything, full sorting
        # is acceptable and simpler.

        if top_k >= total:

            return np.argsort(
                scores
            )[::-1]

        indices = np.argpartition(
            scores,
            -top_k
        )[-top_k:]

        indices = indices[
            np.argsort(
                scores[indices]
            )[::-1]
        ]

        return indices


    # ========================================================
    # Semantic Job Search
    # ========================================================

    def semantic_job_search(
        self,
        resume_text: str,
        top_k: int = TOP_K_JOBS
    ):
        """
        Perform semantic job matching.
        """

        top_k = min(
            max(1, int(top_k)),
            TOP_K_JOBS
        )

        query = self.embed(
            resume_text
        )

        # ----------------------------------------------------
        # Embeddings are normalized.
        #
        # Therefore:
        #
        # cosine_similarity(A, B)
        #
        # is equivalent to:
        #
        # A @ B.T
        #
        # This avoids sklearn temporary matrices.
        # ----------------------------------------------------

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
                    float(scores[idx]),
                    4
                ),

                "category": str(
                    row.get(
                        "category",
                        ""
                    )
                ),

                "description": str(
                    row.get(
                        "description",
                        ""
                    )
                ),

                "requirements": str(
                    row.get(
                        "requirements",
                        ""
                    )
                ),

                "benefits": str(
                    row.get(
                        "benefits",
                        ""
                    )
                )
            })

        del query
        del scores

        gc.collect()

        return results


    # ========================================================
    # TF-IDF Scores
    # ========================================================

    def tfidf_scores(
        self,
        resume_text: str
    ):
        """
        Calculate lexical similarity between the resume
        and all available jobs.
        """

        query = self.vectorizer.transform(
            [resume_text]
        )

        # ----------------------------------------------------
        # TF-IDF vectors are normally L2 normalized by the
        # sklearn TfidfVectorizer.
        #
        # Therefore dot product is sufficient for cosine
        # similarity and avoids sklearn overhead.
        # ----------------------------------------------------

        scores = (
            self.job_tfidf_matrix @ query.T
        ).toarray().ravel()

        del query

        return scores


    # ========================================================
    # Hybrid Job Search
    # ========================================================

    def hybrid_job_search(
        self,
        resume_text: str,
        top_k: int = TOP_K_JOBS
    ):
        """
        Combine semantic and keyword similarity.

        Final score:

            semantic × SEMANTIC_WEIGHT
            +
            TF-IDF × TFIDF_WEIGHT
        """

        top_k = min(
            max(1, int(top_k)),
            TOP_K_JOBS
        )

        # ====================================================
        # Semantic Similarity
        # ====================================================

        semantic = self.embed(
            resume_text
        )

        semantic_scores = np.dot(
            self.job_embeddings,
            semantic
        )

        # ====================================================
        # TF-IDF Similarity
        # ====================================================

        lexical_scores = self.tfidf_scores(
            resume_text
        )

        # ====================================================
        # Hybrid Score
        # ====================================================

        final_scores = (
            semantic_scores
            * SEMANTIC_WEIGHT
            +
            lexical_scores
            * TFIDF_WEIGHT
        )

        # ====================================================
        # Top-K
        # ====================================================

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

                "category": str(
                    row.get(
                        "category",
                        ""
                    )
                ),

                "description": str(
                    row.get(
                        "description",
                        ""
                    )
                ),

                "requirements": str(
                    row.get(
                        "requirements",
                        ""
                    )
                ),

                "benefits": str(
                    row.get(
                        "benefits",
                        ""
                    )
                )
            })

        # ====================================================
        # Cleanup
        # ====================================================

        del semantic
        del semantic_scores
        del lexical_scores
        del final_scores

        gc.collect()

        return jobs


    # ========================================================
    # Interview Question Retrieval
    # ========================================================

    def interview_questions(
        self,
        resume_text: str,
        top_k: int = TOP_K_INTERVIEWS
    ):
        """
        Retrieve interview questions most relevant
        to the candidate's resume.
        """

        top_k = min(
            max(1, int(top_k)),
            TOP_K_INTERVIEWS
        )

        query = self.embed(
            resume_text
        )

        # ----------------------------------------------------
        # Stored interview vectors are normalized.
        #
        # Dot product = cosine similarity.
        #
        # Keeping them in float16 avoids an unnecessary
        # ~22 MB float32 permanent copy.
        # ----------------------------------------------------

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

                "question": str(
                    row.get(
                        "question",
                        ""
                    )
                ),

                "answer": str(
                    row.get(
                        "answer",
                        ""
                    )
                ),

                "role": str(
                    row.get(
                        "role",
                        ""
                    )
                ),

                "category": str(
                    row.get(
                        "category",
                        ""
                    )
                ),

                "difficulty": str(
                    row.get(
                        "difficulty",
                        ""
                    )
                ),

                "experience": str(
                    row.get(
                        "experience",
                        ""
                    )
                )
            })

        del query
        del scores

        gc.collect()

        return questions


# ============================================================
# Global Singleton
# ============================================================

engine = PredictionEngine()
