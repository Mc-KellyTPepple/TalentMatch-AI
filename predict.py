"""
===============================================================
TalentMatch AI

Production Prediction Engine

Responsibilities:

    • Load all trained artifacts once
    • Semantic Job Matching
    • TF-IDF Job Matching
    • Hybrid Job Ranking
    • Interview Question Retrieval

Designed for:
    Render Free
    512 MB RAM

The model is loaded once through a singleton engine.
===============================================================
"""

# ============================================================
# Imports
# ============================================================

import gzip
import json

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

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
# Singleton Prediction Engine
# ============================================================

class PredictionEngine:

    """
    Loads the AI model and trained artifacts once.

    This is important for Render Free because repeatedly
    loading the SentenceTransformer model would consume
    unnecessary memory and CPU.
    """

    _instance = None

    # ========================================================

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            cls._instance._loaded = False

        return cls._instance

    # ========================================================

    def __init__(self):

        if self._loaded:

            return

        print(
            "=================================================="
        )

        print(
            "Loading TalentMatch AI prediction engine..."
        )

        print(
            "=================================================="
        )

        # ====================================================
        # Sentence Transformer
        # ====================================================

        print(
            "Loading embedding model..."
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )

        # ====================================================
        # Job Embeddings
        # ====================================================

        print(
            "Loading job embeddings..."
        )

        self.job_embeddings = np.load(
            JOB_EMBEDDINGS
        )["embeddings"].astype(
            np.float32
        )

        # ====================================================
        # Job Metadata
        # ====================================================

        print(
            "Loading job metadata..."
        )

        self.job_metadata = pd.read_parquet(
            JOB_METADATA
        )

        # ----------------------------------------------------
        # IMPORTANT COMPATIBILITY FIX
        #
        # Normalize all metadata column names to lowercase.
        #
        # This prevents problems if the training dataset used:
        #
        # Category
        # Description
        # Requirements
        # Benefits
        #
        # or:
        #
        # category
        # description
        # requirements
        # benefits
        # ----------------------------------------------------

        self.job_metadata.columns = [
            str(column)
            .strip()
            .lower()
            for column in self.job_metadata.columns
        ]

        print(
            "Job metadata columns:",
            list(self.job_metadata.columns)
        )

        # ====================================================
        # Interview Embeddings
        # ====================================================

        print(
            "Loading interview embeddings..."
        )

        self.interview_embeddings = np.load(
            INTERVIEW_EMBEDDINGS
        )["embeddings"].astype(
            np.float32
        )

        # ====================================================
        # Interview Metadata
        # ====================================================

        print(
            "Loading interview metadata..."
        )

        self.interview_metadata = pd.read_parquet(
            INTERVIEW_METADATA
        )

        # Normalize interview metadata column names too.
        self.interview_metadata.columns = [
            str(column)
            .strip()
            .lower()
            for column in self.interview_metadata.columns
        ]

        # ====================================================
        # TF-IDF Model
        # ====================================================

        print(
            "Loading TF-IDF model..."
        )

        self.vectorizer = joblib.load(
            TFIDF_MODEL
        )

        # ====================================================
        # Build TF-IDF Job Matrix
        # ====================================================

        print(
            "Building TF-IDF job matrix..."
        )

        # Use only available text columns.
        #
        # This is safer than assuming every dataset contains
        # exactly the same fields.

        text_columns = [
            column
            for column in [
                "category",
                "description",
                "requirements",
                "benefits",
            ]
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

            # Fallback:
            # combine every metadata column.

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

        self.job_tfidf_matrix = (
            self.vectorizer.transform(
                job_documents
            )
        )

        # Free temporary memory.

        del job_documents

        # ====================================================
        # Skills
        # ====================================================

        print(
            "Loading skill database..."
        )

        with gzip.open(
            SKILLS,
            "rt",
            encoding="utf-8"
        ) as f:

            self.skills = set(
                json.load(f)
            )

        # ====================================================
        # Final State
        # ====================================================

        self._loaded = True

        print(
            "=================================================="
        )

        print(
            "TalentMatch AI prediction engine ready."
        )

        print(
            "=================================================="
        )

    # ========================================================
    # Embedding
    # ========================================================

    def embed(
        self,
        text
    ):
        """
        Convert resume text into a normalized
        MiniLM embedding.
        """

        embedding = self.model.encode(

            text,

            normalize_embeddings=True,

            convert_to_numpy=True,

            show_progress_bar=False

        )

        return embedding.astype(
            np.float32
        )

    # ========================================================
    # Semantic Job Search
    # ========================================================

    def semantic_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS
    ):
        """
        Perform semantic job matching.
        """

        query = self.embed(
            resume_text
        )

        scores = cosine_similarity(

            query.reshape(
                1,
                -1
            ),

            self.job_embeddings

        )[0]

        indices = np.argsort(
            scores
        )[::-1][
            :top_k
        ]

        results = []

        for idx in indices:

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

                # FIXED:
                # Metadata columns are now lowercase.

                "category": row.get(
                    "category",
                    ""
                ),

                "description": row.get(
                    "description",
                    ""
                ),

                "requirements": row.get(
                    "requirements",
                    ""
                ),

                "benefits": row.get(
                    "benefits",
                    ""
                )

            })

        return results

    # ========================================================
    # TF-IDF Scores
    # ========================================================

    def tfidf_scores(
        self,
        resume_text
    ):
        """
        Compute lexical similarity between the resume
        and all job descriptions.
        """

        query = self.vectorizer.transform(
            [resume_text]
        )

        scores = cosine_similarity(

            query,

            self.job_tfidf_matrix

        )[0]

        return scores

    # ========================================================
    # Hybrid Job Search
    # ========================================================

    def hybrid_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS
    ):
        """
        Combine:

            Semantic similarity
            +
            TF-IDF keyword similarity

        into a single hybrid score.
        """

        # ----------------------------------------------------
        # Semantic similarity
        # ----------------------------------------------------

        semantic = self.embed(
            resume_text
        )

        semantic_scores = cosine_similarity(

            semantic.reshape(
                1,
                -1
            ),

            self.job_embeddings

        )[0]

        # ----------------------------------------------------
        # TF-IDF similarity
        # ----------------------------------------------------

        lexical_scores = self.tfidf_scores(
            resume_text
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        final_scores = (

            semantic_scores
            * SEMANTIC_WEIGHT

            +

            lexical_scores
            * TFIDF_WEIGHT

        )

        # ----------------------------------------------------
        # Top jobs
        # ----------------------------------------------------

        indices = np.argsort(
            final_scores
        )[::-1][
            :top_k
        ]

        jobs = []

        for idx in indices:

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

                # FIXED METADATA ACCESS

                "category": row.get(
                    "category",
                    ""
                ),

                "description": row.get(
                    "description",
                    ""
                ),

                "requirements": row.get(
                    "requirements",
                    ""
                ),

                "benefits": row.get(
                    "benefits",
                    ""
                )

            })

        return jobs

    # ========================================================
    # Interview Question Retrieval
    # ========================================================

    def interview_questions(
        self,
        resume_text,
        top_k=TOP_K_INTERVIEWS
    ):
        """
        Retrieve interview questions most relevant
        to the candidate's resume.
        """

        query = self.embed(
            resume_text
        )

        scores = cosine_similarity(

            query.reshape(
                1,
                -1
            ),

            self.interview_embeddings

        )[0]

        indices = np.argsort(
            scores
        )[::-1][
            :top_k
        ]

        questions = []

        for idx in indices:

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

                "question": row.get(
                    "question",
                    ""
                ),

                "answer": row.get(
                    "answer",
                    ""
                ),

                "role": row.get(
                    "role",
                    ""
                ),

                "category": row.get(
                    "category",
                    ""
                ),

                "difficulty": row.get(
                    "difficulty",
                    ""
                ),

                "experience": row.get(
                    "experience",
                    ""
                )

            })

        return questions


# ============================================================
# Global Prediction Engine
# ============================================================

engine = PredictionEngine()
