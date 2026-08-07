"""
============================================================
TalentMatch AI
Production Prediction Engine
============================================================

Responsibilities:

• Load trained artifacts once
• Semantic job matching
• TF-IDF job matching
• Hybrid job ranking
• Interview question retrieval

Designed for:

Render Free
512 MB RAM
CPU inference

Memory strategy:

• Singleton model
• CPU only
• float32 embeddings
• No unnecessary embedding copies
• Limited result counts
• No duplicate model instances
• Temporary objects released immediately
============================================================
"""

import gzip
import json
import gc

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
# Prediction Engine
# ============================================================

class PredictionEngine:

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
            "Starting TalentMatch AI prediction engine..."
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

        # Make sure inference does not create
        # unnecessary gradients.

        self.model.eval()

        # ====================================================
        # Job Embeddings
        # ====================================================

        print(
            "Loading job embeddings..."
        )

        loaded_jobs = np.load(
            JOB_EMBEDDINGS
        )

        self.job_embeddings = loaded_jobs[
            "embeddings"
        ]

        # Keep only one usable float32 copy.

        if self.job_embeddings.dtype != np.float32:

            self.job_embeddings = (
                self.job_embeddings.astype(
                    np.float32
                )
            )

        # Release NPZ container.

        del loaded_jobs

        # ====================================================
        # Job Metadata
        # ====================================================

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
            for column in self.job_metadata.columns
        ]

        # ====================================================
        # Interview Embeddings
        # ====================================================

        print(
            "Loading interview embeddings..."
        )

        loaded_interviews = np.load(
            INTERVIEW_EMBEDDINGS
        )

        self.interview_embeddings = (
            loaded_interviews["embeddings"]
        )

        if (
            self.interview_embeddings.dtype
            != np.float32
        ):

            self.interview_embeddings = (
                self.interview_embeddings.astype(
                    np.float32
                )
            )

        del loaded_interviews

        # ====================================================
        # Interview Metadata
        # ====================================================

        print(
            "Loading interview metadata..."
        )

        self.interview_metadata = pd.read_parquet(
            INTERVIEW_METADATA
        )

        self.interview_metadata.columns = [
            str(column)
            .strip()
            .lower()
            for column in self.interview_metadata.columns
        ]

        # ====================================================
        # TF-IDF
        # ====================================================

        print(
            "Loading TF-IDF vectorizer..."
        )

        self.vectorizer = joblib.load(
            TFIDF_MODEL
        )

        # ====================================================
        # TF-IDF Job Matrix
        # ====================================================

        print(
            "Building TF-IDF job index..."
        )

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

        gc.collect()

        # ====================================================
        # Ready
        # ====================================================

        self._loaded = True

        print(
            "TalentMatch AI prediction engine ready."
        )

    # ========================================================
    # Embedding
    # ========================================================

    def embed(
        self,
        text: str
    ):

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
    # Semantic Job Search
    # ========================================================

    def semantic_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS
    ):

        top_k = min(
            max(1, int(top_k)),
            TOP_K_JOBS
        )

        query = self.embed(
            resume_text
        )

        scores = cosine_similarity(
            query.reshape(1, -1),
            self.job_embeddings
        )[0]

        indices = np.argpartition(
            scores,
            -top_k
        )[-top_k:]

        indices = indices[
            np.argsort(
                scores[indices]
            )[::-1]
        ]

        results = []

        for idx in indices:

            row = self.job_metadata.iloc[
                int(idx)
            ]

            results.append({

                "score": round(
                    float(scores[idx]),
                    4
                ),

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

        del query
        del scores

        gc.collect()

        return results

    # ========================================================
    # TF-IDF
    # ========================================================

    def tfidf_scores(
        self,
        resume_text
    ):

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

        top_k = min(
            max(1, int(top_k)),
            TOP_K_JOBS
        )

        # ----------------------------------------------------
        # Semantic score
        # ----------------------------------------------------

        semantic = self.embed(
            resume_text
        )

        semantic_scores = cosine_similarity(
            semantic.reshape(1, -1),
            self.job_embeddings
        )[0]

        # ----------------------------------------------------
        # TF-IDF score
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
        # Efficient top-k selection
        #
        # np.argpartition avoids fully sorting thousands
        # of jobs.
        # ----------------------------------------------------

        indices = np.argpartition(
            final_scores,
            -top_k
        )[-top_k:]

        indices = indices[
            np.argsort(
                final_scores[indices]
            )[::-1]
        ]

        jobs = []

        for idx in indices:

            row = self.job_metadata.iloc[
                int(idx)
            ]

            jobs.append({

                "score": round(
                    float(final_scores[idx]),
                    4
                ),

                "semantic_score": round(
                    float(semantic_scores[idx]),
                    4
                ),

                "tfidf_score": round(
                    float(lexical_scores[idx]),
                    4
                ),

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

        # ----------------------------------------------------
        # Release temporary arrays
        # ----------------------------------------------------

        del semantic
        del semantic_scores
        del lexical_scores
        del final_scores

        gc.collect()

        return jobs

    # ========================================================
    # Interview Questions
    # ========================================================

    def interview_questions(
        self,
        resume_text,
        top_k=TOP_K_INTERVIEWS
    ):

        top_k = min(
            max(1, int(top_k)),
            TOP_K_INTERVIEWS
        )

        query = self.embed(
            resume_text
        )

        scores = cosine_similarity(
            query.reshape(1, -1),
            self.interview_embeddings
        )[0]

        indices = np.argpartition(
            scores,
            -top_k
        )[-top_k:]

        indices = indices[
            np.argsort(
                scores[indices]
            )[::-1]
        ]

        questions = []

        for idx in indices:

            row = self.interview_metadata.iloc[
                int(idx)
            ]

            questions.append({

                "score": round(
                    float(scores[idx]),
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

        del query
        del scores

        gc.collect()

        return questions


# ============================================================
# Global Singleton
# ============================================================

engine = PredictionEngine()
