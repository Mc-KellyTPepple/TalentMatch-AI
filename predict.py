"""
=========================================================
TalentMatch AI
Prediction Engine
=========================================================

Loads every trained artifact once and performs:

• Semantic Job Matching
• TF-IDF Matching
• Hybrid Ranking
• Interview Question Retrieval

Designed for Render Free (512 MB)
=========================================================
"""

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

########################################################################
# Singleton Prediction Engine
########################################################################

class PredictionEngine:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            cls._instance._loaded = False

        return cls._instance

    ############################################################

    def __init__(self):

        if self._loaded:
            return

        print("Loading AI artifacts...")

        ########################################################
        # Sentence Transformer
        ########################################################

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )

        ########################################################
        # Job Embeddings
        ########################################################

        self.job_embeddings = np.load(
            JOB_EMBEDDINGS
        )["embeddings"].astype(np.float32)

        self.job_metadata = pd.read_parquet(
            JOB_METADATA
        )

        ########################################################
        # Interview Embeddings
        ########################################################

        self.interview_embeddings = np.load(
            INTERVIEW_EMBEDDINGS
        )["embeddings"].astype(np.float32)

        self.interview_metadata = pd.read_parquet(
            INTERVIEW_METADATA
        )

        ########################################################
        # TF-IDF
        ########################################################
        
        self.vectorizer = joblib.load(
            TFIDF_MODEL
        )
        
        # Build the TF-IDF matrix once during startup
        job_documents = (
            self.job_metadata
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )
        
        self.job_tfidf_matrix = self.vectorizer.transform(
            job_documents
        )
        
        del job_documents

        ########################################################
        # Skills
        ########################################################

        with gzip.open(
            SKILLS,
            "rt",
            encoding="utf8"
        ) as f:

            self.skills = set(json.load(f))

        self._loaded = True

        print("Prediction engine ready.")

    ############################################################

    def embed(self, text):

        embedding = self.model.encode(

            text,

            normalize_embeddings=True,

            convert_to_numpy=True

        )

        return embedding.astype(np.float32)

    ############################################################

    def semantic_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS
    ):

        query = self.embed(resume_text)

        scores = cosine_similarity(

            query.reshape(1, -1),

            self.job_embeddings

        )[0]

        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in indices:

            row = self.job_metadata.iloc[idx]

            results.append({

                "score": round(float(scores[idx]), 4),

                "category": row.get("Category", ""),

                "description": row.get("Description", ""),

                "requirements": row.get("Requirements", ""),

                "benefits": row.get("Benefits", "")

            })

        return results

    ############################################################

    def tfidf_scores(self, resume_text):
    """
    Compute lexical similarity between the resume
    and all job descriptions.
    """

    query = self.vectorizer.transform([resume_text])

    scores = cosine_similarity(
        query,
        self.job_tfidf_matrix
    )[0]

    return scores

    ############################################################

    def hybrid_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS
    ):

        semantic = self.embed(resume_text)

        semantic_scores = cosine_similarity(

            semantic.reshape(1, -1),

            self.job_embeddings

        )[0]

        lexical_scores = self.tfidf_scores(
            resume_text
        )

        final_scores = (

            semantic_scores * SEMANTIC_WEIGHT +

            lexical_scores * TFIDF_WEIGHT

        )

        indices = np.argsort(
            final_scores
        )[::-1][:top_k]

        jobs = []

        for idx in indices:

            row = self.job_metadata.iloc[idx]

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

                "category": row.get("Category", ""),

                "description": row.get("Description", ""),

                "requirements": row.get("Requirements", ""),

                "benefits": row.get("Benefits", "")

            })

        return jobs

    ############################################################

    def interview_questions(
        self,
        resume_text,
        top_k=TOP_K_INTERVIEWS
    ):

        query = self.embed(resume_text)

        scores = cosine_similarity(

            query.reshape(1, -1),

            self.interview_embeddings

        )[0]

        indices = np.argsort(
            scores
        )[::-1][:top_k]

        questions = []

        for idx in indices:

            row = self.interview_metadata.iloc[idx]

            questions.append({

                "score": round(
                    float(scores[idx]),
                    4
                ),

                "question": row["question"],

                "answer": row["answer"],

                "role": row["role"],

                "category": row["category"],

                "difficulty": row["difficulty"],

                "experience": row["experience"]

            })

        return questions


########################################################################
# Global Instance
########################################################################

engine = PredictionEngine()
