"""
TalentMatch AI
Production Prediction Engine

Responsibilities
----------------
- Lazy-load MiniLM
- Load precomputed job embeddings
- Load precomputed TF-IDF matrix
- Load TF-IDF vectorizer
- Perform semantic matching
- Perform TF-IDF matching
- Perform hybrid ranking
- Retrieve interview questions
- Maintain low-memory CPU inference
- Provide job-index information for downstream ranking/explanations

IMPORTANT
---------
The job TF-IDF matrix was generated using:

    category
    description
    requirements
    benefits

with:

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

It must NOT be loaded using scipy.sparse.load_npz().
"""

# ===============================================================
# ENVIRONMENT
# ===============================================================

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    "/tmp/huggingface/transformers",
)
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    "/tmp/huggingface/sentence_transformers",
)

# ===============================================================
# STANDARD LIBRARY
# ===============================================================

import gc
import threading
import time
from pathlib import Path

# ===============================================================
# NUMPY / PANDAS / JOBLIB
# ===============================================================

import joblib
import numpy as np
import pandas as pd

# ===============================================================
# CONFIG
# ===============================================================

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

# ===============================================================
# LOGGER
# ===============================================================

_MODULE_START = time.perf_counter()


def _log(message, *args):
    elapsed = time.perf_counter() - _MODULE_START

    if args:
        try:
            message = message.format(*args)
        except Exception:
            message = (
                str(message)
                + " "
                + " ".join(str(x) for x in args)
            )

    print(
        f"[TalentMatch {elapsed:8.3f}s] {message}",
        flush=True,
    )


_log("predict.py imported.")


# ===============================================================
# PREDICTION ENGINE
# ===============================================================

class PredictionEngine:

    _instance = None
    _instance_lock = threading.Lock()

    # ===========================================================
    # SINGLETON
    # ===========================================================

    def __new__(cls):

        if cls._instance is None:

            with cls._instance_lock:

                if cls._instance is None:

                    cls._instance = super().__new__(cls)

                    cls._instance._initialized = False

        return cls._instance

    # ===========================================================
    # INITIALIZATION
    # ===========================================================

    def __init__(self):

        if self._initialized:
            return

        # -------------------------------------------------------
        # Embedding model
        # -------------------------------------------------------

        self.model = None

        # -------------------------------------------------------
        # Job artifacts
        # -------------------------------------------------------

        self.job_embeddings = None
        self.job_metadata = None
        self.vectorizer = None
        self.job_tfidf_matrix = None

        # -------------------------------------------------------
        # Interview artifacts
        # -------------------------------------------------------

        self.interview_embeddings = None
        self.interview_metadata = None

        # -------------------------------------------------------
        # State
        # -------------------------------------------------------

        self._model_loaded = False
        self._jobs_loaded = False
        self._interviews_loaded = False
        self._loading = False

        self._prediction_lock = threading.RLock()

        self._initialized = True

        _log("PredictionEngine initialized.")
        _log("Heavy artifacts will load lazily.")

    # ===========================================================
    # STATUS
    # ===========================================================

    def is_loaded(self):

        return bool(
            self._model_loaded
            and self._jobs_loaded
        )

    def load_status(self):

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
            "tfidf_vectorizer_loaded": (
                self.vectorizer is not None
            ),
            "tfidf_matrix_loaded": (
                self.job_tfidf_matrix is not None
            ),
            "interview_embeddings_loaded": (
                self.interview_embeddings is not None
            ),
            "interview_metadata_loaded": (
                self.interview_metadata is not None
            ),
        }

    # ===========================================================
    # FILE VALIDATION
    # ===========================================================

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

    # ===========================================================
    # TORCH CONFIGURATION
    # ===========================================================

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

    # ===========================================================
    # LOAD EMBEDDINGS
    # ===========================================================

    @staticmethod
    def _load_embeddings(path):

        path = Path(path)

        PredictionEngine._require_file(path)

        # -------------------------------------------------------
        # NPY
        # -------------------------------------------------------

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

        # -------------------------------------------------------
        # NPZ
        # -------------------------------------------------------

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

    # ===========================================================
    # LOAD CUSTOM TF-IDF CSR MATRIX
    # ===========================================================

    @staticmethod
    def _load_custom_tfidf_matrix(path):

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

        if np.any(indices < 0):

            raise RuntimeError(
                "TF-IDF matrix contains negative indices."
            )

        if indices.size > 0:

            if int(indices.max()) >= shape[1]:

                raise RuntimeError(
                    "TF-IDF matrix contains an index "
                    "outside the feature dimension."
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

        matrix.sort_indices()

        return matrix

    # ===========================================================
    # LOAD EMBEDDING MODEL
    # ===========================================================

    def _load_embedding_model(self):

        if self.model is not None:
            return

        with self._prediction_lock:

            if self.model is not None:
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

    # ===========================================================
    # LOAD JOB ARTIFACTS
    # ===========================================================

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

                # ------------------------------------------------
                # Model
                # ------------------------------------------------

                self._load_embedding_model()

                # ------------------------------------------------
                # Job embeddings
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Job metadata
                # ------------------------------------------------

                self._require_file(
                    JOB_METADATA
                )

                self.job_metadata = pd.read_parquet(
                    JOB_METADATA
                )

                self.job_metadata.columns = [
                    str(c).strip().lower()
                    for c in self.job_metadata.columns
                ]

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

                # ------------------------------------------------
                # TF-IDF vectorizer
                # ------------------------------------------------

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

                # ------------------------------------------------
                # TF-IDF matrix
                # ------------------------------------------------

                self.job_tfidf_matrix = (
                    self._load_custom_tfidf_matrix(
                        JOB_TFIDF_MATRIX
                    )
                )

                _log(
                    "TF-IDF matrix shape: {}",
                    self.job_tfidf_matrix.shape,
                )

                # ------------------------------------------------
                # Alignment checks
                # ------------------------------------------------

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

    # ===========================================================
    # LOAD INTERVIEW ARTIFACTS
    # ===========================================================

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

    # ===========================================================
    # EMBEDDING
    # ===========================================================

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

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim > 1:

            embedding = embedding.reshape(-1)

        return embedding

    # ===========================================================
    # TOP INDICES
    # ===========================================================

    @staticmethod
    def _top_indices(scores, top_k):

        if scores is None:

            return np.empty(
                0,
                dtype=np.int64,
            )

        scores = np.asarray(scores)

        if scores.ndim != 1:

            scores = scores.reshape(-1)

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

    # ===========================================================
    # TF-IDF SCORING
    # ===========================================================

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

    # ===========================================================
    # TF-IDF EVIDENCE
    #
    # Returns terms that actually exist in:
    # - the resume query
    # - the selected job's TF-IDF representation
    #
    # This is used by ranking_engine.py.
    # ===========================================================

    def get_tfidf_evidence(
        self,
        resume_text,
        job_index,
        max_terms=8,
    ):

        self._load_jobs()

        try:

            job_index = int(job_index)

        except Exception:

            return [], []

        if (
            job_index < 0
            or job_index >= self.job_tfidf_matrix.shape[0]
        ):

            return [], []

        try:

            query = self.vectorizer.transform(
                [str(resume_text)]
            )

            job_vector = (
                self.job_tfidf_matrix.getrow(
                    job_index
                )
            )

            query_indices = set(
                int(x)
                for x in query.indices
            )

            job_indices = set(
                int(x)
                for x in job_vector.indices
            )

            matched_indices = (
                query_indices
                & job_indices
            )

            # Terms present in the resume but absent
            # from the selected job representation.
            missing_indices = (
                query_indices
                - job_indices
            )

            feature_names = (
                self.vectorizer.get_feature_names_out()
            )

            # Rank matched terms by combined importance.
            matched_scored = []

            for feature_idx in matched_indices:

                query_pos = np.where(
                    query.indices == feature_idx
                )[0]

                job_pos = np.where(
                    job_vector.indices == feature_idx
                )[0]

                if len(query_pos) == 0:
                    continue

                if len(job_pos) == 0:
                    continue

                query_value = float(
                    query.data[
                        query_pos[0]
                    ]
                )

                job_value = float(
                    job_vector.data[
                        job_pos[0]
                    ]
                )

                importance = (
                    query_value
                    * job_value
                )

                matched_scored.append(
                    (
                        importance,
                        str(
                            feature_names[
                                feature_idx
                            ]
                        ),
                    )
                )

            matched_scored.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            matched_terms = [
                term
                for _, term
                in matched_scored[
                    :max_terms
                ]
            ]

            # Missing terms are sorted according to
            # their TF-IDF importance in the resume.
            missing_scored = []

            for feature_idx in missing_indices:

                query_pos = np.where(
                    query.indices == feature_idx
                )[0]

                if len(query_pos) == 0:
                    continue

                importance = float(
                    query.data[
                        query_pos[0]
                    ]
                )

                missing_scored.append(
                    (
                        importance,
                        str(
                            feature_names[
                                feature_idx
                            ]
                        ),
                    )
                )

            missing_scored.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            missing_terms = [
                term
                for _, term
                in missing_scored[
                    :max_terms
                ]
            ]

            del query
            del job_vector

            return (
                matched_terms,
                missing_terms,
            )

        except Exception as exc:

            _log(
                "TF-IDF evidence warning: {}",
                repr(exc),
            )

            return [], []

    # ===========================================================
    # SEMANTIC JOB SEARCH
    # ===========================================================

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

            results.append(
                self._job_result(
                    row=row,
                    job_index=idx,
                    score=float(scores[idx]),
                    semantic_score=float(scores[idx]),
                    tfidf_score=0.0,
                )
            )

        return results

    # ===========================================================
    # TF-IDF ONLY
    # ===========================================================

    def tfidf_scores(
        self,
        resume_text,
    ):

        self._load_jobs()

        return self._tfidf_scores_from_text(
            resume_text
        )

    # ===========================================================
    # HYBRID JOB SEARCH
    # ===========================================================

    def hybrid_job_search(
        self,
        resume_text,
        top_k=TOP_K_JOBS,
    ):

        start = time.perf_counter()

        self._load_jobs()

        # -------------------------------------------------------
        # ONE RESUME EMBEDDING
        # -------------------------------------------------------

        embed_start = time.perf_counter()

        query = self.embed(
            resume_text
        )

        _log(
            "Resume embedding generated in {:.3f}s.",
            time.perf_counter()
            - embed_start,
        )

        # -------------------------------------------------------
        # SEMANTIC
        # -------------------------------------------------------

        semantic_start = time.perf_counter()

        semantic_scores = np.dot(
            self.job_embeddings,
            query,
        )

        semantic_scores = np.asarray(
            semantic_scores,
            dtype=np.float32,
        )

        _log(
            "Semantic scoring completed in {:.3f}s.",
            time.perf_counter()
            - semantic_start,
        )

        # -------------------------------------------------------
        # TF-IDF
        # -------------------------------------------------------

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

        # -------------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------------

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

        # -------------------------------------------------------
        # HYBRID SCORE
        # -------------------------------------------------------

        final_scores = (
            semantic_scores
            * float(SEMANTIC_WEIGHT)
            +
            lexical_scores
            * float(TFIDF_WEIGHT)
        )

        final_scores = np.asarray(
            final_scores,
            dtype=np.float32,
        )

        # -------------------------------------------------------
        # TOP K
        # -------------------------------------------------------

        indices = self._top_indices(
            final_scores,
            top_k,
        )

        jobs = []

        for idx in indices:

            idx = int(idx)

            row = self.job_metadata.iloc[idx]

            jobs.append(
                self._job_result(
                    row=row,
                    job_index=idx,
                    score=float(
                        final_scores[idx]
                    ),
                    semantic_score=float(
                        semantic_scores[idx]
                    ),
                    tfidf_score=float(
                        lexical_scores[idx]
                    ),
                )
            )

        _log(
            "Hybrid search completed in {:.3f}s.",
            time.perf_counter() - start,
        )

        return jobs

    # ===========================================================
    # JOB RESULT
    # ===========================================================

    def _job_result(
        self,
        row,
        job_index,
        score,
        semantic_score,
        tfidf_score,
    ):

        result = {

            "score":
                float(score),

            "semantic_score":
                float(semantic_score),

            "tfidf_score":
                float(tfidf_score),

            "category":
                self._safe_value(
                    row.get(
                        "category",
                        "",
                    )
                ),

            "title":
                self._safe_value(
                    row.get(
                        "title",
                        "",
                    )
                ),

            "description":
                self._safe_value(
                    row.get(
                        "description",
                        "",
                    )
                ),

            "requirements":
                self._safe_value(
                    row.get(
                        "requirements",
                        "",
                    )
                ),

            "benefits":
                self._safe_value(
                    row.get(
                        "benefits",
                        "",
                    )
                ),

            # Internal alignment information.
            # ranking_engine removes this before returning
            # the final API response.
            "_job_index":
                int(job_index),
        }

        # Preserve stable identifiers if available.
        for key in (
            "job_id",
            "id",
            "listing_id",
        ):

            if row.get(key) not in (
                None,
                "",
            ):

                result[key] = row.get(key)
                break

        # Preserve company/location when available.
        for key in (
            "company",
            "company_name",
            "employer",
            "organization",
            "organisation",
        ):

            value = row.get(key)

            if value not in (
                None,
                "",
            ):

                result["company"] = (
                    self._safe_value(value)
                )
                break

        for key in (
            "location",
            "city",
            "country",
            "job_location",
        ):

            value = row.get(key)

            if value not in (
                None,
                "",
            ):

                result["location"] = (
                    self._safe_value(value)
                )
                break

        return result

    # ===========================================================
    # INTERVIEW QUESTIONS
    # ===========================================================

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
                    float(scores[idx]),

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

    # ===========================================================
    # SAFE VALUE
    # ===========================================================

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

    # ===========================================================
    # CLEANUP JOBS
    # ===========================================================

    def _cleanup_jobs(self):

        self.job_embeddings = None
        self.job_metadata = None
        self.vectorizer = None
        self.job_tfidf_matrix = None

        self._jobs_loaded = False

        gc.collect()

    # ===========================================================
    # CLEANUP INTERVIEWS
    # ===========================================================

    def _cleanup_interviews(self):

        self.interview_embeddings = None
        self.interview_metadata = None

        self._interviews_loaded = False

        gc.collect()

    # ===========================================================
    # SHUTDOWN
    # ===========================================================

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


# ===============================================================
# GLOBAL SINGLETON
# ===============================================================

engine = PredictionEngine()

_log(
    "Global prediction engine created."
)

_log(
    "No Hugging Face model loaded at import time."
)
