"""
TalentMatch AI — Clean Job Index Rebuilder
------------------------------------------
Run this ONCE after replacing ranking_engine.py.

It uses the SAME configured JOB_DATASET and model/artifact paths
already defined in config.py.

What it fixes:
- removes exact duplicate postings
- removes near-identical duplicate postings
- preserves title/company fields when the source dataset contains them
- regenerates job embeddings so metadata and embeddings remain aligned
- rebuilds the TF-IDF vectorizer and sparse job matrix

No hard-coded dataset path is introduced.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    EMBEDDING_MODEL,
    JOB_DATASET,
    JOB_EMBEDDINGS,
    JOB_METADATA,
    TFIDF_MODEL,
)


DUPLICATE_SIMILARITY = 0.92
EMBED_BATCH_SIZE = 64


def norm(value) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_col(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def job_text(row, columns):
    parts = []
    for col in columns:
        if col is not None:
            value = str(row.get(col, "") or "")
            if value:
                parts.append(value)
    return " ".join(parts)


def main():
    df = pd.read_csv(JOB_DATASET)
    df.columns = [str(c).strip().lower() for c in df.columns]

    title = first_col(
        df,
        ["title", "job_title", "jobtitle", "position", "role"],
    )
    company = first_col(
        df,
        ["company", "company_name", "employer", "organization", "organisation"],
    )
    category = first_col(df, ["category"])
    description = first_col(df, ["description"])
    requirements = first_col(df, ["requirements", "requirement"])
    benefits = first_col(df, ["benefits"])

    text_columns = [
        c for c in
        [title, company, category, description, requirements, benefits]
        if c is not None
    ]

    print(f"Original job rows: {len(df):,}")
    print("Detected title column:", title)
    print("Detected company column:", company)
    print("Detected category column:", category)

    # Exact duplicate removal.
    fingerprints = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(norm)
        .map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())
    )

    df = df.loc[~fingerprints.duplicated()].reset_index(drop=True)

    # Near-duplicate removal. This is intentionally performed only
    # on the job text, never on category alone.
    kept = []
    kept_text = []

    for idx, row in df.iterrows():
        text = norm(job_text(row, text_columns))

        duplicate = False
        if text:
            for previous in kept_text:
                if SequenceMatcher(None, text, previous).ratio() >= DUPLICATE_SIMILARITY:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(idx)
            kept_text.append(text)

    df = df.iloc[kept].reset_index(drop=True)

    print(f"Clean job rows: {len(df):,}")
    print(f"Removed duplicates: {len(fingerprints) - len(df):,}")

    # Build semantic documents.
    documents = [
        job_text(row, text_columns)
        for _, row in df.iterrows()
    ]

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)

    # Preserve ALL useful source metadata, not only category/description/etc.
    # This lets the UI show the actual job title and company.
    metadata = df.copy()
    metadata.to_parquet(
        JOB_METADATA,
        compression="brotli",
        index=False,
    )

    np.savez_compressed(
        JOB_EMBEDDINGS,
        embeddings=np.ascontiguousarray(embeddings, dtype=np.float16),
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        dtype=np.float32,
        stop_words="english",
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    joblib.dump(
        vectorizer,
        TFIDF_MODEL,
        compress=3,
    )

    matrix_path = Path(str(TFIDF_MODEL)).parent / "job_tfidf_matrix.npz"
    save_npz(matrix_path, tfidf_matrix)

    print("\nINDEX REBUILD COMPLETE")
    print("Metadata:", JOB_METADATA)
    print("Embeddings:", JOB_EMBEDDINGS)
    print("TF-IDF:", TFIDF_MODEL)
    print("TF-IDF matrix:", matrix_path)
    print("Final jobs:", len(metadata))
    print("Embedding shape:", embeddings.shape)


if __name__ == "__main__":
    main()
