"""
Dataset Builder

Creates production-ready datasets
for TalentMatch AI.
"""

import pandas as pd
from pathlib import Path

from config import *

class DatasetBuilder:

    def __init__(self):

        self.resume_df = None

        self.job_df = None

        self.interview_df = None

    # ------------------------------------------

    def load_datasets(self):

        print("Loading datasets...")

        self.resume_df = pd.read_csv(

            RESUME_DATASET

        )

        self.job_df = pd.read_csv(

            JOB_DATASET

        )

        self.interview_df = pd.read_csv(

            INTERVIEW_DATASET

        )

    # ------------------------------------------

    def remove_duplicates(self):

        self.resume_df.drop_duplicates(

            inplace=True

        )

        self.job_df.drop_duplicates(

            inplace=True

        )

        self.interview_df.drop_duplicates(

            inplace=True

        )

    # ------------------------------------------

    def normalize_text(self):

        for dataframe in [

            self.resume_df,

            self.job_df,

            self.interview_df

        ]:

            dataframe.columns = [

                c.lower().strip()

                for c in dataframe.columns

            ]

    # ------------------------------------------

    def validate(self):

        assert len(self.resume_df) > 0

        assert len(self.job_df) > 0

        assert len(self.interview_df) > 0

    # ------------------------------------------

    def save(self):

        self.resume_df.to_csv(

            PROCESSED_DATA_DIR /

            "resumes.csv",

            index=False

        )

        self.job_df.to_csv(

            PROCESSED_DATA_DIR /

            "job_descriptions.csv",

            index=False

        )

        self.interview_df.to_csv(

            PROCESSED_DATA_DIR /

            "interview_questions.csv",

            index=False

        )

    # ------------------------------------------

    def build(self):

        self.load_datasets()

        self.remove_duplicates()

        self.normalize_text()

        self.validate()

        self.save()

        print("Datasets prepared successfully.")

if __name__ == "__main__":

    DatasetBuilder().build()
