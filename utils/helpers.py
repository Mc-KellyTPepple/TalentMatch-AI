"""
TalentMatch AI
General Helper Functions
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

# ----------------------------------------------------

def ensure_directory(path: Path):

    path.mkdir(

        parents=True,

        exist_ok=True

    )


# ----------------------------------------------------

def load_json(path: Path):

    with open(path, "r", encoding="utf-8") as file:

        return json.load(file)


# ----------------------------------------------------

def save_json(data, path: Path):

    ensure_directory(path.parent)

    with open(path, "w", encoding="utf-8") as file:

        json.dump(

            data,

            file,

            indent=4,

            ensure_ascii=False

        )


# ----------------------------------------------------

def save_pickle(obj, path: Path):

    ensure_directory(path.parent)

    with open(path, "wb") as file:

        pickle.dump(obj, file)


# ----------------------------------------------------

def load_pickle(path: Path):

    with open(path, "rb") as file:

        return pickle.load(file)


# ----------------------------------------------------

def safe_divide(a, b):

    if b == 0:

        return 0.0

    return a / b


# ----------------------------------------------------

def percentage(value):

    return round(

        value * 100,

        2

    )


# ----------------------------------------------------

def unique_preserve_order(items):

    seen = set()

    output = []

    for item in items:

        if item not in seen:

            seen.add(item)

            output.append(item)

    return output


# ----------------------------------------------------

if __name__ == "__main__":

    print(

        unique_preserve_order(

            [1,2,2,3,1,4]

        )

    )
