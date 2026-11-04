"""Pytest bootstrap: make sure every NLTK corpus the engine needs is present
BEFORE any test imports/calls the NLP functions (CI machines start empty)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nlp_utils import ensure_nltk_data

ensure_nltk_data()
