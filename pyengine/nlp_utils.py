"""
nlp_utils.py — Core NLP building blocks for the AI Grammar Correction project.

Implements / wraps the five fundamental NLP techniques used in the pipeline:

    1. Tokenization          (nltk.sent_tokenize / word_tokenize)
    2. Stemming              (nltk PorterStemmer)
    3. Lemmatization         (nltk WordNetLemmatizer, POS-aware)
    4. POS Tagging + NER     (nltk averaged_perceptron_tagger + ne_chunk)
    5. Sentiment Analysis    (nltk VADER SentimentIntensityAnalyzer)

Every function is intentionally small and well commented so each technique
can be studied / demonstrated individually (e.g. in a college viva).
"""

import re

import nltk
from nltk import pos_tag, ne_chunk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------------------------------------------------------------------------
# Lazy singletons (created once per process, reused for every request)
# ---------------------------------------------------------------------------
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
_vader = None

# Map Penn-Treebank POS tags -> WordNet POS for accurate lemmatization
_WN_POS = {
    "J": "a",   # adjective
    "V": "v",   # verb
    "N": "n",   # noun
    "R": "r",   # adverb
}


def ensure_nltk_data():
    """Download any missing NLTK data at runtime (safe: skips if present)."""
    packages = [
        "punkt", "punkt_tab", "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4",
        "vader_lexicon", "maxent_ne_chunker", "maxent_ne_chunker_tab",
        "words",
    ]
    for pkg in packages:
        try:
            nltk.data.find(_resource_path(pkg))
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def _resource_path(pkg):
    """Return the NLTK search path hint for a package name."""
    mapping = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "vader_lexicon": "sentiment/vader_lexicon",
        "maxent_ne_chunker": "chunkers/maxent_ne_chunker",
        "maxent_ne_chunker_tab": "chunkers/maxent_ne_chunker_tab",
        "words": "corpora/words",
    }
    return mapping.get(pkg, f"corpora/{pkg}")


# ---------------------------------------------------------------------------
# 1. TOKENIZATION
# ---------------------------------------------------------------------------
def tokenize_sentences(text: str) -> list[str]:
    """Sentence-level tokenization using NLTK's Punkt sentence splitter."""
    return [s for s in sent_tokenize(text) if s.strip()]


def tokenize_words(text: str) -> list[str]:
    """Word-level tokenization using NLTK's Treebank word tokenizer."""
    return word_tokenize(text)


def detokenize(tokens: list[str]) -> str:
    """
    Rebuild natural text from word tokens (undo word_tokenize spacing).

    word_tokenize splits "don't" -> ["do", "n't"] and puts punctuation on
    its own; naive " ".join() would produce ugly output like "don ' t ." .
    This detokenizer fixes spacing around punctuation and contractions.
    """
    tokens = [t.replace("``", '"').replace("''", '"') for t in tokens]
    text = " ".join(tokens)
    # no space before closing punctuation
    text = re.sub(r'\s+([.,!?;:%)"])', r"\1", text)
    # no space after opening punctuation
    text = re.sub(r'([(\[$"])\s+', r"\1", text)
    # glue English contractions: "do n't" -> "don't", "' s" -> "'s"
    text = re.sub(r"\s+n't\b", "n't", text)
    text = re.sub(r"\s+'(s|re|ve|ll|d|m|t)\b", r"'\1", text)
    # glue hyphens: "well - known" -> "well-known"
    text = re.sub(r"\s-\s", "-", text)
    # ellipsis
    text = re.sub(r"\s?\.\s?\.\s?\.", "...", text)
    # Treebank quote tokens -> plain double quotes
    text = text.replace("``", '"').replace("''", '"')
    return text.strip()


# ---------------------------------------------------------------------------
# 2. STEMMING
# ---------------------------------------------------------------------------
def stem_word(word: str) -> str:
    """Porter stemming: crudely chops affixes ('running'->'run', 'flies'->'fli')."""
    return _stemmer.stem(word)


# ---------------------------------------------------------------------------
# 3. LEMMATIZATION (POS-aware)
# ---------------------------------------------------------------------------
def lemmatize_word(word: str, tag: str | None = None) -> str:
    """
    WordNet lemmatization using the token's POS tag for dictionary-accurate
    results ('running'->'run', 'better'->'good', 'mice'->'mouse').
    """
    wn_pos = _WN_POS.get(tag[0], "n") if tag else "n"
    lemma = _lemmatizer.lemmatize(word.lower(), wn_pos)
    # second pass for verbs tagged oddly (e.g. 'went' tagged NN)
    if lemma == word.lower() and tag and tag.startswith("V"):
        lemma = _lemmatizer.lemmatize(word.lower(), "v")
    return lemma


# ---------------------------------------------------------------------------
# 4. POS TAGGING + NAMED ENTITY RECOGNITION
# ---------------------------------------------------------------------------
def pos_tag_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Penn-Treebank POS tagging via the averaged perceptron tagger."""
    return pos_tag(tokens)


def extract_entities(text: str) -> list[dict]:
    """
    Named Entity Recognition using NLTK's ne_chunk over POS-tagged tokens.
    Returns a list of {text, type} for PERSON / GPE / ORGANIZATION etc.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged, binary=False)

    entities = []
    for subtree in tree:
        if hasattr(subtree, "label"):
            # ne_chunk groups consecutive tokens of one entity into a subtree
            words = [leaf[0] for leaf in subtree.leaves()]
            entities.append(
                {
                    "text": " ".join(words),
                    "type": subtree.label(),  # PERSON, GPE, ORGANIZATION...
                }
            )
    return entities


# ---------------------------------------------------------------------------
# 5. SENTIMENT ANALYSIS (VADER)
# ---------------------------------------------------------------------------
def analyze_sentiment(text: str) -> dict:
    """
    VADER sentiment analysis: returns positive / neutral / negative ratios,
    a compound score in [-1, +1] and a human-readable label.
    """
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "positive": round(scores["pos"], 3),
        "neutral": round(scores["neu"], 3),
        "negative": round(scores["neg"], 3),
        "compound": round(compound, 3),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------
def text_stats(text: str) -> dict:
    """Basic lexical statistics shown on the dashboard."""
    words = [w for w in word_tokenize(text) if re.search(r"[A-Za-z0-9]", w)]
    sentences = tokenize_sentences(text)
    unique = {w.lower() for w in words}
    avg_len = round(len(words) / max(len(sentences), 1), 1)
    diversity = round(len(unique) / max(len(words), 1), 2)
    return {
        "wordCount": len(words),
        "sentenceCount": len(sentences),
        "avgSentenceLength": avg_len,
        "lexicalDiversity": diversity,
    }
