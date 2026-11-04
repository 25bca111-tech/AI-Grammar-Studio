"""
classifier.py — Machine-Learning grammaticality classifier.

A Naive Bayes classifier (nltk.classify.NaiveBayesClassifier) that predicts
whether a text is GRAMMATICAL or UNGRAMMATICAL, from hand-crafted linguistic
features:

    * POS distribution          (noun / verb / adj / adv / other ratios)
    * length statistics         (words per sentence, average word length)
    * error-pattern flags       (duplicate word, bad article, agreement
                                 mismatch, lowercase start, missing verb,
                                 missing terminal punctuation)

Training data: clean sentences from the Brown corpus (label GRAMMATICAL) plus
programmatic corruption of those same sentences (label UNGRAMMATICAL).
The trained model is pickled to model/nb_model.pkl together with its
hold-out accuracy.
"""

import pickle
import random
import re
from pathlib import Path

from nltk import NaiveBayesClassifier, classify
from nltk.corpus import brown, words as words_corpus

MODEL_PATH = Path(__file__).parent / "model" / "nb_model.pkl"

from grammar_rules import (IRREGULAR_PAST, NUMBER_WORDS, _is_vowel_sound,
                           to_base_form, to_singular)
from nlp_utils import pos_tag_tokens


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
COARSE = {"N": "noun", "V": "verb", "J": "adj", "R": "adv"}


def extract_features(text: str, tagged=None) -> dict:
    """Turn raw text into the feature dict consumed by the classifier."""
    words = [w for w in re.findall(r"[A-Za-z']+", text) if w]
    n = max(len(words), 1)

    if tagged is None:
        from nlp_utils import pos_tag_tokens
        tagged = pos_tag_tokens(words)

    pos_counts = {"noun": 0, "verb": 0, "adj": 0, "adv": 0, "other": 0}
    for _, tag in tagged:
        key = COARSE.get(tag[0], "other")
        pos_counts[key] += 1

    # --- error-pattern flags -------------------------------------------
    low = [w.lower() for w in words]
    has_dup = any(low[i] == low[i + 1] for i in range(len(low) - 1))
    bad_article = any(
        (low[i] == "a" and _is_vowel_sound(low[i + 1]))
        or (low[i] == "an" and not _is_vowel_sound(low[i + 1]))
        for i in range(len(low) - 1)
    )
    agree_err = any(
        (low[i] in ("he", "she", "it") and tagged[i + 1][1] == "VBP")
        or (low[i] in ("they", "we", "you") and tagged[i + 1][1] == "VBZ")
        for i in range(len(low) - 1)
    )
    wrong_past = any(w.lower() in IRREGULAR_PAST for w in words)
    has_verb = any(t[1].startswith("VB") or t[1].startswith("MD") for t in tagged)
    start_lower = bool(words) and words[0][0].islower()
    ends_punct = bool(text) and text.rstrip()[-1:] in ".!?"
    number_plur = any(
        low[i] in NUMBER_WORDS and tagged[i + 1][1] == "NN"
        for i in range(len(low) - 1)
    )

    return {
        "noun_ratio": round(pos_counts["noun"] / n, 2),
        "verb_ratio": round(pos_counts["verb"] / n, 2),
        "adj_ratio": round(pos_counts["adj"] / n, 2),
        "adv_ratio": round(pos_counts["adv"] / n, 2),
        "other_ratio": round(pos_counts["other"] / n, 2),
        "len_bucket": "short" if len(words) <= 5 else ("med" if len(words) <= 15 else "long"),
        "avg_word_len": round(sum(len(w) for w in words) / n, 1),
        "has_dup_word": has_dup,
        "bad_article": bad_article,
        "agreement_err": agree_err,
        "wrong_past": wrong_past,
        "has_verb": has_verb,
        "starts_lower": start_lower,
        "ends_punct": ends_punct,
        "number_needs_plural": number_plur,
    }


# ---------------------------------------------------------------------------
# Training-data generation
# ---------------------------------------------------------------------------
def _clean_sentences(limit: int) -> list[str]:
    """Clean, reasonably-lengthed sentences from the Brown corpus."""
    import nltk
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    seen, out = set(), []
    for words in brown.sents(categories=["news", "editorial", "fiction",
                                         "hobbies", "reviews"]):
        text = " ".join(words).replace(" ,", ",").replace(" .", ".")
        text = re.sub(r"\s+", " ", text).strip()
        text = text.rstrip("-`''\",:;")
        n_words = len(text.split())
        if not (5 <= n_words <= 25):
            continue
        if sum(ch.isalpha() for ch in text) < 0.7 * len(text):
            continue
        if text[:1].islower():
            text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
        if text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
        if len(out) >= limit:
            break
    return out


def corrupt(sentence: str, rng: random.Random) -> str:
    """Inject 1-2 realistic mistakes into a clean sentence."""
    words = sentence.split()
    ops = [
        _op_dup, _op_lower, _op_agreement, _op_article, _op_past,
        _op_missing_punct, _op_number,
    ]
    rng.shuffle(ops)
    applied = 0
    for op in ops:
        if applied >= 2:
            break
        if op(words, rng):
            applied += 1
    return " ".join(words)


def _op_dup(words, rng):
    idxs = [i for i, w in enumerate(words) if w.isalpha()]
    if not idxs:
        return False
    i = rng.choice(idxs)
    words.insert(i + 1, words[i])
    return True


def _op_lower(words, rng):
    if not words[0][:1].isupper():
        return False
    words[0] = words[0].lower()
    return True


def _op_agreement(words, rng):
    tagged = list(zip(words, [t for _, t in __import__(
        "nlp_utils").pos_tag_tokens(words)]))
    for i, (w, tag) in enumerate(tagged[:-1]):
        if w.lower() in ("he", "she", "it") and tag == "VBZ":
            words[i + 1] = to_base_form(words[i + 1].lower())
            return True
        if w.lower() in ("they", "we", "you") and tag == "VBP":
            from grammar_rules import to_3rd_person
            words[i + 1] = to_3rd_person(words[i + 1].lower())
            return True
    return False


def _op_article(words, rng):
    for i, w in enumerate(words[:-1]):
        if w.lower() == "a" and _is_vowel_sound(words[i + 1]):
            words[i] = "an" if words[i].islower() else "An"  # introduce error? no
            words[i] = "a"
            return True
        if w.lower() == "an" and not _is_vowel_sound(words[i + 1]):
            words[i] = "a" if words[i].islower() else "A"
            return True
    return False


def _op_past(words, rng):
    """Reverse-map correct irregular past to the wrong '-ed' form."""
    for i, w in enumerate(words):
        for wrong, right in IRREGULAR_PAST.items():
            if w.lower() == right:
                words[i] = wrong
                return True
    return False


def _op_missing_punct(words, rng):
    if words[-1][-1:] in ".!?":
        words[-1] = words[-1].rstrip(".!?")
        return True
    return False


def _op_number(words, rng):
    tagged = __import__("nlp_utils").pos_tag_tokens(words)
    for i, (w, tag) in enumerate(tagged[:-1]):
        if w.lower() in NUMBER_WORDS and tag == "NN":
            from grammar_rules import to_plural
            # already plural after 'to_plural'? then singularize instead
            words[i + 1] = words[i + 1]
            return True
    return False


def build_dataset(clean: list[str], rng: random.Random):
    labeled = []
    for s in clean:
        labeled.append((s, "grammatical"))
        labeled.append((corrupt(s, rng), "ungrammatical"))
    rng.shuffle(labeled)
    return labeled


# ---------------------------------------------------------------------------
# Train / load
# ---------------------------------------------------------------------------
def train_model(n_sentences: int = 1500, seed: int = 42):
    """Train the Naive Bayes classifier and pickle it with its accuracy."""
    rng = random.Random(seed)
    clean = _clean_sentences(n_sentences)
    dataset = build_dataset(clean, rng)

    feature_sets = []
    for text, label in dataset:
        try:
            feature_sets.append((extract_features(text), label))
        except Exception:
            continue

    split = int(len(feature_sets) * 0.8)
    train_set, test_set = feature_sets[:split], feature_sets[split:]

    clf = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(clf, test_set)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"classifier": clf, "accuracy": round(accuracy, 4),
                     "train_size": len(train_set), "test_size": len(test_set),
                     "features": list(train_set[0][0].keys())}, f)
    print(f"Trained on {len(train_set)} samples, "
          f"test accuracy = {accuracy:.4f} -> {MODEL_PATH}")
    return clf, accuracy


def load_model():
    """Load the pickled model; retrain automatically if missing."""
    if not MODEL_PATH.exists():
        train_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(model_pack, text: str) -> dict:
    """Classify a text and return label + confidence."""
    clf = model_pack["classifier"]
    try:
        dist = clf.prob_classify(extract_features(text))
        label = dist.max()
        confidence = round(dist.prob(label), 3)
    except Exception:
        return {"label": "unknown", "confidence": 0.0}
    return {
        "label": label,
        "confidence": confidence,
        "accuracy": model_pack.get("accuracy"),
        "trainSize": model_pack.get("train_size"),
        "testSize": model_pack.get("test_size"),
    }
