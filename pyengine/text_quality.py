"""
text_quality.py — Writing-quality analytics for the AI Grammar Correction
project. These are the "extra features" that go beyond plain correction:

    1. Grammar score        0-100 number derived from error density
    2. Readability          Flesch Reading Ease + Flesch-Kincaid Grade Level
    3. Style suggestions    passive voice, long sentences, wordy phrases,
                            adverb overuse, repeated sentence starts

All functions are pure text-in / dict-out so they are trivial to test and to
demonstrate individually (e.g. in a college viva).
"""

import re

from nlp_utils import tokenize_sentences, tokenize_words, pos_tag_tokens

# ---------------------------------------------------------------------------
# 1. SYLLABLE ESTIMATION (needed by both readability formulas)
# ---------------------------------------------------------------------------
# Standard heuristic: count groups of vowels, subtract silent trailing "e",
# treat "le" at word end as a syllable ("table" -> 2). Minimum of 1 syllable.
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    """Estimate the number of syllables in one lowercase word."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = _VOWEL_GROUP.findall(w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1  # silent trailing e: "time" -> 1, "table" keeps its "le"
    return max(n, 1)


def _flesch(words: list[str], n_sentences: int) -> tuple[float, float]:
    """Return (flesch_reading_ease, flesch_kincaid_grade)."""
    if not words or n_sentences <= 0:
        return 0.0, 0.0
    n_words = len(words)
    syllables = sum(count_syllables(w) for w in words)
    wps = n_words / n_sentences                     # words per sentence
    spw = syllables / n_words                       # syllables per word
    ease = 206.835 - 1.015 * wps - 84.6 * spw
    grade = 0.39 * wps + 11.8 * spw - 15.59
    return round(ease, 1), round(grade, 1)


def readability_label(ease: float) -> str:
    """Map a Flesch Reading Ease score to a human label (Flesch's table)."""
    if ease >= 90:
        return "Very easy"
    if ease >= 80:
        return "Easy"
    if ease >= 70:
        return "Fairly easy"
    if ease >= 60:
        return "Plain English"
    if ease >= 50:
        return "Fairly difficult"
    if ease >= 30:
        return "Difficult"
    return "Very difficult"


def readability(text: str) -> dict:
    """
    Flesch Reading Ease (0-100, higher = easier) and Flesch-Kincaid Grade
    Level (approx. US school grade needed to read the text).
    """
    sentences = tokenize_sentences(text)
    words = [w for w in tokenize_words(text) if re.search(r"[A-Za-z]", w)]
    ease, grade = _flesch(words, len(sentences))
    return {
        "fleschReadingEase": ease,
        "fleschKincaidGrade": grade,
        "label": readability_label(ease) if words else "—",
        "syllables": sum(count_syllables(w) for w in words),
    }


# ---------------------------------------------------------------------------
# 2. GRAMMAR SCORE
# ---------------------------------------------------------------------------
def grammar_score(n_corrections: int, n_style: int, word_count: int) -> dict:
    """
    0-100 quality score. Errors are measured per 100 words so short and long
    texts are comparable: each grammar correction costs ~4 points per 100
    words, style suggestions cost at most 10 points overall.
    """
    if word_count == 0:
        return {"score": 100, "label": "No text"}
    err_rate = n_corrections / word_count * 100.0     # errors per 100 words
    score = 100 - err_rate * 4 - min(n_style, 10)
    score = int(round(max(0, min(100, score))))
    if score >= 90:
        label = "Excellent"
    elif score >= 75:
        label = "Good"
    elif score >= 60:
        label = "Fair"
    else:
        label = "Needs work"
    return {"score": score, "label": label}


# ---------------------------------------------------------------------------
# 3. STYLE SUGGESTIONS (advisory only — never auto-corrected)
# ---------------------------------------------------------------------------
_BE_FORMS = {"be", "am", "is", "are", "was", "were", "been", "being"}

# wordy phrase -> tighter alternative (classic plain-English advice)
_WORDY = [
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin spite of the fact that\b", "although"),
    (r"\ba large number of\b", "many"),
    (r"\bhas the ability to\b", "can"),
    (r"\bin the event that\b", "if"),
]

MAX_WORDS_PER_SENTENCE = 28


def style_suggestions(text: str) -> list[dict]:
    """
    Detect stylistic weaknesses. These are returned as suggestions for the
    user to consider — the engine deliberately does NOT rewrite them, so the
    author keeps control of their voice.
    """
    suggestions: list[dict] = []
    sentences = tokenize_sentences(text)

    for sent in sentences:
        # -- passive voice: any "be" form followed by a past participle ----
        toks = tokenize_words(sent)
        tagged = pos_tag_tokens(toks)
        for i, (tok, tag) in enumerate(tagged[:-1]):
            if tok.lower() in _BE_FORMS:
                for j in range(i + 1, min(i + 3, len(tagged))):
                    if tagged[j][1] == "VBN" and tagged[j][0].lower() not in _BE_FORMS:
                        suggestions.append({
                            "type": "passive-voice",
                            "message": "Passive voice detected — consider rewriting "
                                       "in the active voice for a stronger sentence.",
                            "sentence": sent,
                            "match": f"{tok} … {tagged[j][0]}",
                        })
                        break
                else:
                    continue
                break

        # -- very long sentences -------------------------------------------
        words = [w for w in toks if re.search(r"[A-Za-z0-9]", w)]
        if len(words) > MAX_WORDS_PER_SENTENCE:
            suggestions.append({
                "type": "long-sentence",
                "message": f"This sentence has {len(words)} words — try splitting "
                           f"it (over {MAX_WORDS_PER_SENTENCE} words is hard to read).",
                "sentence": sent,
                "match": "",
            })

        # -- wordy phrases ---------------------------------------------------
        low = sent.lower()
        for pattern, fix in _WORDY:
            m = re.search(pattern, low)
            if m:
                suggestions.append({
                    "type": "wordy-phrase",
                    "message": f"Wordy phrase — consider replacing "
                               f"'{m.group(0)}' with '{fix}'.",
                    "sentence": sent,
                    "match": m.group(0),
                })

    # -- adverb overuse (-ly adverbs in > 6% of words) ----------------------
    toks_all = [w for w in tokenize_words(text) if re.search(r"[A-Za-z]", w)]
    if toks_all:
        tagged_all = pos_tag_tokens(toks_all)
        ly = [t for t, tag in tagged_all
              if tag == "RB" and t.lower().endswith("ly")]
        if len(ly) > max(2, len(toks_all) * 0.06):
            suggestions.append({
                "type": "adverb-overuse",
                "message": f"{len(ly)} '-ly' adverbs — prefer strong verbs "
                           f"(e.g. 'walked quickly' → 'hurried').",
                "sentence": "",
                "match": ", ".join(ly[:4]),
            })

    # -- repeated sentence starts -------------------------------------------
    for a, b in zip(sentences, sentences[1:]):
        wa = tokenize_words(a)
        wb = tokenize_words(b)
        if wa and wb and wa[0].lower() == wb[0].lower():
            suggestions.append({
                "type": "repeated-start",
                "message": f"Two consecutive sentences start with "
                           f"'{wa[0]}' — vary your openings.",
                "sentence": b,
                "match": wa[0],
            })
            break  # one example is enough

    return suggestions
