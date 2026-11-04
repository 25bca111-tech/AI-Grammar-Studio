"""
test_all.py — Automated test suite for the AI Grammar Correction project.

Run with:  cd pyengine && python3 -m pytest test_all.py -v

Covers every layer of the pipeline:
    * nlp_utils      — tokenization, stemming, lemmatization, NER, sentiment
    * grammar_rules  — conjugation helpers + every correction rule
    * classifier     — model quality and predictions
    * engine         — end-to-end JSON contract
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from nlp_utils import (analyze_sentiment, detokenize, extract_entities,
                       lemmatize_word, stem_word, text_stats,
                       tokenize_sentences, tokenize_words, pos_tag_tokens)
from grammar_rules import (to_3rd_person, to_base_form, to_plural,
                           to_singular, analyze_sentence)
from classifier import load_model, predict
from engine import run
from text_quality import (count_syllables, grammar_score, readability,
                          style_suggestions)


# ===========================================================================
# 1. Tokenization
# ===========================================================================
class TestTokenization:
    def test_sentence_split(self):
        sents = tokenize_sentences("Dr. Smith went home. He was tired!")
        assert len(sents) == 2
        assert sents[1].endswith("!")

    def test_word_split(self):
        assert tokenize_words("don't stop") == ["do", "n't", "stop"]

    def test_detokenizer_contractions(self):
        assert detokenize(["do", "n't", "stop", "."]) == "don't stop."

    def test_detokenizer_punctuation(self):
        assert detokenize(["Hello", ",", "world", "!"]) == "Hello, world!"

    def test_detokenizer_quotes(self):
        assert detokenize(["``", "Hi", "''", "."]) == '"Hi".'


# ===========================================================================
# 2. Stemming vs Lemmatization
# ===========================================================================
class TestStemmingLemmatization:
    def test_porter_stem(self):
        assert stem_word("running") == "run"
        assert stem_word("easily") == "easili"   # stems are crude by design

    def test_lemmatizer_verb(self):
        assert lemmatize_word("running", "VBG") == "run"

    def test_lemmatizer_noun_plural(self):
        assert lemmatize_word("mice", "NNS") == "mouse"

    def test_lemmatizer_adjective(self):
        assert lemmatize_word("better", "JJR") == "good"

    def test_stem_vs_lemma_differ(self):
        # the classic teaching example: stemming chops, lemmatization maps
        assert stem_word("better") != lemmatize_word("better", "JJR")


# ===========================================================================
# 3. NER
# ===========================================================================
class TestNER:
    def test_person(self):
        ents = extract_entities("Albert Einstein visited London.")
        types = {e["type"] for e in ents}
        assert "PERSON" in types

    def test_location(self):
        ents = extract_entities("The meeting was held in Paris.")
        assert any(e["type"] == "GPE" and e["text"] == "Paris" for e in ents)


# ===========================================================================
# 4. Sentiment Analysis
# ===========================================================================
class TestSentiment:
    def test_positive(self):
        r = analyze_sentiment("I absolutely love this amazing product!")
        assert r["label"] == "positive" and r["compound"] > 0.3

    def test_negative(self):
        r = analyze_sentiment("This is terrible and I hate it.")
        assert r["label"] == "negative" and r["compound"] < -0.3

    def test_neutral(self):
        assert analyze_sentiment("The chair is on the floor.")["label"] == "neutral"

    def test_scores_sum(self):
        r = analyze_sentiment("Great food, awful service.")
        assert abs(r["positive"] + r["neutral"] + r["negative"] - 1) < 0.01


# ===========================================================================
# 5. Conjugation helpers
# ===========================================================================
class TestConjugation:
    @pytest.mark.parametrize("base,third", [
        ("walk", "walks"), ("watch", "watches"), ("carry", "carries"),
        ("go", "goes"), ("have", "has"), ("miss", "misses"),
    ])
    def test_third_person(self, base, third):
        assert to_3rd_person(base) == third

    @pytest.mark.parametrize("third,base", [
        ("walks", "walk"), ("watches", "watch"), ("carries", "carry"),
        ("goes", "go"), ("has", "have"), ("likes", "like"),
        ("misses", "miss"), ("runs", "run"),
    ])
    def test_base_form(self, third, base):
        assert to_base_form(third) == base

    @pytest.mark.parametrize("sing,plur", [
        ("apple", "apples"), ("box", "boxes"), ("baby", "babies"),
        ("child", "children"), ("man", "men"), ("foot", "feet"),
    ])
    def test_plural_roundtrip(self, sing, plur):
        assert to_plural(sing) == plur and to_singular(plur) == sing


# ===========================================================================
# 6. Grammar rules (input -> expected corrected substring)
# ===========================================================================
GOOD = [
    ("clean simple", "She walks to school every day."),
    ("subjunctive kept", "If I were you, I would wait."),
    ("its possessive kept", "The dog wagged its tail happily."),
    ("proper nouns kept", "Samsung unveiled the Galaxy in Seoul."),
]


@pytest.mark.parametrize("label,text", GOOD)
def test_no_false_positives(label, text):
    result = run(text)
    assert result["correctedText"] == text, f"false positives on: {label}"


@pytest.mark.parametrize("bad,good", [
    ("he go to work.", "he goes to work."),
    ("they is happy.", "they are happy."),
    ("we was late.", "we were late."),
    ("i like tea.", "I like tea."),
    ("she buyed a car.", "she bought a car."),
    ("a apple", "an apple"),
    ("an banana", "a banana"),
    ("two apple", "two apples"),
    ("much books", "many books"),
    ("i dont know.", "I don't know."),
    ("its a nice day.", "it's a nice day."),
    ("should of gone", "should have gone"),
    ("better then mine", "better than mine"),
    ("discuss about it", "discuss it"),
    ("return back home", "return home"),
    ("married with her", "married to her"),
    ("she have finished.", "she has finished."),
    ("he can plays.", "he can play."),
    ("it dont work.", "it doesn't work."),
    ("i like it alot", "I like it a lot"),
])
def test_corrections(bad, good):
    result = run(bad)
    corrected = result["correctedText"].lower()
    assert good.lower() in corrected, f"{bad!r} -> {corrected!r}"


def test_repeated_word_removed():
    assert "the the" not in run("I saw the the dog.").correctedText if False else \
        "the the" not in run("I saw the the dog.")["correctedText"]


def test_fragment_flagged_not_deleted():
    r = run("Because I was tired.")
    assert r["correctedText"] == "Because I was tired."   # text kept
    assert any(c["rule"] == "sentence-fragment" for c in r["corrections"])


def test_spelling_fixed():
    r = run("this is a beutiful garden")
    assert "beautiful" in r["correctedText"]


def test_homeworks_uncountable():
    r = run("She gave us much homeworks yesterday.")
    assert "homework" in r["correctedText"]
    assert "homeworks" not in r["correctedText"]


# ===========================================================================
# 7. ML classifier
# ===========================================================================
class TestClassifier:
    @classmethod
    def setup_class(cls):
        cls.model = load_model()

    def test_accuracy_threshold(self):
        assert self.model["accuracy"] >= 0.90

    def test_clean_is_grammatical(self):
        out = predict(self.model, "She walks to school every morning.")
        assert out["label"] == "grammatical"

    def test_corrupted_is_ungrammatical(self):
        out = predict(self.model, "he buyed two apple and dont go home")
        assert out["label"] == "ungrammatical"

    def test_confidence_present(self):
        out = predict(self.model, "The sun rises in the east.")
        assert 0.0 <= out["confidence"] <= 1.0


# ===========================================================================
# 8. Engine end-to-end contract
# ===========================================================================
class TestEngine:
    def test_response_shape(self):
        r = run("John bought two apples in New York.")
        for key in ("correctedText", "corrections", "highlights", "pipeline",
                    "entities", "sentiment", "stats", "classifier"):
            assert key in r

    def test_highlight_bounds(self):
        r = run("he go home.")
        for h in r["highlights"]:
            assert 0 <= h["start"] <= h["end"] <= len(r["correctedText"])

    def test_token_table_fields(self):
        r = run("The children were running quickly.")
        row = r["pipeline"]["tokens"][0]
        assert set(row) == {"token", "pos", "stem", "lemma"}

    def test_stats(self):
        r = run("One two three four five. Six seven eight nine ten.")
        assert r["stats"]["wordCount"] == 10
        assert r["stats"]["sentenceCount"] == 2

    def test_empty_is_safe(self):
        r = run("   ")
        assert r["correctedText"] == ""


# ===========================================================================
# 9. Writing quality (extra features)
# ===========================================================================
class TestSyllables:
    def test_basic_counts(self):
        assert count_syllables("cat") == 1
        assert count_syllables("time") == 1        # silent e
        assert count_syllables("table") == 2       # trailing -le kept
        assert count_syllables("running") == 2
        assert count_syllables("beautiful") == 3

    def test_never_zero(self):
        assert count_syllables("xyz") >= 1


class TestReadability:
    def test_easy_text_scores_high(self):
        r = readability("I see the cat. The cat is big. It runs to me.")
        assert r["fleschReadingEase"] > 70
        assert r["label"] in ("Very easy", "Easy", "Fairly easy")

    def test_hard_text_scores_low(self):
        r = readability(
            "The implementation of comprehensive administrative "
            "reorganization necessitated considerable deliberation."
        )
        assert r["fleschReadingEase"] < 60
        assert r["fleschKincaidGrade"] > 6

    def test_empty(self):
        r = readability("")
        assert r["label"] == "—"


class TestGrammarScore:
    def test_clean_text_is_perfect(self):
        assert grammar_score(0, 0, 50) == {"score": 100, "label": "Excellent"}

    def test_errors_lower_score(self):
        clean = grammar_score(0, 0, 50)["score"]
        dirty = grammar_score(3, 0, 50)["score"]
        assert dirty < clean

    def test_bounds(self):
        assert grammar_score(500, 0, 10)["score"] == 0
        assert grammar_score(0, 99, 10)["score"] >= 0


class TestStyleSuggestions:
    def test_passive_detected(self):
        s = style_suggestions("The report was written by the committee.")
        assert any(x["type"] == "passive-voice" for x in s)

    def test_wordy_phrase_detected(self):
        s = style_suggestions("We left early in order to avoid traffic.")
        assert any(x["type"] == "wordy-phrase" for x in s)

    def test_long_sentence_detected(self):
        s = style_suggestions(
            "The team reviewed the plan and the budget and the timeline "
            "and the risks and the rewards and the alternatives and the "
            "costs and the benefits before making a final decision together."
        )
        assert any(x["type"] == "long-sentence" for x in s)

    def test_clean_text_no_suggestions(self):
        assert style_suggestions("She walks to school every morning.") == []


class TestEngineQualityFields:
    def test_response_has_new_fields(self):
        r = run("She walks to school every day.")
        assert set(r["quality"]) == {"score", "label"}
        assert {"fleschReadingEase", "fleschKincaidGrade", "label",
                "syllables"} <= set(r["readability"])
        assert isinstance(r["style"], list)

    def test_clean_text_full_score(self):
        r = run("She walks to school every day.")
        assert r["quality"]["score"] == 100

    def test_bad_text_lower_score(self):
        clean = run("She walks to school every day.")["quality"]["score"]
        bad = run("he buyed two apple and dont go home.")["quality"]["score"]
        assert bad < clean
