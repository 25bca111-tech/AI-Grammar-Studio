"""
engine.py — AI Grammar Correction pipeline orchestrator.

Reads a JSON request from stdin, runs the complete NLP pipeline and prints a
JSON result to stdout:

    request :  { "text": "..." }
    response:  {
        correctedText, corrections[], highlightRanges[],
        pipeline:   { sentences, tokens[{token,pos,stem,lemma}] },
        entities:   [...], sentiment: {...}, stats: {...},
        classifier: { label, confidence, accuracy, ... }
    }

Pipeline order:
    1. Tokenization        (sentences + words, via NLTK)
    2. Rule-based grammar  (grammar_rules.py, full suite)
    3. Spelling            (TextBlob spell-check on dictionary-unknown words)
    4. ML classification   (Naive Bayes, classifier.py)
    5. NER                 (ne_chunk)
    6. Sentiment analysis  (VADER)
    7. Statistics          (tokenization/stemming/lemmatization table)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# -- Dependency guard ----------------------------------------------------
# If NLTK/TextBlob are missing, don't dump a raw traceback on stderr.
# Instead emit a clean JSON error the web UI can show as-is, with the
# exact command needed to fix it.
try:
    from nlp_utils import (analyze_sentiment, detokenize, ensure_nltk_data,
                           extract_entities, lemmatize_word, pos_tag_tokens,
                           stem_word, text_stats, tokenize_sentences,
                           tokenize_words)
    import grammar_rules
    from grammar_rules import RULE_LABELS, analyze_sentence
    from classifier import load_model, predict
    from text_quality import grammar_score, readability, style_suggestions
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", None) or "nltk"
    print(json.dumps({
        "error": (
            f"The Python NLP engine is missing the '{missing}' package. "
            f"Fix: open a terminal in the project root and run "
            f"\"cd pyengine && python3 -m pip install -r requirements.txt\", "
            f"then restart the app."
        )
    }))
    sys.exit(1)

MAX_TABLE_TOKENS = 120   # cap the token table sent to the UI
MAX_TEXT_LEN = 5000      # safety cap on accepted input


def correct_text(text: str) -> tuple[str, list[dict], list[dict]]:
    """
    Run all grammar rules + spelling over the text.
    Returns (corrected_text, corrections, highlight_ranges) where
    highlight_ranges are char offsets INSIDE the corrected text.
    """
    # -- 0. whitespace normalisation ------------------------------------
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return "", [], []

    sentences = tokenize_sentences(text)
    corrections = []          # accumulated correction dicts
    corrected_sentences = []  # rebuilt sentence strings
    sent_offsets = []         # char offset of each sentence in the result

    for sent in sentences:
        tokens, tagged, found = analyze_sentence(sent)

        # -- 1. merge: one correction per token, earlier rule wins --------
        # (a) sentence-fragment findings are FLAGS only — they never claim
        #     tokens, so other rules like capitalization still apply;
        # (b) capitalization fixes are case_only — applied as a post-pass
        #     so they never block e.g. an article fix on the same token.
        chosen_flags = [c for c in found if c["rule"] == "sentence-fragment"]
        found_applied = [c for c in found if c["rule"] not in
                         ("sentence-fragment",)]
        case_fixes = [c for c in found_applied if c.get("case_only")]
        span_rules = [c for c in found_applied if not c.get("case_only")]
        span_owner = {}
        chosen = []
        for corr in sorted(span_rules, key=lambda c: -c["tokenCount"]):
            idxs = range(corr["tokenIndex"],
                         corr["tokenIndex"] + corr["tokenCount"])
            if any(i in span_owner for i in idxs):
                continue  # overlaps an already-placed correction
            for i in idxs:
                span_owner[i] = True
            chosen.append(corr)
        chosen.sort(key=lambda c: c["tokenIndex"])

        # -- 2. spelling pass (only on tokens untouched by other rules) ---
        chosen += spell_check(tokens, tagged, span_owner)

        # -- 3. apply replacements to tokens -------------------------------
        new_tokens = list(tokens)
        for corr in chosen:
            i, n = corr["tokenIndex"], corr["tokenCount"]
            sugg = corr["suggestion"]
            if sugg == "":
                for k in range(i, i + n):
                    new_tokens[k] = None        # deletion
            elif n == 1:
                new_tokens[i] = sugg            # single-token replacement
            else:
                # multi-token span -> first token holds new text, rest None
                parts = sugg.split()
                new_tokens[i] = " ".join(parts)
                for k in range(i + 1, i + n):
                    new_tokens[k] = None
        # -- 3b. capitalization post-pass (composes with replaced tokens) --
        for corr in case_fixes:
            i = corr["tokenIndex"]
            if i >= len(new_tokens) or new_tokens[i] is None:
                continue
            if new_tokens[i] == corr["original"]:
                new_tokens[i] = corr["suggestion"]      # token untouched
            else:
                new_tokens[i] = new_tokens[i].capitalize()  # composed fix
            chosen.append(corr)

        kept = [t for t in new_tokens if t]
        corrected = detokenize(kept) if kept else detokenize(tokens)

        # -- 4. record corrections with highlight ranges ------------------
        sent_off = sum(len(s) + 1 for s in corrected_sentences)
        for corr in chosen + chosen_flags:
            i, n = corr["tokenIndex"], corr["tokenCount"]
            # char offset of token i within THIS sentence's corrected text
            prefix = detokenize([t for t in new_tokens[:i] if t])
            start = sent_off + len(prefix)
            if corr["suggestion"] == "":
                # deleted span: highlight the surviving neighbours
                nxt = detokenize([t for t in new_tokens[:i] if t])
                start = sent_off + len(nxt)
                end = start
            else:
                end = sent_off + len(detokenize(
                    [t for t in new_tokens[:i + 1] if t]))
            corr["sentence"] = sent
            corrections.append(corr)
            corr["_hl"] = (start, max(end, start + len(corr["suggestion"])))

        corrected_sentences.append(corrected)

    corrected_text = " ".join(corrected_sentences)

    # -- 5. text-level punctuation pass ---------------------------------
    if corrected_text and corrected_text[-1] not in ".!?…":
        if corrected_text[-1] in ",;:-":
            corrected_text = corrected_text[:-1]
        corrected_text += "."
        corrections.append(dict(
            tokenIndex=0, tokenCount=0, original="", suggestion="",
            rule="punctuation", sentence=sentences[-1] if sentences else "",
            message="The text did not end with punctuation; a period was added.",
            _hl=(len(corrected_text) - 1, len(corrected_text))))

    # clean up leftovers: " ," artifacts from deletions
    corrected_text = re.sub(r"\s+,(\s+)", r",\1", corrected_text)
    # collapse spaced runs of bare punctuation ("! ! !" -> "!!!")
    corrected_text = re.sub(r"([!?.]) (?=[!?.])", r"\1", corrected_text)
    corrected_text = re.sub(r"\s{2,}", " ", corrected_text).strip()

    # convert _hl tuples into plain dicts for JSON
    highlights = [dict(start=hl[0], end=hl[1],
                       rule=c["rule"], suggestion=c["suggestion"])
                  for c in corrections for hl in [c.pop("_hl")]]
    return corrected_text, corrections, highlights


# ---------------------------------------------------------------------------
# Spelling via TextBlob (dictionary-gated to avoid mangling proper nouns)
# ---------------------------------------------------------------------------
_SPELL_VOCAB = None
_SPELL_CACHE = {}


def _vocab():
    global _SPELL_VOCAB
    if _SPELL_VOCAB is None:
        from nltk.corpus import words as words_corpus
        _SPELL_VOCAB = {w.lower() for w in words_corpus.words()}
    return _SPELL_VOCAB


# uncountable nouns that are commonly (and wrongly) given plural forms
UNCOUNTABLE_SPELL = {"homework", "information", "advice", "furniture",
                     "equipment", "luggage", "baggage", "feedback",
                     "research", "software", "hardware", "housework"}


def spell_check(tokens, tagged, span_owner):
    """Suggest spellings for lowercase dictionary-unknown words."""
    corrections = []
    vocab = _vocab()
    for i, tok in enumerate(tokens):
        if i in span_owner:
            continue
        if not tok.isalpha() or len(tok) < 4 or tok.isupper():
            continue
        low = tok.lower()
        if low in vocab:
            continue
        tag = tagged[i][1]
        if tag in ("NNP", "NNPS"):
            continue  # proper noun — leave it alone
        # spelling suggestions must never singularize a legit plural
        is_plural = any(
            low.endswith(ending) and len(low) > len(ending)
            and (low[:-len(ending)] in vocab
                 or (ending == "ies" and low[:-len(ending)] + "y" in vocab))
            for ending in ("s", "es", "ies")
        )
        if is_plural:
            # except wrong plurals of uncountable nouns: "homeworks"->"homework"
            for u in UNCOUNTABLE_SPELL:
                if low == u + "s" or low == u + "es":
                    corrections.append(dict(
                        tokenIndex=i, tokenCount=1, original=tok,
                        suggestion=u, rule="spelling",
                        message=f"'{u}' is uncountable — it has no plural form."))
                    break
            continue
        sug = _spell_suggest(low)
        if sug and sug != low:
            corrections.append(dict(
                tokenIndex=i, tokenCount=1, original=tok, suggestion=sug,
                rule="spelling",
                message=f"Possible misspelling of '{sug}'."))
    return corrections


def _spell_suggest(word: str) -> str | None:
    """TextBlob spellcheck, cached, gated by confidence."""
    if word in _SPELL_CACHE:
        return _SPELL_CACHE[word]
    sug = None
    try:
        from textblob import TextBlob
        candidates = TextBlob(word).words[0].spellcheck()
        if candidates:
            best, conf = candidates[0]
            if conf >= 0.7 and best.lower() != word and best.lower() in _vocab():
                sug = best.lower()
    except Exception:
        sug = None
    _SPELL_CACHE[word] = sug
    return sug


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(text: str) -> dict:
    ensure_nltk_data()
    text = text[:MAX_TEXT_LEN]

    model_pack = load_model()
    ml = predict(model_pack, text) if text.strip() else {}

    corrected, corrections, highlights = correct_text(text)

    # ---- tokenization / stemming / lemmatization table (on corrected text)
    rows = []
    for sent in tokenize_sentences(corrected):
        toks = tokenize_words(sent)
        for tok, tag in pos_tag_tokens(toks):
            if not re.search(r"[A-Za-z0-9]", tok):
                continue  # skip bare punctuation rows in the UI table
            rows.append({
                "token": tok,
                "pos": tag,
                "stem": stem_word(tok),
                "lemma": lemmatize_word(tok, tag),
            })
            if len(rows) >= MAX_TABLE_TOKENS:
                break
        if len(rows) >= MAX_TABLE_TOKENS:
            break

    # ---- NER + sentiment on the corrected text -------------------------
    try:
        entities = extract_entities(corrected)
    except Exception:
        entities = []
    sentiment = analyze_sentiment(corrected)
    stats = text_stats(corrected)

    # ---- group corrections for the UI ----------------------------------
    for c in corrections:
        c["ruleLabel"] = RULE_LABELS.get(c["rule"], c["rule"])

    # ---- writing-quality analytics (extra features) ----------------------
    style = style_suggestions(corrected)
    quality = grammar_score(
        n_corrections=len([c for c in corrections
                           if c["rule"] != "sentence-fragment"]),
        n_style=len(style),
        word_count=stats["wordCount"],
    )

    return {
        "correctedText": corrected,
        "corrections": corrections,
        "highlights": highlights,
        "pipeline": {"tokens": rows, "truncated": len(rows) >= MAX_TABLE_TOKENS},
        "entities": entities,
        "sentiment": sentiment,
        "stats": stats,
        "classifier": ml,
        "quality": quality,
        "readability": readability(corrected),
        "style": style,
    }


if __name__ == "__main__":
    try:
        req = json.loads(sys.stdin.read() or "{}")
        result = run(req.get("text", ""))
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:  # last-resort error contract for the API route
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
