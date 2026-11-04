"""
grammar_rules.py — Rule-based grammar & spelling correction engine ("Full Suite").

Each rule function scans ONE sentence (already tokenized + POS-tagged) and
appends Correction dicts to a shared list. The engine (engine.py) applies the
non-conflicting corrections and rebuilds the corrected text.

Correction dict format:
    {
        "tokenIndex": int,   # index of first token in the sentence
        "tokenCount": int,   # how many tokens the correction spans
        "original":   str,   # original text of the span
        "suggestion": str,   # replacement text ("" = delete span)
        "rule":       str,   # rule id (shown as badge in the UI)
        "message":    str,   # human-readable explanation
    }

Rules covered (Full Suite):
    1.  capitalization          – sentence start, standalone "i"
    2.  repeated-words          – "the the"
    3.  subject-verb-agreement  – "he go", "they is", "we was", "dogs runs"
    4.  irregular-past-tense    – "buyed" -> "bought"
    5.  article-agreement       – "a apple" -> "an apple"
    6.  number-noun-agreement   – "two apple" -> "two apples", "much people"
    7.  confusion-pairs         – its/it's, your/you're, then/than, should of
    8.  preposition-usage       – "discuss about", "return back", "married with"
    9.  missing-contractions    – "dont" -> "don't"
    10. punctuation             – comma before "but", missing terminal "."
    11. sentence-fragment       – no verb / subordinate clause alone
    12. spelling                – TextBlob spell-check on unknown words
"""

import re

from nlp_utils import lemmatize_word, pos_tag_tokens, tokenize_words

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# 4. Common wrongly-formed irregular past tense -> correct form
IRREGULAR_PAST = {
    "buyed": "bought", "goed": "went", "runned": "ran", "eated": "ate",
    "catched": "caught", "thinked": "thought", "teached": "taught",
    "bringed": "brought", "falled": "fell", "finded": "found",
    "gived": "gave", "knowed": "knew", "maked": "made", "taked": "took",
    "writed": "wrote", "comed": "came", "breaked": "broke",
    "choosed": "chose", "drawed": "drew", "drived": "drove",
    "flied": "flew", "growed": "grew", "holded": "held", "layed": "laid",
    "leaved": "left", "losed": "lost", "meeted": "met", "payed": "paid",
    "readed": "read", "sended": "sent", "sleeped": "slept",
    "speaked": "spoke", "standed": "stood", "swimmed": "swam",
    "telled": "told", "understanded": "understood", "winned": "won",
    "heared": "heard", "feeled": "felt", "builded": "built",
    "cutted": "cut", "hitted": "hit", "hurted": "hurt", "letted": "let",
    "putted": "put", "selled": "sold", "shaked": "shook",
    "shooted": "shot", "shutted": "shut", "stoled": "stole",
    "striked": "struck", "beginned": "began", "ringed": "rang",
    "dranked": "drank", "singed": "sang", "spended": "spent",
}

# 3. Verb conjugation helpers
_KEEP_SINGULAR = {"has", "does", "goes", "was", "is"}


def to_3rd_person(verb: str) -> str:
    """walk->walks, watch->watches, carry->carries, go->goes, have->has."""
    if verb in ("have",):
        return "has"
    if verb in ("be",):
        return "is"
    if re.search(r"(ch|sh|x|z|s|o)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


def to_base_form(verb: str) -> str:
    """walks->walk, watches->watch, misses->miss, carries->carry, goes->go, has->have."""
    irregular = {"has": "have", "does": "do", "goes": "go", "was": "be",
                 "is": "be", "am": "be", "aren't": "be"}
    if verb in irregular:
        return irregular[verb]
    if re.search(r"ies$", verb) and len(verb) > 4:
        return verb[:-3] + "y"
    if re.search(r"(ch|sh|x|z|s|o)es$", verb):
        return verb[:-2]
    if re.search(r"s$", verb):
        return verb[:-1]
    return verb


# 6. Number words that require a plural noun
NUMBER_WORDS = {"two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "many", "several", "few", "both", "numerous",
                "various", "hundred", "thousand", "million"}

IRREGULAR_PLURAL = {
    "child": "children", "man": "men", "woman": "women",
    "person": "people", "foot": "feet", "tooth": "teeth",
    "mouse": "mice", "goose": "geese", "die": "dice",
}

IRREGULAR_SINGULAR = {v: k for k, v in IRREGULAR_PLURAL.items()}

UNCOUNTABLE = {"money", "water", "furniture", "information", "advice",
               "rice", "sugar", "equipment", "homework", "news", "people",
               "police", "sheep", "fish", "series", "species", "aircraft"}

# "much" only fits uncountables; flag it before common countable nouns
COUNTABLE_COMMON = {"people", "students", "books", "cars", "things",
                    "ideas", "questions", "times", "friends", "words",
                    "options", "problems", "reasons", "days", "years",
                    "hours", "minutes", "ways", "places"}

# words starting with pronounced "yu" sound -> take "a", not "an"
_YU_SOUND = ("uni", "use", "usu", "euro", "one", "ubiq", "ukr")

SILENT_H = {"hour", "hourly", "honest", "honestly", "honesty", "heir",
            "honour", "honor", "herb"}

# 8. Wordy / wrong preposition phrases: (span, replacement, message)
PHRASE_FIXES = [
    (("discuss", "about"), ("discuss",), "'discuss' is already transitive; drop 'about'"),
    (("discusses", "about"), ("discusses",), "'discuss' is already transitive; drop 'about'"),
    (("discussed", "about"), ("discussed",), "'discuss' is already transitive; drop 'about'"),
    (("return", "back"), ("return",), "'back' is redundant after 'return'"),
    (("returned", "back"), ("returned",), "'back' is redundant after 'return'"),
    (("revert", "back"), ("revert",), "'back' is redundant after 'revert'"),
    (("repeat", "again"), ("repeat",), "'again' is redundant after 'repeat'"),
    (("married", "with"), ("married", "to"), "the standard collocation is 'married to'"),
    (("comprises", "of"), ("comprises",), "'comprise' takes a direct object; drop 'of'"),
    (("each", "and", "every"), ("every",), "'each and every' is wordy; use 'every'"),
    (("due", "to", "the", "fact", "that"), ("because",), "wordy; use 'because'"),
    (("in", "spite", "of", "the", "fact", "that"), ("although",), "wordy; use 'although'"),
    (("very", "unique"), ("unique",), "'unique' is absolute; it cannot be graded"),
    (("good", "in"), ("good", "at"), "the standard collocation is 'good at'"),
]

# 9. Missing apostrophes in common contractions
CONTRACTIONS = {
    "dont": "don't", "cant": "can't", "wont": "won't", "isnt": "isn't",
    "arent": "aren't", "wasnt": "wasn't", "werent": "weren't",
    "didnt": "didn't", "doesnt": "doesn't", "hadnt": "hadn't",
    "hasnt": "hasn't", "havent": "haven't", "couldnt": "couldn't",
    "shouldnt": "shouldn't", "wouldnt": "wouldn't", "mustnt": "mustn't",
    "youre": "you're", "theyre": "they're", "theres": "there's",
    "thats": "that's", "whats": "what's", "heres": "here's",
}

# words that may start a subordinate clause (fragment hint)
_SUBORDINATING = {"because", "although", "since", "while", "whereas",
                  "unless", "whenever", "wherever", "even", "if"}

# common base verbs for the singular-noun + verb agreement rule
_COMMON_VERBS = {"run", "go", "eat", "write", "read", "play", "work",
                 "make", "take", "give", "know", "think", "come", "want",
                 "need", "like", "love", "say", "see", "use", "find",
                 "tell", "ask", "seem", "feel", "try", "leave", "call",
                 "help", "talk", "stop", "start", "live", "believe", "hold"}


def _is_vowel_sound(word: str) -> bool:
    """Heuristic: does the word start with a vowel SOUND?"""
    w = word.lower()
    if not w:
        return False
    if w in SILENT_H:
        return True
    if w.startswith(_YU_SOUND):
        return False          # "a university", "a European"
    if re.match(r"^(8|11|18)\b", w):   # "an 8", "an 11", "an 18"
        return True
    return w[0] in "aeiou"


def to_plural(noun: str) -> str:
    """apple->apples, box->boxes, baby->babies, child->children."""
    low = noun.lower()
    if low in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[low]
    if re.search(r"(s|x|z|ch|sh)$", low):
        return noun + "es"
    if re.search(r"[^aeiou]y$", low):
        return noun[:-1] + "ies"
    return noun + "s"


def to_singular(noun: str) -> str:
    """apples->apple, babies->baby, children->child."""
    low = noun.lower()
    if low in IRREGULAR_SINGULAR:
        return IRREGULAR_SINGULAR[low]
    if re.search(r"ies$", low):
        return noun[:-3] + "y"
    if re.search(r"(ches|shes|xes|zes|ses)$", low):
        return noun[:-2]
    if re.search(r"s$", low):
        return noun[:-1]
    return noun


# ---------------------------------------------------------------------------
# Rule functions — each appends Correction dicts
# ---------------------------------------------------------------------------

def rule_capitalization(tokens, tagged, out):
    """1. Capitalize the first alphabetic word of a sentence and standalone 'i'.

    Marked case_only=True: these fixes are applied AFTER all other rules so
    they never block a correction on the same token (e.g. 'a apple' ->
    'An apple', not just 'A apple').
    """
    for i, tok in enumerate(tokens):
        if tok[:1].isalpha():
            if tok[:1].islower():
                out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                                suggestion=tok.capitalize(),
                                rule="capitalization", case_only=True,
                                message="Sentences should start with a capital letter."))
            break  # only the first alphabetic token matters
    for i, tok in enumerate(tokens):
        if tok == "i":
            out.append(dict(tokenIndex=i, tokenCount=1, original="i",
                            suggestion="I", rule="capitalization", case_only=True,
                            message="The pronoun 'I' is always capitalized."))


def rule_repeated_words(tokens, tagged, out):
    """2. Consecutive duplicate words: 'the the', 'is is'."""
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if (a.lower() == b.lower() and a.isalpha()
                and a.lower() not in {"had", "that", "very", "no", "ten"}):
            out.append(dict(tokenIndex=i + 1, tokenCount=1, original=b,
                            suggestion="", rule="repeated-words",
                            message=f"Repeated word: '{a}' appears twice in a row."))


# copulas must be handled by their own special rules, never conjugated
_COPULAS = {"is", "are", "was", "were", "am", "be", "been", "being"}

# object-pronoun -> subject-pronoun ("john and him goes" -> "he goes")
SUBJECT_PRONOUN = {"him": "he", "her": "she", "them": "they",
                   "me": "I", "us": "we"}

# subject-pronoun -> object-pronoun after a preposition ("with we" -> "with us")
OBJECT_PRONOUN = {"i": "me", "we": "us", "they": "them",
                  "he": "him", "she": "her"}

_PREPS = {"with", "for", "to", "at", "by", "from", "behind", "beside",
          "between", "without", "near", "towards", "against", "about"}


def match_case(original: str, suggestion: str) -> str:
    """Preserve the original token's capitalization on the suggestion."""
    if original[:1].isupper() and suggestion[:1].isalpha():
        return suggestion[0].upper() + suggestion[1:]
    return suggestion


def rule_subject_verb_agreement(tokens, tagged, out):
    """3. Pronoun/noun <-> verb agreement."""
    for i, (tok, tag) in enumerate(tagged):
        low = tok.lower()
        prev = tokens[i - 1].lower() if i > 0 else ""
        prev_tag = tagged[i - 1][1] if i > 0 else ""

        # --- copulas first (was/were/is/are) — never conjugate these ----
        if low == "was" and prev in ("they", "we", "you"):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="were", rule="subject-verb-agreement",
                            message=f"'{prev.capitalize()}' is plural: use 'were'."))
        elif (low == "were" and prev == "i"
              and not (i >= 2 and tokens[i - 2].lower() == "if")):
            # keep the subjunctive "if I were ..." untouched
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="was", rule="subject-verb-agreement",
                            message="'I was' is the standard past form."))
        elif low == "were" and prev in ("he", "she", "it"):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="was", rule="subject-verb-agreement",
                            message=f"'{prev.capitalize()} was' is correct in the past."))
        elif low == "is" and prev in ("they", "we", "you"):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="are", rule="subject-verb-agreement",
                            message=f"'{prev.capitalize()}' is plural: use 'are'."))
        elif low == "are" and prev in ("he", "she", "it"):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="is", rule="subject-verb-agreement",
                            message=f"'{prev.capitalize()}' is singular: use 'is'."))
        elif low == "am" and prev not in ("i", "", "not") and prev_tag not in ("PRP", "CC"):
            pass  # 'am' after a non-'I' subject — too risky, skip

        # --- I + VBZ  ->  am / base form -------------------------------
        elif prev == "i" and tag == "VBZ":
            base = to_base_form(low)
            fix = "am" if low == "is" else base
            if fix != low:
                out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                                suggestion=fix, rule="subject-verb-agreement",
                                message=f"'I' takes the base form; use '{fix}'."))

        # --- he/she/it + plural verb  ->  3rd person -------------------
        elif (prev in ("he", "she", "it") and tag in ("VB", "VBP")
              and low not in _KEEP_SINGULAR and low not in _COPULAS):
            fix = to_3rd_person(low)
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=fix, rule="subject-verb-agreement",
                            message=f"'{prev.capitalize()}' is singular: use '{fix}'."))

        # --- they/we/you + 3rd-person verb  ->  base form --------------
        elif prev in ("they", "we", "you") and tag == "VBZ":
            fix = to_base_form(low)
            if fix != low:
                out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                                suggestion=fix, rule="subject-verb-agreement",
                                message=f"'{prev.capitalize()}' is plural: use '{fix}'."))

        # --- plural noun + 3rd-person verb  ->  base -------------------
        elif (tag == "VBZ" and i > 0 and prev_tag == "NNS"
              and low not in _KEEP_SINGULAR):
            fix = to_base_form(low)
            if fix != low:
                out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                                suggestion=fix, rule="subject-verb-agreement",
                                message=f"'{tokens[i-1]}' is plural: use '{fix}'."))

        # --- singular noun + base verb (common verbs only) -------------
        # (skip when the 'noun' is actually a mistyped contraction like
        #  'dont' — the tagger often labels those NN)
        elif (tag == "VBP" and i > 0 and prev_tag == "NN"
              and low in _COMMON_VERBS and low not in _COPULAS
              and prev not in CONTRACTIONS and prev not in SOLID_FIXES
              and not any(w in ("and", "or", "but") for w in tokens[max(0, i - 2):i])):
            fix = to_3rd_person(low)
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=fix, rule="subject-verb-agreement",
                            message=f"'{tokens[i-1]}' is singular: use '{fix}'."))

        # --- object pronoun used as subject ("...and him goes") --------
        elif (low in SUBJECT_PRONOUN and i + 1 < len(tagged)
              and tagged[i + 1][1].startswith("VB")
              and (i == 0 or tokens[i - 1].lower() == "and")):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=SUBJECT_PRONOUN[low],
                            rule="pronoun-case",
                            message=f"Use the subject pronoun '{SUBJECT_PRONOUN[low]}' "
                                    f"before a verb."))

        # --- subject pronoun after a preposition ("with we") -----------
        elif (low in OBJECT_PRONOUN and i > 0 and prev in _PREPS):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=OBJECT_PRONOUN[low],
                            rule="pronoun-case",
                            message=f"After the preposition '{prev}', use the "
                                    f"object pronoun '{OBJECT_PRONOUN[low]}'."))

        # --- negative contractions: it dont -> it doesn't ----------------
        # (the subject may sit across a conjunction: "he ... and dont go")
        elif low == "dont" and _effective_subject(tokens, i) in ("he", "she", "it"):
            subj = _effective_subject(tokens, i)
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="doesn't", rule="subject-verb-agreement",
                            message=f"'{subj.capitalize()}' is singular: use 'doesn't'."))
        elif low == "doesnt" and _effective_subject(tokens, i) in ("they", "we", "you", "i"):
            subj = _effective_subject(tokens, i)
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="don't", rule="subject-verb-agreement",
                            message=f"'{subj.capitalize()}' takes the plural form 'don't'."))

        # --- verb after negated auxiliary: "doesn't likes" -> "like" ----
        elif (tag in ("VB", "VBP", "VBZ") and i >= 2
              and tokens[i - 1].lower() in ("n't", "not")
              and tokens[i - 2].lower() in ("does", "do", "did")):
            fix = to_base_form(low)
            if fix != low:
                out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                                suggestion=fix, rule="subject-verb-agreement",
                                message=f"After '{tokens[i-2]}{tokens[i-1]}' use "
                                        f"the base form '{fix}'."))

        # --- verb after a modal: "can plays" -> "can play" --------------
        elif (tag in ("VB", "VBP", "VBZ") and i > 0
              and prev_tag.startswith("MD")):
            fix = to_base_form(low)
            if fix != low:
                out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                                suggestion=fix, rule="subject-verb-agreement",
                                message=f"After the modal '{tokens[i-1]}' use "
                                        f"the base form '{fix}'."))


def rule_irregular_past(tokens, tagged, out):
    """4. Wrong irregular past tense forms: 'buyed' -> 'bought'."""
    for i, tok in enumerate(tokens):
        fix = IRREGULAR_PAST.get(tok.lower())
        if fix:
            base = lemmatize_word(fix, "VBD")  # lemma of the correct form
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=fix, rule="irregular-past-tense",
                            message=f"'{tok}' is not a word; the past tense of "
                                    f"'{base}' is '{fix}'."))


def rule_articles(tokens, tagged, out):
    """5. a/an agreement with the following word's sound."""
    for i, tok in enumerate(tokens[:-1]):
        low = tok.lower()
        if low not in ("a", "an"):
            continue
        nxt = tokens[i + 1]
        if not re.search(r"[A-Za-z]", nxt):
            continue
        if low == "a" and _is_vowel_sound(nxt):
            fix = "An" if tok[0].isupper() else "an"
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=fix, rule="article-agreement",
                            message=f"Use '{fix.lower()}' before the vowel sound "
                                    f"in '{nxt}'."))
        elif low == "an" and not _is_vowel_sound(nxt):
            fix = "A" if tok[0].isupper() else "a"
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=fix, rule="article-agreement",
                            message=f"Use '{fix.lower()}' before the consonant "
                                    f"sound in '{nxt}'."))


def rule_number_noun(tokens, tagged, out):
    """6. Number words need plural nouns; 'much' vs 'many'; a/an + plural."""
    for i, tok in enumerate(tokens[:-1]):
        low = tok.lower()
        nxt, nxt_tag = tokens[i + 1], tagged[i + 1][1]

        # two apple -> two apples
        if (low in NUMBER_WORDS and nxt_tag == "NN"
                and nxt.lower() not in UNCOUNTABLE):
            fix = to_plural(nxt)
            if fix != nxt:
                out.append(dict(tokenIndex=i + 1, tokenCount=1, original=nxt,
                                suggestion=fix, rule="number-noun-agreement",
                                message=f"'{low}' requires a plural noun: '{fix}'."))

        # much people -> many people (plurals always take 'many')
        elif low == "much" and (nxt.lower() in COUNTABLE_COMMON or nxt_tag == "NNS"):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="many", rule="number-noun-agreement",
                            message=f"Use 'many' with countable plural nouns like '{nxt}'."))

        # a apples -> an apple
        elif low in ("a", "an") and nxt_tag == "NNS":
            fix = to_singular(nxt)
            out.append(dict(tokenIndex=i + 1, tokenCount=1, original=nxt,
                            suggestion=fix, rule="number-noun-agreement",
                            message=f"'{low}' is singular, so use the singular noun '{fix}'."))


def rule_confusion_pairs(tokens, tagged, out):
    """7. its/it's, your/you're, then/than, should of, there is/are..."""
    for i, tok in enumerate(tokens):
        low = tok.lower()
        nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
        nxt_tag = tagged[i + 1][1] if i + 1 < len(tokens) else ""

        # its + determiner -> it's  ("its a nice day"; keep "its tail")
        if low == "its" and nxt in ("a", "an", "the", "my", "his", "her",
                                    "their", "our", "not", "so", "very",
                                    "been", "going", "important", "time",
                                    "possible", "clear"):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=match_case(tok, "it's"),
                            rule="confusion-pairs",
                            message="'it's' means 'it is' or 'it has' here."))

        # your welcome -> you're welcome (only as the response, not
        # "your welcome gift" — welcome must end the clause)
        elif (low == "your" and nxt == "welcome"
              and (i + 2 >= len(tokens)
                   or not re.search(r"[A-Za-z0-9]", tokens[i + 2]))):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=match_case(tok, "you're"),
                            rule="confusion-pairs",
                            message="'you're' = 'you are': \"you're welcome\"."))

        # should of -> should have
        elif low in ("should", "could", "would", "must", "might") and nxt == "of":
            out.append(dict(tokenIndex=i + 1, tokenCount=1, original="of",
                            suggestion="have", rule="confusion-pairs",
                            message=f"'{low} of' is incorrect; use '{low} have'."))

        # better then -> better than
        elif low == "then" and (i > 0 and tokens[i - 1].lower() in
                                ("better", "worse", "more", "less", "rather",
                                 "other", "bigger", "smaller", "faster", "higher")):
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion="than", rule="confusion-pairs",
                            message="'than' is used for comparisons."))

        # there is many/there was two -> there are/there were
        elif (low in ("is", "was") and i > 0
              and tokens[i - 1].lower() == "there"
              and (nxt_tag in ("NNS",) or nxt in NUMBER_WORDS)):
            fix = "are" if low == "is" else "were"
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=fix, rule="subject-verb-agreement",
                            message=f"'{nxt}' is plural: use 'there {fix}'."))

        # more better / most best -> double comparative
        elif low in ("better", "best") and i > 0 and tokens[i - 1].lower() in ("more", "most"):
            out.append(dict(tokenIndex=i - 1, tokenCount=1,
                            original=tokens[i - 1], suggestion="",
                            rule="confusion-pairs",
                            message=f"'{low}' is already comparative/superlative; "
                                    f"drop '{tokens[i-1].lower()}'."))


def rule_prepositions(tokens, tagged, out):
    """8. Wrong / wordy preposition phrases."""
    lowered = [t.lower() for t in tokens]
    for phrase, fix, msg in PHRASE_FIXES:
        n = len(phrase)
        for i in range(len(lowered) - n + 1):
            if tuple(lowered[i:i + n]) == phrase:
                out.append(dict(tokenIndex=i, tokenCount=n,
                                original=" ".join(tokens[i:i + n]),
                                suggestion=" ".join(fix), rule="preposition-usage",
                                message=msg))


# 9a. Solid (no-space) typos -> correct multi-word forms
SOLID_FIXES = {
    "alot": "a lot", "aswell": "as well", "eachother": "each other",
    "incase": "in case", "inspite": "in spite", "nofwhere": "nowhere",
}


def rule_contractions(tokens, tagged, out):
    """9. Missing apostrophes + solid typos: 'dont' -> \"don't\", 'alot' -> 'a lot'."""
    for i, tok in enumerate(tokens):
        low = tok.lower()
        fix = CONTRACTIONS.get(low) or SOLID_FIXES.get(low)
        if fix:
            out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                            suggestion=fix,
                            rule="missing-contraction" if low in CONTRACTIONS else "spelling",
                            message=(f"Missing apostrophe: '{tok}' should be '{fix}'."
                                     if low in CONTRACTIONS else
                                     f"'{tok}' should be written as '{fix}'.")))


def _has_verb(tagged, lo, hi) -> bool:
    return any(t.startswith("VB") or t.startswith("MD") for _, t in tagged[lo:hi])


def rule_punctuation(tokens, tagged, out):
    """10. Comma before 'but' only when it joins two verb-containing clauses."""
    for i, tok in enumerate(tokens):
        if (tok.lower() == "but" and i >= 2 and i < len(tokens) - 2
                and _has_verb(tagged, 0, i) and _has_verb(tagged, i + 1, len(tagged))):
            prev = tokens[i - 1]
            if re.search(r"[A-Za-z0-9]", prev):
                out.append(dict(tokenIndex=i, tokenCount=1, original=tok,
                                suggestion=", but", rule="punctuation",
                                message="A comma goes before 'but' when it "
                                        "joins two clauses."))


def rule_fragment(tokens, tagged, out):
    """11. Sentence-fragment detection (flag only, no replacement)."""
    if not tokens:
        return
    verbs = [t for _, t in tagged if t.startswith("VB") or t.startswith("MD")]
    words = [t for t in tokens if t.isalpha()]
    if len(words) >= 3 and not verbs:
        out.append(dict(tokenIndex=0, tokenCount=len(tokens),
                        original=" ".join(tokens), suggestion="",
                        rule="sentence-fragment",
                        message="Possible fragment: this sentence has no verb."))
    elif tokens[0].lower() in _SUBORDINATING and "," not in tokens:
        out.append(dict(tokenIndex=0, tokenCount=len(tokens),
                        original=" ".join(tokens), suggestion="",
                        rule="sentence-fragment",
                        message=f"Possible fragment: starts with '{tokens[0]}' "
                                f"but the main clause seems missing."))


# Ordered rule pipeline — earlier rules win when two touch the same token
RULE_PIPELINE = [
    rule_capitalization,
    rule_repeated_words,
    rule_irregular_past,
    rule_subject_verb_agreement,
    rule_articles,
    rule_number_noun,
    rule_confusion_pairs,
    rule_prepositions,
    rule_contractions,
    rule_punctuation,
    rule_fragment,
]

RULE_LABELS = {
    "capitalization": "Capitalization",
    "repeated-words": "Repeated Word",
    "irregular-past-tense": "Tense",
    "subject-verb-agreement": "Agreement",
    "article-agreement": "Article",
    "number-noun-agreement": "Number",
    "confusion-pairs": "Word Choice",
    "preposition-usage": "Preposition",
    "missing-contraction": "Apostrophe",
    "punctuation": "Punctuation",
    "pronoun-case": "Pronoun Case",
    "sentence-fragment": "Fragment",
    "spelling": "Spelling",
}


def _clause_subject(tokens: list[str], i: int) -> str | None:
    """
    Find the subject governing token i, walking backwards over conjunctions
    and commas. Needed for shared-subject clauses: in "he bought two apples
    AND dont go home", the subject of 'dont' is 'he', two tokens back.
    Returns the lowercase subject token, or None.
    """
    j = i - 1
    while j >= 0 and tokens[j].lower() in ("and", "or", "but", ","):
        j -= 1
    return tokens[j].lower() if j >= 0 else None


_PERSON_PRONOUNS = {"he", "she", "it", "they", "we", "you", "i"}


def _effective_subject(tokens: list[str], i: int) -> str | None:
    """
    Resolve who the subject of token i really is.

    1. nearest candidate over a conjunction ("it dont work" -> 'it')
    2. shared-subject fallback: if that candidate is not a pronoun but the
       sentence opened with one ("HE bought two apples and dont go"), the
       opening pronoun governs the whole sentence.
    """
    subj = _clause_subject(tokens, i)
    if subj in _PERSON_PRONOUNS:
        return subj
    # fallback: sentence-initial pronoun + a conjunction between it and i
    if tokens and tokens[0].lower() in _PERSON_PRONOUNS and any(
        t.lower() in ("and", "or", "but") for t in tokens[1:i]
    ):
        return tokens[0].lower()
    return subj


def analyze_sentence(sentence: str) -> tuple[list[str], list[tuple[str, str]], list[dict]]:
    """Tokenize + tag one sentence and run every rule over it."""
    tokens = tokenize_words(sentence)
    tagged = pos_tag_tokens(tokens)
    out = []
    for rule_fn in RULE_PIPELINE:
        try:
            rule_fn(tokens, tagged, out)
        except Exception:
            continue  # one bad match must never break the whole pipeline
    return tokens, tagged, out
