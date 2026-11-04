"""Edge-case sweep — probe the engine with tricky inputs and print results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from engine import run  # noqa: E402

CASES = [
    # (label, input, must_contain substrings in corrected text, must_not_contain)
    ("fragment + capital i", "because i was tired. i slept.",
     ["Because I was tired", "I slept"], []),
    ("comma but short phrase", "He is small but strong and fast.",
     ["He is small but strong"], [", but"]),
    ("comma but two clauses", "I like tea but he likes coffee.",
     ["tea, but he likes"], []),
    ("much + irregular plural", "The teacher gave us much homeworks and much assignments.",
     ["many assignments"], ["much assignments"]),
    ("dont after singular", "It dont work and she dont care.",
     ["It doesn't work", "she doesn't care"], []),
    ("doesnt after plural", "They doesnt know the answer.",
     ["They don't know"], []),
    ("doesnt + VBZ", "He doesn't likes pizza.",
     ["doesn't like"], ["likes"]),
    ("modal + VBZ", "She can plays the violin very well.",
     ["can play"], ["plays"]),
    ("your welcome gift (false pos)", "Your welcome gift arrived this morning.",
     ["Your welcome gift"], ["you're"]),
    ("your welcome (true pos)", "Thanks for the help. Your welcome!",
     ["You're welcome"], []),
    ("alot / aswell", "I like it alot and the food is good aswell.",
     ["a lot", "as well"], ["alot", "aswell"]),
    ("clean text untouched", "She walks to school every day and studies hard at night.",
     ["She walks to school every day and studies hard at night."], []),
    ("proper noun safe", "Samsung unveiled Samsung Galaxy in Seoul on Monday.",
     ["Samsung"], []),
    ("emoji & numbers", "I love this product!!! 10/10 would recommend to everyone :)",
     ["love"], []),
    ("if I were subjunctive", "If I were richer, I would travel the world.",
     ["If I were richer"], ["If I was"]),
    ("he goes correct", "He goes to the gym every morning before work.",
     ["He goes to the gym"], []),
    ("an hour / a university", "It took an hour and a university is nearby.",
     ["an hour", "a university"], []),
    ("two men irregular plural context", "There is many children in the park.",
     ["There are many children"], []),
    ("a apples double fix", "She ate a apples and two banana.",
     ["an apple", "two bananas"], []),
    ("long clean sentence", "The researchers published their findings in a peer-reviewed journal last March after months of careful verification.",
     ["peer-reviewed journal"], []),
    ("punct only", "!!!", ["!"], []),
    ("single word", "Hello", ["Hello"], []),
]

failures = 0
for label, text, musts, must_nots in CASES:
    r = run(text)
    out = r["correctedText"]
    problems = [f"missing {m!r}" for m in musts if m not in out] + \
               [f"should NOT contain {m!r}" for m in must_nots if m in out]
    status = "OK " if not problems else "FAIL"
    if problems:
        failures += 1
    print(f"[{status}] {label}\n     -> {out}")
    for p in problems:
        print(f"     !! {p}")
print(f"\n{failures} failing cases out of {len(CASES)}")
sys.exit(1 if failures else 0)
