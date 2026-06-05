from collections import Counter
import re

def score_document(text):

    words = text.split()

    length_score = min(len(words) / 100, 1.0)

    repetition_score = 1 - (
        max(Counter(words).values()) / max(len(words), 1)
    )

    special_count = len(
        re.findall(r"[^a-zA-Z0-9\s]", text)
    )

    noise_score = 1 - (
        special_count / max(len(text), 1)
    )

    final_score = (
        0.4 * length_score +
        0.3 * repetition_score +
        0.3 * noise_score
    )

    return round(final_score, 3)

docs = [
    "This is a high quality article about machine learning.",
    "buy buy buy buy buy now now now",
    "@@@###$$$%%%%"
]

for d in docs:
    print(score_document(d), d)
