"""English answer metrics using SQuAD-style normalization."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not common:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def response_scores(prediction: str, references: list[str], answered: bool) -> dict:
    if not references:
        return {"f1": None, "em": None}
    if not answered:
        return {"f1": 0.0, "em": 0.0}
    return {
        "f1": max(token_f1(prediction, reference) for reference in references),
        "em": max(exact_match(prediction, reference) for reference in references),
    }
