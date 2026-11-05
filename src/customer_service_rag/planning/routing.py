"""Normalize model-provided routing labels without using dataset metadata."""

DOMAINS = {
    "ssa": "ssa", "social security": "ssa", "social_security": "ssa",
    "dmv": "dmv", "driving license": "dmv", "va": "va", "veterans": "va",
    "studentaid": "studentaid", "student aid": "studentaid", "student_aid": "studentaid",
}


def normalize_domain(label: str) -> str:
    return DOMAINS.get(label.casefold().strip(), "unknown")


def route_visible_dialogue(message: str, history: list[dict]) -> str:
    cues = {
        "dmv": ("dmv", "vehicle", "driver", "license", "atv"),
        "ssa": ("social security", "medicare", "retirement", "survivor"),
        "va": ("veteran", "veterans affairs", "gi bill", "va benefit"),
        "studentaid": ("student aid", "student loan", "fafsa", "federal student"),
    }
    texts = [message.casefold(), " ".join(item["text"] for item in history).casefold()]
    scores = {
        domain: sum(3 * (cue in texts[0]) + (cue in texts[1]) for cue in terms)
        for domain, terms in cues.items()
    }
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[0] if scores[ranked[0]] > scores[ranked[1]] else "unknown"
