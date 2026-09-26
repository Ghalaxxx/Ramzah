"""Conservative concept filter, not a sentence generator or clinical validator."""

import re
import unicodedata


def normalize_arabic(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub("[\u064b-\u065f\u0670ـ]", "", text)
    return text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})).strip()


def concepts_in(text, concepts):
    return set(concept_counts(text, concepts))


def concept_counts(text, concepts):
    text = normalize_arabic(text)
    counts = {}
    for concept, variants in concepts.items():
        spans = set()
        for v in variants:
            # Arabic word boundary, optional conjunction for coordinated phrases.
            pattern = (
                r"(?<![\u0621-\u064a])[وفبكل]?"
                + re.escape(normalize_arabic(v))
                + r"(?![\u0621-\u064a])"
            )
            spans.update(match.span() for match in re.finditer(pattern, text))
        if spans:
            counts[concept] = len(spans)
    return counts


def validate_candidate(words, sentence, concepts):
    expected = set(words)
    counts = concept_counts(sentence, concepts)
    found = set(counts)
    text = normalize_arabic(sentence)
    forbidden = [
        "نوبة",
        "ازمة قلبية",
        "سرطان",
        "تشخيص",
        "جرعة",
        "حبة",
        "ملغ",
        "انتحار",
        "وفاة",
        "عملية جراحية",
    ]
    malformed = ["لدي عيادة", "لدي موعدي", "احتاج الى للطبيب"]
    reasons = []
    if not sentence.strip():
        reasons.append("empty")
    if found != expected:
        reasons.append("concept_mismatch")
    repeated = sorted(
        concept for concept, count in counts.items() if count > words.count(concept)
    )
    if repeated:
        reasons.append("repeated_concept")
    if any(x in text for x in forbidden):
        reasons.append("unsupported_clinical_claim")
    if any(x in text for x in malformed):
        reasons.append("malformed_generation")
    if re.search(r"\d", text):
        reasons.append("invented_number")
    if re.search(r"(?<![\u0621-\u064a])(?:لا|ليس|لم|لن|بدون)(?![\u0621-\u064a])", text):
        reasons.append("unsupported_negation")
    return dict(
        valid=not reasons,
        expected=sorted(expected),
        found=sorted(found),
        missing=sorted(expected - found),
        added=sorted(found - expected),
        repeated=repeated,
        reasons=reasons,
    )
