from ramzah.semantics import validate_candidate
from scripts.create_nlp_dataset import CONCEPTS, targets


def test_preserves_pain_location_without_diagnosis():
    assert validate_candidate(["ألم", "قلب"], "أشعر بألم في القلب.", CONCEPTS)["valid"]
    assert not validate_candidate(["ألم", "قلب"], "أعاني من نوبة قلبية.", CONCEPTS)[
        "valid"
    ]


def test_rejects_changed_time_and_clinic():
    assert not validate_candidate(
        ["موعد", "قلب", "اليوم"], "أود الاستفسار عن موعد غدًا بخصوص الأسنان.", CONCEPTS
    )["valid"]
    assert validate_candidate(
        ["موعد", "قلب", "اليوم"], targets(["موعد", "قلب", "اليوم"])[0], CONCEPTS
    )["valid"]


def test_confirmed_appointment_gloss_may_be_realized_as_booking():
    assert validate_candidate(
        ["موعد", "قلب", "اليوم"], "لدي موعد اليوم بخصوص القلب.", CONCEPTS
    )["valid"]


def test_rejects_repeated_unrequested_concept():
    result = validate_candidate(
        ["ألم", "ضغط الدم"], "أحتاج إلى معلومات عن الألم وضغط الدم وضغط الدم.", CONCEPTS
    )
    assert not result["valid"]
    assert result["repeated"] == ["ضغط الدم"]
