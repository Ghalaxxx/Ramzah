"""The employee and speech routes must stay closed until explicit confirmation."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import ramzah.api as api_module
from ramzah.api import app

client = TestClient(app)


def test_confirmation_gate():
    sid = client.post("/sessions").json()["session_id"]
    assert client.get(f"/sessions/{sid}/employee").status_code == 409
    assert client.get(f"/sessions/{sid}/speech.wav").status_code == 409
    assert (
        client.post(
            f"/sessions/{sid}/confirm", json={"sentence": "أحتاج إلى المساعدة"}
        ).status_code
        == 409
    )


def test_upload_validation():
    sid = client.post("/sessions").json()["session_id"]
    response = client.post(
        f"/sessions/{sid}/signs",
        files={"file": ("example.txt", b"not video", "text/plain")},
    )
    assert response.status_code == 415
    assert client.get(f"/sessions/{sid}").json()["glosses"] == []


def test_unknown_session():
    assert client.get("/sessions/unknown").status_code == 404


def test_production_api_health():
    assert client.get("/api/health").status_code == 200


def test_user_authored_fallback_after_model_abstains(monkeypatch):
    class AbstainingRealizer:
        concepts = json.loads(
            (Path(__file__).resolve().parents[1] / "data/nlp/concepts.json").read_text(
                encoding="utf8"
            )
        )

        def generate(self, glosses):
            return {
                "input": glosses,
                "candidates": [],
                "requires_confirmation": True,
                "rejected_count": 0,
                "uncertainty": True,
            }

    monkeypatch.setattr(api_module, "realizer", lambda: AbstainingRealizer())
    sid = client.post("/api/sessions").json()["session_id"]
    assert (
        client.put(
            f"/api/sessions/{sid}/glosses", json={"glosses": ["موعد", "قلب", "اليوم"]}
        ).status_code
        == 200
    )
    assert client.get(f"/api/sessions/{sid}/employee").status_code == 409
    abstention = client.post(f"/api/sessions/{sid}/sentences").json()
    assert abstention["candidates"] == []
    assert abstention["message"].startswith("تعذر فهم المقصود")
    accepted = client.post(
        f"/api/sessions/{sid}/confirm", json={"sentence": "لدي موعد اليوم بخصوص القلب."}
    )
    assert accepted.status_code == 200
    assert accepted.json()["source"] == "user_edit"
    assert (
        client.get(f"/api/sessions/{sid}/employee").json()["sentence"]
        == "لدي موعد اليوم بخصوص القلب."
    )
