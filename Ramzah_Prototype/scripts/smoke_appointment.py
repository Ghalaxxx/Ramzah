"""Document the NLP-only appointment example through the actual API."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ramzah.api import app

client = TestClient(app)
sid = client.post("/api/sessions").json()["session_id"]
words = ["موعد", "قلب", "اليوم"]
edited = client.put(f"/api/sessions/{sid}/glosses", json={"glosses": words})
edited.raise_for_status()
generated = client.post(f"/api/sessions/{sid}/sentences")
generated.raise_for_status()
assert generated.json()["candidates"], (
    "Promoted neural model abstained on appointment example"
)
assert client.get(f"/api/sessions/{sid}/employee").status_code == 409
choice = generated.json()["candidates"][0]
confirmed = client.post(f"/api/sessions/{sid}/confirm", json={"sentence": choice})
confirmed.raise_for_status()
result = {
    "input_source": "manual NLP-only glosses; no CV appointment recognition claim",
    "glosses": words,
    "generated": generated.json(),
    "confirmed": confirmed.json(),
    "employee": client.get(f"/api/sessions/{sid}/employee").json(),
}
Path("reports/integration/appointment.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
