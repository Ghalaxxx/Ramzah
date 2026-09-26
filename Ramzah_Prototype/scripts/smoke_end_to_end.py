"""Exercise actual video, CV, neural generation, confirmation, employee view and TTS."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ramzah.api import app

rows = json.loads(Path("data/cv/manifest.json").read_text(encoding="utf8"))
examples = [
    next(row for row in rows if row["split"] == "test" and row["class_id"] == class_id)
    for class_id in ["0116", "0100", "0088"]
]
client = TestClient(app)
sid = client.post("/api/sessions").json()["session_id"]
predictions = []
for example in examples:
    with open(example["path"], "rb") as source:
        response = client.post(
            f"/api/sessions/{sid}/signs",
            files={"file": (Path(example["path"]).name, source, "video/mp4")},
        )
    response.raise_for_status()
    predictions.append(response.json())
assert client.get(f"/api/sessions/{sid}/employee").status_code == 409
assert client.get(f"/api/sessions/{sid}/speech.wav").status_code == 409

# In a live UI the signer reviews this step. Here use the audited sample's label
# only to exercise the downstream path, not to inflate the CV prediction metric.
reviewed = [example["arabic_gloss"] for example in examples]
response = client.put(f"/api/sessions/{sid}/glosses", json={"glosses": reviewed})
response.raise_for_status()
response = client.post(f"/api/sessions/{sid}/sentences")
response.raise_for_status()
candidates = response.json()["candidates"]
result = {
    "samples": [example["path"] for example in examples],
    "reviewed_glosses": reviewed,
    "cv_predictions": predictions,
    "candidates": candidates,
}
if candidates:
    response = client.post(
        f"/api/sessions/{sid}/confirm", json={"sentence": candidates[0]}
    )
    response.raise_for_status()
    employee = client.get(f"/api/sessions/{sid}/employee")
    employee.raise_for_status()
    audio = client.get(f"/api/sessions/{sid}/speech.wav")
    audio.raise_for_status()
    assert audio.content[:4] == b"RIFF"
    result.update(confirmed=employee.json(), speech_bytes=len(audio.content))
else:
    result["abstained"] = True
Path("reports/integration").mkdir(parents=True, exist_ok=True)
Path("reports/integration/smoke.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
