"""Download pinned official AraT5 research weights, without executing remote code."""

import hashlib
import json
from pathlib import Path

import requests

evidence = Path("evidence/arat5small_api.txt")
meta = (
    json.loads(evidence.read_text())
    if evidence.exists()
    else {
        "id": "UBC-NLP/AraT5-msa-small",
        "sha": "5c46ab988cba0cdb4358a218602801f25c29c6d5",
    }
)
rev = meta["sha"]
out = Path("checkpoints/nlp/base")
out.mkdir(parents=True, exist_ok=True)
records = []
for name in [
    "config.json",
    "pytorch_model.bin",
    "spiece.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "README.md",
]:
    p = out / name
    if not p.exists():
        url = f"https://huggingface.co/UBC-NLP/AraT5-msa-small/resolve/{rev}/{name}"
        partial = p.with_suffix(p.suffix + ".part")
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            if existing and r.status_code != 206:
                existing = 0
            with partial.open("ab" if existing else "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        partial.rename(p)
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    records.append(dict(file=name, sha256=digest, bytes=p.stat().st_size))
    print(name, p.stat().st_size, flush=True)
Path("evidence").mkdir(exist_ok=True)
Path("evidence/nlp_downloads.json").write_text(
    json.dumps(dict(repo=meta["id"], revision=rev, files=records), indent=2)
)
