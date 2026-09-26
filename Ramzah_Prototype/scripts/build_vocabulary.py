import collections
import csv
import json
from pathlib import Path

SELECTED = {88, 92, 100, 101, 103, 108, 109, 110, 115, 116, 129, 132}
labels = json.loads(Path("data/labels.json").read_text(encoding="utf8"))
rows = json.loads(Path("data/cv/manifest.json").read_text(encoding="utf8"))
counts = collections.Counter(r["class_id"] for r in rows)
aliases = {"0109": "مختبر", "0110": "أشعة"}
for cls, word in aliases.items():
    labels[cls]["arabic_gloss"] = word
Path("data/labels.json").write_text(
    json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf8"
)
with Path("data/ramzah_vocabulary.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class_id",
            "arabic_gloss",
            "original_label",
            "dataset_source",
            "num_samples",
            "selected",
            "notes",
        ],
    )
    writer.writeheader()
    for cls, label in labels.items():
        selected = int(cls) in SELECTED and counts[cls] > 0
        writer.writerow(
            dict(
                class_id=cls,
                arabic_gloss=label["arabic_gloss"],
                original_label=label["original_label"],
                dataset_source="KArSL official raw health test archives",
                num_samples=counts[cls],
                selected=str(selected).lower(),
                notes=(
                    "Local usable extracted clips, not full corpus count. "
                    + (
                        "Canonical Arabic alias/spelling of workbook label. "
                        if cls in aliases
                        else ""
                    )
                )
                + (
                    "No claim of Saudi expert semantic verification."
                    if selected
                    else "Not in selected local training data."
                ),
            )
        )
