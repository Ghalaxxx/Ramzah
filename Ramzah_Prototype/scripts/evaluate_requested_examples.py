"""Record the requested trace example, word-order variants and unseen groups."""

import json
from pathlib import Path

from ramzah.nlp import GlossRealizer

model = GlossRealizer("checkpoints/nlp/contextual_v3")
splits = {}
for split in ("train", "validation", "test"):
    splits[split] = [
        json.loads(line)
        for line in Path(f"data/nlp/{split}.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]

requested = [
    ["موعد", "قلب", "اليوم"],
    ["قلب", "موعد", "اليوم"],
    ["اليوم", "موعد", "قلب"],
]
requested_results = []
for glosses in requested:
    result = model.generate(glosses)
    requested_results.append(
        {
            "input_glosses": glosses,
            "model_input": " ".join(glosses),
            "model_generated": result["candidates"],
            "generation_method": result["generation_method"],
        }
    )

unseen = []
seen_train_rows = {(row["source"], row["target"]) for row in splits["train"]}
for row in splits["test"]:
    if row["group_id"] in {item["group_id"] for item in unseen}:
        continue
    assert (row["source"], row["target"]) not in seen_train_rows
    result = model.generate(row["glosses"])
    unseen.append(
        {
            "group_id": row["group_id"],
            "input_glosses": row["glosses"],
            "reference": row["target"],
            "model_generated": result["candidates"][0]
            if result["candidates"]
            else None,
            "all_candidates": result["candidates"],
            "exact_pair_held_out_from_training": True,
        }
    )
    if len(unseen) == 8:
        break

report = {
    "checkpoint": "checkpoints/nlp/contextual_v3",
    "trace": "recognized_words -> tokenizer -> fine-tuned T5 -> model.generate -> tokenizer.decode",
    "requested_word_orders": requested_results,
    "unseen_test_examples": unseen,
}
Path("reports/nlp/requested_examples.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
