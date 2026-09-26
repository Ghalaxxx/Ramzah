"""Continue contextual fine-tuning with equal sampling per semantic group."""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


def read(split):
    return [
        json.loads(x)
        for x in Path(f"data/nlp/{split}.jsonl").read_text(encoding="utf8").splitlines()
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.set_num_threads(4)
    source = Path("checkpoints/nlp/contextual_initial")
    out = Path("checkpoints/nlp/contextual_v3")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(source, legacy=True)
    retained = json.loads((source / "retained_ids.json").read_text())
    compact = {old: i for i, old in enumerate(retained)}

    def encode(row):
        return (
            [compact[x] for x in tokenizer(row["source"])["input_ids"]],
            [compact[x] for x in tokenizer(row["target"])["input_ids"]],
        )

    groups = defaultdict(list)
    for row in read("train"):
        groups[row["group_id"]].append(encode(row))
    validation = [encode(row) for row in read("validation")]
    model = T5ForConditionalGeneration.from_pretrained(source)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    pad = compact[tokenizer.pad_token_id]

    def batch(items):
        sx = max(len(x) for x, y in items)
        sy = max(len(y) for x, y in items)
        inputs = torch.tensor(
            [x + [pad] * (sx - len(x)) for x, y in items], device=device
        )
        labels = torch.tensor(
            [y + [-100] * (sy - len(y)) for x, y in items], device=device
        )
        return {
            "input_ids": inputs,
            "attention_mask": (inputs != pad).long(),
            "labels": labels,
        }

    best = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(a.epochs):
        rows = []
        for items in groups.values():
            rows.extend(random.choices(items, k=12))
        random.shuffle(rows)
        model.train()
        losses = []
        for index in range(0, len(rows), a.batch_size):
            optimizer.zero_grad()
            loss = model(**batch(rows[index : index + a.batch_size])).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        vl = []
        with torch.inference_mode():
            for index in range(0, len(validation), a.batch_size):
                vl.append(
                    float(model(**batch(validation[index : index + a.batch_size])).loss)
                )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": float(np.mean(vl)),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            model.save_pretrained(out, safe_serialization=True)
            tokenizer.save_pretrained(out)
            (out / "retained_ids.json").write_text(json.dumps(retained))
    metadata = {
        "source_checkpoint": str(source),
        "sampling": "12 random examples per semantic group per epoch",
        "epochs": a.epochs,
        "best_validation_loss": best,
        "seconds": time.perf_counter() - started,
        "device": device,
        "research_only_rights": True,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    Path("reports/nlp/contextual_training.json").write_text(
        json.dumps(history, indent=2)
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
