"""Fine-tune AraT5 on context-specific Arabic targets with no style prompts."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


def read(split):
    return [
        json.loads(line)
        for line in Path(f"data/nlp/{split}.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(51)
    random.seed(51)
    np.random.seed(51)
    torch.set_num_threads(4)
    base = Path("checkpoints/nlp/base")
    out = Path("checkpoints/nlp/contextual_initial")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(base, legacy=True)
    train, validation, test = read("train"), read("validation"), read("test")

    # The pretrained tokenizer is fixed. Vocabulary compaction retains pretrained
    # rows needed to encode/decode the declared corpus; only train rows update weights.
    keep = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id}
    for row in train + validation + test:
        for field in ("source", "target"):
            keep.update(tokenizer(row[field])["input_ids"])
    retained = sorted(keep)
    compact = {original: index for index, original in enumerate(retained)}

    def encode(row):
        return (
            [compact[x] for x in tokenizer(row["source"])["input_ids"]],
            [compact[x] for x in tokenizer(row["target"])["input_ids"]],
        )

    train_encoded = [encode(row) for row in train]
    validation_encoded = [encode(row) for row in validation]

    model = T5ForConditionalGeneration.from_pretrained(base, weights_only=True)
    shared = model.shared.weight.detach().clone()[retained]
    lm = model.lm_head.weight.detach().clone()[retained]
    model.set_input_embeddings(nn.Embedding.from_pretrained(shared, freeze=False))
    output = nn.Linear(model.config.d_model, len(retained), bias=False)
    output.weight.data.copy_(lm)
    model.set_output_embeddings(output)
    model.config.vocab_size = len(retained)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
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
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        random.shuffle(train_encoded)
        model.train()
        losses = []
        for index in range(0, len(train_encoded), args.batch_size):
            optimizer.zero_grad()
            loss = model(**batch(train_encoded[index : index + args.batch_size])).loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for index in range(0, len(validation_encoded), args.batch_size):
                validation_losses.append(
                    float(
                        model(
                            **batch(validation_encoded[index : index + args.batch_size])
                        ).loss
                    )
                )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": float(np.mean(validation_losses)),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            stale = 0
            model.save_pretrained(out, safe_serialization=True)
            tokenizer.save_pretrained(out)
            (out / "retained_ids.json").write_text(json.dumps(retained))
        else:
            stale += 1
        if stale >= 4:
            break
    metadata = {
        "base_model": "UBC-NLP/AraT5-msa-small",
        "base_revision": "5c46ab988cba0cdb4358a218602801f25c29c6d5",
        "training": "full gradient fine-tuning from pretrained weights",
        "source_format": "plain gloss sequence",
        "train_examples": len(train),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "retained_pretrained_subwords": len(retained),
        "epochs_run": len(history),
        "best_validation_loss": best,
        "device": device,
        "seconds": time.perf_counter() - started,
        "research_only_rights": True,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    Path("reports/nlp/contextual_initial_training.json").write_text(
        json.dumps(history, indent=2)
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
