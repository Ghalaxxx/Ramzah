"""Adapt verified KArSL weights; validation determines checkpoint and confidence."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from ramzah.cv import load_model, normalize_keypoints


def tensors(rows, classes):
    xs = []
    ys = []
    masks = []
    for row in rows:
        pts = normalize_keypoints(
            np.load(row["keypoints"], allow_pickle=False)["keypoints"]
        )
        xs.append(np.pad(pts, ((0, 64 - len(pts)), (0, 0), (0, 0))))
        masks.append([1] * len(pts) + [0] * (64 - len(pts)))
        ys.append(classes.index(row["class_id"]))
    if not xs:
        raise ValueError("Empty data split")
    return (
        torch.tensor(np.stack(xs)),
        torch.tensor(masks, dtype=torch.float32),
        torch.tensor(ys),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = json.loads(Path("data/cv/manifest.json").read_text(encoding="utf8"))
    classes = sorted(set(x["class_id"] for x in rows))
    train = [x for x in rows if x["split"] == "train"]
    val = [x for x in rows if x["split"] == "validation"]
    model = load_model("checkpoints/signbart-karsl502")
    old = model.classification_head.out_proj
    ids = [int(model.config.label2id[c]) for c in classes]
    head = nn.Linear(old.in_features, len(classes))
    with torch.no_grad():
        head.weight.copy_(old.weight[ids])
        head.bias.copy_(old.bias[ids])
    model.classification_head.out_proj = head
    model.config.id2label = {i: c for i, c in enumerate(classes)}
    model.config.label2id = {c: i for i, c in enumerate(classes)}
    model.to(device)
    tx, tm, ty = (x.to(device) for x in tensors(train, classes))
    vx, vm, vy = (x.to(device) for x in tensors(val, classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    out = Path("checkpoints/cv/best")
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    best_state = None
    stale = 0
    t = time.perf_counter()
    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(len(tx), device=device)
        losses = []
        for idx in order.split(a.batch_size):
            optimizer.zero_grad()
            x = tx[idx] + torch.randn_like(tx[idx]) * 0.005
            logits = model(x, tm[idx])
            loss = nn.functional.cross_entropy(logits, ty[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            vl = model(vx, vm)
            vloss = nn.functional.cross_entropy(vl, vy).item()
            acc = (vl.argmax(1) == vy).float().mean().item()
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=float(np.mean(losses)),
                validation_loss=vloss,
                validation_accuracy=acc,
            )
        )
        if vloss < best:
            best = vloss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0:
            print(json.dumps(history[-1]), flush=True)
        if stale >= 20:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(vx, vm).detach()
    # Fit scalar temperature solely on validation.
    logt = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / logt.exp().clamp(0.05, 20), vy)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(logt.exp().clamp(0.05, 20))
    probs = (logits / temperature).softmax(1)
    conf, pred = probs.max(1)
    correct = pred.eq(vy)
    threshold = None
    threshold_stats = None
    for candidate in sorted(set(conf.cpu().tolist())):
        keep = conf >= candidate
        n = int(keep.sum())
        success = int(correct[keep].sum())
        if n < 20:
            continue
        # Wilson lower 95% bound; require >=80% instead of picking a cosmetic percentage.
        rate = success / n
        z = 1.96
        lower = (
            rate
            + z * z / (2 * n)
            - z * ((rate * (1 - rate) + z * z / (4 * n)) / n) ** 0.5
        ) / (1 + z * z / n)
        if lower >= 0.8:
            threshold = candidate
            threshold_stats = dict(n=n, correct=success, wilson_lower=lower)
            break
    save_file(best_state, str(out / "model.safetensors"))
    model.config.to_json_file(out / "config.json")
    metadata = dict(
        temperature=temperature,
        threshold=threshold,
        threshold_validation=threshold_stats,
        seed=42,
        train_clips=len(train),
        validation_clips=len(val),
        seconds=time.perf_counter() - t,
        upstream_checkpoint="tinh2312/SignBart-KArSL-ALL-502",
        independence="Signer 03 excluded from adaptation; upstream ALL training exposure unknown. NOT clean signer-independent evaluation.",
        preprocessing="MediaPipe Holistic Tasks 1.0.1; 32 uniformly sampled native frames; 75 xy landmarks; per-part normalization",
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    Path("reports/cv").mkdir(parents=True, exist_ok=True)
    Path("reports/cv/training.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
