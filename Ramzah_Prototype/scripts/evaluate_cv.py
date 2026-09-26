import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from train_cv import tensors

from ramzah.cv import load_model

torch.set_num_threads(4)
rows = json.loads(Path("data/cv/manifest.json").read_text(encoding="utf8"))
rows = [x for x in rows if x["split"] == "test"]
model = load_model("checkpoints/cv/best")
classes = [model.config.id2label[i] for i in range(len(model.config.id2label))]
x, m, y = tensors(rows, classes)
meta = json.loads(Path("checkpoints/cv/best/metadata.json").read_text())
times = []
ls = []
with torch.inference_mode():
    model(x[:1], m[:1])
    for i in range(len(x)):
        t = time.perf_counter()
        ls.append(model(x[i : i + 1], m[i : i + 1]))
        times.append(time.perf_counter() - t)
logits = torch.cat(ls)
probs = (logits / meta["temperature"]).softmax(1)
pred = probs.argmax(1)
conf = probs.max(1).values
p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
report = dict(
    n=len(y),
    accuracy=accuracy_score(y, pred),
    precision_macro=p,
    recall_macro=r,
    f1_macro=f,
    top3_accuracy=float(
        (probs.topk(min(3, len(classes)), 1).indices == y[:, None])
        .any(1)
        .float()
        .mean()
    ),
    low_confidence_rate=float((conf < meta["threshold"]).float().mean())
    if meta["threshold"] is not None
    else 1.0,
    classifier_latency_ms_median=float(np.median(times) * 1000),
    classifier_latency_ms_p95=float(np.percentile(times, 95) * 1000),
    latency_scope="Cached landmarks only; excludes decoding and MediaPipe",
    per_class=classification_report(
        y,
        pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    ),
    confusion_matrix=confusion_matrix(
        y, pred, labels=list(range(len(classes)))
    ).tolist(),
    independence=meta["independence"],
    confidence_threshold=meta["threshold"],
)
out = Path("reports/cv")
out.mkdir(exist_ok=True, parents=True)
(out / "evaluation.json").write_text(json.dumps(report, indent=2))
np.savez_compressed(
    out / "test_predictions.npz",
    logits=logits.numpy(),
    labels=y.numpy(),
    classes=np.array(classes),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(report["confusion_matrix"])
ax.set_xticks(range(len(classes)), classes, rotation=90)
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("Predicted class ID")
ax.set_ylabel("Actual class ID")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(out / "confusion_matrix.png")
plt.close(fig)
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ["per_class", "confusion_matrix"]},
        indent=2,
    )
)
