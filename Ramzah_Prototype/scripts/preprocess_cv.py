import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ramzah.cv import extract_keypoints


def process(row):
    dest = Path("data/cv/keypoints") / (row["sha256"] + ".npz")
    try:
        if not dest.exists():
            points, quality = extract_keypoints(row["path"])
            np.savez_compressed(dest, keypoints=points, quality=json.dumps(quality))
        return {**row, "keypoints": dest.as_posix()}, None
    except (ValueError, RuntimeError) as e:
        return None, dict(path=row["path"], error=str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="reports/dataset/manifest.csv")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8-sig")))
    out = Path("data/cv/keypoints")
    out.mkdir(parents=True, exist_ok=True)
    result = []
    failures = []
    rows = [r for r in rows if r["corrupt"] != "True" and r["missing_label"] != "True"]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, (item, error) in enumerate(pool.map(process, rows)):
            if item:
                result.append(item)
            if error:
                failures.append(error)
            if (i + 1) % 10 == 0:
                print(
                    f"{i + 1}/{len(rows)} processed, {len(failures)} failed", flush=True
                )
    # Disjoint source clips; final signer 03 held out from adaptation only.
    # Upstream ALL checkpoint may already have seen every signer/sample.
    groups = {}
    for row in result:
        groups.setdefault((row["signer_id"], row["class_id"]), []).append(row)
    for (sid, cls), items in groups.items():
        items.sort(key=lambda x: hashlib.sha256(x["path"].encode()).hexdigest())
        for i, row in enumerate(items):
            row["split"] = (
                "test" if sid == "03" else ("validation" if i < 2 else "train")
            )
    seen = {}
    for row in result:
        for key in ["sha256", "decoded_sha256"]:
            identity = (key, row[key])
            if identity in seen and seen[identity] != row["split"]:
                raise ValueError("Duplicate crosses splits")
            seen[identity] = row["split"]
    Path("data/cv/manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    Path("reports/dataset/preprocessing_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print("Prepared", len(result), "sequences", flush=True)


if __name__ == "__main__":
    main()
