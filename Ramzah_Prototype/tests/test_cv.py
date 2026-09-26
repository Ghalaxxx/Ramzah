import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ramzah.cv import load_model, normalize_keypoints, video_frames


def test_invalid_video_is_not_a_prediction(tmp_path):
    f = tmp_path / "bad.mp4"
    f.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        video_frames(f)


def test_preprocessing_rejects_corrupt_and_empty_points():
    with pytest.raises(ValueError):
        normalize_keypoints(np.zeros((0, 75, 2)))
    with pytest.raises(ValueError):
        normalize_keypoints(np.full((3, 75, 2), np.nan))
    with pytest.raises(ValueError):
        normalize_keypoints(np.zeros((3, 25, 2)))


def test_missing_hands_stay_finite():
    x = np.zeros((100, 75, 2), dtype=np.float32)
    x[:, 11:17] = np.array(
        [[0.3, 0.2], [0.6, 0.2], [0.2, 0.4], [0.7, 0.4], [0.1, 0.6], [0.8, 0.6]]
    )
    y = normalize_keypoints(x)
    assert y.shape == (64, 75, 2)
    assert np.isfinite(y).all()
    assert (y[:, 33:] == 0).all()


def test_actual_checkpoint_maps_all_502_classes():
    torch.set_num_threads(2)
    if not Path("checkpoints/signbart-karsl502/config.json").exists():
        pytest.skip(
            "Optional upstream SignBART checkpoint is not redistributed in the ready package"
        )
    model = load_model("checkpoints/signbart-karsl502")
    labels = json.loads(Path("data/labels.json").read_text(encoding="utf8"))
    assert set(model.config.id2label.values()) == set(labels)
    x = torch.zeros(1, 8, 75, 2)
    with torch.inference_mode():
        y = model(x, torch.ones(1, 8))
    assert y.shape == (1, 502) and torch.isfinite(y).all()
    # Successful shape/weight test is not a sign-recognition accuracy claim.
