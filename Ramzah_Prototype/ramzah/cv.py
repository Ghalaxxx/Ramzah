"""Actual video -> MediaPipe landmarks -> checkpoint classifier.

The released Hugging Face weights use Hugging Face BART layers, unlike the
later hand-written upstream implementation. Load every tensor strictly.
This is a visual coordinate model: no text encoder or language generation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartDecoder, BartEncoder

ROOT = Path(__file__).resolve().parents[1]
PARTS = [list(range(11, 17)), list(range(33, 54)), list(range(54, 75))]


def normalize_keypoints(points, max_frames=64):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (75, 2) or len(points) == 0:
        raise ValueError("Expected nonempty T x 75 x 2 landmarks")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks contain nonfinite values")
    points = np.clip(points, 0, 1).copy()
    if len(points) > max_frames:
        points = points[np.linspace(0, len(points) - 1, max_frames).astype(int)]
    # Match the published per-part, per-frame square bounding boxes, including
    # zero-filled absent points. Do not change normalization after training.
    for frame in points:
        for part in PARTS:
            xy = frame[part]
            low = xy.min(axis=0)
            high = xy.max(axis=0)
            span = high - low
            side = span.max()
            margin = (side - span) / 2 + 0.05 * side
            lo = np.clip(low - margin, 0, 1)
            hi = np.clip(high + margin, 0, 1)
            width = hi - lo
            frame[part] = np.divide(xy - lo, width, out=xy.copy(), where=width != 0)
    return points


class Projection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj_x1 = nn.Linear(len(c.joint_idx), c.d_model)
        self.proj_y1 = nn.Linear(len(c.joint_idx), c.d_model)


class ClassificationHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dropout = nn.Dropout(c.classifier_dropout)
        self.out_proj = nn.Linear(c.d_model, len(c.id2label))

    def forward(self, x):
        return self.out_proj(self.dropout(x))


class SignBart(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)
        self.projection = Projection(config)
        self.classification_head = ClassificationHead(config)

    def forward(self, points, mask):
        xy = points[:, :, self.config.joint_idx, :]
        x = self.projection.proj_x1(xy[..., 0])
        y = self.projection.proj_y1(xy[..., 1])
        enc = self.encoder(
            inputs_embeds=x, attention_mask=mask, return_dict=True
        ).last_hidden_state
        dec = self.decoder(
            inputs_embeds=y,
            attention_mask=mask,
            encoder_hidden_states=enc,
            encoder_attention_mask=mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        last = mask.long().sum(1) - 1
        return self.classification_head(
            dec[torch.arange(len(dec), device=dec.device), last]
        )


def load_model(checkpoint):
    checkpoint = Path(checkpoint)
    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf8"))
    # Legacy saved config has num_hidden_layers=6 alongside encoder_layers=2.
    # Transformers 5 aliases the former onto the latter; actual weights have 2.
    raw.pop("num_hidden_layers", None)
    config = BartConfig(**raw)
    config._attn_implementation = "eager"
    model = SignBart(config)
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    return model.eval()


def video_frames(path, max_seconds=20, max_frames=1200):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Invalid or unsupported video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video has no valid frame rate")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(frames) >= max_frames or len(frames) / fps > max_seconds:
                raise ValueError("Record one sign in a clip of 20 seconds or less")
            if max(frame.shape[:2]) > 640:
                scale = 640 / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError("Video contains no decodable frames")
    return frames, fps


def extract_keypoints(
    path, task_path=ROOT / "checkpoints/holistic_landmarker.task", sample_frames=32
):
    import mediapipe as mp

    frames, fps = video_frames(path)
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    points = []
    pose_present = 0
    hands_present = 0
    indices = np.linspace(0, len(frames) - 1, min(sample_frames, len(frames))).astype(
        int
    )
    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as detector:
        for i in indices:
            frame = frames[i]
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
                round(i * 1000 / fps),
            )
            row = np.zeros((75, 2), dtype=np.float32)
            for name, start, count in [
                ("pose_landmarks", 0, 33),
                ("left_hand_landmarks", 33, 21),
                ("right_hand_landmarks", 54, 21),
            ]:
                pts = getattr(result, name)
                if pts:
                    # Tasks API versions may expose single-person lists nested.
                    if isinstance(pts[0], list):
                        pts = pts[0]
                    row[start : start + count] = [[p.x, p.y] for p in pts[:count]]
            pose_present += int(np.any(row[:33]))
            hands_present += int(np.any(row[33:]))
            points.append(row)
    quality = dict(
        decoded_frames=len(frames),
        frames=len(indices),
        fps=fps,
        pose_fraction=pose_present / len(indices),
        hand_fraction=hands_present / len(indices),
    )
    if quality["pose_fraction"] < 0.5 or quality["hand_fraction"] < 0.25:
        raise ValueError(
            "لم أتمكن من التعرف على الإشارة بوضوح. أظهر اليدين والجزء العلوي من الجسم."
        )
    return np.stack(points), quality


class SignPredictor:
    def __init__(self, checkpoint=None):
        checkpoint = Path(checkpoint or ROOT / "checkpoints/cv/best")
        self.model = load_model(checkpoint)
        self.metadata = (
            json.loads((checkpoint / "metadata.json").read_text(encoding="utf8"))
            if (checkpoint / "metadata.json").exists()
            else {}
        )
        self.labels = json.loads((ROOT / "data/labels.json").read_text(encoding="utf8"))
        self.temperature = float(self.metadata.get("temperature", 1))
        self.threshold = self.metadata.get("threshold")

    @torch.inference_mode()
    def predict(self, path, k=3):
        points, quality = extract_keypoints(path)
        x = torch.from_numpy(normalize_keypoints(points)).unsqueeze(0)
        logits = self.model(x, torch.ones(x.shape[:2]))
        probs = (logits / self.temperature).softmax(-1)[0]
        values, ids = probs.topk(min(k, len(probs)))
        top = [
            dict(
                word=self.labels[str(self.model.config.id2label[int(i)])][
                    "arabic_gloss"
                ],
                class_id=str(self.model.config.id2label[int(i)]),
                confidence=float(v),
            )
            for v, i in zip(values, ids)
        ]
        accepted = self.threshold is not None and top[0]["confidence"] >= self.threshold
        return dict(
            prediction=top[0]["word"],
            confidence=top[0]["confidence"],
            top_k=top,
            needs_review=not accepted,
            confidence_kind="temperature_scaled_softmax"
            if self.temperature != 1
            else "uncalibrated_softmax",
            quality=quality,
        )


_predictor = None


def predict_sign(path):
    global _predictor
    if _predictor is None:
        _predictor = SignPredictor()
    return _predictor.predict(path)
