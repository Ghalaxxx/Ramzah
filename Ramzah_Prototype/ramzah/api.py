"""Prototype API: video recognition, editable glosses, reviewed text, confirmed delivery."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .semantics import validate_candidate

app = FastAPI(title="Ramzah prototype", version="0.1")
router = APIRouter()
MAX_VIDEO_BYTES = 80 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Conversation:
    glosses: list[str] = field(default_factory=list)
    recognitions: list[dict] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    chosen: str | None = None
    confirmed: bool = False
    generation_attempted: bool = False
    sentence_source: str | None = None


_sessions: dict[str, Conversation] = {}
_lock = Lock()
_predictor = None
_realizer = None
_speech = None


def session_or_404(session_id: str) -> Conversation:
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session")
    return session


def predictor():
    global _predictor
    if _predictor is None:
        from .cv import SignPredictor

        _predictor = SignPredictor()
    return _predictor


def realizer():
    global _realizer
    if _realizer is None:
        from .nlp import GlossRealizer

        _realizer = GlossRealizer()
    return _realizer


def speech():
    global _speech
    if _speech is None:
        from .tts import ArabicSpeech

        _speech = ArabicSpeech()
    return _speech


class GlossEdit(BaseModel):
    glosses: list[str] = Field(min_length=1, max_length=12)


class Confirmation(BaseModel):
    sentence: str = Field(min_length=1, max_length=500)


@router.get("/health")
def health():
    return {
        "status": "ready"
        if (ROOT / "checkpoints/cv/best/model.safetensors").exists()
        and (ROOT / "checkpoints/nlp/contextual_v3/model.safetensors").exists()
        else "models_missing",
        "cv_checkpoint": (ROOT / "checkpoints/cv/best/model.safetensors").exists(),
        "nlp_checkpoint": (
            ROOT / "checkpoints/nlp/contextual_v3/model.safetensors"
        ).exists(),
        "tts_checkpoint": (ROOT / "checkpoints/tts/mms-ara/model.safetensors").exists(),
    }


@router.post("/sessions")
def create_session():
    sid = uuid4().hex
    with _lock:
        _sessions[sid] = Conversation()
    return {"session_id": sid, "glosses": [], "confirmed": False}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    s = session_or_404(session_id)
    return {
        "session_id": session_id,
        "glosses": s.glosses,
        "recognitions": s.recognitions,
        "candidates": s.candidates,
        "chosen": s.chosen,
        "confirmed": s.confirmed,
        "generation_attempted": s.generation_attempted,
        "sentence_source": s.sentence_source,
    }


@router.post("/sessions/{session_id}/signs")
async def recognize(session_id: str, file: UploadFile = File(...)):
    s = session_or_404(session_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".webm", ".mov", ".avi"}:
        raise HTTPException(415, "Upload a supported video file")
    content = await file.read(MAX_VIDEO_BYTES + 1)
    if len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "Video exceeds 80 MB")
    if not content:
        raise HTTPException(400, "Empty video")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(content)
        path = temp.name
    try:
        prediction = predictor().predict(path)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        os.unlink(path)
    s.recognitions.append(prediction)
    if not prediction["needs_review"]:
        s.glosses.append(prediction["prediction"])
    s.candidates.clear()
    s.chosen = None
    s.confirmed = False
    s.generation_attempted = False
    s.sentence_source = None
    return {**prediction, "glosses": s.glosses}


@router.put("/sessions/{session_id}/glosses")
def edit_glosses(session_id: str, edit: GlossEdit):
    s = session_or_404(session_id)
    supported = realizer().concepts
    if any(g not in supported for g in edit.glosses):
        raise HTTPException(
            422, "One or more glosses are outside the supported vocabulary"
        )
    s.glosses = list(edit.glosses)
    s.candidates.clear()
    s.chosen = None
    s.confirmed = False
    s.generation_attempted = False
    s.sentence_source = None
    return {"glosses": s.glosses, "confirmed": False}


@router.post("/sessions/{session_id}/sentences")
def generate(session_id: str):
    s = session_or_404(session_id)
    if not s.glosses:
        raise HTTPException(409, "Review and add at least one gloss first")
    try:
        result = realizer().generate(s.glosses)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    s.candidates = result["candidates"]
    s.chosen = None
    s.confirmed = False
    s.generation_attempted = True
    s.sentence_source = None
    if not s.candidates:
        result["message"] = (
            "تعذر فهم المقصود بوضوح، يرجى تعديل الكلمات أو المحاولة مرة أخرى."
        )
    return result


@router.post("/sessions/{session_id}/confirm")
def confirm(session_id: str, choice: Confirmation):
    s = session_or_404(session_id)
    # The user may edit a generated sentence, but only if it passes the same guard.
    if not s.generation_attempted:
        raise HTTPException(409, "Try neural sentence generation before confirmation")
    concepts = realizer().concepts
    review = validate_candidate(s.glosses, choice.sentence, concepts)
    # A decoded model candidate may be confirmed after the user sees its warning.
    # User-authored edits still need to pass the strict semantic guard.
    if not review["valid"] and choice.sentence.strip() not in s.candidates:
        raise HTTPException(
            422,
            {
                "message": "Sentence does not preserve reviewed concepts",
                "review": review,
            },
        )
    s.chosen = choice.sentence.strip()
    s.sentence_source = "model_candidate" if s.chosen in s.candidates else "user_edit"
    s.confirmed = True
    return {"confirmed": True, "sentence": s.chosen, "source": s.sentence_source}


@router.get("/sessions/{session_id}/employee")
def employee(session_id: str):
    s = session_or_404(session_id)
    if not s.confirmed or not s.chosen:
        raise HTTPException(409, "The user has not confirmed a sentence")
    return {"sentence": s.chosen, "glosses": s.glosses, "source": s.sentence_source}


@router.get("/sessions/{session_id}/speech.wav")
def speech_audio(session_id: str):
    s = session_or_404(session_id)
    if not s.confirmed or not s.chosen:
        raise HTTPException(409, "The user has not confirmed a sentence")
    return Response(content=speech().wav_bytes(s.chosen), media_type="audio/wav")


app.include_router(router)
app.include_router(router, prefix="/api", include_in_schema=False)
