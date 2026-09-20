"""FastAPI wrapper around the filler-ASR Engine. Wiring only — model logic lives in engine.py.

Config via env vars (see run_server.sh):
    RUN_DIR      required — run dir with vocab.json + checkpoint (picks the SA depth)
    CKPT         checkpoint path (default <RUN_DIR>/best.pt)
    ENCODER      override frozen HuBERT path (default: from ckpt args)
    MAX_SECONDS  reject longer audio (default 60)
    LOG_JSONL    per-request log file (default <RUN_DIR>/serve_requests.jsonl)

Endpoints:
    GET  /health           liveness
    GET  /info             which model is answering
    POST /transcribe       multipart audio upload (mic recording / any file)
    POST /transcribe_path  JSON {"path": "/speech/..."} for files already on the box
    GET  /                 demo recorder page (static/index.html)
"""
import os
import json
import time
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse

from engine import Engine
from audio_io import AudioError, load_audio_bytes, load_audio_path
from schemas import TranscribeResponse, PathRequest, InfoResponse

RUN_DIR = os.environ.get("RUN_DIR", "")
CKPT = os.environ.get("CKPT", "")
ENCODER = os.environ.get("ENCODER", "")
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "60"))
LOG_JSONL = os.environ.get("LOG_JSONL", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
engine = None
gpu_lock = threading.Lock()      # one forward at a time — single model, single GPU
log_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global engine
    assert RUN_DIR, "set RUN_DIR env var (run dir with vocab.json + best.pt)"
    engine = Engine.load(RUN_DIR, ckpt=CKPT, model_checkpoint=ENCODER)
    engine.warmup()
    print("[app] ready", flush=True)
    yield


app = FastAPI(title="filler-ASR", lifespan=lifespan)


def _log(record):
    path = LOG_JSONL or os.path.join(RUN_DIR, "serve_requests.jsonl")
    with log_lock, open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _run(wave, duration, source, raw):
    with gpu_lock:
        result = engine.transcribe(wave, include_frame_tokens=raw)
    _log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source,
          "duration_s": result["duration_s"], "total_frames": result["total_frames"],
          "n_keep": result["n_keep"], "inference_ms": result["inference_ms"],
          "hyp": result["hyp"]})
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.get("/info", response_model=InfoResponse)
def info():
    return {**engine.info, "max_seconds": MAX_SECONDS}


@app.post("/transcribe", response_model=TranscribeResponse, response_model_exclude_none=True)
def transcribe(file: UploadFile = File(...), raw: bool = Query(False)):
    data = file.file.read()
    try:
        wave, duration = load_audio_bytes(data, MAX_SECONDS)
    except AudioError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _run(wave, duration, file.filename or "upload", raw)


@app.post("/transcribe_path", response_model=TranscribeResponse, response_model_exclude_none=True)
def transcribe_path(req: PathRequest):
    try:
        wave, duration = load_audio_path(req.path, MAX_SECONDS)
    except AudioError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _run(wave, duration, req.path, req.raw)


@app.get("/")
def index():
    return FileResponse(os.path.join(_HERE, "static", "index.html"))
