"""Audio ingest: uploaded bytes or a server-local path → mono float32 @ 16 kHz.

Fast path: soundfile (wav/flac/ogg). Fallback: ffmpeg subprocess for everything a
browser or phone might send (webm/opus, mp3, m4a, ...) and for resampling anything
that isn't already 16 kHz. Raises AudioError with a client-safe message on bad input.
"""
import io
import subprocess

import numpy as np
import soundfile as sf

TARGET_SR = 16000


class AudioError(Exception):
    """Bad/undecodable/too-long audio — maps to a 4xx response."""


def _ffmpeg_decode(data=None, path=None):
    """Decode anything ffmpeg understands → mono float32 @ 16 kHz."""
    src = ["-i", "pipe:0"] if data is not None else ["-i", path]
    cmd = ["ffmpeg", "-v", "error", *src,
           "-ac", "1", "-ar", str(TARGET_SR), "-f", "f32le", "pipe:1"]
    try:
        p = subprocess.run(cmd, input=data, capture_output=True, timeout=120)
    except FileNotFoundError:
        raise AudioError("ffmpeg not available on the server")
    except subprocess.TimeoutExpired:
        raise AudioError("audio decode timed out")
    if p.returncode != 0 or not p.stdout:
        raise AudioError(f"could not decode audio: {p.stderr.decode(errors='ignore')[:200]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _to_mono_16k(arr, sr, raw=None, path=None):
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        # let ffmpeg do the resample from the original container (better than ad-hoc numpy)
        arr = _ffmpeg_decode(data=raw, path=path)
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_audio_bytes(data, max_seconds):
    """Uploaded file content → (wave, duration_s)."""
    if not data:
        raise AudioError("empty upload")
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32")
        wave = _to_mono_16k(arr, sr, raw=data)
    except AudioError:
        raise
    except Exception:
        wave = _ffmpeg_decode(data=data)          # webm/opus/mp3/m4a etc.
    return _check(wave, max_seconds)


def load_audio_path(path, max_seconds):
    """Server-local audio file → (wave, duration_s)."""
    try:
        arr, sr = sf.read(path, dtype="float32")
        wave = _to_mono_16k(arr, sr, path=path)
    except AudioError:
        raise
    except FileNotFoundError:
        raise AudioError(f"no such file: {path}")
    except Exception:
        wave = _ffmpeg_decode(path=path)
    return _check(wave, max_seconds)


def _check(wave, max_seconds):
    duration = len(wave) / TARGET_SR
    if duration < 0.1:
        raise AudioError(f"audio too short ({duration:.2f}s)")
    if duration > max_seconds:
        raise AudioError(f"audio too long ({duration:.1f}s > max {max_seconds}s); "
                         "the SA head is full self-attention (O(T^2))")
    return wave, duration
