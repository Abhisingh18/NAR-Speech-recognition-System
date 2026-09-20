"""Pydantic request/response models — the JSON contract of the service."""
from typing import Optional, List

from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    hyp: str                      # decode capped at first </s>  (the real output)
    hyp_uncapped: str             # whole frame axis, but specials still stripped
    hyp_raw: str                  # RAW frame string: every frame's token, incl <s> </s> <fill> <pad> '|'
    leaked: bool                  # hyp != hyp_uncapped (garbage past </s> existed)
    duration_s: float
    total_frames: int             # encoder frames T (~50/s)
    n_keep: int                   # duration budget ceil(duration_s * frames_per_sec)
    eos_frame: Optional[int]      # frame index of the first </s> (None = never emitted)
    inference_ms: float
    rtf: float                    # inference time / audio duration
    frame_tokens: Optional[List[str]] = None   # per-frame tokens incl <fill> (only if raw=1)


class PathRequest(BaseModel):
    path: str                     # server-local audio file
    raw: bool = False             # include per-frame tokens in the response


class InfoResponse(BaseModel):
    run_dir: str
    ckpt: str
    num_sa_layers: int
    use_pe: bool
    encoder: str
    frames_per_sec: int
    trained_step: Optional[int]
    best_wer: Optional[float]
    device: str
    sampling_rate: int
    max_seconds: float
