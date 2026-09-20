# filler-ASR serving

FastAPI wrapper around a trained SA+PE+lm_head head on frozen HuBERT-xlarge.
One model per process; the SA depth is picked simply by pointing `RUN` at a run dir —
`num_sa_layers` is counted from the checkpoint keys (same auto-recovery as `../infer.py`).

```
engine.py     load-once model engine (no web code; python engine.py --run_dir ... --wav ... works standalone)
audio_io.py   uploaded bytes / server path → mono float32 16 kHz (soundfile, ffmpeg fallback for webm/opus/mp3)
schemas.py    JSON contract
app.py        FastAPI wiring: lifespan load, GPU lock, endpoints, request log
static/       browser mic-recorder demo page
run_server.sh launcher (GPU pinning + conda env + uvicorn)
```

## Launch

```bash
RUN=full960_sa8_pe_fps25_lin GPU=1 bash run_server.sh                  # foreground, port 8677
setsid env RUN=full960_sa8_pe_fps25_lin GPU=1 bash run_server.sh > serve.log 2>&1 &   # background
```

Env knobs: `RUN` (required), `GPU` (default 1), `PORT` (8677), `CKPT` (best.pt),
`MAX_SECONDS` (60), `ENCODER` (override frozen HuBERT path), `LOG_JSONL`
(default `<run_dir>/serve_requests.jsonl` — every request's hyp + geometry is appended there).

## Endpoints

```bash
curl localhost:8677/health
curl localhost:8677/info

# upload a file (mic recording, wav, flac, mp3, webm — anything ffmpeg reads)
curl -s -F file=@utt.flac localhost:8677/transcribe | python -m json.tool

# file already on the box
curl -s -X POST localhost:8677/transcribe_path \
     -H 'Content-Type: application/json' \
     -d '{"path": "/speech/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"}'

# add per-frame tokens (incl <fill>) to the response
curl -s -F file=@utt.flac 'localhost:8677/transcribe?raw=true'
```

Browser demo: open `http://<box>:8677/` — record from the mic or pick a file.

## Response

| field | meaning |
|---|---|
| `hyp` | decode capped at the first `</s>` frame — the real output (matches infer.py) |
| `hyp_uncapped` | pure full unstripped hypothesis over the whole frame axis |
| `leaked` | hyp ≠ hyp_uncapped (garbage existed past `</s>`) |
| `duration_s` | audio length |
| `total_frames` | encoder frames T (~50/s) |
| `n_keep` | duration budget `ceil(duration_s × frames_per_sec)` (fps from ckpt args, 25) |
| `eos_frame` | frame index of the first `</s>` (null = never emitted) |
| `inference_ms`, `rtf` | latency and real-time factor |

## Notes

- `--workers` must stay 1 (each uvicorn worker would load its own model copy).
  Want sa4 and sa8 live at once? Run two processes on two ports.
- A `threading.Lock` serializes forwards — concurrent uploads queue, they don't collide.
- Inputs longer than `MAX_SECONDS` are rejected 422: the SA head is full self-attention, O(T²).
- Model is trained on LibriSpeech read speech; live-mic WER will be worse than test-clean —
  that's domain mismatch, not plumbing.
