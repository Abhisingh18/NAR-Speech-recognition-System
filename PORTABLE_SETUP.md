# 🚚 Portable setup — run this training as ANY user, on a common space

> For `abhishek@…`, `akshaya@…`, or a fresh machine — how to stand up the env, point at the code
> and models, **format your data**, and launch training from scratch. Visual + copy-paste.

---

## 🧠 What actually has to be "portable" (only 4 things)

```
        ┌────────────────────────────────────────────────────────────────────┐
        │  1. ENV      conda env with fairseq + transformers 5.5 + torch      │
        │  2. CODE     3 python trees (this project + 2 dependencies)         │
        │  3. ENCODER  the frozen data2vec-aqc .pt (SSL or CTC-finetuned)     │
        │  4. DATA     your audio + a jsonl manifest (the important one ⬇)    │
        └────────────────────────────────────────────────────────────────────┘
   Everything is wired through ENV VARS — no code edits needed to relocate.
```

The scripts read these environment variables (defaults = tomson's paths):

| env var | what it points at | default |
|---|---|---|
| `FILLER_ROOT` | dir with `filler_sa_reference.py` (+ `train_sa_pe_lm_head/`) | `/speech/tomson/filler_asr` |
| `SLAM_SRC` | dir with `slam_llm/models/encoder.py` (data2vec loader) | `/speech/tomson/SMEAR-MoE-ASR/src` |
| `ENCODER_PT` | the frozen data2vec-aqc checkpoint | akshaya's `24kup_model/checkpoint_best.pt` |
| `VOCAB_DIR` | dir with your `vocab.json` | `./vocab_hi` |
| `TRAIN_MANIFEST` / `DEV_MANIFEST` | your jsonl data | smear hi jsonls |
| `OUT` | where checkpoints/logs go | `./runs/…` |
| `ENV` | conda env name **or prefix path** | `indic-nar-filler_asr` |

> **Set those 7 and it runs anywhere.** No file needs editing.

---

## 🗂️ Recommended common-space layout

Put everything under one **group-writable** directory so every user shares it:

```
/speech/shared/indic-nar/                     ← chmod g+rwx, same unix group for all users
├── env/                                       ← conda PREFIX env (shared, activate by path)
├── code/
│   ├── indic-nar-filler_asr/                  ← this project (copy of tomson's)
│   ├── filler_asr/                            ← dependency (filler_sa_reference.py, train_sa_pe_lm_head/)
│   └── SMEAR-MoE-ASR/src/                     ← dependency (slam_llm data2vec loader)
├── models/
│   └── data2vec-aqc/…checkpoint_best.pt       ← the frozen encoder
├── data/
│   ├── audio/…*.wav                           ← 16 kHz mono wavs
│   ├── train.jsonl   dev.jsonl                ← manifests (see below)
│   └── vocab_hi/vocab.json                    ← built from train.jsonl
└── runs/                                       ← outputs
```

> **Permissions matter most.** If you instead point at tomson's originals, they must be
> **group-readable** (`chmod -R g+rX`) and you must share a unix group. Copying into `/speech/shared`
> avoids all permission surprises.

---

## 📀 THE DATA — how it must look (most important)

Training reads a **JSON-Lines manifest**: **one JSON object per line**, one line per utterance.

```json
{"key": "utt_000001", "source": "/speech/shared/indic-nar/data/audio/utt_000001.wav", "target": "उल्टा चश्मा उसमें", "language": "hi"}
{"key": "utt_000002", "source": "/speech/shared/indic-nar/data/audio/utt_000002.wav", "target": "और बाइकिंग ट्रिप में बहुत ही अच्छा", "language": "hi"}
```

```
 ┌───────────┬──────────────────────────────────────────────────────────────────────┐
 │ FIELD     │ REQUIRED?  MEANING / RULES                                             │
 ├───────────┼──────────────────────────────────────────────────────────────────────┤
 │ source    │ YES  absolute path to the audio.  (alias: "audio")                    │
 │ target    │ YES  the raw transcript, any case/punctuation.  (alias: "text")       │
 │ key       │ no   utterance id (ignored by the loader; nice for debugging)         │
 │ language  │ no   ignored by the loader                                            │
 └───────────┴──────────────────────────────────────────────────────────────────────┘
```

**Audio rules (hard requirements):**
```
   • format     : WAV, readable by soundfile
   • sample rate: 16 000 Hz   ← asserted; resample first if not
   • channels   : mono         ← stereo is auto-averaged to mono, but mono is cleaner
```
Resample anything that isn't 16 kHz mono, e.g.:
```bash
ffmpeg -i in.wav -ac 1 -ar 16000 out.wav      # per file
# or: sox in.wav -r 16000 -c 1 out.wav
```

**The transcript is cleaned automatically** by `normalize_hi` at train time — you do **not**
pre-clean it. It applies: NFC normalize → drop punctuation/symbols/zero-width/foreign → keep
Devanagari + single spaces. So `"उल्टा, चश्मा।"` in your file becomes `"उल्टा चश्मा"` for training.
*(Only requirement: the transcript is Devanagari Hindi. Other language → swap in a `normalize_<lang>`.)*

> **Minimal manifest you can hand-write to test:** two fields per line —
> `{"source": "/abs/x.wav", "target": "कुछ हिंदी वाक्य"}` — that's genuinely all it needs.

---

## 🧱 Step 1 — Create the shared env (once, ~5 min)

Cleanest for multi-user: a **prefix env** in the common space (everyone activates the same path).

```bash
# clone the proven env into a SHARED prefix (needs read access to tomson's env once)
conda create -p /speech/shared/indic-nar/env --clone /speech/tomson/miniconda3/envs/indic-nar-filler_asr -y
# use it (by PATH, not name):
conda activate /speech/shared/indic-nar/env      # or:  conda run -p /speech/shared/indic-nar/env …
```
If you can't read tomson's env, build from the same base:
```bash
conda create -p /speech/shared/indic-nar/env --clone /speech/tomson/miniconda3/envs/smear-moe-gemma3 -y
```
> Contains: torch 2.4, transformers 5.5, **fairseq 0.12** (needed by data2vec), omegaconf, soundfile, wandb.
> The scripts default `ENV=indic-nar-filler_asr` (a name); to use the shared **prefix**, pass
> `ENV=/speech/shared/indic-nar/env`.

---

## 🧬 Step 2 — Get the code (copy to common space)

```bash
mkdir -p /speech/shared/indic-nar/code
cp -r /speech/tomson/indic-nar-filler_asr        /speech/shared/indic-nar/code/
cp -r /speech/tomson/filler_asr                  /speech/shared/indic-nar/code/     # dependency
cp -r /speech/tomson/SMEAR-MoE-ASR               /speech/shared/indic-nar/code/     # dependency (need src/)
chmod -R g+rwX /speech/shared/indic-nar
```
(Or skip copying and just set `FILLER_ROOT`/`SLAM_SRC` to tomson's dirs **if** they're group-readable.)

---

## 🔤 Step 3 — Build the vocabulary from YOUR train manifest

```bash
cd /speech/shared/indic-nar/code/indic-nar-filler_asr
FILLER_ROOT=/speech/shared/indic-nar/code/filler_asr \
conda run -p /speech/shared/indic-nar/env python text_norm_hi.py \
    --jsonls /speech/shared/indic-nar/data/train.jsonl /speech/shared/indic-nar/data/dev.jsonl \
    --out_dir /speech/shared/indic-nar/data/vocab_hi
# → prints char count; writes vocab_hi/vocab.json  (rebuild whenever data or normalize_hi changes)
```

---

## ⚡ Step 3.5 (OPTIONAL) — Precompute the tap cache (big speedup for large data)

The data2vec-aqc encoder is frozen + eval-pinned, so its 1024-d tap for a wav never changes. Compute
it **once** and train straight from disk — the encoder is then dropped off the GPU entirely.

```bash
cd /speech/shared/indic-nar/code/indic-nar-filler_asr
ENV=/speech/shared/indic-nar/env \
FILLER_ROOT=/speech/shared/indic-nar/code/filler_asr \
SLAM_SRC=/speech/shared/indic-nar/code/SMEAR-MoE-ASR/src \
ENCODER_PT=/speech/shared/indic-nar/models/data2vec-aqc/checkpoint_best.pt \
VOCAB_DIR=/speech/shared/indic-nar/data/vocab_hi \
TRAIN_MANIFEST=/speech/shared/indic-nar/data/train.jsonl \
DEV_MANIFEST=/speech/shared/indic-nar/data/dev.jsonl \
CACHE=/speech/shared/indic-nar/data/tapcache_d2v_hi \
GPUS=1,2,3,4 bash extract_taps_d2v.sh
# writes CACHE/{train,dev}/{taps.f16,index.npy,manifest.jsonl,PREPARED}. Resumable: a split
# with a PREPARED marker is skipped. Invalidated only if ENCODER_PT changes.
```
Then at Step 4, add `TAP_CACHE=/speech/shared/indic-nar/data/tapcache_d2v_hi` to the exports — the
run reads taps from disk (no live encoder), so it's faster and uses ~1.2 GB less VRAM.

## 🔥 Step 4 — Smoke, then launch (all paths via env vars)

```bash
cd /speech/shared/indic-nar/code/indic-nar-filler_asr
export ENV=/speech/shared/indic-nar/env
export FILLER_ROOT=/speech/shared/indic-nar/code/filler_asr
export SLAM_SRC=/speech/shared/indic-nar/code/SMEAR-MoE-ASR/src
export ENCODER_PT=/speech/shared/indic-nar/models/data2vec-aqc/checkpoint_best.pt
export VOCAB_DIR=/speech/shared/indic-nar/data/vocab_hi
export TRAIN_MANIFEST=/speech/shared/indic-nar/data/train.jsonl
export DEV_MANIFEST=/speech/shared/indic-nar/data/dev.jsonl
export OUT=/speech/shared/indic-nar/runs/my_first_run

# 1) smoke (40 steps, 1 GPU) — proves the whole path
SMOKE=1 GPU=6 bash run.sh

# 2) real run (detached, 3 GPUs) — inherits the exports above
GPU=6,7,8 bash launch_real.sh          # log → logs/ ; ckpts → $OUT ; stop → kill -TERM -<pgid>
```

---

## ⚙️ How a run flows (so you know what each step touches)

```
   your train.jsonl ──► text_norm_hi.py ──► vocab.json
                                                │
   your *.wav (16k) ─────────────┐              │
                                 ▼              ▼
                          run.sh / train.py  (env vars set paths)
                                 │
           ┌─────────────────────┼──────────────────────┐
           ▼                     ▼                      ▼
   FROZEN data2vec-aqc     8-block A-CMLM head     iterative-decode eval
   (ENCODER_PT, live)      (trains, ~100M)         (best.pt by dev iter_WER)
                                 │
                                 ▼
                        $OUT/best.pt, latest.pt, tokenizer
```

---

## 🧯 Portability gotchas (read before you blame the code)

```
 ✔ 16 kHz mono WAV        — else the collator asserts. Resample first.
 ✔ absolute paths in jsonl — "source" must be a full path the compute node can see.
 ✔ group perms           — env, code, models, data all g+rX; share one unix group.
 ✔ conda by PREFIX        — pass ENV=/speech/shared/indic-nar/env (a path), not a per-user name.
 ✔ keep --lr_schedule exponential — the transformers LR-scheduler import needs a newer peft; the
                            exponential path (default) avoids it.
 ✔ GPUs                   — launch_real.sh refuses busy GPUs; pass GPU=<free ids>. NCCL uses loopback.
 ✔ rebuild vocab + normalize TOGETHER — the char set must match what normalize_hi keeps.
 ✔ tap-cached (optional)  — precompute the frozen tap ONCE, then train straight from it and the
                            encoder is dropped off the GPU. See "Step 3.5" below. Big win for big data.
```

---

## 🌍 New machine (not this host) — extra steps

1. Install miniconda; recreate the env from a spec: on the source host
   `conda list -p /speech/shared/indic-nar/env --explicit > env.txt`, then on the new host
   `conda create -p <prefix> --file env.txt` (fairseq may need `pip install fairseq==0.12.2` from source if the wheel is missing).
2. Copy `code/`, `models/` (the encoder .pt), and your `data/`.
3. Set the 7 env vars to the new locations. Done.
```
```
