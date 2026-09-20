# 🏃 RUNBOOK — data2vec-aqc + Hindi A-CMLM (step by step, do-it-yourself)

A follow-along guide: environment → data → model → smoke → real run → monitor.
Companion to `TRAINING_EXPLAINED.md` (the "why"); this file is the "do this".

---

## 🗺️ The whole pipeline at a glance

```
                        ┌───────────────────────────────────────────────┐
   STAGE 0  (once)      │  ENV:  clone smear-moe-gemma3                  │
                        │        → conda env  indic-nar-filler_asr       │
                        │        (fairseq 0.12 + transformers 5.5 + torch)│
                        └───────────────────────┬───────────────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────────────┐
        ▼                                        ▼                                        ▼
┌──────────────────┐              ┌─────────────────────────────┐         ┌───────────────────────────┐
│ STAGE 1  DATA    │              │ STAGE 2  MODEL              │         │ MODELS ON DISK            │
│ jsonl transcripts│              │ FROZEN data2vec-aqc  (1024d)│◄────────│ encoder .pt               │
│  → normalize_hi  │              │   + 8 self-attention blocks │         │  • CTC-finetuned Hindi    │
│  → vocab.json    │              │   + char lm_head            │         │  • or SSL                 │
└────────┬─────────┘              └──────────────┬──────────────┘         └───────────────────────────┘
         │                                       │
         └──────────────►  STAGE 3  COLLATOR  ◄──┘        (audio + masked chars → training batch)
                                   │
                                   ▼
              STAGE 4  TRAIN (A-CMLM masked-LM)  ──►  eval (iterative decode)  ──►  best.pt
```

**Only the head (≈100 M) trains. The encoder (313 M) is frozen and just produces features.**

---

## 🧱 STAGE 0 — Environment (once)

> **Why clone `smear-moe-gemma3`?** data2vec-aqc needs **fairseq**; the A-CMLM head needs
> **transformers 5.x**. `filler_asr`'s env (torch 2.7) can't install fairseq 0.12; `smear-moe-gemma3`
> already has both working. Cloning gives isolation without fighting dependencies.

```bash
# 1. clone the env (≈5 min, ~10 GB)
conda create --clone smear-moe-gemma3 -n indic-nar-filler_asr -y

# 2. verify the two worlds coexist
PYTHONPATH="/speech/tomson/filler_asr:/speech/tomson/SMEAR-MoE-ASR/src" \
conda run -n indic-nar-filler_asr python - <<'PY'
import filler_sa_reference                       # A-CMLM head (transformers 5.5)
from slam_llm.models.encoder import Data2VecAQCEncoder   # data2vec-aqc (fairseq)
import torch, transformers, fairseq
print("torch", torch.__version__, "| transformers", transformers.__version__, "| fairseq", fairseq.__version__)
print("OK: both stacks import together")
PY
```

```
  ┌─────────────────────────────────────────────────────────────┐
  │ ⚠  KNOWN SKEW:  transformers 5.5 LR-scheduler import pulls    │
  │    peft.PeftMixedModel (env peft is older) → ImportError.     │
  │    FIX (already in train.py): that import is LAZY; we use the │
  │    exponential LR (local LambdaLR). Keep --lr_schedule        │
  │    exponential and you never hit it.                          │
  └─────────────────────────────────────────────────────────────┘
```

**Models already on disk (nothing to download):**
```
encoder (frozen, pick one — loader drops the CTC head either way):
  • CTC-finetuned Hindi :  /speech/akshaya/fairseq_expxx/hindi/24kup_model/checkpoint_best.pt   ← we use this
  • SSL (lang-agnostic) :  /speech/tomson/exps/speech-recog/models/data2vec-aqc/SPRING_INX_data2vec_aqc_SSL.pt
both are data2vec-aqc LARGE: 1024-d, 24 layers, 16 kHz, 50 fps.   (no HuBERT / no LLM needed)
```

---

## 📚 STAGE 1 — Data & vocabulary

Input manifest line (`.jsonl`):
```json
{"key":"hi_...","source":".../hi_....wav","target":"उल्टा, चश्मा।","language":"hi"}
```

Build the Devanagari character vocab (run whenever data or `normalize_hi` changes):
```bash
cd /speech/tomson/indic-nar-filler_asr
conda run -n indic-nar-filler_asr python text_norm_hi.py \
    --jsonls /speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/train_hi.jsonl \
             /speech/tomson/exps/speech-recog/data/smear-more-hi-ta-te-ma/val_hi.jsonl \
    --out_dir vocab_hi
# → prints char count; writes vocab_hi/vocab.json + vocab_hi/chars_report.txt
```

```
  ┌───────────────────────────────────────────────────────────────────┐
  │ ⚠  vocab.json and normalize_hi MUST be rebuilt TOGETHER.            │
  │    normalize_hi decides which characters survive; vocab.json must   │
  │    list exactly those. Change one → regenerate the other.           │
  └───────────────────────────────────────────────────────────────────┘
```

### 🔬 What the text becomes (one clip, traced end to end)

```
 RAW target                "उल्टा, चश्मा।"
     │  normalize_hi:  NFC  +  drop punct/symbols/zero-width/foreign  +  keep Devanagari
     ▼
 CLEAN                     "उल्टा चश्मा"
     │  replace(" ","|")  +  split into Unicode CODEPOINTS  (matra ा, virama ्, nukta each = 1 token)
     ▼
 CHAR TOKENS              उ  ल  ्  ट  ा  |  च  श  ्  म  ा
     │  _labels_for:  pin char i → frame i, then pad the silent tail with <fill>   (length = T = 50·dur)
     ▼
 POSITIONAL TARGET (len T=535 for a 10.7 s clip)
   <s> उ ल ् ट ा | च श ् म ा </s></s></s> <fill> <fill> … <fill>
     │  CMLM masking:  randomly hide a subset of content frames
     ▼
 text_input_ids :  <s>  उ  <mask>  ्  ट  ा  |  च  <mask>  म  ा  …   ← model INPUT (holes to fill)
 labels         : -100 -100   ल   -100 -100 … -100   श   -100 -100 …  ← LOSS only on the holes
```

> **Key idea:** the model sees the audio + the *unmasked* characters and must reconstruct the
> masked ones. At inference every position starts as `<mask>` and is filled iteratively.

---

## 🧠 STAGE 2 — The model (tensor shapes as data flows)

```
   wav  [B, S]                          16 kHz mono, per-utterance normalized
     │
     │  ╔═══════════════════════════════════════════════════════╗
     │  ║  FROZEN  data2vec-aqc  (313 M, requires_grad=False)    ║   CTC head PHYSICALLY ABSENT
     │  ║   7×conv (320× ↓)  →  24× Conformer/Transformer blocks ║   → tap taken BEFORE any CTC proj
     │  ╚═══════════════════════════════════════════════════════╝
     ▼
   tap  [B, T, 1024]                    T = S/320  ≈  50 frames / sec
     │
     ├─►  (train only) mask 20–30 % of frames in spans of 10   → learn to fill from context
     ├─►  + symmetric ALiBi position bias                      → relative positions, length-robust
     ├─►  + E_text(text_input_ids)  [B, T, 1024]   (ADDITION)  → fuse the partly-masked characters
     ▼
   [B, T, 1024]
     │  ╔══════════════════════════════════════════════╗
     │  ║  8 × self-attention block  (TRAINABLE ≈100 M) ║   d=1024, 16 heads, ffn 4096, post-norm
     │  ╚══════════════════════════════════════════════╝
     ▼
   [B, T, 1024]  ──►  lm_head (1024 → V≈66)  ──►  logits [B, T, V]
     │
     ▼
   framewise cross-entropy on MASKED slots only    (fill_weight=0.03 down-weights the <fill> majority)
```

Sanity numbers (from the smoke): encoder **313.3 M frozen / 0 trainable**, head **≈100.9 M trainable**,
tap dim **1024**, and `data2vec T == compute_output_length(S)` for every clip (frame↔label aligned).

---

## 🔥 STAGE 3+4 — Smoke, then the real run

### ① Smoke (2 min, 1 GPU) — proves the whole path before committing GPUs
```bash
cd /speech/tomson/indic-nar-filler_asr
SMOKE=1 GPU=6 \
  ENCODER_PT=/speech/akshaya/fairseq_expxx/hindi/24kup_model/checkpoint_best.pt \
  bash run.sh
# expect: builds model, trains 40 steps, eval + best.pt saved, DONE
```

### ② Real run (150 epochs, detached, multi-GPU)
```bash
GPU=6,7,8 bash launch_real.sh
#  → pid/pgid printed; log in logs/train_ctc_*.log ; checkpoints in runs/…/
#  → stop with:  kill -TERM -<PGID>
```

**The recipe baked into `run.sh` (matches the proven English run):**
```
 ┌───────────────────┬──────────────┐   ┌────────────────────┬──────────────┐
 │ positions         │ ALiBi        │   │ fill_weight        │ 0.03         │
 │ self-attn layers  │ 8            │   │ mask (span×len)    │ 20–30 % ×10  │
 │ frames/sec (fps)  │ 50           │   │ eos_repeat         │ 3            │
 │ peak LR / warmup  │ 2e-4 / 5000  │   │ schedule           │ exp → 1 %    │
 │ epochs            │ 150          │   │ select best.pt by  │ dev iter_WER │
 │ batch × accum × G │ 16 × 4 × 3   │   │ eff batch          │ 192          │
 └───────────────────┴──────────────┘   └────────────────────┴──────────────┘
```

```
  ┌───────────────────────────────────────────────────────────────────┐
  │ ⚠  OOM LESSON:  peak GPU mem = batch × (longest clip's T).          │
  │    The collator pads every clip to the batch's longest, so one 28 s │
  │    clip inflates the whole batch. batch 32 → OOM on 48 GB.          │
  │    FIX (baked in): batch 16 + PYTORCH_CUDA_ALLOC_CONF=expandable_.. │
  └───────────────────────────────────────────────────────────────────┘
```

---

## 📈 STAGE 5 — Monitor & read the signals

```bash
tail -f logs/train_ctc_*.log
```
```
 WATCH THESE (leading → lagging):
   iter-CER    ──►  should fall fast (4.0 → <1 in the first ~1k steps)   ← realistic quality
   content_acc ──►  should climb off ~0.003                              ← acoustics→char working
   fill_rate   ──►  should recover to ~0.7 after an early dip            ← <fill> tail learned
   distinct    ──►  should widen (→ most of the vocab)                   ← not collapsed
   WER         ──►  LAST to move; stays ~1.0 until characters sharpen    ← don't panic early
```

Checkpoints land in `runs/hi_d2vaqcCTC_.../` : `best.pt` (argmin dev iter_WER), `latest.pt`, tokenizer.

---

## 🎛️ STAGE 6 — Adapt for best-case / bigger runs

```
 GOAL                        DO THIS
 ─────────────────────────   ────────────────────────────────────────────────────────────
 faster + no OOM + scale     TAP CACHE (implemented): bash extract_taps_d2v.sh   (encoder run
                             ONCE -> memmap), then  TAP_CACHE=<dir> GPU=.. bash launch_real.sh
                             -> head trains from cache, length-bucketed (~3x faster, OOM-proof)
 4000 h Hindi                cache first (≈1.5 TB fp16), then scale; do NOT run live at 4000 h
 match reference LR          keep eff batch ~256 (add a 4th GPU, or raise accum)
 another language            change only normalize_<lang> + vocab_<lang>; model is agnostic
 cheaper encoder             --encoder_tgt_layer N   (early-exit a shallower layer)
 resume                      RESUME=runs/…/latest.pt  (via run.sh)
```

---

### 📂 Files (all in `/speech/tomson/indic-nar-filler_asr/`)
```
 text_norm_hi.py     normalize_hi + build vocab_hi/vocab.json + Hindi tokenizer
 model_d2v_acmlm.py  Data2VecEncoderAdapter + build_d2v_acmlm + Hindi collator
 train.py            training loop (filler_asr train.py + 4 edits)
 run.sh              recipe + env (PYTHONPATH, expandable_segments, NCCL lo)
 launch_real.sh      detached multi-GPU launch (setsid) + GPU guard + pid/pgid
 smoke.py            forward/backward/geometry sanity (no training loop)
 TRAINING_EXPLAINED.md   the "why" + full diff vs data2vec-aqc CTC + hyperparam notes
 RUNBOOK.md          ← this file
```
