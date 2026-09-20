# data2vec-aqc + Hindi A-CMLM — how the training works, and how it differs

## 0. One-paragraph mental model
We take a **frozen** data2vec-aqc encoder (the same model akshaya CTC-fine-tuned for Hindi; we
use its body and throw away the CTC head), read out a per-frame feature "tap" at 50 fps, and train
a **small 100 M-param head** (8 self-attention blocks + a character `lm_head`) to fill in masked
characters — a **non-autoregressive masked-LM over frames** (A-CMLM), not CTC. Only the head learns.

---

## 1. Data pipeline — step by step, what each step does to the data

Input line (`train_hi.jsonl`):
```
{"key": "...", "source": ".../hi_..._.wav", "target": "उल्टा चश्मा उसमें", "language": "hi"}
```

**Step 1 — `load_rows`** → `{audio_path, text}`. Accepts `source/target` or `audio/text`. No change to content.

**Step 2 — read audio** (`soundfile`): WAV → float32 array, mono, **16 kHz** (asserted). `उल्टा…` stays text.

**Step 3 — `Wav2Vec2FeatureExtractor(do_normalize=True)`**: per-utterance **zero-mean / unit-variance**
of the raw waveform → `input_values [S]`. *This is exactly the normalization data2vec expects*
(fairseq `normalize=true` = `F.layer_norm(x, x.shape)`); numerically identical. Also yields an
`attention_mask` after padding (1=real sample, 0=pad).

**Step 4 — frame count** `T = compute_output_length(S)`: the 7-layer conv stack (kernels
[10,3,3,3,3,2,2], strides [5,2,2,2,2,2,2]) downsamples 320× → **50 frames/sec**. e.g. 10.7 s → 535 frames.

**Step 5 — `normalize_hi(text)`** (the Hindi-specific bit):
  1. **NFC normalize** — collapses precomposed nukta letters (ज़ U+095B, ड़, फ़ …) to a *consistent*
     `base + ़` (U+093C) encoding. Without this the corpus mixes both forms → two token sequences
     for the same word.
  2. **Drop** all punctuation / symbols / control-format (categories P*, S*, C*) — including danda `।`,
     comma, `?`, `:` and the zero-width ZWSP/ZWNJ/ZWJ noise — **even inside the Devanagari block**.
  3. **Drop** anything outside the Devanagari block (foreign Latin code-switch, etc.).
  4. Keep single spaces.
  `"उल्टा, चश्मा।"` → `"उल्टा चश्मा"`.

**Step 6 — `_labels_for(text, T)`** (positional character target of length T):
  - `text.replace(" ", "|")` → word boundaries become the `|` token.
  - char-tokenize **every Unicode codepoint** → ids (matra ा, virama ्, nukta ़ are each their own token;
    conjunct क्ष = क + ् + ष).
  - lay them at the **head** of the frame axis: `[<s>] + char_ids + [</s>]×eos_repeat(3) + <fill> × (T − len)`.
  - So char *i* is pinned to frame *i*; the long silent/After-speech tail is `<fill>`.
    `उल्टा…` → `<s> उ ल ् ट ा | च श ् म ा … </s></s></s> <fill> <fill> …`  (length exactly T).

**Step 7 — CMLM masking** (what makes it "conditional masked-LM"):
  - `n_keep = ceil(dur × fps=50) = T` (all frames graded).
  - sample a masking rate `p` (U(0,1), or forced 1.0 for 15 % of clips); with prob. it's a full mask.
  - randomly pick a subset of content positions `[1, n_keep)` → set them to `<mask>` in
    **`text_input_ids`**, keep the truth in **`labels`** there and `-100` (ignore) everywhere else.
  - result: the head **sees** the unmasked chars + the audio, must **predict** the masked ones.

**Step 8 — pad the batch**: `input_values [B,S]`, `attention_mask [B,S]`, `text_input_ids [B,T]`,
`labels [B,T]` (pad→-100). **Every clip in a batch is padded to the longest clip's T** — this is why one
28 s clip inflates a whole batch (the OOM we hit; fixed by batch 16).

**Step 9 — model forward** (`Data2VecEncoderAdapter` + `FillerHubertACMLM`):
```
raw wav → [FROZEN data2vec-aqc body, CTC head absent] → tap  [B,T,1024]
        → (train) 20–30 % span-mask of the tap
        → + symmetric ALiBi position bias
        → + E_text(text_input_ids)          # ADDITION fusion of the (partly-masked) text
        → 8 × post-norm self-attention (d=1024, 16 heads, ffn 4096)
        → lm_head (1024 → V)  → logits [B,T,V]
```

**Step 10 — loss** `framewise_ce_loss`: cross-entropy on the **masked slots only** (`-100` elsewhere),
with **`fill_weight=0.03`** down-weighting the dominant `<fill>` class so the loss budget goes to real
characters. Token-exact averaging over the grad-accum × DDP window.

**Decoding (eval / inference)** — `iterative_decode`: start all-`<mask>`, run the head, keep the most
confident predictions, re-mask the rest, repeat N=32 times (mask-predict). Non-autoregressive: the whole
utterance is refined in parallel, unlike CTC's single greedy/beam pass.

---

## 2. How this differs from the ORIGINAL data2vec-aqc CTC training (akshaya's fairseq run)

| aspect | akshaya: data2vec-aqc **CTC** (fairseq) | this: data2vec-aqc **A-CMLM** (ours) |
|---|---|---|
| **encoder** | data2vec-aqc, **fully fine-tuned** | **same body, FROZEN**; tap taken *before* the CTC head |
| **trained params** | whole encoder (~313 M) + CTC proj | only the 100 M head (SA + lm_head + text_embed + mask_embed) |
| **objective** | **CTC** loss (alignment-free, blank token, monotonic collapse) | **framewise masked-CE** (A-CMLM); predict masked chars |
| **decode** | greedy / beam CTC collapse | **iterative mask-predict** (N steps, non-autoregressive) |
| **target format** | `.ltr`: every codepoint space-sep, word sep ` \| `; **keeps punctuation** (।,?,:) ; **no NFC** | same codepoint split, but **NFC-normalized + punctuation stripped**; adds `<s>/</s>/<fill>/<mask>` and pins char→frame |
| **special tokens** | `<s></s><pad><unk>` + `\|` (CTC blank = `<pad>`) | + `<fill>` (silence/tail class) + `<mask>` (input-only) |
| **frame budget** | CTC handles length via blank; no explicit budget | explicit `n_keep = dur×50`, `<fill>` tail |
| **framework** | fairseq (`fairseq-hydra-train`, `wav2vec_ctc`) | PyTorch + HF transformers, DDP `torchrun` |
| **audio norm** | fairseq `normalize=true` | `Wav2Vec2FeatureExtractor(do_normalize=True)` — *same math* |

**Same:** the encoder architecture/weights, 16 kHz, 50 fps, codepoint-level Devanagari tokenization,
`\|` word separator.  **Different:** we freeze the encoder and only learn a head; we optimize a
masked-LM (not CTC); we clean the text more (NFC + no punctuation); we decode iteratively.

vs the **English HuBERT** A-CMLM this was ported from: only the **encoder** (HuBERT-xlarge 1280-d, 48 layers
→ data2vec-aqc 1024-d, 24 layers) and the **text side** (26 Latin letters → ~64 Devanagari codepoints; NFC;
`normalize_hi` instead of lowercase+ASCII-strip) change. The A-CMLM head, masking, ALiBi, `<fill>`, loss,
iterative decode are **byte-for-byte identical**.

---

## 3. Are the hyperparameters okay / optimized? Is there hope?

**Inherited unchanged from the proven English run** (ALiBi, fw=0.03, fps=50, 8 SA, mask 20–30 %×10,
eos=3, lr 2e-4, warmup 5000, exp-decay to 1 %, 150 ep). **Changed for this box:** eff batch 256→**192**
(3 GPUs × 16 × 4) and batch 32→16 (OOM fix).

- **Okay?** Yes — it's a sane, proven config and the live curves confirm healthy learning.
- **Fully optimized? No.** Honest gaps:
  1. **Speed/stability:** we run the frozen encoder **live every step** (recomputed every epoch) and use
     **random-length batches**. The English run instead used a **precomputed tap cache + length-bucketed
     batches** → ~3× faster and no padding-driven OOM. This is the #1 optimization to add.
  2. **eff batch 192 vs 256:** lr 2e-4 was tuned for 256; 192 is close enough, but if you add a 4th GPU or
     raise accum, keep eff batch ~256 to match the reference lr.
  3. **150 epochs on ~100 h (43 k utts):** the English run had 960 h. 150 passes on 100 h risks head
     overfitting — but `best.pt` is selected on dev iter-WER, so you're protected; watch for a dev-loss uptick.
  4. **fill_weight 0.03** is aggressive (drove the early all-`|` collapse). It recovered, so leave it — but it's
     the first knob to revisit if content stalls.
- **Hope?** Yes. iter-CER 4.0→0.55 in 3 k steps *before LR peaks*, content_acc rising, fill recovered. CER is the
  leading indicator; WER should start dropping below 1.0 once characters sharpen (watch evals 5 k–15 k).

---

## 4. Should you jump straight to 4000 h Hindi?

**Not directly, and not yet.** Sequence it:
1. **Finish/inspect this ~100 h run** to confirm the approach converges to a sane WER. If 100 h can't break
   WER meaningfully below 1.0, more data won't fix a method bug — diagnose first.
2. **Before scaling, switch to the tap-cache + length-bucketed path** (mandatory at 4000 h):
   - Running the frozen 313 M encoder live for 4000 h **every epoch** is enormously wasteful. Precompute the
     data2vec tap **once** → memmap, then train the head from cache (this is exactly filler_asr's
     `precompute_taps.py` + `data_tapcached.py`, adapted to data2vec 1024-d).
   - **Disk:** 4000 h × 3600 s × 50 fps × 1024 × 2 B ≈ **~1.5 TB** fp16 (English 960 h @1280-d was 442 GB).
     Confirm you have the space; otherwise cache a subset or tap a lower layer.
   - Length-bucketing then removes the OOM entirely and lets you push batch size back up.
3. **Then scale**: more data massively helps a char model. Expect the real win at 4000 h, with a longer
   schedule (fewer epochs needed — total *steps* is what matters) and multi-GPU.

Rule of thumb: **method-correctness on 100 h → cache → scale to 4000 h.** Jumping straight to 4000 h *live*
would be slow, OOM-prone, and would burn days before you even know the method works on Hindi.

---

## 5. Minimal, correct setup for a best-case run (env → models → scripts → launch)

### 5.1 Environment (once)
data2vec-aqc needs **fairseq**; the A-CMLM head needs **transformers 5.x**. `filler_asr` (torch 2.7) can't take
fairseq 0.12.2, so clone the env that already has both working:
```bash
conda create --clone smear-moe-gemma3 -n indic-nar-filler_asr -y
# has: torch 2.4.1, transformers 5.5, fairseq 0.12.2, omegaconf, soundfile, numpy, wandb
```
One known skew: `transformers.optimization` imports `peft.PeftMixedModel` (env peft is older). We use the
**exponential LR (local LambdaLR)**, so that import is made **lazy** in `train.py` — nothing to fix if you keep
`--lr_schedule exponential`.

### 5.2 Models (already on disk)
- **Encoder (frozen), two choices** — the loader (`Data2VecAQCEncoder`) drops the CTC head either way:
  - CTC-finetuned Hindi (what we're using): `/speech/akshaya/fairseq_expxx/hindi/24kup_model/checkpoint_best.pt`
  - Language-agnostic SSL: `/speech/tomson/exps/speech-recog/models/data2vec-aqc/SPRING_INX_data2vec_aqc_SSL.pt`
  - Both are data2vec-aqc **large, 1024-d, 16 kHz, 50 fps**. (Source: SPRING-INX / IIT-M SpeechLab.)
- **No HuBERT / no LLM needed** — this task is encoder+head only.

### 5.3 Code (all in `/speech/tomson/indic-nar-filler_asr/`, deps resolved via PYTHONPATH)
- `text_norm_hi.py` — `normalize_hi` + build `vocab_hi/vocab.json` + Hindi tokenizer.
- `model_d2v_acmlm.py` — `Data2VecEncoderAdapter` + `build_d2v_acmlm` + Hindi collator.
- `train.py` — the ported training loop (filler_asr train.py + 4 edits).
- `run.sh` — recipe + env (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, PYTHONPATH, NCCL lo).
- `launch_real.sh` — detached (`setsid`) 3-GPU launch with GPU-free guard + pid/pgid.
`PYTHONPATH = /speech/tomson/filler_asr : …/filler_asr/train_sa_pe_lm_head : /speech/tomson/SMEAR-MoE-ASR/src : .`

### 5.4 Run it
```bash
cd /speech/tomson/indic-nar-filler_asr
# 1. vocab (rebuild if you change normalize_hi or the data):
conda run -n indic-nar-filler_asr python text_norm_hi.py \
    --jsonls .../train_hi.jsonl .../val_hi.jsonl --out_dir vocab_hi
# 2. quick smoke (40 steps, 1 GPU):
SMOKE=1 GPU=6 ENCODER_PT=.../24kup_model/checkpoint_best.pt bash run.sh
# 3. real run (detached, 3 GPU):
GPU=6,7,8 bash launch_real.sh          # log in logs/, ckpts in runs/…/, stop: kill -TERM -<pgid>
```
**Correctness note:** `vocab_hi` and `normalize_hi` must be regenerated **together** — if you change what
`normalize_hi` keeps/drops (e.g. danda), rebuild `vocab.json` so the char set matches the targets.

### 5.5 Best-case adaptations
- **Add the tap cache + length-bucketed sampler** before any large/long run (speed + OOM-proof) — port
  `precompute_taps.py`/`data_tapcached.py` to data2vec 1024-d.
- **Batch**: peak mem = batch × longest-clip-T; keep batch ≤16 on 48 GB for live/random batching, or bucket.
- **eff batch ~256** to match the reference lr (add GPUs or raise accum).
- **Encoder layer tap**: `--encoder_tgt_layer N` to early-exit (cheaper, sometimes better features).
- **Other languages**: only `normalize_*` + `vocab_*` change; the model/recipe are language-agnostic.
