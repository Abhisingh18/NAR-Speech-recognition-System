# Conversation Log — HuBERT Fine-tuning (`filler_asr`) & SLAM-ASR Integration

This document distills the working session on the `filler_asr` project: first **understanding
the trained encoder**, then **repurposing it as the speech encoder for a SLAM-ASR
(encoder → linear projector → LLM) pipeline**. For the full standalone project deep-dive see
[report.md](report.md); this file captures the Q&A reasoning and the downstream design decisions.

---

## Part 0 — What the project is (one paragraph)

`filler_asr` fine-tunes **`facebook/hubert-base-ls960`** (HuBERT-base: CNN feature extractor +
12-layer Transformer, hidden 768, ~94M params) into a **character-level ASR** model on the full
**960h LibriSpeech**. Its defining quirk: it **does not use CTC**. Instead it uses **framewise
cross-entropy** with a **`<fill>` token** — the transcript is placed **left-packed**, one token per
output frame, and all remaining frames are labelled `<fill>`. Vocab = 33 (`'`, a–z, `|`, `<unk>`,
`<pad>`, `<s>`, `</s>`, `<fill>`). Result ≈ 49% WER (the method is a research probe; CTC normally wins).

---

## Part 1 — Understanding the encoder

### 1.1 Audio file → model input (no embedding lookup)

- A `.wav`/`.flac` is decoded by `datasets` → a **1-D float waveform**, resampled to **16 kHz**.
- `Wav2Vec2FeatureExtractor` (`feature_size=1, do_normalize=True`) only **normalizes**
  (zero-mean/unit-var per utterance); it is **not** a spectrogram. Output stays 1-D, same length.
- Shape into the model: `input_values (B, T_s)` where `T_s` = #audio samples.
- **Key point:** audio is *not* embedded via a lookup table. The raw waveform is reshaped
  `input_values[:, None] → (B, 1, T_s)` (1 input channel = raw amplitude), and **the CNN itself
  produces the embedding**.

### 1.2 CNN feature extractor & frame rate

7 conv1d layers, all 512 channels: `kernels [10,3,3,3,3,2,2]`, `strides [5,2,2,2,2,2,2]`.

- Total stride **= 320** → **frame rate = 16000/320 = 50 frames/sec = 20 ms/frame**.
- Output `(B, T, 512)` → projection `Linear(512→768)` → `(B, T, 768)`.
- Exact frame count: `compute_output_length(L)` (mirrors the conv arithmetic):

```python
def compute_output_length(L):
    for k, s in zip([10,3,3,3,3,2,2], [5,2,2,2,2,2,2]):
        L = (L - k)//s + 1
    return L
# 1s→49, 2s→99, 5s→249, 9s→449, 10s→499, 30s→1499   (≈ duration×50, minus ~1 for conv edges)
```

### 1.3 SpecAugment-style masking

- Config-driven, **inside `HubertModel`**, not in the data pipeline.
- This checkpoint: `apply_spec_augment=true`, `mask_time_prob=0.05`, `mask_time_length=10`,
  `mask_time_min_masks=2`; **feature masking off** (`mask_feature_prob=0.0`).
- Masks ~5% of frames in **contiguous 10-frame (200 ms) blocks**, replacing them with a learned
  `masked_spec_embed`. **Training-mode only** — disabled at `eval()`. ⚠️ When reusing as a frozen
  encoder you must `eval()` it (and/or set `apply_spec_augment=False`) or it corrupts features.

### 1.4 The loss — framewise CE vs CTC

`FillerASRTrainer.compute_loss` ([model.py:247](model.py#L247)) overrides the default to use
**`nn.CrossEntropyLoss(ignore_index=-100)`** per frame:

```
L = (1/N) · Σ_(valid b,t)  − log softmax(z[b,t])[ y[b,t] ]
```

- Logits `(B,T,33)` and labels `(B,T)` are flattened to `(B·T,33)` and `(B·T,)`.
- `-100` (padding) frames are skipped. No softmax layer exists in the model — `forward` returns
  raw logits; softmax/argmax happen only in loss/decoding.
- **CTC contrast:** CTC needs *no* per-frame labels — it adds a blank token and minimizes
  `−log Σ_π Π_t p(π_t)` over all alignments that collapse to the transcript (forward-backward DP).
  This project replaces that with a **fixed, hard, left-packed alignment** + plain CE. That fixed
  (acoustically wrong) alignment is why outputs show doubled/garbled letters and ~49% WER.

### 1.5 `dataset.py` pipeline

1. Load + concat train-clean-100/360 + train-other-500 (≈281,241 utts); dev-clean as `"test"`.
2. `remove_special_characters`: strip ``,?.!-;:"`` and lowercase.
3. `cast_column("audio", 16kHz)`.
4. `prepare_dataset` per utterance:
   - `input_values` = normalized waveform; `input_length` = #samples.
   - `T = compute_output_length(input_length)` (frames the model will emit).
   - text: space→`|`, tokenize → char ids; wrap `[<s>] + ids + [</s>]`.
   - truncate text if `len+2 > T`; then **pad with `<fill>` up to length T** (left-packed).
5. `extract_vocab` builds the 33-token vocab (chars sorted, then `| <unk> <pad> <s> </s>`, + `<fill>=32`).

### 1.6 What a prepared sample looks like (before Arrow caching)

```python
{
  'input_values':  [0.0142, -0.0876, ...],   # float32, len = #samples (NORMALIZED waveform)
  'input_length':  32000,
  'output_length': 99,                        # = T (frames)
  'labels':        [30, 8, 5, 27, 18, 1, 14, 31, 32, 32, ..., 32]   # len = T, left-packed + <fill>
}
```
Only `input_values` and `labels` are actually consumed downstream. Each row has a different length;
they're padded **per-batch** later, not here.

### 1.7 Padding & batching (filler training)

- **Dynamic, per-batch** (`collator.py`, `padding=True`): each batch padded to its own longest member.
  There is **no global max** (that would require `padding="max_length"`).
- Effective batch = `per_device(16) × grad_accum(4) × GPUs(4) = 256`; ~1,099 steps/epoch.
- Collator: `feature_extractor.pad` waveforms (+ attention mask), `tokenizer.pad` labels, then
  label pads → `-100`.

### 1.8 The `<fill>` token & "frames from the left" at inference

- The checkpoint is a **per-frame classifier**: argmax each frame, drop specials, `|`→space.
- Trained contract: *frame i emits the i-th left-packed token*; after the transcript it emits `<fill>`.
  The model learns to gather content and dump it at the left, then fill.
- **You never set "how many frames from the left."** It's emergent — the boundary = where the model
  starts predicting `</s>`/`<fill>`. Decoding strips all specials, so the exact boundary doesn't matter.

---

## Part 2 — Repurposing as a SLAM-ASR encoder

Goal: `HuBERT (frozen) → linear projector → Vicuna`, training only the projector (+ optional LoRA),
LLM loss = next-token CE on the transcript.

### 2.1 lm_head logits (33-d) vs hidden states (768-d)

- `lm_head = Linear(768, 33)`. Its **output** is `model(...).logits` `(B,T,33)` — **already pre-softmax**;
  there is no softmax layer to "strip."
- **Two valid feature taps** — they change the projector's input dim:
  | Tap | Shape | Projector input | Notes |
  |---|---|---|---|
  | hidden states (lm_head **input**) | (T,768) | **768** | rich, standard SLAM-ASR, already LayerNorm'd by encoder |
  | lm_head **output** (logits) | (T,33) | **33** | compact "soft characters", lossy, **needs normalization** |
- Final decision in this session: **strip `lm_head` + dropout, use the 768-d hidden states**
  (standard SLAM-ASR). ⚠️ This means the features are **acoustically aligned, NOT left-packed** — so
  Option-C / `<fill>` boundary / left-trim logic **no longer applies** to them.

### 2.2 Normalization (if using 33-d logits, no softmax)

`LayerNorm(33)` (recommended) / `RMSNorm` / L2-normalize / z-score — put it as the projector's first
layer. Needed for raw logits because they're un-normalized; **not** needed for 768-d hidden (the
encoder's internal LayerNorms already condition them, exactly like standard HuBERT→linear ASR).

### 2.3 The checkpoint — verified

Using **`experiments/filler_asr_960h_triphase/checkpoint-164850`** (the only checkpoint with weights):

- ✅ Full `FillerHubertModel`: 213 tensors, `hubert.*` + `lm_head.weight [33,768]` + `lm_head.bias [33]`.
- config: `vocab_size=33`, hidden 768, 12 layers, `architectures=['FillerHubertModel']`,
  `apply_spec_augment=true` (⚠️ disable), bos/eos/pad = 30/31/29.
- Provenance: final step of the **150-epoch triphase run**, `global_step=164850`, dev **WER ≈ 0.50**
  (the `cer≈7.9` is the eval-padding artifact, not real CER). The better `_lr_epochs` run (37.5% WER)
  has **no saved weights** locally.
- ⚠️ `experiments/.../model/` is the **pre-training** snapshot (saved before `trainer.train()`), NOT
  the fine-tuned encoder. Get the trained encoder via `FillerHubertModel.from_pretrained(ckpt).hubert`.

Loading:
```python
cfg = HubertConfig.from_pretrained(CKPT); cfg.apply_spec_augment = False
encoder = FillerHubertModel.from_pretrained(CKPT, config=cfg).hubert   # 768-d, lm_head/dropout dropped
encoder.eval(); [p.requires_grad_(False) for p in encoder.parameters()]
```
Only `model.safetensors` + `config.json` + tokenizer files are needed for inference; ignore
`optimizer.pt`/`scheduler.pt`/`rng_state_*` (resume-only; the 4 rng files confirm the 4-GPU run).

### 2.4 Frame-rate maths & the `T//5` hypothesis (experimental)

- HuBERT native = **50 fps**. To approximate "~10 char-frames/sec" the hypothesis keeps the **first
  `N_i = T_i // 5` frames from the left** (no downsampling, feature dim stays 768).
- Per-utterance, variable: `T_valid = [99,249,449] → N = [19,49,89]` for `[2s,5s,9s]`.
- Use **exact `T_valid//5`** (from the frame mask), not `duration×50/5` (off by ~1).
- **Mechanical note:** since these are acoustic frames, "first `T//5`" = the **first 20% of the audio
  in time**, not a downsample of the whole clip. (Uniform downsample would be `hidden[:, ::5]`.)
  This was implemented as stated (left slice) per the experimental hypothesis.

### 2.5 Frame-rate measured from the data (heuristic)

From `libri960_train_text` + `segments` (chars/sec across utterances):

| Metric | mean | median | p95 | p99 | max |
|---|---|---|---|---|---|
| chars/sec | 14.3 | 14.5 | 18.5 | 19.9 | 22.0 |
| frames/char | 3.55 | 3.36 | 5.1 | 6.7 | 13.3 |
| content fraction (labels/T) | 0.29 | 0.30 | 0.38 | 0.40 | 0.46 |

→ ~14–15 chars/sec; ~71% of frames are `<fill>` in the original model. If a duration-based cap is
ever needed, use **rate ≈ 20** (p99), not 15 (mean) — over-cover is cheap, truncation is permanent.

### 2.6 Padding & masking across encoder and LLM (two independent stages)

- **Encoder padding** (waveform → `T_max`) is **scaffolding** to batch the convs. For `[2s,5s,9s]`
  padded to 9s, every row → `T_max=449`; real frames `[99,249,449]` via the **frame mask**
  (`hubert._get_feature_vector_attention_mask`). **You read `T_i` off the mask — no recompute.**
- The slice resets the length: `[19,49,89]` → re-pad to `N_max=89`. **The 449 is discarded; the LLM
  never sees it.**
- **LLM gets its OWN, smaller padding** (`N_max=89`, then `+prompt+target`). Encoder padding is *not*
  reused.
- **Attention with the mask:** the `attention_mask (B,L)` becomes an additive bias
  (`0` real, `≈-inf` pad) on the key axis → softmax gives padded keys weight 0. Padded queries are
  computed then discarded (`labels=-100`). Combined with the causal mask in the LLM.
- **Cost:** vanilla attention computes the full `L_max×L_max` matrix *then* masks → padding costs
  `O(L²)` FLOPs/memory + KV-cache. Hence slicing (449→89) matters; bucketing (`group_by_length`)
  reduces residual pad. Right-pad for training, left-pad for generation.
- **Projector ignores masks** — it's pointwise on the feature axis (768→d), processes every position
  (incl. pad), and masking is applied only at the LLM.

### 2.7 Variable length / batching (the reassurance)

- LLM handles **arbitrary** length but not **ragged** length: you pad to a rectangular `(B,L,d)` per
  batch and pass `attention_mask`; across batches `L` varies freely. **No fixed 30 s** (that's a
  Whisper-encoder constraint; HuBERT is fully length-flexible).
- Padding max is **per-batch** (dynamic), not global, unless you force `padding="max_length"`.
- Only the **LLM context window** is a hard ceiling; keep an outlier truncation cap as a safeguard.

### 2.8 The naive "pass-all-frames" baseline

Before frame selection: pass **all** HuBERT frames, freeze everything but a linear projector.

```python
class NaiveSlamASR(nn.Module):
    def __init__(self, hubert_ckpt, llm_name, prompt="Transcribe the speech:\n"):
        super().__init__()
        cfg = HubertConfig.from_pretrained(hubert_ckpt); cfg.apply_spec_augment = False
        self.encoder = FillerHubertModel.from_pretrained(hubert_ckpt, config=cfg).hubert
        self.encoder.eval(); [p.requires_grad_(False) for p in self.encoder.parameters()]
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16)
        [p.requires_grad_(False) for p in self.llm.parameters()]
        self.tok = AutoTokenizer.from_pretrained(llm_name); self.embed = self.llm.get_input_embeddings()
        self.projector = nn.Sequential(nn.Linear(768,2048), nn.GELU(),
                                       nn.Linear(2048, self.llm.config.hidden_size))
        self.register_buffer("prompt_ids",
            self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0], persistent=False)

    def train(self, mode=True):
        super().train(mode); self.encoder.eval(); self.llm.eval(); return self

    @torch.no_grad()
    def encode(self, input_values, sample_mask):
        h = self.encoder(input_values, attention_mask=sample_mask).last_hidden_state
        fmask = self.encoder._get_feature_vector_attention_mask(h.shape[1], sample_mask)
        return h, fmask

    def forward(self, input_values, sample_mask, target_ids, target_mask):
        B, dev = input_values.size(0), input_values.device
        h, fmask  = self.encode(input_values, sample_mask)                    # (B,T,768),(B,T)
        audio_emb = self.projector(h.float()).to(self.embed.weight.dtype)     # (B,T,d)
        p_ids = self.prompt_ids.to(dev).unsqueeze(0).expand(B, -1)
        prompt_emb, prompt_msk = self.embed(p_ids), torch.ones(B, p_ids.size(1), device=dev)
        target_emb = self.embed(target_ids)
        inputs_embeds  = torch.cat([prompt_emb, audio_emb, target_emb], dim=1)
        attention_mask = torch.cat([prompt_msk, fmask, target_mask], dim=1)
        ignore = torch.full((B, prompt_emb.size(1)+audio_emb.size(1)), -100, device=dev, dtype=torch.long)
        labels = torch.cat([ignore, target_ids.masked_fill(target_mask==0, -100)], dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
```

Frame selection later = a 2-line change: replace `h, fmask` with the sliced `audio, audio_mask`
(`N = T_valid//5`, `audio = h[:, :N.max()]`, `audio_mask = arange(N.max()) < N[:,None]`).

Caveats: add **EOS** to targets; keep the **fp32→fp16 cast** on the projector output; if unstable with
internal audio padding, pass `position_ids = attention_mask.long().cumsum(-1).sub(1).clamp(min=0)`.

---

## Key decisions & open hypotheses

- **Encoder = `checkpoint-164850`'s `.hubert`**, frozen + `eval()` + `apply_spec_augment=False`.
- **Feature = 768-d hidden states** (lm_head + dropout stripped) → projector input 768.
- **Experimental hypothesis (implement first, analyze later):** keep **first `T//5` frames from the
  left** (acoustic frames; = first 20% of audio in time), no downsampling, 768-d preserved.
- **Train only the projector** (LLM frozen; LoRA optional later).
- **Missing for the LLM phase:** `peft` (LoRA), `sentencepiece` (Llama tokenizer); `Jinja2`/`protobuf`
  already present. `numpy<2` pinned for the torch 2.7+cu126 stack.

## Numbers cheat-sheet

- Frame rate: **50 fps**, 20 ms/frame, stride **320**. `T ≈ duration×50`.
- `[2s,5s,9s]` → samples `[32000,80000,144000]` → frames `[99,249,449]` → `T//5` `[19,49,89]`.
- Effective training batch 256; checkpoint = 150 epochs, dev WER ≈ 0.50.
- chars/sec ≈ 14.3 (mean), content fraction ≈ 0.29.
- Vocab 33; `<fill>`=32, `<pad>`=29, `<s>`=30, `</s>`=31.
```
