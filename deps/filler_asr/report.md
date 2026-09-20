# Project Deep-Dive: HuBERT Fine-tuning on LibriSpeech — Complete Technical Report

## 0. TL;DR (5-sentence executive summary)

This project fine-tunes Facebook's **HuBERT-base** speech model (`facebook/hubert-base-ls960`) into a **character-level automatic speech recognition (ASR)** system on the full **960-hour LibriSpeech** corpus. Its defining and unusual feature is that it **does not use CTC loss** — the standard objective for this kind of model — but instead invents a **"filler-token" framewise cross-entropy** scheme: every 20 ms acoustic frame is given an explicit character label, the transcript is placed left-aligned at the start, and all remaining frames are labelled with a special `<fill>` token. Training is done with the Hugging Face `Trainer` (subclassed), a custom three-phase ("triphase") learning-rate schedule, and a two-stage freezing curriculum (train only the output head first, then everything except the CNN feature extractor). The end artifact is a HuBERT encoder plus a linear classification head that maps each frame to one of 33 character/special tokens; the best run reached ~**37.5 % WER** on dev-clean during training and **48.8 % WER / 14.5 % CER** on test-clean at inference. The project exists as a **research experiment** comparing this naïve framewise-CE alignment scheme against the conventional CTC approach — and the results document why CTC is normally preferred.

---

## 1. Project Overview

**Purpose and motivation.** The repository (internally named `filler_asr`, authored by "Arjun", run on a 10×A6000 GPU server) is a research codebase that explores an **alternative training objective for end-to-end ASR**. Conventional self-supervised speech encoders (wav2vec 2.0, HuBERT) are fine-tuned for transcription using **Connectionist Temporal Classification (CTC)**, which elegantly solves the problem that we know *what* was said but not *exactly when* each character was spoken (the "alignment problem"). This project deliberately replaces CTC with a much simpler idea: assign every output frame a hard label and train with ordinary per-frame cross-entropy. Frames that have no character assigned to them get a dedicated **`<fill>`** token — hence the name "filler ASR".

**What is being trained.** A pretrained HuBERT-base encoder is adapted so that, for each of its ~50 frames-per-second of output, it emits a probability distribution over a 33-symbol vocabulary (26 letters, apostrophe, word-delimiter `|`, and the special tokens `<unk> <pad> <s> </s> <fill>`). A single newly-initialised linear layer (`lm_head`) sits on top of the frozen-then-unfrozen encoder.

**What problem this solves (and the research angle).** Nominally it solves character-level English speech-to-text. The real point, though, is methodological: *can a transformer speech model learn to transcribe if you hand it a fixed, acoustically-naïve frame alignment instead of letting CTC marginalise over all alignments?* The answer the repository documents is "partially, but worse" — the model learns recognisable transcriptions yet produces characteristic doubled-letter garbling (e.g. `tens` → `tens`, but `provision` → `rovvision`) because the fixed left-aligned target does not match where sounds actually occur in time.

**End artifact capability.** After training you obtain a model that takes a raw 16 kHz waveform and outputs a character string transcription. It is a working, if not state-of-the-art, English ASR model. (For reference, a properly CTC-fine-tuned HuBERT-base reaches ~6–7 % WER on test-clean; this method reaches ~49 %.)

---

## 2. Repository Structure

Annotated tree of every tracked file (the `wandb/`, `__pycache__/`, and `data/` directories are run artifacts / caches, summarised rather than listed file-by-file).

```
filler_asr/
├── train.py                  # ENTRY POINT for training. Builds tokenizer, feature extractor,
│                             #   dataset, model, custom Trainer; runs trainer.train().
├── model.py                  # Two classes: FillerHubertModel (HuBERT + linear head) and
│                             #   FillerASRTrainer (subclass of HF Trainer with custom loss,
│                             #   triphase LR schedule, and freezing curriculum).
├── dataset.py                # Loads 960h LibriSpeech, cleans text, extracts char vocab,
│                             #   builds the framewise "filler" labels (the core trick).
├── collator.py               # DataCollatorForFillerASR: pads waveforms + labels into batches,
│                             #   masks padding labels to -100.
├── utils.py                  # compute_metrics: argmax-decode logits, compute WER & CER.
├── inference.py              # ENTRY POINT for inference/eval on test-clean or a single file.
├── test_ds.py                # Throwaway script: prints LibriSpeech config names. Not central.
├── config.json               # The active hyperparameter config (triphase, lr 3e-4, 200 epochs).
├── run_training.sh           # Launch script: activates conda env, runs torchrun (DDP).
├── vocab.json                # A STALE root-level vocab (special-tokens-first ordering).
│                             #   NOT the vocab actually used — see §3/§11. Authoritative vocab
│                             #   is regenerated per-experiment into experiments/<name>/vocab.json.
├── analysis.ipynb            # Near-empty exploratory notebook. One markdown note:
│                             #   "Check inference text - Should <fill> be weighed down in loss?"
├── .gitignore                # Ignores ./data (the multi-hundred-GB LibriSpeech cache).
├── .vscode/settings.json     # Editor config (conda env manager). Not central.
├── results/
│   └── test_clean_results.txt# Saved inference output: WER 0.4880, CER 0.1449 + all hyp/ref pairs.
├── experiments/              # Per-run saved artifacts (config, tokenizer, processor, vocab).
│   ├── filler_asr_960h/                    # earliest experiment variant
│   ├── filler_asr_960h_triphase/           # adds triphase LR schedule
│   └── filler_asr_960h_triphase_lr_epochs/ # active experiment (matches config.json)
│       ├── model/config.json    # HubertConfig with vocab_size=33, bos=30/eos=31/pad=29.
│       ├── vocab.json           # AUTHORITATIVE vocab (chars-first ordering).
│       ├── tokenizer_config.json# Wav2Vec2CTCTokenizer settings + added <fill> token.
│       ├── preprocessor_config.json # Wav2Vec2FeatureExtractor settings (16kHz, normalize).
│       └── added_tokens.json    # {"<fill>": 32}
└── wandb/                     # 11 Weights & Biases run directories (logs, configs, metrics).
                              #   The final run run-20260515_165213-p71qjaz5 reached the
                              #   reported metrics over 219,800 steps / 200 epochs / ~5.8 days.
```

**Note on the three `experiments/` folders.** They are three successive iterations of the *same* method, differing only in LR schedule and epoch budget (their model `config.json` files are byte-identical HuBERT-base configs). Only the configuration/tokenizer/processor files are checked in; the **actual trained weights live in `checkpoint-*/` directories that are not present in the repo** (they would be written under `experiments/<name>/` at train time but are excluded/absent here).

---

## 3. Dataset: LibriSpeech

**What LibriSpeech is.** LibriSpeech is a widely-used, freely-available English read-speech corpus derived from audiobooks in the LibriVox project, aligned to their source texts. It totals roughly **1,000 hours** of 16 kHz English speech. It is partitioned by recording quality ("clean" = easier/clearer speakers, "other" = noisier/harder) and split into standard training, development, and test sets. Each example is a short utterance (a sentence or sentence fragment), a few seconds to ~30 seconds long, paired with its uppercase punctuation-free reference transcript.

**Which splits are used and for what.** Defined in [dataset.py](dataset.py#L47-L58):

| Split (HF name) | LibriSpeech name | Hours | Role in this project | # utterances (approx) |
|-----------------|------------------|-------|----------------------|------------------------|
| `clean / train.100` | train-clean-100 | 100 | training | 28,539 |
| `clean / train.360` | train-clean-360 | 360 | training | 104,014 |
| `other / train.500` | train-other-500 | 500 | training | 148,688 |
| `clean / validation` | dev-clean | ~5.4 | **eval during training** (`dataset["test"]`) | 2,703 |
| `clean / test` | test-clean | ~5.4 | **final inference eval** (in `inference.py` only) | 2,620 |

The three training splits are concatenated into the full **"960h"** training set (~**281,241** utterances total — this number is confirmed by the step math: 281,241 ÷ 256 effective-batch ≈ 1,099 steps/epoch × 200 epochs = 219,800 steps, exactly the `global_step` in the wandb logs). Note a slightly confusing naming choice: the variable `dataset["test"]` is actually **dev-clean**, used for periodic evaluation during training; the real **test-clean** is only touched by `inference.py`.

**How the dataset is loaded in code.** Via the Hugging Face `datasets` library, `load_dataset("librispeech_asr", "clean"/"other", split=...)` in [dataset.py:53-58](dataset.py#L53-L58), cached to a local `./data` directory (git-ignored). The splits are merged with `concatenate_datasets` ([dataset.py:57](dataset.py#L57)).

**Raw data format.** Each record has an `audio` column (a dict with `array` = float waveform, `sampling_rate`, `path`) and a `text` column (the transcript). Audio is force-resampled to **16 kHz** mono via `cast_column("audio", Audio(sampling_rate=16000))` ([dataset.py:72](dataset.py#L72)).

**Dataset-specific preprocessing** ([dataset.py:65-72](dataset.py#L65-L72)):
1. **Text normalisation** — `remove_special_characters` strips the characters `, ? . ! - ; : "` via the regex `CHARS_TO_IGNORE_REGEX` and lowercases the text. The result is lowercase letters, spaces, and apostrophes only.
2. **Resampling** — guarantees 16 kHz.
3. **Vocabulary extraction** — `extract_vocab` ([dataset.py:18-45](dataset.py#L18-L45)) iterates the *training* text, collects every unique non-space character, sorts them, and assigns indices, then appends `| <unk> <pad> <s> </s>`. For LibriSpeech the sorted character set is `'` (apostrophe) then `a–z`, giving the authoritative vocab in [experiments/filler_asr_960h_triphase_lr_epochs/vocab.json](experiments/filler_asr_960h_triphase_lr_epochs/vocab.json): `'`=0, a=1…z=26, `|`=27, `<unk>`=28, `<pad>`=29, `<s>`=30, `</s>`=31, plus `<fill>`=32 added afterwards → **33 tokens**.

---

## 4. HuBERT Background (what the reader needs to know)

**What HuBERT is.** HuBERT ("**Hu**bert = **H**idden-**u**nit **BERT**", Hsu et al., 2021, Facebook AI) is a **self-supervised speech representation model**. Its job in pretraining is not transcription but learning rich acoustic features from *unlabelled* audio. Architecturally it is: a **convolutional feature extractor** (turns the raw waveform into ~50 feature vectors per second) followed by a **BERT-style Transformer encoder** (12 layers for the base model).

**How it was pretrained (the clever part).** HuBERT is trained in iterative rounds:
1. **Offline k-means clustering.** Take some acoustic features (MFCCs in round 1; HuBERT's own intermediate layer outputs in later rounds), run k-means to assign every frame to one of *k* clusters. These cluster IDs are the "hidden units" — pseudo-labels that stand in for unknown phonetic categories.
2. **Masked prediction (BERT-style).** Randomly mask spans of the input frames, feed the audio through the model, and ask it to **predict the cluster ID of the masked frames** from context. This is a classification loss over the k-means codebook, computed only on masked positions.
3. **Iterate.** Re-cluster using better features from the trained model and repeat. `facebook/hubert-base-ls960` was pretrained this way on the 960-hour LibriSpeech audio (no transcripts used).

**How it differs from wav2vec 2.0 and Whisper.**
- **vs wav2vec 2.0:** wav2vec 2.0 uses a *contrastive* loss with a learned, quantized codebook trained jointly end-to-end. HuBERT instead uses *offline k-means targets* and a plain cross-entropy masked-prediction loss — simpler and often more stable. Architecturally the two are nearly identical (CNN + Transformer), which is why this codebase reuses wav2vec2 tokenizer/feature-extractor classes for a HuBERT model.
- **vs Whisper:** Whisper is a *supervised*, *sequence-to-sequence* (encoder–decoder with attention) model trained on 680k hours of weakly-labelled (audio, text) pairs; it outputs text autoregressively and needs no fine-tuning. HuBERT is *self-supervised*, encoder-only, produces frame-level features, and **must be fine-tuned** with a task head (CTC or, here, the filler scheme) to transcribe.

**What the pretrained model provides vs what fine-tuning adds.** Pretraining gives a HuBERT encoder whose 768-dimensional frame embeddings encode phonetic/linguistic content — but it has **no notion of characters or text**. Fine-tuning adds (a) a brand-new linear `lm_head` mapping 768-dim frames to the 33-token vocabulary, and (b) the supervised signal (here, framewise CE against filler labels) that teaches the encoder+head to map acoustics to characters.

**Exact model variant used.** From [config.json:6](config.json#L6) and every `experiments/*/model/config.json`:
- **Checkpoint:** `facebook/hubert-base-ls960` (Hugging Face Hub).
- **`model_type`: hubert**, **`num_hidden_layers`: 12**, **`hidden_size`: 768**, **`num_attention_heads`: 12**, **`intermediate_size`: 3072**, **`num_feat_extract_layers`: 7**.
- **Parameter count:** ~**94.4 M** total (12-layer base). The CNN feature extractor is ~4.2 M of those; the rest (~90 M) is the feature projection + Transformer. The new `lm_head` adds exactly **25,377** params (768×33 + 33 bias).
- The config carries `vocab_size: 33`, `bos_token_id: 30`, `eos_token_id: 31`, `pad_token_id: 29` — overwritten from the tokenizer in [train.py:107-110](train.py#L107-L110).

---

## 5. Architecture: Complete Forward Pass

The forward pass lives in [model.py:35-66](model.py#L35-L66) (`FillerHubertModel.forward`). It is: **raw waveform → CNN feature extractor → feature projection → 12 Transformer layers → dropout → linear head → per-frame logits**. Every dimension below is derived from the model config and the convolution arithmetic in [dataset.py:8-16](dataset.py#L8-L16), not assumed.

### 5.1 Input Specification

- **Raw audio:** a 1-D float32 array sampled at **16,000 Hz** (`sampling_rate=16000`, `feature_size=1` in the `Wav2Vec2FeatureExtractor`, [train.py:65-71](train.py#L65-L71)). After the feature extractor it is **zero-mean/unit-variance normalised** per utterance (`do_normalize=True`).
- **Shape into the model:** `input_values` of shape **(B, T_s)** where `B` = batch size and `T_s` = number of audio samples (e.g. 4 s → 64,000 samples).
- **Padding/truncation:** there is *no fixed truncation*. Within a batch, waveforms are right-padded to the longest member with `padding_value=0.0`, and an **attention mask** (1 = real, 0 = pad) is built by the collator/feature-extractor. Label construction can truncate text if the transcript would not fit in the frame budget (see §5.6).

### 5.2 Feature Extraction (CNN Encoder)

HuBERT-base's feature extractor is **7 1-D convolution layers** with GELU activations and group normalization on the first layer (`feat_extract_norm: group`, `conv_bias: false`). From the config:
- **`conv_dim`:** `[512, 512, 512, 512, 512, 512, 512]` — every layer outputs 512 channels.
- **`conv_kernel`:** `[10, 3, 3, 3, 3, 2, 2]`.
- **`conv_stride`:** `[5, 2, 2, 2, 2, 2, 2]`.

Each layer reduces the time axis by `(L − kernel) // stride + 1`. The combined stride product is **5×2×2×2×2×2×2 = 320**, i.e. the model emits roughly **one frame per 320 audio samples = one frame per 20 ms = ~50 frames/second**. This exact arithmetic is reimplemented in `compute_output_length` ([dataset.py:8-16](dataset.py#L8-L16)) so the dataset code can predict the frame count `T` for label construction.

Worked example for a 4-second clip (64,000 samples):

| Conv layer | kernel | stride | input length | output length |
|-----------:|:------:|:------:|-------------:|--------------:|
| 0 | 10 | 5 | 64,000 | 12,799 |
| 1 | 3 | 2 | 12,799 | 6,399 |
| 2 | 3 | 2 | 6,399 | 3,199 |
| 3 | 3 | 2 | 3,199 | 1,599 |
| 4 | 3 | 2 | 1,599 | 799 |
| 5 | 2 | 2 | 799 | 399 |
| 6 | 2 | 2 | 399 | **199** |

So 4 s → **T = 199 frames**, each a **512-dim** vector. Output of this stage: **(B, 512, T)**, transposed to **(B, T, 512)**.

### 5.3 Feature Projection

A linear layer projects 512 → **768** (`hidden_size`), with a LayerNorm (`feat_proj_layer_norm: true`) and dropout (`feat_proj_dropout: 0.1`). Output: **(B, T, 768)**.

### 5.4 Positional Convolution + Transformer Encoder

A grouped 1-D convolution (`num_conv_pos_embeddings: 128`, `num_conv_pos_embedding_groups: 16`) adds relative positional information, then **12 Transformer encoder layers** run. Each layer: 12-head self-attention (768-dim, so 64 dims/head), a feed-forward block 768→3072→768 with GELU, residual connections, LayerNorm (post-norm, since `do_stable_layer_norm: false`), and dropouts of 0.1. **SpecAugment-style masking** (`apply_spec_augment: true`, `mask_time_prob: 0.05`, `mask_time_length: 10`) is applied to the features **during training only**. `layerdrop: 0.1` randomly skips whole layers in training. Output (`last_hidden_state`): **(B, T, 768)**.

### 5.5 (Implicit) — what the model takes from HuBERT

`FillerHubertModel.forward` calls `self.hubert(...)` and takes `outputs[0]` = the last hidden state, shape **(B, T, 768)** ([model.py:46-54](model.py#L46-L54)). The intermediate hidden states/attentions are passed through but unused by the head.

### 5.6 Output Head (Fine-tuning specific)

On top of HuBERT:
1. **Dropout** with `final_dropout = 0.1` ([model.py:14](model.py#L14), [model.py:55](model.py#L55)).
2. **`lm_head`** = `nn.Linear(768, 33)` ([model.py:15](model.py#L15), [model.py:56](model.py#L56)).

Output **logits** shape: **(B, T, 33)**. Position `[b, t, :]` is an unnormalised score distribution over the 33 vocabulary symbols for **frame _t_ of utterance _b_**. After a softmax (implicit in cross-entropy) and argmax, each frame is classified as exactly one character or special token. **Each output position represents one ~20 ms acoustic frame**, and the model is being asked: "which character (or `<fill>`) belongs at this frame?"

The returned object is a `CausalLMOutput` with `loss=None` (loss is deliberately computed in the Trainer, not the model — see §6.2), `logits`, `hidden_states`, `attentions` ([model.py:61-66](model.py#L61-L66)).

### 5.7 Complete Dimension Map (summary table)

(B = batch, T_s = audio samples, T = frames ≈ T_s/320, derived from config `conv_*` and `hidden_size`/`vocab_size`.)

| Stage | Operation | Input Shape | Output Shape | Notes |
|-------|-----------|-------------|--------------|-------|
| 0 | Raw waveform | (B, T_s) | (B, T_s) | 16 kHz, float32, normalised |
| 1 | CNN feature extractor (7 conv layers) | (B, T_s) | (B, T, 512) | stride product = 320; ~50 fps |
| 2 | Feature projection (Linear + LN + dropout) | (B, T, 512) | (B, T, 768) | 512 → `hidden_size`=768 |
| 3 | Positional conv embedding | (B, T, 768) | (B, T, 768) | grouped conv, kernel 128 |
| 4 | 12× Transformer encoder layers | (B, T, 768) | (B, T, 768) | 12 heads, FFN 3072, SpecAugment in train |
| 5 | Final dropout | (B, T, 768) | (B, T, 768) | `final_dropout`=0.1 |
| 6 | `lm_head` Linear | (B, T, 768) | (B, T, 33) | 768×33+33 = 25,377 params |
| 7 | argmax over last dim (decode) | (B, T, 33) | (B, T) | one token id per frame |
| 8 | tokenizer decode (drop specials, no grouping) | (B, T) | strings | `|`→space, `<fill>/<s>/</s>/<pad>` removed |

---

## 6. Fine-tuning Strategy

### 6.1 What is Frozen vs Trainable

Freezing is dynamic and curriculum-based, implemented in `FillerASRTrainer` ([model.py:145-181](model.py#L145-L181)) and `FillerHubertModel`'s helper methods ([model.py:20-33](model.py#L20-L33)):

- **CNN feature extractor:** **frozen for the whole run** when `freeze_feature_extractor: true` (the config's setting). `freeze_feature_extractor()` calls `self.hubert.feature_extractor._freeze_parameters()` ([model.py:20-21](model.py#L20-L21)). This is standard practice — the low-level acoustic CNN is already well-trained and freezing it stabilises fine-tuning. ~4.2 M params frozen.
- **Two-phase curriculum** driven by `classifier_only_train_ratio` (= **0.1** in config):
  - **Phase A — "classifier_only" (first 10 % of steps).** `freeze_all_except_classifier()` ([model.py:23-27](model.py#L23-L27)) sets `requires_grad=False` on everything and re-enables only `lm_head`. Only **25,377 params** train. The wandb log confirms this: it prints *"Freezing model: training lm_head only."* at step 0. Purpose: let the randomly-initialised head settle before perturbing the pretrained encoder.
  - **Phase B — "full_except_feature_extractor" (remaining 90 %).** `unfreeze_all_except_feature_extractor(freeze_feature_extractor=True)` ([model.py:29-33](model.py#L29-L33)) re-enables grads on the Transformer + projection + head but re-freezes the CNN. **~90 M params** train.
  - The switch happens inside `training_step` → `_maybe_update_freezing_stage` ([model.py:168-181](model.py#L168-L181), [model.py:243-245](model.py#L243-L245)), at `global_step ≥ ceil(max_steps × 0.1)`. If `classifier_only_train_ratio ≤ 0`, the model goes straight to Phase B.
- **No parameter-efficient method** (no LoRA, adapters, or prefix tuning). This is a **full fine-tune** of the encoder (minus the CNN).

**Trainable vs total parameter count:** Total ≈ **94.4 M**. Phase A trainable = **25,377** (≈0.03 %). Phase B trainable ≈ **90 M** (everything except the ~4.2 M CNN).

### 6.2 Loss Function

**This is the crux of the project.** The model is configured as if it were a CTC model (it carries `ctc_loss_reduction`, the tokenizer is `Wav2Vec2CTCTokenizer`), but the Trainer **overrides `compute_loss`** to use plain **cross-entropy** instead ([model.py:247-272](model.py#L247-L272)):

```python
labels = inputs.pop("labels")          # (B, T) — one target id per frame
outputs = model(**inputs)              # logits (B, T, 33); labels withheld so no CTC
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
min_seq_len = min(logits.shape[1], labels.shape[1])   # safety truncation
loss = loss_fct(logits.view(-1, 33), labels.view(-1)) # framewise CE
```

- It is **not CTC**. There is no blank token, no alignment marginalisation, no length-collapse. Instead **every frame is a classification example**, and loss is the mean cross-entropy over all (non-`-100`) frames in the batch.
- **`ignore_index=-100`** means padded-frame labels (set to −100 by the collator, §7.1) contribute nothing to the loss.
- **Mathematical form:** for batch with frames indexed by `i`, label `y_i`, logits `z_i ∈ ℝ³³`:
  `L = (1/N) Σ_i −log( softmax(z_i)[y_i] )`, summed over the `N` frames where `y_i ≠ −100`.
- **Input/output length relationship.** Because there is no CTC collapse, the **target length must equal the frame count T**. This is exactly why the dataset builds labels of length `T` (transcript at the front, `<fill>` for the rest, §5.6 / [dataset.py:74-101](dataset.py#L74-L101)). The `min_seq_len` truncation ([model.py:266-268](model.py#L266-L268)) guards against off-by-one mismatches between the collator's frame count and `lm_head`'s actual output length.
- **Why this matters / the catch.** Most frames in any utterance are labelled `<fill>` (only ~`len(text)+2` of `T` frames carry characters), so `<fill>` massively dominates the loss — a class-imbalance problem the author explicitly flagged in `analysis.ipynb`: *"Should `<fill>` be weighed down in loss?"* It currently is **not** down-weighted.

### 6.3 Optimizer and Schedule

From [train.py:39-62](train.py#L39-L62) and the custom scheduler in [model.py:117-241](model.py#L117-L241):

- **Optimizer:** Hugging Face default **AdamW** (no override of the optimizer *class*; `create_optimizer` only customises weight-decay grouping — decay applies to weights but **not** biases/LayerNorm params, [model.py:183-221](model.py#L183-L221)).
- **Learning rate:** `learning_rate = 3e-4` (config); `weight_decay = 0.005`; AdamW betas/eps left at HF defaults (0.9, 0.999, 1e-8).
- **LR schedule — custom "triphase"** (`lr_schedule_type: "triphase"`, ratios `[0.1, 0.4, 0.5]`). `create_scheduler` installs a `LambdaLR` ([model.py:223-241](model.py#L223-L241)) implementing three phases over the total `num_training_steps` ([model.py:117-133](model.py#L117-L133)):
  1. **Warmup (first 10 %):** LR ramps **linearly 0 → 3e-4**.
  2. **Constant (next 40 %):** LR held at **3e-4**.
  3. **Linear decay (final 50 %):** LR decays **3e-4 → 0**.
  The log confirms the warmup (early LRs like 6.7e-7 rising) and the near-zero final LR (`2.73e-7` at step 219,800). Note: `TrainingArguments.lr_scheduler_type` is logged as `"linear"`, but the subclass's `create_scheduler` **replaces** it with triphase whenever `lr_schedule_type=="triphase"`.
- **Gradient clipping:** HF `Trainer` default **max-grad-norm = 1.0** (logs show `grad_norm` hovering ~1–3, consistent with clipping at 1.0 not being the active config default... the default is 1.0; values >1 are pre-clip norms).
- **Gradient accumulation:** `gradient_accumulation_steps = 4`.
- **Mixed precision:** `fp16=True` when CUDA is available; **gradient checkpointing on** (`gradient_checkpointing=True`) to save memory.

---

## 7. Training Pipeline

### 7.1 Data Collation / Batching

`DataCollatorForFillerASR` ([collator.py](collator.py)) handles variable-length audio and labels:
- Splits each feature dict into `input_values` (waveform) and `labels`.
- **Waveforms** are padded with the feature extractor (`feature_extractor.pad(..., padding=True)`), `padding_value=0.0`, right side; this also returns the **attention mask** (1=real,0=pad).
- **Labels** are padded with `tokenizer.pad(...)`. Crucially, label positions that are padding (per the label attention mask) are **replaced with `-100`** via `masked_fill` ([collator.py:29](collator.py#L29)) so cross-entropy ignores them.
- Output batch dict: `input_values (B,T_s)`, `attention_mask (B,T_s)`, `labels (B,T)`.

### 7.2 Training Loop

- **Framework:** Hugging Face **`Trainer` API**, subclassed as `FillerASRTrainer` ([model.py:69](model.py#L69)). No raw PyTorch loop, no Lightning.
- **Distributed:** launched with **`torchrun`** (DDP) via `run_training.sh`, `--nproc_per_node = num_processes` (= **4** from config) on a 10-GPU A6000 box. `ddp_find_unused_parameters=True` (needed because the freezing curriculum leaves some params grad-less in Phase A).
- **Effective batch size:** `per_device_train_batch_size (16) × gradient_accumulation_steps (4) × world_size (4) = 256`.
- **Steps per epoch:** ⌈281,241 / 256⌉ = **1,099**. With `num_train_epochs = 200` → **219,800 optimizer steps** (matches wandb `global_step`). Total wall-clock ≈ **504,818 s ≈ 5.84 days**.
- **Logging:** every `logging_steps = 50` steps to **Weights & Biases** (`report_to="wandb"`, `WANDB_PROJECT="filler_asr"`). Logged: `loss`, `grad_norm`, `learning_rate`, `epoch`.
- **Checkpointing:** every `save_steps = 1000` steps, keeping `save_total_limit = 2` most recent. On startup, `train.py` auto-detects and **resumes from the newest `checkpoint-*`** dir ([train.py:148-155](train.py#L148-L155)).
- **Artifacts saved on the main process** before training: the base HuBERT (`model.hubert.save_pretrained(.../model)`), feature extractor, and tokenizer into `experiments/<exp_name>/` ([train.py:122-125](train.py#L122-L125)).

### 7.3 Evaluation During Training

- **When:** every `eval_steps = 500` (`eval_strategy="steps"`), on **dev-clean**.
- **Metrics:** **WER** (Word Error Rate) and **CER** (Character Error Rate) via the `evaluate` library (`jiwer` backend) in `get_compute_metrics_fn` ([utils.py](utils.py)).
- **Decoding:** **greedy argmax** over logits (`np.argmax(pred_logits, axis=-1)`), then `tokenizer.batch_decode(..., skip_special_tokens=True, group_tokens=False)`. The `group_tokens=False` is critical and deliberate ([utils.py:18-21](utils.py#L18-L21)): a CTC tokenizer would normally *merge adjacent identical tokens*, but here there is no CTC, so merging must be disabled or real doubled letters would be wrongly collapsed.
- **Normalisation:** labels' `-100` are reset to `pad_token_id` before decoding so they can be turned into strings ([utils.py:13-14](utils.py#L13-L14)); special tokens (`<fill> <s> </s> <pad>`) are dropped by `skip_special_tokens=True`; `|` becomes a space.
- **A known metric artifact:** the in-training `eval/cer` is wildly inflated (it starts at **14.0 = 1400 %** and ends ~**8.9**), while `eval/wer` is sane (1.09 → **0.375**). The cause: during batched eval, short utterances are padded to the longest in the batch, and the model still predicts *characters* for those padded frames; with `group_tokens=False` these produce many spurious inserted characters, exploding CER. The single-utterance inference path (`inference.py`, no batch padding) does not suffer this and reports the realistic **CER 0.1449** on test-clean.

---

## 8. Input → Output: End-to-End Walkthrough

Concrete example using the first test-clean utterance (from `results/test_clean_results.txt`), reference = **"concord returned to its place amidst the tents"**. Assume the audio is ~4.0 s long.

**Step 1 — Raw audio.** 4.0 s × 16,000 Hz = **64,000 samples**, shape `(64000,)`, float32.

**Step 2 — Feature extraction (normalise).** `Wav2Vec2FeatureExtractor` zero-mean/unit-var normalises → `input_values` shape `(1, 64000)` (batch of 1).

**Step 3 — Frame count.** `compute_output_length(64000)` = **199** frames (table in §5.2). So `T = 199`.

**Step 4 — Label construction (training only)** ([dataset.py:83-101](dataset.py#L83-L101)). Text → lowercase, spaces → `|`: `concord|returned|to|its|place|amidst|the|tents`. Tokenize to char ids (≈47 tokens incl. delimiters). Then:
`labels = [<s>] + char_ids + [</s>]` (≈49 tokens), then pad with `<fill>` to length 199:
`[<s>, c, o, n, c, o, r, d, |, r, e, t, …, s, </s>, <fill>, <fill>, …, <fill>]` (199 entries; ~150 are `<fill>`). The transcript is **left-aligned at the front**; the model is told nothing about *when* "concord" is actually spoken in the 4 seconds.

**Step 5 — Forward pass.** `input_values (1,64000)` → CNN → `(1,199,512)` → projection → `(1,199,768)` → 12 Transformer layers → `(1,199,768)` → dropout → `lm_head` → **logits `(1,199,33)`**.

**Step 6 — Loss (training).** Cross-entropy between `logits (1,199,33)` and `labels (1,199)`, averaged over the 199 frames (padding-only frames would be −100, but here T fills exactly). The `<fill>` frames dominate.

**Step 7 — Decode (inference).** `argmax` over dim −1 → `(1,199)` token ids. `batch_decode(skip_special_tokens=True, group_tokens=False)`: drop `<s>/</s>/<fill>`, map `|`→space, concatenate the rest. Because frames where a phoneme is held produce repeated character predictions and the alignment is acoustically fixed, the output comes out as **"concord returned to its place amids the tens"** — recognisable but with dropped/garbled trailing characters (`amidst`→`amids`, `tents`→`tens`). This single example's errors illustrate the method's systematic weakness.

---

## 9. What the Model Produces After Training

**Saved artifacts** (under `experiments/<exp_name>/`):
- `model/config.json` — the `HubertConfig` (with `vocab_size=33` etc.).
- `vocab.json`, `tokenizer_config.json`, `added_tokens.json` — the `Wav2Vec2CTCTokenizer` (chars + specials + `<fill>`).
- `preprocessor_config.json` — the `Wav2Vec2FeatureExtractor` settings.
- `checkpoint-*/` directories — the **actual trained weights** (`model.safetensors` incl. fine-tuned encoder + `lm_head`), optimizer/scheduler state, RNG, trainer state. (These are produced at train time and are **not** committed in this repo snapshot.)
- `results/test_clean_results.txt` — final WER/CER and every hypothesis/reference pair.

**How to load the trained model for inference** (mirrors `inference.py`):

```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, HubertConfig
from model import FillerHubertModel
import torch

exp = "experiments/filler_asr_960h_triphase_lr_epochs"
ckpt = exp + "/checkpoint-219800"   # newest checkpoint dir

tok = Wav2Vec2CTCTokenizer(exp + "/vocab.json", unk_token="<unk>", pad_token="<pad>",
                           word_delimiter_token="|", bos_token="<s>", eos_token="</s>")
tok.add_special_tokens({"additional_special_tokens": ["<fill>"]})
fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                              do_normalize=True, return_attention_mask=True)

cfg = HubertConfig.from_pretrained(ckpt)
cfg.vocab_size = len(tok); cfg.pad_token_id = tok.pad_token_id
cfg.bos_token_id = tok.bos_token_id; cfg.eos_token_id = tok.eos_token_id
model = FillerHubertModel.from_pretrained(ckpt, config=cfg, ignore_mismatched_sizes=True).eval()
```

**Minimal inference** (from [inference.py:24-30](inference.py#L24-L30)):

```python
inputs = fe(audio_array, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits                       # (1, T, 33)
ids = torch.argmax(logits, dim=-1)                    # (1, T)
text = tok.batch_decode(ids, skip_special_tokens=True, group_tokens=False)[0]
```

---

## 10. How to Run This Project

**Setup.**
1. Create the conda env (the launch script expects one named **`asr_filler`**). Key dependencies (from the run's `requirements.txt`): `torch==2.7.0+cu126`, `transformers==5.8.0`, `datasets==4.8.5`, `accelerate==1.13.0`, `evaluate==0.4.6`, `jiwer==4.0.0`, `wandb==0.26.1`, `soundfile`, `torchaudio`, `torchcodec`, `numpy==1.26.4`. Python 3.10.
2. Log in to Weights & Biases (`wandb login`) or set `WANDB_MODE=offline`.
3. The first run will **download ~960 h of LibriSpeech** into `./data` (git-ignored, hundreds of GB).

**Run training** (multi-GPU via the provided script):
```bash
bash run_training.sh          # reads num_processes from config.json, calls torchrun
```
or directly:
```bash
torchrun --nproc_per_node=4 --master_port=29501 train.py --config_path config.json
# single GPU / quick smoke test:
python train.py --config_path config.json --max_steps 100
```

**Key configurable hyperparameters** ([config.json](config.json)) and what they control:

| Key | Value | Controls |
|-----|-------|----------|
| `model_checkpoint` | `facebook/hubert-base-ls960` | which pretrained HuBERT to fine-tune |
| `learning_rate` | `3e-4` | peak LR (with triphase warmup/decay) |
| `per_device_train_batch_size` | `16` | per-GPU batch |
| `gradient_accumulation_steps` | `4` | micro-batches per optimizer step |
| `num_train_epochs` | `200` | training length |
| `lr_schedule_type` | `triphase` | selects the custom 3-phase LR scheduler |
| `lr_schedule_ratios` | `[0.1,0.4,0.5]` | warmup / constant / decay fractions |
| `classifier_only_train_ratio` | `0.1` | fraction of steps training only `lm_head` |
| `freeze_feature_extractor` | `true` | keep the CNN frozen throughout |
| `weight_decay` | `0.005` | AdamW weight decay (weights only) |
| `eval_steps` / `save_steps` / `logging_steps` | 500 / 1000 / 50 | eval / checkpoint / log cadence |
| `num_processes` | `4` | DDP world size **and** dataloader/`map` workers |
| `max_steps` | `-1` | overrides epoch budget if set |

**Run evaluation / inference** ([inference.py](inference.py)):
```bash
# Full test-clean WER/CER (writes results/test_clean_results.txt):
python inference.py --config_path config.json
# Single file:
python inference.py --config_path config.json --audio_path my.wav --output_path out.txt
```

---

## 11. Key Design Decisions & Non-Obvious Details

1. **Framewise cross-entropy instead of CTC (the headline choice).** *What:* every frame gets a hard character/`<fill>` label and CE is applied per frame ([model.py:247-272](model.py#L247-L272)). *Why:* simplicity / a research probe into whether fixed alignments can replace CTC's alignment marginalisation. *Impact:* trains and converges, but transcriptions are systematically garbled with doubled/dropped letters because the fixed left-aligned target ignores real acoustic timing — the model partly follows the (wrong) training alignment and partly the acoustics. This is the central reason the WER is ~49 % rather than ~6 %.

2. **The `<fill>` token & left-aligned labels.** *What:* transcript packed at the front, `<fill>` for the rest, up to `T` frames ([dataset.py:87-100](dataset.py#L87-L100)). *Why:* CE needs target length = T. *Impact:* severe class imbalance — most targets are `<fill>` — so the loss is dominated by predicting "nothing here". The author's own notebook asks whether `<fill>` should be down-weighted; it currently is not.

3. **A CTC tokenizer used in a non-CTC way.** *What:* `Wav2Vec2CTCTokenizer` is used, but decoding always passes `group_tokens=False` ([utils.py:20-21](utils.py#L20-L21), [inference.py:21](inference.py#L21)). *Why:* CTC tokenizers merge repeated adjacent tokens; that would wrongly collapse legitimately repeated letters here. *Impact:* repeated-frame predictions survive into the output (the doubled-letter artifact), but disabling grouping is the correct choice given no CTC.

4. **Two-stage freezing curriculum.** *What:* train `lm_head` only for the first 10 % of steps, then unfreeze everything but the CNN ([model.py:168-181](model.py#L168-L181)). *Why:* lets the random head warm up before disturbing pretrained weights — a common stabilisation trick. *Impact:* `ddp_find_unused_parameters=True` is required, and the freeze switch is logged.

5. **Custom "triphase" LR schedule.** *What:* linear warmup → constant plateau → linear decay, ratios `[0.1,0.4,0.5]` ([model.py:117-133](model.py#L117-L133)). *Why:* a long constant phase at peak LR for the bulk of training, common in speech fine-tuning. *Impact:* overrides HF's logged `linear` scheduler.

6. **`loss=None` from the model; loss computed in Trainer.** *What:* `forward` returns `loss=None` ([model.py:62](model.py#L62)); `compute_loss` does the CE. *Why:* prevents HF's HubertForCTC-style automatic CTC loss; gives full control. *Impact:* you cannot call this model with `labels=` and expect a loss outside the custom Trainer.

7. **Frozen CNN feature extractor.** Standard, stabilising; ~4.2 M params never updated.

8. **`min_seq_len` truncation safety net** ([model.py:266-268](model.py#L266-L268)). Guards against rare ±1 mismatches between the collator's predicted `T` (from `compute_output_length`) and the model's actual conv output length.

9. **Stale root `vocab.json` vs authoritative experiment vocab.** *What:* the root `vocab.json` lists specials first (`<pad>`=0…`<fill>`=4, then letters); the experiment `vocab.json` (and the model config: `bos=30/eos=31/pad=29`) lists chars first. *Why/Impact:* the training pipeline **regenerates** vocab via `extract_vocab` into `experiments/<exp>/vocab.json` and uses *that*; the root file is leftover and is **not** the one the trained model uses. A reader must use the experiment vocab to interpret token ids.

10. **`dataset["test"]` is dev-clean, not test-clean.** Naming trap ([dataset.py:58](dataset.py#L58)): the "test" used during training is actually the dev split; true test-clean is only in `inference.py`.

11. **Inflated in-training CER.** Eval batching pads short clips; the model emits characters on padded frames, exploding CER (starts at 1400 %). Real CER (single-utterance inference) is 14.5 %. A subtle but important caveat when reading the wandb numbers.

12. **`remove_unused_columns=False`.** Required ([train.py:61](train.py#L61)) so HF doesn't strip the custom `input_values`/`labels` columns before the collator sees them.

13. **Three near-identical experiment folders** track the method's iteration (plain → triphase → triphase+more-epochs); their model configs are byte-identical, so the *only* differences are training schedule/budget in `config.json`.

---

## 12. Glossary

- **HuBERT** — Hidden-unit BERT; a self-supervised speech model (CNN + Transformer) pretrained by predicting offline k-means cluster IDs of masked audio frames. Here: `facebook/hubert-base-ls960`, 12 layers, 768-dim, ~94 M params.
- **Self-supervised pretraining** — learning from unlabelled data via a pretext task (masked prediction), producing reusable features.
- **Fine-tuning** — adapting a pretrained model to a labelled downstream task (here: speech→characters).
- **CTC (Connectionist Temporal Classification)** — the standard ASR loss that aligns a short transcript to many audio frames by introducing a "blank" token and summing over all valid alignments; lets you train without frame-level labels. **Not used here.**
- **Cross-entropy loss** — classification loss `−log p(correct class)`; used here per frame.
- **`<fill>` token** — this project's special label for frames not covered by the transcript; the source of the "filler" name.
- **LibriSpeech** — ~1000 h English read-speech corpus from audiobooks; splits clean/other × train/dev/test. "960h" = train-clean-100 + train-clean-360 + train-other-500.
- **WER (Word Error Rate)** — edit distance over words ÷ reference word count; lower is better.
- **CER (Character Error Rate)** — same but over characters.
- **Feature extractor (CNN encoder)** — the 7-layer 1-D conv stack turning a 16 kHz waveform into ~50 frames/s of 512-dim vectors (downsample factor 320).
- **Transformer encoder** — stack of self-attention + feed-forward layers producing contextualised frame embeddings (768-dim).
- **`lm_head`** — the linear layer (768→33) mapping each frame embedding to vocabulary logits.
- **Logits** — raw, pre-softmax scores; shape `(B, T, 33)` here.
- **Frame** — one ~20 ms time step of model output (≈50 per second).
- **Greedy/argmax decoding** — pick the highest-probability token per frame (no beam search, no language model).
- **SpecAugment** — training-time masking of feature regions for regularisation.
- **AdamW** — Adam optimizer with decoupled weight decay.
- **Warmup / triphase schedule** — LR ramp-up then plateau then decay.
- **DDP (DistributedDataParallel) / torchrun** — multi-GPU data-parallel training launcher.
- **Gradient accumulation** — summing gradients over several micro-batches before an optimizer step to emulate a larger batch.
- **Gradient checkpointing** — trading compute for memory by recomputing activations in the backward pass.
- **Weights & Biases (wandb)** — experiment-tracking dashboard receiving the training logs.

---

## 13. Questions You Should Be Able to Answer After Reading This

**Q1: What is HuBERT and why is it used here instead of training from scratch?**
A1: HuBERT is a self-supervised speech encoder (CNN feature extractor + 12-layer Transformer, 768-dim, ~94 M params) pretrained on 960 h of unlabelled LibriSpeech audio by predicting offline k-means cluster IDs of masked frames. It is used because it already encodes rich phonetic/acoustic structure in its frame embeddings; fine-tuning just adds a small linear head and supervised signal, which needs far less labelled data and compute than training an ASR model from random initialisation. The exact checkpoint is `facebook/hubert-base-ls960`.

**Q2: What does the CNN feature extractor do and what is its output dimension?**
A2: It is 7 stacked 1-D convolutions (kernels `[10,3,3,3,3,2,2]`, strides `[5,2,2,2,2,2,2]`, all 512 channels) that turn a raw 16 kHz waveform `(B, T_s)` into `(B, T, 512)`, where `T ≈ T_s/320` (≈50 frames/s, one frame per 20 ms). For a 4 s clip (64,000 samples) it outputs 199 frames. The 512-dim features are then projected to 768.

**Q3: What is the fine-tuning objective — is it CTC?**
A3: No. The model carries CTC-flavoured config and a CTC tokenizer, but the custom `FillerASRTrainer.compute_loss` overrides it to use **framewise cross-entropy** (`nn.CrossEntropyLoss(ignore_index=-100)`) between per-frame logits `(B,T,33)` and per-frame labels `(B,T)`. Every frame is a classification example; there is no blank token or alignment marginalisation.

**Q4: What is the `<fill>` token and why does it exist?**
A4: Cross-entropy requires a target for every frame, but a transcript is much shorter than the frame count `T`. So the transcript (wrapped in `<s>…</s>`) is placed left-aligned at the start and all remaining frames are labelled `<fill>`. `<fill>` (id 32) means "no character here". It is dropped at decode time via `skip_special_tokens=True`. Most frames end up being `<fill>`, causing class imbalance.

**Q5: What vocabulary does the model predict over, and how big is it?**
A5: 33 tokens: apostrophe `'`=0, letters `a`–`z`=1–26, word delimiter `|`=27, `<unk>`=28, `<pad>`=29, `<s>`=30, `</s>`=31, `<fill>`=32. Derived by `extract_vocab` from training transcripts plus appended special tokens. (The root `vocab.json` uses a different, stale ordering and is not what the trained model uses.)

**Q6: Trace the tensor shapes from waveform to output.**
A6: `(B,T_s)` waveform → CNN → `(B,T,512)` → feature projection → `(B,T,768)` → positional conv + 12 Transformer layers → `(B,T,768)` → final dropout → `lm_head` Linear(768→33) → logits `(B,T,33)`. Argmax over the last dim gives `(B,T)` token ids, decoded to a string.

**Q7: Where does the number 768 come from, and 33?**
A7: 768 is HuBERT-base's `hidden_size` (from the config); it is the Transformer/embedding width. 33 is `vocab_size`, set in `train.py` to `len(tokenizer)` (32 base tokens + the added `<fill>`), and written into the model config.

**Q8: What is frozen and what is trainable during fine-tuning?**
A8: The CNN feature extractor (~4.2 M params) is frozen the whole run (`freeze_feature_extractor=true`). Then a curriculum: for the first 10 % of steps only `lm_head` (25,377 params) trains; for the remaining 90 %, everything except the CNN trains (~90 M params). No LoRA/adapters — it's a full fine-tune of the encoder.

**Q9: What is the "triphase" LR schedule?**
A9: A custom `LambdaLR`: with ratios `[0.1, 0.4, 0.5]` over the total steps, LR ramps linearly 0→3e-4 (first 10 %), holds at 3e-4 (next 40 %), then decays linearly to 0 (final 50 %). It overrides the HF default `linear` scheduler whenever `lr_schedule_type=="triphase"`.

**Q10: How is variable-length audio batched?**
A10: `DataCollatorForFillerASR` right-pads waveforms with 0.0 (and builds an attention mask) and pads labels; padded label positions are set to `-100` so cross-entropy ignores them. A `min_seq_len` truncation in `compute_loss` reconciles any ±1 length mismatch between predicted and actual frame counts.

**Q11: What dataset and splits are used?**
A11: LibriSpeech. Training = train-clean-100 + train-clean-360 + train-other-500 (≈960 h, ~281,241 utterances). Evaluation during training = dev-clean (loaded as `dataset["test"]`). Final test = test-clean, only in `inference.py`. Audio is resampled to 16 kHz; text is lowercased and stripped of punctuation.

**Q12: How are predictions decoded and evaluated?**
A12: Greedy argmax per frame, then `tokenizer.batch_decode(skip_special_tokens=True, group_tokens=False)` — `group_tokens=False` prevents CTC-style merging of repeated letters. Metrics are WER and CER via `evaluate`/`jiwer`. References are lowercased and punctuation-stripped to match.

**Q13: What were the actual results?**
A13: Best run (200 epochs, 219,800 steps, ~5.8 days on 4×A6000): dev-clean `eval/wer ≈ 0.375`. On test-clean via `inference.py`: **WER 0.4880, CER 0.1449**. The in-training `eval/cer` (~8.9) is inflated by an eval-padding artifact and is not the real CER. These are far from CTC-fine-tuned HuBERT (~6 % WER), which is the point of the experiment.

**Q14: Why do the transcriptions have doubled/garbled letters?**
A14: Because the training targets are acoustically naïve — the transcript is left-packed one-token-per-frame with no real alignment — while at inference the model genuinely sees sounds spread across many frames. Without CTC's collapse and with `group_tokens=False`, frames where a phoneme is sustained emit repeated characters (e.g. `provision`→`rovvision`), and the fixed alignment causes drops, inflating WER/CER.

**Q15: How do you run training and inference, and what's the entry point?**
A15: Training entry point is `train.py`, launched via `run_training.sh` (which `conda activate asr_filler` then `torchrun --nproc_per_node=4 train.py --config_path config.json`). It reads `config.json`, builds the tokenizer/feature-extractor/dataset/model, and runs the custom `FillerASRTrainer`, auto-resuming from the newest checkpoint. Inference/eval entry point is `inference.py` (`python inference.py --config_path config.json` for test-clean, or `--audio_path file.wav` for a single clip).

---

*Report generated by automated codebase analysis. All dimensional claims are derived
directly from source code inspection (model `config.json` conv/hidden/vocab fields,
the `compute_output_length` convolution arithmetic in `dataset.py`, the `forward`/
`compute_loss` methods in `model.py`, and the collator/tokenizer code), and all training
numbers are cross-checked against the `wandb/` run logs.*

---
---

# Part II — Downstream Integration: Reusing the Filler Model as a Frozen Encoder for SLAM-ASR

*This part documents the planned downstream design (worked out interactively): the trained
`FillerHubertModel` (HuBERT + `lm_head`) is reused as a **frozen speech encoder** whose output is
fed through a small linear projector into a decoder-only LLM (Vicuna-7B), in the **SLAM-ASR** style
("An Embarrassingly Simple Approach for LLM with Strong ASR Capacity"). All shapes and numbers below
are consistent with Part I and with measurements on the local `segments` / `libri960_train_text` files.*

---

## 14. Goal & Mental Model

The pipeline is **`HuBERT → lm_head → LayerNorm → linear projector → Vicuna`**:

```
waveform ─► [FROZEN  HuBERT + lm_head] ─► per-frame features ─► [trainable projector] ─► LLM tokens ─► Vicuna ─► text
            └────────── encoder ──────────┘                     (norm + MLP)            (soft prompt)
```

- The encoder is **frozen**; only the **projector** (and optionally **LoRA** on Vicuna) is trained.
- Two possible feature taps from the encoder:
  - **`lm_head` logits — `(T, 33)`** (pre-softmax): compact, *already close to text* (soft character scores). Chosen here.
  - **hidden states — `(T, 768)`**: the richer, standard SLAM-ASR representation (fallback).
- The LLM is treated as a language model that reads the projected audio embeddings as **soft-prompt tokens** and autoregressively emits the transcript.

**Key conceptual point:** the filler model already produces a near-character-level, left-packed sequence,
so its `lm_head` output is an unusually "text-like" encoder output — which is the motivation for feeding it to an LLM.

---

## 15. The Checkpoint to Use

**Use `experiments/filler_asr_960h_triphase/checkpoint-164850`** — verified by reading
`model.safetensors`:

| Property | Value |
|---|---|
| `lm_head.weight` / `lm_head.bias` | **present** — `[33, 768]` / `[33]` (the trained head survives) |
| Tensor count | 213 (`hubert.*` + `lm_head.*` + `masked_spec_embed`) — full `FillerHubertModel` |
| `config.json` | `vocab_size=33`, `hidden=768`, `12 layers`, `architectures=['FillerHubertModel']`, bos/eos/pad = 30/31/29 |
| `apply_spec_augment` | **`true`** → must be disabled at load (see §16) |
| Tokenizer | self-contained in dir (`vocab.json`+`added_tokens.json` `<fill>:32`+`tokenizer_config.json`) |
| Provenance | final step of the 150-epoch *triphase* run (`global_step=164850`, dev WER ≈ **0.50**) |

> **Critical gotcha:** [train.py:123](train.py#L123) saved only `model.hubert.save_pretrained(.../model)`,
> so **`experiments/<exp>/model/` is HuBERT-only — the `lm_head` is NOT there.** Only the `Trainer`
> `checkpoint-XXXX/` directories contain the head. `checkpoint-164850` is the **only** local checkpoint
> with weights (the `triphase_lr_epochs` run scored better, ~0.375 WER, but its weights were never saved here).

Verification snippet (no torch needed):
```python
import json, struct
with open(f"{CKPT}/model.safetensors","rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n))
assert "lm_head.weight" in header, "this checkpoint has no head — use a checkpoint-XXXX dir"
```

---

## 16. Loading & Freezing the Encoder

Reuse the **`FillerHubertModel` class** and let `from_pretrained` match keys automatically:

```python
from transformers import HubertConfig, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from model import FillerHubertModel          # custom class — required (arch=FillerHubertModel)

cfg = HubertConfig.from_pretrained(CKPT); cfg.apply_spec_augment = False
encoder = FillerHubertModel.from_pretrained(CKPT, config=cfg).eval()
for p in encoder.parameters(): p.requires_grad = False
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(CKPT)        # rebuilds incl. <fill>=32
fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
                              padding_value=0.0, do_normalize=True, return_attention_mask=True)
```

**Why not `HubertModel.from_pretrained`:** a bare `HubertModel` (a) has no `lm_head` and (b) expects
un-prefixed keys (`encoder.*`) while the checkpoint stores `hubert.*` — so *every* key mismatches and
you silently get a randomly-initialized encoder.

### Considerations checklist

1. **`lm_head` presence** — verify first (§15).
2. **Freezing is two things:** `requires_grad=False` **and** keep the encoder in `eval()`. When the parent
   SLAM module calls `.train()`, PyTorch recursively re-enables **dropout + SpecAugment** inside the frozen
   encoder, corrupting "fixed" features. Defend with a `train()` override + `torch.no_grad()`:
   ```python
   def train(self, mode=True):
       super().train(mode); self.encoder.eval(); return self
   ```
   Also `cfg.apply_spec_augment=False` as a belt-and-suspenders.
3. **Tokenizer/vocab consistency** — load the same 33-token vocab; `<fill>` must resolve to **32**.
4. **dtype** — match Vicuna (fp16/bf16); keep `LayerNorm` in fp32; ensure projector output dtype equals the
   LLM `embed_tokens` dtype before concatenation.
5. **Device** — for a sharded Vicuna (`device_map="auto"`), move audio embeds to the device of
   `llm.get_input_embeddings().weight`.
6. **You only need 3 files** from the checkpoint for a frozen encoder: `model.safetensors`, `config.json`,
   tokenizer files. Ignore `optimizer.pt` (722 MB), `scheduler.pt`, `scaler.pt`, `rng_state_*` (resume-only).

**Optional reorganization (`lm_head` inside the projector):** obtain `encoder.lm_head` from the same
`from_pretrained` and use it as the frozen first layer of the projector — functionally identical, with
**zero manual weight copying**, and leaves the door open to later unfreeze the head.

---

## 17. Choosing Which Frames to Keep (the fill/content problem)

Because the transcript is left-packed and the rest is `<fill>`, most frames are content-free. Measured on
the local data (2,620 utterances; speaking rate generalizes to 960h):

| Metric | mean | median | p95 | p99 | max |
|---|---|---|---|---|---|
| chars/second (speaking rate) | **14.3** | 14.5 | 18.5 | 19.9 | 22.0 |
| frames per character (`T`/tokens) | 3.55 | 3.36 | 5.1 | 6.7 | 13.3 |
| content fraction (`labels`/`T`) | **0.29** | 0.30 | 0.38 | 0.40 | 0.46 |
| content frames (`chars`+2) | 109 | 87 | 263 | 366 | 578 |

So **~71% of frames are `<fill>`**, and each character spans **~3.5 acoustic frames**. Three selection strategies:

- **Option A — global fixed `N`.** Set `N = p99/max` (≈366–578). Simple but wasteful for short clips.
- **Option B — per-utterance duration heuristic.** `N_i = ceil(R · duration_i) + margin`, with **`R ≈ 20`**
  (p99; *not* 15 = mean, which truncates ~half the utterances). You know duration at inference → deployable.
- **Option C — dynamic `<fill>` boundary (recommended).** Run the encoder, `argmax` the logits, keep frames
  up to the last non-`<fill>`/non-special frame. Uses the model's own learned boundary; argmax is used only
  to *locate* the cut, **not** as the downstream feature.

> **Current interim decision:** *pass ALL frames* (no trimming) — see §20. Selection (Option C, capped by B)
> is deferred to a later stage.

---

## 18. Normalization of the 33-dim logits (no softmax)

Standard HuBERT-for-CTC applies **no norm** between encoder and `lm_head` (the 768-d hidden is already
LayerNorm-conditioned by the encoder). But the **`lm_head` logits are raw, un-normalized** — so when you
feed *them*, normalize first:

| Method | Effect | Params |
|---|---|---|
| **`LayerNorm(33)`** (recommended) | per-frame zero-mean/unit-var + learnable γ,β | yes |
| `RMSNorm(33)` | per-frame RMS scale (matches LLaMA/Vicuna internals) | small |
| L2 normalize (`F.normalize`) | unit-length per frame | none |
| z-score | LayerNorm without affine | none |

(`log_softmax` is technically softmax-family — avoid if staying strictly pre-softmax.) Optionally drop the
5 special-token columns first (`33 → 28`) for a pure-character signal.

---

## 19. Projector & LLM Integration

**Projector (pointwise, per frame; only fixes the feature dim, never the length):**
- feeding **33-d logits** → projector is **`33 → 2048 → 4096`**.
- feeding **768-d hidden** → projector is **`768 → 2048 → 4096`**.
- `4096` = Vicuna-7B hidden size; `2048` = your chosen bottleneck.

**Variable length — why no fixed 30 s padding is needed:**
- The projector is per-frame; the LLM (decoder-only, RoPE positions) consumes any `seq_len` natively.
- The **30 s requirement is a Whisper-encoder property** (fixed 1500-frame absolute positions). **HuBERT is
  fully length-flexible** — you already rely on this in Part I (variable-length training + dynamic padding).
- Padding appears **only for batching**: pad audio embeds to the **batch max** `N`, add an attention mask;
  at inference (batch=1) there is no padding. 1 s and 30 s both flow through unchanged.

**LLM assembly & loss:**
```
inputs_embeds = [ prompt_embeds ][ audio_embeds (N, 4096) ][ target_text_embeds ]
labels        =   -100 …            -100 … (all audio)        real target ids
loss          = next-token CE on the TARGET text tokens only
```
Loss masking is essential — the audio/prompt positions are `-100`, so even `<fill>` audio frames never
become loss targets; they only enter the LLM's attention context.

**What trains / what you save:** projector (+ optional LoRA on Vicuna). Encoder and LLM-base frozen.
**Save only the projector (+ LoRA)** — tiny checkpoints, no drift of the frozen parts.

**Optional big speedup:** since the encoder is frozen and deterministic, **precompute and cache the
`lm_head` logits** for the dataset once (in `eval()`/no-SpecAugment mode), then train the projector+LLM on
cached features — removes HuBERT from the training loop.

---

## 20. Interim Stage: "Pass All Frames"

For now, the full `(B, T, 33)` is passed with **no trimming** (selection comes later). Effects:

- **Forward pass (unchanged):** `hubert → (B,T,768) → lm_head → (B,T,33)`, `T = compute_output_length(T_s) ≈ T_s/320`,
  padded per batch to `T_max`. From the data: mean `T ≈ 371`, **max ≈ 1747** (a 35 s clip); ~71% of frames are `<fill>`.
- **Dimensions:** `(B,T_max,33) → LayerNorm(33) → projector → (B,T_max,4096)` → concat → `(B, P+T_max+G, 4096)`.
- **Loss formula unchanged:** LLM next-token CE on the `G` target tokens only; fill frames are context, never targets.
- **Cost you accept now:** longer LLM sequences (O(L²) attention; a 35 s clip ≈ 1747 audio tokens can pressure
  Vicuna's 4096 context), two masks (batch-pad vs real frames; fill frames are *real* and attended), and higher
  activation memory (mitigate with gradient checkpointing + fp16/bf16). The pipeline is **correct** as-is —
  pass-all is purely an efficiency/context trade-off that Option C will later reduce by shrinking `T_max`.

---

## 21. Environment Additions for the Downstream Phase

Part I's `requirements.txt` (CUDA 12.6 stack: `torch 2.7.0+cu126`, `transformers 5.8.0`, `accelerate 1.13.0`,
DDP via NCCL `cu12`, fp16 + gradient checkpointing) fully covers the **encoder side**. For
`…→ projector → Vicuna` you additionally need:

- **`peft`** — LoRA adapters on Vicuna.
- **`sentencepiece`** — LLaMA/Vicuna tokenizer (the slow tokenizer needs it; `protobuf`/`tokenizers`/`Jinja2`
  are already present — `Jinja2` handles the Vicuna chat template).
- **`bitsandbytes`** (optional) — 4/8-bit Vicuna to fit alongside the encoder on one GPU.

Keep `numpy==1.26.4` (pinned `<2` for this torch ABI). The inert `nvidia-*-cu11` wheels in the env are dead
cruft (torch is `cu126`); a clean downstream env can drop them.

---

## 22. Reference Skeleton

```python
class HubertLMHVicuna(nn.Module):
    def __init__(self, ckpt, llm, rate=20.0, margin=5, pass_all=True):
        super().__init__()
        cfg = HubertConfig.from_pretrained(ckpt); cfg.apply_spec_augment = False
        self.encoder = FillerHubertModel.from_pretrained(ckpt, config=cfg).eval()
        for p in self.encoder.parameters(): p.requires_grad = False
        self.norm = nn.LayerNorm(33)
        self.proj = nn.Sequential(nn.Linear(33, 2048), nn.GELU(), nn.Linear(2048, 4096))
        self.llm  = llm                                   # frozen base (+ optional LoRA)
        self.rate, self.margin, self.pass_all = rate, margin, pass_all

    def train(self, mode=True):
        super().train(mode); self.encoder.eval(); return self    # keep encoder eval

    @torch.no_grad()
    def _logits(self, x, attn):
        return self.encoder(x, attention_mask=attn).logits        # (B,T,33)

    def encode_audio(self, input_values, attn):
        logits = self._logits(input_values, attn)                 # (B,T,33)
        T = logits.size(1)
        if self.pass_all:                                         # §20 interim
            Ns = attn.new_full((logits.size(0),), T)              # keep all frames
        else:                                                     # Option B cap (+ C later)
            durs = attn.sum(-1).float() / 16000.0
            Ns = torch.clamp((self.rate * durs).ceil().long() + self.margin, max=T)
        Nmax  = int(Ns.max())
        feats = logits[:, :Nmax, :]                               # (B,Nmax,33)
        amask = (torch.arange(Nmax, device=Ns.device)[None, :] < Ns[:, None])
        audio = self.proj(self.norm(feats))                       # (B,Nmax,4096)
        return audio, amask

    def forward(self, input_values, attn, prompt_ids, target_ids):
        audio, amask = self.encode_audio(input_values, attn)
        emb = self.llm.get_input_embeddings()
        # concat [prompt][audio][target] embeds; labels=-100 on prompt+audio, ids on target;
        # build attention_mask; return self.llm(inputs_embeds=..., attention_mask=..., labels=...).loss
        ...
```

---

*Part II documents the intended downstream architecture and the load/freeze/selection/normalization
decisions reached during analysis. It reuses the verified `checkpoint-164850` and is consistent with the
training facts established in Part I; the speaking-rate statistics were measured from the local
`segments` and `libri960_train_text` files.*
