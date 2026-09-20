# `train_sa_pe_lm_head` — Architecture & Data-Flow Report

A line-by-line account of the model and the data pipeline used by this directory:
what each tensor's shape is, how audio becomes per-frame logits, how the positional
`<fill>` target is built, how padding and masking work, how batches are formed, and
how the loss is computed.

The training scripts (`train.py`, `data.py`) live here; the **model, collator,
label/mask logic and loss** are imported unchanged from the verified reference at
`../filler_sa_reference.py`. File references below use that path or the local file.

**Concrete running example.** Throughout, we trace a single utterance of **4.00 s
of 16 kHz mono audio = 64,000 samples**, batched with `B` other clips. Key model
dimensions (HuBERT-xlarge, from `models/hubert-xlarge-ls960-ft/config.json`):

| symbol | value | meaning |
|---|---|---|
| `S` | 64,000 | audio samples (this clip) |
| `T` | **199** | encoder frames for this clip (≈50 fps) |
| `d` | **1280** | `hidden_size` (model width) |
| heads | **16** | attention heads (`d_k = 1280/16 = 80`) |
| ffn | **5120** | SA feed-forward inner dim (`intermediate_size`) |
| layers (frozen) | 48 | HuBERT transformer layers |
| layers (trained) | 2 | new self-attention layers |
| `V` | **33** | output vocab (32 chars/specials + `<fill>`) |

---

## 1. Architecture at a glance

```
 audio (S samples)
   │  Wav2Vec2FeatureExtractor  (zero-mean/unit-var normalize)
   ▼
 input_values (B, S)  + attention_mask (B, S)            ── sample level
   │  [FROZEN] HuBERT-xlarge: 7-layer conv (÷320) + 48 transformer layers
   ▼
 last_hidden_state  (B, T, 1280)                          ── the "tap"
   │  ⊕  sinusoidal positional encoding  (T, 1280)        ── fixed, no params
   ▼
 (B, T, 1280)
   │  src_key_padding_mask (B, T)  derived from attention_mask
   │  [TRAINABLE] 2 × post-norm TransformerEncoderLayer (MHSA+FFN)
   ▼
 (B, T, 1280)
   │  Dropout(final_dropout=0.0 → no-op)
   │  [TRAINABLE] lm_head: Linear(1280 → 33)
   ▼
 logits (B, T, 33)
   │  framewise CrossEntropy vs positional labels (B, T), ignore_index=-100
   ▼
 scalar loss
```

Only the **2 SA layers + lm_head** carry gradients (~**39.4 M** params). The whole
HuBERT body — conv feature extractor *and* all 48 transformer layers — is frozen,
so the tap `last_hidden_state` comes out with `requires_grad=False`; its activations
are never stored for backprop (memory ≈ encoder forward + a small head graph). There
is **no caching**: this forward runs on every training and eval step.

---

## 2. Frame geometry — how `T` is computed

`filler_sa_reference.py:56`
```python
def compute_output_length(input_length):
    length = input_length
    for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        length = (length - kernel) // stride + 1
    return length
```
This replays HuBERT's 7 conv layers (kernels `[10,3,3,3,3,2,2]`, strides
`[5,2,2,2,2,2,2]`) on a length, giving the encoder frame count. Net downsample is
`5·2·2·2·2·2·2 = 320×`, i.e. one frame per ~20 ms → ~50 fps. For our example:

```
64000 →(k10,s5) 12799 →(k3,s2) 6399 →(k3,s2) 3199 →(k3,s2) 1599
      →(k3,s2)   799 →(k2,s2)  399 →(k2,s2)  199   ⇒  T = 199
```
The collator uses this to size each utterance's per-frame label vector **before**
the audio ever reaches the GPU, so labels and encoder output line up frame-for-frame.

---

## 3. Data ingestion — manifest → rows → dataset

`data.py:25`
```python
def load_rows(manifest_path, limit=None):
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({
                "audio_path": d.get("source") or d.get("audio"),
                "text": d.get("target") or d.get("text") or "",
            })
            if limit and len(rows) >= limit:
                break
    return rows
```
Each manifest line is JSON like
`{"key": "...", "source": "/.../103-1240-0000.wav", "target": "CHAPTER ONE ..."}`.
We keep **only** `{audio_path, text}` — no waveforms in RAM. Text is left **raw**
(uppercase, punctuated); normalization happens in exactly one place (the collator,
§4.2). With `--train_n 0` the full 281,241-utterance manifest is loaded.

`data.py:53`
```python
class ManifestDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self):        return len(self.rows)
    def __getitem__(self, i): return self.rows[i]
```
A trivial map-style `Dataset`. `__getitem__` returns the dict row; the actual audio
decode is deferred to the collator, so DataLoader workers parallelize disk I/O.

---

## 4. The collator — the heart of the pipeline

`DataCollatorFillerASR` (`filler_sa_reference.py:229`) turns a list of row dicts into
a padded GPU-ready batch. It does four jobs: (a) load + feature-extract audio,
(b) build the positional `<fill>` target, (c) compute the duration budget,
(d) pad and produce the two masks.

### 4.1 Per-utterance loop: audio → features + label

`filler_sa_reference.py:264`
```python
def __call__(self, features):
    input_features, label_features, keep = [], [], []
    for feat in features:
        arr, sr = sf.read(feat["audio_path"], dtype="float32")   # decode wav → (S,)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)                                # stereo → mono
        assert sr == SAMPLING_RATE, f"expected 16kHz, got {sr}"
        iv = self.fe(arr, sampling_rate=SAMPLING_RATE).input_values[0]   # normalize → (S,)
        input_features.append({"input_values": iv})
        T = compute_output_length(len(iv))                        # frames for THIS clip (199)
        label_features.append({"input_ids": self._labels_for(feat["text"], T)})
        keep.append(math.ceil(len(iv) / SAMPLING_RATE * FRAMES_PER_SEC))  # graded-frame budget
```
- `sf.read` decodes the wav to a float32 array `arr` of shape `(S,)` (here `S=64000`).
- `self.fe(...)` is the `Wav2Vec2FeatureExtractor` with `do_normalize=True`: it
  subtracts the mean and divides by the std of `arr` (per-utterance), returning
  `input_values` of the same shape `(S,)`. `feature_size=1` means the raw waveform
  *is* the feature — HuBERT's conv stack is the real feature extractor.
- `T = compute_output_length(len(iv))` → 199 for this clip.
- `keep` stores the **duration budget** `n_keep = ceil(S/16000 · 20) = ceil(4·20) = 80`
  frames (see §4.3).

### 4.2 The positional `<fill>` label

`filler_sa_reference.py:248`
```python
def _labels_for(self, text, T):
    text = self._normalize(text).replace(" ", "|")   # lowercase, strip punct, space→'|'
    ids = self.tok(text).input_ids                   # raw char ids (no bos/eos)
    if len(ids) + 2 > T:                              # truncate if transcript > frames
        ids = ids[: T - 2]
    labels = [self.bos_id] + ids + [self.eos_id]      # [<s>] + chars + [</s>]
    labels = labels + [self.fill_id] * (T - len(labels))  # pad tail with <fill>
    return labels
```
`_normalize` (`filler_sa_reference.py:243`) lowercases and strips `,?.!-;:"`; spaces
become the word-delimiter token `|`. The transcript is then **laid at the head of
the frame axis**, and the remaining frames are filled with `<fill>` (id 32):

```
frame:  0     1   2   3   4  5  6  7  ...        51  52    53 ...        198
label: <s>    m   i   s   t  e  r  |  ...        </s> <fill> <fill> ... <fill>
       └──── transcript chars (one id per frame) ────┘ └──── <fill> tail ────┘
```
So `labels` is a length-`T` (=199) integer vector. This pins char *i* to frame *i*:
a **positional**, not acoustically-aligned, target. (It is the implemented design;
this report documents it, it does not change it.) Vocab/ids come from the 33-token
tokenizer built in `filler_sa_reference.py:214`, where `<fill>` is added as id 32 and
`assert len(tok) == 33`.

### 4.3 Duration budget — which frames are graded

`FRAMES_PER_SEC = 20` (`filler_sa_reference.py:51`). The encoder runs at ~50 fps, but
only the first `n_keep = ceil(dur_s · 20)` frames are **graded** by the loss; the rest
of the `<fill>` tail is ignored. For 4 s, `n_keep = 80`. This caps how much of the
long `<fill>` tail contributes to the loss, so the loss is dominated by the transcript
region plus a short fill margin rather than by hundreds of trivial tail frames.

### 4.4 Padding and the two masks

`filler_sa_reference.py:283`
```python
    batch = self.fe.pad(input_features, padding=True, return_tensors="pt")
    labels_batch = self.tok.pad(label_features, padding=True, return_tensors="pt")
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    for i, n_keep in enumerate(keep):
        labels[i, n_keep:] = -100
    batch["labels"] = labels
    return batch
```
Two independent padding operations, on two different axes:

1. **Audio padding** — `self.fe.pad(..., padding=True)` right-pads every `input_values`
   in the batch to the longest audio length `S_max` (samples). It returns:
   - `batch["input_values"]` : `(B, S_max)` float32
   - `batch["attention_mask"]` : `(B, S_max)` int — **1 = real sample, 0 = pad**.
   This **sample-level** mask is the single source of truth for padding; the model
   later converts it to a frame-level mask (§5.3).

2. **Label padding** — `self.tok.pad(..., padding=True)` right-pads every label list to
   the longest label length (= the largest `T` in the batch, `T_max` frames). Then
   **two maskings turn padded/ungraded positions into `-100`** (PyTorch CE's
   `ignore_index`):
   - `masked_fill(attention_mask.ne(1), -100)` — label positions that are **pad frames**
     (beyond a clip's own `T`) → `-100`.
   - `labels[i, n_keep:] = -100` — everything **past the duration budget** → `-100`.

   Net: `batch["labels"]` is `(B, T_max)` long, where each entry is either a real
   class id (`0..32`) in the graded region `[0, n_keep)`, or `-100` everywhere else.

So one batch dict is: `input_values (B,S_max)`, `attention_mask (B,S_max)`,
`labels (B,T_max)`. Note `S_max` and `T_max` are *batch-dynamic* — padding is to the
longest member of each batch, not a global max.

---

## 5. The model forward

`FillerHubertSAModel.forward` (`filler_sa_reference.py:172`). Input: `input_values`
`(B,S_max)` and `attention_mask` `(B,S_max)`; output: `logits (B,T_max,33)`.

### 5.1 Frozen HuBERT tap

`filler_sa_reference.py:177`
```python
        outputs = self.hubert(
            input_values, attention_mask=attention_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # the embedding tap   → (B, T, 1280)
```
`self.hubert` (a full `HubertModel`, built in `__init__` at
`filler_sa_reference.py:101`) runs the 7 conv layers (÷320) then 48 transformer
layers. We take `last_hidden_state` (`outputs[0]`), shape **`(B, T, 1280)`** — for the
example, `(B, 199, 1280)`. All HuBERT params have `requires_grad=False` (frozen via
`unfreeze_all_except_feature_extractor`, §10), and the input audio needs no grad, so
`hidden_states` is a `requires_grad=False` leaf into the trainable head.

### 5.2 Sinusoidal positional encoding (the ⊕)

`filler_sa_reference.py:185`
```python
        if self.use_sinusoidal_pe:
            B, T, dim = hidden_states.shape
            pe = sinusoidal_positional_encoding(T, dim, hidden_states.device, hidden_states.dtype)
            hidden_states = hidden_states + self.pe_scale * pe.unsqueeze(0)
```
`sinusoidal_positional_encoding` (`filler_sa_reference.py:71`):
```python
    pos = torch.arange(seq_len).unsqueeze(1)                       # (T, 1)
    div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))  # (dim/2,)
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)                            # even dims = sin
    pe[:, 1::2] = torch.cos(pos * div)                            # odd dims  = cos
    return pe.to(dtype)
```
This is the fixed Vaswani-2017 PE, shape `(T, 1280)`, values in `[-1,1]`, **no
parameters**. `pe.unsqueeze(0)` makes it `(1, T, 1280)` and it broadcasts-adds to
`(B, T, 1280)` (scaled by `pe_scale=1.0`). Why it's needed: self-attention is
permutation-equivariant, so without an absolute position signal the SA layers
couldn't realize the *positional* target (char *i* at frame *i*). PE is the only
thing giving the head a notion of "which frame is this."

### 5.3 Frame-level key-padding mask

`filler_sa_reference.py:192`
```python
        src_key_padding_mask = None
        if attention_mask is not None:
            feat_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask)        # (B, T): 1=real frame
            src_key_padding_mask = ~feat_mask.bool()           # (B, T): True=PAD frame
```
The `attention_mask` is **sample-level** `(B, S_max)`. `_get_feature_vector_attention_mask`
runs the same conv-downsample arithmetic as §2 to turn it into a **frame-level** mask
`feat_mask (B, T)` (1 = real frame). We invert it to `src_key_padding_mask (B,T)` where
**`True` marks a pad frame to be ignored** — the convention `nn.TransformerEncoder`
expects. This is how padded frames are excluded from attention (§5.4), keeping pad
positions from polluting real frames' representations.

### 5.4 Self-attention stack (the trainable core)

Built in `__init__` (`filler_sa_reference.py:116`):
```python
        sa_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,    # 1280
            nhead=nhead,                   # 16  → d_k = 80
            dim_feedforward=dim_ff,        # 5120
            dropout=sa_dropout,            # 0.1
            activation="gelu",
            batch_first=True,              # tensors are (B, T, d)
            norm_first=False,              # POST-norm: x = LayerNorm(x + Sublayer(x))
        )
        self.sa = nn.TransformerEncoder(sa_layer, num_layers=n_layers, enable_nested_tensor=False)
```
Applied in `forward` (`filler_sa_reference.py:199`):
```python
        hidden_states = self.sa(hidden_states, src_key_padding_mask=src_key_padding_mask)
```
Each of the `n_layers=2` layers does, on `(B,T,1280)`:
1. **Multi-head self-attention** — Q,K,V are linear projections `1280→1280`, split into
   16 heads of width 80. Scores `(B,16,T,T)`; `src_key_padding_mask` sets pad-key
   columns to `-inf` before softmax so they get zero weight. Output recombined to
   `(B,T,1280)`.
2. **Add & LayerNorm** — `x = LayerNorm(x + MHSA(x))` (post-norm).
3. **FFN** — `1280 → 5120 → GELU → 1280`.
4. **Add & LayerNorm** — `x = LayerNorm(x + FFN(x))`.

Shape is preserved end to end: in `(B,T,1280)`, out `(B,T,1280)`. These ~39 M params
(2 layers) plus the head are the only trainable weights. (Their `nn.MultiheadAttention`
in-proj weights are explicitly xavier-initialized in `_init_weights`,
`filler_sa_reference.py:135`, because HuBERT's initializer skips them.)

### 5.5 Dropout + classification head

`filler_sa_reference.py:202`
```python
        hidden_states = self.dropout(hidden_states)   # Dropout(final_dropout=0.0) → no-op here
        logits = self.lm_head(hidden_states)          # Linear(1280 → 33)
```
`self.dropout = nn.Dropout(config.final_dropout)` and this checkpoint has
`final_dropout=0.0`, so this dropout is currently a **no-op** (regularization lives
inside the SA layers via `sa_dropout=0.1`). `self.lm_head` is `Linear(1280, 33)` —
loaded with `ignore_mismatched_sizes=True` and **reinitialized** because the pretrained
CTC head was `32`-wide (this is the `lm_head 32→33 MISMATCH` line in the load report).
Output: **`logits (B, T, 33)`** — one distribution over the 33 classes per frame.

The forward returns `CausalLMOutput(logits=logits, ...)` (`filler_sa_reference.py:207`);
callers read `.logits`.

---

## 6. The loss — framewise cross-entropy

`filler_sa_reference.py:300`
```python
def framewise_ce_loss(logits, labels, fill_id=32, fill_weight=None):
    V = logits.size(-1)                                  # 33
    weight = None
    if fill_weight is not None:                          # default None → unweighted
        weight = torch.ones(V, device=logits.device)
        weight[fill_id] = fill_weight
    loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
    return loss_fct(logits.reshape(-1, V), labels.reshape(-1))
```
- `logits.reshape(-1, V)` flattens `(B,T,33) → (B·T, 33)`.
- `labels.reshape(-1)` flattens `(B,T) → (B·T,)`.
- `ignore_index=-100` means every position we marked `-100` in §4.4 (pad frames and
  beyond-budget frames) contributes **nothing** to the loss and is excluded from the
  mean. The loss is the average negative log-likelihood over the **graded** frames only.
- `fill_weight` is `None` by default (`--fill_weight` unset) → the dominant `<fill>`
  class is weighted equally; the logs show no `<fill>` collapse, so this is left off.

A random-init head would score ≈ `ln(33) ≈ 3.50`; the run reaches well below that.

---

## 7. Training loop — autocast, grad-accumulation, clipping, schedule

Setup (`train.py:157`–`169`):
```python
    model = build_model(args.model_checkpoint, tok, use_sinusoidal_pe=args.pe, device=dev)
    trainable = [p for p in model.parameters() if p.requires_grad]            # SA + lm_head
    train_loader = DataLoader(ManifestDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, total_steps)
```
- `shuffle=True` reshuffles utterance order each epoch; `collate_fn=collator` runs §4;
  `num_workers=8` decodes audio in parallel; `drop_last=True` keeps batch shapes regular.
- AdamW (`betas=(0.9,0.98)`, `eps=1e-6`) over **only** the trainable head params; cosine
  schedule with `--warmup` linear-warmup steps (train.py default 500; `run.sh` passes 2000)
  over `total_steps`. **`total_steps` is derived from `--epochs`**, so the LR decay is
  correctly sized to whatever epoch count you pass.

Inner loop (`train.py:224`):
```python
        for batch in train_loader:
            labels = batch.pop("labels").to(dev)                      # (B, T_max)
            inp = {k: v.to(dev) for k, v in batch.items()}            # input_values, attention_mask
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lg = model(**inp).logits.float()                      # (B, T_max, 33), upcast to fp32
            m = min(lg.shape[1], labels.shape[1])                     # align frame axes
            loss = framewise_ce_loss(lg[:, :m], labels[:, :m], fill_id=fill_id, fill_weight=args.fill_weight)
            (loss / args.grad_accum).backward()                       # scale for accumulation
            accum_loss += loss.item(); seen += labels.shape[0]; micro += 1
            if micro % args.grad_accum != 0:
                continue                                              # keep accumulating grads

            gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)   # clip to 1.0 (returns PRE-clip norm)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
```
Key points:
- **bf16 autocast** for the forward (matches the cached experiments); `.float()` upcasts
  logits so the CE is computed in fp32 for stability.
- `m = min(...)` guards the rare off-by-one between logit frames and label frames; both
  are sliced to `m` before the loss.
- **Gradient accumulation**: `loss` is divided by `grad_accum` and `.backward()` is called
  every micro-batch, but `opt.step()` only fires every `grad_accum` micro-batches — so the
  **effective batch = `batch_size · grad_accum`** (default `16·2 = 32`). `step` counts
  optimizer steps, not micro-batches.
- **Clipping**: `clip_grad_norm_(trainable, 1.0)` rescales gradients so their global norm
  ≤ 1.0. Its return value `gnorm` is the **pre-clip** norm (what the logs print). This is
  what keeps training stable.
- `sched.step()` advances the cosine LR each optimizer step.

Logging / eval / checkpoint cadence (`train.py:239`–`253`): every `--log_steps` it prints
and logs `train/{loss,graded_frame_acc,grad_norm}`, `learning_rate`, throughput; every
`--eval_steps` it calls `log_eval`; every `--save_steps` it writes `latest.pt`.

---

## 8. Evaluation & metrics

`evaluate` (`train.py:75`) mirrors the training forward (no grad, `model.eval()`), then
computes, per dev batch:
```python
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model(**inp).logits.float()                 # (B, T, 33)
        m = min(lg.shape[1], labels.shape[1]); lg = lg[:, :m]; lb = labels[:, :m]
        loss_s += framewise_ce_loss(lg, lb, fill_id=fill_id).item(); loss_n += 1
        pred = lg.argmax(-1); graded = lb != -100; corr = (pred == lb) & graded
```
- **Frame metrics** over the graded region (`lb != -100`): `graded_frame_acc`,
  `content_frame_acc` (graded frames whose label ≠ `<fill>`), `fill_frame_acc`,
  `fill_pred_rate`, predictive `entropy`, and `distinct_nonfill_tokens` (how many of the
  33 classes the model actually emits — a collapse detector).
- **String metrics** — for each utterance the graded frames are argmax-decoded
  (`tok.batch_decode(pred, skip_special_tokens=True, group_tokens=False)`), the reference
  is normalized the same way, and Levenshtein (`lev`, `train.py:34`) gives **WER**
  (word edits / ref words), **CER** (char edits / ref chars), and **SER** (sentence error).
- **Samples table** (`train.py:203`): the wandb `Table` logs `full_frames | hyp | ref` for
  a few clips — the qualitative view.

`log_eval` (`train.py:189`) prints the scalar line, logs scalars + the table to wandb, and
saves `best.pt` whenever `eval/WER` improves (`train.py:210`). Model selection is therefore
by **WER**, independent of the (possibly rising) eval loss.

Because decoding uses `group_tokens=False`, a phoneme spanning several frames yields
**repeated characters** in the hypothesis (e.g. `quilter → qiilter`); this is a direct,
expected consequence of the framewise positional target and per-frame argmax, and it is
reflected in the reported WER/CER.

---

## 9. End-to-end dimension walkthrough (one batch)

Tracing shapes for a batch of `B` clips whose longest audio is `S_max` samples and whose
largest frame count is `T_max` (≤ `compute_output_length(S_max)`):

| stage | tensor | shape | dtype |
|---|---|---|---|
| collate: audio | `input_values` | `(B, S_max)` | f32 |
| collate: audio mask | `attention_mask` | `(B, S_max)` | int (1=real) |
| collate: labels | `labels` | `(B, T_max)` | long (−100=ignore) |
| HuBERT tap | `last_hidden_state` | `(B, T, 1280)` | bf16 (autocast) |
| ⊕ PE | `hidden_states` | `(B, T, 1280)` | bf16 |
| frame mask | `src_key_padding_mask` | `(B, T)` | bool (True=pad) |
| 2× SA | `hidden_states` | `(B, T, 1280)` | bf16 |
| dropout (no-op) | `hidden_states` | `(B, T, 1280)` | bf16 |
| lm_head | `logits` | `(B, T, 33)` | bf16 → `.float()` f32 |
| loss flatten | `logits / labels` | `(B·T, 33) / (B·T,)` | f32 / long |
| loss | scalar | `()` | f32 |

(`T` from the encoder and `T_max` from label padding are aligned by the `m = min(...)`
slice in §7/§8.)

---

## 10. What trains, what's frozen, and the checkpoint

`build_model` (`filler_sa_reference.py:320`) sets the vocab to 33, enables PE, loads the
pretrained HuBERT with `ignore_mismatched_sizes=True` (reinit the 32→33 lm_head), then:
```python
    model.freeze_feature_extractor()                  # conv extractor frozen
    model.unfreeze_all_except_feature_extractor()     # everything frozen EXCEPT sa.* and lm_head.*
```
`unfreeze_all_except_feature_extractor` (`filler_sa_reference.py:161`) sets
`requires_grad=False` on all params and the whole `self.hubert`, then `True` only on
`self.sa` and `self.lm_head` → **~39.4 M trainable**.

Checkpoints are therefore **head-only**. `head_state_dict` (`train.py:48`) keeps just keys
starting `sa.` or `lm_head.`; `save_ckpt` (`train.py:53`) bundles that with optimizer +
scheduler + step/epoch/best-WER (~150 MB, not the 1.2 GB full model). `load_ckpt`
(`train.py:58`) restores into a freshly `build_model`'d instance with
`load_state_dict(..., strict=False)` (the frozen HuBERT keys are simply already present).
The tokenizer is saved into `--out_dir` once (`train.py:144`) for inference.

---

### Summary of the trainable signal path
`audio → [frozen conv ÷320] → [frozen 48× transformer] → tap (B,T,1280) → ⊕PE →
[trained 2× SA] → [Linear 1280→33] → (B,T,33) → framewise CE on positional labels`,
with the **sample-level attention mask** driving both the **frame key-padding mask**
(inside attention) and the **label −100 mask** (in the loss), and the **duration budget**
limiting graded frames to `ceil(dur·20)`.
