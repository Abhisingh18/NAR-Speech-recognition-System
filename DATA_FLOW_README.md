# indic-nar-filler_asr — Data & Flow Reference

End-to-end trace of the **Hindi A‑CMLM** pipeline: from `run.sh` → `train.py` → the data
modules → `filler_sa_reference.py`, with special focus on **how a Hindi target string becomes
the per‑frame label vector** `[<s>] + chars + [</s>]×3 + <fill>×tail` (with `|` word
delimiters). Every computation step is shown with its file:line and with tensor
shapes / dims / FLOPs.

> TL;DR: this project is a **thin Hindi + data2vec skin over the English `filler_asr` code**.
> The label construction, masking, sampler, loss and SA head are all **imported unchanged** from
> `/speech/tomson/filler_asr`. Only three things differ: the **encoder** (data2vec‑aqc, 1024‑d,
> instead of HuBERT‑xlarge 1280‑d), the **text normalizer** (`normalize_hi`), and the **vocab**
> (corpus‑built Devanagari `vocab_hi/`).

---

## 0. File map (who owns what)

| Concern | File | Notes |
|---|---|---|
| Launcher / all knobs | `run.sh` | builds `torchrun/python train.py …` |
| Trainer, DDP, loss loop | `train.py` | this repo's own copy |
| **Hindi text norm + vocab** | `text_norm_hi.py` | `normalize_hi`, `build_vocab`, `build_acmlm_tokenizer_hi` |
| Encoder swap + collator factory | `model_d2v_acmlm.py` | data2vec‑aqc adapter, `build_collator` |
| Cached‑tap dataset/collator | `data_tapcached_d2v.py` | subclasses English classes, dim 1024 + Hindi norm |
| Tap cache builder | `precompute_taps_d2v.py` | one‑time frozen‑encoder dump |
| Manifest reader | `/speech/tomson/filler_asr/train_sa_pe_lm_head/data.py` | `load_rows`, `ManifestDataset` |
| **Label construction + masking + SA head + loss** | `/speech/tomson/filler_asr/filler_sa_reference.py` | reused byte‑for‑byte |
| Memmap dataset + sampler + tap collator | `/speech/tomson/filler_asr/train_sa_pe_lm_head/data_tapcached.py` | reused |

**Concrete data for the default run** (`run.sh`):
- encoder `SPRING_INX_data2vec_aqc_SSL.pt` (frozen, 1024‑d, ~50 fps)
- train `train_hi.jsonl` = **43,337 clips**, dev `val_hi.jsonl` = **355 clips**
- vocab `vocab_hi/` = **63 Devanagari chars** + `| <unk> <pad> <s> </s>` = 68 entries; tokenizer then
  adds `<fill>`(68) `<mask>`(69) → **len 70**
- recipe: `pos_mode=alibi`, `fps=50`, `eos_repeat=3`, `fill_weight=0.03`, 8 SA layers,
  `mask20‑30×10`, batch 32 × accum 2, `iter_wer` selection. **LIVE encoder by default** (no tap cache).

---

## 1. Flow: `run.sh` → `train.py`

`run.sh` sets `FILLER_ROOT`, `SLAM_SRC`, `PYTHONPATH` so the shared English modules + the
data2vec loader resolve ([run.sh:20‑23](run.sh)), then assembles `ARGS` and launches
`train.py` under `torchrun` (multi‑GPU) or `python` (1 GPU) ([run.sh:68‑92](run.sh)).

Key flags passed for the default Hindi run:
```
--acmlm --pe --pos_mode alibi --model_checkpoint <d2v.pt> --vocab_dir vocab_hi
--frames_per_sec 50 --eos_repeat 3 --fill_weight 0.03 --num_sa_layers 8
--mask_prob 0.20 --mask_prob_max 0.30 --mask_length 10 --batch_size 32 --grad_accum 2
--select_metric iter_wer --eval_decode_steps 32 …   (--tap_cache only if TAP_CACHE set)
```

Inside `train.py` the tokenizer, collator, dataset, and model are chosen by two switches:
`args.acmlm` and `args.tap_cache`.

```
tok = build_acmlm_tokenizer_hi(vocab_dir)                       # train.py:323  → len 70, <mask>=69
if acmlm and tap_cache:  collator = DataCollatorACMLMTapCached  # train.py:331  (cached 1024-d taps)
elif acmlm:              collator = build_collator(..., normalize_fn=normalize_hi)  # train.py:338 (LIVE)
```
- **LIVE path (default):** `build_collator` ([model_d2v_acmlm.py:111](model_d2v_acmlm.py)) returns a
  `_DataCollatorACMLMNorm` = the English `DataCollatorACMLM` with `_normalize` overridden to
  `normalize_hi`. Reads audio every step.
- **Cached path (`--tap_cache`):** `DataCollatorACMLMTapCached`
  ([data_tapcached_d2v.py:32](data_tapcached_d2v.py)) = the English cached collator with
  `_normalize = normalize_hi` and dim 1024.

Both paths call the **same** `_labels_for` in `filler_sa_reference.py`. Data prep (§3) is identical;
only where the tap comes from differs.

---

## 2. The two data paths (tensor shapes)

```
                        raw target text (Devanagari string)          raw waveform (16 kHz mono)
                                     │                                          │
LIVE  (default) ────────────────────┤                                          │
  ManifestDataset[i] = {audio_path, text}   data.py:50                          │
        │                            │                          Wav2Vec2FeatureExtractor(do_normalize)
        │                            │                          → input_values [B,S], attn_mask [B,S]
        │                            │                                          │
        │                            ▼                                          ▼
        │                 _labels_for(text,T)  ← §3            frozen data2vec-aqc.extract_features
        │                     labels/text_ids                       tap [B,T,1024]   model_d2v_acmlm.py:76
        │                            └──────────────┐                            │
        │                                           ▼                            ▼
        └────────────────────────────────►  DataCollatorACMLM.__call__  →  {input_values, attention_mask,
                                             filler_sa_reference.py:894      text_input_ids, labels}

CACHED (--tap_cache) ───────────────
  precompute_taps_d2v.py (ONCE) → taps.f16 memmap (ΣTᵢ,1024) + index.npy + manifest.jsonl
        │
  TapCacheDataset[i] = {tap [T,1024] fp16, text, T}   data_tapcached.py:58
        │
  LengthBucketedDistributedBatchSampler → length-homogeneous, DDP-sharded batches   data_tapcached.py:64
        │
  DataCollatorACMLMTapCached.__call__ → {tap [B,Tmax,1024], pad_mask [B,Tmax], text_input_ids, labels}
        data_tapcached.py:117  (calls _labels_for + normalize_hi)
```

`compute_output_length` ([filler_sa_reference.py:59](/speech/tomson/filler_asr/filler_sa_reference.py))
gives the frame count `T` from the sample count via the shared 320× conv geometry
(kernels `[10,3,3,3,3,2,2]`, strides `[5,2,2,2,2,2,2]`) → ~50 fps (≈20 ms/frame). data2vec‑aqc uses
the **same** conv stack, so the frame↔label geometry is unchanged from HuBERT.

---

## 3. Hindi data preparation — target text → `[<s>] … [</s>]×3 … <fill>×tail`

This is the part you asked about, step by step. It is one method, `_labels_for`, fed by
`normalize_hi`. **All steps below are verified against the running code.**

### 3.0 Where the text starts
- Manifest line: `{"key","source","target":"उल्टा चश्मा उसमें जेठालाल  का","language":"hi"}`.
- `load_rows` ([data.py:27](/speech/tomson/filler_asr/train_sa_pe_lm_head/data.py)) reads it into
  `{audio_path, text}`, **RAW** (no normalization here).
- The collator calls `_labels_for(text, T)` per clip
  ([DataCollatorACMLM: filler_sa_reference.py:905](/speech/tomson/filler_asr/filler_sa_reference.py);
  cached: [data_tapcached.py:128](/speech/tomson/filler_asr/train_sa_pe_lm_head/data_tapcached.py)).

### 3.1 `_labels_for(text, T)` — the 5 computation steps
Source: [filler_sa_reference.py:604‑619](/speech/tomson/filler_asr/filler_sa_reference.py).

```python
text = self._normalize(text).replace(" ", "|")   # (1) Hindi normalize  (2) space → '|'
ids  = self.tok(text).input_ids                   # (3) character tokenize → id list
n_specials = 1 + self.eos_repeat                   #     <s> + </s>×3  = 4
if len(ids) + n_specials > T:                      # (4) truncate transcript if it won't fit in T frames
    ids = ids[: T - n_specials]
labels = [self.bos_id] + ids + [self.eos_id]*self.eos_repeat   # (5a) prepend <s>, append </s>×3
labels = labels + [self.fill_id]*(T - len(labels))             # (5b) pad tail with <fill> to length T
return labels                                                   # len == T
```

**Step (1) — `normalize_hi`** ([text_norm_hi.py:30](text_norm_hi.py)):
1. `unicodedata.normalize("NFC", text)` — recompose so nukta forms (ज़/ड़) encode consistently as
   base + `U+093C`.
2. Iterate codepoints; **collapse runs of whitespace to a single space** (note the double space in
   the example is collapsed).
3. **Keep** a char only if it is in the Devanagari block `U+0900–U+097F` **and** its Unicode category
   does **not** start with `P` (punctuation) or `S` (symbol). This drops danda `।`/`॥`, `?:;$`,
   and zero‑width `ZWSP/ZWNJ/ZWJ`, and drops anything outside the block (Latin code‑switch, digits).
4. `.strip()`.

**Step (2)** — spaces → `|` (the word‑delimiter token, id **63**).

**Step (3) — character tokenization** — `self.tok(text).input_ids`. `self.tok` is a
`Wav2Vec2CTCTokenizer` built from `vocab_hi/vocab.json` by `build_acmlm_tokenizer_hi`
([text_norm_hi.py:90](text_norm_hi.py)). It splits the string into **individual Unicode
codepoints** and maps each via the vocab (unknown → `<unk>`=64). Crucial Devanagari detail: a
"letter" like **उल्टा** is *not* one token — matras, virama, anusvara are **separate codepoints,
each its own id**. Example: `उल्टा` = `उ ल ् ट ा` = 5 tokens (one of them the virama `U+094D`).

**Step (4)** — if the transcript (chars + 4 specials) exceeds the frame budget `T`, truncate the
char ids so the whole label vector fits in `T`.

**Step (5)** — lay the transcript at the **head** of the frame axis:
`<s>`(66) → chars → `</s>`(67)×`eos_repeat`(3) → `<fill>`(68) padding out to length `T`. This pins
char *i* to frame *i* (positional, non‑acoustic — the design's defining property).

### 3.2 Token id table (this run's `vocab_hi`)
Verified: `len(tok)=70`.

| id range | tokens |
|---|---|
| `0 … 62` | 63 Devanagari chars (freq‑ordered: `ा`=0, `े`=1, …) |
| `63` | `|` word delimiter |
| `64` | `<unk>` |
| `65` | `<pad>` |
| `66` | `<s>` (bos) |
| `67` | `</s>` (eos) |
| `68` | `<fill>` (added; **last real class** — `lm_head` predicts it) |
| `69` | `<mask>` (added; **input‑only**, never a label) |

`cfg.vocab_size = len(tok) - 1 = 69` → `lm_head` outputs **69** classes (`<mask>` excluded).
`text_embed = nn.Embedding(70, 1024)` embeds all 70 ids including `<mask>`.

### 3.3 Fully worked example (verified output)
Input `"उल्टा चश्मा उसमें जेठालाल  का"` (note the **double space**), `eos_repeat=3`.

```
RAW        : 'उल्टा चश्मा उसमें जेठालाल  का'
normalize_hi (NFC, Devanagari-only, collapse ws)  →  'उल्टा चश्मा उसमें जेठालाल का'
space → '|'                                        →  'उल्टा|चश्मा|उसमें|जेठालाल|का'   (28 codepoints)

character ids (28):
 [30,14,12,29,0, 63, 24,39,12,11,0, 63, 30,5,11,1,8, 63, 17,1,47,0,14,0,14, 63, 2,0]
   उ=30 ल=14 ्=12 ट=29 ा=0 | चश्मा… | उसमें(…anusvara ं=8) | जेठालाल | का
```
With a chosen `T = 36` (28 chars + `<s>` + `</s>`×3 + 4‑frame `<fill>` tail), `_labels_for` returns:

```
labels (len 36) =
[66,  30,14,12,29,0, 63, 24,39,12,11,0, 63, 30,5,11,1,8, 63, 17,1,47,0,14,0,14, 63, 2,0,  67,67,67,  68,68,68,68]
 <s>  └──────────────── 28 char ids (| = 63 between words) ────────────────┘  └ </s>×3 ┘  └ <fill>×4 ┘
frame: 0    1 ..............................................................  28 29 30 31  32 33 34 35
```
In a real clip `T` is the encoder frame count (`compute_output_length`, ~50/s), so the `<fill>` tail
is usually long (silence/after‑speech frames all labelled `<fill>`).

---

## 4. A‑CMLM masking (the training objective) — collator

After `_labels_for` builds the positional target `y`, the collator masks a random subset of text
positions and supervises only those. Cached version:
[data_tapcached.py:117‑140](/speech/tomson/filler_asr/train_sa_pe_lm_head/data_tapcached.py); live
version identical logic ([filler_sa_reference.py:894‑931](/speech/tomson/filler_asr/filler_sa_reference.py)).

Per clip (`fps=50` ⇒ `n_keep = min(T, ceil(T·fps/50)) = T`, i.e. the whole clip is maskable):
```python
text_ids = y.clone()                       # model input: full answer everywhere by default
labels   = full(T, -100)                   # loss ignores everything by default
p = 1.0 (prob 0.15)  else  U(0,1)          # force_full_mask_prob=0.15
sel = { i in [1, n_keep) : rand()<p }      # never mask <s> at position 0; ≥1 slot guaranteed
text_ids[sel] = <mask>(69)                 # masked INPUT slots
labels[sel]   = y[sel]                     # supervise ONLY masked slots
```
So the network must reconstruct the masked chars from **audio (tap) + surrounding text**. Eval uses
`fixed_p=1.0` (fully masked = one‑shot decode‑from‑audio) ([train.py:333](train.py)).

**Do not confuse two different masks:**
1. **Text/CMLM mask** (here, in the collator) — masks *label tokens* with `<mask>`.
2. **Audio span mask** `mask20‑30×10` (in the model, train‑only) — masks *tap frames* with a learned
   `mask_embed`, spans of 10 frames (200 ms), rate `U[0.20,0.30]`
   ([filler_sa_reference.py:528‑537 / _span_mask_and_pe:768](/speech/tomson/filler_asr/filler_sa_reference.py)).

### Batch tensors out of the collator
| tensor | shape | dtype | meaning |
|---|---|---|---|
| `tap` (cached) / `input_values` (live) | `[B,Tmax,1024]` / `[B,S]` | fp16 / fp32 | frozen features / raw audio |
| `pad_mask` (cached) / `attention_mask` (live) | `[B,Tmax]` / `[B,S]` | bool / long | True=pad / 1=valid |
| `text_input_ids` | `[B,Tmax]` | long | `y` with `<mask>` at masked slots; tail `<fill>` |
| `labels` | `[B,Tmax]` | long | `y` at masked slots, `-100` elsewhere |

---

## 5. Model forward — dims, shapes, FLOPs

Model = **frozen data2vec‑aqc body** + **trainable A‑CMLM head** (8 SA layers + text embedding +
lm_head). Built by `build_d2v_acmlm` ([model_d2v_acmlm.py:120](model_d2v_acmlm.py)); the head is the
verified `FillerHubertACMLM` at width 1024, with `self.hubert` swapped for `Data2VecEncoderAdapter`.

```
input                              shape                 op / file
──────────────────────────────────────────────────────────────────────────────────────────
waveform  (live)                   [B, S]                Wav2Vec2FeatureExtractor(do_normalize)
  └─ frozen data2vec-aqc           [B, T, 1024]          adapter.forward → extract_features   model_d2v_acmlm.py:76
     (OR cached tap, skips this)   [B, T, 1024]
tap ── span-mask (train) + PE      [B, T, 1024]          _span_mask_and_pe    filler_sa_reference.py:768
  · pos_mode=alibi ⇒ NO sinusoidal add; ALiBi enters as an additive attn bias instead
text_input_ids                     [B, T]                → text_embed         nn.Embedding(70,1024)
fuse by ADDITION                   [B, T, 1024]          h = tap_pe + text_embed(ids)   :810
8× SA encoder layer (post-norm)    [B, T, 1024]          _run_sa (+ALiBi bias)          :485
  · per layer: MHSA(16 heads, dk=64) + FFN(1024→4096→1024, GELU), residual+LayerNorm
dropout → lm_head                  [B, T, 69]            nn.Linear(1024,69)             :812
```

**Parameter widths**
- encoder dim `d = 1024`; SA heads `h = 16` ⇒ head dim `dk = 64`; FFN `dff = 4096`; layers `L = 8`.
- `text_embed`: 70 × 1024 ≈ 71.7 K params. `lm_head`: 1024 × 69 + 69 ≈ 70.7 K.
- SA head params ≈ `L·(4·d² + 2·d·dff) = 8·(4·1.05e6 + 2·1024·4096) ≈ 8·(4.19e6+8.39e6) ≈ 1.0e8` ≈
  **~100 M trainable** (SA) + embeds/head + `mask_embed(1024)`. The frozen data2vec body
  (~0.3 B) is **not** trained.

**FLOPs (trainable SA head, per sequence of length `T`)** — MACs≈, FLOPs = 2×MAC:
```
per layer MAC = 4·T·d²   (Q,K,V,O proj)
              + 2·T²·d    (QKᵀ scores + context·V)
              + 2·T·d·dff (FFN two matmuls)
              = 12,582,912·T + 2,048·T²
×8 layers      = 100,663,296·T + 16,384·T²    MAC
FLOPs ≈ 2×     ≈ 2.01e8·T + 3.28e4·T²
```
Worked (T = 500 frames ≈ 10 s): `2.01e8·500 + 3.28e4·500² ≈ 1.01e11 + 8.2e9 ≈ 1.09e11 FLOPs`
≈ **0.11 TFLOP/seq** forward (backward ≈ 2× more). `lm_head`≈`2·T·d·69≈1.4e5·T` and `text_embed`
(gather) are negligible.

**Frozen encoder cost (live path):** data2vec‑aqc LARGE ≈ 24 transformer layers at the same
`d=1024,dff=4096` ⇒ ≈ **3× the 8‑layer head** ≈ 0.33 TFLOP/seq **forward only**, plus the conv
feature extractor. It dominates the per‑step cost and is recomputed every step in live mode — which
is exactly why `precompute_taps_d2v.py` exists: dump the deterministic tap **once**
(`taps.f16`, `ΣTᵢ×1024×2` bytes) and skip the encoder entirely (`--tap_cache`).

---

## 6. Loss & DDP token‑exact averaging

Training step ([train.py:596](train.py)):
```python
lg = model(**inp).logits.float()            # [B,T,69]
loss_sum = framewise_ce_loss(lg[:, :m], lb, fill_id=68, fill_weight=0.03, reduction="sum")
```
- `framewise_ce_loss` ([filler_sa_reference.py:657](/speech/tomson/filler_asr/filler_sa_reference.py)):
  per‑frame CrossEntropy over 69 classes, `ignore_index=-100` (so **only the CMLM‑masked slots
  count**), with the `<fill>`(68) class **down‑weighted to 0.03** (it dominates; otherwise the model
  collapses to predicting `<fill>`).
- `reduction="sum"`: accumulate raw loss sums + a weighted token count over the whole
  grad‑accum × DDP window, `all_reduce` the token count ([train.py:614](train.py)), rescale grads by
  `world/global_tokens` → an **exact token‑weighted mean**, not a biased mean‑of‑means. This is why
  the cached path uses `LengthBucketedDistributedBatchSampler` — it guarantees **equal per‑rank batch
  counts** so ranks stay in lockstep at the boundary all‑reduce.

---

## 7. End‑to‑end diagram (Hindi, cached path)

```
 train_hi.jsonl                                clips/*.wav (16 kHz)
 {"target":"उल्टा … का"}                              │
        │                                             │  (ONE-TIME) precompute_taps_d2v.py
        │                                             ▼
        │                           frozen data2vec-aqc.extract_features → tap [T,1024] fp16
        │                                             │
        │                                    taps.f16 memmap + index.npy + manifest.jsonl
        │                                             │
        ▼                                             ▼
 normalize_hi (NFC, Deva-only)              TapCacheDataset[i] = {tap[T,1024], text, T}
        │  text_norm_hi.py:30                         │            data_tapcached.py:58
        ▼                                             ▼
 space→'|' ; tok(...)→char ids           LengthBucketedDistributedBatchSampler (len-bucket, DDP)
        │                                             │            data_tapcached.py:64
        ▼                                             ▼
 _labels_for  [<s>]+chars+[</s>]×3+<fill>×tail  ── DataCollatorACMLMTapCached.__call__ ──►
        │  filler_sa_reference.py:604                 │  data_tapcached.py:117
        │                                             │  • pad taps → [B,Tmax,1024], pad_mask[B,Tmax]
        │                                             │  • text_input_ids: <mask> at Bernoulli(p) slots
        └───────────────────────────────────────────►│  • labels: y at masked, -100 else
                                                      ▼
                                   tap+PE ⊕ text_embed → 8× SA(+ALiBi) → lm_head [B,Tmax,69]
                                                      ▼
                                   framewise CE (ignore -100, <fill> wt 0.03), token-exact DDP mean
```

In the **live** default run, replace the "ONE‑TIME precompute + memmap + TapCacheDataset + Sampler"
column with: `ManifestDataset` → `DataCollatorACMLM(normalize_fn=normalize_hi)` reads the wav and
runs the frozen encoder **inside every step** ([train.py:414‑419](train.py)); the label branch on the
left is identical.

---

## 8. English → Hindi diffs (everything else is shared)

| Stage | English `filler_asr` | Hindi here |
|---|---|---|
| Encoder / tap dim | HuBERT‑xlarge, 1280 | data2vec‑aqc, **1024** |
| Text normalize | `_normalize` (lowercase, strip ASCII punct) | **`normalize_hi`** (NFC, Devanagari‑only) — text_norm_hi.py:30 |
| Vocab | hard‑coded 33 (a–z…) | **corpus‑built 63 Deva chars** → `vocab_hi/` — text_norm_hi.py:69 |
| Tokenizer | `build_acmlm_tokenizer` (assert len 34) | **`build_acmlm_tokenizer_hi`** (size‑agnostic, len 70) — :90 |
| SA FFN width `dff` | 5120 | **4096** (model_d2v_acmlm.py:121) |
| Label layout `_labels_for` | filler_sa_reference.py:604 | **inherited, unchanged** |
| Dataset / sampler / CMLM mask / loss | filler_asr | **inherited, unchanged** |

---

*Line numbers are against the code as of this writing; the label/mask/loss logic lives in
`/speech/tomson/filler_asr/filler_sa_reference.py` and
`/speech/tomson/filler_asr/train_sa_pe_lm_head/data_tapcached.py`, imported here.*
