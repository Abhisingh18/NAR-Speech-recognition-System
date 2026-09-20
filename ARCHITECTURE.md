# Architecture — indic-nar-filler_asr (A-CMLM on frozen data2vec-aqc, Hindi)

Non-autoregressive **A-CMLM** ASR head (from `filler_asr`) with the acoustic encoder swapped
from HuBERT-xLarge to a **frozen data2vec-aqc large**, trained on Hindi (Devanagari char tokens).
PDF version: [architecture.pdf](architecture.pdf) (rendered by [make_arch_d2v_acmlm.py](make_arch_d2v_acmlm.py)).

```
 raw audio            +---------------------------------------------------------------------------+
 (16 kHz mono)        |  waveform x   (B, N_samples)   fp32, per-utterance LayerNorm (do_normalize) |
                      +---------------------------------------------------------------------------+
                                             |
                                             v
 #============= FROZEN  data2vec-aqc  large  (313.3M, requires_grad=False, eval-pinned) =============#
 # fairseq data2vec_audio body via Data2VecAQCEncoder  ->  CTC head + AQC (quantizer/project_q/EMA)  #
 #                                                          heads are DROPPED (not instantiated)     #
 # (1) Conv feature extractor  7 x Conv1d (512 ch)   kernels [10,3,3,3,3,2,2] strides [5,2,2,2,2,2,2]#
 #     320x downsample -> 50 fps    extractor_mode=layer_norm    (B,512,T)->(B,T,512)                #
 #                                                                 4.21M par   ~5 GFLOP/s audio      #
 # (2) LayerNorm + feature projection   Linear 512 -> 1024                     (B, T, 1024)          #
 # (3) conv positional embed (k=128,g=16)  +  24 x Transformer encoder layer  (POST-norm)            #
 #     each: MHSA(d=1024,H=16,dk=64) ->Add&LN-> FFN(1024->4096->1024,GELU) ->Add&LN                  #
 #     linear/token = 8d^2+4·d·ff = 25.2 MFLOP    attn/token = 4·T·d          309M par ~33 GFLOP/s   #
 #============================================|=====================================================#
                                             v      TAP = extract_features(layer=None)  BEFORE CTC
                      +------------------------------------------------+
                      |  tap  (B, T, 1024)   deterministic (encoder eval)|
                      +------------------------------------------------+
                                             |
  text_input_ids               (train only) |  span-mask 20-30% of frames, spans len 10,
  (B,T) Devanagari char ids  +  <mask>       |  replaced by learned mask_embed (1024)
        |                                    v
        v                        +-----------------------------------+
  +---------------+              | + symmetric ALiBi position bias    |  (pos_mode=alibi: no PE add)
  |  text_embed   |              +-----------------------------------+
  |  70 x 1024    |----(ADD)------------------> h = tap(+mask) + E_text(text_input_ids)
  +---------------+   fusion                    |
                                                v
       #============ 8 x  POST-norm Self-Attention layer   (TRAINABLE) ============#
       #  MHSA(d=1024,H=16)+ALiBi ->Add&LN-> FFN(1024->4096->1024,GELU) ->Add&LN    #
       #  linear/token = 25.2 MFLOP   100.6M par              ~10 GFLOP/s audio     #
       #===============================|==========================================#
                                       v
                      +------------------------------------------------+
                      |  Dropout(0.1) -> lm_head  Linear 1024 -> 69     |   logits (B, T, 69)
                      +------------------------------------------------+
                                       v
            framewise CrossEntropy on MASKED slots ONLY (CMLM), ignore_index=-100, fill_weight
            target y = [<s>] + char_ids(| for space) + [</s>]x3 + <fill> x (T-len);  n_keep=ceil(dur*50)=T
```

## Block-by-block (intuition · shapes · FLOPs)

**A. The A-CMLM idea (unchanged).** Non-autoregressive: the whole transcript is laid on the
acoustic frame axis (char *i* → frame *i*), a random subset of char slots is masked, and the
model predicts them from the frozen acoustic tap **plus** the surrounding unmasked chars. Inference
starts 100% masked and iterates mask-predict. Only the encoder feeding the tap changed.

**B. Frozen encoder — the one real change.** `data2vec-aqc large` (SPRING-INX SSL), loaded via
fairseq `Data2VecAQCEncoder` which builds **only** the inner `data2vec_audio` body — the CTC head
and AQC pretraining modules (quantizer/project_q/contr_proj/EMA) are physically absent, so
"tap before CTC" == `extract_features()`. 320× conv → 50 fps (same geometry as HuBERT) + 24
post-norm layers @ d=1024. `wav (B,N) → (B,T,1024)`. Pinned to `eval()` under `model.train()` →
deterministic tap (verified). ~313M params, **~33 GFLOP/s audio** (vs HuBERT-xLarge's ~102).

**C. Tap → span-mask → position → text fusion (reused verbatim).** (1) tap `(B,T,1024)`.
(2) train-only span-mask: 20–30% of valid frames, contiguous spans of 10 (~200 ms), overwritten
by a learned `mask_embed(1024)` — inpainting augmentation. (3) symmetric **ALiBi** bias into
attention (no sinusoidal add) — holds `</s>`/tail structure on long clips. (4) `text_embed(70×1024)`
on the char ids (with `<mask>=69` at masked slots), fused by **addition**: `h = tap(+mask) + E_text`.

**D. 8 SA layers → lm_head → loss.** 8 post-norm Transformer layers (d=1024, H=16, ff=4096; the
only width change from filler_asr's 1280/5120). `Dropout(0.1) → lm_head 1024→69 → logits (B,T,69)`.
Loss = framewise CE on **masked slots only** (`-100` elsewhere), with `fill_weight` down-weighting
`<fill>`. Trainable surface = SA + lm_head + text_embed + mask_embed ≈ **100.8M**; encoder = 0
trainable tensors (asserted).

**E. Targets / vocab / 50-fps budget (Hindi).** Collator: `normalize_hi` (NFC + drop **all**
punct/symbol/foreign/zero-width) → spaces→`|` → per-**codepoint** char ids. Devanagari is
codepoint-level: each consonant, vowel, matra, virama `्`, nukta `़`, anusvara `ं`, candrabindu
`ँ`, visarga `ः` is its own token; conjuncts spelled via virama (`क्ष = क ् ष`). Vocab = 63
Devanagari + `|` + `<unk><pad><s></s>` + `<fill>` + `<mask>` → `cfg.vocab_size=69`. Frame budget:
Hindi char rate p50=12, p99=19, max=29 tok/s vs 50 fps → **0% truncation**, ~283 spare frames
median. The long `<fill>` tail is why `fill_weight` exists.

## Changes vs filler_asr A-CMLM
Reference run: `train_acmlm_alibi_fps50_fw003_exp150_4gpu`.

| aspect | filler_asr A-CMLM | indic-nar-filler_asr (this) |
|---|---|---|
| acoustic encoder | frozen HuBERT-xLarge ~963M, 48L d1280 | frozen data2vec-aqc **313M, 24L d1024** |
| encoder framework | HF transformers `HubertModel` | **fairseq** `data2vec_audio` (`Data2VecAQCEncoder`) |
| CTC head removal | `HubertModel` (CTC never built) | inner body only; **CTC + AQC heads dropped** |
| tap | `last_hidden_state (B,T,1280)` | `extract_features (B,T,1024)`, before CTC |
| feature width d | 1280 | **1024** |
| input normalization | `Wav2Vec2FeatureExtractor(do_normalize)` | same (== fairseq `layer_norm`) |
| frame rate | 50 fps | 50 fps (same 320× conv) |
| SA head | 8 layers, d1280, ff5120 | 8 layers, **d1024, ff4096** |
| lm_head | 1280 → 33 | **1024 → 69** |
| text_embed | 34 × 1280 | **70 × 1024** |
| vocabulary | 33 English char (a–z `'` `|` +`<fill>`) | **69** (63 Devanagari + `|` + specials + `<fill>`) |
| tokenization | Latin char | **Devanagari codepoint** (matra/virama/nukta split) |
| text normalization | lowercase + strip ASCII punct | **NFC + drop all punct/symbol/foreign/zero-width** |
| model integration | subclass `HubertPreTrainedModel` | **encoder swap**: `self.hubert → Data2VecEncoderAdapter` |
| conda env | `filler_asr` (torch2.7, tf5.8, no fairseq) | **`indic-nar-filler_asr`** (clone of `smear-moe-gemma3`: torch2.4, tf5.5, fairseq 0.12.2) |
| training data | LibriSpeech 960h (English) | **IndicASR / AI4B Hindi** (~43k utts) |
| tap cache | 442 GB precomputed memmap | live encoder forward (no cache yet) |

**Unchanged:** A-CMLM masking (20–30% span-mask×10, force-full p=0.15), addition text fusion,
symmetric ALiBi, `eos_repeat=3`, positional `<fill>` targets, framewise-CE-on-masked + `fill_weight`,
iterative mask-predict decode, 50-fps budget, ~100.8M trainable head.
