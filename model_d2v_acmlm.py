"""indic-nar-filler_asr — A-CMLM head on a FROZEN data2vec-aqc encoder.

Same non-autoregressive A-CMLM recipe as filler_asr (8 self-attention blocks + text
embedding fused by ADDITION + lm_head, 20-30% tap span-masking, symmetric ALiBi), but the
frozen HuBERT-xlarge tap is REPLACED by a frozen data2vec-aqc (SPRING-INX) body. The tap is
taken BEFORE the CTC head (Data2VecAQCEncoder instantiates only the inner data2vec_audio
body, so the CTC proj is physically absent).

Design: we do NOT re-implement the A-CMLM head. We build the verified FillerHubertACMLM at
the data2vec width (1024) and swap its `self.hubert` for a thin adapter that makes the fairseq
data2vec-aqc encoder quack like transformers' HubertModel for the two calls the head makes:
    self.hubert(input_values, attention_mask=..., return_dict=True) -> (feats,)   [B,T,1024]
    self.hubert._get_feature_vector_attention_mask(T, attention_mask) -> [B,T] long
Everything downstream of the tap (span mask, PE/ALiBi, ADDITION text fusion, decode, loss)
is reused byte-for-byte from the verified pipeline, so only the encoder + feature width change.
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

# --- reuse the verified A-CMLM head, tokenizer, collator, loss, frame geometry ---
_HERE = os.path.dirname(os.path.abspath(__file__))
FILLER_ROOT = os.environ.get("FILLER_ROOT", os.path.join(_HERE, "deps", "filler_asr"))        # local copy
if FILLER_ROOT not in sys.path:
    sys.path.insert(0, FILLER_ROOT)
from filler_sa_reference import (                       # noqa: E402
    FillerHubertACMLM, build_acmlm_tokenizer, DataCollatorACMLM,
    framewise_ce_loss, compute_output_length, SAMPLING_RATE,
)
from transformers import HubertConfig, Wav2Vec2FeatureExtractor   # noqa: E402

# --- data2vec-aqc loader (drops the CTC head) from the SLAM-LLM / SMEAR stack ---
SLAM_SRC = os.environ.get("SLAM_SRC", os.path.join(_HERE, "deps", "SMEAR-MoE-ASR", "src"))   # local copy
if SLAM_SRC not in sys.path:
    sys.path.insert(0, SLAM_SRC)
from slam_llm.models.encoder import Data2VecAQCEncoder             # noqa: E402

D2V_CKPT = os.environ.get(                                          # Kannada CTC-finetuned (abhishek's copy)
    "D2V_CKPT",
    "/speech/abhishek/kannada_data/encoders/SPRING_INX_data2vec_aqc_Kannada.pt")
D2V_DIM = 1024                                          # SPRING-INX data2vec-aqc is LARGE (1024-d)


class _FeatureExtractorShim:
    """Provides hubert.feature_extractor._freeze_parameters() so the reused freeze helpers
    (and any code touching self.hubert.feature_extractor) keep working after the swap."""

    def __init__(self, owner):
        self._owner = owner

    def _freeze_parameters(self):
        for p in self._owner.parameters():
            p.requires_grad = False


class Data2VecEncoderAdapter(nn.Module):
    """Wraps a frozen Data2VecAQCEncoder to look like transformers' HubertModel.

    The tap is `extract_features` output (data2vec-aqc, BEFORE any CTC head). Same 320x conv
    stack as HuBERT/wav2vec2 -> 50 fps, so filler_asr's compute_output_length gives the frame
    count and the frame<->label geometry is unchanged.
    """

    def __init__(self, d2v_encoder):
        super().__init__()
        self.d2v = d2v_encoder                          # Data2VecAQCEncoder (.model, .tgt_layer)
        self.feature_extractor = _FeatureExtractorShim(self)

    def forward(self, input_values, attention_mask=None, return_dict=None,
                output_attentions=None, output_hidden_states=None, **kw):
        # input_values: [B, S] raw waveform, already per-utterance normalized (do_normalize=True,
        # numerically == fairseq F.layer_norm(x, x.shape)). padding_mask: True = PAD (fairseq).
        padding_mask = attention_mask.eq(0) if attention_mask is not None else None
        feats = self.d2v.extract_features(input_values, padding_mask)   # [B, T, 1024]
        return (feats,)

    def _get_feature_vector_attention_mask(self, feat_T, attention_mask):
        """Map the sample-level attention_mask -> frame-level valid mask [B, feat_T] (1 = valid),
        using the (HuBERT-identical) conv geometry. Clamp to the actual feature length feat_T."""
        lens = attention_mask.long().sum(-1)                            # samples per clip
        feat_lens = torch.tensor(
            [compute_output_length(int(n)) for n in lens.tolist()],
            device=attention_mask.device, dtype=torch.long).clamp(max=feat_T)
        ar = torch.arange(feat_T, device=attention_mask.device)[None, :]
        return (ar < feat_lens[:, None]).long()


def build_feature_extractor():
    """Raw-waveform, per-utterance-normalized feature extractor (do_normalize=True is the same
    per-clip zero-mean/unit-var that fairseq's normalize=true / F.layer_norm applies)."""
    return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLING_RATE,
                                    padding_value=0.0, do_normalize=True,
                                    return_attention_mask=True)


class _DataCollatorACMLMNorm(DataCollatorACMLM):
    """DataCollatorACMLM with a pluggable text normalizer. Default (normalize_fn=None) keeps the
    English behavior (lowercase + strip ASCII punct) so the LibriSpeech smoke is unchanged; pass
    normalize_hi for Hindi so targets are NFC + punctuation-stripped, matching vocab_hi/vocab.json."""

    def __init__(self, *a, normalize_fn=None, **kw):
        super().__init__(*a, **kw)
        self._norm_fn = normalize_fn

    def _normalize(self, text):
        return self._norm_fn(text) if self._norm_fn is not None else super()._normalize(text)


def build_collator(tokenizer, frames_per_sec=50, eos_repeat=3,
                   force_full_mask_prob=0.15, fixed_p=None, normalize_fn=None):
    """DataCollatorACMLM over data2vec-normalized raw audio. normalize_fn=normalize_hi for Hindi."""
    return _DataCollatorACMLMNorm(build_feature_extractor(), tokenizer,
                                  frames_per_sec=frames_per_sec, eos_repeat=eos_repeat,
                                  force_full_mask_prob=force_full_mask_prob, fixed_p=fixed_p,
                                  normalize_fn=normalize_fn)


def build_d2v_acmlm(tokenizer, *, d2v_ckpt=D2V_CKPT, encoder_dim=D2V_DIM,
                    num_sa_layers=8, sa_nhead=16, sa_dim_feedforward=4096, sa_dropout=0.1,
                    use_sinusoidal_pe=True, pos_mode="alibi",
                    mask_prob=0.20, mask_prob_max=0.30, mask_length=10,
                    tgt_layer=None, device="cpu"):
    """Build the A-CMLM head at the data2vec width, then swap in the frozen data2vec-aqc body."""
    # --- config: A-CMLM head at encoder_dim; num_hidden_layers=1 keeps the throwaway HubertModel
    #     body (immediately replaced by the adapter) tiny. Direct construction (NOT from_pretrained)
    #     runs init on the real device -> avoids the transformers-5.x meta-init text_embed NaN. ---
    assert encoder_dim % sa_nhead == 0, f"encoder_dim {encoder_dim} not divisible by sa_nhead {sa_nhead}"
    cfg = HubertConfig()
    cfg.hidden_size = encoder_dim
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = sa_nhead
    cfg.vocab_size = len(tokenizer) - 1                 # real classes only (<mask> is input-only)
    cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id = (
        tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    cfg.num_sa_layers = num_sa_layers
    cfg.sa_nhead = sa_nhead
    cfg.sa_dim_feedforward = sa_dim_feedforward
    cfg.sa_dropout = sa_dropout
    cfg.final_dropout = sa_dropout
    cfg.pos_mode = pos_mode                             # "sinusoidal" | "alibi" | "rope"
    cfg.use_sinusoidal_pe = use_sinusoidal_pe
    cfg.pe_scale = 1.0
    cfg.sa_mask_prob = mask_prob
    cfg.sa_mask_prob_max = mask_prob_max
    cfg.sa_mask_length = mask_length
    cfg.apply_spec_augment = False

    model = FillerHubertACMLM(cfg)                      # head built at encoder_dim on real device

    # --- swap the encoder: frozen data2vec-aqc body, CTC head physically absent ---
    enc = Data2VecAQCEncoder.load(
        SimpleNamespace(encoder_path=d2v_ckpt, encoder_tgt_layer=tgt_layer))
    model.hubert = Data2VecEncoderAdapter(enc)          # replaces (and GCs) the throwaway body

    # --- belt-and-suspenders: re-init text_embed on the real device ---
    with torch.no_grad():
        nn.init.normal_(model.text_embed.weight, mean=0.0, std=0.02)

    # --- freeze EVERYTHING except the A-CMLM head (SA + lm_head + mask_embed + text_embed) ---
    for p in model.parameters():
        p.requires_grad = False
    for p in model.sa.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True
    if getattr(model, "mask_embed", None) is not None:
        model.mask_embed.requires_grad = True
    for p in model.text_embed.parameters():
        p.requires_grad = True

    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and not torch.isfinite(p).all()]
    assert not bad, f"non-finite trainable params after init: {bad}"
    return model.to(device)
