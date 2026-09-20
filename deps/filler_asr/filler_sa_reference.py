"""
filler_asr — Self-Attention model: single-file, fully-commented REFERENCE.

This file reproduces the reviewed architecture (results/arch_detailed.png) end-to-end,
one block per diagram block, consolidating code that normally lives across several files:

    audio ─► Wav2Vec2FeatureExtractor (normalize)
          ─► [FROZEN HuBERT-xlarge body, CTC head dropped] ─► TAP last_hidden_state (B,T,1280)
          ─► ⊕ sinusoidal positional encoding            (fixed, no params)
          ─► 2 × post-norm Self-Attention layer          (TRAINABLE)
                 MHSA(+frame mask) → Add&LayerNorm → FFN → Add&LayerNorm
          ─► Dropout(0.1) ─► lm_head (1280→33) ─► logits (B,T,33)

    loss  ─► framewise CrossEntropyLoss(ignore_index=-100)
             targets   : positional <fill> labels  ([<s>]+chars+[</s>]+<fill>×tail)   (label prep)
             loss mask : duration budget  n_keep=ceil(dur·25)  → labels[n_keep:]=-100  + pad=-100
             frame mask: src_key_padding_mask inside attention (pad frames ignored)

Mirrors: model_sa.py (model), collator.py + data_local.py (label prep + masks),
         dataset.compute_output_length (frame count), model.py (framewise CE).
Only the 2 SA layers + lm_head are trained (~39.4M, ~4%); HuBERT stays frozen.
"""
import os
import re
import math
import json
import argparse

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    HubertConfig,
    HubertModel,
    HubertPreTrainedModel,
)
from transformers.models.hubert.modeling_hubert import _compute_mask_indices
from transformers.modeling_outputs import CausalLMOutput


# ============================================================================
# BLOCK 0 — constants (vocab + frame geometry)
# ============================================================================
# 33-token character vocab: a-z=1..26, '=0, |=27, <unk>=28, <pad>=29, <s>=30,
# </s>=31, <fill>=32.  <fill> is the "filler" class the model emits on non-content frames.
FILL_TOKEN = "<fill>"
SAMPLING_RATE = 16000
# Duration budget: only the first ceil(dur_s * FRAMES_PER_SEC) label frames are graded;
# the rest of the <fill> tail is ignored in the loss (see BLOCK 5).  This is the "frames/sec budget thing".
FRAMES_PER_SEC = 25
# Punctuation stripped from the reference transcript before char-tokenizing (training normalization).
CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"]'


def compute_output_length(input_length):
    """Encoder frame count T for an audio of `input_length` samples.

    Standard HuBERT/Wav2Vec2 conv stack (kernels [10,3,3,3,3,2,2], strides [5,2,2,2,2,2,2]);
    320x downsample → ~50 fps (≈20 ms/frame).  Used to size the per-frame label vector.
    """
    length = input_length
    for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        length = (length - kernel) // stride + 1
    return length


# ============================================================================
# BLOCK 1 — sinusoidal positional encoding  (the ⊕ PE in the diagram)
# ============================================================================
def sinusoidal_positional_encoding(seq_len, dim, device, dtype):
    """Fixed Vaswani-2017 PE, shape (seq_len, dim), values in [-1, 1] — NO parameters.

        PE[p, 2i]   = sin( p / 10000^(2i/dim) )
        PE[p, 2i+1] = cos( p / 10000^(2i/dim) )

    Added to the tapped HuBERT features so the (otherwise permutation-equivariant)
    self-attention layers get an absolute notion of frame position.
    """
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)        # (L, 1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))                                       # (dim/2,)
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


# ----------------------------------------------------------------------------
# BLOCK 1b — symmetric ALiBi (opt-in alternative to sinusoidal PE; config.pos_mode="alibi")
# ----------------------------------------------------------------------------
def _alibi_slopes(nheads, device, dtype):
    """Geometric ALiBi slopes 2^(-8/n), 2^(-16/n), ... 2^-8 (Press et al. 2021).
    Exact for power-of-2 n (nhead=16 here -> 2^-0.5 ... 2^-8)."""
    start = 2.0 ** (-8.0 / nheads)
    return torch.tensor([start ** (i + 1) for i in range(nheads)], device=device, dtype=dtype)


def symmetric_alibi_mask(seq_len, nheads, batch, device, dtype):
    """SYMMETRIC (bidirectional-encoder) ALiBi additive attention bias:

        bias[h, i, j] = -slope_h * |i - j|

    Returned as a (batch*nheads, seq_len, seq_len) FLOAT tensor — the (N*num_heads, L, S)
    attn_mask layout nn.MultiheadAttention expects, ordered batch-outer / head-inner. It is
    ADDED to the pre-softmax attention logits, so nearer frames get more weight and the bias
    extrapolates to sequence lengths unseen in training. No absolute position is used, so it
    replaces (does not stack with) the sinusoidal PE. Bidirectional |i-j| (not the causal
    lower-triangular form) because the SA stack attends both directions.
    """
    i = torch.arange(seq_len, device=device)
    dist = (i[None, :] - i[:, None]).abs().to(dtype)                    # (L, L) = |i-j|
    slopes = _alibi_slopes(nheads, device, dtype)                      # (H,)
    bias = -slopes[:, None, None] * dist[None]                         # (H, L, L)
    return bias.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * nheads, seq_len, seq_len)


# ----------------------------------------------------------------------------
# BLOCK 1c — RoPE (rotary positional embedding) + NTK-aware scaling
#   Opt-in third positional scheme (config.pos_mode == "rope"), alternative to the
#   sinusoidal add (BLOCK 1) and the ALiBi bias (BLOCK 1b).
#
#   Idea: instead of ADDING position (sinusoidal) or BIASING the logits (ALiBi),
#   RoPE ROTATES each per-head query/key vector by an angle proportional to its
#   frame index. Because R(mθ)ᵀ R(nθ) = R((n−m)θ), the attention logit qₘ·kₙ then
#   depends only on the RELATIVE offset (m−n) — a relative scheme implemented with
#   purely per-token (absolute) ops. That rotation must happen to q,k *inside*
#   attention, which nn.MultiheadAttention does not expose, so pos_mode=="rope"
#   swaps in the small custom post-norm encoder below (RoPETransformerEncoder).
#   It is numerically identical to the nn.TransformerEncoderLayer used elsewhere
#   (post-norm, GELU FFN, same dims/dropout) except q,k are rotated before SDPA.
#
#   NTK-aware context extension (bloc97, 2023): to run on sequences LONGER than
#   those seen in training without retraining, rescale the rotary BASE rather than
#   the positions —  base' = base · s^(d/(d−2)) —  which barely touches the
#   high-frequency dims (local detail) while interpolating the low-frequency dims
#   that would otherwise extrapolate off-distribution. "Dynamic" NTK sets the
#   stretch from the ACTUAL length, s = max(1, T / orig_len), so in-distribution
#   sequences (T ≤ orig_len ⇒ s = 1) are byte-for-byte plain RoPE and only longer
#   ones get scaled. This is the best no-fine-tune long-context option for RoPE.
# ----------------------------------------------------------------------------
def _rope_ntk_base(head_dim, base, seq_len, orig_len, ntk_factor, dynamic):
    """Effective RoPE base after (optional) NTK-aware scaling.

        dynamic=True  -> s = max(1, seq_len / orig_len)   (Dynamic NTK: no-op at
                         or below the trained length orig_len)
        dynamic=False -> s = ntk_factor                   (static NTK; s<=1 = off)

    Returns base unchanged when s <= 1, else base * s ** (d / (d - 2))."""
    s = (max(1.0, seq_len / float(orig_len)) if dynamic else float(ntk_factor))
    if s <= 1.0:
        return float(base)
    return float(base) * (s ** (head_dim / (head_dim - 2.0)))


def build_rope_cache(seq_len, head_dim, device, base=10000.0,
                     orig_len=1024, ntk_factor=1.0, dynamic=True):
    """Precompute (cos, sin), each (seq_len, head_dim) in float32, for rotate_half:

        inv_freq[i] = 1 / base_eff ** (2i / head_dim),   i = 0 .. head_dim/2 - 1
        freqs       = outer(positions, inv_freq)          -> (L, head_dim/2)
        emb         = [freqs, freqs]                       -> (L, head_dim)
        cos, sin    = emb.cos(), emb.sin()

    Kept in float32 (the rotation is applied in float32 for bf16-safe precision)."""
    base_eff = _rope_ntk_base(head_dim, base, seq_len, orig_len, ntk_factor, dynamic)
    inv_freq = 1.0 / (base_eff ** (torch.arange(0, head_dim, 2, device=device,
                                                dtype=torch.float32) / head_dim))   # (d/2,)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)                   # (L,)
    freqs = torch.outer(t, inv_freq)                                               # (L, d/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                        # (L, d)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    """(-x2, x1) split on the last dim — the rotate_half convention (GPT-NeoX/Llama)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Rotate q,k (each (B, H, L, dk)) by the angles in cos/sin ((L, dk)).
    Done in float32 then cast back so bf16 training keeps rotation precision."""
    cos = cos.float()[None, None]                                                  # (1,1,L,dk)
    sin = sin.float()[None, None]
    qf, kf = q.float(), k.float()
    q_rot = qf * cos + _rotate_half(qf) * sin
    k_rot = kf * cos + _rotate_half(kf) * sin
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class RoPESelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE applied to q,k.

    Mirrors nn.MultiheadAttention's math (separate q/k/v/out projections, scaled
    dot-product, key-padding mask) but rotates q,k with apply_rope() first, so the
    logits depend on relative frame offset. Uses F.scaled_dot_product_attention
    (flash / mem-efficient kernels when eligible). No all-masked query row is
    possible (the key-padding mask is shared across queries and every valid frame
    is its own valid key), so softmax stays finite — this mode is free of the MHA
    fast-path float-mask NaN that _mha_fastpath_disabled() works around for ALiBi."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)   # (B,H,T,dk)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        attn_mask = None
        if key_padding_mask is not None:                                          # (B,T) True=pad
            attn_mask = (~key_padding_mask).view(B, 1, 1, T)                       # (B,1,1,T) True=attend
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0))                   # (B,H,T,dk)
        out = out.transpose(1, 2).reshape(B, T, self.nhead * self.head_dim)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    """Post-norm Transformer encoder layer (== nn.TransformerEncoderLayer with
    norm_first=False) but with RoPE self-attention:
        x = LayerNorm(x + dropout(SelfAttn(x)))
        x = LayerNorm(x + dropout(FFN(x))),   FFN = Linear -> GELU -> Linear."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = RoPESelfAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, cos, sin, src_key_padding_mask=None):
        a = self.self_attn(x, cos, sin, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout1(a))
        f = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(f))
        return x


class RoPETransformerEncoder(nn.Module):
    """Drop-in stand-in for nn.TransformerEncoder used only when pos_mode=='rope'.

    forward(src, mask=None, src_key_padding_mask=None) is signature-compatible with
    nn.TransformerEncoder, so the existing _run_sa() calls it UNCHANGED (`mask` is
    always None in this mode and is ignored). Builds the (cos,sin) rotary cache ONCE
    per forward (positions are shared across all layers) with the (dynamic) NTK-aware
    base read from the config."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers,
                 rope_theta=10000.0, rope_orig_len=1024, rope_ntk_factor=1.0,
                 rope_ntk_dynamic=True):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)])
        self.head_dim = d_model // nhead
        self.rope_theta = rope_theta
        self.rope_orig_len = rope_orig_len
        self.rope_ntk_factor = rope_ntk_factor
        self.rope_ntk_dynamic = rope_ntk_dynamic

    def forward(self, src, mask=None, src_key_padding_mask=None):
        T = src.shape[1]
        cos, sin = build_rope_cache(T, self.head_dim, src.device,
                                    base=self.rope_theta, orig_len=self.rope_orig_len,
                                    ntk_factor=self.rope_ntk_factor,
                                    dynamic=self.rope_ntk_dynamic)
        if src_key_padding_mask is not None and src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask != 0                        # -> bool True=pad
        x = src
        for layer in self.layers:
            x = layer(x, cos, sin, src_key_padding_mask=src_key_padding_mask)
        return x


@contextlib.contextmanager
def _mha_fastpath_disabled():
    """Disable nn.MultiheadAttention's fused fast path for the duration of the block.

    WHY: at eval / under no_grad the fused encoder-layer fast path mishandles a float attn_mask
    combined with padding — it emits NaNs for EVERY position of any padded batch element
    (PyTorch 2.x). Merging the padding into the float mask does not help; only the standard path
    is correct. Training already skips the fast path (self.training=True), so this changes nothing
    there. Scope is limited to nn.MultiheadAttention, so HuBERT's own attention is untouched, and
    the previous global setting is restored on exit."""
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)


# ============================================================================
# BLOCK 2 — the model:  frozen HuBERT tap → PE → 2× post-norm SA → head
# ============================================================================
class FillerHubertSAModel(HubertPreTrainedModel):
    """Frozen HuBERT-xlarge encoder (feature extractor) + 2 NEW self-attention layers + new lm_head."""

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size

        # --- frozen body: full HuBERT-xlarge (conv feat-extractor + projection + pos-conv + 48 layers).
        #     We tap its last_hidden_state; the original CTC head is dropped (not instantiated here).
        self.hubert = HubertModel(config)

        # --- self-attention stack hyperparameters (read from config, with HuBERT-scale defaults).
        n_layers   = getattr(config, "num_sa_layers", 4)
        nhead      = getattr(config, "sa_nhead", config.num_attention_heads)          # 16
        dim_ff     = getattr(config, "sa_dim_feedforward", config.intermediate_size)  # 5120
        sa_dropout = getattr(config, "sa_dropout", config.final_dropout)             # 0.1
        self.sa_nhead = nhead                                                        # needed to shape the ALiBi mask

        # --- positional scheme (BLOCK 1 / 1b). Gated so old checkpoints load & behave unchanged:
        #     pos_mode defaults to "sinusoidal" (absent from old configs) and the sinusoidal add
        #     stays gated on use_sinusoidal_pe, so no-PE checkpoints are still no-PE. Only
        #     pos_mode=="alibi" changes anything: it drops the PE add and feeds a symmetric ALiBi
        #     additive bias into the SA stack instead.
        self.pos_mode          = getattr(config, "pos_mode", "sinusoidal")           # "sinusoidal" | "alibi"
        self.use_sinusoidal_pe = getattr(config, "use_sinusoidal_pe", False)
        self.pe_scale          = getattr(config, "pe_scale", 1.0)

        # --- input-frame masking switches (train-time only). Gated so no-mask checkpoints
        #     load unchanged: mask_embed only exists when sa_mask_prob > 0.
        #     Masking is applied to the tapped HuBERT features BEFORE the PE add, so masked
        #     slots keep their position info but lose their content — the SA stack has to
        #     fill the hole from context (the frozen encoder can't help).
        self.sa_mask_prob     = getattr(config, "sa_mask_prob", 0.0)       # target masked fraction
        self.sa_mask_prob_max = getattr(config, "sa_mask_prob_max", 0.0)   # >prob → per-batch U[prob,max]
        self.sa_mask_length   = getattr(config, "sa_mask_length", 10)      # span length in frames (10 = 200ms)
        if self.sa_mask_prob > 0:
            self.mask_embed = nn.Parameter(torch.empty(config.hidden_size).uniform_())

        # --- RoPE (BLOCK 1c) config. Only read/used when pos_mode=="rope"; harmless otherwise.
        self.rope_theta       = getattr(config, "rope_theta", 10000.0)
        self.rope_orig_len    = getattr(config, "rope_orig_len", 1024)
        self.rope_ntk_factor  = getattr(config, "rope_ntk_factor", 1.0)
        self.rope_ntk_dynamic = getattr(config, "rope_ntk_dynamic", True)

        # --- 2× post-norm Transformer encoder layer (TRAINABLE).
        #     norm_first=False  => POST-NORM:  x1 = LayerNorm( x + Sublayer(x) )
        #     Each layer = MHSA + Add&Norm + FFN(1280→5120→1280, GELU) + Add&Norm.
        #     pos_mode=="rope" swaps in the RoPETransformerEncoder twin (rotates q,k inside
        #     attention; same post-norm/GELU math); every other mode keeps nn.TransformerEncoder.
        if self.pos_mode == "rope":
            self.sa = RoPETransformerEncoder(
                d_model=config.hidden_size, nhead=nhead, dim_feedforward=dim_ff,
                dropout=sa_dropout, num_layers=n_layers,
                rope_theta=self.rope_theta, rope_orig_len=self.rope_orig_len,
                rope_ntk_factor=self.rope_ntk_factor, rope_ntk_dynamic=self.rope_ntk_dynamic)
        else:
            sa_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,     # 1280
                nhead=nhead,                    # 16 heads (dk = 80)
                dim_feedforward=dim_ff,         # 5120
                dropout=sa_dropout,             # 0.1
                activation="gelu",
                batch_first=True,               # (B, T, d)
                norm_first=False,               # post-norm (like HuBERT's encoder layers)
            )
            self.sa = nn.TransformerEncoder(sa_layer, num_layers=n_layers, enable_nested_tensor=False)

        # --- final dropout + classification head (TRAINABLE).  lm_head maps 1280 → 33 char classes.
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    # ---- weight init: HuBERT's _init_weights skips nn.MultiheadAttention's raw in_proj
    #      params (they'd be left NaN). Initialize them explicitly so the SA layers start sane.
    def _init_weights(self, module):
        super()._init_weights(module)
        # mask_embed is a raw Parameter on the root module: from_pretrained treats it as a
        # missing key and re-inits via _init_weights, which would otherwise skip it (same
        # class of problem as the MultiheadAttention in_proj below).
        if isinstance(module, FillerHubertSAModel) and getattr(module, "mask_embed", None) is not None:
            module.mask_embed.data.uniform_()
        if isinstance(module, RoPESelfAttention):
            # match the nn.MultiheadAttention init below (xavier proj, zero bias) so the
            # RoPE SA head starts from the same distribution as the sinusoidal/ALiBi heads.
            for lin in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
                nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            else:
                nn.init.xavier_uniform_(module.q_proj_weight)
                nn.init.xavier_uniform_(module.k_proj_weight)
                nn.init.xavier_uniform_(module.v_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    # ---- freezing helpers (the "FROZEN vs TRAINABLE" split in the diagram) ----
    def freeze_feature_extractor(self):
        """Freeze the conv feature extractor (always frozen)."""
        self.hubert.feature_extractor._freeze_parameters()

    def freeze_all_except_classifier(self):
        """Optional warm-up phase: train only lm_head."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = True

    def unfreeze_all_except_feature_extractor(self, freeze_feature_extractor=True,
                                              unfreeze_top_hubert=0):
        """Train the SA layers + lm_head; optionally also the TOP `unfreeze_top_hubert`
        HuBERT encoder layers (gradual unfreezing).

        unfreeze_top_hubert=0 (default) reproduces the original regime EXACTLY: the
        whole HuBERT body stays frozen.  >0 additionally unfreezes the last N encoder
        transformer layers (hubert.encoder.layers[-N:]) + the encoder's final
        LayerNorm, so the acoustic features can adapt to the char/left-pack task.
        The conv feature_extractor is ALWAYS frozen.
        """
        for p in self.parameters():
            p.requires_grad = False
        for p in self.hubert.parameters():
            p.requires_grad = False
        for p in self.sa.parameters():
            p.requires_grad = True
        for p in self.lm_head.parameters():
            p.requires_grad = True
        if getattr(self, "mask_embed", None) is not None:
            self.mask_embed.requires_grad = True
        if unfreeze_top_hubert and unfreeze_top_hubert > 0:
            enc = self.hubert.encoder
            n = min(int(unfreeze_top_hubert), len(enc.layers))
            for layer in enc.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(enc, "layer_norm"):                 # stable-layer-norm final LN
                for p in enc.layer_norm.parameters():
                    p.requires_grad = True
        # conv feature_extractor stays frozen regardless of the above
        self.hubert.feature_extractor._freeze_parameters()

    def train(self, mode: bool = True):
        """Keep the FROZEN HuBERT encoder in eval mode even under model.train().

        HuBERT is an inference-only feature extractor here (requires_grad=False), but the base
        nn.Module.train() would still flip it to train mode, activating layerdrop=0.1 (~5 of 48
        layers randomly skipped per forward) plus hidden/attention/feat-proj dropout=0.1. That
        makes the "frozen tap" non-deterministic and feeds the head a train-time feature
        distribution it never sees at eval. Pin the encoder to eval so the tap is deterministic
        and train==eval. (Audio-tap masking in the forward is gated on self.training, which stays
        True, so that intended augmentation is unaffected.)
        """
        super().train(mode)
        if self.hubert is not None:          # tap-cache training sets hubert=None (encoder unused)
            self.hubert.eval()
        return self

    def _sa_attn_mask(self, batch, seq_len, device, dtype):
        """Additive attn bias for the SA stack: symmetric ALiBi when pos_mode=='alibi', else
        None (mask=None -> byte-identical to the original self.sa(...) call in every other mode)."""
        if getattr(self, "pos_mode", "sinusoidal") != "alibi":
            return None
        return symmetric_alibi_mask(seq_len, self.sa_nhead, batch, device, dtype)

    def _run_sa(self, hidden_states, src_key_padding_mask):
        """Run the SA stack. Non-ALiBi modes call self.sa exactly as before (mask omitted).
        ALiBi mode adds the symmetric bias as attn_mask, with the MHA fast path disabled so
        padded batch elements don't go NaN at eval (see _mha_fastpath_disabled)."""
        attn_mask = self._sa_attn_mask(hidden_states.shape[0], hidden_states.shape[1],
                                       hidden_states.device, hidden_states.dtype)
        if attn_mask is None:
            return self.sa(hidden_states, src_key_padding_mask=src_key_padding_mask)
        # ALiBi supplies a FLOAT attn_mask. Cast the (bool) padding mask to the SAME float type
        # so PyTorch doesn't warn about — and eventually drop support for — a mixed float/bool
        # mask pair. Padded keys -> -inf (ignored by softmax); numerically identical to the bool
        # path, just future-proof and warning-free.
        if src_key_padding_mask is not None and src_key_padding_mask.dtype == torch.bool:
            src_key_padding_mask = torch.zeros_like(
                src_key_padding_mask, dtype=attn_mask.dtype
            ).masked_fill(src_key_padding_mask, float("-inf"))
        with _mha_fastpath_disabled():
            return self.sa(hidden_states, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

    def forward(self, input_values, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, labels=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- (a) FROZEN HuBERT body → TAP last_hidden_state (B, T, 1280).
        outputs = self.hubert(
            input_values, attention_mask=attention_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # the embedding tap

        # --- (b) FRAME MASK: map the sample-level attention_mask to a frame-level
        #         key-padding mask (True = pad → ignored inside MHSA).
        src_key_padding_mask = None
        feat_mask = None
        if attention_mask is not None:
            feat_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask)
            src_key_padding_mask = ~feat_mask.bool()

        # --- (c) INPUT MASKING (train only): replace ~sa_mask_prob of the valid frames with
        #         the learned mask_embed, in contiguous spans of sa_mask_length frames.
        #         Done BEFORE the PE add so masked slots keep position but lose content.
        if self.training and self.sa_mask_prob > 0:
            B, T, _ = hidden_states.shape
            p = self.sa_mask_prob
            if self.sa_mask_prob_max > p:            # per-batch rate ~ U[prob, prob_max]
                p = float(torch.empty(1).uniform_(p, self.sa_mask_prob_max))
            mask_idx = _compute_mask_indices((B, T), mask_prob=p, mask_length=self.sa_mask_length,
                                             attention_mask=feat_mask, min_masks=1)
            mask_idx = torch.from_numpy(mask_idx).to(hidden_states.device)
            hidden_states = hidden_states.clone()
            hidden_states[mask_idx] = self.mask_embed.to(hidden_states.dtype)

        # --- (d) ⊕ add fixed sinusoidal PE (BLOCK 1). Gated by config.use_sinusoidal_pe.
        #         Skipped under pos_mode=="alibi" (ALiBi supplies position as an attn bias instead).
        if self.use_sinusoidal_pe and self.pos_mode not in ("alibi", "rope"):
            B, T, dim = hidden_states.shape
            pe = sinusoidal_positional_encoding(T, dim, hidden_states.device, hidden_states.dtype)
            hidden_states = hidden_states + self.pe_scale * pe.unsqueeze(0)

        # --- (e) post-norm self-attention (residuals + LayerNorms are inside nn.TransformerEncoder).
        #         ALiBi (pos_mode=="alibi") enters as an additive attn bias; else unchanged.
        hidden_states = self._run_sa(hidden_states, src_key_padding_mask)

        # --- (f) final dropout → lm_head → per-frame logits (B, T, 33).
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]
        return CausalLMOutput(loss=None, logits=logits,
                              hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# ============================================================================
# BLOCK 3 — tokenizer (33-token char vocab incl. <fill>)
# ============================================================================
def build_tokenizer(vocab_dir):
    """Build the 33-token char tokenizer from a dir holding vocab.json (+ <fill> as an added token)."""
    tok = Wav2Vec2CTCTokenizer(
        os.path.join(vocab_dir, "vocab.json"),
        unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
        bos_token="<s>", eos_token="</s>",
    )
    tok.add_special_tokens({"additional_special_tokens": [FILL_TOKEN]})
    assert len(tok) == 33 and tok.convert_tokens_to_ids(FILL_TOKEN) == 32
    return tok


# ============================================================================
# BLOCK 4 — DATA / LABEL PREP  (the tan boxes) + both maskings
# ============================================================================
class DataCollatorFillerASR:
    """Loads audio, builds the positional <fill> target, and applies the duration-budget loss mask.

    Produces per batch:  input_values, attention_mask  (model inputs)
                         labels                         (per-frame targets; -100 = ignored in loss)
    """

    def __init__(self, feature_extractor, tokenizer, frames_per_sec=FRAMES_PER_SEC,
                 eos_repeat=1):
        self.fe = feature_extractor
        self.tok = tokenizer
        self.fill_id = tokenizer.convert_tokens_to_ids(FILL_TOKEN)
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.frames_per_sec = frames_per_sec        # duration budget (frames/sec graded); default = FRAMES_PER_SEC
        # EXPERIMENT: emit </s> on `eos_repeat` consecutive frames at the end of the
        # transcript instead of 1.  Reinforces the content→<fill> boundary (more </s>
        # gradient) and counteracts the boundary-fuzzing of fill down-weighting.
        # eos_repeat=1 reproduces the original target byte-for-byte.
        self.eos_repeat = int(eos_repeat)

    @staticmethod
    def _normalize(text):
        """Reference normalization: drop punctuation, lowercase."""
        return re.sub(CHARS_TO_IGNORE_REGEX, "", text).lower()

    def _labels_for(self, text, T):
        """POSITIONAL <fill> target of length T (the diagram's "_labels_for [positional]" box).

        Lay the transcript at the HEAD of the frame axis and pad the tail with <fill>:
            [<s>] + char_ids + [</s>]×eos_repeat + <fill> × (T - len)
        Spaces become the word-delimiter '|'.  (Note: this pins char i to frame i — the
        non-acoustic placement we diagnosed; kept here because it IS the implemented design.)
        """
        text = self._normalize(text).replace(" ", "|")
        ids = self.tok(text).input_ids                      # raw char ids (no bos/eos)
        n_specials = 1 + self.eos_repeat                    # <s> + </s>×eos_repeat
        if len(ids) + n_specials > T:                       # truncate if transcript longer than frames
            ids = ids[: T - n_specials]
        labels = [self.bos_id] + ids + [self.eos_id] * self.eos_repeat
        labels = labels + [self.fill_id] * (T - len(labels))  # <fill> tail
        return labels

    def __call__(self, features):
        """features: list of {"audio_path": str, "text": str}."""
        input_features, label_features, keep = [], [], []
        for feat in features:
            # --- load + feature-extract audio (normalize) ---
            arr, sr = sf.read(feat["audio_path"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            assert sr == SAMPLING_RATE, f"expected 16kHz, got {sr}"
            iv = self.fe(arr, sampling_rate=SAMPLING_RATE).input_values[0]
            input_features.append({"input_values": iv})

            # --- positional <fill> target sized to the encoder frame count T ---
            T = compute_output_length(len(iv))
            label_features.append({"input_ids": self._labels_for(feat["text"], T)})

            # --- duration budget: number of graded frames for this clip ---
            keep.append(math.ceil(len(iv) / SAMPLING_RATE * self.frames_per_sec))

        # --- pad audio (gives attention_mask → used as FRAME MASK in the model) ---
        batch = self.fe.pad(input_features, padding=True, return_tensors="pt")
        # --- pad labels; pad positions → -100 (LOSS MASK part 1) ---
        labels_batch = self.tok.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # --- LOSS MASK part 2 (duration budget): ignore everything past n_keep frames ---
        for i, n_keep in enumerate(keep):
            labels[i, n_keep:] = -100

        batch["labels"] = labels
        return batch


# ============================================================================
# BLOCK 5 — LOSS:  framewise cross-entropy
# ============================================================================
def framewise_ce_loss(logits, labels, fill_id=32, fill_weight=None, reduction="mean"):
    """Per-frame CrossEntropyLoss; -100 frames (pad + beyond-budget) are ignored.

        L = mean over graded frames of  -log softmax(logits)[label]

    The implemented training uses UNWEIGHTED CE (fill_weight=None). Pass fill_weight<1 to
    down-weight the dominant <fill> class (optional; recommended if it collapses to <fill>).
    reduction="sum" returns the (weighted) SUM over graded frames instead of the mean -- used by
    the trainer to average EXACTLY over all tokens in a grad-accum x DDP window (see train.py).
    """
    V = logits.size(-1)
    weight = None
    if fill_weight is not None:
        weight = torch.ones(V, device=logits.device)
        weight[fill_id] = fill_weight
    loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100, reduction=reduction)
    return loss_fct(logits.reshape(-1, V), labels.reshape(-1))


# ============================================================================
# BLOCK 6 — model construction (config wiring, like train_local.py)
# ============================================================================
def build_model(model_checkpoint, tokenizer, num_sa_layers=4, sa_nhead=16,
                sa_dim_feedforward=5120, sa_dropout=0.1,
                use_sinusoidal_pe=True, pe_scale=1.0, device="cpu", unfreeze_top_hubert=0,
                mask_prob=0.0, mask_prob_max=0.0, mask_length=10, pos_mode="sinusoidal",
                rope_theta=10000.0, rope_orig_len=1024, rope_ntk_factor=1.0,
                rope_ntk_dynamic=True):
    """Load the frozen pretrained HuBERT-xlarge body + fresh SA layers + lm_head."""
    cfg = HubertConfig.from_pretrained(model_checkpoint)
    cfg.vocab_size = len(tokenizer)                                  # 33
    cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id = (
        tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    cfg.num_sa_layers = num_sa_layers
    cfg.sa_nhead = sa_nhead
    cfg.sa_dim_feedforward = sa_dim_feedforward
    cfg.sa_dropout = sa_dropout
    cfg.pos_mode = pos_mode                                          # BLOCK 1b/1c: "sinusoidal" | "alibi" | "rope"
    cfg.rope_theta = rope_theta                                      # BLOCK 1c: RoPE base + (dynamic) NTK scaling
    cfg.rope_orig_len = rope_orig_len
    cfg.rope_ntk_factor = rope_ntk_factor
    cfg.rope_ntk_dynamic = rope_ntk_dynamic
    cfg.use_sinusoidal_pe = use_sinusoidal_pe                        # BLOCK 1 switch
    cfg.pe_scale = pe_scale
    cfg.sa_mask_prob = mask_prob                                     # train-time input masking (0 = off)
    cfg.sa_mask_prob_max = mask_prob_max
    cfg.sa_mask_length = mask_length
    cfg.apply_spec_augment = False                                  # deterministic
    model = FillerHubertSAModel.from_pretrained(
        model_checkpoint, config=cfg, ignore_mismatched_sizes=True)  # lm_head reinit (32→33)
    model.freeze_feature_extractor()
    model.unfreeze_all_except_feature_extractor(unfreeze_top_hubert=unfreeze_top_hubert)  # SA + lm_head (+ top-N HuBERT if >0)
    return model.to(device)


# ============================================================================
# BLOCK 6b — A-CMLM: audio-conditioned masked-LM head (ADDITION fusion, no gamma)
# ============================================================================
# Same recipe as BLOCK 2/6 (frozen HuBERT tap, 20-30% audio-tap masking, PE, 8 SA,
# lm_head->33) PLUS a small text embedding fused by ADDITION:
#       h = tap (+audio-mask) + PE + E_text(text_input_ids)  ->  SA -> lm_head
# <mask> is an INPUT-only id (= vocab_size = 33); it is never predicted (lm_head stays 33).
MASK_TOKEN = "<mask>"


def build_acmlm_tokenizer(vocab_dir):
    """33-token filler tokenizer + <mask> as an added token -> 34 tokens, <mask>=33."""
    tok = build_tokenizer(vocab_dir)
    tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    assert len(tok) == 34 and tok.convert_tokens_to_ids(MASK_TOKEN) == 33
    return tok


class FillerHubertACMLM(FillerHubertSAModel):
    """FillerHubertSAModel + a text-embedding input, fused by ADDITION (no gamma, no extra LN).

    Small-init text embedding means step 0 ~ the audio-only parent model, then the text
    pathway grows during training.  encode_audio()/decode_from_tap() let the iterative
    decoder cache the frozen 1B-param HuBERT tap ONCE instead of recomputing it N times.
    """

    def __init__(self, config):
        super().__init__(config)
        self.mask_token_id = config.vocab_size                        # 33 = <mask> input id
        self.text_embed = nn.Embedding(config.vocab_size + 1, config.hidden_size)   # 34 x 1280
        nn.init.normal_(self.text_embed.weight, mean=0.0, std=0.02)   # small: start ~ audio-only

    def _align_text(self, ids, T):
        """Pad(<pad>)/crop text_input_ids to the encoder frame count T (handles off-by-one)."""
        B, L = ids.shape
        if L == T:
            return ids
        if L > T:
            return ids[:, :T]
        return torch.cat([ids, ids.new_full((B, T - L), self.config.pad_token_id)], dim=1)

    # ---- audio path (tap + audio-mask + PE): cache this ONCE at inference ----
    def encode_audio(self, input_values, attention_mask=None):
        outputs = self.hubert(input_values, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs[0]
        src_key_padding_mask = feat_mask = None
        if attention_mask is not None:
            feat_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask)
            src_key_padding_mask = ~feat_mask.bool()
        hidden_states = self._span_mask_and_pe(hidden_states, feat_mask)
        return hidden_states, src_key_padding_mask

    # ---- shared post-tap processing: train-only span-mask + fixed PE.
    #      Identical whether the tap came from HuBERT live (encode_audio) or the cache
    #      (encode_from_tap) -> the two paths are numerically equivalent downstream.
    def _span_mask_and_pe(self, hidden_states, feat_mask):
        if self.training and self.sa_mask_prob > 0:                    # 20-30% audio-tap span mask
            B, T, _ = hidden_states.shape
            p = self.sa_mask_prob
            if self.sa_mask_prob_max > p:
                p = float(torch.empty(1).uniform_(p, self.sa_mask_prob_max))
            mask_idx = _compute_mask_indices((B, T), mask_prob=p, mask_length=self.sa_mask_length,
                                             attention_mask=feat_mask, min_masks=1)
            mask_idx = torch.from_numpy(mask_idx).to(hidden_states.device)
            hidden_states = hidden_states.clone()
            hidden_states[mask_idx] = self.mask_embed.to(hidden_states.dtype)
        if self.use_sinusoidal_pe and self.pos_mode not in ("alibi", "rope"):  # + fixed sinusoidal PE (skip under ALiBi/RoPE)
            B, T, dim = hidden_states.shape
            pe = sinusoidal_positional_encoding(T, dim, hidden_states.device, hidden_states.dtype)
            hidden_states = hidden_states + self.pe_scale * pe.unsqueeze(0)
        return hidden_states

    # ==== TAP CACHE support: precompute the raw frozen tap once, train from it ====
    @torch.no_grad()
    def hubert_tap(self, input_values, attention_mask=None):
        """RAW frozen HuBERT last_hidden_state (B,T,1280) + per-clip valid frame counts.
        Deterministic (encoder pinned to eval by train()) -> compute ONCE, cache, reuse every
        epoch. This is exactly the tensor encode_audio() feeds into span-mask + PE."""
        out = self.hubert(input_values, attention_mask=attention_mask, return_dict=True)
        hs = out[0]
        feat_len = None
        if attention_mask is not None:
            fm = self.hubert._get_feature_vector_attention_mask(hs.shape[1], attention_mask)
            feat_len = fm.long().sum(-1)
        return hs, feat_len

    def encode_from_tap(self, raw_tap, src_key_padding_mask=None):
        """Cached-tap twin of encode_audio(): span-mask (train only) + PE on a PRECOMPUTED tap.
        Uses the SAME _span_mask_and_pe() as the live path, so it is numerically identical
        downstream of HuBERT."""
        feat_mask = (~src_key_padding_mask) if src_key_padding_mask is not None else None
        hidden_states = self._span_mask_and_pe(raw_tap, feat_mask)
        return hidden_states, src_key_padding_mask

    # ---- text path + SA + head: re-run every decode step (tap stays cached) ----
    def decode_from_tap(self, tap_pe, text_input_ids, src_key_padding_mask=None):
        text_input_ids = self._align_text(text_input_ids, tap_pe.shape[1])
        h = tap_pe + self.text_embed(text_input_ids)                  # ADDITION fusion
        h = self._run_sa(h, src_key_padding_mask)                     # ALiBi bias (opt-in) or unchanged
        logits = self.lm_head(self.dropout(h))
        return CausalLMOutput(loss=None, logits=logits, hidden_states=None, attentions=None)

    def forward(self, input_values=None, attention_mask=None, text_input_ids=None,
                labels=None, tap=None, pad_mask=None, return_dict=None, **kwargs):
        if tap is not None:                                           # cached-tap path (no HuBERT)
            tap_pe, skpm = self.encode_from_tap(tap, pad_mask)
        else:
            tap_pe, skpm = self.encode_audio(input_values, attention_mask)
        if text_input_ids is None:                                    # default: 100% masked
            B, T = tap_pe.shape[:2]
            text_input_ids = torch.full((B, T), self.mask_token_id,
                                        dtype=torch.long, device=tap_pe.device)
        return self.decode_from_tap(tap_pe, text_input_ids, skpm)


def build_acmlm_model(model_checkpoint, tokenizer, num_sa_layers=8, sa_nhead=16,
                      sa_dim_feedforward=5120, sa_dropout=0.1, use_sinusoidal_pe=True,
                      pe_scale=1.0, device="cpu", mask_prob=0.20, mask_prob_max=0.30,
                      mask_length=10, pos_mode="sinusoidal",
                      rope_theta=10000.0, rope_orig_len=1024, rope_ntk_factor=1.0,
                      rope_ntk_dynamic=True):
    """Frozen HuBERT + text embedding + SA + lm_head.  mask_prob defaults KEEP the
    20-30% audio-tap masking enabled (as in the parent run)."""
    cfg = HubertConfig.from_pretrained(model_checkpoint)
    cfg.vocab_size = len(tokenizer) - 1                               # 33 real classes (<mask> excluded)
    cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id = (
        tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    cfg.num_sa_layers = num_sa_layers
    cfg.sa_nhead = sa_nhead
    cfg.sa_dim_feedforward = sa_dim_feedforward
    cfg.sa_dropout = sa_dropout
    cfg.pos_mode = pos_mode                                           # BLOCK 1b/1c: "sinusoidal" | "alibi" | "rope"
    cfg.rope_theta = rope_theta                                       # BLOCK 1c: RoPE base + (dynamic) NTK scaling
    cfg.rope_orig_len = rope_orig_len
    cfg.rope_ntk_factor = rope_ntk_factor
    cfg.rope_ntk_dynamic = rope_ntk_dynamic
    cfg.use_sinusoidal_pe = use_sinusoidal_pe
    cfg.pe_scale = pe_scale
    cfg.sa_mask_prob = mask_prob
    cfg.sa_mask_prob_max = mask_prob_max
    cfg.sa_mask_length = mask_length
    cfg.apply_spec_augment = False
    model = FillerHubertACMLM.from_pretrained(model_checkpoint, config=cfg,
                                              ignore_mismatched_sizes=True)
    # --- CRITICAL: re-init text_embed on the REAL device after from_pretrained. ---
    # text_embed is a "missing key" (not in the HuBERT checkpoint). Under transformers 5.x
    # from_pretrained builds on the meta device, so the nn.init.normal_ in __init__ runs on a
    # meta tensor (no-op) and _init_weights does NOT cover this Embedding -> the weight is left
    # as raw torch.empty memory (seen as zeros / FLT_MAX / NaN across launches). When it lands on
    # NaN the whole model is poisoned from step 0 (NaN loss, ~0 acc, in bf16 AND fp32). Force a
    # proper small init here, where the tensor is materialized on a real device.
    with torch.no_grad():
        nn.init.normal_(model.text_embed.weight, mean=0.0, std=0.02)
    model.freeze_feature_extractor()
    model.unfreeze_all_except_feature_extractor()                     # SA + lm_head + mask_embed
    for p in model.text_embed.parameters():                          # + the text embedding
        p.requires_grad = True
    # fail loud instead of silently NaN-ing if any trainable param is still non-finite
    bad = [n for n, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
    assert not bad, f"non-finite trainable params after init: {bad}"
    return model.to(device)


class DataCollatorACMLM(DataCollatorFillerASR):
    """Like DataCollatorFillerASR, but ALSO masks the text label (CMLM) and returns it.

    Per clip:  build y (unchanged) ; n_keep = ceil(dur*fps) ; sample p ~ U(0,1) (or fixed_p) ;
    mask a Bernoulli(p) subset of positions 1..n_keep-1 (never <s>=0) ->
        text_input_ids : <mask> at masked, y elsewhere ; tail (>= n_keep) = fixed <fill>
        labels         : y at masked, -100 elsewhere      (loss = framewise CE on masked slots)
    """

    def __init__(self, feature_extractor, tokenizer, frames_per_sec=FRAMES_PER_SEC,
                 eos_repeat=1, force_full_mask_prob=0.15, fixed_p=None):
        super().__init__(feature_extractor, tokenizer, frames_per_sec=frames_per_sec,
                         eos_repeat=eos_repeat)
        self.mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        self.pad_id = tokenizer.pad_token_id
        self.force_full_mask_prob = force_full_mask_prob              # fraction of clips forced p=1.0
        self.fixed_p = fixed_p                                        # None=U(0,1); set 1.0 for eval

    def __call__(self, features):
        input_features, text_list, label_list = [], [], []
        for feat in features:
            arr, sr = sf.read(feat["audio_path"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            assert sr == SAMPLING_RATE, f"expected 16kHz, got {sr}"
            iv = self.fe(arr, sampling_rate=SAMPLING_RATE).input_values[0]
            input_features.append({"input_values": iv})

            T = compute_output_length(len(iv))
            y = torch.tensor(self._labels_for(feat["text"], T), dtype=torch.long)
            n_keep = min(T, math.ceil(len(iv) / SAMPLING_RATE * self.frames_per_sec))

            text_in = y.clone()
            labels = torch.full((T,), -100, dtype=torch.long)
            if n_keep > 1:
                p = self.fixed_p if self.fixed_p is not None else (
                    1.0 if torch.rand(1).item() < self.force_full_mask_prob else torch.rand(1).item())
                region = torch.arange(1, n_keep)
                sel = region[torch.rand(region.numel()) < p]
                if sel.numel() == 0:                                 # guarantee >=1 masked slot
                    sel = region[torch.randint(region.numel(), (1,))]
                text_in[sel] = self.mask_id
                labels[sel] = y[sel]
            text_list.append(text_in)
            label_list.append(labels)

        batch = self.fe.pad(input_features, padding=True, return_tensors="pt")
        maxT = max(t.numel() for t in text_list)
        text_ids = torch.full((len(text_list), maxT), self.pad_id, dtype=torch.long)
        labels = torch.full((len(label_list), maxT), -100, dtype=torch.long)
        for i, (t, l) in enumerate(zip(text_list, label_list)):
            text_ids[i, :t.numel()] = t
            labels[i, :l.numel()] = l
        batch["text_input_ids"] = text_ids
        batch["labels"] = labels
        return batch


# ============================================================================
# BLOCK 7 — smoke test: one forward + loss on a few clips (no training)
# ============================================================================
def _smoke(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer(args.vocab_dir)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    rows = [json.loads(l) for l in open(args.manifest)][: args.n]
    rows = [{"audio_path": r.get("audio") or r.get("source"),
             "text": r.get("text") or r.get("target")} for r in rows]

    collator = DataCollatorFillerASR(fe, tok)
    batch = collator(rows)
    labels = batch.pop("labels").to(dev)
    batch = {k: v.to(dev) for k, v in batch.items()}

    model = build_model(args.model_checkpoint, tok, use_sinusoidal_pe=args.pe, device=dev).train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logits = model(**batch).logits
    loss = framewise_ce_loss(logits, labels, fill_id=tok.convert_tokens_to_ids(FILL_TOKEN),
                             fill_weight=args.fill_weight)
    graded = (labels != -100)
    print(f"clips={len(rows)}  PE={args.pe}  trainable={n_train/1e6:.2f}M")
    print(f"input_values={tuple(batch['input_values'].shape)}  logits={tuple(logits.shape)}  "
          f"labels={tuple(labels.shape)}  graded_frames={int(graded.sum())}")
    print(f"framewise CE loss = {loss.item():.4f}   (random-init reference ≈ ln(33) = {math.log(33):.4f})")


# ============================================================================
# BLOCK 8 — overfit driver: train N clips for E epochs (full-batch) to see if it learns
# ============================================================================
def _decode(ids, tok):
    """Argmax frame ids → transcript (strip specials/<fill>, '|'→space)."""
    specials = {tok.pad_token_id, tok.bos_token_id, tok.eos_token_id,
                tok.convert_tokens_to_ids(FILL_TOKEN), tok.unk_token_id}
    return "".join(tok.convert_ids_to_tokens(int(i)) for i in ids
                   if int(i) not in specials).replace("|", " ").strip()


def _train(args):
    """Full-batch overfit of N clips for E epochs. If the architecture/target are learnable,
    loss → ~0, graded-frame-acc → ~1, and the clips are reproduced. Otherwise it plateaus."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer(args.vocab_dir)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)

    # --- take the first N clips, build one full batch (positional <fill> targets + masks) ---
    rows = [json.loads(l) for l in open(args.train_manifest) if l.strip()][: args.n]
    rows = [{"audio_path": r.get("audio") or r.get("source"),
             "text": r.get("text") or r.get("target")} for r in rows]
    batch = DataCollatorFillerASR(fe, tok)(rows)
    labels = batch.pop("labels").to(dev)
    batch = {k: v.to(dev) for k, v in batch.items()}

    # --- model: frozen HuBERT + PE + 2 SA layers + lm_head (BLOCK 6) ---
    model = build_model(args.model_checkpoint, tok, use_sinusoidal_pe=args.pe, device=dev).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="filler_asr",
                         name=f"reference_overfit{args.n}_{args.epochs}ep_{'PE' if args.pe else 'noPE'}",
                         config={"n": args.n, "epochs": args.epochs, "lr": args.lr, "pe": args.pe,
                                 "fill_weight": args.fill_weight, "trainable_M": round(n_train/1e6, 2)})
        print("wandb run:", wandb.run.url, flush=True)
    print(f"device={dev}  clips={len(rows)}  PE={args.pe}  trainable={n_train/1e6:.2f}M  "
          f"lr={args.lr}  epochs={args.epochs}", flush=True)

    # --- optimize ONLY the SA layers + lm_head with framewise CE (BLOCK 5) ---
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.005)
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        m = min(logits.shape[1], labels.shape[1])
        loss = framewise_ce_loss(logits[:, :m], labels[:, :m], fill_id=fill_id, fill_weight=args.fill_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        with torch.no_grad():
            graded = labels[:, :m] != -100
            acc = ((logits[:, :m].argmax(-1) == labels[:, :m]) & graded).sum().item() / max(1, graded.sum().item())
        if run:
            run.log({"train/loss": loss.item(), "train/graded_frame_acc": acc, "epoch": epoch}, step=epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.4f}  graded-frame-acc={acc:.3f}", flush=True)

    # --- reproduction check: does it spit back the (graded) transcripts? ---
    model.eval()
    with torch.no_grad():
        pred = model(**batch).logits.argmax(-1)
    n_ok = 0
    print("\n--- reproduction (graded region) ---", flush=True)
    for i in range(len(rows)):
        gold = _decode(labels[i][labels[i] != -100], tok)
        keep = (labels[i] != -100).nonzero(as_tuple=True)[0]
        hyp = _decode(pred[i][keep], tok)
        ok = hyp == gold; n_ok += int(ok)
        print(f"[{i}] gold: {gold[:70]}\n    hyp : {hyp[:70]}   {'OK' if ok else 'MISMATCH'}", flush=True)
    print(f"\nreproduced {n_ok}/{len(rows)} clips   (final loss={loss.item():.4f}, acc={acc:.3f})", flush=True)
    if run:
        run.summary["reproduced"] = n_ok
        run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="filler_asr SA reference — build, smoke, or overfit-train")
    ap.add_argument("--model_checkpoint", default="/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft")
    ap.add_argument("--vocab_dir", default="/speech/tomson/filler_asr/experiments/Hubert_SA_2gpu_lazy_lr2e3")
    ap.add_argument("--manifest", default="/speech/tomson/filler_asr/data/test_clean.jsonl")
    ap.add_argument("--train_manifest",
                    default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pe", action="store_true", help="enable sinusoidal PE")
    ap.add_argument("--fill_weight", type=float, default=None, help="down-weight <fill> in CE (optional)")
    ap.add_argument("--train", action="store_true", help="run the overfit trainer (BLOCK 8) instead of smoke")
    ap.add_argument("--wandb", action="store_true", help="log curves to wandb")
    args = ap.parse_args()
    (_train if args.train else _smoke)(args)
