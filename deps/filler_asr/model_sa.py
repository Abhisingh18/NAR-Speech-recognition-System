"""
Self-attention variant of the filler_asr model (professor's design).

  frozen HuBERT encoder (CTC-finetuned, CTC head dropped)
    -> tap last_hidden_state (B, T, hidden)
    -> 2 NEW self-attention layers      (trainable)
    -> NEW lm_head                       (trainable)
  trained with framewise cross-entropy on the "filler" target.

The HuBERT encoder is used as a FROZEN feature extractor; only the new
self-attention layers + lm_head learn. Reuses FillerASRTrainer (triphase LR,
freezing curriculum, framewise CE) from model.py unchanged.

Configurable via these (optional) attributes on the HubertConfig:
  num_sa_layers      (default 4)
  sa_nhead           (default config.num_attention_heads, e.g. 16)
  sa_dim_feedforward (default config.intermediate_size, e.g. 5120)
  sa_dropout         (default config.final_dropout)
  use_sinusoidal_pe  (default False)  -- add a fixed sinusoidal positional
                     encoding to the tapped features BEFORE the new SA layers.
                     plain nn.TransformerEncoderLayer carries no positional info
                     and HuBERT's own pos-conv is frozen, so without this the new
                     SA layers are permutation-equivariant over time. Gated + off
                     by default so existing (no-PE) checkpoints load identically.
  pe_scale           (default 1.0) -- multiplier on the PE before adding; lets you
                     match the PE magnitude to the HuBERT hidden-state scale.
"""
import math

import torch
import torch.nn as nn
from transformers import HubertPreTrainedModel, HubertModel
from transformers.modeling_outputs import CausalLMOutput


def sinusoidal_positional_encoding(seq_len, dim, device, dtype):
    """Standard Transformer (Vaswani et al.) fixed sinusoidal PE: (seq_len, dim).
    PE[p, 2i]=sin(p / 10000^(2i/dim)),  PE[p, 2i+1]=cos(p / 10000^(2i/dim))."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)   # (L,1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))                                  # (dim/2,)
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class FillerHubertSAModel(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hubert = HubertModel(config)

        n_layers = getattr(config, "num_sa_layers", 4)
        nhead = getattr(config, "sa_nhead", config.num_attention_heads)
        dim_ff = getattr(config, "sa_dim_feedforward", config.intermediate_size)
        sa_dropout = getattr(config, "sa_dropout", config.final_dropout)
        self.use_sinusoidal_pe = getattr(config, "use_sinusoidal_pe", False)
        self.pe_scale = getattr(config, "pe_scale", 1.0)

        sa_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=sa_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,          # post-norm, like HuBERT's encoder layers
        )
        self.sa = nn.TransformerEncoder(
            sa_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def _init_weights(self, module):
        # HuBERT's _init_weights handles Linear/LayerNorm/Conv but NOT
        # nn.MultiheadAttention's raw in_proj params -> they'd stay uninitialized
        # (NaN) after from_pretrained. Initialize them here.
        super()._init_weights(module)
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

    # ------- freezing helpers (called by FillerASRTrainer) -------
    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()

    def freeze_all_except_classifier(self):
        """Phase A: warm up the new head only."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = True

    def unfreeze_all_except_feature_extractor(self, freeze_feature_extractor=True):
        """SA variant: the WHOLE HuBERT encoder stays frozen (feature extractor);
        train only the new self-attention layers + lm_head.
        (Name kept for FillerASRTrainer compatibility.)"""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.hubert.parameters():
            p.requires_grad = False
        for p in self.sa.parameters():
            p.requires_grad = True
        for p in self.lm_head.parameters():
            p.requires_grad = True

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (B, T, hidden)  -- the embedding tap

        # fixed sinusoidal positional encoding added to the tap BEFORE the new SA
        # layers (plain nn.TransformerEncoderLayer has none of its own). Off unless
        # config.use_sinusoidal_pe is set, so no-PE checkpoints are unaffected.
        if self.use_sinusoidal_pe:
            B, T, dim = hidden_states.shape
            pe = sinusoidal_positional_encoding(T, dim, hidden_states.device, hidden_states.dtype)
            hidden_states = hidden_states + self.pe_scale * pe.unsqueeze(0)

        # frame-level padding mask for the new self-attention (True = pad -> ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            feat_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            src_key_padding_mask = ~feat_mask.bool()

        hidden_states = self.sa(hidden_states, src_key_padding_mask=src_key_padding_mask)
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutput(
            loss=None,  # loss computed in FillerASRTrainer
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
