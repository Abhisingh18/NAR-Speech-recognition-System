"""Render the A-CMLM (addition fusion) pipeline: collator+masking, train fwd/bwd, iterative infer.

Outputs: omni_style_acmlm_pipeline.png   (run: python make_omni_acmlm_diagram.py)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# palette
C_FROZEN = "#cfd8dc"   # frozen / no-grad
C_TRAIN  = "#bbdefb"   # trainable
C_DATA   = "#fff3c4"   # data / tensors
C_LOSS   = "#f8bbd0"   # loss
C_INFER  = "#c8e6c9"   # inference
C_NOTE   = "#eceff1"
EDGE     = "#455a64"


def box(ax, x, y, w, h, text, fc, fs=9, bold=False, ec=EDGE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=5)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.6, style="-", ls="solid"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style + "|>",
                                 mutation_scale=14, color=color, lw=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2, zorder=4))


def title(ax, t):
    ax.text(0.012, 0.955, t, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="left", color="#263238")


fig, axes = plt.subplots(3, 1, figsize=(15.5, 20.5))
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# =====================================================================================
# PANEL A — COLLATOR + MASKING (data prep)
# =====================================================================================
ax = axes[0]
title(ax, "A.  COLLATOR + MASKING   (per clip, at batch time — no caching)")

box(ax, 0.02, 0.72, 0.30, 0.16,
    'JSONL manifest row\n{"source": ".../x.wav",\n "target": "CAT"}', C_DATA, fs=9)
arrow(ax, 0.32, 0.80, 0.37, 0.80)

box(ax, 0.37, 0.80, 0.28, 0.14,
    "read wav 16 kHz mono\nWav2Vec2FeatureExtractor\n-> input_values, attention_mask", C_DATA, fs=8.5)
box(ax, 0.37, 0.62, 0.28, 0.14,
    "T = compute_output_length(len)\n(HuBERT frame count = full\nlength budget; no dur. model)", C_NOTE, fs=8.5)
arrow(ax, 0.51, 0.80, 0.51, 0.76)

box(ax, 0.68, 0.72, 0.30, 0.22,
    "framewise label  y  (len T)\n"
    "y = [<s>] + char_ids\n"
    "        + [</s>]x3\n"
    "        + [<fill>] x (T - len)", C_DATA, fs=9)
arrow(ax, 0.65, 0.83, 0.68, 0.83)

# masking box
box(ax, 0.68, 0.40, 0.30, 0.24,
    "MASK SAMPLING (CMLM)\n"
    "p ~ U(0,1)   (+~15% forced p=1)\n"
    "mask each pos (except idx0=<s>)\n"
    "i.i.d. Bernoulli(p)\n"
    "grade FULL length, fill_weight=0.1", C_TRAIN, fs=8.5)
arrow(ax, 0.83, 0.72, 0.83, 0.64)

# concrete example strip
ax.text(0.02, 0.545, "example  target = \"cat\",  T = 12,  p = 0.5,  masked = {2,4,7,9,10}",
        fontsize=9, fontweight="bold", color="#263238")
rows = [
    ("y  (target)      ", ["^", "c", "a", "t", "$", "$", "$", ".", ".", ".", ".", "."], "#fff3c4"),
    ("text_input_ids   ", ["^", "c", "M", "t", "M", "$", "$", "M", ".", "M", "M", "."], "#bbdefb"),
    ("labels (loss)    ", ["-", "-", "a", "-", "$", "-", "-", ".", "-", ".", ".", "-"], "#f8bbd0"),
]
x0, y0, cw = 0.20, 0.47, 0.045
for r, (lab, cells, fc) in enumerate(rows):
    yy = y0 - r * 0.075
    ax.text(0.02, yy + 0.025, lab, fontsize=8.6, family="monospace", va="center")
    for c, ch in enumerate(cells):
        box(ax, x0 + c * cw, yy, cw * 0.9, 0.05, ch, fc, fs=8.5)
ax.text(0.20, 0.135, "legend:  ^=<s>   $=</s>   .=<fill>   M=<mask>(=33, input only)   "
        "-=ignored(-100)   loss only on masked slots", fontsize=8.3, color="#37474f")
box(ax, 0.02, 0.04, 0.96, 0.055,
    "OUTPUTS per batch:   input_values, attention_mask (audio)   +   text_input_ids (T)   +   labels (T)",
    C_NOTE, fs=9, bold=True)

# =====================================================================================
# PANEL B — TRAINING forward + backward
# =====================================================================================
ax = axes[1]
title(ax, "B.  TRAINING   (audio-conditioned masked-LM;  forward -> loss -> backward)")

# inputs
box(ax, 0.02, 0.78, 0.16, 0.10, "input_values\nattention_mask", C_DATA, fs=8.5)
box(ax, 0.02, 0.50, 0.16, 0.10, "text_input_ids\n(masked y)", C_DATA, fs=8.5)

# frozen encoder
box(ax, 0.22, 0.76, 0.20, 0.14, "FROZEN  HuBERT-xlarge\n(requires_grad=False)\n[snowflake]", C_FROZEN, fs=8.7, bold=True)
arrow(ax, 0.18, 0.83, 0.22, 0.83)
box(ax, 0.46, 0.77, 0.13, 0.12, "tap\n(B,T,1280)", C_DATA, fs=8.7)
arrow(ax, 0.42, 0.83, 0.46, 0.83)

# text embedding
box(ax, 0.22, 0.49, 0.20, 0.12, "E_text : Embedding(34,1280)\n(TRAINABLE)", C_TRAIN, fs=8.5, bold=True)
arrow(ax, 0.18, 0.55, 0.22, 0.55)
box(ax, 0.46, 0.50, 0.13, 0.10, "gamma *\n(learn scalar)", C_TRAIN, fs=8.5)
arrow(ax, 0.42, 0.55, 0.46, 0.55)

# fusion (addition)
box(ax, 0.63, 0.63, 0.15, 0.14, "(+)  ADD\ntap + gamma*E_text\n+ sinusoidal PE", C_TRAIN, fs=8.5, bold=True)
arrow(ax, 0.59, 0.83, 0.705, 0.77)     # tap -> add
arrow(ax, 0.59, 0.55, 0.705, 0.63)     # gamma*E_text -> add

box(ax, 0.63, 0.50, 0.15, 0.09, "LayerNorm\n(input norm)", C_TRAIN, fs=8.5)
arrow(ax, 0.705, 0.63, 0.705, 0.59)
box(ax, 0.63, 0.36, 0.15, 0.10, "8x post-norm\nSelf-Attention\n(TRAINABLE)", C_TRAIN, fs=8.5, bold=True)
arrow(ax, 0.705, 0.50, 0.705, 0.46)
box(ax, 0.82, 0.36, 0.15, 0.10, "dropout ->\nlm_head\n(1280 -> 33)", C_TRAIN, fs=8.5)
arrow(ax, 0.78, 0.41, 0.82, 0.41)
box(ax, 0.82, 0.52, 0.15, 0.09, "logits\n(B,T,33)", C_DATA, fs=8.7)
arrow(ax, 0.895, 0.46, 0.895, 0.52)

# loss
box(ax, 0.60, 0.16, 0.30, 0.12,
    "framewise CrossEntropy  (MASKED slots only)\n"
    "labels=-100 elsewhere;  w(<fill>)=0.1, else 1", C_LOSS, fs=8.7, bold=True)
arrow(ax, 0.895, 0.52, 0.895, 0.42)
arrow(ax, 0.82, 0.22, 0.90, 0.22)               # logits path down via lm_head already; connect labels
box(ax, 0.02, 0.16, 0.16, 0.10, "labels (T)\n(from collator)", C_DATA, fs=8.5)
arrow(ax, 0.18, 0.21, 0.60, 0.21)
arrow(ax, 0.895, 0.36, 0.895, 0.28)             # logits -> loss

# backward (dashed red) to TRAINABLE only
for (x1, y1) in [(0.75, 0.16), (0.75, 0.16)]:
    pass
arrow(ax, 0.60, 0.19, 0.30, 0.40, color="#c62828", lw=1.5, ls="dashed")   # -> SA
arrow(ax, 0.60, 0.19, 0.30, 0.53, color="#c62828", lw=1.5, ls="dashed")   # -> E_text
arrow(ax, 0.60, 0.19, 0.885, 0.36, color="#c62828", lw=1.5, ls="dashed")  # -> lm_head
ax.text(0.31, 0.29, "backward (grad)\nonly to trainable:\nE_text, gamma, LN,\n8xSA, lm_head  (~39.5M)",
        fontsize=8.2, color="#c62828", va="center")
ax.text(0.24, 0.70, "NO grad to\nHuBERT (frozen 1B)", fontsize=8.2, color="#455a64", va="center")

box(ax, 0.02, 0.02, 0.96, 0.05,
    "AdamW lr 2e-4, warmup 5000, linear decay, wd 5e-3, clip 1.0, bf16 | 4-way DDP, eff batch 256, 100 ep, LS-960",
    C_NOTE, fs=8.7)

# =====================================================================================
# PANEL C — INFERENCE (OmniVoice iterative unmasking)
# =====================================================================================
ax = axes[2]
title(ax, "C.  INFERENCE   (OmniVoice iterative unmask;  no duration model, no layer penalty)")

box(ax, 0.02, 0.80, 0.17, 0.10, "audio", C_DATA, fs=9)
box(ax, 0.22, 0.78, 0.20, 0.13, "FROZEN HuBERT\n-> tap (1,T,1280)\ncomputed ONCE", C_FROZEN, fs=8.6, bold=True)
arrow(ax, 0.19, 0.85, 0.22, 0.85)
box(ax, 0.46, 0.79, 0.24, 0.12, "ids = [<s>, <mask> x (T-1)]\nmasked = all but idx0", C_INFER, fs=8.6)
arrow(ax, 0.42, 0.85, 0.46, 0.85)

# loop box
box(ax, 0.30, 0.34, 0.66, 0.36, "", "#f1f8e9", ec="#7cb342", lw=1.6)
ax.text(0.315, 0.665, "for n = 1 .. N   (N=32):", fontsize=10, fontweight="bold", color="#33691e")
steps = [
    "r_n = tau*(n/N) / (1 + (tau-1)*(n/N))          # schedule (tau=0.1, back-loaded)",
    "k_n = round(r_n*M) - round(r_{n-1}*M)          # NEW commits this step",
    "logp = log_softmax( Model(tap, ids) )          # re-reads WHOLE grid",
    "conf = max_v logp ;  pred = argmax_v logp      # value = greedy",
    "conf[pred==<fill>] -= fill_penalty             # optional",
    "sel = gumbel_top_k( conf[masked]/temp , k_n )  # temp-sampled ORDER (temp=5)",
    "ids[sel] = pred[sel]                           # COMMIT (permanent)",
]
for i, s in enumerate(steps):
    ax.text(0.325, 0.625 - i * 0.040, s, fontsize=8.3, family="monospace", color="#1b5e20")
arrow(ax, 0.58, 0.79, 0.63, 0.70)
arrow(ax, 0.63, 0.55, 0.42, 0.79, color="#7cb342", ls="dashed", lw=1.4)   # loop back
ax.text(0.235, 0.60, "each step\nfeeds committed\nids back in", fontsize=7.8, color="#558b2f", va="center")

# readout
box(ax, 0.30, 0.20, 0.66, 0.09,
    "READOUT:  ids -> symbols ;  cut at first </s> ;  drop <fill>/<pad> ;  '|' -> space   ->  \"cat\"",
    C_INFER, fs=9, bold=True)
arrow(ax, 0.63, 0.34, 0.63, 0.29)

# commit trace strip (back-loaded: fill/boundary first)
ax.text(0.02, 0.145, "commit trace ('·'=masked):", fontsize=8.3, fontweight="bold")
trace = [
    ("start ", ["^", "·", "·", "·", "·", "·", "·", "."]),
    ("mid   ", ["^", "·", "·", "·", "$", "·", "·", "."]),
    ("end   ", ["^", "c", "a", "t", "$", "$", "$", "."]),
]
x0, cw = 0.30, 0.05
for r, (lab, cells) in enumerate(trace):
    yy = 0.10 - r * 0.045
    ax.text(0.16, yy + 0.018, lab, fontsize=8, family="monospace")
    for c, ch in enumerate(cells):
        fc = "#c8e6c9" if ch != "·" else "#ffffff"
        box(ax, x0 + c * cw, yy, cw * 0.9, 0.035, ch, fc, fs=8)
ax.text(0.66, 0.075, "order: length/boundary first,\nletters last (confidence-driven)",
        fontsize=8, color="#33691e", va="center")

plt.tight_layout(pad=1.2)
out = os.path.join(HERE, "omni_style_acmlm_pipeline.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", out)
