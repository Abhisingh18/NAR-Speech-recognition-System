"""
Generates a detailed architecture + training schematic for the filler_asr project.
All shapes/numbers are taken directly from the codebase (see report.md).
Output: diagram.png (high-DPI, portrait, sized to sit on one PDF page).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ---- palette -------------------------------------------------------------
C_INPUT   = "#E8EEF7"; E_INPUT  = "#3B6CA8"   # data / IO
C_FROZEN  = "#DCE3EA"; E_FROZEN = "#6B7785"   # frozen (CNN)
C_TRAIN   = "#DDEFD8"; E_TRAIN  = "#3E7D32"   # trainable encoder
C_HEAD    = "#FBE3C6"; E_HEAD   = "#C77A12"   # new head
C_LOSS    = "#F6D9DC"; E_LOSS   = "#B23A48"   # loss / labels
C_NOTE    = "#FFFFFF"; E_NOTE   = "#999999"   # side notes
TXT = "#1a1a1a"

fig, ax = plt.subplots(figsize=(11.0, 15.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 144); ax.axis("off")

def box(x, y, w, h, fc, ec, title, sub="", tcolor=TXT, ts=11, ss=8.5, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                       linewidth=1.6, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(p)
    cy = y + h/2
    if sub:
        ax.text(x+w/2, cy+h*0.16, title, ha="center", va="center",
                fontsize=ts, fontweight="bold" if bold else "normal", color=tcolor, zorder=3)
        ax.text(x+w/2, cy-h*0.24, sub, ha="center", va="center",
                fontsize=ss, color="#333333", zorder=3)
    else:
        ax.text(x+w/2, cy, title, ha="center", va="center",
                fontsize=ts, fontweight="bold" if bold else "normal", color=tcolor, zorder=3)
    return (x+w/2, y, y+h)  # cx, ybottom, ytop

def arrow(x, y0, y1, label="", lx_off=2.2, color="#2b2b2b"):
    a = FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                        linewidth=1.8, color=color, zorder=1)
    ax.add_patch(a)
    if label:
        ax.text(x+lx_off, (y0+y1)/2, label, ha="left", va="center",
                fontsize=9.0, color="#0b3d66", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=3)

# ============================ TITLE =======================================
ax.text(50, 141.5, "filler_asr — HuBERT-base ASR: End-to-End Forward Pass & Training",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#11304f")
ax.text(50, 138.6, "facebook/hubert-base-ls960  ·  framewise cross-entropy (not CTC)  ·  vocab=33  ·  hidden=768",
        ha="center", va="center", fontsize=10, color="#444444", style="italic")

# ============================ MAIN FORWARD COLUMN =========================
cx = 31; w = 40
nodes = []

t = box(cx-w/2, 130.0, w, 5.6, C_INPUT, E_INPUT,
        "Raw waveform", "16 kHz mono, per-utterance normalized")
nodes.append(t)
arrow(cx, 130.0, 124.6, "input_values  (B, T_s)")

t = box(cx-w/2, 113.0, w, 11.0, C_FROZEN, E_FROZEN,
        "CNN Feature Extractor  (FROZEN)",
        "7 conv1d layers  ·  channels=512\nkernels [10,3,3,3,3,2,2]\nstrides [5,2,2,2,2,2,2]  ·  downsample ÷320")
nodes.append(t)
arrow(cx, 113.0, 107.6, "(B, T, 512)   T ≈ T_s / 320  ·  ~50 frames/s")

t = box(cx-w/2, 99.0, w, 6.2, C_TRAIN, E_TRAIN,
        "Feature Projection", "Linear 512 → 768  +  LayerNorm  +  dropout")
nodes.append(t)
arrow(cx, 99.0, 93.6, "(B, T, 768)")

t = box(cx-w/2, 78.5, w, 13.0, C_TRAIN, E_TRAIN,
        "Transformer Encoder  (TRAINABLE)",
        "pos-conv (k=128, groups=16)\n12 layers · 12 heads · FFN 768→3072→768\nGELU · dropout 0.1 · SpecAugment (train)")
nodes.append(t)
arrow(cx, 78.5, 73.1, "last_hidden_state  (B, T, 768)")

t = box(cx-w/2, 67.0, w, 5.2, C_TRAIN, E_TRAIN,
        "Final Dropout", "p = 0.1")
nodes.append(t)
arrow(cx, 67.0, 61.6, "(B, T, 768)")

t = box(cx-w/2, 54.5, w, 6.4, C_HEAD, E_HEAD,
        "lm_head  (NEW · TRAINABLE)", "Linear 768 → 33    (25,377 params)")
nodes.append(t)
arrow(cx, 54.5, 49.1, "logits  (B, T, 33)")

t = box(cx-w/2, 43.0, w, 5.6, C_INPUT, E_INPUT,
        "argmax over vocab", "one token id per 20 ms frame")
nodes.append(t)
arrow(cx, 43.0, 37.6, "ids  (B, T)")

t = box(cx-w/2, 30.5, w, 6.6, C_INPUT, E_INPUT,
        "Decode (tokenizer)",
        "skip_special_tokens=True\ngroup_tokens=False   ·   '|' → space")
nodes.append(t)
arrow(cx, 30.5, 25.1, "")

box(cx-w/2, 19.0, w, 5.6, "#EAF7EE", "#2f7d3a",
    "Transcription (text)", "\"concord returned to its place ...\"")

# ============================ RIGHT: LABELS + LOSS ========================
rx = 78; rw = 38

ax.text(rx, 135.0, "TRAINING TARGET  (the \"filler\" trick)", ha="center",
        fontsize=10.5, fontweight="bold", color="#7a2730")

box(rx-rw/2, 119.5, rw, 13.5, C_LOSS, E_LOSS,
    "Frame-level label of length T",
    "text packed LEFT, rest = <fill>:\n[ <s>, c, o, n, c, o, r, d, |, ...,\n  </s>, <fill>, <fill>, ..., <fill> ]\n(most frames are <fill>)", ts=10, ss=8.5)

# dashed link from label box down to loss, and a tie to logits level
arrow(rx, 119.5, 96.0, "")
box(rx-rw/2, 88.5, rw, 7.0, C_LOSS, E_LOSS,
    "CrossEntropyLoss",
    "ignore_index = -100  (pads)\nper-frame  ·  NO CTC / no blank", ts=10, ss=8.5)

# connect logits to loss
con = FancyArrowPatch((cx+w/2, 57.7), (rx-rw/2, 90.5),
                      arrowstyle="-|>", mutation_scale=14, linewidth=1.6,
                      color="#B23A48", linestyle=(0,(5,3)), zorder=1,
                      connectionstyle="arc3,rad=-0.25")
ax.add_patch(con)
ax.text(60.5, 74.0, "logits vs labels", fontsize=8.5, color="#7a2730",
        rotation=58, style="italic")

# ---- freezing curriculum -------------------------------------------------
box(rx-rw/2, 73.0, rw, 12.5, C_NOTE, E_NOTE,
    "Two-phase freezing",
    "CNN extractor: frozen always (~4.2M)\nPhase A (first 10% steps):\n   train lm_head only (25k)\nPhase B (next 90%):\n   train all EXCEPT CNN (~90M)", ts=10, ss=8.3, bold=True)

# ---- triphase LR ---------------------------------------------------------
lx, ly, lwd, lht = rx-rw/2, 56.0, rw, 13.5
box(lx, ly, lwd, lht, C_NOTE, E_NOTE, "", "")
ax.text(lx+lwd/2, ly+lht-1.6, "Triphase LR schedule", ha="center",
        fontsize=10, fontweight="bold", color=TXT)
ax.text(lx+lwd/2, ly+lht-3.6, "peak 3e-4  ·  ratios [0.1, 0.4, 0.5]", ha="center",
        fontsize=8.2, color="#333")
# mini LR plot inside the box
px0, px1 = lx+3.5, lx+lwd-3.0
py0, py1 = ly+2.2, ly+6.8
warm = px0 + (px1-px0)*0.10
const = px0 + (px1-px0)*0.50
ax.add_line(Line2D([px0, warm, const, px1],
                   [py0, py1, py1, py0], color="#3B6CA8", linewidth=2.2))
ax.add_line(Line2D([px0, px1],[py0, py0], color="#888", linewidth=0.8))
ax.text(warm, py0-1.3, "warmup", ha="center", fontsize=6.8, color="#444")
ax.text((warm+const)/2, py0-1.3, "constant", ha="center", fontsize=6.8, color="#444")
ax.text((const+px1)/2, py0-1.3, "decay", ha="center", fontsize=6.8, color="#444")

# ============================ BOTTOM: NUMERIC EXAMPLE =====================
ex_y = 8.0
ax.add_patch(FancyBboxPatch((6, ex_y), 88, 8.6, boxstyle="round,pad=0.5,rounding_size=1.2",
             linewidth=1.4, facecolor="#F4F1FA", edgecolor="#6A4FA3", zorder=2))
ax.text(50, ex_y+6.7, "Worked dimension example  (4.0 s clip)", ha="center",
        fontsize=10.5, fontweight="bold", color="#3d2a6b")
ax.text(50, ex_y+3.0,
        "64,000 samples  →[÷320]→  T = 199 frames  →  logits (1, 199, 33)  →  argmax (1, 199)  →  decoded text\n"
        "Effective batch 256 = 16/GPU × 4 accum × 4 GPUs   ·   1,099 steps/epoch × 200 epochs = 219,800 steps",
        ha="center", fontsize=8.8, color="#2a2150")

# ============================ LEGEND ======================================
leg = [("Data / IO", C_INPUT, E_INPUT),
       ("Frozen (CNN)", C_FROZEN, E_FROZEN),
       ("Trainable encoder", C_TRAIN, E_TRAIN),
       ("New head", C_HEAD, E_HEAD),
       ("Labels / loss", C_LOSS, E_LOSS)]
lx0 = 7.0
for i,(name,fc,ec) in enumerate(leg):
    yy = 2.6
    xx = lx0 + i*18.4
    ax.add_patch(Rectangle((xx, yy), 2.6, 2.0, facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax.text(xx+3.3, yy+1.0, name, va="center", fontsize=8.2, color=TXT)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig("diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
print("diagram.png written")
