"""Render all figures + equation images for the filler_sa_reference explainer PDF."""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm"})

C_FROZEN = "#dbe9f6"; C_TRAIN = "#fde3c8"; C_DATA = "#f3ecd8"; C_LOSS = "#e8d5e8"
E_FROZEN = "#3b73b9"; E_TRAIN = "#d97b29"; E_DATA = "#a08c4a"; E_LOSS = "#8e4d8e"


# ---------------------------------------------------------------- equations
def eq(name, tex, fs=15, pad=0.12):
    fig = plt.figure(figsize=(0.1, 0.1))
    t = fig.text(0, 0, tex, fontsize=fs)
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight",
                pad_inches=pad, facecolor="white")
    plt.close(fig)

eq("eq_conv.png",
   r"$L_{\mathrm{out}}=\left\lfloor \frac{L_{\mathrm{in}}-k}{s}\right\rfloor+1"
   r"\qquad (k,s)\in\{(10,5),(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)\}"
   r"\qquad T\approx \frac{L}{320}\ \ (\approx 50\ \mathrm{frames/s})$")
eq("eq_pe.png",
   r"$PE_{p,\,2i}=\sin\!\left(\frac{p}{10000^{2i/d}}\right)\qquad "
   r"PE_{p,\,2i+1}=\cos\!\left(\frac{p}{10000^{2i/d}}\right)\qquad "
   r"\lambda_i = 2\pi\cdot 10000^{2i/d}\in[2\pi,\ 2\pi\!\cdot\!10^4]$")
eq("eq_qkv.png",
   r"$Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad X\in\mathbb{R}^{T\times 1280},"
   r"\ \ W_{Q,K,V}\in\mathbb{R}^{1280\times 1280}\ \ (16\ \mathrm{heads},\ d_k=80)$")
eq("eq_attn.png",
   r"$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V"
   r"\qquad M_{tj}=-\infty\ \ \mathrm{if}\ j\ \mathrm{is\ a\ pad\ frame,\ else}\ 0$")
eq("eq_postnorm.png",
   r"$x'=\mathrm{LN}(x+\mathrm{MHSA}(x))\qquad "
   r"y=\mathrm{LN}(x'+\mathrm{FFN}(x'))\qquad "
   r"\mathrm{FFN}(u)=\mathrm{GELU}(uW_1+b_1)\,W_2+b_2$")
eq("eq_labels.png",
   r"$y_0=\langle s\rangle\quad\ \ y_t=c_t\ (1\leq t\leq N)\quad\ \ "
   r"y_{N+1}=\langle /s\rangle\quad\ \ y_t=\langle fill\rangle\ (N{+}1<t<T)"
   r"\qquad N=\#\mathrm{chars\ (spaces}\to |\,)$")
eq("eq_nkeep.png",
   r"$n_{\mathrm{keep}}=\left\lceil \mathrm{dur}_s\times 25\right\rceil\qquad "
   r"y_t\leftarrow -100\ \ \mathrm{for}\ t\geq n_{\mathrm{keep}}\ \ \mathrm{or}\ t\ \mathrm{is\ label\ padding}$")
eq("eq_ce.png",
   r"$\mathcal{L}=-\frac{1}{|G|}\sum_{(b,t)\in G}\log\,"
   r"\mathrm{softmax}(z_{b,t})_{y_{b,t}}\qquad "
   r"G=\{(b,t):y_{b,t}\neq -100\}\qquad z_{b,t}\in\mathbb{R}^{33}$")
eq("eq_ln33.png",
   r"$\mathcal{L}_{\mathrm{init}}\approx -\log\frac{1}{33}=\ln 33\approx 3.497$")
eq("eq_params.png",
   r"$P_{\mathrm{layer}}=(4d^2+5d)_{\mathrm{MHSA}}"
   r"+(2df+f+d)_{\mathrm{FFN}}+(4d)_{\mathrm{2LN}}"
   r"=19{,}677{,}440\quad(d{=}1280,\ f{=}5120)$")
eq("eq_total.png",
   r"$8\times 19{,}677{,}440+(1280\times 33+33)_{\mathrm{lm\ head}}"
   r"=157{,}461{,}793\ \approx\ 157.46\mathrm{M\ trainable}\quad"
   r"(\mathrm{vs}\ \sim\!964\mathrm{M\ frozen\ HuBERT})$")
eq("eq_lr.png",
   r"$\eta(s)=\eta_{\max}\cdot\frac{s}{2000}\ \ (s\leq 2000)\qquad"
   r"\eta(s)=\eta_{\max}\cdot\frac{73200-s}{73200-2000}\ \ (s>2000)"
   r"\qquad \eta_{\max}=2\times10^{-4}$")
eq("eq_eff.png",
   r"$B_{\mathrm{eff}}=32_{\mathrm{batch}}\times 2_{\mathrm{accum}}\times 6_{\mathrm{GPUs}}=384"
   r"\qquad \mathrm{steps/epoch}=\left\lfloor\frac{281241/6}{32}\right\rfloor / 2=732"
   r"\qquad 100\times732=73{,}200\ \mathrm{steps}$")


# ---------------------------------------------------------------- fig: architecture
def box(ax, x, y, w, h, text, fc, ec, fs=9.2, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

def arrow(ax, x0, y0, x1, y1, label="", ls="-", col="#444444"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color=col, linestyle=ls))
    if label:
        ax.text((x0+x1)/2 + 0.015, (y0+y1)/2, label, fontsize=8.4,
                ha="left", va="center", color="#333333", family="monospace")

fig, ax = plt.subplots(figsize=(8.6, 10.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
X, W = 0.06, 0.50
ys = [0.935, 0.83, 0.665, 0.55, 0.455, 0.315, 0.20, 0.10]
H = 0.062
box(ax, X, ys[0], W, H, "raw audio  16 kHz mono\n(B, L) float samples", C_DATA, E_DATA)
box(ax, X, ys[1], W, H, "Wav2Vec2FeatureExtractor\nzero-mean / unit-var normalize + batch pad", C_DATA, E_DATA)
box(ax, X, ys[2], W, 0.115,
    "FROZEN  HuBERT-xlarge  (~964M)\n7-layer conv stack: 320× downsample → ~50 fps\n"
    "48 transformer layers, d=1280\nCTC head dropped — tap last_hidden_state", C_FROZEN, E_FROZEN)
box(ax, X, ys[3], W, 0.055, "⊕  sinusoidal PE  (fixed, no params)\npe_scale · PE(T,1280)", C_FROZEN, E_FROZEN)
box(ax, X, ys[4], W, 0.045, "(optional, OFF in this run)\ntrain-time span masking of the tap", "#eeeeee", "#999999", fs=8.4)
box(ax, X, ys[5], W, 0.095,
    "TRAINABLE  8 × post-norm SA layers  (157.4M)\nMHSA(16 heads, d_k=80) → Add&LN\n"
    "FFN 1280→5120→1280 (GELU) → Add&LN", C_TRAIN, E_TRAIN)
box(ax, X, ys[6], W, 0.055, "Dropout(0.1)  →  lm_head: Linear 1280 → 33\n(TRAINABLE, 42k params)", C_TRAIN, E_TRAIN)
box(ax, X, ys[7], W, 0.055, "per-frame logits\n(B, T, 33)", C_DATA, E_DATA)

xm = X + W/2
arrow(ax, xm, ys[0], xm, ys[1]+H, "(B, L)")
arrow(ax, xm, ys[1], xm, ys[2]+0.115, "input_values (B, L)\nattention_mask (B, L)")
arrow(ax, xm, ys[2], xm, ys[3]+0.055, "tap (B, T, 1280)   T ≈ L/320")
arrow(ax, xm, ys[3], xm, ys[4]+0.045, "(B, T, 1280)")
arrow(ax, xm, ys[4], xm, ys[5]+0.095, "(B, T, 1280)")
arrow(ax, xm, ys[5], xm, ys[6]+0.055, "(B, T, 1280)")
arrow(ax, xm, ys[6], xm, ys[7]+0.055, "(B, T, 33)")

# right column: label prep + loss
XR, WR = 0.64, 0.32
box(ax, XR, 0.83, WR, 0.062, "transcript text\nlowercase, strip punctuation", C_DATA, E_DATA)
box(ax, XR, 0.665, WR, 0.095,
    "positional <fill> target  (B, T)\n[<s>] + chars + [</s>] + <fill>×tail\nchar i pinned to frame i", C_DATA, E_DATA)
box(ax, XR, 0.50, WR, 0.085,
    "loss mask\npad → −100\nt ≥ n_keep = ⌈dur·25⌉ → −100", C_LOSS, E_LOSS)
box(ax, XR, 0.20, WR, 0.075, "framewise CrossEntropy\nignore_index = −100\nmean over graded frames", C_LOSS, E_LOSS)
xr = XR + WR/2
arrow(ax, xr, 0.83, xr, 0.76, "char ids")
arrow(ax, xr, 0.665, xr, 0.585, "labels (B, T)")
arrow(ax, xr, 0.50, xr, 0.275, "labels, −100 masked")
arrow(ax, X+W, ys[7]+0.0275, XR+0.02, 0.235, "logits")
ax.text(xr, 0.155, "scalar loss  →  backward through SA + lm_head only",
        ha="center", fontsize=8.6, style="italic")
ax.set_title("filler_asr SA pipeline — frozen HuBERT tap → PE → SA×8 → lm_head, framewise CE",
             fontsize=11.5)
fig.savefig(os.path.join(OUT, "fig_arch.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- fig: labels + masks timeline
# real clip 1089-134686-0000: dur=10.44 s, T=521, n_keep=261, transcript 155 chars -> N+2 = 157
T, nk, Nlab = 521, 261, 157
fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.6), height_ratios=[1, 1.35])
ax = axes[0]
ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_yticks([])
ax.broken_barh([(0, 1)], (0.25, 0.5), color="#4a9c4a")
ax.broken_barh([(1, Nlab-2)], (0.25, 0.5), color="#5a8ac6")
ax.broken_barh([(Nlab-1, 1)], (0.25, 0.5), color="#c65a5a")
ax.broken_barh([(Nlab, nk-Nlab)], (0.25, 0.5), color="#d9c86a")
ax.broken_barh([(nk, T-nk)], (0.25, 0.5), color="#d8d8d8")
ax.axvline(nk, color="#8e4d8e", lw=1.8, ls="--")
ax.text(1, 0.86, "<s>", fontsize=8, color="#2a6c2a")
ax.text(Nlab/2, 0.86, "chars+| (one per frame, N=155)", fontsize=8.5, ha="center", color="#2a5a96")
ax.text(Nlab+4, 0.86, "</s>", fontsize=8, color="#963a3a")
ax.text((Nlab+nk)/2, 0.05, "<fill> (graded)", fontsize=8.5, ha="center", color="#8a7a20")
ax.text((nk+T)/2, 0.05, "ignored: label = −100", fontsize=8.5, ha="center", color="#666666")
ax.text(nk + 8, 0.87, "n_keep = ⌈ 10.44 · 25 ⌉ = 261", fontsize=8.8, ha="left", color="#8e4d8e")
ax.set_title("Per-frame target for a real clip (1089-134686-0000): dur = 10.44 s  →  T = 521 frames @ 50 fps",
             fontsize=9.8)
ax.set_xlabel("encoder frame index t", fontsize=8.5)

ax = axes[1]
ax.set_xlim(0, T); ax.set_ylim(-0.2, 1.35); ax.set_yticks([]); ax.set_xticks([])
ax.broken_barh([(0, T)], (0.95, 0.28), color="#cfe3cf")
ax.text(T/2, 1.09, "audio: words are SPOKEN all across the 10.44 s (frames 0..521)",
        ha="center", fontsize=8.8, color="#2a6c2a")
ax.broken_barh([(0, Nlab)], (0.0, 0.28), color="#b8cdf0")
ax.broken_barh([(Nlab, T-Nlab)], (0.0, 0.28), color="#e6e6e6")
ax.text(Nlab/2, 0.14, "labels: chars packed at the HEAD (frames 0..157)", ha="center", fontsize=8.6, color="#2a5a96")
for f0, f1 in [(60, 25), (200, 75), (350, 120), (480, 152)]:
    ax.add_patch(FancyArrowPatch((f0, 0.95), (f1, 0.30), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.1, color="#b05050",
                                 connectionstyle="arc3,rad=0.25"))
ax.text(T*0.72, 0.58, "self-attention must TRANSPORT acoustic evidence\nleftwards to the positional slot of each char",
        fontsize=8.8, color="#b05050", ha="center", style="italic")
ax.set_title("Why the head needs self-attention + PE: the target is positional, not acoustically aligned",
             fontsize=9.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_labels.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- fig: PE heatmap + rows
L, d = 260, 1280
pos = np.arange(L)[:, None]
div = np.exp(np.arange(0, d, 2) * (-math.log(10000.0) / d))
pe = np.zeros((L, d)); pe[:, 0::2] = np.sin(pos*div); pe[:, 1::2] = np.cos(pos*div)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), width_ratios=[1.35, 1])
im = axes[0].imshow(pe[:, :160].T, aspect="auto", cmap="RdBu", origin="lower")
axes[0].set_xlabel("frame position p"); axes[0].set_ylabel("dimension i (first 160 of 1280)")
axes[0].set_title("PE(p, i): each dim is a sinusoid of its own wavelength", fontsize=9.5)
fig.colorbar(im, ax=axes[0], fraction=0.04)
for i, c in zip([4, 40, 100], ["#c44", "#48a", "#7a5"]):
    axes[1].plot(pe[:, i], lw=1.1, color=c, label=f"dim {i}")
axes[1].legend(fontsize=8); axes[1].set_xlabel("frame position p")
axes[1].set_title("fast dims → fine position, slow dims → coarse position", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_pe.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- fig: training curve + LR
steps, wer, cer = [], [], []
for line in open("/tmp/sa8_eval_curve.txt"):
    s, w, c = line.split(); steps.append(int(s)); wer.append(float(w)); cer.append(float(c))
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
ax = axes[0]
ax.plot(steps, wer, "-o", ms=3, lw=1.3, color="#c0392b", label="dev WER (500 clips)")
ax.plot(steps, np.clip(cer, 0, 1.05), "-s", ms=2.6, lw=1.1, color="#2980b9", label="dev CER (clipped)")
ax.axhline(0.135, color="#888", lw=0.8, ls=":")
ax.annotate("best 0.135 @ 40k → best.pt", xy=(40000, 0.135), xytext=(30000, 0.42),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.0, color="#555"))
ax.set_ylim(0, 1.05); ax.set_xlabel("optimizer step"); ax.set_ylabel("error rate")
ax.legend(fontsize=8); ax.set_title("dev-clean curve (uncapped greedy decode)", fontsize=9.5)
ax = axes[1]
s = np.arange(0, 73201, 100)
lr = np.where(s <= 2000, s/2000, (73200 - s)/(73200 - 2000)) * 2e-4
ax.plot(s, lr, lw=1.4, color="#8e44ad")
ax.axvline(2000, color="#888", lw=0.8, ls=":"); ax.text(3200, 1.9e-4, "warmup ends (2k)", fontsize=8)
ax.set_xlabel("optimizer step"); ax.set_ylabel("learning rate")
ax.set_title("linear schedule: 0 → 2e-4 → 0 over 73,200 steps", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_curve.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- fig: the three masks
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
Tm = 60
ax = axes[0]
lens = [60, 44, 30]
m = np.zeros((3, Tm))
for i, l in enumerate(lens): m[i, l:] = 1
ax.imshow(m, aspect="auto", cmap="Greys", vmin=0, vmax=1.6)
ax.set_yticks(range(3)); ax.set_yticklabels([f"utt{i}  T={l}" for i, l in enumerate(lens)], fontsize=8)
ax.set_title("1) FRAME mask (attention)\npad frames excluded as MHSA keys", fontsize=8.8)
ax.set_xlabel("frame", fontsize=8)
ax = axes[1]
m = np.zeros((3, Tm))
for i, (l, n, nkp) in enumerate(zip(lens, [18, 14, 10], [30, 22, 15])):
    m[i, n:nkp] = 0.45          # graded fill
    m[i, nkp:] = 1.0            # -100
ax.imshow(m, aspect="auto", cmap="Greys", vmin=0, vmax=1.6)
ax.set_yticks(range(3)); ax.set_yticklabels(["", "", ""])
ax.set_title("2) LOSS mask (−100)\nbeyond n_keep + label pad ignored in CE", fontsize=8.8)
ax.set_xlabel("dark = −100, grey = graded <fill>, white = chars", fontsize=7.6)
ax = axes[2]
m = np.zeros((3, Tm))
rng = np.random.RandomState(3)
for i, l in enumerate(lens):
    for st in rng.choice(range(l-10), 2, replace=False): m[i, st:st+10] = 1
    m[i, l:] = 0.25
ax.imshow(m, aspect="auto", cmap="Oranges", vmin=0, vmax=1.4)
ax.set_yticks(range(3)); ax.set_yticklabels(["", "", ""])
ax.set_title("3) INPUT masking (optional, OFF here)\n200 ms spans → learned mask_embed", fontsize=8.8)
ax.set_xlabel("frame", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_masks.png"), dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("figures written to", OUT)
