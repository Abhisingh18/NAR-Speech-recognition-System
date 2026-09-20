"""
Full architecture schematic of the filler_asr SA model (Pillow renderer).
Mirrors model_sa.py exactly: frozen HuBERT-XLarge feature extractor + encoder
(CTC head dropped) -> tap last_hidden_state -> 2 NEW post-norm self-attention
encoder layers (each = MHSA + Add&Norm + FFNN + Add&Norm) -> dropout -> lm_head.
Output: full_arch.png
"""
from PIL import Image, ImageDraw, ImageFont

# ---- dims (xlarge SA config) ----
d, V, H, dff = 1280, 33, 16, 5120
dk = d // H

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
def f(path, s): return ImageFont.truetype(path, s)
ft   = f(FB, 30)   # title
fh   = f(FB, 22)   # box headers
fr   = f(FR, 17)   # body
fm   = f(FM, 15)   # mono / dims
fs   = f(FR, 14)   # small

# ---- colors ----
C_BG     = (250, 250, 252)
C_FROZEN = (210, 222, 240)   # blue-grey  (frozen)
C_FRBRD  = (90, 120, 165)
C_TRAIN  = (215, 238, 215)   # green      (trainable)
C_TRBRD  = (70, 140, 70)
C_SUB    = (255, 248, 225)   # cream      (sub-layer)
C_SUBBRD = (200, 165, 70)
C_ADD    = (245, 222, 230)   # pink       (add & norm)
C_ADDBRD = (180, 90, 120)
C_IO     = (235, 235, 235)
C_IOBRD  = (120, 120, 120)
C_TXT    = (25, 25, 30)
C_GREY   = (110, 110, 115)
C_RES    = (190, 60, 60)     # residual arrow

W, Hh = 1180, 2240
img = Image.new("RGB", (W, Hh), C_BG)
dr = ImageDraw.Draw(img)

def ctext(cx, y, s, font, fill=C_TXT, anchor="mm"):
    dr.text((cx, y), s, font=font, fill=fill, anchor=anchor)

def box(x0, y0, x1, y1, fill, brd, w=2, r=14):
    dr.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=brd, width=w)

def varrow(x, y0, y1, color=(60, 60, 65), w=3):
    dr.line([x, y0, x, y1], fill=color, width=w)
    dr.polygon([(x-7, y1-11), (x+7, y1-11), (x, y1)], fill=color)

CX = W // 2

# ---------- title ----------
ctext(CX, 38, "filler_asr  —  Self-Attention model (xlarge)", ft)
ctext(CX, 68, "model_sa.py : FillerHubertSAModel", f(FM, 16), C_GREY)

y = 100

# ---------- input ----------
box(CX-180, y, CX+180, y+50, C_IO, C_IOBRD)
ctext(CX, y+18, "input_values", fr)
ctext(CX, y+37, "raw waveform  (B, samples @16kHz)", fs, C_GREY)
y += 50
varrow(CX, y, y+34); y += 34

# =================== FROZEN HuBERT ===================
fb_top = y
box(CX-430, y, CX+430, y+300, C_FROZEN, C_FRBRD, w=3)
ctext(CX, y+24, "HuBERT-XLarge-ls960-ft   (FROZEN feature extractor)", fh, (40, 60, 100))
ctext(CX, y+47, "CTC head dropped  •  requires_grad = False", fs, C_GREY)

# CNN feature extractor
iy = y + 70
box(CX-360, iy, CX+360, iy+46, (235,240,250), C_FRBRD, w=1)
ctext(CX, iy+15, "Conv feature extractor  (7 conv layers, stride)", fr)
ctext(CX, iy+33, "waveform -> (B, T, 1280)", fm, C_GREY)
varrow(CX, iy+46, iy+74)

# Transformer encoder (xN, post-norm, residual)
iy2 = iy + 74
box(CX-360, iy2, CX+360, iy2+118, (235,240,250), C_FRBRD, w=1)
ctext(CX, iy2+20, "Transformer encoder  x48  (post-norm, residual)", fr)
ctext(CX, iy2+44, "each: MHSA(16 heads) -> Add&Norm -> FFNN -> Add&Norm", fs, C_GREY)
ctext(CX, iy2+66, "d_model = 1280   dff = 5120", fm, C_GREY)
ctext(CX, iy2+90, "==>  tap  last_hidden_state  (B, T, 1280)", f(FM, 15), (40,60,100))
y = fb_top + 300
varrow(CX, y, y+40); y += 40
ctext(CX-470, y-20, "hidden_states", fm, C_GREY, anchor="lm")

# =================== TRAINABLE SA STACK ===================
tr_top = y
SA_H = 64 + 360 + 26 + 360 + 22   # top pad + layer1 + gap + layer2 + bottom pad
box(CX-470, y, CX+470, y+SA_H, C_TRAIN, C_TRBRD, w=3)
ctext(CX, y+26, "NEW Self-Attention stack   x2   (TRAINABLE)", fh, (30, 100, 40))
ctext(CX, y+49, "nn.TransformerEncoder  •  post-norm (norm_first=False)", fs, C_GREY)

def encoder_layer(cy, label):
    """draw one post-norm encoder layer; returns bottom y."""
    lx0, lx1 = CX-410, CX+410
    lh = 360
    box(lx0, cy, lx1, cy+lh, (228, 244, 228), C_TRBRD, w=2)
    ctext(CX, cy+22, label, fh, (30, 100, 40))
    in_y = cy + 42

    # ---- sub-layer 1: MHSA ----
    s1 = cy + 52
    box(CX-300, s1, CX+300, s1+58, C_SUB, C_SUBBRD, w=2)
    ctext(CX, s1+18, "Multi-Head Self-Attention", fr)
    ctext(CX, s1+39, f"H={H} heads,  d_k={dk}   (Q,K,V,O: 1280x1280)", fm, C_GREY)
    varrow(CX, s1+58, s1+86)
    # add & norm 1
    a1 = s1 + 86
    box(CX-300, a1, CX+300, a1+44, C_ADD, C_ADDBRD, w=2)
    ctext(CX, a1+22, "Add  &  LayerNorm", fr)
    varrow(CX, a1+44, a1+72)
    # residual arc 1 (input -> add1)
    rx = CX - 345
    dr.line([(CX, in_y), (rx, in_y)], fill=C_RES, width=2)
    dr.line([(rx, in_y), (rx, a1+22)], fill=C_RES, width=2)
    dr.line([(rx, a1+22), (CX-300, a1+22)], fill=C_RES, width=2)
    dr.polygon([(CX-300-1, a1+22-6),(CX-300-1, a1+22+6),(CX-289, a1+22)], fill=C_RES)
    ctext(rx-6, (in_y+a1+22)//2, "residual", fs, C_RES, anchor="rm")

    # ---- sub-layer 2: FFNN ----
    mid_y = a1 + 72
    s2 = a1 + 72
    box(CX-300, s2, CX+300, s2+58, C_SUB, C_SUBBRD, w=2)
    ctext(CX, s2+18, "FFNN   (position-wise)", fr)
    ctext(CX, s2+39, "Linear 1280->5120 -> GELU -> Linear 5120->1280", fm, C_GREY)
    varrow(CX, s2+58, s2+86)
    # add & norm 2
    a2 = s2 + 86
    box(CX-300, a2, CX+300, a2+44, C_ADD, C_ADDBRD, w=2)
    ctext(CX, a2+22, "Add  &  LayerNorm", fr)
    # residual arc 2
    rx2 = CX + 345
    dr.line([(CX, mid_y), (rx2, mid_y)], fill=C_RES, width=2)
    dr.line([(rx2, mid_y), (rx2, a2+22)], fill=C_RES, width=2)
    dr.line([(rx2, a2+22), (CX+300, a2+22)], fill=C_RES, width=2)
    dr.polygon([(CX+300+1, a2+22-6),(CX+300+1, a2+22+6),(CX+289, a2+22)], fill=C_RES)
    ctext(rx2+6, (mid_y+a2+22)//2, "residual", fs, C_RES, anchor="lm")
    return cy + lh

b = encoder_layer(y + 64, "Encoder layer 1")
varrow(CX, b, b+26)
b = encoder_layer(b + 26, "Encoder layer 2")
y = tr_top + SA_H
varrow(CX, y, y+34); y += 34

# ---------- dropout ----------
box(CX-200, y, CX+200, y+44, C_TRAIN, C_TRBRD, w=2)
ctext(CX, y+22, "Dropout  (final_dropout)", fr)
y += 44
varrow(CX, y, y+30); y += 30

# ---------- lm_head ----------
box(CX-260, y, CX+260, y+54, C_TRAIN, C_TRBRD, w=3)
ctext(CX, y+18, "lm_head   (TRAINABLE)", fr, (30, 100, 40))
ctext(CX, y+38, f"Linear  1280 -> {V}   (vocab)", fm, C_GREY)
y += 54
varrow(CX, y, y+30); y += 30

# ---------- logits / loss ----------
box(CX-300, y, CX+300, y+58, C_IO, C_IOBRD)
ctext(CX, y+18, f"logits   (B, T, {V})", fr)
ctext(CX, y+39, "framewise cross-entropy  vs  filler target", fs, C_GREY)

# ---------- legend ----------
ly = Hh - 40
def chip(x, c, b, label):
    dr.rounded_rectangle([x, ly, x+26, ly+18], radius=4, fill=c, outline=b, width=2)
    dr.text((x+34, ly+9), label, font=fs, fill=C_TXT, anchor="lm")
chip(40,  C_FROZEN, C_FRBRD, "frozen")
chip(180, C_TRAIN,  C_TRBRD, "trainable")
chip(340, C_SUB,    C_SUBBRD,"sub-layer")
chip(500, C_ADD,    C_ADDBRD,"add & norm")
dr.line([(680, ly+9),(716, ly+9)], fill=C_RES, width=2)
dr.text((724, ly+9), "residual connection", font=fs, fill=C_RES, anchor="lm")

img.save("full_arch.png")
print("wrote full_arch.png", img.size)
