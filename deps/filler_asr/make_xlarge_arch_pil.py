"""
Architecture schematic (Pillow renderer, no matplotlib needed) for the
XLarge filler_asr variant:
    frozen HuBERT-XLarge-ls960-ft (CTC head removed)
      -> 2 NEW self-attention layers
      -> NEW lm_head
    trained with framewise cross-entropy on the "filler" target.
Dimensions use hidden=1280 (xlarge), vocab=33, with a worked numeric example.
Output: xlarge_arch.png
"""
from PIL import Image, ImageDraw, ImageFont

# ---------------- numbers ----------------
d, V, Hh, dff = 1280, 33, 16, 5120
dk = d // Hh

def sa_params(d, dff):
    attn = 4 * (d*d + d)
    ffn  = (d*dff + dff) + (dff*d + d)
    ln   = 2 * (2*d)
    return attn + ffn + ln
per_sa = sa_params(d, dff); two_sa = 2*per_sa
head_p = d*V + V; new_total = two_sa + head_p

def out_len(n):
    for k, s in zip([10,3,3,3,3,2,2], [5,2,2,2,2,2,2]):
        n = (n-k)//s + 1
    return n
SAMPLES = 160000; T_ex = out_len(SAMPLES)

# ---------------- fonts ----------------
FD = "/usr/share/fonts/truetype/dejavu/"
def F(name, size): return ImageFont.truetype(FD+name, size)
f_title = F("DejaVuSans-Bold.ttf", 34)
f_sub   = F("DejaVuSans.ttf", 20)
f_bt    = F("DejaVuSans-Bold.ttf", 23)
f_bs    = F("DejaVuSans.ttf", 16)
f_lab   = F("DejaVuSans-Bold.ttf", 16)
f_pan   = F("DejaVuSans-Bold.ttf", 18)
f_mono  = F("DejaVuSansMono.ttf", 17)
f_monob = F("DejaVuSansMono-Bold.ttf", 18)
f_sec   = F("DejaVuSans-Bold.ttf", 19)

# ---------------- palette ----------------
INPUT=("#E8EEF7","#3B6CA8"); FROZEN=("#DCE3EA","#6B7785")
SA=("#DDEFD8","#3E7D32"); HEAD=("#FBE3C6","#C77A12")
LOSS=("#F6D9DC","#B23A48"); NOTE=("#FFFFFF","#9a9a9a")
OUTC=("#EAF7EE","#2f7d3a"); TXT="#1a1a1a"

W, H = 1580, 2280
img = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(img)

def ctext(cx, cy, s, font, fill=TXT, anchor="mm"):
    dr.text((cx, cy), s, font=font, fill=fill, anchor=anchor)

def box(x, y, w, h, colors, title, sub="", tfont=f_bt, sfont=f_bs):
    fc, ec = colors
    dr.rounded_rectangle([x, y, x+w, y+h], radius=14, fill=fc, outline=ec, width=3)
    cx = x+w/2
    if sub:
        ctext(cx, y+h*0.34, title, tfont)
        lines = sub.split("\n")
        ly = y+h*0.50
        for ln in lines:
            ctext(cx, ly, ln, sfont, fill="#333"); ly += sfont.size+4
    else:
        ctext(cx, y+h/2, title, tfont)

def panel(x, y, w, h, ec, label):
    dr.rounded_rectangle([x, y, x+w, y+h], radius=18, outline=ec, width=3)
    dr.text((x+14, y+8), label, font=f_pan, fill=ec, anchor="lm")

def arrow(x, y0, y1, label="", color=(43,43,43), labcol="#0b3d66"):
    hl, hw = 16, 8
    dr.line([(x, y0), (x, y1-hl)], fill=color, width=4)
    dr.polygon([(x, y1), (x-hw, y1-hl), (x+hw, y1-hl)], fill=color)
    if label:
        ty = (y0+y1)/2
        bb = dr.textbbox((x+14, ty), label, font=f_lab, anchor="lm")
        dr.rectangle([bb[0]-4, bb[1]-3, bb[2]+4, bb[3]+3], fill="white")
        dr.text((x+14, ty), label, font=f_lab, fill=labcol, anchor="lm")

# ---------------- title ----------------
ctext(W/2, 40, "filler_asr (XLarge variant) — HuBERT-XLarge-ft  +  2 Self-Attention  +  new head", f_title, "#11304f")
ctext(W/2, 80, f"facebook/hubert-xlarge-ls960-ft   ·   framewise cross-entropy (not CTC)   ·   "
               f"vocab={V}   ·   hidden={d}   ·   CTC head removed, embeddings tapped", f_sub, "#444")

# ---------------- main column geometry ----------------
cx = 470; bw = 660; x0 = cx - bw//2
GAP = 60

# pass 1: positions
rects = {}
y = 120
def place(key, h):
    global y
    rects[key] = (x0, y, bw, h); y += h + GAP

place("wave", 70)
place("cnn", 104)
place("proj", 80)
place("tr", 104)
place("sa1", 116)
place("sa2", 116)
place("head", 84)
place("argmax", 76)
place("decode", 84)
place("text", 76)

def r(key): return rects[key]

# panels behind groups
fx, fy, fw, fh = r("cnn")[0]-16, r("cnn")[1]-30, bw+32, (r("tr")[1]+r("tr")[3]) - r("cnn")[1] + 46
panel(fx, fy, fw, fh, FROZEN[1], "FROZEN · HuBERT-XLarge-ls960-ft  (already CTC-finetuned)")
nx, ny, nw, nh = r("sa1")[0]-16, r("sa1")[1]-30, bw+32, (r("head")[1]+r("head")[3]) - r("sa1")[1] + 46
panel(nx, ny, nw, nh, SA[1], f"NEW · TRAINABLE   (2x Self-Attention + lm_head  ~ {new_total/1e6:.1f}M params)")

# pass 2: draw boxes + arrows
def cy_top(key): return r(key)[1]
def cy_bot(key): return r(key)[1]+r(key)[3]

box(*r("wave"), INPUT, "Raw waveform", "16 kHz mono, per-utterance normalized")
arrow(cx, cy_bot("wave"), cy_top("cnn"), "input_values  (B, T_s)")

box(*r("cnn"), FROZEN, "CNN Feature Extractor",
    "7 conv1d · channels 512 · kernels [10,3,3,3,3,2,2]\nstrides [5,2,2,2,2,2,2] · downsample /320")
arrow(cx, cy_bot("cnn"), cy_top("proj"), "(B, T, 512)   T = T_s/320  ·  ~50 frames/s")

box(*r("proj"), FROZEN, "Feature Projection", f"Linear 512 -> {d}  +  LayerNorm")
arrow(cx, cy_bot("proj"), cy_top("tr"), f"(B, T, {d})")

box(*r("tr"), FROZEN, "Transformer Encoder",
    f"48 layers · 16 heads · FFN {d}->{dff}->{d}\n(CTC classification head removed)")
arrow(cx, cy_bot("tr"), cy_top("sa1"),
      f"last_hidden_state  (B, T, {d})   <- EMBEDDING TAP", color=(178,58,72), labcol="#7a2730")

box(*r("sa1"), SA, "Self-Attention Layer 1  (NEW)",
    f"d={d} · {Hh} heads (dk={dk}) · FFN {d}->{dff}->{d}\nA = softmax(QK^T / sqrt(dk)) V · residual + LN")
arrow(cx, cy_bot("sa1"), cy_top("sa2"), f"(B, T, {d})")

box(*r("sa2"), SA, "Self-Attention Layer 2  (NEW)",
    f"d={d} · {Hh} heads (dk={dk}) · FFN {d}->{dff}->{d}\nbidirectional · no causal mask · pad-masked")
arrow(cx, cy_bot("sa2"), cy_top("head"), f"(B, T, {d})")

box(*r("head"), HEAD, "lm_head  (NEW)", f"Linear {d} -> {V}    ({head_p:,} params)")
arrow(cx, cy_bot("head"), cy_top("argmax"), f"logits  (B, T, {V})")

box(*r("argmax"), INPUT, "argmax over vocab", "one token id per 20 ms frame")
arrow(cx, cy_bot("argmax"), cy_top("decode"), "ids  (B, T)")

box(*r("decode"), INPUT, "Decode (tokenizer)",
    "skip_special_tokens=True · group_tokens=False · '|'->space")
arrow(cx, cy_bot("decode"), cy_top("text"), "")

box(*r("text"), OUTC, "Transcription (text)", "\"concord returned to its place ...\"")

# ---------------- right column ----------------
rx0 = 940; rw = 510
ctext(rx0+rw/2, 130, "TRAINING TARGET  (the \"filler\" trick)", f_sec, "#7a2730")
box(rx0, 150, rw, 150, LOSS, "Frame-level label, length T",
    "text packed LEFT, rest = <fill>:\n[ <s>, c, o, n, c, o, r, d, |, ...,\n  </s>, <fill>, ..., <fill> ]\n(~70% of frames are <fill>)", f_bt, f_bs)
arrow(rx0+rw/2, 300, 360, "")

box(rx0, 360, rw, 110, LOSS, "CrossEntropyLoss",
    f"logits (B,T,{V})  vs  labels (B,T)\nignore_index = -100 · per-frame · no CTC", f_bt, f_bs)

# dashed connector logits-level -> loss box
def dline(p0, p1, color, dash=18, gap=12, width=3):
    import math
    x0,y0=p0; x1,y1=p1; L=math.hypot(x1-x0,y1-y0); n=int(L//(dash+gap))
    for i in range(n+1):
        a=(i*(dash+gap))/L; b=min((i*(dash+gap)+dash)/L,1.0)
        dr.line([(x0+(x1-x0)*a, y0+(y1-y0)*a),(x0+(x1-x0)*b, y0+(y1-y0)*b)], fill=color, width=width)
lx0, ly0 = cx+bw//2, cy_top("head")+r("head")[3]/2
dline((lx0, ly0), (rx0, 420), (178,58,72))
dr.polygon([(rx0, 420),(rx0-15,414),(rx0-12,429)], fill=(178,58,72))

box(rx0, 530, rw, 150, NOTE, "Freezing strategy",
    "Whole HuBERT-XLarge: FROZEN\n(used as a feature extractor)\nTrain ONLY the 2 SA layers + lm_head\n(~39.4M trainable of ~1.0B total)", f_bt, f_bs)

box(rx0, 710, rw, 160, NOTE, "Self-attention math (per layer)",
    "Q,K,V = X.Wq , X.Wk , X.Wv\nA = softmax( QK^T / sqrt(dk) )   (T x T)\nU = LN( X + (A.V).Wo )\nout = LN( U + FFN(U) )", f_bt, f_mono)

box(rx0, 900, rw, 150, NOTE, "Parameter budget",
    f"per SA layer : {per_sa:,}\n2 SA layers  : {two_sa:,}\nlm_head      : {head_p:,}\nNEW total    : {new_total:,}  (~{new_total/1e6:.1f}M)", f_bt, f_mono)

# ---------------- bottom worked example ----------------
ey = 1660
dr.rounded_rectangle([60, ey, W-60, ey+470], radius=16, fill="#F4F1FA", outline="#6A4FA3", width=3)
ctext(W/2, ey+40, "Worked example — one 10.0 s clip", f_title, "#3d2a6b")
ctext(W/2, ey+95,
      f"audio (1, {SAMPLES:,})  ->[CNN /320]->  embeddings (1, {T_ex}, {d})  ->[2x SA]->  (1, {T_ex}, {d})  "
      f"->[lm_head]->  logits (1, {T_ex}, {V})  ->[argmax]->  ids (1, {T_ex})  -> text", f_bs, "#2a2150")
ctext(W/2, ey+135,
      f"T = {T_ex} frames   ·   each frame = a {d}-dim vector   ·   each frame -> {V} class scores   ·   "
      f"one symbol decoded per 20 ms", f_bs, "#444")

dr.text((100, ey+195), "Per-frame target vs prediction (first frames of \"concord ...\"):", font=f_sec, fill="#3d2a6b", anchor="lm")
dr.text((100, ey+245), "frame :   1     2     3     4     5     6     7     8     9    ...   498    499", font=f_mono, fill="#222", anchor="lm")
dr.text((100, ey+285), "target:  <s>    c     o     n     c     o     r     d     |    ...  <fill>  <fill>", font=f_mono, fill="#222", anchor="lm")
dr.text((100, ey+325), "argmax:  <s>    c     o     n     c     o     r     d     |    ...  <fill>  <fill>", font=f_mono, fill="#222", anchor="lm")
dr.text((100, ey+380), "loss = mean over graded frames of  -log p(correct symbol)        (init ~ log(33) = 3.50)", font=f_monob, fill="#3d2a6b", anchor="lm")
dr.text((100, ey+420), "decode: drop <s>/</s>/<fill>, '|'->space   ->   \"concord returned to its place ...\"", font=f_mono, fill="#222", anchor="lm")

# ---------------- legend ----------------
leg = [("Data / IO", INPUT), ("Frozen HuBERT-XLarge", FROZEN), ("New self-attention", SA),
       ("New head", HEAD), ("Labels / loss", LOSS)]
lx = 70; lyb = H-40
for name,(fc,ec) in leg:
    dr.rounded_rectangle([lx, lyb, lx+26, lyb+20], radius=4, fill=fc, outline=ec, width=2)
    dr.text((lx+34, lyb+10), name, font=f_bs, fill=TXT, anchor="lm")
    lx += 300

img.save("xlarge_arch.png")
print("xlarge_arch.png written  size:", img.size)
print(f"per-SA={per_sa:,}  2SA={two_sa:,}  head={head_p:,}  new_total={new_total:,}  T_ex={T_ex}")
