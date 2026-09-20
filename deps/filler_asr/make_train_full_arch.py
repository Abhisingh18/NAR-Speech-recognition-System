"""Full TRAINING-pipeline schematic for experiment Hubert_SA_2gpu_lazy_lr2e3.

Everything included: audio + transcript inputs, lazy data/label prep (positional
<fill> target + duration-budget mask), the frozen HuBERT-xlarge body, the trainable
2 SA layers + lm_head, the framewise-CE loss, and this run's hyperparameters.
Mirrors model_sa.py / data_local.py / collator.py / config_xlarge_sa_2gpu_lazy.json.
Output: results/train_full_arch.png
"""
import os
from PIL import Image, ImageDraw, ImageFont

d, V, H, dff = 1280, 33, 16, 5120
dk = d // H

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
def f(p, s): return ImageFont.truetype(p, s)
ft  = f(FB, 30); fh = f(FB, 20); fr = f(FR, 16); fm = f(FM, 14); fs = f(FR, 13); fsb = f(FB, 15)

C_BG=(250,250,252); C_TXT=(25,25,30); C_GREY=(110,110,115)
C_FROZEN=(210,222,240); C_FRBRD=(90,120,165)
C_TRAIN=(212,236,212);  C_TRBRD=(60,140,60)
C_SUB=(255,248,225);    C_SUBBRD=(200,165,70)
C_DATA=(232,226,244);   C_DATABRD=(120,95,170)
C_LOSS=(247,219,226);   C_LOSSBRD=(185,80,110)
C_IO=(235,235,235);     C_IOBRD=(120,120,120)
C_RES=(190,60,60)

W, Hh = 1560, 2360
img = Image.new("RGB", (W, Hh), C_BG)
dr = ImageDraw.Draw(img)

def ctext(cx, y, s, font, fill=C_TXT, anchor="mm"):
    dr.text((cx, y), s, font=font, fill=fill, anchor=anchor)
def ltext(x, y, s, font, fill=C_TXT):
    dr.text((x, y), s, font=font, fill=fill, anchor="lm")
def box(x0,y0,x1,y1,fill,brd,w=2,r=12):
    dr.rounded_rectangle([x0,y0,x1,y1], radius=r, fill=fill, outline=brd, width=w)
def varrow(x,y0,y1,color=(60,60,65),w=3):
    dr.line([x,y0,x,y1], fill=color, width=w)
    dr.polygon([(x-7,y1-12),(x+7,y1-12),(x,y1)], fill=color)

def stackbox(cx, w, y, header, lines, fill, brd, tag=None, lh=21, padtop=30, padbot=12):
    """Draw a box centered at cx with bold header + body lines. Returns bottom y."""
    h = padtop + lh*len(lines) + padbot
    x0, x1 = cx-w//2, cx+w//2
    box(x0, y, x1, y+h, fill, brd)
    ctext(cx, y+16, header, fh)
    yy = y+padtop+8
    for ln in lines:
        ctext(cx, yy, ln, fr); yy += lh
    if tag:
        tw = dr.textlength(tag, font=fsb)
        box(x1-tw-16, y-11, x1+2, y+13, brd, brd, r=8)
        ctext(x1-tw//2-7, y+1, tag, fsb, (255,255,255))
    return y+h

# ============================ title ============================
ctext(W//2, 36, "filler_asr — TRAINING pipeline   (exp: Hubert_SA_2gpu_lazy_lr2e3)", ft)
ctext(W//2, 66, "FillerHubertSAModel  ·  frozen HuBERT-xlarge tap + 2 self-attn layers + lm_head  ·  framewise CE on positional <fill> target", f(FR,15), C_GREY)

CXM = 690          # main (audio→model) column center
WM  = 600
CXL = 240          # left (text→labels) column center
WL  = 380

# ============================ LEFT: data / label prep ============================
ctext(CXL, 100, "DATA PREP  (data_local.py, lazy)", fsb, C_DATABRD)
yl = 116
yl = stackbox(CXL, WL, yl, "Transcript (jsonl 'text')", ['"HELLO I AM TOM"'], C_IO, C_IOBRD)
varrow(CXL, yl, yl+24); yl += 24
yl = stackbox(CXL, WL, yl, "normalize", ["lowercase, strip [,?.!-;:\"]", "spaces -> '|'  =>  hello|i|am|tom"], C_DATA, C_DATABRD)
varrow(CXL, yl, yl+24); yl += 24
yl = stackbox(CXL, WL, yl, "char tokenize (33-vocab)", ["h e l l o | i | a m | t o m", "ids = [8,5,12,12,15,27,9,..]"], C_DATA, C_DATABRD)
varrow(CXL, yl, yl+24); yl += 24
yl = stackbox(CXL, WL, yl, "_labels_for(text, T)", ["[<s>] + ids + [</s>]", "+ <fill>(32) x (T - len)", "=> positional left-pack"], C_DATA, C_DATABRD)
varrow(CXL, yl, yl+24); yl += 24
yl = stackbox(CXL, WL, yl, "duration-budget mask (collator)", ["n_keep = ceil(dur_s x 25)", "labels[n_keep:] = -100  (ignore)"], C_DATA, C_DATABRD)
varrow(CXL, yl, yl+24); yl += 24
yl = stackbox(CXL, WL, yl, "labels  (B, T)", ["target ids per frame; -100 ignored"], C_SUB, C_SUBBRD)
labels_bottom = yl

# ============================ MAIN: audio -> model ============================
ym = 116
ym = stackbox(CXM, WM, ym, "Audio (jsonl 'audio')", ["WAV 16 kHz mono"], C_IO, C_IOBRD)
varrow(CXM, ym, ym+24); ym += 24
ym = stackbox(CXM, WM, ym, "Wav2Vec2FeatureExtractor", ["do_normalize (zero-mean/unit-var)", "-> input_values (B, samples)"], C_SUB, C_SUBBRD)
varrow(CXM, ym, ym+26); ym += 26

# frozen body container
fb_top = ym
ym = stackbox(CXM, WM, ym, "Conv Feature Extractor", ["7 x Conv1d  k[10,3,3,3,3,2,2]  s[5,2,2,2,2,2,2]", "320x downsample  ->  ~50 fps (20 ms/frame)"], C_FROZEN, C_FRBRD, tag="FROZEN")
varrow(CXM, ym, ym+18); ym += 18
ym = stackbox(CXM, WM, ym, "Feature Projection", ["LayerNorm + Linear 512 -> 1280 + dropout"], C_FROZEN, C_FRBRD, tag="FROZEN")
varrow(CXM, ym, ym+18); ym += 18
ym = stackbox(CXM, WM, ym, "Pos Conv Embed", ["Conv1d k=128, groups=16  (relative pos)"], C_FROZEN, C_FRBRD, tag="FROZEN")
varrow(CXM, ym, ym+18); ym += 18
ym = stackbox(CXM, WM, ym, "Transformer Encoder  x 48", ["each: MHSA(16 heads) + Add&Norm", "FFN 1280 -> 5120 -> 1280 (GELU) + Add&Norm"], C_FROZEN, C_FRBRD, tag="FROZEN")
fb_bot = ym
# frozen bracket
dr.line([CXM-WM//2-12, fb_top, CXM-WM//2-12, fb_bot], fill=C_FRBRD, width=3)
dr.text((CXM-WM//2-18, (fb_top+fb_bot)//2), "HuBERT-xlarge-ls960-ft  (CTC head dropped)",
        font=fsb, fill=C_FRBRD, anchor="mm")  # rotated-ish label placed left
varrow(CXM, ym, ym+26); ym += 26
ym = stackbox(CXM, WM, ym, "TAP: last_hidden_state", ["hidden  (B, T, 1280)"], C_IO, C_IOBRD)
varrow(CXM, ym, ym+26); ym += 26

# trainable head container
tr_top = ym
ym = stackbox(CXM, WM, ym, "Self-Attention layer  x 2   (post-norm)", [
    "MultiheadAttention  16 heads (dk=80)  + Add&Norm",
    "FFN  Linear 1280->5120 -> GELU -> 5120->1280  + Add&Norm",
    "src_key_padding_mask from feature attn-mask"], C_TRAIN, C_TRBRD, tag="TRAINABLE")
varrow(CXM, ym, ym+18); ym += 18
ym = stackbox(CXM, WM, ym, "Dropout(0.1)", ["final_dropout"], C_TRAIN, C_TRBRD, tag="TRAINABLE")
varrow(CXM, ym, ym+18); ym += 18
ym = stackbox(CXM, WM, ym, "lm_head", ["Linear 1280 -> 33  (reinit; base CTC head dropped)"], C_TRAIN, C_TRBRD, tag="TRAINABLE")
tr_bot = ym
dr.line([CXM-WM//2-12, tr_top, CXM-WM//2-12, tr_bot], fill=C_TRBRD, width=3)
varrow(CXM, ym, ym+26); ym += 26
ym = stackbox(CXM, WM, ym, "logits", ["(B, T, 33)  -> argmax per frame at decode"], C_IO, C_IOBRD)

# ============================ LOSS ============================
varrow(CXM, ym, ym+30); ym += 30
loss_top = ym
ym = stackbox(CXM, WM, ym, "framewise CROSS-ENTROPY  loss", [
    "CrossEntropyLoss(ignore_index=-100)",
    "L = mean over graded frames of  -log softmax(logits)[label]",
    "grad -> ONLY 2 SA layers + lm_head  (~39.4M, ~4%)"], C_LOSS, C_LOSSBRD)
loss_cy = (loss_top+ym)//2

# elbow arrow: labels (left col) -> loss box
ly = labels_bottom
dr.line([CXL, ly, CXL, loss_cy], fill=C_DATABRD, width=3)
dr.line([CXL, loss_cy, CXM-WM//2, loss_cy], fill=C_DATABRD, width=3)
dr.polygon([(CXM-WM//2-12,loss_cy-7),(CXM-WM//2-12,loss_cy+7),(CXM-WM//2,loss_cy)], fill=C_DATABRD)
ctext((CXL+CXM-WM//2)//2, loss_cy-12, "labels (B,T)", fs, C_DATABRD)

# ============================ CONFIG + LEGEND panel ============================
py = ym + 40
box(50, py, W-50, py+255, (255,255,255), (180,180,185))
ctext(W//2, py+22, "Training configuration  (config_xlarge_sa_2gpu_lazy.json + checkpoint)", fh)
col1 = [
    "base: facebook/hubert-xlarge-ls960-ft (HubertForCTC, vocab 32)",
    "vocab: 33  (a-z, ', |, <unk>,<pad>,<s>,</s>, <fill>=32)",
    "frame rate ~50 fps (20 ms); label budget 20 tok/s",
    "loss: framewise CE, ignore_index = -100",
    "freeze_feature_extractor = True; whole 48-layer body frozen",
]
col2 = [
    "optim: AdamW  lr 2e-3, linear sched, warmup 1000, wd 0.005",
    "batch: 32 / GPU x 2 GPU x 4 grad-accum  = eff 256",
    "epochs 30 ; eval & save every 1000 steps",
    "data: LibriSpeech 960h  (train 281,241 ; dev 500)",
    "trainable: 2 SA layers + lm_head  ~39.4M  (~4%)",
]
yy = py+52
for a, b in zip(col1, col2):
    ltext(70, yy, "• "+a, fr); ltext(W//2+20, yy, "• "+b, fr); yy += 26

# legend
ly2 = py+218
def chip(x, c, b, label):
    box(x, ly2, x+26, ly2+18, c, b); ltext(x+34, ly2+9, label, fs)
chip(70,  C_FROZEN, C_FRBRD, "frozen (HuBERT body)")
chip(330, C_TRAIN,  C_TRBRD, "trainable (SA + lm_head)")
chip(640, C_DATA,   C_DATABRD,"data / label prep")
chip(900, C_LOSS,   C_LOSSBRD,"loss / target")
chip(1140,C_SUB,    C_SUBBRD, "tensor / sub-op")

out = "results/train_full_arch.png"
os.makedirs("results", exist_ok=True)
img.crop((0, 0, W, py+275)).save(out)
print("wrote", out, "size", img.crop((0,0,W,py+275)).size)
