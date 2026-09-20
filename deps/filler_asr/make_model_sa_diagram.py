"""
Architecture diagram generated to match the ACTUAL code in model_sa.py
(FillerHubertSAModel) -- module names, forward path, freezing, and the
parameter counts verified by the CPU dry run.
Output: model_sa_arch.png  (Pillow renderer, no matplotlib needed)
"""
from PIL import Image, ImageDraw, ImageFont

# ---- numbers (config defaults + dry-run-verified param counts) ----
d, V, Hh, dff = 1280, 33, 16, 5120
dk = d // Hh
N_SA = 2
TOT_M, TRAIN_M, SA_M, HEAD_P, HUB_M = 1001.89, 39.40, 39.35, 42273, 962.50

FD = "/usr/share/fonts/truetype/dejavu/"
def F(n, s): return ImageFont.truetype(FD + n, s)
f_title = F("DejaVuSans-Bold.ttf", 33)
f_sub   = F("DejaVuSans.ttf", 19)
f_bt    = F("DejaVuSans-Bold.ttf", 22)
f_bs    = F("DejaVuSans.ttf", 15)
f_mono  = F("DejaVuSansMono.ttf", 15)
f_monob = F("DejaVuSansMono-Bold.ttf", 16)
f_lab   = F("DejaVuSansMono-Bold.ttf", 15)
f_pan   = F("DejaVuSans-Bold.ttf", 17)
f_sec   = F("DejaVuSans-Bold.ttf", 18)

INPUT=("#E8EEF7","#3B6CA8"); FROZEN=("#DCE3EA","#6B7785")
SA=("#DDEFD8","#3E7D32"); HEAD=("#FBE3C6","#C77A12")
LOSS=("#F6D9DC","#B23A48"); NOTE=("#FFFFFF","#9a9a9a"); CODE=("#F4F1FA","#6A4FA3")
OUTC=("#EAF7EE","#2f7d3a"); TXT="#1a1a1a"

W, Hpx = 1600, 2120
img = Image.new("RGB", (W, Hpx), "white"); dr = ImageDraw.Draw(img)

def ctext(cx, cy, s, font, fill=TXT, anchor="mm"): dr.text((cx, cy), s, font=font, fill=fill, anchor=anchor)
def box(x, y, w, h, colors, title, sub="", tfont=f_bt, sfont=f_bs):
    fc, ec = colors
    dr.rounded_rectangle([x, y, x+w, y+h], radius=13, fill=fc, outline=ec, width=3)
    if sub:
        ctext(x+w/2, y+h*0.32, title, tfont)
        ly = y+h*0.50
        for ln in sub.split("\n"):
            ctext(x+w/2, ly, ln, sfont, fill="#333"); ly += sfont.size+4
    else:
        ctext(x+w/2, y+h/2, title, tfont)
def panel(x, y, w, h, ec, label):
    dr.rounded_rectangle([x, y, x+w, y+h], radius=16, outline=ec, width=3)
    dr.text((x+13, y+8), label, font=f_pan, fill=ec, anchor="lm")
def arrow(x, y0, y1, label="", color=(43,43,43), labcol="#0b3d66"):
    dr.line([(x, y0), (x, y1-15)], fill=color, width=4)
    dr.polygon([(x, y1), (x-8, y1-15), (x+8, y1-15)], fill=color)
    if label:
        ty=(y0+y1)/2; bb=dr.textbbox((x+13, ty), label, font=f_lab, anchor="lm")
        dr.rectangle([bb[0]-4,bb[1]-3,bb[2]+4,bb[3]+3], fill="white")
        dr.text((x+13, ty), label, font=f_lab, fill=labcol, anchor="lm")

# ---- title ----
ctext(W/2, 38, "FillerHubertSAModel  —  architecture straight from model_sa.py", f_title, "#11304f")
ctext(W/2, 76, f"code-faithful  ·  hidden={d}  ·  {N_SA} self-attention layers (nhead={Hh}, ff={dff})  ·  "
               f"vocab={V}  ·  encoder FROZEN, SA+head trainable", f_sub, "#444")

cx=440; bw=620; x0=cx-bw//2; GAP=58
rects={}; y=120
def place(k,h):
    global y; rects[k]=(x0,y,bw,h); y+=h+GAP
def r(k): return rects[k]
def top(k): return r(k)[1]
def bot(k): return r(k)[1]+r(k)[3]

place("in", 64)
place("hub", 120)
place("sa", 120)
place("drop", 56)
place("head", 70)
place("out", 78)

# frozen panel around hubert
fx,fy,fw,fh = r("hub")[0]-16, r("hub")[1]-26, bw+32, r("hub")[3]+40
panel(fx,fy,fw,fh, FROZEN[1], "FROZEN  ·  self.hubert = HubertModel(config)")
# trainable panel around sa+drop+head
nx,ny,nw,nh = r("sa")[0]-16, r("sa")[1]-26, bw+32, (bot("head"))-r("sa")[1]+34
panel(nx,ny,nw,nh, SA[1], f"NEW · TRAINABLE   (self.sa + self.lm_head  =  {TRAIN_M:.2f}M params)")

box(*r("in"), INPUT, "input_values  (B, T_s)", "+ attention_mask   ·   16 kHz, normalized")
arrow(cx, bot("in"), top("hub"), "forward(input_values, attention_mask)")

box(*r("hub"), FROZEN, "self.hubert(...)   ->   outputs",
    "HubertModel: CNN feat-extractor (/320) + feature_projection\n"
    f"+ 48 transformer layers (16 heads, d={d})   [xlarge-ft, CTC head dropped]")
arrow(cx, bot("hub"), top("sa"), f"hidden_states = outputs[0]   (B, T, {d})   <- embedding tap",
      color=(178,58,72), labcol="#7a2730")

box(*r("sa"), SA, f"self.sa  =  nn.TransformerEncoder  (num_layers={N_SA})",
    f"TransformerEncoderLayer: MHA(d={d}, nhead={Hh}, dk={dk})\n"
    f"FFN {d}->{dff}->{d} GELU · post-norm · batch_first · pad-masked")
arrow(cx, bot("sa"), top("drop"), f"hidden_states = self.sa(hidden_states, src_key_padding_mask)   (B, T, {d})")

box(*r("drop"), SA, "self.dropout  =  nn.Dropout(p=0.1)", "")
arrow(cx, bot("drop"), top("head"), f"(B, T, {d})")

box(*r("head"), HEAD, "self.lm_head  =  nn.Linear(1280 -> 33)", f"{HEAD_P:,} params")
arrow(cx, bot("head"), top("out"), f"logits = self.lm_head(hidden_states)   (B, T, {V})")

box(*r("out"), LOSS, "return CausalLMOutput(loss=None, logits=...)",
    "loss=None  ->  framewise CrossEntropyLoss computed\nin FillerASRTrainer.compute_loss (ignore_index=-100)")

# ---------------- right column ----------------
rx=910; rw=620
def rbox(y,h,colors,title,sub,tf=f_bt,sf=f_bs): box(rx,y,rw,h,colors,title,sub,tf,sf)

ctext(rx+rw/2, 118, "__init__  (the 4 modules)", f_sec, "#11304f")
rbox(135, 128, NOTE, "",
     "self.hubert  = HubertModel(config)            # frozen encoder\n"
     "self.sa      = nn.TransformerEncoder(layer,2) # NEW\n"
     "self.dropout = nn.Dropout(final_dropout)\n"
     "self.lm_head = nn.Linear(hidden, vocab)       # NEW", sf=f_mono)

ctext(rx+rw/2, 295, "one self-attention layer (inside self.sa)", f_sec, "#3E7D32")
rbox(312, 150, SA, "",
     "nn.TransformerEncoderLayer(\n"
     f"   d_model={d}, nhead={Hh}, dk={dk},\n"
     f"   dim_feedforward={dff}, activation='gelu',\n"
     "   batch_first=True, norm_first=False)\n"
     "U = LN(x + MHA(x));  out = LN(U + FFN(U))", sf=f_mono)

ctext(rx+rw/2, 492, "freezing helpers (used by FillerASRTrainer)", f_sec, "#6B7785")
rbox(510, 158, NOTE, "",
     "freeze_feature_extractor()    -> CNN frozen\n"
     "freeze_all_except_classifier()-> phase A: lm_head only\n"
     "unfreeze_all_except_feature_extractor():\n"
     "   phase B: train self.sa + lm_head,\n"
     "   WHOLE hubert stays FROZEN (feature extractor)", sf=f_mono)

ctext(rx+rw/2, 700, "_init_weights override  (the NaN fix)", f_sec, "#B23A48")
rbox(718, 128, LOSS, "",
     "HuBERT _init_weights skips nn.MultiheadAttention\n"
     "-> from_pretrained left self_attn.in_proj_bias = NaN.\n"
     "Override xavier-inits in_proj_weight / in_proj_bias\n"
     "(+ zeros bias) so the SA layers start finite.", sf=f_mono)

ctext(rx+rw/2, 878, "parameter budget  (dry-run verified)", f_sec, "#3d2a6b")
rbox(896, 150, CODE, "",
     f"total       = {TOT_M:.2f} M\n"
     f"trainable   = {TRAIN_M:.2f} M   (sa {SA_M:.2f}M + head {HEAD_P/1e3:.1f}K)\n"
     f"frozen hubert = {HUB_M:.2f} M\n"
     "hubert params with grad = 0   (encoder frozen)", sf=f_mono)

# ---------------- bottom: actual forward() code ----------------
ey=1100
dr.rounded_rectangle([60, ey, W-60, ey+330], radius=14, fill=CODE[0], outline=CODE[1], width=3)
ctext(W/2, ey+34, "def forward(...)  —  verbatim path from model_sa.py", f_bt, "#3d2a6b")
code = [
 "outputs       = self.hubert(input_values, attention_mask=attention_mask, ...)",
 "hidden_states = outputs[0]                                      # (B, T, 1280)  embedding tap",
 "feat_mask     = self.hubert._get_feature_vector_attention_mask(T, attention_mask)",
 "src_key_padding_mask = ~feat_mask.bool()                        # True = padded frame -> ignore",
 "hidden_states = self.sa(hidden_states, src_key_padding_mask=src_key_padding_mask)",
 "hidden_states = self.dropout(hidden_states)",
 "logits        = self.lm_head(hidden_states)                     # (B, T, 33)",
 "return CausalLMOutput(loss=None, logits=logits, ...)",
]
yy=ey+70
for ln in code:
    dr.text((90, yy), ln, font=f_mono, fill="#222", anchor="lm"); yy+=31

# ---------------- legend ----------------
leg=[("Data / IO",INPUT),("Frozen HuBERT-XLarge",FROZEN),("New self-attention",SA),
     ("New head",HEAD),("Labels / loss",LOSS)]
lx=70; lyb=Hpx-42
for name,(fc,ec) in leg:
    dr.rounded_rectangle([lx,lyb,lx+26,lyb+20], radius=4, fill=fc, outline=ec, width=2)
    dr.text((lx+34,lyb+10), name, font=f_bs, fill=TXT, anchor="lm"); lx+=300

img.save("model_sa_arch.png")
print("model_sa_arch.png written", img.size)
