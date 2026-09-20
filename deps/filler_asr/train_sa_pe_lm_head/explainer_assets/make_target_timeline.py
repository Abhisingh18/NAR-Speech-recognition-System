"""
Label-construction + duration-budget timeline (Pillow, repo house style).

Shows, on a real 50-fps frame axis for a 4.0 s clip (T=199, n_keep=100), how the
positional <fill> target is laid out, which frames the CE loss grades, and how
Stage-2's strip keeps exactly the graded budget — the pure-<fill> tail is dropped.

Proportions use the measured LibriSpeech mean char-rate ~14.6 chars/s:
    content = [<s>] + ~58 chars + [</s>] = 60 frames  ->  ends at frame 59 (<< n_keep=100)
Renders: target_budget_timeline.png
"""
from PIL import Image, ImageDraw, ImageFont

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
def f(p, s): return ImageFont.truetype(p, s)
ftitle=f(FB,30); fsub=f(FR,18); fhead=f(FB,20); fbody=f(FR,17); fmono=f(FM,16); fsmall=f(FR,15); fchip=f(FB,15)

C_BG=(250,250,252); C_TXT=(25,25,30); C_GREY=(110,110,115)
C_BOS=(255,214,214); C_BOSB=(180,70,70)
C_CHAR=(215,238,215); C_CHARB=(70,140,70)
C_EOS=(255,214,214); C_EOSB=(180,70,70)
C_FILLG=(255,233,204); C_FILLGB=(210,130,40)      # graded fill (inside budget)
C_FILLI=(226,226,230); C_FILLIB=(150,150,158)     # ignored fill (beyond budget, -100)
C_KEEP=(60,140,70); C_DROP=(150,150,158); C_BUD=(120,90,170)

# ---- frame geometry (real numbers) ----
T = 199; NKEEP = 100
BOS, EOS = 1, 1
CHARS = 58                      # round(14.6 * 4)
CONTENT = BOS + CHARS + EOS     # 60
FILL_GRADED = NKEEP - CONTENT   # 40
FILL_IGN = T - NKEEP            # 99

W, H = 1500, 640
img = Image.new("RGB", (W, H), C_BG)
dr = ImageDraw.Draw(img)
def ct(x,y,s,fnt,fill=C_TXT,a="mm"): dr.text((x,y),s,font=fnt,fill=fill,anchor=a)

ct(W//2, 40, "Label construction + duration budget", ftitle)
ct(W//2, 74, "4.0 s clip · encoder T = compute_output_length(64000) = 199 frames (~50 fps) · n_keep = ⌈25·4⌉ = 100", fsub, fill=C_GREY)

# ---- the frame bar ----
BX0, BX1 = 90, W-90
BW = BX1 - BX0
BY0, BY1 = 200, 268
def fx(frame): return BX0 + BW * frame / T          # frame index -> x

# segment [start,end) frames -> draw
def seg(a, b, fill, brd, label=None, tcol=C_TXT):
    x0, x1 = fx(a), fx(b)
    dr.rectangle([x0, BY0, x1, BY1], fill=fill, outline=brd, width=2)
    if label and (x1 - x0) > 40:
        ct((x0+x1)/2, (BY0+BY1)/2, label, fbody, fill=tcol)

seg(0, BOS, C_BOS, C_BOSB)
seg(BOS, BOS+CHARS, C_CHAR, C_CHARB, f"{CHARS} char frames  (transcript)", C_TXT)
seg(BOS+CHARS, CONTENT, C_EOS, C_EOSB)
seg(CONTENT, NKEEP, C_FILLG, C_FILLGB, f"<fill> ×{FILL_GRADED}")
seg(NKEEP, T, C_FILLI, C_FILLIB, f"<fill> ×{FILL_IGN}  (beyond budget → labels = -100)", C_GREY)

# tiny <s> / </s> callouts
ct(fx(0.5), BY0-16, "<s>", fsmall, fill=C_BOSB)
ct(fx(CONTENT-0.5), BY0-16, "</s>", fsmall, fill=C_EOSB)

# frame ticks
for fr in [0, CONTENT, NKEEP, T]:
    x = fx(fr)
    dr.line([x, BY1, x, BY1+8], fill=C_GREY, width=2)
    ct(x, BY1+22, str(fr), fmono, fill=C_GREY)

# ---- n_keep budget line ----
xk = fx(NKEEP)
dr.line([xk, BY0-40, xk, BY1+40], fill=C_BUD, width=3)
ct(xk, BY0-56, "n_keep = 100  (duration budget, 25 fps)", fchip, fill=C_BUD)

# ---- GRADED / IGNORED brackets ----
def bracket(a, b, y, color, label):
    x0, x1 = fx(a), fx(b)
    dr.line([x0, y, x0, y+10], fill=color, width=3)
    dr.line([x1, y, x1, y+10], fill=color, width=3)
    dr.line([x0, y, x1, y], fill=color, width=3)
    ct((x0+x1)/2, y+30, label, fhead, fill=color)

bracket(0, NKEEP, 330, C_KEEP, "GRADED  [0,100)  → framewise CE  (60 content + 40 <fill>)")
bracket(NKEEP, T, 330, C_DROP, "IGNORED  [100,199)  → -100")

# ---- Stage-2 strip arrow (keeps the graded budget) ----
sy = 430
dr.line([fx(0), sy, fx(NKEEP), sy], fill=C_KEEP, width=6)
dr.polygon([(fx(NKEEP)-2, sy-8),(fx(NKEEP)-2, sy+8),(fx(NKEEP)+10, sy)], fill=C_KEEP)
ct((fx(0)+fx(NKEEP))/2, sy-20, "Stage-2 STRIP keeps [0,100)  — all content + short <fill> pad", fbody, fill=C_KEEP)
dr.line([fx(NKEEP), sy, fx(T), sy], fill=C_DROP, width=6)
ct((fx(NKEEP)+fx(T))/2, sy-20, "discarded (pure <fill>/silence tail)", fbody, fill=C_DROP)

# ---- token strip caption ----
ct(W//2, 486, "target  =  [<s>] + char_ids + [</s>] + <fill> × (T − len)      loss-mask: labels[n_keep:] = -100 , pad = -100",
   fmono, fill=C_TXT)

# ---- evidence callout box ----
ey0 = 520
dr.rounded_rectangle([BX0, ey0, BX1, ey0+86], radius=12, fill=(238,244,238), outline=C_CHARB, width=2)
ct(BX0+18, ey0+22, "VERIFIED on 25,000 real train utts (analyze_targets.py):", fchip, fill=C_CHARB, a="lm")
ct(BX0+18, ey0+48, "•  </s> graded 100%   •  transcript truncated 0%   •  content clipped 0%   •  utts with 0 graded <fill> 0%",
   fsmall, fill=C_TXT, a="lm")
ct(BX0+18, ey0+70, "•  <fill> = 41% of graded frames (balanced)   •  char-rate ≤ 22.98/s  <  25 fps budget  → left-packing is well-posed",
   fsmall, fill=C_TXT, a="lm")

img.save("target_budget_timeline.png")
print("wrote target_budget_timeline.png", img.size)
