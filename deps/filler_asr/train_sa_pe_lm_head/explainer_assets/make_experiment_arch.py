"""
Experiment schematic (Pillow, repo house style): the two deltas vs the mask baseline
  (1) </s> emitted on 3 consecutive end frames (was 1)  -> sharper content/<fill> boundary
  (2) framewise CE weighted per class: w=1 everywhere, w=0.1 on <fill>
Everything else identical: frozen HuBERT + sinusoidal PE + N-SA + lm_head, 25-frame budget.
Renders: experiment_target_loss.png
"""
from PIL import Image, ImageDraw, ImageFont
FB="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
def f(p,s): return ImageFont.truetype(p,s)
ftitle=f(FB,30); fsub=f(FR,18); fhead=f(FB,20); fbody=f(FR,17); fmono=f(FM,15); fsmall=f(FR,14); fchip=f(FB,15)
C_BG=(250,250,252); C_TXT=(25,25,30); C_GREY=(110,110,115)
C_CHAR=(215,238,215); C_CHARB=(70,140,70)
C_EOS=(255,206,206); C_EOSB=(200,50,50)
C_FILLG=(255,233,204); C_FILLGB=(210,130,40)
C_FILLI=(226,226,230); C_FILLIB=(150,150,158)
C_BUD=(120,90,170); C_RED=(200,40,40); C_KEEP=(60,140,70)

# geometry (4.0 s clip): T=199, n_keep=100, ~58 chars, now +3 </s>
T=199; NKEEP=100; BOS=1; CHARS=58; EOSN=3
CONTENT=BOS+CHARS+EOSN            # 62
FILL_G=NKEEP-CONTENT             # 38
FILL_I=T-NKEEP                    # 99
W,H=1520,760
img=Image.new("RGB",(W,H),C_BG); dr=ImageDraw.Draw(img)
def ct(x,y,s,fnt,fill=C_TXT,a="mm"): dr.text((x,y),s,font=fnt,fill=fill,anchor=a)
ct(W//2,38,"Experiment:  3× </s> boundary   +   <fill>-weighted CE (w=0.1)",ftitle)
ct(W//2,70,"unchanged: frozen HuBERT-xlarge + sinusoidal PE + N×SA + lm_head · 25-frame duration budget",fsub,fill=C_GREY)

BX0,BX1=80,W-80; BW=BX1-BX0
def fx(fr): return BX0+BW*fr/T
BY0,BY1=170,232
def seg(a,b,fill,brd,label=None,tcol=C_TXT,bw=2):
    x0,x1=fx(a),fx(b); dr.rectangle([x0,BY0,x1,BY1],fill=fill,outline=brd,width=bw)
    if label and (x1-x0)>34: ct((x0+x1)/2,(BY0+BY1)/2,label,fbody,fill=tcol)
seg(0,BOS,C_CHAR,C_CHARB)
seg(BOS,BOS+CHARS,C_CHAR,C_CHARB,f"{CHARS} char frames (transcript)")
seg(BOS+CHARS,CONTENT,C_EOS,C_EOSB,None,bw=3)                      # 3 </s>
seg(CONTENT,NKEEP,C_FILLG,C_FILLGB,f"<fill> ×{FILL_G}")
seg(NKEEP,T,C_FILLI,C_FILLIB,f"<fill> ×{FILL_I}  (beyond budget -> -100)",C_GREY)
ct(fx(0.5),BY0-15,"<s>",fsmall,fill=C_CHARB)
ct(fx((BOS+CHARS+CONTENT)/2),BY0-15,"</s>×3",fchip,fill=C_EOSB)
for fr in [0,CONTENT,NKEEP,T]:
    x=fx(fr); dr.line([x,BY1,x,BY1+7],fill=C_GREY,width=2); ct(x,BY1+20,str(fr),fmono,fill=C_GREY)
xk=fx(NKEEP); dr.line([xk,BY0-36,xk,BY1+34],fill=C_BUD,width=3)
ct(xk,BY0-50,"n_keep = 100  (25-frame budget, unchanged)",fchip,fill=C_BUD)

# ---- per-frame CE WEIGHT row ----
WY0,WY1=300,344
ct(BX0-14,(WY0+WY1)/2,"CE weight",fsmall,fill=C_TXT,a="rm")
dr.rectangle([fx(0),WY0,fx(CONTENT),WY1],fill=(225,244,225),outline=C_CHARB,width=2)
ct(fx(CONTENT/2),(WY0+WY1)/2,"w = 1   (content + </s>)",fbody,fill=C_CHARB)
dr.rectangle([fx(CONTENT),WY0,fx(NKEEP),WY1],fill=(255,238,214),outline=C_FILLGB,width=2)
ct(fx((CONTENT+NKEEP)/2),(WY0+WY1)/2,"w = 0.1",fbody,fill=C_FILLGB)
dr.rectangle([fx(NKEEP),WY0,fx(T),WY1],fill=(238,238,240),outline=C_FILLIB,width=2)
ct(fx((NKEEP+T)/2),(WY0+WY1)/2,"ignored (-100)",fbody,fill=C_GREY)

# ---- loss formula ----
ct(W//2,400,"L  =  Σₜ  w[yₜ] · (−log softmax(zₜ)[yₜ])   /   Σₜ w[yₜ]        w=1 (content,</s>) · w=0.1 (<fill>)",fmono,fill=C_TXT)
ct(W//2,430,"∂L/∂zₜ[c] = (w[yₜ]/D)·(pₜ[c] − 1[c=yₜ]) ,   D = |content| + 0.1·|fill|",fmono,fill=C_TXT)

# ---- change callouts ----
def callout(x0,y0,x1,y1,title,lines):
    dr.rounded_rectangle([x0,y0,x1,y1],radius=12,fill=(255,240,240),outline=C_RED,width=2)
    ct(x0+16,y0+20,title,fchip,fill=C_RED,a="lm")
    for i,ln in enumerate(lines): ct(x0+16,y0+46+i*24,ln,fsmall,fill=C_TXT,a="lm")
callout(BX0,478,BX0+680,588,"CHANGE 1 — </s> ×3  (was ×1)",
        ["• +2 graded frames become </s> (fewer graded <fill>)",
         "• 3× the </s> gradient → crisper content→<fill> boundary",
         "• counteracts boundary-fuzzing from CHANGE 2 · decode still caps @ first </s>"])
callout(BX1-680,478,BX1,588,"CHANGE 2 — <fill> weight = 0.1",
        ["• fill frames: gradient × ~0.16  (≈6× weaker)",
         "• content frames: gradient × ~1.585  (denominator shrinks)",
         "• effective content LR ↑ ~1.6× → consider base-LR × 0.63 for clean A/B"])

# ---- verified strip ----
dr.rounded_rectangle([BX0,610,BX1,678],radius=12,fill=(238,244,238),outline=C_KEEP,width=2)
ct(BX0+16,634,"VERIFIED (verify_experiment.py, CPU):",fchip,fill=C_KEEP,a="lm")
ct(BX0+16,660,"eos_repeat=1 ≡ original target · budget mask unchanged (graded = first n_keep) · "
              "weighted-CE == Σwℓ/Σw · grads = (w/D)(p−1) · ignored frames grad 0",fsmall,fill=C_TXT,a="lm")
ct(W//2,712,"Stage 2 unaffected: strip is duration-based and never reads </s>; encoder rebuild loads sa.* unchanged.",fsmall,fill=C_GREY)
img.save("experiment_target_loss.png"); print("wrote experiment_target_loss.png",img.size)
