"""Detailed, non-overlapping schematic of the IMPLEMENTED filler_asr SA training.

frozen HuBERT-xlarge tap -> sinusoidal PE (add) -> 2 self-attention layers EXPANDED
(MHSA + residual + LayerNorm + FFN + residual + LayerNorm) -> dropout -> lm_head ->
logits -> framewise CE. Shows the two distinct maskings (frame mask in attention,
loss mask in CE) and the positional <fill> target + duration budget.
Output: results/arch_detailed.png
"""
import os, math
from PIL import Image, ImageDraw, ImageFont

FB="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
def f(p,s): return ImageFont.truetype(p,s)
ftitle=f(FB,32); fhdr=f(FB,19); fbody=f(FR,16); fmono=f(FM,14); fsm=f(FR,13); fchip=f(FB,14)

BG=(250,250,252); TXT=(22,22,28); GREY=(120,120,126)
FROZEN=(208,221,241); FROZEN_B=(78,112,162)
TRAIN=(210,235,210); TRAIN_B=(56,138,56)
PE=(228,219,246); PE_B=(120,92,176)
RES=(225,128,40)
FMASK=(206,236,234); FMASK_B=(38,148,140)
LMASK=(250,222,236); LMASK_B=(196,80,140)
LOSS=(250,214,205); LOSS_B=(208,86,58)
DATA=(247,238,214); DATA_B=(188,150,60)
IO=(236,236,238); IO_B=(120,120,126)
NORM=(255,247,222); NORM_B=(206,170,72)

W,H=1780,1980
img=Image.new("RGB",(W,H),BG); dr=ImageDraw.Draw(img)

def ctext(cx,y,s,fo,fill=TXT): dr.text((cx,y),s,font=fo,fill=fill,anchor="mm")
def ltext(x,y,s,fo,fill=TXT): dr.text((x,y),s,font=fo,fill=fill,anchor="lm")
def rrect(x0,y0,x1,y1,fill,brd,w=2,r=12): dr.rounded_rectangle([x0,y0,x1,y1],radius=r,fill=fill,outline=brd,width=w)
def down(x,y0,y1,color=(60,60,66),w=3):
    dr.line([x,y0,x,y1],fill=color,width=w); dr.polygon([(x-7,y1-12),(x+7,y1-12),(x,y1)],fill=color)
def plus(cx,cy,r=17,color=(50,50,56)):
    dr.ellipse([cx-r,cy-r,cx+r,cy+r],fill=(255,255,255),outline=color,width=3)
    dr.line([cx-9,cy,cx+9,cy],fill=color,width=3); dr.line([cx,cy-9,cx,cy+9],fill=color,width=3)
def chip(x1,y_top,label,brd):
    tw=dr.textlength(label,font=fchip); rrect(x1-tw-18,y_top-12,x1+2,y_top+14,brd,brd,r=8)
    ctext(x1-tw//2-8,y_top+1,label,fchip,(255,255,255))
def vbox(cx,w,y,header,lines,fill,brd,tag=None,tagbrd=None,lh=21,pt=30,pb=12):
    h=pt+lh*len(lines)+pb; x0,x1=cx-w//2,cx+w//2
    rrect(x0,y,x1,y+h,fill,brd); ctext(cx,y+16,header,fhdr); yy=y+pt+7
    for ln in lines: ctext(cx,yy,ln,fbody); yy+=lh
    if tag: chip(x1,y,tag,tagbrd or brd)
    return y+h,(x0,x1)

# ---------------- title ----------------
ctext(W//2,34,"filler_asr SA model — IMPLEMENTED training (detailed)",ftitle)
ctext(W//2,66,"frozen HuBERT-xlarge tap  +  sinusoidal PE  +  2 self-attention layers  +  lm_head   •   framewise cross-entropy",f(FR,15),GREY)

CX=720; SW=440
RES_LANE1=CX+SW//2+46
RES_LANE2=CX+SW//2+92

# ---------------- spine: input -> frozen body -> tap ----------------
y=104
y,_=vbox(CX,SW,y,"Audio input",["WAV 16 kHz mono  ->  (B, samples)"],IO,IO_B)
down(CX,y,y+30); y+=30
y,_=vbox(CX,SW,y,"Wav2Vec2FeatureExtractor",["normalize (zero-mean / unit-var)  ->  input_values"],NORM,NORM_B)
down(CX,y,y+30); y+=30
y,_=vbox(CX,SW,y,"HuBERT-xlarge body  (CTC head dropped)",
         ["7x Conv feat-extractor  ->  ~50 fps (20 ms / frame)",
          "feature projection (LN + Linear 512->1280)",
          "pos-conv embed (Conv1d k=128, g=16)",
          "48x Transformer encoder layers"],FROZEN,FROZEN_B,tag="FROZEN",tagbrd=FROZEN_B)
down(CX,y,y+28); y+=28
y,_=vbox(CX,SW,y,"TAP  last_hidden_state",["hidden  (B, T, 1280)"],IO,IO_B)

# ---------------- sinusoidal PE add ----------------
down(CX,y,y+34); y+=34
pe_cy=y+17; plus(CX,pe_cy)
pe_x0=CX+140; pe_x1=pe_x0+360
rrect(pe_x0,pe_cy-46,pe_x1,pe_cy+46,PE,PE_B)
ctext((pe_x0+pe_x1)//2,pe_cy-28,"sinusoidal positional encoding",fhdr,PE_B)
ctext((pe_x0+pe_x1)//2,pe_cy-5,"PE[p,2i]=sin(p/10000^(2i/d))",fmono)
ctext((pe_x0+pe_x1)//2,pe_cy+15,"PE[p,2i+1]=cos(...)   fixed, + pe_scale",fsm)
sg=pe_x0+14; dr.line([(sg+k,pe_cy+33-int(6*math.sin(k/9.0))) for k in range(0,150,3)],fill=PE_B,width=2)
dr.line([pe_x0,pe_cy,CX+19,pe_cy],fill=PE_B,width=3); dr.polygon([(CX+19,pe_cy-7),(CX+19,pe_cy+7),(CX+4,pe_cy)],fill=PE_B)
ctext((CX+pe_x0)//2,pe_cy-13,"add",fsm,PE_B)
y=pe_cy+17
down(CX,y,y+62); y+=62           # generous gap so the SA header clears the PE box

# ================= SA LAYER (expanded) x2 =================
sa_top=y
hb=34
rrect(CX-SW//2,y,CX+SW//2,y+hb,TRAIN,TRAIN_B)
ctext(CX,y+hb//2,"SELF-ATTENTION LAYER    x 2    (post-norm,  TRAINABLE)",fhdr,TRAIN_B)
y+=hb+20
li_x0,li_x1=CX-135,CX+135
rrect(li_x0,y,li_x1,y+28,IO,IO_B,r=10); ctext(CX,y+14,"layer input   x",fmono)
res_a_y=y+14; y+=28
down(CX,y,y+24); y+=24

mh_top=y
y,(mhx0,mhx1)=vbox(CX,SW,y,"Multi-Head Self-Attention",["16 heads (dk = 80)   +   dropout"],TRAIN,TRAIN_B)
mh_cy=(mh_top+y)//2
down(CX,y,y+26); y+=26
addA_cy=y+17; plus(CX,addA_cy); y=addA_cy+17
down(CX,y,y+22); y+=22
y,_=vbox(CX,SW,y,"Add & LayerNorm",["x1 = LayerNorm( x + MHSA(x) )"],NORM,NORM_B)
res_b_y=y-3
down(CX,y,y+26); y+=26
y,_=vbox(CX,SW,y,"Feed-Forward Network",["Linear 1280 -> 5120","GELU","Linear 5120 -> 1280   +   dropout"],TRAIN,TRAIN_B)
down(CX,y,y+26); y+=26
addB_cy=y+17; plus(CX,addB_cy); y=addB_cy+17
down(CX,y,y+22); y+=22
y,_=vbox(CX,SW,y,"Add & LayerNorm",["x2 = LayerNorm( x1 + FFN(x1) )"],NORM,NORM_B)
sa_bot=y

# residual / skip arrows (emanate from box right edges -> dedicated lanes -> into the (+) )
def residual(start_x,y_start,add_cy,lane,label):
    dr.line([start_x,y_start,lane,y_start],fill=RES,width=3)
    dr.line([lane,y_start,lane,add_cy],fill=RES,width=3)
    dr.line([lane,add_cy,CX+18,add_cy],fill=RES,width=3)
    dr.polygon([(CX+18,add_cy-7),(CX+18,add_cy+7),(CX+3,add_cy)],fill=RES)
    ctext(lane+62,(y_start+add_cy)//2,label,fsm,RES)
residual(li_x1, res_a_y, addA_cy, RES_LANE1, "residual / skip")
residual(CX+SW//2, res_b_y, addB_cy, RES_LANE2, "residual / skip")

# frame mask (LEFT of MHSA, upper-left)
fm_x1=CX-SW//2-58; fm_x0=fm_x1-350
rrect(fm_x0,mh_cy-48,fm_x1,mh_cy+48,FMASK,FMASK_B)
ctext((fm_x0+fm_x1)//2,mh_cy-29,"FRAME MASK  (attention)",fhdr,FMASK_B)
ctext((fm_x0+fm_x1)//2,mh_cy-6,"src_key_padding_mask",fmono)
ctext((fm_x0+fm_x1)//2,mh_cy+15,"= ~feature_attn_mask  (True = pad, ignored)",fsm)
dr.line([fm_x1,mh_cy,mhx0-2,mh_cy],fill=FMASK_B,width=3); dr.polygon([(mhx0-2,mh_cy-7),(mhx0-2,mh_cy+7),(mhx0+12,mh_cy)],fill=FMASK_B)

# ---------------- head ----------------
y=sa_bot
down(CX,y,y+32); y+=32
y,_=vbox(CX,SW,y,"Dropout(0.1)",["final_dropout"],TRAIN,TRAIN_B,tag="TRAINABLE",tagbrd=TRAIN_B)
down(CX,y,y+26); y+=26
y,_=vbox(CX,SW,y,"lm_head",["Linear 1280 -> 33"],TRAIN,TRAIN_B,tag="TRAINABLE",tagbrd=TRAIN_B)
down(CX,y,y+26); y+=26
y,_=vbox(CX,SW,y,"logits",["(B, T, 33)"],IO,IO_B)

# ---------------- loss ----------------
down(CX,y,y+34); y+=34
loss_top=y
y,(lx0,lx1)=vbox(CX,SW,y,"framewise CROSS-ENTROPY",
        ["CE(ignore_index = -100)","L = mean over GRADED frames of","   -log softmax(logits)[ label ]"],LOSS,LOSS_B)
loss_cy=(loss_top+y)//2

# ================= LEFT: target prep + loss mask (lower-left, below frame mask) =================
LX=235; LW=360
ly=mh_cy+150
ctext(LX,ly-26,"TARGET / LABEL PREP   (data_local.py)",fchip,DATA_B)
ly,_=vbox(LX,LW,ly,"transcript -> chars",['"hello i am tom" -> hello|i|am|tom'],DATA,DATA_B)
down(LX,ly,ly+22); ly+=22
ly,_=vbox(LX,LW,ly,"_labels_for(text, T)  [positional]",["[<s>] + char_ids + [</s>]","+ <fill>(32) x (T - len)"],DATA,DATA_B)
down(LX,ly,ly+22); ly+=22
ly,_=vbox(LX,LW,ly,"LOSS MASK  (duration budget)",["n_keep = ceil(dur_s x 25)","labels[n_keep:] = -100 ;  pad = -100"],LMASK,LMASK_B)
down(LX,ly,ly+22); ly+=22
ly,_=vbox(LX,LW,ly,"labels  (B, T)",["graded ids   |   -100 = ignored"],IO,IO_B)
labels_bot=ly

dr.line([LX,labels_bot,LX,loss_cy],fill=LMASK_B,width=3)
dr.line([LX,loss_cy,lx0-2,loss_cy],fill=LMASK_B,width=3)
dr.polygon([(lx0-2,loss_cy-7),(lx0-2,loss_cy+7),(lx0+12,loss_cy)],fill=LMASK_B)
ctext((LX+lx0)//2,loss_cy-13,"targets  +  loss-mask",fsm,LMASK_B)

# gradient note
down(CX,y,y+30); y+=30
rrect(CX-SW//2,y,CX+SW//2,y+44,(255,255,255),(170,170,176))
ctext(CX,y+22,"gradient flows to  ONLY  the 2 SA layers + lm_head   (~39.4M, ~4%)   —   PE is fixed, HuBERT frozen",f(FR,14))
y+=44

# ---------------- legend ----------------
ly2=y+34
rrect(50,ly2,W-50,ly2+128,(255,255,255),(178,178,184))
ctext(W//2,ly2+22,"legend",fhdr)
def lchip(x,yy,c,b,label): rrect(x,yy-11,x+26,yy+11,c,b); ltext(x+34,yy,label,fsm)
r1=ly2+60
lchip(80,r1,FROZEN,FROZEN_B,"frozen (HuBERT body)"); lchip(380,r1,TRAIN,TRAIN_B,"trainable (SA + lm_head)")
lchip(720,r1,PE,PE_B,"sinusoidal PE (add)");        lchip(1010,r1,NORM,NORM_B,"Add & LayerNorm")
r2=ly2+98
lchip(80,r2,FMASK,FMASK_B,"frame mask (attention)"); lchip(380,r2,LMASK,LMASK_B,"loss mask (CE ignore -100)")
lchip(720,r2,LOSS,LOSS_B,"loss");                    lchip(1010,r2,DATA,DATA_B,"target / label prep")
dr.line([1300,r1,1342,r1],fill=RES,width=4); ltext(1352,r1,"residual / skip",fsm)

out="results/arch_detailed.png"; os.makedirs("results",exist_ok=True)
img.crop((0,0,W,ly2+150)).save(out)
print("wrote",out,"size",img.crop((0,0,W,ly2+150)).size)
