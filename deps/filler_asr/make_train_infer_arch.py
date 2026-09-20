"""
Training vs Inference architecture schematics for the PROPOSED filler_asr SA
model (Pillow renderer). Shared trunk:
  input -> FROZEN HuBERT-XLarge tap -> + sinusoidal PE -> 2 SA layers
  -> dropout -> lm_head -> logits (B, T, 33),  T = compute_output_length ~ dur*50
Training head: fill targets to T, loss-mask to first n_keep=ceil(dur*25), CE.
Inference head: argmax -> crop to first n_keep -> strip fill/specials -> text.
Outputs: train_arch.png, inference_arch.png
"""
from PIL import Image, ImageDraw, ImageFont

d, V, H, dff = 1280, 33, 16, 5120
dk = d // H

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
def f(p, s): return ImageFont.truetype(p, s)
ft, fh, fr, fm, fs = f(FB,30), f(FB,22), f(FR,17), f(FM,15), f(FR,14)

C_BG=(250,250,252)
C_FROZEN=(210,222,240); C_FRBRD=(90,120,165)
C_TRAIN=(215,238,215);  C_TRBRD=(70,140,70)
C_SUB=(255,248,225);    C_SUBBRD=(200,165,70)
C_ADD=(245,222,230);    C_ADDBRD=(180,90,120)
C_NEW=(255,233,204);    C_NEWBRD=(210,130,40)
C_MASK=(226,216,240);   C_MASKBRD=(120,90,170)
C_INF=(214,235,238);    C_INFBRD=(60,140,150)
C_IO=(235,235,235);     C_IOBRD=(120,120,120)
C_TXT=(25,25,30); C_GREY=(110,110,115); C_RES=(190,60,60)

W = 1180
CX = W // 2


def build(mode):
    assert mode in ("train", "infer")
    img = Image.new("RGB", (W, 2900), C_BG)
    dr = ImageDraw.Draw(img)

    def ct(cx, y, s, font, fill=C_TXT, anchor="mm"):
        dr.text((cx, y), s, font=font, fill=fill, anchor=anchor)
    def box(x0,y0,x1,y1,fill,brd,w=2,r=14):
        dr.rounded_rectangle([x0,y0,x1,y1], radius=r, fill=fill, outline=brd, width=w)
    def varr(x,y0,y1,color=(60,60,65),w=3):
        dr.line([x,y0,x,y1], fill=color, width=w)
        dr.polygon([(x-7,y1-11),(x+7,y1-11),(x,y1)], fill=color)

    # ---- title ----
    title = "filler_asr  —  TRAINING" if mode=="train" else "filler_asr  —  INFERENCE"
    ct(CX, 38, title, ft)
    sub = ("proposed SA model: + sinusoidal PE,  duration frame-budget loss mask"
           if mode=="train" else
           "proposed SA model: + sinusoidal PE,  decode = crop to ceil(dur*25) frames")
    ct(CX, 68, sub, f(FM,15), C_GREY)

    y = 100
    box(CX-180,y,CX+180,y+50,C_IO,C_IOBRD)
    ct(CX,y+18,"input_values",fr); ct(CX,y+37,"raw waveform  (B, samples @16kHz)",fs,C_GREY)
    y+=50; varr(CX,y,y+34); y+=34

    # ---- frozen hubert ----
    fb=y
    box(CX-430,y,CX+430,y+300,C_FROZEN,C_FRBRD,w=3)
    ct(CX,y+24,"HuBERT-XLarge-ls960-ft   (FROZEN feature extractor)",fh,(40,60,100))
    ct(CX,y+47,"CTC head dropped  •  requires_grad = False",fs,C_GREY)
    iy=y+70
    box(CX-360,iy,CX+360,iy+46,(235,240,250),C_FRBRD,w=1)
    ct(CX,iy+15,"Conv feature extractor  (7 conv layers, stride /320)",fr)
    ct(CX,iy+33,"waveform -> (B, T, 1280)",fm,C_GREY)
    varr(CX,iy+46,iy+74)
    iy2=iy+74
    box(CX-360,iy2,CX+360,iy2+118,(235,240,250),C_FRBRD,w=1)
    ct(CX,iy2+20,"Transformer encoder  x48  (post-norm, residual)",fr)
    ct(CX,iy2+44,"each: MHSA(16 heads) -> Add&Norm -> FFNN -> Add&Norm",fs,C_GREY)
    ct(CX,iy2+66,"d_model = 1280   dff = 5120",fm,C_GREY)
    ct(CX,iy2+90,"==>  tap  last_hidden_state  (B, T, 1280),  T = ceil~ dur*50",f(FM,15),(40,60,100))
    y=fb+300; varr(CX,y,y+38); y+=38
    ct(CX-470,y-19,"hidden_states",fm,C_GREY,anchor="lm")

    # ---- (1) sinusoidal PE ----
    box(CX-390,y,CX+390,y+66,C_NEW,C_NEWBRD,w=3)
    ct(CX,y+21,"(1)  + Sinusoidal Positional Encoding   [NEW]",fr,(170,95,20))
    ct(CX,y+43,"h = hidden_states + PE(t)   ·   PE fixed (non-learned)   ·   (B, T, 1280)",fm,C_GREY)
    y+=66; varr(CX,y,y+32); y+=32

    # ---- SA stack ----
    tr=y
    SA_H=64+360+26+360+22
    box(CX-470,y,CX+470,y+SA_H,C_TRAIN,C_TRBRD,w=3)
    ct(CX,y+26,"NEW Self-Attention stack   x2   (TRAINABLE)",fh,(30,100,40))
    ct(CX,y+49,"nn.TransformerEncoder  •  post-norm (norm_first=False)",fs,C_GREY)

    def enc(cy,label):
        box(CX-410,cy,CX+410,cy+360,(228,244,228),C_TRBRD,w=2)
        ct(CX,cy+22,label,fh,(30,100,40))
        in_y=cy+42
        s1=cy+52
        box(CX-300,s1,CX+300,s1+58,C_SUB,C_SUBBRD,w=2)
        ct(CX,s1+18,"Multi-Head Self-Attention",fr)
        ct(CX,s1+39,f"H={H} heads,  d_k={dk}   (Q,K,V,O: 1280x1280)",fm,C_GREY)
        varr(CX,s1+58,s1+86)
        a1=s1+86
        box(CX-300,a1,CX+300,a1+44,C_ADD,C_ADDBRD,w=2); ct(CX,a1+22,"Add  &  LayerNorm",fr)
        varr(CX,a1+44,a1+72)
        rx=CX-345
        dr.line([(CX,in_y),(rx,in_y)],fill=C_RES,width=2)
        dr.line([(rx,in_y),(rx,a1+22)],fill=C_RES,width=2)
        dr.line([(rx,a1+22),(CX-300,a1+22)],fill=C_RES,width=2)
        dr.polygon([(CX-301,a1+16),(CX-301,a1+28),(CX-289,a1+22)],fill=C_RES)
        ct(rx-6,(in_y+a1+22)//2,"residual",fs,C_RES,anchor="rm")
        mid=a1+72; s2=a1+72
        box(CX-300,s2,CX+300,s2+58,C_SUB,C_SUBBRD,w=2)
        ct(CX,s2+18,"FFNN   (position-wise)",fr)
        ct(CX,s2+39,"Linear 1280->5120 -> GELU -> Linear 5120->1280",fm,C_GREY)
        varr(CX,s2+58,s2+86)
        a2=s2+86
        box(CX-300,a2,CX+300,a2+44,C_ADD,C_ADDBRD,w=2); ct(CX,a2+22,"Add  &  LayerNorm",fr)
        rx2=CX+345
        dr.line([(CX,mid),(rx2,mid)],fill=C_RES,width=2)
        dr.line([(rx2,mid),(rx2,a2+22)],fill=C_RES,width=2)
        dr.line([(rx2,a2+22),(CX+300,a2+22)],fill=C_RES,width=2)
        dr.polygon([(CX+301,a2+16),(CX+301,a2+28),(CX+289,a2+22)],fill=C_RES)
        ct(rx2+6,(mid+a2+22)//2,"residual",fs,C_RES,anchor="lm")
        return cy+360

    b=enc(y+64,"Encoder layer 1"); varr(CX,b,b+26); b=enc(b+26,"Encoder layer 2")
    y=tr+SA_H; varr(CX,y,y+32); y+=32

    # ---- dropout ----
    box(CX-220,y,CX+220,y+44,C_TRAIN,C_TRBRD,w=2)
    if mode=="train":
        ct(CX,y+22,"Dropout  (final_dropout, active)",fr)
    else:
        ct(CX,y+22,"Dropout  (eval: disabled / identity)",fr,C_GREY)
    y+=44; varr(CX,y,y+28); y+=28

    # ---- lm_head ----
    box(CX-260,y,CX+260,y+54,C_TRAIN,C_TRBRD,w=3)
    ct(CX,y+18,"lm_head   (TRAINABLE)",fr,(30,100,40))
    ct(CX,y+38,f"Linear  1280 -> {V}   (vocab)",fm,C_GREY)
    y+=54; varr(CX,y,y+28); y+=28

    # ---- logits ----
    box(CX-300,y,CX+300,y+54,C_IO,C_IOBRD)
    ct(CX,y+18,f"logits   (B, T, {V})",fr); ct(CX,y+38,"per-frame class scores",fs,C_GREY)
    y+=54; varr(CX,y,y+34); y+=34

    if mode=="train":
        # targets box (side) feeding the loss
        ty=y+8
        box(CX+150,ty,CX+470,ty+118,C_NEW,C_NEWBRD,w=2)
        ct(CX+310,ty+22,"targets  (prep time)",fr,(170,95,20))
        ct(CX+310,ty+48,"[<s>]+chars+[</s>]",fm,C_GREY)
        ct(CX+310,ty+70,"+ <fill> x (T - len)",fm,C_GREY)
        ct(CX+310,ty+94,"length = T",fm,C_GREY)
        dr.line([(CX+150,ty+59),(CX+95,ty+59)],fill=C_NEWBRD,width=2)
        dr.polygon([(CX+95,ty+53),(CX+95,ty+65),(CX+84,ty+59)],fill=C_NEWBRD)

        # loss-mask box
        box(CX-460,y,CX+90,y+118,C_MASK,C_MASKBRD,w=3)
        ct(CX-185,y+22,"Duration frame-budget loss mask  [NEW]",fr,(95,60,150))
        ct(CX-185,y+48,"n_keep = ceil(dur * 25)",fm,(60,40,110))
        ct(CX-185,y+72,"frames n_keep..T-1  -> -100",fs,C_GREY)
        ct(CX-185,y+94,"cross-batch <pad>   -> -100",fs,C_GREY)
        y+=118; varr(CX,y,y+32); y+=32
        box(CX-330,y,CX+330,y+58,C_IO,C_IOBRD)
        ct(CX,y+18,"Framewise Cross-Entropy",fr)
        ct(CX,y+39,"over the first n_keep frames only  (ignore_index = -100)",fs,C_GREY)
        y+=58
    else:
        # argmax
        box(CX-330,y,CX+330,y+50,C_INF,C_INFBRD,w=3)
        ct(CX,y+16,"per-frame argmax",fr,(20,90,100))
        ct(CX,y+35,f"logits (B,T,{V}) -> token ids (B, T)",fm,C_GREY)
        y+=50; varr(CX,y,y+30); y+=30
        # crop
        box(CX-360,y,CX+360,y+62,C_INF,C_INFBRD,w=3)
        ct(CX,y+20,"crop to frame budget",fr,(20,90,100))
        ct(CX,y+42,"keep ids[:, :n_keep],  n_keep = ceil(dur * 25)",fm,C_GREY)
        y+=62; varr(CX,y,y+30); y+=30
        # strip
        box(CX-400,y,CX+400,y+86,C_INF,C_INFBRD,w=3)
        ct(CX,y+20,"strip & detokenize",fr,(20,90,100))
        ct(CX,y+44,"remove <fill>, <pad>, <s>, </s>",fm,C_GREY)
        ct(CX,y+66,"map '|' -> space   (no CTC collapse needed)",fm,C_GREY)
        y+=86; varr(CX,y,y+30); y+=30
        # text
        box(CX-300,y,CX+300,y+54,C_IO,C_IOBRD)
        ct(CX,y+18,"predicted transcript",fr)
        ct(CX,y+38,"\"the cat sat on the mat\"",fm,C_GREY)
        y+=54

    # ---- legend ----
    ly=y+34
    def chip(x,c,b,label):
        dr.rounded_rectangle([x,ly,x+24,ly+16],radius=4,fill=c,outline=b,width=2)
        dr.text((x+32,ly+8),label,font=fs,fill=C_TXT,anchor="lm")
    chip(30,C_FROZEN,C_FRBRD,"frozen"); chip(140,C_TRAIN,C_TRBRD,"trainable")
    chip(270,C_NEW,C_NEWBRD,"new")
    if mode=="train": chip(360,C_MASK,C_MASKBRD,"loss mask")
    else: chip(360,C_INF,C_INFBRD,"decode")
    chip(500,C_ADD,C_ADDBRD,"add & norm")
    dr.line([(640,ly+8),(674,ly+8)],fill=C_RES,width=2)
    dr.text((682,ly+8),"residual",font=fs,fill=C_RES,anchor="lm")

    out = img.crop((0,0,W, ly+34))
    name = "train_arch.png" if mode=="train" else "inference_arch.png"
    out.save(name)
    print("wrote", name, out.size)


build("train")
build("infer")
