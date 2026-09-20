"""
Two-stage filler_asr pipeline schematics (Pillow renderer, repo house style).

Renders three slide-ready PNGs with tensor shapes on every edge, using the
running example of a 4.0 s @ 16 kHz clip:
    64000 samples -> T=199 encoder frames -> duration budget = 100 (25 fps).

  pipeline_stage1.png  : STAGE 1  frozen HuBERT + PE + N-SA + lm_head  (filler head, trains ~4-14%)
  pipeline_stage2.png  : STAGE 2  HubertEmbeddingGenerator + strip + linear projector + frozen Vicuna-7B
  pipeline_full.png    : both stages stacked, with the fps=25 hand-off contract

Colour / font conventions copied from make_train_infer_arch.py.
"""
from PIL import Image, ImageDraw, ImageFont

# ---- geometry of the running example -------------------------------------------------
d, V, H, dff, LLM = 1280, 33, 16, 5120, 4096
dk = d // H
SEC, SR = 4.0, 16000
NSAMP = int(SEC * SR)                 # 64000
T = 199                               # compute_output_length(64000)
NKEEP = 100                           # ceil(25 * 4)

# ---- fonts ---------------------------------------------------------------------------
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
def f(p, s): return ImageFont.truetype(p, s)
ftitle = f(FB, 34); fsub = f(FR, 19); fhead = f(FB, 23)
fbody  = f(FR, 18); fmono = f(FM, 16); fshape = f(FM, 17); fnote = f(FR, 15); flabel = f(FB, 15)

# ---- palette (from repo) -------------------------------------------------------------
C_BG=(250,250,252)
C_FROZEN=(210,222,240); C_FRBRD=(90,120,165)
C_TRAIN=(215,238,215);  C_TRBRD=(70,140,70)
C_SUB=(255,248,225);    C_SUBBRD=(200,165,70)
C_ADD=(245,222,230);    C_ADDBRD=(180,90,120)
C_OP=(255,233,204);     C_OPBRD=(210,130,40)
C_MASK=(226,216,240);   C_MASKBRD=(120,90,170)
C_IO=(235,235,235);     C_IOBRD=(120,120,120)
C_TXT=(25,25,30); C_GREY=(110,110,115); C_SHAPE=(20,70,120)

KIND = {  # kind -> (fill, border)
    "io":     (C_IO, C_IOBRD),
    "frozen": (C_FROZEN, C_FRBRD),
    "train":  (C_TRAIN, C_TRBRD),
    "add":    (C_ADD, C_ADDBRD),
    "op":     (C_OP, C_OPBRD),
    "mask":   (C_MASK, C_MASKBRD),
    "loss":   (C_SUB, C_SUBBRD),
}

W = 1560
BOXW = 820
BX0 = 80
BX1 = BX0 + BOXW          # 900
CX = (BX0 + BX1) // 2     # 490  (box-column centre; titles/legend also use this)
PAD_TITLE = 155
GAP = 34          # vertical gap between boxes (arrow lives here)
LINEH = 26
TAGH = 24         # extra top space reserved for the corner role-chip


def _txt(dr, x, y, s, font, fill=C_TXT, anchor="mm"):
    dr.text((x, y), s, font=font, fill=fill, anchor=anchor)


def _measure(node):
    """height of a node box from its text lines (+ a reserved row for the tag chip)"""
    n = len(node["lines"])
    return 26 + (TAGH if node.get("tag") else 0) + n * LINEH


def _draw_box(dr, y, node):
    fill, brd = KIND[node["kind"]]
    h = _measure(node)
    w = node.get("w", 3)
    dr.rounded_rectangle([BX0, y, BX1, y + h], radius=14, fill=fill, outline=brd, width=w)
    # role tag on its own row so it never collides with the (centred) title
    tag = node.get("tag")
    y0 = y + 16
    if tag:
        _txt(dr, BX0 + 16, y0, tag, flabel, fill=brd, anchor="lm")
        y0 += TAGH
    for i, ln in enumerate(node["lines"]):
        fnt = fhead if (i == 0 and node.get("bold")) else fbody
        _txt(dr, CX, y0 + i * LINEH + 9, ln, fnt)
    return h


def _arrow(dr, y0, y1, shape=None, note=None, dashed=False):
    x = CX
    if dashed:
        yy = y0
        while yy < y1 - 10:
            dr.line([x, yy, x, min(yy + 9, y1 - 10)], fill=(90,90,95), width=3)
            yy += 16
    else:
        dr.line([x, y0, x, y1 - 10], fill=(60,60,65), width=3)
    # arrowhead
    dr.polygon([(x-7, y1-11), (x+7, y1-11), (x, y1-1)], fill=(60,60,65))
    if shape:
        _txt(dr, BX1 + 24, (y0 + y1)//2, shape, fshape, fill=C_SHAPE, anchor="lm")
    if note:
        _txt(dr, BX0 - 24, (y0 + y1)//2, note, fnote, fill=C_GREY, anchor="rm")


def render(nodes, title, subtitle, path, legend=True):
    # first pass: total height
    y = PAD_TITLE
    heights = []
    for i, nd in enumerate(nodes):
        h = _measure(nd); heights.append(h)
        y += h + (GAP if i < len(nodes) - 1 else 0)
    total = y + 120
    img = Image.new("RGB", (W, total), C_BG)
    dr = ImageDraw.Draw(img)

    _txt(dr, W // 2, 50, title, ftitle)
    _txt(dr, W // 2, 92, subtitle, fsub, fill=C_GREY)

    y = PAD_TITLE
    for i, nd in enumerate(nodes):
        h = _draw_box(dr, y, nd)
        if i < len(nodes) - 1:
            _arrow(dr, y + h, y + h + GAP,
                   shape=nd.get("out"), note=nd.get("note"), dashed=nd.get("dash_out", False))
        y += h + GAP

    if legend:
        items0 = [("frozen", C_FROZEN, C_FRBRD), ("trained", C_TRAIN, C_TRBRD),
                  ("add/PE", C_ADD, C_ADDBRD), ("op (strip/scatter)", C_OP, C_OPBRD),
                  ("mask", C_MASK, C_MASKBRD), ("loss/target", C_SUB, C_SUBBRD)]
        legw = sum(26 + 8 + int(dr.textlength(t, font=fnote)) + 26 for t, _, _ in items0) - 26
        lx, ly = (W - legw) // 2, total - 70
        for txt, fl, br in items0:
            dr.rounded_rectangle([lx, ly, lx+26, ly+20], radius=5, fill=fl, outline=br, width=2)
            _txt(dr, lx+34, ly+10, txt, fnote, fill=C_TXT, anchor="lm")
            lx += 26 + 8 + int(dr.textlength(txt, font=fnote)) + 26
    img.save(path)
    print("wrote", path, img.size)
    return img


# =====================================================================================
# STAGE 1  — filler head (training view)
# =====================================================================================
stage1 = [
    dict(kind="io", tag="INPUT", lines=[
        "audio waveform  (4.0 s @ 16 kHz)",
        "Wav2Vec2FeatureExtractor · normalize + attention_mask"],
        out="input_values (B, 64000)   attention_mask (B, 64000)"),
    dict(kind="frozen", tag="FROZEN", bold=True, lines=[
        "HuBERT-xlarge body",
        "conv 320↓ (~50 fps) · +proj · pos-conv · 48 layers",
        "TAP last_hidden_state"],
        out="(B, 199, 1280)     T = compute_output_length(64000) = 199"),
    dict(kind="mask", tag="TRAIN-ONLY", lines=[
        "tap masking (SpecAugment-style)",
        "p ~ U[0.10, 0.20] · span 10 · hidden[mask] = mask_embed(1280)"],
        out="(B, 199, 1280)"),
    dict(kind="add", tag="⊕ no params", lines=[
        "+ fixed sinusoidal positional encoding"],
        out="(B, 199, 1280)"),
    dict(kind="train", tag="TRAINED", bold=True, lines=[
        "N × post-norm Self-Attention   (sa8 → N=8)",
        "d=1280 · H=16 (dk=80) · ff=5120 · gelu",
        "LN(x+MHSA(x)) → LN(x+FFN(x))  · src_key_padding_mask",
        "≈ 19.68M / layer   ← post-SA tap reused by Stage 2"],
        out="(B, 199, 1280)"),
    dict(kind="train", tag="TRAINED", lines=[
        "Dropout(0.1) → lm_head  Linear(1280 → 33)"],
        out="(B, 199, 33)  logits"),
    dict(kind="loss", tag="LOSS", bold=True, lines=[
        "target = [<s>] + chars + [</s>] + <fill>×tail",
        "framewise CrossEntropy(ignore_index = -100)",
        "loss-mask: labels[n_keep:] = -100 ,  n_keep = ⌈25·sec⌉ = 100",
        "→ graded = first 100 frames (left-packed transcript)"]),
]

# =====================================================================================
# STAGE 2  — SLAM-LLM (frozen encoder + strip + projector + frozen Vicuna)
# =====================================================================================
stage2 = [
    dict(kind="io", tag="INPUT", lines=[
        "audio (B, 64000) + attention_mask   ·   text prompt",
        '"USER: Transcribe speech to text.\\n ASSISTANT:"'],
        out="audio (B, 64000)"),
    dict(kind="frozen", tag="FROZEN", bold=True, lines=[
        "HubertEmbeddingGenerator  =  HuBERT + PE + N×SA",
        "lm_head DROPPED → taps the post-SA embeddings",
        "(weights: HuBERT body + sa.* from Stage-1 best.pt)"],
        out="(B, 199, 1280)   post-SA tap"),
    dict(kind="op", tag="OP · strip", lines=[
        "STRIP frames before projector",
        "max_budget = modality_mask.sum(1).max()·k = 100·1",
        "encoder_outs[:, :100, :]   (drop <fill>/silence tail)"],
        out="(B, 100, 1280)"),
    dict(kind="train", tag="TRAINED (only this)", bold=True, lines=[
        'projector "linear" = EncoderProjectorConcat (k=1)',
        "Linear(1280→2048) → ReLU → Linear(2048→4096)",
        "≈ 11.0M params  →  44 MB fp32 model.pt"],
        out="(B, 100, 4096)"),
    dict(kind="op", tag="OP · scatter", lines=[
        "SCATTER into LLM input at the 100 audio-placeholder slots",
        "inputs_embeds = proj_audio  ⊕  embed_tokens(prompt+answer)·(~modality_mask)",
        "dataset reserves audio_length = round(25·sec) = 100 slots"],
        out="[audio×100 | prompt | answer | </s>]  →  (B, seq, 4096)"),
    dict(kind="frozen", tag="FROZEN", bold=True, lines=[
        "Vicuna-7B-v1.5  (llm_dim = 4096)",
        "train: next-token CE on answer tokens (audio+prompt = -100)",
        "decode: generate(num_beams=4, max_new_tokens=200)"],
        out="transcript tokens → text"),
]


def render_full():
    """Both stages stacked into one tall master figure with the hand-off contract."""
    contract = dict(kind="op", tag="HAND-OFF CONTRACT", bold=True, lines=[
        "Stage-1  n_keep = ⌈25·sec⌉      ≡      Stage-2  audio_length = round(25·sec)",
        "fps = 25 on both sides  →  the stripped tail is exactly the <fill> Stage 1 learned to emit",
        "best.pt = { sa.*, lm_head.*, mask_embed, args }  →  loaded as Stage-2 encoder"])
    nodes = (stage1
             + [contract]
             + stage2)
    # add a divider label by tweaking subtitle
    render(nodes,
           "filler_asr  ·  two-stage SLAM-ASR pipeline",
           "frozen HuBERT left-packer (Stage 1)  →  strip  →  frozen Vicuna-7B transcriber (Stage 2)   ·   example: 4.0 s clip",
           "pipeline_full.png")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    render(stage1,
           "STAGE 1  ·  filler head  (trains ~4–14%)",
           "frozen HuBERT-xlarge tap  →  +PE  →  N self-attention  →  char lm_head  ·  example: 4.0 s clip",
           "pipeline_stage1.png")
    render(stage2,
           "STAGE 2  ·  SLAM-LLM  (trains only the 11M projector)",
           "post-SA tap  →  strip to budget  →  linear projector  →  scatter  →  frozen Vicuna-7B",
           "pipeline_stage2.png")
    render_full()
