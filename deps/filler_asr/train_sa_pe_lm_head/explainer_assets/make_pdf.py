"""Build the filler_sa_reference.py + full960_sa8 run explainer PDF (reportlab)."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, Preformatted, KeepTogether)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(os.path.dirname(HERE), "filler_sa_reference_explained.pdf")

PAGE_W, PAGE_H = A4
MARG = 1.9 * cm
AVAIL_W = PAGE_W - 2 * MARG

# ---------------------------------------------------------------- styles
ACCENT = colors.HexColor("#1f4e79")
ORANGE = colors.HexColor("#c55a11")
GREY = colors.HexColor("#555555")
LBG = colors.HexColor("#f4f4f4")

S_TITLE = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=19, leading=24,
                         textColor=ACCENT, spaceAfter=6)
S_SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=11.5, leading=15,
                       textColor=GREY, spaceAfter=14)
S_H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18,
                      textColor=ACCENT, spaceBefore=16, spaceAfter=6)
S_H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                      textColor=ORANGE, spaceBefore=10, spaceAfter=4)
S_BODY = ParagraphStyle("b", fontName="Helvetica", fontSize=10.2, leading=14.4,
                        alignment=TA_JUSTIFY, spaceAfter=6)
S_BUL = ParagraphStyle("bul", parent=S_BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
S_CAP = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.8, leading=11.5,
                       textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
S_CODE = ParagraphStyle("code", fontName="Courier", fontSize=8.0, leading=10.4)
S_TCELL = ParagraphStyle("tc", fontName="Helvetica", fontSize=9.0, leading=11.8)
S_TCELLB = ParagraphStyle("tcb", parent=S_TCELL, fontName="Helvetica-Bold",
                          textColor=colors.white)
S_TMONO = ParagraphStyle("tm", parent=S_TCELL, fontName="Courier", fontSize=8.4)

def C(s):  # inline code
    return f'<font name="Courier" size="9">{s}</font>'

FILL = C("&lt;fill&gt;")
EOS = C("&lt;/s&gt;")
BOS = C("&lt;s&gt;")

story = []
def P(txt, style=S_BODY): story.append(Paragraph(txt, style))
def SP(h=6): story.append(Spacer(1, h))

def IMG(name, width=None, maxw=AVAIL_W, caption=None):
    path = os.path.join(HERE, name)
    iw, ih = ImageReader(path).getSize()
    w = min(width or maxw, maxw)
    img = Image(path, width=w, height=ih * w / iw)
    img.hAlign = "CENTER"
    if caption:
        story.append(KeepTogether([img, Paragraph(caption, S_CAP)]))
    else:
        story.append(img)

def EQIMG(name, scale=0.62):
    """Equation PNG at natural-ish size (220 dpi render -> scale to page)."""
    path = os.path.join(HERE, name)
    iw, ih = ImageReader(path).getSize()
    w = min(iw * 72.0 / 220.0 * scale * 1.35, AVAIL_W)   # tuned display width
    img = Image(path, width=w, height=ih * w / iw)
    img.hAlign = "CENTER"
    story.append(img); SP(4)

def CODE(text, title=None):
    rows = [[Preformatted(text, S_CODE)]]
    t = Table(rows, colWidths=[AVAIL_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LBG),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    parts = []
    if title: parts.append(Paragraph(title, S_H2))
    parts.append(t)
    story.append(KeepTogether(parts)); SP(6)

def TAB(header, rows, widths, mono_cols=()):
    data = [[Paragraph(h, S_TCELLB) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), S_TMONO if i in mono_cols else S_TCELL)
                     for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b5c2d2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
    story.append(t); SP(8)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8); canvas.setFillColor(GREY)
    canvas.drawString(MARG, 1.05 * cm,
                      "filler_sa_reference.py + run full960_sa8_pe_fps25_lin_100ep_bs32acc2 - explained")
    canvas.drawRightString(PAGE_W - MARG, 1.05 * cm, f"page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#cccccc")); canvas.setLineWidth(0.5)
    canvas.line(MARG, 1.3 * cm, PAGE_W - MARG, 1.3 * cm)
    canvas.restoreState()

# ================================================================ TITLE
P("Filler-ASR: a Self-Attention Read-out Head on a Frozen HuBERT-xlarge", S_TITLE)
P("A detailed, intuitive walkthrough of <font name='Courier' size='10'>filler_sa_reference.py</font> "
  "and the full-LibriSpeech training run "
  "<font name='Courier' size='10'>full960_sa8_pe_fps25_lin_100ep_bs32acc2</font> "
  "&nbsp;|&nbsp; July 2026", S_SUB)

TAB(["", "TL;DR"],
    [["Idea", "Instead of CTC, treat ASR as per-frame classification: pack the transcript characters "
              "at the head of the frame axis, label every remaining frame with a special &lt;fill&gt; "
              "class, and train a small self-attention head with plain cross-entropy on top of a "
              "completely frozen HuBERT-xlarge encoder."],
     ["Trainable", "8 post-norm SA layers + linear head = 157.46M params (~14%); the ~964M HuBERT body is frozen"],
     ["Data", "LibriSpeech 960h - 281,241 train utterances; dev-clean (500 clips) for model selection"],
     ["Compute", "6-GPU DDP, bf16 autocast, effective batch 384, 100 epochs = 73,200 optimizer steps"],
     ["Result", "test-clean (2,620 utts): WER 12.54% / CER 5.66% / SER 36.0% with greedy, &lt;/s&gt;-capped decode "
                "(best dev WER 13.5% at step 40k)"]],
    [2.6*cm, AVAIL_W - 2.6*cm])

# ================================================================ 1 BIG PICTURE
P("1&nbsp;&nbsp;The big picture - what problem is this solving, and how", S_H1)
P("Modern ASR on top of a self-supervised encoder (wav2vec2 / HuBERT) almost always uses "
  "<b>CTC</b>: the model emits a distribution per frame, and a dynamic-programming loss "
  "marginalises over all monotonic alignments between frames and characters. This project asks a "
  "deliberately different question: <i>can we skip alignment entirely and make ASR a plain "
  "per-frame classification problem?</i>")
P("The trick is the <b>" + FILL + " token</b>. The 26 letters, the word separator " + C("|") +
  ", and a few specials form a 33-class vocabulary. For each clip we build a target sequence "
  "exactly as long as the encoder output (one label per 20 ms frame): the transcript characters are "
  "packed at the <b>head</b> of the frame axis - character <i>i</i> is pinned to frame <i>i</i> - "
  "then a " + EOS + " marker, then " + FILL + " everywhere else. Every frame now has exactly one "
  "correct answer, so the loss is ordinary cross-entropy. No CTC blank, no forward-backward, no "
  "alignment lattice.")
P("Why bother? Three reasons:")
P("• <b>Probing:</b> with HuBERT completely frozen, whatever WER this reaches measures how "
  "linearly-accessible the transcript is from the encoder's last layer - the head cannot "
  "re-learn acoustics.", S_BUL)
P("• <b>Cheapness:</b> only the head trains (~14% of parameters). The frozen encoder needs no "
  "backward pass and no stored activations, so memory is roughly forward-only.", S_BUL)
P("• <b>Simplicity:</b> the whole learning signal is a softmax + CE per frame - trivially "
  "debuggable, and the single-file reference reproduces it end-to-end.", S_BUL)
P("The catch is that this target is <b>positional, not acoustic</b>. The word spoken in second 8 "
  "of a clip contributes its letters near frame ~120, not near frame ~400 where its sound lives. "
  "A per-frame linear classifier on HuBERT features therefore <i>cannot</i> solve it (an earlier "
  "head-only experiment in this project sat at WER close to 1.0). Making it work needed three additions, and each "
  "one maps to a block in the reference file:")
P("• <b>Sinusoidal positional encoding</b> (Block 1) - the head must know 'which slot am I?' to "
  "place characters by index.", S_BUL)
P("• A <b>duration budget</b> in the loss (Block 4) - only the first ceil(dur x 25) frames are "
  "graded, so the model is not forced to emit " + FILL + " over the entire long tail.", S_BUL)
P("• A stack of <b>8 trainable self-attention layers</b> (Block 2) - enough capacity to "
  "<i>transport</i> acoustic evidence from where sounds occur to the head-of-sequence slots where "
  "labels sit.", S_BUL)

story.append(PageBreak())

# ================================================================ 2 PIPELINE + DIMS
P("2&nbsp;&nbsp;End-to-end pipeline and the flow of dimensions", S_H1)
P("The whole system is one diagram. Blue is frozen, orange trains, tan is data preparation, "
  "purple is the loss path. Shapes use B = batch, L = audio samples, T = encoder frames.")
IMG("fig_arch.png", width=15.2*cm,
    caption="Figure 1 - the full pipeline of filler_sa_reference.py. Left column: the model forward "
            "path with tensor shapes. Right column: label preparation in the collator and the "
            "framewise CE loss. Only the orange blocks receive gradients.")
P("<b>A real worked example.</b> Clip " + C("1089-134686-0000") + " from test-clean is 10.44 s of "
  "16 kHz audio. Following it through every stage:")
TAB(["stage", "tensor / value", "how it is computed"],
    [["raw audio", "L = 167,040 samples", "10.44 s x 16,000 Hz"],
     ["feature extractor", "(1, 167040) + attention_mask", "zero-mean/unit-variance normalise, batch pad"],
     ["conv stack (7 layers)", "167040 -&gt; 33407 -&gt; 16703 -&gt; 8351 -&gt; 4175 -&gt; 2087 -&gt; 1043 -&gt; T = 521",
      "floor((L-k)/s)+1 per layer; 320x total downsample"],
     ["HuBERT tap", "(1, 521, 1280)", "last_hidden_state of the frozen 48-layer encoder"],
     ["+ sinusoidal PE", "(1, 521, 1280)", "adds PE(521, 1280), fixed, no parameters"],
     ["8 x SA layers", "(1, 521, 1280)", "shape preserved; content mixed across time"],
     ["lm_head", "logits (1, 521, 33)", "Linear 1280 -&gt; 33 per frame"],
     ["labels", "(1, 521) ints", "&lt;s&gt; + 155 chars + &lt;/s&gt; = 157 head labels, then &lt;fill&gt;"],
     ["duration budget", "n_keep = ceil(10.44 x 25) = 261", "labels[261:] = -100 (ignored in loss)"],
     ["graded frames", "261 of 521", "157 content + 104 graded &lt;fill&gt;"]],
    [3.4*cm, 6.0*cm, AVAIL_W - 9.4*cm], mono_cols=(1,))
P("Note the asymmetry that defines this whole design: the encoder produces ~50 frames per second, "
  "but read speech carries only ~12-18 characters per second. So even with one character per "
  "frame, the transcript occupies well under half of the frame axis - the rest is " + FILL +
  " by construction. Everything else in the file (budget, masks, capped decoding) manages that tail.")

# ================================================================ 3 BLOCK 0
P("3&nbsp;&nbsp;Block 0 - vocabulary and frame geometry", S_H1)
P("<b>The 33-token vocabulary.</b> A character tokenizer (" + C("Wav2Vec2CTCTokenizer") + ") over "
  "LibriSpeech text, plus " + FILL + " appended as an extra special token:")
TAB(["id", "token(s)", "meaning"],
    [["0", "'", "apostrophe (it's, don't)"],
     ["1-26", "a ... z", "letters"],
     ["27", "|", "word delimiter - spaces become | before tokenising"],
     ["28 / 29", "&lt;unk&gt; / &lt;pad&gt;", "unknown / padding"],
     ["30 / 31", "&lt;s&gt; / &lt;/s&gt;", "transcript start / end markers - &lt;/s&gt; doubles as the "
      "model's own 'speech ended here' estimate at decode time"],
     ["32", "&lt;fill&gt;", "the filler class: 'no character is assigned to this frame'"]],
    [1.7*cm, 3.0*cm, AVAIL_W - 4.7*cm], mono_cols=(1,))
P("<b>Frame geometry.</b> " + C("compute_output_length()") + " replays the HuBERT/Wav2Vec2 "
  "convolutional feature extractor arithmetic to predict exactly how many frames T the encoder "
  "will emit for L input samples - the label vector must be sized to T <i>before</i> the model runs:")
EQIMG("eq_conv.png")
P("Seven conv layers with kernels [10,3,3,3,3,2,2] and strides [5,2,2,2,2,2,2] multiply to a "
  "320x downsample: 16,000 samples/s / 320 = 50 frames/s, i.e. one frame per 20 ms. This "
  "formula is why the worked example above lands on exactly T = 521 - and the collator computes "
  "the same number independently so labels and logits line up frame-for-frame.")

story.append(PageBreak())

# ================================================================ 4 BLOCK 1 PE
P("4&nbsp;&nbsp;Block 1 - sinusoidal positional encoding (the +PE in the diagram)", S_H1)
P("Self-attention is <b>permutation-equivariant</b>: shuffle the input frames and the outputs "
  "shuffle identically. Nothing in Q.K dot products knows <i>where</i> a frame sits. For most "
  "ASR losses that is survivable (CTC only needs order via the encoder's own conv position "
  "information), but here it is fatal: the target says 'character 12 lives at frame 12', so the "
  "head must be able to address frames <i>by absolute index</i>. The fix is the classic "
  "Vaswani-2017 fixed encoding, added once to the tapped features:")
EQIMG("eq_pe.png")
IMG("fig_pe.png", width=15.5*cm,
    caption="Figure 2 - left: the PE matrix; every dimension i oscillates with its own wavelength. "
            "Right: three sample dimensions - fast ones distinguish neighbouring frames, slow ones "
            "encode coarse position.")
P("<b>Intuition - a multi-hand clock.</b> Each pair of dimensions (sin, cos) is a hand rotating "
  "at its own speed: dimension 0 completes a cycle every ~6 frames (a 'second hand'), the last "
  "dimensions take tens of thousands of frames (an 'hour hand'). Reading all 1280 hands together "
  "gives every frame position a unique, smoothly-varying fingerprint. Two useful properties:")
P("• <b>Bounded and comparable:</b> values live in [-1, 1], the same scale as layer-normalised "
  "features, so a plain addition works (scaled by " + C("pe_scale") + ", 1.0 here).", S_BUL)
P("• <b>Relative offsets are easy:</b> PE(p+k) is a fixed linear function (a rotation) of PE(p), "
  "so attention can learn 'look k frames to the right' as easily as 'look at frame 12'.", S_BUL)
P("Because it has <b>zero parameters</b> it costs nothing and generalises to any length. In the "
  "code it is gated by " + C("config.use_sinusoidal_pe") + " so older no-PE checkpoints still load; "
  "this run trains with PE on (" + C("--pe") + "). One subtle ordering detail: the optional input "
  "masking (Section 7) happens <i>before</i> the PE add, so a masked frame keeps its position "
  "fingerprint while losing its content - exactly what a fill-in-from-context objective wants.")

# ================================================================ 5 BLOCK 2 MODEL
P("5&nbsp;&nbsp;Block 2 - the model: frozen tap, 8 SA layers, lm_head", S_H1)
P("<b>(a) The frozen tap.</b> " + C("FillerHubertSAModel") + " embeds a full " + C("HubertModel") +
  " (conv extractor + 48 transformer layers, d = 1280) and simply reads its "
  + C("last_hidden_state") + " - shape (B, T, 1280). The original CTC head of the "
  + C("hubert-xlarge-ls960-ft") + " checkpoint is never instantiated. Every HuBERT parameter has "
  + C("requires_grad=False") + ", which has a second, quieter benefit: autograd stores no "
  "activations inside the encoder, so the 1B-parameter body costs only forward memory. "
  "The encoder is a pure feature pump.")
P("<b>(b) The trainable self-attention stack.</b> Eight standard " +
  C("nn.TransformerEncoderLayer") + "s in <b>post-norm</b> arrangement (matching HuBERT's own "
  "encoder style), 16 heads, FFN width 5120, dropout 0.1, GELU. Per layer, per frame:")
EQIMG("eq_qkv.png")
EQIMG("eq_attn.png")
P("Each of the 16 heads projects the 1280-dim frames into its own 80-dim subspace, scores every "
  "frame against every other frame (the T x T matrix QK^T / sqrt(80)), softmaxes those scores into "
  "attention weights, and averages the value vectors with them. The additive mask M implements the "
  "<b>frame mask</b>: batch-padding frames get a score of -infinity, i.e. zero attention weight, so short "
  "clips in a batch are unaffected by their padding. The 16 head outputs are concatenated back to "
  "1280 dims and mixed by an output projection. Then the residual + LayerNorm + FFN sandwich:")
EQIMG("eq_postnorm.png")
P("<b>Division of labour intuition:</b> MHSA is the <i>transport</i> layer - it is the only place "
  "information crosses time, and with 8 stacked layers evidence can hop from an acoustic frame to "
  "its distant positional slot in several attention steps. The FFN is per-frame <i>computation</i> - "
  "3.9x wider than the model dim, it does the local 'given what was gathered here, which character "
  "is this?' classification work.")
P("<b>(c) Head and parameter arithmetic.</b> After the stack: " + C("Dropout(0.1)") + " then " +
  C("lm_head = Linear(1280, 33)") + " per frame. Counting parameters (d = 1280, f = 5120):")
EQIMG("eq_params.png")
EQIMG("eq_total.png")
P("This matches the training log line " + C("trainable=157.46M") + " exactly. It also explains the "
  "checkpoint size: 157.46M floats x 4 bytes = 630 MB of weights, plus two AdamW moment buffers of "
  "the same size = ~1.9 GB, which is precisely what " + C("best.pt") + " weighs on disk. "
  "(The docstring's '~39.4M, ~4%' figure is for the 2-layer reference default; this run trains 8 layers.)")
P("<b>(d) A real-world initialisation bug-fix worth knowing.</b> HuggingFace's " +
  C("_init_weights") + " for HuBERT does not know about " + C("nn.MultiheadAttention") + "'s fused " +
  C("in_proj_weight") + " parameter, so when " + C("from_pretrained") + " re-initialises the keys "
  "missing from the checkpoint (all the new SA layers), those tensors would be left as allocated "
  "garbage. The class overrides " + C("_init_weights") + " to Xavier-init them (and the optional " +
  C("mask_embed") + ") explicitly - without this the head trains from NaN-ish weights.")
P("<b>(e) Freezing helpers.</b> " + C("unfreeze_all_except_feature_extractor()") + " freezes "
  "everything, then re-enables the SA stack + lm_head. An optional " + C("unfreeze_top_hubert=N") +
  " argument additionally opens the top N HuBERT layers for gradual unfreezing (with a "
  "discriminative, smaller LR in train.py) - unused in this run: the whole body stayed frozen.")

story.append(PageBreak())

# ================================================================ 6 BLOCK 3 + 7 BLOCK 4
P("6&nbsp;&nbsp;Blocks 3-4 - label preparation: the heart of the method", S_H1)
P("The collator " + C("DataCollatorFillerASR") + " turns raw (audio path, text) rows into model-ready "
  "batches. Per clip it does four things: load + normalise audio, build the positional target, "
  "compute the duration budget, and pad everything into batch tensors.")
P("<b>Text normalisation.</b> Strip punctuation " + C("[,?.!-;:\"]") + ", lowercase, replace "
  "spaces with the word delimiter " + C("|") + ". LibriSpeech references are already clean, so this "
  "mainly guarantees train/eval symmetry.")
P("<b>The positional target.</b> With T frames (from the Block-0 formula) and N normalised "
  "characters:")
EQIMG("eq_labels.png")
P("If a transcript were longer than T - 2 frames it is truncated (essentially never fires at "
  "50 fps). The label vector is exactly T long: every frame has a class.")
P("<b>The duration budget - the 'frames-per-second' idea.</b> Grading all T frames would make "
  "~70% of the training signal 'predict " + FILL + "', and the model can drift toward the "
  "majority class. Instead only the first n_keep frames are graded:")
EQIMG("eq_nkeep.png")
P("25 graded frames per second is a deliberate midpoint: comfortably above any real character "
  "rate (~12-18 chars/s, so the transcript always fits), yet only half the 50 fps frame rate - "
  "so the model still learns a stretch of graded " + FILL + " after " + EOS + " (it must know "
  "when speech <i>content ends</i>), while the far tail costs nothing. Frames beyond n_keep are "
  "trained on <b>nothing at all</b> - which becomes important at decode time (Section 9).")
IMG("fig_labels.png", width=16.3*cm,
    caption="Figure 3 - top: the actual label layout for the 10.44 s worked example (T = 521, "
            "n_keep = 261, 157 head labels). Bottom: why this is hard - sounds are spread over the "
            "whole clip but labels sit at the head; attention must carry evidence leftwards.")
P("<b>The three masks - do not confuse them.</b> The reference is emphatic about this and it is "
  "the most common source of confusion in the codebase:")
TAB(["mask", "lives where", "mechanism", "purpose"],
    [["1. frame / attention mask", "inside every MHSA", "key-padding scores set to -inf",
      "batch-padding frames of short clips are invisible to attention"],
     ["2. loss mask", "labels tensor", "label = -100, CE ignore_index",
      "(a) label padding and (b) all frames past n_keep are simply not graded"],
     ["3. input masking (OFF here)", "on the HuBERT tap, before PE", "200 ms spans replaced by a learned mask_embed",
      "optional SpecAugment-style regulariser: the SA stack must reconstruct masked content from context"]],
    [4.1*cm, 3.4*cm, 4.6*cm, AVAIL_W - 12.1*cm])
IMG("fig_masks.png", width=16.5*cm,
    caption="Figure 4 - the three masks on a toy 3-clip batch. Only masks 1 and 2 are active in "
            "this run (mask_prob = 0).")

# ================================================================ 8 LOSS
P("7&nbsp;&nbsp;Block 5 - the loss: framewise cross-entropy", S_H1)
P("Given logits z (B, T, 33) and labels y (B, T), both flattened:")
EQIMG("eq_ce.png")
P("That is the entire loss - one softmax per frame, averaged over graded frames G. Sanity anchor "
  "used by the smoke test: an untrained head should score about the entropy of a uniform guess,")
EQIMG("eq_ln33.png")
P("and the observed initial loss ~4.05 in this run's step-0 eval is in that ballpark (slightly "
  "higher because the re-initialised head is not exactly uniform). <b>Class balance:</b> inside the "
  "graded region roughly 60% of frames are content and 40% are " + FILL + " (for the worked "
  "example: 157 of 261 vs 104 of 261) - mild enough that the implemented training uses "
  "<b>unweighted</b> CE. A " + C("fill_weight") + " knob exists to down-weight " + FILL + " if a "
  "configuration collapses to all-filler; it stayed unused here, and the trained model's "
  "eval " + C("fill_pred_rate") + " of ~0.41 confirms it predicts filler at almost exactly the "
  "target rate.")

story.append(PageBreak())

# ================================================================ 9 BLOCKS 6-8
P("8&nbsp;&nbsp;Blocks 6-8 - wiring, smoke test, and the overfit proof", S_H1)
P("<b>Block 6 - build_model().</b> Loads the pretrained config, overwrites vocab_size to 33 and "
  "attaches all SA/PE/masking hyper-parameters to it, then calls " + C("from_pretrained(...,"
  " ignore_mismatched_sizes=True)") + ". The pretrained checkpoint is a CTC model with a 32-class "
  "head; the flag lets the 33-class " + C("lm_head") + " be freshly re-initialised instead of "
  "erroring. SpecAugment is disabled for determinism. Finally the freeze helpers set the "
  "trainable set: SA + lm_head only.")
P("<b>Block 7 - smoke test.</b> One collated batch, one forward, one loss. It prints shapes, "
  "trainable-parameter count and the loss vs the ln 33 reference - a 30-second check that "
  "tokenizer, collator, masks and model agree about T before any GPU-hours are spent.")
P("<b>Block 8 - the overfit driver.</b> The classic learnability experiment: take N=10 clips, "
  "full-batch AdamW (lr 2e-3), a few hundred epochs. If the architecture + target are coherent, "
  "loss must go to ~0 and the model must reproduce the graded transcripts verbatim; if the target "
  "were unlearnable, it plateaus. Running it with and without " + C("--pe") + " is "
  "how the PE requirement was established empirically: same data, same head, PE off fails to "
  "memorise even 10 clips, PE on reproduces them. " + C("_decode()") + " maps argmax frame ids to "
  "text by dropping specials/" + FILL + " and turning " + C("|") + " into spaces - the same "
  "stripping used everywhere downstream.")

# ================================================================ 10 TRAINING RUN
P("9&nbsp;&nbsp;The training run: full960_sa8_pe_fps25_lin_100ep_bs32acc2", S_H1)
P("The run trains the reference architecture (imported <b>unchanged</b> from "
  "filler_sa_reference.py by train.py - model, collator, loss are the same objects) on the full "
  "960 h manifest with no feature caching: the frozen encoder runs a fresh forward every step.")
TAB(["knob", "value", "notes"],
    [["encoder", "hubert-xlarge-ls960-ft", "frozen; ~964M params, already ASR-finetuned (its own CTC head discarded)"],
     ["head", "8 SA layers + lm_head, PE on", "157.46M trainable"],
     ["objective", "framewise CE, fps budget 25", "unweighted; ignore_index -100"],
     ["data", "281,241 train utts / 500 dev-clean utts", "lazy audio loading, 8 dataloader workers per rank"],
     ["parallelism", "6-GPU DDP (torchrun), bf16 autocast", "physical GPUs 1,4,6,7,8,9; NCCL over loopback"],
     ["batch", "32 per GPU x 2 accum x 6 GPUs = 384", "grad accumulation with no_sync() between boundaries"],
     ["optimizer", "AdamW lr 2e-4, betas (0.9, 0.98), eps 1e-6, wd 0.005", "grad-norm clip 1.0"],
     ["schedule", "linear: 2,000-step warmup then decay to 0", "sized to 73,200 total steps ('lin' in the run name)"],
     ["epochs / steps", "100 epochs = 73,200 steps", "732 steps/epoch (see arithmetic below)"],
     ["eval / ckpt", "every 2,000 steps on 500 dev clips", "best.pt on dev-WER improvement; latest.pt for resume"]],
    [2.9*cm, 6.3*cm, AVAIL_W - 9.2*cm])
EQIMG("eq_eff.png")
EQIMG("eq_lr.png")
P("<b>DDP mechanics worth noting.</b> Each rank owns 1/6 of the data via " +
  C("DistributedSampler") + " (reshuffled every epoch). During the first of each pair of "
  "micro-batches the loop wraps backward in " + C("model.no_sync()") + " so gradients are only "
  "all-reduced at the accumulation boundary - halving communication. The loss is divided by the "
  "accumulation factor so gradient magnitudes match a true batch-384 step. Forward runs in "
  "bf16 autocast; logits are cast back to fp32 before the CE for numerical safety. Checkpoints "
  "store <b>only head keys</b> (" + C("sa.*") + ", " + C("lm_head.*") + ") plus optimizer/scheduler "
  "state - the frozen HuBERT never changes, so it is reloaded from the pretrained checkpoint at "
  "inference and the head is applied with " + C("strict=False") + ".")
IMG("fig_curve.png", width=16.5*cm,
    caption="Figure 5 - left: dev-clean WER/CER over the run (evaluated every 2,000 steps with "
            "UNCAPPED greedy decode, hence pessimistic and noisy - see Section 9). Best dev WER "
            "0.135 at step 40k became best.pt. Right: the linear LR schedule.")
P("<b>How the run actually unfolded.</b> At step 0 the untrained head scores WER 1.0 and CER 7.96 "
  "- it emits near-random characters over all 521 frames, so hypotheses are many times longer "
  "than references (CER can exceed 1). Through warmup the filler structure is learned first "
  "(fill_pred_rate climbs to ~0.41 and stays pinned there), then content accuracy grinds upward "
  "for tens of thousands of steps - this is the slow part: the head is learning to route evidence "
  "across time. Dev WER bottoms out at <b>0.135 around step 40k (~epoch 55)</b>. After that the "
  "curve oscillates and drifts up while prediction entropy keeps falling (0.15 -&gt; 0.07): the "
  "classic signature of a head over-sharpening on a converged schedule - more epochs stopped "
  "helping. Because " + C("best.pt") + " snapshots only on dev-WER improvement, the over-run "
  "wasted compute but not quality.")

story.append(PageBreak())

# ================================================================ 11 INFERENCE
P("10&nbsp;&nbsp;Inference: greedy decode, and why it must be capped", S_H1)
P("Decoding is deliberately primitive - <b>argmax per frame</b>, no beam, no language model:")
CODE("logits = model(input_values, attention_mask).logits          # (B, T, 33)\n"
     "pred   = logits.argmax(-1)                                   # (B, T) int ids\n"
     "hit    = (pred == eos_id).nonzero()                          # first </s> = model's own end estimate\n"
     "capped = pred[: hit[0]]        # ... or pred[: n_keep] - dump_decodes.py uses min(first </s>, n_keep)\n"
     "hyp    = tok.decode(capped, skip_special_tokens=True, group_tokens=False)   # NO CTC collapse\n"
     "hyp    = hyp.replace('|', ' ')")
P("<b>Why cap at all?</b> Every frame past n_keep was labelled -100 during training - the model "
  "received <i>zero</i> gradient about them. Whatever it emits there is undefined behaviour, and "
  "in practice it leaks stray characters. An uncapped decode therefore appends garbage after the "
  "real transcript and inflates WER badly (this is exactly why the in-training eval curve in "
  "Figure 5 is noisy and pessimistic - " + C("evaluate()") + " decodes uncapped). The fix costs "
  "one line: cut at the first " + EOS + " the model emits (its own duration estimate) or at "
  "n_keep, whichever comes first.")
P("<b>Why group_tokens=False (no CTC-style duplicate merging)?</b> In CTC, consecutive repeats "
  "are alignment artefacts and must be collapsed. Here the target places <i>exactly one</i> "
  "character per frame - a repeated letter in the output is a genuine double letter ('ll' in "
  "'hello'), and collapsing destroys it. The decode-variant sweep on this run's best.pt proves the "
  "point empirically:")
TAB(["decode variant", "WER", "CER", "SER", "verdict"],
    [["cut at first &lt;/s&gt; (no collapse)", "<b>12.54%</b>", "5.66%", "36.0%", "best - used everywhere"],
     ["cut at &lt;/s&gt; + collapse duplicates", "20.59%", "7.41%", "81.9%", "collapse merges real double letters"],
     ["cut at first run of 3+ &lt;fill&gt;", "12.54%", "5.66%", "36.0%", "equivalent end detector"],
     ["&lt;/s&gt; or fill-run, whichever first", "12.54%", "5.66%", "36.0%", "ditto"]],
    [5.6*cm, 2.1*cm, 2.1*cm, 2.1*cm, AVAIL_W - 11.9*cm])
P("<b>A real decode from this run</b> (test-clean, greedy, capped). The model emits the frame "
  "stream on top; stripping specials and mapping " + C("|") + " to spaces yields the hypothesis:")
CODE("frames : <s> h e | h o p e d | t h e r e | w o u l d | b e | s t e w ... s a u c e </s> <fill> x260\n"
     "HYP    : he hoped there would be stew for dinner turnips and carrots and bruised\n"
     "         potatoes and fat mutton pieces to be ladled out in thick peppered flour\n"
     "         fattened sauce                                            (WER 0.000)\n"
     "-----------------------------------------------------------------------------\n"
     "HYP    : hello berty any good in your mind          REF: hello bertie ...  (1 word err)")
P("The second clip shows the characteristic error mode: <i>berty</i> for <i>bertie</i> - "
  "acoustically perfect, orthographically wrong. A char-level greedy decoder has no lexicon or "
  "language model to prefer the correct spelling, which is why WER (12.5%) is more than double "
  "CER (5.7%): errors are mostly single-letter misspellings inside otherwise-correct words. "
  "This is also why a downstream LLM-correction pass over these decodes is a natural add-on "
  "(explored elsewhere in this project).")

# ================================================================ 12 RESULTS
P("11&nbsp;&nbsp;Results, honest limitations, and where to go next", S_H1)
TAB(["metric (test-clean, 2,620 utts, greedy + &lt;/s&gt; cap)", "value"],
    [["Word Error Rate (WER)", "12.54%"],
     ["Character Error Rate (CER)", "5.66%"],
     ["Sentence Error Rate (SER)", "35.99%"],
     ["best dev-clean WER during training (uncapped eval)", "13.5% @ step 40,000"],
     ["trainable / total parameters", "157.46M / ~1.12B (~14%)"]],
    [11.2*cm, AVAIL_W - 11.2*cm])
P("For calibration: the same frozen encoder <i>with its native CTC head</i> sits around ~2% WER on "
  "test-clean. The gap is the price of the deliberately non-acoustic target - and simultaneously "
  "the interesting finding: a frozen encoder plus 8 attention layers can solve 'transcribe by "
  "writing characters into indexed slots' to within 12.5% WER using nothing but per-frame CE.")
P("<b>Known limitations (from the code comments and the eval analysis):</b>")
P("• <b>Positional pinning is non-acoustic.</b> Character i sits at frame i regardless of when it "
  "is spoken, so the head must effectively count characters while listening. Errors compound "
  "rightward: one wrong slot early shifts everything after it (a big contributor to SER).", S_BUL)
P("• <b>Greedy + no LM.</b> Spelling errors (berty/bertie) go uncorrected; a lexicon-constrained "
  "or LLM-rescored decode would recover many of them cheaply.", S_BUL)
P("• <b>In-training eval is uncapped</b>, so the Figure-5 curve overstates error and adds noise; "
  "model selection still worked, but capped eval would give a cleaner signal (and was later "
  "implemented in the eval_analysis / dump_decodes tooling).", S_BUL)
P("• <b>Long clips stress the design</b>: transport distance grows with T while the graded-tail "
  "fraction shrinks - the truncation guard exists but the geometry gets harder.", S_BUL)
P("<b>Natural next steps</b>, all wired but unused in this run: train-time input masking "
  "(mask_prob &gt; 0) as a regulariser; gradual unfreezing of the top HuBERT layers with the "
  "discriminative LR path; a " + C("fill_weight") + " sweep; and LLM-based generative error "
  "correction over the capped decodes.")

story.append(PageBreak())

# ================================================================ APPENDIX
P("Appendix A - file map", S_H1)
TAB(["file", "role"],
    [["filler_asr/filler_sa_reference.py", "single-file reference: Blocks 0-8 (model, tokenizer, collator, loss, smoke, overfit)"],
     ["train_sa_pe_lm_head/train.py", "full-data training loop: DDP, bf16, grad accum, eval, best/latest checkpoints, wandb"],
     ["train_sa_pe_lm_head/data.py", "manifest loader + re-export of the reference collator"],
     ["train_sa_pe_lm_head/run.sh", "launcher: env, GPU pinning (PCI_BUS_ID), NCCL loopback, torchrun"],
     ["train_sa_pe_lm_head/train_sa8_100ep_6gpu.sh", "THIS run's launcher: tmux-detached, GPU/ckpt-clobber preflights, 6 GPUs"],
     ["train_sa_pe_lm_head/infer.py", "decode engine: rebuilds arch from ckpt keys, &lt;/s&gt;-capped greedy, WER/CER/SER"],
     ["train_sa_pe_lm_head/dump_decodes.py", "per-utterance dump: RAW vs NKEEP frame streams + capped hyp vs ref + variants"],
     ["runs/full960_sa8_pe_fps25_lin_100ep_bs32acc2/", "best.pt / latest.pt (1.9 GB: head + AdamW state), tokenizer files, decode dumps + summaries"]],
    [7.3*cm, AVAIL_W - 7.3*cm], mono_cols=(0,))
P("Appendix B - reproducing the run", S_H1)
CODE("# smoke test (30 s sanity check of shapes + ln(33) loss)\n"
     "cd /speech/tomson/filler_asr\n"
     "python filler_sa_reference.py --n 4 --pe\n"
     "\n"
     "# overfit proof (should reproduce all 10 clips)\n"
     "python filler_sa_reference.py --train --n 10 --epochs 200 --pe\n"
     "\n"
     "# the full 100-epoch, 6-GPU run (tmux-detached; preflights GPUs + out dir)\n"
     "cd /speech/tomson/filler_asr/train_sa_pe_lm_head\n"
     "bash train_sa8_100ep_6gpu.sh          # GPUS=1,4,6,7,8,9  EPOCHS=100  BATCH=32 ACCUM=2\n"
     "\n"
     "# decode test-clean with the trained head\n"
     "python infer.py --run_dir runs/full960_sa8_pe_fps25_lin_100ep_bs32acc2 \\\n"
     "                --manifest .../librispeech_test_clean.jsonl")
CODE("# loading the trained head for your own inference\n"
     "from filler_sa_reference import build_tokenizer, build_model\n"
     "import torch\n"
     "run = 'runs/full960_sa8_pe_fps25_lin_100ep_bs32acc2'\n"
     "tok = build_tokenizer(run)                       # tokenizer was saved into the run dir\n"
     "ck  = torch.load(f'{run}/best.pt', map_location='cpu')\n"
     "model = build_model('/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft',\n"
     "                    tok, num_sa_layers=8, use_sinusoidal_pe=True, device='cuda')\n"
     "model.load_state_dict(ck['head'], strict=False)  # head keys only; hubert already loaded\n"
     "model.eval()")
P("<i>Generated from the source tree on 2026-07-05. All numbers are read from the run's own "
  "logs and decode summaries (train_sa8_100ep_*.log, librispeech_test_clean.dump.summary / "
  ".strip_variants.summary).</i>", S_CAP)

doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, leftMargin=MARG, rightMargin=MARG,
                        topMargin=1.6*cm, bottomMargin=1.7*cm,
                        title="filler_sa_reference.py explained",
                        author="tomson + Claude")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT_PDF)
