#!/usr/bin/env python
"""Render a detailed PDF: the A-CMLM/ALiBi architecture diagram (verbatim, monospace)
followed by an in-depth explanation of TRAINING and INFERENCE.

Uses only matplotlib (PdfPages) + DejaVu Sans Mono (box-drawing glyphs). A tiny flowing
layout engine paginates a list of content elements across landscape-A4 pages.

    python make_arch_train_infer_pdf.py
"""
import logging
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"

# ---------------------------------------------------------------- page geometry
PAGE_W, PAGE_H = 11.69, 8.27           # landscape A4 (inches)
ML, MR, MT, MB = 0.55, 0.45, 0.50, 0.45
USABLE_W = PAGE_W - ML - MR
TOP_IN = MT
BOTTOM_LIMIT = PAGE_H - MB

DIAGRAM_TITLE = "Filler-ASR : A-CMLM ASR head on a frozen HuBERT-xLarge (ALiBi, fw0.03)"


# ==================================================================== the diagram
DIAGRAM = r"""
                      +-------------------------------------------------------------------+
 raw audio            |  waveform x   shape (B, N_samples)   16 kHz mono, fp32 (normalized)|
 (16 kHz)             +-------------------------------------------------------------------+
                                            |
                                            v
   #=========================== FROZEN  HuBERT-xLarge  (~963M, requires_grad=False, eval-pinned) ==========================#
   #                                                                                                                        #
   #  (1) Conv feature extractor  7 x Conv1d                                                                                #
   #      channels 1->512->...->512  kernels [10,3,3,3,3,2,2]  strides [5,2,2,2,2,2,2]  (320x downsample)                   #
   #      out: (B, 512, T)  ->  transpose  ->  (B, T, 512)                                  FLOPs ~  5.0 GFLOP/s audio      #
   #                                            |                                                                           #
   #  (2) LayerNorm + feature projection  Linear 512->1280                   (B, T, 1280)   FLOPs ~  0.13 GFLOP/s           #
   #                                            |                                                                           #
   #  (3) Positional conv-embed (kernel 128, groups 16)  +  48 x Transformer encoder layers                                #
   #      each layer:  MHSA(d=1280,H=16,dk=80) -> Add&LN -> FFN(1280->5120->1280,GELU) -> Add&LN                            #
   #      per-layer/token linear = 8d^2 + 4*d*ff = 39.3 MFLOP ;  attn(QK^T+AV) = 4*T*d = 3.15 MFLOP (T=615)                 #
   #      48 layers              ------------------------------------------> TAP last_hidden_state  FLOPs ~ 102 GFLOP/s     #
   #========================================|===============================================================================#
                                            v
                 +------------------------------------------------+   <-- DETERMINISTIC tap (encoder in eval: no layerdrop
                 |  tap   (B, T, 1280)  fp16 in cache             |       / dropout). Precomputed ONCE into data/tapcache/
                 |  [precompute_taps.py -> taps.f16 memmap]       |       {train,dev}, reused every epoch -> the 963M
                 +------------------------------------------------+       encoder is OFF the training loop.
                                            |
   #============================== TRAINABLE HEAD  (157.5M) ================================================================#
   #                                        |                                                                              #
   #  (A) train-only span mask  20-30% of valid frames, spans of 10 frames (200 ms) <- learned mask_embed (1280,)         #
   #      _compute_mask_indices -> replace tap rows in-place          (B, T, 1280)         FLOPs ~0                        #
   #                                        |                                                                              #
   #  (B) POSITION = ALiBi (pos_mode="alibi")  -> NO sinusoidal add, NO RoPE.                                             #
   #      symmetric bias  b[h,i,j] = -slope_h * |i-j|,  slopes 2^-0.5 ... 2^-8                                            #
   #      built as (B*H, T, T) float, ADDED to pre-softmax attn logits inside every SA layer                             #
   #                                        |                                                                              #
   #  (C) (+) TEXT EMBEDDING  E_text : (34->1280)   text_input_ids (B,T) -> (B,T,1280)                                    #
   #      ADDITION fusion:  h = tap(+mask) + E_text(text_ids)         (B, T, 1280)         FLOPs ~ lookup (negligible)     #
   #                                        |                                                                              #
   #  (D) 8 x post-norm SA layers  (nn.TransformerEncoder, norm_first=False, batch_first)                                 #
   #      per layer:  MHSA(1280,16,dk80)+ALiBi -> Add&LN -> FFN(1280->5120->1280,GELU) -> Add&LN                          #
   #      x = LN(x + Dropout(MHSA(x)+alibi));   x = LN(x + Dropout(FFN(x)))                                               #
   #      per-layer/token linear = 39.3 MFLOP ;  attn = 3.15 MFLOP (T=615)                                                #
   #      8 layers                          |                         (B, T, 1280)         FLOPs ~ 17 GFLOP/s audio        #
   #                                        v                                                                              #
   #  (E) Dropout(0.1) -> lm_head  Linear 1280->33                    (B, T, 33) = logits   FLOPs ~ 0.004 GFLOP/s          #
   #========================================|===============================================================================#
                                            v
                                     logits (B, T, 33)
                        +-------------------+-------------------------+
          TRAIN         v                                           v        INFER (mask-predict)
  framewise CE(ignore_index=-100)                       iterative_decode(): seed pos0=<s>,
  graded ONLY on masked CMLM slots in [1, n_keep)       1..n_keep-1=<mask>, tail=<fill>;
  fill_weight=0.03 down-weights the <fill> class        N=32 passes, re-run (D)+(E) each pass,
  n_keep = min(T, ceil(T*fps/50)),  fps=50 => n_keep=T  Gumbel-top-k commit (tau=0.1, temp=1.0),
                                                        cut at first </s>  ->  hypothesis
"""


# =============================================================== content elements
# Each element is (kind, payload). kind in:
#   h1,h2,h3 : headings ; body : paragraph ; bullet : "- " item ; mono : verbatim block
#   table    : (header_list, rows) ; rule ; space ; pagebreak ; diagram
def T(*paras):
    return "\n".join(paras)


CONTENT = []
def h1(t): CONTENT.append(("h1", t))
def h2(t): CONTENT.append(("h2", t))
def h3(t): CONTENT.append(("h3", t))
def body(t): CONTENT.append(("body", t))
def bullet(t): CONTENT.append(("bullet", t))
def mono(t): CONTENT.append(("mono", t))
def table(hdr, rows): CONTENT.append(("table", (hdr, rows)))
def rule(): CONTENT.append(("rule", None))
def space(pt=6): CONTENT.append(("space", pt))
def pagebreak(): CONTENT.append(("pagebreak", None))


# ---------------------------------------------------------------- TITLE PAGE
CONTENT.append(("title", None))

# ---------------------------------------------------------------- 1. DIAGRAM
h1("1.  Architecture  (dimensions, shapes, FLOPs workflow)")
body("The model is a small trainable HEAD on top of a large FROZEN acoustic encoder. Audio is turned "
     "into a fixed 50-fps frame grid by HuBERT-xLarge; the head then predicts, per frame, one of 33 "
     "character classes. Positions are supplied by symmetric ALiBi (an additive attention bias), not by "
     "an absolute positional encoding. The head is an audio-conditioned masked language model (A-CMLM): "
     "a small text embedding is ADDED to the acoustic tap, so at inference the transcript is produced by "
     "iterative mask-predict decoding rather than a single forward pass.")
pagebreak()
CONTENT.append(("figcap", "Figure 1  -  end-to-end data / dimension / FLOPs flow "
                          "(each block annotated with its output shape and cost)"))
CONTENT.append(("diagram", None))
pagebreak()
body("Notation: B = batch, T = encoder frames (~50 fps = 320x downsample of 16 kHz), d = 1280 hidden, "
     "ff = 5120 FFN, H = 16 heads, dk = 80 head dim, V = 33 output classes, text-input vocab 34 (adds "
     "<mask>). FLOPs use the MAC = 2 FLOPs convention; \"/s\" = per second of audio (~50 frames). The "
     "example clip is 12.3 s => T ~ 615 frames (only the length-dependent attention term depends on this).")

h2("1.1  Per-stage cost and parameter split")
table(["Stage", "Params", "FLOPs / frame", "FLOPs / s audio"],
      [["Conv feature extractor", "~4.2M frozen", "~0.10 G", "~5 G"],
       ["HuBERT 48x transformer", "~960M frozen", "~2.04 G", "~102 G"],
       ["Head: 8x SA + lm_head", "157.5M TRAINABLE", "~0.34 G", "~17 G"]])
space(3)
body("Where the compute lands in the two workflows (the whole point of the tap cache and of mask-predict "
     "decoding is visible here):")
table(["Workflow", "Encoder 963M", "Head 157.5M", "Total ~"],
      [["Training step (tap-cached)", "SKIPPED (precomputed)", "fwd 17 + bwd ~34", "~51 GFLOP/s audio"],
       ["Training step (no cache)", "107 GFLOP/s fwd", "fwd 17 + bwd 34", "~158 GFLOP/s (=> ~3x cache win)"],
       ["Inference (N=32 decode)", "107 GFLOP/s (tap ONCE)", "32 x 17 = 544", "~651 GFLOP/s audio"]])

h2("1.2  Trainable head parameters (exactly what best.pt / latest.pt store, ~150 MB)")
table(["Component", "Count"],
      [["8 x SA layer  (in_proj 4.92M + out_proj 1.64M + FFN 13.11M + 2 LN) = 19.68M each", "157,419,520"],
       ["lm_head  (1280 x 33 + 33)", "42,273"],
       ["text_embed  (34 x 1280)", "43,520"],
       ["mask_embed  (1280,)", "1,280"],
       ["TOTAL trainable", "157,506,593  (~157.5M)"]])
body("Only requires_grad=True tensors are written to the checkpoint (head_state_dict); the 963M frozen "
     "HuBERT is reloaded from its base checkpoint at inference, so a run checkpoint is ~150 MB, not ~1.2 GB.")

pagebreak()

# ---------------------------------------------------------------- 2. TRAINING
h1("2.  Training  (in depth)")

h2("2.1  What is being learned:  the positional <fill> target")
body("The head is trained to place the transcript at the HEAD of the frame axis and to emit a filler class "
     "on every remaining frame. For a clip with T frames the per-frame target y is built by _labels_for():")
mono("    y = [<s>] + char_ids + [</s>] x eos_repeat + <fill> x (T - len)\n"
     "    spaces -> '|' (word delimiter) ; eos_repeat = 3 (reinforces the content->fill boundary)")
body("So token i is pinned to frame i (a deliberate, non-acoustic left-packing). The 33-class vocabulary is "
     "a-z (1..26), apostrophe (0), '|' (27), <unk> 28, <pad> 29, <s> 30, </s> 31, <fill> 32. <mask> (33) is an "
     "INPUT-only id used by the CMLM masking below; it is never a prediction target, so lm_head stays 33-way.")

h2("2.2  The duration budget  (which frames are graded)")
body("Not all frames are supervised. Only the first n_keep frames form the CONTENT region; the rest are the "
     "un-graded <fill> tail. With the tap cache, n_keep = min(T, ceil(T * fps / 50)). Because this run uses "
     "fps = 50, ceil(T*50/50) = T, so n_keep = T: the ENTIRE frame grid is inside the content region and "
     "eligible for supervision. (At fps = 25, only the first half would be graded.)")

h2("2.3  A-CMLM masking  (how each clip becomes a training example)")
body("This is a masked-language-model objective conditioned on audio. Per clip the collator "
     "(DataCollatorACMLMTapCached) does:")
bullet("Sample a masking rate p. With probability force_full_mask_prob = 0.15 set p = 1.0 (fully masked "
       "clip); otherwise p ~ Uniform(0,1). Training therefore sees every masking density from near-0 to 100%.")
bullet("Choose the masked set: over positions [1, n_keep) (position 0 = <s> is never masked), select each "
       "position independently with probability p (guaranteeing at least one).")
bullet("Build the two tensors: text_input_ids = <mask> at masked positions, the true token y elsewhere, and "
       "<fill> on the tail [n_keep, T). labels = y at masked positions, -100 everywhere else.")
body("Result: the model is given the audio plus a partially-revealed transcript and must fill the holes. "
     "The loss is computed ONLY on the masked slots, which is exactly the sub-problem the iterative decoder "
     "solves one commit at a time at inference. The eval collator instead fixes p = 1.0 (a deterministic, "
     "fully-masked reconstruction) so the eval loss is comparable across steps.")

h2("2.4  A SECOND, independent masking:  the acoustic span mask")
body("Separately from the CMLM text mask, the forward pass span-masks the ACOUSTIC tap (train only): 20-30% "
     "of valid frames, in contiguous spans of 10 frames (200 ms), are overwritten by a single learned "
     "mask_embed vector before positions/text are added (_span_mask_and_pe). The per-batch rate is drawn "
     "U[0.20, 0.30]. This forces the SA stack to reconstruct locally-missing acoustic context from "
     "neighbours (the frozen encoder cannot help), and is a regulariser distinct from the CMLM text mask.")

h2("2.5  The forward pass actually run each step  (cached-tap A-CMLM)")
mono("    tap (B,T,1280) fp16   [read from memmap, NO HuBERT forward]\n"
     "      -> (train) 20-30% span-mask with mask_embed\n"
     "      -> (ALiBi mode: NO sinusoidal PE add)\n"
     "      -> h = tap + E_text(text_input_ids)            # ADDITION fusion, 34->1280 embedding\n"
     "      -> 8 x post-norm SA layer, ALiBi bias (B*H,T,T) added to pre-softmax logits\n"
     "      -> Dropout(0.1) -> lm_head(1280->33) -> logits (B,T,33)")
body("Two correctness details the code is careful about: (i) the frozen HuBERT is pinned to eval() even under "
     "model.train() so the tap is deterministic (no layerdrop / dropout) - which is precisely what makes the "
     "tap cacheable; (ii) text_embed is force-re-initialised on the real device after from_pretrained, because "
     "under transformers 5.x the meta-device build would otherwise leave it as uninitialised memory (NaN "
     "poisoning from step 0).")

h2("2.6  Loss:  framewise cross-entropy with token-exact averaging")
body("The per-frame loss is CrossEntropy(ignore_index=-100) over the 33 classes, evaluated only on graded "
     "(masked) frames. The <fill> class is down-weighted by fill_weight = 0.03 - this ablation pulls the loss "
     "budget back onto CONTENT characters (ALiBi, not fill_weight, is what holds the </s>/tail structure, so "
     "the boundary stays sharp while spelling sharpens). Crucially the trainer uses reduction=\"sum\", not "
     "\"mean\": it accumulates raw per-frame loss SUMS across the grad-accum window AND across DDP ranks, "
     "all-reduces the global weighted token count, then rescales the summed gradient by world/global_tokens. "
     "That yields an EXACT token-weighted mean over the whole effective batch instead of a biased mean-of-means.")
mono("    weighted denom = (#content graded frames) + fill_weight * (#<fill> graded frames)")

h2("2.7  Optimisation schedule")
table(["Knob", "Value", "Note"],
      [["optimizer", "AdamW", "betas (0.9, 0.98), eps 1e-6, weight_decay 0.005"],
       ["peak LR", "2e-4", "single param group (head only; no unfrozen HuBERT)"],
       ["warmup", "5000 steps", "linear 0 -> peak"],
       ["decay", "exponential -> 1% peak", "geometric gamma = 0.01^(1/decay_steps)"],
       ["grad clip", "1.0", "on the token-normalised gradient"],
       ["batch", "32 x accum 2 x 4 GPU = 256", "effective batch; DDP no_sync during accumulation"],
       ["epochs", "150", "eval + checkpoint every 5000 steps"]])
body("DDP detail: model.no_sync() is used on every micro-step except the accumulation boundary, so gradients "
     "are all-reduced once per optimizer step (not once per micro-batch). find_unused_parameters=False.")

h2("2.8  Evaluation and checkpoint selection  (the iter_wer fix)")
body("At each eval the model reports two families of metrics: (a) one-shot reconstruction metrics from the "
     "p=1.0 collator (eval/loss, frame accuracies, entropy), and (b) a REALISTIC dev WER produced by actually "
     "running the 32-step iterative decoder on up to 200 dev clips (eval/iter_WER). best.pt is selected by "
     "select_metric = iter_wer, i.e. argmin of the iterative-decode dev WER - NOT argmin one-shot loss. This "
     "matters because on ALiBi the one-shot MLM loss saturates early (~ep69) while the iterative-decode WER "
     "keeps falling to ep150; selecting on loss would have shipped a checkpoint ~55% worse than the true best.")
body("For this run best.pt landed at step 110000 / epoch 101 with dev iter_WER = 0.1025 (10.25%). The "
     "checkpoint stores head tensors + optimizer + scheduler + step/epoch/best_val + args, so a run is exactly "
     "resumable with RESUME=<out>/latest.pt.")

h2("2.9  Tap cache  (why a training step is ~3x cheaper)")
body("Because the frozen tap is a deterministic function of the audio, precompute_taps.py runs the 963M "
     "encoder ONCE over the corpus and writes a ragged fp16 memmap (taps.f16 ~442 GB for train-960h, plus "
     "index.npy and manifest.jsonl). Training then reads taps straight from disk and never touches HuBERT, "
     "removing 107 GFLOP/s-audio of encoder forward from every step. Length-bucketed, DDP-sharded batches "
     "(LengthBucketedDistributedBatchSampler) keep per-rank batch counts equal (required by the token-exact "
     "all-reduce) and remove ~28% padding waste.")

pagebreak()

# ---------------------------------------------------------------- 3. INFERENCE
h1("3.  Inference  (in depth):  OmniVoice-style iterative mask-predict decoding")
body("The head is a masked LM, so decoding is not a single forward pass - it is a discrete masked-diffusion "
     "sampler (iterative_decode) that starts from an all-masked content region and commits a growing subset "
     "of frames each step, re-reading the whole grid every time. There is NO duration model and NO "
     "autoregression: the canvas length is fixed by the audio, and every step attends bidirectionally.")

h2("3.1  Seeding the canvas")
mono("    T       = compute_output_length(#samples)          # full canvas (attention + ALiBi span)\n"
     "    n_keep  = min(T, ceil(dur_sec * fps))              # fps=50 -> n_keep = T (decode whole grid)\n"
     "    ids[0]          = <s>                               # seeded, never masked\n"
     "    ids[1 : n_keep] = <mask>                            # the M = n_keep-1 content slots to fill\n"
     "    ids[n_keep : T] = <fill>                            # fixed tail scaffold (never decoded)")
body("The tap is computed ONCE (encode_audio -> encode_from_tap cache); only the text pathway "
     "(decode_from_tap: text-embed add + 8 SA layers + lm_head) is re-run each step.")

h2("3.2  The commit schedule  (how many frames are frozen each step)")
body("With N = 32 steps and back-loaded parameter tau = 0.1, the cumulative unmasked fraction after step n is")
mono("    r(n) = (tau * x) / (1 + (tau - 1) * x),   x = n/N,   r(0)=0, r(N)=1")
body("k_n = round(r(n)*M) - round(r(n-1)*M) new slots are committed at step n (remainder forced at the last "
     "step). tau = 0.1 is strongly back-loaded: very few commits early (when context is thin and the model is "
     "unsure), an avalanche of easy commits late (when most of the sentence is already fixed).")

h2("3.3  What one decode step does")
bullet("logp = log_softmax(model(tap, ids)); the tap is cached, so this is only the 8-SA-layer + lm_head pass.")
bullet("conf, pred = logp.max(-1): value is the greedy argmax token per frame; conf is its log-prob. An "
       "optional fill_penalty subtracts from conf where pred == <fill> (unused here; fill_penalty = 0.0).")
bullet("Position sampling: among still-masked candidate frames, pick the k_n to commit by Gumbel-top-k on "
       "conf/temp (temp = 1.0). This is temperature-controlled sampling of WHICH positions to freeze - the "
       "model commits its most confident frames first, with a little Gumbel noise to break ties.")
bullet("Commit: ids[chosen] = pred[chosen], and those frames leave the masked set permanently.")
body("So confidence decides ORDER (which frames freeze when) while the greedy argmax decides the VALUE. After "
     "N steps every content slot is filled.")

h2("3.4  Optional Mask-Predict refinement  (self-correction)")
body("Because each commit is permanent, an early low-confidence guess (classic vowel-mush) can never be "
     "revised by the fill-in loop alone. The optional remask stage (remask_rounds passes) re-masks the "
     "lowest-confidence remask_frac of committed CONTENT tokens and re-predicts them with full left+right "
     "context. The shipped decode for this run runs WITHOUT remask (parity with the reported numbers); it is "
     "available for a quality/latency trade-up.")

h2("3.5  Readout")
mono("    seq = ids up to (not including) the FIRST </s>\n"
     "    text = decode(seq, skip_special_tokens=True) ; '|' -> space ; strip")
body("Cutting at the first </s> is what stops the transcript instead of running through the <fill> tail, so "
     "no trailing garbage leaks in.")

h2("3.6  Inference cost")
body("The tap is 107 GFLOP/s-audio, paid ONCE. Each of the N=32 decode steps re-runs only the head "
     "(~17 GFLOP/s-audio), so the head dominates: 32 x 17 = 544, total ~651 GFLOP/s-audio. Latency scales "
     "linearly in N; N is the main quality/speed dial (alongside tau, temp, and remask).")

h2("3.7  Exact decode command used for the reported WER")
mono("    steps 32 | tau 0.1 | temp 1.0 | fill_penalty 0.0 | seed 0 | (no remask)\n"
     "    driver : decode_iterative.py  via  decode_alibi_fw003_lat_vs_best.sh\n"
     "    example: bash decode_alibi_fw003_lat_vs_best.sh 0 snap_best_for_decode.pt iter32_best \\\n"
     "                  test_clean test_other")

h2("3.8  Why mask-predict (not CTC / autoregression) for this head")
bullet("The training objective IS masked in-filling, so mask-predict decoding matches training exactly - "
       "hence selecting best.pt on iter_WER rather than one-shot loss.")
bullet("Bidirectional context at every step (ALiBi is symmetric) lets late, easy frames inform earlier "
       "uncertain ones through the confidence ordering.")
bullet("A fixed audio-derived canvas removes the need for a separate duration/length model and keeps the "
       "frame<->position pinning from training intact.")

rule()
body("Source of record: model in filler_sa_reference.py ; trainer in train.py ; launcher "
     "train_acmlm_alibi_fps50_fw003_exp150_4gpu_nohup.sh -> run.sh -> torchrun train.py ; decoder in "
     "filler_asr_omni_style_decoding.py / decode_iterative.py. Run dir: "
     "runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep.")


# ==================================================================== renderer
def fit_mono_fontsize(text, usable_w, max_pt=7.2):
    maxlen = max((len(l) for l in text.splitlines()), default=1)
    adv = 0.6 / 72.0  # char advance in inches per pt
    return min(max_pt, usable_w / (maxlen * adv))


STYLE = {  # kind -> (font, size_pt, weight, color, leading_mult, top_pad_pt)
    "h1":     (SANS, 14.0, "bold", "#0b3d66", 1.55, 12),
    "h2":     (SANS, 11.0, "bold", "#134b7a", 1.5, 9),
    "h3":     (SANS, 9.5, "bold", "#1a1a1a", 1.45, 6),
    "body":   (SANS, 8.6, "normal", "#161616", 1.42, 2),
    "bullet": (SANS, 8.6, "normal", "#161616", 1.42, 2),
    "mono":   (MONO, 7.4, "normal", "#0a0a0a", 1.28, 3),
    "table":  (MONO, 7.6, "normal", "#0a0a0a", 1.5, 4),
}
BODY_WRAP = 128
MONO_DIAGRAM_FS = fit_mono_fontsize(DIAGRAM, USABLE_W, max_pt=7.2)


def wrap_body(text, width):
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return out


def fmt_table(hdr, rows):
    cols = len(hdr)
    w = [len(str(hdr[i])) for i in range(cols)]
    for r in rows:
        for i in range(cols):
            w[i] = max(w[i], len(str(r[i])))
    def line(cells):
        return "  ".join(str(cells[i]).ljust(w[i]) for i in range(cols)).rstrip()
    sep = "  ".join("-" * w[i] for i in range(cols))
    return [line(hdr), sep] + [line(r) for r in rows]


class Renderer:
    def __init__(self, pdf):
        self.pdf = pdf
        self._new_page()

    def _new_page(self):
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.fig.patch.set_facecolor("white")
        self.y = TOP_IN            # inches from top
        self.page_no = getattr(self, "page_no", 0) + 1
        # footer
        self.fig.text(0.5, MB * 0.4 / PAGE_H, f"{self.page_no}",
                      ha="center", va="center", fontsize=7, color="#888888", family=SANS)

    def _flush(self):
        self.fig.text(ML / PAGE_W, 1 - (MT * 0.35) / PAGE_H, DIAGRAM_TITLE,
                      ha="left", va="center", fontsize=6.5, color="#9aa4ad", family=SANS)
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def _ensure(self, need_in):
        if self.y + need_in > BOTTOM_LIMIT:
            self._flush()
            self._new_page()

    def _draw_line(self, x_in, text, font, size, weight, color):
        yfrac = 1 - (self.y / PAGE_H)
        self.fig.text(x_in / PAGE_W, yfrac, text, ha="left", va="top",
                      fontsize=size, fontweight=weight, color=color, family=font)

    def emit_lines(self, lines, kind, x_in=ML):
        font, size, weight, color, lead_m, top_pad = STYLE[kind]
        lead_in = size * lead_m / 72.0
        self.y += top_pad / 72.0
        for ln in lines:
            self._ensure(lead_in)
            self._draw_line(x_in, ln, font, size, weight, color)
            self.y += lead_in

    def add(self, kind, payload):
        if kind == "title":
            self._title_page(); return
        if kind == "pagebreak":
            self._flush(); self._new_page(); return
        if kind == "space":
            self.y += payload / 72.0; return
        if kind == "rule":
            self._ensure(0.12)
            yfrac = 1 - (self.y / PAGE_H)
            self.fig.add_artist(plt.Line2D([ML / PAGE_W, 1 - MR / PAGE_W], [yfrac, yfrac],
                                           color="#c9d2da", lw=0.6, transform=self.fig.transFigure))
            self.y += 0.10; return
        if kind == "figcap":
            self.emit_lines([payload], "h3"); return
        if kind == "diagram":
            lines = DIAGRAM.split("\n")
            nlines = len(lines)
            lead_m = 1.16
            avail_h = BOTTOM_LIMIT - self.y - 0.05
            size_w = fit_mono_fontsize(DIAGRAM, USABLE_W, max_pt=7.2)
            size_h = avail_h * 72.0 / (nlines * lead_m)
            size = min(size_w, size_h, 7.2)
            lead_in = size * lead_m / 72.0
            self.y += 3 / 72.0
            for ln in lines:
                self._draw_line(ML, ln, MONO, size, "normal", "#111111")
                self.y += lead_in
            return
        if kind in ("h1", "h2", "h3"):
            self.emit_lines([payload], kind); return
        if kind == "body":
            self.emit_lines(wrap_body(payload, BODY_WRAP), "body"); return
        if kind == "bullet":
            font, size, weight, color, lead_m, top_pad = STYLE["bullet"]
            wrapped = textwrap.wrap(payload, width=BODY_WRAP - 2)
            wrapped = ["- " + wrapped[0]] + ["  " + w for w in wrapped[1:]]
            self.emit_lines(wrapped, "bullet"); return
        if kind == "mono":
            self.emit_lines(payload.split("\n"), "mono"); return
        if kind == "table":
            hdr, rows = payload
            self.emit_lines(fmt_table(hdr, rows), "table"); return

    def _title_page(self):
        self.fig.add_artist(plt.Line2D([ML / PAGE_W, 1 - MR / PAGE_W], [0.74, 0.74],
                                       color="#0b3d66", lw=1.4, transform=self.fig.transFigure))
        self.fig.text(0.5, 0.82, "Filler-ASR", ha="center", va="center",
                      fontsize=30, fontweight="bold", color="#0b3d66", family=SANS)
        self.fig.text(0.5, 0.775, "A-CMLM ASR head on a frozen HuBERT-xLarge  -  architecture, training & inference",
                      ha="center", va="center", fontsize=12, color="#2a2a2a", family=SANS)
        meta = [
            "run:  full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep",
            "encoder:  HuBERT-xLarge (48 layers, d=1280), FROZEN (~963M)     head:  8 SA layers + lm_head (157.5M, TRAINABLE)",
            "positions:  symmetric ALiBi        fusion:  tap + text-embedding (addition)        decode:  32-step iterative mask-predict",
            "objective:  audio-conditioned masked-LM,  fill_weight = 0.03,  fps = 50,  eos_repeat = 3,  span-mask 20-30% x10",
            "training:  4x DDP, eff batch 256, AdamW 2e-4, warmup 5000, exp decay -> 1%, 150 epochs, tap-cached",
            "best.pt:  step 110000 / epoch 101,  dev 32-step iterative-decode WER = 10.25%",
        ]
        yy = 0.66
        for m in meta:
            self.fig.text(0.5, yy, m, ha="center", va="center", fontsize=8.6, color="#333333", family=MONO)
            yy -= 0.033
        self.fig.text(0.5, 0.18,
                      "Sections:   1) Architecture (shapes, dims, FLOPs)     2) Training (in depth)     3) Inference (in depth)",
                      ha="center", va="center", fontsize=9, color="#0b3d66", family=SANS)
        self._flush(); self._new_page()


def build(out_pdf):
    with PdfPages(out_pdf) as pdf:
        r = Renderer(pdf)
        for kind, payload in CONTENT:
            if kind == "title":
                r._title_page()
            else:
                r.add(kind, payload)
        r._flush()
    print("wrote", out_pdf)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/" \
        "full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep/" \
        "ARCH_TRAIN_INFER_alibi_fw003.pdf"
    build(out)
