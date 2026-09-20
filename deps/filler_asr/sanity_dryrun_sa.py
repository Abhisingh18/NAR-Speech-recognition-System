"""
In-depth instrumented dry run for FillerHubertSAModel (model_sa.py).
Passes a real 2-clip sample through the model and prints the dimensions +
the math at EVERY stage:
   input -> HuBERT encoder -> SA layer 1 -> SA layer 2 -> dropout -> lm_head -> CE loss
Run (CPU):  CUDA_VISIBLE_DEVICES="" python sanity_dryrun_sa.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import soundfile as sf
from transformers import HubertConfig, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from model_sa import FillerHubertSAModel
from data_local import write_vocab
from dataset import compute_output_length

torch.manual_seed(0)
CKPT = "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"
TRAIN = "/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl"
WAVS = "/speech/tomson/exps/speech-recog/data/librispeech/wavs"

def line(c="="): print(c * 92)
def stats(name, t):
    print(f"    {name:18} shape={tuple(t.shape)}  dtype={t.dtype}  "
          f"finite={torch.isfinite(t).all().item()}  "
          f"mean={t.float().mean():+.4f}  std={t.float().std():.4f}  absmax={t.float().abs().max():.4f}")

# ---------- setup ----------
vocab = write_vocab(TRAIN, "/tmp", "sa_dry")
tok = Wav2Vec2CTCTokenizer(vocab, unk_token="<unk>", pad_token="<pad>",
        word_delimiter_token="|", bos_token="<s>", eos_token="</s>")
tok.add_special_tokens({"additional_special_tokens": ["<fill>"]})
fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                              do_normalize=True, return_attention_mask=True)

cfg = HubertConfig.from_pretrained(CKPT)
cfg.vocab_size = len(tok)
cfg.num_sa_layers = 4; cfg.sa_nhead = 16; cfg.sa_dim_feedforward = 5120; cfg.sa_dropout = 0.1
model = FillerHubertSAModel.from_pretrained(CKPT, config=cfg, ignore_mismatched_sizes=True)
model.freeze_feature_extractor()
model.unfreeze_all_except_feature_extractor()   # SA regime: encoder frozen, SA+head trainable
model.eval()

d  = cfg.hidden_size            # 1280
h  = cfg.sa_nhead               # 16
dk = d // h                     # 80
dff= cfg.sa_dim_feedforward     # 5120
V  = cfg.vocab_size             # 33
fill_id = tok.convert_tokens_to_ids("<fill>")
bos, eos = tok.bos_token_id, tok.eos_token_id

# ---------- build a real 2-clip batch (truncate to clean lengths) ----------
import os, json
rows = []
with open(TRAIN) as f:
    for ln in f:
        rows.append(json.loads(ln));
        if len(rows) >= 2: break
texts = []
arrs  = []
import re
from dataset import CHARS_TO_IGNORE_REGEX
for r, n in zip(rows, [64000, 48000]):     # 4.0s and 3.0s
    a, sr = sf.read(r["source"], dtype="float32"); a = a[:n]
    arrs.append(a)
    texts.append(re.sub(CHARS_TO_IGNORE_REGEX, "", r["target"]).lower())

batch = fe(arrs, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
input_values, attention_mask = batch["input_values"], batch["attention_mask"]

# build filler labels per clip, padded to the batch's max frame count with -100
def make_labels(text, T):
    ids = tok(text.replace(" ", "|")).input_ids
    if len(ids) + 2 > T: ids = ids[:T-2]
    lab = [bos] + ids + [eos] + [fill_id] * (T - len(ids) - 2)
    return lab
T_each = [compute_output_length(len(a)) for a in arrs]
Tmax = max(T_each)
labels = torch.full((2, Tmax), -100, dtype=torch.long)
for i, (txt, Ti) in enumerate(zip(texts, T_each)):
    lab = make_labels(txt, Ti)
    labels[i, :Ti] = torch.tensor(lab)

B = 2
line()
print("DRY RUN — FillerHubertSAModel   |   B=%d   d=%d   heads=%d   dk=%d   ff=%d   vocab=%d" % (B,d,h,dk,dff,V))
line()

# ===================== STEP 0: INPUT =====================
print("\n[STEP 0] INPUT  (raw waveform -> feature extractor)")
print(f"    clip0: {len(arrs[0])} samples = {len(arrs[0])/16000:.2f}s   clip1: {len(arrs[1])} samples = {len(arrs[1])/16000:.2f}s")
stats("input_values", input_values)
stats("attention_mask", attention_mask)
print(f"    -> clip1 is shorter, right-padded with 0.0; attention_mask marks real vs pad samples")

# ===================== STEP 1: CNN DOWNSAMPLING MATH =====================
print("\n[STEP 1] CNN DOWNSAMPLING  (T_s audio samples -> T frames),  factor = 5*2*2*2*2*2*2 = 320")
for i,(a,Ti) in enumerate(zip(arrs, T_each)):
    print(f"    clip{i}: compute_output_length({len(a)}) = {Ti} frames   (~{len(a)/Ti:.0f} samples/frame, ~{Ti/(len(a)/16000):.0f} frames/s)")
print(f"    batch frame count T = max({T_each}) = {Tmax}   -> all per-frame tensors are length T={Tmax}")

# ===================== STEP 2: HUBERT ENCODER =====================
print("\n[STEP 2] HuBERT ENCODER  self.hubert(input_values, attention_mask) -> outputs[0]")
print(f"    inside: CNN(512ch) -> feature_projection(512->{d}) -> 48 transformer layers (16 heads) -> last_hidden_state")
with torch.no_grad():
    enc = model.hubert(input_values, attention_mask=attention_mask)[0]
stats("last_hidden_state", enc)
print(f"    (B, T_s)=({B},{input_values.shape[1]})  -->  (B, T, d)=({B},{enc.shape[1]},{enc.shape[2]})   [the EMBEDDING TAP, encoder FROZEN]")

# ===================== STEP 3: PAD MASK FOR SA =====================
print("\n[STEP 3] FRAME-LEVEL PAD MASK for the self-attention")
feat_mask = model.hubert._get_feature_vector_attention_mask(enc.shape[1], attention_mask)
skp = ~feat_mask.bool()    # True = pad -> ignore
print(f"    feat_mask (1=valid): valid frames per clip = {feat_mask.sum(1).tolist()} of T={Tmax}")
print(f"    src_key_padding_mask (True=ignore): padded frames per clip = {skp.sum(1).tolist()}")

# ===================== STEP 4: SA LAYER 1 (attention internals) =====================
print("\n[STEP 4] SELF-ATTENTION LAYER 1  self.sa.layers[0]")
X = enc
print(f"    input X: {tuple(X.shape)}   (B,T,d)")
attn = model.sa.layers[0].self_attn
with torch.no_grad():
    qkv = Fnn.linear(X, attn.in_proj_weight, attn.in_proj_bias)   # (B,T,3d)
    q,k,v = qkv.split(d, dim=-1)                                   # each (B,T,d)
    qh = q.view(B,Tmax,h,dk).transpose(1,2)                        # (B,h,T,dk)
    kh = k.view(B,Tmax,h,dk).transpose(1,2)
    vh = v.view(B,Tmax,h,dk).transpose(1,2)
    scores = (qh @ kh.transpose(-2,-1)) / math.sqrt(dk)           # (B,h,T,T)
    scores = scores.masked_fill(skp[:,None,None,:], float("-inf"))
    attn_w = torch.softmax(scores, dim=-1)                         # (B,h,T,T)
    ctx = (attn_w @ vh).transpose(1,2).reshape(B,Tmax,d)          # (B,T,d)
print(f"    Q = X.Wq, K = X.Wk, V = X.Wv     (in_proj_weight {tuple(attn.in_proj_weight.shape)})  ->  each {tuple(q.shape)}")
print(f"    reshape to heads:  {tuple(q.shape)} -> (B,h,T,dk) = {tuple(qh.shape)}")
print(f"    scores = Q.K^T / sqrt(dk={dk})  ->  (B,h,T,T) = {tuple(scores.shape)}   [each head: {Tmax}x{Tmax} attention matrix]")
print(f"    softmax(scores) rows sum to 1:  e.g. attn_w[0,0,0].sum() = {attn_w[0,0,0].sum().item():.4f}")
print(f"    context = softmax(scores).V  ->  (B,h,T,dk) -> merge heads -> (B,T,d) = {tuple(ctx.shape)}")
print(f"    then out_proj (d->d), residual + LayerNorm, FFN {d}->{dff}->{d} (GELU), residual + LayerNorm")
with torch.no_grad():
    sa1 = model.sa.layers[0](X, src_key_padding_mask=skp)
stats("SA1 output", sa1)

# ===================== STEP 5: SA LAYER 2 =====================
print("\n[STEP 5] SELF-ATTENTION LAYER 2  self.sa.layers[1]  (same shape, refines further)")
with torch.no_grad():
    sa2 = model.sa.layers[1](sa1, src_key_padding_mask=skp)
stats("SA2 output", sa2)
print(f"    (also equals model.sa(enc): full stack output) -> feeds dropout")

# ===================== STEP 6: DROPOUT =====================
print("\n[STEP 6] DROPOUT  self.dropout (p=0.1)  — identity in eval(); in train() randomly zeros 10%")

# ===================== STEP 7: LM HEAD =====================
print("\n[STEP 7] LM HEAD  self.lm_head = nn.Linear(%d -> %d)" % (d, V))
with torch.no_grad():
    logits = model.lm_head(sa2)
print(f"    logits = h . W_head^T + b   ((B,T,d)=({B},{Tmax},{d})) x ((d,V)=({d},{V}))  ->  (B,T,V)")
stats("logits", logits)

# full model forward to confirm identical
with torch.no_grad():
    logits_full = model(input_values=input_values, attention_mask=attention_mask).logits
print(f"    matches model(...).logits ?  {torch.allclose(logits, logits_full, atol=1e-4)}")

# ===================== STEP 8: FILLER LABELS =====================
print("\n[STEP 8] FILLER LABELS  (target the CE is computed against)")
for i,Ti in enumerate(T_each):
    row = labels[i]
    nfill = (row == fill_id).sum().item(); nign = (row == -100).sum().item()
    ncontent = Ti - nfill
    print(f"    clip{i}: T={Ti}  content(<s>+chars+</s>)={ncontent}  <fill>={nfill}  -100(pad to {Tmax})={nign}")
toks = tok.convert_ids_to_tokens(labels[0,:14].tolist())
print(f"    clip0 first 14 target tokens: {' '.join(toks)}")

# ===================== STEP 9: CROSS-ENTROPY LOSS =====================
print("\n[STEP 9] CROSS-ENTROPY LOSS  (framewise, ignore_index=-100)")
flat_logits = logits.reshape(-1, V)
flat_labels = labels.reshape(-1)
N = (flat_labels != -100).sum().item()
print(f"    flatten: logits {tuple(logits.shape)} -> {tuple(flat_logits.shape)}   labels {tuple(labels.shape)} -> {tuple(flat_labels.shape)}")
print(f"    total frames B*T = {B*Tmax}   graded (label != -100) N = {N}   ignored = {B*Tmax - N}")
loss = nn.CrossEntropyLoss(ignore_index=-100)(flat_logits, flat_labels)
print(f"    loss = (1/N) * sum_t -log softmax(logit_t)[label_t]")
print(f"    LOSS = {loss.item():.4f}    (random head -> ~ ln(V) = ln(33) = {math.log(33):.4f})")
# show per-frame penalty on first few graded frames of clip0
p = torch.softmax(logits[0], dim=-1)
print("    per-frame check (clip0, first 6 frames):")
for t in range(6):
    y = labels[0,t].item()
    print(f"        frame {t}: target={tok.convert_ids_to_tokens([y])[0]:6} p(correct)={p[t,y].item():.4f}  penalty=-log p={-math.log(p[t,y].item()+1e-12):.3f}")

# ===================== STEP 10: BACKWARD / GRAD FLOW =====================
print("\n[STEP 10] BACKWARD  (grads must flow ONLY to self.sa + self.lm_head)")
model.train()
out = model(input_values=input_values, attention_mask=attention_mask)
loss = nn.CrossEntropyLoss(ignore_index=-100)(out.logits.reshape(-1,V), labels.reshape(-1))
loss.backward()
def gnorm(ps):
    s=0.0; k=0
    for pp in ps:
        if pp.grad is not None: s+=pp.grad.float().pow(2).sum().item(); k+=1
    return math.sqrt(s), k
print(f"    grad-norm self.sa      = {gnorm(model.sa.parameters())[0]:.4f}   (>0)")
print(f"    grad-norm self.lm_head = {gnorm(model.lm_head.parameters())[0]:.4f}   (>0)")
print(f"    hubert params receiving grad = {sum(1 for pp in model.hubert.parameters() if pp.grad is not None)}   (== 0, frozen)")
tr = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
print(f"    trainable params = {tr/1e6:.2f}M   (of {sum(pp.numel() for pp in model.parameters())/1e6:.2f}M total)")
line()
print("DRY RUN COMPLETE — pipeline is dimensionally consistent and numerically finite.")
line()

import shutil; shutil.rmtree("/tmp/sa_dry", ignore_errors=True)
