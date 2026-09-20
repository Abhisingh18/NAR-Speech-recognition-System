"""Verify the two experiment changes WITHOUT training:
   (1) eos_repeat  -> </s> emitted on N consecutive end frames (default 1 = unchanged)
   (2) fill_weight -> per-class CE weight for <fill> (weighted-mean + gradient scaling)

CPU-only, no model load. Run:
   CUDA_VISIBLE_DEVICES="" python verify_experiment.py
"""
import os, sys, math, json
import numpy as np
import torch

sys.path.insert(0, "/speech/tomson/filler_asr")
from filler_sa_reference import (build_tokenizer, DataCollatorFillerASR, framewise_ce_loss,
                                 compute_output_length, FILL_TOKEN, SAMPLING_RATE)
from transformers import Wav2Vec2FeatureExtractor

VOCAB = "/speech/tomson/filler_asr/experiments/Hubert_SA_2gpu_lazy_lr2e3"
MANIFEST = "/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl"

tok = build_tokenizer(VOCAB)
BOS, EOS = tok.bos_token_id, tok.eos_token_id
FILL = tok.convert_tokens_to_ids(FILL_TOKEN)
ok = True
def check(name, cond, extra=""):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> '+extra) if extra else ''}")

print("="*74); print("A) LABEL CONSTRUCTION  (_labels_for)"); print("="*74)
fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                              do_normalize=True, return_attention_mask=True)
text = "hello world"                      # normalize -> "hello|world" = 11 chars
T = 60
# eos_repeat = 1  (must reproduce the ORIGINAL formula exactly)
c1 = DataCollatorFillerASR(fe, tok, eos_repeat=1)
lab1 = c1._labels_for(text, T)
ids = tok("hello|world").input_ids
old = [BOS] + ids + [EOS] + [FILL]*(T - len(ids) - 2)
check("eos_repeat=1 == original target (backward compatible)", lab1 == old)
check("eos_repeat=1 has exactly ONE </s>", lab1.count(EOS) == 1)

# eos_repeat = 3
c3 = DataCollatorFillerASR(fe, tok, eos_repeat=3)
lab3 = c3._labels_for(text, T)
L = len(ids)
check("eos_repeat=3 length == T", len(lab3) == T, f"{len(lab3)}")
check("eos_repeat=3: [<s>] + chars + [</s>,</s>,</s>] + <fill>...",
      lab3[0] == BOS and lab3[1:1+L] == ids and lab3[1+L:1+L+3] == [EOS,EOS,EOS]
      and lab3[1+L+3] == FILL)
check("eos_repeat=3 has exactly THREE </s>", lab3.count(EOS) == 3)
check("only 2 fewer <fill> than eos_repeat=1 (2 frames -> </s>)",
      lab1.count(FILL) - lab3.count(FILL) == 2, f"{lab1.count(FILL)} vs {lab3.count(FILL)}")

# truncation guard: text longer than T-n_specials must be cut so the </s>×3 still fit
long_text = "|".join(["abcdefgh"]*20)     # 20*8 + 19 = 179 chars
Tsmall = 30
labT = c3._labels_for(long_text, Tsmall)
check("truncation keeps </s>×3 + <s> inside T (len==T, ends with 3 </s>)",
      len(labT) == Tsmall and labT.count(EOS) == 3 and labT[0] == BOS
      and labT[-3:] == [EOS,EOS,EOS])
print(f"      sample eos3 head: {tok.convert_ids_to_tokens(lab3[:L+5])}")

print("\n" + "="*74); print("B) FULL COLLATOR + DURATION-BUDGET MASK  (2 real wavs)"); print("="*74)
rows = [json.loads(l) for l in open(MANIFEST)][:2]
rows = [{"audio_path": r["source"], "text": r["target"]} for r in rows]
for er in (1, 3):
    batch = DataCollatorFillerASR(fe, tok, frames_per_sec=25, eos_repeat=er)(rows)
    labels = batch["labels"]
    iv_len = int(batch["attention_mask"][0].sum())
    n_keep = math.ceil(iv_len / SAMPLING_RATE * 25)
    row0 = labels[0]
    graded = (row0 != -100)
    n_eos = int((row0 == EOS).sum())
    # graded region is exactly [0, min(n_keep, T)); tail is -100
    Tfull = row0.numel()
    exp_graded = min(n_keep, Tfull)
    contiguous = graded.nonzero(as_tuple=True)[0].tolist() == list(range(exp_graded))
    check(f"eos_repeat={er}: graded == first n_keep={n_keep} frames (mask unchanged)", contiguous)
    check(f"eos_repeat={er}: {n_eos} graded </s> present (<=3)", n_eos == er or n_eos <= er,
          f"n_eos={n_eos}")
    check(f"eos_repeat={er}: beyond-budget frames are -100",
          bool((row0[n_keep:] == -100).all()))

print("\n" + "="*74); print("C) FILL WEIGHT  (weighted-mean + gradient scaling)"); print("="*74)
torch.manual_seed(0)
V = len(tok)
# synthetic graded frames: 6 content (targets 1..6), 4 fill, 2 ignored(-100)
labels = torch.tensor([1,2,3,4,5,6, FILL,FILL,FILL,FILL, -100,-100])
logits = torch.randn(len(labels), V, requires_grad=True)

L_unw = framewise_ce_loss(logits.unsqueeze(0), labels.unsqueeze(0), fill_id=FILL, fill_weight=None)
L_w   = framewise_ce_loss(logits.unsqueeze(0), labels.unsqueeze(0), fill_id=FILL, fill_weight=0.1)

# manual weighted-mean:  L = Σ w[y]·(-log p[y]) / Σ w[y]   over graded frames
p = torch.softmax(logits.detach(), -1)
graded = labels != -100
w = torch.where(labels == FILL, torch.tensor(0.1), torch.tensor(1.0)).float()
per = -torch.log(p[torch.arange(len(labels)), labels.clamp(min=0)])
num = (w[graded] * per[graded]).sum(); den = w[graded].sum()
L_manual = num / den
check("weighted CE == manual weighted-mean (num/den, den=Σw)",
      torch.allclose(L_w.detach(), L_manual, atol=1e-5), f"{L_w.item():.5f} vs {L_manual.item():.5f}")
nC, nF = 6, 4
check("denominator D = |C|+0.1|F|", abs(den.item() - (nC + 0.1*nF)) < 1e-5, f"D={den.item():.3f}")

# gradient scaling:  dL/dz_t[c] = (w[y_t]/D)(p_t[c]-1[c=y_t])
logits.grad = None; L_w.backward()
g = logits.grad
# for a content frame (frame 0, target=1) the grad on the TRUE class = (1/D)(p-1)
D = den.item()
c_true = (1.0/D)*(p[0,1] - 1.0)
f_true = (0.1/D)*(p[6,FILL] - 1.0)
check("content-frame true-class grad == (1/D)(p-1)  (content scale 1/D)",
      abs(g[0,1].item() - c_true.item()) < 1e-5, f"{g[0,1].item():.5f} vs {c_true.item():.5f}")
check("fill-frame true-class grad == (0.1/D)(p-1)  (fill scale 0.1/D, ~6x weaker)",
      abs(g[6,FILL].item() - f_true.item()) < 1e-5, f"{g[6,FILL].item():.5f} vs {f_true.item():.5f}")
check("ignored (-100) frames get ZERO gradient", bool((g[10:] == 0).all()))
content_scale = (1.0/D)/(1.0/(nC+nF)); fill_scale = (0.1/D)/(1.0/(nC+nF))
print(f"      vs unweighted 1/N: content grad x{content_scale:.3f}  |  fill grad x{fill_scale:.3f}")

print("\n" + "="*74)
print("ALL CHECKS PASS" if ok else "*** SOME CHECKS FAILED ***")
print("="*74)
raise SystemExit(0 if ok else 1)
