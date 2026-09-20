"""
Pre-flight verification of the WHOLE SA pipeline before launching training.
Checks: config, manifests, audio, vocab, collator/labels, model load,
forward, loss, grad-flow. Prints a PASS/FAIL checklist.  Run on CPU.
"""
import os, json, math, sys
import torch, torch.nn as nn, soundfile as sf
from transformers import HubertConfig, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from data_local import build_datasets, write_vocab, DataCollatorLazyFillerASR, load_manifest
from model_sa import FillerHubertSAModel

CFG = sys.argv[1] if len(sys.argv) > 1 else "config_xlarge_sa_full.json"
ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> '+detail) if detail else ''}")

print("="*80); print(f"PRE-FLIGHT CHECK  ({CFG})"); print("="*80)
cfg = json.load(open(CFG))

print("\n1) CONFIG")
check("use_self_attention = true", cfg.get("use_self_attention") is True)
check("num_train_epochs = 20", cfg.get("num_train_epochs") == 20, str(cfg.get("num_train_epochs")))
check("triphase schedule", cfg.get("lr_schedule_type") == "triphase")
check("model_checkpoint exists", os.path.isdir(cfg["model_checkpoint"]), cfg["model_checkpoint"])
check("train_jsonl exists", os.path.isfile(cfg["train_jsonl"]))
check("dev_jsonl exists", os.path.isfile(cfg["dev_jsonl"]))
eff = cfg["per_device_train_batch_size"]*cfg["gradient_accumulation_steps"]*2
print(f"      effective batch (2 GPU) = {cfg['per_device_train_batch_size']}x{cfg['gradient_accumulation_steps']}x2 = {eff}")

print("\n2) MANIFESTS + AUDIO")
tr = load_manifest(cfg["train_jsonl"]); dv = load_manifest(cfg["dev_jsonl"])
check("train rows", len(tr) > 250000, f"{len(tr)} utts")
check("dev rows", len(dv) > 2000, f"{len(dv)} utts")
p0 = tr[0]["audio_path"]
check("sample wav exists", os.path.isfile(p0), p0)
info = sf.info(p0)
check("wav is 16 kHz mono", info.samplerate == 16000 and info.channels == 1,
      f"{info.samplerate}Hz {info.channels}ch")

print("\n3) VOCAB + TOKENIZER")
vocab = write_vocab(cfg["train_jsonl"], "/tmp", "preflight")
tok = Wav2Vec2CTCTokenizer(vocab, unk_token="<unk>", pad_token="<pad>",
        word_delimiter_token="|", bos_token="<s>", eos_token="</s>")
tok.add_special_tokens({"additional_special_tokens": ["<fill>"]})
check("vocab size = 33", len(tok) == 33, str(len(tok)))
check("<fill> id = 32", tok.convert_tokens_to_ids("<fill>") == 32)

print("\n4) COLLATOR + FILLER LABELS")
fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                              do_normalize=True, return_attention_mask=True)
ds = build_datasets(cfg["train_jsonl"], cfg["dev_jsonl"], dev_limit=4)
coll = DataCollatorLazyFillerASR(fe, tok)
batch = coll([ds["test"][0], ds["test"][1]])
B = batch["input_values"].shape[0]; T = batch["labels"].shape[1]
check("batch keys", set(batch) == {"input_values", "attention_mask", "labels"}, str(list(batch)))
check("input_values finite", torch.isfinite(batch["input_values"]).all().item())
check("labels have <fill> or content", (batch["labels"] >= 0).any().item())
check("labels mask padding to -100", (batch["labels"] == -100).any().item() or B == 1)
print(f"      input_values {tuple(batch['input_values'].shape)}  labels {tuple(batch['labels'].shape)}")

print("\n5) MODEL LOAD + FORWARD + LOSS + GRAD")
hc = HubertConfig.from_pretrained(cfg["model_checkpoint"]); hc.vocab_size = len(tok)
hc.num_sa_layers = cfg["num_sa_layers"]; hc.sa_nhead = cfg["sa_nhead"]
hc.sa_dim_feedforward = cfg["sa_dim_feedforward"]; hc.sa_dropout = cfg["sa_dropout"]
m = FillerHubertSAModel.from_pretrained(cfg["model_checkpoint"], config=hc, ignore_mismatched_sizes=True)
m.freeze_feature_extractor(); m.unfreeze_all_except_feature_extractor(); m.train()
out = m(input_values=batch["input_values"], attention_mask=batch["attention_mask"])
check("logits shape (B,T,33)", tuple(out.logits.shape) == (B, T, 33), str(tuple(out.logits.shape)))
check("logits finite", torch.isfinite(out.logits).all().item())
loss = nn.CrossEntropyLoss(ignore_index=-100)(out.logits.reshape(-1, 33), batch["labels"].reshape(-1))
check("loss finite ~ ln(33)", torch.isfinite(loss).item() and 2.5 < loss.item() < 5.0, f"{loss.item():.3f}")
loss.backward()
def gn(ps): return math.sqrt(sum(p.grad.float().pow(2).sum().item() for p in ps if p.grad is not None))
check("grad flows to self.sa", gn(m.sa.parameters()) > 0, f"{gn(m.sa.parameters()):.3f}")
check("grad flows to lm_head", gn(m.lm_head.parameters()) > 0, f"{gn(m.lm_head.parameters()):.3f}")
check("hubert frozen (0 grads)", sum(1 for p in m.hubert.parameters() if p.grad is not None) == 0)
tr_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
check("trainable ~ 39.4M", abs(tr_params/1e6 - 39.40) < 0.5, f"{tr_params/1e6:.2f}M")

import shutil; shutil.rmtree("/tmp/preflight", ignore_errors=True)
print("\n" + "="*80)
print("PRE-FLIGHT RESULT:", "ALL PASS — pipeline verified, safe to launch." if ok else "FAILURES ABOVE — do NOT launch.")
print("="*80)
sys.exit(0 if ok else 1)
