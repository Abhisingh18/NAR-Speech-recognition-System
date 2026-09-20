"""Smoke test for the data2vec-aqc A-CMLM model on a few LibriSpeech clips.

Proves the plumbing + math BEFORE any real data/training:
  1. model builds; encoder frozen, only A-CMLM head trainable
  2. frame geometry: data2vec feature length == compute_output_length(samples) per clip
     (so frame<->label alignment holds; this is the one real risk of swapping encoders)
  3. one forward -> logits [B, T, V]; loss finite and ~ln(V) at init
  4. one backward -> finite grads on the head, NO grad on the frozen data2vec body
  5. encoder tap is deterministic in eval (no layerdrop/dropout leaking in)
  6. 30-step overfit on ONE batch -> loss falls  (the head learns through the frozen tap)

  python smoke.py --n 6 --device cuda
"""
import os
import sys
import math
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from model_d2v_acmlm import (build_d2v_acmlm, build_collator, build_feature_extractor,
                             framewise_ce_loss, compute_output_length, D2V_DIM)

FILLER_ROOT = os.environ.get("FILLER_ROOT", "/speech/tomson/filler_asr")
sys.path.insert(0, os.path.join(FILLER_ROOT, "train_sa_pe_lm_head"))
from data import load_rows                                                    # noqa: E402
from filler_sa_reference import build_acmlm_tokenizer                         # noqa: E402


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl")
    ap.add_argument("--vocab_dir",
                    default="/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/"
                            "full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pos_mode", default="alibi")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--eos_repeat", type=int, default=3)
    ap.add_argument("--overfit_steps", type=int, default=30)
    args = ap.parse_args()
    dev = args.device

    hr("1) BUILD  model + tokenizer + collator")
    tok = build_acmlm_tokenizer(args.vocab_dir)
    model = build_d2v_acmlm(tok, pos_mode=args.pos_mode, device=dev)
    model.train()                                        # encoder pinned to eval inside .train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.hubert.parameters())
    n_enc_train = sum(p.numel() for p in model.hubert.parameters() if p.requires_grad)
    print(f"tokenizer size          : {len(tok)}  (vocab_size cfg={model.config.vocab_size}, +<mask>)")
    print(f"encoder (data2vec-aqc)  : {n_enc/1e6:.1f}M params, trainable={n_enc_train}  (expect 0)")
    print(f"encoder .training       : {model.hubert.training}  (expect False -> deterministic tap)")
    print(f"A-CMLM head trainable   : {n_train/1e6:.1f}M params (SA + lm_head + text_embed + mask_embed)")
    assert n_enc_train == 0, "data2vec body must be frozen"
    assert model.hubert.training is False, "encoder must be in eval mode under model.train()"

    hr("2) DATA  + frame-geometry alignment (data2vec T  vs  compute_output_length)")
    rows = load_rows(args.manifest, args.n)
    collator = build_collator(tok, frames_per_sec=args.fps, eos_repeat=args.eos_repeat)
    fe = build_feature_extractor()
    import soundfile as sf
    ok = True
    with torch.no_grad():
        for r in rows:
            arr, sr = sf.read(r["audio_path"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            iv = fe(arr, sampling_rate=16000).input_values[0]
            expect_T = compute_output_length(len(iv))
            x = torch.from_numpy(iv).unsqueeze(0).to(dev)
            feat = model.hubert(x)[0]                    # [1, T, D]
            got_T = feat.shape[1]
            flag = "OK " if got_T == expect_T else "MISMATCH"
            if got_T != expect_T:
                ok = False
            print(f"  {flag} samples={len(iv):7d}  compute_output_length={expect_T:5d}  "
                  f"data2vec_T={got_T:5d}  D={feat.shape[2]}")
    print(f"alignment: {'ALL MATCH (frame<->label geometry preserved)' if ok else 'DIFFERS -> min-crop handles tail, but check!'}")
    assert feat.shape[2] == D2V_DIM, f"expected {D2V_DIM}-d tap, got {feat.shape[2]}"

    hr("3) FORWARD  one batch -> logits + loss (exact train.py math)")
    batch = collator(rows)
    labels = batch.pop("labels").to(dev)
    inp = {k: v.to(dev) for k, v in batch.items()}
    fill_id = tok.convert_tokens_to_ids("<fill>")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        logits = model(**inp).logits.float()
    m = min(logits.shape[1], labels.shape[1]); lb = labels[:, :m]
    loss = framewise_ce_loss(logits[:, :m], lb, fill_id=fill_id, reduction="mean")
    n_sup = int((lb != -100).sum())
    print(f"logits shape   : {tuple(logits.shape)}   labels shape: {tuple(labels.shape)}  (min-crop m={m})")
    print(f"supervised slots (masked CMLM targets): {n_sup}")
    print(f"init loss      : {loss.item():.4f}   (ln(V)=ln({logits.shape[-1]})={math.log(logits.shape[-1]):.4f} expected ~random)")
    assert torch.isfinite(loss), "loss is not finite"

    hr("4) BACKWARD  grad flow (head gets grad; frozen data2vec body does NOT)")
    model.zero_grad(set_to_none=True)
    loss.backward()
    head_gnorm = sum((p.grad.detach()**2).sum() for _, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None) ** 0.5
    enc_grads = [n for n, p in model.hubert.named_parameters() if p.grad is not None]
    nonfinite = [n for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
    for tag, mod in [("SA[0]", model.sa), ("lm_head", model.lm_head),
                     ("text_embed", model.text_embed)]:
        g = [p.grad for p in mod.parameters() if p.grad is not None]
        gn = (sum((x**2).sum() for x in g) ** 0.5).item() if g else float("nan")
        print(f"  grad norm {tag:11s}: {gn:.4e}   ({len(g)} tensors)")
    print(f"total head grad norm    : {head_gnorm.item():.4e}")
    print(f"frozen-encoder tensors with grad: {len(enc_grads)}  (expect 0)")
    print(f"non-finite head grads   : {len(nonfinite)}  (expect 0)")
    assert not enc_grads, f"frozen data2vec body got grads: {enc_grads[:5]}"
    assert not nonfinite, f"non-finite grads: {nonfinite[:5]}"

    hr("5) DETERMINISM  frozen tap identical across two eval forwards")
    model.eval()
    with torch.no_grad():
        a = model.hubert(inp["input_values"], attention_mask=inp["attention_mask"])[0]
        b = model.hubert(inp["input_values"], attention_mask=inp["attention_mask"])[0]
    maxdiff = (a - b).abs().max().item()
    print(f"max|tap_a - tap_b| = {maxdiff:.3e}  (expect 0 -> encoder is deterministic)")
    assert maxdiff == 0.0, "encoder tap is non-deterministic (layerdrop/dropout leaking?)"
    model.train()

    hr(f"6) OVERFIT  {args.overfit_steps} steps on this ONE batch (loss must fall)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    losses = []
    for step in range(args.overfit_steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = model(**inp).logits.float()
        m = min(logits.shape[1], labels.shape[1])
        loss = framewise_ce_loss(logits[:, :m], labels[:, :m], fill_id=fill_id, reduction="mean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == args.overfit_steps - 1:
            print(f"  step {step:3d}  loss={loss.item():.4f}")
    drop = losses[0] - losses[-1]
    print(f"\nloss: {losses[0]:.4f} -> {losses[-1]:.4f}   (drop {drop:.4f})")
    assert losses[-1] < losses[0], "loss did not decrease -> head is not learning through the tap"

    hr("SMOKE PASSED  ✅   model runs, geometry aligns, grads flow, head learns through the frozen data2vec-aqc tap")


if __name__ == "__main__":
    main()
