"""A-CMLM NaN diagnostic smoke.

Drives the EXACT training code path (real filler_sa_reference + data modules)
on a handful of clips and checks finiteness at every stage, so we can localize
where a NaN first appears (forward tap / SA / logits / loss / a specific grad).

Run (single GPU):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=10 \
    python nan_smoke.py --dtype bf16 --n_clips 24 --n_steps 12
  ... --dtype fp32   # repeat in fp32

It mirrors train.py: fps=50, eos_repeat=3, mask 0.2-0.3 x10, PE, sa=8, fill_weight=0.03.
"""
import os, sys, argparse, contextlib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load_rows, ManifestDataset  # noqa
from filler_sa_reference import (FILL_TOKEN, build_acmlm_tokenizer, build_acmlm_model,
                                 DataCollatorACMLM, framewise_ce_loss)
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader


def finite(t):
    return bool(torch.isfinite(t).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_clips", type=int, default=24)
    ap.add_argument("--n_steps", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--eos_repeat", type=int, default=3)
    ap.add_argument("--fill_weight", type=float, default=0.03)
    ap.add_argument("--vocab_dir", default=os.path.join(HERE, "runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1"))
    ap.add_argument("--manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl")
    ap.add_argument("--ckpt", default="/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft")
    ap.add_argument("--anomaly", action="store_true", help="enable autograd anomaly detection (slow, precise)")
    args = ap.parse_args()

    dev = args.device
    if dev.startswith("cuda"):
        print(f"[dev] {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
    print(f"[cfg] dtype={args.dtype} fps={args.fps} eos_repeat={args.eos_repeat} "
          f"fill_weight={args.fill_weight} n_clips={args.n_clips} n_steps={args.n_steps} batch={args.batch}")

    torch.manual_seed(0)
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    tok = build_acmlm_tokenizer(args.vocab_dir)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    collator = DataCollatorACMLM(fe, tok, frames_per_sec=args.fps, eos_repeat=args.eos_repeat)
    eval_collator = DataCollatorACMLM(fe, tok, frames_per_sec=args.fps, eos_repeat=args.eos_repeat, fixed_p=1.0)

    rows = load_rows(args.manifest, args.n_clips)
    model = build_acmlm_model(args.ckpt, tok, num_sa_layers=8, use_sinusoidal_pe=True, device=dev,
                              mask_prob=0.20, mask_prob_max=0.30, mask_length=10)

    # sanity: are any freshly-initialized trainable params non-finite at init?
    bad_init = [n for n, p in model.named_parameters() if p.requires_grad and not finite(p)]
    print(f"[init] trainable non-finite params: {bad_init or 'NONE'}")

    loader = DataLoader(ManifestDataset(rows), batch_size=args.batch, shuffle=False, collate_fn=collator)
    eval_loader = DataLoader(ManifestDataset(rows[:args.batch]), batch_size=args.batch, shuffle=False,
                             collate_fn=eval_collator)

    def autocast():
        if args.dtype == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ---- (0) eval-style forward (eval mode, no grad) : mirrors [eval @ 0] ----
    model.eval()
    with torch.no_grad():
        b = next(iter(eval_loader))
        labels = b.pop("labels").to(dev)
        inp = {k: v.to(dev) for k, v in b.items()}
        with autocast():
            lg = model(**inp).logits.float()
        m = min(lg.shape[1], labels.shape[1])
        loss = framewise_ce_loss(lg[:, :m], labels[:, :m], fill_id=fill_id, fill_weight=args.fill_weight)
        print(f"[eval] logits_finite={finite(lg)} loss={loss.item():.4f} loss_finite={finite(loss)} "
              f"graded_frames={(labels[:, :m] != -100).sum().item()}")

    # ---- (1..N) training steps with per-stage finite checks ----
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2e-4, weight_decay=0.005, betas=(0.9, 0.98), eps=1e-6)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    model.train()
    step = 0
    for b in loader:
        step += 1
        labels = b.pop("labels").to(dev)
        inp = {k: v.to(dev) for k, v in b.items()}
        graded = (labels != -100).sum().item()
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(**inp).logits
            lg = out.float()
        m = min(lg.shape[1], labels.shape[1])
        loss = framewise_ce_loss(lg[:, :m], labels[:, :m], fill_id=fill_id, fill_weight=args.fill_weight)
        logits_ok = finite(lg)
        loss_ok = finite(loss)
        loss.backward()
        # find the FIRST trainable param with a non-finite grad
        bad_grad = None
        for n, p in trainable:
            if p.grad is not None and not finite(p.grad):
                bad_grad = n
                break
        gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        flag = "" if (logits_ok and loss_ok and bad_grad is None and finite(gnorm)) else "   <<< NON-FINITE"
        print(f"[step {step:2d}] graded={graded:5d} logits_finite={logits_ok} loss={loss.item():.4f} "
              f"loss_finite={loss_ok} first_bad_grad={bad_grad} gnorm={gnorm:.3f}{flag}")
        if step >= args.n_steps:
            break
    print("[done]")


if __name__ == "__main__":
    main()
