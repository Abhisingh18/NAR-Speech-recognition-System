"""Overfit 10 clips / 100 epochs on the PRETRAINED HuBERT-xlarge (ll60k).

Same architecture + same training strategy as filler_sa_reference.py — the ONLY
difference is the starting encoder weights:

    ls960-ft : CTC-FINETUNED on LibriSpeech 960h  (HubertForCTC; char-aware features)
               -> used in every previous overfit run.
    ll60k    : SELF-SUPERVISED pretrain on Libri-Light 60k h  (HubertModel; no CTC head)
               -> hubert-xlarge-ls960-ft was finetuned FROM this, so identical architecture.
               from_pretrained maps its encoder into self.hubert; lm_head starts fresh.

Everything else is reused unchanged from filler_sa_reference.py:
  frozen HuBERT tap -> sinusoidal PE -> 2 post-norm SA layers -> lm_head,
  positional <fill> target + duration-budget loss mask, framewise CrossEntropy.

This isolates the question: does starting from the generic self-supervised encoder
(instead of the CTC-finetuned one) change the collapse?  (The target is unchanged,
so the expectation is the same <fill>-floor — but this confirms it for ll60k.)

Run on one GPU:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
      python overfit_pretrained_ll60k.py --pe --n 10 --epochs 100 --wandb
"""
import argparse

import torch
from transformers import Wav2Vec2FeatureExtractor

# reuse the reviewed building blocks (model, data/label prep, loss, decode)
from filler_sa_reference import (
    FILL_TOKEN,
    build_tokenizer,
    DataCollatorFillerASR,
    build_model,
    framewise_ce_loss,
    _decode,
)

PRETRAINED_LL60K = "/speech/tomson/filler_asr/models/hubert-xlarge-ll60k"   # <-- the difference


def main():
    ap = argparse.ArgumentParser(description="Overfit 10 clips on PRETRAINED ll60k encoder")
    ap.add_argument("--model_checkpoint", default=PRETRAINED_LL60K,
                    help="pretrained (NOT CTC-finetuned) HuBERT-xlarge")
    ap.add_argument("--vocab_dir", default="/speech/tomson/filler_asr/experiments/Hubert_SA_2gpu_lazy_lr2e3")
    ap.add_argument("--train_manifest",
                    default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pe", action="store_true", help="enable sinusoidal PE")
    ap.add_argument("--fill_weight", type=float, default=None, help="down-weight <fill> in CE (optional)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer(args.vocab_dir)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN)

    # --- first N clips -> one full batch (positional <fill> targets + masks) ---
    import json
    rows = [json.loads(l) for l in open(args.train_manifest) if l.strip()][: args.n]
    rows = [{"audio_path": r.get("audio") or r.get("source"),
             "text": r.get("text") or r.get("target")} for r in rows]
    batch = DataCollatorFillerASR(fe, tok)(rows)
    labels = batch.pop("labels").to(dev)
    batch = {k: v.to(dev) for k, v in batch.items()}

    # --- model: frozen PRETRAINED ll60k encoder + PE + 2 SA layers + lm_head ---
    model = build_model(args.model_checkpoint, tok, use_sinusoidal_pe=args.pe, device=dev).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="filler_asr",
                         name=f"ll60k_pretrained_overfit{args.n}_{args.epochs}ep_{'PE' if args.pe else 'noPE'}",
                         config={"encoder": "ll60k_pretrained", "n": args.n, "epochs": args.epochs,
                                 "lr": args.lr, "pe": args.pe, "fill_weight": args.fill_weight,
                                 "trainable_M": round(n_train/1e6, 2)})
        print("wandb run:", wandb.run.url, flush=True)
    print(f"encoder=ll60k(pretrained)  device={dev}  clips={len(rows)}  PE={args.pe}  "
          f"trainable={n_train/1e6:.2f}M  lr={args.lr}  epochs={args.epochs}", flush=True)

    # --- optimize ONLY the SA layers + lm_head with framewise CE ---
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.005)
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        m = min(logits.shape[1], labels.shape[1])
        loss = framewise_ce_loss(logits[:, :m], labels[:, :m], fill_id=fill_id, fill_weight=args.fill_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        with torch.no_grad():
            graded = labels[:, :m] != -100
            acc = ((logits[:, :m].argmax(-1) == labels[:, :m]) & graded).sum().item() / max(1, graded.sum().item())
        if run:
            run.log({"train/loss": loss.item(), "train/graded_frame_acc": acc, "epoch": epoch}, step=epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.4f}  graded-frame-acc={acc:.3f}", flush=True)

    # --- reproduction: FULL per-frame output (everything) + stripped hyp + ref ---
    model.eval()
    with torch.no_grad():
        pred = model(**batch).logits.argmax(-1)
    n_ok = 0
    print("\n--- reproduction (full frames | stripped | ref) ---", flush=True)
    for i in range(len(rows)):
        keep = (labels[i] != -100).nonzero(as_tuple=True)[0]
        full = " ".join(tok.convert_ids_to_tokens(int(x)) for x in pred[i][keep])   # raw, nothing stripped
        hyp = _decode(pred[i][keep], tok)                                            # special-char removed
        gold = _decode(labels[i][labels[i] != -100], tok)
        ok = hyp == gold; n_ok += int(ok)
        print(f"[{i}] Full: {full[:160]}", flush=True)
        print(f"    Hyp : {hyp[:80]}", flush=True)
        print(f"    Ref : {gold[:80]}    {'OK' if ok else 'MISMATCH'}", flush=True)
    print(f"\nreproduced {n_ok}/{len(rows)} clips   (final loss={loss.item():.4f}, acc={acc:.3f})", flush=True)
    if run:
        run.summary["reproduced"] = n_ok
        run.finish()


if __name__ == "__main__":
    main()
