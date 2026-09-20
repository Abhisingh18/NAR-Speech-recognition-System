"""Overfit N clips for E epochs (full-batch) and log curves to wandb. No checkpoints.

Same model / loss / label pipeline as the real lazy training:
  * FillerHubertSAModel (frozen HuBERT-xlarge + 2 new SA layers + new lm_head)
  * DataCollatorLazyFillerASR  -> positional <fill> labels, duration budget mask
  * CrossEntropyLoss(ignore_index=-100), constant lr from the config (no warmup/sched)

Logs per epoch to wandb: train/loss, train/graded_frame_acc, learning_rate.
This is a "can it even memorize N clips?" diagnostic -- watch whether loss -> 0
(it can learn) or plateaus at the <fill>-prior floor (supervision is unlearnable).

Run on one GPU:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python overfit_wandb.py
"""
import os
import json
import argparse

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertConfig

import wandb
from model_sa import FillerHubertSAModel
from data_local import load_manifest, DataCollatorLazyFillerASR
from infer_test_clean_local import build_tokenizer


def decode(ids, tok):
    specials = {tok.pad_token_id, tok.bos_token_id, tok.eos_token_id,
                tok.convert_tokens_to_ids("<fill>"), tok.unk_token_id}
    chars = [tok.convert_ids_to_tokens(int(i)) for i in ids if int(i) not in specials]
    return "".join(chars).replace("|", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="config_xlarge_sa_2gpu_lazy.json")
    ap.add_argument("--tokenizer_dir", default="experiments/Hubert_SA_2gpu_lazy_lr2e3")
    ap.add_argument("--n", type=int, default=10, help="number of clips to overfit")
    ap.add_argument("--epochs", type=int, default=100, help="full-batch updates")
    ap.add_argument("--pe", action="store_true",
                    help="add sinusoidal positional encoding to the SA layers")
    ap.add_argument("--pe_scale", type=float, default=1.0)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config_path))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lr = cfg["learning_rate"]
    pe_tag = f"_PE{('' if args.pe_scale == 1.0 else f'x{args.pe_scale:g}')}" if args.pe else "_noPE"
    run_name = args.run_name or f"overfit{args.n}_{args.epochs}ep_lr{lr:g}{pe_tag}"

    tok = build_tokenizer(args.tokenizer_dir)
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )

    rows = load_manifest(cfg["train_jsonl"], limit=args.n)
    collator = DataCollatorLazyFillerASR(fe, tok)
    batch = collator(rows)
    batch = {k: v.to(dev) for k, v in batch.items()}
    labels = batch.pop("labels")
    graded_per_clip = [int((labels[i] != -100).sum()) for i in range(len(rows))]

    hc = HubertConfig.from_pretrained(cfg["model_checkpoint"])
    hc.vocab_size = len(tok)
    hc.pad_token_id, hc.bos_token_id, hc.eos_token_id = (
        tok.pad_token_id, tok.bos_token_id, tok.eos_token_id)
    hc.num_sa_layers = cfg.get("num_sa_layers", 4)
    hc.sa_nhead = cfg.get("sa_nhead", hc.num_attention_heads)
    hc.sa_dim_feedforward = cfg.get("sa_dim_feedforward", hc.intermediate_size)
    hc.sa_dropout = cfg.get("sa_dropout", hc.final_dropout)
    hc.use_sinusoidal_pe = args.pe
    hc.pe_scale = args.pe_scale

    model = FillerHubertSAModel.from_pretrained(
        cfg["model_checkpoint"], config=hc, ignore_mismatched_sizes=True).to(dev)
    model.freeze_feature_extractor()
    model.unfreeze_all_except_feature_extractor(freeze_feature_extractor=True)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)

    wandb.init(
        project=cfg.get("wandb_project", "filler_asr"),
        name=run_name,
        config={
            "task": "overfit_diagnostic",
            "n_clips": args.n, "epochs": args.epochs, "learning_rate": lr,
            "weight_decay": cfg.get("weight_decay", 0.0),
            "num_sa_layers": hc.num_sa_layers, "sa_nhead": hc.sa_nhead,
            "sa_dim_feedforward": hc.sa_dim_feedforward,
            "use_sinusoidal_pe": hc.use_sinusoidal_pe, "pe_scale": hc.pe_scale,
            "trainable_M": round(n_train / 1e6, 2),
            "graded_frames_per_clip": graded_per_clip,
            "model_checkpoint": cfg["model_checkpoint"],
        },
    )
    print(f"wandb run: {wandb.run.url}", flush=True)
    print(f"device={dev}  clips={len(rows)}  trainable={n_train/1e6:.2f}M  lr={lr}  "
          f"epochs={args.epochs}  graded/clip={graded_per_clip}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg.get("weight_decay", 0.0))
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        m = min(logits.shape[1], labels.shape[1])
        loss = loss_fct(logits[:, :m].reshape(-1, logits.size(-1)),
                        labels[:, :m].reshape(-1))
        if not torch.isfinite(loss):
            print(f"epoch {epoch}: NON-FINITE loss -- lr too high / instability", flush=True)
            wandb.finish(exit_code=1); raise SystemExit(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        with torch.no_grad():
            graded = labels[:, :m] != -100
            acc = ((logits[:, :m].argmax(-1) == labels[:, :m]) & graded).sum().item() \
                / max(1, graded.sum().item())
        wandb.log({"train/loss": loss.item(), "train/graded_frame_acc": acc,
                   "learning_rate": lr, "epoch": epoch}, step=epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.4f}  graded-frame-acc={acc:.3f}", flush=True)

    # reproduction check
    model.eval()
    with torch.no_grad():
        pred = model(**batch).logits.argmax(-1)
    print("\n--- reproduction (graded region only) ---", flush=True)
    n_ok = 0
    table = wandb.Table(columns=["i", "gold", "hyp", "match"])
    for i in range(len(rows)):
        gold = decode(labels[i][labels[i] != -100], tok)
        keep = (labels[i] != -100).nonzero(as_tuple=True)[0]
        hyp = decode(pred[i][keep], tok)
        ok = hyp == gold
        n_ok += int(ok)
        table.add_data(i, gold, hyp, "OK" if ok else "MISMATCH")
        print(f"[{i}] gold: {gold}\n    hyp : {hyp}   {'OK' if ok else 'MISMATCH'}", flush=True)

    wandb.log({"reproduction": table})
    wandb.run.summary["reproduced_exact"] = n_ok
    wandb.run.summary["reproduced_frac"] = n_ok / len(rows)
    wandb.run.summary["final_loss"] = loss.item()
    wandb.run.summary["final_graded_frame_acc"] = acc
    print(f"\nreproduced {n_ok}/{len(rows)} clips exactly", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
