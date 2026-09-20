"""
Gate 2 -- "can it learn?" smoke test. Take a handful of real cached examples and
train ONLY the new SA layers + lm_head (HuBERT frozen) on that tiny set for a few
hundred steps. If the model is wired correctly, framewise CE -> ~0 and the model
reproduces the (un-masked) transcripts. If it cannot memorize ~8 clips, the
problem is a bug / LR / init / stability issue -- fix it BEFORE the 960h run.

Replicates the exact training loss (CrossEntropyLoss(ignore_index=-100) over the
masked labels from DataCollatorForFillerASR) -- model.py:compute_loss.

Run on one GPU:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python overfit_tiny.py
"""
import json
import argparse
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, HubertConfig

from model_sa import FillerHubertSAModel
from collator import DataCollatorForFillerASR, FRAMES_PER_SEC
from data_cached import build_cached_datasets


def decode(ids, tok):
    specials = {tok.pad_token_id, tok.bos_token_id, tok.eos_token_id,
                tok.convert_tokens_to_ids("<fill>"), tok.unk_token_id}
    chars = [tok.convert_ids_to_tokens(int(i)) for i in ids if int(i) not in specials]
    return "".join(chars).replace("|", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="config_xlarge_sa_2gpu.json")
    ap.add_argument("--n", type=int, default=8, help="number of clips to overfit")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    cfg = json.load(open(args.config_path))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = cfg["cache_dir"]
    import os
    if not os.path.exists(os.path.join(cache_dir, "PREPARED")):
        raise SystemExit(
            f"Cache not prepared at {cache_dir}. Run first:\n"
            f"  python prepare_data_local.py --config_path {args.config_path}")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(cache_dir)
    tok = Wav2Vec2CTCTokenizer.from_pretrained(cache_dir)

    ds = build_cached_datasets(cfg, fe, tok)["train"].select(range(args.n))
    collator = DataCollatorForFillerASR(feature_extractor=fe, tokenizer=tok)
    batch = collator([ds[i] for i in range(len(ds))])
    batch = {k: v.to(dev) for k, v in batch.items()}
    labels = batch.pop("labels")

    hc = HubertConfig.from_pretrained(cfg["model_checkpoint"])
    hc.vocab_size = len(tok)
    hc.pad_token_id, hc.bos_token_id, hc.eos_token_id = (
        tok.pad_token_id, tok.bos_token_id, tok.eos_token_id)
    hc.num_sa_layers = cfg.get("num_sa_layers", 4)
    hc.sa_nhead = cfg.get("sa_nhead", hc.num_attention_heads)
    hc.sa_dim_feedforward = cfg.get("sa_dim_feedforward", hc.intermediate_size)
    hc.sa_dropout = cfg.get("sa_dropout", hc.final_dropout)

    model = FillerHubertSAModel.from_pretrained(
        cfg["model_checkpoint"], config=hc, ignore_mismatched_sizes=True).to(dev)
    model.freeze_feature_extractor()
    model.unfreeze_all_except_feature_extractor(freeze_feature_extractor=True)  # train SA+head
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"device={dev}  clips={args.n}  frames_per_sec(mask)={FRAMES_PER_SEC}  "
          f"trainable params={n_train/1e6:.2f}M  lr={cfg['learning_rate']}")

    opt = torch.optim.AdamW(trainable, lr=cfg["learning_rate"],
                            weight_decay=cfg.get("weight_decay", 0.0))
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        m = min(logits.shape[1], labels.shape[1])
        loss = loss_fct(logits[:, :m].reshape(-1, logits.size(-1)),
                        labels[:, :m].reshape(-1))
        if not torch.isfinite(loss):
            print(f"step {step}: NON-FINITE loss ({loss.item()}) -- LR too high / "
                  f"post-norm instability. Lower lr or add warmup."); raise SystemExit(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                graded = labels[:, :m] != -100
                acc = ((logits[:, :m].argmax(-1) == labels[:, :m]) & graded).sum().item() \
                    / max(1, graded.sum().item())
            print(f"step {step:4d}  loss={loss.item():.4f}  graded-frame-acc={acc:.3f}")

    # reproduction check
    model.eval()
    with torch.no_grad():
        pred = model(**batch).logits.argmax(-1)
    print("\n--- reproduction (graded region only) ---")
    for i in range(len(ds)):
        gold = decode(labels[i][labels[i] != -100], tok)
        keep = (labels[i] != -100).nonzero(as_tuple=True)[0]
        hyp = decode(pred[i][keep], tok)
        print(f"[{i}] gold: {gold}")
        print(f"    hyp : {hyp}   {'OK' if hyp == gold else 'MISMATCH'}")


if __name__ == "__main__":
    main()
