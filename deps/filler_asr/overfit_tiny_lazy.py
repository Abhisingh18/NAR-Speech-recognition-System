"""Gate-2 "can it learn?" smoke test -- LAZY data path (no prepared Arrow cache).

Same intent as overfit_tiny.py: take ~8 real clips, train ONLY the new SA layers
+ lm_head (HuBERT frozen) for a few hundred steps with the EXACT training loss
(framewise CrossEntropyLoss(ignore_index=-100) over the duration-budgeted, <fill>-
padded labels from DataCollatorLazyFillerASR). If the model is wired correctly it
should drive loss -> ~0 and reproduce the (graded-region) transcripts on these 8
clips. If it CANNOT memorize 8 clips, the problem is a bug/LR/init/stability issue.
If it CAN overfit 8 clips but the 960h run still gives WER~1.0, the supervision
signal (positional <fill> target) is the problem, not the model.

Differs from overfit_tiny.py only in the data source: it reads the lazy jsonl
manifest + builds labels at batch time (data_local.DataCollatorLazyFillerASR),
instead of data_cached.build_cached_datasets (which needs the 220GB cache).

Run on one GPU:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python overfit_tiny_lazy.py
"""
import json
import argparse

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertConfig

from model_sa import FillerHubertSAModel
from data_local import load_manifest, DataCollatorLazyFillerASR
from infer_test_clean_local import build_tokenizer

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))


def decode(ids, tok):
    specials = {tok.pad_token_id, tok.bos_token_id, tok.eos_token_id,
                tok.convert_tokens_to_ids("<fill>"), tok.unk_token_id}
    chars = [tok.convert_ids_to_tokens(int(i)) for i in ids if int(i) not in specials]
    return "".join(chars).replace("|", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="config_xlarge_sa_2gpu_lazy.json")
    ap.add_argument("--tokenizer_dir",
                    default="experiments/Hubert_SA_2gpu_lazy_lr2e3",
                    help="dir with the 33-token vocab.json + added_tokens.json")
    ap.add_argument("--n", type=int, default=8, help="number of clips to overfit")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    cfg = json.load(open(args.config_path))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = build_tokenizer(args.tokenizer_dir)           # 33-token char tok incl <fill>=32
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )

    rows = load_manifest(cfg["train_jsonl"], limit=args.n)  # [{audio_path, text}, ...]
    collator = DataCollatorLazyFillerASR(fe, tok)           # loads audio + builds labels
    batch = collator(rows)
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
    model.unfreeze_all_except_feature_extractor(freeze_feature_extractor=True)  # SA+head only
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"device={dev}  clips={len(rows)}  trainable params={n_train/1e6:.2f}M  "
          f"lr={cfg['learning_rate']}  steps={args.steps}", flush=True)
    print(f"label frames T (per clip): {[int((labels[i] != -100).sum()) for i in range(len(rows))]} graded", flush=True)

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
                  f"post-norm instability. Lower lr or add warmup.", flush=True)
            raise SystemExit(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            with torch.no_grad():
                graded = labels[:, :m] != -100
                acc = ((logits[:, :m].argmax(-1) == labels[:, :m]) & graded).sum().item() \
                    / max(1, graded.sum().item())
            print(f"step {step:4d}  loss={loss.item():.4f}  graded-frame-acc={acc:.3f}", flush=True)

    # reproduction check (graded region only)
    model.eval()
    with torch.no_grad():
        pred = model(**batch).logits.argmax(-1)
    print("\n--- reproduction (graded region only) ---", flush=True)
    n_ok = 0
    for i in range(len(rows)):
        gold = decode(labels[i][labels[i] != -100], tok)
        keep = (labels[i] != -100).nonzero(as_tuple=True)[0]
        hyp = decode(pred[i][keep], tok)
        ok = hyp == gold
        n_ok += int(ok)
        print(f"[{i}] gold: {gold}", flush=True)
        print(f"    hyp : {hyp}   {'OK' if ok else 'MISMATCH'}", flush=True)
    print(f"\nreproduced {n_ok}/{len(rows)} clips exactly", flush=True)


if __name__ == "__main__":
    main()
