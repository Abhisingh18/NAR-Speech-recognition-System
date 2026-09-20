"""Scaled training (mini-batch) with FULL diagnostic plotting to wandb.

Same architecture / target as filler_sa_reference.py (frozen HuBERT tap + sinusoidal PE
+ 2 post-norm SA layers + lm_head; positional <fill> target + duration-budget mask;
framewise CE) — but mini-batch over N train clips for E epochs, with a held-out dev set
and every diagnostic plot:

  curves      : train/loss, eval/loss, eval/WER, eval/CER, eval/SER, lr, grad_norm
  collapse    : <fill> prediction-rate, content-frame-acc, fill-frame-acc,
                graded-frame-acc, prediction entropy, #distinct non-<fill> tokens
  distribution: predicted-token histogram, target-token histogram
  position    : accuracy vs frame-position (10 bins)
  health      : samples/sec
  qualitative : sample Table (FULL per-frame output | stripped hyp | ref), every eval

Run (1 Ada GPU):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
    python train_scale_plots.py --n 10000 --epochs 100 --pe --wandb
"""
import os, re, json, time, math, argparse

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup

from filler_sa_reference import (
    FILL_TOKEN, build_tokenizer, DataCollatorFillerASR, build_model, framewise_ce_loss,
)

CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"]'


# ---------- helpers ----------
def norm_ref(text):
    return re.sub(CHARS_TO_IGNORE, "", text).lower().strip()

def levenshtein(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m; ri = ref[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ri == hyp[j - 1] else 1))
        prev = cur
    return prev[m]

def load_rows(path, limit):
    rows = []
    for l in open(path):
        if not l.strip(): continue
        d = json.loads(l)
        rows.append({"audio_path": d.get("audio") or d.get("source"),
                     "text": d.get("text") or d.get("target")})
        if len(rows) >= limit: break
    return rows

class ListDS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


# ---------- evaluation: every metric + sample dump ----------
@torch.no_grad()
def evaluate(model, loader, tok, fill_id, device, dump_k=6):
    model.eval()
    V = len(tok)
    w_err = w_tot = c_err = c_tot = s_err = s_tot = 0
    graded_correct = graded_total = 0
    content_correct = content_total = 0
    fill_correct = fill_total = 0
    fill_pred = 0
    ent_sum = ent_cnt = 0.0
    pred_hist = torch.zeros(V); tgt_hist = torch.zeros(V)
    POSB = 10; pos_corr = torch.zeros(POSB); pos_tot = torch.zeros(POSB)
    loss_sum = loss_cnt = 0.0
    samples = []

    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits
        logits = logits.float()
        m = min(logits.shape[1], labels.shape[1])
        lg, lb = logits[:, :m], labels[:, :m]
        loss_sum += framewise_ce_loss(lg, lb, fill_id=fill_id).item(); loss_cnt += 1

        pred = lg.argmax(-1)                                   # (B, m)
        graded = lb != -100
        # ---- frame accuracies ----
        corr = (pred == lb) & graded
        graded_correct += corr.sum().item(); graded_total += graded.sum().item()
        is_fill = graded & (lb == fill_id)
        is_cont = graded & (lb != fill_id)
        fill_correct += (corr & is_fill).sum().item(); fill_total += is_fill.sum().item()
        content_correct += (corr & is_cont).sum().item(); content_total += is_cont.sum().item()
        # ---- <fill> prediction rate ----
        fill_pred += ((pred == fill_id) & graded).sum().item()
        # ---- entropy + histograms ----
        p = torch.softmax(lg, -1)
        ent = -(p * (p.clamp_min(1e-9)).log()).sum(-1)        # (B, m)
        ent_sum += ent[graded].sum().item(); ent_cnt += graded.sum().item()
        pred_hist += torch.bincount(pred[graded].cpu(), minlength=V).float()
        tgt_hist += torch.bincount(lb[graded].cpu(), minlength=V).float()
        # ---- position-binned accuracy ----
        B, T = lb.shape
        posidx = torch.arange(T, device=device).expand(B, T)
        b = (posidx.float() / max(1, T) * POSB).long().clamp(max=POSB - 1)
        for k in range(POSB):
            sel = graded & (b == k)
            pos_tot[k] += sel.sum().item(); pos_corr[k] += (corr & sel).sum().item()
        # ---- WER / CER / SER + sample dump ----
        ids = pred  # already argmax
        hyps = tok.batch_decode(ids, skip_special_tokens=True, group_tokens=False)
        for i in range(lb.shape[0]):
            keep = (lb[i] != -100).nonzero(as_tuple=True)[0]
            ref = norm_ref(tok.batch_decode(lb[i][keep].masked_fill(lb[i][keep] < 0, fill_id).unsqueeze(0),
                                            skip_special_tokens=True, group_tokens=False)[0])
            hyp = hyps[i].strip()
            rw, hw = ref.split(), hyp.split()
            w_err += levenshtein(rw, hw); w_tot += len(rw)
            c_err += levenshtein(ref, hyp); c_tot += len(ref)
            s_err += int(hyp != ref); s_tot += 1
            if len(samples) < dump_k:
                full = " ".join(tok.convert_ids_to_tokens(int(x)) for x in ids[i][keep])
                samples.append((full, hyp, ref))

    model.train()
    nf = max(1, graded_total)
    return {
        "eval/loss": loss_sum / max(1, loss_cnt),
        "eval/WER": w_err / max(1, w_tot), "eval/CER": c_err / max(1, c_tot),
        "eval/SER": s_err / max(1, s_tot),
        "eval/graded_frame_acc": graded_correct / nf,
        "eval/content_frame_acc": content_correct / max(1, content_total),
        "eval/fill_frame_acc": fill_correct / max(1, fill_total),
        "eval/fill_pred_rate": fill_pred / nf,
        "eval/pred_entropy": ent_sum / max(1, ent_cnt),
        "eval/distinct_nonfill_tokens": int(((pred_hist > 0).sum().item()) - (1 if pred_hist[fill_id] > 0 else 0)),
    }, (pred_hist, tgt_hist), (pos_corr / pos_tot.clamp_min(1)), samples


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", default="/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft")
    ap.add_argument("--vocab_dir", default="/speech/tomson/filler_asr/experiments/Hubert_SA_2gpu_lazy_lr2e3")
    ap.add_argument("--train_manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl")
    ap.add_argument("--dev_manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dev_n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.005)
    ap.add_argument("--pe", action="store_true")
    ap.add_argument("--fill_weight", type=float, default=None)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer(args.vocab_dir)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    collator = DataCollatorFillerASR(fe, tok)

    train_rows = load_rows(args.train_manifest, args.n)
    dev_rows = load_rows(args.dev_manifest, args.dev_n)
    train_loader = DataLoader(ListDS(train_rows), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.num_workers, drop_last=True)
    eval_loader = DataLoader(ListDS(dev_rows), batch_size=args.eval_batch, shuffle=False,
                             collate_fn=collator, num_workers=2)

    model = build_model(args.model_checkpoint, tok, use_sinusoidal_pe=args.pe, device=dev).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    total_steps = args.epochs * len(train_loader)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, total_steps)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="filler_asr",
                         name=f"scale{args.n}_{args.epochs}ep_{'PE' if args.pe else 'noPE'}_{os.path.basename(args.model_checkpoint)}",
                         config={**vars(args), "trainable_M": round(n_train / 1e6, 2), "total_steps": total_steps})
        print("wandb run:", wandb.run.url, flush=True)
    print(f"train={len(train_rows)} dev={len(dev_rows)} steps/epoch={len(train_loader)} "
          f"total_steps={total_steps} trainable={n_train/1e6:.2f}M PE={args.pe}", flush=True)

    def log_eval(step):
        scal, (predh, tgth), posacc, samples = evaluate(model, eval_loader, tok, fill_id, dev)
        msg = (f"[eval @ {step}] WER={scal['eval/WER']:.3f} CER={scal['eval/CER']:.3f} "
               f"SER={scal['eval/SER']:.3f} loss={scal['eval/loss']:.3f} "
               f"content_acc={scal['eval/content_frame_acc']:.3f} fill_rate={scal['eval/fill_pred_rate']:.3f} "
               f"entropy={scal['eval/pred_entropy']:.3f} distinct={scal['eval/distinct_nonfill_tokens']}")
        print(msg, flush=True)
        if samples:
            f0, h0, r0 = samples[0]
            print(f"   sample Full: {f0[:140]}\n   sample Hyp : {h0[:80]}\n   sample Ref : {r0[:80]}", flush=True)
        if run:
            log = dict(scal); log["step"] = step
            for k in range(len(posacc)):
                log[f"pos_acc/bin{k}"] = float(posacc[k])
            # histograms over token ids (weighted)
            try:
                import wandb
                toks = list(range(len(tok)))
                log["dist/pred_tokens"] = wandb.Histogram(np_histogram=(predh.numpy(), np.arange(len(tok) + 1)))
                log["dist/target_tokens"] = wandb.Histogram(np_histogram=(tgth.numpy(), np.arange(len(tok) + 1)))
                tbl = wandb.Table(columns=["step", "full_frames", "hyp_stripped", "ref"])
                for fl, hy, rf in samples:
                    tbl.add_data(step, fl, hy, rf)
                log["samples"] = tbl
            except Exception as e:
                print("plot-log warn:", e, flush=True)
            run.log(log, step=step)

    step = 0; t0 = time.time(); seen = 0
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            labels = batch.pop("labels").to(dev)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.float()
            m = min(logits.shape[1], labels.shape[1])
            loss = framewise_ce_loss(logits[:, :m], labels[:, :m], fill_id=fill_id, fill_weight=args.fill_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
            step += 1; seen += labels.shape[0]

            if step % args.log_steps == 0:
                with torch.no_grad():
                    g = labels[:, :m] != -100
                    acc = ((logits[:, :m].argmax(-1) == labels[:, :m]) & g).sum().item() / max(1, g.sum().item())
                sps = seen / (time.time() - t0)
                print(f"ep{epoch} step{step}/{total_steps} loss={loss.item():.4f} acc={acc:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} gnorm={gnorm:.2f} {sps:.1f} smp/s", flush=True)
                if run:
                    run.log({"train/loss": loss.item(), "train/graded_frame_acc": acc,
                             "train/grad_norm": float(gnorm), "learning_rate": sched.get_last_lr()[0],
                             "throughput/samples_per_s": sps, "epoch": epoch}, step=step)

            if step % args.eval_steps == 0:
                log_eval(step)

    log_eval(step)
    if run: run.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
