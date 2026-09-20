"""Scaled training with CACHED tap features + FULL diagnostic plotting.

Phase 1: run the FROZEN HuBERT encoder ONCE over N train + dev clips, cache the tap
         (B,T,1280) features + positional <fill> labels in RAM (~15 GB for 10k).
Phase 2: train ONLY the 2 SA layers + lm_head (sinusoidal PE + framewise CE) for E
         epochs on the cached features — the 1B encoder is never re-run, so 100 epochs
         is fast.  Same architecture/target as filler_sa_reference.py.

Logs every plot from the plan to wandb:
  curves: train/loss, eval/{loss,WER,CER,SER}, lr, grad_norm
  collapse: fill_pred_rate, content/fill/graded frame-acc, entropy, distinct tokens
  distribution: predicted & target token histograms
  position: accuracy vs frame-position (10 bins)
  qualitative: sample Table (FULL frames | stripped hyp | ref)

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
    python train_cached_plots.py --n 10000 --epochs 100 --pe --wandb
"""
import os, re, json, time, argparse

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup

from filler_sa_reference import (
    FILL_TOKEN, build_tokenizer, DataCollatorFillerASR, build_model,
    framewise_ce_loss, sinusoidal_positional_encoding,
)

CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"]'
def norm_ref(t): return re.sub(CHARS_TO_IGNORE, "", t).lower().strip()
def lev(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0]*m; ri = ref[i-1]
        for j in range(1, m+1): cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if ri==hyp[j-1] else 1))
        prev = cur
    return prev[m]

def load_rows(path, limit):
    rows = []
    for l in open(path):
        if not l.strip(): continue
        d = json.loads(l)
        rows.append({"audio_path": d.get("audio") or d.get("source"), "text": d.get("text") or d.get("target")})
        if len(rows) >= limit: break
    return rows


# ---------- Phase 1: cache frozen tap features + labels ----------
@torch.no_grad()
def build_cache(model, rows, collator, device, bs=16, tag=""):
    """Returns list of (hidden fp16 [L,1280] cpu, labels long [L] cpu)."""
    cache = []
    model.eval()                                   # deterministic frozen features (dropout off)
    for s in range(0, len(rows), bs):
        batch = collator(rows[s:s+bs])
        labels = batch.pop("labels")
        inp = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model.hubert(**inp).last_hidden_state             # (B,T,1280)
        fmask = model.hubert._get_feature_vector_attention_mask(hidden.shape[1], inp["attention_mask"])
        lens = fmask.sum(-1).tolist()
        for i, L in enumerate(lens):
            L = int(L); Ll = min(L, labels.shape[1])
            cache.append((hidden[i, :Ll].half().cpu().contiguous(), labels[i, :Ll].cpu().contiguous()))
        if (s // bs) % 25 == 0:
            print(f"  cache[{tag}] {s+len(lens)}/{len(rows)}", flush=True)
    return cache

class CacheDS(Dataset):
    def __init__(self, cache): self.c = cache
    def __len__(self): return len(self.c)
    def __getitem__(self, i): return self.c[i]

def collate_cache(items):
    Tmax = max(h.shape[0] for h, _ in items); B = len(items); D = items[0][0].shape[1]
    hid = torch.zeros(B, Tmax, D, dtype=torch.float16)
    lab = torch.full((B, Tmax), -100, dtype=torch.long)
    kpm = torch.ones(B, Tmax, dtype=torch.bool)          # True = pad
    for i, (h, l) in enumerate(items):
        L = h.shape[0]; hid[i, :L] = h; lab[i, :L] = l; kpm[i, :L] = False
    return hid, lab, kpm


# ---------- head-only forward (no encoder): PE -> SA -> dropout -> lm_head ----------
def head_forward(model, hidden, key_padding_mask):
    if model.use_sinusoidal_pe:
        B, T, d = hidden.shape
        hidden = hidden + model.pe_scale * sinusoidal_positional_encoding(T, d, hidden.device, hidden.dtype).unsqueeze(0)
    h = model.sa(hidden, src_key_padding_mask=key_padding_mask)
    h = model.dropout(h)
    return model.lm_head(h)


# ---------- evaluation: every metric + sample dump ----------
@torch.no_grad()
def evaluate(model, loader, tok, fill_id, device, dump_k=6):
    was_training = model.training; model.eval()
    V = len(tok)
    w_err=w_tot=c_err=c_tot=s_err=s_tot=0
    gc=gt=cc=ct=fc=ft=fp=0; ent_s=ent_n=0.0
    predh=torch.zeros(V); tgth=torch.zeros(V)
    POSB=10; pcorr=torch.zeros(POSB); ptot=torch.zeros(POSB)
    loss_s=loss_n=0.0; samples=[]
    for hid, lab, kpm in loader:
        hid=hid.to(device).float(); lab=lab.to(device); kpm=kpm.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = head_forward(model, hid, kpm).float()
        m=min(lg.shape[1], lab.shape[1]); lg=lg[:,:m]; lb=lab[:,:m]
        loss_s += framewise_ce_loss(lg, lb, fill_id=fill_id).item(); loss_n += 1
        pred=lg.argmax(-1); graded=lb!=-100; corr=(pred==lb)&graded
        gc+=corr.sum().item(); gt+=graded.sum().item()
        isf=graded&(lb==fill_id); isc=graded&(lb!=fill_id)
        fc+=(corr&isf).sum().item(); ft+=isf.sum().item()
        cc+=(corr&isc).sum().item(); ct+=isc.sum().item()
        fp+=((pred==fill_id)&graded).sum().item()
        p=torch.softmax(lg,-1); ent=-(p*p.clamp_min(1e-9).log()).sum(-1)
        ent_s+=ent[graded].sum().item(); ent_n+=graded.sum().item()
        predh+=torch.bincount(pred[graded].cpu(),minlength=V).float()
        tgth+=torch.bincount(lb[graded].cpu(),minlength=V).float()
        B,T=lb.shape; pos=(torch.arange(T,device=device).expand(B,T).float()/max(1,T)*POSB).long().clamp(max=POSB-1)
        for k in range(POSB):
            sel=graded&(pos==k); ptot[k]+=sel.sum().item(); pcorr[k]+=(corr&sel).sum().item()
        hyps=tok.batch_decode(pred, skip_special_tokens=True, group_tokens=False)
        for i in range(lb.shape[0]):
            keep=(lb[i]!=-100).nonzero(as_tuple=True)[0]
            ref=norm_ref(tok.batch_decode(lb[i][keep].unsqueeze(0), skip_special_tokens=True, group_tokens=False)[0])
            hyp=hyps[i].strip()
            w_err+=lev(ref.split(),hyp.split()); w_tot+=len(ref.split())
            c_err+=lev(ref,hyp); c_tot+=len(ref); s_err+=int(hyp!=ref); s_tot+=1
            if len(samples)<dump_k:
                full=" ".join(tok.convert_ids_to_tokens(int(x)) for x in pred[i][keep])
                samples.append((full,hyp,ref))
    if was_training: model.train()
    nf=max(1,gt)
    scal={"eval/loss":loss_s/max(1,loss_n),"eval/WER":w_err/max(1,w_tot),"eval/CER":c_err/max(1,c_tot),
          "eval/SER":s_err/max(1,s_tot),"eval/graded_frame_acc":gc/nf,"eval/content_frame_acc":cc/max(1,ct),
          "eval/fill_frame_acc":fc/max(1,ft),"eval/fill_pred_rate":fp/nf,"eval/pred_entropy":ent_s/max(1,ent_n),
          "eval/distinct_nonfill_tokens":int((predh>0).sum().item()-(1 if predh[fill_id]>0 else 0))}
    return scal,(predh,tgth),(pcorr/ptot.clamp_min(1)),samples


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", default="/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft")
    ap.add_argument("--vocab_dir", default="/speech/tomson/filler_asr/experiments/Hubert_SA_2gpu_lazy_lr2e3")
    ap.add_argument("--train_manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl")
    ap.add_argument("--dev_manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--dev_n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.005)
    ap.add_argument("--pe", action="store_true"); ap.add_argument("--fill_weight", type=float, default=None)
    ap.add_argument("--eval_steps", type=int, default=500); ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--wandb", action="store_true")
    args=ap.parse_args()

    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=build_tokenizer(args.vocab_dir); fill_id=tok.convert_tokens_to_ids(FILL_TOKEN)
    fe=Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    collator=DataCollatorFillerASR(fe, tok)
    train_rows=load_rows(args.train_manifest,args.n); dev_rows=load_rows(args.dev_manifest,args.dev_n)

    model=build_model(args.model_checkpoint, tok, use_sinusoidal_pe=args.pe, device=dev)
    trainable=[p for p in model.parameters() if p.requires_grad]; n_train=sum(p.numel() for p in trainable)

    print(f"[phase1] caching tap features: train={len(train_rows)} dev={len(dev_rows)} ...", flush=True)
    t=time.time()
    train_cache=build_cache(model, train_rows, collator, dev, tag="train")
    dev_cache=build_cache(model, dev_rows, collator, dev, tag="dev")
    del model.hubert; torch.cuda.empty_cache()                       # free the 1B encoder
    model.train()
    print(f"[phase1] cached in {time.time()-t:.0f}s; encoder freed", flush=True)

    train_loader=DataLoader(CacheDS(train_cache), batch_size=args.batch_size, shuffle=True, collate_fn=collate_cache)
    eval_loader=DataLoader(CacheDS(dev_cache), batch_size=args.batch_size, shuffle=False, collate_fn=collate_cache)
    total_steps=args.epochs*len(train_loader)
    opt=torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.98), eps=1e-6)
    sched=get_cosine_schedule_with_warmup(opt, args.warmup, total_steps)

    run=None
    if args.wandb:
        import wandb
        run=wandb.init(project="filler_asr",
            name=f"cached_scale{args.n}_{args.epochs}ep_{'PE' if args.pe else 'noPE'}_{os.path.basename(args.model_checkpoint)}",
            config={**vars(args),"trainable_M":round(n_train/1e6,2),"total_steps":total_steps})
        print("wandb run:", wandb.run.url, flush=True)
    print(f"[phase2] steps/epoch={len(train_loader)} total_steps={total_steps} trainable={n_train/1e6:.2f}M PE={args.pe}", flush=True)

    def log_eval(step):
        scal,(predh,tgth),posacc,samples=evaluate(model, eval_loader, tok, fill_id, dev)
        print(f"[eval @ {step}] WER={scal['eval/WER']:.3f} CER={scal['eval/CER']:.3f} SER={scal['eval/SER']:.3f} "
              f"loss={scal['eval/loss']:.3f} content_acc={scal['eval/content_frame_acc']:.3f} "
              f"fill_rate={scal['eval/fill_pred_rate']:.3f} entropy={scal['eval/pred_entropy']:.3f} "
              f"distinct={scal['eval/distinct_nonfill_tokens']}", flush=True)
        if samples:
            f0,h0,r0=samples[0]
            print(f"   Full: {f0[:140]}\n   Hyp : {h0[:80]}\n   Ref : {r0[:80]}", flush=True)
        if run:
            log=dict(scal)
            for k in range(len(posacc)): log[f"pos_acc/bin{k}"]=float(posacc[k])
            try:
                import wandb
                log["dist/pred_tokens"]=wandb.Histogram(np_histogram=(predh.numpy(), np.arange(len(tok)+1)))
                log["dist/target_tokens"]=wandb.Histogram(np_histogram=(tgth.numpy(), np.arange(len(tok)+1)))
                tbl=wandb.Table(columns=["step","full_frames","hyp_stripped","ref"])
                for fl,hy,rf in samples: tbl.add_data(step,fl,hy,rf)
                log["samples"]=tbl
            except Exception as e:
                print("plot warn:",e, flush=True)
            run.log(log, step=step)

    step=0; t0=time.time(); seen=0
    for epoch in range(1,args.epochs+1):
        for hid,lab,kpm in train_loader:
            hid=hid.to(dev).float(); lab=lab.to(dev); kpm=kpm.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lg=head_forward(model,hid,kpm).float()
            m=min(lg.shape[1],lab.shape[1])
            loss=framewise_ce_loss(lg[:,:m],lab[:,:m],fill_id=fill_id,fill_weight=args.fill_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm=torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); sched.step()
            step+=1; seen+=lab.shape[0]
            if step%args.log_steps==0:
                with torch.no_grad():
                    g=lab[:,:m]!=-100; acc=((lg[:,:m].argmax(-1)==lab[:,:m])&g).sum().item()/max(1,g.sum().item())
                sps=seen/(time.time()-t0)
                print(f"ep{epoch} step{step}/{total_steps} loss={loss.item():.4f} acc={acc:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} gnorm={gnorm:.2f} {sps:.0f} smp/s", flush=True)
                if run: run.log({"train/loss":loss.item(),"train/graded_frame_acc":acc,"train/grad_norm":float(gnorm),
                                 "learning_rate":sched.get_last_lr()[0],"throughput/samples_per_s":sps,"epoch":epoch}, step=step)
            if step%args.eval_steps==0: log_eval(step)
    log_eval(step)
    if run: run.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
