"""Full-LibriSpeech training of the SA + PE + lm_head head on a FROZEN HuBERT-xlarge.

Pipeline (NO caching — forward pass through the 1B encoder every step):

    audio ─► Wav2Vec2FeatureExtractor ─► [FROZEN HuBERT-xlarge] ─► tap (B,T,1280)
          ─► ⊕ sinusoidal PE ─► 2× post-norm Self-Attention ─► dropout ─► lm_head (→33)
    loss  ─► framewise CrossEntropy on the positional <fill> target (duration budget)

Only the 2 SA layers + lm_head train (~39.4M).  The encoder produces a
requires_grad=False tap, so its internal activations are never retained for
backward — memory is ~encoder-forward + a small head graph.

What is logged to wandb (intentionally trimmed):
    train  : train/loss, train/graded_frame_acc, train/grad_norm, train/lr,
                   throughput/samples_per_s, epoch
    eval   : eval/{loss,WER,CER,SER,graded_frame_acc,content_frame_acc,fill_frame_acc,
                   fill_pred_rate,pred_entropy,distinct_nonfill_tokens}
    chart  : all of the above are scalar series → wandb line charts
    table  : qualitative sample Table (full frames | stripped hyp | ref)  [key: samples]
    config : every metric above is gated by --log_config (default log_config.json);
             set a metric's flag to false there to stop logging that series.
    REMOVED: per-position accuracy bins + predicted/target token histograms.

Checkpoints (head-only state dict + optimizer/scheduler) are written to --out_dir:
    latest.pt  (every --save_steps)   best.pt  (on dev-loss improvement)
Resume with --resume <path>.

Launch: see run.sh
"""
import os
import re
import sys
import json
import time
import argparse
import contextlib

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

# verified model/collator/loss from the parent filler_asr dir (see data.py for path wiring)
from data import load_rows, ManifestDataset, DataCollatorFillerASR
from filler_sa_reference import (FILL_TOKEN, SAMPLING_RATE, build_tokenizer, build_model,
                                 framewise_ce_loss,
                                 build_acmlm_tokenizer, build_acmlm_model, DataCollatorACMLM)
# A-CMLM iterative (mask-predict) decoder — used to report a REALISTIC dev WER at validation,
# matching how the model is actually run at inference (see filler_asr_omni_style_decoding.py).
from filler_asr_omni_style_decoding import iterative_decode, build_specials, readout
from data_tapcached import (TapCacheDataset, DataCollatorACMLMTapCached,     # --tap_cache: train from
                            LengthBucketedDistributedBatchSampler)          #   precomputed HuBERT taps


# ---------- logging config (metric on/off toggles, see log_config.json) ----------
def load_log_config(path):
    """Return {metric_name: bool}.  Missing file → empty dict (everything stays ON)."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def gate_log(d, cfg):
    """Keep only metrics whose flag is not explicitly false; unknown keys default ON."""
    return {k: v for k, v in d.items() if cfg.get(k, True)}


# ---------- text normalization + Levenshtein (for WER/CER/SER) ----------
CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"]'
def norm_ref(t): return re.sub(CHARS_TO_IGNORE, "", t).lower().strip()
def lev(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0]*m; ri = ref[i-1]
        for j in range(1, m+1): cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if ri == hyp[j-1] else 1))
        prev = cur
    return prev[m]


# ---------- checkpoint helpers (head only: SA + lm_head, ~150 MB not ~1.2 GB) ----------
def head_state_dict(model):
    # Save EVERY trainable parameter: the SA head, lm_head, text_embed, mask_embed, AND any
    # HuBERT encoder layers unfrozen via --unfreeze_top_hubert. The old name-prefix filter
    # dropped the unfrozen hubert.* params, so gradual-unfreezing silently lost its fine-tuned
    # encoder on every save/resume. Frozen params (requires_grad=False) are excluded and
    # restored from the base checkpoint. For the fully-frozen run this is identical to before.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}

def save_ckpt(path, model, opt, sched, step, epoch, best_val, args):
    torch.save({"head": head_state_dict(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "epoch": epoch, "best_val": best_val, "args": vars(args)}, path)

def load_ckpt(path, model, opt=None, sched=None):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["head"], strict=False)             # only head keys present → missing hubert keys OK
    if opt is not None and "opt" in ck: opt.load_state_dict(ck["opt"])
    if sched is not None and "sched" in ck: sched.load_state_dict(ck["sched"])
    # accept legacy "best_wer" ckpts (older runs selected on WER); worst = +inf (lower is better)
    return ck.get("step", 0), ck.get("epoch", 1), ck.get("best_val", ck.get("best_wer", float("inf")))


# ---------- distributed (DDP) ----------
def setup_distributed():
    """Init NCCL if launched via torchrun (WORLD_SIZE>1); else single-process.
    Returns (ddp, rank, world, local_rank, device)."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, dist.get_rank(), dist.get_world_size(), local_rank, f"cuda:{local_rank}"
    return False, 0, 1, 0, ("cuda" if torch.cuda.is_available() else "cpu")

def barrier(ddp):
    if ddp: dist.barrier()


# ---------- evaluation: scalars + qualitative samples (no plots/histograms) ----------
@torch.no_grad()
def evaluate(model, loader, tok, fill_id, device, dump_k=10, max_sample_sec=10.0,
             decode_steps=0, decode_n=0):
    was_training = model.training; model.eval()
    V = len(tok); eos_id = tok.eos_token_id
    w_err = w_tot = c_err = c_tot = s_err = s_tot = 0
    gc = gt = cc = ct = fc = ft = fp = 0; ent_s = ent_n = 0.0
    predh = torch.zeros(V); loss_s = loss_n = 0.0; samples = []
    # --- optional A-CMLM iterative (mask-predict) decode: realistic dev WER matching inference.
    #     Keeps loss + one-shot metrics untouched (best.pt still selected by eval/loss).
    do_iter = decode_steps > 0 and hasattr(model, "decode_from_tap")
    iw_err = iw_tot = ic_err = ic_tot = is_err = is_tot = 0
    n_iter_done = 0; specials = igen = None
    if do_iter:
        specials = build_specials(tok, device)
        igen = torch.Generator(device=device); igen.manual_seed(0)   # deterministic eval decode
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inp = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model(**inp).logits.float()
        m = min(lg.shape[1], labels.shape[1]); lg = lg[:, :m]; lb = labels[:, :m]
        loss_s += framewise_ce_loss(lg, lb, fill_id=fill_id).item(); loss_n += 1
        pred = lg.argmax(-1); graded = lb != -100; corr = (pred == lb) & graded
        gc += corr.sum().item(); gt += graded.sum().item()
        isf = graded & (lb == fill_id); isc = graded & (lb != fill_id)
        fc += (corr & isf).sum().item(); ft += isf.sum().item()
        cc += (corr & isc).sum().item(); ct += isc.sum().item()
        fp += ((pred == fill_id) & graded).sum().item()
        p = torch.softmax(lg, -1); ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
        ent_s += ent[graded].sum().item(); ent_n += graded.sum().item()
        predh += torch.bincount(pred[graded].cpu(), minlength=V).float()
        # Cut each hyp at the model's FIRST predicted </s>: set that frame and everything
        # after it to <pad> (a skipped special) so batch_decode stops there instead of
        # running through post-eos frames -> trailing garbage. skip_special_tokens DELETES
        # </s> rather than stopping on it, so without this the tail leaks in (see eval n_keep cap).
        after_eos = (pred == eos_id).cumsum(dim=1) > 0
        pred_dec = pred.masked_fill(after_eos, tok.pad_token_id)
        hyps = tok.batch_decode(pred_dec, skip_special_tokens=True, group_tokens=False)
        for i in range(lb.shape[0]):
            keep = (lb[i] != -100).nonzero(as_tuple=True)[0]
            ref = norm_ref(tok.batch_decode(lb[i][keep].unsqueeze(0), skip_special_tokens=True, group_tokens=False)[0])
            hyp = hyps[i].strip()
            w_err += lev(ref.split(), hyp.split()); w_tot += len(ref.split())
            c_err += lev(ref, hyp); c_tot += len(ref); s_err += int(hyp != ref); s_tot += 1
            # --- iterative (N-step mask-predict) decode of THIS clip -> realistic-inference WER ---
            if do_iter and (decode_n == 0 or n_iter_done < decode_n):
                graded_i = (lb[i] != -100)
                nk = int(graded_i.nonzero().max().item()) + 1 if graded_i.any() else 0
                if nk >= 1:
                    if "tap" in inp:
                        tap_pe, skpm = model.encode_from_tap(inp["tap"][i:i+1], inp["pad_mask"][i:i+1])
                    else:
                        tap_pe, skpm = model.encode_audio(inp["input_values"][i:i+1], inp["attention_mask"][i:i+1])
                    Ti = tap_pe.shape[1]
                    def _lf(text_ids, _tp=tap_pe, _sk=skpm):
                        return model.decode_from_tap(_tp, text_ids.unsqueeze(0), _sk).logits[0]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ids = iterative_decode(_lf, Ti, specials, content_len=min(nk, Ti),
                                               N=decode_steps, generator=igen)
                    ihyp = readout(ids, tok, eos_id).strip()
                    iw_err += lev(ref.split(), ihyp.split()); iw_tot += len(ref.split())
                    ic_err += lev(ref, ihyp); ic_tot += len(ref)
                    is_err += int(ihyp != ref); is_tot += 1
                    n_iter_done += 1
            # qualitative samples: first dump_k clips shorter than max_sample_sec
            dur_i = (inp["attention_mask"][i].sum().item() / SAMPLING_RATE if "attention_mask" in inp
                     else (~inp["pad_mask"][i]).sum().item() / 50.0)   # cached: valid frames / 50 fps
            if len(samples) < dump_k and dur_i < max_sample_sec:
                full = " ".join(tok.convert_ids_to_tokens(int(x)) for x in pred[i][keep])
                samples.append((full, hyp, ref))
    if was_training: model.train()
    nf = max(1, gt)
    scal = {"eval/loss": loss_s/max(1, loss_n), "eval/WER": w_err/max(1, w_tot), "eval/CER": c_err/max(1, c_tot),
            "eval/SER": s_err/max(1, s_tot), "eval/graded_frame_acc": gc/nf, "eval/content_frame_acc": cc/max(1, ct),
            "eval/fill_frame_acc": fc/max(1, ft), "eval/fill_pred_rate": fp/nf, "eval/pred_entropy": ent_s/max(1, ent_n),
            "eval/distinct_nonfill_tokens": int((predh > 0).sum().item() - (1 if predh[fill_id] > 0 else 0))}
    if do_iter and is_tot > 0:                                        # realistic iterative-decode WER
        scal["eval/iter_WER"] = iw_err/max(1, iw_tot); scal["eval/iter_CER"] = ic_err/max(1, ic_tot)
        scal["eval/iter_SER"] = is_err/max(1, is_tot); scal["eval/iter_n"] = is_tot
        scal["eval/iter_steps"] = decode_steps
    return scal, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", default="/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft")
    ap.add_argument("--vocab_dir", default="/speech/tomson/filler_asr/experiments/Hubert_SA_2gpu_lazy_lr2e3")
    ap.add_argument("--train_manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl")
    ap.add_argument("--dev_manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_dev_clean.jsonl")
    ap.add_argument("--out_dir", default="/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa_pe_lmhead")
    ap.add_argument("--train_n", type=int, default=0, help="0 = full manifest")
    ap.add_argument("--dev_n", type=int, default=500, help="0 = full dev set (slower eval)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2, help="effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr_schedule", choices=["linear", "cosine", "exponential"], default="linear",
                    help="LR decay after warmup")
    ap.add_argument("--lr_exp_final_ratio", type=float, default=0.01,
                    help="exponential schedule only: final LR as a fraction of peak at the last step")
    ap.add_argument("--num_sa_layers", type=int, default=4, help="# self-attention layers in the head")
    ap.add_argument("--frames_per_sec", type=int, default=25,
                    help="duration budget: graded <fill> frames per second")
    ap.add_argument("--eos_repeat", type=int, default=1,
                    help="emit </s> on this many consecutive frames at end of transcript "
                         "(1 = original target; e.g. 3 reinforces the content/<fill> boundary)")
    ap.add_argument("--mask_prob", type=float, default=0.0,
                    help="train-time input masking of the HuBERT tap before SA (0 = off)")
    ap.add_argument("--mask_prob_max", type=float, default=0.0,
                    help="if > mask_prob, per-batch rate sampled U[mask_prob, mask_prob_max]")
    ap.add_argument("--mask_length", type=int, default=10,
                    help="masked span length in frames (10 frames = 200ms at 50fps)")
    ap.add_argument("--unfreeze_top_hubert", type=int, default=0,
                    help="also train the top N HuBERT encoder layers (0 = keep whole body frozen)")
    ap.add_argument("--hubert_lr", type=float, default=0.0,
                    help="LR for the unfrozen HuBERT layers (0 = use --lr; recommend ~1e-5)")
    ap.add_argument("--weight_decay", type=float, default=0.005)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8, help="audio decode is I/O-bound; keep the GPU fed")
    ap.add_argument("--pe", action="store_true", default=True); ap.add_argument("--no_pe", dest="pe", action="store_false")
    ap.add_argument("--pos_mode", choices=["sinusoidal", "alibi", "rope"], default="sinusoidal",
                    help="positional scheme for the SA head: 'sinusoidal' (default, = original absolute PE), "
                         "'alibi' (symmetric ALiBi additive bias; no absolute PE), or 'rope' (rotary PE: "
                         "rotates q,k inside attention -> relative positions, with dynamic NTK-aware length "
                         "extension). 'alibi'/'rope' ignore --pe/--no_pe (they replace the absolute PE).")
    # --- RoPE / NTK knobs (pos_mode=="rope" only) -------------------------------------
    ap.add_argument("--rope_theta", type=float, default=10000.0,
                    help="RoPE base θ. Larger = longer native wavelengths / more long-context headroom "
                         "(Llama-3 uses 500000). Default 10000.")
    ap.add_argument("--rope_orig_len", type=int, default=1024,
                    help="reference (trained) length for dynamic NTK: sequences with T<=this are plain "
                         "RoPE; longer ones get NTK-scaled. 1024 frames ~= 20s at 50fps.")
    ap.add_argument("--rope_ntk_factor", type=float, default=1.0,
                    help="STATIC NTK stretch s (used only when --no_rope_ntk_dynamic). base'=base*s^(d/(d-2)); "
                         "s<=1 disables. Ignored in the default dynamic mode.")
    ap.add_argument("--rope_ntk_dynamic", action="store_true", default=True,
                    help="dynamic NTK: derive the stretch from the actual length, s=max(1,T/rope_orig_len) "
                         "(default; no penalty in-distribution).")
    ap.add_argument("--no_rope_ntk_dynamic", dest="rope_ntk_dynamic", action="store_false",
                    help="disable dynamic NTK; use the fixed --rope_ntk_factor instead.")
    ap.add_argument("--fill_weight", type=float, default=None)
    # --- validation: optional A-CMLM iterative (mask-predict) decode for a realistic dev WER ---
    ap.add_argument("--eval_decode_steps", type=int, default=0,
                    help="0 = one-shot (p=1.0) reconstruction eval only (original behavior). >0 = ALSO run "
                         "N-step iterative mask-predict decode at eval and report eval/iter_{WER,CER,SER}. "
                         "ACMLM only. Best-checkpoint selection stays on eval/loss regardless.")
    ap.add_argument("--eval_decode_n", type=int, default=200,
                    help="cap on #dev clips iteratively decoded per eval (0 = all). Keeps eval fast.")
    ap.add_argument("--select_metric", choices=["loss", "iter_wer"], default="loss",
                    help="metric that selects best.pt (lower=better). 'loss' = one-shot MLM dev loss "
                         "(default, proper scoring rule). 'iter_wer' = realistic N-step iterative-decode "
                         "dev WER (needs --eval_decode_steps>0; use when MLM loss saturates before decode "
                         "WER does, e.g. ALiBi). Falls back to loss if iter WER unavailable.")
    ap.add_argument("--eval_steps", type=int, default=2000); ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--debug_steps", type=int, default=5,
                    help="print a per-step grad/token averaging audit for the first N optimizer steps (0=off)")
    ap.add_argument("--dump_k", type=int, default=10, help="# qualitative eval samples to log")
    ap.add_argument("--sample_max_sec", type=float, default=10.0,
                    help="only log qualitative samples for clips shorter than this (seconds)")
    ap.add_argument("--log_config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_config.json"),
                    help="JSON {metric_name: true/false} toggling which series get logged to wandb")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = run --epochs; else cap optimizer steps")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init_from", default="",
                    help="warm-start SA+lm_head weights from this ckpt but start a FRESH "
                         "optimizer/scheduler (for gradual unfreezing); != --resume")
    ap.add_argument("--eval_at_start", action="store_true", default=True)
    ap.add_argument("--no_eval_at_start", dest="eval_at_start", action="store_false")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--acmlm", action="store_true",
                    help="audio-conditioned masked-LM (A-CMLM): mask the text label + iterative decode "
                         "(adds text embedding, addition fusion; keeps 20-30%% audio-tap masking)")
    ap.add_argument("--tap_cache", default="",
                    help="dir with precomputed HuBERT taps (precompute_taps.py), holding train/ and dev/ "
                         "subdirs. Trains the head straight from the cached tap -> skips the frozen encoder "
                         "forward every step (~3x faster). Uses length-bucketed batches (no padding waste).")
    args = ap.parse_args()
    log_cfg = load_log_config(args.log_config)

    ddp, rank, world, local_rank, dev = setup_distributed()
    is_main = (rank == 0)
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    barrier(ddp)

    tok = (build_acmlm_tokenizer(args.vocab_dir) if args.acmlm else build_tokenizer(args.vocab_dir))
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN)
    if is_main:
        tok.save_pretrained(args.out_dir)                              # for later inference
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    if args.acmlm and args.tap_cache:
        # CACHED-tap A-CMLM: collator pads precomputed taps + builds the CMLM text/labels (no audio)
        collator = DataCollatorACMLMTapCached(tok, frames_per_sec=args.frames_per_sec,
                                              eos_repeat=args.eos_repeat)
        eval_collator = DataCollatorACMLMTapCached(tok, frames_per_sec=args.frames_per_sec,
                                                   eos_repeat=args.eos_repeat, fixed_p=1.0)
    elif args.acmlm:
        collator = DataCollatorACMLM(fe, tok, frames_per_sec=args.frames_per_sec,
                                     eos_repeat=args.eos_repeat)
        # eval uses a DETERMINISTIC full-content mask (p=1.0) -> one-shot reconstruction proxy
        eval_collator = DataCollatorACMLM(fe, tok, frames_per_sec=args.frames_per_sec,
                                          eos_repeat=args.eos_repeat, fixed_p=1.0)
    else:
        collator = DataCollatorFillerASR(fe, tok, frames_per_sec=args.frames_per_sec,
                                         eos_repeat=args.eos_repeat)
        eval_collator = collator

    if args.tap_cache:
        train_rows = dev_rows = None                                   # datasets come from the tap cache
    else:
        train_rows = load_rows(args.train_manifest, args.train_n or None)
        dev_rows = load_rows(args.dev_manifest, args.dev_n or None)

    if args.acmlm:
        model = build_acmlm_model(args.model_checkpoint, tok, num_sa_layers=args.num_sa_layers,
                                  use_sinusoidal_pe=args.pe, device=dev,
                                  mask_prob=args.mask_prob, mask_prob_max=args.mask_prob_max,
                                  mask_length=args.mask_length, pos_mode=args.pos_mode,
                                  rope_theta=args.rope_theta, rope_orig_len=args.rope_orig_len,
                                  rope_ntk_factor=args.rope_ntk_factor,
                                  rope_ntk_dynamic=args.rope_ntk_dynamic)
    else:
        model = build_model(args.model_checkpoint, tok, num_sa_layers=args.num_sa_layers,
                            use_sinusoidal_pe=args.pe, device=dev,
                            unfreeze_top_hubert=args.unfreeze_top_hubert,
                            mask_prob=args.mask_prob, mask_prob_max=args.mask_prob_max,
                            mask_length=args.mask_length, pos_mode=args.pos_mode,
                            rope_theta=args.rope_theta, rope_orig_len=args.rope_orig_len,
                            rope_ntk_factor=args.rope_ntk_factor,
                            rope_ntk_dynamic=args.rope_ntk_dynamic)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    raw_model = model                                                  # unwrapped: used for eval + state_dict
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    trainable = [p for p in model.parameters() if p.requires_grad]     # SA + lm_head (+ top-N HuBERT if unfrozen)

    # Warm-start WEIGHTS ONLY (no optimizer/scheduler/step) from a prior head ckpt.
    # For gradual unfreezing: continue from best.pt's SA+lm_head but start a FRESH
    # optimizer + warmup-decay schedule (the old opt state is head-only and would not
    # match a head+hubert param set). No-op when --init_from is unset.
    if args.init_from and os.path.exists(args.init_from):
        _ck = torch.load(args.init_from, map_location="cpu")
        _sd = _ck.get("head", _ck)
        raw_model.load_state_dict(_sd, strict=False)
        if is_main:
            print(f"[init_from] warm-started {sum(v.numel() for v in _sd.values())/1e6:.1f}M head "
                  f"params from {args.init_from} (fresh optimizer/schedule)", flush=True)

    if args.tap_cache:
        from torch.utils.data import Subset
        train_ds = TapCacheDataset(os.path.join(args.tap_cache, "train"))
        dev_ds = TapCacheDataset(os.path.join(args.tap_cache, "dev"))
        if args.dev_n:
            dev_ds = Subset(dev_ds, range(min(args.dev_n, len(dev_ds))))
        # length-bucketed, DDP-sharded BATCH sampler (equal per-rank batch counts -> DDP stays in
        # lockstep for the token-exact all-reduce; length homogeneity removes the ~28% pad waste)
        _lens = train_ds.lengths()[:args.train_n] if args.train_n else train_ds.lengths()
        train_sampler = LengthBucketedDistributedBatchSampler(
            _lens, args.batch_size, num_replicas=world, rank=rank, shuffle=True, seed=0)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collator,
                                  num_workers=args.num_workers, pin_memory=True)
        eval_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=eval_collator, num_workers=max(2, args.num_workers // 2))
    else:
        train_ds = ManifestDataset(train_rows)
        dev_ds = ManifestDataset(dev_rows)
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp else None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=(train_sampler is None), sampler=train_sampler,
                                  collate_fn=collator, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        eval_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=eval_collator, num_workers=max(2, args.num_workers // 2))

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    # Discriminative LR: give the (optionally) unfrozen HuBERT layers a smaller LR than
    # the SA head, so gradual unfreezing adapts the backbone without erasing its
    # pretrained features. With defaults (no unfreeze / hubert_lr=0) this collapses to a
    # single group at args.lr -- byte-identical to the original optimizer.
    _hub  = [p for n, p in raw_model.named_parameters() if p.requires_grad and n.startswith("hubert.")]
    _head = [p for n, p in raw_model.named_parameters() if p.requires_grad and not n.startswith("hubert.")]
    if _hub and args.hubert_lr > 0:
        param_groups = [{"params": _head, "lr": args.lr}, {"params": _hub, "lr": args.hubert_lr}]
        if is_main:
            print(f"[optim] discriminative LR: head={args.lr:.1e} ({len(_head)} tensors) "
                  f"hubert={args.hubert_lr:.1e} ({len(_hub)} tensors)", flush=True)
    else:
        param_groups = trainable
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)
    if args.lr_schedule == "exponential":
        from torch.optim.lr_scheduler import LambdaLR
        _warm = min(args.warmup, total_steps)
        _decay = max(1, total_steps - _warm)
        _final = max(1e-8, args.lr_exp_final_ratio)
        _gamma = _final ** (1.0 / _decay)              # geometric: lr(end) = _final * peak
        def _exp_lr(step):
            if _warm > 0 and step < _warm:
                return step / max(1, _warm)            # linear warmup 0 -> 1
            return max(_final, _gamma ** (step - _warm))
        sched = LambdaLR(opt, _exp_lr)
    else:
        make_sched = (get_linear_schedule_with_warmup if args.lr_schedule == "linear"
                      else get_cosine_schedule_with_warmup)
        sched = make_sched(opt, args.warmup, total_steps)

    eff_batch = args.batch_size * args.grad_accum * world
    step = 0; start_epoch = 1; best_val = float("inf"); resume_skip = 0
    if args.resume and os.path.exists(args.resume):
        step, _, best_val = load_ckpt(args.resume, raw_model, opt, sched)   # same file on all ranks → consistent
        # Step-accurate resume: find the epoch that CONTAINS `step` and SKIP the micro-batches
        # already consumed within it, instead of replaying the whole epoch from the top (which
        # double-counted steps and re-trained already-seen data). set_epoch seeds the sampler,
        # so skipping reproduces the exact remaining order.
        if steps_per_epoch > 0:
            start_epoch = step // steps_per_epoch + 1
            resume_skip = (step % steps_per_epoch) * args.grad_accum
        if is_main:
            print(f"[resume] from {args.resume}: step={step} -> epoch={start_epoch} "
                  f"(skip {resume_skip} micro-batches) best_val={best_val:.4f}", flush=True)

    run = None
    if args.wandb and is_main:
        import wandb
        mask_tag = (f"_mask{args.mask_prob:g}-{args.mask_prob_max:g}x{args.mask_length}"
                    if args.mask_prob > 0 else "")
        exp_tag = (f"_eos{args.eos_repeat}" if args.eos_repeat != 1 else "") + \
                  (f"_fw{args.fill_weight:g}" if args.fill_weight is not None else "") + \
                  (f"_{args.pos_mode}" if args.pos_mode != "sinusoidal" else "")
        run = wandb.init(project="filler_asr",
                         name=f"full960_sa{args.num_sa_layers}_pe_lmhead_{'PE' if args.pe else 'noPE'}{mask_tag}{exp_tag}_{os.path.basename(args.model_checkpoint)}",
                         config={**vars(args), "trainable_M": round(n_train/1e6, 2),
                                 "train_utts": len(train_ds), "dev_utts": len(dev_ds), "world_size": world,
                                 "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                                 "effective_batch": eff_batch})
        print("wandb run:", wandb.run.url, flush=True)
    if is_main:
        print(f"[setup] world={world} train={len(train_ds)} dev={len(dev_ds)} trainable={n_train/1e6:.2f}M "
              + (f"tap_cache=ON " if args.tap_cache else "")
              + f"pos_mode={args.pos_mode} PE={args.pe} sa_layers={args.num_sa_layers} fps={args.frames_per_sec} sched={args.lr_schedule} "
              + f"mask={args.mask_prob:g}-{args.mask_prob_max:g}x{args.mask_length} "
              + f"eff_batch={eff_batch} steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)
        off = [k for k, v in log_cfg.items() if not v]
        print(f"[log] config={args.log_config} disabled={off or 'none'}", flush=True)
        # --- change #5 audit: exactly what goes into best.pt/latest.pt (trainable tensors only) ---
        _hsd = head_state_dict(raw_model)
        _hub_in = sum(1 for k in _hsd if k.startswith("hubert."))
        print(f"[ckpt-audit] checkpoint will save {len(_hsd)} trainable tensors / "
              f"{sum(v.numel() for v in _hsd.values())/1e6:.2f}M params  (hubert tensors included={_hub_in})", flush=True)
        # --- eval-strategy audit: which metric selects best.pt ---
        _selm = "eval/iter_WER" if args.select_metric == "iter_wer" else "eval/loss"
        print(f"[select] best.pt = argmin {_selm}; other metrics are diagnostics only", flush=True)
        # --- transformation audit: how one training clip becomes the positional target (+ CMLM mask) ---
        try:
            _sample = train_ds[0] if args.tap_cache else train_rows[0]  # cached path: dataset item, not a manifest row
            _b0 = collator([_sample]); _lab = _b0["labels"][0]
            _Tv = _lab.numel(); _ngr = int((_lab != -100).sum())
            if args.acmlm:
                _tin = _b0["text_input_ids"][0]
                _view = " ".join(f"{i}:{tok.convert_ids_to_tokens(int(_tin[i]))}"
                                 + ("" if _lab[i] == -100 else "->" + tok.convert_ids_to_tokens(int(_lab[i])))
                                 for i in range(min(24, _Tv)))
                print(f"[data-audit] ACMLM transform, 1 train clip: T={_Tv} graded(masked)_slots={_ngr}\n"
                      f"   frame:input[->CEtarget], first {min(24,_Tv)}/{_Tv}  (<mask>=input hole; -> = supervised slot)\n"
                      f"   {_view}", flush=True)
            else:
                _view = " ".join(f"{i}:{tok.convert_ids_to_tokens(int(_lab[i]))}"
                                 for i in range(min(24, _Tv)) if _lab[i] != -100)
                print(f"[data-audit] positional target, 1 train clip: T={_Tv} graded_frames={_ngr}\n   {_view}", flush=True)
        except Exception as _e:
            print(f"[data-audit] skipped ({_e})", flush=True)

    def log_eval(step):
        nonlocal best_val
        scal, samples = evaluate(raw_model, eval_loader, tok, fill_id, dev,
                                 dump_k=args.dump_k, max_sample_sec=args.sample_max_sec,
                                 decode_steps=args.eval_decode_steps, decode_n=args.eval_decode_n)
        print(f"[eval @ {step}] WER={scal['eval/WER']:.3f} CER={scal['eval/CER']:.3f} SER={scal['eval/SER']:.3f} "
              f"loss={scal['eval/loss']:.3f} content_acc={scal['eval/content_frame_acc']:.3f} "
              f"fill_rate={scal['eval/fill_pred_rate']:.3f} entropy={scal['eval/pred_entropy']:.3f} "
              f"distinct={scal['eval/distinct_nonfill_tokens']}", flush=True)
        if "eval/iter_WER" in scal:
            print(f"[eval @ {step}] iter{scal['eval/iter_steps']}  WER={scal['eval/iter_WER']:.3f} "
                  f"CER={scal['eval/iter_CER']:.3f} SER={scal['eval/iter_SER']:.3f} "
                  f"(mask-predict decode, n={scal['eval/iter_n']} clips)", flush=True)
        if samples:
            f0, h0, r0 = samples[0]
            print(f"   Full: {f0[:140]}\n   Hyp : {h0[:80]}\n   Ref : {r0[:80]}", flush=True)
        if run:
            log = gate_log(scal, log_cfg)
            if log_cfg.get("samples", True):
                try:
                    import wandb
                    tbl = wandb.Table(columns=["step", "full_frames", "hyp_stripped", "ref"])
                    for fl, hy, rf in samples: tbl.add_data(step, fl, hy, rf)
                    log["samples"] = tbl
                except Exception as e:
                    print("table warn:", e, flush=True)
            if log: run.log(log, step=step)
        # checkpoint best by the configured selection metric (lower = better).
        #   loss     : held-out one-shot MLM loss (strictly proper scoring rule, decode-bug-immune) [default]
        #   iter_wer : realistic N-step iterative-decode dev WER (matches deployment; use when one-shot
        #              loss saturates before decode WER does -- e.g. ALiBi, where best.pt-by-loss lands ~ep69
        #              while iterative-decode WER keeps falling to ep150).
        sel_key = "eval/iter_WER" if args.select_metric == "iter_wer" else "eval/loss"
        if sel_key not in scal: sel_key = "eval/loss"           # iter_wer requested but decode-eval off -> fall back
        if scal[sel_key] < best_val:
            best_val = scal[sel_key]
            save_ckpt(os.path.join(args.out_dir, "best.pt"), raw_model, opt, sched, step, cur_epoch, best_val, args)
            print(f"   ** new best {sel_key}={best_val:.4f} -> best.pt", flush=True)

    cur_epoch = start_epoch
    if args.eval_at_start and not args.resume:
        barrier(ddp)
        if is_main: log_eval(0)
        barrier(ddp)

    model.train()
    if is_main:                                                        # change #1 audit: encoder must be eval + frozen
        _hg = sum(p.requires_grad for p in raw_model.hubert.parameters())
        print(f"[frozen-audit] after model.train(): hubert.training={raw_model.hubert.training} "
              f"hubert_trainable_tensors={_hg}  (expect False / 0 -> no layerdrop, no dropout, deterministic tap)", flush=True)
    t0 = time.time(); seen = 0; done = False
    for epoch in range(start_epoch, args.epochs + 1):
        cur_epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)                            # reshuffle shards each epoch
        opt.zero_grad(set_to_none=True); micro = 0
        tok_acc = 0.0; loss_acc = 0.0                                 # token(weight) + loss SUMS over the current grad-accum window
        for batch in train_loader:
            micro += 1
            if epoch == start_epoch and micro <= resume_skip:         # step-accurate resume: skip already-trained micro-batches
                continue
            labels = batch.pop("labels").to(dev)
            inp = {k: v.to(dev) for k, v in batch.items()}
            is_boundary = (micro % args.grad_accum == 0)
            # no_sync() during accumulation so DDP all-reduces grads ONLY on the boundary micro-step
            sync_ctx = model.no_sync() if (ddp and not is_boundary) else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    lg = model(**inp).logits.float()
                m = min(lg.shape[1], labels.shape[1]); lb = labels[:, :m]
                # SUM reduction (not mean): accumulate raw per-frame loss sums and normalize the
                # gradient by the GLOBAL token count at the boundary -> an EXACT token-weighted mean
                # over the whole grad-accum x DDP window, instead of a biased mean-of-means.
                loss_sum = framewise_ce_loss(lg[:, :m], lb, fill_id=fill_id,
                                             fill_weight=args.fill_weight, reduction="sum")
                loss_sum.backward()
                graded = lb != -100
                if args.fill_weight is not None:                      # denom = Σ class weights over graded frames
                    w_den = ((graded & (lb != fill_id)).sum()
                             + args.fill_weight * (graded & (lb == fill_id)).sum())
                else:
                    w_den = graded.sum()
                tok_acc += float(w_den); loss_acc += float(loss_sum.detach())
            seen += labels.shape[0]
            if not is_boundary:
                continue

            # Normalize the accumulated SUM-gradient to the exact token-weighted mean over the whole
            # window across ranks. DDP averaged grads by 1/world at the boundary backward, so
            # multiplying by world/global_tokens recovers Σ_ranks Σ_tokens grad / Σ_ranks tokens.
            if ddp:
                _t = torch.tensor([tok_acc], device=dev); dist.all_reduce(_t)
                global_tok = max(1.0, _t.item())
            else:
                global_tok = max(1.0, tok_acc)
            for p in trainable:
                if p.grad is not None:
                    p.grad.mul_(world / global_tok)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            win_loss = loss_acc / max(1.0, tok_acc)                   # rank-local window mean (logging only)
            _tok_local, _loss_local, _scale = tok_acc, loss_acc, world / global_tok
            tok_acc = 0.0; loss_acc = 0.0
            step += 1
            if is_main and step <= args.debug_steps:                  # change #3 audit: token-exact averaging
                print(f"[grad-audit step{step}] micro/window={args.grad_accum} tok_local={_tok_local:.0f} "
                      f"tok_global={global_tok:.0f} grad_scale=world/tok={_scale:.3e} "
                      f"loss_sum_local={_loss_local:.1f} win_loss={win_loss:.4f} "
                      f"gnorm_preclip={gnorm:.3f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if step % args.log_steps == 0 and is_main:
                with torch.no_grad():
                    g = lb != -100
                    acc = ((lg[:, :m].argmax(-1) == lb) & g).sum().item() / max(1, g.sum().item())
                sps = seen * world / (time.time() - t0)              # aggregate throughput
                print(f"ep{epoch} step{step}/{total_steps} loss={win_loss:.4f} acc={acc:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} gnorm={gnorm:.2f} {sps:.0f} smp/s", flush=True)
                if run:
                    tl = gate_log({"train/loss": win_loss, "train/graded_frame_acc": acc, "train/grad_norm": float(gnorm),
                                   "train/lr": sched.get_last_lr()[0], "throughput/samples_per_s": sps, "epoch": epoch}, log_cfg)
                    if tl: run.log(tl, step=step)
            if step % args.eval_steps == 0:
                barrier(ddp)
                if is_main: log_eval(step)
                barrier(ddp)
            if step % args.save_steps == 0:
                if is_main: save_ckpt(os.path.join(args.out_dir, "latest.pt"), raw_model, opt, sched, step, epoch, best_val, args)
                barrier(ddp)
            if args.max_steps and step >= args.max_steps:
                done = True; break
        if done:
            break

    barrier(ddp)
    if is_main:
        log_eval(step)
        save_ckpt(os.path.join(args.out_dir, "latest.pt"), raw_model, opt, sched, step, cur_epoch, best_val, args)
        if run: run.finish()
        print("DONE", flush=True)
    barrier(ddp)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()