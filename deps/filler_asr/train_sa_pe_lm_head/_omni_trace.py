"""Instrumented omni-style (mask-predict) decode trace for the English ALiBi A-CMLM model.

Two modes:
  --search K   : decode the first K test-clean clips with 10<=dur<=15s, print WER (to pick a clean one)
  --pick IDX   : full 32-step TRACE of that clip (schedule math + per-step masking + exact calcs)

Reuses the EXACT decode math from filler_asr_omni_style_decoding (schedule_r, commit_counts,
gumbel_top_k, make_logits_fn, readout); the loop is re-implemented here only to expose conf/pred/
chosen for printing (identical arithmetic to iterative_decode).
"""
import os, sys, json, math, argparse
import torch, torch.nn.functional as F

HERE = "/speech/tomson/filler_asr/train_sa_pe_lm_head"
sys.path.insert(0, "/speech/tomson/filler_asr"); sys.path.insert(0, HERE)
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from data import load_rows
from filler_sa_reference import build_acmlm_tokenizer, build_acmlm_model, compute_output_length
from filler_asr_omni_style_decoding import (schedule_r, commit_counts, gumbel_top_k,
                                            make_logits_fn, readout, build_specials)

RUN = "/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_alibi_fps50_mask20-30x10_eos3_fw0.03_exp_150ep"
MAN = "/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl"


def lev(a, b):
    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[m]


def load_model(dev, latest=False):
    ck = torch.load(os.path.join(RUN, "latest.pt" if latest else "best.pt"), map_location="cpu")
    ca = ck.get("args", {}) or {}
    enc = ca.get("model_checkpoint") or "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"
    tok = build_acmlm_tokenizer(RUN)
    model = build_acmlm_model(enc, tok, num_sa_layers=int(ca.get("num_sa_layers", 8)),
                              use_sinusoidal_pe=bool(ca.get("pe", True)), device=dev,
                              mask_prob=float(ca.get("mask_prob", 0.20)),
                              mask_prob_max=float(ca.get("mask_prob_max", 0.30)),
                              pos_mode=ca.get("pos_mode", "alibi"))
    model.load_state_dict(ck["head"], strict=False); model.eval()
    fps = int(ca.get("frames_per_sec", 50))
    print(f"[ckpt] {'latest' if latest else 'best'}.pt step={ck.get('step')} best_wer={ck.get('best_wer')} "
          f"train_fps={fps} pos_mode={ca.get('pos_mode')}", flush=True)
    return model, tok, fps


def candidates(lo=10.0, hi=15.0, limit=40):
    out = []
    for i, r in enumerate(load_rows(MAN)):
        d = sf.info(r["audio_path"]).frames / 16000.0
        if lo <= d <= hi:
            out.append((i, d, r))
        if len(out) >= limit:
            break
    return out


def decode_plain(model, tok, specials, fps, r, dev, N=32):
    from filler_asr_omni_style_decoding import iterative_decode
    arr, sr = sf.read(r["audio_path"], dtype="float32"); arr = arr.mean(1) if arr.ndim > 1 else arr
    fe = FE; iv = fe(arr, sampling_rate=16000, return_tensors="pt")
    im = iv["input_values"].to(dev); am = iv.get("attention_mask"); am = am.to(dev) if am is not None else None
    T = compute_output_length(im.shape[-1]); n_keep = min(T, math.ceil(len(arr) / 16000 * fps))
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        lf = make_logits_fn(model, im, am)
        ids = iterative_decode(lf, T, specials, content_len=n_keep, N=N, generator=gen)
    return readout(ids, tok, specials["eos"])


FE = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                              do_normalize=True, return_attention_mask=True)

CHARS_IGN = __import__("re")
def norm(t): return CHARS_IGN.sub(r'[\,\?\.\!\-\;\:\"]', "", t).lower().strip()


def trace_decode(model, tok, specials, fps, r, dev, N=32, tau=0.1, temp=5.0, seed=0):
    """Re-implements iterative_decode with per-step printing of the EXACT calculations."""
    arr, sr = sf.read(r["audio_path"], dtype="float32"); arr = arr.mean(1) if arr.ndim > 1 else arr
    dur = len(arr) / 16000.0
    iv = FE(arr, sampling_rate=16000, return_tensors="pt")
    im = iv["input_values"].to(dev); am = iv.get("attention_mask"); am = am.to(dev) if am is not None else None
    T = compute_output_length(im.shape[-1]); n_keep = min(T, math.ceil(dur * fps))
    mask_id, fill_id, eos_id = specials["mask"], specials["fill"], specials["eos"]
    ref = norm(r["text"])
    print("=" * 100)
    print(f"CLIP  {os.path.basename(r['audio_path'])}   dur={dur:.2f}s")
    print(f"REF   {ref}")
    print(f"GEOMETRY  T(frames)={T}  n_keep(content, fps={fps})={n_keep}  ->  M=n_keep-1={n_keep-1} slots to unmask")
    print("=" * 100)

    # ---- schedule (exact) ----
    M = n_keep - 1
    counts = commit_counts(N, tau, M)
    print(f"\nSCHEDULE  r_n = tau*x/(1+(tau-1)*x),  x=n/N,  tau={tau}  (back-loaded: few early, avalanche late)")
    print(f"  {'n':>3} {'x=n/N':>7} {'r_n':>7} {'cum=round(r_n*M)':>17} {'k_n(new)':>9}")
    cum = 0
    for n in range(1, N + 1):
        rn = schedule_r(n, N, tau); c = round(rn * M)
        if n <= 6 or n >= N - 3 or n % 6 == 0:
            print(f"  {n:>3} {n/N:>7.3f} {rn:>7.3f} {c:>17} {counts[n-1]:>9}")
        cum = c
    print(f"  (sum of k_n = {sum(counts)} = M = {M})")

    # ---- decode loop (instrumented; identical math to iterative_decode) ----
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    ids = torch.full((T,), fill_id, dtype=torch.long, device=dev)
    ids[:n_keep] = mask_id
    masked = torch.zeros(T, dtype=torch.bool, device=dev); masked[:n_keep] = True
    ids[0] = specials["bos"]; masked[0] = False

    def partial(ids, masked):
        # readout-so-far: content region, '·' for masked, cut at first committed </s>
        toks = []
        for i in range(1, n_keep):
            if masked[i]:
                toks.append("·")
            else:
                t = ids[i].item()
                if t == eos_id:
                    break
                ch = tok.convert_ids_to_tokens(t)
                toks.append(" " if ch == "|" else ("▪" if ch == "<fill>" else ch))
        return "".join(toks)

    print(f"\nDECODE  N={N} steps  temp={temp}  ('·'=masked, '▪'=<fill>, space='|')\n")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        lf = make_logits_fn(model, im, am)
        for n, k in enumerate(counts, start=1):
            if k <= 0:
                continue
            logp = F.log_softmax(lf(ids).float(), dim=-1)
            conf, pred = logp.max(dim=-1)                       # greedy value, confidence=max log-prob
            cand = masked.nonzero(as_tuple=True)[0]
            sel_local = gumbel_top_k(conf[cand] / temp, k, generator=gen)
            chosen = cand[sel_local]
            ids[chosen] = pred[chosen]; masked[chosen] = False
            # commit stats
            pc = pred[chosen]
            n_fill = int((pc == fill_id).sum()); n_eos = int((pc == eos_id).sum())
            n_real = k - n_fill - n_eos
            probs = conf[chosen].exp()                          # softmax prob of the committed token
            show_steps = (n <= 4 or n >= N - 2 or n % 8 == 0)
            print(f"step {n:2d} | +{k:<3d} new  (real={n_real} fill={n_fill} eos={n_eos})  "
                  f"masked_left={int(masked.sum()):3d}  avg_conf={probs.mean().item():.3f}")
            if show_steps:
                # exact: show up to 8 newly-committed content slots (pos: char @ prob)
                items = []
                order = chosen[pred[chosen] != fill_id][:8]
                for p in order.tolist():
                    ch = tok.convert_ids_to_tokens(ids[p].item())
                    items.append(f"f{p}:{'|' if ch=='|' else ch}@{logp[p].max().exp().item():.2f}")
                if items:
                    print("        commits: " + "  ".join(items))
                print(f"        text: |{partial(ids, masked)}|")
    hyp = readout(ids, tok, eos_id)
    print("\n" + "=" * 100)
    print(f"FINAL HYP : {hyp}")
    print(f"REF       : {ref}")
    we = lev(ref.split(), hyp.split()); ce = lev(ref, hyp)
    print(f"WER = {we}/{len(ref.split())} = {we/max(1,len(ref.split())):.3f}   CER = {ce}/{len(ref)} = {ce/max(1,len(ref)):.3f}")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=int, default=0)
    ap.add_argument("--pick", type=int, default=-1, help="row index into the 10-15s candidate list")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, fps = load_model(dev, latest=args.latest)
    specials = build_specials(tok, dev)
    cand = candidates()
    print(f"[cand] {len(cand)} test-clean clips in 10-15s\n")
    if args.search:
        for j, (i, d, r) in enumerate(cand[:args.search]):
            hyp = decode_plain(model, tok, specials, fps, r, dev, N=args.steps)
            ref = norm(r["text"]); we = lev(ref.split(), hyp.split())
            print(f"  [{j:2d}] idx={i} dur={d:.1f}s WER={we}/{len(ref.split())}  hyp: {hyp[:70]}")
    elif args.pick >= 0:
        trace_decode(model, tok, specials, fps, cand[args.pick][2], dev, N=args.steps)


if __name__ == "__main__":
    main()
