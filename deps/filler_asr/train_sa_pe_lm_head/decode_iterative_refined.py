"""EXPERIMENTAL improved-refinement decode (NEW code; does NOT modify iterative_decode,
train.py, or the model). Phase 1 (confidence-ordered fill-in) is identical to the omni decoder;
Phase 2 replaces the plain `--remask` with a smarter Mask-Predict that:
  * refines ONLY the transcript body  [1, first </s>)  -- never the </s> or the <fill> tail
  * BANS <fill> at those body slots when re-predicting  (kills the deletion side-effect)
  * selects re-mask targets by smallest top1-top2 MARGIN (or lowest conf)   [--select]
  * ACCEPTS a re-prediction only if it scores >= the old token (no regressions)  [--no_accept off]
  * anneals the re-mask fraction hi->lo over the rounds                          [--frac_hi/--frac_lo]

Default (no flags) reproduces the ORIGINAL loop with remask OFF, so this file is a superset used
only for the max-quality experiment. Reporting matches decode_iterative.py ([done]/[errs] lines).

  python decode_iterative_refined.py --run_dir <run> --ckpt <ckpt> --manifest <jsonl> \
      --out_dir <dir> --steps 32 --refine_rounds 6
"""
import os, re, sys, json, math, argparse
import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import (FILL_TOKEN, build_acmlm_tokenizer, build_acmlm_model,
                                 compute_output_length)
from filler_asr_omni_style_decoding import (build_specials, make_logits_fn,
                                            commit_counts, gumbel_top_k, readout)

DEFAULT_CKPT = "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"
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
def sdi(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1] and dp[i][j] == dp[i-1][j-1]: i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1]+1: S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j]+1: D += 1; i -= 1
        else: I += 1; j -= 1
    return S, D, I
def count_sa(sd):
    idx = {int(m.group(1)) for k in sd for m in [re.match(r"sa\.layers\.(\d+)\.", k)] if m}
    return (max(idx)+1) if idx else 0
def read_manifest(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        rows.append({"key": d.get("key") or d.get("id") or f"utt{len(rows)}",
                     "audio_path": d.get("source") or d.get("audio"),
                     "text": d.get("target") or d.get("text") or ""})
    return rows


@torch.no_grad()
def iterative_decode_refined(logits_fn, T, specials, *, content_len=None, N=32, tau=0.1, temp=1.0,
                             refine_rounds=6, frac_hi=0.20, frac_lo=0.05, select="margin",
                             ban_fill=True, accept_if_better=True, generator=None):
    mask_id, fill_id, bos_id, eos_id = specials["mask"], specials["fill"], specials["bos"], specials["eos"]
    dev = specials.get("device", "cpu")
    content_len = T if content_len is None else min(content_len, T)
    NEG = float("-inf")

    ids = torch.full((T,), fill_id, dtype=torch.long, device=dev)
    ids[:content_len] = mask_id
    masked = torch.zeros(T, dtype=torch.bool, device=dev); masked[:content_len] = True
    if bos_id is not None:
        ids[0] = bos_id; masked[0] = False
    M = int(masked.sum().item())

    def logp_all(x): return F.log_softmax(logits_fn(x).float(), dim=-1)      # (T, V)

    # ---- Phase 1: confidence-ordered fill-in (identical to the original loop) ----
    for n, k in enumerate(commit_counts(N, tau, M), start=1):
        if k <= 0: continue
        lp = logp_all(ids); conf, pred = lp.max(-1)
        cand = masked.nonzero(as_tuple=True)[0]
        chosen = cand[gumbel_top_k(conf[cand] / temp, k, generator=generator)]
        ids[chosen] = pred[chosen]; masked[chosen] = False

    # ---- Phase 2: improved refinement (body only; fill banned; accept-if-better) ----
    for r in range(max(0, refine_rounds)):
        hits = (ids[:content_len] == eos_id).nonzero(as_tuple=True)[0]
        eos_pos = int(hits[0]) if hits.numel() else content_len          # start of </s>/tail
        if eos_pos <= 1: break
        body = torch.arange(1, eos_pos, device=dev)                      # transcript chars only
        frac = frac_hi + (frac_lo - frac_hi) * (r / max(1, refine_rounds - 1))
        k_re = max(1, int(round(frac * body.numel())))

        lp = logp_all(ids)
        if ban_fill: lp[body, fill_id] = NEG
        pb = lp[body]                                                    # (Lb, V)
        if select == "margin":
            top2 = pb.topk(2, dim=-1).values
            score = top2[:, 0] - top2[:, 1]                              # small margin => uncertain
        else:                                                           # 'conf' of current token
            score = pb.gather(-1, ids[body].unsqueeze(-1)).squeeze(-1)
        worst = body[torch.topk(-score, min(k_re, body.numel())).indices]

        old = ids[worst].clone()
        ids[worst] = mask_id
        lp2 = logp_all(ids)
        if ban_fill: lp2[body, fill_id] = NEG
        ids[worst] = lp2[worst].argmax(-1)

        if accept_if_better:                                            # revert if the old token was better
            lp3 = logp_all(ids)
            if ban_fill: lp3[body, fill_id] = NEG
            new = ids[worst]
            new_c = lp3[worst].gather(-1, new.unsqueeze(-1)).squeeze(-1)
            old_c = lp3[worst].gather(-1, old.unsqueeze(-1)).squeeze(-1)
            rev = old_c > new_c
            ids[worst[rev]] = old[rev]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", required=True); ap.add_argument("--out_dir", default="")
    ap.add_argument("--fps", type=int, default=0); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--steps", type=int, default=32); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refine_rounds", type=int, default=6)
    ap.add_argument("--frac_hi", type=float, default=0.20); ap.add_argument("--frac_lo", type=float, default=0.05)
    ap.add_argument("--select", choices=["margin", "conf"], default="margin")
    ap.add_argument("--no_ban_fill", action="store_true"); ap.add_argument("--no_accept", action="store_true")
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    out_dir = args.out_dir or args.run_dir; os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.manifest))[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu"); sd = ck["head"]; cargs = ck.get("args", {}) or {}
    fps = args.fps or int(cargs.get("frames_per_sec") or 50)

    tok = build_acmlm_tokenizer(args.run_dir)
    eos_id = tok.eos_token_id
    dec_specials = build_specials(tok, dev)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    model = build_acmlm_model(cargs.get("model_checkpoint") or DEFAULT_CKPT, tok,
                              num_sa_layers=count_sa(sd) or int(cargs.get("num_sa_layers", 8)),
                              use_sinusoidal_pe=bool(cargs.get("pe", True)), device=dev,
                              mask_prob=float(cargs.get("mask_prob") or 0.20),
                              mask_prob_max=float(cargs.get("mask_prob_max") or 0.30),
                              mask_length=int(cargs.get("mask_length") or 10),
                              pos_mode=cargs.get("pos_mode", "sinusoidal"))
    model.load_state_dict(sd, strict=False); model.eval()
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    print(f"[ckpt] {ckpt} step={ck.get('step')} pos_mode={cargs.get('pos_mode','sinusoidal')} fps={fps}", flush=True)
    print(f"[refine] steps={args.steps} rounds={args.refine_rounds} frac={args.frac_hi}->{args.frac_lo} "
          f"select={args.select} ban_fill={not args.no_ban_fill} accept={not args.no_accept}", flush=True)

    rows = read_manifest(args.manifest)
    if args.n: rows = rows[:args.n]
    w_err = w_tot = c_err = c_tot = s_err = s_tot = 0; n_sub = n_del = n_ins = 0
    fh = open(os.path.join(out_dir, f"{tag}.hyp.txt"), "w"); fr = open(os.path.join(out_dir, f"{tag}.ref.txt"), "w")
    for bi, r in enumerate(rows):
        arr, sr = sf.read(r["audio_path"], dtype="float32")
        if arr.ndim > 1: arr = arr.mean(1)
        iv = fe(arr, sampling_rate=16000, return_tensors="pt")
        input_values = iv["input_values"].to(dev)
        am = iv.get("attention_mask"); am = am.to(dev) if am is not None else None
        T = compute_output_length(input_values.shape[-1])
        n_keep = min(T, math.ceil(len(arr) / 16000 * fps))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits_fn = make_logits_fn(model, input_values, am)
            ids = iterative_decode_refined(logits_fn, T, dec_specials, content_len=n_keep, N=args.steps,
                                           tau=args.tau, temp=args.temp, refine_rounds=args.refine_rounds,
                                           frac_hi=args.frac_hi, frac_lo=args.frac_lo, select=args.select,
                                           ban_fill=not args.no_ban_fill, accept_if_better=not args.no_accept,
                                           generator=gen)
        hyp = readout(ids.cpu(), tok, eos_id).strip()
        ref = norm_ref(r["text"]); rw = ref.split()
        we = lev(rw, hyp.split()); ce = lev(ref, hyp); S, D, I = sdi(rw, hyp.split())
        n_sub += S; n_del += D; n_ins += I
        w_err += we; w_tot += len(rw); c_err += ce; c_tot += len(ref); s_err += int(hyp != ref); s_tot += 1
        fh.write(hyp + "\n"); fr.write(ref + "\n")
        if (bi + 1) % 200 == 0:
            print(f"  ..{bi+1}/{len(rows)}  running WER={w_err/max(1,w_tot):.4f}", flush=True)
    fh.close(); fr.close()
    WER = w_err/max(1,w_tot); CER = c_err/max(1,c_tot); SER = s_err/max(1,s_tot)
    print(f"[done] {os.path.basename(args.run_dir)} {tag} n={s_tot} steps={args.steps} "
          f"WER={WER:.4f} CER={CER:.4f} SER={SER:.4f}", flush=True)
    print(f"[errs] words={w_tot}  sub={n_sub} del={n_del} ins={n_ins}  WER_check={(n_sub+n_del+n_ins)/max(1,w_tot):.4f}", flush=True)


if __name__ == "__main__":
    main()
