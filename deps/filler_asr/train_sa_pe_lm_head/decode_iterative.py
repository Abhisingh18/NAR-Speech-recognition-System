"""Decode a manifest with the OmniVoice-style ITERATIVE masked-diffusion decoder
(filler_asr_omni_style_decoding.py) at a CONFIGURABLE number of steps, then sweep the
hypothesis-stripping variants and report WER/CER/SER per variant.

Same reporting harness as decode_train_infer.py (n_keep cap + stripping variants + hyp/ref
files); the ONLY difference is the decode:
  * decode_train_infer.py -> ONE forward pass, argmax           (== training inference, N=1)
  * this script           -> iterative_decode(..., N=--steps)   (commit a fraction / step)

Pick the step count with --steps (16 / 32 / 64 / 100 / ...). --steps 1 reproduces the
one-shot training-inference decode.

  python decode_iterative.py --run_dir runs/<acmlm_run> \
      --manifest .../librispeech_test_clean.jsonl --out_dir <dir> \
      --steps 32 --tau 0.1 --temp 5 [--fill_penalty 0.0] [--remask]
"""
import os, re, sys, json, math, argparse
import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # parent filler_asr dir holds the reference module
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import (FILL_TOKEN, build_acmlm_tokenizer, build_acmlm_model,
                                 compute_output_length)
# reuse the SAME iterative loop the omni script uses -> single source of truth
from filler_asr_omni_style_decoding import iterative_decode, build_specials, make_logits_fn

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
    """Edit distance WITH a backtrace -> (substitutions, deletions, insertions).
    D = ref word missing from hyp ; I = extra hyp word ; total edits S+D+I == lev()."""
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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            rows.append({"key": d.get("key") or d.get("id") or f"utt{len(rows)}",
                         "audio_path": d.get("source") or d.get("audio"),
                         "text": d.get("target") or d.get("text") or ""})
    return rows


@torch.no_grad()
def main():
    import soundfile as sf
    from transformers import Wav2Vec2FeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--fps", type=int, default=0, help="0 = read frames_per_sec from ckpt args (matches training budget)")
    ap.add_argument("--n", type=int, default=0, help="0 = whole manifest")
    ap.add_argument("--fillrun_k", type=int, default=3,
                    help="cut the hyp at the first run of >=K consecutive <fill> frames")
    # ---- iterative-decode config ----
    ap.add_argument("--steps", type=int, default=32, help="# iterative unmasking steps (1 = one-shot / training inference)")
    ap.add_argument("--tau", type=float, default=0.1, help="schedule shape: <1 back-loaded, 1 linear, >1 front-loaded")
    ap.add_argument("--temp", type=float, default=5.0, help="position-selection temperature (Gumbel-top-k)")
    ap.add_argument("--fill_penalty", type=float, default=0.0, help="subtract from <fill> confidence before committing")
    ap.add_argument("--remask", action="store_true", help="add Mask-Predict refinement rounds after the fill-in decode")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the Gumbel position sampling")
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    out_dir = args.out_dir or args.run_dir; os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.manifest))[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu"); sd = ck["head"]; cargs = ck.get("args", {}) or {}
    fps = args.fps or int(cargs.get("frames_per_sec") or 50)

    tok = build_acmlm_tokenizer(args.run_dir)                          # 34-token vocab (incl <mask>)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN); eos_id = tok.eos_token_id
    specials = {tok.bos_token_id, eos_id, tok.pad_token_id, tok.unk_token_id, fill_id}
    dec_specials = build_specials(tok, dev)                            # dict(mask/bos/eos/pad/fill/device) for the loop
    bar_id = tok.convert_tokens_to_ids("|"); fillrun_k = args.fillrun_k
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    model = build_acmlm_model(cargs.get("model_checkpoint") or DEFAULT_CKPT, tok,
                              num_sa_layers=count_sa(sd) or int(cargs.get("num_sa_layers", 8)),
                              use_sinusoidal_pe=bool(cargs.get("pe", True)), device=dev,
                              mask_prob=float(cargs.get("mask_prob") or 0.20),
                              mask_prob_max=float(cargs.get("mask_prob_max") or 0.30),
                              mask_length=int(cargs.get("mask_length") or 10),
                              pos_mode=cargs.get("pos_mode", "sinusoidal"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = list(unexpected) + [k for k in missing
                              if k.startswith(("sa.", "lm_head.", "text_embed.")) or k == "mask_embed"]
    assert not bad, f"head load mismatch: {bad[:6]}"
    model.eval()                                                      # eval() -> train-only audio-mask is OFF
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    print(f"[ckpt] {ckpt}  sa_layers={count_sa(sd)}  step={ck.get('step')}  "
          f"best_val={ck.get('best_val', ck.get('best_wer'))}  decode fps={fps} "
          f"(train fps={cargs.get('frames_per_sec','?')})", flush=True)
    print(f"[decode] iterative  steps={args.steps} tau={args.tau} temp={args.temp} "
          f"fill_penalty={args.fill_penalty} remask={args.remask} seed={args.seed}", flush=True)

    rows = read_manifest(args.manifest)
    if args.n: rows = rows[:args.n]

    # ---- hypothesis stripping variants (identical to decode_train_infer.py) ----
    def _cut_idx(ids, mode):
        eos_at = next((i for i, x in enumerate(ids) if x == eos_id), None)
        fill_at, run = None, 0
        for i, x in enumerate(ids):
            if x == fill_id:
                run += 1
                if run >= fillrun_k: fill_at = i - fillrun_k + 1; break
            else:
                run = 0
        cands = []
        if mode in ("eos", "eos_or_fillrun") and eos_at is not None:  cands.append(eos_at)
        if mode in ("fillrun", "eos_or_fillrun") and fill_at is not None: cands.append(fill_at)
        return min(cands) if cands else len(ids)

    def _to_text(ids, collapse):
        if collapse:
            merged = []
            for x in ids:
                if not merged or merged[-1] != x: merged.append(x)
            ids = merged
        out = []
        for x in ids:
            if x in specials: continue
            out.append(" " if x == bar_id else tok.convert_ids_to_tokens(int(x)))
        return "".join(out).strip()

    def strip_hyp(ids, mode="eos", collapse=False):
        return _to_text(ids[:_cut_idx(ids, mode)], collapse)

    VARIANTS = [                                       # (name, cut mode, collapse)
        ("eos",                        "eos",            False),  # <- canonical (matches infer.py / omni readout)
        ("eos+collapse",               "eos",            True),
        (f"fillrun{fillrun_k}",        "fillrun",        False),
        (f"eos_or_fillrun{fillrun_k}", "eos_or_fillrun", False),
        ("eos|fillrun+coll",           "eos_or_fillrun", True),
    ]
    agg = {name: {"we": 0, "ce": 0, "se": 0} for name, _, _ in VARIANTS}

    w_err = w_tot = c_err = c_tot = s_err = s_tot = 0
    n_sub = n_del = n_ins = 0                                          # word-level S/D/I on the canonical hyp
    fdet = open(os.path.join(out_dir, f"{tag}.decodes.txt"), "w")
    fhyp = open(os.path.join(out_dir, f"{tag}.hyp.txt"), "w")
    fref = open(os.path.join(out_dir, f"{tag}.ref.txt"), "w")
    for bi, r in enumerate(rows):
        arr, sr = sf.read(r["audio_path"], dtype="float32")
        if arr.ndim > 1: arr = arr.mean(1)
        iv = fe(arr, sampling_rate=16000, return_tensors="pt")
        input_values = iv["input_values"].to(dev)
        attention_mask = iv.get("attention_mask")
        attention_mask = attention_mask.to(dev) if attention_mask is not None else None
        T = compute_output_length(input_values.shape[-1])
        n_keep = min(T, math.ceil(len(arr) / 16000 * fps))            # duration budget (matches training)
        # ITERATIVE DECODE: tap cached once, text grid re-read each of --steps commits.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits_fn = make_logits_fn(model, input_values, attention_mask)
            ids = iterative_decode(logits_fn, T, dec_specials, content_len=n_keep, N=args.steps,
                                   tau=args.tau, temp=args.temp, fill_penalty=args.fill_penalty,
                                   remask=args.remask, generator=gen)
        pred = ids.cpu()
        dur = len(arr) / 16000
        ref = norm_ref(r["text"])
        raw_toks  = " ".join(tok.convert_ids_to_tokens([int(x) for x in pred]))
        keep_ids  = pred[:n_keep].tolist()
        keep_toks = " ".join(tok.convert_ids_to_tokens(keep_ids))
        ref_w = ref.split()
        variant_hyps = {name: strip_hyp(keep_ids, mode, coll) for name, mode, coll in VARIANTS}
        hyp = variant_hyps["eos"]                       # canonical
        we = lev(ref_w, hyp.split()); ce = lev(ref, hyp)
        S, D, I = sdi(ref_w, hyp.split()); n_sub += S; n_del += D; n_ins += I
        w_err += we; w_tot += len(ref_w); c_err += ce; c_tot += len(ref); s_err += int(hyp != ref); s_tot += 1
        wer = we / max(1, len(ref_w))
        for name, h in variant_hyps.items():
            agg[name]["we"] += lev(ref_w, h.split()); agg[name]["ce"] += lev(ref, h); agg[name]["se"] += int(h != ref)

        fdet.write(f"================ {r['key']}  dur={dur:.2f}s  T={T}  n_keep={n_keep}  WER={wer:.3f} ================\n")
        fdet.write(f"RAW   : {raw_toks}\n")
        fdet.write(f"NKEEP : {keep_toks}\n")
        fdet.write(f"HYP   : {hyp}\n")
        fdet.write(f"REF   : {ref}\n\n")
        fhyp.write(hyp + "\n")
        fref.write(ref + "\n")
        if (bi + 1) % 200 == 0:
            print(f"  ..{bi+1}/{len(rows)}  running WER={w_err/max(1,w_tot):.4f}", flush=True)
    fdet.close(); fhyp.close(); fref.close()

    WER, CER, SER = w_err/max(1, w_tot), c_err/max(1, c_tot), s_err/max(1, s_tot)
    E = max(1, n_sub + n_del + n_ins)                                 # total word edits (== w_err)
    print(f"\n[done] {os.path.basename(args.run_dir)} {tag} n={s_tot} steps={args.steps} "
          f"WER={WER:.4f} CER={CER:.4f} SER={SER:.4f}", flush=True)
    print(f"[errs] words={w_tot}  sub={n_sub} del={n_del} ins={n_ins}  "
          f"(S/D/I = {n_sub/E*100:.0f}/{n_del/E*100:.0f}/{n_ins/E*100:.0f}%)  "
          f"WER_check={(n_sub+n_del+n_ins)/max(1,w_tot):.4f}", flush=True)
    for nm in (f"{tag}.decodes.txt", f"{tag}.hyp.txt", f"{tag}.ref.txt"):
        print(f"[out ] {os.path.join(out_dir, nm)}", flush=True)
    with open(os.path.join(out_dir, f"{tag}.dump.summary"), "w") as f:
        json.dump({"run": os.path.basename(args.run_dir), "set": tag, "n": s_tot,
                   "decode": "iterative", "steps": args.steps, "tau": args.tau, "temp": args.temp,
                   "fill_penalty": args.fill_penalty, "remask": args.remask,
                   "WER": WER, "CER": CER, "SER": SER,
                   "words": w_tot, "sub": n_sub, "del": n_del, "ins": n_ins,
                   "cap": "min(first </s>, n_keep)", "ckpt": ckpt}, f, indent=2)

    # ---- stripping-variant comparison (all still capped inside n_keep) ----
    print(f"\n=== stripping-variant comparison  (n={s_tot}, steps={args.steps}, fillrun_k={fillrun_k}) ===", flush=True)
    print(f"{'variant':22s} {'WER':>8s} {'CER':>8s} {'SER':>8s}", flush=True)
    variant_rows = []
    for name, mode, coll in VARIANTS:
        a = agg[name]
        vw, vc, vs = a["we"]/max(1, w_tot), a["ce"]/max(1, c_tot), a["se"]/max(1, s_tot)
        variant_rows.append({"variant": name, "cut": mode, "collapse": coll, "WER": vw, "CER": vc, "SER": vs})
        print(f"{name:22s} {vw:8.4f} {vc:8.4f} {vs:8.4f}", flush=True)
    best = min(variant_rows, key=lambda r: r["WER"])
    print(f"[best] {best['variant']}  WER={best['WER']:.4f}  (baseline eos={variant_rows[0]['WER']:.4f})", flush=True)
    with open(os.path.join(out_dir, f"{tag}.strip_variants.summary"), "w") as f:
        json.dump({"run": os.path.basename(args.run_dir), "set": tag, "n": s_tot, "decode": "iterative",
                   "steps": args.steps, "fillrun_k": fillrun_k, "ckpt": ckpt,
                   "variants": variant_rows, "best": best}, f, indent=2)
    print(f"[out ] {os.path.join(out_dir, tag + '.strip_variants.summary')}", flush=True)


if __name__ == "__main__":
    main()
