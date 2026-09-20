"""Decode test_clean with the EXACT training-time inference of the A-CMLM head, then
sweep the hypothesis-stripping variants and report WER/CER/SER per variant.

This mirrors train.py::evaluate() (the wandb hyp/ref table + eval/loss decode):
  * ONE forward pass on a 100%-masked text canvas  (== eval collator fixed_p=1.0)
  * frame argmax -> full per-frame token sequence   (NON-autoregressive, single step)
  * cut at the FIRST predicted </s>, drop specials + <fill>, '|' -> space

It is the A-CMLM twin of dump_decodes.py (which drives the NON-acmlm parent model): the
only change is the model/tokenizer (build_acmlm_*) and the one-shot masked forward. All
the n_keep capping + stripping-variant comparison machinery is identical, so the numbers
line up directly against the omni-style step sweep (sweep_steps.sh) as its N=1 baseline.

  python decode_train_infer.py --run_dir runs/<acmlm_run> \
      --manifest .../librispeech_test_clean.jsonl --out_dir <dir>
"""
import os, re, sys, json, math, argparse
import torch

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # parent filler_asr dir holds the reference module
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import (FILL_TOKEN, SAMPLING_RATE, build_acmlm_tokenizer,
                                 build_acmlm_model, compute_output_length)

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
    print(f"[ckpt] {ckpt}  sa_layers={count_sa(sd)}  step={ck.get('step')}  "
          f"best_val={ck.get('best_val', ck.get('best_wer'))}  decode fps={fps} "
          f"(train fps={cargs.get('frames_per_sec','?')})", flush=True)

    rows = read_manifest(args.manifest)
    if args.n: rows = rows[:args.n]

    # ---- hypothesis stripping variants (identical to dump_decodes.py) ----
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
        ("eos",                        "eos",            False),  # <- canonical (== training eval)
        ("eos+collapse",               "eos",            True),
        (f"fillrun{fillrun_k}",        "fillrun",        False),
        (f"eos_or_fillrun{fillrun_k}", "eos_or_fillrun", False),
        ("eos|fillrun+coll",           "eos_or_fillrun", True),
    ]
    agg = {name: {"we": 0, "ce": 0, "se": 0} for name, _, _ in VARIANTS}

    w_err = w_tot = c_err = c_tot = s_err = s_tot = 0
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
        # TRAINING INFERENCE: text_input_ids=None -> model seeds a 100%-masked canvas
        # (== eval collator fixed_p=1.0) and reconstructs it in ONE forward pass.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(input_values=input_values, attention_mask=attention_mask).logits.float()
        pred = lg.argmax(-1)[0].cpu()
        T = pred.numel()
        n_keep = min(T, math.ceil(len(arr) / 16000 * fps))            # duration budget (matches training)
        dur = len(arr) / 16000
        ref = norm_ref(r["text"])
        raw_toks  = " ".join(tok.convert_ids_to_tokens([int(x) for x in pred]))
        keep_ids  = pred[:n_keep].tolist()
        keep_toks = " ".join(tok.convert_ids_to_tokens(keep_ids))
        ref_w = ref.split()
        variant_hyps = {name: strip_hyp(keep_ids, mode, coll) for name, mode, coll in VARIANTS}
        hyp = variant_hyps["eos"]                       # canonical (== training eval decode)
        we = lev(ref_w, hyp.split()); ce = lev(ref, hyp)
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
    print(f"\n[done] {os.path.basename(args.run_dir)} {tag} n={s_tot} WER={WER:.4f} CER={CER:.4f} SER={SER:.4f}", flush=True)
    for nm in (f"{tag}.decodes.txt", f"{tag}.hyp.txt", f"{tag}.ref.txt"):
        print(f"[out ] {os.path.join(out_dir, nm)}", flush=True)
    with open(os.path.join(out_dir, f"{tag}.dump.summary"), "w") as f:
        json.dump({"run": os.path.basename(args.run_dir), "set": tag, "n": s_tot, "decode": "train_infer_1shot",
                   "WER": WER, "CER": CER, "SER": SER, "cap": "min(first </s>, n_keep)", "ckpt": ckpt}, f, indent=2)

    # ---- stripping-variant comparison (all still capped inside n_keep) ----
    print(f"\n=== stripping-variant comparison  (n={s_tot}, fillrun_k={fillrun_k}) ===", flush=True)
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
        json.dump({"run": os.path.basename(args.run_dir), "set": tag, "n": s_tot, "decode": "train_infer_1shot",
                   "fillrun_k": fillrun_k, "ckpt": ckpt, "variants": variant_rows, "best": best}, f, indent=2)
    print(f"[out ] {os.path.join(out_dir, tag + '.strip_variants.summary')}", flush=True)


if __name__ == "__main__":
    main()
