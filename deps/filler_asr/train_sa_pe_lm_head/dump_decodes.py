"""Dump full decode chain per utterance + two line-aligned text files (hyp / ref).

For every utt (batch_size=1 so frames = this clip's own encoder frames, no batch padding):
  RAW   : the COMPLETE frame argmax, ALL <fill> + special tokens kept
  NKEEP : RAW truncated to the duration ceiling n_keep = ceil(dur*25)  (still with specials/fill)
  HYP   : cleaned final sentence = NKEEP cut at first </s>, specials+<fill> removed, '|'->space
  WER   : Levenshtein(HYP, REF) / |REF words|     (same metric as infer.py)

Writes into --out_dir (default = run_dir):
  <tag>.decodes.txt : human-readable per-utt chain (RAW/NKEEP/HYP/REF/WER)
  <tag>.hyp.txt     : cleaned hypotheses, ONE PER LINE   (line i  <-> )
  <tag>.ref.txt     : ground-truth references, ONE PER LINE (line i)
  <tag>.dump.summary: overall WER/CER/SER + n
"""
import os, re, sys, json, math, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # parent filler_asr dir holds the reference module
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import (FILL_TOKEN, SAMPLING_RATE, build_tokenizer, build_model,
                                 DataCollatorFillerASR)

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

class Rows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", default="/speech/tomson/filler_asr/data/test_clean.jsonl")
    ap.add_argument("--out_dir", default=""); ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--n", type=int, default=0, help="0 = whole manifest")
    ap.add_argument("--fillrun_k", type=int, default=3,
                    help="cut the hyp at the first run of >=K consecutive <fill> frames")
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    out_dir = args.out_dir or args.run_dir
    tag = os.path.splitext(os.path.basename(args.manifest))[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu"); sd = ck["head"]; cargs = ck.get("args", {}) or {}
    tok = build_tokenizer(args.run_dir)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN); eos_id = tok.eos_token_id
    specials = {tok.bos_token_id, eos_id, tok.pad_token_id, tok.unk_token_id, fill_id}
    bar_id = tok.convert_tokens_to_ids("|"); fillrun_k = args.fillrun_k
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    col = DataCollatorFillerASR(fe, tok, frames_per_sec=args.fps)
    def collate(feats):
        b = col(feats); b["_keys"] = [f["key"] for f in feats]; b["_texts"] = [f["text"] for f in feats]; return b
    # mask-trained ckpts carry a mask_embed param; masking is train-only (gated on
    # self.training) but the param must exist on the model for it to load.
    mask_prob = float(cargs.get("mask_prob") or 0.0) or (1.0 if "mask_embed" in sd else 0.0)
    model = build_model(cargs.get("model_checkpoint") or DEFAULT_CKPT, tok, num_sa_layers=count_sa(sd),
                        use_sinusoidal_pe=bool(cargs.get("pe", True)), device=dev,
                        mask_prob=mask_prob,
                        mask_prob_max=float(cargs.get("mask_prob_max") or 0.0),
                        mask_length=int(cargs.get("mask_length") or 10))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = list(unexpected) + [k for k in missing if k.startswith(("sa.", "lm_head.")) or k == "mask_embed"]
    assert not bad, f"head load mismatch: {bad[:6]}"
    model.eval()
    print(f"[ckpt] {ckpt}  sa_layers={count_sa(sd)}  step={ck.get('step')}  best_wer={ck.get('best_wer')}", flush=True)

    rows = read_manifest(args.manifest)
    if args.n: rows = rows[:args.n]
    loader = DataLoader(Rows(rows), batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)

    # ---- hypothesis stripping variants (all operate on the n_keep frame window) ----
    #   cut modes : where to truncate the frame stream before removing specials/<fill>
    #     eos            -> at first </s>                        (original behaviour)
    #     fillrun        -> at first run of >=K consecutive <fill>
    #     eos_or_fillrun -> whichever comes first (robust to a missing </s>)
    #   collapse : CTC-style merge of consecutive duplicate frames (de-smears w o k k -> w o k)
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
        ("eos",                        "eos",            False),  # <- original canonical HYP
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
    for bi, batch in enumerate(loader):
        key = batch.pop("_keys")[0]; raw_text = batch.pop("_texts")[0]
        labels = batch.pop("labels")
        inp = {k: v.to(dev) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model(**inp).logits.float()
        pred = lg.argmax(-1)[0].cpu()
        T = pred.numel(); n_keep = int((labels[0] != -100).sum())
        dur = inp["attention_mask"][0].sum().item() / SAMPLING_RATE
        ref = norm_ref(raw_text)
        raw_toks  = " ".join(tok.convert_ids_to_tokens([int(x) for x in pred]))
        keep_ids  = pred[:n_keep].tolist()
        keep_toks = " ".join(tok.convert_ids_to_tokens(keep_ids))
        ref_w = ref.split()
        variant_hyps = {name: strip_hyp(keep_ids, mode, coll) for name, mode, coll in VARIANTS}
        hyp = variant_hyps["eos"]                       # canonical (backward compatible)
        we = lev(ref_w, hyp.split()); ce = lev(ref, hyp)
        w_err += we; w_tot += len(ref_w); c_err += ce; c_tot += len(ref); s_err += int(hyp != ref); s_tot += 1
        wer = we / max(1, len(ref_w))
        for name, h in variant_hyps.items():
            agg[name]["we"] += lev(ref_w, h.split()); agg[name]["ce"] += lev(ref, h); agg[name]["se"] += int(h != ref)

        fdet.write(f"================ {key}  dur={dur:.2f}s  T={T}  n_keep={n_keep}  WER={wer:.3f} ================\n")
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
        json.dump({"run": os.path.basename(args.run_dir), "set": tag, "n": s_tot,
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
        json.dump({"run": os.path.basename(args.run_dir), "set": tag, "n": s_tot,
                   "fillrun_k": fillrun_k, "ckpt": ckpt, "variants": variant_rows, "best": best}, f, indent=2)
    print(f"[out ] {os.path.join(out_dir, tag + '.strip_variants.summary')}", flush=True)


if __name__ == "__main__":
    main()
