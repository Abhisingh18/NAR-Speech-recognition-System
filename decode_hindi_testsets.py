"""Iterative (A-CMLM omni-style) decode of the Hindi test sets with the trained data2vec-aqc head,
then score with akshaya's NFC-normalized WER.

Real-inference decode (NOT oracle length): content_len = n_keep = min(T, ceil(dur*fps)); at the
trained fps=50 that is the whole frame grid, and </s> terminates the transcript (readout cuts at
the first </s>). Mirrors filler_asr_omni_style_decoding.run_real, but builds the data2vec head via
build_d2v_acmlm and loads the CTC-encoder checkpoint the model was trained with.

Usage:
    CUDA_VISIBLE_DEVICES=0 python decode_hindi_testsets.py \
        --run_dir runs/hi_d2vaqcCTC_sa8_ACMLM_alibi_fps50_fw003_exp150 --steps 32
"""
import os
import sys
import math
import json
import time
import argparse
import subprocess

import numpy as np
import torch
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FILLER_ROOT = os.environ.get("FILLER_ROOT", "/speech/tomson/filler_asr")
sys.path.insert(0, FILLER_ROOT)
sys.path.insert(0, os.path.join(FILLER_ROOT, "train_sa_pe_lm_head"))

from transformers import Wav2Vec2FeatureExtractor                                   # noqa: E402
from filler_sa_reference import compute_output_length                              # noqa: E402
from filler_asr_omni_style_decoding import (iterative_decode, build_specials,       # noqa: E402
                                            readout, make_logits_fn)
from model_d2v_acmlm import build_d2v_acmlm                                         # noqa: E402
from text_norm_hi import build_acmlm_tokenizer_hi                                   # noqa: E402

DATA_FS = "/speech/akshaya/fairseq_expxx/hindi/data_fs"
WER_SCRIPT = "/speech/akshaya/OMNI_ASR/decodes/indic_normalise_werNFC.py"
# display name -> data_fs subdir (order = report order)
TESTSETS = [
    ("indictts",       "indictts_hindi_test"),
    ("fleurs",         "fleurs_hindi_test"),
    ("kathbath",       "kathbath_hindi_test"),
    ("kathbath_noisy", "kathbath_noisy_hindi_test"),
    ("evaliitm",       "eval_IITM_Hindi"),
    ("commonvoice",    "commonvoice_hindi_test"),
]


def read_tsv_wrd(setdir):
    """Return list of (abs_wav_path, transcript) from fairseq test.tsv + test.wrd."""
    tsv, wrd = os.path.join(setdir, "test.tsv"), os.path.join(setdir, "test.wrd")
    with open(tsv) as f:
        root = f.readline().strip()
        rels = [ln.split("\t")[0] for ln in f if ln.strip()]
    with open(wrd) as f:
        refs = [ln.rstrip("\n") for ln in f]
    assert len(rels) == len(refs), f"{setdir}: tsv {len(rels)} != wrd {len(refs)}"
    return [(os.path.join(root, r), t) for r, t in zip(rels, refs)]


@torch.no_grad()
def decode_set(model, tok, specials, fe, rows, fps, steps, dev, name, out_dir, limit=0):
    """Decode one test set, write kaldi-style .ref/.hyp keyed by index. Return (ref_f, hyp_f)."""
    if limit:
        rows = rows[:limit]
    ref_f = os.path.join(out_dir, f"{name}.ref")
    hyp_f = os.path.join(out_dir, f"{name}.hyp")
    eos_id = specials["eos"]
    t0 = time.time()
    with open(ref_f, "w") as fr, open(hyp_f, "w") as fh:
        for i, (wav, ref) in enumerate(rows):
            key = f"{name}_{i:06d}"
            arr, sr = sf.read(wav, dtype="float32")
            assert sr == 16000, f"{wav} is {sr}Hz, expected 16000"
            if arr.ndim > 1:
                arr = arr.mean(1)
            iv = fe(arr, sampling_rate=16000, return_tensors="pt")
            input_values = iv["input_values"].to(dev)
            am = iv.get("attention_mask")
            am = am.to(dev) if am is not None else None
            T = compute_output_length(input_values.shape[-1])
            n_keep = min(T, math.ceil(len(arr) / 16000 * fps))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits_fn = make_logits_fn(model, input_values, am)     # caches the tap once
                ids = iterative_decode(logits_fn, T, specials, content_len=n_keep, N=steps)
            hyp = readout(ids, tok, eos_id)
            fr.write(f"{key} {ref}\n")
            fh.write(f"{key} {hyp}\n")
            if (i + 1) % 100 == 0:
                print(f"  [{name}] {i+1}/{len(rows)}  {(i+1)/(time.time()-t0):.1f} utt/s", flush=True)
    print(f"  [{name}] done {len(rows)} utts in {time.time()-t0:.0f}s", flush=True)
    return ref_f, hyp_f


def score(ref_f, hyp_f, out_dir, name):
    """Run akshaya's NFC WER scorer, return (WER, SER, nwords) parsed from its output file."""
    wer_f = os.path.join(out_dir, f"{name}.wer")
    subprocess.run([sys.executable, WER_SCRIPT, ref_f, hyp_f, wer_f], check=True)
    wer = ser = nw = None
    with open(wer_f) as f:
        for ln in f:
            if ln.startswith("%WER"):
                wer = float(ln.split()[1]); nw = int(ln.split("/")[1].split(",")[0])
            elif ln.startswith("%SER"):
                ser = float(ln.split()[1])
    return wer, ser, nw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--encoder_pt", default="", help="override; default = the ckpt's training encoder")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N clips per set (smoke)")
    ap.add_argument("--only", default="", help="comma-sep subset of set names to run")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    ck = torch.load(ckpt, map_location="cpu")
    cargs = ck.get("args", {}) or {}
    enc = args.encoder_pt or cargs.get("model_checkpoint")
    fps = int(cargs.get("frames_per_sec", 50))
    vocab_dir = cargs.get("vocab_dir") or os.path.join(HERE, "vocab_hi")
    out_dir = args.out_dir or os.path.join(args.run_dir, f"decode_testsets_N{args.steps}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ckpt] {ckpt}  step={ck.get('step')} epoch={ck.get('epoch')}", flush=True)
    print(f"[enc ] {enc}", flush=True)
    print(f"[cfg ] fps={fps} steps={args.steps} sa={cargs.get('num_sa_layers')} pos={cargs.get('pos_mode')} "
          f"pe={cargs.get('pe')} dev={dev}", flush=True)

    tok = build_acmlm_tokenizer_hi(vocab_dir)
    specials = build_specials(tok, dev)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    model = build_d2v_acmlm(tok, d2v_ckpt=enc, num_sa_layers=int(cargs.get("num_sa_layers", 8)),
                            use_sinusoidal_pe=bool(cargs.get("pe", True)),
                            pos_mode=cargs.get("pos_mode", "alibi"),
                            mask_prob=float(cargs.get("mask_prob", 0.20)),
                            mask_prob_max=float(cargs.get("mask_prob_max", 0.30)),
                            mask_length=int(cargs.get("mask_length", 10)),
                            tgt_layer=cargs.get("encoder_tgt_layer"), device=dev)
    missing, unexpected = model.load_state_dict(ck["head"], strict=False)
    bad = list(unexpected) + [k for k in missing
                              if k.startswith(("sa.", "lm_head.", "text_embed.")) or k == "mask_embed"]
    assert not bad, f"head load mismatch: {bad[:6]}"
    model.eval()

    want = set(s.strip() for s in args.only.split(",") if s.strip())
    results = []
    for name, sub in TESTSETS:
        if want and name not in want:
            continue
        setdir = os.path.join(DATA_FS, sub)
        rows = read_tsv_wrd(setdir)
        print(f"\n== {name}  ({len(rows)} utts) -> {setdir} ==", flush=True)
        ref_f, hyp_f = decode_set(model, tok, specials, fe, rows, fps, args.steps, dev,
                                  name, out_dir, limit=args.limit)
        wer, ser, nw = score(ref_f, hyp_f, out_dir, name)
        results.append((name, len(rows) if not args.limit else min(args.limit, len(rows)), nw, wer, ser))
        print(f"  [{name}] %WER {wer}  %SER {ser}  (words={nw})", flush=True)

    print("\n" + "=" * 62)
    print(f"{'testset':<16}{'utts':>7}{'words':>9}{'WER%':>9}{'SER%':>9}")
    print("-" * 62)
    for name, n, nw, wer, ser in results:
        print(f"{name:<16}{n:>7}{nw:>9}{wer:>9}{ser:>9}")
    print("=" * 62)
    summ = os.path.join(out_dir, "SUMMARY.tsv")
    with open(summ, "w") as f:
        f.write("testset\tutts\twords\tWER\tSER\n")
        for name, n, nw, wer, ser in results:
            f.write(f"{name}\t{n}\t{nw}\t{wer}\t{ser}\n")
    print(f"[out] per-set .ref/.hyp/.wer + SUMMARY.tsv -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
