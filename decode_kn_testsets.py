"""Iterative (A-CMLM mask-predict) decode of the Kannada test sets + WER/CER (+ wandb logging).

Kannada twin of decode_hindi_testsets.py. Differences:
  * test sets come from the jsonl written by prep_kn_manifests.py (data_kn/test_<name>.jsonl)
  * refs AND hyps are both passed through normalize_kn before scoring (same text space as training)
  * scoring is done in-process (word S/D/I + char edits), no external WER script
  * the decode is SEEDED per test set (--seed) so re-running the same checkpoint gives the same numbers
  * writes per_utt.tsv (duration, S/D/I, ...) that plot_kn_run.py turns into inference plots
  * --wandb logs the summary + per-set bar charts + a sample table to wandb

  CUDA_VISIBLE_DEVICES=6 python decode_kn_testsets.py --run_dir runs/kn_en_... --steps 32 [--wandb]
"""
import os
import sys
import glob
import json
import math
import time
import argparse

import torch
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FILLER_ROOT", os.path.join(HERE, "deps", "filler_asr"))   # imported modules read this
FILLER_ROOT = os.environ["FILLER_ROOT"]
sys.path.insert(0, FILLER_ROOT)
sys.path.insert(0, os.path.join(FILLER_ROOT, "train_sa_pe_lm_head"))
os.environ.setdefault("SLAM_SRC", os.path.join(HERE, "deps", "SMEAR-MoE-ASR", "src"))

from transformers import Wav2Vec2FeatureExtractor                                   # noqa: E402
from filler_sa_reference import compute_output_length                              # noqa: E402
from filler_asr_omni_style_decoding import (iterative_decode, build_specials,       # noqa: E402
                                            readout, make_logits_fn)
from model_d2v_acmlm import build_d2v_acmlm                                         # noqa: E402
from text_norm_kn import normalize_kn, build_acmlm_tokenizer_kn                     # noqa: E402
from wer_utils import align_counts, edit_distance                                   # noqa: E402

PER_UTT_COLS = ["testset", "key", "duration", "language", "ref_words", "hyp_words", "S", "D", "I",
                "char_edits", "ref_chars"]


@torch.no_grad()
def decode_set(model, tok, specials, fe, rows, fps, steps, dev, name, out_dir, seed, per_utt, samples,
               remask=False, remask_rounds=4, remask_frac=0.15):
    eos_id = specials["eos"]
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    tot = dict(S=0, D=0, I=0, N=0, CE=0, CN=0, exact=0)
    t0 = time.time()
    with open(os.path.join(out_dir, f"{name}.ref"), "w") as fr, \
         open(os.path.join(out_dir, f"{name}.hyp"), "w") as fh:
        for i, r in enumerate(rows):
            arr, sr = sf.read(r["source"], dtype="float32")
            assert sr == 16000, f"{r['source']} is {sr}Hz"
            if arr.ndim > 1:
                arr = arr.mean(1)
            iv = fe(arr, sampling_rate=16000, return_tensors="pt")
            x = iv["input_values"].to(dev)
            am = iv.get("attention_mask")
            am = am.to(dev) if am is not None else None
            T = compute_output_length(x.shape[-1])
            n_keep = min(T, math.ceil(len(arr) / 16000 * fps))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                ids = iterative_decode(make_logits_fn(model, x, am), T, specials,
                                       content_len=n_keep, N=steps, generator=gen,
                                       remask=remask, remask_rounds=remask_rounds, remask_frac=remask_frac)
            hyp = normalize_kn(readout(ids, tok, eos_id))
            ref = normalize_kn(r.get("target_raw") or r["target"])
            rw, hw = ref.split(), hyp.split()
            S, D, I = align_counts(rw, hw)
            rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
            ce = edit_distance(rc, hc)
            tot["S"] += S; tot["D"] += D; tot["I"] += I; tot["N"] += len(rw)
            tot["CE"] += ce; tot["CN"] += len(rc); tot["exact"] += int(ref == hyp)
            key = r.get("key") or f"{name}_{i:06d}"
            fr.write(f"{key} {ref}\n")
            fh.write(f"{key} {hyp}\n")
            per_utt.append(dict(testset=name, key=key, duration=round(len(arr) / 16000, 3),
                                language=r.get("language", ""), ref_words=len(rw), hyp_words=len(hw),
                                S=S, D=D, I=I, char_edits=ce, ref_chars=len(rc)))
            if len(samples) < 50 * (1 + len({s[0] for s in samples})) and sum(s[0] == name for s in samples) < 50:
                samples.append((name, key, round(len(arr) / 16000, 2), ref, hyp,
                                round(100 * (S + D + I) / max(1, len(rw)), 1)))
            if (i + 1) % 100 == 0:
                print(f"  [{name}] {i+1}/{len(rows)}  WER so far {100*(tot['S']+tot['D']+tot['I'])/max(1,tot['N']):.2f}  "
                      f"{(i+1)/(time.time()-t0):.1f} utt/s", flush=True)
    N = max(1, tot["N"])
    return dict(testset=name, utts=len(rows), words=tot["N"],
                WER=100 * (tot["S"] + tot["D"] + tot["I"]) / N, CER=100 * tot["CE"] / max(1, tot["CN"]),
                SER=100 * (1 - tot["exact"] / max(1, len(rows))),
                sub=100 * tot["S"] / N, dele=100 * tot["D"] / N, ins=100 * tot["I"] / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="", help="default: <run_dir>/best.pt")
    ap.add_argument("--encoder_pt", default="", help="default: the encoder the ckpt was trained with")
    ap.add_argument("--test_glob", default=os.path.join(HERE, "data_kn", "test_*.jsonl"))
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remask", action="store_true",
                    help="after the fill-in decode, re-mask the least-confident committed chars and "
                         "re-predict them (fixes doubled letters; ~2 WER better on a 20-clip dev probe)")
    ap.add_argument("--remask_rounds", type=int, default=4)
    ap.add_argument("--remask_frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--wandb", action="store_true", help="log summary, bar charts and samples to wandb")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "indic_nar_kn_en"))
    ap.add_argument("--wandb_new_run", action="store_true",
                    help="log to a separate decode run instead of attaching to the training run")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    ck = torch.load(ckpt, map_location="cpu")
    cargs = ck.get("args", {}) or {}
    enc = args.encoder_pt or cargs.get("model_checkpoint")
    fps = int(cargs.get("frames_per_sec", 50))
    ck_tag = os.path.splitext(os.path.basename(ckpt))[0]
    rm_tag = f"_remask{args.remask_rounds}" if args.remask else ""
    out_dir = args.out_dir or os.path.join(args.run_dir, f"decode_kn_testsets_{ck_tag}_N{args.steps}{rm_tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt] {ckpt} step={ck.get('step')} epoch={ck.get('epoch')}\n[enc ] {enc}\n[cfg ] fps={fps} "
          f"steps={args.steps} lang={cargs.get('lang')} seed={args.seed}", flush=True)

    # same vocab.json the model was trained with (saved in the ckpt args); fall back to vocab_kn/
    vocab_dir = cargs.get("vocab_dir") or os.path.join(HERE, "vocab_kn")
    if not os.path.exists(os.path.join(vocab_dir, "vocab.json")):
        vocab_dir = os.path.join(HERE, "vocab_kn")
    print(f"[tok ] vocab_dir={vocab_dir}", flush=True)
    tok = build_acmlm_tokenizer_kn(vocab_dir)
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
    assert model.lm_head.out_features == len(tok) - 1, "vocab size differs from the checkpoint's lm_head"
    model.eval()

    results, per_utt, samples = [], [], []
    for path in sorted(glob.glob(args.test_glob)):
        name = os.path.basename(path)[len("test_"):-len(".jsonl")]
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if args.limit:
            rows = rows[:args.limit]
        print(f"\n== {name} ({len(rows)} utts) ==", flush=True)
        m = decode_set(model, tok, specials, fe, rows, fps, args.steps, dev, name, out_dir, args.seed,
                       per_utt, samples, remask=args.remask, remask_rounds=args.remask_rounds,
                       remask_frac=args.remask_frac)
        results.append(m)
        print(f"  [{name}] WER {m['WER']:.2f}  CER {m['CER']:.2f}  SER {m['SER']:.1f}  "
              f"(sub {m['sub']:.1f} / del {m['dele']:.1f} / ins {m['ins']:.1f})", flush=True)

    cols = ["testset", "utts", "words", "WER", "CER", "SER", "sub", "dele", "ins"]
    lines = ["\t".join(cols)] + ["\t".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols)
                                 for r in results]
    with open(os.path.join(out_dir, "SUMMARY.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "per_utt.tsv"), "w") as f:
        f.write("\t".join(PER_UTT_COLS) + "\n")
        for u in per_utt:
            f.write("\t".join(str(u[c]) for c in PER_UTT_COLS) + "\n")
    print("\n" + "\n".join(lines) + f"\n[out] {out_dir}", flush=True)

    if args.wandb:
        log_to_wandb(args, ckpt, ck, enc, ck_tag, cols, results, per_utt, samples)


def log_to_wandb(args, ckpt, ck, enc, ck_tag, cols, results, per_utt, samples):
    """All inference graphs as NATIVE wandb charts/tables. Attaches to the TRAINING run
    (<run_dir>/wandb_run_id.txt written by train.py) so train + dev + test graphs share one page;
    --wandb_new_run (or no id file) logs to a separate decode run in the same project."""
    import wandb
    tag = f"test_{ck_tag}_N{args.steps}" + (f"_remask{args.remask_rounds}" if args.remask else "")               # e.g. test_best_N32 -> several decodes never collide
    id_file = os.path.join(args.run_dir, "wandb_run_id.txt")
    if os.path.exists(id_file) and not args.wandb_new_run:
        project, run_id = open(id_file).read().split()
        run = wandb.init(project=project, id=run_id, resume="must")
        print(f"[wandb] attached to training run {project}/{run_id}", flush=True)
    else:
        run = wandb.init(project=args.wandb_project, job_type="decode",
                         name=f"decode_{os.path.basename(os.path.normpath(args.run_dir))}_{ck_tag}_N{args.steps}")
    run.config.update({f"{tag}/ckpt": ckpt, f"{tag}/ckpt_step": ck.get("step"), f"{tag}/encoder": enc,
                       f"{tag}/decode_steps": args.steps, f"{tag}/seed": args.seed}, allow_val_change=True)
    log = {}

    # 1) WER / CER / SER per test set  + 2) substitution / deletion / insertion breakdown
    summ = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in results])
    log[f"{tag}/summary"] = summ
    for metric, title in [("WER", "WER"), ("CER", "CER"), ("SER", "Sentence error rate"),
                          ("sub", "Substitutions (% ref words)"), ("dele", "Deletions (% ref words)"),
                          ("ins", "Insertions (% ref words)")]:
        log[f"{tag}/{metric}_by_testset"] = wandb.plot.bar(summ, "testset", metric, title=f"{title} - {tag}")
    for r in results:
        for metric in ("WER", "CER", "SER", "sub", "dele", "ins"):
            run.summary[f"{tag}/{r['testset']}/{metric}"] = r[metric]

    # 3) WER by utterance duration (one bar chart per test set + one table)
    buckets = [(0, 3, "<3s"), (3, 6, "3-6s"), (6, 10, "6-10s"), (10, 15, "10-15s"), (15, 1e9, ">15s")]
    dur_rows = []
    for r in results:
        us = [u for u in per_utt if u["testset"] == r["testset"]]
        rows = []
        for lo, hi, bl in buckets:
            b = [u for u in us if lo <= u["duration"] < hi]
            n = sum(u["ref_words"] for u in b)
            if n:
                w = 100 * sum(u["S"] + u["D"] + u["I"] for u in b) / n
                rows.append([f"{bl} (n={len(b)})", w])
                dur_rows.append([r["testset"], bl, len(b), w])
        t = wandb.Table(columns=["duration", "WER"], data=rows)
        log[f"{tag}/{r['testset']}/wer_by_duration"] = wandb.plot.bar(
            t, "duration", "WER", title=f"{r['testset']}: WER by duration")
    log[f"{tag}/wer_by_duration_table"] = wandb.Table(columns=["testset", "bucket", "utts", "WER"], data=dur_rows)

    # 4) per-utterance WER distribution + 5) hyp/ref length ratio (histograms per test set)
    for r in results:
        us = [u for u in per_utt if u["testset"] == r["testset"]]
        uw = [[min(200.0, 100 * (u["S"] + u["D"] + u["I"]) / max(1, u["ref_words"]))] for u in us]
        lr = [[min(2.0, u["hyp_words"] / u["ref_words"])] for u in us if u["ref_words"]]
        log[f"{tag}/{r['testset']}/utterance_wer_hist"] = wandb.plot.histogram(
            wandb.Table(data=uw, columns=["utt_WER"]), "utt_WER", title=f"{r['testset']}: per-utterance WER % (clip 200)")
        log[f"{tag}/{r['testset']}/length_ratio_hist"] = wandb.plot.histogram(
            wandb.Table(data=lr, columns=["hyp_over_ref_words"]), "hyp_over_ref_words",
            title=f"{r['testset']}: hyp/ref word-length ratio (clip 2)")

    # 6) tables: every utterance's scores + the first 50 ref/hyp pairs per test set
    log[f"{tag}/per_utt"] = wandb.Table(columns=PER_UTT_COLS, data=[[u[c] for c in PER_UTT_COLS] for u in per_utt])
    log[f"{tag}/samples"] = wandb.Table(columns=["testset", "key", "dur_s", "ref", "hyp", "utt_WER"],
                                        data=[list(s) for s in samples])
    run.log(log)
    print(f"[wandb] logged {len(log)} test charts/tables under '{tag}/' -> {run.url}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
