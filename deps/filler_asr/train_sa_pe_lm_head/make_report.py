"""Aggregate shard_*.jsonl from decode_report_test_clean.py into REPORT.txt, all_utts.jsonl, gt_vs_pred.txt.

Usage:
    python make_report.py --decode_dir <path to decode output dir> [--steps N] [--tau T] [--temp T] [--ckpt_name NAME]

Reads shard_*.jsonl, sorts by original index, and generates:
    REPORT.txt       : overall WER/CER/SER + first 10 + 10 worst examples
    all_utts.jsonl   : all utterances sorted by original index
    gt_vs_pred.txt   : line-aligned GT / PRED pairs for quick visual inspection
"""
import os
import re
import json
import glob
import argparse


def _lev(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m; ri = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ri == b[j - 1] else 1))
        prev = cur
    return prev[m]


def _sdi(a, b):
    n, m = len(a), len(b); dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    i, j = n, m; S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]: i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1: S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1: D += 1; i -= 1
        else: I += 1; j -= 1
    return S, D, I


_clean = lambda h: " ".join(re.sub(r"(<fill>)+", " ", h).split())   # drop the <fill> readout-leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode_dir", required=True, help="directory containing shard_*.jsonl files")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_name", default="", help="checkpoint name for the report header")
    args = ap.parse_args()

    # ---- load all shards ----
    shard_files = sorted(glob.glob(os.path.join(args.decode_dir, "shard_*.jsonl")))
    if not shard_files:
        print(f"ERROR: no shard_*.jsonl found in {args.decode_dir}")
        return

    utts = []
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if line:
                    utts.append(json.loads(line))

    # sort by original index (if present)
    if utts and "idx" in utts[0]:
        utts.sort(key=lambda u: u["idx"])

    n = len(utts)
    w_err = sum(u["w_err"] for u in utts)
    w_tot = sum(u["w_tot"] for u in utts)
    c_err = sum(u["c_err"] for u in utts)
    c_tot = sum(u["c_tot"] for u in utts)
    s_err = sum(u["s_err"] for u in utts)

    WER = w_err / max(1, w_tot)
    CER = c_err / max(1, c_tot)
    SER = s_err / max(1, n)
    exact = n - s_err

    # ---- count utts with no </s> emitted ----
    no_eos = sum(1 for u in utts if "</s>" not in u.get("full", ""))

    # ---- honest CER (drop the <fill> readout-leak) + word S/D/I on cleaned hyps ----
    ce_c = ct_c = wS = wD = wI = 0
    for u in utts:
        hc = _clean(u.get("hyp", "")); ref = u.get("ref", "")
        ce_c += _lev(ref, hc); ct_c += len(ref)
        s, d, i = _sdi(ref.split(), hc.split()); wS += s; wD += d; wI += i
    CER_clean = ce_c / max(1, ct_c)
    Ework = wS + wD + wI

    # ---- per-utt WER for ranking ----
    for u in utts:
        u["utt_wer"] = u["w_err"] / max(1, u["w_tot"])

    # ---- write all_utts.jsonl ----
    all_utts_path = os.path.join(args.decode_dir, "all_utts.jsonl")
    with open(all_utts_path, "w") as f:
        for u in utts:
            f.write(json.dumps(u) + "\n")
    print(f"[out] {all_utts_path}")

    # ---- write gt_vs_pred.txt ----
    gt_vs_pred_path = os.path.join(args.decode_dir, "gt_vs_pred.txt")
    with open(gt_vs_pred_path, "w") as f:
        for u in utts:
            key = u.get("key", "")
            ref = u.get("ref", "")
            hyp = u.get("hyp", "")
            uwer = u["utt_wer"]
            f.write(f"[{key}]  WER={uwer:.1%}\n")
            f.write(f"  GT   : {ref}\n")
            f.write(f"  PRED : {hyp}\n\n")
    print(f"[out] {gt_vs_pred_path}")

    # ---- write REPORT.txt ----
    ckpt_label = args.ckpt_name or "checkpoint"
    report_path = os.path.join(args.decode_dir, "REPORT.txt")
    with open(report_path, "w") as f:
        f.write(f"A-CMLM iterative decode — LibriSpeech test-clean\n")
        f.write(f"{'='*60}\n")
        f.write(f"checkpoint : {ckpt_label}\n")
        f.write(f"testset    : librispeech test-clean\n")
        f.write(f"utterances : {n}\n")
        f.write(f"decoder    : OmniVoice-style iterative A-CMLM (N={args.steps}, tau={args.tau}, temp={args.temp}, seed={args.seed})\n")
        f.write(f"hyp fix    : cut@first </s>, then drop specials incl <fill> (raw kept as hyp_raw)\n")
        f.write(f"-----------------------------------------------\n")
        f.write(f"WER = {WER:.2%}   ({w_err}/{w_tot} words)\n")
        f.write(f"CER = {CER:.2%}   ({c_err}/{c_tot} chars)   [raw]\n")
        f.write(f"CER = {CER_clean:.2%}   ({ce_c}/{ct_c} chars)   [<fill>-leak cleaned]\n")
        f.write(f"SER = {SER:.2%}   ({s_err}/{n} sentences)\n")
        f.write(f"word S/D/I  : Sub {wS/max(1,w_tot):.2%}  Del {wD/max(1,w_tot):.2%}  Ins {wI/max(1,w_tot):.2%}"
                f"  (of errors {wS/max(1,Ework)*100:.0f}/{wD/max(1,Ework)*100:.0f}/{wI/max(1,Ework)*100:.0f})\n")
        f.write(f"Exact-match : {exact}/{n} ({exact/n:.1%})\n")
        f.write(f"utts with no </s> emitted : {no_eos}\n")

        # ---- first 10 ----
        f.write(f"\nFirst 10 (raw-frame FULL + stripped PRED vs GT)\n")
        f.write(f"{'='*60}\n")
        for i, u in enumerate(utts[:10]):
            key = u.get("key", f"utt{i}")
            dur = u.get("dur", 0)
            T = u.get("T", 0)
            nk = u.get("n_keep", 0)
            uwer = u["utt_wer"]
            f.write(f"\n[{i}] {key}  dur={dur:.2f}s T={T} n_keep={nk} uWER={uwer:.1%}\n")
            f.write(f"  GT   : {u.get('ref', '')}\n")
            f.write(f"  PRED : {u.get('hyp', '')}\n")
            full = u.get("full", "")
            if full:
                f.write(f"  FULL : {full}\n")

        # ---- 10 worst (>= 5 words) ----
        long_utts = [u for u in utts if u["w_tot"] >= 5]
        long_utts.sort(key=lambda u: -u["utt_wer"])
        f.write(f"\n\n10 WORST (>=5 words)\n")
        f.write(f"{'='*60}\n")
        for u in long_utts[:10]:
            idx = u.get("idx", "?")
            key = u.get("key", "")
            dur = u.get("dur", 0)
            T = u.get("T", 0)
            nk = u.get("n_keep", 0)
            uwer = u["utt_wer"]
            f.write(f"\n[{idx}] {key}  dur={dur:.2f}s T={T} n_keep={nk} uWER={uwer:.1%}\n")
            f.write(f"  GT   : {u.get('ref', '')}\n")
            f.write(f"  PRED : {u.get('hyp', '')}\n")
            full = u.get("full", "")
            if full:
                f.write(f"  FULL : {full}\n")

    print(f"[out] {report_path}")
    print(f"\n[summary] n={n}  WER={WER:.4f}  CER_raw={CER:.4f}  CER_clean={CER_clean:.4f}  SER={SER:.4f}  "
          f"S/D/I={wS/max(1,Ework)*100:.0f}/{wD/max(1,Ework)*100:.0f}/{wI/max(1,Ework)*100:.0f}")


if __name__ == "__main__":
    main()
