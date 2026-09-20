"""Per-testset decode breakdown for the Hindi A-CMLM iterative decodes, mirroring the
English filler_asr report (WER/CER/SER + word & char S/D/I + <fill>/</s> termination +
error-char profile + worst examples).

Reads the decode dir written by decode_hindi_testsets.py: per set {name}.ref/.hyp/.wer.
Word-level numbers (WER, SER, ins/del/sub) are taken verbatim from akshaya's .wer summary
(same scorer that produced SUMMARY.tsv); char-level CER/S/D/I + profiles are computed here
using akshaya's own normalize_text so tokenization is identical.

Usage:
    python make_testset_report.py --decode_dir runs/.../decode_testsets_N32 [--worst 8]
"""
import os
import re
import sys
import argparse
from collections import Counter

sys.path.insert(0, "/speech/akshaya/OMNI_ASR/decodes")
from indic_normalise_werNFC import normalize_text          # noqa: E402

TESTSETS = ["indictts", "fleurs", "kathbath", "kathbath_noisy", "evaliitm", "commonvoice"]


def read_keyed(path):
    d = {}
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            k, _, rest = ln.partition(" ")
            d[k] = rest
    return d


def norm_str(s, keep_space=True):
    toks = normalize_text(s.split())        # akshaya: NFC + lower + strip P + ws-split
    return " ".join(toks) if keep_space else "".join(toks)


def char_align(ref, hyp):
    """Levenshtein backtrace over characters. Return (S,D,I,ncorr,nref, sub_pairs, dels, inss)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prow = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            c = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prow[j] + 1, row[j - 1] + 1, prow[j - 1] + c)
    i, j = n, m
    S = D = I = C = 0
    sub_pairs, dels, inss = Counter(), Counter(), Counter()
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            C += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; sub_pairs[(ref[i - 1], hyp[j - 1])] += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; dels[ref[i - 1]] += 1; i -= 1
        else:
            I += 1; inss[hyp[j - 1]] += 1; j -= 1
    return S, D, I, C, n, sub_pairs, dels, inss


def parse_wer_summary(wer_path):
    """Return (wer, ins, del, sub, nwords, ser, nsent) from akshaya's .wer trailer."""
    wer = ser = None
    ins = dele = sub = nw = nsent = 0
    with open(wer_path) as f:
        for ln in f:
            if ln.startswith("%WER"):
                wer = float(ln.split()[1])
                m = re.search(r"\[\s*(\d+)\s*/\s*(\d+),\s*(\d+)\s*ins,\s*(\d+)\s*del,\s*(\d+)\s*sub", ln)
                if m:
                    nw = int(m.group(2)); ins = int(m.group(3)); dele = int(m.group(4)); sub = int(m.group(5))
            elif ln.startswith("%SER"):
                ser = float(ln.split()[1])
                m = re.search(r"\[\s*(\d+)\s*/\s*(\d+)", ln)
                if m:
                    nsent = int(m.group(2))
    return wer, ins, dele, sub, nw, ser, nsent


def parse_per_utt(wer_path):
    """Yield (key, nwords, cor, ins, del, sub) from akshaya's per-utt header lines."""
    pat = re.compile(r"^(\S+)\(nwords=(\d+),cor=(\d+),ins=(\d+),del=(\d+),sub=(\d+)\)")
    out = []
    with open(wer_path) as f:
        for ln in f:
            m = pat.match(ln)
            if m:
                out.append((m.group(1), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)), int(m.group(6))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--worst", type=int, default=8)
    args = ap.parse_args()
    dd = args.decode_dir

    rows = []          # aggregate summary rows
    sections = []      # per-set markdown
    tot = dict(nw=0, wins=0, wdel=0, wsub=0, nsent=0, ssent=0,
               cS=0, cD=0, cI=0, cN=0, leak=0, no_eos=0, nutt=0)
    g_sub, g_del = Counter(), Counter()

    for name in TESTSETS:
        ref_p = os.path.join(dd, f"{name}.ref")
        hyp_p = os.path.join(dd, f"{name}.hyp")
        wer_p = os.path.join(dd, f"{name}.wer")
        if not (os.path.exists(ref_p) and os.path.exists(hyp_p) and os.path.exists(wer_p)):
            continue
        refs, hyps = read_keyed(ref_p), read_keyed(hyp_p)
        keys = list(refs.keys())

        # --- word-level (verbatim from akshaya) ---
        wer, wins, wdel, wsub, nw, ser, nsent = parse_wer_summary(wer_p)

        # --- char-level (computed w/ akshaya normalization, spaces kept) ---
        cS = cD = cI = cN = 0
        sub_pairs, dels = Counter(), Counter()
        leak = no_eos = 0
        max_hyp_len = 0
        for k in keys:
            r = norm_str(refs[k]); h = norm_str(hyps[k])
            max_hyp_len = max(max_hyp_len, len(h))
            if "<fill>" in hyps[k]:
                leak += 1
            # </s> never emitted => <fill> tail leaks as literal text (see omni-readout-fill-leak)
            if "<fill>" in hyps[k]:
                no_eos += 1
            S, D, I, C, N, sp, dl, ins = char_align(r, h)
            cS += S; cD += D; cI += I; cN += N
            sub_pairs.update(sp); dels.update(dl)
        cer = 100.0 * (cS + cD + cI) / max(cN, 1)

        # --- worst utterances by word-error rate (min 4 ref words) ---
        per = parse_per_utt(wer_p)
        def uwer(t):  # (sub+del+ins)/nwords
            _, n, cor, i, d, s = t
            return (s + d + i) / max(n, 1)
        worst = sorted([t for t in per if t[1] >= 4], key=uwer, reverse=True)[:args.worst]

        rows.append((name, len(keys), nw, wer, cer, ser, wins, wdel, wsub, cS, cD, cI, leak))
        for kk, v in [("nw", nw), ("wins", wins), ("wdel", wdel), ("wsub", wsub),
                      ("nsent", nsent), ("cS", cS), ("cD", cD), ("cI", cI), ("cN", cN),
                      ("leak", leak), ("no_eos", no_eos), ("nutt", len(keys))]:
            tot[kk] += v
        tot["ssent"] += round(ser / 100.0 * nsent)
        g_sub.update(sub_pairs); g_del.update(dels)

        # per-set section
        def fmt_pairs(c, n=8):
            return ", ".join(f"'{a}'→'{b}':{v}" if isinstance(a, tuple) is False else ""
                             for (a, b), v in c.most_common(n)) if c else "—"
        def fmt_sub(c, n=8):
            items = []
            for (a, b), v in c.most_common(n):
                aa = "␣" if a == " " else a
                bb = "␣" if b == " " else b
                items.append(f"{aa}→{bb}:{v}")
            return ", ".join(items) if items else "—"
        def fmt_del(c, n=8):
            return ", ".join(f"{('␣' if a==' ' else a)}:{v}" for a, v in c.most_common(n)) if c else "—"

        lines = [f"### {name}",
                 f"- utts **{len(keys)}** | words **{nw}** | ref-chars **{cN}**",
                 f"- **WER {wer:.2f}%**  (S {wsub} / D {wdel} / I {wins})   |   "
                 f"**CER {cer:.2f}%**  (S {cS} / D {cD} / I {cI})   |   **SER {ser:.2f}%**",
                 f"- `<fill>`-leak utts: **{leak}** / {len(keys)}  →  `</s>`-termination "
                 f"**{100.0*(len(keys)-no_eos)/len(keys):.2f}%**   (max hyp {max_hyp_len} chars)",
                 f"- top char subs: {fmt_sub(sub_pairs)}",
                 f"- top char dels: {fmt_del(dels)}",
                 f"- worst {len(worst)} utts (word err):"]
        for kkey, n, cor, i, d, s in worst:
            r = refs.get(kkey, ""); h = hyps.get(kkey, "")
            lines.append(f"  - `{kkey}` n={n} S{s}/D{d}/I{i}")
            lines.append(f"    - ref: {r}")
            lines.append(f"    - hyp: {h}")
        sections.append("\n".join(lines))

    # ---- aggregate ----
    ov_wer = 100.0 * (tot["wins"] + tot["wdel"] + tot["wsub"]) / max(tot["nw"], 1)
    ov_cer = 100.0 * (tot["cS"] + tot["cD"] + tot["cI"]) / max(tot["cN"], 1)
    ov_ser = 100.0 * tot["ssent"] / max(tot["nsent"], 1)

    out = []
    out.append("# Hindi A-CMLM iterative decode — per-testset breakdown\n")
    out.append(f"decode_dir: `{dd}`  |  steps=32  |  scorer: akshaya NFC (word) + char-CER here\n")
    out.append("## Summary\n")
    out.append("| testset | utts | words | WER% | CER% | SER% | word S/D/I | char S/D/I | fill-leak |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for (name, nu, nw, wer, cer, ser, wi, wd, ws, cS, cD, cI, leak) in rows:
        out.append(f"| {name} | {nu} | {nw} | {wer:.2f} | {cer:.2f} | {ser:.2f} "
                   f"| {ws}/{wd}/{wi} | {cS}/{cD}/{cI} | {leak} |")
    out.append(f"| **micro-avg** | {tot['nutt']} | {tot['nw']} | **{ov_wer:.2f}** | "
               f"**{ov_cer:.2f}** | **{ov_ser:.2f}** | "
               f"{tot['wsub']}/{tot['wdel']}/{tot['wins']} | {tot['cS']}/{tot['cD']}/{tot['cI']} "
               f"| {tot['leak']} |")
    out.append("")
    out.append("## Fill-prediction / termination health\n")
    out.append(f"- Total `<fill>`-leak utterances (model never emitted `</s>`, so the `<fill>` "
               f"tail rendered as literal text): **{tot['leak']} / {tot['nutt']}**.")
    out.append(f"- `</s>`-termination rate across all sets: "
               f"**{100.0*(tot['nutt']-tot['no_eos'])/max(tot['nutt'],1):.3f}%**.")
    out.append("- (Contrast: the English long-audio A-CMLM leaked on >20 s clips — see "
               "omni-readout-fill-leak. These Hindi test clips are short read/telephone speech "
               "so the readout cut fires on every utterance and CER is NOT leak-inflated.)")
    out.append("")
    out.append("## Global error-char profile (char alignment, all sets)\n")
    def _c(ch): return "␣" if ch == " " else ch
    out.append("- top substitutions (ref→hyp): " +
               ", ".join(f"{_c(a)}→{_c(b)}:{v}" for (a, b), v in g_sub.most_common(15)))
    out.append("- top deletions (ref char dropped): " +
               ", ".join(f"{_c(a)}:{v}" for a, v in g_del.most_common(15)))
    out.append("")
    out.append("## Per-testset detail\n")
    out += sections

    txt = "\n".join(out) + "\n"
    rep = os.path.join(dd, "REPORT.md")
    with open(rep, "w") as f:
        f.write(txt)
    print(txt)
    print(f"[out] {rep}")


if __name__ == "__main__":
    main()
