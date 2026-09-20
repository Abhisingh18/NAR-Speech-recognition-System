"""By-duration structural analysis of an iterative decode dump (decode_iterative.py output).

For each clip parses the per-frame RAW prediction and measures, per duration bin:
    got/exp </s>  : first predicted </s> frame  /  expected position (1 + transcript length)
                    ==1.0 -> </s> lands where the transcript ends; <1 -> fires early (deletions)
    tail-leak %   : fraction of post-</s> frames that are NON-<fill> (content leaking into the tail)
    fill-in-body  : # of <fill> frames BEFORE </s> (mid-sentence gaps; the FILL_INTRUDE mode)
    letters       : # of real content tokens emitted in the body (0 => collapsed to |/specials)
    no-</s>       : clips that never emit </s> (the NO_EOS mode)
    WER %         : word-level, hyp vs ref
Plus corr(expected,got) and the error-mass split (%words / %errors per bin).

Usage: python analyze_eos_by_duration.py --decode_dir <dir with *.decodes.txt> [--bins 3,6,10,15]
"""
import os, re, glob, argparse
import numpy as np


def load(decode_dir):
    f = glob.glob(os.path.join(decode_dir, "*.decodes.txt"))
    assert f, f"no *.decodes.txt in {decode_dir}"
    recs = []; cur = {}
    for line in open(f[0]):
        m = re.match(r"^====.* (\S+)  dur=([\d.]+)s  T=(\d+)  n_keep=(\d+)", line)
        if m:
            if cur: recs.append(cur)
            cur = {"key": m.group(1), "dur": float(m.group(2)), "T": int(m.group(3)), "nk": int(m.group(4))}
        elif line.startswith("RAW"): cur["raw"] = line.split(":", 1)[1].split()
        elif line.startswith("REF"): cur["ref"] = line.split(":", 1)[1].strip()
        elif line.startswith("HYP"): cur["hyp"] = line.split(":", 1)[1].strip()
    if cur: recs.append(cur)
    return recs, os.path.basename(f[0])


def lev(a, b):
    n, m = len(a), len(b); dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return dp[m]


def analyze(r):
    toks = r.get("raw", [])
    eos = next((i for i, t in enumerate(toks) if t == "</s>"), None)
    exp = 1 + len(re.sub(r"[^a-z' ]", "", r["ref"].lower()).replace(" ", "|"))
    body_end = eos if eos is not None else len(toks)
    nonfill_after = sum(1 for t in toks[body_end:] if t not in ("<fill>", "</s>"))
    letters = sum(1 for t in toks[:body_end] if t not in ("<s>", "</s>", "<fill>", "|"))
    fill_in_body = sum(1 for t in toks[:body_end] if t == "<fill>")
    return eos, exp, nonfill_after, letters, fill_in_body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--bins", default="3,6,10,15", help="duration cut points (s), comma-sep")
    args = ap.parse_args()
    recs, fname = load(args.decode_dir)
    for r in recs: r["a"] = analyze(r)
    cuts = [0] + [float(x) for x in args.bins.split(",")] + [1e9]

    print(f"\n=== by-duration structural report : {args.decode_dir} ({fname}, n={len(recs)}) ===")
    print(f"{'dur bin':>9} {'#clips':>6} {'avgT':>5} {'exp@':>5} {'got@':>5} {'got/exp':>7} "
          f"{'tailleak%':>9} {'fillbody':>8} {'letters':>7} {'noeos':>5} {'WER%':>6}")
    tot_w = tot_e = 0; masses = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        g = [r for r in recs if lo <= r["dur"] < hi]
        if not g: continue
        eos_got = [r["a"][0] for r in g if r["a"][0] is not None]
        ratios = [r["a"][0] / r["a"][1] for r in g if r["a"][0] is not None and r["a"][1] > 0]
        leak = [100 * r["a"][2] / max(1, (r["T"] - (r["a"][0] or r["T"]))) for r in g if r["a"][0] is not None]
        w = sum(len(r["ref"].split()) for r in g if "hyp" in r)
        e = sum(lev(r["ref"].split(), r["hyp"].split()) for r in g if "hyp" in r)
        tot_w += w; tot_e += e; masses.append((f"{lo:g}-{hi if hi < 1e8 else ''}", w, e))
        name = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        print(f"{name:>9} {len(g):>6} {np.mean([r['T'] for r in g]):>5.0f} "
              f"{np.mean([r['a'][1] for r in g]):>5.0f} {np.mean(eos_got) if eos_got else float('nan'):>5.0f} "
              f"{np.mean(ratios) if ratios else float('nan'):>7.2f} {np.mean(leak) if leak else 0:>9.2f} "
              f"{np.mean([r['a'][4] for r in g]):>8.2f} {np.mean([r['a'][3] for r in g]):>7.1f} "
              f"{sum(1 for r in g if r['a'][0] is None):>5} {100*e/max(1,w):>6.2f}")
    xs = [r["a"][1] for r in recs if r["a"][0] is not None]
    ys = [r["a"][0] for r in recs if r["a"][0] is not None]
    print(f"\ncorr(expected,got </s>) = {np.corrcoef(xs, ys)[0,1]:.3f}   "
          f"clips-with-</s> = {len(ys)}/{len(recs)}   mean letters = {np.mean([r['a'][3] for r in recs]):.1f}")
    print(f"overall WER = {100*tot_e/max(1,tot_w):.2f}%   (words={tot_w}, errs={tot_e})")
    print("error mass:  " + " | ".join(f"{n}s: {100*w/max(1,tot_w):.0f}%w {100*e/max(1,tot_e):.0f}%err" for n, w, e in masses))


if __name__ == "__main__":
    main()
