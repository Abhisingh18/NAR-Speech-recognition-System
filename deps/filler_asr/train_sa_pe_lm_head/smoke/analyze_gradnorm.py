#!/usr/bin/env python3
"""Analyse the grad-norm trace from a smoke_gradnorm.sh log.

The training print line (train.py) is:
    ep{e} step{s}/{tot} loss={..} acc={..} lr={..} gnorm={..} {sps} smp/s
`gnorm` is the total grad norm BEFORE clipping (what clip_grad_norm_ returns), so
gnorm > clip  <=>  clipping was active on that step.

Question we are answering (from the autopsy note):
  Does gnorm sit PINNED at the clip threshold for a long stretch (clip binding,
  doing more work than the LR schedule), or does it fall below clip after the
  initial transient (schedule in control)?

Usage:  python analyze_gradnorm.py <log> [--clip 1.0] [--warmup 30]
Safe to run on a partial (still-growing) log; re-run to refresh.
"""
import argparse, math, re, sys

LINE = re.compile(
    r"step(\d+)/\d+\s+loss=(\S+)\s+acc=(\S+)\s+lr=(\S+)\s+gnorm=(\S+)")


def fnum(x):
    """float, tolerating nan/inf tokens; None if unparseable."""
    try:
        return float(x)
    except ValueError:
        xl = x.lower()
        if "nan" in xl:
            return float("nan")
        if "inf" in xl:
            return float("inf")
        return None


def pct(vals, q):
    if not vals:
        return float("nan")
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round(q / 100.0 * (len(v) - 1)))))
    return v[i]


def finite(vals):
    return [v for v in vals if v is not None and math.isfinite(v)]


def summarize(name, rows, clip):
    g = [r["gnorm"] for r in rows]
    gf = finite(g)
    n = len(rows)
    n_nan = sum(1 for v in g if v is None or not math.isfinite(v))
    n_clip = sum(1 for v in gf if v > clip)
    frac = (100.0 * n_clip / len(gf)) if gf else float("nan")
    med = pct(gf, 50)
    p90 = pct(gf, 90)
    mx = max(gf) if gf else float("nan")
    mn = min(gf) if gf else float("nan")
    print(f"  {name:<22} n={n:<5} median={med:6.3f}  p90={p90:6.3f}  "
          f"min={mn:6.3f}  max={mx:8.3f}  clipped(>{clip:g})={frac:5.1f}%"
          + (f"  NON-FINITE={n_nan}" if n_nan else ""))
    return med, frac


def bar(v, vmax, width=32):
    if not math.isfinite(v) or vmax <= 0:
        return "?" * 4
    return "#" * max(1, int(round(width * v / vmax)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=30,
                    help="steps <= warmup are treated as the LR-ramp phase")
    ap.add_argument("--bin", type=int, default=10, help="trajectory bin width (steps)")
    a = ap.parse_args()

    rows = []
    try:
        with open(a.log) as f:
            for ln in f:
                m = LINE.search(ln)
                if not m:
                    continue
                step, loss, acc, lr, gn = m.groups()
                rows.append(dict(step=int(step), loss=fnum(loss), acc=fnum(acc),
                                 lr=fnum(lr), gnorm=fnum(gn)))
    except FileNotFoundError:
        sys.exit(f"no such log: {a.log}")

    if not rows:
        print(f"no 'step.. gnorm=' lines yet in {a.log}")
        print("(model may still be building / loading data -- try again shortly)")
        return
    rows.sort(key=lambda r: r["step"])

    clip = a.clip
    warm = [r for r in rows if r["step"] <= a.warmup]
    post = [r for r in rows if r["step"] > a.warmup]

    print(f"\n=== grad-norm smoke analysis ===")
    print(f"log   : {a.log}")
    print(f"clip  : {clip:g}    warmup(assumed) : {a.warmup}    "
          f"steps logged : {rows[0]['step']}..{rows[-1]['step']}  ({len(rows)} pts)\n")

    # NaN gate -- the smoke doubles as a sanity check that text_embed init is sound.
    bad = [r for r in rows if r["gnorm"] is None or not math.isfinite(r["gnorm"])
           or (r["loss"] is not None and not math.isfinite(r["loss"]))]
    if bad:
        print(f"!! NON-FINITE loss/gnorm at {len(bad)} step(s); first = step {bad[0]['step']}")
        print("!! that is the NaN failure class, not a clipping question -- stop and fix init.\n")

    summarize("overall", rows, clip)
    if warm:
        summarize(f"warmup (<= {a.warmup})", warm, clip)
    if post:
        med_post, frac_post = summarize(f"post-warmup (> {a.warmup})", post, clip)
    else:
        med_post = frac_post = float("nan")
        print("  (no post-warmup steps yet)")

    # when does gnorm first drop below clip, and does it STAY there?
    first_below = next((r["step"] for r in rows if r["gnorm"] is not None
                        and math.isfinite(r["gnorm"]) and r["gnorm"] < clip), None)
    sustained = None
    W = 10
    for i in range(len(rows) - W + 1):
        win = rows[i:i + W]
        if all(r["gnorm"] is not None and math.isfinite(r["gnorm"]) and r["gnorm"] < clip
               for r in win):
            sustained = win[0]["step"]
            break
    print()
    print(f"  first step with gnorm < clip        : "
          f"{first_below if first_below is not None else 'never'}")
    print(f"  first run of {W} consecutive < clip   : "
          f"{sustained if sustained is not None else 'never (still binding)'}")

    # trajectory: mean gnorm per bin, so pinning is visible at a glance
    print(f"\n  trajectory (mean gnorm per {a.bin} steps):")
    binned = {}
    for r in rows:
        gf = r["gnorm"]
        if gf is None or not math.isfinite(gf):
            continue
        b = (r["step"] - 1) // a.bin
        binned.setdefault(b, []).append(gf)
    means = {b: sum(v) / len(v) for b, v in binned.items()}
    vmax = max(means.values()) if means else 1.0
    for b in sorted(means):
        lo, hi = b * a.bin + 1, (b + 1) * a.bin
        mean = means[b]
        flag = "  <-- <clip" if mean < clip else ""
        print(f"    {lo:4d}-{hi:<4d} | {mean:7.3f} {bar(mean, vmax)}{flag}")

    # verdict
    print("\n  verdict:")
    if bad:
        print("    [NaN]  non-finite values present -- this is not a clipping story;")
        print("           the run is diverging/uninitialised. Fix before reading gnorm.")
    elif not post:
        print("    [WAIT] no post-warmup steps yet; let it run further, then re-analyze.")
    elif sustained is not None:
        print(f"    [OK]   gnorm settles below clip (sustained from step {sustained}).")
        print("           Clipping is NOT the binding constraint -- the LR schedule is in")
        print("           control. clip=1.0 is fine; nothing to change.")
    elif frac_post >= 80 and med_post > clip:
        print(f"    [BINDING] post-warmup median gnorm={med_post:.2f} > clip={clip:g}, and")
        print(f"           {frac_post:.0f}% of post-warmup steps are clipped. The clip is the")
        print("           active constraint, not the schedule. With Adam the direction still")
        print("           survives, but if this persists deep into the real run consider a")
        print(f"           looser clip (retry: CLIP=5.0 bash smoke_gradnorm.sh) or accept it.")
    else:
        print(f"    [MIXED] post-warmup median gnorm={med_post:.2f}, {frac_post:.0f}% clipped.")
        print("           Partly binding; watch the real run's first few hundred steps to see")
        print("           whether the trajectory keeps falling below clip.")
    print()


if __name__ == "__main__":
    main()
