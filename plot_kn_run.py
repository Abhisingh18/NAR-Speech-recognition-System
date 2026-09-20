"""Plot EVERYTHING for an A-CMLM run: training + eval curves (parsed from the train log) and
test-set inference results (from one or more decode dirs). Writes PNGs + CSVs (table view).

  python plot_kn_run.py \
      --log logs/train_kn_20260915_XXXXXX.log \
      --decode_dir runs/kn_en_.../decode_kn_testsets_N32 \
      --out_dir runs/kn_en_.../plots

  --log accepts several files (e.g. the original run + a RESUME run); later values win per step.
  --decode_dir accepts several dirs (e.g. best.pt vs latest.pt, N16 vs N32) -> grouped bars.
  Only numpy + matplotlib needed:  /speech/abhishek/miniconda3/envs/slam_llm/bin/python
  --wandb uploads every PNG to a wandb run (needs wandb in the python env).

Outputs in --out_dir:
  train/*.png        train loss, frame acc, grad norm, LR, throughput     (+ training_curves.png grid)
  eval/*.png         WER, CER, SER (one-shot vs iterative), dev loss, content acc, <fill> rate,
                     entropy, distinct tokens                              (+ eval_curves.png grid)
  inference/*.png    per-testset WER/CER/SER, S/D/I error breakdown, WER by duration,
                     per-utterance WER distribution, hyp/ref length ratio
  *.csv              the numbers behind every chart
"""
import os
import re
import csv
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wer_utils import align_counts, edit_distance                            # noqa: E402

# ---- reference palette (dataviz skill, light mode; validated slots 1-3) ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 150,
})

TRAIN_RE = re.compile(r"^ep(\d+) step(\d+)/(\d+) loss=([\d.]+) acc=([\d.]+) lr=([\d.eE+-]+) "
                      r"gnorm=([\d.]+|nan|inf) (\d+) smp/s")
EVAL_RE = re.compile(r"^\[eval @ (\d+)\] WER=([\d.]+) CER=([\d.]+) SER=([\d.]+) loss=([\d.]+) "
                     r"content_acc=([\d.]+) fill_rate=([\d.]+) entropy=([\d.]+) distinct=(\d+)")
ITER_RE = re.compile(r"^\[eval @ (\d+)\] iter(\d+)\s+WER=([\d.]+) CER=([\d.]+) SER=([\d.]+)")
BEST_RE = re.compile(r"\*\* new best (\S+)=([\d.]+)")


# ============================================================================ parsing
def parse_logs(paths):
    train, ev, it = {}, {}, {}
    best, iter_steps, last_eval = [], None, None
    for p in paths:
        for line in open(p, errors="replace"):
            line = line.rstrip("\n")
            if (m := TRAIN_RE.match(line)):
                s = int(m.group(2))
                train[s] = dict(step=s, epoch=int(m.group(1)), loss=float(m.group(4)), acc=float(m.group(5)),
                                lr=float(m.group(6)), gnorm=float(m.group(7)), sps=float(m.group(8)))
            elif (m := EVAL_RE.match(line)):
                s = int(m.group(1)); last_eval = s
                ev[s] = dict(step=s, WER=float(m.group(2)), CER=float(m.group(3)), SER=float(m.group(4)),
                             loss=float(m.group(5)), content_acc=float(m.group(6)), fill_rate=float(m.group(7)),
                             entropy=float(m.group(8)), distinct=int(m.group(9)))
            elif (m := ITER_RE.match(line)):
                s = int(m.group(1)); iter_steps = int(m.group(2))
                it[s] = dict(step=s, iter_WER=float(m.group(3)), iter_CER=float(m.group(4)), iter_SER=float(m.group(5)))
            elif (m := BEST_RE.search(line)) and last_eval is not None:
                best.append((last_eval, m.group(1), float(m.group(2))))
    return ([train[k] for k in sorted(train)], [ev[k] for k in sorted(ev)],
            [it[k] for k in sorted(it)], best, iter_steps)


def ema(v, span):
    if len(v) == 0 or span <= 1:
        return np.asarray(v, float)
    a, out, acc = 2.0 / (span + 1), [], v[0]
    for x in v:
        acc = a * x + (1 - a) * acc
        out.append(acc)
    return np.asarray(out)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ============================================================================ chart helpers
def finish(ax, title, ylabel, xlabel="optimizer step"):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK2, annotation_clip=False)


def raw_and_smoothed(ax, x, y, title, ylabel, logy=False, ref=None):
    """Noisy per-log-step series: raw in muted hairline + EMA in slot 1 (legend: 2 series)."""
    span = max(1, len(y) // 40)
    ax.plot(x, y, color=MUTED, lw=0.7, alpha=0.55, label="raw")
    sm = ema(y, span)
    ax.plot(x, sm, color=SERIES[0], lw=1.6, label=f"EMA (span {span})")
    if len(x):
        end_label(ax, x[-1], sm[-1], f"{sm[-1]:.3g}")
    if ref is not None:
        ax.axhline(ref[0], color=AXIS, lw=0.9)
        ax.annotate(ref[1], (x[0] if len(x) else 0, ref[0]), xytext=(2, 3), textcoords="offset points",
                    fontsize=8, color=MUTED)
    if logy:
        ax.set_yscale("log")
    ax.legend(loc="best")              # matplotlib picks the emptiest corner -> never on top of the curve
    finish(ax, title, ylabel)


def single_line(ax, x, y, title, ylabel, pct=False, logy=False, fmt="{:.3g}"):
    y = np.asarray(y, float) * (100 if pct else 1)
    if len(x) <= 60:      # sparse series (eval points): markers help; dense ones (train log) -> line only
        ax.plot(x, y, color=SERIES[0], lw=1.6, marker="o", ms=3.5, mec=SURFACE, mew=1.2)
    else:
        ax.plot(x, y, color=SERIES[0], lw=1.6)
    if len(x):
        ax.plot([x[-1]], [y[-1]], "o", ms=5, color=SERIES[0], mec=SURFACE, mew=1.2)
        end_label(ax, x[-1], y[-1], fmt.format(y[-1]) + ("%" if pct else ""))
    if logy:
        ax.set_yscale("log")
    finish(ax, title, ylabel)


def clip_pct(ax, arrays, cap=150):
    mx = max((np.nanmax(a) for a in arrays if len(a)), default=0)
    if mx > cap:
        ax.set_ylim(0, cap)
        ax.annotate(f"early values > {cap}% clipped", (0.99, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8, color=MUTED)


def two_lines(ax, x1, y1, lab1, x2, y2, lab2, title, best_step=None):
    """one-shot (slot 1) vs iterative decode (slot 2), both in %; legend + end labels."""
    y1 = np.asarray(y1, float) * 100
    y2 = np.asarray(y2, float) * 100
    ax.plot(x1, y1, color=SERIES[0], lw=1.6, marker="o", ms=3.5, mec=SURFACE, mew=1.2, label=lab1)
    if len(x2):
        ax.plot(x2, y2, color=SERIES[1], lw=1.6, marker="o", ms=3.5, mec=SURFACE, mew=1.2, label=lab2)
    for x, y in ((x1, y1), (x2, y2)):
        if len(x):
            end_label(ax, x[-1], y[-1], f"{y[-1]:.1f}%")
    if best_step is not None and len(x2) and best_step in list(x2):
        i = list(x2).index(best_step)
        ax.plot([best_step], [y2[i]], "o", ms=9, mfc="none", mec=INK, mew=1.2)
        ax.annotate(f"best.pt {y2[i]:.1f}%", (best_step, y2[i]), xytext=(-8, 12), textcoords="offset points",
                    ha="right", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=5))
    clip_pct(ax, [y1, y2])
    ax.legend(loc="best")
    finish(ax, title, "%")


def save_grid(panels, ncols, path, suptitle):
    """panels: list of callables(ax). Saves the grid AND one PNG per panel next to it."""
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k < n:
            panels[k][1](ax)
        else:
            ax.axis("off")
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)
    sub = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0].split("_")[0])
    os.makedirs(sub, exist_ok=True)
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        fn(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(sub, f"{name}.png"))
        plt.close(fig)


# ============================================================================ training + eval
def plot_training(train, ev, it, best, iter_steps, out):
    x = [r["step"] for r in train]
    panels = [
        ("train_loss", lambda ax: raw_and_smoothed(ax, x, [r["loss"] for r in train],
                                                   "Train loss (masked-slot CE, fill-weighted)", "loss")),
        ("train_graded_frame_acc", lambda ax: raw_and_smoothed(ax, x, [100 * r["acc"] for r in train],
                                                               "Train accuracy on masked slots", "%")),
        ("train_grad_norm", lambda ax: raw_and_smoothed(ax, x, [r["gnorm"] for r in train],
                                                        "Gradient norm (pre-clip)", "L2 norm", logy=True,
                                                        ref=(1.0, "clip = 1.0"))),
        ("train_lr", lambda ax: single_line(ax, x, [r["lr"] for r in train], "Learning rate", "lr",
                                            fmt="{:.2e}")),
        ("throughput", lambda ax: single_line(ax, x, [r["sps"] for r in train],
                                              "Throughput (all GPUs)", "samples / s", fmt="{:.0f}")),
        ("epoch", lambda ax: single_line(ax, x, [r["epoch"] for r in train], "Epoch", "epoch", fmt="{:.0f}")),
    ]
    for _, _fn in panels:
        pass
    # single_line draws markers; for dense train series markers are noise -> thin them
    save_grid(panels, 3, os.path.join(out, "training_curves.png"), "Training")

    ex = [r["step"] for r in ev]
    ix = [r["step"] for r in it]
    lab_iter = f"iterative ({iter_steps} steps)" if iter_steps else "iterative"
    best_iter = [b for b in best if "iter" in b[1]]
    best_step = best_iter[-1][0] if best_iter else None
    ep = [
        ("wer", lambda ax: two_lines(ax, ex, [r["WER"] for r in ev], "one-shot", ix, [r["iter_WER"] for r in it],
                                     lab_iter, "Dev WER", best_step)),
        ("cer", lambda ax: two_lines(ax, ex, [r["CER"] for r in ev], "one-shot", ix, [r["iter_CER"] for r in it],
                                     lab_iter, "Dev CER", best_step)),
        ("ser", lambda ax: two_lines(ax, ex, [r["SER"] for r in ev], "one-shot", ix, [r["iter_SER"] for r in it],
                                     lab_iter, "Dev sentence error rate", best_step)),
        ("eval_loss", lambda ax: single_line(ax, ex, [r["loss"] for r in ev],
                                             "Dev loss (one-shot, unweighted CE)", "loss")),
        ("content_frame_acc", lambda ax: single_line(ax, ex, [r["content_acc"] for r in ev],
                                                     "Dev accuracy on character frames", "%", pct=True,
                                                     fmt="{:.1f}")),
        ("fill_pred_rate", lambda ax: single_line(ax, ex, [r["fill_rate"] for r in ev],
                                                  "Share of frames predicted <fill>", "%", pct=True, fmt="{:.1f}")),
        ("pred_entropy", lambda ax: single_line(ax, ex, [r["entropy"] for r in ev],
                                                "Prediction entropy (nats)", "nats")),
        ("distinct_tokens", lambda ax: single_line(ax, ex, [r["distinct"] for r in ev],
                                                   "Distinct non-<fill> tokens predicted", "tokens", fmt="{:.0f}")),
    ]
    if ev:
        save_grid(ep, 3, os.path.join(out, "eval_curves.png"), "Dev evaluation")
    write_csv(os.path.join(out, "train_log.csv"), train)
    merged = {r["step"]: dict(r) for r in ev}
    for r in it:
        merged.setdefault(r["step"], {"step": r["step"]}).update(r)
    write_csv(os.path.join(out, "eval_log.csv"), [merged[k] for k in sorted(merged)])


# ============================================================================ inference
def read_kaldi(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        k, _, t = line.partition(" ")
        d[k] = t.strip()
    return d


def load_durations(decode_dir, test_glob):
    dur = {}
    pu = os.path.join(decode_dir, "per_utt.tsv")
    if os.path.exists(pu):
        for r in csv.DictReader(open(pu), delimiter="\t"):
            dur[r["key"]] = float(r["duration"])
    if not dur and test_glob:
        for p in glob.glob(test_glob):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if r.get("key") is not None and r.get("duration"):
                        dur[r["key"]] = float(r["duration"])
    return dur


def score_decode_dir(ddir, test_glob):
    durs = load_durations(ddir, test_glob)
    sets = {}
    for ref_f in sorted(glob.glob(os.path.join(ddir, "*.ref"))):
        name = os.path.basename(ref_f)[:-4]
        hyp_f = ref_f[:-4] + ".hyp"
        if not os.path.exists(hyp_f):
            continue
        refs, hyps = read_kaldi(ref_f), read_kaldi(hyp_f)
        utts = []
        for k, ref in refs.items():
            hyp = hyps.get(k, "")
            rw, hw = ref.split(), hyp.split()
            S, D, I = align_counts(rw, hw)
            rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
            utts.append(dict(key=k, duration=durs.get(k), ref_words=len(rw), hyp_words=len(hw), S=S, D=D, I=I,
                             char_edits=edit_distance(rc, hc), ref_chars=len(rc), exact=int(ref == hyp)))
        sets[name] = utts
    return sets


def set_metrics(utts):
    N = max(1, sum(u["ref_words"] for u in utts))
    C = max(1, sum(u["ref_chars"] for u in utts))
    S, D, I = (sum(u[k] for u in utts) for k in "SDI")
    return dict(utts=len(utts), words=N, WER=100 * (S + D + I) / N, CER=100 * sum(u["char_edits"] for u in utts) / C,
                SER=100 * (1 - sum(u["exact"] for u in utts) / max(1, len(utts))),
                sub=100 * S / N, dele=100 * D / N, ins=100 * I / N)


def hbar_grouped(ax, names, groups, metric, title):
    """groups: [(label, {set: metrics})]. Horizontal bars, value at the tip, 2px surface gap."""
    n_g = len(groups)
    h = min(0.8 / n_g, 0.34)
    y = np.arange(len(names))
    for gi, (lab, met) in enumerate(groups):
        vals = [met.get(s, {}).get(metric, np.nan) for s in names]
        pos = y - 0.4 + h * (gi + 0.5) + (0.8 - h * n_g) / 2
        ax.barh(pos, vals, height=h, color=SERIES[gi % len(SERIES)], edgecolor=SURFACE, linewidth=1.5, label=lab)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", (v, p), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(1.0, np.nanmax([m.get(metric, 0) for _, g in groups for m in g.values()] or [1])) * 1.15)
    if n_g > 1:
        ax.legend(loc="lower right")
    ax.set_title(title)
    ax.set_xlabel("%")
    ax.tick_params(length=0)


def plot_inference(decode_dirs, labels, test_glob, out):
    os.makedirs(out, exist_ok=True)
    scored = [(lab, score_decode_dir(d, test_glob)) for d, lab in zip(decode_dirs, labels)]
    names = sorted({s for _, sets in scored for s in sets})
    if not names:
        print("[inference] no .ref/.hyp found in", decode_dirs)
        return
    groups = [(lab, {s: set_metrics(u) for s, u in sets.items()}) for lab, sets in scored]
    rows = [dict(decode=lab, testset=s, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()})
            for lab, met in groups for s, m in met.items()]
    write_csv(os.path.join(out, "inference_summary.csv"), rows)

    # 1) WER / CER / SER per test set
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(names) * max(1, len(groups)) + 1.8), squeeze=False)
    for ax, metric, title in zip(axes[0], ["WER", "CER", "SER"],
                                 ["Word error rate", "Character error rate", "Sentence error rate"]):
        hbar_grouped(ax, names, groups, metric, title)
    fig.suptitle("Test sets", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "testset_wer_cer_ser.png"))
    plt.close(fig)

    for lab, sets in scored:
        tag = re.sub(r"[^\w.-]+", "_", lab)
        met = {s: set_metrics(u) for s, u in sets.items()}

        # 2) error breakdown: substitutions / deletions / insertions (stacked, % of ref words)
        fig, ax = plt.subplots(figsize=(8, 0.6 * len(met) + 1.6))
        left = np.zeros(len(met))
        ks = list(met)
        for ci, (key, nm) in enumerate([("sub", "substitutions"), ("dele", "deletions"), ("ins", "insertions")]):
            vals = np.array([met[s][key] for s in ks])
            ax.barh(ks, vals, left=left, height=0.5, color=SERIES[ci], edgecolor=SURFACE, linewidth=2, label=nm)
            for yi, (l, v) in enumerate(zip(left, vals)):
                if v >= 4:
                    ax.text(l + v / 2, yi, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white")
            left += vals
        for yi, tot in enumerate(left):
            ax.annotate(f"WER {tot:.1f}%", (tot, yi), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK2)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, left.max() * 1.25 if len(left) else 1)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)   # below the axis, never on a bar
        ax.set_title(f"Error breakdown - {lab}")
        ax.set_xlabel("% of reference words")
        ax.tick_params(length=0)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"error_breakdown_{tag}.png"))
        plt.close(fig)

        # 3) WER by utterance duration (small multiples, one panel per test set)
        bins = [0, 3, 6, 10, 15, 1e9]
        blabels = ["<3 s", "3-6 s", "6-10 s", "10-15 s", ">15 s"]
        have_dur = any(u["duration"] for us in sets.values() for u in us)
        dur_rows = []
        if have_dur:
            nc = min(2, len(sets))
            nr = (len(sets) + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 3.3 * nr), squeeze=False)
            for ax, (s, us) in zip(axes.flat, sets.items()):
                wers, ns = [], []
                for lo, hi, bl in zip(bins[:-1], bins[1:], blabels):
                    b = [u for u in us if u["duration"] is not None and lo <= u["duration"] < hi]
                    n = sum(u["ref_words"] for u in b)
                    w = 100 * sum(u["S"] + u["D"] + u["I"] for u in b) / n if n else np.nan
                    wers.append(w); ns.append(len(b))
                    dur_rows.append(dict(decode=lab, testset=s, bucket=bl, utts=len(b), WER=round(w, 2) if n else ""))
                xs = np.arange(len(blabels))
                ax.bar(xs, np.nan_to_num(wers), width=0.55, color=SERIES[0])
                for xi, w, n in zip(xs, wers, ns):
                    if not np.isnan(w):
                        ax.annotate(f"{w:.1f}", (xi, w), xytext=(0, 3), textcoords="offset points", ha="center",
                                    fontsize=8.5, color=INK2)
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{bl}\nn={n}" for bl, n in zip(blabels, ns)])
                ax.grid(axis="x", visible=False)
                ax.set_title(f"{s}: WER by duration")
                ax.set_ylabel("WER %")
                ax.tick_params(length=0)
            for ax in list(axes.flat)[len(sets):]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"wer_by_duration_{tag}.png"))
            plt.close(fig)
            write_csv(os.path.join(out, f"wer_by_duration_{tag}.csv"), dur_rows)

        # 4) per-utterance WER distribution + 5) hyp/ref word-length ratio (small multiples)
        nc = min(2, len(sets))
        nr = (len(sets) + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 3.2 * nr), squeeze=False)
        cats = ["0", "0-20", "20-50", "50-100", ">100"]
        for ax, (s, us) in zip(axes.flat, sets.items()):
            w = [100 * (u["S"] + u["D"] + u["I"]) / max(1, u["ref_words"]) for u in us]
            cnt = [sum(v == 0 for v in w), sum(0 < v <= 20 for v in w), sum(20 < v <= 50 for v in w),
                   sum(50 < v <= 100 for v in w), sum(v > 100 for v in w)]
            share = [100 * c / max(1, len(w)) for c in cnt]
            xs = np.arange(len(cats))
            ax.bar(xs, share, width=0.55, color=SERIES[0])
            for xi, v in zip(xs, share):
                ax.annotate(f"{v:.0f}%", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8.5, color=INK2)
            ax.set_xticks(xs)
            ax.set_xticklabels([c + ("" if c == "0" else "") for c in cats])
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{s}: utterances by WER")
            ax.set_xlabel("utterance WER %")
            ax.set_ylabel("% of utterances")
            ax.tick_params(length=0)
        for ax in list(axes.flat)[len(sets):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"utterance_wer_distribution_{tag}.png"))
        plt.close(fig)

        fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 3.2 * nr), squeeze=False)
        for ax, (s, us) in zip(axes.flat, sets.items()):
            r = np.array([u["hyp_words"] / u["ref_words"] for u in us if u["ref_words"] > 0])
            ax.hist(np.clip(r, 0, 2), bins=np.linspace(0, 2, 41), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
            ax.axvline(1.0, color=INK2, lw=1)
            med = float(np.median(r)) if len(r) else float("nan")
            ax.annotate(f"median {med:.2f}", (0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                        fontsize=8.5, color=INK2)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{s}: hypothesis / reference length")
            ax.set_xlabel("hyp words / ref words (clipped at 2)")
            ax.set_ylabel("utterances")
            ax.tick_params(length=0)
        for ax in list(axes.flat)[len(sets):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"length_ratio_{tag}.png"))
        plt.close(fig)

        write_csv(os.path.join(out, f"per_utt_{tag}.csv"),
                  [dict(testset=s, **u) for s, us in sets.items() for u in us])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="*", default=[], help="train log(s) from run_kn.sh / launch_real.sh")
    ap.add_argument("--decode_dir", nargs="*", default=[], help="decode output dir(s) with <set>.ref/<set>.hyp")
    ap.add_argument("--labels", nargs="*", default=None, help="legend label per --decode_dir")
    ap.add_argument("--test_glob", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "data_kn", "test_*.jsonl"),
                    help="jsonl with key+duration (used when a decode dir has no per_utt.tsv)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--wandb", action="store_true", help="upload all PNGs to a wandb run")
    ap.add_argument("--wandb_project", default="indic_nar_kn_en")
    ap.add_argument("--wandb_name", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.log:
        train, ev, it, best, iter_steps = parse_logs(args.log)
        print(f"[log] train points={len(train)} eval points={len(ev)} iter-decode points={len(it)} best marks={len(best)}")
        if train or ev:
            plot_training(train, ev, it, best, iter_steps, args.out_dir)
    if args.decode_dir:
        labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.decode_dir]
        assert len(labels) == len(args.decode_dir), "--labels must match --decode_dir"
        plot_inference(args.decode_dir, labels, args.test_glob, os.path.join(args.out_dir, "inference"))

    pngs = sorted(glob.glob(os.path.join(args.out_dir, "**", "*.png"), recursive=True))
    print(f"[out] {len(pngs)} PNGs + CSVs -> {args.out_dir}")
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, job_type="plots",
                         name=args.wandb_name or f"plots_{os.path.basename(os.path.normpath(args.out_dir))}")
        run.log({os.path.relpath(p, args.out_dir)[:-4]: wandb.Image(p) for p in pngs})
        run.finish()


if __name__ == "__main__":
    main()
