"""Figures for the A-CMLM pipeline PDF: masking (uniform+Bernoulli) and the decode schedule."""
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- masking figure
def masking_fig():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    rng = np.random.default_rng(0)

    # (a) three Bernoulli draws over n=8 slots at different p
    ax = axs[0]
    ps = [0.25, 0.9, 1.0]
    labels = ["p=0.25 (uniform draw)", "p=0.90 (uniform draw)", "p=1.00 (forced 15%)"]
    n = 8
    for r, (p, lab) in enumerate(zip(ps, labels)):
        coins = rng.random(n)
        masked = coins < p
        for c in range(n):
            fc = "#c62828" if masked[c] else "#e8eef2"
            ax.add_patch(plt.Rectangle((c, -r), 0.9, 0.8, fc=fc, ec="#455a64"))
            ax.text(c + 0.45, -r + 0.4, "M" if masked[c] else "·", ha="center", va="center",
                    fontsize=9, color="white" if masked[c] else "#455a64")
        ax.text(-0.3, -r + 0.4, lab, ha="right", va="center", fontsize=8)
        ax.text(n + 0.2, -r + 0.4, f"{masked.sum()} masked", ha="left", va="center", fontsize=8, color="#c62828")
    ax.set_xlim(-4.2, n + 2.5); ax.set_ylim(-2.6, 1.1); ax.axis("off")
    ax.set_title("(a) one Bernoulli(p) coin per slot  ->  masked where uniform < p", fontsize=9)

    # (b) E[#masked]=n*p and the marginal 1/2 from p~U(0,1)
    ax = axs[1]
    p = np.linspace(0, 1, 100)
    ax.plot(p, n * p, color="#1565c0", lw=2, label="E[#masked | p] = n·p")
    ax.axhline(n * 0.5, color="#2e7d32", ls="--", lw=1.5, label="avg over p~U(0,1): each slot masked ½")
    ax.fill_between(p, n * p - np.sqrt(n * p * (1 - p)), n * p + np.sqrt(n * p * (1 - p)),
                    color="#1565c0", alpha=0.12, label="± std  √(n·p(1−p))")
    ax.set_xlabel("masking rate  p"); ax.set_ylabel("# masked (of n=8)")
    ax.set_title("(b) count is Binomial(n,p): mean n·p, spread √(n·p(1−p))", fontsize=9)
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(HERE, "acmlm_masking.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); print("wrote", out)


# ---------------------------------------------------------------- schedule figure
def schedule_fig():
    N, tau, M = 32, 0.1, 100
    n = np.arange(0, N + 1)
    def r(x, t): return (t * x) / (1 + (t - 1) * x)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))

    ax = axs[0]
    ax.plot(n / N, r(n / N, 0.1), color="#6a1b9a", lw=2, marker="o", ms=3, label="τ=0.1 (back-loaded)")
    ax.plot(n / N, r(n / N, 1.0), color="#9e9e9e", lw=1.5, ls="--", label="τ=1 (linear)")
    ax.set_xlabel("step fraction  n/N"); ax.set_ylabel("cumulative fraction unmasked  rₙ")
    ax.set_title("(a) schedule rₙ = τ(n/N)/(1+(τ−1)(n/N))", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axs[1]
    cum = np.round(r(n / N, tau) * M).astype(int)
    k = np.diff(cum); k[-1] += M - cum[-1] if cum[-1] < M else 0
    ax.bar(np.arange(1, N + 1), k, color="#8e24aa")
    ax.set_xlabel("decode step n"); ax.set_ylabel("# NEW slots committed  kₙ")
    ax.set_title(f"(b) commits/step (M={M}): few early, avalanche at the end", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = os.path.join(HERE, "acmlm_schedule.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); print("wrote", out)


if __name__ == "__main__":
    masking_fig()
    schedule_fig()
