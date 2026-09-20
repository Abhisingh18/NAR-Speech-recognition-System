"""
Plot eval WER and CER vs training step for a filler_asr run.

Reads the eval history from the latest checkpoint's trainer_state.json
(log_history), which the HF Trainer writes at every save. Re-run anytime to
refresh as training progresses.  WER/CER also stream live to wandb.

Usage:
  python plot_wer_cer.py [--exp experiments/Hubert_SA_finetuning] [--out wer_cer.png]

NOTE: the in-training eval CER is inflated by the batched-eval padding artifact
(model emits chars on padded frames). eval WER is the reliable curve; real CER
comes from single-utterance inference.
"""
import os, glob, json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_history(exp_dir):
    ckpts = glob.glob(os.path.join(exp_dir, "checkpoint-*"))
    if not ckpts:
        return []
    latest = max(ckpts, key=lambda d: int(d.rsplit("-", 1)[-1]))
    state = json.load(open(os.path.join(latest, "trainer_state.json")))
    rows = []
    for h in state.get("log_history", []):
        if "eval_wer" in h:
            rows.append((h.get("step"), h.get("eval_wer"), h.get("eval_cer"), h.get("eval_loss")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="experiments/Hubert_SA_finetuning")
    ap.add_argument("--out", default="wer_cer.png")
    args = ap.parse_args()

    rows = load_eval_history(args.exp)
    if not rows:
        print(f"No eval history yet in {args.exp} (need the first checkpoint/eval). "
              f"WER/CER are also live in wandb.")
        return
    steps = [r[0] for r in rows]
    wer   = [r[1] for r in rows]
    cer   = [r[2] for r in rows]
    print(f"{len(rows)} eval points; latest: step={steps[-1]} WER={wer[-1]:.4f} CER={cer[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(steps, wer, "-o", color="#1f77b4", lw=2, ms=4)
    ax1.set_title("dev WER vs step"); ax1.set_xlabel("step"); ax1.set_ylabel("WER")
    ax1.grid(alpha=0.3); ax1.axhline(min(wer), ls="--", c="#1f77b4", alpha=0.4)
    ax1.annotate(f"best {min(wer):.3f}", (steps[wer.index(min(wer))], min(wer)))

    ax2.plot(steps, cer, "-o", color="#d62728", lw=2, ms=4)
    ax2.set_title("dev CER vs step  (inflated by eval-padding)"); ax2.set_xlabel("step"); ax2.set_ylabel("CER")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{os.path.basename(args.exp)}  —  eval metrics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
