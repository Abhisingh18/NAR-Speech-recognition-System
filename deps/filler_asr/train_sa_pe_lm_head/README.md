# train_sa_pe_lm_head — full-LibriSpeech SA + PE + lm_head pipeline

Trains the **2 self-attention layers + sinusoidal PE + lm_head** (~39.4M params)
on top of a **frozen HuBERT-xlarge** encoder, over the **whole** LibriSpeech 960h
train set, with **no feature caching** — the encoder runs a forward pass every step.

This is the clean, full-data successor to `../train_cached_plots.py`. It keeps the
useful logging (train + eval scalars → charts, plus the qualitative sample table)
and drops the position-bin and token-histogram plots.

## Files
| file | role |
|---|---|
| `train.py` | training loop: full data, forward-every-step, eval, checkpointing, wandb |
| `data.py`  | manifest loader + tiny in-memory dataset; re-exports the lazy collator |
| `run.sh`   | single-GPU launcher with sensible full-data defaults |

The **model, collator, label/mask logic, and loss are imported unchanged** from
`../filler_sa_reference.py` (the verified reference), so the target/masking are
identical to the cached experiments — only data feeding and logging differ.

## What gets logged (trimmed, as requested)
- **train**: `train/loss`, `train/graded_frame_acc`, `train/grad_norm`, `learning_rate`, `throughput/samples_per_s`
- **eval**: `eval/{loss,WER,CER,SER,graded_frame_acc,content_frame_acc,fill_frame_acc,fill_pred_rate,pred_entropy,distinct_nonfill_tokens}`
- **chart**: every scalar above renders as a wandb line chart
- **table**: `samples` — full per-frame tokens | stripped hyp | ref
- **removed**: per-position accuracy bins, predicted/target token histograms

## Run
```bash
cd /speech/tomson/filler_asr/train_sa_pe_lm_head
# foreground:
bash run.sh
# detached (preferred on the shared box):
nohup bash run.sh > train_full960.log 2>&1 &
echo $! > run.pid ; tail -f train_full960.log
```
Override via env: `GPU=6 EPOCHS=8 BATCH=12 ACCUM=4 bash run.sh`.
Encoder choice: `CKPT=/speech/tomson/filler_asr/models/hubert-xlarge-ll60k bash run.sh`.

> GPU is a **PCI-BUS index** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). Pick a free one with
> `nvidia-smi`; mind the +1 physical/CUDA offset noted in the project memory.

## How many epochs is reasonable?
Full train = **281,241 utterances**. At the default `batch_size=16, grad_accum=2`
(effective batch 32) that is **~8,800 optimizer steps per epoch**.

- The 10k-sample probe reached `content_acc≈0.33` after seeing ~96k samples
  (~1,500 steps). **One full epoch already shows ~3× more data than that whole run.**
- A frozen-feature linear-ish head converges fast: expect dev-WER to **plateau
  within ~3–5 epochs**. The cosine schedule (`total_steps = epochs × steps/epoch`)
  is sized to whatever `--epochs` you pass.

**Recommendation: set `--epochs 10` as an upper bound, but watch dev WER and
early-stop on a plateau — 3–5 epochs is the realistic sweet spot.** `best.pt`
always holds the lowest-WER checkpoint, so over-running only wastes compute, not
quality. Use `--max_steps` to hard-cap if you prefer a fixed update budget.

### Cost note
With the 1B encoder in the forward path, throughput is roughly the encoder-forward
rate (~25–35 samples/s on one GPU), so **~2–3 h per epoch single-GPU**. The encoder
tap is `requires_grad=False`, so its activations aren't retained for backward — if
you OOM, lower `--batch_size` (and raise `--grad_accum` to keep the effective batch).

## Checkpoints & resume
Written to `--out_dir` (default `runs/full960_sa_pe_lmhead/`):
- `latest.pt` every `--save_steps`, `best.pt` on dev-WER improvement.
- Head-only (`sa.*` + `lm_head.*`) state dict + optimizer + scheduler + step — ~150 MB, not the full 1.2 GB model.
- Tokenizer is saved into `--out_dir` once for inference.

Resume: `RESUME=runs/full960_sa_pe_lmhead/latest.pt bash run.sh`
(or `python train.py --resume <path> ...`).

### Load a trained head for inference
```python
from filler_sa_reference import build_tokenizer, build_model, FILL_TOKEN
import torch
tok = build_tokenizer("runs/full960_sa_pe_lmhead")            # tokenizer was saved there
model = build_model("/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft",
                    tok, use_sinusoidal_pe=True, device="cuda")
ck = torch.load("runs/full960_sa_pe_lmhead/best.pt", map_location="cpu")
model.load_state_dict(ck["head"], strict=False)              # head keys only; hubert already loaded
model.eval()
```

## Known limitation (unchanged from the reference)
The positional `<fill>` target pins char *i* to frame *i* and the eval decode uses
`group_tokens=False`, so multi-frame phonemes emit repeated characters
(`quilter → qiilter`). That inflates CER/WER even when the transcript is recognizable.
A CTC-style collapse at decode (merge consecutive duplicates, drop blank/`<fill>`)
would cut WER substantially with no retraining — a sensible next step, but out of
scope for this training pipeline.
