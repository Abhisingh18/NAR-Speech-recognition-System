Subject: Hindi NAR-ASR (data2vec-aqc) — it's training; docs + one review ask

Hi Akshaya, Abhishek,

Quick share on the Hindi non-autoregressive ASR I've been putting together. It's the filler_asr
A-CMLM recipe (a FROZEN speech encoder + a small 8-block self-attention character head, decoded by
iterative mask-predict), but with the encoder swapped from the English HuBERT over to the
**data2vec-aqc CTC Hindi model** — I tap it *before* the CTC head and only train the ~100M head on
top. It's running now and looks healthy: dev iter-WER is down to ~0.68 and still falling (~29 of
150 epochs).

Three docs (attached) — best read in this order:
  • RUNBOOK.md            — step-by-step to actually run it (env → data → train), with diagrams.
  • PORTABLE_SETUP.md     — how to set it up under YOUR user / a common space, and exactly how the
                            data must look (jsonl manifest, 16 kHz mono wav). Read this if you want
                            to run it yourselves.
  • OMNI_DECODE_EXPLAINED.md — how the mask-predict ("omni-style") decoding works, walked through a
                            real, error-free example. The conceptual one.
  (paths: /speech/tomson/indic-nar-filler_asr/{RUNBOOK,PORTABLE_SETUP,OMNI_DECODE_EXPLAINED}.md)

Where everything lives:
  • conda env : indic-nar-filler_asr  at  /speech/tomson/miniconda3/envs/indic-nar-filler_asr
                (clone of smear-moe-gemma3: fairseq 0.12 + transformers 5.5 + torch 2.4).
                Use it by name, or clone it into a shared prefix — see PORTABLE_SETUP.md §1.
  • all scripts for this run :  /speech/tomson/indic-nar-filler_asr/
                (train.py, run.sh, launch_real.sh, model_d2v_acmlm.py, text_norm_hi.py, plus the
                 tap-cache scripts precompute_taps_d2v.py / data_tapcached_d2v.py / extract_taps_d2v.sh)
  • current run log + checkpoints :
                /speech/tomson/indic-nar-filler_asr/logs/train_ctc_20260910_001508.log
                /speech/tomson/indic-nar-filler_asr/runs/hi_d2vaqcCTC_sa8_ACMLM_alibi_fps50_fw003_exp150/

One thing I'd genuinely like a second pair of eyes on — the **best-checkpoint saving logic**. It's
in train.py: `save_ckpt()` / `head_state_dict()` (saves the ~100M trainable head plus optimizer +
scheduler state) and the selection inside `log_eval()` (best.pt = argmin dev iter-WER, which re-runs
a 200-clip iterative decode at every eval). I honestly haven't looked much into this — whether the
selection metric, the eval cost, or what/how we actually save is the right call — so if you could
look into optimising it, that would be really helpful.

Happy to walk through any of it.

Thanks,
Tomson
