"""Precompute the frozen HuBERT tap for every clip ONCE -> ragged fp16 memmap.

Valid because the encoder is pinned to eval (deterministic tap). Reusable across ALL downstream
hyperparameters (fps, mask, eos_repeat, fill_weight, LR, epochs, #SA layers); only invalidated if
the HuBERT checkpoint / feature extractor changes.

Layout written to --cache_dir:
    index.npy       (N,2) int64  [row_offset, T_i]
    manifest.jsonl  N lines {"text","T"} aligned to index
    taps.f16        memmap (sum T_i, 1280) float16
    PREPARED        marker (bytes written) once complete

Single-GPU:  python precompute_taps.py --cache_dir <dir> [--limit N]
Multi-GPU :  torchrun --nproc_per_node=4 precompute_taps.py --cache_dir <dir>
             (ranks write DISJOINT clip ranges of the same memmap; index/manifest by rank 0)
"""
import os
import sys
import json
import math
import argparse

import numpy as np
import torch
import soundfile as sf

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import (SAMPLING_RATE, compute_output_length,        # noqa: E402
                                 build_acmlm_tokenizer, build_acmlm_model)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_rows                                                    # noqa: E402
from transformers import Wav2Vec2FeatureExtractor                            # noqa: E402
from data_tapcached import cache_paths, TAP_DIM                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_train_960h.jsonl")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--model_checkpoint", default="/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft")
    ap.add_argument("--vocab_dir", default="/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N clips (smoke)")
    ap.add_argument("--dim", type=int, default=TAP_DIM)
    args = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dev = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    is_main = (rank == 0)

    os.makedirs(args.cache_dir, exist_ok=True)
    taps_f, index_f, man_f = cache_paths(args.cache_dir)

    # ---- pass 1 (rank 0): lengths from headers (no decode) -> index + manifest + allocate memmap ----
    rows = load_rows(args.manifest, args.limit or None)               # [{audio_path, text}]
    Ts = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        n = sf.info(r["audio_path"]).frames
        Ts[i] = compute_output_length(n)
    offsets = np.zeros(len(rows), dtype=np.int64)
    offsets[1:] = np.cumsum(Ts)[:-1]
    total = int(Ts.sum())
    index = np.stack([offsets, Ts], axis=1)                          # (N,2)

    if is_main:
        np.save(index_f, index)
        with open(man_f, "w") as f:
            for r, T in zip(rows, Ts):
                f.write(json.dumps({"text": r["text"], "T": int(T)}) + "\n")
        # create/truncate the memmap file to the right size
        np.memmap(taps_f, dtype=np.float16, mode="w+", shape=(total, args.dim)).flush()
        print(f"[precompute] N={len(rows)} total_frames={total} "
              f"size={total*args.dim*2/1e9:.1f} GB -> {taps_f}", flush=True)
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier()

    # ---- model + feature extractor (encoder only used) ----
    tok = build_acmlm_tokenizer(args.vocab_dir)
    model = build_acmlm_model(args.model_checkpoint, tok, device=dev).eval()
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLING_RATE, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)

    taps = np.memmap(taps_f, dtype=np.float16, mode="r+", shape=(total, args.dim))

    # ---- pass 2: this rank fills its disjoint clip range ----
    my = list(range(rank, len(rows), world)) if ddp else list(range(len(rows)))
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available() else torch.autocast(device_type="cpu", enabled=False))
    done = 0
    for s in range(0, len(my), args.batch_size):
        idxs = my[s:s + args.batch_size]
        feats = []
        for i in idxs:
            arr, sr = sf.read(rows[i]["audio_path"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            feats.append({"input_values": fe(arr, sampling_rate=SAMPLING_RATE).input_values[0]})
        batch = fe.pad(feats, padding=True, return_tensors="pt")
        iv = batch["input_values"].to(dev)
        am = batch["attention_mask"].to(dev)
        with torch.no_grad(), autocast:
            hs, feat_len = model.hubert_tap(iv, am)                   # (B,Tmax,1280), (B,)
        hs = hs.float().cpu().numpy()
        for b, i in enumerate(idxs):
            T = int(index[i, 1]); off = int(index[i, 0])
            fl = int(feat_len[b]) if feat_len is not None else T
            assert fl >= T, f"HuBERT frames {fl} < index T {T} for clip {i}"
            taps[off:off + T] = hs[b, :T].astype(np.float16)
        done += len(idxs)
        if is_main and (s // args.batch_size) % 50 == 0:
            print(f"[precompute rank0] {done}/{len(my)} clips", flush=True)
    taps.flush()

    if ddp:
        torch.distributed.barrier()
    if is_main:
        with open(os.path.join(args.cache_dir, "PREPARED"), "w") as f:
            f.write(f"{total} {args.dim}\n")
        print("[precompute] DONE", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
