"""Precompute the FROZEN data2vec-aqc tap for every clip ONCE -> ragged fp16 memmap.

data2vec twin of filler_asr/train_sa_pe_lm_head/precompute_taps.py. Valid because the encoder is
frozen + eval-pinned (deterministic tap). The tap is EXACTLY what the live path feeds the SA head
(model.hubert_tap == data2vec extract_features on the do_normalize'd waveform), so cached and live
training are numerically identical downstream.

Layout written to --cache_dir  (identical to the HuBERT cache, only dim changes 1280->1024):
    index.npy       (N,2) int64  [row_offset, T_i]
    manifest.jsonl  N lines {"text","T"} aligned to index   (text = RAW; normalize_hi at train time)
    taps.f16        memmap (sum T_i, 1024) float16
    PREPARED        marker "total dim" once complete

Single :  python precompute_taps_d2v.py --cache_dir <dir> --manifest <jsonl> [--limit N] [--device cpu]
Multi  :  torchrun --nproc_per_node=4 precompute_taps_d2v.py --cache_dir <dir> --manifest <jsonl>
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FILLER_ROOT = os.environ.get("FILLER_ROOT", "/speech/tomson/filler_asr")
sys.path.insert(0, FILLER_ROOT)
sys.path.insert(0, os.path.join(FILLER_ROOT, "train_sa_pe_lm_head"))

from filler_sa_reference import SAMPLING_RATE, compute_output_length                # noqa: E402
from data import load_rows                                                          # noqa: E402
from data_tapcached import cache_paths                                              # noqa: E402
from transformers import Wav2Vec2FeatureExtractor                                   # noqa: E402
from model_d2v_acmlm import build_d2v_acmlm, D2V_CKPT, D2V_DIM                       # noqa: E402
from text_norm_hi import build_acmlm_tokenizer_hi                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--encoder_pt", default=os.environ.get("ENCODER_PT", D2V_CKPT))
    ap.add_argument("--vocab_dir", default=os.path.join(HERE, "vocab_hi"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N clips (smoke)")
    ap.add_argument("--dim", type=int, default=D2V_DIM)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dev = args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and dev.startswith("cuda"):
        torch.cuda.set_device(local_rank)
    is_main = (rank == 0)

    os.makedirs(args.cache_dir, exist_ok=True)
    taps_f, index_f, man_f = cache_paths(args.cache_dir)

    # ---- pass 1 (rank 0): lengths from headers (no decode) -> index + manifest + allocate memmap ----
    rows = load_rows(args.manifest, args.limit or None)
    Ts = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        Ts[i] = compute_output_length(sf.info(r["audio_path"]).frames)
    offsets = np.zeros(len(rows), dtype=np.int64)
    offsets[1:] = np.cumsum(Ts)[:-1]
    total = int(Ts.sum())
    index = np.stack([offsets, Ts], axis=1)

    if is_main:
        np.save(index_f, index)
        with open(man_f, "w") as f:
            for r, T in zip(rows, Ts):
                f.write(json.dumps({"text": r["text"], "T": int(T)}, ensure_ascii=False) + "\n")
        np.memmap(taps_f, dtype=np.float16, mode="w+", shape=(total, args.dim)).flush()
        print(f"[precompute-d2v] N={len(rows)} total_frames={total} dim={args.dim} "
              f"size={total*args.dim*2/1e9:.2f} GB -> {taps_f}", flush=True)
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier()

    # ---- model (encoder only used via hubert_tap) + feature extractor ----
    tok = build_acmlm_tokenizer_hi(args.vocab_dir)
    model = build_d2v_acmlm(tok, d2v_ckpt=args.encoder_pt, device=dev).eval()
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLING_RATE, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)

    taps = np.memmap(taps_f, dtype=np.float16, mode="r+", shape=(total, args.dim))

    my = list(range(rank, len(rows), world)) if ddp else list(range(len(rows)))
    use_cuda = dev.startswith("cuda")
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_cuda else torch.autocast(device_type="cpu", enabled=False))
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
            hs, feat_len = model.hubert_tap(iv, am)                    # (B,Tmax,1024), (B,)
        hs = hs.float().cpu().numpy()
        for b, i in enumerate(idxs):
            T = int(index[i, 1]); off = int(index[i, 0])
            fl = int(feat_len[b]) if feat_len is not None else T
            assert fl >= T, f"data2vec frames {fl} < index T {T} for clip {i}"
            taps[off:off + T] = hs[b, :T].astype(np.float16)
        done += len(idxs)
        if is_main and (s // args.batch_size) % 50 == 0:
            print(f"[precompute-d2v rank0] {done}/{len(my)} clips", flush=True)
    taps.flush()

    if ddp:
        torch.distributed.barrier()
    if is_main:
        with open(os.path.join(args.cache_dir, "PREPARED"), "w") as f:
            f.write(f"{total} {args.dim}\n")
        print("[precompute-d2v] DONE", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
