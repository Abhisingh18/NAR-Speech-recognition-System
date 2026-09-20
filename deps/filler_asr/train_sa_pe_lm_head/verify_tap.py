"""CPU verification of the deterministic-tap / tap-cache implementation.

No GPU, no audio files, no touching the live run. Checks:
  1. DETERMINISM      : hubert_tap(x) twice (eval) -> bitwise identical  (=> cacheable)
  2. LENGTH           : compute_output_length(L) == HuBERT valid frames  (index bookkeeping is exact)
  3. FP16 ROUNDTRIP   : fp16-stored tap == fp32 tap within fp16 rounding
  4. PATH EQUIVALENCE : model(input_values=...) == model(tap=cached, ...)  (eval, same text)  -> the
                        cached forward reproduces the live forward exactly
  5. BUCKETING        : LengthBucketedDistributedBatchSampler -> length-homogeneous, DDP disjoint+equal
  6. DATASET/COLLATOR : memmap roundtrip + CMLM collate shapes/masks

Run:  conda run -n filler_asr python verify_tap.py
"""
import os
import sys
import json
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import (compute_output_length, build_acmlm_tokenizer,   # noqa: E402
                                 build_acmlm_model)
from data_tapcached import (TapCacheDataset, LengthBucketedDistributedBatchSampler,   # noqa: E402
                            DataCollatorACMLMTapCached, cache_paths, TAP_DIM)

VOCAB = "/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_pe_fps25_lin_100ep_mask20-30x10_eos3_fw0.1"
CKPT = "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"
n_pass = n_fail = 0


def check(name, ok, detail=""):
    global n_pass, n_fail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    n_pass += ok
    n_fail += (not ok)


def main():
    torch.manual_seed(0)
    print("building ACMLM model on CPU (loads frozen hubert-xlarge) ...", flush=True)
    tok = build_acmlm_tokenizer(VOCAB)
    model = build_acmlm_model(CKPT, tok, device="cpu").eval()

    # two synthetic clips (2.0s, 3.1s) -> batch with attention_mask
    L = [int(16000 * 2.0), int(16000 * 3.1)]
    iv = torch.zeros(2, max(L))
    am = torch.zeros(2, max(L), dtype=torch.long)
    for b, l in enumerate(L):
        iv[b, :l] = torch.randn(l)
        am[b, :l] = 1

    print("\n== 1. determinism + 2. length + 3. fp16 roundtrip ==", flush=True)
    with torch.no_grad():
        hs1, fl1 = model.hubert_tap(iv, am)
        hs2, fl2 = model.hubert_tap(iv, am)
    check("hubert_tap deterministic (eval, 2 runs identical)", torch.equal(hs1, hs2))
    exp = [compute_output_length(l) for l in L]
    check("compute_output_length == HuBERT valid frames",
          [int(x) for x in fl1] == exp, f"{[int(x) for x in fl1]} vs {exp}")
    rt = hs1.half().float()
    md = (rt - hs1).abs().max().item()
    check("fp16 roundtrip within rounding", md < 0.05, f"max|Δ|={md:.2e}, tap absmax={hs1.abs().max():.2f}")

    print("\n== 4. path equivalence: model(input_values) == model(tap=cached) ==", flush=True)
    Tmax = hs1.shape[1]
    pad = torch.ones(2, Tmax, dtype=torch.bool)
    for b in range(2):
        pad[b, :int(fl1[b])] = False
    tii = torch.randint(0, 34, (2, Tmax))                    # arbitrary text ids (incl <mask>=33)
    with torch.no_grad():
        out_full = model(input_values=iv, attention_mask=am, text_input_ids=tii).logits
        out_cached = model(tap=hs1, pad_mask=pad, text_input_ids=tii).logits
    diff = (out_full - out_cached).abs().max().item()
    check("cached-path logits == live-path logits", diff < 1e-4, f"max|Δlogits|={diff:.2e}")

    print("\n== 5. length-bucketed DDP sampler ==", flush=True)
    rng = np.random.default_rng(1)
    lengths = rng.integers(40, 900, size=200)
    bs, world = 8, 4
    samplers = [LengthBucketedDistributedBatchSampler(lengths, bs, num_replicas=world, rank=r, seed=7)
                for r in range(world)]
    per_rank = len(samplers[0])
    counts_ok = all(len(s) == per_rank for s in samplers)
    all_batches, spreads = [], []
    for s in samplers:
        for batch in s:
            all_batches.append(tuple(sorted(batch)))
            L_ = lengths[batch]
            spreads.append(int(L_.max() - L_.min()))
    sizes_ok = all(len(b) == bs for b in all_batches)
    disjoint = len(set(all_batches)) == len(all_batches)           # no batch shared across ranks
    covered = len({i for b in all_batches for i in b})
    n_full = (len(lengths) // bs // world) * bs * world
    homog = max(spreads)                                            # worst within-batch length spread
    check("equal batch count per rank", counts_ok, f"{per_rank}/rank x {world} ranks")
    check("batch size uniform", sizes_ok)
    check("ranks disjoint (no shared batch)", disjoint)
    check("coverage == n_full (drop_last)", covered == n_full, f"{covered} of {n_full}")
    check("batches length-homogeneous (small spread)", homog < 60, f"worst spread={homog} frames")
    s0a = list(samplers[0])
    G0 = {tuple(sorted(b)) for s in samplers for b in list(s)}          # global batch set, epoch 0
    for s in samplers:
        s.set_epoch(1)
    s0b = list(samplers[0])
    G1 = {tuple(sorted(b)) for s in samplers for b in list(s)}          # global batch set, epoch 1
    check("set_epoch reshuffles (global batch set preserved, rank-0 order changes)",
          G0 == G1 and s0a != s0b, f"|G|={len(G0)} preserved, rank0 changed={s0a != s0b}")

    print("\n== 6. dataset memmap roundtrip + CMLM collator ==", flush=True)
    with tempfile.TemporaryDirectory() as d:
        Ts = [5, 9, 7]
        off = np.array([0, 5, 14], dtype=np.int64)
        idx = np.stack([off, np.array(Ts)], 1)
        total = sum(Ts)
        mm = np.memmap(cache_paths(d)[0], dtype=np.float16, mode="w+", shape=(total, TAP_DIM))
        ref = (np.arange(total * TAP_DIM).reshape(total, TAP_DIM) % 7).astype(np.float16)
        mm[:] = ref; mm.flush()
        np.save(cache_paths(d)[1], idx)
        with open(cache_paths(d)[2], "w") as f:
            for t, T in zip(["the cat", "hello world foo", "a b c"], Ts):
                f.write(json.dumps({"text": t, "T": int(T)}) + "\n")
        ds = TapCacheDataset(d)
        it1 = ds[1]
        slice_ok = it1["tap"].shape == (9, TAP_DIM) and np.allclose(it1["tap"].numpy(), ref[5:14])
        check("dataset slices correct clip from memmap", slice_ok, f"tap.shape={tuple(it1['tap'].shape)}")
        col = DataCollatorACMLMTapCached(tok, frames_per_sec=50, eos_repeat=3)
        batch = col([ds[0], ds[1], ds[2]])
        Tm = max(Ts)
        shapes_ok = (batch["tap"].shape == (3, Tm, TAP_DIM) and batch["pad_mask"].shape == (3, Tm)
                     and batch["text_input_ids"].shape == (3, Tm) and batch["labels"].shape == (3, Tm))
        check("collator output shapes", shapes_ok)
        pad_ok = bool(batch["pad_mask"][2, 7:].all()) and not bool(batch["pad_mask"][2, :7].any())
        check("pad_mask matches per-clip T", pad_ok)
        sup = (batch["labels"] != -100)
        mask_at_sup = (batch["text_input_ids"][sup] == col.mask_id).all().item()
        check("supervised slots are <mask> in input, >=1 per clip",
              bool(mask_at_sup) and all((sup[b]).sum() >= 1 for b in range(3)))

    print(f"\n==== {n_pass} passed, {n_fail} failed ====", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
