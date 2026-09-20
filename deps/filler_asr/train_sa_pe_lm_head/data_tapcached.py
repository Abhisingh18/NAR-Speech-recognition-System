"""Tap-cache data path for the A-CMLM run.

The frozen HuBERT tap is a deterministic function of the audio (encoder pinned to eval),
so it is precomputed ONCE (precompute_taps.py) into a ragged memmap and reused every epoch,
removing the 48-layer encoder from every training step.

Cache layout (see precompute_taps.py):
    taps.f16      memmap (sum_i T_i, 1280) float16   -- all clips' frames stacked
    index.npy     (N, 2) int64  -> [row_offset, T_i] per clip
    manifest.jsonl  N lines {"text": ..., "T": ...}  (aligned to index rows)

This module provides:
    TapCacheDataset                       -- serves {tap (T,1280), text, T} via memmap slice
    LengthBucketedDistributedBatchSampler -- length-homogeneous batches, DDP-sharded
    DataCollatorACMLMTapCached            -- pads taps + builds the CMLM text/labels
"""
import os
import sys
import json
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../filler_asr
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import DataCollatorFillerASR, MASK_TOKEN       # noqa: E402

TAP_DIM = 1280


def cache_paths(cache_dir):
    return (os.path.join(cache_dir, "taps.f16"),
            os.path.join(cache_dir, "index.npy"),
            os.path.join(cache_dir, "manifest.jsonl"))


class TapCacheDataset(Dataset):
    """Serves {tap (T,1280) float16, text, T} from the ragged memmap + index (no audio, no HuBERT)."""

    def __init__(self, cache_dir, dim=TAP_DIM):
        taps_f, index_f, man_f = cache_paths(cache_dir)
        self.index = np.load(index_f)                                  # (N,2) [row_offset, T]
        total = int(self.index[-1, 0] + self.index[-1, 1]) if len(self.index) else 0
        self.taps = np.memmap(taps_f, dtype=np.float16, mode="r", shape=(total, dim))
        self.texts = [json.loads(l)["text"] for l in open(man_f)]
        assert len(self.texts) == len(self.index), "manifest/index length mismatch"
        self.dim = dim

    def __len__(self):
        return len(self.index)

    def lengths(self):
        return self.index[:, 1]

    def __getitem__(self, i):
        off, T = int(self.index[i, 0]), int(self.index[i, 1])
        tap = torch.from_numpy(np.array(self.taps[off:off + T], dtype=np.float16))   # copy -> writable
        return {"tap": tap, "text": self.texts[i], "T": T}


class LengthBucketedDistributedBatchSampler(Sampler):
    """Yield batches of `batch_size` clips of SIMILAR length (sort -> chunk), with the batch ORDER
    shuffled per epoch, sharded across DDP ranks with equal batch counts (drop_last semantics).

    Length homogeneity kills the ~28% padding waste of random batching; equal per-rank batch counts
    keep DDP ranks in lockstep (required by the token-exact all-reduce in train.py)."""

    def __init__(self, lengths, batch_size, num_replicas=1, rank=0, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.world = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        order = np.argsort(self.lengths, kind="stable")               # short -> long
        n_full = (len(order) // self.batch_size) * self.batch_size    # drop the ragged tail
        self._batches = [order[i:i + self.batch_size].tolist()
                         for i in range(0, n_full, self.batch_size)]
        nb = (len(self._batches) // self.world) * self.world          # divisible by world
        self._batches = self._batches[:nb]
        self._per_rank = nb // self.world

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self._per_rank

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:                                              # shuffle ORDER, not composition
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)
        return iter(batches[self.rank::self.world])                   # this rank's share


class DataCollatorACMLMTapCached(DataCollatorFillerASR):
    """Pads cached taps and builds the A-CMLM text/labels (identical masking scheme to
    DataCollatorACMLM, but on cached taps instead of audio). Produces:
        tap (B,Tmax,1280) f16 | pad_mask (B,Tmax) bool True=pad | text_input_ids | labels
    n_keep is derived from T (taps are 50 fps): n_keep = min(T, ceil(T * fps / 50))."""

    def __init__(self, tokenizer, frames_per_sec=50, eos_repeat=3,
                 force_full_mask_prob=0.15, fixed_p=None, dim=TAP_DIM):
        super().__init__(feature_extractor=None, tokenizer=tokenizer,
                         frames_per_sec=frames_per_sec, eos_repeat=eos_repeat)
        self.mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        self.pad_id = tokenizer.pad_token_id
        self.force_full_mask_prob = force_full_mask_prob
        self.fixed_p = fixed_p
        self.dim = dim

    def __call__(self, items):
        B = len(items)
        Tmax = max(it["T"] for it in items)
        taps = torch.zeros(B, Tmax, self.dim, dtype=torch.float16)
        pad = torch.ones(B, Tmax, dtype=torch.bool)                   # True = padding
        text_ids = torch.full((B, Tmax), self.pad_id, dtype=torch.long)
        labels = torch.full((B, Tmax), -100, dtype=torch.long)
        for b, it in enumerate(items):
            T = it["T"]
            taps[b, :T] = it["tap"]
            pad[b, :T] = False
            y = torch.tensor(self._labels_for(it["text"], T), dtype=torch.long)   # positional target
            text_ids[b, :T] = y                                       # default input = target (fill tail incl.)
            n_keep = min(T, math.ceil(T * self.frames_per_sec / 50.0))
            if n_keep > 1:                                            # CMLM: mask a subset of [1, n_keep)
                p = self.fixed_p if self.fixed_p is not None else (
                    1.0 if torch.rand(1).item() < self.force_full_mask_prob else torch.rand(1).item())
                region = torch.arange(1, n_keep)
                sel = region[torch.rand(region.numel()) < p]
                if sel.numel() == 0:
                    sel = region[torch.randint(region.numel(), (1,))]
                text_ids[b, sel] = self.mask_id                       # <mask> at masked slots
                labels[b, sel] = y[sel]                               # supervise only masked slots
        return {"tap": taps, "pad_mask": pad, "text_input_ids": text_ids, "labels": labels}
