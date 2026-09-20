"""data2vec + Hindi twin of filler_asr's data_tapcached.py.

Reuses the memmap dataset, the length-bucketed DDP sampler, and the CMLM tap collator UNCHANGED,
overriding only two things:
  * dim 1280 -> 1024  (data2vec-aqc width)
  * text normalization -> normalize_hi  (NFC + strip punctuation, matching vocab_hi)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FILLER_ROOT = os.environ.get("FILLER_ROOT", os.path.join(HERE, "deps", "filler_asr"))   # local copy
sys.path.insert(0, os.path.join(FILLER_ROOT, "train_sa_pe_lm_head"))

from data_tapcached import (                                            # noqa: E402
    TapCacheDataset as _TapCacheDataset,
    DataCollatorACMLMTapCached as _DataCollatorACMLMTapCached,
    LengthBucketedDistributedBatchSampler, cache_paths,
)
from text_norm_hi import normalize_hi                                   # noqa: E402

D2V_DIM = 1024


class TapCacheDataset(_TapCacheDataset):
    """Same memmap dataset, defaulting to the data2vec tap width (1024)."""
    def __init__(self, cache_dir, dim=D2V_DIM):
        super().__init__(cache_dir, dim=dim)


class DataCollatorACMLMTapCached(_DataCollatorACMLMTapCached):
    """CMLM tap collator at dim 1024, with a pluggable target normalizer (default: Hindi
    normalize_hi; pass normalize_fn=normalize_kn for Kannada+English)."""
    def __init__(self, tokenizer, frames_per_sec=50, eos_repeat=3,
                 force_full_mask_prob=0.15, fixed_p=None, dim=D2V_DIM, normalize_fn=None):
        super().__init__(tokenizer, frames_per_sec=frames_per_sec, eos_repeat=eos_repeat,
                         force_full_mask_prob=force_full_mask_prob, fixed_p=fixed_p, dim=dim)
        self._norm_fn = normalize_fn or normalize_hi

    def _normalize(self, text):
        return self._norm_fn(text)
