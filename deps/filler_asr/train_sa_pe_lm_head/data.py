"""Full-LibriSpeech data for the SA + PE + lm_head pipeline  (NO caching).

Unlike train_cached_plots.py, the frozen HuBERT encoder is run on every step
(forward pass every time).  So this module holds ONLY the tiny jsonl manifest in
memory ({audio_path, text} per row) and decodes + feature-extracts the audio
lazily inside the collator at batch time.

The positional <fill> label construction, duration-budget loss mask and frame
mask are reused unchanged from the verified reference (filler_sa_reference.py)
so this pipeline is bit-identical in target/masking to the cached experiments.
"""
import os
import sys
import json

from torch.utils.data import Dataset

# reuse the verified model + collator from the parent filler_asr directory
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from filler_sa_reference import DataCollatorFillerASR  # noqa: E402  (re-exported below)

__all__ = ["load_rows", "ManifestDataset", "DataCollatorFillerASR"]


def load_rows(manifest_path, limit=None):
    """Read a LibriSpeech jsonl manifest into [{audio_path, text}, ...].

    Accepts both schemas seen on this box: {"source","target"} and {"audio","text"}.
    Text is left RAW here — DataCollatorFillerASR normalizes (lowercase + strip
    punctuation) internally, so normalization happens in exactly one place.
    """
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({
                "audio_path": d.get("source") or d.get("audio"),
                "text": d.get("target") or d.get("text") or "",
            })
            if limit and len(rows) >= limit:
                break
    return rows


class ManifestDataset(Dataset):
    """Tiny in-memory list of {audio_path, text}; audio decoded lazily in the collator."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]
