"""
Local-LibriSpeech data path for the filler_asr smoke test.

Why this exists: the original dataset.py pulls 960h from HuggingFace
(load_dataset("librispeech_asr")) and caches decoded waveforms to disk
(~220GB) -- infeasible on this box. The full LibriSpeech is already on disk
as wavs + jsonl manifests, so here we:
  * read the local jsonl manifests ({"key","source","target"}),
  * keep only {audio_path, text} in the dataset (tiny, no waveform cache),
  * load + feature-extract audio LAZILY in the collator at batch time,
  * build the same "filler" labels as dataset.prepare_dataset.

Reuses (unchanged) from the original code:
  compute_output_length, extract_vocab, CHARS_TO_IGNORE_REGEX  (dataset.py)
"""
import re
import json
import math
import soundfile as sf
from datasets import Dataset, DatasetDict

from dataset import compute_output_length, extract_vocab, CHARS_TO_IGNORE_REGEX
from collator import FRAMES_PER_SEC, SAMPLING_RATE  # single source for the frame budget


def _norm(text):
    return re.sub(CHARS_TO_IGNORE_REGEX, "", text).lower()


def load_manifest(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            audio = d.get("source") or d.get("audio")
            text = d.get("target") or d.get("text") or ""
            rows.append({"audio_path": audio, "text": _norm(text)})
            if limit and len(rows) >= limit:
                break
    return rows


def write_vocab(train_jsonl, out_dir, exp_name):
    """Main-process-only: extract char vocab from training transcripts."""
    rows = load_manifest(train_jsonl)
    ds = Dataset.from_dict({"text": [r["text"] for r in rows]})
    return extract_vocab(ds, out_dir, exp_name)  # reuses original extract_vocab


def build_datasets(train_jsonl, dev_jsonl, dev_limit=None):
    train_ds = Dataset.from_list(load_manifest(train_jsonl))
    dev_ds = Dataset.from_list(load_manifest(dev_jsonl, limit=dev_limit))
    return DatasetDict({"train": train_ds, "test": dev_ds})


class DataCollatorLazyFillerASR:
    """Loads audio from disk and builds filler labels at batch time."""

    def __init__(self, feature_extractor, tokenizer):
        self.fe = feature_extractor
        self.tok = tokenizer
        self.fill_id = tokenizer.convert_tokens_to_ids("<fill>")
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def _labels_for(self, text, T):
        text = text.replace(" ", "|")
        ids = self.tok(text).input_ids
        if len(ids) + 2 > T:
            ids = ids[: T - 2]
        labels = [self.bos_id] + ids + [self.eos_id]
        labels = labels + [self.fill_id] * (T - len(labels))
        return labels

    def __call__(self, features):
        input_features, label_features, keep = [], [], []
        for feat in features:
            arr, sr = sf.read(feat["audio_path"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            assert sr == 16000, f"expected 16kHz, got {sr} for {feat['audio_path']}"
            iv = self.fe(arr, sampling_rate=16000).input_values[0]
            input_features.append({"input_values": iv})
            T = compute_output_length(len(iv))
            label_features.append({"input_ids": self._labels_for(feat["text"], T)})
            keep.append(math.ceil(len(iv) / SAMPLING_RATE * FRAMES_PER_SEC))

        batch = self.fe.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.tok.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # duration frame-budget mask: keep only the first n_keep frames per
        # utterance, ignore the rest of the <fill> tail in the loss.
        for i, n_keep in enumerate(keep):
            labels[i, n_keep:] = -100

        batch["labels"] = labels
        return batch
