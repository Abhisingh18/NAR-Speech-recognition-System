"""
Shared builder for the PRECOMPUTED (Arrow-cached) filler_asr data path.

Replicates the original dataset.get_dataset() behaviour -- feature-extract audio
+ build the framewise <fill> labels with datasets.Dataset.map(num_proc=...) --
and PERSISTS the result as a *named* Arrow cache on disk (the same idea as the
reference box's cache-*.arrow files). The first run (prepare_data_local.py)
computes it; every later run (train_cached.py) reuses the exact same cache files.

Why a named cache_file_name instead of save_to_disk:
  save_to_disk would write a SECOND full copy of the ~220GB input_values next to
  the map cache (peak ~440GB > the ~450GB free on the shared /speech). Pointing
  map at an explicit cache_file_name keeps it to a SINGLE ~220GB copy, and is
  closer to how the original/reference pipeline persisted things anyway.

Both prepare_data_local.py and train_cached.py import build_cached_datasets() so
they hit the identical cache_file_name + num_proc -- reuse requires both to match
(keep `num_processes` the same in the config between prepare and train).

Reuses (unchanged): compute_output_length (dataset.py), load_manifest (data_local.py).
"""
import os
import soundfile as sf
from datasets import Dataset, DatasetDict

from dataset import compute_output_length
from data_local import load_manifest


def _make_prepare_fn(feature_extractor, tokenizer):
    """Closure mirroring dataset.prepare_dataset, reading audio from a local path."""
    fill_id = tokenizer.convert_tokens_to_ids("<fill>")
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    def prepare(batch):
        arr, sr = sf.read(batch["audio_path"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        assert sr == 16000, f"expected 16kHz, got {sr} for {batch['audio_path']}"
        iv = feature_extractor(arr, sampling_rate=16000).input_values[0]
        T = compute_output_length(len(iv))

        text = batch["text"].replace(" ", "|")
        ids = tokenizer(text).input_ids
        if len(ids) + 2 > T:
            ids = ids[: T - 2]
        labels = [bos_id] + ids + [eos_id]
        labels = labels + [fill_id] * (T - len(labels))

        return {"input_values": iv, "labels": labels}

    return prepare


def _build_split(rows, prepare_fn, num_proc, cache_file):
    ds = Dataset.from_list(rows)  # {audio_path, text}
    return ds.map(
        prepare_fn,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        cache_file_name=cache_file,   # persistent, single copy on /speech
        load_from_cache_file=True,    # reuse if already computed
        desc=f"precompute -> {os.path.basename(cache_file)}",
    )


def cache_files(cache_dir):
    return {
        "train": os.path.join(cache_dir, "train_prepared.arrow"),
        "test": os.path.join(cache_dir, "dev_prepared.arrow"),
    }


def build_cached_datasets(config, feature_extractor, tokenizer):
    """Compute-or-load the precomputed {train,test} Arrow datasets (input_values + labels)."""
    cache_dir = config["cache_dir"]
    num_proc = config.get("num_processes", 8)
    os.makedirs(cache_dir, exist_ok=True)
    prepare_fn = _make_prepare_fn(feature_extractor, tokenizer)
    cf = cache_files(cache_dir)

    train_rows = load_manifest(config["train_jsonl"])
    dev_rows = load_manifest(config["dev_jsonl"])  # full dev cached; dev_limit applied at train time

    train_ds = _build_split(train_rows, prepare_fn, num_proc, cf["train"])
    dev_ds = _build_split(dev_rows, prepare_fn, num_proc, cf["test"])
    return DatasetDict({"train": train_ds, "test": dev_ds})
