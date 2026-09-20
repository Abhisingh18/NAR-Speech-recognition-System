"""
Run ONCE before training to warm the precomputed Arrow cache (feature extraction
+ framewise <fill> labels) for the SA filler_asr run, from LOCAL wavs (no
HuggingFace download). train_cached.py then reads this cache directly.

This is the heavy step: it decodes + feature-extracts all ~281k training wavs and
writes ~220GB of float32 input_values to <cache_dir>. Run it on its own (single
process driver, parallel map workers) inside tmux:

  conda activate filler_asr
  python prepare_data_local.py --config_path config_xlarge_sa_cached.json

Idempotent: re-running with the cache already present is a no-op (cache hit).
"""
import os
import json
import shutil
import argparse

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

from data_local import write_vocab
from data_cached import build_cached_datasets, cache_files


def build_tokenizer(cache_dir, train_jsonl, out_dir, exp_name):
    """Extract char vocab once into cache_dir; build the same tokenizer as train.py."""
    vocab_file = os.path.join(cache_dir, "vocab.json")
    if not os.path.exists(vocab_file):
        print("Extracting vocabulary from local train manifest ...")
        src = write_vocab(train_jsonl, out_dir, exp_name)  # writes experiments/<exp>/vocab.json
        shutil.copy(src, vocab_file)
    tok = Wav2Vec2CTCTokenizer(
        vocab_file, unk_token="<unk>", pad_token="<pad>",
        word_delimiter_token="|", bos_token="<s>", eos_token="</s>",
    )
    tok.add_special_tokens({"additional_special_tokens": ["<fill>"]})
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config_xlarge_sa_cached.json")
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = json.load(f)

    cache_dir = config["cache_dir"]
    out_dir = config.get("output_dir", "experiments")
    exp_name = config.get("exp_name", "Hubert_SA_cached")
    os.makedirs(cache_dir, exist_ok=True)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    tokenizer = build_tokenizer(cache_dir, config["train_jsonl"], out_dir, exp_name)

    print(f"Precomputing into {cache_dir} (num_proc={config.get('num_processes', 8)}) ...")
    dataset = build_cached_datasets(config, feature_extractor, tokenizer)
    print(f"Done. train={len(dataset['train'])}  dev={len(dataset['test'])}")

    # Persist the extractor + tokenizer alongside the cache so train_cached.py
    # loads byte-identical objects (and therefore the same label ids).
    feature_extractor.save_pretrained(cache_dir)
    tokenizer.save_pretrained(cache_dir)

    # Sentinel so train_cached.py refuses to start (and silently recompute under
    # DDP) before the cache is ready.
    cf = cache_files(cache_dir)
    with open(os.path.join(cache_dir, "PREPARED"), "w") as f:
        json.dump({"train_cache": cf["train"], "dev_cache": cf["test"],
                   "num_processes": config.get("num_processes", 8)}, f, indent=2)
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
