import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, HubertConfig
from datasets import load_dataset, Audio
from model import FillerHubertModel
import evaluate
import numpy as np


def load_audio_array(audio_path, sampling_rate=16000):
    audio_ds = load_dataset("audio", data_files={"audio":[audio_path]}, split="audio")
    audio_ds = audio_ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return audio_ds[0]["audio"]["array"]


def decode_predictions(logits, tokenizer):
    predicted_ids = torch.argmax(logits, dim=-1)
    return tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, group_tokens=False)[0]


def infer_audio(audio, model, feature_extractor, tokenizer):
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    logits = model(**inputs).logits
    return decode_predictions(logits, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to a single audio file to transcribe")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save the transcription")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to cache/load datasets")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    out_dir = config.get("output_dir", "experiments")
    exp_name = config.get("exp_name", "filler_asr_960h")
    dataset_name = config.get("dataset_name", "librispeech_asr")

    exp_dir = os.path.join(out_dir, exp_name)
    vocab_file = os.path.join(exp_dir, "vocab.json")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ['<fill>']})

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    checkpoint = None
    if os.path.exists(exp_dir):
        checkpoints = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Loading checkpoint: {checkpoint}")

    if not checkpoint:
        print("No checkpoint found. Exiting.")
        return

    hubert_config = HubertConfig.from_pretrained(checkpoint)
    hubert_config.vocab_size = len(tokenizer)
    hubert_config.pad_token_id = tokenizer.pad_token_id
    hubert_config.bos_token_id = tokenizer.bos_token_id
    hubert_config.eos_token_id = tokenizer.eos_token_id

    model = FillerHubertModel.from_pretrained(
        checkpoint,
        config=hubert_config,
        ignore_mismatched_sizes=True
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    if args.audio_path:
        audio = load_audio_array(args.audio_path)
        transcription = infer_audio(audio, model, feature_extractor, tokenizer)
        print("Transcription:", transcription)
        if args.output_path:
            with open(args.output_path, "w", encoding="utf-8") as out_file:
                out_file.write(transcription)
        return

    print("Loading test-clean dataset...")
    test_dataset = load_dataset(dataset_name, "clean", split="test", cache_dir=os.path.join(os.path.dirname(__file__), args.data_dir))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions = []
    references = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_dataset):
            audio = batch["audio"]["array"]
            transcription = infer_audio(audio, model, feature_extractor, tokenizer)

            # Normalize reference text
            ref_str = batch["text"].lower().replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace(";", "").replace(":", "").replace('"', "")

            predictions.append(transcription)
            references.append(ref_str)

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print(f"Test-Clean WER: {wer:.4f}")
    print(f"Test-Clean CER: {cer:.4f}")

    # Save results to file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "test_clean_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Avg WER: {wer:.4f}\n")
        f.write(f"Avg CER: {cer:.4f}\n\n")
        for pred, ref in zip(predictions, references):
            f.write(f"Hyp: {pred}\n")
            f.write(f"Ref: {ref}\n\n")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
