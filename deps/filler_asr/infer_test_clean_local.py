"""Filler-HuBERT character-level inference on a LOCAL LibriSpeech test-clean manifest.

Why this script (instead of the repo's inference.py):
  * inference.py imports model.py, which imports FillerASRTrainer ->
    `from transformers.trainer import is_optimizer_factory`. That symbol only
    exists in transformers 5.x. The only envs on this host are transformers 4.x,
    so model.py cannot be imported here.
  * inference.py also pulls test-clean via `load_dataset("librispeech_asr", ...)`
    (HF hub download). We instead read a local manifest (see prepare_test_clean.py)
    that references wavs already on disk.

This script reproduces ONLY the FillerHubertModel (HuBERT body + dropout + lm_head)
inline -- no Trainer -- so the checkpoint loads cleanly under transformers 4.48.3.
Decoding is the original recipe: per-frame argmax over the 33-token char vocab,
`skip_special_tokens=True, group_tokens=False` (no CTC merge of repeats).

WER/CER are computed with a dependency-free Levenshtein (no jiwer/evaluate needed),
corpus-aggregated (sum of edits / sum of reference length), matching jiwer's
default corpus reduction.

Run (single-utterance, no batch padding -> realistic CER):
  CUDA_VISIBLE_DEVICES=6 python infer_test_clean_local.py
  CUDA_VISIBLE_DEVICES=6 python infer_test_clean_local.py --limit 20   # quick smoke
"""
import os
import re
import json
import argparse

import torch
import torch.nn as nn
import soundfile as sf
from tqdm import tqdm
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    HubertConfig,
    HubertModel,
    HubertPreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"]'  # identical to dataset.py training normalization
DEFAULT_CKPT = "/speech/tomson/exps/speech-recog/models/asr_filler_hubert/checkpoint-164850"


class FillerHubertModel(HubertPreTrainedModel):
    """Inference-only copy of model.py's FillerHubertModel (no Trainer dependency).

    Identical module layout -> state_dict keys `hubert.*` + `lm_head.*` map 1:1,
    so `from_pretrained(checkpoint)` loads the trained head and body exactly.
    """

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hubert = HubertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(self, input_values, attention_mask=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.hubert(input_values, attention_mask=attention_mask, return_dict=return_dict)
        hidden_states = self.dropout(outputs[0])      # dropout is a no-op in eval()
        logits = self.lm_head(hidden_states)
        if not return_dict:
            return (logits,) + outputs[1:]
        return CausalLMOutput(loss=None, logits=logits,
                              hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def build_tokenizer(checkpoint):
    """Build the 33-token char tokenizer (incl. <fill>=32) from the checkpoint's vocab.json.

    We construct it explicitly (like the repo's inference.py) instead of
    `from_pretrained(checkpoint)`: the checkpoint's tokenizer_config.json was
    written by transformers 5.x and stores `extra_special_tokens` in a form that
    4.48.3's loader rejects (`'list' object has no attribute 'keys'`). vocab.json
    + explicit special tokens sidesteps that entirely.
    """
    vocab_file = os.path.join(checkpoint, "vocab.json")
    tok = Wav2Vec2CTCTokenizer(
        vocab_file,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        bos_token="<s>",
        eos_token="</s>",
    )
    tok.add_special_tokens({"additional_special_tokens": ["<fill>"]})
    assert len(tok) == 33, f"expected 33 tokens, got {len(tok)}"
    assert tok.convert_tokens_to_ids("<fill>") == 32, "fill token id must be 32"
    return tok


def load_wav(path, target_sr=16000):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), sr, target_sr).numpy()
    return audio


def normalize_ref(text):
    return re.sub(CHARS_TO_IGNORE_REGEX, "", text).lower().strip()


def levenshtein(ref, hyp):
    """Edit distance (S+D+I) between two token sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--manifest", default=os.path.join(HERE, "data", "test_clean.jsonl"))
    ap.add_argument("--output", default=os.path.join(HERE, "results", "test_clean_local_results.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N utterances (smoke test)")
    args = ap.parse_args()

    print(f"checkpoint: {args.checkpoint}")
    print(f"manifest:   {args.manifest}")
    print(f"device:     {args.device}")

    tokenizer = build_tokenizer(args.checkpoint)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )

    cfg = HubertConfig.from_pretrained(args.checkpoint)
    cfg.apply_spec_augment = False  # deterministic; eval() also disables it
    model = FillerHubertModel.from_pretrained(args.checkpoint, config=cfg).eval().to(args.device)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"utterances: {len(rows)}")

    w_err = w_tot = c_err = c_tot = 0
    results = []
    with torch.no_grad():
        for d in tqdm(rows):
            audio = load_wav(d["audio"])
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(args.device)
            logits = model(input_values=input_values).logits           # (1, T, 33)
            ids = torch.argmax(logits, dim=-1)                          # (1, T)
            hyp = tokenizer.batch_decode(ids, skip_special_tokens=True, group_tokens=False)[0].strip()
            ref = normalize_ref(d["text"])

            rw, hw = ref.split(), hyp.split()
            w_err += levenshtein(rw, hw); w_tot += len(rw)
            c_err += levenshtein(ref, hyp); c_tot += len(ref)
            results.append((d["key"], hyp, ref))

    wer = w_err / max(1, w_tot)
    cer = c_err / max(1, c_tot)
    print(f"\nTest-Clean WER: {wer:.4f}  ({w_err}/{w_tot} words)")
    print(f"Test-Clean CER: {cer:.4f}  ({c_err}/{c_tot} chars)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Manifest:   {args.manifest}  ({len(rows)} utts)\n")
        f.write(f"Avg WER: {wer:.4f}\n")
        f.write(f"Avg CER: {cer:.4f}\n\n")
        for key, hyp, ref in results:
            f.write(f"Key: {key}\nHyp: {hyp}\nRef: {ref}\n\n")
    print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
