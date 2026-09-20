"""Zero-shot LLM error correction (GER) for filler-HuBERT SA8+PE decodes.

Reads strip_variants decode outputs, prompts an instruction LLM to correct the
ASR hypothesis, and reports corpus WER before/after.

Modes:
  clean : input is the already-cleaned hypothesis (HYP line)
  raw   : input is the raw token stream (<s> spaced chars | ... </s> <fill>...),
          the LLM must both parse the format and fix errors

Usage:
  python correct_llm.py --mode clean --n 300
  python correct_llm.py --mode clean            # full 2620
  python correct_llm.py --mode clean --gate --decodes <run_dir>/l2arctic_test.decodes.txt
"""
import argparse
import json
import re
import string
import time

import jiwer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DECODES = "/speech/tomson/filler_asr/train_sa_pe_lm_head/eval_analysis/strip_variants/test_clean.decodes.txt"

SYSTEM = (
    "You are a transcription corrector for an English automatic speech "
    "recognition (ASR) system transcribing audiobook recordings."
)

PROMPT_CLEAN = """The following ASR transcript of a classic-literature audiobook may contain errors: misspelled words, dropped or repeated letters, words split or merged, or short stretches of garbled characters. Correct it to the most plausible original English sentence.

Rules:
- Output ONLY the corrected transcript, nothing else.
- All lowercase, no punctuation except apostrophes in contractions and possessives.
- ONLY fix words that are clearly misspelled or garbled (not valid English words). Never replace a valid word with a synonym or a different valid word.
- The text is from old books: archaic or unusual words and grammar (ye, thee, hath, whilst) are usually CORRECT — keep them exactly as they are.
- Do not change word spacing of names or compounds; do not add or drop words.
- If unsure about a word, leave it unchanged. If the transcript already looks correct, output it unchanged.

Transcript: {hyp}"""

PROMPT_RAW = """The following is the raw output of a character-level ASR system. Format: it starts with <s>, each letter of a word is separated by spaces, words are separated by | , the sentence ends at </s>, and everything after that is <fill> padding to be ignored. The characters may contain errors: misspelled words, dropped or repeated letters, or garbled stretches.

Convert it to the most plausible original English sentence.

Rules:
- Output ONLY the corrected sentence, nothing else.
- All lowercase, no punctuation except apostrophes in contractions and possessives.
- Keep every word the speaker plausibly said; do not paraphrase, add, or drop content.

Raw ASR output: {hyp}"""

KEEP = set(string.ascii_lowercase + string.digits + "' ")
GATE_VOCAB = "/speech/tomson/filler_asr/train_sa_pe_lm_head/eval_analysis/llm_correction/gate_vocab.txt"


def is_suspicious(hyp, vocab):
    """True if the hypothesis contains any out-of-vocabulary token."""
    return any(w.strip("'") not in vocab for w in hyp.split())


def normalize(text):
    text = text.lower().strip()
    # strip surrounding quotes/labels the model sometimes adds
    text = re.sub(r"^(corrected transcript|transcript|corrected sentence|output)\s*:\s*", "", text)
    text = text.replace("’", "'")
    text = "".join(c if c in KEEP else " " for c in text)
    return " ".join(text.split())


def load_decodes(path):
    utts = []
    cur = None
    with open(path) as f:
        for line in f:
            m = re.match(r"^=+\s+(\S+)\s+dur=", line)
            if m:
                cur = {"utt": m.group(1)}
                utts.append(cur)
            elif line.startswith("RAW   :"):
                cur["raw"] = line[7:].strip()
            elif line.startswith("HYP   :"):
                cur["hyp"] = line[7:].strip()
            elif line.startswith("REF   :"):
                cur["ref"] = line[7:].strip()
    return utts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", choices=["clean", "raw"], default="clean")
    ap.add_argument("--n", type=int, default=0, help="subset size, 0 = all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate", action="store_true",
                    help="only apply LLM correction to utts with OOV tokens")
    ap.add_argument("--decodes", default=DECODES,
                    help="dump_decodes .decodes.txt to correct")
    args = ap.parse_args()

    utts = load_decodes(args.decodes)
    if args.n:
        utts = utts[: args.n]
    out_path = args.out or f"corrected.{args.mode}.n{len(utts)}.jsonl"

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    template = PROMPT_CLEAN if args.mode == "clean" else PROMPT_RAW
    prompts = []
    for u in utts:
        inp = u["hyp"] if args.mode == "clean" else u["raw"]
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": template.format(hyp=inp)},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    t0 = time.time()
    results = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for j, seq in enumerate(gen):
            new = seq[enc["input_ids"].shape[1] :]
            results.append(normalize(tok.decode(new, skip_special_tokens=True)))
        done = min(i + args.batch_size, len(prompts))
        print(f"[{done}/{len(prompts)}] {time.time()-t0:.0f}s", flush=True)

    refs = [u["ref"] for u in utts]
    hyps = [u["hyp"] for u in utts]
    corr = [c if c else h for c, h in zip(results, hyps)]  # empty output -> keep hyp

    n_gated = 0
    if args.gate:
        vocab = set(open(GATE_VOCAB).read().split())
        gated = []
        for h, c in zip(hyps, corr):
            if is_suspicious(h, vocab):
                gated.append(c)
            else:
                gated.append(h)
                n_gated += c != h
        corr = gated

    wer_before = jiwer.wer(refs, hyps)
    wer_after = jiwer.wer(refs, corr)
    improved = worsened = unchanged_wer = 0
    with open(out_path, "w") as f:
        for u, c in zip(utts, corr):
            wb = jiwer.wer(u["ref"], u["hyp"])
            wa = jiwer.wer(u["ref"], c)
            improved += wa < wb
            worsened += wa > wb
            unchanged_wer += wa == wb
            f.write(json.dumps({
                "utt": u["utt"], "ref": u["ref"], "hyp": u["hyp"],
                "corrected": c, "wer_before": wb, "wer_after": wa,
            }) + "\n")

    summary = {
        "model": args.model, "mode": args.mode, "n": len(utts),
        "corpus_wer_before": wer_before, "corpus_wer_after": wer_after,
        "utts_improved": improved, "utts_worsened": worsened,
        "utts_same_wer": unchanged_wer,
        "gate": args.gate, "llm_edits_rejected_by_gate": n_gated,
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(summary, indent=2))
    with open(out_path.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
