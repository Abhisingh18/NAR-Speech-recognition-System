"""GER (Qwen2.5-7B) error correction on the A-CMLM iterative decode (all_utts.jsonl).

Reuses correct_llm.py's SYSTEM/PROMPT_CLEAN/normalize/is_suspicious + OOV gate.
Cleans the <fill> readout-leak from the hyp first. Reports corpus WER + word S/D/I
before/after, gated (LLM only touches utts with an OOV token).
"""
import argparse, json, re, time, sys, os
import jiwer, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from correct_llm import SYSTEM, PROMPT_CLEAN, normalize, is_suspicious, GATE_VOCAB

ALL = "/speech/tomson/filler_asr/train_sa_pe_lm_head/runs/full960_sa8_ACMLM_pe_fps25_mask20-30x10_eos3_fw0.1/omni_decode_test_clean_20260826_124905/all_utts.jsonl"

def sdi(a, b):
    n, m = len(a), len(b); dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
    i, j = n, m; S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1] and dp[i][j] == dp[i-1][j-1]: i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1]+1: S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j]+1: D += 1; i -= 1
        else: I += 1; j -= 1
    return S, D, I

def corpus_sdi(refs, hyps):
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        s, d, i = sdi(r.split(), h.split()); S += s; D += d; I += i; N += len(r.split())
    return S, D, I, N

def fmt(tag, refs, hyps):
    S, D, I, N = corpus_sdi(refs, hyps); E = S + D + I
    return (f"{tag:22} WER={E/N*100:6.2f}%  "
            f"Sub={S/N*100:5.2f}% Del={D/N*100:5.2f}% Ins={I/N*100:5.2f}%  "
            f"(S/D/I {S/E*100:.0f}/{D/E*100:.0f}/{I/E*100:.0f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--all", default=ALL, help="path to decode all_utts.jsonl")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(_HERE, "corrected.acmlm.gated.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.all)]
    if args.n: rows = rows[: args.n]
    for r in rows:                                     # clean the <fill> readout leak
        r["hyp_clean"] = re.sub(r"(<fill>)+", " ", r["hyp"]).strip()
        r["hyp_clean"] = " ".join(r["hyp_clean"].split())

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": PROMPT_CLEAN.format(hyp=r["hyp_clean"])}],
        tokenize=False, add_generation_prompt=True) for r in rows]

    t0 = time.time(); results = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i:i+args.batch_size], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for seq in gen:
            results.append(normalize(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)))
        print(f"[{min(i+args.batch_size,len(prompts))}/{len(prompts)}] {time.time()-t0:.0f}s", flush=True)

    refs = [r["ref"] for r in rows]
    hyps = [r["hyp_clean"] for r in rows]
    corr_raw = [c if c else h for c, h in zip(results, hyps)]

    vocab = set(open(GATE_VOCAB).read().split())
    n_gated = 0; corr = []
    for h, c in zip(hyps, corr_raw):
        if is_suspicious(h, vocab): corr.append(c)
        else: corr.append(h); n_gated += (c != h)

    with open(args.out, "w") as f:
        for r, c in zip(rows, corr):
            f.write(json.dumps({"utt": r["key"], "ref": r["ref"], "hyp": r["hyp_clean"],
                                "corrected": c,
                                "wer_before": jiwer.wer(r["ref"], r["hyp_clean"]),
                                "wer_after": jiwer.wer(r["ref"], c)}) + "\n")

    print("\n=== A-CMLM + external LM (Qwen2.5-7B GER, OOV-gated) ===")
    print(fmt("before (raw decode)", refs, hyps))
    print(fmt("after  (GER gated)", refs, corr))
    print(f"gate: LLM edits rejected on in-vocab utts = {n_gated}   time = {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
