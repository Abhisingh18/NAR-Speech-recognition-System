"""OmniVoice-style iterative decoding for the audio-conditioned masked-LM (A-CMLM) ASR head.

INFERENCE engine for the redesigned model (ADDITION fusion, no gamma):
    h = tap (+audio-mask, train-only) + PE + E_text(text_ids) -> 8 SA -> lm_head -> (B,T,33)

It borrows OmniVoice's discrete masked-diffusion sampler, minus the parts ASR doesn't need:
    * NO duration model  -> canvas length T = HuBERT frame count (from the audio).
    * NO layer penalty    -> single text stream (not 8 RVQ codebooks).

Length handling (matches training's duration budget):
    T      = compute_output_length(#samples)          # full canvas (PE + attention)
    n_keep = ceil(dur_sec * 25)                        # CONTENT region = first n_keep frames
    seed   : pos0=<s> ; pos 1..n_keep-1 = <mask> ; pos n_keep..T-1 = <fill> (fixed scaffold)
    decode : iteratively unmask only the n_keep content slots ; then cut at first </s>.

Per step:
    r_n  = tau*(n/N) / (1 + (tau-1)*(n/N))             # back-loaded schedule, r_0=0, r_N=1
    k_n  = round(r_n*M) - round(r_{n-1}*M)             # NEW commits this step  (M = n_keep-1)
    logp = log_softmax( Model(tap, ids) )              # re-reads the grid; tap cached ONCE
    conf = max_v logp ;  pred = argmax_v logp          # value = greedy (deterministic)
    conf[pred==<fill>] -= fill_penalty                 # optional
    sel  = gumbel_top_k( conf[masked]/temp , k_n )     # temperature-sampled ORDER
    ids[sel] = pred[sel]                               # commit (permanent)

Runnable NOW without a checkpoint (drives the same loop against a mock model):
    python filler_asr_omni_style_decoding.py --selftest --text cat --steps 12

Real decode (after training with train.py --acmlm):
    python filler_asr_omni_style_decoding.py \
        --run_dir runs/<acmlm_run> --manifest <test.jsonl> --steps 32 --tau 0.1 --temp 5
"""
import os
import re
import sys
import json
import time
import math
import argparse

import torch
import torch.nn.functional as F

# same path wiring as data.py / infer.py: make the parent filler_asr dir importable
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

FRAMES_PER_SEC = 25          # legacy default; the real fps is read from the ckpt's
                             # training args (or --fps), so decode matches training.
MASK_TOKEN = "<mask>"


# ============================================================================
# 1) SCHEDULE  (OmniVoice Eq. 3)
# ============================================================================
def schedule_r(n, N, tau):
    """Cumulative fraction of content slots unmasked after step n.  r(0)=0, r(N)=1.
    tau<1 back-loaded (few early, avalanche late) ; tau=1 linear ; tau>1 front-loaded."""
    x = n / N
    return (tau * x) / (1.0 + (tau - 1.0) * x)


def commit_counts(N, tau, M):
    """Per-step count of NEW slots so cumulative matches round(r_n * M)."""
    counts, prev = [], 0
    for n in range(1, N + 1):
        cum = round(schedule_r(n, N, tau) * M)
        counts.append(max(0, cum - prev))
        prev = cum
    counts[-1] += M - prev            # guarantee all committed by step N
    return counts


# ============================================================================
# 2) POSITION SELECTION  (temperature-sampled == Gumbel-top-k, no replacement)
# ============================================================================
def gumbel_top_k(scores, k, generator=None):
    k = min(k, scores.numel())
    if k <= 0:
        return scores.new_empty(0, dtype=torch.long)
    u = torch.rand(scores.shape, generator=generator, device=scores.device).clamp_(1e-9, 1 - 1e-9)
    g = -torch.log(-torch.log(u))
    return torch.topk(scores + g, k).indices


# ============================================================================
# 3) THE DECODER  (model-agnostic: driven by logits_fn(text_ids) -> (T, V))
# ============================================================================
@torch.no_grad()
def iterative_decode(logits_fn, T, specials, *, content_len=None, N=32, tau=0.1, temp=5.0,
                     fill_penalty=0.0, seed_bos=True, remask=False, remask_rounds=4, remask_frac=0.15,
                     logits_fn_uncond=None, cfg_w=1.0, generator=None, trace=None):
    """Run OmniVoice-style iterative unmasking. Returns the final token id vector (T,).

    content_len : # of frames to decode (= n_keep). Slots >= content_len are fixed <fill>.
                  None -> decode the whole canvas.
    specials    : dict(mask=, bos=, eos=, pad=, fill=, device=)
    remask      : after the fill-in decode, run `remask_rounds` Mask-Predict refinement passes
                  (re-mask the lowest-confidence `remask_frac` of committed content tokens, then
                  re-predict them with full left+right context) -> self-corrects vowel-mush etc.
    """
    mask_id, fill_id = specials["mask"], specials["fill"]
    dev = specials.get("device", "cpu")
    content_len = T if content_len is None else min(content_len, T)

    ids = torch.full((T,), fill_id, dtype=torch.long, device=dev)   # tail scaffold = <fill>
    ids[:content_len] = mask_id                                     # content region masked
    masked = torch.zeros(T, dtype=torch.bool, device=dev)
    masked[:content_len] = True
    if seed_bos:
        ids[0] = specials["bos"]
        masked[0] = False
    M = int(masked.sum().item())

    for n, k in enumerate(commit_counts(N, tau, M), start=1):
        if k <= 0:
            continue
        logp = F.log_softmax(logits_fn(ids).float(), dim=-1)
        if logits_fn_uncond is not None and cfg_w != 1.0:           # classifier-free guidance
            logp_u = F.log_softmax(logits_fn_uncond(ids).float(), dim=-1)
            logp = logp_u + cfg_w * (logp - logp_u)

        conf, pred = logp.max(dim=-1)                               # value = argmax (greedy)
        if fill_penalty:
            conf = conf - fill_penalty * (pred == fill_id).float()  # don't over-commit <fill>

        cand = masked.nonzero(as_tuple=True)[0]
        chosen = cand[gumbel_top_k(conf[cand] / temp, k, generator=generator)]
        ids[chosen] = pred[chosen]                                  # COMMIT (permanent)
        masked[chosen] = False

        if trace is not None:
            trace(n, k, ids.clone(), masked.clone())

    # ---- Mask-Predict refinement (self-correction): after the canvas is filled, repeatedly
    #      re-mask the lowest-confidence committed CONTENT tokens and re-predict them with full
    #      left+right context.  The per-step commit above is PERMANENT, so without this pass an
    #      early low-confidence guess (e.g. a vowel-mush 'e') can never be revised.
    if remask and content_len > 1:
        content = torch.arange(1, content_len, device=dev)          # skip bos@0 and the <fill> tail
        k_re = max(1, int(round(remask_frac * content.numel())))
        for r in range(remask_rounds):
            cur = F.log_softmax(logits_fn(ids).float(), -1)[content].gather(
                -1, ids[content].unsqueeze(-1)).squeeze(-1)         # conf of each current token
            worst = content[torch.topk(-cur, min(k_re, content.numel())).indices]
            ids[worst] = mask_id                                    # re-mask least-confident
            ids[worst] = F.log_softmax(logits_fn(ids).float(), -1)[worst].argmax(-1)   # re-predict
            if trace is not None:
                trace(N + r + 1, -k_re, ids.clone(), torch.zeros(T, dtype=torch.bool, device=dev))
    return ids


# ============================================================================
# 4) READOUT  (matches infer.py: cut@first </s>, no CTC collapse, '|' -> space)
# ============================================================================
def readout(ids, tok, eos_id):
    seq = ids.tolist()
    if eos_id in seq:
        seq = seq[:seq.index(eos_id)]
    return tok.decode(seq, skip_special_tokens=True, group_tokens=False).strip()


def build_specials(tok, device):
    return dict(mask=tok.convert_tokens_to_ids(MASK_TOKEN), bos=tok.bos_token_id,
                eos=tok.eos_token_id, pad=tok.pad_token_id,
                fill=tok.convert_tokens_to_ids("<fill>"), device=device)


# ============================================================================
# 5) REAL MODEL PATH
# ============================================================================
def make_logits_fn(model, input_values, attention_mask):
    """text_ids (T,) -> logits (T,V). Caches the HuBERT tap via encode_audio/decode_from_tap."""
    tap, skpm = model.encode_audio(input_values, attention_mask)     # ONCE
    def fn(text_ids):
        return model.decode_from_tap(tap, text_ids.unsqueeze(0), skpm).logits[0]
    return fn


def run_real(args):
    import soundfile as sf
    from transformers import Wav2Vec2FeatureExtractor
    from data import load_rows
    from filler_sa_reference import (build_acmlm_tokenizer, build_acmlm_model,
                                     compute_output_length)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    ck = torch.load(ckpt, map_location="cpu")
    cargs = ck.get("args", {}) or {}
    enc = args.model_checkpoint or cargs.get("model_checkpoint") \
        or "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"

    tok = build_acmlm_tokenizer(args.run_dir)                        # 34-token vocab (incl <mask>)
    specials = build_specials(tok, dev)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    model = build_acmlm_model(enc, tok, num_sa_layers=int(cargs.get("num_sa_layers", 8)),
                              use_sinusoidal_pe=bool(cargs.get("pe", True)), device=dev,
                              mask_prob=float(cargs.get("mask_prob", 0.20)),
                              mask_prob_max=float(cargs.get("mask_prob_max", 0.30)),
                              pos_mode=cargs.get("pos_mode", "sinusoidal"))
    missing, unexpected = model.load_state_dict(ck["head"], strict=False)
    bad = [k for k in unexpected] + [k for k in missing
                                     if k.startswith(("sa.", "lm_head.", "text_embed.")) or k == "mask_embed"]
    assert not bad, f"head load mismatch: {bad[:6]}"
    model.eval()
    # DECODE BUDGET must match TRAINING: n_keep = min(T, ceil(dur * fps)).
    # Default fps = the frames_per_sec the model was TRAINED with (from the ckpt args);
    # --fps overrides. fps=50 (the fps50 run) -> ceil(dur*50) >= T -> n_keep = T (decode
    # the whole grid); fps=25 (parent) -> first-half content region, as before.
    fps = args.fps if args.fps is not None else int(cargs.get("frames_per_sec", FRAMES_PER_SEC))
    print(f"[ckpt] {ckpt}  step={ck.get('step')} best_wer={ck.get('best_wer')}  "
          f"decode fps={fps} (train fps={cargs.get('frames_per_sec', '?')})", flush=True)

    rows = load_rows(args.manifest, args.dev_n or None)
    tag = os.path.splitext(os.path.basename(args.manifest))[0]
    out = args.out or os.path.join(args.run_dir, f"omni_infer_{tag}.jsonl")
    t0 = time.time()
    with open(out, "w") as fout:
        for i, r in enumerate(rows):
            arr, sr = sf.read(r["audio_path"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(1)
            iv = fe(arr, sampling_rate=16000, return_tensors="pt")
            input_values = iv["input_values"].to(dev)
            attention_mask = iv.get("attention_mask")
            attention_mask = attention_mask.to(dev) if attention_mask is not None else None
            T = compute_output_length(input_values.shape[-1])
            n_keep = min(T, math.ceil(len(arr) / 16000 * fps))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits_fn = make_logits_fn(model, input_values, attention_mask)
                ids = iterative_decode(logits_fn, T, specials, content_len=n_keep, N=args.steps,
                                       tau=args.tau, temp=args.temp, fill_penalty=args.fill_penalty,
                                       remask=args.remask)
            hyp = readout(ids, tok, specials["eos"])
            fout.write(json.dumps({"audio": r["audio_path"], "ref": r["text"], "hyp": hyp}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  ..{i+1}/{len(rows)}  {(i+1)/(time.time()-t0):.1f} utt/s", flush=True)
    print(f"[done] {len(rows)} utts -> {out}  ({time.time()-t0:.0f}s, N={args.steps} tau={args.tau})")


# ============================================================================
# 6) SELFTEST  (no conda model: mock logits + fake tokenizer)
# ============================================================================
_ID2CH = {0: "'", 27: "|", 28: "<unk>", 29: "<pad>", 30: "<s>", 31: "</s>", 32: "<fill>", 33: "<mask>"}
_ID2CH.update({i: chr(ord('a') + i - 1) for i in range(1, 27)})
_CH2ID = {v: k for k, v in _ID2CH.items()}


class FakeTok:
    bos_token_id, eos_token_id, pad_token_id = 30, 31, 29
    def convert_tokens_to_ids(self, t): return _CH2ID[t]
    def decode(self, ids, skip_special_tokens=True, group_tokens=False):
        specials = {28, 29, 30, 31, 32, 33}
        out = [" " if i == 27 else _ID2CH.get(i, "") for i in ids
               if not (skip_special_tokens and i in specials)]
        return re.sub(r"\s+", " ", "".join(out)).strip()


def framewise_label(text, T, eos_repeat=3):
    ids = [_CH2ID["|"] if c == " " else _CH2ID[c] for c in text.lower()]
    body = [30] + ids + [31] * eos_repeat
    return body + [32] * (T - len(body))


def selftest(args):
    tok = FakeTok()
    target = args.text or "cat"
    body = 1 + len(target) + 3                                       # <s> + chars + </s>x3
    content_len = min(args.T or (body + 5), (args.T or (body + 5)))
    T = args.T or (body + 10)
    y = torch.tensor(framewise_label(target, T))
    specials = dict(mask=33, bos=30, eos=31, pad=29, fill=32, device="cpu")
    print(f"target='{target}'  T={T}  n_keep(content_len)={content_len}")

    gen = torch.Generator().manual_seed(0)
    def mock(text_ids):                                             # high logit on truth; fill extra-confident
        lg = torch.randn(T, 33, generator=gen) * 0.3
        for i, t in enumerate(y.tolist()):
            lg[i, t] += 6.0 + (3.0 if t == 32 else 0.0)
        return lg

    sym = {30: '^', 31: '$', 32: '.', 29: '_'}
    def show(n, k, ids, masked):
        row = "".join("·" if m else (sym.get(i.item(), _ID2CH[i.item()][0])) for i, m in zip(ids, masked))
        print(f"  step {n:2d}  +{k:<3d} committed={int((~masked).sum())}/{T}  |{row}|")

    print(f"\n  tau={args.tau} N={args.steps}  ('·'=<mask> ^=<s> $=</s> .=<fill>)")
    ids = iterative_decode(mock, T, specials, content_len=content_len, N=args.steps, tau=args.tau,
                           temp=args.temp, fill_penalty=args.fill_penalty, remask=args.remask,
                           generator=gen, trace=show)
    hyp = readout(ids, tok, 31)
    print(f"\n  READOUT -> '{hyp}'   {'OK' if hyp == target else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--text", default=None)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl")
    ap.add_argument("--model_checkpoint", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dev_n", type=int, default=0)
    ap.add_argument("--fps", type=int, default=None,
                    help="decode duration budget frames/sec; default = model's training frames_per_sec "
                         "(read from ckpt args). fps=50 -> n_keep=min(ceil(dur*50),T)=T (whole grid).")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--fill_penalty", type=float, default=0.0)
    ap.add_argument("--remask", action="store_true")
    args = ap.parse_args()
    if args.selftest or not args.run_dir:
        selftest(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
