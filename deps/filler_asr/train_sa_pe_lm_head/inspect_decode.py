"""Inspect REAL decodes of a trained SA+PE+lm_head filler head, stage by stage.

For each of the first --n short utts it prints, side by side:
  [FRAMES]
    0 raw full pred           : argmax over the WHOLE padded row (what batch_decode sees)
    1 keep-window (n_keep)     : pred over the graded duration budget = the `full_frames` column
    2 beyond-ceiling leak      : non-<fill> tokens AFTER n_keep  (== the garbage capping removes)
    3 target (label) frames    : the per-frame CE target  ([<s>]+chars+[</s>]+<fill> tail)
  [DECODES]
    A strip, NO collapse, FULL  : tok.batch_decode(pred, group_tokens=False)   <- CURRENT eval/infer hyp (buggy)
    B strip, NO collapse, CAP   : decode(keep window, cut at </s>)             <- TRUE n_keep output
    C strip, CTC COLLAPSE, CAP  : same but group_tokens=True (merges repeats)
  [LOSS] the thing CE is computed on: per-frame (pred|target) over graded frames,
         framewise CE loss for this utt, content-frame acc, fill-frame acc.

Decode of (A) is bit-identical to infer.py / train.evaluate(); loss is filler_sa_reference.framewise_ce_loss.

Launch: see inspect_decode.sh
"""
import os, re, argparse
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from data import load_rows, ManifestDataset
from filler_sa_reference import (FILL_TOKEN, SAMPLING_RATE, build_tokenizer, build_model,
                                 DataCollatorFillerASR, framewise_ce_loss)

DEFAULT_CKPT = "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"
CHARS_TO_IGNORE = r'[\,\?\.\!\-\;\:\"]'
def norm_ref(t): return re.sub(CHARS_TO_IGNORE, "", t).lower().strip()

def lev(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0]*m; ri = ref[i-1]
        for j in range(1, m+1): cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if ri == hyp[j-1] else 1))
        prev = cur
    return prev[m]

def count_sa_layers(sd):
    idx = {int(m.group(1)) for k in sd for m in [re.match(r"sa\.layers\.(\d+)\.", k)] if m}
    return (max(idx) + 1) if idx else 0

def wer(ref, hyp): return lev(ref.split(), hyp.split()) / max(1, len(ref.split()))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", default="/speech/tomson/exps/speech-recog/data/librispeech/librispeech_test_clean.jsonl")
    ap.add_argument("--n", type=int, default=4, help="# utts to inspect")
    ap.add_argument("--max_sec", type=float, default=8.0, help="only inspect clips shorter than this (readability)")
    ap.add_argument("--fps", type=int, default=25, help="duration budget frames/sec (MUST match training=25)")
    ap.add_argument("--show_frames", type=int, default=0, help="truncate frame dumps to first N tokens (0=full)")
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu"); head_sd = ck["head"]; cargs = ck.get("args", {}) or {}
    n_sa = count_sa_layers(head_sd)
    enc = cargs.get("model_checkpoint") or DEFAULT_CKPT
    use_pe = bool(cargs.get("pe", True))
    print(f"[ckpt] {ckpt}\n[arch] sa_layers={n_sa} pe={use_pe} fps={args.fps} "
          f"(step={ck.get('step')} best_wer={ck.get('best_wer')})\n", flush=True)

    tok = build_tokenizer(args.run_dir)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN); eos_id = tok.eos_token_id
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    collator = DataCollatorFillerASR(fe, tok, frames_per_sec=args.fps)   # fps -> graded keep window + loss mask
    def collate(feats):
        b = collator(feats); b["_texts"] = [f["text"] for f in feats]; return b

    model = build_model(enc, tok, num_sa_layers=n_sa, use_sinusoidal_pe=use_pe, device=dev)
    model.load_state_dict(head_sd, strict=False); model.eval()

    # grab enough short clips
    rows = [r for r in load_rows(args.manifest, None)]
    loader = DataLoader(ManifestDataset(rows), batch_size=8, shuffle=False, collate_fn=collate, num_workers=4)

    def toks(ids):  # ids: list[int] -> "a b c | d e"
        s = tok.convert_ids_to_tokens([int(x) for x in ids])
        if args.show_frames: s = s[:args.show_frames] + (["…"] if len(s) > args.show_frames else [])
        return " ".join(s)

    def decode(ids, collapse):
        return tok.decode([int(x) for x in ids], skip_special_tokens=True, group_tokens=collapse).strip()

    shown = 0
    for batch in loader:
        texts = batch.pop("_texts"); labels = batch["labels"]
        inp = {k: v.to(dev) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model(**inp).logits.float()
        pred = lg.argmax(-1).cpu()
        for i in range(len(texts)):
            dur = inp["attention_mask"][i].sum().item() / SAMPLING_RATE
            if dur >= args.max_sec: continue
            ref = norm_ref(texts[i])
            lb = labels[i]
            keep = (lb != -100).nonzero(as_tuple=True)[0]
            n_keep = int(keep.numel()); T = pred.shape[1]
            p = pred[i]
            kp = p[keep]                                   # keep-window pred ids  (== full_frames)
            beyond = p[n_keep:]                            # everything past the budget
            beyond_garbage = [x for x in beyond.tolist() if x != fill_id]
            # cut keep window at first </s>
            hit = (kp == eos_id).nonzero(as_tuple=True)[0]
            cap = kp[:hit[0]] if hit.numel() else kp

            hypA = decode(p, collapse=False)               # FULL row, no collapse  (current buggy hyp)
            hypB = decode(cap, collapse=False)             # capped, no collapse    (true output)
            hypC = decode(cap, collapse=True)              # capped, CTC-collapsed

            # per-utt loss + frame accuracies over graded frames
            m = min(lg.shape[1], labels.shape[1])
            ce = framewise_ce_loss(lg[i:i+1, :m], labels[i:i+1].to(dev)[:, :m], fill_id=fill_id).item()
            g = lb[:m]; pr = p[:m]; graded = g != -100
            corr = (pr == g) & graded
            isf = graded & (g == fill_id); isc = graded & (g != fill_id)
            cont_acc = (corr & isc).sum().item() / max(1, isc.sum().item())
            fill_acc = (corr & isf).sum().item() / max(1, isf.sum().item())
            # content-frame mismatches (ignore fills) for the loss view
            mism = [(int(j), tok.convert_ids_to_tokens(int(pr[j])), tok.convert_ids_to_tokens(int(g[j])))
                    for j in range(m) if isc[j] and pr[j] != g[j]]

            print("="*100)
            print(f"UTT {shown}  dur={dur:.2f}s  T_frames={T}  n_keep={n_keep}  "
                  f"content_frames={int(isc.sum())}  fill_frames={int(isf.sum())}")
            print(f"REF : {ref}")
            print("-"*40 + " FRAMES " + "-"*40)
            print(f"[1] keep-window (=full_frames) : {toks(kp)}")
            print(f"[2] beyond-ceiling leak ({len(beyond_garbage)} non-fill of {beyond.numel()}) : {toks(beyond_garbage)}")
            print(f"[3] target frames (CE label)   : {toks(lb[keep])}")
            print("-"*40 + " DECODES " + "-"*39)
            print(f"[A] strip,  no-collapse, FULL  (current/buggy) : '{hypA}'   WER={wer(ref,hypA):.3f}")
            print(f"[B] strip,  no-collapse, CAP   (TRUE n_keep)   : '{hypB}'   WER={wer(ref,hypB):.3f}")
            print(f"[C] strip,  CTC-collapse, CAP                  : '{hypC}'   WER={wer(ref,hypC):.3f}")
            print("-"*40 + " LOSS VIEW " + "-"*38)
            print(f"framewise CE(this utt)={ce:.3f}  content_acc={cont_acc:.3f}  fill_acc={fill_acc:.3f}")
            print(f"content-frame mismatches (pos pred|target): "
                  + (" ".join(f"{j}:{pp}|{gg}" for j, pp, gg in mism[:30]) or "none") )
            shown += 1
            if shown >= args.n:
                return


if __name__ == "__main__":
    main()
