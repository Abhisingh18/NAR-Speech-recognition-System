"""How often does the PREDICTION actually contain </s>, and where does it land vs n_keep?"""
import os, re, argparse, math
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
from data import load_rows, ManifestDataset
from filler_sa_reference import FILL_TOKEN, SAMPLING_RATE, build_tokenizer, build_model, DataCollatorFillerASR

DEFAULT_CKPT = "/speech/tomson/filler_asr/models/hubert-xlarge-ls960-ft"
def count_sa(sd):
    idx = {int(m.group(1)) for k in sd for m in [re.match(r"sa\.layers\.(\d+)\.", k)] if m}
    return (max(idx)+1) if idx else 0

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--manifest", default="/speech/tomson/filler_asr/data/test_clean.jsonl")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    ckpt = args.ckpt or os.path.join(args.run_dir, "best.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu"); sd = ck["head"]
    tok = build_tokenizer(args.run_dir)
    fill_id = tok.convert_tokens_to_ids(FILL_TOKEN); eos_id = tok.eos_token_id
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    col = DataCollatorFillerASR(fe, tok, frames_per_sec=args.fps)
    model = build_model(ck.get("args",{}).get("model_checkpoint") or DEFAULT_CKPT, tok,
                        num_sa_layers=count_sa(sd), use_sinusoidal_pe=bool(ck.get("args",{}).get("pe",True)), device=dev)
    model.load_state_dict(sd, strict=False); model.eval()

    rows = load_rows(args.manifest, None)
    loader = DataLoader(ManifestDataset(rows), batch_size=16, shuffle=False, collate_fn=col, num_workers=4)
    N=no_eos=eos_in_keep=eos_after_keep=leak_after_eos=garbage_no_eos=0
    for batch in loader:
        labels = batch.pop("labels")
        inp = {k:v.to(dev) for k,v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = model(**inp).logits.float()
        pred = lg.argmax(-1).cpu()
        for i in range(pred.shape[0]):
            N += 1
            p = pred[i]
            n_keep = int((labels[i] != -100).sum())
            hit = (p == eos_id).nonzero(as_tuple=True)[0]
            if hit.numel()==0:
                no_eos += 1
                # any non-fill char in the pad region past n_keep? (leak that capping CANNOT remove)
                if (p[n_keep:] != fill_id).any(): garbage_no_eos += 1
            else:
                e = int(hit[0])
                if e < n_keep: eos_in_keep += 1
                else: eos_after_keep += 1
                if (p[e+1:] != fill_id).any(): leak_after_eos += 1
    pct = lambda x: f"{x} ({100*x/N:.1f}%)"
    print(f"\nrun={os.path.basename(args.run_dir)}  set={os.path.basename(args.manifest)}  N={N}")
    print(f"  </s> predicted somewhere : {pct(N-no_eos)}")
    print(f"     - within n_keep       : {pct(eos_in_keep)}")
    print(f"     - only AFTER n_keep   : {pct(eos_after_keep)}")
    print(f"     - non-fill after </s> (leak capping removes) : {pct(leak_after_eos)}")
    print(f"  NO </s> at all           : {pct(no_eos)}")
    print(f"     - of those, garbage past n_keep (cap falls back to full row) : {pct(garbage_no_eos)}")

if __name__ == "__main__":
    main()
