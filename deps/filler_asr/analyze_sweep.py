"""Same <fill>/</s> target breakdown across (dataset, fps) combos.

Reads each manifest's (dur, T, L) ONCE (audio-header read is the slow part),
then evaluates every fps budget instantly from the cache.
"""
import sys, re, json, math, argparse
import numpy as np
sys.path.insert(0, '/speech/tomson/filler_asr')
import soundfile as sf
from filler_sa_reference import compute_output_length, CHARS_TO_IGNORE_REGEX

SR = 16000
_rx = re.compile(CHARS_TO_IGNORE_REGEX)
def norm_len(t): return len(_rx.sub('', t).lower().replace(' ', '|'))

def load_cache(path, limit=0):
    dur, T, L = [], [], []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        ns = sf.info(d.get('audio') or d.get('source')).frames
        dur.append(ns / SR)
        T.append(compute_output_length(ns))
        L.append(norm_len(d.get('text') or d.get('target') or ''))
        if limit and len(dur) >= limit: break
    return np.array(dur), np.array(T, dtype=np.int64), np.array(L, dtype=np.int64)

def summarize(cache, fps):
    dur, T, L = cache
    n_keep = np.ceil(dur * fps).astype(np.int64)
    graded_total = np.minimum(n_keep, T)
    prefix = np.minimum(L + 2, T)          # <s> + L chars + </s>
    eos_pos = prefix - 1
    eos_graded = eos_pos < graded_total
    graded_content = np.minimum(prefix, graded_total)
    graded_fill = graded_total - graded_content
    return {
        "n": len(dur),
        "eos_not_graded": int((~eos_graded).sum()),
        "zero_fill": int((graded_fill == 0).sum()),
        "fill_pct_overall": graded_fill.sum() / graded_total.sum() * 100,
        "fill_frac_med": float(np.median(graded_fill / graded_total)) * 100,
        "gfill_med": float(np.median(graded_fill)),
        "gfill_p99": float(np.percentile(graded_fill, 99)),
        "nkeep_med": float(np.median(n_keep)),
        "cps_med": float(np.median(L / dur)),
        "dur_med": float(np.median(dur)),
    }

SETS = {
    "train_960h":  "/speech/tomson/filler_asr/data/train_960h.jsonl",
    "dev_clean":   "/speech/tomson/filler_asr/data/dev_clean.jsonl",
    "dev_other":   "/speech/tomson/filler_asr/data/dev_other.jsonl",
    "test_clean":  "/speech/tomson/filler_asr/data/test_clean.jsonl",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    caches = {name: load_cache(path, args.limit) for name, path in SETS.items()}

    print("\n############ TABLE A — all sets at fps=25 (the trained budget) ############")
    hdr = f"{'set':<12}{'n':>7}{'dur_med':>9}{'cps_med':>9}{'fill%_all':>11}{'fillfrac_med':>14}{'gfill_med':>11}{'eos_miss':>10}"
    print(hdr); print("-"*len(hdr))
    for name, cache in caches.items():
        s = summarize(cache, 25)
        print(f"{name:<12}{s['n']:>7}{s['dur_med']:>9.2f}{s['cps_med']:>9.2f}{s['fill_pct_overall']:>10.2f}%"
              f"{s['fill_frac_med']:>13.1f}%{s['gfill_med']:>11.0f}{s['eos_not_graded']:>7}"
              f" ({s['eos_not_graded']/s['n']*100:.3f}%)")

    print("\n############ TABLE B — fps budget sweep ############")
    for name in ["train_960h", "test_clean"]:
        cache = caches[name]
        print(f"\n--- {name} (n={len(cache[0])}) ---")
        hdr = f"{'fps':>5}{'nkeep_med':>11}{'fill%_all':>11}{'fillfrac_med':>14}{'gfill_med':>11}{'gfill_p99':>11}{'eos_miss':>11}{'zero_fill':>11}"
        print(hdr); print("-"*len(hdr))
        for fps in [15, 20, 25, 30, 40, 50]:
            s = summarize(cache, fps)
            print(f"{fps:>5}{s['nkeep_med']:>11.0f}{s['fill_pct_overall']:>10.2f}%{s['fill_frac_med']:>13.1f}%"
                  f"{s['gfill_med']:>11.0f}{s['gfill_p99']:>11.0f}"
                  f"{s['eos_not_graded']:>7} ({s['eos_not_graded']/s['n']*100:4.2f}%)"
                  f"{s['zero_fill']:>7} ({s['zero_fill']/s['n']*100:4.2f}%)")
