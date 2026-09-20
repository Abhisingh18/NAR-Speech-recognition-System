"""Root-cause diagnosis of long-audio degradation for an A-CMLM decode.

For each clip we parse the raw per-frame prediction (`full`) and locate:
    expected_eos = len(ref)+1        # target </s> index (1 char/frame at head + <s>)
    actual_eos   = index of FIRST </s> in the frames (None = never emitted)
    fill_in_body = # of <fill> frames BEFORE actual_eos (or before n_keep if no eos)
and classify the dominant failure:
    OK           : WER < 5%
    EARLY_EOS    : </s> fired at < 0.85*expected  -> tail truncated (deletions)
    NO_EOS       : </s> never fired -> <fill>+garbage leak into hyp
    FILL_INTRUDE : >=5 <fill> frames inside the body -> mid-sentence gaps
    DRIFT        : eos ~ on time but WER>=5% -> spelling/substitution drift

Usage: python diagnose_long.py --decode_dir <dir> [--start 10]
"""
import os, re, json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode_dir", required=True)
    ap.add_argument("--start", type=int, default=10)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(args.decode_dir, "all_utts.jsonl"))]
    rows = [r for r in rows if float(r['dur']) >= args.start]

    for r in rows:
        toks = r['full'].split(); nk = int(r['n_keep'])
        exp = len(r['ref']) + 1
        eos = next((i for i, t in enumerate(toks) if t == '</s>'), None)
        body_end = eos if eos is not None else nk
        r['_exp'] = exp
        r['_eos'] = eos
        r['_fill_body'] = sum(1 for t in toks[:body_end] if t == '<fill>')
        wer = float(r['utt_wer']) * 100
        if eos is None:
            cat = 'NO_EOS'
        elif eos < 0.85 * exp:
            cat = 'EARLY_EOS'
        elif r['_fill_body'] >= 5:
            cat = 'FILL_INTRUDE'
        elif wer < 5:
            cat = 'OK'
        else:
            cat = 'DRIFT'
        r['_cat'] = cat
        r['_wer'] = wer

    # --- category rollup ---
    cats = ['OK', 'DRIFT', 'EARLY_EOS', 'NO_EOS', 'FILL_INTRUDE']
    print(f"subset >= {args.start}s : {len(rows)} clips\n")
    print(f"{'category':<14}{'clips':>6}{'share':>8}{'meanWER':>9}{'words':>8}{'errW':>7}{'%oferr':>8}")
    tot_err = sum(int(r['w_err']) for r in rows)
    for c in cats:
        sub = [r for r in rows if r['_cat'] == c]
        if not sub: continue
        ew = sum(int(r['w_err']) for r in sub); ww = sum(int(r['w_tot']) for r in sub)
        mw = sum(r['_wer'] for r in sub) / len(sub)
        print(f"{c:<14}{len(sub):>6}{len(sub)/len(rows)*100:>7.1f}%{mw:>8.1f}%{ww:>8}{ew:>7}{ew/max(1,tot_err)*100:>7.1f}%")

    # --- eos timing by duration bin ---
    print(f"\n{'dur':>5}{'clips':>6}{'exp_eos':>8}{'act_eos':>8}{'act/exp':>8}{'%early':>7}{'%noeos':>7}{'fillbody':>9}")
    bins = {}
    for r in rows:
        bins.setdefault(int(float(r['dur'])), []).append(r)
    for b in sorted(bins):
        sub = bins[b]; n = len(sub)
        exp = sum(r['_exp'] for r in sub) / n
        withe = [r for r in sub if r['_eos'] is not None]
        act = sum(r['_eos'] for r in withe) / len(withe) if withe else float('nan')
        ratio = act / exp if withe else float('nan')
        early = sum(r['_cat'] == 'EARLY_EOS' for r in sub) / n * 100
        noeos = sum(r['_eos'] is None for r in sub) / n * 100
        fb = sum(r['_fill_body'] for r in sub) / n
        act_s = f"{act:>8.0f}" if withe else f"{'--':>8}"
        rat_s = f"{ratio:>7.2f}" if withe else f"{'--':>7}"
        print(f"{str(b)+'s':>5}{n:>6}{exp:>8.0f}{act_s}{rat_s} {early:>6.0f}%{noeos:>6.0f}%{fb:>9.1f}")

if __name__ == "__main__":
    main()
