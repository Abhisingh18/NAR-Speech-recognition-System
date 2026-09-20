"""Frame/character heuristics across test, dev and train full-dumps.

Each model frame (~20 ms of HuBERT output) emits ONE token. The model emits a
content region (<s> ... text ... </s>) followed (mostly) by a trailing run of
<fill>. Before passing to the projector we strip <fill> and keep only the
content frames. This script answers, per split and combined (ALL):

  1. How many frames/chars come BEFORE the first <fill>  (the content budget).
  2. How many content frames are "must required" per REFERENCE char, then per
     HYPOTHESIS char  (frames-per-char).
  3. The 1-second heuristic: how many characters does 1 s of audio yield -- the
     floor (min, "at least") and the ceiling (max, "budget for").

Token classes (vocab.json):
  specials  : <pad> <s> </s> <unk> <fill>
  delimiter : |              (word boundary)
  letters   : a-z '          (actual text characters)

Per utterance we measure:
  T            = num_frames
  n_fill       = num_fill
  content      = T - n_fill            (non-fill frames = what the projector sees
                                         if you strip only <fill>)
  first_fill   = index of first <fill> (= frames before <fill> appears; T if none)
  letters      = # frames that are an a-z/' token (pure text chars)
  audio_sec, len_ref, len_hyp          (from the dump)

Run:
  python analyze_frame_heuristics.py
"""
import os
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

SPLITS = [
    ("test_clean", os.path.join(RES, "test_clean_fulldump.jsonl")),
    ("dev_clean",  os.path.join(RES, "dev_clean_fulldump.jsonl")),
    ("dev_other",  os.path.join(RES, "dev_other_fulldump.jsonl")),
    ("train_960h", os.path.join(RES, "train_960h_fulldump.jsonl")),
]

FILL = "<fill>"
LETTERS = set("abcdefghijklmnopqrstuvwxyz'")


def scan(path):
    """Stream one jsonl dump -> dict of per-utterance numpy arrays."""
    T, nfill, content, first_fill, letters = [], [], [], [], []
    audio, lref, lhyp = [], [], []
    trailing_pure = 0
    n = 0
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        toks = d["frame_tokens"]
        t = len(toks)
        nf = d["num_fill"]
        try:
            ff = toks.index(FILL)
        except ValueError:
            ff = t                                  # no <fill> at all
        let = sum(1 for x in toks if x in LETTERS)
        # is every <fill> a single trailing block?  (content | <fill>...<fill>)
        if ff + nf == t and all(x == FILL for x in toks[ff:]):
            trailing_pure += 1

        T.append(t); nfill.append(nf); content.append(t - nf)
        first_fill.append(ff); letters.append(let)
        audio.append(d["audio_len_sec"]); lref.append(d["len_ref"]); lhyp.append(d["len_hyp"])
        n += 1

    return {
        "n": n,
        "T": np.array(T, float),
        "n_fill": np.array(nfill, float),
        "content": np.array(content, float),
        "first_fill": np.array(first_fill, float),
        "letters": np.array(letters, float),
        "audio": np.array(audio, float),
        "len_ref": np.array(lref, float),
        "len_hyp": np.array(lhyp, float),
        "trailing_pure": trailing_pure,
    }


def merge(parts):
    keys = ["T", "n_fill", "content", "first_fill", "letters", "audio", "len_ref", "len_hyp"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["n"] = sum(p["n"] for p in parts)
    out["trailing_pure"] = sum(p["trailing_pure"] for p in parts)
    return out


def stat(a):
    return dict(mean=a.mean(), std=a.std(), min=a.min(),
                p1=np.percentile(a, 1), p5=np.percentile(a, 5),
                p50=np.percentile(a, 50), p95=np.percentile(a, 95),
                p99=np.percentile(a, 99), max=a.max())


def row(name, a):
    s = stat(a)
    return (f"  {name:<26} mean={s['mean']:8.2f}  min={s['min']:8.2f}  "
            f"p1={s['p1']:8.2f}  p50={s['p50']:8.2f}  p99={s['p99']:8.2f}  "
            f"max={s['max']:8.2f}")


def report(name, D, out):
    safe_ref = np.maximum(1.0, D["len_ref"])
    safe_hyp = np.maximum(1.0, D["len_hyp"])
    aud = D["audio"]

    out.append(f"\n{'='*100}\n{name}   (n={D['n']} utts)   "
               f"fill-purely-trailing: {D['trailing_pure']}/{D['n']} "
               f"({100*D['trailing_pure']/max(1,D['n']):.2f}%)\n{'='*100}")

    out.append("[1] Frames/characters BEFORE <fill> appears  (content budget, 1 frame = 1 char):")
    out.append(row("first_fill (frames)", D["first_fill"]))
    out.append(row("content = T - n_fill", D["content"]))
    out.append(row("letters only (a-z')", D["letters"]))

    out.append("\n[2] Frames 'must required' per text character  (content frames / chars):")
    out.append(row("content / len_REF", D["content"] / safe_ref))
    out.append(row("content / len_HYP", D["content"] / safe_hyp))
    out.append(row("first_fill / len_REF", D["first_fill"] / safe_ref))

    out.append("\n[3] Characters per 1 s of audio  (min = floor / 'at least', max = budget):")
    out.append(row("len_REF / sec", D["len_ref"] / aud))
    out.append(row("len_HYP / sec", D["len_hyp"] / aud))
    out.append(row("content frames / sec", D["content"] / aud))
    out.append(row("letters / sec", D["letters"] / aud))


def main():
    parts = {}
    out = []
    for name, path in SPLITS:
        print(f"scanning {name} ...", flush=True)
        parts[name] = scan(path)
        report(name, parts[name], out)

    ALL = merge(list(parts.values()))
    report("ALL (test + dev_clean + dev_other + train)", ALL, out)

    # Headline floors/ceilings on the combined set.
    aud = ALL["audio"]
    ref_ps = ALL["len_ref"] / aud
    hyp_ps = ALL["len_hyp"] / aud
    con_ps = ALL["content"] / aud
    out.append("\n" + "#" * 100)
    out.append("HEADLINE (combined ALL):")
    out.append(f"  Reference  : 1 s of audio yields at least {ref_ps.min():.2f} chars "
               f"(p1={np.percentile(ref_ps,1):.2f}), mean {ref_ps.mean():.2f}, up to {ref_ps.max():.2f}")
    out.append(f"  Hypothesis : 1 s of audio yields at least {hyp_ps.min():.2f} chars "
               f"(p1={np.percentile(hyp_ps,1):.2f}), mean {hyp_ps.mean():.2f}, up to {hyp_ps.max():.2f}")
    out.append(f"  Content frames (non-fill kept for projector): min {con_ps.min():.2f}/s, "
               f"mean {con_ps.mean():.2f}/s, max {con_ps.max():.2f}/s")
    out.append("#" * 100)

    text = "\n".join(out)
    print(text)
    with open(os.path.join(RES, "frame_heuristics.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nsaved -> {os.path.join(RES, 'frame_heuristics.txt')}")


if __name__ == "__main__":
    main()
