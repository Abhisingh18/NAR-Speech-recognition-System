"""
Re-score the filler-ASR W&B sample dumps with the output CAPPED to the duration
ceiling (n_keep) -- i.e. cut the hypothesis at the first </s> and drop everything
after it (the beyond-ceiling frames that leak into batch_decode as trailing garbage).

For each utterance we compare:
   old_hyp  : the hyp_stripped column from the W&B table
              == tok.batch_decode(pred over the WHOLE padded sequence) with specials+<fill>
                 removed  -> includes hallucinated chars emitted PAST n_keep
   cap_hyp  : decode of full_frames (the graded keep-window) cut at first </s>
              == the duration-ceiling-capped output the user wants

WER/CER are the same Levenshtein the training loop uses (infer.py / train.py).
3 runs (SA depth recovered from the [setup] lines of the 30ep logs):
   ep8.csv -> SA-8  @ step 24000   (dev WER logged 0.300, 500 utts)
   4ep.csv -> SA-6  @ step 30000   (dev WER logged 0.340, 500 utts)
   ep6.csv -> SA-4  @ step 24000   (dev WER logged 0.382, 500 utts)
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
SPECIALS = {"<s>", "</s>", "<pad>", "<unk>", "<fill>"}

# ---- Levenshtein identical to infer.py / train.py ----
def lev(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0: return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m; ri = ref[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ri == hyp[j - 1] else 1))
        prev = cur
    return prev[m]

def frames_to_words(full_prefix):
    """Reproduce batch_decode(skip_special, group_tokens=False) but on the keep window,
    CUT at the first </s>. char tokens concatenated, '|' -> space, specials dropped."""
    toks = full_prefix.split()
    out = []
    for t in toks:
        if t == "</s>":          # duration-ceiling cap: stop at end-of-sentence
            break
        if t in SPECIALS:        # drop <s>/<pad>/<unk>/<fill>
            continue
        out.append(" " if t == "|" else t)
    return "".join(out).strip()

REFS = [
 "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
 "nor is mister quilter's manner less interesting than his matter",
 "he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca",
 "it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression",
 "on the general principles of art mister quilter writes with equal lucidity",
 "painting he tells us is of a different quality to mathematics and finish in art is adding more fact",
 "as for etchings they are of two kinds british and foreign",
 "near the fire and the ornaments fred brought home from india on the mantel board",
 "only unfortunately his own work never does get good",
 "mister quilter has missed his chance for he has failed even to make himself the tupper of painting",
]

# full_frames content-prefix (everything up to the first <fill>) per run, in utt order
FULL = {
"SA-8 (ep8 @24k)": [
 "<s> m i s t e r | q u i l t e r | i s | t h e | a p o s t l e | o f | t h e | m i d d l e | c l a s s e s | a n d | w e | a r e | g l a d | t o | w e l c o m e | h i s | g o s p e l </s>",
 "<s> n o r | i s | m i s t e r | q u i l t e r ' s | m a n n e r | l e s s | i n t e r e s t i n g | t h a n | h i s | m a t t e r </s>",
 "<s> h e | h a s | g r a v e | d o u b t s | w h e t h e r | s i r | f r e d e r i c k | l a y g h o o ' s s w o r k | i s | r e a a l l y g g r e e k | a f t e r | a l l | a n d | a n n | d i c c o v e r | i | i i t | u u t | l i t t l e | o f | r o c k y | i t h a a </s>",
 "<s> i t | i s | o b v i o u s l y | u n n e c e s s a r y | f o r | u s | t o | p o i n t | o u t | h o w | l u m i n o u s | t h e s e | c r i t i c i m s s | r r e h h w w | e l i c a t e | i n | e x p r e s s i o n </s>",
 "<s> o n | t h e | g e n e r a l | p r i n c i p l e s | o f | a r t | m i s t e r | q u i l t e r | w r i t e s | w i t h | e q u a l | l u c i d i t y </s>",
 "<s> p a i n t i n g | h e | t e l l s | u s | i s | o f | a | d i f f e r e n t | q u a l i t y | t o | m a t h e m a t i c s | a n d | f i n i s h | i n | a r t | i s | a d d i n g | m o r e | f a c t </s>",
 "<s> a s | f o r | e t c h i n g s | t h e y | a r e | o f | t w o | k i n d s | b r i t i s h | a n d | f o r e i g n </s>",
 "<s> n e a r | t h e | f i r e | a n d | t h e | o r n a m e n t s | f r e d | b r o u g h t | h o m e | f r o m | i n d i a | o n | t h e | m a n t a l | b o a r d </s>",
 "<s> o n l y | u n f o r t u n a t e l y | h i s | o w n | w o r k | n e v e r | d o e s | g e t | g o o d </s>",
 "<s> m i s t e r | q u i l t e r | h a s | m i s s e d | h i s | c h a n c e | f o r | h e | h a s | f a i l e d | e v e n | t o | m a k e | h i m s e l f | t h e | t u p p e r | o f | p a i n t i n g </s>",
],
"SA-6 (4ep @30k)": [
 "<s> m i s t e r | q u i l t e r | i s | t h e | a p o s t l e | o f | t h e | m i d d l e | c l a s s e s | a n d w w e | a r e | g l a d | o | | w e l c o m e | h i s | g o s p e l </s>",
 "<s> n o r | i s | m i s t e r | q u i l t e r ' s | m a n n e r | l e s s | i n t e r e s t i n g | t h a n | h i s | m a d e r </s> </s>",
 "<s> h e | h a s | g r a v e | d o u b t s | w h e t h e r | s i r | f r e d e r i c k | l a y h t o n ' s | w o r k | i s r r e a l y y | r r e e k | a f t e r | a l l a a d | c a n | d i c o v v r | | n n | i t | b u t | l i t t l e | o f | r o c k y | t t h a a a",
 "<s> i t | i s | o b v i o u s l y | u n e e e e s s a r | f o r | u s | t t | p o i n t | o u t | h o w | l u m i o o u s | t h s e | c r i t i c i i m m | | r r e h o w | d e l i c a t e | i n | e x p r e s s i o n </s>",
 "<s> o n | t h e | g e n e r a l | p r i n c i p l e s | o f | a r t | m i s t e r | q u i l t e r | w r i t e s | w i t h | e q u a l | l u c i d i t y </s>",
 "<s> p a i n t i n g | h e | t e l l s | u s | i s | o f | a | i i f e r e n t | q u a a l i t y | t o | m t t e m a t i c s | a n d | f i n i s h | i n | a r t | i s | a d d i n g | m o r e | f a c </s>",
 "<s> a s | f o r | e t c h i n g s | t h e y a r e | o f | t t o o | k i n d s | b r i i s h | a n d | f o r e i g n </s>",
 "<s> n e a r | t h e | f i r e | a n d | t h e | o r n a m e t s s | f r d | b r o u g h t | h o m e e | f o o m | i n d i a | o n | t h e m m a n t a l | b o a r d </s>",
 "<s> o n l y | u n f o r t u n a t e l y | h i s | o w n | o r k k | n e v e r | d o e s | g e | | g o d </s>",
 "<s> m i s t e r | k u i l t e r | h a s | m i s s e d | h i s | c h a n c e | f o r | h e | h a s f a i l e d | e v e n | t o | m a k e | h i m s e l f | t h h | | t u p p e r | o f | p a i t i i g </s>",
],
"SA-4 (ep6 @24k)": [
 "<s> m i s t e r | q u i l t e r | i s | t h e | a p o s t l e | o f | t h e | m i d d l e | c l a s s e | | n d | w e | a r e | g l a d | t o | w e l c o m e | h i s | g o s p e l </s>",
 "<s> n o r | i s | m i s t e r | q u i l t e r ' s | m a n n e r | l e s s | i n t e r e t i n g | t h a n | h i s | | m a d d e </s>",
 "<s> h e | h a s | g r a v e | d o u b t s | w h e t h e r | s i r | f r e d e r i c | l l y t t o n ' s | w o r k | i s | r e a l l y | g r e e k | f t t e r a l l | a n d | c a n | d i s c o v e r | i n | i | b u t t | l i t t l | | o f | r c k y | | t h h a c </s>",
 "<s> i t | i s | o b v i o u s l y | u n e c e s s a r y | f o r | u s | t o | p o i n t | o u t | h o w | l m m i o u s | t h e s s | c c i t i c i s m s | a r e | h o w | d e l i c c t e | i n | e x p r e s s i o n </s>",
 "<s> o n | t h e | g e n e r a l | p r i n c i p l e s | o f | a r t | m i s t e r | q u i l t e r | w r i t e s | w i t h e e u u l l l u c c i i t y </s>",
 "<s> p a i n t i n g | h e | t e l l s | u s | i s | o f | a | d i f e r e n t | | q a l i t y | t o | m a t h e m a t i c s | a n d | f i n s h | i n | a a r | i i s | a d d i n g | m r e | f a a c </s>",
 "<s> a s | f o r | e t c h i n g s | t h e y | a r e | o f | t w o | k i n d s | b r i t i s h | a n d f f r e e i g </s>",
 "<s> n e a r | t h e | f i r e | a n d | t h e | o r n m e e t s | f r e d | b r o u g h t | h o m e | f r o m | i n d i a | o n | t h e | m a n t a l | b o a r d </s>",
 "<s> o n l y | u n f o r t n a t e l y | h i s | o w n | w o r k | n e v e r | d o e s | g e t | g o o d </s>",
 "<s> m i s t e r | q u i l t e r | h a s | m i s s e d | h i s | c h a n c e | f o r | h e | h a s f a i l e d | e v e n | t o | m a k e | h i m s e l f | t h e | t u p p e r | o f | p i n t i n g </s>",
],
}

# old hyp_stripped column (the leaked/uncapped decode) per run, utt order
OLD = {
"SA-8 (ep8 @24k)": [
 "mister quilter is the apostle of the middle classes and we are glad to welcome his gospelsi isti",
 "nor is mister quilter's manner less interesting than his matter",
 "he has grave doubts whether sir frederick layghoo'sswork is reaallyggreek after all and ann diccover i iit uut little of rocky ithaa s  s   anal r llnhlalalllananl anal an alantllltwllts woslawall'stttllfeawtlllll  aaluuubuu tahooubleub''onn'woubbtattothaete habaytttn'ss oubbralaltwsuets woubtkaabbut btubnswhubbb doubbs oubbt  oubbs obu",
 "it is obviously unnecessary for us to point out how luminous these criticimss rrehhww elicate in expression ieeiiirreweeewneoreerrrsearrrrarrreeseesysimuiniyyisyyyn syy syyytyy yut sy mscysmssry  ssasy msicsmssyy uuly  lucly yt ly  slyy",
 "on the general principles of art mister quilter writes with equal lucidityr",
 "painting he tells us is of a different quality to mathematics and finish in art is adding more factrlsellselllseelll",
 "as for etchings they are of two kinds british and foreign",
 "near the fire and the ornaments fred brought home from india on the mantal board",
 "only unfortunately his own work never does get good",
 "mister quilter has missed his chance for he has failed even to make himself the tupper of paintingngn inasseshass smista hs",
],
"SA-6 (4ep @30k)": [
 "mister quilter is the apostle of the middle classes andwwe are glad o  welcome his gospel",
 "nor is mister quilter's manner less interesting than his mader",
 "he has grave doubts whether sir frederick layhton's work isrrealyy rreek after allaad can dicovvr  nn it but little of rocky tthaaaeerrrrrrrrrrr",
 "it is obviously uneeeessar for us tt point out how lumioous thse criticiimm  rrehow delicate in expressionsssss",
 "on the general principles of art mister quilter writes with equal lucidity",
 "painting he tells us is of a iiferent quaality to mttematics and finish in art is adding more fac",
 "as for etchings theyare of ttoo kinds briish and foreign",
 "near the fire and the ornametss frd brought homee foom india on themmantal board",
 "only unfortunately his own orkk never does ge  god",
 "mister kuilter has missed his chance for he hasfailed even to make himself thh  tupper of paitiig",
],
"SA-4 (ep6 @24k)": [
 "mister quilter is the apostle of the middle classe  nd we are glad to welcome his gospel",
 "nor is mister quilter's manner less intereting than his  madde",
 "he has grave doubts whether sir frederic llytton's work is really greek ftterall and can discover in i butt littl  of rcky  thhacoooddoodddddddddooooddoodddoe ddoooddoooddddool   worrrr do wwl oaeeeee do g  rrda dowaee   dd",
 "it is obviously unecessary for us to point out how lmmious thess cciticisms are how deliccte in expressionyiiiuiisoobooous eeoooooooouis ob",
 "on the general principles of art mister quilter writes witheeuulllucciity",
 "painting he tells us is of a diferent  qality to mathematics and finsh in aar iis adding mre faacee",
 "as for etchings they are of two kinds british andffreeig",
 "near the fire and the ornmeets fred brought home from india on the mantal board",
 "only unfortnately his own work never does get good",
 "mister quilter has missed his chance for he hasfailed even to make himself the tupper of pinting",
],
}

RUNS = ["SA-4 (ep6 @24k)", "SA-6 (4ep @30k)", "SA-8 (ep8 @24k)"]   # ascending depth
LOGGED_DEV_WER = {"SA-4 (ep6 @24k)": 0.382, "SA-6 (4ep @30k)": 0.340, "SA-8 (ep8 @24k)": 0.300}

# ---------------- score ----------------
rows = []
agg = {}
for run in RUNS:
    we_o = we_c = ce_o = ce_c = wtot = ctot = 0
    se_o = se_c = 0
    for i, ref in enumerate(REFS):
        old = OLD[run][i].strip()
        cap = frames_to_words(FULL[run][i])
        rw, ow, cw = ref.split(), old.split(), cap.split()
        ewo, ewc = lev(rw, ow), lev(rw, cw)
        eco, ecc = lev(ref, old), lev(ref, cap)
        we_o += ewo; we_c += ewc; ce_o += eco; ce_c += ecc
        wtot += len(rw); ctot += len(ref)
        se_o += int(old != ref); se_c += int(cap != ref)
        rows.append({
            "run": run, "utt": i + 1,
            "ref_words": len(rw),
            "old_wer": ewo / len(rw), "cap_wer": ewc / len(rw),
            "old_cer": eco / len(ref), "cap_cer": ecc / len(ref),
            "trailing_words_removed": len(ow) - len(cw),
            "cap_hyp": cap, "old_hyp": old, "ref": ref,
        })
    agg[run] = {
        "WER_old": we_o / wtot, "WER_cap": we_c / wtot,
        "CER_old": ce_o / ctot, "CER_cap": ce_c / ctot,
        "SER_old": se_o / len(REFS), "SER_cap": se_c / len(REFS),
        "logged_dev_wer": LOGGED_DEV_WER[run],
    }

# ---------------- write per-utt cleaned csv ----------------
with open(os.path.join(OUT, "cleaned_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "utt", "ref_words", "old_wer", "cap_wer",
                                      "old_cer", "cap_cer", "trailing_words_removed",
                                      "ref", "cap_hyp", "old_hyp"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---------------- console report ----------------
print("\n================ AGGREGATE (10 short dev utts) ================")
print(f"{'run':<18}{'WER_old':>9}{'WER_cap':>9}{'Δabs':>8}{'Δrel%':>8}"
      f"{'CER_old':>9}{'CER_cap':>9}{'SER_old':>9}{'SER_cap':>9}{'devWER*':>9}")
for run in RUNS:
    a = agg[run]
    drel = 100 * (a["WER_old"] - a["WER_cap"]) / a["WER_old"]
    print(f"{run:<18}{a['WER_old']:>9.3f}{a['WER_cap']:>9.3f}"
          f"{a['WER_old']-a['WER_cap']:>8.3f}{drel:>8.1f}"
          f"{a['CER_old']:>9.3f}{a['CER_cap']:>9.3f}"
          f"{a['SER_old']:>9.3f}{a['SER_cap']:>9.3f}{a['logged_dev_wer']:>9.3f}")
print("* devWER = WER logged during training on 500 dev utts (uncapped decode)")

print("\n================ PER-UTT (capped WER, %) ================")
print(f"{'utt':>4} {'ref_w':>5} | " + " | ".join(f"{r.split()[0]:>5}" for r in RUNS) + "   removed(old trailing words)")
for i in range(len(REFS)):
    line = f"{i+1:>4} {len(REFS[i].split()):>5} | "
    cells = []
    rem = []
    for run in RUNS:
        rr = next(r for r in rows if r["run"] == run and r["utt"] == i + 1)
        cells.append(f"{100*rr['cap_wer']:>5.0f}")
        rem.append(rr["trailing_words_removed"])
    print(line + " | ".join(cells) + "   " + str(rem))

# ---------------- PLOTS ----------------
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

# 1) training curves from the logs (uncapped dev WER)
CURVES = {
"SA-8": ([0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000],
         [1.000,0.947,0.854,0.764,0.741,0.811,0.776,0.612,0.574,0.571,0.354,0.254,0.300]),
"SA-6": ([0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000],
         [0.999,0.937,0.894,0.834,0.733,0.794,0.699,0.618,0.594,0.568,0.501,0.458,0.451,0.414,0.379,0.340]),
"SA-4": ([0,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000],
         [1.000,0.955,0.922,0.800,0.787,0.723,0.638,0.565,0.594,0.561,0.455,0.446,0.382]),
}
CONTENT_ACC = {
"SA-8": [0.017,0.287,0.280,0.457,0.450,0.429,0.589,0.645,0.710,0.710,0.813,0.820,0.831],
"SA-6": [0.024,0.239,0.362,0.297,0.486,0.421,0.499,0.472,0.542,0.513,0.551,0.612,0.585,0.568,0.611,0.626],
"SA-4": [0.025,0.258,0.282,0.423,0.394,0.426,0.526,0.519,0.384,0.317,0.488,0.405,0.464],
}
COL = {"SA-8": "#1b9e77", "SA-6": "#d95f02", "SA-4": "#7570b3"}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k in ["SA-8", "SA-6", "SA-4"]:
    xs, ys = CURVES[k]
    ax[0].plot(xs, ys, "-o", ms=3, color=COL[k], label=k)
    ax[1].plot(xs, CONTENT_ACC[k], "-o", ms=3, color=COL[k], label=k)
ax[0].set(title="Dev WER vs step (uncapped, 500 utts)", xlabel="optimizer step", ylabel="WER")
ax[0].axhline(0.30, ls=":", c="grey", lw=.8); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set(title="Content-frame accuracy vs step", xlabel="optimizer step", ylabel="content_acc")
ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_training_curves.png")); plt.close(fig)

# 2) old vs capped aggregate WER (10-utt subset)
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(RUNS)); w = 0.35
old = [agg[r]["WER_old"] for r in RUNS]
cap = [agg[r]["WER_cap"] for r in RUNS]
b1 = ax.bar(x - w/2, old, w, label="old hyp (uncapped / leaked)", color="#bdbdbd")
b2 = ax.bar(x + w/2, cap, w, label="capped at </s> (n_keep)", color="#2c7fb8")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.005, f"{b.get_height():.3f}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([r.split()[0] for r in RUNS])
ax.set(title="WER on 10 short dev utts: leaked decode vs duration-capped",
       ylabel="WER (micro, errors/ref-words)")
ax.legend(); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_old_vs_capped.png")); plt.close(fig)

# 3) per-utterance capped WER heatmap + trailing-garbage removed
mat = np.array([[next(r for r in rows if r["run"]==run and r["utt"]==i+1)["cap_wer"]
                 for i in range(len(REFS))] for run in RUNS])
rem = np.array([[next(r for r in rows if r["run"]==run and r["utt"]==i+1)["trailing_words_removed"]
                 for i in range(len(REFS))] for run in RUNS])
fig, ax = plt.subplots(2, 1, figsize=(11, 5.5))
im0 = ax[0].imshow(mat*100, aspect="auto", cmap="OrRd", vmin=0, vmax=40)
ax[0].set_yticks(range(len(RUNS))); ax[0].set_yticklabels([r.split()[0] for r in RUNS])
ax[0].set_xticks(range(len(REFS))); ax[0].set_xticklabels([f"u{i+1}" for i in range(len(REFS))])
ax[0].set_title("Capped WER per utterance (%)")
for i in range(len(RUNS)):
    for j in range(len(REFS)):
        ax[0].text(j, i, f"{mat[i,j]*100:.0f}", ha="center", va="center", fontsize=8)
fig.colorbar(im0, ax=ax[0], fraction=.025)
im1 = ax[1].imshow(rem, aspect="auto", cmap="Purples", vmin=0, vmax=max(1, rem.max()))
ax[1].set_yticks(range(len(RUNS))); ax[1].set_yticklabels([r.split()[0] for r in RUNS])
ax[1].set_xticks(range(len(REFS))); ax[1].set_xticklabels([f"u{i+1}" for i in range(len(REFS))])
ax[1].set_title("Trailing garbage words removed by capping (old #words - capped #words)")
for i in range(len(RUNS)):
    for j in range(len(REFS)):
        ax[1].text(j, i, f"{rem[i,j]}", ha="center", va="center", fontsize=8)
fig.colorbar(im1, ax=ax[1], fraction=.025)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_per_utt.png")); plt.close(fig)

print("\nwrote:", os.path.join(OUT, "cleaned_results.csv"))
print("wrote:", os.path.join(OUT, "fig1_training_curves.png"))
print("wrote:", os.path.join(OUT, "fig2_old_vs_capped.png"))
print("wrote:", os.path.join(OUT, "fig3_per_utt.png"))
