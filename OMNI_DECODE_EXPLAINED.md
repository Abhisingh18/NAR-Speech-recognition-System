# 🎨 How the A-CMLM model *decodes* — the omni-style mask-predict, explained

> A walkthrough of how our non-autoregressive ASR head turns audio into text by **iteratively
> unmasking** a grid of characters. Built around one real, error-free decode.
> (Same algorithm for English-HuBERT and Hindi-data2vec — only the encoder/vocab differ.)

---

## 🧩 The one-sentence idea

> **Instead of writing the transcript left-to-right one letter at a time, the model lays down a
> grid of `?` (masks) — one slot per 20 ms of audio — and fills them in over 32 passes,
> committing the letters it is most sure about first, like solving a crossword.**

```
   Autoregressive / CTC            A-CMLM (this model)
   ───────────────────            ────────────────────
   t → h → e → _ → d → …          ??????????????????????   step 0  (all masked)
   one step per letter            ??e????????f???·o?e??a…   step 24 (confident bits)
   left to right, ~140 steps       ·he dul· ···t fe·l ·ore…  step 31
                                   the dull light fell more… step 32  ✅  (done in 32 passes)
```

---

## 🎬 The example we'll trace

| | |
|---|---|
| **clip** | `1089-134686-0006.wav`, 10.55 s |
| **reference** | *the dull light fell more faintly upon the page whereon another equation began to unfold itself slowly and to spread abroad its widening tail* |
| **result** | **WER 0 / 24 words**, CER 1/140 (one stray space) ✅ |

---

## 📐 Step A — Set up the canvas (geometry)

The audio is 10.55 s. The encoder emits **one feature frame per 20 ms → 50 frames/sec**:

```
   10.55 s  ×  50 fps   =   527 frames  =  527 character slots

   frame:   0    1    2    3    4   …                                        526
          ┌────┬────┬────┬────┬────┬───────────────────────────────────────┬────┐
   seed:  │<s> │ ?  │ ?  │ ?  │ ?  │  ? ? ? ?  (all masked)  ? ? ? ? ? ? ?   │ ?  │
          └────┴────┴────┴────┴────┴───────────────────────────────────────┴────┘
            ▲fixed          └──────────────── M = 526 slots to fill ─────────────┘
```

Frame 0 is pinned to `<s>`. The other **526 slots** must be filled. Slots where nobody is speaking
will become `<fill>` (a "silence / nothing here" token); slots with speech become letters or `|` (space).

---

## 🕰️ Step B — The unmasking schedule (how many per pass)

We don't unmask everything at once. A schedule decides **how many slots to commit on each of the
32 passes**:

```
        τ · x                            n
r(n) = ─────────────         x = ───   (τ = 0.1  →  "back-loaded")
       1 + (τ−1)·x                N

k(n) = round(r(n)·M) − round(r(n−1)·M)      ← new commits on pass n
```

Because **τ = 0.1**, `r(n)` stays tiny for a long time and then shoots up — so the model commits
**very few slots early, then avalanches at the end**:

```
 pass n :   1    2    3    5    12    18    24     28    29    30    31    32
 r(n)   : .003 .007 .010 .018 .057  .114  .231   .38   .49   .60   .76   1.00
 commit : ┃▎    ▎    ▎    ▍     ▊    █▋    ███▌   ██…   …42   …57   …82   …128
 k(n)   :  2    1    2    3     4     6     14     33    42    57    82   128
                                                  └──────── 342 of 526 here ────────┘
```

> **Why back-load?** Early on the model has almost no committed context, so it should only place the
> few slots it is *extremely* sure about. Once a scaffold exists, the rest snap into place quickly.

**Worked number (pass 5):** `r(5)=0.1·(5/32)/(1−0.9·(5/32))=0.0182` → `round(0.0182·526)=10`;
previous was `7`; so **k(5)=3**. All the k's sum to exactly 526.

---

## 🎞️ Step C — Watch the sentence develop

Each pass: run the head once, look at every masked slot, and **commit the k(n) the model is most
confident about**. Here is the speech region actually developing (`·` = still masked):

```
 pass  8   ·····················o····a···········································  ← first letters
 pass 24   ··e············f··· ·o·e··a···l·········e···ge···e···n·····h········  ← sparse anchors
 pass 30   ··e d·l· ····t f··· ·o·e ·a·n·ly u··n··he ·age ·he·e·n·a··th··…       ← words emerging
 pass 31   ·he dul· ····t fe·l ·ore fa·n·ly upon ·he page ·he·e·n a··the· ·equat·on…
 pass 32   the dull light fell more faintly upon the page whereon another equation … tail  ✅
```

It reads like a **Polaroid developing** — or a crossword where the sure letters go in first and the
rest become obvious once they're surrounded.

---

## 🔊 Step D — The key behaviour: **silence first, words last**

Look at what *kind* of token each pass commits (`real` = a letter/space, `fill` = silence):

```
 pass  1 :  +2   (real 0 | fill 2)   ← commits SILENCE, no letters yet
 pass  5 :  +3   (real 0 | fill 3)
 pass  8 :  +3   (real 1 | fill 2)   ← first real letter
 pass 24 :  +14  (real 2 | fill 12)
 pass 30 :  +57  (real 18 | fill 39) ← the word avalanche
 pass 32 :  +128 (real 36 | fill 92) ← finishes everything
```

> **Why?** The `<fill>` frames (after the sentence ends, ~200 ms gaps) are *trivially* predictable —
> nothing is being said — so they carry the **highest confidence** and get committed first. Real
> letters are individually ambiguous until their neighbours exist, so they land in the late cascade.
> The model literally **carves out the silence to reveal where the words are**, then fills the words.

---

## 🎯 Step E — How it picks *which* slots (the exact math per pass)

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  logp   = log_softmax( head(current_grid) )        # [527 frames, 33] │
 │  conf, pred = logp.max(dim = -1)                    # confidence & best char per slot │
 │  chosen = gumbel_top_k( conf[masked] / temp , k )   # pick k masked slots  (temp = 5) │
 │  grid[chosen] = pred[chosen]                        # COMMIT — permanent │
 └──────────────────────────────────────────────────────────────────────┘
```

- **conf** = the model's confidence (max log-probability) at each still-masked slot.
- **gumbel-top-k @ temp 5** = "pick the k most confident slots, but with some randomness in the tie-breaks."
- Committed slots are **frozen** for the rest of the decode.

On this clean clip **average confidence ≈ 1.00 on every pass**, so the randomness never flips a
correct choice — which is exactly *why the result is error-free*. Real commit records from the trace:

```
 pass 30 commits:  frame138→"t"@1.00   frame112→"p"@1.00   frame42→"|"(space)@1.00   frame127→"s"@1.00 …
 pass 32 commits:  frame13→"h"@1.00     frame28→"i"@1.00     frame11→"i"@1.00 …
```

Only **one shared thing runs 32 times** — the small self-attention head. The heavy encoder runs
**once** (its features are cached), so 32 passes is cheap.

---

## ✅ Step F — The result

```
 HYP :  the dull light fell more faintly upon the page whereon another  equation … widening tail
 REF :  the dull light fell more faintly upon the page whereon another equation  … widening tail
 ────────────────────────────────────────────────────────────────────────────────────────────
 WER = 0 / 24 = 0.000        CER = 1 / 140 = 0.007   (the single error = one doubled space)
```

---

## 🧠 Mental model to explain it

> **It's a crossword, not a typewriter.**
> The model is handed a blank grid sized to the audio. Each pass it pencils in only the squares it's
> sure of — first the obvious blanks (silence), then the letters that anchor words, then everything
> else cascades. After 32 passes the whole grid is inked, and reading left-to-right up to the first
> `</s>` gives the transcript. Because every pass sees the **whole** grid (left *and* right context),
> a letter can be chosen using the words on *both* sides — unlike left-to-right decoding.

| term | plain meaning |
|---|---|
| **frame / slot** | one 20 ms character position (50 per second of audio) |
| **`<mask>`** | "unknown — to be filled" |
| **`<fill>`** | "silence / nothing spoken here" |
| **confidence** | how sure the model is about a slot's best character |
| **schedule (τ=0.1)** | commit few early, many late (back-loaded) |
| **N = 32** | number of refinement passes |
| **temp = 5** | randomness in which confident slots get picked first |

*Reproduce this trace:* `python train_sa_pe_lm_head/_omni_trace.py --pick 1` (English model, GPU with the ckpt).
